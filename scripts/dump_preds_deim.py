#!/usr/bin/env python3
"""Dump DEIMv2 detections to a common JSON, one file per (variant, fold).

WHY A DUMP AND NOT A METRICS SCRIPT. The confusion matrix for a model has to pool all three
cross-validation folds -- each fold's checkpoint scored on its own held-out fold, which
between them cover all 90 annotations without ever touching the test set. Nothing about that
pooling is architecture-specific, so the architecture-specific part stops at "produce
detections" and everything downstream is shared with the YOLO arm (scripts/pool_metrics.py).

This half runs ON THE TRAINING BOX. Rebuilding a DINOv3 backbone on the laptop would mean
cloning DEIMv2 there and fetching its ViT distillation weights, to compute numbers that are
a few hundred KB of JSON. Inference happens where the environment already works.

TWO DETAILS THAT SILENTLY CORRUPT THE OUTPUT IF MISSED, both taken from
tools/inference/torch_inf.py:

  Normalisation is backbone-dependent. DINOv3 variants (s/m/l/x) expect ImageNet mean/std;
  HGNetv2 variants (atto/femto/pico/n) expect raw [0,1] tensors. Applying the wrong one does
  not raise -- it just makes the model worse, which reads as a bad model.

  The EMA weights are the trained model. checkpoint['ema']['module'] is what evaluation used
  and therefore what produced the reported AP; checkpoint['model'] is the raw non-averaged
  copy and scores lower. Loading the wrong key gives numbers that disagree with the AP table
  for no visible reason.

Class indices: the COCO json is 1-indexed (1..3) and remap_mscoco_category is False, so the
head emits 1..3 and slot 0 -- which `num_classes: 4` creates and nothing ever trains -- is
dead. Output is converted to 0-indexed thumbout/openhand/closedhand here.

Slot 0 is DROPPED, not remapped. A well-trained variant never emits it (deimv2_n produced
4500 detections across a fold without a single one), but a barely-trained one does: atto,
at AP 0.065, emits it as noise. Mapping it to class -1 would corrupt every downstream count,
so it is discarded and the discard is recorded per dump. Any label above 3 aborts the run,
because that would mean the indexing assumption itself is wrong rather than merely noisy.

ONE CHECKPOINT PER SUBPROCESS, and this is not defensive tidiness. DEIM's config system
registers modules in a process-global registry, so building a second model in the same
interpreter inherits state from the first. The failure is not clean: `pico` loaded alone
reports raw [0,1] input (correct for its HGNetv2 backbone), while the same checkpoint loaded
sixth in a batch reports ImageNet normalisation and silently produces different detections.
The DINOv3 variants are luckier and simply die with a state_dict mismatch. Since one of those
two outcomes is silent, every checkpoint gets a fresh interpreter.

    python3 dump_preds_deim.py --configs deim_configs --ckpts best_ckpts \
                               --images deim_data/images --out preds_deim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLASS_NAMES = ["thumbout", "openhand", "closedhand"]
SCORE_FLOOR = 0.001         # collect nearly everything; thresholding happens downstream
FOLDS = ["fold0", "fold1", "fold2"]


def build_model(repo: Path, cfg_path: Path, ckpt: Path, device: str):
    sys.path.insert(0, str(repo))
    import torch
    import torch.nn as nn
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(cfg_path), resume=str(ckpt))
    # The backbone's ImageNet weights are downloaded on construction and then immediately
    # overwritten by the checkpoint. Skipping the download is not an optimisation here: on a
    # box with no outbound access it is the difference between running and not.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(str(ckpt), map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Deployed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    model = Deployed().to(device).eval()
    size = cfg.yaml_cfg["eval_spatial_size"]
    vit = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    used_ema = "ema" in checkpoint
    return model, size, vit, used_ema


def run_fold(model, size, vit, images: list[Path], device: str):
    import torch
    import torchvision.transforms as T
    from PIL import Image

    ops = [T.Resize(tuple(size)), T.ToTensor()]
    if vit:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tf = T.Compose(ops)

    out = {}
    hist = {}                      # raw label -> count, kept for the audit trail
    dropped, dropped_max_conf = 0, 0.0
    for p in images:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        x = tf(im).unsqueeze(0).to(device)
        orig = torch.tensor([[w, h]]).to(device)
        with torch.no_grad():
            labels, boxes, scores = model(x, orig)
        lab = labels[0].cpu().numpy()
        box = boxes[0].cpu().numpy()
        scr = scores[0].cpu().numpy()
        dets = []
        for l, b, s in zip(lab, box, scr):
            if s < SCORE_FLOOR:
                continue
            l = int(l)
            hist[l] = hist.get(l, 0) + 1
            if l > 3:
                raise SystemExit(f"predicted class id {l} > 3. The head has 4 slots, so the "
                                 f"1-indexed COCO assumption is wrong -- fix the mapping "
                                 f"before trusting any confusion matrix.")
            if l == 0:                                # untrained slot, noise only
                dropped += 1
                dropped_max_conf = max(dropped_max_conf, float(s))
                continue
            dets.append({"cls": l - 1,                # 1-indexed COCO -> 0-indexed
                         "conf": round(float(s), 6),
                         "box": [round(float(v), 2) for v in b]})
        out[p.stem] = sorted(dets, key=lambda d: -d["conf"])
    return out, hist, dropped, dropped_max_conf


def dump_one(args, ck: Path):
    """Everything for a single checkpoint. Runs in its own interpreter -- see module docs."""
    stem = ck.stem.split("__")[0]                     # "<variant>_<fold>__<stg1|stg2>"
    variant, fold = stem.rsplit("_", 1)
    cfg_path = args.configs / f"deimv2_{variant}_{fold}.yml"
    if not cfg_path.exists():
        raise SystemExit(f"no config {cfg_path}")
    # --group overrides which images to score. Default is the checkpoint's own held-out fold
    # (cross-validation); --group test runs the same checkpoint against the untouched test
    # set, which is how each model gets a test number without ever selecting on it.
    group = args.group or fold
    imgs = sorted((args.images / group).glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {args.images / group}")

    model, size, vit, used_ema = build_model(args.repo, cfg_path, ck, args.device)
    preds, hist, dropped, drop_conf = run_fold(model, size, vit, imgs, args.device)

    n_det = sum(len(v) for v in preds.values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"deimv2_{variant}_{fold}.json").write_text(json.dumps({
        "arch": "deimv2", "model": variant, "fold": fold, "scored_on": group, "geom": "hbb",
        "imgsz": list(size), "vit_normalised": vit, "used_ema": used_ema,
        "checkpoint": ck.name, "score_floor": SCORE_FLOOR,
        "raw_label_histogram": {str(k): v for k, v in sorted(hist.items())},
        "dropped_slot0": dropped, "dropped_slot0_max_conf": round(drop_conf, 6),
        "class_names": CLASS_NAMES, "preds": preds}))
    # A slot-0 detection above the operating threshold would actually reach the confusion
    # matrix if it were not dropped, so surface that case rather than hiding it in JSON.
    flag = ""
    if dropped:
        flag = f"  slot0 dropped {dropped} (max conf {drop_conf:.3f})"
        if drop_conf >= 0.25:
            flag += "  <-- ABOVE OPERATING THRESHOLD"
    print(f"RESULT  {stem:18s} {len(imgs):3d} imgs  {n_det:6d} dets  "
          f"{'ema' if used_ema else 'RAW'}  {'norm' if vit else 'raw01'}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/workspace/midi/DEIMv2"))
    ap.add_argument("--configs", type=Path, default=Path("/workspace/midi/deim_configs"))
    ap.add_argument("--ckpts", type=Path, default=Path("/workspace/midi/best_ckpts"))
    ap.add_argument("--images", type=Path, default=Path("/workspace/midi/deim_data/images"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/midi/preds_deim"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--group", default=None,
                    help="image group to score; default = the checkpoint's own held-out fold")
    ap.add_argument("--one", type=Path, default=None,
                    help="internal: process exactly this checkpoint, in this process")
    args = ap.parse_args()

    if args.one:
        dump_one(args, args.one)
        return

    ckpts = [c for c in sorted(args.ckpts.glob("*.pth"))
             if not args.variants or c.stem.split("__")[0].rsplit("_", 1)[0] in args.variants]
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpts}")
    args.out.mkdir(parents=True, exist_ok=True)

    import subprocess
    done, failed = 0, []
    for i, ck in enumerate(ckpts, 1):
        stem = ck.stem.split("__")[0]
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one", str(ck),
               "--repo", str(args.repo), "--configs", str(args.configs),
               "--ckpts", str(args.ckpts), "--images", str(args.images),
               "--out", str(args.out), "--device", args.device]
        if args.group:
            cmd += ["--group", args.group]
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), None)
        if p.returncode != 0 or line is None:
            tail = (p.stdout + p.stderr).strip().splitlines()
            failed.append((stem, tail[-1][:200] if tail else f"rc={p.returncode}"))
            print(f"  [{i}/{len(ckpts)}] FAIL {stem}: {failed[-1][1][:120]}")
            continue
        print(f"  [{i}/{len(ckpts)}]" + line[len("RESULT"):])
        done += 1

    print(f"\n{done} dump(s) -> {args.out}")
    if failed:
        print(f"{len(failed)} FAILED:")
        for s, m in failed:
            print(f"  {s}: {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
