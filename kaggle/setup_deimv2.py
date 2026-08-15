#!/usr/bin/env python3
"""Prepare DEIMv2 on Kaggle: clone, patch, wire up the data, write configs.

DEIMv2's defaults are written for 118k COCO images. On 30 images per fold several of them
are not merely suboptimal, they stop training happening at all. This script applies the
minimum set of changes and prints what it did, so the paper can state it.

  total_batch_size 32 -> 4, and drop_last -> False
      floor(30/32) = 0 batches. The dataloader is empty and training silently does nothing.
      Open, unanswered DEIM issue #82.

  warmup_iter 2000 -> 200
      At ~8 iterations per epoch that is 250 epochs of pure warm-up, so the learning rate
      never reaches its useful range. The DEIM maintainer diagnosed exactly this in issue #5.

  every epoch-indexed schedule rescaled to ONE schedule shared by all variants
      flat_epoch, no_aug_epoch, policy.epoch, collate_fn.stop_epoch, mixup_epochs,
      copyblend_epochs and matcher_change_epoch are each written against that variant's own
      COCO budget, and those budgets differ enormously: 132 epochs for s, 68 for l, 500 for
      atto, 160 for n. Forcing `epoches: 120` on top of them without touching the rest does
      NOT give every variant a 120-epoch version of its recipe -- it gives each one an
      arbitrary slice of a different recipe.

      Concretely, DEIM trains in two stages and switches at collate_fn.stop_epoch, where it
      turns off augmentation and reloads best_stg1.pth. With `epoches: 120` and the stock
      values, s (stop 120), atto (468) and n (148) NEVER REACH THE SWITCH -- they train 120
      epochs of stage 1 and no fine-tune stage at all -- while l (60) and x (50) do reach it.
      Comparing those numbers to each other measures the schedule, not the architecture.

      So all of it is pinned to a single schedule, stated here so the paper can quote it:
      120 epochs, flat_epoch 64, no_aug_epoch 20, stop_epoch 100, matcher_change_epoch 90.
      The relationships are the repo's own: stop_epoch = epoches - no_aug_epoch, and
      flat_epoch = 4 + epoches // 2 (both hold in every stock config).

  num_classes 4, not 3
      Categories are 1-indexed (1..3), so the head needs 4 slots. DEIMKit documents
      classes + 1, and DEIM issue #12 reports num_classes=1 failing where 2 worked.

  requirements.txt is NOT installed
      It pins torch==2.5.1 / torchvision==0.20.1, which would replace Kaggle's CUDA-matched
      build and break every GPU op. Only the genuinely missing packages are installed.

Usage:
    !python setup_deimv2.py --data /kaggle/input/midi-gesture-v2 \
                            --ckpt /kaggle/input/deimv2-checkpoints --work /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Intellindust-AI-Lab/DEIMv2.git"
VARIANTS = {
    "atto": "deimv2_hgnetv2_atto_coco.pth", "femto": "deimv2_hgnetv2_femto_coco.pth",
    "pico": "deimv2_hgnetv2_pico_coco.pth", "n": "deimv2_hgnetv2_n_coco.pth",
    "s": "deimv2_dinov3_s_coco.pth", "m": "deimv2_dinov3_m_coco.pth",
    "l": "deimv2_dinov3_l_coco.pth", "x": "deimv2_dinov3_x_coco.pth",
}
GROUPS = ["fold0", "fold1", "fold2", "test"]



def resolve_data(root: Path) -> Path:
    """Find the real data root under a Kaggle mount, whatever layout Kaggle used.

    Classic datasets mount at /kaggle/input/<slug>/. Data Hub mounts at
    /kaggle/input/datasets/<user>/<slug>/<slug>/, and `--dir-mode zip` adds one more level by
    preserving the uploaded folder. Rather than encode any of that, search for the marker
    files. Getting this wrong yields "0 images found" and a model trained on nothing, so it is
    worth being thorough.
    """
    if not root.exists():
        raise SystemExit(
            f"{root} does not exist.\n"
            f"  Kaggle mounts vary: /kaggle/input/<slug>/ for classic datasets,\n"
            f"  /kaggle/input/datasets/<user>/<slug>/<slug>/ for Data Hub.\n"
            f"  Find it with:  !find /kaggle/input -name folds.json")

    def ok(p: Path) -> bool:
        return (p / "folds.json").exists() and (p / "manifest.csv").exists()

    if ok(root):
        return root
    for depth in range(1, 6):
        for cand in sorted(root.glob("/".join(["*"] * depth))):
            if cand.is_dir() and ok(cand):
                print(f"  data root: {cand}")
                return cand
    raise SystemExit(f"no folds.json + manifest.csv anywhere under {root}"
                     f"   try:  !find {root} -name folds.json")

def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=False, **kw)


def stage_data(data: Path, work: Path):
    """COCO tree DEIMv2 expects: images/<group>/ + annotations/instances_<group>.json.

    Built under /kaggle/working because /kaggle/input is read-only and DEIMv2 writes
    alongside its data. Uses the HBB geometry -- DEIMv2 predicts upright boxes only, so the
    OBB arm stays YOLO26-internal.
    """
    root = work / "deim_data"
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    manifest = {}
    import csv
    with (data / "manifest.csv").open() as fh:
        for r in csv.DictReader(fh):
            manifest[r["id"]] = r
    folds = json.loads((data / "folds.json").read_text())
    id_of = {r["original"].replace(".jpg", ""): r["id"] for r in manifest.values()}

    for g in GROUPS:
        d = root / "images" / g
        d.mkdir(parents=True, exist_ok=True)
        for stem, grp in folds["photo_group"].items():
            if grp != g:
                continue
            pid = id_of[stem]
            src = data / "hbb" / "images" / f"{pid}.jpg"
            dst = d / f"{pid}.jpg"
            if not dst.exists():
                shutil.copy2(src, dst)
        shutil.copy2(data / "coco" / f"instances_{g}.json",
                     root / "annotations" / f"instances_{g}.json")
        n = len(list(d.glob("*.jpg")))
        print(f"    {g}: {n} images")

    # 3-fold CV means training on the OTHER TWO folds. COCO takes a single annotation file,
    # so they are merged here -- with ids re-issued, because each per-fold json numbers its
    # images from 1 and a naive concatenation would collide silently.
    for k, held in enumerate(GROUPS[:3]):
        others = [g for g in GROUPS[:3] if g != held]
        d = root / "images" / f"train_fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        merged = {"images": [], "annotations": [], "categories": None}
        next_img, next_ann = 1, 1
        for g in others:
            js = json.loads((root / "annotations" / f"instances_{g}.json").read_text())
            merged["categories"] = js["categories"]
            remap = {}
            for im in js["images"]:
                remap[im["id"]] = next_img
                merged["images"].append({**im, "id": next_img})
                src = root / "images" / g / im["file_name"]
                dst = d / im["file_name"]
                if not dst.exists():
                    shutil.copy2(src, dst)
                next_img += 1
            for an in js["annotations"]:
                merged["annotations"].append({**an, "id": next_ann,
                                              "image_id": remap[an["image_id"]]})
                next_ann += 1
        (root / "annotations" / f"instances_train_fold{k}.json").write_text(
            json.dumps(merged, indent=1))
        print(f"    train_fold{k}: {len(merged['images'])} images, "
              f"{len(merged['annotations'])} annotations  (= {' + '.join(others)})")
    return root


def patch_repo(repo: Path):
    """Apply the torchvision v2 transform rename if this torchvision needs it."""
    import torchvision
    tv = tuple(int(x) for x in torchvision.__version__.split(".")[:2])
    print(f"  torchvision {torchvision.__version__}")
    if tv < (0, 21):
        print("    < 0.21, no transform patch needed")
        return
    n = 0
    for p in repo.rglob("*.py"):
        s = p.read_text()
        o = s
        s = re.sub(r"\bdef _get_params\b", "def make_params", s)
        s = re.sub(r"\bdef _transform\b", "def transform", s)
        s = re.sub(r"\bself\._get_params\b", "self.make_params", s)
        s = re.sub(r"\bself\._transform\b", "self.transform", s)
        if s != o:
            p.write_text(s)
            n += 1
    print(f"    patched {n} file(s) for the torchvision >=0.21 v2 transform API (upstream PR #139)")


def write_config(repo: Path, out: Path, variant: str, fold: str, data_root: Path,
                 ckpt: Path, epochs: int, batch: int):
    k = GROUPS.index(fold)
    # Warm-up must be a small fraction of the run, not a fixed 2000. With 30 images at batch 4
    # that is 8 iterations per epoch, so the COCO default would spend 250 epochs warming up and
    # the learning rate would never reach its useful range (DEIM issue #5).
    n_train = len(json.loads(
        (data_root / "annotations" / f"instances_train_fold{k}.json").read_text())["images"])
    iters_per_epoch = max(1, -(-n_train // batch))
    warmup = max(20, int(0.05 * iters_per_epoch * epochs))
    cfg = out / f"deimv2_{variant}_{fold}.yml"
    # The base config must match the checkpoint's backbone. atto/femto/pico/n are HGNetv2;
    # s/m/l/x ship as DINOv3. The repo has BOTH families for s/m/l/x, so pointing at the wrong
    # one loads DINOv3 weights into an HGNetv2 graph -- a mismatch that surfaces as an opaque
    # state_dict error 40 minutes into a sweep. Derive it from the checkpoint filename instead.
    base_cfg = repo / "configs" / "deimv2" / (Path(ckpt).stem + ".yml")
    if not base_cfg.exists():
        raise SystemExit(f"no base config for {variant}: {base_cfg}")
    # One schedule for every variant, derived from OUR epoch budget rather than inherited from
    # whichever COCO recipe this backbone happened to ship with. Without this, s/atto/n never
    # reach the stage-2 switch and l/x do -- see the module docstring.
    no_aug = max(4, round(0.167 * epochs))          # 20 at 120 epochs
    stop_epoch = epochs - no_aug                    # the repo's own relationship
    flat_epoch = 4 + epochs // 2                    # ditto, stated in every stock config
    matcher_change = round(0.75 * epochs)
    cfg.write_text(f"""__include__: [ '{base_cfg}' ]

# 3 gesture classes, 1-indexed in the COCO json, so the head needs 4 slots
num_classes: 4
remap_mscoco_category: False

train_dataloader:
  total_batch_size: {batch}          # 32 would give floor(30/32)=0 batches -> empty loader
  dataset:
    img_folder: {data_root}/images/train_fold{k}
    ann_file: {data_root}/annotations/instances_train_fold{k}.json
    transforms:
      policy:
        epoch: [4, {flat_epoch}, {stop_epoch}]   # [warm, flat, stop], the stock pattern
  collate_fn:
    stop_epoch: {stop_epoch}               # stage 1 -> stage 2: aug off, reload best_stg1.pth
    mixup_epochs: [4, {flat_epoch}]
    copyblend_epochs: [4, {stop_epoch}]
  drop_last: False                   # with 30 images, True discards the only partial batch

val_dataloader:
  total_batch_size: {batch}
  dataset:
    img_folder: {data_root}/images/{fold}
    ann_file: {data_root}/annotations/instances_{fold}.json

epoches: {epochs}
flat_epoch: {flat_epoch}
no_aug_epoch: {no_aug}
DEIMCriterion:
  matcher:
    matcher_change_epoch: {matcher_change}

lr_warmup:
  warmup_iter: {warmup}                    # 5% of {iters_per_epoch * epochs} total iters
                                     # ({iters_per_epoch} iters/epoch x {epochs} epochs)

output_dir: {out}/runs/{variant}_{fold}
tuning: {ckpt.resolve()}
""")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--skip-clone", action="store_true")
    args = ap.parse_args()
    args.data = resolve_data(args.data)

    repo = args.work / "DEIMv2"
    print("=== 1. repository ===")
    if not repo.exists() and not args.skip_clone:
        run(["git", "clone", "--depth", "1", REPO_URL, str(repo)])
    print(f"  {repo}  {'present' if repo.exists() else 'MISSING'}")

    print("\n=== 2. dependencies ===")
    print("  NOT running pip install -r requirements.txt (it pins torch==2.5.1)")
    run([sys.executable, "-m", "pip", "install", "-q",
         "faster-coco-eval", "calflops", "omegaconf", "loguru", "easydict"])

    print("\n=== 3. torch / torchvision compatibility ===")
    if repo.exists():
        patch_repo(repo)

    print("\n=== 4. data ===")
    data_root = stage_data(args.data, args.work)

    print("\n=== 5. configs ===")
    cfg_dir = args.work / "deim_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for v in args.variants:
        ck = (args.ckpt / VARIANTS[v]).resolve()
        if not ck.exists():
            print(f"  {v}: checkpoint missing ({ck.name}) - skipped")
            continue
        for fold in GROUPS[:3]:
            write_config(repo, cfg_dir, v, fold, data_root, ck, args.epochs, args.batch)
            made += 1
    print(f"  wrote {made} config(s) -> {cfg_dir}")

    print("\n=== next ===")
    print(f"  cd {repo} && python train.py -c {cfg_dir}/deimv2_n_fold0.yml --use-amp --seed 42")
    print("\n  Before trusting a run, assert the loader is non-empty:")
    print("    len(train_dataloader) > 0     # DEIM issue #82")
    print("    warm-up ends inside the first 10% of epochs")


if __name__ == "__main__":
    main()
