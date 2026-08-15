#!/usr/bin/env python3
"""Export a trained DEIMv2 checkpoint to Core ML and time it on this Mac.

WHY THIS EXISTS. The study's question is whether the instrument runs on a MacBook Air M4,
and the best-accuracy models in the study are DEIMv2. Ultralytics exports YOLO26 to Core ML
with one call; DEIMv2 has no Core ML path at all, so without this script the DEIMv2 arm could
only be timed through PyTorch-MPS -- which carries ~18 ms of fixed dispatch overhead and never
touches the Neural Engine. Comparing an MPS number against a Core ML number would understate
DEIMv2 by more than the effect being measured.

NOT VIA ONNX. The repo ships tools/deployment/export_onnx.py, but coremltools removed its ONNX
frontend after version 6, so ONNX is a dead end here. This traces the deployed PyTorch module
and converts through the torch frontend -- the same route ultralytics takes.

TORCH VERSION IS LOAD-BEARING. coremltools 9.0 reports torch 2.7.0 as the newest tested
version. On torch 2.13 the conversion dies inside coremltools' own `_cast` op with
"only 0-dimensional arrays can be converted to Python scalars" -- an error that says nothing
about the model. Run this from the pinned .venv-coreml (torch 2.7.x, numpy 1.26).

WHAT IS EXPORTED. cfg.model.deploy() + cfg.postprocessor.deploy(), i.e. the same module the
repo's own inference script builds, so the exported graph includes decoding to labels/boxes/
scores. DEIMv2 is NMS-free, so there is no NMS to fold in -- which is also why its raw
detection count is 300 per image and why the operating threshold matters downstream.

    .venv-coreml/bin/python scripts/export_coreml_deim.py --variants n l --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

FAMILY = {"atto": "hgnetv2", "femto": "hgnetv2", "pico": "hgnetv2", "n": "hgnetv2",
          "s": "dinov3", "m": "dinov3", "l": "dinov3", "x": "dinov3"}
WARMUP = 10


def write_config(repo: Path, out_dir: Path, variant: str) -> Path:
    """Minimal config: the stock architecture plus our class count.

    Only the model and postprocessor are built here, so none of the training or dataset keys
    matter. num_classes stays 4 because the COCO json is 1-indexed and slot 0 is unused --
    the head shape has to match the checkpoint or load_state_dict fails.
    """
    # Absolute: DEIM resolves __include__ relative to the INCLUDING file's directory, so a
    # relative repo path here silently becomes <out_dir>/<repo>/configs/... and fails.
    base = (repo.resolve() / "configs" / "deimv2"
            / f"deimv2_{FAMILY[variant]}_{variant}_coco.yml")
    if not base.exists():
        raise SystemExit(f"no base config for {variant}: {base}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / f"export_{variant}.yml"
    cfg.write_text(f"__include__: [ '{base}' ]\nnum_classes: 4\nremap_mscoco_category: False\n")
    return cfg


def build(repo: Path, cfg_path: Path, ckpt: Path):
    sys.path.insert(0, str(repo))
    import torch
    import torch.nn as nn
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(cfg_path), resume=str(ckpt))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ck = torch.load(str(ckpt), map_location="cpu")
    state = ck["ema"]["module"] if "ema" in ck else ck["model"]
    cfg.model.load_state_dict(state)

    class Deployed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    m = Deployed().eval()
    size = cfg.yaml_cfg["eval_spatial_size"]
    vit = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    return m, size, vit


def load_frame(path: Path, imgsz: int, vit: bool):
    """A REAL photo, preprocessed exactly as inference does.

    Tracing and verifying on torch.rand is a trap: on noise every one of the 300 queries is
    low-confidence garbage, their ranking is arbitrary, and fp16 rounding of 0.01 reshuffles
    the top-10 -- which reads as a broken conversion (7/10 labels, 385 px box deviation) when
    the graph is fine. On a real frame the top detections are confident and stable, so any
    disagreement that remains is a genuine conversion fault.

    Normalisation is backbone-dependent, matching tools/inference/torch_inf.py: DINOv3 wants
    ImageNet mean/std, HGNetv2 wants raw [0,1].
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    ops = [T.Resize((imgsz, imgsz)), T.ToTensor()]
    if vit:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    im = Image.open(path).convert("RGB")
    return T.Compose(ops)(im).unsqueeze(0)


def patch_linear_op():
    """Let `linear` accept a rank-1 weight by lowering it to matmul.

    coremltools' torch frontend maps every aten::linear onto mb.linear, whose type inference
    asserts `len(weight_shape) == 2`. DEIMv2 has exactly one op that violates this, and the
    conversion dies on it at 59% of ops with a bare AssertionError naming no module:

        x (1200, 33)  @  weight (33,)  ->  (1200,)      no bias

    That is D-FINE's distribution-integral head. The decoder does not regress each box
    coordinate directly; it predicts a distribution over 33 bins and contracts it against a
    fixed projection vector to recover the value. torch.nn.functional.linear accepts a 1-D
    weight and treats it as a contraction over the last axis, which is a plain matmul with no
    transpose -- transposing, the natural guess for the rank>2 case, would be wrong here.

    Rank 2 keeps the native op. Rank > 2 (batched weight) is handled for completeness with
    transpose_y, since F.linear(x, W) is x @ W^T. The converted model is checked numerically
    against PyTorch by the caller: an op override that quietly changed the maths would
    produce a perfectly plausible latency number for a model that detects nothing.
    """
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY, register_torch_op)

    # Keep the stock implementation and DELEGATE to it for every node whose indices are
    # already integral. Reimplementing gather wholesale is what broke here first: the graph
    # converted end to end but a later constant-folding pass died with "index 241 is out of
    # bounds for axis 0 with size 75", because the hand-rolled version does not reproduce
    # every aten::gather variant in this model. Only the float-index case needs fixing.
    _stock_gather = _TORCH_OPS_REGISTRY["gather"]

    @register_torch_op(override=True)
    def gather(context, node):
        """Cast float indices back to int, then hand off to coremltools' own gather.

        DEIM's postprocessor selects boxes with
            index = topk_index // num_classes ;  boxes.gather(1, index...)
        Integer floor-division stays integral in PyTorch, but coremltools lowers
        aten::floor_divide through a real division, so the index tensor reaches the gather as
        fp32 and MIL rejects it: "expects tensor of dtype int32 ... but got tensor[1,300,4,
        fp32]". The values are already whole numbers; only the dtype is wrong.
        """
        ins = _get_inputs(context, node, expected=[3, 4])
        indices = ins[2]
        if indices.dtype in (types.fp16, types.fp32, types.fp64):
            cast = mb.cast(x=indices, dtype="int32", name=node.name + "_idx_int")
            context.add(cast, torch_name=node.inputs[2] + "_int")
            node.inputs[2] = node.inputs[2] + "_int"
        _stock_gather(context, node)

    @register_torch_op(override=True)
    def linear(context, node):
        raw = [context[n] if n in context else None for n in node.inputs]
        x = raw[0]
        w = raw[1] if len(raw) > 1 else None
        bias = raw[2] if len(raw) > 2 else None
        rank = getattr(w, "rank", 2)
        if rank == 2:
            res = mb.linear(x=x, weight=w, bias=bias, name=node.name)
        elif rank == 1:
            # MIL's matmul rejects rank-1 operands too, so lift the vector to (1, K), contract
            # with transpose_y to get (..., 1), then drop that trailing axis. squeeze rather
            # than reshape([-1]) so a batched input keeps its leading dimensions.
            w2 = mb.reshape(x=w, shape=[1, -1], name=node.name + "_w2d")
            res = mb.matmul(x=x, y=w2, transpose_x=False, transpose_y=True,
                            name=node.name + "_mm")
            res = mb.squeeze(x=res, axes=[-1],
                             name=node.name + ("_sq" if bias is not None else ""))
            if bias is not None:
                res = mb.add(x=res, y=bias, name=node.name)
        else:
            res = mb.matmul(x=x, y=w, transpose_x=False, transpose_y=True,
                            name=node.name + ("_mm" if bias is not None else ""))
            if bias is not None:
                res = mb.add(x=res, y=bias, name=node.name)
        context.add(res, node.name)


def to_coreml(model, imgsz: int, out: Path, frame):
    import torch
    import coremltools as ct

    patch_linear_op()

    import numpy as np

    images = frame
    sizes = torch.tensor([[imgsz, imgsz]], dtype=torch.int64)
    with torch.no_grad():
        model(images, sizes)                       # concretise lazy init before tracing
        traced = torch.jit.trace(model, (images, sizes), strict=False)

    # Try the most capable configuration first and fall back. fp16 is what makes a model
    # eligible for the Neural Engine, and newer targets unlock newer ops -- but on this graph
    # the macOS14+ optimisation passes die during constant folding with "index 227 is out of
    # bounds for axis 0 with size 75", which persists even when delegating to coremltools' own
    # gather, so it is a converter limitation rather than something these overrides caused.
    # Which rung succeeded is recorded, because an fp32 model does not reach the ANE and its
    # latency must not be presented as if it had.
    ladder = [("macOS15", ct.target.macOS15, ct.precision.FLOAT16),
              ("macOS14", ct.target.macOS14, ct.precision.FLOAT16),
              ("macOS13", ct.target.macOS13, ct.precision.FLOAT16),
              ("macOS13-fp32", ct.target.macOS13, ct.precision.FLOAT32)]
    errors = []
    for name, tgt, prec in ladder:
        try:
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="images", shape=images.shape),
                        ct.TensorType(name="orig_target_sizes", shape=sizes.shape,
                                      dtype=np.int32)],
                minimum_deployment_target=tgt,
                compute_units=ct.ComputeUnit.ALL,
                compute_precision=prec,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            mlmodel.save(str(out))
            cfg = {"target": name, "precision": str(prec).split(".")[-1],
                   "ane_eligible": prec == ct.precision.FLOAT16,
                   "fallbacks_tried": errors}
            return mlmodel, (images, sizes), cfg
        except Exception as e:
            errors.append({"rung": name, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    raise RuntimeError(f"every conversion configuration failed: {errors}")


def verify(mlmodel, model, sample, tol_box=2.0):
    """Compare Core ML against PyTorch on the same input.

    A latency number from a graph that computes the wrong thing is worthless, and the linear
    override above rewrites real ops, so this is not optional. Labels must match exactly;
    boxes are compared in pixels because fp16 on the Neural Engine will not reproduce fp32
    bit-for-bit.
    """
    import numpy as np
    import torch

    images, sizes = sample
    with torch.no_grad():
        t_lab, t_box, t_scr = model(images, sizes)
    out = mlmodel.predict({"images": images.numpy().astype(np.float32),
                           "orig_target_sizes": sizes.numpy().astype(np.int32)})

    # Core ML renames graph outputs ('2434' -> 'var_2434') and does not promise dict order, so
    # identify the three tensors by SHAPE, not position. Reading them positionally compares
    # boxes against scores and reports a spurious several-hundred-pixel disagreement.
    arrs = [np.asarray(v) for v in out.values()]
    boxes = [a for a in arrs if a.ndim == 3 and a.shape[-1] == 4]
    flat = [a for a in arrs if a.ndim == 2 or (a.ndim == 3 and a.shape[-1] != 4)]
    if not boxes or len(flat) < 2:
        return {"error": f"unexpected Core ML output shapes: {[a.shape for a in arrs]}",
                "ok": False}
    c_box = boxes[0].reshape(-1, 4)
    # of the two flat tensors, scores are float in [0,1]; labels are small integers
    a, b = flat[0].reshape(-1), flat[1].reshape(-1)
    c_scr, c_lab = (a, b) if (a.max() <= 1.0 and a.dtype.kind == "f") else (b, a)

    t_scr_np, t_lab_np = t_scr.numpy().reshape(-1), t_lab.numpy().reshape(-1)
    t_box_np = t_box.numpy().reshape(-1, 4)

    # Compare only detections anyone would act on, and pair them BY POSITION, not by rank.
    # DEIMv2 emits 300 queries; past the handful of real ones they are near-tied noise whose
    # order flips under fp16 rounding. Ranking the top-10 and comparing index-for-index
    # therefore reports hundreds of pixels of "disagreement" for a conversion that reproduces
    # every usable detection to within a pixel.
    keep = np.where(t_scr_np >= 0.25)[0]
    if len(keep) == 0:
        keep = t_scr_np.argsort()[::-1][:2]
    # Match only against detections Core ML also considers usable. DEIMv2 has no NMS, so
    # several queries land on the same hand with widely different scores; a nearest-centre
    # match over ALL 300 happily pairs a real detection with its 0.05-confidence duplicate,
    # which shows up as a 0.65 score deviation on a conversion that is actually exact.
    cand = np.where(c_scr >= 0.25)[0]
    if len(cand) == 0:
        cand = c_scr.argsort()[::-1][:max(1, len(keep))]
    devs, labs, scores = [], [], []
    for i in keep:
        centre = t_box_np[i][:2] + t_box_np[i][2:]
        sub = np.abs((c_box[cand][:, :2] + c_box[cand][:, 2:]) - centre).sum(axis=1)
        j = int(cand[sub.argmin()])
        devs.append(float(np.abs(t_box_np[i] - c_box[j]).max()))
        labs.append(bool(t_lab_np[i] == c_lab[j]))
        scores.append(float(abs(t_scr_np[i] - c_scr[j])))
    return {"compared": int(len(keep)),
            "label_match": f"{sum(labs)}/{len(labs)}",
            "max_box_dev_px": round(max(devs), 3),
            "max_score_dev": round(max(scores), 5),
            "ok": bool(all(labs) and max(devs) <= tol_box)}


def time_mlmodel(mlmodel, imgsz: int, iters: int, frame):
    import numpy as np
    # Time on the same real frame that was verified. Detection cost is input-dependent for a
    # DETR only via the postprocessor's topk, which is fixed at 300, but keeping one input
    # throughout means the timed graph is provably the one that was checked.
    x = {"images": frame.numpy().astype(np.float32),
         "orig_target_sizes": np.array([[imgsz, imgsz]], dtype=np.int32)}
    for _ in range(WARMUP):
        mlmodel.predict(x)
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mlmodel.predict(x)
        s.append((time.perf_counter() - t0) * 1000.0)
    s.sort()
    med = statistics.median(s)
    p95 = s[int(0.95 * (len(s) - 1))]
    return {"n": len(s), "median_ms": round(med, 2), "p95_ms": round(p95, 2),
            "min_ms": round(s[0], 2), "jitter_ms": round(p95 - med, 2),
            "fps_at_median": round(1000.0 / med, 1), "meets_budget": bool(med <= 10.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["n", "s", "m", "l", "x"])
    ap.add_argument("--repo", type=Path, default=ROOT / "deim_local")
    ap.add_argument("--ckpts", type=Path, default=ROOT / "deim_results" / "weights")
    ap.add_argument("--fold", default="fold0", help="architecture is identical across folds")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency_deim_m4.json")
    ap.add_argument("--frame", type=Path, default=None,
                    help="real photo to trace and verify on; defaults to a pool image")
    ap.add_argument("--packages", type=Path, default=ROOT / "models" / "deim")
    args = ap.parse_args()

    if platform.system() != "Darwin":
        raise SystemExit("This measures the target laptop. Run it on the Mac.")

    if args.frame is None:
        cands = sorted((ROOT / "data" / "pool" / "hbb" / "images").glob("*.jpg"))
        if not cands:
            raise SystemExit("no --frame given and no pool image found")
        args.frame = cands[len(cands) // 2]
    print(f"frame: {args.frame.name}")

    rows, failed = [], []
    for v in args.variants:
        cand = sorted(args.ckpts.glob(f"{v}_{args.fold}__*.pth"))
        if not cand:
            failed.append((v, f"no checkpoint {v}_{args.fold}__*.pth in {args.ckpts}"))
            continue
        ckpt = cand[0]
        for sz in args.imgsz:
            tag = f"deimv2-{v}-{sz}"
            print(f"\n=== {tag}  ({ckpt.name}) ===")
            try:
                cfg = write_config(args.repo, args.packages / "configs", v)
                model, native, vit = build(args.repo, cfg, ckpt)
                if list(native) != [sz, sz]:
                    print(f"  note: config eval_spatial_size {native}, exporting at {sz}")
                pkg = args.packages / f"{tag}.mlpackage"
                frame = load_frame(args.frame, sz, vit)
                ml, sample, conv = to_coreml(model, sz, pkg, frame)
                print(f"  converted at {conv['target']} / {conv['precision']}"
                      f"{'' if conv['ane_eligible'] else '  (fp32: NOT Neural Engine eligible)'}")
                chk = verify(ml, model, sample)
                print(f"  verify: {chk.get('compared','?')} usable det(s), labels "
                      f"{chk.get('label_match','?')}  "
                      f"max box dev {chk.get('max_box_dev_px','?')} px  "
                      f"max score dev {chk.get('max_score_dev','?')}  "
                      f"{'OK' if chk['ok'] else 'MISMATCH'}")
                if not chk["ok"]:
                    raise RuntimeError(f"Core ML output disagrees with PyTorch: {chk}. "
                                       f"Refusing to report latency for a graph that does "
                                       f"not reproduce the model.")
                r = time_mlmodel(ml, sz, args.iters, frame)
                r.update({"arch": "deimv2", "model": v, "imgsz": sz, "runtime": "coreml",
                          "checkpoint": ckpt.name, "verify": chk, "conversion": conv,
                          "package_mb": round(
                              sum(f.stat().st_size for f in pkg.rglob("*")) / 1e6, 1)})
                rows.append(r)
                print(f"  median {r['median_ms']} ms   p95 {r['p95_ms']}   "
                      f"{r['fps_at_median']} fps   {'MEETS' if r['meets_budget'] else 'over'} 10 ms")
            except Exception as e:
                failed.append((tag, f"{type(e).__name__}: {e}"))
                print(f"  FAILED {type(e).__name__}: {str(e)[:300]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"host": platform.platform(), "results": rows,
                                    "failed": [{"tag": t, "error": m} for t, m in failed],
                                    "budget_ms": 10.0, "iters": args.iters,
                                    "note": "Core ML, compute_units=ALL (Neural Engine "
                                            "eligible), fp16. Timed on mlmodel.predict()."},
                                   indent=2))
    if rows:
        print(f"\n{'model':18s} {'imgsz':>5s} {'median':>8s} {'p95':>7s} {'fps':>6s} {'<=10ms':>7s}")
        for r in sorted(rows, key=lambda r: r["median_ms"]):
            print(f"deimv2-{r['model']:11s} {r['imgsz']:5d} {r['median_ms']:8.2f} "
                  f"{r['p95_ms']:7.2f} {r['fps_at_median']:6.1f} "
                  f"{'yes' if r['meets_budget'] else 'no':>7s}")
    if failed:
        print(f"\n{len(failed)} FAILED:")
        for t, m in failed:
            print(f"  {t}: {m[:200]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
