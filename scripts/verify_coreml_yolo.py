#!/usr/bin/env python3
"""Check that a YOLO26 Core ML package still reproduces its PyTorch checkpoint.

WHY THIS EXISTS. The DEIMv2 exporter verifies numerically before it will report a latency,
and that check earned its place: four DEIMv2 variants convert without error and then produce
boxes 213-277 px wrong. The YOLO arm never had the equivalent -- its exports were trusted
because ultralytics produced them without complaint. That is an asymmetry in the study's own
standard of evidence, so this closes it.

METHOD, matching scripts/export_coreml_deim.py's verify(): run both models on the same REAL
photograph, keep only detections above the operating threshold, and pair them BY POSITION
rather than by rank -- ranking is unstable under fp16 when two detections have close scores.
Reports max box deviation in pixels.

    python3 scripts/verify_coreml_yolo.py --weights models/hbb-l.pt \
        --package models/hbb-l_320.mlpackage --imgsz 320
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONF = 0.25
TOL_PX = 2.0


def dets(res, obb):
    src = res.obb if obb else res.boxes
    if src is None or len(src) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    conf = src.conf.cpu().numpy()
    cls = src.cls.cpu().numpy()
    if obb:
        poly = src.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        box = np.stack([poly[:, :, 0].min(1), poly[:, :, 1].min(1),
                        poly[:, :, 0].max(1), poly[:, :, 1].max(1)], axis=1)
    else:
        box = src.xyxy.cpu().numpy()
    return box, conf, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    obb = args.package.name.startswith("obb")
    task = "obb" if obb else "detect"
    if args.frame is None:
        c = sorted((ROOT / "data/pool/hbb/images").glob("*.jpg"))
        args.frame = c[len(c) // 2]

    pt = YOLO(str(args.weights))
    ml = YOLO(str(args.package), task=task)
    r_pt = pt.predict(str(args.frame), imgsz=args.imgsz, device="cpu", conf=CONF, verbose=False)[0]
    r_ml = ml.predict(str(args.frame), imgsz=args.imgsz, device="cpu", conf=CONF, verbose=False)[0]
    b_pt, c_pt, k_pt = dets(r_pt, obb)
    b_ml, c_ml, k_ml = dets(r_ml, obb)

    print(f"  frame              {args.frame.name}")
    print(f"  PyTorch  {args.weights.name}: {len(b_pt)} detection(s) above conf {CONF}")
    print(f"  Core ML  {args.package.name}: {len(b_ml)} detection(s) above conf {CONF}")

    if len(b_pt) == 0 or len(b_ml) == 0:
        print("  VERDICT: FAIL — one side produced no usable detections")
        raise SystemExit(1)

    devs, labs, scores = [], [], []
    for i in range(len(b_pt)):
        centre = b_pt[i][:2] + b_pt[i][2:]
        j = int(np.abs((b_ml[:, :2] + b_ml[:, 2:]) - centre).sum(axis=1).argmin())
        devs.append(float(np.abs(b_pt[i] - b_ml[j]).max()))
        labs.append(bool(k_pt[i] == k_ml[j]))
        scores.append(float(abs(c_pt[i] - c_ml[j])))

    ok = all(labs) and max(devs) <= TOL_PX and len(b_pt) == len(b_ml)
    rec = {"weights": args.weights.name, "package": args.package.name, "imgsz": args.imgsz,
           "frame": args.frame.name, "conf": CONF,
           "n_pytorch": int(len(b_pt)), "n_coreml": int(len(b_ml)),
           "label_match": f"{sum(labs)}/{len(labs)}",
           "max_box_dev_px": round(max(devs), 3),
           "max_score_dev": round(max(scores), 5),
           "tolerance_px": TOL_PX, "ok": ok}
    print(f"  labels matched     {rec['label_match']}")
    print(f"  max box deviation  {rec['max_box_dev_px']} px   (tolerance {TOL_PX} px)")
    print(f"  max score deviation {rec['max_score_dev']}")
    print(f"  VERDICT: {'PASS — package reproduces the checkpoint' if ok else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=2))
        print(f"  wrote {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
