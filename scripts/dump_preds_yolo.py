#!/usr/bin/env python3
"""Dump YOLO26 detections to the same JSON format as dump_preds_deim.py.

Runs on the Mac against the 30 checkpoints in midi_results/weights/, named
"<geom>-<size>-fold<k>-<imgsz>.pt". Each is scored on ITS OWN held-out fold, which is the
only split it never saw; pooling the three covers all 90 annotations while leaving the test
set untouched. scripts/pool_metrics.py does the pooling for both arms.

OBB detections keep their four corners. Downstream IoU is computed on the polygon with
cv2.intersectConvexConvex, never with ultralytics' ProbIoU surrogate -- which is optimistic
by roughly +0.11 and would put the OBB arm on a different scale from the HBB arm in a study
whose whole point is comparing them.

    python3 dump_preds_yolo.py --weights midi_results/weights --out results/preds_yolo
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CLASS_NAMES = ["thumbout", "openhand", "closedhand"]
SCORE_FLOOR = 0.001
FOLDS = ["fold0", "fold1", "fold2"]
NAME_RE = re.compile(r"^(hbb|obb)-([nsmlx])-fold([012])-(\d+)$")


def fold_images(pool: Path, folds_json: Path, manifest: Path, geom: str):
    """photo id -> group, using the same source of truth the training splits came from."""
    import csv
    groups = json.loads(folds_json.read_text())["photo_group"]
    ids = {}
    with manifest.open() as fh:
        for r in csv.DictReader(fh):
            ids[r["original"].replace(".jpg", "")] = r["id"]
    out = {g: [] for g in FOLDS + ["test"]}
    for stem, grp in groups.items():
        out[grp].append(pool / geom / "images" / f"{ids[stem]}.jpg")
    return {g: sorted(v) for g, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("midi_results/weights"))
    ap.add_argument("--pool", type=Path, default=Path("data/pool"))
    ap.add_argument("--folds", type=Path, default=Path("data/splits/folds.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/pool/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/preds_yolo"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--group", default=None,
                    help="image group to score; default = the checkpoint's own held-out fold")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    images = {g: fold_images(args.pool, args.folds, args.manifest, g) for g in ("hbb", "obb")}
    ckpts = sorted(args.weights.glob("*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.weights}")

    done, skipped = 0, []
    for ck in ckpts:
        m = NAME_RE.match(ck.stem)
        if not m:
            skipped.append((ck.name, "filename does not parse"))
            continue
        geom, size, k, imgsz = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        fold = f"fold{k}"
        obb = geom == "obb"
        # Default: score a checkpoint on the fold it was held out from (cross-validation).
        # --group test points every checkpoint at the untouched test set instead.
        group = args.group or fold
        imgs = images[geom][group]

        model = YOLO(str(ck))
        preds = {}
        n_det = 0
        for p in imgs:
            r = model.predict(str(p), imgsz=imgsz, device=args.device,
                              conf=SCORE_FLOOR, verbose=False)[0]
            src = r.obb if obb else r.boxes
            dets = []
            if src is not None and len(src):
                confs = src.conf.cpu().numpy()
                clss = src.cls.cpu().numpy().astype(int)
                if obb:
                    polys = src.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
                    for poly, c, cl in zip(polys, confs, clss):
                        dets.append({"cls": int(cl), "conf": round(float(c), 6),
                                     "poly": [[round(float(x), 2), round(float(y), 2)]
                                              for x, y in poly]})
                else:
                    for b, c, cl in zip(src.xyxy.cpu().numpy(), confs, clss):
                        dets.append({"cls": int(cl), "conf": round(float(c), 6),
                                     "box": [round(float(v), 2) for v in b]})
            dets.sort(key=lambda d: -d["conf"])
            preds[p.stem] = dets
            n_det += len(dets)

        tag = f"yolo26_{geom}-{size}-{imgsz}_{fold}"
        (args.out / f"{tag}.json").write_text(json.dumps({
            "arch": "yolo26", "model": f"{geom}-{size}-{imgsz}", "fold": fold,
            "scored_on": group, "geom": geom,
            "imgsz": imgsz, "checkpoint": ck.name, "score_floor": SCORE_FLOOR,
            "class_names": CLASS_NAMES, "preds": preds}))
        print(f"  {tag:30s} {len(imgs):3d} imgs  {n_det:6d} dets")
        done += 1

    print(f"\n{done} dump(s) -> {args.out}")
    for n, why in skipped:
        print(f"  skipped {n}: {why}")


if __name__ == "__main__":
    main()
