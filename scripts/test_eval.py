#!/usr/bin/env python3
"""AP50-95 on the held-out test set, computed identically for every model.

WHY NOT JUST CALL ultralytics val(). Because it would make the headline table incomparable.
The two validators do not use the same geometry, which is verifiable in the installed source:

    ultralytics/models/yolo/obb/val.py:94      iou = batch_probiou(...)
    ultralytics/models/yolo/detect/val.py:317  iou = box_iou(...)

So oriented boxes are matched with ProbIoU, a Gaussian surrogate, while axis-aligned boxes get
real box IoU -- and neither is on DEIMv2's scale.

The SIZE of that discrepancy was measured on this dataset rather than assumed
(results/probiou_check.json, 30 cells at 640 px, ultralytics minus exact-IoU):

    hbb   -0.0005 ± 0.0279     control -- the two agree, so this evaluator is sound
    obb   +0.1777 ± 0.0255     ultralytics over-reports oriented boxes by ~0.18

That is large enough to reverse an OBB-vs-HBB conclusion, and it did: OBB cells looked like
the best in the study until they were rescored here.

This computes COCO-style AP from the prediction dumps with EXACT geometry for all three:
polygon intersection via cv2.intersectConvexConvex for OBB, box IoU otherwise. One
definition, one matching rule, one table.

WHAT IS REPORTED. AP averaged over IoU 0.50:0.05:0.95 with 101-point interpolation (the COCO
definition), per class and then averaged. Each model has three fold checkpoints; every one is
scored on the SAME untouched test set and reported as mean +- SD, which is the same shape as
the cross-validation numbers and directly comparable to them.

THE TEST SET IS NOT A SELECTION TOOL. Choose the model on cross-validation, then read its row
here. With 30 annotations one instance moves a class AP by 3.3 points, so differences under
about 0.10 between two models are not distinguishable -- the table shows whether the CV
ranking broadly held, not a re-ranking.

    python3 scripts/test_eval.py --preds results/preds_yolo_test results/preds_deim_test
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import CLASS_NAMES, box_iou, load_ground_truth, poly_iou  # noqa: E402

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
RECALL_POINTS = np.linspace(0, 1, 101)


def as_shape(det, obb):
    if obb:
        poly = np.array(det["poly"], dtype=float)
        return {"cls": det["cls"], "conf": det["conf"], "poly": poly,
                "box": [poly[:, 0].min(), poly[:, 1].min(),
                        poly[:, 0].max(), poly[:, 1].max()]}
    b = det["box"]
    return {"cls": det["cls"], "conf": det["conf"], "box": b,
            "poly": np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])}


def iou_of(a, b, obb):
    return poly_iou(a["poly"], b["poly"]) if obb else box_iou(a["box"], b["box"])


def average_precision(dets, gts, obb, thr):
    """COCO-style AP for ONE class at ONE IoU threshold.

    dets: [(conf, image_id, shape)] across all images. gts: {image_id: [shape]}.
    Detections are consumed in confidence order and each ground truth can be claimed once,
    which is what makes a duplicate detection a false positive rather than a second hit.
    """
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return None
    dets = sorted(dets, key=lambda d: -d[0])
    claimed = collections.defaultdict(set)
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for i, (_, img, d) in enumerate(dets):
        best, best_iou = None, thr
        for j, g in enumerate(gts.get(img, [])):
            if j in claimed[img]:
                continue
            v = iou_of(d, g, obb)
            if v >= best_iou:
                best, best_iou = j, v
        if best is None:
            fp[i] = 1
        else:
            claimed[img].add(best)
            tp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    # make precision monotonically decreasing, then sample at the 101 recall points
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


def evaluate(dump: dict, pool: Path):
    geom = dump["geom"]
    obb = geom == "obb"
    labels_dir = pool / geom / "labels"
    from PIL import Image

    per_class_dets = collections.defaultdict(list)
    per_class_gts = collections.defaultdict(lambda: collections.defaultdict(list))
    for pid, dets in dump["preds"].items():
        img = pool / geom / "images" / f"{pid}.jpg"
        with Image.open(img) as im:
            w, h = im.size
        for g in load_ground_truth(labels_dir / f"{pid}.txt", w, h, obb):
            per_class_gts[g["cls"]][pid].append(g)
        for d in dets:
            s = as_shape(d, obb)
            per_class_dets[s["cls"]].append((s["conf"], pid, s))

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        aps = [average_precision(per_class_dets.get(c, []), per_class_gts.get(c, {}), obb, t)
               for t in IOU_THRESHOLDS]
        aps = [a for a in aps if a is not None]
        if not aps:
            per_class[name] = None
            continue
        per_class[name] = {"ap50_95": round(float(np.mean(aps)), 6),
                           "ap50": round(float(aps[0]), 6)}
    vals = [v["ap50_95"] for v in per_class.values() if v]
    v50 = [v["ap50"] for v in per_class.values() if v]
    return {"per_class": per_class,
            "mAP50_95": round(float(np.mean(vals)), 6) if vals else None,
            "mAP50": round(float(np.mean(v50)), 6) if v50 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    ap.add_argument("--pool", type=Path, default=Path("data/pool"))
    ap.add_argument("--out", type=Path, default=Path("results/test_set.json"))
    args = ap.parse_args()

    files = [f for d in args.preds for f in sorted(Path(d).glob("*.json"))]
    if not files:
        raise SystemExit(f"no dumps in {args.preds}")

    groups = collections.defaultdict(list)
    for f in files:
        d = json.loads(f.read_text())
        on = d.get("scored_on", d.get("fold"))
        if on != "test":
            print(f"  skip {f.name}: scored_on={on}, not the test set")
            continue
        groups[(d["arch"], d["model"])].append(d)

    rows = []
    for (arch, model), dumps in sorted(groups.items()):
        per_fold = []
        for d in sorted(dumps, key=lambda d: d["fold"]):
            r = evaluate(d, args.pool)
            per_fold.append({"fold": d["fold"], **r})
        vals = [f["mAP50_95"] for f in per_fold if f["mAP50_95"] is not None]
        rows.append({
            "arch": arch, "model": model, "n_checkpoints": len(per_fold),
            "test_mAP50_95_mean": round(statistics.mean(vals), 6) if vals else None,
            "test_mAP50_95_sd": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
            "test_mAP50_mean": round(statistics.mean(
                [f["mAP50"] for f in per_fold if f["mAP50"] is not None]), 6) if vals else None,
            "per_fold": per_fold})
        print(f"  {arch}/{model:14s} test mAP50-95 "
              f"{rows[-1]['test_mAP50_95_mean']:.4f} ± {rows[-1]['test_mAP50_95_sd']:.4f}")

    rows.sort(key=lambda r: -(r["test_mAP50_95_mean"] or 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "split": "test", "n_annotations": 30,
        "iou_thresholds": IOU_THRESHOLDS,
        "geometry": "exact polygon IoU for OBB (cv2.intersectConvexConvex), box IoU otherwise "
                    "-- NOT ultralytics' ProbIoU, so OBB/HBB/DETR are on one scale",
        "protocol": "each model's three fold checkpoints scored on the same untouched test "
                    "set; reported as mean ± SD across checkpoints",
        "caveat": "30 annotations: one instance moves a class AP by ~3.3 points. Differences "
                  "under ~0.10 are not distinguishable. Select on cross-validation, not here.",
        "results": rows}, indent=2))
    print(f"\n{'model':26s} {'test mAP50-95':>16s}")
    print("-" * 46)
    for r in rows:
        print(f"{r['arch']+'/'+r['model']:26s} "
              f"{r['test_mAP50_95_mean']:.4f} ± {r['test_mAP50_95_sd']:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
