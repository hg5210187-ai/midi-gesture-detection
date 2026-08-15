#!/usr/bin/env python3
"""Confusion matrix, ROC/AUC and per-class P/R/F1 for a trained detector.

None of this exists in ultralytics' reported metrics, which give mAP, dataset-level precision
and recall, and nothing else. The three additions here:

1. A 4x4 CONFUSION MATRIX -- 3 classes plus background, in both directions:
       rows    ground truth  {thumbout, openhand, closedhand, background}
       columns prediction    {thumbout, openhand, closedhand, background}
   A ground-truth hand with no matching detection lands in the background COLUMN (a miss).
   A detection matching no ground-truth hand lands in the background ROW (a false alarm).
   The background/background cell is meaningless and always 0.

2. Per-hand one-vs-rest ROC and AUC. See the honest caveat below.

3. Per-class precision, recall and F1 at the operating threshold.

MATCHING. Greedy by descending confidence: each detection takes the highest-IoU unclaimed
ground-truth box above IOU_THRESH. Greedy rather than Hungarian because with two hands per
image they almost never compete, and greedy is what the detection literature uses.

IoU IS COMPUTED EXACTLY, including for oriented boxes -- rotated-polygon intersection via
cv2.intersectConvexConvex, not the Gaussian ProbIoU surrogate ultralytics uses internally.
ProbIoU is optimistic and disagrees with true IoU near the threshold, so an OBB score computed
with it is not on the same scale as an axis-aligned score computed with box IoU. Since this
study compares the two geometries directly, both must use the same definition.

WHAT THE AUC DOES AND DOES NOT MEASURE. It is computed only over ground-truth hands the
detector already found and localised at IoU >= 0.5. A model that finds 20 of 30 hands and
labels all 20 correctly scores 1.00. Read it next to n_scored/n_gt and next to the background
column of the confusion matrix, which is where the missed hands are counted.

Ultralytics returns one confidence and one class per box, not a full class distribution. The
per-class score for a ground-truth hand is therefore the highest-confidence detection of that
class overlapping it at IoU >= 0.5, and 0 where no such detection exists. That is what the
detector actually emits after NMS; it is not a softmax over classes, and the scores do not sum
to 1, so one-vs-rest is the correct reading.

    python3 scripts/metrics.py --weights models/best.pt --data data/yolo/hbb/fold0.yaml \
                               --split val --out results/metrics_fold0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["thumbout", "openhand", "closedhand"]
BACKGROUND = "background"
IOU_THRESH = 0.50
CONF_THRESH = 0.25          # operating point for the confusion matrix and P/R/F1
SCORE_FLOOR = 0.001         # detections below this are not collected at all


def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Exact IoU between two convex quadrilaterals given as (4,2) corner arrays."""
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    if inter <= 0:
        return 0.0
    area_a = cv2.contourArea(a32)
    area_b = cv2.contourArea(b32)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_iou(a, b) -> float:
    """IoU between two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def load_ground_truth(label_path: Path, w: int, h: int, obb: bool):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().split("\n"):
        t = line.split()
        if not t:
            continue
        cls = int(t[0])
        v = [float(x) for x in t[1:]]
        if obb:
            pts = np.array([[v[i] * w, v[i + 1] * h] for i in range(0, 8, 2)])
            out.append({"cls": cls, "poly": pts,
                        "box": [pts[:, 0].min(), pts[:, 1].min(),
                                pts[:, 0].max(), pts[:, 1].max()]})
        else:
            cx, cy, bw, bh = v[0] * w, v[1] * h, v[2] * w, v[3] * h
            box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            out.append({"cls": cls, "box": box,
                        "poly": np.array([[box[0], box[1]], [box[2], box[1]],
                                          [box[2], box[3]], [box[0], box[3]]])})
    return out


def predictions(result, obb: bool):
    out = []
    src = getattr(result, "obb", None) if obb else result.boxes
    if src is None or len(src) == 0:
        return out
    confs = src.conf.cpu().numpy()
    clss = src.cls.cpu().numpy().astype(int)
    if obb:
        polys = src.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        for p, c, k in zip(polys, confs, clss):
            out.append({"cls": int(k), "conf": float(c), "poly": p,
                        "box": [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]})
    else:
        for b, c, k in zip(src.xyxy.cpu().numpy(), confs, clss):
            out.append({"cls": int(k), "conf": float(c), "box": b.tolist(),
                        "poly": np.array([[b[0], b[1]], [b[2], b[1]],
                                          [b[2], b[3]], [b[0], b[3]]])})
    return sorted(out, key=lambda d: -d["conf"])


def iou_of(a, b, obb: bool) -> float:
    return poly_iou(a["poly"], b["poly"]) if obb else box_iou(a["box"], b["box"])


def evaluate(weights: Path, data_yaml: Path, split: str, obb: bool, imgsz: int,
             device: str, conf_thresh: float):
    from ultralytics import YOLO
    import yaml

    cfg = yaml.safe_load(Path(data_yaml).read_text())
    listing = Path(cfg[split])
    images = [Path(p) for p in listing.read_text().splitlines() if p.strip()]

    model = YOLO(str(weights))
    n = len(CLASS_NAMES)
    cm = np.zeros((n + 1, n + 1), dtype=int)          # rows GT, cols predicted
    y_true, y_score, n_gt, n_scored = [], [], 0, 0

    for img_path in images:
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        gts = load_ground_truth(label_path, w, h, obb)
        n_gt += len(gts)

        res = model.predict(str(img_path), imgsz=imgsz, device=device,
                            conf=SCORE_FLOOR, verbose=False)[0]
        preds = predictions(res, obb)

        # --- confusion matrix at the operating threshold ---
        kept = [p for p in preds if p["conf"] >= conf_thresh]
        claimed = set()
        for p in kept:
            best, best_iou = None, IOU_THRESH
            for i, g in enumerate(gts):
                if i in claimed:
                    continue
                v = iou_of(p, g, obb)
                if v >= best_iou:
                    best, best_iou = i, v
            if best is None:
                cm[n, p["cls"]] += 1                  # false alarm -> background row
            else:
                claimed.add(best)
                cm[gts[best]["cls"], p["cls"]] += 1
        for i, g in enumerate(gts):
            if i not in claimed:
                cm[g["cls"], n] += 1                  # missed -> background column

        # --- per-hand class scores, over all detections regardless of threshold ---
        for g in gts:
            scores = [0.0] * n
            hit = False
            for p in preds:
                if iou_of(p, g, obb) >= IOU_THRESH:
                    scores[p["cls"]] = max(scores[p["cls"]], p["conf"])
                    hit = True
            if hit:
                n_scored += 1
                y_true.append(g["cls"])
                y_score.append(scores)

    return cm, np.array(y_true), np.array(y_score), n_gt, n_scored


def roc_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score, roc_curve
    out = {"per_class": {}, "curves": {}}
    if len(y_true) == 0:
        return out
    aucs = []
    for i, name in enumerate(CLASS_NAMES):
        pos = (y_true == i).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            out["per_class"][name] = None            # AUC undefined with one class present
            continue
        a = float(roc_auc_score(pos, y_score[:, i]))
        fpr, tpr, _ = roc_curve(pos, y_score[:, i])
        out["per_class"][name] = round(a, 6)
        out["curves"][name] = {"fpr": [round(float(x), 5) for x in fpr],
                               "tpr": [round(float(x), 5) for x in tpr]}
        aucs.append(a)
    out["macro"] = round(float(np.mean(aucs)), 6) if aucs else None
    return out


def prf(cm):
    n = len(CLASS_NAMES)
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": round(p, 6), "recall": round(r, 6),
                     "f1": round(f, 6), "tp": tp, "fp": fp, "fn": fn}
    return out


def render(cm, title, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CLASS_NAMES + [BACKGROUND]
    rown = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(rown == 0, 1, rown))

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == len(CLASS_NAMES) and j == len(CLASS_NAMES):
                ax.text(j, i, "—", ha="center", va="center", color="#999")
                continue
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 0.55 else "#222")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=CONF_THRESH)
    ap.add_argument("--obb", action="store_true", help="oriented boxes (exact polygon IoU)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--figure", type=Path, default=None)
    args = ap.parse_args()

    cm, y_true, y_score, n_gt, n_scored = evaluate(
        args.weights, args.data, args.split, args.obb, args.imgsz, args.device, args.conf)

    labels = CLASS_NAMES + [BACKGROUND]
    print(f"confusion matrix   rows=truth  cols=predicted   "
          f"(IoU>={IOU_THRESH}, conf>={args.conf})\n")
    print(" " * 13 + "".join(f"{l[:9]:>11s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>12s} " + "".join(f"{cm[i, j]:>11d}" for j in range(len(labels))))

    per = prf(cm)
    print(f"\n{'class':12s} {'precision':>10s} {'recall':>8s} {'f1':>7s} "
          f"{'tp':>4s} {'fp':>4s} {'fn':>4s}")
    for name, d in per.items():
        print(f"{name:12s} {d['precision']:10.4f} {d['recall']:8.4f} {d['f1']:7.4f} "
              f"{d['tp']:4d} {d['fp']:4d} {d['fn']:4d}")

    roc = roc_auc(y_true, y_score)
    print(f"\nper-hand one-vs-rest AUC   (scored {n_scored}/{n_gt} ground-truth hands)")
    for name in CLASS_NAMES:
        v = roc["per_class"].get(name)
        print(f"  {name:12s} {'n/a' if v is None else f'{v:.4f}'}")
    if roc.get("macro") is not None:
        print(f"  {'macro':12s} {roc['macro']:.4f}")
    if n_scored < n_gt:
        print(f"  NOTE: {n_gt - n_scored} hand(s) had no detection at IoU>={IOU_THRESH} and are "
              f"absent from the AUC. They appear in the background column above.")

    payload = {
        "weights": str(args.weights), "data": str(args.data), "split": args.split,
        "imgsz": args.imgsz, "conf": args.conf, "iou": IOU_THRESH, "obb": args.obb,
        "labels": labels, "confusion_matrix": cm.tolist(),
        "per_class": per, "auc": roc, "n_gt": n_gt, "n_scored": n_scored,
        "caveat": ("AUC is computed only over ground-truth hands localised at IoU>=0.5; it "
                   "measures classification given detection, not detection. Per-class scores "
                   "are post-NMS detection confidences, not a softmax, so they do not sum to 1."),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    if args.figure:
        render(cm, f"{args.weights.stem} · {args.split}", args.figure)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
