#!/usr/bin/env python3
"""Pool the 3 cross-validation folds into ONE confusion matrix and ROC per model.

The brief asks for a confusion matrix for every model. A per-fold matrix is built from 30
annotations, which is too few to read -- a single misclassification moves a cell by 3%. So
each model's three fold checkpoints are each scored on the fold they were held out from, and
the results are pooled: 90 annotations per matrix, every one of them predicted by a model
that never trained on it. The 30-annotation test set is not touched here at all.

This is deliberately NOT a re-implementation. Matching, exact polygon IoU, one-vs-rest AUC,
P/R/F1 and the figure all come from scripts/metrics.py; this script only decides which
detections to feed them and how to group the results.

INPUT is the dump format written by dump_preds_yolo.py and dump_preds_deim.py, so the two
architectures reach this point on equal terms -- same IoU definition, same matching rule.

THE OPERATING THRESHOLD IS NOT SHARED, AND THAT IS THE POINT. A confusion matrix needs one,
and 0.25 is the ultralytics convention -- but YOLO26 applies NMS and DEIMv2 does not. DEIMv2
emits 300 unsuppressed queries per image, so at 0.25 several of them survive on the same
hand; the first is matched and the rest are counted as false alarms. Measured here, that is
10-25 false alarms per model for YOLO26 against 44-150 for DEIMv2, and it drags deimv2_m to
macro-F1 0.46 despite an AP50-95 of 0.829. Reporting that as a property of the model would
be wrong: it is a property of the threshold.

So each model is reported at ITS OWN macro-F1-optimal threshold, swept over 0.05..0.95, with
the fixed-0.25 matrix kept alongside for reference. The caveat that comes with it: the
threshold is chosen on the same predictions being scored, so it is an upper bound on what a
held-out threshold would give. It is stated per model in the output. AUC is unaffected --
it is threshold-free, which is exactly why it is the metric to lead with when comparing the
two architectures.

    python3 scripts/pool_metrics.py --preds results/preds_yolo results/preds_deim
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import (CLASS_NAMES, BACKGROUND, IOU_THRESH, CONF_THRESH,  # noqa: E402
                     box_iou, poly_iou, load_ground_truth, prf, render, roc_auc)

N = len(CLASS_NAMES)


def render_roc(roc: dict, title: str, path: Path):
    """One-vs-rest ROC per class. Curves come straight from the AUC computation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#bbb", zorder=1)
    colours = {"thumbout": "#d1495b", "openhand": "#2a9d8f", "closedhand": "#3d5a80"}
    for name in CLASS_NAMES:
        c = roc.get("curves", {}).get(name)
        a = roc.get("per_class", {}).get(name)
        if not c:
            continue
        ax.plot(c["fpr"], c["tpr"], lw=2, color=colours[name], zorder=2,
                label=f"{name}  AUC {a:.3f}" if a is not None else name)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def image_sizes(pool: Path, geom: str) -> dict:
    """(w, h) per photo id, read from the JPEG header rather than decoding the pixels."""
    from PIL import Image
    out = {}
    for p in sorted((pool / geom / "images").glob("*.jpg")):
        with Image.open(p) as im:
            out[p.stem] = im.size
    return out


def as_shape(det: dict, obb: bool) -> dict:
    """Normalise a dumped detection into the {box, poly} pair the IoU helpers expect."""
    if obb:
        poly = np.array(det["poly"], dtype=float)
        return {"cls": det["cls"], "conf": det["conf"], "poly": poly,
                "box": [poly[:, 0].min(), poly[:, 1].min(),
                        poly[:, 0].max(), poly[:, 1].max()]}
    b = det["box"]
    return {"cls": det["cls"], "conf": det["conf"], "box": b,
            "poly": np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])}


def iou_of(a, b, obb: bool) -> float:
    return poly_iou(a["poly"], b["poly"]) if obb else box_iou(a["box"], b["box"])


def accumulate(dumps: list[dict], pool: Path, thresholds: list[float]):
    """Fold one model's dumps into a confusion matrix PER threshold, plus the AUC inputs.

    The IoU between every detection and every ground-truth hand does not depend on the
    threshold, so it is computed once and reused across the sweep. Matching does depend on
    it -- a detection removed by a higher threshold frees the hand it had claimed -- so the
    greedy match is replayed for each one.
    """
    geom = dumps[0]["geom"]
    obb = geom == "obb"
    sizes = image_sizes(pool, geom)
    labels_dir = pool / geom / "labels"

    cms = {t: np.zeros((N + 1, N + 1), dtype=int) for t in thresholds}
    y_true, y_score = [], []
    n_gt = n_scored = n_img = 0

    for d in dumps:
        for pid, dets in d["preds"].items():
            if pid not in sizes:
                raise SystemExit(f"{pid} in dump but not in {pool/geom/'images'}")
            w, h = sizes[pid]
            gts = load_ground_truth(labels_dir / f"{pid}.txt", w, h, obb)
            if not gts:
                raise SystemExit(f"no ground truth for {pid} in {labels_dir}")
            n_gt += len(gts)
            n_img += 1
            preds = sorted((as_shape(x, obb) for x in dets), key=lambda p: -p["conf"])
            ious = [[iou_of(p, g, obb) for g in gts] for p in preds]

            for t in thresholds:
                cm = cms[t]
                claimed = set()
                for pi, p in enumerate(preds):
                    if p["conf"] < t:
                        continue                     # sorted by conf, but keep it explicit
                    best, best_iou = None, IOU_THRESH
                    for gi in range(len(gts)):
                        if gi in claimed:
                            continue
                        if ious[pi][gi] >= best_iou:
                            best, best_iou = gi, ious[pi][gi]
                    if best is None:
                        cm[N, p["cls"]] += 1         # false alarm -> background row
                    else:
                        claimed.add(best)
                        cm[gts[best]["cls"], p["cls"]] += 1
                for gi, g in enumerate(gts):
                    if gi not in claimed:
                        cm[g["cls"], N] += 1         # missed -> background column

            # --- per-hand class scores, every detection regardless of threshold ---
            for gi, g in enumerate(gts):
                scores = [0.0] * N
                hit = False
                for pi, p in enumerate(preds):
                    if ious[pi][gi] >= IOU_THRESH:
                        scores[p["cls"]] = max(scores[p["cls"]], p["conf"])
                        hit = True
                if hit:
                    n_scored += 1
                    y_true.append(g["cls"])
                    y_score.append(scores)

    return cms, np.array(y_true), np.array(y_score), n_gt, n_scored, n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    ap.add_argument("--pool", type=Path, default=Path("data/pool"))
    ap.add_argument("--out", type=Path, default=Path("results/pooled"))
    ap.add_argument("--figures", type=Path, default=Path("figures/confusion"))
    ap.add_argument("--roc", type=Path, default=Path("figures/roc"))
    ap.add_argument("--conf", type=float, default=CONF_THRESH)
    args = ap.parse_args()

    files = [f for d in args.preds for f in sorted(Path(d).glob("*.json"))]
    if not files:
        raise SystemExit(f"no dumps found in {args.preds}")

    groups = defaultdict(list)
    for f in files:
        d = json.loads(f.read_text())
        groups[(d["arch"], d["model"])].append(d)

    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.roc.mkdir(parents=True, exist_ok=True)
    sweep = [round(0.05 * i, 2) for i in range(1, 20)]        # 0.05 .. 0.95
    if args.conf not in sweep:
        sweep.append(args.conf)
    sweep.sort()

    def macro_f1(cm):
        p = prf(cm)
        return float(np.mean([p[c]["f1"] for c in CLASS_NAMES]))

    rows = []
    for (arch, model), dumps in sorted(groups.items()):
        folds = sorted(d["fold"] for d in dumps)
        if folds != ["fold0", "fold1", "fold2"]:
            print(f"  SKIP {arch}/{model}: folds {folds}, need all three to pool")
            continue
        cms, y_true, y_score, n_gt, n_scored, n_img = accumulate(dumps, args.pool, sweep)
        roc = roc_auc(y_true, y_score)

        best_t = max(sweep, key=lambda t: (macro_f1(cms[t]),
                                           -int(cms[t][N, :N].sum())))   # ties -> fewer FAs
        cm, cm_fixed = cms[best_t], cms[args.conf]
        per, per_fixed = prf(cm), prf(cm_fixed)
        acc = int(np.trace(cm[:N, :N])) / n_gt if n_gt else 0.0

        tag = f"{arch}_{model}"
        payload = {
            "arch": arch, "model": model, "folds": folds,
            "pooled_over": "each fold checkpoint scored on its own held-out fold",
            "test_set_used": False,
            "geom": dumps[0]["geom"], "imgsz": dumps[0]["imgsz"], "iou": IOU_THRESH,
            "labels": CLASS_NAMES + [BACKGROUND],
            "operating_threshold": best_t,
            "confusion_matrix": cm.tolist(), "per_class": per,
            "accuracy_over_all_gt": round(acc, 6),
            "fixed_threshold": args.conf,
            "confusion_matrix_at_fixed": cm_fixed.tolist(), "per_class_at_fixed": per_fixed,
            "threshold_sweep": [{"conf": t, "macro_f1": round(macro_f1(cms[t]), 6),
                                 "miss": int(cms[t][:N, N].sum()),
                                 "false_alarm": int(cms[t][N, :N].sum())} for t in sweep],
            "auc": roc, "n_images": n_img, "n_gt": n_gt, "n_scored": n_scored,
            "caveats": [
                f"The operating threshold {best_t} maximises macro-F1 on these same "
                f"predictions, so per-class P/R/F1 are an upper bound; a threshold picked on "
                f"held-out data would score no higher. AUC does not depend on it.",
                f"AUC covers only ground-truth hands localised at IoU>={IOU_THRESH} "
                f"({n_scored}/{n_gt}); it measures classification given detection. Missed "
                f"hands appear in the background column, not the AUC.",
            ],
        }
        (args.out / f"{tag}.json").write_text(json.dumps(payload, indent=2))
        render(cm, f"{arch} · {model} · 3-fold pooled (n={n_gt}, conf≥{best_t})",
               args.figures / f"{tag}.png")
        render_roc(roc, f"{arch} · {model} · one-vs-rest ROC (n={n_scored}/{n_gt} hands)",
                   args.roc / f"{tag}.png")

        rows.append({"arch": arch, "model": model, "auc": roc.get("macro"),
                     "acc": acc, "n_gt": n_gt, "n_scored": n_scored, "conf": best_t,
                     "miss": int(cm[:N, N].sum()), "fa": int(cm[N, :N].sum()),
                     "f1": macro_f1(cm), "f1_at_fixed": macro_f1(cm_fixed),
                     "fa_at_fixed": int(cm_fixed[N, :N].sum())})
        macro = roc.get("macro")
        macro_s = "n/a" if macro is None else f"{macro:.4f}"
        print(f"  {tag:28s} conf {best_t:.2f}  acc {acc:.4f}  macroAUC {macro_s}  "
              f"miss {int(cm[:N, N].sum()):3d}  false-alarm {int(cm[N, :N].sum()):3d}")

    rows.sort(key=lambda r: (-(r["auc"] or 0), -r["f1"]))
    print(f"\n{'model':26s} {'conf':>5s} {'acc':>7s} {'macroAUC':>9s} {'macroF1':>8s} "
          f"{'miss':>5s} {'FA':>4s} {'scored':>8s}  {'F1@0.25':>8s} {'FA@0.25':>8s}")
    print("-" * 102)
    for r in rows:
        print(f"{r['arch']+'/'+r['model']:26s} {r['conf']:5.2f} {r['acc']:7.4f} "
              f"{(r['auc'] if r['auc'] is not None else float('nan')):9.4f} {r['f1']:8.4f} "
              f"{r['miss']:5d} {r['fa']:4d} {r['n_scored']:4d}/{r['n_gt']:<4d}  "
              f"{r['f1_at_fixed']:8.4f} {r['fa_at_fixed']:8d}")

    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nper-model json -> {args.out}\nfigures        -> {args.figures}")


if __name__ == "__main__":
    main()
