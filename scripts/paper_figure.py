#!/usr/bin/env python3
"""The publication figure: confusion matrix + ROC for the deployed model, side by side.

WHY ONE FIGURE AND NOT TWO. A confusion matrix at a single operating threshold and a
threshold-free ROC answer different halves of the same question, and a reader needs both to
judge a detector. The matrix says what the model does at the point it would actually ship;
the ROC says how much of that is the threshold's doing. Printed apart they get compared
across a page turn.

PANEL A -- 4x4 confusion matrix, counts with row-normalised percentages beneath. Rows are
ground truth, columns prediction, plus a background row and column:
    background COLUMN = a ground-truth hand with no detection (a miss)
    background ROW    = a detection matching no hand (a false alarm)
The background/background cell is meaningless and is left blank rather than printed as 0.

PANEL B -- one-vs-rest ROC per class, computed only over hands localised at IoU >= 0.5, so it
measures classification GIVEN detection. Read it next to the background column, which is
where the missed hands are.

Colours are shared between panels: each class keeps its hue in the ROC and its diagonal cell
in the matrix, so the eye can move between them without a legend lookup.

    python3 scripts/paper_figure.py --model yolo26_hbb-s-320
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASSES = ["thumbout", "openhand", "closedhand"]
LABELS = CLASSES + ["background"]
CLASS_COLOUR = {"thumbout": "#c1435a", "openhand": "#2a8c7f", "closedhand": "#3d5a80"}


def panel_matrix(ax, cm, title):
    n = len(CLASSES)
    rown = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(rown == 0, 1, rown))

    # Greyscale ground so the diagonal can carry class colour without fighting a colormap.
    ax.imshow(np.zeros_like(norm), cmap="Greys", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == n and j == n:
                continue
            a = norm[i, j]
            if i == j and i < n:
                base = CLASS_COLOUR[CLASSES[i]]
            elif j == n:
                base = "#c9a227"          # miss
            elif i == n:
                base = "#8d97a3"          # false alarm
            else:
                base = "#c1435a"          # wrong class
            ax.add_patch(__import__("matplotlib").patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, facecolor=base, alpha=min(0.12 + 0.88 * a, 1.0),
                edgecolor="white", lw=1.5, zorder=1))
            txt = f"{cm[i, j]}"
            sub = f"{a*100:.0f}%"
            ax.text(j, i - 0.10, txt, ha="center", va="center", zorder=2,
                    fontsize=13, fontweight="bold",
                    color="white" if a > 0.45 else "#1a1a1a")
            ax.text(j, i + 0.24, sub, ha="center", va="center", zorder=2,
                    fontsize=8.5, color="white" if a > 0.45 else "#666")
    ax.text(n, n, "—", ha="center", va="center", color="#bbb", fontsize=12, zorder=2)

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=28, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=9.5)
    ax.set_xlabel("predicted", fontsize=10.5)
    ax.set_ylabel("ground truth", fontsize=10.5)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlim(-0.5, len(LABELS) - 0.5)
    ax.set_ylim(len(LABELS) - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def panel_roc(ax, roc, n_scored, n_gt):
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), lw=1, color="#c8c8c8", zorder=1)
    for name in CLASSES:
        c = roc.get("curves", {}).get(name)
        a = roc.get("per_class", {}).get(name)
        if not c:
            continue
        ax.plot(c["fpr"], c["tpr"], lw=2.2, color=CLASS_COLOUR[name], zorder=3,
                solid_capstyle="round",
                label=f"{name}   {a:.3f}" if a is not None else name)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate", fontsize=10.5)
    ax.set_ylabel("true positive rate", fontsize=10.5)
    ax.set_title(f"One-vs-rest ROC  ({n_scored}/{n_gt} hands localised at IoU ≥ 0.5)",
                 fontsize=11, pad=10)
    leg = ax.legend(loc="lower right", fontsize=9, frameon=False,
                    title="AUC", title_fontsize=9)
    leg._legend_box.align = "left"
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.12, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26_hbb-s-320")
    ap.add_argument("--pooled", type=Path, default=ROOT / "results" / "pooled")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default=None, help="title text; defaults to the model name")
    args = ap.parse_args()

    d = json.loads((args.pooled / f"{args.model}.json").read_text())
    cm = np.array(d["confusion_matrix"])
    label = args.label or f"{d['arch']} · {d['model']}"
    out = args.out or ROOT / "figures" / f"paper_{args.model}.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    panel_matrix(axes[0], cm,
                 f"Confusion matrix  (n={d['n_gt']}, conf ≥ {d['operating_threshold']})")
    panel_roc(axes[1], d["auc"], d["n_scored"], d["n_gt"])
    fig.suptitle(f"{label} — 3-fold cross-validation, pooled",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.008,
             "Each fold checkpoint scored on its own held-out fold; the test set is not used "
             "here. Background column = missed hands, background row = false alarms.",
             ha="center", fontsize=8.2, color="#666")
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))       # vector, for the paper
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
