#!/usr/bin/env python3
"""The aggregate confusion matrix: every model that learned the task, summed.

WHY AN AGGREGATE. Each per-model matrix in figures/confusion/ rests on 90 annotations, where
one mistake moves a cell by ~1%. Summing the models that actually learned the task shows
whether the error STRUCTURE is a property of the task or an accident of any one model. It is
the former, and that is the study's most robust finding.

WHAT "SUCCEEDED IN TRAINING" MEANS HERE: macro AUC >= 0.9. That admits 35 of the 38 models and
excludes exactly the three DEIMv2 variants below the capacity floor (atto, femto, pico), whose
AUC sits at chance and whose confusions would be noise rather than signal.

READ WITH CARE -- the 35 models are NOT independent trials. They are 35 models scored on the
SAME 90 annotations, so a hand that is intrinsically ambiguous is counted up to 35 times. The
matrix therefore shows the SHAPE of the error reliably and its MAGNITUDE only in the sense of
"how often, across configurations". It is not 3,150 independent observations.

Colour encodes the KIND of error, and every cell carries its count, so colour is never the
sole cue -- which is what keeps it readable in greyscale and for colour-blind readers.

    python3 scripts/aggregate_figure.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["thumbout", "openhand", "closedhand"]
LABELS = CLASSES + ["background"]
AUC_FLOOR = 0.9
CLASS_COLOUR = {"thumbout": "#c1435a", "openhand": "#2a8c7f", "closedhand": "#3d5a80"}
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8984"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8984"),
}


def collect():
    tot = np.zeros((4, 4), int)
    models, per = [], {c: [] for c in CLASSES}
    for f in sorted(glob.glob(str(ROOT / "results/pooled/*.json"))):
        if "summary" in f:
            continue
        d = json.loads(Path(f).read_text())
        if (d["auc"].get("macro") or 0) < AUC_FLOOR:
            continue
        tot += np.array(d["confusion_matrix"])
        models.append(f"{d['arch']}/{d['model']}")
        for c in CLASSES:
            v = d["auc"]["per_class"].get(c)
            if v is not None:
                per[c].append(v)
    return tot, models, per


def render(mode: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    t = THEME[mode]
    cm, models, per = collect()
    n = len(CLASSES)
    rown = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(rown == 0, 1, rown))

    off = [(i, j, int(cm[i, j])) for i in range(n) for j in range(n) if i != j]
    total_conf = sum(c for _, _, c in off)
    pair = sum(c for i, j, c in off if {CLASSES[i], CLASSES[j]} == {"thumbout", "closedhand"})

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    ax.imshow(np.zeros_like(norm), cmap="Greys", vmin=0, vmax=1)

    for i in range(4):
        for j in range(4):
            if i == n and j == n:
                ax.text(j, i, "—", ha="center", va="center", color=t["ink3"], fontsize=13)
                continue
            a = norm[i, j]
            if i == j and i < n:
                base = CLASS_COLOUR[CLASSES[i]]
            elif j == n:
                base = "#c9a227"      # a ground-truth hand with no detection
            elif i == n:
                base = "#8d97a3"      # a detection matching no hand
            else:
                base = "#c1435a"      # wrong class
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, zorder=1,
                                   facecolor=base, alpha=min(.12 + .88 * a, 1.0),
                                   edgecolor=t["surface"], lw=2.2))
            ax.text(j, i - .09, f"{cm[i, j]:,}", ha="center", va="center", zorder=2,
                    fontsize=15, fontweight="bold",
                    color="white" if a > .45 else "#1a1a1a")
            ax.text(j, i + .25, f"{a*100:.1f}%", ha="center", va="center", zorder=2,
                    fontsize=9, color="white" if a > .45 else "#666")

    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=26, ha="right", fontsize=10.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(LABELS, fontsize=10.5)
    ax.set_xlabel("predicted", fontsize=11.5, color=t["ink2"])
    ax.set_ylabel("ground truth", fontsize=11.5, color=t["ink2"])
    ax.tick_params(length=0, colors=t["ink2"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-.5, 3.5); ax.set_ylim(3.5, -.5)

    fig.suptitle("Aggregate error structure — all 35 models that learned the task",
                 fontsize=13.5, fontweight="bold", color=t["ink"], y=0.975)
    ax.set_title(f"summed over {len(models)} models × 90 annotations   ·   "
                 f"macro AUC ≥ {AUC_FLOOR}   ·   percentages are row-normalised",
                 fontsize=9.5, color=t["ink3"], pad=12)

    aucs = "     ".join(f"{c} {np.mean(per[c]):.4f}" for c in CLASSES)
    fig.text(0.5, 0.105,
             f"{pair:,} of {total_conf:,} class confusions ({pair/total_conf*100:.1f}%) are "
             f"thumbout ↔ closedhand.  Only {total_conf-pair} involve openhand.",
             ha="center", fontsize=11, color=t["ink"], fontweight="bold")
    fig.text(0.5, 0.068, f"mean per-class AUC:     {aucs}",
             ha="center", fontsize=9.2, color=t["ink2"])
    fig.text(0.5, 0.026,
             "The 35 models are not independent trials — they are scored on the same 90 "
             "annotations, so an intrinsically ambiguous hand is counted repeatedly.\n"
             "The matrix shows the shape of the error reliably; read its magnitude as "
             "\"how often across configurations\", not as 3,150 independent observations.",
             ha="center", fontsize=8.2, color=t["ink3"])
    fig.tight_layout(rect=[0, 0.135, 1, 0.945])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    fig.savefig(out.with_suffix(".pdf"), facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out.name} + .pdf   ({len(models)} models, {pair}/{total_conf} pair)")


if __name__ == "__main__":
    for m in ("light", "dark"):
        render(m, ROOT / "figures" / f"aggregate_confusion_{m}.png")
