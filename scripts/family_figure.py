#!/usr/bin/env python3
"""YOLO26 vs DEIMv2 — the cross-family comparison figure.

WHY THIS IS SEPARATE FROM tradeoff.png. That figure compares HBB against OBB and the three
input resolutions, all WITHIN YOLO26; DEIMv2 does not appear in it at all. This one compares
the two architecture families, which is a different question with a different evidence base:
accuracy is fully measurable for both, latency is not.

PANEL A -- accuracy vs capacity. Both families, every variant, 3-fold CV mAP50-95 with the
fold SD as the error bar. Both read at 640 px so the comparison is matched: DEIMv2 is
architecturally fixed at 640 and cannot be run at another resolution. Log x-axis because the
parameter counts span 0.53 M to 58.8 M.

PANEL B -- accuracy vs measured latency on the MacBook Air M4, Core ML. YOLO26 contributes
all 30 cells. DEIMv2 contributes ONE point, and that asymmetry is the finding rather than an
omission: of its eight variants, four (the DINOv3 line) convert to Core ML without error and
then produce boxes 213-277 px wrong, so no latency is reported for them; three sit below the
capacity floor and were never exported; only `n` converts and reproduces PyTorch exactly.

Accuracy for BOTH families is exact-IoU, recomputed from the prediction dumps -- never
ultralytics' reported mAP, which scores OBB with ProbIoU and runs +0.175 optimistic on this
dataset (results/probiou_check.json).

    python3 scripts/family_figure.py
"""
from __future__ import annotations
import glob, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET_MS = 10.0

FAMILY = {
    "yolo26": {"light": "#2a78d6", "dark": "#3987e5", "label": "YOLO26"},
    "deimv2": {"light": "#eb6834", "dark": "#d95926", "label": "DEIMv2"},
}
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8984",
                  grid="#e6e5e0", band="#eef6ee"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8984",
                  grid="#2e2d2a", band="#1e2a20"),
}
YOLO_SIZES = ["n", "s", "m", "l", "x"]
DEIM_VARIANTS = ["atto", "femto", "pico", "n", "s", "m", "l", "x"]


def params():
    """Parameter count per variant, read from the trained checkpoints."""
    import torch, warnings
    warnings.filterwarnings("ignore")
    out = {"yolo26": {}, "deimv2": {}}
    for s in YOLO_SIZES:
        ck = torch.load(ROOT / f"midi_results/weights/hbb-{s}-fold0-640.pt",
                        map_location="cpu", weights_only=False)
        m = ck.get("ema") or ck.get("model")
        out["yolo26"][s] = sum(p.numel() for p in m.parameters()) / 1e6
    for v in DEIM_VARIANTS:
        g = glob.glob(str(ROOT / f"deim_results/weights/{v}_fold0__*.pth"))
        if not g:
            continue
        ck = torch.load(g[0], map_location="cpu", weights_only=False)
        sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
        out["deimv2"][v] = sum(t.numel() for t in sd.values()
                               if hasattr(t, "numel")) / 1e6
    return out


def accuracy():
    """3-fold CV mAP50-95, exact IoU, for both families."""
    yolo = {}
    per_cell = json.loads((ROOT / "results/cv_exact_iou.json").read_text())["per_cell"]
    for k, v in per_cell.items():
        g, s, z = k.split("-")
        if g == "hbb" and z == "640":
            yolo[s] = (v["mean"], v["sd"])
    by = {}
    for r in json.loads((ROOT / "results/deim_evaluator_check.json").read_text()):
        by.setdefault(r["model"], []).append(r["exact"])
    deim = {v: (statistics.mean(x), statistics.stdev(x) if len(x) > 1 else 0.0)
            for v, x in by.items()}
    return yolo, deim


def latency():
    """Core ML medians. YOLO: every cell. DEIMv2: only what verified numerically."""
    yolo = json.loads((ROOT / "results/tradeoff.json").read_text())["points"]
    deim, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results/latency_deim*.json"))):
        for r in json.loads(Path(f).read_text()).get("results", []):
            if r.get("verify", {}).get("ok") and r["model"] not in seen:
                seen.add(r["model"])
                deim.append({"model": r["model"], "ms": r["median_ms"], "imgsz": r["imgsz"]})
    return yolo, deim


def render(mode: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEME[mode]
    P = params()
    acc_y, acc_d = accuracy()
    lat_y, lat_d = latency()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.patch.set_facecolor(t["surface"])
    for ax in (axA, axB):
        ax.set_facecolor(t["surface"])
        ax.grid(alpha=0.30, lw=0.6, color=t["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(t["grid"])
        ax.tick_params(colors=t["ink2"], labelsize=9)

    # ---- Panel A: accuracy vs capacity, both families at 640 px ----
    for fam, order, acc in (("yolo26", YOLO_SIZES, acc_y), ("deimv2", DEIM_VARIANTS, acc_d)):
        c = FAMILY[fam][mode]
        # the two families nearly coincide in x around 10 M and 20-30 M, so one family's
        # labels go above its markers and the other's below -- otherwise they overprint
        dy = 14 if fam == "yolo26" else -20
        # Selective labels, not one per point. DEIMv2 gets all eight because its shape IS the
        # story (the cliff); YOLO26's line is flat and unremarkable, so only its endpoints are
        # named. Labelling every point in both families overprints where they converge.
        show = set(order) if fam == "deimv2" else {order[0], order[-1]}
        pts = [(P[fam][v], acc[v][0], acc[v][1], v) for v in order
               if v in acc and v in P[fam]]
        axA.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=2, color=c,
                 zorder=3, solid_capstyle="round", label=FAMILY[fam]["label"])
        axA.errorbar([p[0] for p in pts], [p[1] for p in pts],
                     yerr=[p[2] for p in pts], fmt="o", ms=8, color=c, capsize=3,
                     lw=1.4, zorder=4, markeredgecolor=t["surface"], markeredgewidth=1.6)
        for x, yv, _, v in pts:
            if v in show:
                axA.annotate(v, (x, yv), textcoords="offset points", xytext=(0, dy),
                             ha="center", fontsize=8.5, color=c)
    axA.set_xscale("log")
    axA.set_xlabel("parameters (M, log scale)", fontsize=10, color=t["ink2"])
    axA.set_ylabel("mAP50-95   (3-fold CV, exact IoU)", fontsize=10, color=t["ink2"])
    axA.set_title("A · Accuracy vs capacity — both at 640 px", fontsize=11.5,
                  color=t["ink"], pad=10, loc="left")
    axA.set_ylim(-0.05, 1.02)
    axA.legend(fontsize=9.5, frameon=False, loc="center right", labelcolor=t["ink2"])
    axA.annotate("capacity floor — below this,\nDEIMv2 does not learn the task",
                 xy=(2.4, 0.42), xytext=(0.60, 0.60), fontsize=8.2, color=t["ink3"],
                 ha="left", arrowprops=dict(arrowstyle="->", color=t["ink3"], lw=0.9,
                                            connectionstyle="arc3,rad=-0.2"))

    # ---- Panel B: accuracy vs measured M4 latency ----
    axB.axvspan(0, BUDGET_MS, color=t["band"], zorder=0)
    axB.axvline(BUDGET_MS, color="#2a8c7f", lw=1.3, ls="--", zorder=1)
    cy = FAMILY["yolo26"][mode]
    axB.scatter([p["ms"] for p in lat_y], [p["map"] for p in lat_y], s=42, color=cy,
                zorder=3, edgecolor=t["surface"], linewidth=1.4,
                label=f"YOLO26 — {len(lat_y)} configurations")
    cd = FAMILY["deimv2"][mode]
    for r in lat_d:
        m = acc_d.get(r["model"])
        if not m:
            continue
        axB.errorbar([r["ms"]], [m[0]], yerr=[m[1]], fmt="D", ms=10, color=cd, capsize=3,
                     lw=1.4, zorder=4, markeredgecolor=t["surface"], markeredgewidth=1.6,
                     label="DEIMv2 — the 1 variant that converts correctly")
        axB.annotate(f"deimv2-{r['model']}", (r["ms"], m[0]), textcoords="offset points",
                     xytext=(0, 15), ha="center", fontsize=8.5, color=t["ink3"])
    axB.set_xlabel("Core ML latency on MacBook Air M4 — median ms", fontsize=10, color=t["ink2"])
    axB.set_ylabel("mAP50-95   (3-fold CV, exact IoU)", fontsize=10, color=t["ink2"])
    axB.set_title("B · Accuracy vs measured latency", fontsize=11.5,
                  color=t["ink"], pad=10, loc="left")
    axB.legend(fontsize=8.6, frameon=False, loc="lower right", labelcolor=t["ink2"])
    xmax = max([p["ms"] for p in lat_y] + [r["ms"] for r in lat_d]) * 1.08
    axB.set_xlim(0, xmax)
    ys = [p["map"] for p in lat_y] + [acc_d[r["model"]][0] for r in lat_d if r["model"] in acc_d]
    ymin, ymax = min(ys) - 0.07, max(ys) + 0.11
    axB.set_ylim(ymin, ymax)
    axB.text(BUDGET_MS * 0.93, ymin + 0.012, "10 ms budget", rotation=90,
             va="bottom", ha="right", fontsize=8, color="#2a8c7f")

    fig.suptitle("YOLO26 vs DEIMv2", fontsize=13.5, fontweight="bold",
                 color=t["ink"], y=0.985)
    fig.text(0.5, 0.030,
             "Panel B asymmetry is a result, not an omission: four DEIMv2 variants (the DINOv3 "
             "line — s, m, l, x) convert to Core ML without error and then produce",
             ha="center", fontsize=8.2, color=t["ink3"])
    fig.text(0.5, 0.008,
             "boxes 213–277 px wrong, so no latency is reported for them; atto/femto/pico sit "
             "below the capacity floor and were never exported.",
             ha="center", fontsize=8.2, color=t["ink3"])
    fig.tight_layout(rect=[0, 0.062, 1, 0.952])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    fig.savefig(out.with_suffix(".pdf"), facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out.name} + .pdf")


if __name__ == "__main__":
    for m in ("light", "dark"):
        render(m, ROOT / "figures" / f"family_comparison_{m}.png")
