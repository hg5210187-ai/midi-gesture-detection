#!/usr/bin/env python3
"""Sustained latency on the M4: every 15-minute run, with the thermal-start effect shown.

WHY THIS FIGURE EXISTS. The sustained results were the study's most consequential latency
finding and lived only in a table. They carry two things a table states but does not show:
how little margin separates a configuration that holds the budget from one that breaches, and
that the SAME configuration can land on either side depending on whether the machine was cold.

WHAT IS PLOTTED. One row per 15-minute run, sorted by median. The dot is the median, the
whisker runs to p95 -- p95 matters as much as the median for an instrument, because variable
lag cannot be anticipated the way uniform lag can. Where a configuration has both a cold and a
preheated reading, a connector shows the shift between them.

COLOUR IS NOT LOAD-BEARING. Every row carries its verdict as text, so the figure reads in
greyscale and for colour-blind readers; the two hues are a pair validated earlier for CVD
separation (dE 9.2 on the all-pairs list) and only reinforce the label.

    python3 scripts/sustained_figure.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET = 10.0
C = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8984",
                   grid="#e6e5e0", band="#eef6ee", hold="#1baf7a", breach="#eb6834",
                   pre="#b9b8b2"),
     "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8984",
                   grid="#2e2d2a", band="#1e2a20", hold="#199e70", breach="#d95926",
                   pre="#55544f")}


def load():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/latency_sustained*.json"))):
        d = json.loads(Path(f).read_text())
        for i, r in enumerate(d["results"]):
            rows.append(dict(
                pkg=r["package"].replace(".mlpackage", "").replace("_", "@"),
                cold=(i == 0), med=r["overall_median_ms"], p95=r["overall_p95_ms"],
                holds=r["holds_budget_throughout"], breach=r.get("first_breach_at_s")))
    rows.sort(key=lambda r: r["med"])
    return rows


def render(mode: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = C[mode]
    rows = load()
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.2, 0.64 * n + 2.6))
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])

    ax.axvspan(0, BUDGET, color=t["band"], zorder=0)
    ax.axvline(BUDGET, color="#2a8c7f", lw=1.4, ls="--", zorder=2)

    ys = list(range(n))[::-1]
    for y, r in zip(ys, rows):
        col = t["hold"] if r["holds"] else t["breach"]
        ax.plot([r["med"], r["p95"]], [y, y], color=col, lw=2.4, alpha=.45,
                solid_capstyle="round", zorder=3)
        ax.scatter([r["p95"]], [y], s=26, color=col, alpha=.55, zorder=3)
        ax.scatter([r["med"]], [y], s=104, color=col, zorder=4,
                   edgecolor=t["surface"], linewidth=1.6)
        ax.text(r["med"], y + .30, f"{r['med']:.2f}", ha="center", va="bottom",
                fontsize=8.6, color=t["ink"], fontweight="bold")
        verdict = "holds all 15 windows" if r["holds"] else f"breach @ {r['breach']:.0f}s"
        ax.text(r["p95"] + .45, y, verdict, va="center", fontsize=8.8, color=col)

    by = {}
    for y, r in zip(ys, rows):
        by.setdefault(r["pkg"], []).append((y, r))
    for pkg, pair in by.items():
        if len(pair) == 2:
            (y1, r1), (y2, r2) = sorted(pair, key=lambda p: p[1]["med"])
            ax.annotate("", xy=(r2["med"], y2), xytext=(r1["med"], y1),
                        arrowprops=dict(arrowstyle="<-", color=t["pre"], lw=1.4,
                                        connectionstyle="arc3,rad=0.22"), zorder=2)
            ax.text(min(r1["med"], r2["med"]) - 0.55, (y1 + y2) / 2,
                    f"+{r2['med'] - r1['med']:.2f} ms\nwhen preheated", ha="right",
                    va="center", fontsize=7.8, color=t["ink3"], style="italic")

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['pkg']}   {'cold' if r['cold'] else 'preheated'}" for r in rows],
                       fontsize=9.5)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color(t["ink"] if r["cold"] else t["ink3"])
    ax.set_xlim(-3.4, max(r["p95"] for r in rows) + 7.0)
    ax.set_ylim(-0.9, n - 0.15)
    ax.set_xlabel("sustained latency over 15 minutes — median, whisker to p95 (ms)",
                  fontsize=10.5, color=t["ink2"])
    ax.grid(axis="x", alpha=.28, lw=.6, color=t["grid"])
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.text(BUDGET - .16, -0.78, "10 ms budget", rotation=90, ha="right", va="bottom",
            fontsize=8, color="#2a8c7f")

    fig.suptitle("Sustained latency on the MacBook Air M4 — every 15-minute run",
                 fontsize=13, fontweight="bold", color=t["ink"], y=0.985)
    fig.text(0.5, 0.030,
             "Thermal start decides two of these verdicts: hbb-l@320 and hbb-s@320 each breach "
             "when preheated and hold from cold. The two gaps agree\nto within 0.2 ms (+1.25 "
             "and +1.45), so the preheating penalty looks like a stable property of the "
             "machine rather than a per-model quirk.",
             ha="center", fontsize=8.4, color=t["ink3"])
    fig.tight_layout(rect=[0, 0.070, 1, 0.955])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    fig.savefig(out.with_suffix(".pdf"), facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out.name} + .pdf   ({n} runs)")


if __name__ == "__main__":
    for m in ("light", "dark"):
        render(m, ROOT / "figures" / f"sustained_latency_{m}.png")
