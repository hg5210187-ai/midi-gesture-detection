#!/usr/bin/env python3
"""The accuracy-vs-latency plot: the one figure the deployability claim rests on.

Everything else in the study feeds this. Accuracy comes from 3-fold cross-validation
(mean +- SD over folds, measured on held-out folds only). Latency comes from Core ML on the
actual MacBook Air M4. A model is deployable only if it clears the 10 ms budget AND holds up
on accuracy, and neither number alone identifies it.

WHY LATENCY IS THE X AXIS AND CORE ML IS THE ONLY RUNTIME SHOWN. PyTorch-MPS carries ~18 ms
of fixed dispatch overhead that swamps every architectural difference -- on MPS a 24x range of
model capacity spans about 2 ms and is not even monotonic in resolution. Plotting it would
show a vertical smear that says nothing. Core ML is also what a shipped instrument would use.

THE BUDGET LINE IS AN ENGINEERING TARGET, NOT A PERCEPTUAL THRESHOLD. The camera alone costs
30-50 ms of a ~55-80 ms chain, so 10 ms for the model is a design decision about where to
spend the budget, not a claim that 11 ms is audible.

    python3 scripts/tradeoff.py --out figures/tradeoff.png
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUDGET_MS = 10.0


def load_accuracy(path: Path):
    """(geom, size, imgsz) -> mean, sd, n from the EXACT-IoU rescoring.

    Not ultralytics' own numbers: those match OBB with ProbIoU and HBB with box IoU, which
    over-reports oriented boxes by +0.175 on this dataset (results/probiou_check.json) and
    would put half the points on this plot at the wrong height.
    """
    d = json.loads(path.read_text())["per_cell"]
    out = {}
    for k, v in d.items():
        g, s, z = k.split("-")
        out[(g, s, int(z))] = {"mean": v["mean"], "sd": v["sd"], "n": v["n"]}
    return out


def load_sustained():
    """(geom, size, imgsz) -> sustained median ms, where a 15-minute run exists.

    Burst latency overstates what a fanless machine delivers during a performance: the
    measured penalty is +0.7 to +2.1 ms and it arrives as a step after 5-9 minutes. Where a
    sustained number exists it replaces the burst one, because that is the regime an
    instrument actually runs in.
    THERMAL START MATTERS AS MUCH AS THE MODEL. Within one batch file only the FIRST model
    began on a cold machine; every later one inherited the heat of the 15-minute run before it
    (60 s of cooldown does not undo that). hbb-s@320 measured 9.38 ms preheated and 7.93 ms
    cold -- a 1.45 ms difference that is larger than the gap between several models. So a cold
    reading always wins over a preheated one for the same cell, and preheated-only cells are
    flagged rather than quietly plotted as if they were comparable.
    """
    out = {}
    for p in sorted((ROOT / "results").glob("latency_sustained*.json")):
        for i, r in enumerate(json.loads(p.read_text())["results"]):
            stem = r["package"].replace(".mlpackage", "")
            name, _, z = stem.rpartition("_")
            g, _, s = name.partition("-")
            # This figure is the YOLO26 arm only. DEIMv2 sustained files live in the same
            # directory and do not use the "<geom>-<size>_<imgsz>" convention, so skip
            # anything that does not parse rather than crashing on it.
            if g not in ("hbb", "obb") or not z.isdigit():
                continue
            k = (g, s, int(z))
            cold = (i == 0)                       # first model in a batch starts cold
            prev = out.get(k)
            if prev and prev["cold"] and not cold:
                continue                          # never let a preheated run overwrite a cold one
            out[k] = {"ms": r["overall_median_ms"],
                      "holds": r["holds_budget_throughout"], "cold": cold}
    return out





def load_latency(paths):
    """(geom, size, imgsz) -> median ms, Core ML only."""
    out = {}
    for p in paths:
        if not p.exists():
            continue
        for r in json.load(p.open())["results"]:
            if r["runtime"] != "coreml":
                continue
            stem = r["weights"]                     # "hbb-n" or "hbb-n-fold0-640"
            parts = stem.split("-")
            geom, size = parts[0], parts[1]
            out[(geom, size, r["imgsz"])] = r["median_ms"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracy", type=Path, default=ROOT / "results" / "cv_exact_iou.json")
    ap.add_argument("--latency", type=Path, nargs="+", default=[
        ROOT / "results" / "latency_m4_640.json",
        ROOT / "results" / "latency_m4_lowres.json",
        ROOT / "results" / "latency_m4_bigmodels_lowres.json"])
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "tradeoff.png")
    ap.add_argument("--table", type=Path, default=ROOT / "results" / "tradeoff.json")
    args = ap.parse_args()

    acc = load_accuracy(args.accuracy)
    lat = load_latency(args.latency)
    keys = sorted(set(acc) & set(lat))
    missing_lat = sorted(set(acc) - set(lat))
    missing_acc = sorted(set(lat) - set(acc))
    if not keys:
        raise SystemExit("no cell has both an accuracy and a latency number yet")

    sus = load_sustained()
    pts = [{"geom": g, "size": s, "imgsz": z,
            "ms": sus.get((g, s, z), {}).get("ms", lat[(g, s, z)]),
            "burst_ms": lat[(g, s, z)],
            "sustained": (g, s, z) in sus,
            "map": acc[(g, s, z)]["mean"], "sd": acc[(g, s, z)]["sd"],
            "folds": acc[(g, s, z)]["n"],
            # A SUSTAINED RUN IS PASS/FAIL ON EVERY WINDOW, NOT ON THE MEDIAN. hbb-m@320
            # medians 9.96 ms and would read as deployable, but it crosses the budget at ~6
            # minutes and sits above it for the last nine windows. For an instrument that is
            # a failure, so where a sustained run exists its verdict governs; only cells with
            # burst data alone fall back to comparing the median.
            "meets": (sus[(g, s, z)]["holds"] if (g, s, z) in sus
                      else lat[(g, s, z)] <= BUDGET_MS)}
           for g, s, z in keys]

    # Pareto front over the deployable region: nothing is both faster AND more accurate.
    ok = [p for p in pts if p["meets"]]
    front = [p for p in ok if not any(q["ms"] <= p["ms"] and q["map"] > p["map"] for q in ok)]
    front.sort(key=lambda p: p["ms"])

    print(f"{'cell':18s} {'imgsz':>5s} {'ms':>7s} {'mAP50-95':>16s} {'<=10ms':>7s}")
    print("-" * 60)
    for p in sorted(pts, key=lambda p: -p["map"]):
        print(f"{p['geom']+'-'+p['size']:18s} {p['imgsz']:5d} {p['ms']:7.2f} "
              f"{p['map']:.4f} ± {p['sd']:.4f} {'YES' if p['meets'] else 'no':>7s}")
    print(f"\nPareto front within budget ({len(front)} point(s)):")
    for p in front:
        print(f"  {p['geom']}-{p['size']}@{p['imgsz']}  {p['ms']:.2f} ms  {p['map']:.4f}")
    if missing_lat:
        print(f"\n{len(missing_lat)} cell(s) have accuracy but no latency: "
              f"{', '.join(f'{g}-{s}@{z}' for g, s, z in missing_lat[:8])}")
    if missing_acc:
        print(f"{len(missing_acc)} cell(s) have latency but no accuracy: "
              f"{', '.join(f'{g}-{s}@{z}' for g, s, z in missing_acc[:8])}")

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(
        {"budget_ms": BUDGET_MS, "points": pts,
         "pareto_within_budget": front,
         "missing_latency": [list(k) for k in missing_lat],
         "missing_accuracy": [list(k) for k in missing_acc]}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.axvspan(0, BUDGET_MS, color="#2a9d8f", alpha=0.07, zorder=0)
    ax.axvline(BUDGET_MS, color="#2a9d8f", lw=1.4, ls="--", zorder=1)

    marker = {320: "o", 416: "s", 640: "^"}
    colour = {"hbb": "#3d5a80", "obb": "#d1495b"}
    for p in pts:
        ax.errorbar(p["ms"], p["map"], yerr=p["sd"], fmt=marker.get(p["imgsz"], "o"),
                    color=colour[p["geom"]], ms=7, capsize=3, lw=1,
                    alpha=1.0 if p["meets"] else 0.35, zorder=3)
        ax.annotate(f"{p['size']}", (p["ms"], p["map"]), textcoords="offset points",
                    xytext=(7, -3), fontsize=7.5, color="#444")
    if len(front) > 1:
        ax.plot([p["ms"] for p in front], [p["map"] for p in front],
                color="#666", lw=1, ls=":", zorder=2)

    # placed after the points so the axis limits are final
    ax.text(BUDGET_MS * 0.97, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "10 ms budget", rotation=90, va="bottom", ha="right", fontsize=8, color="#2a9d8f")

    handles = [plt.Line2D([], [], color=colour[g], marker="o", ls="", label=g)
               for g in ("hbb", "obb")]
    handles += [plt.Line2D([], [], color="#888", marker=m, ls="", label=f"{z} px")
                for z, m in marker.items()]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right", ncol=2)
    ax.set_xlabel("Core ML latency on MacBook Air M4 — median ms")
    ax.set_ylabel("mAP50-95  (3-fold mean ± SD)")
    ax.set_title("Accuracy vs latency: shaded region is deployable", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.15, lw=0.6)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"\nwrote {args.out}\nwrote {args.table}")


if __name__ == "__main__":
    main()
