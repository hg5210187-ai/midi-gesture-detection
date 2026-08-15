#!/usr/bin/env python3
"""Sustained-load latency: does the model hold the budget while someone is actually playing?

THE GAP THIS CLOSES. export_coreml.py measures bursts -- 80 iterations, a few seconds. A
gesture instrument runs continuously for minutes or hours, and the MacBook Air is FANLESS, so
sustained inference heats the package and the SoC throttles. A burst benchmark cannot see
that, and "7.09 ms" from a cold machine is not the same claim as "7.09 ms during a set".

WHAT IS REPORTED. Latency is bucketed into fixed windows over the whole run, so the shape over
time is visible rather than collapsed into one median. The number that matters is the drift
between the first window and the last, and whether any window's median crosses the budget.
A model that starts at 7 ms and ends at 11 ms passes a burst test and fails in performance.

WHY NO THERMAL SENSOR READING. `pmset -g therm` reports nothing on this machine and
`powermetrics` needs root, so there is no unprivileged thermal telemetry to record. That is
fine: latency over time is the quantity of interest, and it is the effect rather than a proxy
for it. The absence is stated rather than papered over.

The same code path as the burst benchmark -- ultralytics predict() on the .mlpackage -- so the
two sets of numbers are directly comparable.

    python3 scripts/sustained.py --packages models/hbb-n_320.mlpackage --imgsz 320 --minutes 15
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUDGET_MS = 10.0
WARMUP = 20


def host() -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return "?"
    return {"chip": sysctl("machdep.cpu.brand_string"), "model": sysctl("hw.model"),
            "ncpu": sysctl("hw.ncpu"), "os": platform.platform()}


def windows(samples, stamps, n_windows: int):
    """Bucket by WALL-CLOCK time, not by sample count.

    Bucketing by count would hide the effect being measured: if the machine throttles, later
    windows contain fewer iterations, and equal-count buckets would silently stretch to cover
    more time and blur the transition.
    """
    if not samples:
        return []
    t0, t1 = stamps[0], stamps[-1]
    span = max(t1 - t0, 1e-9) / n_windows
    out = []
    for w in range(n_windows):
        lo, hi = t0 + w * span, t0 + (w + 1) * span
        sel = [s for s, t in zip(samples, stamps) if lo <= t < hi or (w == n_windows - 1 and t == hi)]
        if not sel:
            continue
        sel_sorted = sorted(sel)
        out.append({"window": w + 1,
                    "t_start_s": round(lo - t0, 1), "t_end_s": round(hi - t0, 1),
                    "n": len(sel),
                    "median_ms": round(statistics.median(sel_sorted), 2),
                    "p95_ms": round(sel_sorted[int(0.95 * (len(sel_sorted) - 1))], 2),
                    "max_ms": round(sel_sorted[-1], 2)})
    return out


def run_one(pkg: Path, frame: str, imgsz: int, minutes: float, n_windows: int):
    from ultralytics import YOLO
    # A .mlpackage carries no task metadata, so ultralytics guesses -- and it guesses
    # "detect". For an oriented-box model that decodes the head wrongly, which would make the
    # OBB latency figure describe something other than the model. Take it from the name.
    task = "obb" if pkg.name.startswith("obb") else "detect"
    model = YOLO(str(pkg), task=task)
    for _ in range(WARMUP):
        model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)

    samples, stamps = [], []
    t_start = time.perf_counter()
    deadline = t_start + minutes * 60
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
        stamps.append(t1)

    win = windows(samples, stamps, n_windows)
    s = sorted(samples)
    first, last = win[0]["median_ms"], win[-1]["median_ms"]
    breaches = [w for w in win if w["median_ms"] > BUDGET_MS]
    return {
        "package": pkg.name, "imgsz": imgsz, "minutes": minutes,
        "iterations": len(samples),
        "overall_median_ms": round(statistics.median(s), 2),
        "overall_p95_ms": round(s[int(0.95 * (len(s) - 1))], 2),
        "first_window_median_ms": first, "last_window_median_ms": last,
        "drift_ms": round(last - first, 2),
        "drift_pct": round(100.0 * (last - first) / first, 1) if first else None,
        "windows": win,
        "budget_ms": BUDGET_MS,
        "holds_budget_throughout": len(breaches) == 0,
        "first_breach_window": breaches[0]["window"] if breaches else None,
        "first_breach_at_s": breaches[0]["t_start_s"] if breaches else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", type=Path, nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", required=True,
                    help="one per package, or a single value for all")
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--windows", type=int, default=15)
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--cooldown", type=float, default=60.0,
                    help="seconds idle between models, so one run does not preheat the next")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency_sustained_m4.json")
    args = ap.parse_args()

    if platform.system() != "Darwin":
        raise SystemExit("This measures the target laptop. Run it on the Mac.")
    sizes = args.imgsz if len(args.imgsz) == len(args.packages) else \
        [args.imgsz[0]] * len(args.packages)

    frame = args.frame
    if frame is None:
        cands = sorted((ROOT / "data" / "pool" / "hbb" / "images").glob("*.jpg"))
        frame = cands[len(cands) // 2]

    h = host()
    print(f"host  {h['chip']}  {h['model']}  ({h['ncpu']} cpu)")
    print(f"frame {frame.name}")
    print(f"{len(args.packages)} model(s) x {args.minutes} min, "
          f"{args.windows} windows, {args.cooldown}s cooldown between\n")

    results = []
    for i, (pkg, sz) in enumerate(zip(args.packages, sizes)):
        if i:
            print(f"  cooling down {args.cooldown:.0f}s ...")
            time.sleep(args.cooldown)
        print(f"=== {pkg.name} @ {sz} for {args.minutes} min ===")
        r = run_one(pkg, str(frame), sz, args.minutes, args.windows)
        results.append(r)
        print(f"  {r['iterations']} iterations   overall median {r['overall_median_ms']} ms")
        print(f"  first window {r['first_window_median_ms']} -> last {r['last_window_median_ms']} "
              f"ms   drift {r['drift_ms']:+.2f} ms ({r['drift_pct']:+.1f}%)")
        if r["holds_budget_throughout"]:
            print(f"  HOLDS the {BUDGET_MS} ms budget for the whole run")
        else:
            print(f"  BREACHES budget from window {r['first_breach_window']} "
                  f"(~{r['first_breach_at_s']:.0f}s in)")
        print("   window medians: " +
              " ".join(f"{w['median_ms']:.1f}" for w in r["windows"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "host": h, "frame": str(frame), "budget_ms": BUDGET_MS,
        "timed_window": "end-to-end ultralytics predict() on the .mlpackage, same path as the "
                        "burst benchmark in export_coreml.py",
        "thermal_telemetry": "unavailable without root (pmset -g therm reports nothing on this "
                             "machine, powermetrics needs sudo); latency over time is reported "
                             "instead, which is the effect rather than a proxy for it",
        "note": "The MacBook Air is fanless, so sustained load is the regime a real instrument "
                "runs in. Burst figures from export_coreml.py describe a cold machine.",
        "results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
