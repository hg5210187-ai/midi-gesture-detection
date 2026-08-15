#!/usr/bin/env python3
"""Export to Core ML and measure inference latency on this machine.

The study's question is whether a gesture instrument is playable on a consumer laptop, so the
number that matters is measured here, on the MacBook Air M4 -- not on the rented GPU that
trained the model. A datacentre GPU answers a question nobody asked.

WHY CORE ML AND NOT JUST PyTorch-MPS. PyTorch dispatches each operation from Python as a
separate kernel launch. Core ML compiles the whole graph ahead of time and can schedule it on
the Apple Neural Engine, which the MPS path never touches. That is the difference between
~22 ms and a plausible single-digit figure, and it is the only lever besides input resolution
that moves latency by more than a few percent.

WHAT IS TIMED. End-to-end `model.predict()` -- preprocess, inference, postprocess -- with an
explicit synchronise on both sides. Ultralytics' own `Profile` only synchronises CUDA, so on
MPS it stops the clock at kernel *enqueue* and reports a figure several times too low.

MEASUREMENT DISCIPLINE, because a laptop is a noisy instrument:
  - warm-up iterations discarded (first call compiles kernels and allocates)
  - several repeats, with the model order shuffled between them, so thermal drift cannot
    masquerade as an architecture effect
  - median and p95 reported, never the mean: the distribution has a long right tail
  - jitter = p95 - median, which is what a musician actually feels. Uniform lag can be
    anticipated; variable lag cannot.
  - thermal state captured before and after

    python3 scripts/export_coreml.py --weights models/*.pt --imgsz 320 416 640
    python3 scripts/export_coreml.py --weights models/best.pt --no-export   # PyTorch only
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_FRAME = ROOT / "data" / "pool" / "hbb" / "images"

WARMUP = 15
ITERS = 100
REPEATS = 3
ORDER_SEED = 7

# study/latency_policy.py tiers, restated so this script stands alone
TIERS = [("A", 16.0, "hard real-time, keeps up with 60 fps"),
         ("B", 33.0, "one frame at 30 fps"),
         ("C", 100.0, "interactive, drops frames"),
         ("D", 150.0, "audibly late"),
         ("E", float("inf"), "not a musical controller")]
MAX_JITTER_MS = 10.0        # trained musicians' inter-onset SD is ~10-20 ms
MODEL_BUDGET_MS = 10.0      # engineering target, NOT a perceptual threshold: the camera
                            # alone costs 30-50 ms of a ~55-80 ms chain


def tier_of(ms: float) -> str:
    return next(name for name, bound, _ in TIERS if ms <= bound)


def thermal() -> str:
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "n/a"
    except Exception:
        return "n/a"


def host() -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return "?"
    import torch
    return {"chip": sysctl("machdep.cpu.brand_string"), "model": sysctl("hw.model"),
            "ncpu": sysctl("hw.ncpu"), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__}


def synchronise(device: str):
    import torch
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_model(model, frame: str, imgsz: int, device: str, iters: int):
    for _ in range(WARMUP):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    synchronise(device)
    samples = []
    for _ in range(iters):
        synchronise(device)
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
        synchronise(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def summarise(samples):
    s = sorted(samples)
    med = statistics.median(s)
    p95 = s[int(0.95 * (len(s) - 1))]
    return {"n": len(s), "median_ms": round(med, 2),
            "p90_ms": round(s[int(0.90 * (len(s) - 1))], 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(s[int(0.99 * (len(s) - 1))], 2),
            "min_ms": round(s[0], 2),
            "iqr_ms": round(s[int(0.75 * (len(s) - 1))] - s[int(0.25 * (len(s) - 1))], 2),
            "jitter_ms": round(p95 - med, 2),
            "fps_at_median": round(1000.0 / med, 1),
            "tier": tier_of(med),
            "meets_budget": bool(med <= MODEL_BUDGET_MS),
            "jitter_ok": bool(p95 - med <= MAX_JITTER_MS)}


def export_coreml(weights: Path, imgsz: int, half: bool):
    from ultralytics import YOLO
    out = weights.parent / f"{weights.stem}_{imgsz}.mlpackage"
    if out.exists():
        print(f"    {out.name} exists, reusing")
        return out
    try:
        p = YOLO(str(weights)).export(format="coreml", imgsz=imgsz, half=half, nms=True)
        p = Path(p)
        if p != out:
            p.rename(out)
        return out
    except Exception as e:
        print(f"    export FAILED: {type(e).__name__}: {str(e)[:160]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--frame", type=Path, default=None,
                    help="image to time on; defaults to a pool image")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--no-export", action="store_true", help="PyTorch-MPS only, skip Core ML")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency_m4.json")
    args = ap.parse_args()

    if platform.system() != "Darwin":
        raise SystemExit("This measures the target laptop. Run it on the Mac.")

    frame = args.frame
    if frame is None:
        cands = sorted(DEFAULT_FRAME.glob("*.jpg"))
        if not cands:
            raise SystemExit(f"no frame given and none found in {DEFAULT_FRAME}")
        frame = cands[len(cands) // 2]
    print(f"host   {host()['chip']}  {host()['model']}")
    print(f"frame  {frame.name}")
    print(f"thermal(before)  {thermal()}\n")

    from ultralytics import YOLO

    cells = []
    for w in args.weights:
        for sz in args.imgsz:
            cells.append({"weights": w, "imgsz": sz, "runtime": "pytorch-mps"})
            if not args.no_export:
                cells.append({"weights": w, "imgsz": sz, "runtime": "coreml"})

    if not args.no_export:
        print("=== Core ML export ===")
        for w in args.weights:
            for sz in args.imgsz:
                print(f"  {w.stem} @ {sz}")
                p = export_coreml(w, sz, args.half)
                for c in cells:
                    if c["weights"] == w and c["imgsz"] == sz and c["runtime"] == "coreml":
                        c["package"] = p
        cells = [c for c in cells if c["runtime"] != "coreml" or c.get("package")]
        print()

    rng = random.Random(ORDER_SEED)
    pooled = {}
    print(f"=== timing: {len(cells)} cell(s) x {args.repeats} repeats x {args.iters} iters ===")
    for rep in range(args.repeats):
        order = list(range(len(cells)))
        rng.shuffle(order)                    # thermal drift must not look like an effect
        for idx in order:
            c = cells[idx]
            key = (str(c["weights"]), c["imgsz"], c["runtime"])
            src = str(c["package"]) if c["runtime"] == "coreml" else str(c["weights"])
            try:
                model = YOLO(src)
                s = time_model(model, str(frame), c["imgsz"],
                               "mps" if c["runtime"] == "pytorch-mps" else "cpu", args.iters)
                pooled.setdefault(key, []).extend(s)
            except Exception as e:
                print(f"  {key} FAILED: {type(e).__name__}: {str(e)[:110]}")
                pooled.setdefault(key, [])
        print(f"  repeat {rep+1}/{args.repeats} done")

    rows = []
    for (w, sz, rt), samples in pooled.items():
        if not samples:
            continue
        rows.append({"weights": Path(w).stem, "imgsz": sz, "runtime": rt, **summarise(samples)})
    rows.sort(key=lambda r: r["median_ms"])

    print(f"\n{'model':22s} {'imgsz':>5s} {'runtime':12s} {'median':>8s} {'p95':>7s} "
          f"{'jitter':>7s} {'fps':>6s} {'tier':>4s}  {'<=10ms':>6s}")
    print("-" * 92)
    for r in rows:
        print(f"{r['weights'][:22]:22s} {r['imgsz']:5d} {r['runtime']:12s} "
              f"{r['median_ms']:8.2f} {r['p95_ms']:7.2f} {r['jitter_ms']:7.2f} "
              f"{r['fps_at_median']:6.1f} {r['tier']:>4s}  "
              f"{'yes' if r['meets_budget'] else 'no':>6s}")

    payload = {"host": host(), "frame": str(frame), "iters": args.iters,
               "repeats": args.repeats, "order_seed": ORDER_SEED,
               "thermal_before": thermal(), "results": rows,
               "budget_ms": MODEL_BUDGET_MS, "max_jitter_ms": MAX_JITTER_MS,
               "tiers": [{"tier": t, "max_ms": b, "meaning": m} for t, b, m in TIERS],
               "timed_window": "end-to-end model.predict(), torch.mps.synchronize() both sides",
               "note": ("The model is one term in a chain. Camera exposure, readout and USB "
                        "transport cost 30-50 ms on this machine and are not reducible, so a "
                        "10 ms model budget is an engineering target rather than a perceptual "
                        "threshold.")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nthermal(after)   {thermal()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
