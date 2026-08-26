#!/usr/bin/env python3
"""Sustained-load latency for a DEIMv2 Core ML package.

scripts/sustained.py cannot do this: it loads packages through ultralytics' YOLO(), which
does not read a raw Core ML model. The timing method here is deliberately identical to that
script -- same warm-up, same wall-clock windowing, same reported fields -- so the two arms'
sustained numbers sit in one table without an asterisk.

WHY IT MATTERS FOR THIS STUDY. deimv2-n measures 4.80 ms in a burst, faster than the deployed
YOLO26 configuration. But every YOLO sustained figure is a 15-minute run and the measured
burst->sustained penalty there was +0.5 to +2.1 ms, arriving as a step after 5-9 minutes on
this fanless machine. Comparing a DEIMv2 burst against a YOLO sustained number would flatter
DEIMv2 by roughly the size of the effect being claimed.

    python3 scripts/sustained_deim.py --package models/deim/deimv2-n-640.mlpackage --minutes 15
"""
from __future__ import annotations
import argparse, json, platform, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from sustained import windows, host, BUDGET_MS, WARMUP   # identical method, one definition


def load_frame(path: Path, imgsz: int, vit: bool):
    import torchvision.transforms as T
    from PIL import Image
    ops = [T.Resize((imgsz, imgsz)), T.ToTensor()]
    if vit:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)(Image.open(path).convert("RGB")).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--windows", type=int, default=15)
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if platform.system() != "Darwin":
        raise SystemExit("This measures the target laptop. Run it on the Mac.")

    import numpy as np
    import coremltools as ct

    if args.frame is None:
        cands = sorted((ROOT / "data/pool/hbb/images").glob("*.jpg"))
        args.frame = cands[len(cands) // 2]
    # atto/femto/pico/n are HGNetv2 (raw [0,1]); s/m/l/x are DINOv3 (ImageNet normalised)
    vit = any(v in args.package.name for v in ("-s-", "-m-", "-l-", "-x-"))
    frame = load_frame(args.frame, args.imgsz, vit)

    h = host()
    print(f"host    {h['chip']}  {h['model']}")
    print(f"package {args.package.name}   frame {args.frame.name}   "
          f"{'ImageNet-normalised' if vit else 'raw [0,1]'}")
    print(f"running {args.minutes} min in {args.windows} windows\n")

    ml = ct.models.MLModel(str(args.package))
    x = {"images": frame.numpy().astype(np.float32),
         "orig_target_sizes": np.array([[args.imgsz, args.imgsz]], dtype=np.int32)}
    for _ in range(WARMUP):
        ml.predict(x)

    samples, stamps = [], []
    deadline = time.perf_counter() + args.minutes * 60
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        ml.predict(x)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
        stamps.append(t1)

    win = windows(samples, stamps, args.windows)
    s = sorted(samples)
    breaches = [w for w in win if w["median_ms"] > BUDGET_MS]
    rec = {
        "package": args.package.name, "imgsz": args.imgsz, "minutes": args.minutes,
        "iterations": len(samples),
        "overall_median_ms": round(statistics.median(s), 2),
        "overall_p95_ms": round(s[int(0.95 * (len(s) - 1))], 2),
        "first_window_median_ms": win[0]["median_ms"],
        "last_window_median_ms": win[-1]["median_ms"],
        "drift_ms": round(win[-1]["median_ms"] - win[0]["median_ms"], 2),
        "windows": win, "budget_ms": BUDGET_MS,
        "holds_budget_throughout": not breaches,
        "first_breach_window": breaches[0]["window"] if breaches else None,
        "first_breach_at_s": breaches[0]["t_start_s"] if breaches else None,
    }
    print(f"  {rec['iterations']} iterations   overall median {rec['overall_median_ms']} ms"
          f"   p95 {rec['overall_p95_ms']}")
    print(f"  first window {rec['first_window_median_ms']} -> last "
          f"{rec['last_window_median_ms']} ms   drift {rec['drift_ms']:+.2f} ms")
    print(f"  {'HOLDS' if rec['holds_budget_throughout'] else 'BREACHES'} the "
          f"{BUDGET_MS} ms budget"
          + ("" if rec["holds_budget_throughout"]
             else f" from window {rec['first_breach_window']} (~{rec['first_breach_at_s']:.0f}s)"))
    print("   window medians: " + " ".join(f"{w['median_ms']:.1f}" for w in win))

    out = args.out or ROOT / "results" / f"latency_sustained_{args.package.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"host": h, "frame": str(args.frame),
                               "budget_ms": BUDGET_MS, "results": [rec]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
