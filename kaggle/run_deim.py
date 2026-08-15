#!/usr/bin/env python3
"""Run the DEIMv2 arm: 8 variants x 3 folds, resumable, with a pre-flight guard.

THE GUARD IS THE POINT. DEIMv2's COCO defaults produce an EMPTY dataloader at 30 images
(total_batch_size 32, drop_last True -> floor(30/32) = 0 batches), and training then does
nothing while still printing epochs and exiting 0. Every config is checked for a non-empty
loader before a single run starts, so a silent no-op cannot be mistaken for a bad result.

Results append to results_deim.jsonl after each run; completed runs are skipped on restart.

Usage:
    !python run_deim.py --repo /kaggle/working/DEIMv2 \
                        --configs /kaggle/working/deim_configs --out /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def preflight(cfg: Path) -> tuple[bool, str]:
    """Read the config and reject settings that silently prevent training."""
    text = cfg.read_text()

    def get(key, default=None):
        m = re.search(rf"^\s*{key}\s*:\s*(\S+)", text, re.M)
        return m.group(1) if m else default

    batch = int(get("total_batch_size", 32))
    drop = (get("drop_last", "True") or "True").lower().startswith("t")
    warm = int(get("warmup_iter", 2000))
    epochs = int(get("epoches", 100))
    ncls = int(get("num_classes", 80))

    ann = re.search(r"ann_file:\s*(\S+)", text)
    n_train = 0
    if ann and Path(ann.group(1)).exists():
        n_train = len(json.loads(Path(ann.group(1)).read_text())["images"])

    problems = []
    batches = n_train // batch if drop else -(-n_train // batch)
    if batches == 0:
        problems.append(f"EMPTY DATALOADER: {n_train} images, batch {batch}, "
                        f"drop_last={drop} -> 0 batches")
    if batches and warm > batches * epochs * 0.10:
        problems.append(f"warmup_iter {warm} exceeds 10% of {batches*epochs} total iters")
    if ncls < 4:
        problems.append(f"num_classes {ncls}: categories are 1-indexed, so 3 classes need 4")
    if n_train == 0:
        problems.append("training annotation file missing or empty")

    # Each variant's stock config carries the stop_epoch of ITS OWN COCO budget (120 for s, 60
    # for l, 468 for atto). Override `epoches` alone and some variants sail past the end of
    # training without ever reaching the stage-1 -> stage-2 switch, silently getting a
    # different training regime from their neighbours. Comparing those is comparing schedules.
    stop = int(get("stop_epoch", 0) or 0)
    if stop >= epochs:
        problems.append(f"NO STAGE 2: stop_epoch {stop} >= epoches {epochs}, so augmentation "
                        f"never turns off and this variant is not comparable to one where it does")
    return (not problems), "; ".join(problems) or (f"{n_train} imgs, {batches} batches/epoch, "
                                                   f"stage2 at {stop}/{epochs}")




# Only these are safe to delete. DEIMv2 writes three LOAD-BEARING files that must never be
# touched, and pruning by mtime alone destroys runs:
#   last.pth       written every epoch, the resume point                (det_solver.py:109)
#   best_stg1.pth  RELOADED at the stage-1 -> stage-2 transition        (det_solver.py:81)
#   best_stg2.pth  the final result, and the checkpoint metrics.py needs
# An earlier version of this function kept "the newest 2 .pth by mtime". When a run's best
# score plateaued, best_stg1.pth went stale, was deleted as the oldest, and training died with
# FileNotFoundError at the transition epoch -- after ~60 epochs of GPU time, and only for the
# models that plateau, which is exactly the ones whose numbers you would most want to trust.
DISPOSABLE = re.compile(r"^checkpoint\d+\.pth$")
SETTLE_S = 30.0        # never unlink a file mid-write: that is what produced a 0-byte last.pth


def prune_forever(run_dir: Path, keep: int, stop):
    """Delete the periodic numbered checkpoints, every 20 s, while training runs.

    DEIMv2 writes checkpoint%04d.pth on a fixed period on top of last.pth. At 120 epochs an
    `x` run would write tens of GB and two parallel processes double the rate, so on a capped
    disk they still have to go -- but they are pure duplicates of last.pth, so dropping them
    costs nothing. `keep` retains the newest few as a hedge against a corrupt last.pth.
    """
    while not stop.is_set():
        try:
            now = time.time()
            ws = [w for w in run_dir.rglob("*.pth") if DISPOSABLE.match(w.name)]
            ws = [w for w in ws if now - w.stat().st_mtime > SETTLE_S]
            ws.sort(key=lambda w: w.stat().st_mtime)
            for w in ws[:-keep] if keep else ws:
                try:
                    w.unlink()
                except OSError:
                    pass
        except Exception:
            pass
        stop.wait(20)

def env_fingerprint():
    """Record the machine and stack per run.

    If any part of a sweep ends up on different hardware -- Kaggle T4 today, a rented 4090
    tomorrow -- this is what lets you check whether a difference is the model or the machine.
    Training hardware does not change a model to within fold-to-fold noise, but that is a
    claim the data should support rather than one the methods section merely asserts.
    """
    import torch, platform
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "python": platform.python_version(),
            "host": platform.node()}

def parse_ap(log: str):
    """Best COCO AP over the run. DEIMv2 prints the standard 12-line COCOeval block."""
    best = {}
    for m in re.finditer(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s*all.*?\]\s*=\s*([\d.\-]+)", log):
        try:
            v = float(m.group(1))
            if v >= 0:
                best["ap50_95"] = max(best.get("ap50_95", 0.0), v)
        except ValueError:
            pass
    for m in re.finditer(r"Average Precision.*?IoU=0\.50\s.*?area=\s*all.*?\]\s*=\s*([\d.\-]+)", log):
        try:
            v = float(m.group(1))
            if v >= 0:
                best["ap50"] = max(best.get("ap50", 0.0), v)
        except ValueError:
            pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--configs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--results", type=Path, default=None,
                    help="results file; give each parallel process its own, then merge")
    ap.add_argument("--port-base", type=int, default=29500,
                    help="torchrun rendezvous port; two processes must not share one")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep the best checkpoint per run (needed for confusion matrices)")
    args = ap.parse_args()
    # train.py runs with cwd=repo, so every path handed to it must be absolute
    args.repo = args.repo.resolve()
    args.configs = args.configs.resolve()
    args.out = args.out.resolve()
    if args.results:
        args.results = args.results.resolve()

    cfgs = sorted(args.configs.glob("deimv2_*.yml"))
    if args.variants:
        cfgs = [c for c in cfgs if c.stem.split("_")[1] in args.variants]
    if not cfgs:
        raise SystemExit(f"no configs in {args.configs}")

    print(f"=== pre-flight: {len(cfgs)} config(s) ===")
    ok_cfgs = []
    for c in cfgs:
        ok, msg = preflight(c)
        print(f"  {'OK  ' if ok else 'FAIL'} {c.stem:28s} {msg}")
        if ok:
            ok_cfgs.append(c)
    if not ok_cfgs:
        raise SystemExit("\nno config passed pre-flight; fix setup_deimv2.py before training")
    if args.check_only:
        return

    results = args.results or (args.out / "results_deim.jsonl")
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                r = json.loads(line)
                if "error" not in r:          # a failed run must not block a retry
                    done.add(r["config"])
            except Exception:
                pass
    todo = [c for c in ok_cfgs if c.stem not in done]
    print(f"\n{len(done)} done, {len(todo)} to run\n")

    for i, cfg in enumerate(todo, 1):
        _, variant, fold = cfg.stem.split("_", 2)
        print(f"[{i}/{len(todo)}] {cfg.stem}")
        t0 = time.time()
        # DEIMv2 calls init_process_group(backend="nccl") unconditionally -- there is no
        # single-GPU branch -- so `python train.py` raises before the first step. torchrun
        # sets the rendezvous env vars even for one process. A distinct port per run avoids
        # "address already in use" when the previous socket has not been released yet.
        port = args.port_base + (i % 100)
        run_dir = args.configs / "runs" / f"{variant}_{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        import threading
        stop = threading.Event()
        pruner = threading.Thread(target=prune_forever,
                                  args=(run_dir, 2 if args.keep_checkpoints else 1, stop),
                                  daemon=True)
        pruner.start()
        proc = subprocess.run(
            ["torchrun", "--nproc_per_node=1", f"--master_port={port}",
             "train.py", "-c", str(cfg), "--use-amp", "--seed", "42"],
            cwd=args.repo, capture_output=True, text=True)
        stop.set(); pruner.join(timeout=5)
        if not args.keep_checkpoints:
            for w in run_dir.rglob("*.pth"):
                w.unlink()
        log = proc.stdout + proc.stderr
        (args.out / "logs").mkdir(exist_ok=True)
        (args.out / "logs" / f"{cfg.stem}.log").write_text(log)

        rec = {"config": cfg.stem, "variant": variant, "fold": fold,
               "minutes": round((time.time() - t0) / 60, 2), "returncode": proc.returncode,
               "env": env_fingerprint()}
        rec.update(parse_ap(log))
        if proc.returncode != 0:
            rec["error"] = log.strip().splitlines()[-1][:300] if log.strip() else "no output"
            print(f"   FAILED rc={proc.returncode}: {rec['error'][:120]}")
        else:
            print(f"   AP50-95 {rec.get('ap50_95','?')}  AP50 {rec.get('ap50','?')}  "
                  f"{rec['minutes']:.1f} min")
        with results.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print(f"\nresults -> {results}\nlogs    -> {args.out/'logs'}")


if __name__ == "__main__":
    main()
