#!/usr/bin/env python3
"""YOLO26 training driver for Kaggle. 30 cells: 5 sizes x 2 geometries x 3 folds.

RESUME is the point of this script. A Kaggle session can die at 12 hours, or be interrupted,
and 30 runs will not always fit in one. Every finished cell appends to results.jsonl in the
working directory; on restart, cells already recorded there are skipped. Put results.jsonl in
a Kaggle Dataset (or rely on /kaggle/working persisting between edit sessions) and a
half-finished sweep continues rather than restarting.

PATHS are rebuilt here, never read from the bundle. The split manifests on the source machine
hold absolute paths from that machine; a stale manifest makes ultralytics report "0 images found" and
train on nothing without raising. Regenerating from folds.json means the paths are always the
ones that exist on this machine.

Usage inside a notebook cell:
    !python train_yolo.py --data /kaggle/input/midi-gesture-v2 --out /kaggle/working
    !python train_yolo.py ... --sizes n s --geoms hbb        # a subset
    !python train_yolo.py ... --imgsz-sweep                  # add 320/416 for size n
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CLASS_NAMES = {0: "thumbout", 1: "openhand", 2: "closedhand"}
CV_FOLDS = ["fold0", "fold1", "fold2"]



def resolve_data(root: Path) -> Path:
    """Find the real data root under a Kaggle mount, whatever layout Kaggle used.

    Classic datasets mount at /kaggle/input/<slug>/. Data Hub mounts at
    /kaggle/input/datasets/<user>/<slug>/<slug>/, and `--dir-mode zip` adds one more level by
    preserving the uploaded folder. Rather than encode any of that, search for the marker
    files. Getting this wrong yields "0 images found" and a model trained on nothing, so it is
    worth being thorough.
    """
    if not root.exists():
        raise SystemExit(
            f"{root} does not exist.\n"
            f"  Kaggle mounts vary: /kaggle/input/<slug>/ for classic datasets,\n"
            f"  /kaggle/input/datasets/<user>/<slug>/<slug>/ for Data Hub.\n"
            f"  Find it with:  !find /kaggle/input -name folds.json")

    def ok(p: Path) -> bool:
        return (p / "folds.json").exists() and (p / "manifest.csv").exists()

    if ok(root):
        return root
    for depth in range(1, 6):
        for cand in sorted(root.glob("/".join(["*"] * depth))):
            if cand.is_dir() and ok(cand):
                print(f"  data root: {cand}")
                return cand
    raise SystemExit(f"no folds.json + manifest.csv anywhere under {root}"
                     f"   try:  !find {root} -name folds.json")

def build_splits(data: Path, work: Path, geom: str):
    """Write YAMLs and path manifests using paths that exist on THIS machine."""
    folds = json.loads((data / "folds.json").read_text())
    group_of = folds["photo_group"]
    manifest = {}
    import csv
    with (data / "manifest.csv").open() as fh:
        for r in csv.DictReader(fh):
            manifest[r["original"].replace(".jpg", "")] = r["id"]

    images = data / geom / "images"
    members = {g: [] for g in CV_FOLDS + ["test"]}
    for stem, grp in group_of.items():
        pid = manifest.get(stem)
        if pid is None:
            raise SystemExit(f"{stem} missing from manifest.csv")
        members[grp].append(str(images / f"{pid}.jpg"))
    for g in members:
        members[g].sort()

    sd = work / "splits" / geom
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "test.txt").write_text("\n".join(members["test"]) + "\n")
    yamls = {}
    for k, grp in enumerate(CV_FOLDS):
        fd = sd / f"fold{k}"
        fd.mkdir(exist_ok=True)
        (fd / "val.txt").write_text("\n".join(members[grp]) + "\n")
        (fd / "train.txt").write_text(
            "\n".join(p for o in CV_FOLDS if o != grp for p in members[o]) + "\n")
        y = sd / f"fold{k}.yaml"
        y.write_text(
            f"path: {data / geom}\n"
            f"train: {fd / 'train.txt'}\n"
            f"val: {fd / 'val.txt'}\n"
            f"test: {sd / 'test.txt'}\n"
            "names:\n" + "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items()) + "\n")
        yamls[grp] = y
    return yamls



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

def done_cells(path: Path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            # A FAILED cell is not a done cell. Without this, one CUDA OOM in a parallel
            # sweep is recorded, counted as complete, and skipped by every later resume --
            # so the run finishes "60/60" with a hole in it. Filtering here means a rerun
            # picks the failure up automatically.
            if "error" in r or "mAP50-95" not in r:
                continue
            out.add((r["geom"], r["size"], r["fold"], r["imgsz"]))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--sizes", nargs="+", default=["n", "s", "m", "l", "x"])
    ap.add_argument("--geoms", nargs="+", default=["hbb", "obb"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz-sweep", action="store_true",
                    help="also run size n at 320 and 416 (accuracy for the latency arm)")
    ap.add_argument("--device", default="0",
                    help="cuda index on Kaggle; use mps or cpu to smoke-test locally")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.data = resolve_data(args.data)

    from ultralytics import YOLO
    import torch
    if args.device not in ("cpu", "mps") and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}")
        if "P100" in name:
            raise SystemExit(
                "P100 selected. Kaggle's PyTorch build targets sm_70+ and the P100 is sm_60, "
                "so CUDA kernels are missing and nothing will train.\n"
                "Switch the accelerator to 'GPU T4 x2'.")
        a = torch.zeros(8, device="cuda") + 1        # fail fast rather than 40 runs in
        assert float(a.sum()) == 8
        print("CUDA kernels OK")
    elif args.device in ("cpu", "mps"):
        print(f"device={args.device} (local smoke test)")
    else:
        print("WARNING: no GPU visible")

    results = args.out / "results.jsonl"
    already = done_cells(results)
    print(f"{len(already)} cell(s) already done\n")

    cells = []
    for geom in args.geoms:
        for size in args.sizes:
            for k in args.folds:
                cells.append((geom, size, k, args.imgsz))
    if args.imgsz_sweep:
        # sweep whatever --sizes asks for, not a hardcoded "n"
        for geom in args.geoms:
            for size in args.sizes:
                for k in args.folds:
                    for sz in (320, 416):
                        cells.append((geom, size, k, sz))

    todo = [c for c in cells if (c[0], c[1], CV_FOLDS[c[2]], c[3]) not in already]
    print(f"{len(cells)} cells total, {len(todo)} to run")
    if args.dry_run:
        for c in todo:
            print("   ", c)
        return

    yamls = {g: build_splits(args.data, args.out, g) for g in args.geoms}

    for i, (geom, size, k, imgsz) in enumerate(todo, 1):
        grp = CV_FOLDS[k]
        tag = f"{geom}-{size}-fold{k}-{imgsz}"
        weights = f"yolo26{size}-obb.pt" if geom == "obb" else f"yolo26{size}.pt"
        print(f"\n[{i}/{len(todo)}] {tag}   weights={weights}")
        t0 = time.time()
        try:
            model = YOLO(weights)
            model.train(
                data=str(yamls[geom][grp]), epochs=args.epochs, imgsz=imgsz,
                batch=args.batch, device=args.device, seed=42, deterministic=True,
                project=str(args.out / "runs"), name=tag, exist_ok=True,
                patience=args.epochs,          # early stopping OFF: with 3 folds it fires at
                                               # random and measures luck, not architecture
                val=True, plots=False, verbose=False,
                # The dataset lives on a read-only mount, so ultralytics cannot save its label
                # cache and re-decodes 2238x1492 JPEGs every epoch -- ~20 s/epoch, which makes
                # 30 runs exceed a 12 h session. 60 images at 640 px fit in RAM easily.
                cache="ram", workers=2,
            )
            m = model.val(data=str(yamls[geom][grp]), split="val", imgsz=imgsz,
                          device=args.device, plots=False, verbose=False)
            rec = {
                "geom": geom, "size": size, "fold": grp, "imgsz": imgsz,
                "epochs": args.epochs, "batch": args.batch,
                "mAP50-95": round(float(m.box.map), 6), "mAP50": round(float(m.box.map50), 6),
                "precision": round(float(m.box.mp), 6), "recall": round(float(m.box.mr), 6),
                "per_class": {CLASS_NAMES[int(c)]: {"ap50": round(float(m.box.ap50[j]), 6),
                                                    "ap50_95": round(float(m.box.ap[j]), 6)}
                              for j, c in enumerate(m.box.ap_class_index)},
                "minutes": round((time.time() - t0) / 60, 2),
                "env": env_fingerprint(),
                "weights": str(args.out / "runs" / tag / "weights" / "best.pt"),
            }
        except Exception as e:
            rec = {"geom": geom, "size": size, "fold": grp, "imgsz": imgsz,
                   "error": f"{type(e).__name__}: {e}",
                   "minutes": round((time.time() - t0) / 60, 2)}
            print(f"   FAILED: {rec['error']}")
        with results.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if "error" not in rec:
            print(f"   mAP50-95 {rec['mAP50-95']:.4f}   mAP50 {rec['mAP50']:.4f}   "
                  f"{rec['minutes']:.1f} min")

    print(f"\nresults -> {results}")


if __name__ == "__main__":
    main()
