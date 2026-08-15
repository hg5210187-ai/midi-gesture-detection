#!/usr/bin/env python3
"""Build the upload bundle for Kaggle.

Two datasets, kept separate on purpose:

  midi-gesture-v2/     ~40 MB   images, labels (both geometries), COCO json, fold metadata
  deimv2-checkpoints/  ~480 MB  the eight COCO-pretrained .pth files

Splitting them means re-uploading the data after a label fix does not re-upload half a
gigabyte of weights that have not changed.

WHAT IS DELIBERATELY NOT INCLUDED: the split manifests. They hold absolute paths from this
machine (absolute, e.g. /home/you/project/...) which do not exist on Kaggle, and a stale manifest fails in the worst
way -- ultralytics reports "0 images found" and trains on nothing rather than erroring. The
notebook regenerates them from folds.json at runtime, where the real paths are known.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POOL = ROOT / "data" / "pool"
OUT = ROOT / "kaggle" / "upload"
KAGGLE_USER = "shumahara"

DATA_FILES = [
    ("data/pool/manifest.csv", "manifest.csv"),
    ("data/places.csv", "places.csv"),
    ("data/splits/folds.json", "folds.json"),
    ("results/mask_qa_v2.csv", "mask_qa.csv"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "DEIMv2checkpoints")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists; pass --force")
    if OUT.exists():
        shutil.rmtree(OUT)

    data = OUT / "midi-gesture-v2"
    ckpt = OUT / "deimv2-checkpoints"
    data.mkdir(parents=True)
    ckpt.mkdir(parents=True)

    # Exclude machine-local junk. labels.cache in particular embeds absolute paths from
    # whichever machine last ran training here, so shipping it sends stale paths to Kaggle.
    junk = shutil.ignore_patterns("*.cache", ".DS_Store", "__pycache__", "*.pyc")
    for geom in ("obb", "hbb"):
        for sub in ("images", "labels"):
            shutil.copytree(POOL / geom / sub, data / geom / sub, ignore=junk)
    shutil.copytree(POOL / "coco", data / "coco", ignore=junk)
    for src, dst in DATA_FILES:
        p = ROOT / src
        if p.exists():
            shutil.copy2(p, data / dst)

    n = 0
    if args.checkpoints.exists():
        for p in sorted(args.checkpoints.glob("*.pth")):
            shutil.copy2(p, ckpt / p.name)
            n += 1

    import json as _json
    for folder, title, slug in ((data, "MIDI Gesture v2", "midi-gesture-v2"),
                                (ckpt, "DEIMv2 COCO Checkpoints", "deimv2-checkpoints")):
        (folder / "dataset-metadata.json").write_text(_json.dumps(
            {"title": title, "id": f"{KAGGLE_USER}/{slug}",
             "licenses": [{"name": "CC0-1.0"}]}, indent=2))

    (data / "DATASET.md").write_text(
        "# midi-gesture-v2\n\n"
        "60 photos, 120 hand annotations, 3 classes (thumbout, openhand, closedhand),\n"
        "exactly 40 per class. Two hands in every photo.\n\n"
        "```\n"
        "obb/images  obb/labels     rotated boxes,  `cls x1 y1 x2 y2 x3 y3 x4 y4` normalised\n"
        "hbb/images  hbb/labels     upright boxes,  `cls cx cy w h` normalised\n"
        "coco/instances_<group>.json               same boxes, COCO format, category_id 1-3\n"
        "folds.json                                place and group for every photo\n"
        "places.csv, manifest.csv, mask_qa.csv\n"
        "```\n\n"
        "Both geometries come from the same mask contours, so line *i* of an obb label and\n"
        "line *i* of the matching hbb label are the same hand. Instances are ordered by\n"
        "centroid x, so line 1 is always the image-left hand.\n\n"
        "Groups: fold0, fold1, fold2, test -- 15 photos and 10 per class each. A capture\n"
        "place never spans two groups, so no background appears in both train and val.\n\n"
        "Split manifests are NOT included: they would carry absolute paths from another\n"
        "machine. Regenerate them from folds.json.\n"
    )

    def size(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6

    stray = [q for q in OUT.rglob("*") if q.name.endswith(".cache") or q.name == ".DS_Store"]
    for q in stray:
        q.unlink()
    if stray:
        print(f"  removed {len(stray)} stray cache/metadata file(s)")

    print(f"bundle -> {OUT}\n")
    print(f"  midi-gesture-v2/      {size(data):7.1f} MB   "
          f"{len(list((data/'obb'/'images').glob('*')))} images x2 geometries")
    print(f"  deimv2-checkpoints/   {size(ckpt):7.1f} MB   {n} checkpoints")
    print("\nupload each folder as its own Kaggle Dataset, then in the notebook:")
    print("  /kaggle/input/midi-gesture-v2/      and  /kaggle/input/deimv2-checkpoints/")


if __name__ == "__main__":
    main()
