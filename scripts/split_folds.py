#!/usr/bin/env python3
"""Materialise the fold assignment from folds.json.

Writes two things, because they serve different readers:

  data/folds/<group>/photos|masks/   real files, so the split can be eyeballed and a contact
                                     sheet made per fold. Copies, not symlinks -- 60 photos is
                                     ~15 MB and symlinks do not survive being tarred to Kaggle.

  data/splits/*.txt                  newline-delimited absolute paths, which is what ultralytics
                                     consumes directly. 3-fold CV means each fold is the val set
                                     once and the train set twice, so train.txt for fold k is the
                                     union of the other two folds. The test set never appears in
                                     any train or val manifest.

Absolute paths do not survive moving machines; regenerate rather than edit.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHOTOS = ROOT / "photov2"
MASKS = ROOT / "photov2_annotate"
FOLDS_JSON = ROOT / "data" / "splits" / "folds.json"
FOLD_DIR = ROOT / "data" / "folds"
SPLIT_DIR = ROOT / "data" / "splits"
QA = ROOT / "results" / "mask_qa_v2.csv"

CV_FOLDS = ["fold0", "fold1", "fold2"]
CLASSES = ("thumbout", "openhand", "closedhand")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = json.loads(FOLDS_JSON.read_text())
    photo_group = meta["photo_group"]

    classes_of = collections.defaultdict(list)
    for r in csv.DictReader(QA.open()):
        if r.get("cls"):
            classes_of[r["stem"]].append(r["cls"])

    members = collections.defaultdict(list)
    for stem, grp in photo_group.items():
        members[grp].append(stem)
    for grp in members:
        members[grp].sort()

    print(f"  {'group':7s} {'photos':>6s} {'annots':>7s}   " +
          "  ".join(f"{c[:5]:>5s}" for c in CLASSES))
    print("  " + "-" * 52)
    for grp in CV_FOLDS + ["test"]:
        stems = members[grp]
        c = collections.Counter(k for s in stems for k in classes_of[s])
        print(f"  {grp:7s} {len(stems):6d} {sum(c.values()):7d}   " +
              "  ".join(f"{c[k]:5d}" for k in CLASSES))

    if args.dry_run:
        print("\nDRY RUN - nothing written")
        return

    for grp in CV_FOLDS + ["test"]:
        for sub in ("photos", "masks"):
            d = FOLD_DIR / grp / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        for stem in members[grp]:
            src_p = PHOTOS / f"{stem}.jpg"
            src_m = MASKS / f"{stem}_mask.tif"
            if src_p.exists():
                shutil.copy2(src_p, FOLD_DIR / grp / "photos" / src_p.name)
            if src_m.exists():
                shutil.copy2(src_m, FOLD_DIR / grp / "masks" / src_m.name)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {g: [str((PHOTOS / f"{s}.jpg").resolve()) for s in members[g]]
             for g in CV_FOLDS + ["test"]}

    (SPLIT_DIR / "test.txt").write_text("\n".join(paths["test"]) + "\n")
    (SPLIT_DIR / "trainval.txt").write_text(
        "\n".join(p for g in CV_FOLDS for p in paths[g]) + "\n")
    for k, grp in enumerate(CV_FOLDS):
        d = SPLIT_DIR / f"fold{k}"
        d.mkdir(exist_ok=True)
        (d / "val.txt").write_text("\n".join(paths[grp]) + "\n")
        train = [p for other in CV_FOLDS if other != grp for p in paths[other]]
        (d / "train.txt").write_text("\n".join(train) + "\n")

    # the assertion that protects the study: nothing in test may appear in any train or val list
    test = set(paths["test"])
    for k in range(len(CV_FOLDS)):
        for name in ("train", "val"):
            got = set((SPLIT_DIR / f"fold{k}" / f"{name}.txt").read_text().splitlines())
            assert not (got & test), f"fold{k}/{name} overlaps test"
    print("\nassertion passed: test appears in no train or val manifest")

    print(f"\nfiles  -> {FOLD_DIR}/<group>/photos|masks/")
    print(f"lists  -> {SPLIT_DIR}/fold{{0,1,2}}/{{train,val}}.txt, test.txt, trainval.txt")
    for k, grp in enumerate(CV_FOLDS):
        n_tr = len((SPLIT_DIR / f"fold{k}" / "train.txt").read_text().splitlines())
        n_va = len((SPLIT_DIR / f"fold{k}" / "val.txt").read_text().splitlines())
        print(f"   fold{k}:  train {n_tr:3d}   val {n_va:3d}")
    print(f"   test:   {len(paths['test'])} (held out, untouched until the very end)")


if __name__ == "__main__":
    main()
