#!/usr/bin/env python3
"""Install QuPath-exported masks over the automatic ones, merging where QuPath is partial.

Why a merge rather than a plain copy: QuPath writes a complete mask from the annotations in
its project. If only one hand was drawn there -- because the other was already correct in the
automatic pass -- a straight copy would silently delete the good hand. That is the same class
of error as an unlabelled hand, and it fails quietly.

Rule: a class present in the QuPath export always wins for that class. A class present only in
the existing mask is carried over untouched. So drawing one hand adds it; redrawing a hand
replaces it; nothing is ever lost by omission.

    python3 scripts/install_manual_masks.py --dry-run
    python3 scripts/install_manual_masks.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "needs_fixing" / "manual_annotation" / "class_image_mask"
DST = ROOT / "photov2_annotate"
BAK = ROOT / "needs_fixing" / "masks_superseded_auto"

VALUES = {200: "thumbout", 100: "openhand", 255: "closedhand"}


def blobs(mask: np.ndarray, value: int) -> int:
    b = (mask == value).astype(np.uint8)
    if not b.any():
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 200)


def summarise(mask: np.ndarray) -> str:
    parts = [f"{VALUES[v]}x{blobs(mask, v)}" for v in (200, 100, 255) if (mask == v).any()]
    return ", ".join(parts) or "EMPTY"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exports = sorted(args.src.glob("*_mask.tif"))
    if not exports:
        print(f"no exported masks in {args.src}")
        return
    BAK.mkdir(parents=True, exist_ok=True)

    print(f"{'photo':22s} {'existing':26s} {'qupath':26s} {'result':26s}")
    print("-" * 104)
    for src in exports:
        dst = args.dst / src.name
        new = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if new is None:
            print(f"{src.name:22s} UNREADABLE EXPORT — skipped")
            continue
        if new.ndim == 3:
            new = new[..., 0]

        old = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED) if dst.exists() else None
        if old is not None and old.ndim == 3:
            old = old[..., 0]

        if old is None or old.shape != new.shape:
            merged = new
        else:
            merged = new.copy()
            for v in VALUES:                      # carry over classes QuPath did not paint
                if not (new == v).any() and (old == v).any():
                    merged[(old == v) & (merged == 0)] = v

        stem = src.name.replace("_mask.tif", "").replace("Photo on 2026-08-11 at ", "")
        o = summarise(old) if old is not None else "-"
        print(f"{stem:22s} {o:26s} {summarise(new):26s} {summarise(merged):26s}")

        if not args.dry_run:
            if dst.exists():
                shutil.copy2(dst, BAK / dst.name)
            cv2.imwrite(str(dst), merged)

    print("\nDRY RUN — nothing written" if args.dry_run
          else f"\ninstalled {len(exports)} mask(s); previous versions backed up to {BAK}/"
               "\nnow run:  python3 scripts/mask_qa.py")


if __name__ == "__main__":
    main()
