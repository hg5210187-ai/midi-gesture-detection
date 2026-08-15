#!/usr/bin/env python3
"""Convert masks to training labels: two box geometries, two annotation formats.

INPUT   photov2/<original>.jpg  +  photov2_annotate/<original>_mask.tif
        data/splits/folds.json   (place and group per photo, from assign_places.py)

OUTPUT  data/pool/obb/{images,labels}/   rotated boxes,  8 coords   -> YOLO26-OBB
        data/pool/hbb/{images,labels}/   upright boxes,  cx cy w h  -> YOLO26 detect
        data/pool/coco/instances_<group>.json                       -> DEIMv2
        data/pool/manifest.csv
        data/yolo/{obb,hbb}/fold{0,1,2}.yaml + final.yaml

WHY TWO IMAGE TREES rather than one with two label directories: ultralytics derives the
label path from the image path by replacing the last "images" component with "labels", so a
single image tree cannot serve two label sets. Copies, not symlinks -- symlinks do not
survive being tarred up for Kaggle. 60 JPEGs twice is ~36 MB.

FILENAMES are rewritten to e01_p01.jpg. The originals contain spaces, which break shell
globbing, xargs, and anything that splits paths on whitespace. The manifest records the
original name for every file, so provenance is not lost.

INSTANCE ORDER is by centroid x, so line 1 of a label file is always the image-left hand.
That is what makes a left/right swap detectable downstream; a sorted set comparison cannot
see one.

GEOMETRY. Both boxes come from the same contour, so they describe the same instance:
    OBB  cv2.boxPoints(cv2.minAreaRect(contour))   -> 4 corners
    HBB  cv2.boundingRect(contour)                 -> upright extent
Coordinates are normalised then clipped to [0,1]. Clipping matters: ultralytics rejects a
label with any coordinate outside [-0.01, 1.01] as corrupt and silently drops the image, so a
hand touching the frame edge would cost a whole training example. Clipping shears such a box
slightly; dropping the image loses it entirely.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHOTOS = ROOT / "photov2"
MASKS = ROOT / "photov2_annotate"
FOLDS_JSON = ROOT / "data" / "splits" / "folds.json"
POOL = ROOT / "data" / "pool"
YAML_DIR = ROOT / "data" / "yolo"

# mask grey value -> class index, matching mask_to_obb_midi.py:34 exactly
VALUE_TO_IDX = {200: 0, 100: 1, 255: 2}
CLASS_NAMES = {0: "thumbout", 1: "openhand", 2: "closedhand"}
MIN_AREA_PX = 200
CV_FOLDS = ["fold0", "fold1", "fold2"]


def instances(mask: np.ndarray):
    """Every hand blob, left to right by centroid x."""
    found = []
    for value, idx in VALUE_TO_IDX.items():
        binary = (mask == value).astype(np.uint8)
        if not binary.any():
            continue
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_AREA_PX:
                continue
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            found.append({
                "cls": idx, "contour": max(cnts, key=cv2.contourArea),
                "cx": float(centroids[i][0]), "area": int(stats[i, cv2.CC_STAT_AREA]),
            })
    found.sort(key=lambda d: d["cx"])
    return found


def obb_line(inst, w, h):
    pts = cv2.boxPoints(cv2.minAreaRect(inst["contour"]))
    xs = np.clip(pts[:, 0] / w, 0.0, 1.0)
    ys = np.clip(pts[:, 1] / h, 0.0, 1.0)
    if len({(round(x, 6), round(y, 6)) for x, y in zip(xs, ys)}) < 4:
        return None                                  # degenerate after clipping
    coords = " ".join(f"{v:.6f}" for pair in zip(xs, ys) for v in pair)
    return f"{inst['cls']} {coords}"


def hbb_parts(inst, w, h):
    x, y, bw, bh = cv2.boundingRect(inst["contour"])
    cx = min(max((x + bw / 2) / w, 0.0), 1.0)
    cy = min(max((y + bh / 2) / h, 0.0), 1.0)
    nw = min(bw / w, 1.0)
    nh = min(bh / h, 1.0)
    if nw <= 0 or nh <= 0:
        return None, None
    return f"{inst['cls']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}", (x, y, bw, bh)


def write_yaml(path: Path, root: Path, train: Path, val: Path, test: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items())
    path.write_text(
        f"path: {root}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"test: {test}\n"
        f"names:\n{names}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing pool")
    args = ap.parse_args()

    if POOL.exists() and any(POOL.iterdir()) and not args.force:
        raise SystemExit(f"{POOL} is not empty; pass --force to rebuild")
    if POOL.exists():
        shutil.rmtree(POOL)
    for geom in ("obb", "hbb"):
        (POOL / geom / "images").mkdir(parents=True)
        (POOL / geom / "labels").mkdir(parents=True)
    (POOL / "coco").mkdir(parents=True)

    meta = json.loads(FOLDS_JSON.read_text())
    place_of, group_of = meta["photo_place"], meta["photo_group"]

    # deterministic id: place, then capture order within the place
    order = sorted(place_of, key=lambda s: (place_of[s], meta.get("replacements", {}).get(s, s)))
    seq = collections.Counter()
    rows, coco = [], collections.defaultdict(lambda: {"images": [], "annotations": []})
    ann_id = 1

    for stem in order:
        place = place_of[stem]
        seq[place] += 1
        pid = f"{place.lower()}_p{seq[place]:02d}"
        group = group_of[stem]

        img = cv2.imread(str(PHOTOS / f"{stem}.jpg"))
        mask = cv2.imread(str(MASKS / f"{stem}_mask.tif"), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            raise SystemExit(f"{stem}: mask {mask.shape[:2]} != image {(h, w)}")

        insts = instances(mask)
        if len(insts) != 2:
            raise SystemExit(f"{stem}: {len(insts)} instances, expected 2")

        obb_lines, hbb_lines = [], []
        img_id = len(coco[group]["images"]) + 1
        for inst in insts:
            ol = obb_line(inst, w, h)
            hl, rect = hbb_parts(inst, w, h)
            if ol is None or hl is None:
                raise SystemExit(f"{stem}: degenerate box for class {inst['cls']}")
            obb_lines.append(ol)
            hbb_lines.append(hl)
            x, y, bw, bh = rect
            coco[group]["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": inst["cls"] + 1,          # COCO is 1-indexed
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": float(bw * bh), "iscrowd": 0,
            })
            ann_id += 1

        for geom, lines in (("obb", obb_lines), ("hbb", hbb_lines)):
            shutil.copy2(PHOTOS / f"{stem}.jpg", POOL / geom / "images" / f"{pid}.jpg")
            (POOL / geom / "labels" / f"{pid}.txt").write_text("\n".join(lines) + "\n")

        coco[group]["images"].append({"id": img_id, "file_name": f"{pid}.jpg",
                                      "width": w, "height": h})
        rows.append({
            "id": pid, "original": f"{stem}.jpg", "place": place, "group": group,
            "width": w, "height": h,
            "left_class": CLASS_NAMES[insts[0]["cls"]],
            "right_class": CLASS_NAMES[insts[1]["cls"]],
            "left_area": insts[0]["area"], "right_area": insts[1]["area"],
            "label_sequence": " ".join(str(i["cls"]) for i in insts),
        })

    with (POOL / "manifest.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    cats = [{"id": i + 1, "name": n} for i, n in CLASS_NAMES.items()]
    for group, data in coco.items():
        (POOL / "coco" / f"instances_{group}.json").write_text(
            json.dumps({**data, "categories": cats}, indent=1))

    # split manifests + dataset yamls, one set per geometry
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r["id"])
    split_dir = ROOT / "data" / "splits"
    for geom in ("obb", "hbb"):
        imgs = POOL / geom / "images"
        paths = {g: [str((imgs / f"{i}.jpg").resolve()) for i in sorted(v)]
                 for g, v in by_group.items()}
        d = split_dir / geom
        d.mkdir(parents=True, exist_ok=True)
        (d / "test.txt").write_text("\n".join(paths["test"]) + "\n")
        (d / "trainval.txt").write_text("\n".join(p for g in CV_FOLDS for p in paths[g]) + "\n")
        for k, grp in enumerate(CV_FOLDS):
            fd = d / f"fold{k}"
            fd.mkdir(exist_ok=True)
            (fd / "val.txt").write_text("\n".join(paths[grp]) + "\n")
            (fd / "train.txt").write_text(
                "\n".join(p for o in CV_FOLDS if o != grp for p in paths[o]) + "\n")
            write_yaml(YAML_DIR / geom / f"fold{k}.yaml", POOL / geom,
                       fd / "train.txt", fd / "val.txt", d / "test.txt")
        write_yaml(YAML_DIR / geom / "final.yaml", POOL / geom,
                   d / "trainval.txt", d / "trainval.txt", d / "test.txt")

    # ---- verification -------------------------------------------------------------
    errs = []
    for geom, ntok in (("obb", 9), ("hbb", 5)):
        labs = sorted((POOL / geom / "labels").glob("*.txt"))
        if len(labs) != 60:
            errs.append(f"{geom}: {len(labs)} label files, expected 60")
        for p in labs:
            lines = p.read_text().strip().splitlines()
            if len(lines) != 2:
                errs.append(f"{geom}/{p.name}: {len(lines)} lines")
            for ln in lines:
                t = ln.split()
                if len(t) != ntok:
                    errs.append(f"{geom}/{p.name}: {len(t)} tokens, expected {ntok}")
                elif any(not (0.0 <= float(v) <= 1.0) for v in t[1:]):
                    errs.append(f"{geom}/{p.name}: coordinate outside [0,1]")
    # the two geometries must describe the same instances in the same order
    for p in sorted((POOL / "obb" / "labels").glob("*.txt")):
        a = [l.split()[0] for l in p.read_text().split("\n") if l]
        b = [l.split()[0] for l in (POOL / "hbb" / "labels" / p.name).read_text().split("\n") if l]
        if a != b:
            errs.append(f"{p.name}: obb classes {a} != hbb classes {b}")
    tot = collections.Counter()
    for r in rows:
        tot[r["left_class"]] += 1
        tot[r["right_class"]] += 1
    for k, v in tot.items():
        if v != 40:
            errs.append(f"{k}: {v} annotations, expected 40")
    for g, data in coco.items():
        if len(data["images"]) != 15 or len(data["annotations"]) != 30:
            errs.append(f"coco/{g}: {len(data['images'])} images, {len(data['annotations'])} anns")

    print(f"pool -> {POOL}")
    print(f"  {len(rows)} images, {sum(tot.values())} annotations   " +
          "  ".join(f"{k}={v}" for k, v in sorted(tot.items())))
    for g in CV_FOLDS + ["test"]:
        print(f"  {g:7s} {len(by_group[g]):2d} images")
    print(f"  yolo yamls -> {YAML_DIR}/{{obb,hbb}}/fold{{0,1,2}}.yaml, final.yaml")
    print(f"  coco json  -> {POOL}/coco/instances_<group>.json")
    if errs:
        print("\nFAILED:")
        for e in errs[:20]:
            print("  -", e)
        raise SystemExit(1)
    print("\nverification passed: 2 lines per label, coords in [0,1], obb/hbb agree, 40 per class")


if __name__ == "__main__":
    main()
