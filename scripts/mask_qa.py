#!/usr/bin/env python3
"""Quality-assurance pass over the photov2 masks.

Two hands per photo, three classes. A mask is uint8, same HxW as its photo:
    0 background · 100 openhand · 200 thumbout · 255 closedhand

What this checks, in rough order of how badly it burns you if wrong:

1. INSTANCE COUNT. Every photo must yield exactly 2 hand blobs. A missing hand is an
   unlabelled positive -- the detector is actively taught that a hand there is background.
2. GEOMETRY. Blob area, aspect, and whether it touches the frame edge (a hand cut off by
   the frame produces a box that is not the hand).
3. EDGE ALIGNMENT. A correct outline sits on real image gradients. We take the Sobel
   magnitude, normalise by its 99th percentile so exposure does not matter, sample the mean
   gradient along the contour at zero offset, and compare against the best score over a
   small search grid. align = zero/best. 1.000 means nothing beats where the contour
   already is; well below 1 means a shifted contour lies on stronger edges, i.e. the mask
   is offset from the hand.

   Deliberately NOT skin-colour scoring: it misfires on backlit and colour-cast frames,
   which several of these are.

4. SIDE ORDER. Blobs are sorted by centroid x so index 0 is the image-left hand. This is
   what makes a left/right class swap detectable downstream.

Outputs overlays at alpha 0.5 plus contact sheets, because the scores are a triage aid and
not evidence. Look at the pictures.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHOTOS = ROOT / "photov2"
MASKS = ROOT / "photov2_annotate"
OUT_OVERLAY = ROOT / "qa_overlays"
OUT_CSV = ROOT / "results" / "mask_qa_v2.csv"

CLASSES = {
    100: ("openhand", (0, 0, 255)),     # red   (BGR)
    200: ("thumbout", (255, 128, 0)),   # blue
    255: ("closedhand", (0, 255, 0)),   # green
}

ALPHA = 0.5
MIN_AREA_PX = 1500          # below this a blob is noise, not a hand
MIN_ALIGN = 0.80            # below this the outline is probably off the hand
MIN_SHIFT_PX = 6            # ...but only flag if the better position is this far away
EDGE_MARGIN_PX = 3          # a blob within this of the frame edge is clipped


def gradient_map(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    p99 = np.percentile(mag, 99)
    return mag / p99 if p99 > 1e-6 else mag


def sample_contour(grad: np.ndarray, contour: np.ndarray, dx: int, dy: int) -> float:
    h, w = grad.shape
    pts = contour.reshape(-1, 2)
    xs = np.clip(pts[:, 0] + dx, 0, w - 1)
    ys = np.clip(pts[:, 1] + dy, 0, h - 1)
    return float(grad[ys, xs].mean())


def alignment(grad: np.ndarray, contour: np.ndarray, scale: float):
    """Return (align_ratio, (best_dx, best_dy), distance_px)."""
    radius = max(6, int(0.012 * scale))
    step = max(1, radius // 6)
    offsets = list(range(-radius, radius + 1, step))
    if 0 not in offsets:
        offsets.append(0)
        offsets.sort()

    zero = sample_contour(grad, contour, 0, 0)
    best, best_off = zero, (0, 0)
    for dy in offsets:
        for dx in offsets:
            if dx == 0 and dy == 0:
                continue
            s = sample_contour(grad, contour, dx, dy)
            if s > best:
                best, best_off = s, (dx, dy)
    ratio = zero / best if best > 1e-6 else 1.0
    return ratio, best_off, math.hypot(*best_off)


def instances(mask: np.ndarray):
    """Every blob in the mask, sorted left-to-right by centroid x.

    Sorting is what lets a downstream check compare the painted class SEQUENCE against the
    expected left-then-right order, which a sorted set comparison cannot do.
    """
    out = []
    for value, (name, colour) in CLASSES.items():
        binary = (mask == value).astype(np.uint8)
        if not binary.any():
            continue
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 200:            # true speckle; MIN_AREA_PX flags the rest as 'tiny'
                continue
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            out.append(dict(
                value=value, cls=name, colour=colour, area=area, contour=cnt,
                cx=float(centroids[i][0]), cy=float(centroids[i][1]),
                x=x, y=y, w=w, h=h, comp=comp,
            ))
    out.sort(key=lambda d: d["cx"])
    return out


def render(bgr: np.ndarray, insts, header: str) -> np.ndarray:
    over = bgr.copy()
    for d in insts:
        over[d["comp"] > 0] = d["colour"]
    blend = cv2.addWeighted(over, ALPHA, bgr, 1 - ALPHA, 0)
    for i, d in enumerate(insts):
        cv2.drawContours(blend, [d["contour"]], -1, d["colour"], 3)
        label = f'{i}:{d["cls"]}'
        cv2.putText(blend, label, (d["x"], max(30, d["y"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(blend, label, (d["x"], max(30, d["y"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, d["colour"], 2, cv2.LINE_AA)
    cv2.rectangle(blend, (0, 0), (blend.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(blend, header, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return blend


def contact_sheet(paths, out_path, cols=6, cell=460):
    if not paths:
        return
    rows = math.ceil(len(paths) / cols)
    sheet = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = cell / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        r, c = divmod(i, cols)
        y, x = r * cell, c * cell
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, default=PHOTOS)
    ap.add_argument("--masks", type=Path, default=MASKS)
    ap.add_argument("--out", type=Path, default=OUT_OVERLAY)
    ap.add_argument("--csv", type=Path, default=None,
                    help="defaults to results/mask_qa_<out-dir-name>.csv, so a partial run "
                         "never clobbers the full-set report")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or (OUT_CSV if args.out == OUT_OVERLAY
                            else ROOT / "results" / f"mask_qa_{args.out.name}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in args.photos.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows, flagged, overlay_paths = [], [], []

    for p in photos:
        mpath = args.masks / f"{p.stem}_mask.tif"
        bgr = cv2.imread(str(p))
        if bgr is None:
            rows.append(dict(stem=p.stem, n_hands=0, note="UNREADABLE-PHOTO"))
            continue
        if not mpath.exists():
            rows.append(dict(stem=p.stem, n_hands=0, note="NO-MASK"))
            flagged.append(p.stem)
            continue

        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if mask is None:
            rows.append(dict(stem=p.stem, n_hands=0, note="UNREADABLE-MASK"))
            flagged.append(p.stem)
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]

        H, W = bgr.shape[:2]
        if mask.shape[:2] != (H, W):
            rows.append(dict(stem=p.stem, n_hands=0,
                             note=f"SIZE-MISMATCH photo={W}x{H} mask={mask.shape[1]}x{mask.shape[0]}"))
            flagged.append(p.stem)
            continue

        grad = gradient_map(bgr)
        scale = math.hypot(W, H)
        insts = instances(mask)
        notes = []
        if len(insts) != 2:
            notes.append(f"HANDS={len(insts)}-EXPECTED-2")

        seq = " ".join({"thumbout": "0", "openhand": "1", "closedhand": "2"}[d["cls"]]
                       for d in insts)

        for idx, d in enumerate(insts):
            ratio, off, dist = alignment(grad, d["contour"], scale)
            n = []
            if d["area"] < MIN_AREA_PX:
                n.append("tiny")
            if ratio < MIN_ALIGN and dist >= MIN_SHIFT_PX:
                n.append(f"misaligned~{int(dist)}px")
            if (d["x"] <= EDGE_MARGIN_PX or d["y"] <= EDGE_MARGIN_PX
                    or d["x"] + d["w"] >= W - EDGE_MARGIN_PX
                    or d["y"] + d["h"] >= H - EDGE_MARGIN_PX):
                n.append("CLIPPED-BY-FRAME")
            rows.append(dict(
                stem=p.stem, idx=idx,
                side="left" if idx == 0 else "right",
                cls=d["cls"], value=d["value"], area=d["area"],
                area_frac=round(d["area"] / (W * H), 5),
                aspect=round(d["w"] / d["h"], 3) if d["h"] else 0,
                align=round(ratio, 3), shift=f'{off[0]:+d},{off[1]:+d}',
                shift_px=int(dist), n_hands=len(insts), seq=seq,
                note=",".join(n),
            ))
            notes.extend(n)

        if not insts:
            rows.append(dict(stem=p.stem, n_hands=0, seq="", note="EMPTY-MASK"))
            notes.append("EMPTY-MASK")

        head = f'{p.stem}   hands={len(insts)}  seq=[{seq}]'
        if notes:
            head += "  ** " + " ".join(sorted(set(notes)))
            flagged.append(p.stem)
        op = args.out / f"{p.stem}_overlay.jpg"
        cv2.imwrite(str(op), render(bgr, insts, head), [cv2.IMWRITE_JPEG_QUALITY, 90])
        overlay_paths.append(op)

    fields = ["stem", "idx", "side", "cls", "value", "area", "area_frac", "aspect",
              "align", "shift", "shift_px", "n_hands", "seq", "note"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    contact_sheet(overlay_paths, args.out / "_contact_all.jpg")
    bad = [args.out / f"{s}_overlay.jpg" for s in flagged]
    contact_sheet([b for b in bad if b.exists()], args.out / "_contact_flagged.jpg", cols=4, cell=560)

    inst_rows = [r for r in rows if r.get("idx") is not None and "cls" in r]
    print(f"photos            {len(photos)}")
    print(f"hand instances    {len(inst_rows)}   (expected {len(photos)*2})")
    print(f"photos flagged    {len(set(flagged))}")
    aligns = [r["align"] for r in inst_rows if "align" in r]
    if aligns:
        print(f"align  mean {np.mean(aligns):.3f}  min {min(aligns):.3f}")
    print(f"\noverlays -> {args.out}")
    print(f"csv      -> {csv_path}")
    if flagged:
        print("\nFLAGGED:")
        for s in sorted(set(flagged)):
            ns = sorted({r["note"] for r in rows if r.get("stem") == s and r.get("note")})
            print(f"  {s}: {'; '.join(ns)}")


if __name__ == "__main__":
    main()
