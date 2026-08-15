"""
Three-class mask generation for the v2 study (PLAN.md phase (c)).

PLAN.md:239 notes that testopus/make_masks.py cannot emit thumbout. Rather than edit it
-- PLAN.md:221 requires that no existing file be modified -- this imports its geometry
and segmentation wholesale and adds the third branch on top.

Classes follow PLAN.md:27 -- 200 thumbout, 100 openhand, 255 closedhand, 0 background.

Output (default): v2-study/photov2_annotate/
    <stem>_mask.tif        uint8 class mask, same HxW as the photo
    overlays/<stem>_overlay.jpg   alpha=0.5 overlay for eyeball QA
    predictions.csv        per-hand class + the geometric features behind it

Usage:
    python3 v2-study/scripts/make_masks_v2.py
    python3 v2-study/scripts/make_masks_v2.py --photos DIR --out DIR
"""

import os
import sys
import csv
import glob
import argparse
import importlib.util
import collections

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import mediapipe as mp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(REPO, "v2-study")
DEFAULT_PHOTOS = os.path.join(V2, "photov2")
DEFAULT_OUT = os.path.join(V2, "photov2_annotate")

# PLAN.md:27
CLASS_VALUES = {"thumbout": 200, "openhand": 100, "closedhand": 255}
OVERLAY_COLOURS = {"thumbout": (255, 128, 0), "openhand": (0, 0, 255), "closedhand": (0, 255, 0)}


def _load_v1():
    """Import testopus/make_masks.py by path -- it is not a package, and per PLAN.md:221
    it must not be modified, so its segmentation is reused exactly as written."""
    path = os.path.join(REPO, "testopus", "make_masks.py")
    spec = importlib.util.spec_from_file_location("make_masks_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V1 = _load_v1()

WRIST, THUMB_MCP, THUMB_IP, THUMB_TIP = 0, 2, 3, 4
INDEX_MCP, PINKY_MCP, MIDDLE_MCP = 5, 17, 9
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def features(pts):
    """Geometric descriptors, all normalised by palm length so they are scale-free."""
    palm = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])) or 1.0

    # A finger counts as extended when its tip is clearly further from the wrist than
    # its own PIP joint -- curling pulls the tip back toward the wrist.
    extended = sum(
        1
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        if np.linalg.norm(pts[tip] - pts[WRIST]) > np.linalg.norm(pts[pip] - pts[WRIST]) + 0.15 * palm
    )

    # Thumb: extended when the tip stands off the palm. Measured against the index MCP
    # (the knuckle it tucks against in a fist) and cross-checked with straightness at
    # the IP joint, so a thumb folded across the fingers does not read as out.
    tip_from_index = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_MCP])) / palm
    span = float(np.linalg.norm(pts[THUMB_TIP] - pts[PINKY_MCP])) / palm
    a, b = pts[THUMB_IP] - pts[THUMB_MCP], pts[THUMB_TIP] - pts[THUMB_IP]
    cos = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6))
    return dict(palm=palm, extended=extended, tip_from_index=tip_from_index,
                span=span, thumb_straight=cos)


def classify3(f):
    """openhand | thumbout | closedhand.

    Order matters: thumbout and closedhand share a curled-finger posture and differ
    only in the thumb, so fingers are tested first and the thumb only breaks the tie.
    """
    if f["extended"] >= 3:
        return "openhand"
    thumb_out = (f["tip_from_index"] > 0.62 and f["thumb_straight"] > 0.55) or f["span"] > 1.25
    if f["extended"] <= 1 and thumb_out:
        return "thumbout"
    if f["extended"] == 2:
        # Two fingers up is off-protocol for this study; fall to the thumb signal.
        return "openhand" if thumb_out else "closedhand"
    return "closedhand"


def enhance(img):
    """Lift shadows via CLAHE on L. Many of these photos are shot into a window, which
    leaves one hand a near-black silhouette: the landmarker misses it entirely, and
    GrabCut has no contrast to cut on and returns a sliver of the true hand."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _detect(hands, bgr):
    res = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return res.hand_landmarks or []


def hull_fill(mask, pts):
    """Mask area as a fraction of the landmark convex hull. A hand fills most of its own
    hull -- roughly 0.5-0.7 spread open, more for a fist -- so a much lower value means
    GrabCut returned a sliver rather than the hand."""
    hull = cv2.convexHull(pts.astype(np.int32))
    area = cv2.contourArea(hull)
    return (float(mask.sum()) / area) if area > 1 else 0.0


def segment_best(img, enhanced, pts):
    """Segment on the original image.

    An earlier version re-cut on the shadow-lifted copy whenever `hull_fill` looked low.
    That was dropped: on the backlit frames the *landmarks* are wrong, not the cut, so
    the hull shrinks along with the mask and fill stays in its normal range (the worst
    mask in this set scores 1.13, above the median). Fill therefore cannot separate
    those failures, and thresholding it only churned good frames. Those photos are
    caught by the alignment check in overlay_check.py and annotated by hand instead.
    """
    m = V1.clean(V1.cut_at_wrist(V1.segment(img, pts), pts), pts)
    return m, hull_fill(m, pts), 0


def process(path, hands, out_dir, ov_dir):
    img = cv2.imread(path)
    if img is None:
        return [], f"unreadable"
    h, w = img.shape[:2]

    landmarks = _detect(hands, img)
    boosted = img if len(landmarks) >= 2 else enhance(img)
    recovered = 0
    if len(landmarks) < 2:
        # Retry on the shadow-lifted copy and keep any hand the first pass missed.
        found = _detect(hands, boosted)
        have = [np.array([[lm.x * w, lm.y * h] for lm in l], np.float32).mean(0)
                for l in landmarks]
        for cand in found:
            c = np.array([[lm.x * w, lm.y * h] for lm in cand], np.float32).mean(0)
            if all(np.linalg.norm(c - p) > 0.05 * w for p in have):
                landmarks.append(cand)
                have.append(c)
                recovered += 1

    if not landmarks:
        return [], "no hands detected"

    stem = os.path.splitext(os.path.basename(path))[0]
    mask = np.zeros((h, w), np.uint8)
    overlay = img.copy()
    rows = []
    enhanced = enhance(img)

    for i, lms in enumerate(landmarks):
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms], np.float32)
        f = features(pts)
        label = classify3(f)

        m, fill, used_enhanced = segment_best(img, enhanced, pts)
        mask[m > 0] = CLASS_VALUES[label]
        overlay[m > 0] = OVERLAY_COLOURS[label]

        side = "left" if pts[:, 0].mean() < w / 2 else "right"   # image-space side
        rows.append(dict(stem=stem, hand=i, img_side=side, cls=label,
                         area=int(m.sum()), extended=f["extended"],
                         tip_from_index=round(f["tip_from_index"], 3),
                         span=round(f["span"], 3),
                         thumb_straight=round(f["thumb_straight"], 3),
                         fill=round(fill, 3), enhanced=used_enhanced))

    cv2.imwrite(os.path.join(out_dir, stem + "_mask.tif"), mask)
    blend = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    for label, colour in OVERLAY_COLOURS.items():
        binary = np.uint8(mask == CLASS_VALUES[label]) * 255
        if binary.any():
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blend, cnts, -1, colour, 4)
    cv2.imwrite(os.path.join(ov_dir, stem + "_overlay.jpg"), blend,
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    note = "" if len(rows) == 2 else f"{len(rows)} hand(s), expected 2"
    if recovered:
        note = (note + "; " if note else "") + f"{recovered} hand recovered via shadow-lift"
    return rows, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default=DEFAULT_PHOTOS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    # Unusual poses (a fist rotated so the thumb points down) fall below the 0.3 default and the
    # hand is silently dropped -- which is the one failure mode that actively harms training,
    # since an unlabelled hand teaches the detector that a hand is background. Lowering this
    # recovers such hands, at the cost of duplicate/unstable detections that mask_qa.py will flag.
    ap.add_argument("--min-confidence", type=float, default=0.3,
                    help="MediaPipe hand-detection threshold (default 0.3)")
    ap.add_argument("--max-hands", type=int, default=2)
    args = ap.parse_args()

    ov_dir = os.path.join(args.out, "overlays")
    os.makedirs(ov_dir, exist_ok=True)

    photos = sorted(
        (p for ext in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(args.photos, ext))),
        key=os.path.getmtime,          # capture order, to line up with shotlist.csv
    )
    if not photos:
        print(f"no photos in {args.photos}")
        return

    model = next((m for m in V1.MODEL_CANDIDATES if os.path.exists(m)), None)
    if model is None:
        print("hand_landmarker.task not found")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=RunningMode.IMAGE, num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_confidence,
    )

    all_rows, problems = [], []
    with HandLandmarker.create_from_options(options) as hands:
        for idx, p in enumerate(photos, 1):
            rows, note = process(p, hands, args.out, ov_dir)
            for r in rows:
                r["order"] = idx
            all_rows += rows
            if note:
                problems.append((os.path.basename(p), note))

    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["order", "stem", "hand", "img_side", "cls", "area",
                                           "extended", "tip_from_index", "span", "thumb_straight",
                                           "fill", "enhanced"])
        w.writeheader()
        w.writerows(all_rows)

    counts = collections.Counter(r["cls"] for r in all_rows)
    print(f"{len(photos)} photos -> {len(all_rows)} annotations")
    print(f"  class balance: {dict(counts)}   (PLAN.md target: 40 per class)")
    print(f"  masks:    {args.out}/")
    print(f"  overlays: {ov_dir}/")
    if problems:
        print(f"  !! {len(problems)} photo(s) not giving exactly 2 hands:")
        for name, note in problems:
            print(f"     {name}: {note}")


if __name__ == "__main__":
    main()
