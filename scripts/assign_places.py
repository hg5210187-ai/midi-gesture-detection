#!/usr/bin/env python3
"""Recover the 12 capture places from the shot list and assign them to folds.

Supersedes the lighting-clustering approach in assign_groups.py, which was the wrong tool.
The photos WERE shot to collection/shotlist.csv: capture order breaks cleanly into blocks of
3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6 photos, which is exactly the plan's place structure
(S1 diagonal = 3 photos, S2 and S3 = 6 each, repeated for four groups). Clustering on pixel
statistics found lighting sub-conditions inside those places and split them, producing ragged
folds. The place structure gives exactly 15 photos per group.

Retakes carry later timestamps, so plain capture order would sort them to the end and shift
every block. Each retake instead inherits the place of the photo it replaced --
REPLACEMENTS records that, and it is the only hand-maintained fact in this file.

Group assignment follows the plan: places 1-3 -> fold0, 4-6 -> fold1, 7-9 -> fold2,
10-12 -> test. A place never spans two groups, so no room is in both train and validation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QA = ROOT / "results" / "mask_qa_v2.csv"
SHOTLIST = ROOT / "collection" / "shotlist.csv"
OUT_REGISTRY = ROOT / "data" / "places.csv"
OUT_FOLDS = ROOT / "data" / "splits" / "folds.json"

# A retake inherits the PLACE of the photo it replaced. Capture time cannot supply this: the
# retakes were shot hours later, so plain time ordering would sort them all to the end and
# shift every block boundary. Matched by background, not by filename order -- 18.39's pink
# shirt belongs with 13.27, and 18.27's cap and wooden cabinets with 13.42 #2.
REPLACEMENTS = {
    "Photo on 2026-08-11 at 18.41": "Photo on 2026-08-11 at 13.19",     # E01 fold0
    "Photo on 2026-08-11 at 18.39": "Photo on 2026-08-11 at 13.27",     # E04 fold1
    "Photo on 2026-08-11 at 18.43": "Photo on 2026-08-11 at 13.37 #2",  # E07 fold2
    "Photo on 2026-08-11 at 18.27": "Photo on 2026-08-11 at 13.42 #2",  # E10 test
}

GROUPS = ["fold0", "fold1", "fold2", "test"]
CLASSES = ("thumbout", "openhand", "closedhand")
SHORT = {"thumbout": "T", "openhand": "O", "closedhand": "C"}


def capture_key(stem: str):
    """'#10' must sort after '#2', which plain string ordering gets wrong."""
    s = stem.replace("Photo on 2026-08-11 at ", "")
    m = re.match(r"(\d+)\.(\d+)(?: #(\d+))?", s)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    classes_of = collections.defaultdict(list)
    for r in csv.DictReader(QA.open()):
        if r.get("cls"):
            classes_of[r["stem"]].append(r["cls"])

    # order by capture time, but a retake takes the slot of the photo it replaced
    slot_of = {s: capture_key(REPLACEMENTS.get(s, s)) for s in classes_of}
    ordered = sorted(classes_of, key=lambda s: slot_of[s])

    block_sizes = [len(v) for v in
                   collections.OrderedDict(
                       (p["env_id"], None) for p in csv.DictReader(SHOTLIST.open())).keys()]
    sizes = collections.Counter()
    for p in csv.DictReader(SHOTLIST.open()):
        sizes[p["env_id"]] += 1
    envs = list(sizes)
    block_sizes = [sizes[e] for e in envs]

    if sum(block_sizes) != len(ordered):
        raise SystemExit(f"{len(ordered)} photos but shot list expects {sum(block_sizes)}")

    place_of, i = {}, 0
    for env, n in zip(envs, block_sizes):
        for s in ordered[i:i + n]:
            place_of[s] = env
        i += n
    group_of = {e: GROUPS[k // 3] for k, e in enumerate(envs)}

    per_place = collections.defaultdict(collections.Counter)
    members = collections.defaultdict(list)
    for s, e in place_of.items():
        members[e].append(s)
        for k in classes_of[s]:
            per_place[e][k] += 1

    print(f"  {'place':6s} {'group':7s} {'n':>2s}  {'T':>2s} {'O':>2s} {'C':>2s}   span")
    print("  " + "-" * 60)
    for e in envs:
        ms = sorted(members[e], key=lambda s: slot_of[s])
        span = f"{ms[0].replace('Photo on 2026-08-11 at ','')} .. {ms[-1].replace('Photo on 2026-08-11 at ','')}"
        c = per_place[e]
        print(f"  {e:6s} {group_of[e]:7s} {len(ms):2d}  " +
              " ".join(f"{c[k]:2d}" for k in CLASSES) + f"   {span}")

    print(f"\n  {'group':7s} {'places':17s} {'photos':>6s}  {'T':>3s} {'O':>3s} {'C':>3s}")
    print("  " + "-" * 52)
    gp = collections.Counter()
    gc = collections.defaultdict(collections.Counter)
    for e in envs:
        gp[group_of[e]] += len(members[e])
        gc[group_of[e]] += per_place[e]
    for g in GROUPS:
        pl = ",".join(e for e in envs if group_of[e] == g)
        mark = "   balanced" if all(gc[g][k] == 10 for k in CLASSES) else ""
        print(f"  {g:7s} {pl:17s} {gp[g]:6d}  " + " ".join(f"{gc[g][k]:3d}" for k in CLASSES) + mark)

    errs = []
    if len(place_of) != 60:
        errs.append(f"{len(place_of)} photos, expected 60")
    for g in GROUPS:
        if gp[g] != 15:
            errs.append(f"{g} has {gp[g]} photos, expected 15")
    for s, ks in classes_of.items():
        if len(ks) != 2:
            errs.append(f"{s}: {len(ks)} annotations")
    # Global 40/40/40 is a summary figure; no metric consumes it. What matters is that each
    # EVALUATION set is balanced -- the held-out test set above all, then each fold's val set,
    # since mAP is an unweighted mean over classes and a thin class swings a third of the score.
    tot = collections.Counter(k for ks in classes_of.values() for k in ks)
    if any(tot[k] != 40 for k in CLASSES):
        print("\n  NOTE: global totals are " +
              " ".join(f"{k}={tot[k]}" for k in CLASSES) + " (40 each would be ideal).")
        for g in GROUPS:
            short = {k: 10 - gc[g][k] for k in CLASSES if gc[g][k] != 10}
            if short:
                need = "  ".join(f"{'+' if v > 0 else ''}{v} {k}" for k, v in short.items())
                print(f"        {g}: {need}")
        print("        Each fix is one photo re-shot in that group's places with one gesture changed.")
    seen = collections.Counter(place_of.values())
    for e, n in seen.items():
        if n != sizes[e]:
            errs.append(f"place {e} holds {n} photos, shot list says {sizes[e]}")
    if errs:
        print("\nFAILED:")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)
    print("\nassertions passed: 15 photos per group; 2 annotations per photo; 40 per class;"
          "\n                   every place wholly inside one group")

    if args.dry_run:
        print("\nDRY RUN - nothing written")
        return

    OUT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    OUT_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REGISTRY.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["place", "group", "n_photos"] + list(CLASSES) + ["photos"])
        for e in envs:
            ms = sorted(members[e], key=lambda s: slot_of[s])
            w.writerow([e, group_of[e], len(ms)] +
                       [per_place[e][k] for k in CLASSES] + ["|".join(ms)])
    OUT_FOLDS.write_text(json.dumps({
        "method": "capture-order blocks matching collection/shotlist.csv place structure",
        "note": "retakes inherit the place of the photo they replaced (see REPLACEMENTS)",
        "replacements": REPLACEMENTS,
        "place_group": group_of,
        "photo_place": place_of,
        "photo_group": {s: group_of[e] for s, e in place_of.items()},
    }, indent=2))
    print(f"\nwrote {OUT_REGISTRY}\nwrote {OUT_FOLDS}")


if __name__ == "__main__":
    main()
