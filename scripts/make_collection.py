#!/usr/bin/env python3
"""Generate the collection plan: environments.csv, shotlist.csv, shot_list.xlsx.

Deterministic and reproducible -- rerun it any time and you get byte-identical CSVs.
Nothing here reads or writes the retired v1 study.

DESIGN (see ../PLAN.md)
  12 places, one lighting each, 3 per group (fold0/fold1/fold2/test).
  Each group holds one place at each of three fixed SLOTS, so the four groups match on
  every nuisance variable and differ only in which physical place fills each slot.

    S1  diagonal            3 photos   55 cm   0 deg   daylight diffuse
    S2  cycleA + cycleB     6 photos   65 cm  -15 deg  artificial hard
    S3  cycleA + cycleB     6 photos   75 cm  +20 deg  artificial soft/dim

  A balanced TRIPLET is 3 photos giving exactly 2 annotations of each class, one per side.
  Per group: 1 diagonal + 2 cycleA + 2 cycleB = 5 triplets = 15 photos = 30 annotations
             = 10 per class = 5 left + 5 right per class.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTION = HERE.parent / "collection"

# --- classes -----------------------------------------------------------------------------
# mask grayscale -> class index, matching the retired pipeline's convention exactly
THUMB, OPEN, CLOSED = "thumbout", "openhand", "closedhand"
CLASS_IDX = {THUMB: 0, OPEN: 1, CLOSED: 2}
CLASS_CODE = {THUMB: "T", OPEN: "O", CLOSED: "C"}
MASK_VALUE = {THUMB: 200, OPEN: 100, CLOSED: 255}

# --- the three balanced triplets ---------------------------------------------------------
# Each is 3 (left, right) pairs. Every triplet yields exactly 2 of each class,
# one on each side. Verified by assertions at the bottom.
CYCLE_A = [(OPEN, CLOSED), (CLOSED, THUMB), (THUMB, OPEN)]
CYCLE_B = [(OPEN, THUMB), (THUMB, CLOSED), (CLOSED, OPEN)]
DIAGONAL = [(OPEN, OPEN), (CLOSED, CLOSED), (THUMB, THUMB)]

# Hand rotation in the IMAGE PLANE (palm stays parallel to the sensor).
# For the 6-photo slots this pattern gives 2 photos per rotation AND two distinct
# classes at every rotation on BOTH sides -- so rotation never predicts class.
SIX_ROTATION = [0, 30, 60, 60, 30, 0]

# For the 3-photo diagonal slot, rotation would otherwise be perfectly confounded with
# class, so the class->rotation mapping is rotated between groups (a Latin square over
# the three training groups; test reuses one of them).
DIAGONAL_ORDER = {
    "fold0": [OPEN, CLOSED, THUMB],
    "fold1": [CLOSED, THUMB, OPEN],
    "fold2": [THUMB, OPEN, CLOSED],
    "test": [OPEN, THUMB, CLOSED],
}

# --- slot definitions --------------------------------------------------------------------
SLOTS = {
    "S1": dict(
        triplets=["diagonal"], n_photos=3,
        camera_distance_cm=55, camera_tilt_deg=0,
        light_class="daylight", light_hardness="soft", lux_target=800,
        light_source="diffuse window daylight (overcast or shaded), no lamp on",
        sleeve="short sleeves, forearms bare", accessory="none", accessory_side="",
        difficulty="easy",
    ),
    "S2": dict(
        triplets=["cycleA", "cycleB"], n_photos=6,
        camera_distance_cm=65, camera_tilt_deg=-15,
        light_class="artificial", light_hardness="hard", lux_target=300,
        light_source="ceiling light only, curtains or blinds shut",
        sleeve="long sleeves, striped or patterned", accessory="wristwatch",
        accessory_side="image-left", difficulty="medium",
    ),
    "S3": dict(
        triplets=["cycleA", "cycleB"], n_photos=6,
        camera_distance_cm=75, camera_tilt_deg=20,
        light_class="artificial", light_hardness="soft", lux_target=60,
        light_source="one dim lamp only, no ceiling light",
        sleeve="long dark sleeves or hoodie cuffs", accessory="bracelet",
        accessory_side="image-right", difficulty="hard",
    ),
}
SLOT_ORDER = ["S1", "S2", "S3"]
GROUPS = ["fold0", "fold1", "fold2", "test"]

# --- the 12 places -----------------------------------------------------------------------
# Rooms are TO CONFIRM against the actual house. All 12 must be distinct, and the three
# test rooms must not appear anywhere in training. Swap a room freely; keep the slot's
# distance / tilt / lighting / sleeve / accessory exactly as specified.
PLACES = [
    # (group, slot, room, background_desc, session_block, time_of_day)
    ("fold0", "S1", "home office desk",   "plain wall behind the desk",            "A-day",   "daytime"),
    ("fold0", "S2", "kitchen",            "tiled splashback or cupboard fronts",   "A-night", "evening"),
    ("fold0", "S3", "bedroom",            "duvet and folded fabric",               "A-night", "evening"),

    ("fold1", "S1", "living room",        "plain painted wall",                    "B-day",   "daytime"),
    ("fold1", "S2", "hallway",            "closed internal door, flat panel",      "B-night", "evening"),
    ("fold1", "S3", "dining room",        "wooden table and chair backs",          "B-night", "evening"),

    ("fold2", "S1", "spare room",         "bookshelf with book spines (busy)",     "C-day",   "daytime"),
    ("fold2", "S2", "landing or stairs",  "stair wall and banister",               "C-night", "evening"),
    ("fold2", "S3", "entryway",           "coat rack with hanging coats",          "C-night", "evening"),

    ("test",  "S1", "balcony or doorway", "outdoor greenery, depth behind hands",  "X-day",   "daytime"),
    ("test",  "S2", "bathroom",           "white wall tiles, no mirror in frame",  "X-night", "evening"),
    ("test",  "S3", "garage or utility",  "unfinished wall and shelving",          "X-night", "evening"),
]

SUBJECT = "subj01"


def build():
    """Return (environments, shots) as lists of dicts."""
    environments, shots = [], []
    photo_n = 0

    for i, (group, slot, room, background, session, tod) in enumerate(PLACES, start=1):
        env_id = f"E{i:02d}"
        spec = SLOTS[slot]

        # ---- assemble this environment's (left, right) pairs and rotations ----
        if slot == "S1":
            order = DIAGONAL_ORDER[group]
            pairs = [(c, c) for c in order]
            rotations = [0, 30, 60]
            triplet_of = ["diagonal"] * 3
        else:
            pairs = CYCLE_A + CYCLE_B
            rotations = SIX_ROTATION
            triplet_of = ["cycleA"] * 3 + ["cycleB"] * 3

        first_photo = photo_n + 1
        for seq, ((left, right), rot, ttype) in enumerate(zip(pairs, rotations, triplet_of), start=1):
            photo_n += 1
            photo_id = f"P{photo_n:02d}"
            shots.append({
                "photo_id": photo_id,
                "env_id": env_id,
                "slot": slot,
                "seq_in_env": seq,
                "filename": f"{env_id}_{photo_id}.jpg",
                "group": group,
                "triplet_type": ttype,
                "left_class": left,
                "right_class": right,
                "left_code": CLASS_CODE[left],
                "right_code": CLASS_CODE[right],
                # ORDERED left-then-right. mask_instances() sorts blobs by centroid x, so
                # the label file's line order IS left-then-right and a swap is detectable.
                "expected_label_sequence": f"{CLASS_IDX[left]} {CLASS_IDX[right]}",
                "hand_rotation_deg": rot,
                "annotations_expected": 2,
                "captured": "", "mask_done": "", "qa_pass": "", "notes": "",
            })

        environments.append({
            "env_id": env_id,
            "group": group,
            "slot": slot,
            "triplet_types": "+".join(spec["triplets"]),
            "n_photos": spec["n_photos"],
            "photo_ids": f"P{first_photo:02d}-P{photo_n:02d}",
            "room": room,
            "background_desc": background,
            "light_class": spec["light_class"],
            "light_hardness": spec["light_hardness"],
            "light_source": spec["light_source"],
            "lux_target": spec["lux_target"],
            "camera_distance_cm": spec["camera_distance_cm"],
            "camera_tilt_deg": spec["camera_tilt_deg"],
            "sleeve": spec["sleeve"],
            "accessory": spec["accessory"],
            "accessory_side": spec["accessory_side"],
            "difficulty": spec["difficulty"],
            "session_block": session,
            "time_of_day": tod,
            "subject_id": SUBJECT,
            "room_confirmed": "",
            "notes": "TEST - room must not appear in training" if group == "test" else "",
        })

    return environments, shots


# ---------------------------------------------------------------------------------------
# Self-checks. These duplicate design_asserts.py deliberately: a generator that cannot
# verify its own output is not worth trusting.
# ---------------------------------------------------------------------------------------
def verify(environments, shots):
    from collections import Counter, defaultdict

    assert len(shots) == 60, len(shots)
    assert len(environments) == 12, len(environments)
    assert [s["photo_id"] for s in shots] == [f"P{i:02d}" for i in range(1, 61)]
    assert len({s["filename"] for s in shots}) == 60

    for triplet in (CYCLE_A, CYCLE_B, DIAGONAL):
        c = Counter()
        for l, r in triplet:
            c[l] += 1
            c[r] += 1
        assert set(c.values()) == {2}, triplet

    per_group = defaultdict(Counter)
    per_side = defaultdict(Counter)
    per_env = defaultdict(Counter)
    rooms = defaultdict(set)
    sessions = defaultdict(set)

    for s in shots:
        g = s["group"]
        per_group[g][s["left_class"]] += 1
        per_group[g][s["right_class"]] += 1
        per_side[(g, "L")][s["left_class"]] += 1
        per_side[(g, "R")][s["right_class"]] += 1
        per_env[s["env_id"]][s["left_class"]] += 1
        per_env[s["env_id"]][s["right_class"]] += 1

    for g in GROUPS:
        assert sum(per_group[g].values()) == 30, (g, per_group[g])
        assert set(per_group[g].values()) == {10}, (g, per_group[g])
        assert set(per_side[(g, "L")].values()) == {5}, (g, per_side[(g, "L")])
        assert set(per_side[(g, "R")].values()) == {5}, (g, per_side[(g, "R")])

    for env_id, counts in per_env.items():
        assert len(set(counts.values())) == 1, (env_id, counts)

    for e in environments:
        rooms[e["room"]].add(e["group"])
        sessions[e["session_block"]].add(e["group"])
    assert len(rooms) == 12, f"rooms not distinct: {len(rooms)}"
    for room, gs in rooms.items():
        assert len(gs) == 1, f"room {room!r} spans groups {gs}"
    for sess, gs in sessions.items():
        assert len(gs) == 1, f"session {sess!r} spans groups {gs}"

    train_rooms = {e["room"] for e in environments if e["group"] != "test"}
    test_rooms = {e["room"] for e in environments if e["group"] == "test"}
    assert not (train_rooms & test_rooms), train_rooms & test_rooms

    # rotation balance within every environment
    for env_id in {s["env_id"] for s in shots}:
        rots = Counter(s["hand_rotation_deg"] for s in shots if s["env_id"] == env_id)
        assert set(rots) == {0, 30, 60}, (env_id, rots)
        assert len(set(rots.values())) == 1, (env_id, rots)

    # slot profile identical across all four groups
    for slot in SLOT_ORDER:
        rows = [e for e in environments if e["slot"] == slot]
        assert len(rows) == 4
        for key in ("camera_distance_cm", "camera_tilt_deg", "lux_target",
                    "sleeve", "accessory", "difficulty", "light_class"):
            assert len({r[key] for r in rows}) == 1, (slot, key)

    totals = Counter()
    for s in shots:
        totals[s["left_class"]] += 1
        totals[s["right_class"]] += 1
    assert set(totals.values()) == {40}, totals
    return totals


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    envs, shots = build()
    totals = verify(envs, shots)
    write_csv(COLLECTION / "environments.csv", envs)
    write_csv(COLLECTION / "shotlist.csv", shots)
    print(f"environments.csv  {len(envs)} rows")
    print(f"shotlist.csv      {len(shots)} rows")
    print(f"annotations       {sum(totals.values())}  " +
          "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    print("all design assertions passed")
