#!/usr/bin/env python3
"""Render the 60-photo shooting plan as a chart you can hold while photographing.

WHY THIS EXISTS. collection/shotlist.csv holds the plan, but GitHub renders it as a 60x18
table and the .xlsx does not render at all. Neither is usable at the moment of taking a
photo, which is the only moment the plan matters. This turns it into one page: every place,
every photo, which gesture on which hand, at what rotation.

READ IT AS: one row per place, one cell per photo. Each cell is split left|right for the two
hands, coloured by gesture, and ALWAYS lettered T/O/C -- colour never carries identity on its
own, which keeps it readable for colour-blind users and when printed in greyscale.

Palette is the validated categorical set (slots 1-3), which clears CVD and normal-vision
separation on the all-pairs list in both light and dark modes. Do NOT substitute the colours
used in the confusion-matrix figures: that trio fails the chroma floor and lands in the 6-8
CVD warn band.

    python3 scripts/shooting_chart.py            # writes both light and dark PNG
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GESTURE = {
    "thumbout":   {"light": "#2a78d6", "dark": "#3987e5", "letter": "T"},
    "openhand":   {"light": "#eb6834", "dark": "#d95926", "letter": "O"},
    "closedhand": {"light": "#1baf7a", "dark": "#199e70", "letter": "C"},
}
GROUP_LABEL = {"fold0": "FOLD 0", "fold1": "FOLD 1", "fold2": "FOLD 2", "test": "TEST"}
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8984",
                  rule="#e2e1dc", band="#f2f1ec", chip="#ffffff"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8984",
                  rule="#33322f", band="#232320", chip="#1a1a19"),
}


def load():
    envs = {e["env_id"]: e for e in csv.DictReader(open(ROOT / "collection/environments.csv"))}
    shots = list(csv.DictReader(open(ROOT / "collection/shotlist.csv")))
    by_env: dict[str, list] = {}
    for s in shots:
        by_env.setdefault(s["env_id"], []).append(s)
    order = sorted(envs, key=lambda e: ["fold0", "fold1", "fold2", "test"].index(envs[e]["group"]))
    return envs, by_env, order


def render(mode: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch

    t = THEME[mode]
    envs, by_env, order = load()

    ROW_H, CELL_W, CELL_H, GAP = 1.0, 1.42, 0.62, 0.06
    META_W, CELL_PITCH = 7.45, CELL_W + 0.10
    HEAD_H, GROUP_H, FOOT_H = 1.95, 0.70, 1.30
    max_cells = max(len(v) for v in by_env.values())
    n_groups = len({e["group"] for e in envs.values()})
    # width must hold the widest row; height must hold headers + rows + group bands + footer
    fig_w = META_W + max_cells * CELL_PITCH + 0.45
    fig_h = HEAD_H + n_groups * GROUP_H + ROW_H * len(order) + FOOT_H
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    ax.text(0.35, fig_h - 0.52, "Photo shooting plan — 60 photographs",
            fontsize=17, fontweight="bold", color=t["ink"], va="center")
    ax.text(0.35, fig_h - 0.87,
            "12 places × 3 groups + held-out test · both hands in every frame · "
            "2 annotations per photo",
            fontsize=9.5, color=t["ink2"], va="center")

    y = fig_h - 1.32
    lx = 0.35
    for name, d in GESTURE.items():
        ax.add_patch(FancyBboxPatch((lx, y - 0.11), 0.27, 0.27,
                     boxstyle="round,pad=0,rounding_size=0.05",
                     facecolor=d[mode], edgecolor="none"))
        ax.text(lx + 0.135, y + 0.025, d["letter"], fontsize=9.5, fontweight="bold",
                color="#ffffff", ha="center", va="center")
        ax.text(lx + 0.36, y + 0.025, name, fontsize=9.5, color=t["ink2"], va="center")
        lx += 0.36 + len(name) * 0.085 + 0.45
    ax.text(lx, y + 0.025, "cell = one photo, split  left hand │ right hand",
            fontsize=9, color=t["ink3"], style="italic", va="center")

    y -= 0.55
    last_group = None
    for env_id in order:
        e = envs[env_id]
        if e["group"] != last_group:
            y -= 0.26
            ax.text(0.35, y, GROUP_LABEL[e["group"]], fontsize=10.5, fontweight="bold",
                    color=t["ink"], va="center")
            if e["group"] == "test":
                ax.text(1.55, y, "— these 3 places must not appear in training",
                        fontsize=8.8, color=t["ink3"], style="italic", va="center")
            y -= 0.44
            last_group = e["group"]

        ax.add_patch(Rectangle((0.30, y - 0.40), fig_w - 0.62, 0.80,
                     facecolor=t["band"], edgecolor="none", zorder=0))
        ax.text(0.46, y + 0.13, env_id, fontsize=11, fontweight="bold", color=t["ink"], va="center")
        ax.text(0.46, y - 0.16, e["slot"], fontsize=8.5, color=t["ink3"], va="center")
        ax.text(1.05, y + 0.13, e["room"], fontsize=9.5, color=t["ink"], va="center")
        ax.text(1.05, y - 0.16, e["background_desc"][:40], fontsize=8.3, color=t["ink3"], va="center")
        ax.text(4.35, y + 0.13, e["difficulty"], fontsize=8.8, color=t["ink2"], va="center")
        ax.text(4.35, y - 0.16, f"{e['camera_distance_cm']} cm · {e['camera_tilt_deg']}°",
                fontsize=8.3, color=t["ink3"], va="center")
        ax.text(5.62, y + 0.13, f"{e['light_class']} · {e['light_hardness']}",
                fontsize=8.8, color=t["ink2"], va="center")
        ax.text(5.62, y - 0.16, f"{e['lux_target']} lx", fontsize=8.3,
                color=t["ink3"], va="center")

        x = META_W
        for s in by_env[env_id]:
            L, R = GESTURE[s["left_class"]], GESTURE[s["right_class"]]
            ax.add_patch(FancyBboxPatch((x, y - CELL_H / 2), CELL_W, CELL_H,
                         boxstyle="round,pad=0,rounding_size=0.07",
                         facecolor=t["chip"], edgecolor=t["rule"], lw=0.8))
            half = (CELL_W - GAP * 3) / 2
            for i, d in enumerate((L, R)):
                ax.add_patch(FancyBboxPatch((x + GAP + i * (half + GAP), y - CELL_H / 2 + 0.19),
                             half, CELL_H - 0.30,
                             boxstyle="round,pad=0,rounding_size=0.05",
                             facecolor=d[mode], edgecolor="none"))
                ax.text(x + GAP + i * (half + GAP) + half / 2, y + 0.05, d["letter"],
                        fontsize=11.5, fontweight="bold", color="#ffffff",
                        ha="center", va="center")
            ax.text(x + CELL_W / 2, y - CELL_H / 2 + 0.09,
                    f"{s['photo_id']} · {s['hand_rotation_deg']}°",
                    fontsize=7.4, color=t["ink3"], ha="center", va="center")
            x += CELL_PITCH
        y -= ROW_H

    ax.text(0.35, 0.72,
            "Rotation (0° / 30° / 60°) is the wrist angle within each triplet.   "
            "Triplets:  cycleA = O|C, C|T, T|O   ·   cycleB = O|T, T|C, C|O   ·   "
            "diagonal = O|O, C|C, T|T",
            fontsize=8.6, color=t["ink2"], va="center")
    ax.text(0.35, 0.42,
            "Every fold ends with exactly 10 instances of each gesture — the triplets are what "
            "make that true, so do not improvise the pairs.",
            fontsize=8.6, color=t["ink3"], va="center", style="italic")

    fig.savefig(out, dpi=190, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    (ROOT / "figures").mkdir(exist_ok=True)
    for m in ("light", "dark"):
        render(m, ROOT / "figures" / f"shooting_chart_{m}.png")
