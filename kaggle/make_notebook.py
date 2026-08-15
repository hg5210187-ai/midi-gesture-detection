#!/usr/bin/env python3
"""Generate the Kaggle notebook, embedding the driver scripts.

The scripts are embedded via %%writefile rather than shipped as a third Kaggle Dataset. That
keeps the notebook self-contained -- no dataset-version drift between the notebook and the
code it runs, which is the failure mode where you fix a bug locally and Kaggle silently keeps
running last week's version. Regenerate with this script after editing any driver so the
notebook cannot fall out of sync.

    python3 kaggle/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "midi_gesture_kaggle.ipynb"
USER = "shumahara"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().split("\n")}


def embed(name):
    body = (HERE / name).read_text()
    return code(f"%%writefile /kaggle/working/{name}\n{body}")


cells = [
md(f"""
# MIDI gesture study — Kaggle training

60 photos · 120 hand annotations · 3 classes · exactly 40 per class.
Four groups of 15 photos at 10/10/10: `fold0`, `fold1`, `fold2`, `test`.
A capture place never spans two groups, so no background appears in both train and validation.

**Two arms**

| arm | geometry | models | runs |
|---|---|---|---|
| YOLO26 | OBB and axis-aligned | n s m l x | 30 |
| DEIMv2 | axis-aligned | atto femto pico n s m l x | 24 |

**Before running:** Accelerator = **GPU T4 x2** (not P100), Internet **on**,
Persistence = **Variables and Files** so a restart resumes instead of starting over.

The test set is evaluated **once**, at the end, on the single configuration cross-validation
selects. With 54 models and 30 test annotations, choosing on the test set would guarantee an
inflated number.
"""),

md("## 1 · Environment check\n\nThirty seconds here saves a wasted session."),
code(f"""
import torch, os, sys, json
from pathlib import Path

print("torch      ", torch.__version__)
print("cuda avail ", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Settings -> Accelerator -> GPU T4 x2"

name = torch.cuda.get_device_name(0)
print("gpu        ", name)
assert "P100" not in name, (
    "P100 selected. Kaggle's PyTorch targets sm_70+; the P100 is sm_60, so the CUDA kernels "
    "are missing and nothing will train. Switch to GPU T4 x2.")

# prove kernels actually run, rather than trusting the device name
assert float((torch.zeros(8, device="cuda") + 1).sum()) == 8
print("cuda kernels OK")

import torchvision; print("torchvision", torchvision.__version__)

# Kaggle mounts inputs differently depending on how they were attached: classic datasets
# land at /kaggle/input/<slug>/, Data Hub at /kaggle/input/datasets/<user>/<slug>/<slug>/.
# Search for the marker files rather than assuming either.
ROOT = Path("/kaggle/input")
DATA = next((d for d in ROOT.rglob("folds.json")), None)
CKPT = next((d for d in ROOT.rglob("deimv2_hgnetv2_n_coco.pth")), None)
assert DATA is not None, f"folds.json not found under {{ROOT}}. Attach the midi-gesture-v2 dataset."
assert CKPT is not None, f"checkpoints not found under {{ROOT}}. Attach deimv2-checkpoints."
DATA, CKPT = DATA.parent, CKPT.parent
print("DATA", DATA)
print("CKPT", CKPT)
print()
!ls {{DATA}}

# these two paths feed every later cell
%store DATA
%store CKPT
"""),

md("""
## 2 · Dependencies

Kaggle's image does not ship ultralytics. Installed without touching torch: ultralytics needs
`torch>=1.8` and the image has 2.10+cu128, so pip leaves it alone. **If pip reports that it is
installing or upgrading torch, stop** -- the preinstalled build is matched to the driver and
replacing it breaks CUDA.
"""),
code("""
!pip install -q ultralytics
import ultralytics, torch
print("ultralytics", ultralytics.__version__)
print("torch      ", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch lost CUDA - the pip install replaced it"
"""),

md("## 3 · Write the driver scripts"),
embed("train_yolo.py"),
embed("setup_deimv2.py"),
embed("run_deim.py"),

md("""
## 4 · YOLO26 — 30 runs

Both geometries, five sizes, three folds. Appends to `results.jsonl` after every run, so an
interrupted session resumes rather than restarting. Re-run this cell as often as you like.
"""),
code("""
!cd /kaggle/working && python train_yolo.py \\
    --data {DATA} --out /kaggle/working --epochs 100 --imgsz 640 --batch 8
"""),

md("""
### 4b · Resolution sweep (optional)

Accuracy at 320 and 416 for the smallest model, so the latency arm has matching accuracy
numbers. Input size is the largest single lever on inference time.
"""),
code("""
!cd /kaggle/working && python train_yolo.py \\
    --data {DATA} --out /kaggle/working --sizes n --epochs 100 --imgsz-sweep
"""),

md("""
## 5 · DEIMv2 — setup

Clones the repo, patches the torchvision v2 transform API if needed, stages a COCO tree under
`/kaggle/working` (the input mount is read-only), and writes 24 configs.

**`requirements.txt` is deliberately not installed** — it pins `torch==2.5.1`, which would
replace Kaggle's CUDA-matched build and break every GPU operation.
"""),
code("""
!cd /kaggle/working && python setup_deimv2.py \\
    --data {DATA} --ckpt {CKPT} --work /kaggle/working --epochs 120 --batch 4
"""),

md("""
### 5b · Pre-flight

DEIMv2's COCO defaults give an **empty dataloader** at 30 images — `total_batch_size 32` with
`drop_last: True` is `floor(30/32) = 0` batches — and training then does nothing while still
printing epochs and exiting cleanly. This checks every config before any run starts.
"""),
code("""
!cd /kaggle/working && python run_deim.py \\
    --repo /kaggle/working/DEIMv2 --configs /kaggle/working/deim_configs --check-only
"""),

md("### 5c · DEIMv2 — 24 runs\n\nStart with `pico` and `n`; the DINOv3 variants are larger and slower."),
code("""
!cd /kaggle/working && python run_deim.py \\
    --repo /kaggle/working/DEIMv2 --configs /kaggle/working/deim_configs \\
    --out /kaggle/working --variants pico n
"""),
code("""
# the rest, once the first two are known good
!cd /kaggle/working && python run_deim.py \\
    --repo /kaggle/working/DEIMv2 --configs /kaggle/working/deim_configs --out /kaggle/working
"""),

md("## 6 · Results\n\nCross-validation reported as mean ± SD across the three folds (ddof=1)."),
code("""
import json, statistics, collections
from pathlib import Path

rows = []
p = Path("/kaggle/working/results.jsonl")
if p.exists():
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))

ok = [r for r in rows if "error" not in r]
print(f"{len(ok)} successful runs, {len(rows)-len(ok)} failed\\n")

agg = collections.defaultdict(list)
for r in ok:
    agg[(r["geom"], r["size"], r["imgsz"])].append(r["mAP50-95"])

print(f"{'geom':5s} {'size':5s} {'imgsz':>6s} {'n':>2s} {'mAP50-95':>10s} {'SD':>8s}")
print("-" * 46)
for k in sorted(agg):
    v = agg[k]
    sd = statistics.stdev(v) if len(v) > 1 else 0.0   # ddof=1: with n=3 the population SD
    print(f"{k[0]:5s} {k[1]:5s} {k[2]:6d} {len(v):2d} "   # understates the spread by ~18%
          f"{statistics.mean(v):10.4f} {sd:8.4f}")

d = Path("/kaggle/working/results_deim.jsonl")
if d.exists():
    print("\\nDEIMv2")
    dag = collections.defaultdict(list)
    for line in d.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "ap50_95" in r:
                dag[r["variant"]].append(r["ap50_95"])
    for k in sorted(dag):
        v = dag[k]
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"  {k:6s} n={len(v)}  AP50-95 {statistics.mean(v):.4f} +/- {sd:.4f}")
"""),

md("""
## 7 · Collect the outputs

Checkpoints come home so latency can be measured on the target laptop — a datacentre GPU
answers a question nobody asked.
"""),
code("""
import shutil, os
from pathlib import Path
out = Path("/kaggle/working/collected"); out.mkdir(exist_ok=True)
n = 0
for w in Path("/kaggle/working/runs").rglob("weights/best.pt"):
    dst = out / f"{w.parent.parent.name}.pt"
    shutil.copy2(w, dst); n += 1
for f in ("results.jsonl", "results_deim.jsonl"):
    p = Path("/kaggle/working") / f
    if p.exists():
        shutil.copy2(p, out / f)
print(f"{n} checkpoints + results -> {out}")
print(f"total {sum(f.stat().st_size for f in out.rglob('*') if f.is_file())/1e6:.1f} MB")
"""),

md(f"""
## Notes

**Use Save Version → Save & Run All (Commit)** for the full sweep. Interactive sessions die at
12 hours and also when the browser disconnects; a committed run survives both.

**Download afterwards:**
```
kaggle kernels output {USER}/<notebook-slug> -p ./results
```

**If you re-upload the dataset**, bump the version in the Input panel. The notebook otherwise
keeps using the version it was attached to, with no warning — a notebook quietly training on
last week's labels is the most common Kaggle mistake.

**Not done here:** the 4x4 confusion matrix, per-hand AUC-ROC, and Core ML export. Those run
locally on the M4 against the checkpoints collected above, because the latency question is
about that machine.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4, "nbformat_minor": 4,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}")
print(f"  {len(cells)} cells, {OUT.stat().st_size/1024:.0f} KB")
print("  embedded: train_yolo.py, setup_deimv2.py, run_deim.py")
