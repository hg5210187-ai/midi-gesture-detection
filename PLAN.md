# MIDI-Gesture Instrument — Fresh Study Design

## Context

The goal is to establish whether a vision-based hand-gesture MIDI instrument is deployable on a
consumer laptop. The target machine is the development machine: **MacBook Air, Apple M4, 10-core,
16 GB, macOS 15.5** (`Mac16,12`), torch 2.11 with MPS, ultralytics 8.4.45.

**This is a clean-slate study.** Nothing from the previous work is reported. Two lessons carry
forward as *design constraints*, not findings:

1. **Group-aware splitting is mandatory.** The old study's openhand score was inflated from a true
   0.77 to 0.99 because burst frames from one capture minute straddled train and test
   (`data/splits/SPLIT_REPORT.md`). The new design groups whole *environments*.
2. **mAP is an unweighted mean over classes**, so a thin class swings a third of the score. The new
   dataset is exactly balanced by construction.

The previous study also stalled on latency: the smallest model measured **22.13 ms** median on MPS
at `imgsz=640` against a ~10 ms goal. Closing that gap is the study's central question — and the
levers are **input resolution and runtime**, not architecture.

## Locked decisions

| Decision | Value |
|---|---|
| Classes | `0 thumbout`, `1 openhand`, `2 closedhand` |
| Mask grayscale | `200` thumbout · `100` openhand · `255` closedhand · `0` background |
| Photos | Both hands visible → **2 annotations per photo** |
| Cross-validation | **3 folds** (`fold0/1/2`), 15 photos / 30 annotations each |
| Held-out test | 15 photos / 30 annotations, **rooms unused in training** |
| Dataset total | **60 photos · 120 annotations · 40 per class** |
| Environments | **12 places, one lighting each** — 3 per group; a place never straddles a fold |
| Labels | **Both** `labels_obb/` (minAreaRect) and `labels_hbb/` (boundingRect) |
| Models | YOLO26 `n/s/m/l/x` (OBB **and** HBB) + **DEIMv2** (Pico/N/S/M/L, anchor = N) |
| Cross-arch comparison | **On HBB only.** OBB stays a within-YOLO26 analysis |
| Input size | **640 everywhere** — YOLO26-OBB defaults to 1024; pin it to 640 or the arms drift |
| Latency sweep | `imgsz {320, 416, 640}` × runtime `{PyTorch-MPS, Core ML}` |
| Metrics | mAP50-95 (mean ± SD, ddof=1), 4×4 confusion matrix, AUC-ROC, M4 inference time |
| AUC-ROC form | Per-hand one-vs-rest, match at IoU ≥ 0.5, AUC per class + macro |
| Training | Kaggle notebooks, **GPU T4 ×2 — NOT P100** (see below) |
| Location | `MIDI-Dataset/v2-study/` — no existing file modified or deleted |

## Design: the 3-slot skeleton

The atomic unit is a **balanced triplet** — 3 photos giving exactly 2 annotations of each class,
each class once on the left hand and once on the right:

```
cycleA   L=open R=closed | L=closed R=thumb | L=thumb  R=open
cycleB   L=open R=thumb  | L=thumb  R=closed| L=closed R=open
diagonal L=open R=open   | L=closed R=closed| L=thumb  R=thumb
```

**12 places, one lighting each, 3 places per group.** Each of the four groups (fold0, fold1, fold2,
test) contains exactly one place at each of three fixed slots, and the slot pins every nuisance
variable:

| slot | triplets | photos | distance | tilt | light | lux | sleeve | accessory | difficulty |
|---|---|---|---|---|---|---|---|---|---|
| S1 | diagonal | 3 | 55 cm | 0° | daylight, diffuse | 800 | short, bare forearm | none | easy |
| S2 | cycleA + cycleB | 6 | 65 cm | −15° | artificial, hard | 300 | long striped | wristwatch, image-left | medium |
| S3 | cycleA + cycleB | 6 | 75 cm | +20° | artificial, soft | 60 | long dark / hoodie | bracelet, image-right | hard |

Per group that is **1 diagonal + 2 cycleA + 2 cycleB = 5 triplets = 15 photos**, identical in all
four groups.

**This is what makes the ±SD interpretable.** All four groups match exactly on photo-weighted mean
distance (67 cm), mean tilt (+2°), difficulty mix (1 easy / 1 medium / 1 hard), illumination mix
(1 daylight / 2 artificial), triplet mix, and sleeve/accessory conditions. The *only* thing that
differs is which physical place occupies each slot — so fold-to-fold spread measures environment
variance, which is the variance that matters for deployment.

**The diagonal triplet sits at the easiest, nearest, best-lit slot in every group.** It is the only
case where both hands show the same class, so it carries the merge/confusion risk; it must not also
carry the dark, distant, high-tilt risk.

Balance, verified:

| level | photos | annotations | per class | per class per side |
|---|---|---|---|---|
| S1 environment | 3 | 6 | 2 | 1 L, 1 R |
| S2 environment | 6 | 12 | 4 | 2 L, 2 R |
| S3 environment | 6 | 12 | 4 | 2 L, 2 R |
| **group** | **15** | **30** | **10** | **5 L, 5 R** |
| **study** | **60** | **120** | **40** | **20 L, 20 R** |

**Class is orthogonal to place, side, distance, tilt, light, sleeve and accessory — at every
level, including within each individual environment.** That is the sentence for the methods section,
and it is what stops the model reading background instead of gesture.

**Orientation ladder 0° / 30° / 60°** of in-plane rotation, balanced within every environment: one
photo per level in the 3-photo environments, two per level in the 6-photo environments.

**Rooms.** 12 distinct places, 3 per group, **the 3 test places unused anywhere in training**. All
12 backgrounds distinct. Test-room count is the effective sample size behind any "works in an
unseen room" claim — here **n = 3**, stated as such.

**Orientation ladder 0° / 30° / 60°** of in-plane rotation, one per environment, arranged as a
Latin square against class. This exists to make the OBB-vs-HBB contrast measurable.

## What adversarial review caught (5 fatal)

These would have wasted the entire shoot:

1. **Auto-exposure is a function of the gesture.** An open hand fills more frame than a fist, so
   AE/AWB shifts global brightness *with the class* — a shortcut that survives environment-disjoint
   splitting and would have been confirmed, not exposed, by the held-out set. **Fix:** capture with
   locked exposure/gain/WB/focus, plus an 18% grey card taped in a fixed frame corner, and an
   assertion that per-class card luma within an environment differs by less than a tolerance.
   Unrecoverable after the fact.
2. **No shutter trigger exists.** Every photo needs *both* hands holding a gesture — you cannot
   press the shutter. **Fix:** `capture.py` with a delayed CLI trigger, which also removes the
   Photo Booth screen flash, the mirroring ambiguity, and the manual rename step.
3. **0° and 90° give identical OBB and HBB boxes.** The original 0/45/90 ladder left only a third
   of annotations informative. **Fix:** 0/30/60 — HBB/OBB area ratio 1.00 at 0°, 1.87–2.08 at
   30/60°, so 80 of 120 annotations inform the contrast.
4. **The leakage assertion could not fail.** Encoding the fold in the filename and then deriving
   the group key from the filename makes the check tautological. **Fix:** filename is
   `E07_P19.jpg`; the split comes from `data/environments.csv`, hashed into `folds.json`.
5. **Orientation was ambiguous** about the rotation axis, and the natural reading destroys the
   thumbout class. **Fix:** rotation in the image plane, palm parallel to sensor, with a reference
   photo shot during the pilot.

Also fixed: sessions straddling folds, accessories appearing only in test, test skewed harder than
the folds, the sorted label check being blind to left/right swaps, `group_of()` silently falling
through for unknown sources, and no cross-check binding the OBB and HBB label trees.

## Second architecture: DEIMv2 (DAMO-YOLO dropped)

DAMO-YOLO is disqualified on four independent grounds: it is the **only NMS-bearing candidate**, so
the latency gap would be confounded by post-processing in YOLO26's favour; its published 2.78 ms
**excludes post-processing**, so it isn't even citable; **one** checkpoint survives (official
Alibaba links dead since 2024), so it cannot produce a scaling curve; and it has **no Core ML
path** — `nonzero`, boolean masks and `batched_nms` make it structurally untraceable.

DEIMv2 wins on the things a paper is judged on:

| | DEIMv2 | why it matters |
|---|---|---|
| Licence | Apache-2.0 (LICENSE read in full) | publishable; GitHub API mis-reports "Other" |
| Weights | verified live on **3 hosts** (HF safetensors, Google Drive `.pth`, Quark) | DAMO's were dead |
| NMS-free | yes, fixed `[B,300]` output, no `nonzero`/thresholding | comparison varies **one** thing |
| Core ML | **demonstrated** — a downloadable fp16+int8 `.mlpackage` exists; deformable attention lowers to **18 native MIL `resample` ops**, no custom operator | the risk that killed DAMO-YOLO is retired |
| Sizes | **8** variants, 6 at a fixed 640 | a real scaling curve without a resolution confound |
| Size match | Pico 1.51 M / N **3.57 M** vs YOLO26n 2.4 M | RF-DETR-Nano is 30.5 M — a 12× mismatch |

**Anchor on DEIMv2-N** (3.57 M, COCO AP 43.0). Pico/Atto have reduced query counts (200/100) and
had distillation disabled for "limited capacity"; DEIM issue #71 documents the exact failure mode
this study is exposed to — only the majority class learned.

### Config surgery — the single most likely failure point

The official "2 files, ~10 lines" tutorial is wrong at this data scale:

- `total_batch_size: 32` → **4 or 8**, *and* `drop_last: True` → **False**. At 30 images
  `floor(30/32) = 0` batches — **the dataloader is empty and training silently does nothing.**
  Open, unanswered DEIM issue #82.
- `warmup_iter: 2000` → **100–300**. At ~7 iters/epoch the default is ~285 epochs of pure warm-up.
- Rescale every epoch-indexed schedule written for 148 COCO epochs: `epoches`, `flat_epoch`,
  `no_aug_epoch`, `policy.epoch`, `stop_epoch`, `mixup_epochs`, `copyblend_epochs`,
  `matcher_change_epoch`. LR scaling is **manual**, not automatic.
- COCO JSON, **1-indexed `category_id`, `num_classes: 4`** (classes + 1), `remap_mscoco_category: False`.
- Two torch-2.x patches: torchvision v2 transform rename (upstream PR #139, ~8 lines) and
  `weights_only=False` in `engine/solver/_solver.py`. **Never `pip install -r requirements.txt`** —
  it pins `torch==2.5.1` and would break Kaggle's CUDA build.
- Conversion runs in a **separate venv with `torch==2.8.*` + coremltools 9.0**; coremltools'
  `_TORCH_MAX_VERSION` is 2.8.0 and the main environment is on 2.11.

### Honest position on small data

**No published result exists for any DETR-family detector fine-tuned at 30 images.** Nearest
documented cases: D-FINE-Nano on 150 images (single class), DEIMv2-X on 383 images. RF100-VL's
10-shot table shows *every* detector family collapsing from ~55 to ~20 mAP — and contains no
DETR-family model at all. This is a genuine extrapolation 2–5× below the smallest documented case,
and the paper says so.

### Stop rules

- **Training:** after config fixes are *verified applied* (`assert len(train_dataloader) > 0`),
  a fold-0 DEIMv2-N pilot still gives AP 0.0 on ≥2 of 3 classes across **two** hyper-parameter
  attempts. Stop at ~8 h sunk.
- **Export:** fp32 `.mlpackage` fails the `allclose` parity gate *and* re-converting with
  `cross_attn_method: discrete` also fails. Stop at ~8 h sunk.
- **Do NOT stop for:** latency above 10 ms, a flat scaling curve, or fp16 failing where fp32 passes.
  Those are results — report them.
- **Fallback:** RF-DETR-Nano (~18–25 h, native Core ML export, but foreground the 12× capacity
  mismatch). Second fallback: reframe around the data-scale finding itself, with YOLO26 n→x as the
  scaling study and a documented negative result for DETR fine-tuning at this scale. That is a
  publishable paper and the one this dataset can actually support.

### Go/no-go probe before anything else (1.5 h)

Download `DanteLiu/DEIMv2-X-pill-needle`'s `.mlpackage` (fp16, 97 MB, Apache-2.0), load it on the
M4 with coremltools, run one image, and open Xcode's Core ML Performance Report. This tells you
**today**, before any training, whether DEIMv2 Core ML runs and where `resample` dispatches on your
exact hardware. It is the X variant, so its latency is an upper bound — but the dispatch breakdown
transfers.

## Project layout

```
MIDI-Dataset/v2-study/
├── PLAN.md, README.md          plan + "do not reuse" notes on old-code defects
├── collection/
│   ├── shot_list.xlsx          THE deliverable — multi-sheet, live formulas (openpyxl 3.1.5)
│   │     Shot List · Balance Check · Environments · Protocol · Setup Cards
│   ├── shotlist.csv            60 photo rows
│   ├── environments.csv        20-row registry — SOURCE OF TRUTH for env → split
│   └── CAPTURE_PROTOCOL.md
├── photos/ masks/ overlays/
├── data/pool/{images,obb/,hbb/,manifest.csv}   two complete ultralytics roots
├── data/splits/{fold0,fold1,fold2}/, test.txt, *.yaml, folds.json
├── scripts/                    capture.py, build_pool.py, make_splits.py, train_cv.py,
│                               metrics.py, damo_runner.py, design_asserts.py, export_coreml.py
├── kaggle/ results/ figures/ models/
```

Reused unmodified: `mask_to_obb_midi.py::mask_to_obb_lines`, `overlay_check.py`,
`study/plotstyle.py`, `study/latency_policy.py`, `study/benchmark_inference.py`, `YOLOmodels/`.

**No existing file is modified or deleted.** Known defects recorded in `README.md` as warnings: the
root `import qupath.lib.regions.*` groovy (paints closedhand as `1`, silently blank annotations);
`mask_to_obb_midi.py:26` stale `SOURCE_IMAGE_DIR`; empty `updateimg/scripts/convert.groovy`; TIFF
bytes under `.png` in `updateimg/class_image_mask/`; three hardcoded weight paths bypassing
`common.pretrained_path()`.

## Implementation phases

**(a) Before shooting** — generate `environments.csv`, `shotlist.csv`, `shot_list.xlsx` and printed
per-environment setup cards. Run `design_asserts.py` (14 checks, below). Measure the real webcam FOV
with a ruler at 55 and 75 cm; if narrower than 60°, shift the whole distance ladder by a constant.
Walk the house against `environments.csv` and confirm every room and prop. Shoot **E01 as a pilot**
end-to-end — capture → mask → label → overlay QA → train one 5-epoch model — before anything else.

**(b) Shooting** — 8 sessions, none straddling a group, max 3 environments each, ≥24 h before the
test sessions. Locked exposure/WB/focus via `ffmpeg -f avfoundation`, grey card in frame, delayed
trigger. Hard/dark slots shot first while fresh.

**(c) Masks + QA** — generate three-class masks (100/200/255). **The existing
`testopus/make_masks.py` cannot emit thumbout** — it needs a third branch keyed on its existing
`thumb_out` computation at line 72. Then `overlay_check.py` at alpha 0.5 on every one of the 60,
eyeballed, not just scored.

**(d) Pipeline rebuild** — two annotations per image throughout; dual `obb/` and `hbb/` roots
selected by an explicit `--label-set` flag (never a bare env var); `group_of()` keys on env id and
*raises* on unknown sources; fold assignment read from `environments.csv`, not searched.

**(e) Kaggle** — 33 training cells: 7 models × 3 folds at 640, plus `yolo26n-obb` and `yolo26n-hbb`
also at 416 and 320. Manifests hold absolute paths, so `make_splits.py --rebase` runs first.

> **Do not select the P100.** Kaggle's current PyTorch image is built for `sm_70 … sm_120`; the
> P100 is `sm_60`, so CUDA fails with *"no kernel image is available for execution on the device"*.
> This is an open Kaggle bug ([docker-python #1546](https://github.com/Kaggle/docker-python/issues/1546),
> filed 2026-03-20, still open) and it breaks **all** training, not just the second architecture.
> Select **GPU T4 ×2** (`sm_75`). GPU sessions are 12 h, not 9 h; 9 h is the TPU limit.
> Verify `torch.__version__` and a one-line CUDA tensor op in a scratch cell before anything else.

**(f) Latency** — Core ML export, benchmark `imgsz {320,416,640}` × `{PyTorch-MPS, Core ML}`.
Report at realistic webcam resolution *and* at the current still resolution so the preprocess
inflation is visible. New gate: `GATE_MEDIAN_MS = 16.0`, `GATE_P95_MS = 25.0`, pre-registered this
time, plus a jitter clause wiring up the currently-uncalled `MAX_JITTER_MS`.
`MODEL_BUDGET_MS = 10.0` is an engineering budget, not a perceptual tier.

**(g) Metrics and figures** — `metrics.py` emits mAP50-95 (exact IoU as headline, ultralytics
native as an appendix column, and the delta between them as its own result), the 4×4 confusion
matrix, per-hand one-vs-rest ROC/AUC, and per-class P/R/F1. Figures follow `plotstyle.py`.

## Verification

`design_asserts.py` runs 14 checks; 9 are pre-shoot and catch errors before a single photo:

- 12 environments with 3/6/6 photos per slot; `P01..P60`; filenames `{env}_{photo}.jpg` unique
- env→split matches `environments.csv`; 3 envs per split; **no env spans two splits**
- **no room spans train and test**; **no session spans groups**; 12 distinct backgrounds
- slot profile identical across all four groups (distance, tilt, light, sleeve, accessory, difficulty)
- per environment: 2 or 4 per class, evenly split left/right
- per group: 10 per class, 5 per class per side, all 9 (L,R) combos, 1 diagonal + 2 cycleA + 2 cycleB
- orientation: balanced within every environment; Latin square against class globally and per group
- at build: exactly 2 annotations per image; **painted class *sequence* in centroid-x order equals
  `expected_label_sequence`** (catches left/right swaps, which a sorted check cannot); obb and hbb
  trees agree on stems, line counts and class order
- **grey-card luma per class within an environment differs by ≤ tolerance** (the AE shortcut)
- **geometry-only shortcut probe**: a classifier on box position/area/aspect alone must not beat
  chance (0.333)

## Honest limitations for the paper

Written publication-ready in the full design; the load-bearing ones:

- **30 test annotations resolves nothing narrower than ~11 percentage points.** Wilson 95% at 30/30
  is [0.886, 1.000]; per class (n=10) at 10/10 it is [0.722, 1.000]. Taking the environment as the
  unit of independence gives n=5. **Two models differing by fewer than ~5 hands are tied.**
- **Per-hand AUC measures classification given localisation**, not detection — a model finding 20 of
  30 hands and labelling all 20 right scores 1.00. Must be read beside `n_scored/n_gt` and the
  background column.
- **Ultralytics evaluates OBB with the ProbIoU surrogate, not rotated-polygon IoU** — optimistic by
  +0.112, wrong side of the 0.75 threshold 45.5% of the time. So OBB-vs-HBB uses exact IoU for both.
- **Three folds**: 95% interval on the fold mean is mean ± 2.484 × SD.
- **Single subject** — skin tone and hand geometry constant everywhere. Not leakage, but no
  cross-subject claim is available.
- **The camera dominates latency.** Sensor + transport is 30–50 ms of a ~55–80 ms chain, so a 10 ms
  *model* budget is an engineering target, not a perceptual threshold. Reported as such.

## Open decisions

Defaults I will take unless told otherwise: orientation ladder 0/30/60; `FLIPLR = 0.5` with a
rewritten rationale (the recorded reason is now false, and it breaks residual side↔class
correlation); `ddof=1`; `max_det=300` for both runtimes; keep `runs_tune/` archived not pruned.

Settled since: 12 places / one lighting each; DEIMv2 replaces DAMO-YOLO; Kaggle T4 ×2 not P100;
cross-architecture comparison on HBB with input pinned to 640.

Still needs your input: the room inventory (walk the house against `environments.csv` before
shooting), and whether the headline runtime is Core ML or MPS — I default to reporting both,
entering via MPS so a publishable number exists early.
