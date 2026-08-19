# STATUS — v2 study, resume document

**Last updated: 2026-08-13, 18:45 local.**

> **THE STUDY IS COMPLETE. Read `results/RESULTS.md`.**
>
> Answer: **yes, deployable.** The model is **YOLO26-s, axis-aligned, 320 px** —
> 0.8056 ± 0.0437 mAP50-95 (3-fold CV, exact IoU), macro AUC 0.9846, and **7.93 ms sustained
> over 15 minutes from cold** on the M4, holding the 10 ms budget for all 111,884 iterations.
>
> All 90 YOLO cells + 24 DEIMv2 runs trained; 38 models have confusion matrices, AUC-ROC and
> held-out test scores; latency measured burst and sustained on the target laptop.
>
> Two corrections that changed conclusions, both documented in RESULTS.md §4–5: ultralytics
> over-reports OBB by **+0.175** via ProbIoU (so the OBB advantage was an artifact — every
> top cell is HBB), and **sustained latency is 0.5–2.1 ms worse than burst**, which
> disqualifies four configurations that passed the burst test.

Purpose of this file: if the Claude session dies, this is the single place that says what
is done, what is running, what is left, and the exact commands to pick it back up. Anything
asserted here was verified by running something, not remembered. Where a number is
second-hand, it is marked **UNVERIFIED**.

---

## 1. The study

Establish whether a vision-based hand-gesture MIDI instrument is deployable on a consumer
laptop (MacBook Air M4). Two model families are compared on accuracy and on latency
measured **on the M4**, not on the training GPU.

Requirements fixed at the start, all still binding:

| Requirement | State |
|---|---|
| 3-fold cross-validation | done, folds are place-disjoint |
| 30 annotations per fold, classes balanced | verified 10/10/10 per group |
| 30-annotation test set, never trained on | **used once, at the end, for all 38 models** |
| Both hands in every photo | by construction in the shot list |
| mAP50-95 reported mean ± SD across folds | **done** — 90 YOLO cells + 24 DEIMv2 runs, exact IoU |
| **Confusion matrix (4×4) for every model** | **done** — **38 models**, `figures/confusion/` |
| AUC-ROC | **done** — per-class + macro, 38 models, `figures/roc/` |
| Inference time on the M4 | **done** — burst AND 15-min sustained, `results/RESULTS.md` §4 |
| ~10 ms model budget | engineering target; camera alone costs 30–50 ms |

---

## 2. Dataset — verified, frozen

Ran against `data/splits/folds.json` and `data/pool/coco/*.json` on 2026-08-13:

```
fold0  15 imgs 30 anns  {thumbout: 10, openhand: 10, closedhand: 10}
fold1  15 imgs 30 anns  {thumbout: 10, openhand: 10, closedhand: 10}
fold2  15 imgs 30 anns  {thumbout: 10, openhand: 10, closedhand: 10}
test   15 imgs 30 anns  {thumbout: 10, openhand: 10, closedhand: 10}
```

60 photos, 120 annotations, 2 hands each, perfectly balanced. `data/pool/hbb/images` and
`data/pool/obb/images` both hold 60 jpgs. Class index map is `{200: thumbout(0),
100: openhand(1), 255: closedhand(2)}` in the grayscale masks; COCO categories are
1-indexed, which is why DEIMv2 needs `num_classes: 4`.

Folds are disjoint **by place** (12 places), so a model cannot score by memorising a
background. That was rebuilt once already — an earlier version grouped by lighting cluster,
which was wrong; `scripts/assign_places.py` is the correct one and `assign_groups.py` was
deleted.

---

## 3. YOLO26 arm — COMPLETE, 90/90 cells

### Hyperparameters actually used

Recovered from the checkpoints (`TRAINING_CONFIG.md` has the full derivation):

| | |
|---|---|
| epochs | **100** (early stopping off: `patience` = `epochs`) |
| batch | **8** nominal, **64 effective** (accumulation to `nbs=64`) |
| optimizer | **AdamW** |
| lr0 | **0.001429**, linear decay to ×0.01 |
| momentum | **0.9**, weight decay 0.0005 |
| seed | 42, deterministic, AMP on |
| ultralytics | 8.4.45 |

**The checkpoints record `optimizer: auto, lr0: 0.01, momentum: 0.937` and those last two
were NOT used.** Auto mode overrides them: with `nc=3` it resolves to AdamW at
`round(0.002*5/(4+3), 6) = 0.001429`, momentum 0.9. Quoting the recorded `lr0` in a paper
would be wrong by 7× and would name the wrong optimizer.

Four runs (`hbb-s-fold2-416`, `hbb-x-fold0-416`, `hbb-x-fold1-416`, `obb-x-fold2-416`) were
auto-reduced to batch 4 after CUDA OOM during the parallel sweep. Effective batch, weight
decay, optimizer and LR are unchanged by that; only batch-norm statistics differ.


`midi_results/results.jsonl` — 90 rows, 0 errors. `midi_results/weights/` — 90 `.pt` files,
4.5 GB, verified against the expected name set (5 sizes × 2 geometries × 3 folds × 3
resolutions). 640 px trained on Kaggle; 320/416 on a rented RTX 4090.

**The numbers in this section were ultralytics' own and are superseded.** OBB rows are
inflated by ProbIoU (see the box below); the authoritative exact-IoU table is
`results/cv_exact_iou.json`, summarised in **`results/RESULTS.md` §2**. Top cells there:
hbb-m@416 0.8373 ± 0.0056, hbb-l@416 0.8225, hbb-x@416 0.8155 — **all HBB, all 416 or 320.**

Two findings from the completed sweep:

- **Lower resolution is better, not merely faster.** The best cells are at 416 and 320; every
  640 px cell is beaten. With 60 training images, 640 px appears to give the model more room
  to overfit, and hands occupy a large fraction of the frame so 320 px still resolves a thumb.
- The remembered figure `obb-s-416 = 0.8533 ± 0.0117` from an early Kaggle session was
  **wrong** — measured 0.8291 ± 0.0314 by ultralytics, and 0.6277 ± 0.0564 with exact IoU.

> **MEASURED caveat that must reach the paper.** Ultralytics matches OBB with ProbIoU and HBB
> with real box IoU — verifiable at `obb/val.py:94` (`batch_probiou`) vs `detect/val.py:317`
> (`box_iou`). Measured on this dataset over the 30 cells at 640 px
> (`results/probiou_check.json`), ultralytics minus exact-IoU:
>
> | geometry | mean | sd |
> |---|---|---|
> | hbb (control) | −0.0005 | 0.0279 |
> | **obb** | **+0.1777** | 0.0255 |
>
> The HBB control agreeing to −0.0005 is what validates the exact-IoU evaluator. OBB is
> over-reported by ~0.18, which is enough to reverse an OBB-vs-HBB conclusion — and did.
> **Every OBB number quoted from ultralytics in this file is inflated by roughly that much.**
> `scripts/test_eval.py` computes exact polygon IoU and is what the comparison must use.
>
> An earlier version of this note claimed "+0.11 optimistic". That figure had **no source** —
> it was asserted, not measured or cited, and it propagated into code comments before being
> checked. The table above replaces it.

---

## 4. DEIMv2 arm — complete

### The machine

```
ssh vastmidi          # already in ~/.ssh/config: 95.3.33.46 port 14465, ~/.ssh/vast_key
```
RTX 4090, 150 GB disk (5 GB used), `/venv/main` with torch 2.12.0+cu130. Work lives in
`/workspace/midi`. **This is a rented box and it is billing by the hour — destroy it when
the results are pulled down (§6).**

### Results — COMPLETE. 24/24 runs, all `rc=0`, all under the uniform schedule.

AP50-95 on each variant's held-out fold, mean ± SD across the 3 folds:

| variant | fold0 | fold1 | fold2 | mean ± SD |
|---|---|---|---|---|
| atto | 0.050 | 0.115 | 0.029 | 0.065 ± 0.045 |
| femto | 0.066 | 0.107 | 0.239 | 0.137 ± 0.090 |
| pico | 0.087 | 0.110 | 0.072 | 0.090 ± 0.019 |
| **n** | 0.808 | 0.799 | 0.770 | **0.792 ± 0.020** |
| **s** | 0.871 | 0.823 | 0.832 | **0.842 ± 0.026** |
| **m** | 0.856 | 0.839 | 0.791 | **0.829 ± 0.034** |
| **l** | 0.853 | 0.860 | 0.815 | **0.843 ± 0.024** |
| **x** | 0.843 | 0.856 | 0.813 | **0.837 ± 0.022** |

Verified: all 24 logged `Refresh EMA at epoch 100`, so every variant genuinely executed the
stage-2 fine-tune. That is the fix in §4.1 working.

Best in the study: **DEIMv2-L 0.843 ± 0.024** and **DEIMv2-S 0.842 ± 0.026**, statistically
indistinguishable from each other, both above the best YOLO26 640 cell (hbb-s, 0.815 ±
0.037) and with tighter fold-to-fold spread.

### Checkpoints: `best_stg2.pth` if present, ELSE `best_stg1.pth`

18 of 24 runs have `best_stg2.pth`. The other 6 (atto_f2, femto_f1, l_f2, m_f2, s_f1, x_f2)
peaked during stage 1 — at epochs 93/96/99 — and stage 2 never beat it, so DEIMv2 never
wrote a stage-2 best. For those runs **`best_stg1.pth` is the best checkpoint**, and a
retrieval rule that only fetches `best_stg2.pth` silently drops a quarter of the arm.
All 24 best checkpoints total 5.4 GB.

### 4.1 ⚠ Why everything was re-run: the variants were not comparable

DEIM trains in **two stages** and switches at `collate_fn.stop_epoch`, where it turns off
augmentation and reloads `best_stg1.pth` for a fine-tune phase. `stop_epoch` ships **inside
each variant's own COCO recipe**, and those recipes have wildly different budgets:

| variant | stock `epoches` | stock `stop_epoch` | with our `epoches: 120` |
|---|---|---|---|
| s | 132 | 120 | **never switches** — 120 epochs of stage 1 |
| atto | 500 | 468 | **never switches** |
| n | 160 | 148 | **never switches** |
| l | 68 | 60 | switches at 60 |
| x | — | 50 | switches at 50 |

Overriding `epoches: 120` alone does not give each variant a 120-epoch version of its own
recipe. It gives each an arbitrary slice of a *different* recipe. Half the arm got a
fine-tune stage and half never did, so `s` (0.843) vs `l` (0.374) was substantially
**measuring the schedule, not the architecture** — which is precisely the comparison the
paper exists to make.

`setup_deimv2.py`'s docstring claimed these schedules were rescaled. They were not; the
generated config never wrote them. Docstring and code now agree.

**Fixed:** every generated config pins one schedule, and this is the sentence for the
methods section — *all DEIMv2 variants trained 120 epochs with `flat_epoch` 64,
`no_aug_epoch` 20, `stop_epoch` 100, `matcher_change_epoch` 90; the stage-2 fine-tune is
therefore the final 20 epochs for every variant.* The relationships are the repo's own
(`stop_epoch = epoches - no_aug_epoch`, `flat_epoch = 4 + epoches // 2`).

Verified two ways: all 24 configs report identical schedule values, and DEIM's own
`YAMLConfig` loader resolves one of them with all **11 augmentation ops intact** (the nested
override is a deep merge, so pinning `policy.epoch` does not wipe the transform list).
`run_deim.py` pre-flight now **refuses** any config with `stop_epoch >= epoches`.

### Check progress

```bash
ssh vastmidi 'tmux ls; tail -3 /workspace/midi/jobA.log /workspace/midi/jobB.log'
ssh vastmidi 'cd /workspace/midi && python3 -c "
import json,glob
for f in sorted(glob.glob(\"results_*.jsonl\")):
    for l in open(f):
        r=json.loads(l)
        print(\"%-24s rc=%s ap50_95=%s\" % (r.get(\"config\"), r.get(\"returncode\"), r.get(\"ap50_95\",\"-\")))
"'
```

### ⚠ THE RULE: a DEIMv2 row with `rc=1` IS NOT A RESULT

A crashed run still parses an `ap50_95` — the best score it reached **before dying**. It
looks like a real, merely-mediocre number. It is not one.

This already happened today and it was my bug. `prune_forever()` in `run_deim.py` kept "the
newest 2 `.pth` by mtime". DEIMv2 **reloads `best_stg1.pth` at the stage-1→stage-2
transition** (`engine/solver/det_solver.py:81`), and writes `last.pth` every epoch
(`:109`). Once a run's best score plateaued, `best_stg1.pth` became the oldest file, got
deleted, and training died at epoch ~62 of 120 with `FileNotFoundError`. It also caught
`last.pth` mid-write and left it **0 bytes**.

It only bit the models that plateau — which is exactly the models whose numbers you would
most want to trust. It also only bit the variants that *reach* a stage-2 switch, which is
how the much larger schedule bug in §4.1 came to light.

Note `last.pth` is **not** a substitute for the best checkpoint: `det_solver.py:108` guards
its write with `epoch < stop_epoch`, so it stops updating once stage 2 begins. The
checkpoint to keep per run is **`best_stg2.pth`**.

Fixed: the pruner now deletes only `checkpoint\d+\.pth` (periodic duplicates of `last.pth`)
and never `last.pth` / `best_stg1.pth` / `best_stg2.pth`, and skips any file touched in the
last 30 s so it cannot truncate a write. Verified holding at 07:21. The fix is in
`kaggle/run_deim.py` locally **and** synced to `/workspace/midi/scripts/run_deim.py`.

Failed rows are deliberately not counted as "done", so a restart re-runs them by itself.

---

## 5. What is left

**Nothing blocking. The study is complete** — see `results/RESULTS.md`.

Optional, if the paper wants them:
1. Sustained runs longer than 15 min (the plateau looks stable but was not measured further).
2. `hbb-l@320` re-run **from cold** — its breach was measured preheated, so it is unresolved.
3. A Core ML path for DEIMv2's DINOv3 line — those four convert silently wrong (RESULTS.md §9).

## 6. The rented boxes — both finished with

Everything is local and verified: 90 YOLO checkpoints (4.5 GB), 24 DEIMv2 checkpoints
(5.4 GB), all results, logs and prediction dumps. Zero truncated files; both sets were
checked against their expected name sets, which is how a corrupted checkpoint was caught
(see §11). **Both instances can be destroyed / are destroyed.**

## 7. File map

```
scripts/
  make_collection.py     shot list + environments (how the 60 photos were specified)
  assign_places.py       recovers the 12 places from capture order  [correct one]
  mask_qa.py             Sobel edge-alignment QA, alpha=0.5 overlays
  install_manual_masks.py merges QuPath exports, per-class, never overwrites
  build_pool.py          masks -> obb/ + hbb/ ultralytics roots + COCO json
  metrics.py             4x4 confusion matrix + AUC-ROC, exact polygon IoU
  export_coreml.py       M4 latency; refuses to run off Darwin
kaggle/
  train_yolo.py          90-cell YOLO driver, resumable
  setup_deimv2.py        clones/patches DEIMv2, stages COCO, writes configs
  run_deim.py            DEIMv2 runner  [pruner fixed 2026-08-13]
data/
  splits/folds.json      the fold assignment
  pool/manifest.csv      photo -> id
  pool/{hbb,obb,coco}/   the three label geometries
midi_results/            30 YOLO 640 runs: results.jsonl + 30 weights
vast/bundle/             what was shipped to the rented box
```

---

## 8. Findings worth keeping

- **DEIMv2 has a hard capacity floor, and it is not gradual.** `pico` (0.090) → `n` (0.792)
  is a ~9× jump for a ~2× parameter increase. Below `n`, the model does not learn the task
  at 60 training images at all; `atto` 0.065 and `pico` 0.090 are failures, not weak
  results. This is the cleanest finding in the study and it is worth a figure.
- **Above that floor, both families go flat.** DEIMv2 spans 0.792–0.843 from `n` to `x`;
  YOLO26 spans 0.719–0.815 across a 24× parameter range. Once capacity clears the
  threshold, more of it buys essentially nothing at this dataset size. That is the finding
  that matters for a laptop deployment: **the smallest model above the cliff is the right
  one**, and for DEIMv2 that is `n` (0.792 ± 0.020, the tightest SD in the arm).
- **The top of each family:** DEIMv2-L 0.843 ± 0.024 and DEIMv2-S 0.842 ± 0.026 are tied,
  both above the best YOLO26 640 cell (hbb-s 0.815 ± 0.037). But `s`/`m`/`l`/`x` are
  **DINOv3**-backboned while `atto`/`femto`/`pico`/`n` are **HGNetv2**, so the honest
  framing of a DEIMv2 win at the top is *foundation-model pre-training + a DETR head* beats
  YOLO26 — not architecture alone. The HGNetv2 line is the clean architectural comparison,
  and there `n` (0.792) sits just below the best YOLO26 cell (0.815).
- **Fold SD is 0.02–0.04 nearly everywhere**, which is the resolution limit of this
  experiment. Differences smaller than ~0.04 between two working models are not real. Say so
  rather than ranking them.
- **Resolution is not the big latency lever it looks like.** 320 vs 640 saves only ~4.3 ms
  because ~17.5 ms of the PyTorch-MPS path is fixed overhead. That is what makes Core ML
  necessary rather than optional — it is the only remaining lever.
- Early stopping is **off** (`patience=epochs`). With 3 folds it fires at random and would
  measure luck rather than architecture.

---

## 9. Confusion matrices + AUC-ROC — done, 38 models

Full write-up in **`results/METRICS.md`** (18-model version) and
**`results/RESULTS.md` §6** (final, 38 models). Produced by three scripts:

```
scripts/dump_preds_yolo.py   30 YOLO checkpoints -> per-fold detection dumps   (Mac, MPS)
scripts/dump_preds_deim.py   24 DEIMv2 checkpoints -> same format              (rented box)
scripts/pool_metrics.py      pools each model's 3 folds -> CM + ROC + figures  (Mac)
```

Each model's three fold checkpoints are scored on their own held-out folds and pooled, so
every matrix covers all 90 annotations and **the test set stays untouched**.

Two things that would have silently corrupted these numbers, both now handled in code:

- **DEIM's config registry is process-global.** Building a second model in one interpreter
  inherits state from the first. `pico` loaded alone reports raw [0,1] input (correct for its
  HGNetv2 backbone); loaded sixth in a batch it reports ImageNet normalisation and produces
  different detections — silently. The DINOv3 variants die loudly with a state_dict mismatch
  instead. `dump_preds_deim.py` now runs **one checkpoint per subprocess**.
- **`num_classes: 4` creates a dead slot 0** that nothing trains. Well-trained variants never
  emit it; `atto` at AP 0.065 does. It is dropped, not remapped, and the drop is recorded per
  dump. All observed slot-0 detections were below conf 0.05, far under any operating point.

**The threshold matters more than expected.** YOLO26 applies NMS, DEIMv2 does not — at a
fixed conf 0.25, DEIMv2's 300 unsuppressed queries produce 44–150 false alarms against
YOLO26's 10–25, dragging `deimv2/m` to macro-F1 0.46 despite AP50-95 0.829. Every model is
therefore reported at its own F1-optimal threshold, with the fixed-0.25 numbers kept beside
it. AUC is threshold-free and is the metric to lead with.

### What the matrices say

**79 of 80 class confusions are thumbout ↔ closedhand; exactly one involves openhand.**
openhand mean AUC 0.9985, thumbout 0.9508, closedhand 0.9509 — the latter two are confused
with each other and almost nothing else. thumbout is a fist plus an extended thumb, so it
differs from closedhand by one digit; the same pair produced the four annotation errors found
during dataset construction. The vocabulary has exactly one hard pair, and it is the
anatomically adjacent one.

**AP and AUC rank the architectures differently.** DEIMv2 wins AP50-95 and finds more hands
(`deimv2/l` misses 2 of 90; YOLO26 misses 6–19), but YOLO26 takes the top four AUC slots.
DEIMv2 localises better, YOLO26 classifies better once localised. For an instrument a missed
hand is a dropped note and a misclassification is a wrong note — which one to prefer follows
from that, not from either metric alone.

---

## 10. M4 latency — done. The deployability question is answered: yes.

Full write-up in **`results/LATENCY.md`**. Two scripts:
`scripts/export_coreml.py` (YOLO26) and `scripts/export_coreml_deim.py` (DEIMv2, new work).

**8 YOLO26 configurations meet the 10 ms budget**, all at reduced resolution under Core ML:

| model | 320 | 416 | 640 |
|---|---|---|---|
| hbb-n | **7.09** | **8.57** | 13.39 |
| obb-n | **7.22** | **8.89** | 14.50 |
| hbb-s | **7.54** | **9.10** | 14.75 |
| obb-s | **7.57** | **9.30** | 15.07 |

Jitter at those points is 0.39–1.10 ms, far inside the 10 ms tolerance.

### The correction that matters

I reported earlier — from the MPS baseline — that resolution was not a latency lever. **On
Core ML it is.** hbb-n runs 7.09 / 8.57 / 13.39 ms at 320 / 416 / 640, monotonic, a 6.3 ms
range. On MPS the same model spanned 2.4 ms non-monotonically because fixed dispatch overhead
swamped it. Core ML removes that overhead and the model's actual size becomes visible.

Consequence: **the 60 Kaggle resolution-sweep accuracy rows are now the top priority**, not a
deprioritised task. The budget is met at 320/416 and we do not have verified accuracy there.

### The environment is fragile — pin it

`.venv-coreml` is separate from the system Python for a reason. coremltools 9.0 declares
torch **2.7.0** as its newest tested version; on torch 2.13 conversion dies inside coremltools'
own `_cast` with "only 0-dimensional arrays can be converted to Python scalars". Working set:
**torch 2.7.1, numpy 1.26.4, scipy 1.14.1, opencv 4.11, coremltools 9.0**. Installing DEIMv2's
deps pulls numpy 2.x back in and re-breaks it.

### DEIMv2: one variant measurable, four not — and that is a finding

`export_coreml_deim.py` had to fix three converter incompatibilities to get any DEIMv2 model
through: a rank-1 `linear` (D-FINE's distribution-integral head), float `gather` indices from
lowered integer floor-division, and `aten::gather` arity.

- **n (HGNetv2): converts and verifies exactly** — 0.57 px box deviation, 0.0017 score
  deviation against PyTorch. ~5–7 ms at 640.
- **s / m / l / x (DINOv3): convert silently and produce wrong boxes** — 213–277 px deviation,
  all four failing identically while the one HGNetv2 variant is exact. Backbone-specific
  converter fault. **Latency is not reported for these**, and the script refuses to time a
  model that fails verification rather than leaving that to discipline.
- **DEIMv2 cannot change resolution at all**: 320/416 fail with a fixed-size mismatch, because
  the architecture is tied to `eval_spatial_size: 640`.

Confidence note: DEIMv2-n medians across four runs were 4.90 / 5.26 / 6.97 / 13.85 ms. The
outlier followed two failed exports. Before publishing, re-measure with the YOLO arm's
discipline — shuffled repeats over a pre-compiled package. Treat ~5–7 ms as indicative.

If it holds, it is worth a sentence in the paper: DEIMv2-n beats every YOLO26 at 640 on both
latency and mAP (0.792 vs 0.763), plausibly because it is **NMS-free** while the YOLO Core ML
export bakes in NMS, which the ANE does not like.

---

## 11. Two errors caught late, and how

Both were found by cross-checks rather than by the pipeline running cleanly, which is worth
recording because a reviewer will ask.

**A corrupted checkpoint in the results.** `hbb-m-fold1-416.pt` was a 167 MB partial left by
the worker killed during a rebalance; my staging script copied the first matching run
directory rather than the one whose `results.jsonl` recorded a completed run. It emitted 300
detections per image (ultralytics' `max_det` cap) instead of ~7, and scored 0.4957 exact vs
0.8449 as reported — a +0.349 disagreement where the other 44 folds agreed to ±0.03.

It was caught because the exact-IoU evaluator disagreed with ultralytics on exactly one HBB
fold, and HBB is the control where the two should agree. Fixed by fetching the real
checkpoint from `out_m1`; a size scan of all 90 confirmed it was the only anomaly. The
control tightened from +0.0108 ± 0.0549 to +0.0032 ± 0.0188.

**Thermal start contaminating a latency comparison.** Sustained runs batched two models with
60 s between them, so only the first started cold. `hbb-s@320` read 9.38 ms preheated and
7.93 ms cold. The 1.45 ms difference is larger than the gap between several models, and the
preheated reading would have disqualified the model that is now the deployment choice.
`scripts/tradeoff.py` now refuses to let a preheated reading override a cold one.

The general lesson for the methods section: **every quantitative claim here has an artifact
behind it, and the ones that survived did so because an independent measurement agreed.** The
exact-IoU evaluator is trusted because it matches ultralytics on HBB (+0.003) and COCOeval on
DEIMv2 (+0.001) while disagreeing with ProbIoU on OBB (+0.175).
