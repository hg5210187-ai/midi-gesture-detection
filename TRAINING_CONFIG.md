# YOLO26 training configuration — as actually used

Recovered from the 90 trained checkpoints, not from the training script's defaults. Every
value below was read out of `train_args` inside the `.pt` files, which is what ultralytics
records at the end of a run.

Reproduce with:

```bash
python3 -c "
import torch, glob, collections, warnings; warnings.filterwarnings('ignore')
agg = collections.defaultdict(collections.Counter)
for f in sorted(glob.glob('midi_results/weights/*.pt')):
    a = (torch.load(f, map_location='cpu', weights_only=False).get('train_args') or {})
    for k, v in a.items(): agg[k][v] += 1
for k in sorted(agg): print(k, dict(agg[k]))
"
```

---

## The answer

| setting | value | source |
|---|---|---|
| **epochs** | **100** | explicit; identical in all 90 checkpoints |
| **batch (nominal)** | **8** in 86 runs, **4** in 4 runs | see §Batch below |
| **batch (effective)** | **64** in all 90 runs | gradient accumulation to `nbs=64` |
| **optimizer** | **AdamW** | resolved from `optimizer=auto` |
| **learning rate (lr0)** | **0.001429** | resolved from `optimizer=auto` |
| **momentum** | **0.9** | resolved from `optimizer=auto` |
| final LR fraction (`lrf`) | 0.01 | linear decay to lr0 × 0.01 |
| weight decay | 0.0005 | scaled to 0.000500 effective |
| warmup | 3.0 epochs, momentum 0.8, bias LR 0.0 | |
| scheduler | linear (`cos_lr=False`) | |
| AMP | on | |
| seed / deterministic | 42 / True | |
| early stopping | **off** (`patience=100` = `epochs`) | deliberate, see below |
| `close_mosaic` | 10 | mosaic disabled for the last 10 epochs |
| `fliplr` | 0.5 | |

---

## Optimizer and learning rate: why the checkpoint is misleading

The checkpoints record `optimizer: auto`, `lr0: 0.01`, `momentum: 0.937`. **Those last two
were not used.** In `auto` mode ultralytics discards them and derives its own, logging
"ignoring `lr0=...` and `momentum=...`" as it does so (`engine/trainer.py`):

```python
nc = self.data.get("nc", 10)
lr_fit = round(0.002 * 5 / (4 + nc), 6)
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
self.args.warmup_bias_lr = 0.0
```

With this dataset:

```
nc          = 3                                    (thumbout, openhand, closedhand)
lr_fit      = round(0.002 * 5 / (4 + 3), 6)        = 0.001429
iterations  = ceil(30 / max(batch, 64)) * 100      = 100        (<= 10000)
            -> AdamW, lr0 = 0.001429, momentum = 0.9
```

**Corroboration that the auto branch really ran:** every checkpoint records
`warmup_bias_lr: 0.0`, whereas the ultralytics default is `0.1`. Nothing else in the codebase
sets it to zero. The training script never passes it. So the auto path executed in all 90
runs, and the effective optimizer was **AdamW at 0.001429 with momentum 0.9** — not SGD at
0.01, which is what a reader would assume from the recorded `lr0`.

> **Reporting note.** Quoting `lr0 = 0.01` from the checkpoint or from ultralytics' defaults
> would be wrong by a factor of 7, and would name the wrong optimizer family. If a paper
> states hyperparameters, state the resolved ones and say they came from `optimizer=auto`.

---

## Batch: 4 runs were silently reduced

`results.jsonl` reports `batch: 8` for all 90 runs, because it records what was *requested*.
Four checkpoints record `batch: 4`:

```
hbb-s-fold2-416     hbb-x-fold0-416     hbb-x-fold1-416     obb-x-fold2-416
```

All four are 416 px, three are the largest model. They were trained during an 8-way parallel
sweep on one RTX 4090, and ultralytics halves the batch and retries when it hits OOM in the
first epoch (`trainer.py`, max 3 retries, single-GPU only):

```python
self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
LOGGER.warning(f"{error} with batch={old_batch}. Reducing to batch={self.batch_size} ...")
```

**The effect on the experiment is nil, and this is checkable rather than asserted.**
Ultralytics accumulates gradients to a nominal batch of `nbs=64` and scales weight decay by
the same ratio:

| nominal batch | accumulate | effective batch | scaled weight decay | iterations | optimizer |
|---|---|---|---|---|---|
| 8 | 8 | **64** | 0.000500 | 100 | AdamW @ 0.001429 |
| 4 | 16 | **64** | 0.000500 | 100 | AdamW @ 0.001429 |

Effective batch, weight decay, optimizer and learning rate are **identical**. The only real
difference is batch-norm statistics, computed over 4 rather than 8 images per forward pass.

Those four cells are not outliers in the results, which is consistent with a negligible
effect — but the honest statement is that the difference is small and confounded with the
cells themselves, not that it was measured to be zero.

---

## Early stopping is off, deliberately

`patience` equals `epochs`, so it can never trigger. From `train_yolo.py`:

> early stopping OFF: with 3 folds it fires at random and measures luck, not architecture

With 30 training images, validation mAP is noisy enough between epochs that patience-based
stopping halts different folds at different points for reasons unrelated to the model. Every
run therefore sees exactly 100 epochs.

---

## What is *not* recovered

- **Per-epoch loss and metric curves.** Ultralytics writes `results.csv` and `args.yaml` into
  each run directory; those stayed on Kaggle and the rented GPU and were not retrieved. Only
  the final checkpoint and the summary row per run are held locally.
- **The auto-optimizer log line** confirming the resolution at runtime. The derivation above
  reconstructs it from the recorded arguments and the pinned ultralytics source
  (**8.4.45**, also recorded in every checkpoint), and the `warmup_bias_lr: 0.0` fingerprint
  corroborates it — but the original stdout was not kept.

If a reviewer requires the log line itself, re-running one cell for one epoch under
ultralytics 8.4.45 reproduces it in the first few lines of output.

---

## DEIMv2, for contrast

DEIMv2's schedule is not auto-resolved and was pinned explicitly, because its variants ship
mutually incompatible COCO recipes (60–468 epochs). All 24 runs used:

```
epochs 120 · flat_epoch 64 · no_aug_epoch 20 · stop_epoch 100 · matcher_change_epoch 90
total_batch_size 4 · drop_last False · num_classes 4
```

`REPORT.md` §4 explains why the whole arm had to be re-run to make that true.
