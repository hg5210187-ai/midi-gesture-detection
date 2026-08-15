# Hand-Gesture MIDI Instrument — detection study

Is a vision-based hand-gesture MIDI instrument deployable on a consumer laptop?

**Yes.** YOLO26-s at 320 px, axis-aligned, exported to Core ML, sustains **7.93 ms** on a
MacBook Air M4 for 15 continuous minutes without breaching a 10 ms budget, at
**0.8056 ± 0.0437** mAP50-95 (3-fold cross-validation) and **0.9846** macro AUC.

📄 **[REPORT.md](REPORT.md)** — the full study: planning, method, results, figures, and the
reasoning behind each decision, including the ones that turned out to be wrong.

---

## What this is

Three hand gestures — `thumbout`, `openhand`, `closedhand` — detected on **both hands
simultaneously**, converted to MIDI. The study compares two detector families on accuracy
*and* on latency measured on the target laptop rather than on the GPU that trained them.

- **114 training runs**: 90 YOLO26 (5 sizes × 2 box geometries × 3 resolutions × 3 folds) +
  24 DEIMv2 (8 variants × 3 folds)
- **38 models** with pooled confusion matrices, one-vs-rest AUC-ROC, and held-out test scores
- Latency measured **burst and sustained** on an Apple M4

## Headline results

| | |
|---|---|
| Deployed model | YOLO26-s, axis-aligned, 320 px |
| Accuracy (3-fold CV, exact IoU) | 0.8056 ± 0.0437 mAP50-95 |
| Macro AUC-ROC | 0.9846 |
| Held-out test (30 annotations) | 0.6855 ± 0.0500 |
| Sustained latency (15 min, from cold) | 7.93 ms median, p95 8.83 |

Three findings that generalise beyond this instrument:

1. **Capacity is not the binding constraint** at small dataset sizes — both families are flat
   above a threshold, spanning a 24× parameter range for ~0.14 of mAP.
2. **The deployment runtime must be measured before drawing conclusions about deployment.**
   On PyTorch-MPS an 18 ms fixed overhead hides everything; under Core ML the same model spans
   6.3 ms across resolutions, monotonically. Measuring MPS would have led to abandoning the
   axis that made the result possible.
3. **Burst latency is the wrong benchmark for a continuously-running instrument on passively
   cooled hardware.** Four configurations pass a burst test and fail under sustained load.

## Repository layout

```
REPORT.md                  the study, end to end
STATUS.md                  operational record: environment, versions, resume notes
results/RESULTS.md         results, condensed
results/METRICS.md         confusion-matrix and AUC methodology
results/LATENCY.md         latency methodology

scripts/                   dataset construction, evaluation, latency, figures
kaggle/                    training drivers (YOLO26 and DEIMv2)
data/splits/               fold assignment
collection/                the shot list the photos were taken from
figures/                   38 confusion matrices, 38 ROC curves, publication figures
results/                   per-model JSON: matrices, AUC, threshold sweeps, test scores
```

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics==8.4.45          # version is load-bearing, see STATUS.md

# evaluation from prediction dumps (no GPU needed)
python3 scripts/pool_metrics.py --preds results/preds_yolo results/preds_deim
python3 scripts/test_eval.py   --preds results/preds_yolo_test results/preds_deim_test
python3 scripts/tradeoff.py
python3 scripts/paper_figure.py --model yolo26_hbb-s-320
```

Core ML export and latency need a separate environment — coremltools 9.0 caps at torch 2.7:

```bash
python3 -m venv .venv-coreml && source .venv-coreml/bin/activate
pip install coremltools==9.0 "torch==2.7.*" "torchvision==0.22.*" "numpy<2" "scipy<1.15" ultralytics
python3 scripts/export_coreml.py --weights models/hbb-s.pt --imgsz 320
python3 scripts/sustained.py --packages models/hbb-s_320.mlpackage --imgsz 320 --minutes 15
```

DEIMv2 training and inference need the upstream repo, which is **not vendored here**:

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git deim_local
```

## Artifacts not in this repository

Model weights and the image dataset are too large for git and are stored separately.

| artifact | size | where |
|---|---|---|
| 90 YOLO26 checkpoints | 4.5 GB | Google Cloud Storage |
| 24 DEIMv2 checkpoints | 5.4 GB | Google Cloud Storage |
| 8 DEIMv2 COCO-pretrained checkpoints | 455 MB | Google Cloud Storage |
| Core ML exports (30 `.mlpackage`) | 2.0 GB | Google Cloud Storage |
| Source photographs | 17 MB | **not distributed — see below** |

`scripts/upload_gcs.sh` uploads them; the bucket URI is recorded there once set.

## A note on the dataset

The source images are webcam-style photographs of **one person**, showing their face and the
interior of their home, not cropped hands. They are **identifiable personal data** and are
excluded from this repository by `.gitignore`.

Everything needed to reproduce the *analysis* is here — prediction dumps, fold assignments,
per-model metrics — so every number in the report can be recomputed without the images.
Re-training from scratch requires the images, which are not publicly distributed.

**Single-subject dataset.** Every limitation in this study traces back to that: the models
have seen one person's hands, in 12 places, on one day. The accuracy figures should be read
as an upper bound on what a second person would get.

## Citation

If this is useful, please cite the report. Study conducted August 2026.
