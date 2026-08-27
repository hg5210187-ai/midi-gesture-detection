# RESULTS — final

Everything below was measured, and every number has a file behind it. Where a claim was
revised during the study the revision is stated rather than the history quietly dropped.

---

## 1. The answer

**Yes — a vision-based hand-gesture MIDI instrument is deployable on a MacBook Air M4.**

The deployable model is **YOLO26-s, axis-aligned, 320 px**:

| | |
|---|---|
| accuracy (3-fold CV, exact IoU) | **0.8056 ± 0.0437** mAP50-95 |
| accuracy (held-out test, 30 anns) | 0.6855 ± 0.0500 |
| macro AUC-ROC | **0.9846** |
| sustained latency, 15 min from cold | **7.93 ms** median, p95 8.83 |
| budget | holds all 15 windows, 111,884 iterations |
| misses / false alarms (90 anns) | 3 / 3 |

It is the highest-accuracy configuration that holds the 10 ms budget under sustained load.

---

## 2. Accuracy — YOLO26, exact IoU, top 12 of 30 cells

| cell | CV mAP50-95 | test mAP50-95 |
|---|---|---|
| hbb-m@416 | 0.8373 ± 0.0056 | 0.7658 |
| hbb-l@416 | 0.8225 ± 0.0376 | 0.7552 |
| hbb-x@416 | 0.8155 ± 0.0519 | 0.7836 |
| hbb-m@320 | 0.8122 ± 0.0667 | 0.7443 |
| hbb-s@640 | 0.8080 ± 0.0438 | 0.7574 |
| hbb-s@320 | 0.8056 ± 0.0437 | 0.6855 |
| hbb-l@320 | 0.8018 ± 0.0356 | 0.7586 |
| hbb-x@320 | 0.7987 ± 0.0349 | 0.7692 |
| hbb-s@416 | 0.7910 ± 0.0384 | 0.7618 |
| hbb-n@640 | 0.7814 ± 0.0227 | 0.7409 |
| hbb-l@640 | 0.7566 ± 0.0491 | 0.7237 |
| hbb-m@640 | 0.7364 ± 0.0826 | 0.7489 |

## 3. Accuracy — DEIMv2

| model | CV mAP50-95 | test mAP50-95 |
|---|---|---|
| deimv2/l | 0.8428 | 0.7928 |
| deimv2/s | 0.8409 | 0.8014 |
| deimv2/x | 0.8337 | 0.7824 |
| deimv2/m | 0.8295 | 0.7591 |
| deimv2/n | 0.7922 | 0.7428 |
| deimv2/femto | 0.1309 | 0.0787 |
| deimv2/pico | 0.0850 | 0.0397 |
| deimv2/atto | 0.0642 | 0.0117 |

---

## 4. Latency — sustained is the number that matters

Burst benchmarks describe a cold machine. The MacBook Air is **fanless**, and a gesture
instrument runs continuously, so every candidate was also run for 15 minutes:

| model | thermal start | median ms | p95 | first → last window | verdict |
|---|---|---|---|---|---|
| hbb-n@320 | cold | 7.22 | 8.05 | 7.19 → 7.91 | **holds** |
| hbb-s@320 | cold | 7.93 | 8.83 | 7.72 → 8.26 | **holds** |
| obb-s@320 | cold | 8.29 | 9.79 | 7.73 → 8.53 | **holds** |
| hbb-n@416 | cold | 8.88 | 12.47 | 8.74 → 10.84 | breach @ 540s |
| hbb-s@320 | **preheated** | 9.38 | 12.07 | 7.87 → 9.19 | breach @ 300s |
| hbb-m@320 | cold | 9.96 | 10.87 | 9.17 → 10.08 | breach @ 360s |
| hbb-l@320 | **preheated** | 10.60 | 14.43 | 9.76 → 11.33 | breach @ 360s |

Three findings, all of which changed a conclusion:

**Sustained load costs +0.5 to +2.1 ms, and it arrives as a step.** Latency sits flat for
5–9 minutes, transitions to a higher level, and plateaus there. A burst median above roughly
8 ms does not survive.

**Thermal start matters as much as the model.** `hbb-s@320` measured 9.38 ms preheated and
**7.93 ms cold** — a 1.45 ms difference, larger than the gap between several models. Within a
batch only the first model starts cold; 60 s of cooldown does not undo a 15-minute run. Any
sustained comparison must control for this.

**A median under budget is not a pass.** `hbb-m@320` medians 9.96 ms yet is above 10 ms for
its last nine windows. The verdict is per-window, not on the median.

**Core ML is what makes any of this possible.** PyTorch-MPS carries ~18 ms of fixed dispatch
overhead: on MPS the whole 24× capacity range spans ~2 ms and is not even monotonic in
resolution, so resolution looks like a dead end. Under Core ML `hbb-n` runs 7.09 / 8.57 /
13.39 ms at 320 / 416 / 640 — monotonic, and a 6.3 ms range. Core ML also collapses jitter
from up to 12.4 ms to under 2.6 ms.

---

## 5. The measurement error that reversed a conclusion

Ultralytics matches oriented boxes with **ProbIoU** and axis-aligned boxes with real box IoU —
`obb/val.py:94` vs `detect/val.py:317`. Measured over the 30 cells at 640 px
(`results/probiou_check.json`):

| geometry | ultralytics − exact IoU |
|---|---|
| hbb (control) | **-0.0005** |
| obb | **+0.1777** |

The control agreeing to ~0.00 is what validates the exact-IoU evaluator; it was independently
confirmed against COCOeval on the DEIMv2 arm (+0.0008 ± 0.0028, `results/deim_evaluator_check.json`).

**Consequence: the apparent OBB advantage was entirely an artifact.** Rescored with exact
polygon IoU, the top 12 cells in the study are all HBB, and `obb-s@320` — recommended
mid-study on the inflated numbers — drops from 0.837 to **0.627**.

An earlier version of this study's notes claimed ProbIoU ran "+0.11 optimistic". That figure
had no source; it was asserted rather than measured or cited. The measured value is +0.175.

---

## 6. Error structure — one hard pair

Pooled over the 35 models that learned the task (`results/METRICS.md`):

**176 of 179 class confusions (98.3%) are thumbout ↔ closedhand. Only 3 involve openhand.**

| class | mean AUC | worst model |
|---|---|---|
| openhand | 0.9979 | 0.9650 |
| thumbout | 0.9497 | 0.8533 |
| closedhand | 0.9550 | 0.8906 |

thumbout is a fist with the thumb extended — one digit from closedhand — while openhand
differs in global hand shape. The same pair produced the four annotation errors found during
dataset construction. The vocabulary has exactly one hard pair and it is the anatomically
adjacent one, which is a gesture-design problem rather than a model problem.

---

## 7. Generalisation

Mean CV → test gap across all 30 YOLO cells: **-0.0417**; DEIMv2 **−0.052 ± 0.011**.
Uniform across a 6.6× capacity range, which is what an honest generalisation gap looks like —
the test set is four unseen places.

One exception matters for model choice: **`hbb-s@320` has the worst gap in the study, −0.120**,
against a mean of −0.042. Its CV advantage transfers less well than its neighbours'. It is
still the right deployment choice because the alternatives at that latency are worse on both
measures, but the paper should say so.

---

## 8. Capacity

**DEIMv2 has a hard floor.** pico (0.085) → n (0.792) is a ~9× jump for ~2× the parameters.
Below `n` these models do not learn the task at 60 training images at all.

**Above the floor, both families are flat.** DEIMv2 spans 0.792–0.843 from n to x; YOLO26
spans 0.699–0.837 across a 24× parameter range. Once capacity clears the threshold, more of
it buys almost nothing at this dataset size — which is why the smallest model above the
threshold is the right deployment choice.

**Backbone caveat.** DEIMv2 s/m/l/x are DINOv3-backboned; atto/femto/pico/n are HGNetv2. A
DEIMv2 win at the top is *foundation-model pre-training + a DETR head*, not architecture
alone. On the clean HGNetv2-vs-YOLO26 comparison, DEIMv2-n (0.792) sits just below the best
YOLO26 cell (0.837).

---

## 9. What is not established

- **DEIMv2 latency on the M4 is measured for one variant only.** `n` (HGNetv2) converts to
  Core ML and reproduces PyTorch exactly (0.57 px box deviation); s/m/l/x (DINOv3) convert
  silently and produce boxes 213–277 px wrong, so their latency is deliberately not reported.
  DEIMv2 also cannot change resolution — the architecture is fixed at 640.
- **`hbb-l@320` was only measured preheated**, so its breach is unresolved rather than proven.
- **Sustained runs are 15 minutes.** A full performance is longer; the plateau looks stable
  but was not measured beyond that.
- **No thermal telemetry.** `pmset -g therm` reports nothing unprivileged on this machine, so
  latency over time is reported as the effect rather than a proxy for the cause.

---

## 10. Files

```
results/cv_exact_iou.json        30 YOLO cells, exact-IoU CV, per fold and per cell
results/test_set.json            38 models on the held-out test set
results/pooled/                  38 confusion matrices + AUC + threshold sweeps
results/probiou_check.json       the ProbIoU measurement
results/deim_evaluator_check.json  COCOeval vs exact IoU, evaluator validation
results/cv_vs_test.json          per-cell generalisation gaps
results/latency_m4_*.json        burst latency, Core ML + MPS
results/latency_sustained*.json  15-minute runs
results/tradeoff.json            the Pareto front
figures/tradeoff.png             accuracy vs latency, deployable region shaded
figures/confusion/  figures/roc/ 38 each
```
