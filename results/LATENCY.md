# M4 latency — Core ML on the MacBook Air M4

Measured on the target laptop, not the training GPU. `scripts/export_coreml.py` (YOLO26) and
`scripts/export_coreml_deim.py` (DEIMv2). Budget is **10 ms for the model**, an engineering
target rather than a perceptual threshold — the camera alone costs 30–50 ms of the chain.

Method: end-to-end `predict()`, warm-up discarded, **3 repeats with model order shuffled
between them** so thermal drift cannot masquerade as an architecture effect. Median and p95,
never the mean — the distribution has a long right tail. Jitter = p95 − median, which is what
a player actually feels: uniform lag can be anticipated, variable lag cannot.

## YOLO26 — 8 configurations meet the budget

| model | imgsz | median ms | p95 | jitter | fps | ≤10 ms | MPS ms | Core ML speed-up |
|---|---|---|---|---|---|---|---|---|
| hbb-n | 320 | 7.09 | 7.61 | 0.52 | 141 | **yes** | 18.47 | 2.61x |
| obb-n | 320 | 7.22 | 8.31 | 1.10 | 139 | **yes** | 18.96 | 2.63x |
| hbb-s | 320 | 7.54 | 8.13 | 0.59 | 133 | **yes** | 17.64 | 2.34x |
| obb-s | 320 | 7.57 | 8.35 | 0.78 | 132 | **yes** | 18.44 | 2.44x |
| hbb-n | 416 | 8.57 | 8.96 | 0.39 | 117 | **yes** | 19.99 | 2.33x |
| obb-n | 416 | 8.89 | 9.99 | 1.10 | 112 | **yes** | 17.91 | 2.01x |
| hbb-s | 416 | 9.10 | 10.05 | 0.94 | 110 | **yes** | 24.52 | 2.69x |
| obb-s | 416 | 9.30 | 10.17 | 0.87 | 108 | **yes** | 24.59 | 2.64x |
| hbb-n | 640 | 13.39 | 14.82 | 1.44 | 75 | no | 18.03 | 1.35x |
| obb-n | 640 | 14.50 | 17.09 | 2.59 | 69 | no | 18.69 | 1.29x |
| hbb-s | 640 | 14.75 | 16.50 | 1.76 | 68 | no | 26.41 | 1.79x |
| obb-s | 640 | 15.07 | 15.90 | 0.83 | 66 | no | 26.45 | 1.76x |
| hbb-m | 640 | 21.32 | 23.85 | 2.54 | 47 | no | 40.24 | 1.89x |
| obb-m | 640 | 21.47 | 22.65 | 1.18 | 47 | no | 40.36 | 1.88x |
| obb-l | 640 | 22.61 | 23.87 | 1.26 | 44 | no | 47.40 | 2.10x |
| hbb-l | 640 | 22.93 | 24.96 | 2.03 | 44 | no | 45.72 | 1.99x |
| hbb-x | 640 | 34.44 | 40.52 | 6.08 | 29 | no | 93.08 | 2.70x |
| obb-x | 640 | 34.98 | 38.59 | 3.60 | 29 | no | 92.85 | 2.65x |

### Core ML is the whole story, and resolution only became a lever because of it

On **PyTorch-MPS, input resolution does essentially nothing**: hbb-n measured 20.14 / 18.82 /
21.24 ms at 320 / 416 / 640 — a 2.4 ms spread across a 4× change in pixel count, and not even
monotonic. Fixed dispatch overhead dominates, so the model size is invisible.

Under Core ML the same model runs **7.09 / 8.57 / 13.39 ms** — monotonic, and a 6.3 ms range.
The Neural Engine removes the fixed overhead, and only then does resolution start to matter.
Anyone tuning resolution on the MPS numbers would have concluded it was pointless.

Core ML also **collapses jitter**: 0.39–2.59 ms versus up to 12.39 ms on MPS. For a musical
instrument that matters more than a couple of ms of median.

The speed-up grows with model size — 1.35× for n, 2.70× for x — because the fixed overhead is
a smaller fraction of a bigger model's work.

**OBB costs 6–8 ms over HBB** on MPS but only ~0.2–0.4 ms under Core ML. Oriented boxes are
close to free on the ANE.

## DEIMv2 — partly measurable, and the reason matters

There is no Core ML path for DEIMv2; `scripts/export_coreml_deim.py` is new work. Three
converter incompatibilities had to be fixed to get any DEIMv2 model through at all, all
documented in that script: a rank-1 `linear` (D-FINE's distribution-integral head), float
`gather` indices produced by lowering integer floor-division, and `aten::gather` arity.

| variant | backbone | converts | reproduces PyTorch | latency |
|---|---|---|---|---|
| n | HGNetv2 | yes | **yes** — 0.57 px box dev, 0.0017 score dev | ~5–7 ms at 640 |
| s | DINOv3 | yes | **no** — 259 px box dev | not reportable |
| m | DINOv3 | yes | **no** — 260 px box dev | not reportable |
| l | DINOv3 | yes | **no** — 213 px box dev | not reportable |
| x | DINOv3 | yes | **no** — 277 px box dev | not reportable |

**The DINOv3 line converts without error and produces wrong output.** All four fail the same
way — labels correct, boxes hundreds of pixels off — while the one HGNetv2 variant is exact.
That is a backbone-specific converter fault, not noise. Latency for those four is deliberately
**not reported**: a number from a graph that does not reproduce the model is worse than no
number. This is enforced in code, not by convention — the exporter refuses to time a model
that fails verification.

**DEIMv2 cannot change resolution.** 320 and 416 fail with `size of tensor a (100) must match
tensor b (400)`: the architecture is tied to `eval_spatial_size: 640`. The resolution lever
that puts YOLO26 under budget is not available here without retraining.

### Confidence in the DEIMv2-n number is lower than the YOLO numbers

Four measurements of the same model gave medians of 4.90, 5.26, 6.97 and 13.85 ms. The 13.85
run had two failed exports immediately before it; the three clean repeats cluster at 4.9–7.0.
But p95 ran 10.1–15.8 ms, so jitter is materially worse than YOLO26's Core ML jitter of under
2.6 ms. **Before this goes in the paper it needs the same discipline the YOLO arm got** —
repeats with shuffled order, reusing a pre-compiled package rather than re-exporting each
time. Treat ~5–7 ms as indicative, not final.

Taken at face value it is still notable: DEIMv2-n at 640 is faster than any YOLO26 at 640
(best 13.39 ms) despite scoring higher mAP (0.792 vs 0.763 for hbb-n). A plausible reason is
that DEIMv2 is **NMS-free**, while the YOLO Core ML export bakes in NMS, which is not an
ANE-friendly operation. If that holds up it is a genuine architectural advantage for DETRs on
Apple hardware and worth stating.

## What this means for the instrument

The 10 ms budget is met, on real hardware, by 8 YOLO26 configurations — and comfortably: 7.09
ms at 320, 8.57 ms at 416 for hbb-n, with jitter under 1.1 ms. The deployability question the
study exists to answer is **yes**, provided the model is exported to Core ML and run at 320 or
416 rather than 640.

The open accuracy question is what those resolutions cost. The 60 Kaggle resolution-sweep
cells were deprioritised while the MPS numbers suggested resolution was not a lever; the Core
ML numbers reverse that. Those rows are now the highest-value missing data in the study.
