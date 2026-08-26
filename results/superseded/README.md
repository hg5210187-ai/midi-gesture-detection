# Superseded measurements

## latency_deim_n_CONTAMINATED.json

DEIMv2-n measured at **13.85 ms**. Not used.

That run executed immediately after two failed Core ML export attempts in the same batch,
which were still competing for CPU and memory. Four measurements of the same package gave
medians of **4.80, 4.90, 5.26, 6.97** and this single **13.85** — the outlier is the one that
followed the failed exports.

`results/latency_deim_n_clean.json` (4.80 ms, numerically verified against PyTorch at
0.568 px box deviation) is the measurement used in the figures and the report.

Kept rather than deleted so the discrepancy is on the record.
