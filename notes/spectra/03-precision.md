# Precision: bf16 isn't what limits length generalization

Figure: `figures/spectra/precision_bf16_vs_fp32.pdf`.

**Setup.** The paper's DT evals run with `provider.dtype=bfloat16`, and the HF fork computes `β = 2·sigmoid(b)` in bf16. A near-reflection therefore has either |μ| = 1 exactly or |μ| ≤ 0.992. We reran the regular eval harness in fp32 (`scripts/eval_dt.py --dtype float32`) for GDN± and Hybrid± on general finite and soluble data, tiers up to 64K tokens, with the same `limit=100` sample per tier as the bf16 jobs. The comparison is on paired instances only.

**Result.** fp32 matches bf16 within noise, and if anything is slightly worse.

| model | general finite: bf16 → fp32 | soluble: bf16 → fp32 | paired n |
|---|---|---|---|
| GDN± | 86.9% → 84.5% | 100% → 100% | 328 / 325 |
| Hybrid± | 75.0% → 74.7% | 95.7% → 95.1% | 328 / 325 |

The accuracy-vs-length curves overlap, including where they fall.

**Interpretation.** Rounding β near 2 isn't the bottleneck. This agrees with the gate statistics: in GDN±'s memory head, value-token reflections have mean |μ| ≈ 0.89, far below anything bf16 rounding would produce. The leak is learned, not numerical. The spectra fp32 runs (gate statistics, readout) are still in progress.
