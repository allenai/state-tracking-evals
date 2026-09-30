# GDN spectra: what we measure (methods note)

**Question.** Hybrid± and GDN± models learn general-finite (and soluble) StateBench data in distribution, then decay a few × past the training length. Is that because the eigenvalues of the GDN transitions drift or leak at long positions?

**Key fact: the per-token spectrum is closed-form.** Keys are L2-normalized in the kernel, so each head's transition `A_t = α_t (I − β_t k kᵀ)` has exactly two eigenvalues: `α_t` (×127) and `μ_t = α_t(1 − β_t)` along `k_t`. Reading `g = log α` and `β` is reading the spectrum; no eigendecomposition is needed. For ± models, `β = 2σ(·) ∈ (0, 2)`, so `μ ∈ (−1, 1)`, with −1 reachable only in the limit.
- `μ ≈ +1`: the head ignores the token.
- `μ ≈ 0`: delta-rule overwrite (aperiodic-style).
- `μ ≈ −1`: reflection.

**Caveats built into the design.**
- In layer 0, `α` and `β` depend only on the current token (there's no conv on the a/b projections). Any positional drift must come from the residual stream in layers ≥ 1.
- Evals run in bf16, and `sigmoid(b)` is computed in bf16, so `β` near 2 is either exactly 2 or ≤ 1.992 (|μ| ≤ 0.992). That's a precision confound; we rerun in fp32.

**What the pipeline records** (`spectra/`, one Beaker job per model × data × condition). Prompts are prefilled in segments that end on statement boundaries (validated against a one-shot forward pass at job start). A wrapper on each GDN kernel records:
1. **Gate moments and histograms** per layer × head × token role × position bin: reflection mass `P(μ<−0.5)`, leak per reflection `−E log|μ|`, forgetting `−E log α`, and exact `β=2` / `α=1` counts.
2. **Readout at each position.** At ~24 log-spaced checkpoints inside each long prompt, the cache is branched, the gold answer for the state *at that point* is teacher-forced, and every register's value token is scored. This gives accuracy vs position within one sequence, alongside three covariates per register:
   - absolute position;
   - value age (statements since the value entered as a constant);
   - reference/increment hops.
3. **Write retention.** `‖Φ(t,s) k_s‖` for the strongest write per 64-token block, as a function of lag and write position. Exact reflections keep it at 1; forgetting, leaky reflections and overwrites shrink it.
4. **Per-statement spectra.** Eigenvalues of the product of transitions across one whole statement (e.g. `x <- (x + d) mod 100;`).
5. **Residual drift.** Per-role mean input to each layer by position, compared with its in-distribution mean.

**Interventions** (same pipeline):
- snap `β → 2` when `β > 1.98`;
- snap `α → 1` when `α > 0.999`;
- `α → α^0.5`;
- sliding-window the hybrid's attention (last 4K tokens);
- fp32 inference.

**How the hypotheses read out.**
| Hypothesis | Signature |
|---|---|
| Drift (spectrum shifts at long positions) | gate curves change with position; errors track absolute position; residual cosine falls |
| Leak (stable but \|μ\|, α < 1) | flat gate curves; errors track value age; retention decays with lag at every start position; snapping helps |
| Precision (bf16) | fp32 moves the decay point |
| Copy chains (few real reflections) | low reflection mass; errors track hops |

Code: `spectra/` on branch `jacksonp/spectral-analysis`. Figures: `figures/spectra/`.
