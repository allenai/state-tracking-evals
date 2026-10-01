# Baseline spectra: what the gates do at long positions

Runs: one per model (GDN±, Hybrid±, GDN+, Hybrid+) × matched train/eval data, bf16 (as in the paper), final checkpoint. Each covers 8 prompts under 2K tokens and 8 at ~256K tokens, stratified by variable count. Figures are in `figures/spectra/`.

**TL;DR.** The per-token eigenvalues don't decay at long positions. In every ± model the spectrum (including the bimodal mass near μ ≈ −1) is stationary from ~1K to 256K tokens, and residual-stream inputs barely drift. What does predict where each model fails is how long its most persistent head retains a write. That retention is a function of *lag*, identical wherever the write happens. Writes leak through the product of later transitions (inexact reflections and overwrites), not through the forget gate.

**1. No eigenvalue drift** (`gate_spectrum_heatmaps_periodic.pdf`, `gate_summaries_periodic.pdf`).
- In the top ± heads, 40–64% of content tokens have μ < −0.5. The models do use negative eigenvalues; this isn't a pure copy-chain strategy.
- The μ distributions and three summaries are flat from 1K to 200K tokens: reflection mass (~0.5–0.6), leak per reflection (−E log|μ| ≈ 0.22–0.25, i.e. |μ| ≈ 0.8), and forgetting per token. They change only in the first few hundred tokens, during the opening run of constant assignments.
- The + models stay in μ ∈ [0, 1], as they must.

**2. No meaningful residual drift** (`hidden_drift_periodic.pdf`). Every layer's mean input per token role keeps cosine ≥ 0.96 with its in-distribution mean out to 256K tokens, including the layers after the hybrid's NoPE attention. It dips slightly only past ~100K, while failures start around ~10K tokens.

**3. Accuracy within a sequence reproduces the eval decay** (`readout_by_position.pdf`). We teacher-force the gold answer at 24 checkpoints inside each long prompt (placed between swaps). General-finite accuracy drops at 2–5K assignments for the ± models and right after the training cutoff for the + models, the same pattern as the paper's DT figure.

**4. Write retention in each model's memory head predicts the failure point** (`retention_periodic.pdf`). This is the fraction of a write that survives the transitions after it:

| general finite | memory head | retained at 1K / 4K / 16K tokens | readout < 90% at |
|---|---|---|---|
| GDN± | L1 H6 | 0.75 / 0.45 / 0.22 | 2.9K assignments |
| Hybrid± | L5 H1 | 0.34 / 0.06 / 0.01 | 1.4K |
| GDN+ | L6 H6 | 0.07 / 0.03 / 0.004 | 0.36K |
| Hybrid+ | L1 H0 | 0.06 / 0.03 / 0.02 | 0.36K |

- The curves for writes at 1K, 8K, 32K and 131K tokens lie on top of one another, so the dynamics are stationary. The memory is finite and leaks with lag.
- GDN±'s memory head has α ≈ 1 − 6×10⁻⁶ per token, about 2% loss over 4K tokens. Almost all of the loss comes from the Householder products themselves. It reflects on value tokens (97% of them have μ < −0.5), but none of those reflections is exact: the mean |μ| is ≈ 0.89. On semicolons (66% reflect), about half are exact (β = 2 in bf16).
- The ordering GDN± > Hybrid± > + models matches the eval ordering. This is consistent with the Phase 0 finding that GDN±'s horizon shrinks with the number of variables: more values share the head and interfere.

**5. Soluble is solved by rotation in ± models, by something else in GDN+** (`statement_eigenvalues_solvable.pdf`).
- For `x <- (x + d) mod 100;`, the transition products of the ± models' heads have complex eigenvalues on or near the unit circle: a ring for Hybrid± L6 H6, discrete clusters for GDN± L0 H5. That's rotation-based counting, and it doesn't drift inward at long positions.
- GDN+'s eigenvalues are real and near 0, so it can't rotate, yet it is still ≥ 93% accurate out to 23K assignments in the eval.
- A caveat on the readout: Hybrid±'s per-register readout on soluble collapses to ~45% after ~1K assignments, even though its eval multiple-choice accuracy stays ≥ 90%. Soluble distractors are apparently easy to rule out from partial knowledge, so the MC metric overstates its state tracking.

**What we can't separate yet.** In this data, a value's age (statements since its constant write) is almost collinear with position: long prompts have only ~1–64 constants (`dissociation_periodic.pdf`, where mass sits on the diagonal). Telling "errors grow with lag" apart from "errors grow with position" behaviorally needs a test set with late resets. The retention result already points to lag.

**Pending:** fp32 reruns of both the eval harness and the spectra; gate interventions (snap β→2, snap α→1, α→α^0.5, both snaps, hybrid attention window 4K); earlier training checkpoints.
