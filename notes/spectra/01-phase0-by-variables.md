# Phase 0: DT accuracy split by number of variables

Figures: `figures/spectra/accuracy_by_variables_periodic.pdf`, `figures/spectra/accuracy_by_variables_solvable.pdf`. These use the existing DT eval results (bf16, final checkpoints, matched train/eval, in-distribution padding), re-binned by variable count.

**General finite: GDN± and Hybrid± fail differently.** The table gives the first length bin (in assignments) where accuracy drops below 90%:

| model | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|
| GDN± | 46K | 12K | 5.8K | 5.8K |
| Hybrid± | 2.9K | 2.9K | 2.9K | 2.9K |
| GDN+ / Hybrid+ | 0.7K | 0.7K | 0.7K | 0.7–1.4K |

- **GDN±**'s horizon shrinks roughly 8× from m=2 to m≥8. With 2 variables (plus TMP) the swaps are just parity, and GDN± holds ~100% to 30K assignments, dipping to 84% at 46K. So at least for the ± recurrence, length alone doesn't break it; what matters is how many values it must keep apart and how long each is carried.
- **Hybrid±** breaks at ~2.9K assignments (~12K tokens) regardless of m, even on parity. A failure length that ignores problem size points to a positional mechanism rather than state capacity. The obvious suspect is the two NoPE attention layers averaging over ever-longer contexts. We're testing this directly by sliding-windowing attention (`attnwindow` runs).
- **+ models** fail right at the 4× cutoff for every m, as the theory predicts. For m=2 they stay well above chance (75–90%), consistent with a partial solution.

**Soluble: there's very little decay to explain.** Every model, including GDN+ and Hybrid+, is ≥ 93% out to 23K assignments at every m. The exception is Hybrid± at m=4 (dips to ~65% around 6K). GDN+ doing mod-100 increments perfectly out to 90× the training length bears on the paper's open question of whether GDN+ can implement mod. The spectra runs show how (see the per-statement eigenvalue figure).

**What this changes about the plan.** The eigenvalue story has to explain an m-dependent horizon for GDN± and an m-independent one for Hybrid±. The per-position readout covariates (position vs value age vs hops) and the attention-window intervention are aimed at exactly that split.
