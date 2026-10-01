# Training dynamics: decay phases help extrapolation, but a ceiling remains

Figure: `figures/spectra/training_steps_periodic.pdf`. General finite data, checkpoints converted with `spectra/convert.py`. Checkpoint pairs straddle each learning-rate decay of the WSDS schedule. The memory head is fixed at the final checkpoint's most persistent head.

| GDN± (memory head L1 H6) | readout ≤ 256 / 1K–4K / 4K–16K assignments | write retained at 1K / 4K / 16K tokens | leak per value-token reflection |
|---|---|---|---|
| step 3695 (before decay 1) | 0.96 / 0.44 / 0.31 | 0.67 / 0.39 / 0.20 | 0.039 |
| step 4106 (after decay 1) | 0.97 / **0.71** / 0.39 | 0.78 / 0.48 / 0.27 | 0.090 |
| step 7000 | 0.97 / 0.59 / 0.35 | 0.74 / 0.44 / 0.23 | 0.061 |
| step 7800 (before decay 2) | 0.97 / 0.47 / 0.36 | 0.47 / 0.26 / 0.12 | 0.079 |
| step 8211 (final) | 0.97 / **0.79** / 0.40 | 0.75 / 0.45 / 0.22 | 0.114 |

| Hybrid± (memory head L5 H1) | readout ≤ 256 / 1K–4K / 4K–16K | write retained at 1K / 4K / 16K |
|---|---|---|
| step 3627 (before decay 1) | 0.99 / 0.35 / 0.28 | 0.34 / 0.02 / 0.00 |
| step 4030 (after decay 1) | 0.99 / 0.49 / 0.33 | 0.33 / 0.01 / 0.00 |
| step 7656 (before decay 2) | 0.98 / 0.52 / 0.36 | 0.58 / 0.19 / 0.02 |
| step 8000 | 0.99 / 0.61 / 0.37 | 0.32 / 0.05 / 0.01 |
| step 8059 (final) | 0.98 / **0.67** / 0.39 | 0.34 / 0.06 / 0.01 |

**Takeaways.**
- **In-distribution accuracy saturates early (~0.97), but extrapolation keeps improving, and mostly during the decays.** GDN± at 1K–4K assignments goes 0.44 → 0.71 across decay 1 and 0.47 → 0.79 across decay 2. Hybrid± improves across both decays too (0.35 → 0.49, 0.52 → 0.67).
- **The improvement shifts the failure point by ~1.5–2×; it doesn't remove it.** No checkpoint extrapolates past ~4–8K assignments. Accuracy at 4K–16K inches up from 0.31 to 0.40.
- **For GDN±, memory-head retention tracks extrapolation across checkpoints**, consistent with note 02. The leak per reflection *rises* over training (0.04 → 0.11) even as accuracy improves. That agrees with the snap null in note 04: reflection exactness isn't what the model is optimizing, or what limits it.
- **For Hybrid±, retention doesn't track.** Step 7656 has the best retention but mid-range accuracy, so its extrapolation gains come from somewhere other than this head.

**Reading.** Extrapolation is still improving at the end of training, which makes the horizon partly a training-budget effect. Training only rewards memory out to ~256 assignments, and nothing here suggests the model would discover a length-general solution without longer training sequences. The causal test is to train on longer sequences, which checkpoints can't substitute for.
