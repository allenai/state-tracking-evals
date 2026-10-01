# Interventions: what changes long-length accuracy

Figure: `figures/spectra/interventions.pdf`. Gates are rewritten inside the GDN kernel call during a normal forward pass. The metric is per-register readout accuracy at checkpoints within each long prompt (16 prompts per run, bf16 unless noted).

Readout accuracy by assignments read so far:

| GDN±, general finite | ≤ 256 | 1K–4K | 4K–16K | > 16K |
|---|---|---|---|---|
| none | 0.97 | 0.79 | 0.40 | 0.36 |
| snap β→2 where β > 1.98 | 0.97 | 0.81 | 0.40 | 0.37 |
| snap α→1 where α > 0.999 | 0.97 | 0.78 | 0.39 | 0.36 |
| both snaps | 0.97 | 0.79 | 0.38 | 0.36 |
| α → α^0.5 (half the forgetting) | 0.92 | 0.21 | 0.20 | 0.19 |
| float32 | 0.97 | 0.73 | 0.37 | 0.35 |

| Hybrid±, general finite | ≤ 256 | 1K–4K | 4K–16K | > 16K |
|---|---|---|---|---|
| none | 0.98 | 0.67 | 0.39 | 0.35 |
| snap β→2 where β > 1.98 | 0.98 | 0.66 | 0.39 | 0.35 |
| snap α→1 where α > 0.999 | 0.98 | 0.62 | 0.41 | 0.35 |
| α → α^0.5 | 0.99 | 0.40 | 0.27 | 0.29 |
| attention window 4K tokens | 0.98 | 0.65 | 0.40 | 0.35 |
| float32 | 0.98 | 0.63 | 0.40 | 0.35 |

Soluble shows the same pattern. For example, GDN± at > 16K assignments: none 0.85, snaps 0.74–0.87, α^0.5 0.19.

**Takeaways.**
- **Making near-exact gates exact does nothing.** Snapping β > 1.98 to exactly 2, or α > 0.999 to exactly 1, moves accuracy by at most ±0.03 (and ±0.11 at > 16K on soluble). Together with fp32 matching bf16, this rules out rounding near ±1 as the bottleneck.
- **Forgetting is load-bearing.** Halving it (α → α^0.5) breaks the models near the training length and beyond: GDN± general finite at 1K–4K goes 0.79 → 0.21, soluble 0.98 → 0.13. The models rely on the gate to clear stale state, so "less forgetting" isn't a fix.
- **The hybrid's attention isn't the long-range bottleneck.** Restricting both NoPE attention layers to the last 4K tokens leaves accuracy unchanged at every length. That rules out "attention averages over an ever-longer context" as the reason Hybrid± fails at a fixed ~1.4–2.9K assignments.
- **What these interventions didn't test.** The leak the retention analysis points to is in the memory heads' value-token reflections, at β ≈ 1.89 (|μ| ≈ 0.89). The β > 1.98 snap never touches those. A targeted snap (β > 1.75 → 2) in the memory head alone and in all heads is running: GDN± L1 H6, Hybrid± L5 H1. *Result: pending.*
