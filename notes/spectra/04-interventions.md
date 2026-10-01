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
- **Making the memory head's reflections exact doesn't help either.** The retention analysis pointed at GDN±'s memory head (L1 H6), whose value-token reflections sit at β ≈ 1.89 (|μ| ≈ 0.89, never exact). Snapping β > 1.75 → 2, in that head alone or in all heads, makes 100% of those reflections exact (the leak per reflection goes from 0.114 to 0.001) and leaves accuracy unchanged at every length:

| GDN±, general finite | ≤ 256 | 1K–4K | 4K–16K | > 16K |
|---|---|---|---|---|
| none | 0.97 | 0.79 | 0.40 | 0.36 |
| snap β > 1.75 → 2, memory head | 0.97 | 0.80 | 0.42 | 0.35 |
| snap β > 1.75 → 2, all heads | 0.97 | 0.80 | 0.41 | 0.35 |

  Hybrid± is unchanged as well. Its memory head (L5 H1) doesn't reflect on value tokens, so the snap barely touches it.

**What this means for the eigenvalue hypothesis.** Several things are now ruled out as what sets the length horizon:
- position-dependent drift of the spectrum (it's stationary);
- precision (fp32 = bf16);
- rounding near ±1;
- the inexactness of the memory head's reflections;
- forgetting (it's needed);
- the hybrid's attention.

The match between memory-head retention and the failure point (note 02) holds across models, but it's correlational. The remaining candidates are interference between values written along overlapping keys (consistent with GDN±'s horizon shrinking with the number of variables), or the long-range state and readout living somewhere other than the memory head. A head ablation (zero the memory head's output) would separate those.

**Caveat on retention under a snap.** After snapping, the retention probe isn't comparable to baseline: many tokens tie at β = 2, which changes which token counts as a block's "strongest write". Compare snapped runs by accuracy, not retention.
