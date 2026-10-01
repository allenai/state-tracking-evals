# Interventions: what changes long-length accuracy

Figure: `figures/spectra/interventions.pdf`. Gates are rewritten inside the GDN kernel call during a normal forward pass. The metric is per-register readout accuracy at checkpoints within each long prompt (16 prompts per run, bf16 unless noted).

Readout accuracy by assignments read so far:

| GDN±, general finite | ≤ 256 | 1K–4K | 4K–16K | > 16K |
|---|---|---|---|---|
| none | 0.97 | 0.79 | 0.40 | 0.36 |
| snap β→2 where β > 1.98 | 0.97 | 0.81 | 0.40 | 0.37 |
| snap α→1 where α > 0.999 | 0.97 | 0.78 | 0.39 | 0.36 |
| both snaps | 0.97 | 0.79 | 0.38 | 0.36 |
| α → α^0.5 (half the forgetting; corrected) | 0.97 | 0.77 | 0.38 | 0.35 |
| float32 | 0.97 | 0.73 | 0.37 | 0.35 |

| Hybrid±, general finite | ≤ 256 | 1K–4K | 4K–16K | > 16K |
|---|---|---|---|---|
| none | 0.98 | 0.67 | 0.39 | 0.35 |
| snap β→2 where β > 1.98 | 0.98 | 0.66 | 0.39 | 0.35 |
| snap α→1 where α > 0.999 | 0.98 | 0.62 | 0.41 | 0.35 |
| α → α^0.5 (corrected) | 0.98 | 0.65 | 0.40 | 0.35 |
| attention window 4K tokens | 0.98 | 0.65 | 0.40 | 0.35 |
| float32 | 0.98 | 0.63 | 0.40 | 0.35 |

Soluble shows the same pattern. For example, GDN± at > 16K assignments: none 0.85, snaps 0.74–0.87, α^0.5 0.78.

**Takeaways.**
- **Making near-exact gates exact does nothing.** Snapping β > 1.98 to exactly 2, or α > 0.999 to exactly 1, moves accuracy by at most ±0.03 (and ±0.11 at > 16K on soluble). Together with fp32 matching bf16, this rules out rounding near ±1 as the bottleneck.
- **Slowing forgetting doesn't help either.** α → α^0.5 doubles every head's memory timescale and changes accuracy by at most −0.02, near the training length and beyond, on both datasets. An earlier version of this note claimed halving forgetting was destructive. That came from a bug (kernel wrappers stacked per sample, so the k-th sample got α^(0.5^k)); the numbers here are the corrected rerun.
- **The hybrid's attention isn't the long-range bottleneck.** Restricting both NoPE attention layers to the last 4K tokens leaves accuracy unchanged at every length. That rules out "attention averages over an ever-longer context" as the reason Hybrid± fails at a fixed ~1.4–2.9K assignments.
- **Making the memory head's reflections exact doesn't help either.** The retention analysis pointed at GDN±'s memory head (L1 H6), whose value-token reflections sit at β ≈ 1.89 (|μ| ≈ 0.89, never exact). Snapping β > 1.75 → 2, in that head alone or in all heads, makes 100% of those reflections exact (the leak per reflection goes from 0.114 to 0.001) and leaves accuracy unchanged at every length:

| GDN±, general finite | ≤ 256 | 1K–4K | 4K–16K | > 16K |
|---|---|---|---|---|
| none | 0.97 | 0.79 | 0.40 | 0.36 |
| snap β > 1.75 → 2, memory head | 0.97 | 0.80 | 0.42 | 0.35 |
| snap β > 1.75 → 2, all heads | 0.97 | 0.80 | 0.41 | 0.35 |

  Hybrid± is unchanged as well. Its memory head (L5 H1) doesn't reflect on value tokens, so the snap barely touches it.

**Ablations** (`ablations_periodic.pdf`). Each zeroes a GDN head's kernel output; the head's state still updates.

| readout accuracy | ≤ 256 | 256–1K | 1K–4K | 4K–16K | > 16K |
|---|---|---|---|---|---|
| GDN±: none | 0.97 | 1.00 | 0.79 | 0.40 | 0.36 |
| GDN±: zero memory head L1 H6 | **0.14** | **0.17** | 0.16 | 0.16 | 0.14 |
| GDN±: zero top-4 reflection heads | 0.97 | 1.00 | 0.76 | 0.41 | 0.36 |
| GDN±: zero control head L8 H2 | 0.97 | 1.00 | 0.79 | 0.40 | 0.36 |
| Hybrid±: none | 0.98 | 0.98 | 0.67 | 0.39 | 0.35 |
| Hybrid±: zero memory head L5 H1 | 0.98 | 0.98 | 0.66 | 0.40 | 0.35 |
| Hybrid±: zero top-4 reflection heads | 0.98 | 0.98 | 0.63 | 0.41 | 0.35 |
| Hybrid±: zero both attention layers | 0.98 | 0.99 | 0.62 | 0.39 | 0.35 |

- **GDN± carries its state in one head.** Zeroing L1 H6 (the head the retention analysis identified) collapses readout to near chance even in-distribution. Zeroing four heavily reflecting heads, or a control head, does nothing.
- **Hybrid± carries it redundantly.** No single head, set of reflection heads, or even both attention layers is necessary. That is consistent with its memory-head retention not tracking accuracy across checkpoints (note 05).

**What this means for the eigenvalue hypothesis.** For GDN± the state flows through one head, whose write retention tracks extrapolation across models and checkpoints. Yet none of the following affects the length horizon:
- position-dependent drift of the spectrum (it's stationary, per role, to three decimals);
- precision (fp32 = bf16);
- rounding near ±1;
- making that head's reflections exact;
- doubling its memory timescale;
- the hybrid's attention.

What remains is interference inside the memory head: many values carried through one 128-dimensional key space for thousands of steps. That fits the horizon shrinking with the number of variables (note 01). Testing it directly (e.g. key overlap between registers in L1 H6, or state rank over time) is the natural next step.

**Caveat on retention under a snap.** After snapping, the retention probe isn't comparable to baseline: many tokens tie at β = 2, which changes which token counts as a block's "strongest write". Compare snapped runs by accuracy, not retention.
