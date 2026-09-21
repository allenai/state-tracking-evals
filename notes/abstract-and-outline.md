# Abstract revision + paper outline (2026-08-31)

## Phrasing problems in the current abstract draft

1. **"Crohn-Rhodes" → "Krohn–Rhodes"** (spelling; also "expresses" → "express").
2. **"a second, independent measure of complexity: their periodicity."** Periodicity *is* an
   algebraic property of the monoid, so it can't be independent of "the algebraic properties of
   the monoid." What prior work actually keyed on is the *group-theoretic* axis — whether the
   monoid contains non-solvable groups (S5/A5 vs. solvable). The axis you're adding is the
   *group-free* one: aperiodic vs. periodic (does the monoid contain any nontrivial group at
   all), plus the R-trivial refinement below aperiodic. Krohn–Rhodes is what justifies treating
   these as the two coordinates: every finite monoid decomposes into a cascade of groups and
   flip-flops, so "which groups appear" and "how much aperiodic (flip-flop) structure appears"
   are the two natural complexity dials.
3. **"generalized finite-length state-tracking monoid"** — garbled. The formal object is a
   *state-tracking problem* (a word problem), generalized from a single finite monoid to
   assignment traces over an unbounded set of variables (equivalently, an information-lattice /
   register formulation). It's this unbounded-variable generalization — not the Krohn–Rhodes
   decomposition — that unifies state tracking with associative recall: with unboundedly many
   variables and value-only assignments, the problem *is* recall (find the last write); adding
   assignment-by-reference is what buys periodicity. Attribute the unification to the right
   mechanism.
4. **"derived group structure"** — "derived" collides with derived subgroup/commutator
   subgroup; say "the group complexity of its Krohn–Rhodes decomposition" or "which groups
   divide the monoid."
5. **"pretrained language models struggle on aperiodic and periodic tasks"** — the data says
   two different things: accuracy on aperiodic *degrades with context length* (all
   architectures, all scales), while on periodic every pretrained model is *at chance at every
   scale (275M–7B) and context length*. "Struggle" flattens the paper's sharpest empirical
   claim. Also worth one clause: R-trivial is essentially solved (i.e., the failure is
   specifically the complexity axis, not the format).
6. **Direct-training claim is missing.** The DT results are the empirical validation of the
   expressivity theorems (recurrent models length-generalize exactly on the classes they can
   express; positive-eigenvalue GDN fits periodic in-distribution but fails to length-
   generalize, which is precisely the finite-length-vs-asymptotic signature). The abstract
   currently only says "we show how architectures fare."
7. **"connecting the periodicity of periodic monoids to the notion of sensitivity"** —
   redundant, and the logical direction matters: aperiodic word problems are in AC0, AC0
   functions have bounded average sensitivity, so *super-constant sensitivity certifies that a
   task is not aperiodic*. It's a one-way, data-driven lower-bound diagnostic. Say that, and
   then the CruxEval punchline: executed real code has high sensitivity, so real code
   understanding sits in the regime where all current pretrained models fail.
8. **Benchmark description undersells the design.** The matched-triplet construction (each
   periodic source trace deterministically re-rendered as aperiodic and R-trivial versions with
   identical length, query, format, and padding) is what makes the complexity comparisons
   causal rather than correlational. One clause in the abstract earns its keep.

## Revised abstract

> Tracking the state of the world across a long context — the values of variables in code, the
> properties of entities in a document — is a core capability of language models, and one where
> architectures are known to differ sharply. Prior work has characterized the difficulty of
> state tracking through the group-theoretic structure of an underlying finite monoid,
> exemplified by the word problem for $S_5$. We extend this analysis along a second algebraic
> axis exposed by the Krohn–Rhodes decomposition: whether the monoid contains any nontrivial
> group at all (periodicity), and, below that, whether writes are irrevocable (R-triviality).
> We formalize a family of state-tracking problems over unboundedly many variables that unifies
> monoid word problems with associative recall — recall is exactly value-only assignment, and
> assignment by reference is exactly what introduces periodicity — and we prove expressivity
> results locating transformers and (gated) linear RNN variants within the resulting hierarchy:
> in particular, finite-precision gated linear RNNs express periodic problems only when their
> state-transition eigenvalues may be negative. To test these predictions we introduce
> StateBench, a benchmark that renders each underlying assignment trace into matched R-trivial,
> aperiodic, and periodic variants across many surface forms, at context lengths up to $10^6$
> tokens. Directly trained models validate the theory: recurrent architectures length-
> generalize precisely on the classes they can express, while eigenvalue-restricted variants
> fit periodic data in-distribution but fail to generalize. Pretrained models (275M–7B,
> transformer, hybrid, and linear-RNN) solve R-trivial tracking, degrade with context length on
> aperiodic tracking, and perform at chance on periodic tracking at every scale. Finally, since
> aperiodic problems lie in $\mathrm{AC}^0$ and thus have bounded average sensitivity, super-
> constant sensitivity certifies that a task requires periodic state tracking; measuring
> sensitivity over execution traces of real programs, we find that most code sits in this
> regime — squarely where today's models fail.

(~230 words; trim candidates if a venue wants ~150: the R-triviality clause in sentence 3, the
"recall is exactly…" appositive, and the parenthetical model list.)

Hook options for sentence 1 (pick one register):
- Capability framing (used above): state tracking as the thing long contexts are *for*.
- Failure framing: "Language models now read million-token contexts, but reading is not
  remembering: we show that no current pretrained model can track state that is ever
  overwritten by reference."
- Theory framing: "Which state-tracking problems an architecture can express is determined by
  the algebra of the problem, not its surface form."

## Detailed paper outline

### 1. Introduction
- Hook: state tracking as the core semantic demand of long contexts (code execution,
  discourse/entity state).
- Prior work axis: solvable vs. non-solvable groups ($S_5$/$A_5$; Liu et al. 2023, Merrill &
  Sabharwal 2024). Why this axis alone is a poor model of *natural* state tracking: too hard
  (non-solvability may be adversarial for natural text) and too easy (fixed alphabet, no
  entity/variable introduction, no recall component).
- Our two moves: (a) the periodicity axis + unbounded variables → unifies state tracking with
  associative recall; (b) a matched benchmark + sensitivity as the bridge to real data.
- Contributions list (theory / benchmark / DT experiments / PT experiments / sensitivity-in-code).

### 2. A Generalized Model of State Tracking
- 2.1 Word problems and the complexity ladder: R-trivial ⊂ aperiodic ⊂ periodic-solvable ⊂
  non-solvable; Krohn–Rhodes as the justification (groups + flip-flops are the two prime
  factors). Table mapping algebra ↔ logic (LTL[past] / LTL / regular) ↔ circuits (AC0 / TC0 /
  NC1 / L).
- 2.2 The variable-tracking formulation: assignment traces over registers; value assignment
  (aperiodic), first-write-wins (R-trivial), assignment by reference (periodic). State the
  equivalence: with unboundedly many variables, value-only tracking = associative recall.
  (Keep the information-lattice generality for the appendix; the assignment formulation is
  what StateBench instantiates.)
- 2.3 Unbounded-variable complexity results: aperiodic case stays in AC0 (per-variable
  circuits in parallel); periodic case with growing $S_m$ — the NC1 / PNC1 / L-complete
  regimes (JP's bullet list from the June notes).

### 3. Expressivity of Finite-Precision Architectures
- Known: SMAT/UHAT = LTL[past]; GDN with negative eigenvalues = all regular languages.
- New: fixed-precision GDN with positive eigenvalues = exactly LTL (aperiodic). Corollary
  table for hybrids.
- Predictions to carry into experiments: (P1) transformers fail all classes at length; (P2)
  positive-eigenvalue GDN handles R-trivial/aperiodic at any length but not periodic; (P3)
  negative-eigenvalue GDN handles periodic. Emphasize these are *length-generalization*
  predictions — any architecture can fit any finite-length slice.

### 4. StateBench
- Construction: one periodic source trace → derived aperiodic and R-trivial variants with
  identical assignment count, query, formatting, padding (matched triplets → causal
  attribution to complexity class). Surface forms (integer-code, cube-painting, spreadsheet-
  cells, status-lights, people-in-rooms, RULER-style). Long-context sweep (log-spaced up to
  64K assignments, token strata to 1M), padding/distraction axes, dependency-length and
  per-example sensitivity metadata. Multiple-choice scoring with equality-pattern-preserving
  distractors.

### 5. Direct Training Validates the Expressivity Theory
- Setup: 275M {Transformer-RoPE, Transformer-NoPE, Hybrid, GDN-positive, GDN-full} ×
  {r-trivial, aperiodic, periodic (± sensitivity-ordered curricula)}.
- Fig: dt_accuracy_diagonal_by_assignments — GDN generalizes to 50K+ assignments on
  R-trivial/aperiodic; transformers collapse past the training cutoff; on periodic,
  transformers are at chance even in-distribution.
- Fig: dt_accuracy_periodic_by_assignments — GDN-full > Hybrid > GDN-positive in
  length-generalization on periodic; GDN-positive's in-distribution fit + early collapse is
  exactly P2's signature. Curriculum panel (ordering effects).
- Supporting: dependency-length and sensitivity-binned views (accuracy declines with
  sensitivity coverage); regression pulling apart context length vs. assignment count vs.
  dependency length.

### 6. Pretrained Models Fail Periodic State Tracking
- Setup: matched pretrained checkpoints (Transformer/Hybrid/GDN, 275M–7B; 8K pretraining,
  64K long-context extension; external baselines).
- Fig: pt_accuracy_by_context_length_aggregated_max — R-trivial solved; aperiodic degrades
  with context (before the long-context cutoff); periodic at chance at *every* scale and
  architecture, including hybrids and GDN.
- Discussion: scale doesn't buy periodicity (7B = 275M at chance); pretraining does not
  install the mechanism DT shows the architecture can express — a data/objective gap, not
  (only) an architecture gap. Padding-complexity and dependency-length breakouts as
  robustness checks.

### 7. Sensitivity: From Benchmark to Real Data
- Theorem: aperiodic ⇒ AC0 ⇒ bounded average sensitivity; contrapositive gives a data-driven
  certificate of "requires periodic state tracking."
- Method: executing real code linearizes control flow into exactly StateBench's trace object;
  reverse-liveness sensitivity over CruxEval traces.
- Fig: sensitivity_density histogram — median 0.56, large mass at 1.0 ⇒ most real code
  evaluation is in the super-constant-sensitivity regime ⇒ beyond aperiodic ⇒ in the regime
  where §6 shows all pretrained models are at chance. Correlate against published per-problem
  CruxEval solve rates.

### 8. Related Work
State tracking & word problems; linear-RNN expressivity; associative-recall benchmarks
(MQAR/RULER — position ours as the complexity-graded superset); sensitivity in LLM analysis;
dynamic semantics/entity tracking (brief; points to the discussion).

### 9. Discussion & Limitations
Entity tracking / dynamic-semantics reading of the operations (assertion=assignment,
definites=reference, existentials=allocation); finite precision caveats; multiple-choice
format; eager-copy (non-aliasing) reference semantics; what it would take to pretrain-in
periodicity (data with reference chains? negative-eigenvalue hybrids at scale?).
