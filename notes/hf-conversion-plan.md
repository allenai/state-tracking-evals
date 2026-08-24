# Plan: convert StateBench olmo-core checkpoints to HuggingFace format

Goal: convert every trained 275M StateBench checkpoint under
`/weka/oe-training-default/ai2-llm/model-ladders/state-bench/` to HF format for
olmo-eval, writing each converted checkpoint to a sister dir of the original
step dir (suffix convention: `stepN-hf`, matching hybrid-small-suite), with
automatic discovery and skip-if-already-converted.

## Existing machinery (olmo-core, branch jacksonp/state-bench-dt)

- `convert_checkpoint_to_hf(...)` — src/olmo_core/nn/hf/convert_checkpoint.py:46.
  One call: builds model on meta, loads sharded DCP directly (no unshard step),
  converts, saves, optionally validates logit parity. Reads model config from
  the `config.json` that ConfigSaverCallback writes inside every step dir.
- Hybrid support — src/olmo_core/nn/hf/config.py: `is_olmo_hybrid_model` (:256),
  `get_hybrid_hf_config` (:319) emits `model_type: olmo3_5_hybrid`
  (`Olmo3_5HybridForCausalLM`) with `layer_types`, GDN dims, `embed_scale`,
  `use_peri_norm`, `use_head_qk_norm`, `use_attention_gate`. State mapping:
  src/olmo_core/nn/hf/convert.py:874 `convert_hybrid_state_to_hf`.
- Precedent batch driver: src/scripts/train/hybrid-small-suite/convert_checkpoints.py
  (`stepN-hf` sister dirs); run_evals.py launches olmo-eval against `-hf` dirs
  (needs `transformers @ git+...@olmo-3.5-hybrid` fork; vLLM opts:
  enforce_eager=true, add_bos_token=false, prompt_logprobs=1).
- Step dir layout: `stepN/{config.json, data_paths.txt, model_and_optim/, train/}`;
  `dir_is_checkpoint` = `model_and_optim/.metadata` exists
  (src/olmo_core/train/checkpoint.py:304).

## Gaps to close

1. **Ladder config shape**: example CLI
   (src/examples/huggingface/convert_checkpoint_to_hf.py:106) asserts
   `config["dataset"]["tokenizer"]`; ladder configs store it under
   `config["data_loader"]["tokenizer"]`. → batch script calls the library
   function directly and passes the tokenizer explicitly.
2. **Pure-GDN (gdn-sdp, gdn-full)**: `get_hybrid_hf_config` raises
   "Hybrid model must have at least one attention layer" (config.py:346).
   → relax to allow all-`linear_attention` layer_types; verify the HF fork's
   modeling code tolerates it.
3. **Transformer variants (rope/nope) can't use the dense path**: dense route
   emits Olmo2/Olmo3Config, which cannot express gated attention, head QK-norm,
   peri-norm, embed_scale — and transformer-nope would silently get RoPE.
   → route ALL five model types through the hybrid converter (all-
   `full_attention` layer_types for the transformers); add a force-hybrid knob
   (or extend `is_olmo_hybrid_model` to trigger on gated/peri-norm attention).
   **Risk to verify first**: does `olmo3_5_hybrid` handle (a) all-attention and
   all-linear layer_types, (b) RoPE'd full-attention layers? (hybrid-small-suite
   only exercised NoPE attention.) Harness: existing
   src/test/examples/huggingface/convert_checkpoint_to_hf_hybrid_test.py and
   src/scripts/train/hybrid-small-suite/compare_logits_*.py.

## Implementation

A. **olmo-core library**: relax >=1-attention check; audit hybrid config for
   all-linear / all-attention / rope'd-attention cases; add force-hybrid knob;
   parametrize the hybrid conversion test over all five state-bench block
   configs with round-trip logit parity.

B. **Batch script** `src/scripts/train/ladder/convert_state_bench_checkpoints.py`:
   - Arg: checkpoint root (default the state-bench ladder dir). rglob for step
     dirs via `dir_is_checkpoint` → discovers every
     `275M/<model_type>/<distribution>/Cx*/init_seed*/stepN` automatically.
   - Target: sister `stepN-hf/`; skip if it already contains
     `model.safetensors` + `config.json` (incomplete conversions get redone).
   - Calls `convert_checkpoint_to_hf` with tokenizer from
     `data_loader.tokenizer`, dtype bfloat16, validation on CUDA (GDN
     validation needs GPU kernels — run on a weka-mounted GPU session).
   - Flags: `--dry-run`, `--final-only` (default converts every permanent step
     dir — intermediate checkpoints matter for emergence analysis),
     `--skip-validation`.

C. **Verification order**: unit tests → one real checkpoint per family
   (transformer, hybrid, pure-GDN) with validation on →
   compare_logits_vllm.py spot check → full sweep.

Sequencing: the 9 finished transformer runs need the riskiest new path
(forced-hybrid for gated/NoPE attention) — start gap-3 verification first,
before the GDN reruns finish.

Open items:
- Confirm suffix `-hf` (established convention) over the requested prefix.
- olmo-eval side needs the transformers fork installed (see run_evals.py:98).

## Status updates (2026-08-24)

- Implemented: requires_hybrid_hf_format routing, homogeneous-hybrid config support,
  batch converter (olmo-core jacksonp/state-bench-dt).
- The transformers fork's guards rejected homogeneous layer_types, and rope
  standardization silently dropped RoPE for full-attention layers. Fixed on
  `jopetty/transformers@olmo-3.5-hybrid-state-bench` (forked from
  yashassamaga/transformers@olmo-3.5-hybrid); install with
  `pip install 'transformers @ git+https://github.com/jopetty/transformers.git@olmo-3.5-hybrid-state-bench'`.
  Candidate PR back to yashassamaga when settled.
- Known environment issue: fla 0.5.2 fails to import next to the fork's
  transformers 5.0.0.dev0 (fla.layers/fla.models import transformers internals);
  workaround is patching fla's _import_optional_public_module to swallow the
  ImportError so fla.ops/fla.modules stay usable. Affects any env pairing the
  fork with current fla, including Yashas's original convert_checkpoints.py flow.
- Outstanding: GDN-variant logit-parity validation (blocked on the fla import fix);
  check whether vLLM ships its own olmo3_5_hybrid impl with the same homogeneity
  assumptions before launching the olmo-eval matrix.
