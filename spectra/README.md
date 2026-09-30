# spectra

Tests whether the long-length decay of GDN± / Hybrid± models on soluble and general-finite StateBench data comes from their transition eigenvalues. See `notes/spectra/00-method.md` for the reasoning and `notes/spectra/*.md` for results.

## Pieces

| module | runs on | does |
|---|---|---|
| `replay.py` | anywhere | parses integer-code prompts; gives register state after any statement, token roles, and value age / hops |
| `capture.py` | GPU job | wraps each GDN kernel: records `g`, `β`, `k`; applies gate interventions; streaming sinks (gate stats, raw windows, write retention, per-statement spectra, residual drift) |
| `run.py` | GPU job | samples eval rows, validates segmented prefill against a one-shot forward, prefills with checkpoints, branches the cache for per-register readout, scores the five eval choices |
| `launch.py` | laptop | gantry launcher; suites `pilot`, `metadata`, `baseline`, `precision`, `interventions`, `training-steps` |
| `fetch.py` | laptop | downloads finished `spectra-*` jobs into `results/spectra/` |
| `dump_doc_metadata.py` | CPU job | fills `results/doc_metadata/` (the dataset is private) |
| `evals.py`, `outputs.py` | laptop | load eval predictions / job outputs |
| `figures.py`, `style.py` | laptop | figures in `figures/spectra/`, styled like `notebooks/dt_results.ipynb` |

## Workflow

```bash
uv run python -m spectra.launch --suite baseline precision interventions --dry-run   # then without --dry-run
uv run python -m spectra.fetch
uv run python -m spectra.figures all
```

The precision reruns of the regular eval harness use `scripts/eval_dt.py --dtype float32 --group jacksonp-state-bench-precision-fp32`, and are fetched with `scripts/fetch_results.py --group jacksonp-state-bench-precision-fp32 --workspace ai2/beyond-state --completed-only --skip-requests`.

## Tests

```bash
uv run --with pytest --with numpy --with torch \
  --with "transformers @ git+https://github.com/jopetty/transformers.git@olmo-3.5-hybrid-state-bench" \
  pytest spectra/tests
```

CPU tests use the fork's torch fallback kernels. The GPU job's startup check covers segmented-vs-one-shot equivalence with fla's kernels.
