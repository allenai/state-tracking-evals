"""Beaker job entry point: capture gate spectra and per-position readouts.

For each sampled StateBench prompt this prefills the model in segments that
end on statement boundaries, recording the GDN gates through
:class:`spectra.capture.GateRecorder`. At log-spaced checkpoints it snapshots
the recurrent state (and attention KV length), teacher-forces the gold answer
for the state *at that checkpoint*, and scores every register's value token --
giving accuracy as a function of position within one long sequence, aligned
with the gate statistics at the same positions. After the last statement it
also scores the five eval choices, reproducing the eval's own metric.

Usage (inside the job; see spectra/launch.py)::

    python -m spectra.run --model-type gdn-full --training periodic \\
        --eval-class periodic --dtype bfloat16 --out /results/spectra
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from spectra import capture, replay

log = logging.getLogger("spectra")

CHECKPOINT_BASE = Path("/weka/oe-training-default/ai2-llm/model-ladders/state-bench/275M")
# Mirrors scripts/eval_dt.py (DATASET_DIRS, CHECKPOINT_STEPS): final WSDS steps.
TRAINING_DIRS = {
    "r-trivial": "integer-code--r-trivial",
    "aperiodic": "integer-code--aperiodic",
    "periodic": "integer-code--periodic",
    "periodic-sens-curriculum": "integer-code--periodic--sens-curriculum",
    "solvable": "integer-code-modular--solvable-v2",
}
CHECKPOINT_STEPS = {
    "hybrid": (3627, 4030, 7656, 8059),
    "hybrid-sdp": (3627, 4030, 7656, 8059),
    "gdn-sdp": (3695, 4106, 7800, 8211),
    "gdn-full": (3695, 4106, 7800, 8211),
}
MODULAR_TRAINING = {"solvable"}
DATASET_REPO = "allenai/state-bench"
TOKENIZER = "allenai/Olmo-3-1025-7B"


def checkpoint_path(model_type: str, training: str, step: int | None, seed: int = 0) -> Path:
    step = step if step is not None else CHECKPOINT_STEPS[model_type][-1]
    return CHECKPOINT_BASE / model_type / TRAINING_DIRS[training] / "Cx1" / f"init_seed{seed}" / f"step{step}-hf"


def eval_subset(training: str, eval_class: str) -> str:
    formatter = "integer-code-modular" if training in MODULAR_TRAINING else "integer-code"
    return f"{formatter}--{eval_class}"


# --------------------------------------------------------------------------------------
# Sampling


def sample_rows(subset: str, tiers: list[str], per_tier: int, num_variables: list[int] | None,
                max_filler: float, seed: int) -> list[dict]:
    """Deterministic, variable-count-stratified sample of eval rows per tier."""
    from datasets import load_dataset

    rows: list[dict] = []
    for tier in tiers:
        split = load_dataset(DATASET_REPO, subset, split=f"test__{tier}")
        keep = [
            row for row in split
            if float(row.get("filler_proportion") or 0.0) <= max_filler
            and (num_variables is None or int(row["num_variables"]) in num_variables)
        ]
        strata: dict[int, list[dict]] = {}
        for row in keep:
            strata.setdefault(int(row["num_variables"]), []).append(row)
        chosen: list[dict] = []
        rank = lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()  # noqa: E731
        ordered = {m: sorted(group, key=rank) for m, group in sorted(strata.items())}
        # Round-robin over variable counts so each tier covers every stratum.
        while len(chosen) < per_tier and any(ordered.values()):
            for m in list(ordered):
                if ordered[m] and len(chosen) < per_tier:
                    chosen.append(ordered[m].pop(0))
        log.info("tier %s: %d rows kept of %d, sampled %d", tier, len(keep), len(split), len(chosen))
        rows.extend({**row, "tier": tier} for row in chosen)
    return rows


# --------------------------------------------------------------------------------------
# Model and cache plumbing


def load_model(path: Path, dtype: str, device: torch.device):
    from transformers import AutoModelForCausalLM

    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch_dtype, attn_implementation="sdpa")
    return model.to(device).eval()


def new_cache(model):
    module = __import__(type(model).__module__, fromlist=["Olmo3_5HybridDynamicCache"])
    return module.Olmo3_5HybridDynamicCache(config=model.config)


def snapshot(cache) -> dict:
    clone = lambda xs: [x.clone() if x is not None else None for x in xs]  # noqa: E731
    return {
        "recurrent": clone(cache.recurrent_states),
        "conv_q": clone(cache.conv_states_q),
        "conv_k": clone(cache.conv_states_k),
        "conv_v": clone(cache.conv_states_v),
        "kv_len": [k.shape[2] if k is not None else None for k in cache.key_cache],
    }


def restore(cache, saved: dict) -> None:
    cache.recurrent_states = list(saved["recurrent"])
    cache.conv_states_q = list(saved["conv_q"])
    cache.conv_states_k = list(saved["conv_k"])
    cache.conv_states_v = list(saved["conv_v"])
    for index, length in enumerate(saved["kv_len"]):
        if length is not None:
            cache.key_cache[index] = cache.key_cache[index][:, :, :length]
            cache.value_cache[index] = cache.value_cache[index][:, :, :length]


def truncate_kv(cache, window: int) -> None:
    for index, keys in enumerate(cache.key_cache):
        if keys is not None and keys.shape[2] > window:
            cache.key_cache[index] = keys[:, :, -window:].contiguous()
            cache.value_cache[index] = cache.value_cache[index][:, :, -window:].contiguous()


def prefill(model, cache, ids: torch.Tensor) -> None:
    model.model(input_ids=ids[None], past_key_values=cache, use_cache=True)


def continuation_logits(model, cache, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids[None], past_key_values=cache, use_cache=True).logits[0].float()


def segment_plan(layout: replay.TokenLayout, checkpoints: list[int], max_segment: int) -> list[int]:
    """Prefill boundaries (token counts): every checkpoint's end, the prompt's end,
    and enough statement ends in between that no segment exceeds ``max_segment``."""
    ends = [int(e) for e in layout.statement_end_token]
    required = {ends[c] for c in checkpoints} | {ends[-1]}
    boundaries: list[int] = []
    last = previous = 0
    for end in ends:
        if end - last > max_segment and previous > last:
            boundaries.append(previous)
            last = previous
        if end in required:
            boundaries.append(end)
            last = end
        previous = end
    return boundaries


def validate_segmentation(model, device: torch.device, ids: torch.Tensor, tolerance: float) -> dict:
    """Segmented prefill + cached continuation must match one full forward pass."""
    ids = ids.to(device)
    n = ids.shape[0]
    tail = 40
    with torch.inference_mode():
        full = model(input_ids=ids[None]).logits[0, -tail:].float()
        cache = new_cache(model)
        # A long chunked segment, a single-token (decode-path) segment, and another
        # chunked one, then the cached continuation.
        body = n - tail
        cut = sorted({0, body // 3, body // 3 + 1, body})
        for a, b in zip(cut, cut[1:]):
            prefill(model, cache, ids[a:b])
        segmented = continuation_logits(model, cache, ids[-tail:])
    diff = (full.log_softmax(-1) - segmented.log_softmax(-1)).abs().max().item()
    result = {"max_abs_logprob_diff": diff, "tolerance": tolerance, "ok": diff <= tolerance}
    if not result["ok"]:
        raise RuntimeError(f"segmented prefill diverges from a full forward pass: {result}")
    return result


# --------------------------------------------------------------------------------------
# Readout


class Readout:
    """Teacher-forced per-register readout of the model's belief at a checkpoint."""

    def __init__(self, tokenizer, num_values: int, modular: bool):
        labels = [str(i) for i in (range(num_values) if modular else range(1, num_values + 1))]
        self.labels = labels
        ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(i) != 1 for i in ids):
            raise ValueError("value labels are not single tokens; readout needs multi-token scoring")
        self.label_ids = torch.tensor([i[0] for i in ids])
        self.label_index = {label: index for index, label in enumerate(labels)}
        self.tokenizer = tokenizer

    def encode(self, text: str):
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return encoded["input_ids"], encoded["offset_mapping"]

    def score(self, model, cache, state: dict[str, replay.RegisterState], registers: list[str],
              device: torch.device) -> list[dict]:
        text = " " + replay.render_answer(state, registers)
        ids, offsets = self.encode(text)
        logits = continuation_logits(model, cache, torch.tensor(ids, device=device))
        logprobs = logits.log_softmax(-1)
        label_ids = self.label_ids.to(device)
        present = sorted({state[name].value for name in registers})
        present_index = torch.tensor([self.label_index[v] for v in present], device=device)
        rows, cursor = [], 0
        starts = {start: index for index, (start, _) in enumerate(offsets)}
        for name in registers:
            fragment = f"{name} ?= {state[name].value};"
            at = text.index(fragment, cursor)
            value_start = at + len(f"{name} ?= ")
            cursor = at + len(fragment)
            token = starts.get(value_start)
            if token is None or token == 0:
                raise ValueError(f"value token for {name} not aligned in {text!r}")
            row_logprobs = logprobs[token - 1]
            over_labels = row_logprobs[label_ids]
            true = self.label_index[state[name].value]
            rank = int((over_labels > over_labels[true]).sum().item())
            best_present = present_index[over_labels[present_index].argmax()].item()
            rows.append({
                "register": name,
                "value": state[name].value,
                "logprob": float(row_logprobs[label_ids[true]].item()),
                "label_mass": float(over_labels.exp().sum().item()),
                "rank": rank,
                "correct_all_values": rank == 0,
                "correct_present_values": best_present == true,
                "n_present_values": len(present),
                "origin": state[name].origin,
                "hops": state[name].hops,
                "last_write": state[name].last_write,
            })
        return rows


def score_choices(model, cache, tokenizer, choices: list[str], device: torch.device) -> list[float]:
    saved = snapshot(cache)
    scores = []
    for choice in choices:
        ids = tokenizer.encode(" " + choice, add_special_tokens=False)
        # The first choice token is predicted from the last prompt position, which the
        # cache no longer exposes; every choice shares it (" a"), so it cancels in argmax.
        logits = continuation_logits(model, cache, torch.tensor(ids, device=device))
        logprobs = logits.log_softmax(-1)[:-1]
        scores.append(float(logprobs.gather(1, torch.tensor(ids[1:], device=device)[:, None]).sum()))
        restore(cache, saved)
    return scores


# --------------------------------------------------------------------------------------
# One sample


def run_sample(model, tokenizer, row: dict, args, device: torch.device, readout: Readout,
               intervention: capture.Intervention, out_dir: Path) -> dict:
    prefix = row["prefix"]
    program = replay.parse_program(prefix)
    encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor(encoded["input_ids"], device=device)
    layout = replay.token_layout(program, np.array(encoded["offset_mapping"]))
    registers = program.registers
    checkpoints = replay.log_spaced_checkpoints(program, args.checkpoints)
    checkpoint_states = dict(zip(checkpoints, replay.states_at(program, checkpoints)))
    n_tokens = ids.shape[0]
    n_heads = model.config.linear_num_value_heads
    head_dim = model.config.linear_key_head_dim

    position_edges = capture.log2_position_edges(n_tokens)
    hist_edges = position_edges[::2] if len(position_edges) > 6 else position_edges
    if hist_edges[-1] != position_edges[-1]:
        hist_edges = np.append(hist_edges, position_edges[-1])
    recorder = capture.GateRecorder(model, intervention)
    stats = capture.BinnedGateStats(layout.roles, position_edges=position_edges, hist_position_edges=hist_edges,
                                    n_heads=n_heads, device=device)
    window_starts = sorted({min(s, max(n_tokens - args.window, 0)) for s in (0, 1024, 4096, 16384, 65536, 200_000)
                            if s < n_tokens})
    windows = capture.RawWindows([(s, min(s + args.window, n_tokens)) for s in window_starts])
    recorder.sinks = [stats, windows]
    retention = None
    if args.retention:
        retention = capture.ImpulseRetention(n_heads=n_heads, head_dim=head_dim, device=device,
                                             position_edges=position_edges)
        recorder.sinks.append(retention)
    spectra_sink = None
    if args.statement_spectra:
        assignments = program.assignment_indices
        picks = np.unique(np.geomspace(1, len(assignments), num=args.statement_spectra).round().astype(int) - 1)
        chosen = assignments[picks]
        ends = layout.statement_end_token
        ranges = {int(s): (int(ends[s - 1]) if s > 0 else int(np.argmax(layout.statement >= 0)), int(ends[s]))
                  for s in chosen}
        spectra_sink = capture.StatementSpectra(ranges)
        recorder.sinks.append(spectra_sink)
    drift = capture.HiddenDrift(layout.roles, position_edges, device)
    drift.attach(model)

    max_segment = args.max_segment_attention if capture.attention_layers(model) else args.max_segment
    boundaries = segment_plan(layout, checkpoints, max_segment)
    checkpoint_at_token = {int(layout.statement_end_token[c]): c for c in checkpoints}

    readout_rows: list[dict] = []
    cache = new_cache(model)
    started = time.time()
    last = 0
    with torch.inference_mode():
        for boundary in boundaries:
            recorder.recording = drift.recording = True
            recorder.offset = drift.offset = last
            prefill(model, cache, ids[last:boundary])
            recorder.recording = drift.recording = False
            if intervention.attn_window is not None:
                truncate_kv(cache, intervention.attn_window)
            last = boundary
            statement = checkpoint_at_token.get(boundary)
            if statement is None:
                continue
            saved = snapshot(cache)
            rows = readout.score(model, cache, checkpoint_states[statement], registers, device)
            restore(cache, saved)
            assignment_rank = int(np.searchsorted(program.assignment_indices, statement)) + 1
            for r in rows:
                r.update({"statement": statement, "assignments_so_far": assignment_rank, "token": boundary})
            readout_rows.extend(rows)
        choice_scores = score_choices(model, cache, tokenizer, list(row["choices"]), device)
    drift.detach()
    elapsed = time.time() - started

    sample_id = row["id"]
    sample_dir = out_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_dir / "gates.npz", **stats.result())
    np.savez_compressed(sample_dir / "windows.npz", **windows.result())
    np.savez_compressed(sample_dir / "drift.npz", **drift.result())
    if retention is not None:
        np.savez_compressed(sample_dir / "retention.npz", **retention.result())
    if spectra_sink is not None:
        statement_meta = {
            "statement_kind": np.array([program.statements[s].kind for s in spectra_sink.token_ranges]),
            "statement_delta": np.array([
                int(program.statements[s].argument) if program.statements[s].kind == "increment" else -1
                for s in spectra_sink.token_ranges
            ]),
            "statement_index": np.array(list(spectra_sink.token_ranges)),
        }
        np.savez_compressed(sample_dir / "statement_spectra.npz", **spectra_sink.result(), **statement_meta)
    np.savez_compressed(sample_dir / "layout.npz", roles=layout.roles, statement=layout.statement,
                        statement_end_token=layout.statement_end_token)
    with (sample_dir / "readout.jsonl").open("w") as f:
        for r in readout_rows:
            f.write(json.dumps(r) + "\n")
    predicted = int(np.argmax(choice_scores))
    meta = {
        "id": sample_id,
        "tier": row["tier"],
        "complexity": row.get("complexity"),
        "num_variables": int(row["num_variables"]),
        "num_assignments": int(row["num_assignments"]),
        "num_values": int(row.get("num_values") or 100),
        "max_dependency_length": row.get("max_dependency_length"),
        "filler_proportion": row.get("filler_proportion"),
        "n_tokens": n_tokens,
        "registers": registers,
        "checkpoints": checkpoints,
        "choice_scores": choice_scores,
        "gold_choice": int(row["correct_choice_index"]),
        "predicted_choice": predicted,
        "choice_correct": predicted == int(row["correct_choice_index"]),
        "seconds": elapsed,
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("sample %s (%s, m=%d, %d tokens): choice_correct=%s, final readout acc=%.2f, %.0fs",
             sample_id, row["tier"], meta["num_variables"], n_tokens, meta["choice_correct"],
             np.mean([r["correct_present_values"] for r in readout_rows if r["statement"] == checkpoints[-1]]),
             elapsed)
    return meta


# --------------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", required=True, choices=sorted(CHECKPOINT_STEPS))
    parser.add_argument("--training", required=True, choices=sorted(TRAINING_DIRS))
    parser.add_argument("--eval-class", required=True, choices=["r-trivial", "aperiodic", "solvable", "periodic"])
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: final)")
    parser.add_argument("--model-path", type=Path, default=None, help="Override the checkpoint path")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--tiers", nargs="+", default=["tokens_under_2k", "tokens_256k"])
    parser.add_argument("--per-tier", type=int, default=8)
    parser.add_argument("--num-variables", nargs="+", type=int, default=None)
    parser.add_argument("--max-filler", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoints", type=int, default=24, help="Readout checkpoints per sample")
    parser.add_argument("--window", type=int, default=1024, help="Raw gate window length (tokens)")
    parser.add_argument("--max-segment", type=int, default=8192)
    parser.add_argument("--max-segment-attention", type=int, default=2048)
    parser.add_argument("--no-retention", dest="retention", action="store_false")
    parser.add_argument("--statement-spectra", type=int, default=64, help="Statements per sample (0 = off)")
    parser.add_argument("--intervention", type=json.loads, default={},
                        help='JSON, e.g. \'{"snap_reflect": 0.05}\' (see capture.Intervention)')
    parser.add_argument("--validation-tolerance", type=float, default=5e-2)
    parser.add_argument("--out", type=Path, default=Path("/results/spectra"))
    parser.add_argument("--name", default=None, help="Run directory name (default: derived)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = args.model_path or checkpoint_path(args.model_type, args.training, args.step)
    intervention = capture.Intervention(**args.intervention)
    if intervention.heads is not None:
        intervention.heads = [tuple(h) for h in intervention.heads]
    name = args.name or "__".join([
        args.model_type, args.training, args.eval_class, args.dtype, path.name, intervention.name(),
    ]).replace("=", "-").replace(",", "_")
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = load_model(path, args.dtype, device)
    log.info("loaded %s (%s); layer types %s", path, args.dtype, model.config.layer_types)
    subset = eval_subset(args.training, args.eval_class)
    rows = sample_rows(subset, args.tiers, args.per_tier, args.num_variables, args.max_filler, args.seed)
    if not rows:
        raise RuntimeError(f"no rows sampled from {subset} {args.tiers}")

    probe = min(rows, key=lambda r: len(r["prefix"]))
    probe_ids = torch.tensor(tokenizer.encode(probe["prefix"], add_special_tokens=False))
    validation = validate_segmentation(model, device, probe_ids, args.validation_tolerance)
    log.info("segmentation check: %s", validation)

    readout = Readout(tokenizer, int(rows[0].get("num_values") or 100), args.training in MODULAR_TRAINING)
    run_meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_path": str(path),
        "subset": subset,
        "intervention": asdict(intervention),
        "allow_neg_eigval": bool(model.config.linear_allow_neg_eigval),
        "layer_types": list(model.config.layer_types),
        "validation": validation,
        "samples": [],
    }
    for row in rows:
        meta = run_sample(model, tokenizer, row, args, device, readout, intervention, out_dir)
        run_meta["samples"].append(meta)
        (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2))
        torch.cuda.empty_cache()
    log.info("done: %d samples -> %s", len(rows), out_dir)


if __name__ == "__main__":
    main()
