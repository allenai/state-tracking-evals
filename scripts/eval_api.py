"""Runs subsampled state-bench evals against OpenAI, Claude, and Gemini models.

    uv run python scripts/eval_api.py --model gpt-4.1-mini --dry-run
    uv run python scripts/eval_api.py --model gpt-4.1-mini --samples-per-tier 10
    uv run python scripts/eval_api.py --model gpt-5-mini --reasoning-effort low
    uv run python scripts/eval_api.py --model claude-opus-5 --reasoning-effort high
    uv run python scripts/eval_api.py --model gemini-3-pro --reasoning-effort low

The provider follows from the model id (claude-* is Anthropic, gemini-* is
Google, anything else OpenAI) unless --provider names one; each provider's SDK
lives in its own module under scripts/api_providers/. API keys are read from
the project's .env -- OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY or
GEMINI_API_KEY -- along with HF_TOKEN for the gated dataset. Without
--reasoning-effort each model runs at its own API default. Token counts are
normalized across providers: output_tokens is billed output including any
reasoning, and reasoning_tokens is its reasoning share where the API reports it
(Anthropic does not). A reply cut off by --max-output-tokens is recorded with
status 'incomplete' and a declined one with status 'refusal'; both score wrong.
 Examples come from the solvable build (jacksonp-ai2/state-bench-solvable),
the only dataset that renders all four complexity classes -- r-trivial,
aperiodic, solvable, periodic -- from the same periodic sources, all with the
integer-code-modular formatter. Each (complexity, context tier) cell is
subsampled independently with a seeded RNG. Raising --samples-per-tier later
extends the existing sample rather than redrawing it, so a rerun scores only
the added examples.

By default (--task-format generate) the model gets the program alone, is free
to reason, and ends its reply with an `ANSWER: a ?= 3; b ?= 7; ...` line; it
is scored correct only if every variable's final value matches. With
--task-format choice the dataset's candidate final states are instead shown as
lettered options and the model answers with a letter, the closest API analogue
of the pt evals' log-likelihood ranking (chance is 1/len(choices)).

Outputs live under state-bench/data/api-evals/:
    samples/<config>/<tier>__seed<S>.jsonl         sampled docs, shared by all models
    <model>[__<effort>][__mc]/predictions.jsonl    one row per scored example
Reruns skip examples that already have an error-free prediction, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# scripts/ is on sys.path when this runs as a script, but not when
# plot_api_results imports it as scripts.eval_api.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_providers import (  # noqa: E402
    PROVIDERS,
    ProviderError,
    detect_provider,
    load_provider,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "state-bench" / "data" / "api-evals"

DATASET_REPO = "jacksonp-ai2/state-bench-solvable"
FORMATTER = "integer-code-modular"
COMPLEXITIES = ("r-trivial", "aperiodic", "solvable", "periodic")
# Tier names are their upper bound in Dolma tokens (the dataset's token_length).
TIERS = (
    "tokens_under_2k",
    "tokens_4k",
    "tokens_8k",
    "tokens_16k",
    "tokens_32k",
    "tokens_64k",
    "tokens_128k",
    "tokens_256k",
    "tokens_512k",
    "tokens_1m",
)
DEFAULT_TIERS = TIERS[: TIERS.index("tokens_128k") + 1]
# `text` duplicates `prefix` and is never read, so it is left out of the download.
DOC_COLUMNS = [
    "example_id",
    "complexity",
    "prefix",
    "choices",
    "correct_choice_index",
    "num_variables",
    "num_assignments",
    "max_dependency_length",
    "token_length",
    "token_stratum",
]
LETTERS = "ABCDEFGHIJ"
CHOICE_QUESTION = (
    "Which option gives the final value of every variable after all of the "
    "assignments above have executed?"
)
CHOICE_INSTRUCTION = "Answer with the letter of the correct option only."
GENERATE_QUESTION = (
    "What is the final value of every variable after all of the assignments "
    "above have executed? You may reason before answering. End your reply with "
    "exactly one line of the form"
)
GENERATE_INSTRUCTION = "with each <value> replaced by that variable's final value."
# `a ?= 3;` is the dataset's own completion syntax; `=` and `:` are accepted too.
STATE_PAIR_RE = re.compile(r"([A-Za-z_]\w*)\s*(?:\?=|=|:)\s*(-?\d+)")


def config_name(complexity: str) -> str:
    return f"{FORMATTER}--{complexity}"


def sample_docs(
    config: str, tier: str, n: int, seed: int, output_root: Path
) -> list[dict]:
    """First `n` docs of one config/tier's seeded sample, cached on disk.

    The cache is a single growing list per (config, tier, seed): asking for
    more docs than it holds tops it up with rows it does not contain yet, taken
    in the order of a seeded permutation of the split, so a larger sample
    always extends a smaller one and already-scored examples stay in it. Only
    the example_id column is scanned to choose rows; full columns are fetched
    just for the shards holding a new pick.
    """
    folder = output_root / "samples" / config
    cache = folder / f"{tier}__seed{seed}.jsonl"
    if cache.exists():
        docs = [json.loads(line) for line in cache.open()]
    else:
        # Caches from before top-ups were fixed-size, one file per n; the
        # largest seeds the growing list so its scored examples are kept.
        legacy = sorted(
            folder.glob(f"{tier}__n*__seed{seed}.jsonl"),
            key=lambda path: int(path.name.split("__n")[1].split("__")[0]),
        )
        docs = [json.loads(line) for line in legacy[-1].open()] if legacy else []
    if len(docs) >= n:
        return docs[:n]

    fs = HfFileSystem()
    shards = sorted(
        fs.glob(f"datasets/{DATASET_REPO}/data/{config}/test__{tier}-*.parquet")
    )
    if not shards:
        return docs
    shard_ids = [
        pq.ParquetFile(fs.open(shard))
        .read(columns=["example_id"])["example_id"]
        .to_pylist()
        for shard in shards
    ]
    locations = [
        (shard_index, row)
        for shard_index, ids in enumerate(shard_ids)
        for row in range(len(ids))
    ]
    order = list(range(len(locations)))
    random.Random(f"{seed}:{config}:{tier}").shuffle(order)
    have = {doc["example_id"] for doc in docs}
    picks = [
        locations[i]
        for i in order
        if shard_ids[locations[i][0]][locations[i][1]] not in have
    ][: n - len(docs)]

    fetched = {}
    for shard_index in sorted({shard_index for shard_index, _ in picks}):
        rows = [row for index, row in picks if index == shard_index]
        table = pq.ParquetFile(fs.open(shards[shard_index])).read(columns=DOC_COLUMNS)
        fetched.update(
            zip(((shard_index, row) for row in rows), table.take(rows).to_pylist())
        )
    docs.extend(fetched[pick] for pick in picks)

    folder.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as stream:
        for doc in docs:
            stream.write(json.dumps(doc) + "\n")
    return docs


def gold_state(doc: dict) -> dict[str, str]:
    """Final state from the correct choice, which is the dataset completion."""
    return dict(STATE_PAIR_RE.findall(doc["choices"][doc["correct_choice_index"]]))


def build_choice_prompt(doc: dict) -> str:
    options = "\n".join(
        f"{LETTERS[i]}. {choice}" for i, choice in enumerate(doc["choices"])
    )
    return f"{doc['prefix']}\n\n{CHOICE_QUESTION}\n\n{options}\n\n{CHOICE_INSTRUCTION}"


def build_generate_prompt(doc: dict) -> str:
    template = " ".join(f"{var} ?= <value>;" for var in gold_state(doc))
    return (
        f"{doc['prefix']}\n\n{GENERATE_QUESTION}\n\n"
        f"ANSWER: {template}\n\n{GENERATE_INSTRUCTION}"
    )


def parse_state(text: str) -> dict[str, str] | None:
    """Final state from the last ANSWER line, else the last line naming values."""
    lines = text.replace("*", "").replace("`", "").splitlines()
    answer_lines = [line for line in lines if re.search(r"answer\s*:", line, re.I)]
    for line in reversed(answer_lines or lines):
        tail = re.split(r"answer\s*:", line, flags=re.I)[-1]
        pairs = STATE_PAIR_RE.findall(tail)
        if pairs:
            return dict(pairs)
    return None


def parse_letter(text: str, num_choices: int) -> str | None:
    """Pull the chosen letter out of a reply, preferring an explicit answer."""
    valid = LETTERS[:num_choices]
    stripped = text.strip().strip("*().:").upper()
    if len(stripped) == 1 and stripped in valid:
        return stripped
    explicit = re.findall(
        rf"(?:answer|option)\s*(?:is)?\s*[:\-]?\s*\**\(?([{valid}])\b",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()
    bare = re.findall(rf"\b([{valid}])\b", text)
    return bare[-1] if bare else None


async def score_doc(
    provider,
    args: argparse.Namespace,
    doc: dict,
    config: str,
    tier: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    choice = args.task_format == "choice"
    gold = LETTERS[doc["correct_choice_index"]] if choice else gold_state(doc)
    record = {
        "model": args.model,
        "provider": args.provider,
        "reasoning_effort": args.reasoning_effort,
        "task_format": args.task_format,
        "config": config,
        "complexity": doc["complexity"],
        "tier": tier,
        "example_id": doc["example_id"],
        "num_variables": doc["num_variables"],
        "num_assignments": doc["num_assignments"],
        "max_dependency_length": doc["max_dependency_length"],
        "token_length": doc["token_length"],
        "num_choices": len(doc["choices"]),
        "gold": gold,
    }
    prompt = build_choice_prompt(doc) if choice else build_generate_prompt(doc)
    async with semaphore:
        try:
            completion = await provider.complete(
                prompt,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        except ProviderError as error:
            return {**record, "error": str(error)}

    text = completion.text
    if choice:
        prediction = parse_letter(text, len(doc["choices"]))
        scores = {"correct": prediction == gold}
    else:
        prediction = parse_state(text)
        matched = sum((prediction or {}).get(var) == val for var, val in gold.items())
        scores = {
            "correct": matched == len(gold),
            "variable_accuracy": matched / len(gold),
        }
    return {
        **record,
        "prediction": prediction,
        **scores,
        "output_text": text,
        "status": completion.status,
        "stop_reason": completion.stop_reason,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "reasoning_tokens": completion.reasoning_tokens,
        "error": None,
    }


def load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.open():
        row = json.loads(line)
        if not row.get("error"):
            done.add((row["config"], row["tier"], row["example_id"]))
    return done


def drop_errored(path: Path) -> None:
    """Rewrite predictions without errored rows, which a rerun will retry."""
    if not path.exists():
        return
    rows = [line for line in path.open() if not json.loads(line).get("error")]
    path.write_text("".join(rows))


def describe_api_key(key_env: tuple[str, ...], shell_env: set[str]) -> str:
    """Masked API key the provider's SDK will use, with its origin; safe to print."""
    for name in key_env:
        key = os.environ.get(name, "")
        if key:
            source = "shell environment" if name in shell_env else ".env"
            return f"{name}: {key[:8]}...{key[-4:]} ({len(key)} chars, from {source})"
    return f"{' / '.join(key_env)}: not set"


async def check_api_key(provider, model: str) -> str:
    """Confirm the key authenticates and can see `model`; the lookup is free."""
    try:
        await provider.check(model)
    except ProviderError as error:
        return f"key check failed: {error}"
    return f"key check ok: {model} is available"


def model_dir_name(args: argparse.Namespace) -> str:
    name = args.model.replace("/", "_")
    if args.reasoning_effort:
        name += f"__{args.reasoning_effort}"
    if args.task_format == "choice":
        name += "__mc"
    return name


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        required=True,
        help="Model id, e.g. gpt-4.1-mini, claude-opus-5, gemini-3-pro",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=None,
        help="API to call (default: inferred from the model id)",
    )
    parser.add_argument("--samples-per-tier", type=int, default=10)
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(DEFAULT_TIERS),
        choices=TIERS,
        metavar="TIER",
        help=f"Context tiers to sample (default: up to tokens_128k). Choices: {', '.join(TIERS)}",
    )
    parser.add_argument(
        "--complexities", nargs="+", default=list(COMPLEXITIES), choices=COMPLEXITIES
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Reasoning depth (OpenAI reasoning effort, Claude effort with adaptive "
        "thinking, Gemini thinking level); omit for the model's default. Also "
        "becomes part of the output directory name",
    )
    parser.add_argument(
        "--task-format",
        choices=["generate", "choice"],
        default="generate",
        help="generate: write out the final state (default); choice: pick a "
        "lettered option (output directory gets a __mc suffix)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16_384,
        help="Caps reasoning plus answer tokens per request; a reply cut off "
        "before its ANSWER line scores wrong (status 'incomplete')",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Skip examples whose dataset token_length exceeds this (e.g. the model's window)",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and cache docs, report request and token counts, and check the "
        "API key (a free model lookup); runs no evals",
    )
    args = parser.parse_args()

    args.provider = args.provider or detect_provider(args.model)
    # load_dotenv never overrides a variable already in the environment, so a
    # key exported in the shell wins over the one in .env.
    shell_env = set(os.environ)
    load_dotenv(REPO_ROOT / ".env")
    try:
        provider = load_provider(args.provider)
    except (TypeError, ValueError) as error:
        # The Gemini client refuses to construct without a key.
        parser.error(f"could not create the {args.provider} client: {error}")
    if (
        args.reasoning_effort
        and provider.EFFORTS
        and args.reasoning_effort not in provider.EFFORTS
    ):
        parser.error(
            f"--reasoning-effort {args.reasoning_effort} is not accepted by "
            f"{args.provider}; choose from {', '.join(provider.EFFORTS)}"
        )
    print(f"provider: {args.provider}")
    print(describe_api_key(provider.KEY_ENV, shell_env))

    jobs = []
    for complexity in args.complexities:
        config = config_name(complexity)
        for tier in args.tiers:
            docs = sample_docs(
                config, tier, args.samples_per_tier, args.seed, args.output_root
            )
            if not docs:
                print(f"  {config} {tier}: no shards, skipping", file=sys.stderr)
            jobs.extend((doc, config, tier) for doc in docs)

    if args.max_context_tokens:
        kept = [
            job for job in jobs if job[0]["token_length"] <= args.max_context_tokens
        ]
        print(
            f"Skipping {len(jobs) - len(kept)} example(s) over {args.max_context_tokens:,} tokens"
        )
        jobs = kept

    out_path = args.output_root / model_dir_name(args) / "predictions.jsonl"
    done = load_done(out_path)
    pending = [
        job for job in jobs if (job[1], job[2], job[0]["example_id"]) not in done
    ]
    pending_tokens = sum(doc["token_length"] for doc, _, _ in pending)
    print(
        f"{args.model}: {len(jobs)} sampled, {len(jobs) - len(pending)} already scored, "
        f"{len(pending)} to run (~{pending_tokens / 1e6:.2f}M input tokens, Dolma tokenizer)"
    )
    if args.dry_run:
        print(await check_api_key(provider, args.model))
        return
    if not pending:
        return
    # A bad key or model id would otherwise fail every request in the run.
    key_check = await check_api_key(provider, args.model)
    if not key_check.startswith("key check ok"):
        raise SystemExit(f"{key_check}\nNothing was sent; fix the key or model id.")

    drop_errored(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(score_doc(provider, args, doc, config, tier, semaphore))
        for doc, config, tier in pending
    ]
    errors = 0
    correct = 0
    with out_path.open("a") as stream, tqdm(total=len(tasks), desc=args.model) as bar:
        for next_done in asyncio.as_completed(tasks):
            row = await next_done
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            errors += bool(row["error"])
            correct += bool(row.get("correct"))
            bar.set_postfix(
                acc=f"{correct / max(bar.n + 1 - errors, 1):.2f}", errors=errors
            )
            bar.update()
    print(f"Wrote {out_path} ({errors} error(s); rerun to retry them)")


if __name__ == "__main__":
    asyncio.run(main())
