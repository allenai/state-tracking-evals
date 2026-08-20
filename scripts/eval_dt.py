"""Runs state-bench evals on directly-trained models."""

import argparse
import shlex
import subprocess
from pathlib import Path

# Gantry stamps each Beaker job with the git repo at the launch CWD, so launches
# run from the olmo-eval submodule — its pinned commit (which must be pushed to
# GitHub) is the olmo-eval version the job installs.
REPO_ROOT = Path(__file__).resolve().parent.parent
OLMO_EVAL_DIR = REPO_ROOT / "olmo-eval"

GROUP = "jacksonp-state-bench-dt"
CLUSTER = "ai2/saturn"
PRIORITY = "urgent"
NUM_GPUS = 2
WORKSPACE = "ai2/beyond-state"
BUDGET = "ai2/oe-other"

CHECKPOINT_BASE = Path("/weka/oe-training-default/ai2-llm/model-ladders/state-bench/60M")
MODEL_TYPES = ("transformer", "hybrid")
DATASET_TYPES = ("r-trivial", "aperiodic", "periodic")
SEEDS = (0,)

DATASET_DIRS = {
    "r-trivial": "integer-code--r-trivial",
    "aperiodic": "integer-code--aperiodic",
    "periodic": "integer-code--periodic",
}

CHECKPOINT_STEPS = {
    ("hybrid", "r-trivial"): (0, 2034, 2260, 4000, 4294, 4521),
    ("hybrid", "aperiodic"): (0, 2034, 2260, 4000, 4294, 4521),
    ("hybrid", "periodic"): (0, 2034, 2260, 4000, 4294, 4521),
    ("transformer", "r-trivial"): (0, 1971, 2190, 4000, 4161, 4381),
    ("transformer", "aperiodic"): (0, 1971, 2190, 4000, 4161, 4381),
    ("transformer", "periodic"): (0, 1971, 2190, 4000, 4161, 4381),
}

MODEL_ROOTS = {
    (model_type, dataset_type, seed): CHECKPOINT_BASE
    / model_type
    / DATASET_DIRS[dataset_type]
    / "Cx1"
    / f"init_seed{seed}"
    for model_type, dataset_type in CHECKPOINT_STEPS
    for seed in SEEDS
}

CHECKPOINTS = {
    model: {step: model_root / f"step{step}-hf" for step in CHECKPOINT_STEPS[model[:2]]}
    for model, model_root in MODEL_ROOTS.items()
}

OLMO_3_7B_BASE_ID = "allenai/Olmo-3-1025-7B"
# StateBench context tiers, ascending; each tier's name is its upper token bound.
# The full list lives upstream in
# olmo-eval/src/olmo_eval/evals/tasks/state_bench_lc.py. `tokens_2m_plus` is
# declared there but no dataset config carries that split yet, so it is omitted.
NORMAL_TIERS = (
    "tokens_under_2k",
    "tokens_4k",
    "tokens_8k",
    "tokens_16k",
    "tokens_32k",
    "tokens_64k",
    "tokens_128k",
    "tokens_256k",
)
XL_TIERS = ("tokens_512k", "tokens_1m")

# --strata selects tiers; each tier still launches as its own job so every one
# starts with a fresh vLLM process and CUDA allocator state.
TOKEN_STRATA = {
    "normal": NORMAL_TIERS,
    "xl": XL_TIERS,
}

# Nominal context ceiling per tier. Measured prompts reach only 88-94% of these,
# so a tier's window needs room for the scored continuation and nothing more.
# Sizing per tier rather than per band keeps the KV cache off the memory that
# full-prompt logprob scoring needs for its logits.
TIER_CONTEXT_CEILING = {
    "tokens_under_2k": 2_048,
    "tokens_4k": 4_096,
    "tokens_8k": 8_192,
    "tokens_16k": 16_384,
    "tokens_32k": 32_768,
    "tokens_64k": 65_536,
    "tokens_128k": 131_072,
    "tokens_256k": 262_144,
    "tokens_512k": 524_288,
    "tokens_1m": 1_048_576,
}
CONTINUATION_HEADROOM_TOKENS = 2_048

# StateBench scores full-prompt logprobs, so every prefill chunk materializes
# full-vocabulary logits on top of an already-allocated KV cache. These leave
# headroom for those temporaries and cap the per-chunk peak. vLLM only: the HF
# provider ignores gpu_memory_utilization and rejects the other engine kwargs.
STATE_BENCH_VLLM_OVERRIDES = [
    ("provider.kwargs.gpu_memory_utilization", "0.75"),
    ("provider.kwargs.max_num_batched_tokens", "8192"),
    ("provider.kwargs.max_num_seqs", "16"),
]
# Harness-level, so it applies whichever provider serves the model. The default
# harness batches at chunk_size=64, which is far more scored sequences in flight
# than a long-context logprob pass can hold.
BATCH_CHUNK_SIZE = 8
# Fragmentation across chunks is what usually turns a tight run into an OOM.
ALLOCATOR_ENV = "PYTORCH_ALLOC_CONF=expandable_segments:True"

# Instances per tier used by --subsample, for quick passes over the full grid.
SUBSAMPLE_LIMIT = 100


def max_model_len_for(tier: str) -> int:
    """Context window for one tier: its ceiling plus room for continuations."""
    return TIER_CONTEXT_CEILING[tier] + CONTINUATION_HEADROOM_TOKENS

BASE_HARNESS_OVERRIDES = [
    ("provider.num_instances", "{num_gpus}"),
    ("provider.kwargs.enforce_eager", "true"),
    ("provider.kwargs.compilation_config", '{"custom_ops":["-rms_norm"]}'),
    ("provider.add_bos_token", "false"),
    ("provider.prompt_logprobs", "1"),
    ("provider.logprob_temperature", "1.0"),
    ("provider.completion_use_prompt_token_ids", "true"),
    ("provider.completion_client_side_stop_trim", "true"),
    ("provider.completion_sentencepiece_cleanup", "true"),
    (
        "provider.dependencies",
        "[transformers @ git+https://github.com/yashassamaga/transformers.git@hybrid-small-suite]",
    ),
    ("provider.tokenizer", OLMO_3_7B_BASE_ID),
]

TRANSFORMER_HARNESS_OVERRIDES = [
    *BASE_HARNESS_OVERRIDES,
    ("provider.trust_remote_code", "true"),
    ("provider.kwargs.hf_overrides", '{"architectures":["TransformersForCausalLM"]}'),
]

HYBRID_HARNESS_OVERRIDES = [
    ("provider.num_instances", "{num_gpus}"),
    ("provider.kind", "hf"),
    ("provider.trust_remote_code", "true"),
    ("provider.dtype", "bfloat16"),
    (
        "provider.dependencies",
        "[transformers @ git+https://github.com/yashassamaga/transformers.git@hybrid-small-suite]",
    ),
    ("provider.tokenizer", OLMO_3_7B_BASE_ID),
]

HYBRID_IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"
SECRET_ENVS = ["JACKSONP_HF_TOKEN:HF_TOKEN", "jacksonp_GITHUB_TOKEN:GITHUB_TOKEN"]


def get_model_short_name(model_path: str) -> str:
    return "_".join(model_path.rstrip("/").split("/")[-5:]).lower().replace(":", "_")


def build_command(
    model_path: str,
    model_type: str,
    tier: str,
    limit: int | None = None,
    save_requests: bool = True,
    num_gpus: int = NUM_GPUS,
    cluster: str = CLUSTER,
    group: str = GROUP,
    workspace: str = WORKSPACE,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--project",
        str(REPO_ROOT),
        "olmo-eval",
        "beaker",
        "launch",
        "-H",
        "default",
        "-n",
        f"{get_model_short_name(model_path)}-state_bench_integer_code_{tier}",
    ]
    harness_overrides = (
        HYBRID_HARNESS_OVERRIDES if model_type == "hybrid" else TRANSFORMER_HARNESS_OVERRIDES
    )
    if model_type != "hybrid":
        harness_overrides = [*harness_overrides, *STATE_BENCH_VLLM_OVERRIDES]
    for key, value in harness_overrides:
        cmd.extend(["-o", f"{key}={value.replace('{num_gpus}', str(num_gpus))}"])
    cmd.extend(["-o", f"provider.max_model_len={max_model_len_for(tier)}"])
    cmd.extend(["-o", f"batching.chunk_size={BATCH_CHUNK_SIZE}"])
    cmd.extend(["-m", model_path])
    # An `-o` override applies to the preceding `-t`, so the cap goes after it.
    cmd.extend(["-t", f"state_bench:integer_code:{tier}"])
    if limit is not None:
        cmd.extend(["-o", f"limit={limit}"])
    if not save_requests:
        cmd.append("--no-save-requests")
    cmd.extend(["--env", ALLOCATOR_ENV])
    cmd.extend(
        [
            "--gpus",
            str(num_gpus),
            "--priority",
            PRIORITY,
            "--group",
            group,
            "--cluster",
            cluster,
            "--workspace",
            workspace,
            "--budget",
            BUDGET,
            "--inspect",
        ]
    )
    if model_type == "hybrid":
        cmd.extend(["--image", HYBRID_IMAGE])
    for secret_env in SECRET_ENVS:
        cmd.extend(["--secret-env", secret_env])
    return [*cmd, "--no-follow", "-y"]


def resolve_checkpoints(
    model_type: str,
    dataset_type: str,
    seed: int,
    requested_checkpoints: list[int] | None,
    final: bool = False,
) -> list[int]:
    model = (model_type, dataset_type, seed)
    if model not in CHECKPOINTS:
        available = ", ".join(
            f"{available_type}/{available_dataset}/seed{available_seed}"
            for available_type, available_dataset, available_seed in CHECKPOINTS
        )
        raise ValueError(
            f"No HF checkpoints are configured for {model_type}/{dataset_type}/seed{seed}. "
            f"Available combinations: {available}"
        )
    if requested_checkpoints is not None:
        invalid = sorted(set(requested_checkpoints) - set(CHECKPOINTS[model]))
        if invalid:
            valid = ", ".join(str(step) for step in CHECKPOINTS[model])
            raise ValueError(f"Invalid checkpoint(s) for {model}: {invalid}. Valid steps: {valid}")
        return requested_checkpoints
    if final:
        return [max(CHECKPOINTS[model])]
    return list(CHECKPOINTS[model])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        nargs="+",
        choices=MODEL_TYPES,
        default=MODEL_TYPES,
        help="Model architecture(s) to evaluate. Defaults to all architectures.",
    )
    parser.add_argument(
        "--dataset-type",
        nargs="+",
        choices=DATASET_TYPES,
        default=DATASET_TYPES,
        help="Training dataset variant(s) to evaluate. Defaults to all variants.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=SEEDS,
        help="Initialization seed(s) to evaluate. Defaults to all configured seeds.",
    )
    parser.add_argument("--gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--cluster", default=CLUSTER)
    parser.add_argument("--group", default=GROUP, help="Beaker workgroup for launched jobs.")
    parser.add_argument(
        "--workspace", default=WORKSPACE, help="Beaker workspace for launched jobs."
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoints", "-c", nargs="+", type=int)
    checkpoint_group.add_argument(
        "--all-checkpoints",
        dest="final",
        action="store_false",
        help="Evaluate every configured checkpoint.",
    )
    checkpoint_group.add_argument(
        "--final", dest="final", action="store_true", help=argparse.SUPPRESS
    )
    parser.set_defaults(final=True)
    tier_group = parser.add_mutually_exclusive_group()
    tier_group.add_argument(
        "--strata",
        nargs="+",
        choices=TOKEN_STRATA,
        default=["normal"],
        help=(
            "StateBench context bands to evaluate: normal (tiers up to 256k) or "
            "xl (512k and above). Each tier in a band launches as its own job. "
            "Defaults to normal."
        ),
    )
    tier_group.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIER_CONTEXT_CEILING),
        help="Launch these exact tiers, e.g. to retry the one that ran out of memory.",
    )
    parser.add_argument(
        "--no-save-requests",
        dest="save_requests",
        action="store_false",
        help=(
            "Skip saving request payloads, which are huge at long context. The "
            "notebooks join example metadata from the source dataset instead."
        ),
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--subsample",
        dest="limit",
        action="store_const",
        const=SUBSAMPLE_LIMIT,
        help=f"Shorthand for --limit {SUBSAMPLE_LIMIT}.",
    )
    limit_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate at most this many instances per tier, sampled with the "
            "task's own seed. Omit to evaluate every instance."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_tiers(requested_tiers: list[str] | None, strata: list[str]) -> list[str]:
    """Expand --tiers or --strata into tiers, ascending by context length."""
    selected = set(requested_tiers or [tier for band in strata for tier in TOKEN_STRATA[band]])
    return [tier for tier in TIER_CONTEXT_CEILING if tier in selected]


def main() -> None:
    args = parse_args()
    launched = 0
    for model_type in args.model_type:
        for dataset_type in args.dataset_type:
            for tier in resolve_tiers(args.tiers, args.strata):
                for seed in args.seed:
                    checkpoints = resolve_checkpoints(
                        model_type, dataset_type, seed, args.checkpoints, args.final
                    )
                    for checkpoint in checkpoints:
                        model_path = str(CHECKPOINTS[(model_type, dataset_type, seed)][checkpoint])
                        command = build_command(
                            model_path,
                            model_type,
                            tier,
                            limit=args.limit,
                            save_requests=args.save_requests,
                            num_gpus=args.gpus,
                            cluster=args.cluster,
                            group=args.group,
                            workspace=args.workspace,
                        )
                        if args.dry_run:
                            print(shlex.join(command))
                        else:
                            subprocess.run(command, check=True, cwd=OLMO_EVAL_DIR)
                        launched += 1
    print(f"\n{'Would launch' if args.dry_run else 'Launched'} {launched} job(s).")


if __name__ == "__main__":
    main()
