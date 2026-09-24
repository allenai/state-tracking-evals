"""Runs state-bench evals on directly-trained 275M-class models.

Every 275M StateBench variant (the two transformers, the two hybrids, and the two pure-GDN
models) is saved in the HF ``olmo3_5_hybrid`` format, so all of them are served by the
HF provider with the ``jopetty/transformers@olmo-3.5-hybrid-state-bench`` fork installed
in the job (the fork supports the homogeneous layer_types the non-hybrid variants use).

Checkpoint steps are hardcoded per model type (they depend only on the model's
non-embedding parameter count via the WSDS Chinchilla schedule, not on the training
distribution), so this script runs fine from a laptop without weka access; only the
launched jobs read the checkpoint paths.
"""

import argparse
import shlex
import subprocess
from pathlib import Path

# Gantry stamps each Beaker job with the git repo at the launch CWD, so launches
# run from the olmo-eval submodule — its pinned commit (which must be pushed to
# GitHub) is the olmo-eval version the job installs.
REPO_ROOT = Path(__file__).resolve().parent.parent
OLMO_EVAL_DIR = REPO_ROOT / "olmo-eval"

GROUP = "jacksonp-state-bench-dt-275m"
CLUSTER = "ai2/saturn"
PRIORITY = "urgent"
NUM_GPUS = 2
WORKSPACE = "ai2/beyond-state"
BUDGET = "ai2/oe-other"

CHECKPOINT_BASE = Path("/weka/oe-training-default/ai2-llm/model-ladders/state-bench/275M")
MODEL_TYPES = ("transformer-rope", "transformer-nope", "hybrid", "hybrid-sdp", "gdn-sdp", "gdn-full")
DATASET_TYPES = (
    "r-trivial",
    "aperiodic",
    "periodic",
    "periodic-sens-shuffled",
    "periodic-sens-curriculum",
    # Trained on the solvable rewrite of the periodic training sources
    # (state-tracking-solvable-v2); evaluated on the modular test tiers.
    "solvable",
)
SEEDS = (0,)

DATASET_DIRS = {
    "r-trivial": "integer-code--r-trivial",
    "aperiodic": "integer-code--aperiodic",
    "periodic": "integer-code--periodic",
    "periodic-sens-shuffled": "integer-code--periodic--sens-shuffled",
    "periodic-sens-curriculum": "integer-code--periodic--sens-curriculum",
    # The ladder names the v2 run distinctly from the v1 increment-source runs.
    "solvable": "integer-code-modular--solvable-v2",
}

# Each job evaluates every complexity class of one formatter for one context tier, so
# the matched-distribution result is always included. Models trained on the modular
# (solvable) build are evaluated on the modular formatter's test set, whose zero-based
# value vocabulary matches their training data; everything else uses integer-code.
MODULAR_DATASET_TYPES = frozenset({"solvable"})


def eval_formatter_for(dataset_type: str) -> str:
    return "integer_code_modular" if dataset_type in MODULAR_DATASET_TYPES else "integer_code"

# Fixed WSDS checkpoint steps per model type: [period-1 pre-decay, period-1 end,
# period-2 pre-decay, period-2 end(final)], computed from olmo-core's
# RunConfigurator.configure_checkpoint_intervals with each variant's non-embedding
# parameter count and the launched jobs' 524288-token global batch. Final steps
# verified against the trained runs (7451 / 8059 / 8211).
CHECKPOINT_STEPS = {
    "transformer-rope": (3353, 3726, 7078, 7451),
    "transformer-nope": (3353, 3726, 7078, 7451),
    "hybrid": (3627, 4030, 7656, 8059),
    "hybrid-sdp": (3627, 4030, 7656, 8059),  # same param count as hybrid
    "gdn-sdp": (3695, 4106, 7800, 8211),
    "gdn-full": (3695, 4106, 7800, 8211),
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
# starts with a fresh process and CUDA allocator state.
TOKEN_STRATA = {
    "normal": NORMAL_TIERS,
    "xl": XL_TIERS,
}

# Nominal context ceiling per tier. Measured prompts reach only 88-94% of these,
# so a tier's window needs room for the scored continuation and nothing more.
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

# The default harness batches at chunk_size=64, which is far more scored
# sequences in flight than a long-context logprob pass can hold.
BATCH_CHUNK_SIZE = 8
# Fragmentation across chunks is what usually turns a tight run into an OOM.
ALLOCATOR_ENV = "PYTORCH_ALLOC_CONF=expandable_segments:True"

# Instances per tier used by --subsample, for quick passes over the full grid.
SUBSAMPLE_LIMIT = 100


def max_model_len_for(tier: str) -> int:
    """Context window for one tier: its ceiling plus room for continuations."""
    return TIER_CONTEXT_CEILING[tier] + CONTINUATION_HEADROOM_TOKENS


# All 275M variants are olmo3_5_hybrid checkpoints, served by the HF provider with the
# fork that supports homogeneous layer_types (pure-transformer / pure-GDN variants).
HARNESS_OVERRIDES = [
    ("provider.num_instances", "{num_gpus}"),
    ("provider.kind", "hf"),
    ("provider.trust_remote_code", "true"),
    ("provider.dtype", "bfloat16"),
    (
        "provider.dependencies",
        "[transformers @ git+https://github.com/jopetty/transformers.git@olmo-3.5-hybrid-state-bench]",
    ),
    ("provider.tokenizer", OLMO_3_7B_BASE_ID),
]

EVAL_IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"
SECRET_ENVS = ["JACKSONP_HF_TOKEN:HF_TOKEN", "jacksonp_GITHUB_TOKEN:GITHUB_TOKEN"]


def get_model_short_name(model_path: str) -> str:
    return "_".join(model_path.rstrip("/").split("/")[-5:]).lower().replace(":", "_")


# Beaker caps experiment names at 128 characters, and appends a 5-character
# suffix ("-55bd") when a job of the same name already exists, as a rerun does.
BEAKER_NAME_LIMIT = 128
BEAKER_DEDUP_SUFFIX_LEN = 5


def experiment_name(model_path: str, formatter: str, tier: str) -> str:
    """Beaker experiment name for one model/tier job; must leave room for a rerun suffix.

    The notebooks select results by model path and task, and read training
    metadata from the model path, so the name only needs to be unique and
    readable.
    """
    name = f"{get_model_short_name(model_path)}-{formatter}_{tier}"
    if len(name) + BEAKER_DEDUP_SUFFIX_LEN > BEAKER_NAME_LIMIT:
        raise ValueError(
            f"experiment name is {len(name)} characters; Beaker allows "
            f"{BEAKER_NAME_LIMIT - BEAKER_DEDUP_SUFFIX_LEN} before a rerun suffix: {name}"
        )
    return name


def model_root(model_type: str, dataset_type: str, seed: int) -> Path:
    return CHECKPOINT_BASE / model_type / DATASET_DIRS[dataset_type] / "Cx1" / f"init_seed{seed}"


def checkpoint_paths(model_type: str, dataset_type: str, seed: int) -> dict[int, Path]:
    """Map step -> converted checkpoint dir for one run, from the fixed step tables."""
    root = model_root(model_type, dataset_type, seed)
    return {step: root / f"step{step}-hf" for step in CHECKPOINT_STEPS[model_type]}


def build_command(
    model_path: str,
    tier: str,
    formatter: str = "integer_code",
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
        experiment_name(model_path, formatter, tier),
    ]
    for key, value in HARNESS_OVERRIDES:
        cmd.extend(["-o", f"{key}={value.replace('{num_gpus}', str(num_gpus))}"])
    cmd.extend(["-o", f"provider.max_model_len={max_model_len_for(tier)}"])
    cmd.extend(["-o", f"batching.chunk_size={BATCH_CHUNK_SIZE}"])
    cmd.extend(["-m", model_path])
    # An `-o` override applies to the preceding `-t`, so the cap goes after it.
    cmd.extend(["-t", f"state_bench:{formatter}:{tier}"])
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
            "--image",
            EVAL_IMAGE,
        ]
    )
    for secret_env in SECRET_ENVS:
        cmd.extend(["--secret-env", secret_env])
    return [*cmd, "--no-follow", "-y"]


def resolve_checkpoints(
    available: dict[int, Path],
    requested_checkpoints: list[int] | None,
    final: bool,
) -> list[int]:
    if requested_checkpoints is not None:
        invalid = sorted(set(requested_checkpoints) - set(available))
        if invalid:
            valid = ", ".join(str(step) for step in sorted(available))
            raise ValueError(f"Invalid checkpoint(s): {invalid}. Configured steps: {valid}")
        return sorted(requested_checkpoints)
    if final:
        return [max(available)]
    return sorted(available)


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
        help="Evaluate every converted checkpoint instead of only the latest.",
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
    # Tiers ascend outermost so every model's cheap short-context jobs queue
    # before any long-context job: the multi-hour 256k runs would otherwise
    # hog GPU slots ahead of quick jobs for the remaining models.
    for tier in resolve_tiers(args.tiers, args.strata):
        for model_type in args.model_type:
            for dataset_type in args.dataset_type:
                for seed in args.seed:
                    available = checkpoint_paths(model_type, dataset_type, seed)
                    checkpoints = resolve_checkpoints(available, args.checkpoints, args.final)
                    for checkpoint in checkpoints:
                        command = build_command(
                            str(available[checkpoint]),
                            tier,
                            formatter=eval_formatter_for(dataset_type),
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
