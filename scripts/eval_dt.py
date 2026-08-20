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
MODEL_TYPES = ("hybrid", "transformer")
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

# Tiers launch in bands rather than one job per tier: a band's tasks share a
# single job, and therefore one vLLM server sized by the band's largest tier.
TOKEN_STRATA = {
    "normal": (NORMAL_TIERS, 262_144),
    "xl": (XL_TIERS, 1_048_576),
}

# Instances per tier used by --subsample, for quick passes over the full grid.
SUBSAMPLE_LIMIT = 100

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
    tasks: tuple[str, ...],
    stratum: str,
    max_model_len: int,
    limit: int | None = None,
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
        f"{get_model_short_name(model_path)}-state_bench_integer_code_{stratum}",
    ]
    harness_overrides = (
        HYBRID_HARNESS_OVERRIDES if model_type == "hybrid" else TRANSFORMER_HARNESS_OVERRIDES
    )
    for key, value in harness_overrides:
        cmd.extend(["-o", f"{key}={value.replace('{num_gpus}', str(num_gpus))}"])
    cmd.extend(["-o", f"provider.max_model_len={max_model_len}"])
    cmd.extend(["-m", model_path])
    # An `-o` override applies to the preceding `-t`, so a per-task cap has to be
    # repeated after each task rather than passed once for the whole job.
    for task in tasks:
        cmd.extend(["-t", task])
        if limit is not None:
            cmd.extend(["-o", f"limit={limit}"])
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
    parser.add_argument(
        "--strata",
        nargs="+",
        choices=TOKEN_STRATA,
        default=["normal"],
        help=(
            "StateBench context bands to evaluate: normal (tiers up to 256k) or "
            "xl (512k and above). Defaults to normal."
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


def main() -> None:
    args = parse_args()
    for model_type in args.model_type:
        for dataset_type in args.dataset_type:
            for stratum in args.strata:
                tiers, max_model_len = TOKEN_STRATA[stratum]
                tasks = tuple(f"state_bench:integer_code:{tier}" for tier in tiers)
                for seed in args.seed:
                    checkpoints = resolve_checkpoints(
                        model_type, dataset_type, seed, args.checkpoints, args.final
                    )
                    for checkpoint in checkpoints:
                        model_path = str(CHECKPOINTS[(model_type, dataset_type, seed)][checkpoint])
                        command = build_command(
                            model_path,
                            model_type,
                            tasks,
                            stratum,
                            max_model_len,
                            limit=args.limit,
                            num_gpus=args.gpus,
                            cluster=args.cluster,
                            group=args.group,
                            workspace=args.workspace,
                        )
                        if args.dry_run:
                            print(shlex.join(command))
                        else:
                            subprocess.run(command, check=True, cwd=OLMO_EVAL_DIR)


if __name__ == "__main__":
    main()
