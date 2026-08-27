"""Runs state-bench evals on pretrained models."""

import argparse
import json
import subprocess
import time
from pathlib import Path

# Gantry stamps each Beaker job with the git repo at the launch CWD, so launches
# run from the olmo-eval submodule — its pinned commit (which must be pushed to
# GitHub) is the olmo-eval version the job installs.
REPO_ROOT = Path(__file__).resolve().parent.parent
OLMO_EVAL_DIR = REPO_ROOT / "olmo-eval"

GROUP = "jacksonp-state-bench-pt"
DEFAULT_CLUSTER = "ai2/saturn"
NUM_GPUS = 4
WORKSPACE = "ai2/beyond-state"
BUDGET = "ai2/oe-other"
# IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"
IMAGE = "ai2-tylerm/olmo-eval-cu1281-trc2100-amd64"

PRIORITIES = {
    "ai2/saturn": "urgent",
    "ai2/jupiter": "urgent",
    "ai2/holmes": "high",
}
# PRIORITY = PRIORITIES.get(CLUSTER, "high")

CKPT_BASE = "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls"

# The olmo_hybrid_small architecture is served via the transformers/vLLM plugins
# (stock transformers/vLLM, no fork). The eval job pip-installs them from this
# repo's GitHub remote at the branch/commit this launcher runs from, so they must
# be committed and pushed.
PLUGINS_REPO = "git+https://github.com/jopetty/hybrid-small-suite.git"
DEFAULT_PLUGINS_REF = "main"


def _plugin_dependencies_override(plugins_ref: str | None = None) -> str:
    """``provider.dependencies`` override installing both ladder plugins.

    Installs the vLLM and transformers plugins (git subdirectory installs) at
    ``plugins_ref`` so the eval job serves the olmo_hybrid_small architecture on
    stock vLLM/transformers. olmo-eval v1.0.0 JSON-decodes the value, so it must
    be a JSON list of strings.
    """
    ref = plugins_ref or DEFAULT_PLUGINS_REF
    deps = [
        f"vllm-plugin @ {PLUGINS_REPO}@{ref}#subdirectory=plugins/vllm_plugin",
        f"transformers-plugin @ {PLUGINS_REPO}@{ref}#subdirectory=plugins/transformers_plugin",
    ]
    return f"provider.dependencies={json.dumps(deps)}"


# Each stage maps size -> list of HF checkpoint paths.
# Single-model stages have one-element lists; grid searches have multiple.
all_stages: dict[str, dict[str, list[str]]] = {
    "pretraining": {
        "275m": [f"{CKPT_BASE}/hybrid-small-275M-Cx100/step161186-hf/"],
        "275m-transformer": [f"{CKPT_BASE}/hybrid-small-transformer-275M/step161186-hf/"],
        "275m-gdn": [f"{CKPT_BASE}/hybrid-small-gdn-275M/step161186-hf/"],
        "450m": [f"{CKPT_BASE}/hybrid-small-450m-cx100-lr8e-3/step179814-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-810M-Cx100/step269926-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-1.4B-Cx100/step308433-hf/"],
    },
    "midtraining": {
        "275m": [f"{CKPT_BASE}/hybrid-small-midtraining-275M-v2-lr1.6e-3/step38147-hf/"],
        "275m-transformer": [
            f"{CKPT_BASE}/hybrid-small-transformer-275M-midtraining/step38147-hf/"
        ],
        "275m-gdn": [f"{CKPT_BASE}/hybrid-small-gdn-275M-midtraining/step38147-hf/"],
        "450m": [f"{CKPT_BASE}/hybrid-small-midtraining-450m/step38147-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-midtraining-v2-810M-lr4e-4/step23842-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-midtraining-v2-1.4b-lr4e-4/step11921-hf/"],
    },
    "long_context": {
        "275m": [f"{CKPT_BASE}/hybrid-small-long-context-v2-275m/step47684-hf/"],
        "275m-transformer": [f"{CKPT_BASE}/hybrid-small-transformer-275M-lc/step47684-hf/"],
        "275m-gdn": [f"{CKPT_BASE}/hybrid-small-gdn-275M-lc/step47684-hf/"],
        # TODO: still training, update with final step folder.
        # "450m": [f"{CKPT_BASE}/hybrid-small-lc-v2-450m/stepXXXXX-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-long-context-v2-810m/step23842-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-long-context-v2-1.4b/step23842-hf/"],
        "2.7b": [f"{CKPT_BASE}/hybrid-small-lc-2.7B/step23842-hf/"],
        "2.7b-lr5em5": [f"{CKPT_BASE}/hybrid-small-lc-2.7B-lr5em5/step23842-hf/"],
        "2.7b-warmup5k": [f"{CKPT_BASE}/hybrid-small-lc-2.7B-warmup5k/step23842-hf/"],
    },
}

# External baseline models (HF base checkpoints) to compare against.
# Launch with: --model <hf_id> (e.g. --model Qwen/Qwen3.5-2B-Base).
# ctx = native context length; arch notes matter for serving.
#   - qwen3_5 models are hybrid (Gated DeltaNet linear-attention + periodic Gated
#     Attention, dense FFN — not MoE at these sizes) with a vision encoder;
#     served text-only via provider.kwargs.language_model_only=true (harmless no-op
#     on text-only models, so it is now set globally in build_command).
# status: "ok" = model serving was previously smoke-tested through this stack.
BASELINES: dict[str, dict[str, str | int]] = {
    # Qwen3 dense transformers (32k ctx, Apr 2025)
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-0.6B-Base", "ctx": 32768, "status": "ok"},
    "qwen3-1.7b": {"hf": "Qwen/Qwen3-1.7B-Base", "ctx": 32768, "status": "ok"},
    # Qwen3.5 hybrid (Gated DeltaNet + Gated Attention), dense FFN (256k native, Feb 2026)
    "qwen3.5-0.8b": {"hf": "Qwen/Qwen3.5-0.8B-Base", "ctx": 262144, "status": "untested"},
    "qwen3.5-2b": {"hf": "Qwen/Qwen3.5-2B-Base", "ctx": 262144, "status": "ok"},
    # Llama 3.2 dense (128k ctx, Sep 2024)
    "llama3.2-1b": {"hf": "meta-llama/Llama-3.2-1B", "ctx": 131072, "status": "ok"},
    "llama3.2-3b": {"hf": "meta-llama/Llama-3.2-3B", "ctx": 131072, "status": "untested"},
    # Gemma 3 dense (32k ctx, 2025)
    "gemma3-270m": {"hf": "google/gemma-3-270m", "ctx": 32768, "status": "ok"},
    "gemma3-1b": {"hf": "google/gemma-3-1b-pt", "ctx": 32768, "status": "untested"},
    # OLMo 2 dense (4k ctx, Apr 2025) — fully-open; weak/short-ctx, downstream baseline
    "olmo2-1b": {"hf": "allenai/OLMo-2-0425-1B", "ctx": 4096, "status": "untested"},
    # Pythia dense (2k ctx, Apr 2023) — fully-open scaling suite; weak/short-ctx baseline
    "pythia-1b": {"hf": "EleutherAI/pythia-1b", "ctx": 2048, "status": "untested"},
    "pythia-1.4b": {"hf": "EleutherAI/pythia-1.4b", "ctx": 2048, "status": "untested"},
    # SmolLM2 dense llama (8k ctx, Feb 2025) — fully-open; downstream-only baseline
    "smollm2-1.7b": {"hf": "HuggingFaceTB/SmolLM2-1.7B", "ctx": 8192, "status": "untested"},
    # SmolLM3 dense GQA+NoPE (64k native, 128k YaRN; 2025) — fully-open; long-ctx baseline
    "smollm3-3b": {"hf": "HuggingFaceTB/SmolLM3-3B-Base", "ctx": 65536, "status": "untested"},
    # Qwen2.5 dense transformers (32k native ctx, Sep 2024) — prior-gen Qwen trendline
    "qwen2.5-0.5b": {"hf": "Qwen/Qwen2.5-0.5B", "ctx": 32768, "status": "untested"},
    "qwen2.5-1.5b": {"hf": "Qwen/Qwen2.5-1.5B", "ctx": 32768, "status": "untested"},
    # Qwen2.5-3B is qwen-research license (NOT Apache like 0.5B/1.5B)
    "qwen2.5-3b": {"hf": "Qwen/Qwen2.5-3B", "ctx": 32768, "status": "untested"},
    # LFM2.5 hybrid (short-conv + GQA) base checkpoints (32k ctx, Jun 2026) — on-device baseline
    "lfm2.5-230m": {"hf": "LiquidAI/LFM2.5-230M-Base", "ctx": 32768, "status": "untested"},
    "lfm2.5-350m": {"hf": "LiquidAI/LFM2.5-350M-Base", "ctx": 32768, "status": "untested"},
    # OLMo 3 7b
    "olmo-3-7b": {"hf": "allenai/Olmo-3-1025-7B", "ctx": 65536, "status": "ok"},
    "olmo-3-hybrid-7b": {"hf": "allenai/Olmo-Hybrid-7B", "ctx": 65536, "status": "ok"},
}

# StateBench context tiers, ascending; each tier's name is its upper token bound.
# The full list lives upstream in
# olmo-eval/src/olmo_eval/evals/tasks/state_bench_lc.py. `tokens_2m_plus` is
# declared there but no dataset config carries that split yet, so it is omitted.
# Note `tokens_1m` only exists for some formatters (all integer-code and ruler
# configs, plus the periodic variants of the rest).
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
# full-vocabulary logits on top of an already-allocated KV cache. Leave headroom
# for those temporaries and cap request concurrency. max_num_batched_tokens is
# set below at 4096, lower than the 8192 these runs were first tuned with.
STATE_BENCH_VLLM_OVERRIDES = [
    ("provider.kwargs.gpu_memory_utilization", "0.75"),
    ("provider.kwargs.max_num_seqs", "16"),
]
# Harness-level. The default harness batches at chunk_size=64, which is far more
# scored sequences in flight than a long-context logprob pass can hold.
BATCH_CHUNK_SIZE = 8
# Fragmentation across chunks is what usually turns a tight run into an OOM.
ALLOCATOR_ENV = "PYTORCH_ALLOC_CONF=expandable_segments:True"

# Instances per tier used by --subsample, for quick passes over the full grid.
SUBSAMPLE_LIMIT = 100


def max_model_len_for(tier: str) -> int:
    """Context window for one tier: its ceiling plus room for continuations."""
    return TIER_CONTEXT_CEILING[tier] + CONTINUATION_HEADROOM_TOKENS


def resolve_tiers(requested_tiers: list[str] | None, strata: list[str]) -> list[str]:
    """Expand --tiers or --strata into tiers, ascending by context length."""
    selected = set(requested_tiers or [tier for band in strata for tier in TOKEN_STRATA[band]])
    return [tier for tier in TIER_CONTEXT_CEILING if tier in selected]


def experiment_name(model_path: str, tier: str) -> str:
    """Build a deterministic Beaker experiment name for a model."""
    parts = model_path.rstrip("/").split("/")
    model_short = "_".join(parts[-2:]).lower()
    return f"{model_short}-state_bench-{tier}"


def build_command(
    model_path: str,
    num_gpus: int = NUM_GPUS,
    group: str = GROUP,
    plugins_ref: str | None = None,
    cluster: str = DEFAULT_CLUSTER,
    tier: str = "tokens_256k",
    limit: int | None = None,
    save_requests: bool = True,
    workspace: str = WORKSPACE,
) -> list[str]:
    exp_name = experiment_name(model_path, tier)

    priority = PRIORITIES.get(cluster, "high")

    cmd = ["uv", "run", "--project", str(REPO_ROOT), "olmo-eval", "beaker", "launch"]
    cmd += ["-H", "default"]
    cmd += ["-n", exp_name]
    cmd += ["-o", f"provider.num_instances={num_gpus}"]
    cmd += ["-o", "provider.kwargs.enforce_eager=true"]
    cmd += ["-o", "provider.kwargs.mamba_ssm_cache_dtype=float32"]
    # Serve vision-language baselines (e.g. qwen3_5) as text-only. Harmless no-op
    # on text-only models.
    cmd += ["-o", "provider.kwargs.language_model_only=true"]
    cmd += ["-o", "provider.add_bos_token=false"]
    cmd += ["-o", "provider.kind=vllm"]
    cmd += ["-o", "provider.package=wheel"]
    cmd += ["-o", _plugin_dependencies_override(plugins_ref)]
    cmd += ["-o", "provider.kwargs.attention_backend=FLASH_ATTN"]
    cmd += ["-o", f"provider.max_model_len={max_model_len_for(tier)}"]
    cmd += ["-o", "provider.kwargs.max_num_batched_tokens=4096"]
    for key, value in STATE_BENCH_VLLM_OVERRIDES:
        cmd += ["-o", f"{key}={value}"]
    cmd += ["-o", f"batching.chunk_size={BATCH_CHUNK_SIZE}"]
    if model_path.rstrip("/") == "allenai/Olmo-3-1025-7B":
        cmd += ["-o", "provider.kwargs.model_impl=transformers"]
    cmd += ["-m", model_path]
    # An `-o` override applies to the preceding `-t`, so the cap goes after it.
    cmd += ["-t", f"state_bench:{tier}"]
    if limit is not None:
        cmd += ["-o", f"limit={limit}"]
    if not save_requests:
        cmd += ["--no-save-requests"]
    cmd += ["--gpus", str(num_gpus)]
    cmd += ["--retries", "3"]
    cmd += ["--priority", priority]
    cmd += ["--group", group]
    cmd += ["--cluster", cluster]
    cmd += ["--workspace", workspace]
    cmd += ["--budget", BUDGET]
    cmd += ["--image", IMAGE]
    cmd += ["--inspect"]
    cmd += ["--secret-env", "jacksonp_HF_TOKEN:HF_TOKEN"]
    cmd += ["--secret-env", "jacksonp_GITHUB_TOKEN:GITHUB_TOKEN"]
    cmd += ["--env", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"]
    cmd += ["--env", ALLOCATOR_ENV]
    cmd += ["--no-follow"]
    cmd += ["-y"]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["275m", "275m-transformer", "275m-gdn", "450m", "810m", "1.4b", "2.7b", "2.7b-lr5em5", "2.7b-warmup5k"],
        default=["275m"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(all_stages.keys()),
        default=["pretraining"],
    )
    parser.add_argument(
        "--group", type=str, default=GROUP, help="Beaker workgroup for launched jobs."
    )
    parser.add_argument("--gpus", type=int, default=NUM_GPUS, help="Number of GPUs per job.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom model path or HF name (overrides --sizes/--stages)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINES.keys()),
        default=None,
        help="External baseline keys to evaluate using each model's native context length.",
    )
    parser.add_argument(
        "--plugins-ref",
        type=str,
        default=None,
        help=f"Git ref for the vLLM/transformers plugins (default: {DEFAULT_PLUGINS_REF}).",
    )
    parser.add_argument(
        "--delay", type=int, default=0, help="Seconds to wait between launching each eval job"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=DEFAULT_CLUSTER,
        help="Cluster to submit jobs to (default: ai2/saturn).",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=WORKSPACE,
        help=f"Beaker workspace for launched jobs (default: {WORKSPACE}).",
    )
    tier_group = parser.add_mutually_exclusive_group()
    tier_group.add_argument(
        "--strata",
        nargs="+",
        choices=TOKEN_STRATA,
        default=list(TOKEN_STRATA),
        help=(
            "Context bands to evaluate: normal (tiers up to 256k) or xl (512k "
            "and above). Each tier in a band launches as its own job. Defaults "
            "to both."
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
            "Skip saving request payloads, which are huge at long context. Note "
            "pt_results.ipynb reads per-example doc metadata from them."
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
    args = parser.parse_args()
    num_gpus = args.gpus
    group = args.group

    plugins_ref = args.plugins_ref or DEFAULT_PLUGINS_REF

    launched = 0

    def launch_task(
        model_path,
        *,
        label,
        tier,
    ):
        """Launch one StateBench context tier for a model."""
        nonlocal launched
        max_model_len = max_model_len_for(tier)
        cmd = build_command(
            model_path,
            num_gpus,
            group=group,
            plugins_ref=plugins_ref,
            cluster=args.cluster,
            tier=tier,
            limit=args.limit,
            save_requests=args.save_requests,
            workspace=args.workspace,
        )
        print(f"\n=== {label} | {tier} | {num_gpus} GPUs | mml={max_model_len} ===")
        print(" ".join(cmd))
        if not args.dry_run:
            if launched > 0 and args.delay > 0:
                time.sleep(args.delay)
            subprocess.run(cmd, check=True, cwd=OLMO_EVAL_DIR)
        launched += 1

    targets = []
    if args.model:
        targets.append((args.model, "custom"))
    elif args.baselines:
        for key in args.baselines:
            spec = BASELINES[key]
            hf = spec["hf"]
            targets.append((hf, f"baseline {key} ({hf})"))
    else:
        for stage in args.stages:
            checkpoints = all_stages[stage]
            for size in args.sizes:
                if size not in checkpoints:
                    print(f"[skip] {stage} has no size {size}")
                    continue
                for model_path in checkpoints[size]:
                    short_name = model_path.rstrip("/").split("/")[-1]
                    targets.append((model_path, f"{stage}/{size}/{short_name}"))

    for tier in resolve_tiers(args.tiers, args.strata):
        for model_path, label in targets:
            launch_task(
                model_path,
                label=label,
                tier=tier,
            )

    print(f"\n{'Would launch' if args.dry_run else 'Launched'} {launched} job(s).")


if __name__ == "__main__":
    main()
