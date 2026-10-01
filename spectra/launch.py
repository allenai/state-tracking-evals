"""Launch spectra jobs on Beaker with gantry.

Jobs clone this repo at the current commit (which must be pushed), install the
analysis dependencies into the eval image's ``/opt/venv`` against its pinned
torch/CUDA wheels, and run ``python -m spectra.run`` (or the metadata dump).
Results land in each job's result dataset under ``/results``; fetch them with
``spectra/fetch.py``.

Suites (``--suite``) name the experiment grid; ``--dry-run`` prints commands::

    uv run python -m spectra.launch --suite pilot --dry-run
    uv run python -m spectra.launch --suite baseline
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = "ai2/beyond-state"
BUDGET = "ai2/oe-other"
# Any weka-mounted cluster; jobs take whichever has GPUs free first.
CLUSTERS = ("ai2/holmes", "ai2/titan", "ai2/saturn", "ai2/jupiter", "ai2/ceres")
PRIORITY = "urgent"
# The eval image (scripts/eval_dt.py EVAL_IMAGE): torch + CUDA in /opt/venv.
IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"
TRANSFORMERS_FORK = "transformers @ git+https://github.com/jopetty/transformers.git@olmo-3.5-hybrid-state-bench"
HF_HOME = "/weka/oe-eval-default/oyvindt/hf-cache"

# Install into the image's venv, constrained to its torch/CUDA/triton wheels so
# fla's kernels match the preinstalled torch.
INSTALL = " && ".join([
    # A job-local uv cache: the shared weka cache can block on locks while many
    # eval jobs install at once.
    "export VIRTUAL_ENV=/opt/venv UV_CACHE_DIR=/tmp/uv-cache",
    "uv pip freeze | grep -E '^(torch|nvidia-|triton)' > /tmp/cuda-constraints.txt",
    "uv pip install -c /tmp/cuda-constraints.txt numpy datasets flash-linear-attention",
    f"uv --no-config pip install --reinstall-package transformers '{TRANSFORMERS_FORK}' -c /tmp/cuda-constraints.txt",
    f"{'/opt/venv/bin/python'} -c 'import fla, transformers, torch; print(\"env ok\", torch.__version__, transformers.__version__)'",
])
PYTHON = "/opt/venv/bin/python"


@dataclass
class Job:
    name: str
    args: list[str]
    gpus: int = 1
    module: str = "spectra.run"
    env: dict[str, str] = field(default_factory=dict)


def run_job(model: str, training: str, eval_class: str, *, dtype: str = "bfloat16",
            intervention: dict | None = None, step: int | None = None, extra: list[str] | None = None,
            tag: str = "") -> Job:
    args = ["--model-type", model, "--training", training, "--eval-class", eval_class, "--dtype", dtype]
    if intervention:
        args += ["--intervention", json.dumps(intervention)]
    if step is not None:
        args += ["--step", str(step)]
    args += extra or []
    parts = ["spectra", model, training, eval_class, dtype]
    if step is not None:
        parts.append(f"step{step}")
    if tag:
        parts.append(tag)
    return Job(name="-".join(parts), args=args)


MATCHED = [("periodic", "periodic"), ("solvable", "solvable"), ("aperiodic", "aperiodic")]
GDN_MODELS = ["gdn-full", "hybrid", "gdn-sdp", "hybrid-sdp"]
PM_MODELS = ["gdn-full", "hybrid"]


def suite(name: str) -> list[Job]:
    if name == "pilot":
        return [run_job("gdn-full", "periodic", "periodic",
                        extra=["--tiers", "tokens_under_2k", "tokens_64k", "--per-tier", "2"], tag="pilot")]
    if name == "metadata":
        return [Job(name="spectra-doc-metadata", args=[], gpus=0, module="spectra.dump_doc_metadata")]
    if name == "metadata-solvable":
        return [Job(name="spectra-doc-metadata-solvable", args=["--repos", "jacksonp-ai2/state-bench-solvable"],
                    gpus=0, module="spectra.dump_doc_metadata")]
    if name == "baseline":
        return [run_job(m, t, e) for m in GDN_MODELS for t, e in MATCHED]
    if name == "precision":
        return [run_job(m, t, e, dtype="float32") for m in PM_MODELS for t, e in MATCHED[:2]]
    if name == "interventions":
        interventions = {
            "snapreflect": {"snap_reflect": 0.02},
            "snapalpha": {"snap_alpha": 1e-3},
            "alphahalf": {"alpha_power": 0.5},
            "snapboth": {"snap_reflect": 0.02, "snap_alpha": 1e-3},
        }
        jobs = [
            run_job(m, t, e, intervention=iv, tag=tag)
            for m in PM_MODELS for t, e in MATCHED[:2] for tag, iv in interventions.items()
        ]
        jobs += [run_job("hybrid", t, e, intervention={"attn_window": 4096}, tag="attnwindow") for t, e in MATCHED[:2]]
        return jobs
    if name == "interventions-targeted":
        # Snap the memory head's inexact reflections (value tokens sit near beta ~ 1.89),
        # which the beta > 1.98 snap above never touches. Memory heads are the most
        # persistent heads in the baseline retention (figures.persistent_heads).
        memory_heads = {"gdn-full": [[1, 6]], "hybrid": [[5, 1]]}
        return [
            job
            for m in PM_MODELS
            for job in (
                run_job(m, "periodic", "periodic", intervention={"snap_reflect": 0.25, "heads": memory_heads[m]},
                        tag="snapreflect25-memhead"),
                run_job(m, "periodic", "periodic", intervention={"snap_reflect": 0.25}, tag="snapreflect25"),
            )
        ]
    if name == "training-steps":
        # Converted intermediate checkpoints on weka (spectra/convert.py): the two
        # pre-/post-decay pairs of the WSDS schedule plus one extra mid-run save.
        steps = {"gdn-full": (3695, 4106, 7000, 7800), "hybrid": (3627, 4030, 7656, 8000)}
        return [run_job(m, "periodic", "periodic", step=s) for m in PM_MODELS for s in steps[m]]
    raise ValueError(f"unknown suite {name!r}")


SUITES = ("pilot", "metadata", "metadata-solvable", "baseline", "precision", "interventions",
          "interventions-targeted", "training-steps")


def gantry_command(job: Job, *, clusters: list[str], priority: str) -> list[str]:
    command = [
        "gantry", "run",
        "--name", job.name,
        "--workspace", WORKSPACE,
        "--budget", BUDGET,
        "--priority", priority,
        "--beaker-image", IMAGE,
        "--gpus", str(job.gpus),
        "--weka", "oe-training-default:/weka/oe-training-default",
        "--weka", "oe-eval-default:/weka/oe-eval-default",
        "--gh-token-secret", "jacksonp_GITHUB_TOKEN",
        "--secret-env", "HF_TOKEN=jacksonp_HF_TOKEN",
        "--env", f"HF_HOME={HF_HOME}",
        "--env", "PYTORCH_ALLOC_CONF=expandable_segments:True",
        # This repo's submodules use SSH remotes the job cannot clone and the
        # analysis does not need: an unmatched submodule.active pathspec turns
        # gantry's `git submodule update --init` into a no-op.
        "--env", "GIT_CONFIG_COUNT=1",
        "--env", "GIT_CONFIG_KEY_0=submodule.active",
        "--env", "GIT_CONFIG_VALUE_0=__no_submodules__",
        "--no-python",
        "--install", INSTALL,
        "--results", "/results",
        "--no-logs",
        "--yes",
    ]
    for cluster in clusters:
        command += ["--cluster", cluster]
    for key, value in job.env.items():
        command += ["--env", f"{key}={value}"]
    return [*command, "--", PYTHON, "-m", job.module, *job.args]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", nargs="+", required=True, choices=SUITES)
    parser.add_argument("--cluster", nargs="+", default=list(CLUSTERS))
    parser.add_argument("--priority", default=PRIORITY)
    parser.add_argument("--only", nargs="*", default=None, help="Launch only jobs whose name contains one of these")
    parser.add_argument("--names", nargs="*", default=None, help="Launch only jobs with exactly these names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [job for name in args.suite for job in suite(name)]
    if args.only:
        jobs = [job for job in jobs if any(fragment in job.name for fragment in args.only)]
    if args.names:
        jobs = [job for job in jobs if job.name in set(args.names)]
    for job in jobs:
        command = gantry_command(job, clusters=args.cluster, priority=args.priority)
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(["uv", "run", *command], check=True, cwd=REPO_ROOT)
    print(f"\n{'Would launch' if args.dry_run else 'Launched'} {len(jobs)} job(s).")


if __name__ == "__main__":
    main()
