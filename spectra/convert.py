"""Convert intermediate DT checkpoints to HF format on Beaker.

Runs olmo-core's ``src/scripts/train/ladder/convert_state_bench_checkpoints.py``
from the olmo-core submodule (gantry clones its pinned, pushed commit, as in
scripts/train_dt.sh). The script skips anything already converted and
validates logit parity, so re-running is safe. ``--filter`` takes path
substrings relative to the ladder root, e.g. ``gdn-full/integer-code--periodic/``
(the trailing slash keeps ``--periodic--sens-*`` runs out).

    uv run python -m spectra.convert --dry-run
    uv run python -m spectra.convert
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

OLMO_CORE = Path(__file__).resolve().parent.parent / "olmo-core"
SCRIPT = "src/scripts/train/ladder/convert_state_bench_checkpoints.py"
IMAGE = "petew/olmo-core-tch2100cu128-2026-01-23"
TRANSFORMERS_FORK = "transformers @ git+https://github.com/jopetty/transformers.git@olmo-3.5-hybrid-state-bench"
# The runs whose intermediate checkpoints spectra's training-steps suite reads.
DEFAULT_FILTERS = ("gdn-full/integer-code--periodic/", "hybrid/integer-code--periodic/")

INSTALL = " && ".join([
    "export UV_CACHE_DIR=/tmp/uv-cache",
    "command -v uv >/dev/null || pip install -q uv",
    # Constrain to the image's torch/CUDA/triton so the extras can't replace them.
    "pip freeze | grep -iE '^(torch|nvidia-|triton)' > /tmp/cuda-constraints.txt",
    "uv pip install --system -c /tmp/cuda-constraints.txt -e '.[fla]'",
    f"uv pip install --system -c /tmp/cuda-constraints.txt '{TRANSFORMERS_FORK}'",
])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", nargs="+", default=list(DEFAULT_FILTERS))
    parser.add_argument("--cluster", nargs="+", default=["ai2/holmes", "ai2/titan", "ai2/saturn", "ai2/jupiter"])
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--convert-dry-run", action="store_true", help="Pass --dry-run to the converter in the job")
    parser.add_argument("--dry-run", action="store_true", help="Print the gantry command only")
    args = parser.parse_args()

    converter = ["python", SCRIPT]
    for fragment in args.filter:
        converter += ["--filter", fragment]
    if args.convert_dry_run:
        converter.append("--dry-run")
    command = [
        "gantry", "run",
        "--name", "spectra-convert-checkpoints",
        "--workspace", "ai2/beyond-state",
        "--budget", "ai2/oe-other",
        "--priority", args.priority,
        "--beaker-image", IMAGE,
        "--gpus", "1",
        "--weka", "oe-training-default:/weka/oe-training-default",
        "--gh-token-secret", "jacksonp_GITHUB_TOKEN",
        "--no-python",
        "--install", INSTALL,
        "--no-logs",
        "--yes",
    ]
    for cluster in args.cluster:
        command += ["--cluster", cluster]
    command += ["--", *converter]
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(["uv", "run", "--project", str(OLMO_CORE.parent), *command], check=True, cwd=OLMO_CORE)


if __name__ == "__main__":
    main()
