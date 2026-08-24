#!/usr/bin/env python3
"""Fetch state-bench evals on directly-trained models.

Usage:
    uv run scripts/fetch_dt.py --completed-only
    uv run scripts/fetch_dt.py -v  # show per-file download progress
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_results import main  # noqa: E402


# Fetch every directly-trained evaluation group from the workspace that owns it.
BEAKER = [
    {
        "workspace": "ai2/beyond-state",
        "group": "jacksonp-state-bench-dt-275m",
    },
    # {
    #     "workspace": "ai2/linear-rnns",
    #     "group": "jacksonp-state-bench-dt-2",
    # },
]


if __name__ == "__main__":
    for beaker in BEAKER:
        main(
            default_group=beaker["group"],
            default_workspace=beaker["workspace"],
            description="Fetch directly-trained synthetic state-bench results.",
        )
