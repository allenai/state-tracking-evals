#!/usr/bin/env python3
"""Fetch state-bench evals on directly-trained models.

Usage:
    uv run fetch_dt_results.py --completed-only
    uv run fetch_dt_results.py -v  # show per-file download progress
"""

from fetch_results import main


# Fetch every directly-trained evaluation group from the workspace that owns it.
BEAKER = [
    {
        "workspace": "ai2/beyond-state",
        "group": "jacksonp-state-bench-dt-11",
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
