#!/usr/bin/env python3
"""Fetch state-bench evals on pre-trained models.

Usage:
    uv run fetch_sb_lc.py --completed-only
    uv run fetch_sb_lc.py -v  # show per-file download progress
"""

from fetch_results import main

BEAKER = [
    {"workspace": "ai2/beyond-state", "group": "jacksonp-state-bench-lc-10"},
    # {"workspace": "ai2/linear-rnns", "group": "jacksonp-state-bench-lc-8.1"}
]

if __name__ == "__main__":

    for beaker in BEAKER:
        main(
            default_group=beaker["group"],
            default_workspace=beaker["workspace"],
            description="Fetch state-bench length-comparison results.",
        )
