#!/usr/bin/env python3
"""Fetch state-bench evals on pre-trained models.

Usage:
    uv run scripts/fetch_pt.py --completed-only
    uv run scripts/fetch_pt.py -v  # show per-file download progress
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_results import main  # noqa: E402

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
