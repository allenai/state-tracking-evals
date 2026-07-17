#!/usr/bin/env python3
"""Fetch state-bench length-comparison results from Beaker.

Usage:
    uv run fetch_sb_lc.py --completed-only
"""

from fetch_results import main


GROUP = "jacksonp-state-bench-lc-2"
WORKSPACE = "ai2/beyond-state"


if __name__ == "__main__":
    main(
        default_group=GROUP,
        default_workspace=WORKSPACE,
        description="Fetch state-bench length-comparison results.",
    )
