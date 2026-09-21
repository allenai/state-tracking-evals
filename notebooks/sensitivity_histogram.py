#!/usr/bin/env python3
"""Plot a histogram of per-function sensitivity from a scored JSONL file.

Ported from sensitivity-estimate/scripts/sensitivity_histogram.py so the
paper's figure code lives alongside the dt/pt results notebooks. Reads the
scored CruxEval results from the sensitivity-estimate submodule by default,
uses the top-level aesthetics module, and writes into the paper submodule's
figures directory (PDF only, matching the notebooks) unless --output
overrides it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aesthetics as aes  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "sensitivity-estimate/results/cruxeval-test.jsonl"
FIGURES_DIR = PROJECT_ROOT / "state-bench-paper" / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=(
            "Scored JSONL produced by `sensitivity-estimate cruxeval`. "
            f"Defaults to {DEFAULT_INPUT.relative_to(PROJECT_ROOT)}."
        ),
    )
    parser.add_argument("--metric", default="sensitivity_density")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output stem or filename; defaults to "
            f"{FIGURES_DIR.relative_to(PROJECT_ROOT)}/<metric>."
        ),
    )
    parser.add_argument("--title", default=None, help="Defaults to a metric-based title.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    values = [row[args.metric] for row in rows if row.get("completed") and args.metric in row]
    if not values:
        raise SystemExit(f"no completed records with {args.metric!r} in {args.input}")

    metric_label = args.metric.replace("_", " ").capitalize()
    title = args.title or f"CruxEval {metric_label.lower()}"
    figure, axis = aes.half_width_figure(title)
    sns.histplot(values, bins=args.bins, color=aes.NONSEMANTIC_COLOR, ax=axis)
    median = statistics.median(values)
    axis.axvline(
        median,
        linestyle="--",
        linewidth=1,
        color=aes.darken(aes.NONSEMANTIC_COLOR, by=0.4),
        label=f"median = {median:.2f}",
    )
    axis.set_xlabel(metric_label)
    axis.set_ylabel("Functions")
    axis.yaxis.set_major_formatter(aes.NICE_FORMATTER)
    if args.metric in {"sensitivity_density", "sensitivity_coverage"}:
        axis.xaxis.set_major_formatter(aes.PCT_FORMATTER)
    axis.legend(frameon=False, fontsize=7)
    sns.despine(fig=figure)

    output = args.output or FIGURES_DIR / args.metric
    pdf_path, _ = aes.save_figure(output, fig=figure, tight=False)
    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
