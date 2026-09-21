#!/usr/bin/env python3
"""Two-panel sensitivity histogram: StateBench periodic programs and CruxEval.

Combines notebooks/periodic_sensitivity_histograms.py (left panel, one series
per periodic dataset) with notebooks/sensitivity_histogram.py (right panel,
scored CruxEval functions) on one full-line-width canvas. Both panels are
frequency histograms over the same bins and share the y-axis, so the two
distributions can be read against each other directly. Writes into the paper
submodule's figures directory unless --output overrides it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aesthetics as aes  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "state-bench-paper" / "figures"
DEFAULT_OUTPUT = FIGURES_DIR / "sensitivity_histograms"
DEFAULT_STATE_BENCH_INPUT = (
    PROJECT_ROOT / "state-bench/data/state-tracking-long-context-v1/canonical/test.jsonl"
)
DEFAULT_CRUXEVAL_INPUT = PROJECT_ROOT / "sensitivity-estimate/results/cruxeval-test.jsonl"

# Same fixed canvas as the single-row dt/pt summary figures (SUMMARY_* in
# notebooks/dt_results.ipynb): 1.5x the line width by 2.0 in, included at
# \linewidth so text scales identically across those figures. Margins in inches.
FIG_WIDTH_IN = 1.5 * aes.ICLR_PAPER_WIDTH_IN
FIG_HEIGHT_IN = 2.0
MARGIN_LEFT_IN = 0.75
MARGIN_RIGHT_IN = 0.1
MARGIN_TOP_IN = 0.6
MARGIN_BOTTOM_IN = 0.5
BINS = np.linspace(0.0, 1.0, 33)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-bench",
        type=Path,
        nargs="+",
        default=[DEFAULT_STATE_BENCH_INPUT],
        help=(
            "Canonical state-bench JSONL inputs; only periodic records are used. Labels are "
            "inferred from the dataset directory names unless --labels is given."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Legend labels for the state-bench inputs, in the same order.",
    )
    parser.add_argument(
        "--cruxeval",
        type=Path,
        default=DEFAULT_CRUXEVAL_INPUT,
        help="Scored JSONL produced by `sensitivity-estimate cruxeval`.",
    )
    parser.add_argument(
        "--metric",
        default="sensitivity_coverage",
        choices=("sensitivity_coverage", "sensitivity_density"),
        help=(
            "Record field plotted on both panels (default: sensitivity_coverage). Coverage is "
            "the fraction of live variables per state; density further divides by the number "
            "of queried outputs, which for StateBench mostly reflects the variable count."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path stem or file. Defaults to {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)}.",
    )
    return parser.parse_args(argv)


def read_state_bench(path: Path, metric: str) -> list[float]:
    values: list[float] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] != "periodic":
                continue
            values.append(float(record[metric]))
    if not values:
        raise ValueError(f"no periodic records found in {path}")
    return values


def read_cruxeval(path: Path, metric: str) -> list[float]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    values = [float(row[metric]) for row in rows if row.get("completed") and metric in row]
    if not values:
        raise ValueError(f"no completed records with {metric!r} in {path}")
    return values


def infer_label(path: Path) -> str:
    """Infer an aperiodic-mixture label from a periodic dataset directory."""

    dataset_name = path.parents[1].name
    if dataset_name.endswith("-pure"):
        return "0% aperiodic mix"
    match = re.search(r"mixed-(\d+)pct", dataset_name)
    if match:
        return f"{match.group(1)}% aperiodic mix"
    return dataset_name


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    labels = args.labels or [infer_label(path) for path in args.state_bench]
    if len(labels) != len(args.state_bench):
        raise ValueError("--labels must provide exactly one label for each state-bench input")
    if len(set(labels)) != len(labels):
        raise ValueError("state-bench labels must be unique")
    metric_label = args.metric.replace("_", " ").capitalize()

    figure = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    left = MARGIN_LEFT_IN / FIG_WIDTH_IN
    grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        left=left,
        right=1 - MARGIN_RIGHT_IN / FIG_WIDTH_IN,
        top=1 - MARGIN_TOP_IN / FIG_HEIGHT_IN,
        bottom=MARGIN_BOTTOM_IN / FIG_HEIGHT_IN,
        wspace=0.12,
    )
    state_bench_axis = figure.add_subplot(grid[0, 0])
    cruxeval_axis = figure.add_subplot(grid[0, 1], sharey=state_bench_axis)

    palette = dict(zip(labels, sns.color_palette("deep", n_colors=len(labels)), strict=True))
    for path, label in zip(args.state_bench, labels, strict=True):
        sns.histplot(
            x=read_state_bench(path, args.metric),
            bins=BINS,
            stat="proportion",
            element="bars",
            fill=True,
            alpha=0.4,
            color=palette[label],
            label=label,
            ax=state_bench_axis,
        )
    state_bench_axis.legend(title="Aperiodic mix", frameon=False, fontsize=7, title_fontsize=7)
    state_bench_axis.set_title("StateBench periodic programs", fontsize=8, fontweight="regular")

    cruxeval_values = read_cruxeval(args.cruxeval, args.metric)
    sns.histplot(
        x=cruxeval_values,
        bins=BINS,
        stat="proportion",
        color=aes.NONSEMANTIC_COLOR,
        ax=cruxeval_axis,
    )
    median = statistics.median(cruxeval_values)
    cruxeval_axis.axvline(
        median,
        linestyle="--",
        linewidth=1,
        color=aes.darken(aes.NONSEMANTIC_COLOR, by=0.4),
        label=f"median = {median:.2f}",
    )
    cruxeval_axis.legend(frameon=False, fontsize=7)
    cruxeval_axis.set_title("CruxEval functions", fontsize=8, fontweight="regular")

    for axis in (state_bench_axis, cruxeval_axis):
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks(np.linspace(0.0, 1.0, 5))
        axis.xaxis.set_major_formatter(aes.PCT_FORMATTER)
        axis.set_xlabel(metric_label)
        axis.set_ylabel("")
    state_bench_axis.set_ylabel("Proportion")
    cruxeval_axis.tick_params(labelleft=False)
    # The panels sit flush; drop the right panel's leading 0% so it does not
    # collide with the left panel's 100%.
    cruxeval_axis.set_xticklabels(["", "25%", "50%", "75%", "100%"])
    sns.despine(fig=figure)

    figure.suptitle(
        f"{metric_label}: StateBench vs. CruxEval",
        x=left,
        y=0.96,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
    pdf_path, _ = aes.save_figure(args.output, fig=figure, tight=False)
    plt.close(figure)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
