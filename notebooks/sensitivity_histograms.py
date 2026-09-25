#!/usr/bin/env python3
"""Five-panel sensitivity histogram: each StateBench class and CruxEval code.

One panel per StateBench complexity class (R-trivial, Aperiodic, Soluble,
General finite), read from a canonical state-bench test split, followed by a panel of
scored CruxEval functions (notebooks/sensitivity_histogram.py). All panels are
frequency histograms over the same bins and share the y-axis, so the
distributions can be read against each other directly, and each marks its
median. The default input is the state-tracking-solvable-v2 test build, whose
four classes are derived from the same periodic sources, so every panel
describes the same programs. Writes into the paper submodule's figures
directory unless --output overrides it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aesthetics as aes  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "state-bench-paper" / "figures"
DEFAULT_OUTPUT = FIGURES_DIR / "sensitivity_histograms"
DEFAULT_STATE_BENCH_INPUT = (
    PROJECT_ROOT / "state-bench/data/state-tracking-solvable-v2/canonical/test.jsonl"
)
DEFAULT_CRUXEVAL_INPUT = PROJECT_ROOT / "sensitivity-estimate/results/cruxeval-test.jsonl"


def xcolor_tint(base_hex: str, percent: float, *, toward: str = "FFFFFF") -> tuple[float, float, float]:
    """xcolor's ``base!percent!toward`` mix, e.g. ``stbase!80`` (toward white)."""

    base = [int(base_hex[i : i + 2], 16) for i in (0, 2, 4)]
    other = [int(toward[i : i + 2], 16) for i in (0, 2, 4)]
    return tuple(  # type: ignore[return-value]
        (percent / 100 * b + (1 - percent / 100) * o) / 255 for b, o in zip(base, other)
    )


# The paper's hierarchy diagram (state-bench-paper/tikz-figures/hierarchy.tex)
# tints one base color per nested class shell. Bars take each shell's border
# tint (its fills run down to stbase!5, too pale for bars), and bar outlines
# and medians the diagram's label color, stbase!85!black, so the palest panel
# stays legible.
STBASE_HEX = "F28F3B"
STATE_BENCH_EDGE_COLOR = xcolor_tint(STBASE_HEX, 85, toward="000000")
# Panels in increasing group complexity, labelled as in dt_results.
STATE_BENCH_PANELS = {
    "r-trivial": ("R-trivial", xcolor_tint(STBASE_HEX, 80)),
    "aperiodic": ("Aperiodic", xcolor_tint(STBASE_HEX, 60)),
    "solvable": ("Soluble", xcolor_tint(STBASE_HEX, 40)),
    "periodic": ("General finite", xcolor_tint(STBASE_HEX, 25)),
}
CRUXEVAL_LABEL = "CruxEval (code)"
# The same treatment in seaborn's deep blue: a lightened fill, outlined and
# marked at the median in the blue darkened as stbase!85!black is.
CRUXEVAL_HEX = "4C72B0"
CRUXEVAL_COLOR = xcolor_tint(CRUXEVAL_HEX, 70)
CRUXEVAL_EDGE_COLOR = xcolor_tint(CRUXEVAL_HEX, 85, toward="000000")

# Same fixed canvas as the single-row dt/pt summary figures (SUMMARY_* in
# notebooks/dt_results.ipynb): 1.5x the line width by 1.75 in, included at
# \linewidth so text scales identically across those figures. Margins in inches,
# sized to this figure's labels: the left margin fits the y label and ticks, and
# the right margin leaves room for the last panel's centred "100%" tick label.
FIG_WIDTH_IN = 1.5 * aes.ICLR_PAPER_WIDTH_IN
FIG_HEIGHT_IN = 1.75
MARGIN_LEFT_IN = 0.61
MARGIN_RIGHT_IN = 0.21
MARGIN_TOP_IN = 0.35
MARGIN_BOTTOM_IN = 0.5
# 16 bins of 1/16 each keep the bars wide enough to read at this panel size.
BINS = np.linspace(0.0, 1.0, 17)
# Shared y-axis ceiling, just above the tallest bar outside the R-trivial and
# Aperiodic spikes at zero coverage. Taller bars are clipped at the ceiling,
# marked with a break, and labelled with their true height.
Y_MAX = 0.30
# Whole percentages on both axes; the shared PCT_FORMATTER picks a decimal for
# proportions this small.
WHOLE_PERCENT = mtick.PercentFormatter(1.0, decimals=0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-bench",
        type=Path,
        default=DEFAULT_STATE_BENCH_INPUT,
        help=(
            "Canonical state-bench JSONL holding every complexity class (default: the "
            "state-tracking-solvable-v2 test split; build it with `state-bench build "
            "configs/experiments/solvable.yaml --splits test`)."
        ),
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
            "Record field plotted on every panel (default: sensitivity_coverage). Coverage is "
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


def read_state_bench(path: Path, metric: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] in STATE_BENCH_PANELS:
                values[record["complexity"]].append(float(record[metric]))
    missing = [complexity for complexity in STATE_BENCH_PANELS if not values[complexity]]
    if missing:
        raise ValueError(f"no {', '.join(missing)} records found in {path}")
    return values


def read_cruxeval(path: Path, metric: str) -> list[float]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    values = [float(row[metric]) for row in rows if row.get("completed") and metric in row]
    if not values:
        raise ValueError(f"no completed records with {metric!r} in {path}")
    return values


def clip_bar(axis: plt.Axes, bar: object, text_color: object) -> None:
    """Clip a bar at Y_MAX with a white break mark and label its true height."""

    height = bar.get_height()
    bar.set_height(Y_MAX)
    left, width = bar.get_x(), bar.get_width()
    for offset in (0.06, 0.1):
        y = Y_MAX * (1 - offset)
        axis.plot(
            [left, left + width],
            [y - 0.012, y + 0.012],
            color="white",
            linewidth=1.2,
            solid_capstyle="butt",
            zorder=3,
        )
    axis.annotate(
        f"{height:.0%}",
        xy=(left + width, Y_MAX),
        xytext=(2, -1),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6,
        color=text_color,
    )


def plot_panel(
    axis: plt.Axes,
    values: list[float],
    title: str,
    color: object,
    edge_color: object | None = None,
) -> None:
    """Histogram one panel; ``edge_color`` outlines the bars and colors the median."""

    sns.histplot(
        x=values,
        bins=BINS,
        stat="proportion",
        color=color,
        alpha=1.0,
        **({"edgecolor": edge_color, "linewidth": 0.4} if edge_color is not None else {}),
        ax=axis,
    )
    clipped = [bar for bar in axis.patches if bar.get_height() > Y_MAX]
    for bar in clipped:
        clip_bar(axis, bar, edge_color if edge_color is not None else color)
    median = statistics.median(values)
    median_color = edge_color if edge_color is not None else aes.darken(color, by=0.4)
    axis.axvline(median, linestyle="--", linewidth=1, color=median_color)
    # Label on the side of the median with more room, clear of the line.
    left_of_line = median > 0.5
    axis.annotate(
        f"median {median:.2f}",
        # One line lower when a clipped bar's label occupies the top line.
        xy=(median, 0.83 if clipped else 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(-3 if left_of_line else 3, 0),
        textcoords="offset points",
        ha="right" if left_of_line else "left",
        va="top",
        fontsize=6,
    )
    axis.set_title(title, fontsize=8, fontweight="regular")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metric_label = args.metric.replace("_", " ").capitalize()
    state_bench_values = read_state_bench(args.state_bench, args.metric)
    panels = [
        (label, state_bench_values[complexity], color, STATE_BENCH_EDGE_COLOR)
        for complexity, (label, color) in STATE_BENCH_PANELS.items()
    ]
    panels.append(
        (
            CRUXEVAL_LABEL,
            read_cruxeval(args.cruxeval, args.metric),
            CRUXEVAL_COLOR,
            CRUXEVAL_EDGE_COLOR,
        )
    )

    figure = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    grid = figure.add_gridspec(
        nrows=1,
        ncols=len(panels),
        left=MARGIN_LEFT_IN / FIG_WIDTH_IN,
        right=1 - MARGIN_RIGHT_IN / FIG_WIDTH_IN,
        top=1 - MARGIN_TOP_IN / FIG_HEIGHT_IN,
        bottom=MARGIN_BOTTOM_IN / FIG_HEIGHT_IN,
        wspace=0.12,
    )
    axes: list[plt.Axes] = []
    for index, (title, values, color, edge_color) in enumerate(panels):
        axis = figure.add_subplot(grid[0, index], sharey=axes[0] if axes else None)
        axes.append(axis)
        plot_panel(axis, values, title, color, edge_color)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_xticks([0.25, 0.75], minor=True)
        axis.xaxis.set_major_formatter(WHOLE_PERCENT)
        axis.set_xlabel(metric_label if index == 0 else "")
        axis.set_ylabel("Proportion" if index == 0 else "")
        axis.tick_params(labelleft=index == 0)
        if index:
            # The panels sit flush; drop the leading 0% so it does not collide
            # with the previous panel's 100%.
            axis.set_xticklabels(["", "50%", "100%"])
    axes[0].set_ylim(0.0, Y_MAX)
    axes[0].set_yticks([0.0, 0.1, 0.2, 0.3])
    axes[0].yaxis.set_major_formatter(WHOLE_PERCENT)
    sns.despine(fig=figure)

    pdf_path, _ = aes.save_figure(args.output, fig=figure, tight=False)
    plt.close(figure)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
