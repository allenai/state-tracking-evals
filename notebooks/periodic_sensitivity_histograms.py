#!/usr/bin/env python3
"""Compare periodic-program sensitivity-coverage distributions across mixtures.

Ported from state-bench/scripts/periodic_sensitivity_histograms.py so the
paper's figure code lives alongside the dt/pt results notebooks. Uses the
top-level aesthetics module and writes into the paper submodule's figures
directory (PDF only, matching the notebooks) unless --output overrides it.
"""

from __future__ import annotations

import argparse
import json
import re
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
DEFAULT_OUTPUT = FIGURES_DIR / "periodic_sensitivity_histograms"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Canonical JSONL inputs; labels are inferred from their dataset directory names.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels, in the same order as the canonical JSONL inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Destination path stem or file. "
            f"Defaults to {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)}."
        ),
    )
    return parser.parse_args(argv)


def read_coverages(path: Path) -> list[float]:
    coverages: list[float] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["complexity"] != "periodic":
                continue
            coverages.append(float(record["sensitivity_coverage"]))
    if not coverages:
        raise ValueError(f"no periodic records found in {path}")
    return coverages


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
    labels = args.labels or [infer_label(path) for path in args.inputs]
    if len(labels) != len(args.inputs):
        raise ValueError("--labels must provide exactly one label for each input")
    if len(set(labels)) != len(labels):
        raise ValueError("dataset labels must be unique")

    palette = dict(zip(labels, sns.color_palette("deep", n_colors=len(labels)), strict=True))
    figure, axis = aes.half_width_figure("Periodic sensitivity coverage")
    bins = np.linspace(0.0, 1.0, 33)
    for path, label in zip(args.inputs, labels, strict=True):
        sns.histplot(
            x=read_coverages(path),
            bins=bins,
            stat="proportion",
            element="bars",
            fill=True,
            alpha=0.4,
            color=palette[label],
            label=label,
            ax=axis,
        )
    axis.set(
        xlabel="Sensitivity coverage",
        ylabel="Proportion of sequences",
        xlim=(0.0, 1.0),
        xticks=np.linspace(0.0, 1.0, 6),
    )
    axis.xaxis.set_major_formatter(aes.PCT_FORMATTER)
    axis.legend(title="Aperiodic mix", frameon=False, fontsize=7, title_fontsize=7)
    sns.despine(fig=figure)
    aes.save_figure(args.output, fig=figure, tight=False)
    plt.close(figure)


if __name__ == "__main__":
    main()
