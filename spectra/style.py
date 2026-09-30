"""Figure conventions shared by the spectra plots.

These mirror ``notebooks/dt_results.ipynb`` (series colors and markers from
``aesthetics``, Wilson bands, the training-cutoff reference lines, the
fixed-canvas layout with a legend band below the panels) so spectra figures
sit next to the paper's DT figures without restyling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from statsmodels.stats.proportion import proportion_confint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aesthetics as aes  # noqa: E402

FIGURE_DIR = PROJECT_ROOT / "figures" / "spectra"

SERIES_ORDER = list(aes.STATE_BENCH_MODEL_ORDER)
SERIES_COLORS = {series: aes.state_bench_model_color(series) for series in SERIES_ORDER}
SERIES_MARKERS = dict(aes.STATE_BENCH_MODEL_MARKERS)
SERIES_LABELS = dict(aes.STATE_BENCH_MODEL_LABELS)
DATASET_LABEL = {
    "r-trivial": "R-trivial",
    "aperiodic": "Aperiodic",
    "solvable": "Soluble",
    "periodic": "General finite",
}

ASSIGNMENT_REFERENCE = 256
ASSIGNMENT_REFERENCE_COLOR = "#2CA02C"
ASSIGNMENT_EXTRAPOLATION_MULTIPLE = 4
ASSIGNMENT_EXTRAPOLATION_COLOR = "#008B8B"
# Directly-trained sequences hold about 4 tokens per assignment; the token-axis
# cutoff marks the longest training prompt rather than 256 x 4.
TOKEN_REFERENCE = 1024

# Fixed-canvas geometry of the notebook's single-row summary figures.
SUMMARY_FIG_WIDTH_IN = 1.5 * aes.ICLR_PAPER_WIDTH_IN
SUMMARY_FIG_HEIGHT_IN = 1.75
MARGINS_IN = {"left_in": 0.75, "right_in": 0.25, "top_in": 0.35, "bottom_in": 0.5}


def legend_figure(width_in, height_in, legend_handles, *, left_in, right_in, top_in, bottom_in,
                  nrows=1, ncols=1, **grid_kwargs):
    """Fixed-canvas figure with a legend band below the panels (as in dt_results)."""
    layout = aes.bottom_legend_layout(legend_handles, width_in - left_in - right_in)
    total_height_in = height_in + layout.band_in
    fig = plt.figure(figsize=(width_in, total_height_in))
    left, right = left_in / width_in, 1 - right_in / width_in
    grid = fig.add_gridspec(
        nrows=nrows, ncols=ncols, left=left, right=right,
        top=1 - top_in / total_height_in, bottom=(bottom_in + layout.band_in) / total_height_in, **grid_kwargs,
    )
    aes.add_bottom_legend(fig, legend_handles, layout, x_center=(left + right) / 2)
    return fig, grid


def row_figure(ncols, legend_handles, *, height_in=SUMMARY_FIG_HEIGHT_IN, width_in=SUMMARY_FIG_WIDTH_IN,
               nrows=1, **grid_kwargs):
    grid_kwargs.setdefault("wspace", 0.1)
    return legend_figure(width_in, height_in * nrows, legend_handles, nrows=nrows, ncols=ncols,
                         **MARGINS_IN, **grid_kwargs)


def series_handle(series, **kwargs):
    return plt.Line2D([0], [0], color=SERIES_COLORS[series], marker=SERIES_MARKERS[series], linewidth=1.5,
                      label=SERIES_LABELS[series], **kwargs)


def chance_handle():
    return plt.Line2D([0], [0], color=aes.CHANCE_LINE_COLOR, linestyle="--", linewidth=1.25,
                      label="chance accuracy")


def reference_handles(*, extrapolation=True):
    handles = [plt.Line2D([0], [0], color=ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.45,
                          label="training cutoff")]
    if extrapolation:
        handles.append(plt.Line2D([0], [0], color=ASSIGNMENT_EXTRAPOLATION_COLOR, linestyle=":", linewidth=1.45,
                                  label=f"{ASSIGNMENT_EXTRAPOLATION_MULTIPLE}× training cutoff"))
    return handles


def add_reference_lines(ax, reference=ASSIGNMENT_REFERENCE, *, extrapolation=True):
    ax.axvline(reference, color=ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.6)
    if extrapolation:
        ax.axvline(ASSIGNMENT_EXTRAPOLATION_MULTIPLE * reference, color=ASSIGNMENT_EXTRAPOLATION_COLOR,
                   linestyle=":", linewidth=1.6)


def configure_log_axis(ax, lo=None, hi=None):
    ax.set_xscale("log", base=10)
    if lo is not None or hi is not None:
        ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(mtick.LogLocator(base=10, numticks=8))
    ax.xaxis.set_major_formatter(aes.COMPACT_NUMBER_FORMATTER)
    ax.xaxis.set_minor_locator(mtick.LogLocator(base=10, subs=tuple(range(1, 10)), numticks=100))
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())


def configure_accuracy_axis(ax, *, first):
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.yaxis.set_major_formatter(aes.PCT_FORMATTER)
    ax.set_ylabel("Accuracy" if first else "")
    ax.tick_params(labelleft=first)


def panel_title(ax, text):
    ax.set_title(text, fontsize=8, fontweight="regular")


def wilson(successes, n):
    low, high = proportion_confint(successes, n, alpha=0.05, method="wilson")
    return np.asarray(low), np.asarray(high)


def plot_band(ax, x, y, low, high, *, color, marker="o", linestyle="-", label=None, zorder=2):
    ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0, zorder=zorder - 1)
    ax.plot(x, y, marker=marker, ms=3, lw=1.2, color=color, linestyle=linestyle, label=label, zorder=zorder)


def power_of_two_bins(values, lo=8):
    """Assignment-count bins as in dt_results: [2^k, 2^(k+1)), plotted at the geometric mid."""
    values = np.asarray(values, dtype=float)
    top = int(np.floor(np.log2(max(values.max(), lo)))) + 1
    edges = 2.0 ** np.arange(int(np.floor(np.log2(lo))), top + 1)
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    mids = np.sqrt(edges[:-1] * edges[1:])
    return index, mids, edges


def save(fig, name):
    return aes.save_figure(FIGURE_DIR / name, fig=fig, tight=False, save_png=True)
