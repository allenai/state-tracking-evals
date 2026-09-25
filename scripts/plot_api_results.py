"""Plots API-model state-bench accuracy against context length.

    uv run python scripts/plot_api_results.py
    uv run python scripts/plot_api_results.py --models gpt-4.1-mini gpt-5-mini__low
    uv run python scripts/plot_api_results.py --include-mc
    uv run python scripts/plot_api_results.py --push

Reads every <model>/predictions.jsonl that scripts/eval_api.py wrote under
state-bench/data/api-evals/ and draws the API counterpart of the pt_results
aggregated context-length summary (pt_accuracy_by_context_length_aggregated):
one panel per complexity class, one line per model, points at each tier's
median prompt length with Wilson 95% bands. Prompt length is the dataset's
Dolma token_length, the same tokenizer the pt figure measures with, so the
x-axes of the two figures line up regardless of each API model's tokenizer.
Errored requests (e.g. over the model's context window) are left out and
counted. Accuracy is exact final-state match for free-form runs and the chosen
letter for multiple-choice (`__mc`) runs; the dashed chance line is drawn only
for multiple-choice runs, since guessing a full free-form state is ~0.
Multiple-choice runs are left out unless --include-mc is passed or they are
named in --models.

--push publishes a changed figure the way scripts/render_figures.sh --push
does: commit it in state-bench-paper, rebase onto its origin/main (Overleaf
syncs land there), push, then bump the submodule pointer in this repo and push
that too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import aesthetics as aes  # noqa: E402
from scripts.eval_api import COMPLEXITIES, DEFAULT_OUTPUT_ROOT, TIERS  # noqa: E402

# Written next to the notebooks' paper figures, as a PDF only like theirs.
DEFAULT_FIGURE = (
    REPO_ROOT
    / "state-bench-paper"
    / "figures"
    / "api_accuracy_by_context_length_aggregated"
)
PAPER_DIR = REPO_ROOT / "state-bench-paper"
MIN_BIN_EXAMPLES = 3
# Display labels for complexity classes, matching pt_results.
COMPLEXITY_LABEL = {
    "r-trivial": "R-trivial",
    "aperiodic": "Aperiodic",
    "solvable": "Soluble",
    "periodic": "General finite",
}

# Same fixed-canvas geometry as pt_results' paper_figure.
FIG_WIDTH_IN = 1.5 * aes.ICLR_PAPER_WIDTH_IN
MARGIN_LEFT_IN = 0.75
MARGIN_RIGHT_IN = 0.15
MARGIN_TOP_IN = 0.35
MARGIN_BOTTOM_IN = 0.5
PANEL_HEIGHT_IN = 1.75


def load_predictions(
    root: Path, models: list[str] | None, include_mc: bool
) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/predictions.jsonl")):
        name = path.parent.name
        if models and name not in models:
            continue
        if not models and not include_mc and name.endswith("__mc"):
            continue
        rows = [json.loads(line) for line in path.open()]
        if rows:
            frames.append(pd.DataFrame(rows).assign(model=name))
    if not frames:
        raise SystemExit(f"No predictions under {root}")
    return pd.concat(frames, ignore_index=True)


def wilson_interval(successes: pd.Series, n: pd.Series) -> tuple[pd.Series, pd.Series]:
    z = 1.96
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return center - radius, center + radius


def summarize(items: pd.DataFrame) -> pd.DataFrame:
    present_tiers = [tier for tier in TIERS if tier in set(items["tier"])]
    items = items.assign(
        tier=pd.Categorical(items["tier"], present_tiers, ordered=True)
    )
    # Each tier plots at its median prompt length over every model's rows, so
    # all models share x positions within a tier.
    tier_mid = items.groupby("tier", observed=True)["token_length"].median()
    items = items.assign(length_mid=items["tier"].map(tier_mid).astype(float))
    summary = (
        items.groupby(
            ["complexity", "model", "tier", "length_mid"], observed=True, as_index=False
        )
        .agg(
            accuracy=("score", "mean"),
            successes=("score", "sum"),
            n=("score", "size"),
            chance=("chance", "mean"),
        )
        .query("n >= @MIN_BIN_EXAMPLES")
        .copy()
    )
    summary["ci_low"], summary["ci_high"] = wilson_interval(
        summary["successes"], summary["n"]
    )
    return summary


def configure_length_axis(ax: plt.Axes, mids: pd.Series) -> None:
    ax.set_xscale("log", base=10)
    ax.set_xlim(mids.min() / 1.4, mids.max() * 1.4)
    ax.xaxis.set_major_locator(mtick.LogLocator(base=10, numticks=8))
    ax.xaxis.set_major_formatter(aes.COMPACT_NUMBER_FORMATTER)
    ax.xaxis.set_minor_locator(
        mtick.LogLocator(base=10, subs=(1, 2, 3, 4, 5, 6, 7, 8, 9), numticks=100)
    )
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())


def plot(
    summary: pd.DataFrame, models: list[str], complexities: list[str]
) -> plt.Figure:
    palette = sns.color_palette("colorblind", len(models))
    colors = dict(zip(models, palette, strict=True))
    handles = [
        plt.Line2D([0], [0], color=colors[model], marker="s", label=model)
        for model in models
    ]

    panel_right = 1 - MARGIN_RIGHT_IN / FIG_WIDTH_IN
    panel_left = MARGIN_LEFT_IN / FIG_WIDTH_IN
    layout = aes.bottom_legend_layout(
        handles, FIG_WIDTH_IN - MARGIN_LEFT_IN - MARGIN_RIGHT_IN
    )
    height_in = PANEL_HEIGHT_IN + layout.band_in
    fig = plt.figure(figsize=(FIG_WIDTH_IN, height_in))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=len(complexities),
        wspace=0.1,
        left=panel_left,
        right=panel_right,
        top=1 - MARGIN_TOP_IN / height_in,
        bottom=(MARGIN_BOTTOM_IN + layout.band_in) / height_in,
    )
    axes = []
    for col in range(len(complexities)):
        reference = axes[0] if axes else None
        axes.append(fig.add_subplot(grid[0, col], sharex=reference, sharey=reference))
    aes.add_bottom_legend(fig, handles, layout, x_center=(panel_left + panel_right) / 2)

    for ax, complexity in zip(axes, complexities, strict=True):
        subset = summary[summary["complexity"] == complexity]
        for model in models:
            curve = subset[subset["model"] == model].sort_values("length_mid")
            x = curve["length_mid"].to_numpy(float)
            ax.plot(
                x, curve["accuracy"], marker="s", ms=2.5, lw=1.2, color=colors[model]
            )
            ax.fill_between(
                x, curve["ci_low"], curve["ci_high"], color=colors[model], alpha=0.12
            )
        choice_rows = subset[subset["chance"] > 0]
        if len(choice_rows):
            chance = (choice_rows["chance"] * choice_rows["n"]).sum() / choice_rows[
                "n"
            ].sum()
            ax.axhline(
                chance, color=aes.CHANCE_LINE_COLOR, linestyle="--", linewidth=1.25
            )
        configure_length_axis(ax, summary["length_mid"])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.yaxis.set_major_formatter(aes.PCT_FORMATTER)
        ax.set_title(COMPLEXITY_LABEL[complexity], fontsize=9, fontweight="normal")
        ax.label_outer()
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Input length (tokens, log scale)")
    return fig


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def push_figures(paths: list[Path]) -> None:
    """Commit and push changed figures in the paper, then bump its pointer here."""
    relative = []
    for path in paths:
        if not path.resolve().is_relative_to(PAPER_DIR):
            raise SystemExit(
                f"--push only publishes figures inside {PAPER_DIR}: {path}"
            )
        relative.append(str(path.resolve().relative_to(PAPER_DIR)))
    changed = [
        path
        for path in relative
        if git("status", "--porcelain", "--", path, cwd=PAPER_DIR)
    ]
    if not changed:
        print(">> No figure changed; nothing to push.")
        return
    git("fetch", "-q", "origin", cwd=PAPER_DIR)
    git("add", "--", *changed, cwd=PAPER_DIR)
    git(
        "commit",
        "-q",
        "-m",
        "Regenerate API model figures",
        "--",
        *changed,
        cwd=PAPER_DIR,
    )
    git("rebase", "-q", "origin/main", cwd=PAPER_DIR)
    git("push", "-q", "origin", "main", cwd=PAPER_DIR)
    print(f">> state-bench-paper: {git('log', '--oneline', '-1', cwd=PAPER_DIR)}")

    # Only the pointer is committed here, whatever else happens to be staged.
    git(
        "commit",
        "-q",
        "-m",
        "Bump state-bench-paper for regenerated API figures",
        "--",
        "state-bench-paper",
        cwd=REPO_ROOT,
    )
    git("push", "-q", "origin", "main", cwd=REPO_ROOT)
    print(f">> {REPO_ROOT.name}: {git('log', '--oneline', '-1', cwd=REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model directory names under --input-root (default: all)",
    )
    parser.add_argument(
        "--include-mc",
        action="store_true",
        help="Also plot multiple-choice (__mc) runs; implied for any named in --models",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE, help="Path stem")
    parser.add_argument(
        "--png", action="store_true", help="Also write a PNG next to the PDF"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Commit and push the figure in state-bench-paper, then bump the "
        "submodule pointer here (as scripts/render_figures.sh --push does)",
    )
    args = parser.parse_args()

    raw = load_predictions(args.input_root, args.models, args.include_mc)
    errored = raw["error"].notna()
    if errored.any():
        print(f"Leaving out {errored.sum()} errored request(s):")
        print(raw[errored].groupby(["model", "tier"]).size().to_string())
    items = raw[~errored].assign(
        score=lambda df: df["correct"].astype(float),
    )
    # Rows written before --task-format existed are all multiple choice.
    task_format = items.get("task_format", pd.Series("choice", index=items.index))
    is_choice = task_format.fillna("choice").eq("choice")
    items["chance"] = np.where(is_choice, 1.0 / items["num_choices"], 0.0)
    unparsed = items["prediction"].isna()
    if unparsed.any():
        print(
            f"{unparsed.sum()} reply(ies) had no parseable answer and count as wrong:"
        )
        print(items[unparsed].groupby("model").size().to_string())
    if "status" in items:
        cut_off = items["status"].eq("incomplete")
        if cut_off.any():
            print(
                f"{cut_off.sum()} reply(ies) hit --max-output-tokens before finishing"
            )

    models = args.models or sorted(items["model"].unique())
    complexities = [c for c in COMPLEXITIES if c in set(items["complexity"])]
    summary = summarize(items)
    print(
        summary.pivot_table(
            index=["model", "tier"],
            columns="complexity",
            values="accuracy",
            observed=True,
        )[complexities].to_string(float_format="{:.2f}".format)
    )
    fig = plot(summary, models, complexities)
    pdf_path, png_path = aes.save_figure(
        args.output, fig=fig, tight=False, save_png=args.png
    )
    print(f"Saved {pdf_path}" + (f" and {png_path}" if png_path else ""))
    if args.push:
        push_figures([path for path in (pdf_path, png_path) if path])


if __name__ == "__main__":
    main()
