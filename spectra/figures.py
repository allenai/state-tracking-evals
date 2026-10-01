"""Figures for the GDN spectrum analysis.

``uv run python -m spectra.figures <name> [<name> ...]`` writes PDFs (and PNGs
for sharing) to ``figures/spectra/``. Eval-based figures read ``results/``
(fetched with scripts/fetch_dt.py); spectra figures read job outputs fetched
with ``spectra/fetch.py`` into ``results/spectra/``.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectra import style

os.environ.setdefault("HF_HUB_OFFLINE", "1")

GDN_SERIES = ["hybrid", "hybrid-sdp", "gdn-full", "gdn-sdp"]
PM_SERIES = ["hybrid", "gdn-full"]
DT_GROUP = "jacksonp-state-bench-dt"
FP32_GROUP = "jacksonp-state-bench-precision-fp32"


EVAL_SUBSETS = {
    "periodic": "integer-code--periodic",
    "aperiodic": "integer-code--aperiodic",
    "r-trivial": "integer-code--r-trivial",
    "solvable": "integer-code-modular--solvable",
}


def _final_step(df: pd.DataFrame) -> pd.DataFrame:
    final = df.groupby(["architecture", "training_dataset"])["step"].transform("max")
    return df[df["step"] == final]


def _matched(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["training_dataset"] == df["evaluation_dataset"]]


def _binned_accuracy(df: pd.DataFrame, by: list[str], min_examples: int = 3) -> pd.DataFrame:
    index, mids, _ = style.power_of_two_bins(df["num_assignments"])
    df = df.assign(bin_mid=mids[index])
    out = df.groupby([*by, "bin_mid"], as_index=False).agg(
        correct=("correct", "sum"), examples=("correct", "size"), chance=("chance", "mean")
    )
    out = out[out["examples"] >= min_examples].copy()
    out["accuracy"] = out["correct"] / out["examples"]
    out["ci_low"], out["ci_high"] = style.wilson(out["correct"], out["examples"])
    return out


# --------------------------------------------------------------------------------------
# Phase 0


def accuracy_by_variables(eval_dataset: str) -> None:
    """Accuracy vs length for each variable count (m=2 swaps are parity; m=4 is still soluble)."""
    from spectra.evals import load_predictions

    df = _final_step(_matched(load_predictions([DT_GROUP], subsets={EVAL_SUBSETS[eval_dataset]})))
    df = df[(df["evaluation_dataset"] == eval_dataset) & df["architecture"].isin(GDN_SERIES)]
    counts = sorted(df["num_variables"].unique())
    binned = _binned_accuracy(df, ["architecture", "num_variables"])
    series = [s for s in style.SERIES_ORDER if s in set(binned["architecture"])]
    handles = [style.series_handle(s) for s in series] + style.reference_handles() + [style.chance_handle()]
    fig, grid = style.row_figure(len(counts), handles,
                                 title=f"{style.DATASET_LABEL[eval_dataset]}: accuracy by number of variables")
    axes = []
    for i, m in enumerate(counts):
        ax = fig.add_subplot(grid[0, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        panel = binned[binned["num_variables"] == m]
        for s in series:
            curve = panel[panel["architecture"] == s].sort_values("bin_mid")
            if not curve.empty:
                style.plot_band(ax, curve["bin_mid"], curve["accuracy"], curve["ci_low"], curve["ci_high"],
                                color=style.SERIES_COLORS[s], marker=style.SERIES_MARKERS[s])
        if not panel.empty:
            ax.axhline(panel["chance"].mean(), color=style.aes.CHANCE_LINE_COLOR, linestyle="--", linewidth=1.25)
        style.configure_log_axis(ax, 10, binned["bin_mid"].max() * 1.5)
        style.add_reference_lines(ax)
        style.configure_accuracy_axis(ax, first=i == 0)
        style.panel_title(ax, f"{m} variables")
        ax.set_xlabel("Input length (assignments, log scale)" if i == 0 else None)
    style.save(fig, f"accuracy_by_variables_{eval_dataset}")
    plt.close(fig)


def precision() -> None:
    """bf16 (the paper's evals) vs fp32 inference on the same sampled instances."""
    from spectra.evals import load_predictions

    subsets = {EVAL_SUBSETS["periodic"], EVAL_SUBSETS["solvable"]}
    bf16 = load_predictions([DT_GROUP], subsets=subsets)
    fp32 = load_predictions([FP32_GROUP], dtype="float32", subsets=subsets)
    if fp32.empty:
        raise SystemExit("no fp32 results under results/; run scripts/fetch_results.py for the fp32 group")
    both = pd.concat([_final_step(_matched(bf16)), _final_step(_matched(fp32))])
    both = both[both["architecture"].isin(PM_SERIES)]
    # Pair on instances scored under both dtypes, so the comparison is within-example.
    keys = ["architecture", "training_dataset", "native_id"]
    paired = both.groupby(keys)["dtype"].transform("nunique") == 2
    both = both[paired]
    datasets = [d for d in ("solvable", "periodic") if d in set(both["evaluation_dataset"])]
    binned = _binned_accuracy(both, ["architecture", "evaluation_dataset", "dtype"])
    handles = [style.series_handle(s) for s in PM_SERIES] + [
        plt.Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.2, label="bfloat16 (paper)"),
        plt.Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.2, label="float32"),
    ] + style.reference_handles() + [style.chance_handle()]
    fig, grid = style.row_figure(len(datasets), handles, width_in=style.aes.ICLR_PAPER_WIDTH_IN)
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(grid[0, i])
        panel = binned[binned["evaluation_dataset"] == dataset]
        for s in PM_SERIES:
            for dtype, linestyle in (("bfloat16", "-"), ("float32", "--")):
                curve = panel[(panel["architecture"] == s) & (panel["dtype"] == dtype)].sort_values("bin_mid")
                if not curve.empty:
                    style.plot_band(ax, curve["bin_mid"], curve["accuracy"], curve["ci_low"], curve["ci_high"],
                                    color=style.SERIES_COLORS[s], marker=style.SERIES_MARKERS[s], linestyle=linestyle)
        ax.axhline(panel["chance"].mean(), color=style.aes.CHANCE_LINE_COLOR, linestyle="--", linewidth=1.25)
        style.configure_log_axis(ax, 10, binned["bin_mid"].max() * 1.5)
        style.add_reference_lines(ax)
        style.configure_accuracy_axis(ax, first=i == 0)
        style.panel_title(ax, style.DATASET_LABEL[dataset])
        ax.set_xlabel("Input length (assignments, log scale)" if i == 0 else None)
    style.save(fig, "precision_bf16_vs_fp32")
    plt.close(fig)
    summary = both.groupby(["architecture", "evaluation_dataset", "dtype"])["correct"].agg(["mean", "size"])
    print(summary.unstack("dtype"))


# --------------------------------------------------------------------------------------
# Spectra (job outputs)

# Roles that carry a statement's content; header and blank tokens are excluded.
CONTENT_ROLES = ["target", "arrow", "value", "source", "self", "lparen", "plus", "delta", "rparen", "mod",
                 "modulus", "semicolon"]
ID_TOKENS = 2048  # positions inside the training length distribution
BASELINE_RUNS = [(m, d) for d in ("periodic", "solvable") for m in ("hybrid", "gdn-full", "hybrid-sdp", "gdn-sdp")]


def _baseline(model, dataset, **kwargs):
    from spectra import outputs

    return outputs.find(model, dataset, **kwargs)


def reflection_heads(run, k=4, roles=None):
    """Heads ranked by in-distribution mass at mu < -0.5 (the heads that reflect)."""
    edges = run.gates["position_edges"]
    early = edges[1:] <= ID_TOKENS
    mask = run.role_mask(roles or CONTENT_ROLES)
    neg = run.moment("neg")[:, :, mask][..., early].sum(axis=(2, 3))
    count = run.moment("count")[:, :, mask][..., early].sum(axis=(2, 3))
    score = neg / np.maximum(count, 1)
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0][:k]
    layers = run.gates["layers"]
    return [(int(li), int(layers[li]), int(h), float(score[li, h])) for li, h in order]


def _position_axis_edges(edges):
    edges = edges.astype(float).copy()
    edges[0] = max(1.0, edges[1] / 2)
    return edges


SPECTRUM_MODELS = ("hybrid", "gdn-full", "hybrid-sdp", "gdn-sdp")
MU_BINS = 50  # coarsened from the 200 recorded bins
OVERWRITE_MU = 0.2  # |mu| below this counts as an overwrite (beta ~ 1)


def _mu_hist(run):
    """Content-token mu histogram per (layer, head, position bin, mu bin)."""
    hist = run.gates["hist_mu"][:, :, run.role_mask(CONTENT_ROLES)].sum(axis=2)  # (L, H, Bh, 200)
    return hist.reshape(*hist.shape[:3], MU_BINS, -1).sum(-1)


def overwrite_heads(run, k=3, exclude=()):
    """Heads ranked by in-distribution mass at |mu| < OVERWRITE_MU (delta-rule overwrites)."""
    hist = _mu_hist(run)
    early = run.gates["hist_position_edges"][1:] <= ID_TOKENS
    centers = np.linspace(-1, 1, MU_BINS + 1)
    centers = (centers[:-1] + centers[1:]) / 2
    near_zero = np.abs(centers) < OVERWRITE_MU
    total = hist[:, :, early].sum(axis=(2, 3))
    score = hist[:, :, early][..., near_zero].sum(axis=(2, 3)) / np.maximum(total, 1)
    layers = run.gates["layers"]
    ranked = [(int(li), int(layers[li]), int(h), float(score[li, h]))
              for li, h in np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]]
    return [r for r in ranked if (r[1], r[2]) not in set(exclude)][:k]


def selected_heads(run, k=3):
    """(layer index, layer, head, title) for a model's memory head plus k heads chosen by role.

    The memory head is the most persistent one (write retention at a 1K-token lag).
    The others are the top reflectors (mass at mu < -0.5) when the model has
    negative eigenvalues, otherwise the top overwriters (mass at |mu| < 0.2).
    Both rankings use in-distribution positions only.
    """
    li, layer, head, _ = persistent_heads(run)[0]
    ret = _retention(run)
    j = int(np.searchsorted(ret["lag_tokens"], 4096))
    counts = ret["counts"][li, 0, :, j]
    left = float(np.exp((ret["mean"][li, 0, head, :, j] * counts).sum() / max(counts.sum(), 1)))
    chosen = [(li, layer, head, f"memory  L{layer} H{head}  ({left:.0%} left at 4K)")]
    if run.meta.get("allow_neg_eigval", True):
        others = [r for r in reflection_heads(run, k + 1) if (r[1], r[2]) != (layer, head)][:k]
        chosen += [(oli, ol, oh, f"reflects  L{ol} H{oh}  ({score:.0%} \u03bc<\u22120.5)")
                   for oli, ol, oh, score in others]
    else:
        others = overwrite_heads(run, k, exclude=[(layer, head)])
        chosen += [(oli, ol, oh, f"overwrites  L{ol} H{oh}  ({score:.0%} |\u03bc|<{OVERWRITE_MU})")
                   for oli, ol, oh, score in others]
    return chosen


def _density_grid(nrows, ncols, title, *, panel_height_in=1.15):
    width = style.SUMMARY_FIG_WIDTH_IN
    fig = plt.figure(figsize=(width, panel_height_in * nrows + 0.75))
    grid = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.05], left=0.08, right=0.92,
                            top=1 - 0.5 / (panel_height_in * nrows + 0.75), bottom=0.38 / (panel_height_in * nrows + 0.75),
                            hspace=0.6, wspace=0.15)
    fig.suptitle(title, x=0.08, ha="left", fontsize=9, fontweight="bold")
    return fig, grid


def _draw_density(ax, counts, position_edges):
    """Column-normalized mu density (rows: position bins, cols: mu bins) on a log position axis."""
    import seaborn as sns
    from matplotlib.colors import LogNorm

    totals = counts.sum(axis=1, keepdims=True)
    density = np.where(totals > 0, counts / np.maximum(totals, 1), np.nan)
    image = ax.pcolormesh(_position_axis_edges(position_edges), np.linspace(-1, 1, MU_BINS + 1),
                          np.clip(density.T, 1e-4, 1), cmap=sns.color_palette("rocket_r", as_cmap=True),
                          norm=LogNorm(1e-4, 1), shading="flat", rasterized=True)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(style.aes.COMPACT_NUMBER_FORMATTER)
    ax.axvline(style.TOKEN_REFERENCE, color=style.ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.2)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=6)
    return image


def _finish_density_grid(fig, grid, image, axes_rows, models, name):
    for row, (model, axes) in enumerate(zip(models, axes_rows)):
        for col, ax in enumerate(axes):
            ax.tick_params(labelleft=col == 0)
            if col == 0:
                ax.set_ylabel(f"{style.SERIES_LABELS[model]}\n\u03bc", fontsize=7)
            if row == len(models) - 1 and col == 0:
                ax.set_xlabel("Token position (log scale)", fontsize=7)
    if image is not None:
        cbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, -1]))
        cbar.set_label("fraction of tokens at each position", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    style.save(fig, name)
    plt.close(fig)


def gate_spectrum_heatmaps(dataset="periodic", k=3):
    """Per-token mu = alpha * (1 - beta) by position for each model's memory head and k role heads."""
    runs = [(m, _baseline(m, dataset)) for m in SPECTRUM_MODELS]
    runs = [(m, r) for m, r in runs if r is not None]
    fig, grid = _density_grid(len(runs), k + 1,
                              f"{style.DATASET_LABEL[dataset]}: per-token eigenvalue \u03bc = \u03b1(1\u2212\u03b2) "
                              "in selected heads")
    image, axes_rows = None, []
    for row, (_model, run) in enumerate(runs):
        hist = _mu_hist(run)
        axes = []
        for col, (li, _layer, head, title) in enumerate(selected_heads(run, k)):
            ax = fig.add_subplot(grid[row, col])
            image = _draw_density(ax, hist[li, head], run.gates["hist_position_edges"])
            ax.set_title(title, fontsize=6, fontweight="regular")
            axes.append(ax)
        axes_rows.append(axes)
    _finish_density_grid(fig, grid, image, axes_rows, [m for m, _ in runs], f"gate_spectrum_heatmaps_{dataset}")


def gate_spectrum_by_layer(dataset="periodic"):
    """mu density by position, pooled over the heads of each layer."""
    runs = [(m, _baseline(m, dataset)) for m in SPECTRUM_MODELS]
    runs = [(m, r) for m, r in runs if r is not None]
    n_layers = max(len(r.meta["layer_types"]) for _, r in runs)
    fig, grid = _density_grid(len(runs), n_layers,
                              f"{style.DATASET_LABEL[dataset]}: per-token \u03bc by layer (all heads pooled)",
                              panel_height_in=1.0)
    image, axes_rows = None, []
    for row, (_model, run) in enumerate(runs):
        hist = _mu_hist(run)
        gdn_layers = list(run.gates["layers"])
        axes = []
        for layer in range(n_layers):
            ax = fig.add_subplot(grid[row, layer])
            axes.append(ax)
            if layer in gdn_layers:
                image = _draw_density(ax, hist[gdn_layers.index(layer)].sum(axis=0), run.gates["hist_position_edges"])
            else:
                ax.set_ylim(-1, 1)
                ax.set_yticks([-1, 0, 1])
                ax.set_xticks([])
                for spine in ("bottom", "left"):
                    ax.spines[spine].set_visible(False)
                ax.tick_params(left=False)
                ax.text(0.5, 0.5, "attention\nlayer", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color="0.5")
            if row == 0:
                ax.set_title(f"layer {layer}", fontsize=7, fontweight="regular")
        axes_rows.append(axes)
    _finish_density_grid(fig, grid, image, axes_rows, [m for m, _ in runs], f"gate_spectrum_by_layer_{dataset}")


def gate_spectrum_by_head(dataset="periodic"):
    """mu density by position, pooled over layers for each head index.

    Head k in one layer is unrelated to head k in another, so this pools
    unrelated heads: an overview, not a per-mechanism view.
    """
    runs = [(m, _baseline(m, dataset)) for m in SPECTRUM_MODELS]
    runs = [(m, r) for m, r in runs if r is not None]
    n_heads = max(r.gates["hist_mu"].shape[1] for _, r in runs)
    fig, grid = _density_grid(len(runs), n_heads,
                              f"{style.DATASET_LABEL[dataset]}: per-token \u03bc by head index (all GDN layers "
                              "pooled; head k is unrelated across layers)",
                              panel_height_in=1.0)
    image, axes_rows = None, []
    for row, (_model, run) in enumerate(runs):
        hist = _mu_hist(run)
        axes = []
        for head in range(n_heads):
            ax = fig.add_subplot(grid[row, head])
            image = _draw_density(ax, hist[:, head].sum(axis=0), run.gates["hist_position_edges"])
            if row == 0:
                ax.set_title(f"head {head}", fontsize=7, fontweight="regular")
            axes.append(ax)
        axes_rows.append(axes)
    _finish_density_grid(fig, grid, image, axes_rows, [m for m, _ in runs], f"gate_spectrum_by_head_{dataset}")


def gate_spectrum_all_heads(model, dataset="periodic"):
    """Every head of one model: rows are layers, columns are heads.

    Each panel is the column-normalized mu density by token position (content
    tokens). Corner labels give the head's in-distribution reflection mass
    (mu < -0.5, top left) and how much of its strongest write per block is left
    after 4K tokens (top right).
    """
    run = _baseline(model, dataset)
    if run is None:
        return
    hist = _mu_hist(run)
    gdn_layers = list(run.gates["layers"])
    layer_types = run.meta["layer_types"]
    n_layers, n_heads = len(layer_types), hist.shape[1]

    edges = run.gates["position_edges"]
    early = edges[1:] <= ID_TOKENS
    mask = run.role_mask(CONTENT_ROLES)
    neg = run.moment("neg")[:, :, mask][..., early].sum((2, 3)) / np.maximum(
        run.moment("count")[:, :, mask][..., early].sum((2, 3)), 1)
    ret = _retention(run)
    j = int(np.searchsorted(ret["lag_tokens"], 4096))
    counts = ret["counts"][:, 0, :, j]  # (L, start)
    left = np.exp((ret["mean"][:, 0, :, :, j] * counts[:, None]).sum(-1) / np.maximum(counts.sum(-1), 1)[:, None])

    panel_in = 0.82
    width = style.SUMMARY_FIG_WIDTH_IN
    height = panel_in * n_layers * 0.62 + 1.0
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(n_layers, n_heads + 1, width_ratios=[1] * n_heads + [0.06], left=0.07, right=0.92,
                            top=1 - 0.55 / height, bottom=0.45 / height, hspace=0.12, wspace=0.08)
    fig.suptitle(f"{style.SERIES_LABELS[model]}, {style.DATASET_LABEL[dataset]}: per-token \u03bc = \u03b1(1\u2212\u03b2) "
                 "by position, every head", x=0.07, ha="left", fontsize=9, fontweight="bold")
    image = None
    label_box = {"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75}
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = fig.add_subplot(grid[layer, head])
            # Position labels go on the last GDN row (a hybrid ends on an attention layer).
            last_row, first_col = layer == max(gdn_layers), head == 0
            if layer in gdn_layers:
                li = gdn_layers.index(layer)
                image = _draw_density(ax, hist[li, head], run.gates["hist_position_edges"])
                ax.text(0.03, 0.95, f"{neg[li, head]:.0%}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=4.5, color="0.1", bbox=label_box)
                ax.text(0.97, 0.95, f"{left[li, head]:.2f}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=4.5, color="0.1", bbox=label_box)
                ax.tick_params(labelbottom=last_row, labelleft=first_col, labelsize=4.5, length=1.5)
                if not last_row:
                    ax.set_xticklabels([])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if head == n_heads // 2:
                    ax.text(0.0, 0.5, "attention layer", transform=ax.transAxes, ha="center", va="center",
                            fontsize=6, color="0.5")
            if layer == 0:
                ax.set_title(f"H{head}", fontsize=6.5, fontweight="regular")
            if first_col:
                ax.set_ylabel(f"L{layer}", fontsize=6.5, rotation=0, ha="right", va="center")
    if image is not None:
        cbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, -1]))
        cbar.set_label("fraction of tokens at each position (x: token position, log; y: \u03bc \u2208 [\u22121, 1])",
                       fontsize=6)
        cbar.ax.tick_params(labelsize=5)
    fig.text(0.07, 0.15 / height, "corner labels: left = reflection mass (\u03bc<\u22120.5, \u2264 2K tokens); "
             "right = write retained at 4K tokens.", fontsize=5.5, color="0.3")
    style.save(fig, f"gate_spectrum_all_heads_{model}_{dataset}")
    plt.close(fig)


def _head_curves(run, heads, roles):
    """Per-position gate summaries pooled over ``heads`` and ``roles``."""
    mask = run.role_mask(roles)
    pick = lambda f: np.stack([run.moment(f)[li, h][mask].sum(0) for li, _, h, _ in heads]).sum(0)  # noqa: E731
    count, neg = pick("count"), pick("neg")
    return {
        "count": count,
        "reflection_mass": neg / np.maximum(count, 1),
        # Per-reflection leak: -E[log|mu| | mu < -0.5]; 0 for exact reflections.
        "reflection_leak": -pick("log_abs_mu_neg") / np.maximum(neg, 1),
        # Per-token forgetting: -E[log alpha].
        "forgetting": -pick("log_alpha") / np.maximum(count, 1),
        "exact_reflection": pick("beta_is_two") / np.maximum(neg, 1),
    }


def gate_summaries(dataset="periodic", k=4):
    """Reflection mass, reflection leak, and forgetting rate vs position for each model."""
    panels = [
        ("reflection_mass", "Reflection mass  P(μ < −0.5)", "linear"),
        ("reflection_leak", "Leak per reflection  −E log|μ|", "log"),
        ("forgetting", "Forgetting per token  −E log α", "log"),
    ]
    series = [(m, _baseline(m, dataset)) for m in GDN_SERIES]
    series = [(m, r) for m, r in series if r is not None]
    handles = [style.series_handle(m) for m, _ in series] + style.reference_handles(extrapolation=False)
    fig, grid = style.row_figure(len(panels), handles, wspace=0.35,
                                 title=f"{style.DATASET_LABEL[dataset]}: top-{k} reflection heads per model")
    for i, (key, title, scale) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        plotted = []
        for model, run in series:
            curves = _head_curves(run, reflection_heads(run, k), CONTENT_ROLES)
            edges = run.gates["position_edges"]
            mids = np.sqrt(np.maximum(edges[:-1], 1) * edges[1:])
            ok = curves["count"] > 0
            if key == "reflection_leak":
                ok &= curves["reflection_mass"] > 1e-3
            ax.plot(mids[ok], curves[key][ok], marker=style.SERIES_MARKERS[model], ms=3, lw=1.2,
                    color=style.SERIES_COLORS[model])
            plotted.extend(curves[key][ok])
        style.configure_log_axis(ax)
        ax.axvline(style.TOKEN_REFERENCE, color=style.ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.6)
        if scale == "log" and any(v > 0 for v in plotted):
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
        style.panel_title(ax, title)
        if key == "reflection_mass":
            ax.yaxis.set_major_formatter(style.aes.PCT_FORMATTER)
        ax.set_xlabel("Token position (log scale)" if i == 0 else None)
    style.save(fig, f"gate_summaries_{dataset}")
    plt.close(fig)


def _readout_curve(df, x="assignments_so_far", correct="correct_present_values"):
    index, mids, _ = style.power_of_two_bins(df[x], lo=4)
    out = df.assign(bin_mid=mids[index]).groupby("bin_mid", as_index=False).agg(
        correct=(correct, "sum"), n=(correct, "size"), chance=("n_present_values", lambda v: np.mean(1 / v)))
    out["accuracy"] = out["correct"] / out["n"]
    out["ci_low"], out["ci_high"] = style.wilson(out["correct"], out["n"])
    return out[out["n"] >= 5]


def readout_by_position():
    """Per-register readout accuracy at checkpoints inside long sequences."""
    datasets = ["solvable", "periodic"]
    handles = [style.series_handle(m) for m in GDN_SERIES] + style.reference_handles() + [
        plt.Line2D([0], [0], color=style.aes.CHANCE_LINE_COLOR, linestyle="--", linewidth=1.25,
                   label="chance (present values)")]
    fig, grid = style.row_figure(len(datasets), handles, width_in=style.aes.ICLR_PAPER_WIDTH_IN)
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(grid[0, i])
        chance = []
        for model in GDN_SERIES:
            run = _baseline(model, dataset)
            if run is None:
                continue
            curve = _readout_curve(run.readout)
            chance.append(curve["chance"].mean())
            style.plot_band(ax, curve["bin_mid"], curve["accuracy"], curve["ci_low"], curve["ci_high"],
                            color=style.SERIES_COLORS[model], marker=style.SERIES_MARKERS[model])
        if chance:
            ax.axhline(np.mean(chance), color=style.aes.CHANCE_LINE_COLOR, linestyle="--", linewidth=1.25)
        style.configure_log_axis(ax, 4, 1e5)
        style.add_reference_lines(ax)
        style.configure_accuracy_axis(ax, first=i == 0)
        style.panel_title(ax, style.DATASET_LABEL[dataset])
        ax.set_xlabel("Assignments read so far (log scale)" if i == 0 else None)
    style.save(fig, "readout_by_position")
    plt.close(fig)


def dissociation(dataset="periodic"):
    """Readout accuracy by position (x) and value age (y), per model.

    Separating "errors grow with absolute position" from "errors grow with how
    long a value has been carried" needs values of every age at every
    position; mass concentrated on the diagonal means this data cannot tell
    them apart.
    """
    import seaborn as sns

    series = [(m, _baseline(m, dataset)) for m in GDN_SERIES]
    series = [(m, r) for m, r in series if r is not None]
    width = style.SUMMARY_FIG_WIDTH_IN
    fig = plt.figure(figsize=(width, 2.2))
    grid = fig.add_gridspec(1, len(series) + 1, width_ratios=[1] * len(series) + [0.05], left=0.07, right=0.93,
                            top=0.8, bottom=0.2, wspace=0.18)
    cmap = sns.color_palette("rocket", as_cmap=True)
    image = None
    for i, (model, run) in enumerate(series):
        df = run.readout
        df = df[df["num_variables"] > 2]
        pos_index, _, pos_edges = style.power_of_two_bins(df["assignments_so_far"], lo=4)
        age_index, _, age_edges = style.power_of_two_bins(df["age"] + 1, lo=1)
        table = np.full((len(age_edges) - 1, len(pos_edges) - 1), np.nan)
        grouped = df.assign(p=pos_index, a=age_index).groupby(["a", "p"])["correct_present_values"].agg(["mean", "size"])
        for (a, p), (mean, size) in grouped.iterrows():
            if size >= 5:
                table[a, p] = mean
        ax = fig.add_subplot(grid[0, i])
        image = ax.pcolormesh(pos_edges, age_edges, table, cmap=cmap, vmin=0, vmax=1, shading="flat")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(style.aes.COMPACT_NUMBER_FORMATTER)
        ax.yaxis.set_major_formatter(style.aes.COMPACT_NUMBER_FORMATTER)
        ax.plot([1, pos_edges[-1]], [1, pos_edges[-1]], color="0.5", linewidth=0.8, linestyle="--")
        ax.tick_params(labelleft=i == 0, labelsize=6)
        style.panel_title(ax, style.SERIES_LABELS[model])
        ax.set_xlabel("Assignments so far" if i == 0 else None, fontsize=7)
        if i == 0:
            ax.set_ylabel("Value age (assignments)", fontsize=7)
    if image is not None:
        cbar = fig.colorbar(image, cax=fig.add_subplot(grid[0, -1]))
        cbar.set_label("Readout accuracy", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    fig.suptitle(f"{style.DATASET_LABEL[dataset]}: readout accuracy by position and value age", x=0.07, ha="left",
                 fontsize=9, fontweight="bold")
    style.save(fig, f"dissociation_{dataset}")
    plt.close(fig)


def _retention(run):
    """Mean log retention (L, probe, H, start bin, lag bin), counts (L, probe, start, lag), and axes."""
    loaded = run.load_all("retention.npz")
    if not loaded:
        return None
    n_pos = max(r["log_norm_sum"].shape[3] for r in loaded)
    sums = sum(_pad(r["log_norm_sum"], 3, n_pos) for r in loaded)
    counts = sum(_pad(r["count"], 2, n_pos) for r in loaded)
    mean = sums / np.maximum(counts[:, :, None], 1)
    return {
        "mean": mean,
        "counts": counts,
        "lag_tokens": loaded[0]["lag_edges_blocks"][:-1] * int(loaded[0]["block"]),
        "edges": max(loaded, key=lambda r: len(r["position_edges"]))["position_edges"],
        "layers": loaded[0]["layers"],
    }


def persistent_heads(run, k=1, lag=1024):
    """Heads whose strongest writes survive best at ``lag`` tokens (start positions pooled)."""
    ret = _retention(run)
    j = int(np.searchsorted(ret["lag_tokens"], lag))
    sums = (ret["mean"][:, 0, :, :, j] * ret["counts"][:, 0, None, :, j]).sum(-1)
    pooled = sums / np.maximum(ret["counts"][:, 0, :, j].sum(-1), 1)[:, None]  # (L, H)
    order = np.dstack(np.unravel_index(np.argsort(-pooled, axis=None), pooled.shape))[0][:k]
    return [(int(li), int(ret["layers"][li]), int(h), float(np.exp(pooled[li, h]))) for li, h in order]


def retention(dataset="periodic"):
    """How much of a write survives the transitions after it, by lag and by the write's position.

    Each panel shows the model's most persistent head (best write retention at
    a 1K-token lag), one line per write position; overlapping lines mean the
    dynamics are the same wherever the write happens. Dashed: median over all
    heads, positions pooled.
    """
    import seaborn as sns

    series = [(m, _baseline(m, dataset)) for m in GDN_SERIES]
    series = [(m, r) for m, r in series if r is not None and r.load_all("retention.npz")]
    handles = style.reference_handles(extrapolation=False) + [
        plt.Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.1, label="median head")]
    fig, grid = style.row_figure(
        len(series), handles, height_in=1.9,
        title=f"{style.DATASET_LABEL[dataset]}: retention of writes in each model's most persistent head")
    for i, (model, run) in enumerate(series):
        ret = _retention(run)
        li, layer, head, _ = persistent_heads(run)[0]
        counts = ret["counts"][li, 0]  # (start, lag)
        rows = [p for p in range(counts.shape[0]) if (counts[p] > 0).sum() >= 3]
        palette = sns.color_palette("mako_r", n_colors=len(rows) + 2)[2:]
        ax = fig.add_subplot(grid[0, i], sharey=None)
        for color, p in zip(palette, rows):
            ok = counts[p] > 0
            ax.plot(ret["lag_tokens"][ok], np.exp(ret["mean"][li, 0, head, p][ok]), color=color, lw=1.1,
                    marker="o", ms=2, label=style.aes.format_compact_number(ret["edges"][p]))
        pooled_counts = ret["counts"][:, 0].sum(1)  # (L, lag)
        pooled = (ret["mean"][:, 0] * ret["counts"][:, 0, None]).sum(2) / np.maximum(pooled_counts[:, None], 1)
        median = np.exp(np.median(pooled.reshape(-1, pooled.shape[-1]), axis=0))
        ok = pooled_counts.sum(0) > 0
        ax.plot(ret["lag_tokens"][ok], median[ok], color="0.4", linestyle="--", lw=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.5)
        ax.xaxis.set_major_formatter(style.aes.COMPACT_NUMBER_FORMATTER)
        ax.axvline(style.TOKEN_REFERENCE, color=style.ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.4)
        ax.grid(alpha=0.2)
        style.panel_title(ax, f"{style.SERIES_LABELS[model]} \u2014 L{layer} H{head}")
        ax.set_xlabel("Lag (tokens)" if i == 0 else None)
        ax.tick_params(labelleft=i == 0)
        if i == 0:
            ax.set_ylabel("\u2016\u03a6 k\u2016 (write retained)")
        if i == 0:
            ax.legend(title="write at token", fontsize=5.5, title_fontsize=6, frameon=False, loc="lower left",
                      ncol=2)
    style.save(fig, f"retention_{dataset}")
    plt.close(fig)


def _pad(array, axis, size):
    from spectra.outputs import _pad_axis

    return _pad_axis(array, axis, size)


def statement_eigenvalues(dataset="solvable", k=2):
    """Eigenvalues of whole-statement transition products, colored by position."""
    import seaborn as sns
    from matplotlib.colors import LogNorm

    series = [(m, _baseline(m, dataset)) for m in PM_SERIES + ["gdn-sdp"]]
    series = [(m, r) for m, r in series if r is not None]
    fig = plt.figure(figsize=(style.SUMMARY_FIG_WIDTH_IN, 2.3))
    grid = fig.add_gridspec(1, len(series) * k + 1, width_ratios=[1] * (len(series) * k) + [0.06], left=0.06,
                            right=0.93, top=0.8, bottom=0.14, wspace=0.25)
    cmap = sns.color_palette("mako_r", as_cmap=True)
    scatter = None
    theta = np.linspace(0, 2 * np.pi, 400)
    for i, (model, run) in enumerate(series):
        loaded = run.load_all("statement_spectra.npz")
        heads = reflection_heads(run, k)
        for j, (li, layer, head, _) in enumerate(heads):
            ax = fig.add_subplot(grid[0, i * k + j])
            ax.plot(np.cos(theta), np.sin(theta), color="0.6", linewidth=0.6)
            for spectra, sample in zip(loaded, run.sample_dirs):
                layout = np.load(sample / "layout.npz")
                pick = spectra["layer"] == layer
                ends = layout["statement_end_token"][spectra["statement"][pick]]
                eig = spectra["eigenvalues"][pick, head]  # (N, keep)
                pos = np.repeat(ends[:, None], eig.shape[1], axis=1)
                scatter = ax.scatter(eig.real.ravel(), eig.imag.ravel(), c=pos.ravel(), s=2, cmap=cmap,
                                     norm=LogNorm(10, 3e5), linewidths=0, alpha=0.7, rasterized=True)
            ax.set_aspect("equal")
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=6, labelleft=i == 0 and j == 0)
            ax.set_title(f"{style.SERIES_LABELS[model]}\nL{layer} H{head}", fontsize=6.5, fontweight="regular")
    if scatter is not None:
        fig.colorbar(scatter, cax=fig.add_subplot(grid[0, -1]), label="statement end (token)")
    fig.suptitle(f"{style.DATASET_LABEL[dataset]}: eigenvalues of per-statement transition products",
                 x=0.06, ha="left", fontsize=9, fontweight="bold")
    style.save(fig, f"statement_eigenvalues_{dataset}")
    plt.close(fig)


MEMORY_HEADS = {"gdn-full": [[1, 6]], "hybrid": [[5, 1]]}  # spectra/launch.py interventions-targeted


def intervention_variants(model):
    """(label, find kwargs) for every intervention shown for ``model``."""
    variants = [
        ("none", {}),
        ("float32", {"dtype": "float32"}),
        ("snap \u03b2\u21922 (\u03b2 > 1.98)", {"intervention": {"snap_reflect": 0.02}}),
        ("snap \u03b2\u21922 (\u03b2 > 1.75)", {"intervention": {"snap_reflect": 0.25}}),
        ("snap \u03b2\u21922 (\u03b2 > 1.75), memory head", {"intervention": {"snap_reflect": 0.25,
                                                                   "heads": MEMORY_HEADS.get(model)}}),
        ("snap \u03b1\u21921 (\u03b1 > 0.999)", {"intervention": {"snap_alpha": 0.001}}),
        ("snap both", {"intervention": {"snap_reflect": 0.02, "snap_alpha": 0.001}}),
        ("\u03b1 \u2192 \u03b1^0.5", {"intervention": {"alpha_power": 0.5}}),
    ]
    if model == "hybrid":
        variants.append(("attention window 4K", {"intervention": {"attn_window": 4096}}))
    return variants


ABLATION_HEADS = {  # spectra/launch.py "ablations"
    "gdn-full": {"memory": [[1, 6]], "reflection": [[3, 1], [6, 7], [2, 5], [8, 1]], "control": [[8, 2]]},
    "hybrid": {"memory": [[5, 1]], "reflection": [[3, 4], [2, 5], [3, 0], [1, 7]], "control": [[8, 2]]},
}


def ablations(dataset="periodic"):
    """Readout accuracy vs position with heads (or the hybrid's attention) zeroed."""
    import seaborn as sns

    from spectra import outputs

    labels = ["none", "zero memory head", "zero 4 reflection heads", "zero control head", "zero attention (hybrid)"]
    colors = dict(zip(labels[1:], sns.color_palette("colorblind", n_colors=len(labels) - 1)))
    colors["none"] = (0.1, 0.1, 0.1)
    handles = [plt.Line2D([0], [0], color=colors[label], lw=2.0 if label == "none" else 1.2, label=label)
               for label in labels] + style.reference_handles()
    fig, grid = style.row_figure(len(PM_SERIES), handles, width_in=style.aes.ICLR_PAPER_WIDTH_IN,
                                 title=f"{style.DATASET_LABEL[dataset]}: ablations")
    for i, model in enumerate(PM_SERIES):
        ax = fig.add_subplot(grid[0, i])
        variants = [("none", {})] + [
            (f"zero {name}", {"intervention": {"ablate_heads": ABLATION_HEADS[model][kind]}})
            for kind, name in (("memory", "memory head"), ("reflection", "4 reflection heads"),
                               ("control", "control head"))
        ]
        if model == "hybrid":
            variants.append(("zero attention (hybrid)", {"intervention": {"ablate_attention": [4, 9]}}))
        for label, kwargs in variants:
            run = outputs.find(model, dataset, **kwargs)
            if run is None:
                continue
            curve = _readout_curve(run.readout)
            ax.plot(curve["bin_mid"], curve["accuracy"], color=colors[label], lw=2.0 if label == "none" else 1.2,
                    marker="o" if label == "none" else None, ms=2.5, zorder=3 if label == "none" else 2)
        style.configure_log_axis(ax, 4, 1e5)
        style.add_reference_lines(ax)
        style.configure_accuracy_axis(ax, first=i == 0)
        memory = ABLATION_HEADS[model]["memory"][0]
        style.panel_title(ax, f"{style.SERIES_LABELS[model]} (memory head L{memory[0]} H{memory[1]})")
        ax.set_xlabel("Assignments read so far" if i == 0 else None)
    style.save(fig, f"ablations_{dataset}")
    plt.close(fig)


def interventions():
    """Readout accuracy vs position under gate interventions and fp32 inference."""
    import seaborn as sns

    from spectra import outputs

    cells = [(m, d) for d in ("solvable", "periodic") for m in PM_SERIES]
    labels = [label for label, _ in intervention_variants("hybrid")]
    colors = dict(zip(labels[1:], sns.color_palette("colorblind", n_colors=len(labels) - 1)))
    colors["none"] = (0.1, 0.1, 0.1)
    handles = [plt.Line2D([0], [0], color=colors[label], lw=2.0 if label == "none" else 1.1, label=label)
               for label in labels] + style.reference_handles()
    fig, grid = style.row_figure(len(cells), handles)
    for i, (model, dataset) in enumerate(cells):
        ax = fig.add_subplot(grid[0, i])
        for label, kwargs in intervention_variants(model):
            run = outputs.find(model, dataset, **kwargs)
            if run is None:
                continue
            curve = _readout_curve(run.readout)
            ax.plot(curve["bin_mid"], curve["accuracy"], color=colors[label], lw=2.0 if label == "none" else 1.1,
                    marker="o" if label == "none" else None, ms=2.5, zorder=3 if label == "none" else 2)
        style.configure_log_axis(ax, 4, 1e5)
        style.add_reference_lines(ax)
        style.configure_accuracy_axis(ax, first=i == 0)
        style.panel_title(ax, f"{style.SERIES_LABELS[model]}, {style.DATASET_LABEL[dataset]}")
        ax.set_xlabel("Assignments read so far" if i == 0 else None)
    style.save(fig, "interventions")
    plt.close(fig)


def training_steps(dataset="periodic"):
    """Readout accuracy and memory-head retention across training checkpoints.

    The head is the final checkpoint's most persistent head, held fixed so the
    rows show how that head's memory develops. Pairs of checkpoints straddle
    each learning-rate decay of the WSDS schedule.
    """
    import seaborn as sns

    from spectra import outputs

    models = [m for m in PM_SERIES if outputs.find(m, dataset) is not None]
    handles = style.reference_handles()
    fig, grid = style.row_figure(len(models), handles, nrows=2, height_in=1.7, hspace=0.6,
                                 width_in=style.aes.ICLR_PAPER_WIDTH_IN,
                                 title=f"{style.DATASET_LABEL[dataset]}: across training checkpoints")
    for i, model in enumerate(models):
        final = outputs.find(model, dataset)
        li, layer, head, _ = persistent_heads(final)[0]
        steps = sorted({r.step for r in outputs.runs() if r.model_type == model and r.training == dataset
                        and r.dtype == "bfloat16" and not r.intervention and "pilot" not in r.name})
        palette = dict(zip(steps, sns.color_palette("crest", n_colors=len(steps))))
        top = fig.add_subplot(grid[0, i])
        bottom = fig.add_subplot(grid[1, i])
        for step in steps:
            run = outputs.find(model, dataset, step=step)
            curve = _readout_curve(run.readout)
            top.plot(curve["bin_mid"], curve["accuracy"], color=palette[step], lw=1.2, marker="o", ms=2,
                     label=f"step {step}")
            ret = _retention(run)
            if ret is None:
                continue
            counts = ret["counts"][li, 0]
            pooled = (ret["mean"][li, 0, head] * counts).sum(0) / np.maximum(counts.sum(0), 1)
            ok = counts.sum(0) > 0
            bottom.plot(ret["lag_tokens"][ok], np.exp(pooled[ok]), color=palette[step], lw=1.2, marker="o", ms=2)
        style.configure_log_axis(top, 4, 1e5)
        style.add_reference_lines(top)
        style.configure_accuracy_axis(top, first=i == 0)
        style.panel_title(top, style.SERIES_LABELS[model])
        top.legend(fontsize=5.5, frameon=False, loc="lower left")
        top.set_xlabel("Assignments read so far" if i == 0 else None, fontsize=7)
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        bottom.set_ylim(1e-3, 1.5)
        bottom.xaxis.set_major_formatter(style.aes.COMPACT_NUMBER_FORMATTER)
        bottom.axvline(style.TOKEN_REFERENCE, color=style.ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.4)
        bottom.grid(alpha=0.2)
        style.panel_title(bottom, f"memory head L{layer} H{head}: write retained")
        bottom.set_xlabel("Lag (tokens)" if i == 0 else None, fontsize=7)
    style.save(fig, f"training_steps_{dataset}")
    plt.close(fig)


def retention_vs_extrapolation(dataset="periodic", lag=4096):
    """Memory-head retention vs readout accuracy past the training length, one point per run.

    Points cover every model, training checkpoint and gate intervention. A shared
    trend across models and checkpoints is the correlation; the snapped runs,
    whose reflections were made exact, test whether it is causal.
    """
    from spectra import outputs

    handles = [style.series_handle(m, linestyle="none") for m in GDN_SERIES] + [
        plt.Line2D([0], [0], marker="o", color="0.4", linestyle="none", ms=4, label="final checkpoint"),
        plt.Line2D([0], [0], marker="o", color="0.4", markerfacecolor="none", linestyle="none", ms=4,
                   label="earlier checkpoint"),
        plt.Line2D([0], [0], marker="X", color="0.4", linestyle="none", ms=5, label="\u03b2 snapped to 2"),
    ]
    fig, grid = style.row_figure(1, handles, width_in=style.aes.ICLR_PAPER_WIDTH_IN, height_in=2.3,
                                 title=f"{style.DATASET_LABEL[dataset]}: memory retention vs. extrapolation")
    ax = fig.add_subplot(grid[0, 0])
    for model in GDN_SERIES:
        final = outputs.find(model, dataset)
        if final is None:
            continue
        li, layer, head, _ = persistent_heads(final)[0]
        for run in outputs.runs():
            if run.model_type != model or run.training != dataset or run.dtype != "bfloat16" or "pilot" in run.name:
                continue
            snapped = "snap_reflect" in run.intervention
            if run.intervention and not snapped:
                continue
            ret = _retention(run)
            if ret is None:
                continue
            j = int(np.searchsorted(ret["lag_tokens"], lag))
            counts = ret["counts"][li, 0, :, j]
            retained = float(np.exp((ret["mean"][li, 0, head, :, j] * counts).sum() / max(counts.sum(), 1)))
            df = run.readout
            past = df[(df["assignments_so_far"] > 1024) & (df["assignments_so_far"] <= 4096)]
            accuracy = past["correct_present_values"].mean()
            is_final = run.step == final.step
            color = style.SERIES_COLORS[model]
            ax.plot(retained, accuracy, linestyle="none", ms=6 if snapped else 4.5,
                    marker="X" if snapped else style.SERIES_MARKERS[model], color=color,
                    markerfacecolor=color if (is_final or snapped) else "none", alpha=0.55 if snapped else 1.0,
                    zorder=2 if snapped else 3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(style.aes.PCT_FORMATTER)
    ax.grid(alpha=0.2)
    ax.set_xlabel(f"Memory-head write retained at {style.aes.format_compact_number(lag)} tokens")
    ax.set_ylabel("Accuracy, 1K\u20134K assign.")
    style.save(fig, f"retention_vs_extrapolation_{dataset}")
    plt.close(fig)


def hidden_drift(dataset="periodic"):
    """Cosine similarity of each layer's mean input (per role) to its in-distribution mean."""
    import seaborn as sns

    series = [(m, _baseline(m, dataset)) for m in PM_SERIES]
    series = [(m, r) for m, r in series if r is not None]
    fig, grid = style.row_figure(len(series), style.reference_handles(extrapolation=False), height_in=1.9,
                                 width_in=style.aes.ICLR_PAPER_WIDTH_IN,
                                 title=f"{style.DATASET_LABEL[dataset]}: residual-stream drift at each layer's input")
    for i, (model, run) in enumerate(series):
        loaded = run.load_all("drift.npz")
        n_pos = max(d["sum"].shape[2] for d in loaded)
        sums = sum(_pad(d["sum"], 2, n_pos) for d in loaded)  # (L, R, B, d)
        counts = sum(_pad(d["count"], 2, n_pos) for d in loaded)  # (L, R, B)
        edges = max(loaded, key=lambda d: len(d["position_edges"]))["position_edges"]
        mids = np.sqrt(np.maximum(edges[:-1], 1) * edges[1:])
        roles = run.role_mask(CONTENT_ROLES)
        layers = loaded[0]["layers"]
        colors = sns.color_palette("crest", n_colors=len(layers))
        ax = fig.add_subplot(grid[0, i])
        early = edges[1:] <= ID_TOKENS
        for color, (li, layer) in zip(colors, enumerate(layers)):
            means = sums[li] / np.maximum(counts[li], 1)[..., None]  # (R, B, d)
            reference = (sums[li][:, early].sum(1) / np.maximum(counts[li][:, early].sum(1), 1)[:, None])
            cos = np.einsum("rbd,rd->rb", means, reference) / (
                np.linalg.norm(means, axis=-1) * np.linalg.norm(reference, axis=-1)[:, None] + 1e-12)
            weight = counts[li] * roles[:, None]
            curve = (cos * weight).sum(0) / np.maximum(weight.sum(0), 1)
            ok = weight.sum(0) > 0
            is_attention = run.meta["layer_types"][int(layer)] != "linear_attention"
            ax.plot(mids[ok], curve[ok], color=color, lw=1.1, linestyle="--" if is_attention else "-",
                    label=f"L{layer}{' (attn)' if is_attention else ''}")
        style.configure_log_axis(ax)
        ax.axvline(style.TOKEN_REFERENCE, color=style.ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.4)
        ax.grid(alpha=0.2)
        style.panel_title(ax, style.SERIES_LABELS[model])
        ax.set_xlabel("Token position (log scale)" if i == 0 else None)
        if i == 0:
            ax.set_ylabel("cos(mean input, ID mean)")
        ax.legend(fontsize=5, ncol=2, frameon=False, loc="lower left")
    style.save(fig, f"hidden_drift_{dataset}")
    plt.close(fig)


FIGURES = {
    "accuracy-by-variables": lambda: [accuracy_by_variables(d) for d in ("periodic", "solvable")],
    "precision": precision,
    "gate-heatmaps": lambda: [gate_spectrum_heatmaps(d) for d in ("periodic", "solvable")],
    "gate-by-layer": lambda: [gate_spectrum_by_layer(d) for d in ("periodic", "solvable")],
    "gate-by-head": lambda: [gate_spectrum_by_head(d) for d in ("periodic", "solvable")],
    "gate-all-heads": lambda: [gate_spectrum_all_heads(m, d) for d in ("periodic", "solvable")
                               for m in SPECTRUM_MODELS],
    "gate-summaries": lambda: [gate_summaries(d) for d in ("periodic", "solvable")],
    "readout": readout_by_position,
    "dissociation": lambda: [dissociation(d) for d in ("periodic", "solvable")],
    "retention": lambda: [retention(d) for d in ("periodic", "solvable")],
    "statement-eigenvalues": lambda: [statement_eigenvalues(d) for d in ("solvable", "periodic")],
    "interventions": interventions,
    "ablations": ablations,
    "drift": lambda: [hidden_drift(d) for d in ("periodic", "solvable")],
    "training-steps": training_steps,
    "retention-vs-extrapolation": retention_vs_extrapolation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", choices=[*FIGURES, "all"])
    args = parser.parse_args()
    for name in FIGURES if "all" in args.names else args.names:
        print(f">> {name}")
        FIGURES[name]()


if __name__ == "__main__":
    main()
