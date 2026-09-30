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

    df = _final_step(_matched(load_predictions([DT_GROUP])))
    df = df[(df["evaluation_dataset"] == eval_dataset) & df["architecture"].isin(GDN_SERIES)]
    counts = sorted(df["num_variables"].unique())
    binned = _binned_accuracy(df, ["architecture", "num_variables"])
    series = [s for s in style.SERIES_ORDER if s in set(binned["architecture"])]
    handles = [style.series_handle(s) for s in series] + style.reference_handles() + [style.chance_handle()]
    fig, grid = style.row_figure(len(counts), handles)
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
    fig.suptitle(f"{style.DATASET_LABEL[eval_dataset]}: accuracy by number of variables",
                 x=0.75 / style.SUMMARY_FIG_WIDTH_IN, ha="left", fontsize=9, fontweight="bold")
    style.save(fig, f"accuracy_by_variables_{eval_dataset}")
    plt.close(fig)


def precision() -> None:
    """bf16 (the paper's evals) vs fp32 inference on the same sampled instances."""
    from spectra.evals import load_predictions

    bf16 = load_predictions([DT_GROUP])
    fp32 = load_predictions([FP32_GROUP], dtype="float32")
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


def gate_spectrum_heatmaps(dataset="periodic", k=4):
    """Density of mu = alpha * (1 - beta) by token position for each run's top reflection heads."""
    import seaborn as sns
    from matplotlib.colors import LogNorm

    runs = [(m, _baseline(m, dataset)) for m in ("hybrid", "gdn-full", "hybrid-sdp", "gdn-sdp")]
    runs = [(m, r) for m, r in runs if r is not None]
    if not runs:
        raise SystemExit(f"no baseline runs for {dataset}")
    width = style.SUMMARY_FIG_WIDTH_IN
    fig = plt.figure(figsize=(width, 1.25 * len(runs) + 0.6))
    grid = fig.add_gridspec(len(runs), k + 1, width_ratios=[1] * k + [0.05], left=0.08, right=0.93, top=0.9,
                            bottom=0.12, hspace=0.55, wspace=0.12)
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    norm = LogNorm(1e-4, 1)
    image = None
    for row, (model, run) in enumerate(runs):
        mask = run.role_mask(CONTENT_ROLES)
        hist = run.gates["hist_mu"][:, :, mask].sum(axis=2)  # (L, H, Bh, 200)
        hist = hist.reshape(*hist.shape[:3], 50, 4).sum(-1)  # 50 mu bins
        x = _position_axis_edges(run.gates["hist_position_edges"])
        y = np.linspace(-1, 1, 51)
        for col, (li, layer, head, score) in enumerate(reflection_heads(run, k)):
            ax = fig.add_subplot(grid[row, col])
            density = hist[li, head] / np.maximum(hist[li, head].sum(axis=1, keepdims=True), 1)
            density = np.where(hist[li, head].sum(axis=1, keepdims=True) > 0, density, np.nan)
            image = ax.pcolormesh(x, y, np.clip(density.T, 1e-4, 1), cmap=cmap, norm=norm, shading="flat",
                                  rasterized=True)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(style.aes.COMPACT_NUMBER_FORMATTER)
            ax.axvline(style.TOKEN_REFERENCE, color=style.ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.2)
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelleft=col == 0, labelsize=6)
            ax.set_title(f"L{layer} H{head}  ({score:.0%} μ<−0.5)", fontsize=6.5, fontweight="regular")
            if col == 0:
                ax.set_ylabel(f"{style.SERIES_LABELS[model]}\nμ", fontsize=7)
            if row == len(runs) - 1 and col == 0:
                ax.set_xlabel("Token position (log scale)", fontsize=7)
    cax = fig.add_subplot(grid[:, -1])
    fig.colorbar(image, cax=cax, label="fraction of tokens")
    fig.suptitle(f"{style.DATASET_LABEL[dataset]}: per-token eigenvalue μ = α(1−β) by position",
                 x=0.08, ha="left", fontsize=9, fontweight="bold")
    style.save(fig, f"gate_spectrum_heatmaps_{dataset}")
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
    fig, grid = style.row_figure(len(panels), handles, wspace=0.35)
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
    fig.suptitle(f"{style.DATASET_LABEL[dataset]}: top-{k} reflection heads per model",
                 x=0.75 / style.SUMMARY_FIG_WIDTH_IN, ha="left", fontsize=9, fontweight="bold")
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


def _dissociation_table(run):
    import statsmodels.formula.api as smf

    df = run.readout.copy()
    df = df[df["num_variables"] > 2]
    df["log_position"] = np.log2(df["assignments_so_far"])
    df["log_age"] = np.log2(1 + df["age"])
    df["log_hops"] = np.log2(1 + df["hops"])
    df["y"] = df["correct_present_values"].astype(int)
    if df["y"].nunique() < 2:
        return None
    import statsmodels.api as sm

    formula = "y ~ log_position + log_age + log_hops"
    if df["num_variables"].nunique() > 1:
        formula += " + C(num_variables)"
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    # Checkpoints within one sample are correlated; cluster by sample when there are enough.
    if df["sample"].nunique() >= 3:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["sample"])[0]})
    else:
        fit = model.fit(cov_type="HC0")
    rows = []
    for term in ("log_position", "log_age", "log_hops"):
        low, high = fit.conf_int().loc[term]
        rows.append({"term": term, "coef": fit.params[term], "low": low, "high": high})
    return pd.DataFrame(rows)


def dissociation(dataset="periodic"):
    """What predicts a readout error: absolute position, value age, or reference hops?

    Left: accuracy by position (x) and value age (y). Right: per-doubling
    log-odds effects from a binomial GLM with all three covariates (plus the
    variable count), clustered by sample.
    """
    import seaborn as sns

    series = [(m, _baseline(m, dataset)) for m in PM_SERIES]
    series = [(m, r) for m, r in series if r is not None]
    width = style.SUMMARY_FIG_WIDTH_IN
    fig = plt.figure(figsize=(width, 2.1))
    grid = fig.add_gridspec(1, len(series) + 2, width_ratios=[1] * len(series) + [0.05, 1.1], left=0.08,
                            right=0.98, top=0.82, bottom=0.2, wspace=0.35)
    cmap = sns.color_palette("rocket", as_cmap=True)
    image = None
    for i, (model, run) in enumerate(series):
        df = run.readout
        df = df[df["num_variables"] > 2]
        pos_index, pos_mid, pos_edges = style.power_of_two_bins(df["assignments_so_far"], lo=4)
        age_index, age_mid, age_edges = style.power_of_two_bins(df["age"] + 1, lo=1)
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
        ax.plot([1, pos_edges[-1]], [1, pos_edges[-1]], color="white", linewidth=0.8, linestyle="--")
        style.panel_title(ax, style.SERIES_LABELS[model])
        ax.set_xlabel("Assignments so far")
        if i == 0:
            ax.set_ylabel("Value age (assignments)")
    if image is not None:
        fig.colorbar(image, cax=fig.add_subplot(grid[0, len(series)]), label="Readout accuracy")
    ax = fig.add_subplot(grid[0, -1])
    labels = {"log_position": "position", "log_age": "value age", "log_hops": "ref. hops"}
    for j, (model, run) in enumerate(series):
        table = _dissociation_table(run)
        if table is None:
            continue
        y = np.arange(len(table)) + (j - (len(series) - 1) / 2) * 0.25
        ax.errorbar(table["coef"], y, xerr=[table["coef"] - table["low"], table["high"] - table["coef"]],
                    fmt=style.SERIES_MARKERS[model], color=style.SERIES_COLORS[model], ms=3.5, elinewidth=0.8,
                    capsize=1.5)
        ax.set_yticks(np.arange(len(table)), [labels[t] for t in table["term"]])
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.grid(alpha=0.2, axis="x")
    style.panel_title(ax, "log-odds per doubling")
    fig.suptitle(f"{style.DATASET_LABEL[dataset]}: what predicts a readout error?", x=0.08, ha="left",
                 fontsize=9, fontweight="bold")
    style.save(fig, f"dissociation_{dataset}")
    plt.close(fig)


def retention(dataset="periodic", k=4):
    """Surviving norm of a write direction vs lag, by the write's position."""
    import seaborn as sns

    series = [(m, _baseline(m, dataset)) for m in GDN_SERIES]
    series = [(m, r) for m, r in series if r is not None]
    fig, grid = style.row_figure(len(series), style.reference_handles(extrapolation=False), height_in=1.9)
    palette = None
    for i, (model, run) in enumerate(series):
        loaded = run.load_all("retention.npz")
        if not loaded:
            continue
        n_pos = max(r["log_norm_sum"].shape[3] for r in loaded)
        sums = sum(_pad(r["log_norm_sum"], 3, n_pos) for r in loaded)  # (L, P, H, pos, lag)
        counts = sum(_pad(r["count"], 2, n_pos) for r in loaded)  # (L, P, pos, lag)
        block = int(loaded[0]["block"])
        lag_tokens = loaded[0]["lag_edges_blocks"][:-1] * block
        edges = max(loaded, key=lambda r: len(r["position_edges"]))["position_edges"]
        heads = reflection_heads(run, k)
        mean = np.stack([sums[li, 0, h] for li, _, h, _ in heads]).sum(0) / np.maximum(counts[heads[0][0], 0], 1) / len(heads)
        present = counts[heads[0][0], 0] > 0
        rows = [p for p in range(n_pos) if present[p].sum() >= 2]
        palette = sns.color_palette("mako_r", n_colors=len(rows) + 2)[2:]
        ax = fig.add_subplot(grid[0, i])
        for color, p in zip(palette, rows):
            ok = present[p]
            ax.plot(lag_tokens[ok], np.exp(mean[p][ok]), color=color, lw=1.1, marker="o", ms=2,
                    label=style.aes.format_compact_number(edges[p]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2)
        ax.xaxis.set_major_formatter(style.aes.COMPACT_NUMBER_FORMATTER)
        ax.axvline(style.TOKEN_REFERENCE, color=style.ASSIGNMENT_REFERENCE_COLOR, linestyle=":", linewidth=1.4)
        ax.grid(alpha=0.2)
        style.panel_title(ax, style.SERIES_LABELS[model])
        ax.set_xlabel("Lag (tokens)" if i == 0 else None)
        if i == 0:
            ax.set_ylabel("‖Φ k‖ (write retained)")
        if i == len(series) - 1:
            ax.legend(title="write at token", fontsize=5.5, title_fontsize=6, frameon=False, loc="lower left")
    fig.suptitle(f"{style.DATASET_LABEL[dataset]}: retention of the strongest write per block (top-{k} heads)",
                 x=0.75 / style.SUMMARY_FIG_WIDTH_IN, ha="left", fontsize=9, fontweight="bold")
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


INTERVENTIONS = [
    ({}, "none", "-"),
    ({"snap_reflect": 0.02}, "snap β→2", "--"),
    ({"snap_alpha": 0.001}, "snap α→1", "-."),
    ({"snap_reflect": 0.02, "snap_alpha": 0.001}, "snap both", (0, (1, 1))),
    ({"alpha_power": 0.5}, "α → α^0.5", (0, (3, 1, 1, 1, 1, 1))),
    ({"attn_window": 4096}, "attention window 4K", (0, (5, 2))),
]


def interventions():
    """Readout accuracy vs position under gate interventions and fp32 inference."""
    from spectra import outputs

    cells = [(m, d) for d in ("solvable", "periodic") for m in PM_SERIES]
    variants = INTERVENTIONS + [(None, "float32", (0, (1, 2)))]
    handles = [plt.Line2D([0], [0], color="0.3", linestyle=ls, linewidth=1.2, label=label)
               for _, label, ls in variants] + style.reference_handles()
    fig, grid = style.row_figure(len(cells), handles)
    for i, (model, dataset) in enumerate(cells):
        ax = fig.add_subplot(grid[0, i])
        for iv, label, linestyle in variants:
            run = (outputs.find(model, dataset, dtype="float32") if iv is None
                   else outputs.find(model, dataset, intervention=iv))
            if run is None:
                continue
            curve = _readout_curve(run.readout)
            ax.plot(curve["bin_mid"], curve["accuracy"], color=style.SERIES_COLORS[model], linestyle=linestyle,
                    lw=1.2, marker=style.SERIES_MARKERS[model] if not iv else None, ms=2.5)
        style.configure_log_axis(ax, 4, 1e5)
        style.add_reference_lines(ax)
        style.configure_accuracy_axis(ax, first=i == 0)
        style.panel_title(ax, f"{style.SERIES_LABELS[model]}, {style.DATASET_LABEL[dataset]}")
        ax.set_xlabel("Assignments read so far" if i == 0 else None)
    style.save(fig, "interventions")
    plt.close(fig)


def hidden_drift(dataset="periodic"):
    """Cosine similarity of each layer's mean input (per role) to its in-distribution mean."""
    import seaborn as sns

    series = [(m, _baseline(m, dataset)) for m in PM_SERIES]
    series = [(m, r) for m, r in series if r is not None]
    fig, grid = style.row_figure(len(series), style.reference_handles(extrapolation=False), height_in=1.9,
                                 width_in=style.aes.ICLR_PAPER_WIDTH_IN)
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
    fig.suptitle(f"{style.DATASET_LABEL[dataset]}: residual-stream drift at each layer's input",
                 x=0.75 / style.aes.ICLR_PAPER_WIDTH_IN, ha="left", fontsize=9, fontweight="bold")
    style.save(fig, f"hidden_drift_{dataset}")
    plt.close(fig)


FIGURES = {
    "accuracy-by-variables": lambda: [accuracy_by_variables(d) for d in ("periodic", "solvable")],
    "precision": precision,
    "gate-heatmaps": lambda: [gate_spectrum_heatmaps(d) for d in ("periodic", "solvable")],
    "gate-summaries": lambda: [gate_summaries(d) for d in ("periodic", "solvable")],
    "readout": readout_by_position,
    "dissociation": lambda: [dissociation(d) for d in ("periodic", "solvable")],
    "retention": lambda: [retention(d) for d in ("periodic", "solvable")],
    "statement-eigenvalues": lambda: [statement_eigenvalues(d) for d in ("solvable", "periodic")],
    "interventions": interventions,
    "drift": lambda: [hidden_drift(d) for d in ("periodic", "solvable")],
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
