"""Layouts of the arrays spectra jobs write, shared by capture (GPU) and plots (CPU)."""

# Features accumulated by BinnedGateStats, in order.
GATE_FEATURES: tuple[str, ...] = (
    "count",
    "log_alpha",  # sum of g
    "mu",  # sum of mu
    "neg",  # mu < -0.5
    "pos",  # mu > 0.5
    "log_abs_mu_neg",  # sum of log|mu| over mu < -0.5
    "log_abs_mu_pos",  # sum of log|mu| over mu > 0.5
    "beta_is_two",  # beta == 2 exactly (a representable exact reflection)
    "beta_is_zero",  # beta == 0 exactly
    "alpha_is_one",  # g == 0 exactly
)

# Histogram axes: name -> (low, high, bins). Values outside land in the edge bins.
HISTOGRAMS: dict[str, tuple[float, float, int]] = {
    "mu": (-1.0, 1.0, 200),
    # log10(1 - alpha): how far the gate is from "no forgetting".
    "log10_one_minus_alpha": (-8.0, 0.0, 80),
    # log10(1 + mu) for mu < 0: how far a reflection is from exact.
    "log10_one_plus_mu": (-8.0, 0.0, 80),
    # log10(1 - mu) for mu > 0: how far an identity is from exact.
    "log10_one_minus_mu": (-8.0, 0.0, 80),
}
