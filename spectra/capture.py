"""Record, summarize, and intervene on GDN transitions during a forward pass.

Every GDN head applies the per-token transition

    A_t = alpha_t * (I - beta_t * k_t k_t^T),    ||k_t|| = 1

(keys are L2-normalized inside the kernel), whose spectrum is exactly
``alpha_t`` (multiplicity d_k - 1) and ``mu_t = alpha_t * (1 - beta_t)`` along
``k_t``. So the gates ``g = log alpha`` and ``beta`` *are* the per-token
eigenvalues, and recording them needs no eigendecomposition.

``olmo3_5_hybrid``'s GDN module stores its kernels as instance attributes
(``self.chunk_gated_delta_rule`` / ``self.recurrent_gated_delta_rule``), so
:class:`GateRecorder` replaces them per instance with a wrapper that sees the
exact ``q, k, v, g, beta`` the kernel receives -- after the model's own dtype
casts -- can rewrite ``g``/``beta`` (interventions), and hands each segment to
a set of streaming *sinks*:

- :class:`BinnedGateStats` -- moments and histograms of alpha/mu per
  (layer, head, token role, position bin);
- :class:`RawWindows` -- raw gates for a few log-spaced token windows;
- :class:`ImpulseRetention` -- how much of a write direction survives the
  product of later transitions, as a function of lag and start position;
- :class:`StatementSpectra` -- eigenvalues of the transition product over one
  whole statement (e.g. a swap or ``x <- (x + d) mod N``);
- :class:`HiddenDrift` -- per-role residual-stream means by position, for the
  representational-drift hypothesis (a forward pre-hook, not a kernel wrapper).

Everything here streams: a 256K-token prompt is prefilled in segments and no
sink keeps per-token tensors for the whole sequence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from spectra.replay import ROLE_NAMES
from spectra.schema import GATE_FEATURES, HISTOGRAMS

N_ROLES = len(ROLE_NAMES)


# --------------------------------------------------------------------------------------
# Position binning


def log2_position_edges(max_tokens: int) -> np.ndarray:
    """Token-position bin edges: [0, 64), then powers of two up to ``max_tokens``."""
    top = max(7, math.ceil(math.log2(max(max_tokens, 2))) + 1)
    return np.concatenate([[0], 2.0 ** np.arange(6, top)]).astype(np.int64)


def bin_index(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, positions, side="right") - 1, 0, len(edges) - 2)


# --------------------------------------------------------------------------------------
# Interventions


@dataclass
class Intervention:
    """Rewrites of the gates before they reach the kernel.

    Thresholds of 0 (and ``alpha_power == 1``) leave the gates untouched.
    ``heads`` restricts the rewrite to ``(layer, head)`` pairs, ``None`` = all.
    """

    # beta > 2 - snap_reflect  ->  beta = 2   (mu -> -alpha, an exact reflection)
    snap_reflect: float = 0.0
    # beta < snap_identity     ->  beta = 0   (the head skips the token)
    snap_identity: float = 0.0
    # alpha > 1 - snap_alpha   ->  alpha = 1  (no forgetting)
    snap_alpha: float = 0.0
    # g -> alpha_power * g, i.e. alpha -> alpha ** alpha_power (< 1 slows forgetting)
    alpha_power: float = 1.0
    # Hybrid only: keep only the last ``attn_window`` tokens in each attention layer's KV cache.
    attn_window: int | None = None
    heads: list[tuple[int, int]] | None = None

    @property
    def rewrites_gates(self) -> bool:
        return bool(self.snap_reflect or self.snap_identity or self.snap_alpha) or self.alpha_power != 1.0

    @property
    def is_null(self) -> bool:
        return not self.rewrites_gates and self.attn_window is None

    def name(self) -> str:
        if self.is_null:
            return "none"
        parts = [f"{k}={v}" for k, v in asdict(self).items() if k != "heads" and v not in (0.0, 1.0, None)]
        if self.heads is not None:
            parts.append(f"heads={len(self.heads)}")
        return ",".join(parts)

    def head_mask(self, layer: int, n_heads: int, device: torch.device) -> torch.Tensor | None:
        if self.heads is None:
            return None
        mask = torch.zeros(n_heads, dtype=torch.bool, device=device)
        for layer_index, head in self.heads:
            if layer_index == layer:
                mask[head] = True
        return mask

    def apply(self, layer: int, g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rewrites_gates:
            return g, beta
        new_g, new_beta = g, beta
        if self.alpha_power != 1.0:
            new_g = new_g * self.alpha_power
        if self.snap_alpha:
            new_g = torch.where(new_g > math.log1p(-self.snap_alpha), torch.zeros_like(new_g), new_g)
        if self.snap_reflect:
            new_beta = torch.where(new_beta > 2.0 - self.snap_reflect, torch.full_like(new_beta, 2.0), new_beta)
        if self.snap_identity:
            new_beta = torch.where(new_beta < self.snap_identity, torch.zeros_like(new_beta), new_beta)
        mask = self.head_mask(layer, g.shape[-1], g.device)
        if mask is not None:
            new_g = torch.where(mask, new_g, g)
            new_beta = torch.where(mask, new_beta, beta)
        return new_g, new_beta


# --------------------------------------------------------------------------------------
# Kernel wrapping


@dataclass
class Segment:
    """One kernel call's worth of recorded transitions (batch size 1)."""

    layer: int  # decoder layer index
    offset: int  # absolute token position of the first token
    k: torch.Tensor  # (T, H, d_k), raw (the kernel L2-normalizes)
    g: torch.Tensor  # (T, H) float32, log alpha
    beta: torch.Tensor  # (T, H) in the dtype the kernel receives

    @property
    def length(self) -> int:
        return self.g.shape[0]


class Sink(Protocol):
    def observe(self, segment: Segment) -> None: ...


def gdn_layers(model: Any) -> list[tuple[int, Any]]:
    """(decoder layer index, GDN module) for every linear-attention layer."""
    return [
        (index, layer.linear_attn)
        for index, layer in enumerate(model.model.layers)
        if layer.layer_type == "linear_attention"
    ]


def attention_layers(model: Any) -> list[int]:
    return [index for index, layer in enumerate(model.model.layers) if layer.layer_type != "linear_attention"]


class GateRecorder:
    """Wrap every GDN kernel of ``model`` to apply an intervention and feed sinks."""

    def __init__(self, model: Any, intervention: Intervention | None = None):
        self.intervention = intervention or Intervention()
        self.sinks: list[Sink] = []
        self.recording = False
        self.offset = 0
        self.layers = gdn_layers(model)
        for layer_index, module in self.layers:
            module.chunk_gated_delta_rule = self._wrap(layer_index, module.chunk_gated_delta_rule)
            module.recurrent_gated_delta_rule = self._wrap(layer_index, module.recurrent_gated_delta_rule)

    def _wrap(self, layer_index: int, kernel):
        def wrapped(q, k, v, *args, g, beta, **kwargs):
            g, beta = self.intervention.apply(layer_index, g, beta)
            if self.recording:
                if q.shape[0] != 1:
                    raise ValueError("gate recording supports batch size 1 only")
                segment = Segment(layer_index, self.offset, k[0], g[0].float(), beta[0])
                for sink in self.sinks:
                    sink.observe(segment)
            return kernel(q, k, v, *args, g=g, beta=beta, **kwargs)

        return wrapped


# --------------------------------------------------------------------------------------
# Sinks


def _unit(k: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unit-normalize in ``dtype``: normalizing in a lower precision and upcasting
    would make exact reflections look slightly contractive."""
    k = k.to(dtype)
    return k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class BinnedGateStats:
    """Streaming alpha/mu moments and histograms per (layer, head, role, position bin)."""

    def __init__(self, roles: np.ndarray, *, position_edges: np.ndarray, hist_position_edges: np.ndarray,
                 n_heads: int, device: torch.device):
        self.device = device
        self.roles = torch.as_tensor(roles.astype(np.int64), device=device)
        positions = np.arange(len(roles))
        self.position_edges = position_edges
        self.hist_position_edges = hist_position_edges
        self.n_bins = len(position_edges) - 1
        self.n_hist_bins = len(hist_position_edges) - 1
        self.bins = torch.as_tensor(bin_index(positions, position_edges), device=device)
        self.hist_bins = torch.as_tensor(bin_index(positions, hist_position_edges), device=device)
        self.n_heads = n_heads
        self.moments: dict[int, torch.Tensor] = {}
        self.histograms: dict[int, dict[str, torch.Tensor]] = {}

    def _moments(self, layer: int) -> torch.Tensor:
        if layer not in self.moments:
            self.moments[layer] = torch.zeros(
                self.n_heads, N_ROLES * self.n_bins, len(GATE_FEATURES), dtype=torch.float64, device=self.device
            )
        return self.moments[layer]

    def _hist(self, layer: int, name: str) -> torch.Tensor:
        per_layer = self.histograms.setdefault(layer, {})
        if name not in per_layer:
            per_layer[name] = torch.zeros(
                self.n_heads, N_ROLES * self.n_hist_bins * HISTOGRAMS[name][2], dtype=torch.float64,
                device=self.device,
            )
        return per_layer[name]

    def observe(self, segment: Segment) -> None:
        span = slice(segment.offset, segment.offset + segment.length)
        roles, bins, hist_bins = self.roles[span], self.bins[span], self.hist_bins[span]
        g = segment.g.double()  # (T, H)
        beta = segment.beta.double()
        alpha = g.exp()
        mu = alpha * (1.0 - beta)
        log_abs_mu = mu.abs().clamp_min(1e-300).log()
        neg, pos = (mu < -0.5).double(), (mu > 0.5).double()
        features = torch.stack(
            [
                torch.ones_like(g), g, mu, neg, pos, log_abs_mu * neg, log_abs_mu * pos,
                (segment.beta == 2.0).double(), (segment.beta == 0.0).double(), (segment.g == 0.0).double(),
            ],
            dim=-1,
        )  # (T, H, F)
        index = (roles * self.n_bins + bins)[None, :, None].expand(self.n_heads, -1, features.shape[-1])
        self._moments(segment.layer).scatter_add_(1, index, features.permute(1, 0, 2))

        cell = roles * self.n_hist_bins + hist_bins  # (T,)
        values = {
            "mu": mu,
            "log10_one_minus_alpha": torch.log10((-torch.expm1(g)).clamp_min(1e-300)),
            "log10_one_plus_mu": torch.log10((1.0 + mu).clamp_min(1e-300)),
            "log10_one_minus_mu": torch.log10((1.0 - mu).clamp_min(1e-300)),
        }
        keep = {"log10_one_plus_mu": mu < 0, "log10_one_minus_mu": mu > 0}
        for name, value in values.items():
            low, high, n = HISTOGRAMS[name]
            which = ((value - low) / (high - low) * n).floor().clamp(0, n - 1).long()  # (T, H)
            flat = (cell[:, None] * n + which).T  # (H, T)
            weight = keep[name].double().T if name in keep else torch.ones_like(flat, dtype=torch.float64)
            self._hist(segment.layer, name).scatter_add_(1, flat, weight)

    def result(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "position_edges": self.position_edges,
            "hist_position_edges": self.hist_position_edges,
        }
        layers = sorted(self.moments)
        out["layers"] = np.array(layers)
        out["moments"] = np.stack(
            [self.moments[layer].reshape(self.n_heads, N_ROLES, self.n_bins, -1).cpu().numpy() for layer in layers]
        )  # (L, H, R, B, F)
        for name, (_, _, n) in HISTOGRAMS.items():
            out[f"hist_{name}"] = np.stack(
                [
                    self.histograms[layer][name].reshape(self.n_heads, N_ROLES, self.n_hist_bins, n).cpu().numpy()
                    for layer in layers
                ]
            )  # (L, H, R, Bh, n)
        return out


class RawWindows:
    """Raw per-token gates for a few absolute token windows."""

    def __init__(self, windows: list[tuple[int, int]]):
        self.windows = windows
        self.data: dict[tuple[int, int], dict[str, list[np.ndarray]]] = {}

    def observe(self, segment: Segment) -> None:
        start, end = segment.offset, segment.offset + segment.length
        for window_index, (lo, hi) in enumerate(self.windows):
            a, b = max(lo, start), min(hi, end)
            if a >= b:
                continue
            store = self.data.setdefault((segment.layer, window_index), {"g": [], "beta": []})
            store["g"].append(segment.g[a - start : b - start].cpu().numpy().astype(np.float32))
            store["beta"].append(segment.beta[a - start : b - start].float().cpu().numpy())

    def result(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {"windows": np.array(self.windows, dtype=np.int64)}
        for (layer, window_index), store in sorted(self.data.items()):
            out[f"g_L{layer}_W{window_index}"] = np.concatenate(store["g"])
            out[f"beta_L{layer}_W{window_index}"] = np.concatenate(store["beta"])
        return out


def transition_product(k_unit: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Product A_{n-1} ... A_0 over the leading axis of ``(n, ..., d)`` inputs.

    Batched over the trailing axes of ``g``/``beta`` (and the matching axes of
    ``k_unit``); returns ``(..., d, d)``. Later transitions multiply on the left.
    """
    n, d = k_unit.shape[0], k_unit.shape[-1]
    batch = k_unit.shape[1:-1]
    product = torch.eye(d, dtype=k_unit.dtype, device=k_unit.device).expand(*batch, d, d).clone()
    for t in range(n):
        key = k_unit[t]  # (..., d)
        projected = torch.einsum("...d,...de->...e", key, product)  # k^T M
        product = product - beta[t][..., None, None] * key[..., :, None] * projected[..., None, :]
        product = product * g[t].exp()[..., None, None]
    return product


class ImpulseRetention:
    """How long a written direction survives the transitions that follow it.

    A write at token s adds ``beta_s k_s v_s^T`` to the state; at token t its
    contribution is ``Phi(t, s) k_s v_s^T beta_s`` with ``Phi(t, s)`` the product
    of the transitions in between. ``||Phi(t, s) k_s||`` is therefore the
    fraction of that write still readable at t. Exact reflections and
    permutations keep it at 1; forgetting (alpha < 1), overwrites of the same
    key (beta ~ 1), and inexact reflections shrink it.

    Transitions are grouped in absolute blocks of ``block`` tokens. Every
    ``probe_every`` blocks two probes start at a block boundary: ``write``, the
    unit key of the largest-beta token in the preceding block (the strongest
    write there), and ``random``, a fixed random direction (whose decay is
    dominated by alpha alone). The few transitions between the write and the
    boundary are skipped. Log-norms are accumulated per (start position bin,
    lag bin) and clipped at ``floor``.
    """

    PROBE_TYPES = ("write", "random")

    def __init__(self, *, n_heads: int, head_dim: int, device: torch.device, position_edges: np.ndarray,
                 block: int = 64, probe_every: int = 16, floor: float = -50.0,
                 dtype: torch.dtype = torch.float64, seed: int = 0):
        self.n_heads, self.head_dim, self.device, self.dtype = n_heads, head_dim, device, dtype
        self.block, self.probe_every, self.floor = block, probe_every, floor
        self.position_edges = position_edges
        self.n_pos = len(position_edges) - 1
        # Lag bins in blocks: [1, 2), [2, 4), ...
        self.lag_edges = 2 ** np.arange(0, 20)
        self.n_lag = len(self.lag_edges) - 1
        generator = torch.Generator().manual_seed(seed)
        self.random_direction = _unit(torch.randn(n_heads, head_dim, generator=generator), dtype).to(device)
        self.state: dict[int, dict[str, Any]] = {}
        self.sums = {}  # layer -> (P, H, pos, lag)
        self.counts = {}

    def _layer_state(self, layer: int) -> dict[str, Any]:
        if layer not in self.state:
            self.state[layer] = {
                "buffer": None,  # (k_unit, g, beta) for tokens not yet in a full block
                "next_block": 0,
                "last_write_key": None,
                "probes": torch.zeros(self.n_heads, self.head_dim, 0, dtype=self.dtype, device=self.device),
                "log_norm": torch.zeros(self.n_heads, 0, dtype=self.dtype, device=self.device),
                "start_block": np.zeros(0, dtype=np.int64),
                "probe_type": np.zeros(0, dtype=np.int64),
            }
            # Flat (H, probe type * position bin * lag bin) so records are one scatter_add.
            cells = len(self.PROBE_TYPES) * self.n_pos * self.n_lag
            self.sums[layer] = torch.zeros(self.n_heads, cells, dtype=torch.float64, device=self.device)
            self.counts[layer] = np.zeros((len(self.PROBE_TYPES), self.n_pos, self.n_lag), dtype=np.int64)
        return self.state[layer]

    def observe(self, segment: Segment) -> None:
        state = self._layer_state(segment.layer)
        k_unit = _unit(segment.k, self.dtype)
        g, beta = segment.g.to(self.dtype), segment.beta.to(self.dtype)
        if state["buffer"] is not None:
            bk, bg, bb = state["buffer"]
            k_unit, g, beta = torch.cat([bk, k_unit]), torch.cat([bg, g]), torch.cat([bb, beta])
        n_blocks = k_unit.shape[0] // self.block
        used = n_blocks * self.block
        state["buffer"] = (k_unit[used:], g[used:], beta[used:]) if used < k_unit.shape[0] else None
        if n_blocks == 0:
            return
        L, H, D = self.block, self.n_heads, self.head_dim
        kb = k_unit[:used].reshape(n_blocks, L, H, D).transpose(0, 1)  # (L, nb, H, D)
        gb = g[:used].reshape(n_blocks, L, H).transpose(0, 1)
        bb = beta[:used].reshape(n_blocks, L, H).transpose(0, 1)
        products = transition_product(kb, gb, bb)  # (nb, H, D, D)
        strongest = bb.argmax(dim=0)  # (nb, H)
        write_keys = torch.gather(kb, 0, strongest[None, :, :, None].expand(1, -1, -1, D))[0]  # (nb, H, D)

        for local in range(n_blocks):
            block_index = state["next_block"]
            if block_index % self.probe_every == 0 and block_index > 0 and state["last_write_key"] is not None:
                new = torch.stack([state["last_write_key"], self.random_direction], dim=-1)  # (H, D, 2)
                state["probes"] = torch.cat([state["probes"], new], dim=-1)
                state["log_norm"] = torch.cat(
                    [state["log_norm"], torch.zeros(H, 2, dtype=self.dtype, device=self.device)], dim=-1
                )
                state["start_block"] = np.append(state["start_block"], [block_index, block_index])
                state["probe_type"] = np.append(state["probe_type"], [0, 1])
            if state["probes"].shape[-1]:
                probes = products[local] @ state["probes"]  # (H, D, P)
                norms = probes.norm(dim=1).clamp_min(1e-300)  # (H, P)
                state["log_norm"] = (state["log_norm"] + norms.log()).clamp_min(self.floor)
                state["probes"] = probes / norms[:, None, :]
                self._record(segment.layer, state, block_index + 1)
            state["last_write_key"] = write_keys[local]
            state["next_block"] += 1

    def _record(self, layer: int, state: dict[str, Any], boundary: int) -> None:
        lag = boundary - state["start_block"]
        lag_bin = np.clip(np.searchsorted(self.lag_edges, lag, side="right") - 1, 0, self.n_lag - 1)
        pos_bin = bin_index(state["start_block"] * self.block, self.position_edges)
        types = state["probe_type"]
        flat = (types * self.n_pos + pos_bin) * self.n_lag + lag_bin
        index = torch.as_tensor(flat, device=self.device)
        self.sums[layer].scatter_add_(1, index[None].expand(self.n_heads, -1), state["log_norm"].double())
        np.add.at(self.counts[layer], (types, pos_bin, lag_bin), 1)

    def result(self) -> dict[str, np.ndarray]:
        layers = sorted(self.sums)
        return {
            "layers": np.array(layers),
            "block": np.array(self.block),
            "lag_edges_blocks": self.lag_edges,
            "position_edges": self.position_edges,
            "log_norm_sum": np.stack(
                [
                    self.sums[layer].reshape(self.n_heads, len(self.PROBE_TYPES), self.n_pos, self.n_lag)
                    .permute(1, 0, 2, 3).cpu().numpy()
                    for layer in layers
                ]
            ),  # (L, P, H, pos, lag)
            "count": np.stack([self.counts[layer] for layer in layers]),  # (L, P, pos, lag)
        }


class StatementSpectra:
    """Eigenvalues of the transition product across whole statements.

    For each selected statement, multiply the transitions of its tokens and
    keep the ``keep`` eigenvalues farthest from the trivial one (the product of
    the alphas, which every direction untouched by the statement's keys has).
    """

    def __init__(self, token_ranges: dict[int, tuple[int, int]], *, keep: int = 12):
        self.token_ranges = token_ranges
        self.keep = keep
        self.results: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    def observe(self, segment: Segment) -> None:
        start, end = segment.offset, segment.offset + segment.length
        for statement, (lo, hi) in self.token_ranges.items():
            if lo < start or hi > end:
                continue
            span = slice(lo - start, hi - start)
            k_unit = _unit(segment.k[span], torch.float64)
            g, beta = segment.g[span].double(), segment.beta[span].double()
            product = transition_product(k_unit, g, beta).cpu().numpy()  # (H, D, D)
            # Eigendecompose on the CPU: the matrices are tiny, and complex CUDA
            # arithmetic JIT-compiles kernels that fail on some GPU generations.
            eigenvalues = np.linalg.eigvals(product)  # (H, D) complex
            trivial = np.exp(g.sum(dim=0).cpu().numpy())  # (H,)
            distance = np.abs(eigenvalues - trivial[:, None])
            order = np.argsort(-distance, axis=-1)[:, : self.keep]
            self.results[(segment.layer, statement)] = {
                "eigenvalues": np.take_along_axis(eigenvalues, order, axis=1),
                "trivial": trivial,
            }

    def result(self) -> dict[str, np.ndarray]:
        keys = sorted(self.results)
        if not keys:
            return {}
        return {
            "layer": np.array([k[0] for k in keys]),
            "statement": np.array([k[1] for k in keys]),
            "eigenvalues": np.stack([self.results[k]["eigenvalues"] for k in keys]),  # (N, H, keep)
            "trivial": np.stack([self.results[k]["trivial"] for k in keys]),  # (N, H)
        }


@dataclass
class HiddenDrift:
    """Per-(layer, role, position bin) residual-stream sums at every decoder layer's input."""

    roles: np.ndarray
    position_edges: np.ndarray
    device: torch.device
    recording: bool = False
    offset: int = 0
    sums: dict[int, torch.Tensor] = field(default_factory=dict)
    sq_norms: dict[int, torch.Tensor] = field(default_factory=dict)
    counts: dict[int, torch.Tensor] = field(default_factory=dict)
    handles: list[Any] = field(default_factory=list)

    def __post_init__(self):
        self.n_bins = len(self.position_edges) - 1
        self.cells = torch.as_tensor(
            self.roles.astype(np.int64) * self.n_bins + bin_index(np.arange(len(self.roles)), self.position_edges),
            device=self.device,
        )

    def attach(self, model: Any) -> None:
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._hook(index)))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer: int):
        def hook(_module, args):
            if not self.recording:
                return
            hidden = args[0][0].double()  # (T, d)
            cells = self.cells[self.offset : self.offset + hidden.shape[0]]
            n_cells = N_ROLES * self.n_bins
            if layer not in self.sums:
                self.sums[layer] = torch.zeros(n_cells, hidden.shape[1], dtype=torch.float64, device=self.device)
                self.sq_norms[layer] = torch.zeros(n_cells, dtype=torch.float64, device=self.device)
                self.counts[layer] = torch.zeros(n_cells, dtype=torch.float64, device=self.device)
            self.sums[layer].index_add_(0, cells, hidden)
            self.sq_norms[layer].index_add_(0, cells, hidden.pow(2).sum(-1))
            self.counts[layer].index_add_(0, cells, torch.ones_like(cells, dtype=torch.float64))

        return hook

    def result(self) -> dict[str, np.ndarray]:
        layers = sorted(self.sums)
        shape = (N_ROLES, self.n_bins)
        return {
            "layers": np.array(layers),
            "position_edges": self.position_edges,
            "sum": np.stack([self.sums[layer].reshape(*shape, -1).cpu().numpy() for layer in layers]),  # (L, R, B, d)
            "sq_norm": np.stack([self.sq_norms[layer].reshape(shape).cpu().numpy() for layer in layers]),
            "count": np.stack([self.counts[layer].reshape(shape).cpu().numpy() for layer in layers]),
        }
