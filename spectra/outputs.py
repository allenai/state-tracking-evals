"""Load spectra job outputs fetched into ``results/spectra``.

A run directory holds ``run.json`` plus one directory per sample with
``gates.npz``, ``windows.npz``, ``retention.npz``, ``statement_spectra.npz``,
``drift.npz``, ``layout.npz``, ``readout.jsonl`` and ``meta.json`` (see
spectra/run.py). Per-sample accumulators are summed across samples here, on a
common position grid (each sample's grid is a prefix of the longest one).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from spectra.schema import GATE_FEATURES, HISTOGRAMS
from spectra.replay import ROLE_CODES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECTRA_RESULTS = Path(os.environ.get("SPECTRA_RESULTS_DIR", PROJECT_ROOT / "results" / "spectra"))


def _pad_axis(array: np.ndarray, axis: int, size: int) -> np.ndarray:
    if array.shape[axis] == size:
        return array
    pad = [(0, 0)] * array.ndim
    pad[axis] = (0, size - array.shape[axis])
    return np.pad(array, pad)


@dataclass
class Run:
    path: Path
    meta: dict = field(init=False)

    def __post_init__(self):
        self.meta = json.loads((self.path / "run.json").read_text())

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def model_type(self) -> str:
        return self.meta["args"]["model_type"]

    @property
    def training(self) -> str:
        return self.meta["args"]["training"]

    @property
    def eval_class(self) -> str:
        return self.meta["args"]["eval_class"]

    @property
    def dtype(self) -> str:
        return self.meta["args"]["dtype"]

    @property
    def intervention(self) -> dict:
        return {k: v for k, v in self.meta["intervention"].items() if v not in (0.0, 1.0, None)}

    @property
    def step(self) -> int:
        return int(Path(self.meta["model_path"]).name.removeprefix("step").removesuffix("-hf"))

    @property
    def sample_dirs(self) -> list[Path]:
        return [self.path / s["id"] for s in self.meta["samples"] if (self.path / s["id"] / "meta.json").exists()]

    def samples(self, tier: str | None = None) -> list[Path]:
        return [p for p in self.sample_dirs
                if tier is None or json.loads((p / "meta.json").read_text())["tier"] == tier]

    # ---- gates ------------------------------------------------------------------------

    @cached_property
    def gates(self) -> dict[str, np.ndarray]:
        """Gate moments/histograms summed over samples; position grids padded to the longest."""
        loaded = [dict(np.load(p / "gates.npz")) for p in self.sample_dirs]
        longest = max(loaded, key=lambda g: len(g["position_edges"]))
        out = {k: longest[k] for k in ("position_edges", "hist_position_edges", "layers")}
        n_bins = len(longest["position_edges"]) - 1
        n_hist = len(longest["hist_position_edges"]) - 1
        out["moments"] = sum(_pad_axis(g["moments"], 3, n_bins) for g in loaded)
        for name in HISTOGRAMS:
            key = f"hist_{name}"
            # Shorter samples' coarse grids are prefixes only if they end on a shared
            # edge; a sample whose last coarse edge is its length is folded into the
            # bin that contains it on the longest grid.
            total = np.zeros_like(longest[key])
            for g in loaded:
                edges = g["hist_position_edges"]
                target = np.clip(np.searchsorted(longest["hist_position_edges"], edges[:-1], side="right") - 1,
                                 0, n_hist - 1)
                for source_bin, target_bin in enumerate(target):
                    total[:, :, :, target_bin] += g[key][:, :, :, source_bin]
            out[key] = total
        return out

    def moment(self, feature: str) -> np.ndarray:
        """(L, H, R, B) sums of one gate feature."""
        return self.gates["moments"][..., GATE_FEATURES.index(feature)]

    def role_mask(self, roles: list[str] | None) -> np.ndarray:
        n = self.gates["moments"].shape[2]
        if roles is None:
            mask = np.ones(n, dtype=bool)
            mask[ROLE_CODES["header"]] = False
            return mask
        mask = np.zeros(n, dtype=bool)
        mask[[ROLE_CODES[r] for r in roles]] = True
        return mask

    # ---- readout ----------------------------------------------------------------------

    @cached_property
    def readout(self) -> pd.DataFrame:
        frames = []
        for p in self.sample_dirs:
            meta = json.loads((p / "meta.json").read_text())
            frame = pd.read_json(p / "readout.jsonl", lines=True)
            if frame.empty:
                continue
            frame["sample"] = meta["id"]
            frame["tier"] = meta["tier"]
            frame["num_variables"] = meta["num_variables"]
            frame["num_assignments"] = meta["num_assignments"]
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        # Ages in statements (fillers excluded upstream only when max_filler = 0).
        df["age"] = df["statement"] - df["origin"]
        df["since_write"] = df["statement"] - df["last_write"]
        return df

    @cached_property
    def sample_meta(self) -> pd.DataFrame:
        return pd.DataFrame([json.loads((p / "meta.json").read_text()) for p in self.sample_dirs])

    # ---- other sinks ------------------------------------------------------------------

    def load_all(self, name: str) -> list[dict[str, np.ndarray]]:
        return [dict(np.load(p / name)) for p in self.sample_dirs if (p / name).exists()]


def runs(root: Path = SPECTRA_RESULTS) -> list[Run]:
    return [Run(p) for p in sorted(root.iterdir()) if (p / "run.json").exists()] if root.exists() else []


def find(model_type: str, training: str, *, dtype: str = "bfloat16", intervention: dict | None = None,
         step: int | None = None, root: Path = SPECTRA_RESULTS) -> Run | None:
    """The run matching a configuration (the newest if several)."""
    matches = [
        r for r in runs(root)
        if r.model_type == model_type and r.training == training and r.dtype == dtype
        and r.intervention == (intervention or {})
        and (step is None or r.step == step)
        and "pilot" not in r.name
    ]
    if step is None and matches:
        final = max(r.step for r in matches)
        matches = [r for r in matches if r.step == final]
    return matches[-1] if matches else None
