"""Populate the results_io doc-metadata cache from inside a Beaker job.

The StateBench dataset is private, so machines without a Hugging Face token
cannot join eval predictions to per-example metadata. This runs the same
``results_io.cached_doc_metadata`` the notebooks use, writing its cache files
to ``--out``; fetch that directory into ``results/doc_metadata`` and the
notebooks (and spectra.plots) find it with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results_io as rio  # noqa: E402

DATASET = "allenai/state-bench"
SUBSETS = (
    "integer-code--r-trivial",
    "integer-code--aperiodic",
    "integer-code--periodic",
    "integer-code-modular--r-trivial",
    "integer-code-modular--aperiodic",
    "integer-code-modular--periodic",
    "integer-code-modular--solvable",
)
TIERS = (
    "tokens_under_2k", "tokens_4k", "tokens_8k", "tokens_16k", "tokens_32k", "tokens_64k",
    "tokens_128k", "tokens_256k", "tokens_512k", "tokens_1m",
)
# notebooks/dt_results.ipynb DOC_METADATA_FIELDS, plus the fields spectra.plots uses.
FIELDS = (
    "complexity", "num_assignments", "num_variables", "token_length", "max_dependency_length",
    "num_filler_lines", "filler_proportion", "filler_distraction", "sensitivity_coverage",
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/results/doc_metadata"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        for tier in TIERS:
            split = f"test__{tier}"
            try:
                rows = rio.cached_doc_metadata(DATASET, subset, split, FIELDS, args.out)
            except Exception as error:  # noqa: BLE001 - some subset/tier pairs do not exist
                logging.warning("skipped %s/%s: %s", subset, split, error)
                continue
            logging.info("%s/%s: %d rows", subset, split, len(rows))


if __name__ == "__main__":
    main()
