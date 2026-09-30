"""Per-example eval predictions for the directly-trained models, as one table.

A compact counterpart to the loading cells of ``notebooks/dt_results.ipynb``:
the same job selection (``results_io``), training-metadata parsing, doc
metadata join, and in-distribution padding filter, returning one row per
(job, example) so analyses can stratify by any doc field.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import results_io as rio  # noqa: E402

RESULTS_ROOT = PROJECT_ROOT / "results"
DOC_METADATA_CACHE_DIR = RESULTS_ROOT / "doc_metadata"
DOC_METADATA_FIELDS = (
    "complexity", "num_assignments", "num_variables", "token_length", "max_dependency_length",
    "num_filler_lines", "filler_proportion", "filler_distraction", "sensitivity_coverage",
)
# notebooks/dt_results.ipynb: training used filler <= 25% with easy distractors.
TRAIN_MAX_FILLER_PROPORTION = 0.25
TRAIN_DISTRACTION_LEVELS = frozenset({None, "easy"})

ARCHITECTURE_PATTERN = r"transformer-rope|transformer-nope|hybrid-sdp|hybrid|gdn-sdp|gdn-full"
TRAINING_RE = re.compile(
    rf"(?P<architecture>{ARCHITECTURE_PATTERN})[-_/]"
    r"(?P<formatter>integer-code-modular|integer-code)--(?P<training>r-trivial|aperiodic"
    r"|solvable-v2|periodic--sens-shuffled|periodic--sens-curriculum|periodic)(?:[_/]|$)",
    flags=re.IGNORECASE,
)
STEP_RE = re.compile(r"step(?P<step>\d+)-hf")


def dataset_key(formatter: str, complexity: str) -> str:
    """Same keys as the notebook's training_dataset_key."""
    key = complexity.lower().replace("--sens-", "-sens-")
    if key == "solvable-v2":
        return "solvable"
    if formatter.lower() == "integer-code-modular" and key != "solvable":
        return f"modular-{key}"
    return key


def load_predictions(groups: list[str], *, since: str | None = "2026-08-25", dtype: str = "bfloat16",
                     in_distribution_padding: bool = True, subsets: set[str] | None = None) -> pd.DataFrame:
    """One row per (job, example); ``subsets`` limits the dataset subsets read."""
    results = rio.select_latest(rio.load_experiments(RESULTS_ROOT, groups=groups, since=since))
    rows = []
    for result in results:
        haystack = f"{result.get('experiment_name', '')} {result.get('model', '')}"
        training = TRAINING_RE.search(haystack)
        if training is None:
            continue
        step = STEP_RE.search(haystack)
        for task in result.get("tasks", []):
            source = task.get("config", {}).get("data_source") or {}
            if source.get("source_type") != "hf" or not str(source.get("subset", "")).startswith("integer-code"):
                continue
            if subsets is not None and source["subset"] not in subsets:
                continue
            path = rio.artifact_file(result, task, "predictions")
            if path is None:
                continue
            metadata = rio.cached_doc_metadata(
                source["path"], source["subset"], source["split"], DOC_METADATA_FIELDS, DOC_METADATA_CACHE_DIR,
                revision=source.get("revision"),
            )
            formatter = source["subset"].partition("--")[0]
            metric = task.get("primary_metric") or "accuracy:logprob"
            with path.open() as lines:
                for line in lines:
                    prediction = json.loads(line)
                    doc = metadata.get(prediction.get("native_id"))
                    if doc is None:
                        continue
                    if in_distribution_padding and not (
                        float(doc.get("filler_proportion") or 0.0) <= TRAIN_MAX_FILLER_PROPORTION
                        and doc.get("filler_distraction") in TRAIN_DISTRACTION_LEVELS
                    ):
                        continue
                    score = prediction.get("instance_metrics", {})
                    for key in metric.split(":"):
                        score = score.get(key) if isinstance(score, dict) else None
                    if score is None:
                        continue
                    rows.append({
                        "experiment_id": result["experiment_id"],
                        "architecture": training["architecture"].lower(),
                        "training_dataset": dataset_key(training["formatter"], training["training"]),
                        "evaluation_dataset": dataset_key(formatter, str(doc["complexity"])),
                        "step": int(step["step"]) if step else None,
                        "dtype": dtype,
                        "split": source["split"],
                        "native_id": prediction.get("native_id"),
                        "correct": float(score),
                        "chance": 1.0 / max(len(prediction.get("model_output") or ()), 1),
                        "num_assignments": int(doc["num_assignments"]),
                        "num_variables": int(doc["num_variables"]),
                        "token_length": int(doc.get("token_length") or 0),
                        "max_dependency_length": doc.get("max_dependency_length"),
                    })
    return pd.DataFrame(rows)
