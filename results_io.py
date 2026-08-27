"""Loading and job-level filtering for olmo-eval result dumps.

Results accumulate under ``results/<group>/`` as sweeps are launched, so an
analysis selects the jobs it wants instead of relying on a fresh Beaker group
per sweep. :func:`load_experiments` reads each group's ``all_results.json`` and
:func:`select_latest` keeps only the newest job for every model/task pair. Both
run on experiment metadata alone, before any per-example artifact is opened, so
superseded predictions are never read off disk.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

# Beaker experiment IDs are ULIDs: the leading 10 characters are a millisecond
# timestamp in Crockford base32, which orders reruns without any extra metadata.
CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

SUCCEEDED = ("succeeded",)

Experiment = dict[str, Any]
TaskKey = Callable[[Experiment, dict[str, Any]], Any]


def experiment_created_at(experiment_id: str) -> datetime:
    """Return the creation time encoded in a Beaker experiment's ULID."""
    milliseconds = 0
    for character in experiment_id[:10]:
        milliseconds = milliseconds * 32 + CROCKFORD_BASE32.index(character.upper())
    return datetime.fromtimestamp(milliseconds / 1000, UTC)


@lru_cache(maxsize=None)
@lru_cache(maxsize=None)
def dataset_revision(path: str) -> str:
    """Resolve a Hugging Face dataset's current commit SHA.

    Task configs usually leave ``data_source.revision`` unset, which resolves to
    whatever the dataset's default branch holds at read time. Caches of joined
    example metadata therefore have to be keyed by the SHA rather than by a
    moving label, or a regenerated dataset silently reuses the wrong metadata.
    """
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(path).sha
    if sha is None:
        raise ValueError(f"Hugging Face reported no commit SHA for dataset {path!r}")
    return sha


def cached_doc_metadata(
    path: str,
    subset: str,
    split: str,
    fields: Sequence[str],
    cache_dir: Path,
    *,
    id_column: str = "id",
    revision: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Map example id -> selected metadata fields for one dataset split.

    Streams only ``[id_column, *fields]`` from the Hub and caches the rows on
    disk keyed by the resolved commit SHA, so figure regeneration never
    re-downloads or re-resolves dataset shards. A cache written with fewer
    fields than requested is refetched.
    """
    from datasets import load_dataset

    key_prefix = "__".join(part.replace("/", "--") for part in (path, subset, split))
    if revision is None and os.environ.get("HF_HUB_OFFLINE") == "1":
        # Deliberate offline iteration: reuse the newest cached revision on disk
        # instead of resolving the Hub. Only for reruns where the dataset is
        # known unchanged; unset HF_HUB_OFFLINE to restore the SHA check.
        cached = sorted(
            Path(cache_dir).glob(f"{key_prefix}__*.jsonl"),
            key=lambda f: f.stat().st_mtime,
        )
        if not cached:
            raise FileNotFoundError(
                f"HF_HUB_OFFLINE=1 but no cached metadata for {path}/{subset}/{split} "
                f"under {cache_dir}"
            )
        revision = cached[-1].stem.rsplit("__", 1)[-1]
    revision = revision or dataset_revision(path)
    cache_file = Path(cache_dir) / f"{key_prefix}__{revision}.jsonl"
    rows: list[dict[str, Any]] | None = None
    cached_columns: set[str] = set()
    if cache_file.exists():
        rows = [json.loads(line) for line in cache_file.open()]
        if rows:
            cached_columns = set(rows[0])
            if not {id_column, *fields} <= cached_columns:
                rows = None  # cache predates the requested columns
    if rows is None:
        dataset = load_dataset(path, subset, split=split, revision=revision, streaming=True)
        # Refetch the union of the requested and previously cached columns so
        # callers with different field sets or id columns share one cache file
        # instead of repeatedly overwriting each other's.
        wanted = dict.fromkeys([id_column, *fields, *sorted(cached_columns)])
        columns = [c for c in wanted if c in dataset.column_names]
        rows = list(dataset.select_columns(columns))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w") as cache:
            cache.writelines(json.dumps(row) + "\n" for row in rows)
    return {row[id_column]: row for row in rows}


def normalize_task_name(task_name: str) -> str:
    """Strip sampling suffixes so reruns of the same task compare equal.

    A ``limit`` override and the older ``_10pct`` suites evaluate one task at
    different sample sizes, so both collapse to a single name for job selection.
    """
    return task_name.removesuffix("_10pct")


def load_experiments(
    results_root: Path,
    *,
    groups: Sequence[str] | None = None,
    statuses: Sequence[str] | None = SUCCEEDED,
    since: datetime | str | None = None,
    experiment_ids: Iterable[str] | None = None,
) -> list[Experiment]:
    """Load experiment records from the ``all_results.json`` files under a root.

    Each record gains ``results_dir`` and ``created_at``, locating its artifacts
    and ordering reruns. Records are returned oldest first.

    :param groups: Group directory names or glob patterns to read, e.g.
        ``["*-dt*"]`` for one experiment family. Defaults to every group
        directory holding an ``all_results.json``. A pattern matching no group
        is an error, so a typo fails loudly instead of selecting nothing.
    :param statuses: Job statuses to keep, or None to keep every status.
    :param since: Drop jobs created before this time (naive values are UTC).
    :param experiment_ids: Keep only these experiment IDs.
    """
    if isinstance(since, str):
        since = datetime.fromisoformat(since)
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=UTC)

    wanted_ids = None if experiment_ids is None else set(experiment_ids)
    wanted_statuses = None if statuses is None else set(statuses)

    if groups is None:
        group_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    else:
        group_dirs = []
        for pattern in groups:
            matched = sorted(
                path
                for path in results_root.glob(pattern)
                if (path / "all_results.json").exists()
            )
            if not matched:
                raise FileNotFoundError(
                    f"No group under {results_root} matching {pattern!r} has an all_results.json"
                )
            group_dirs.extend(path for path in matched if path not in group_dirs)

    experiments: list[Experiment] = []
    for group_dir in group_dirs:
        results_path = group_dir / "all_results.json"
        if not results_path.exists():
            continue
        for record in json.loads(results_path.read_text()):
            if wanted_ids is not None and record.get("experiment_id") not in wanted_ids:
                continue
            if wanted_statuses is not None and record.get("status") not in wanted_statuses:
                continue
            created_at = experiment_created_at(record["experiment_id"])
            if since is not None and created_at < since:
                continue
            experiments.append(
                {**record, "results_dir": str(group_dir), "created_at": created_at}
            )

    experiments.sort(key=lambda record: (record["created_at"], record["experiment_id"]))
    return experiments


def select_latest(
    experiments: Sequence[Experiment],
    *,
    task_key: TaskKey | None = None,
) -> list[Experiment]:
    """Keep only the newest job for each model/task pair.

    Returns copies whose ``tasks`` hold just the entries that job won; jobs left
    without tasks are dropped. Selection happens per task rather than per job, so
    a rerun covering part of a sweep supersedes only the tasks it repeated.

    :param task_key: Maps a record and one of its tasks to the identity that
        reruns share. Defaults to the model path plus the normalized task name.
    """
    if task_key is None:
        task_key = _default_task_key

    winners: dict[Any, tuple[datetime, str]] = {}
    for record in experiments:
        stamp = (record["created_at"], record["experiment_id"])
        for task in record.get("tasks", []):
            key = task_key(record, task)
            if key not in winners or stamp > winners[key]:
                winners[key] = stamp

    selected: list[Experiment] = []
    for record in experiments:
        stamp = (record["created_at"], record["experiment_id"])
        kept = [
            task for task in record.get("tasks", []) if winners[task_key(record, task)] == stamp
        ]
        if kept:
            selected.append({**record, "tasks": kept})
    return selected


def _default_task_key(record: Experiment, task: dict[str, Any]) -> Any:
    return record.get("model"), normalize_task_name(task.get("task", ""))


def selection_table(experiments: Sequence[Experiment]) -> pd.DataFrame:
    """Summarize the jobs feeding an analysis, one row per job."""
    return pd.DataFrame(
        [
            {
                "created_at": record["created_at"].strftime("%Y-%m-%d %H:%M"),
                "group": record.get("experiment_group"),
                "experiment_id": record["experiment_id"],
                "experiment_name": record.get("experiment_name"),
                "tasks": len(record.get("tasks", [])),
            }
            for record in experiments
        ]
    )


def superseded_table(
    loaded: Sequence[Experiment], selected: Sequence[Experiment]
) -> pd.DataFrame:
    """Report the tasks that :func:`select_latest` dropped as older duplicates."""
    kept = {
        (record["experiment_id"], task.get("task"))
        for record in selected
        for task in record.get("tasks", [])
    }
    rows = []
    for record in loaded:
        tasks = record.get("tasks", [])
        dropped = [
            task for task in tasks if (record["experiment_id"], task.get("task")) not in kept
        ]
        if dropped:
            rows.append(
                {
                    "created_at": record["created_at"].strftime("%Y-%m-%d %H:%M"),
                    "group": record.get("experiment_group"),
                    "experiment_name": record.get("experiment_name"),
                    "dropped_tasks": len(dropped),
                    "kept_tasks": len(tasks) - len(dropped),
                }
            )
    return pd.DataFrame(rows)


def artifact_file(record: Experiment, task: dict[str, Any], kind: str) -> Path | None:
    """Locate one task's ``predictions``/``requests`` JSONL inside its own group.

    Returns None when the artifact is missing or the task hash is ambiguous.
    """
    task_hash = task.get("task_hash")
    if not task_hash:
        return None
    root = Path(record["results_dir"]) / record["experiment_id"] / "main" / kind
    matches = sorted(root.glob(f"*/*_{task_hash[-6:]}-{kind}.jsonl"))
    return matches[0] if len(matches) == 1 else None
