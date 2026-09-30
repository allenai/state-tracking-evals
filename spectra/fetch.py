"""Fetch spectra job outputs from Beaker into ``results/spectra``.

Each successful ``spectra-*`` experiment in the workspace is downloaded once
(its ``spectra/`` result prefix) into ``results/spectra/<experiment name>``;
the metadata job's cache goes to ``results/doc_metadata``.

    uv run python -m spectra.fetch            # every finished spectra job
    uv run python -m spectra.fetch --only pilot
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
WORKSPACE = "ai2/beyond-state"


def experiments(prefix: str) -> list[dict]:
    listing = subprocess.run(
        ["beaker", "workspace", "experiments", WORKSPACE, "--text", prefix, "--format", "json"],
        check=True, capture_output=True, text=True,
    )
    return [e for e in json.loads(listing.stdout) if e.get("name", "").startswith(prefix)]


def result_dataset(experiment: dict) -> str | None:
    detail = subprocess.run(["beaker", "experiment", "get", experiment["id"], "--format", "json"],
                            check=True, capture_output=True, text=True)
    for job in json.loads(detail.stdout)[0].get("jobs", []):
        status = job.get("status", {})
        if status.get("exitCode") == 0 and "finalized" in status:
            return job.get("result", {}).get("beaker")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--refetch", action="store_true")
    args = parser.parse_args()
    for experiment in experiments("spectra-"):
        name = experiment["name"]
        if args.only and not any(fragment in name for fragment in args.only):
            continue
        is_metadata = name == "spectra-doc-metadata"
        target = RESULTS / "doc_metadata" if is_metadata else RESULTS / "spectra" / name
        marker = target / f".fetched-{experiment['id']}"
        if marker.exists() and not args.refetch:
            continue
        dataset = result_dataset(experiment)
        if dataset is None:
            print(f"  [skip] {name}: not finished")
            continue
        staging = RESULTS / ".staging" / experiment["id"]
        if staging.exists():
            shutil.rmtree(staging)
        prefix = "doc_metadata" if is_metadata else "spectra"
        subprocess.run(["beaker", "dataset", "fetch", dataset, "--prefix", prefix, "--output", str(staging)],
                       check=True, capture_output=True)
        source = staging / prefix
        if not is_metadata:
            # One run directory per job; flatten it into the experiment-named target.
            runs = [p for p in source.iterdir() if p.is_dir()]
            source = runs[0] if len(runs) == 1 else source
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            destination = target / item.name
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(item), destination)
        marker.touch()
        shutil.rmtree(staging, ignore_errors=True)
        print(f"  [fetched] {name} -> {target.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
