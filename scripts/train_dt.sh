#!/usr/bin/env bash
#
# Launch the full StateBench direct-training suite: {transformer-rope,
# transformer-nope, hybrid, gdn-sdp, gdn-full} x {r-trivial, aperiodic,
# periodic, periodic-sens-shuffled, periodic-sens-curriculum} at the 275M
# size (the only size this suite uses) -- 25 jobs per seed.
#
# state_bench.py's own `launch` expansion covers only the first three
# distributions when --distribution is omitted; the two sensitivity-ordering
# conditions must be selected explicitly, so this script launches them as
# separate invocations. Passing --distribution restricts the suite to just
# that condition, as before.
#
# Training runs out of the olmo-core submodule rather than this repo. Gantry
# stamps each Beaker job with the git repo it was launched from, so the
# submodule's pinned commit (tracking allenai/OLMo-core `jacksonp/state-bench-dt`)
# is the olmo-core version each job clones and trains with. That submodule is
# its own uv project -- Python 3.11 plus torch -- and stays separate from this
# repo's analysis environment, so `uv run` below resolves against olmo-core's
# dependencies.
#
# Each invocation of state_bench.py's `launch` command already expands over
# model-type and distribution when both are omitted, so this script only needs
# to add an optional seed axis on top of that suite.
#
# Usage:
#   scripts/train_dt.sh [--n_seeds N] [extra args passed to state_bench.py launch]
#
# Examples:
#   scripts/train_dt.sh
#   scripts/train_dt.sh --n_seeds 3
#   scripts/train_dt.sh --dry-run
#   scripts/train_dt.sh --model-type hybrid --distribution periodic

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLMO_CORE_ROOT="$PROJECT_ROOT/olmo-core"

# `configure_launcher` (src/olmo_core/internal/ladder.py) builds the remote job's
# command from `sys.argv[0]` verbatim, so we must invoke state_bench.py with a path
# that's also valid inside the Beaker container's freshly cloned repo: a path
# relative to the repo root, run from the repo root.
STATE_BENCH_PY="src/scripts/train/ladder/state_bench.py"

if [[ ! -f "$OLMO_CORE_ROOT/$STATE_BENCH_PY" ]]; then
    echo "error: $STATE_BENCH_PY not found under $OLMO_CORE_ROOT" >&2
    echo "The olmo-core submodule isn't checked out. Run:" >&2
    echo "    git submodule update --init olmo-core" >&2
    exit 1
fi

n_seeds=1
extra_args=()
has_distribution=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --distribution | --distribution=*)
            has_distribution=1
            extra_args+=("$1")
            shift
            ;;
        --n_seeds)
            n_seeds="$2"
            shift 2
            ;;
        --n_seeds=*)
            n_seeds="${1#*=}"
            shift
            ;;
        --size | --size=*)
            echo "error: --size is fixed at 275M and cannot be overridden" >&2
            exit 1
            ;;
        --max-gpus | --max-gpus=*)
            echo "error: --max-gpus is fixed at 8 and cannot be overridden" >&2
            exit 1
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

cd "$OLMO_CORE_ROOT"

# Jobs clone this commit from GitHub rather than copying the local checkout, so
# report the pin being launched. An unpushed commit or a dirty submodule fails
# inside the launcher, well after this point.
echo "=== olmo-core @ $(git log -1 --format='%h %s') ==="

# With no explicit --distribution, cover the full suite: the default
# three-distribution expansion plus one invocation per sensitivity condition.
launch_batches=("")
if [[ "$has_distribution" -eq 0 ]]; then
    launch_batches+=("periodic-sens-shuffled" "periodic-sens-curriculum")
fi

for ((seed = 0; seed < n_seeds; seed++)); do
    for distribution in "${launch_batches[@]}"; do
        seed_args=()
        [[ "$n_seeds" -gt 1 ]] && seed_args+=(--init-seed "$seed")
        distribution_args=()
        [[ -n "$distribution" ]] && distribution_args+=(--distribution "$distribution")
        echo "=== Launching StateBench suite (init-seed $seed, distribution ${distribution:-default}) ==="
        uv run "$STATE_BENCH_PY" launch --size 275M --max-gpus 8 \
            "${seed_args[@]+"${seed_args[@]}"}" \
            "${distribution_args[@]+"${distribution_args[@]}"}" \
            "${extra_args[@]+"${extra_args[@]}"}"
    done
done
