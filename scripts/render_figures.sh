#!/usr/bin/env bash
# Regenerate the paper figures from the results notebooks.
#
#   scripts/render_figures.sh [dt|pt|all] [--fetch] [--push]
#
#   dt|pt|all   Which notebook(s) to execute (default: dt).
#   --fetch     First pull newly completed eval jobs from Beaker
#               (scripts/fetch_dt.py / fetch_pt.py --completed-only).
#   --push      Commit the changed PDFs in state-bench-paper, rebase onto its
#               origin/main (Overleaf syncs land there), push, then bump the
#               submodule pointer in this repo and push that too.
#
# Notebooks run headlessly with nbconvert. Example metadata is joined from the
# on-disk cache first (HF_HUB_OFFLINE=1); if a new dataset tier is not cached
# yet, the run is retried online using the shell's Hugging Face token. The
# executed notebook is written to a temp file, never back into notebooks/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="dt"
FETCH=0
PUSH=0
for arg in "$@"; do
    case "$arg" in
        dt|pt|all) TARGET="$arg" ;;
        --fetch) FETCH=1 ;;
        --push) PUSH=1 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

case "$TARGET" in
    dt) NOTEBOOKS=(dt_results) ;;
    pt) NOTEBOOKS=(pt_results) ;;
    all) NOTEBOOKS=(dt_results pt_results) ;;
esac

if [[ "$FETCH" -eq 1 ]]; then
    for nb in "${NOTEBOOKS[@]}"; do
        fetcher="scripts/fetch_${nb%_results}.py"
        echo ">> Fetching completed jobs with $fetcher"
        uv run "$fetcher" --completed-only 2>&1 | grep -E "new experiments|\[new\]|Saved" || true
    done
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

run_notebook() {
    local nb="$1"
    local log="$TMP_DIR/$nb.log"
    echo ">> Executing notebooks/$nb.ipynb (offline metadata cache)"
    if HF_HUB_OFFLINE=1 uv run jupyter nbconvert --to notebook --execute "notebooks/$nb.ipynb" \
        --output "$TMP_DIR/$nb.executed.ipynb" --ExecutePreprocessor.timeout=3600 >"$log" 2>&1; then
        return 0
    fi
    if grep -q "no cached metadata" "$log"; then
        echo ">> Uncached dataset tier; retrying online"
        if uv run jupyter nbconvert --to notebook --execute "notebooks/$nb.ipynb" \
            --output "$TMP_DIR/$nb.executed.ipynb" --ExecutePreprocessor.timeout=3600 >"$log" 2>&1; then
            return 0
        fi
    fi
    echo "!! $nb failed; last error lines:" >&2
    grep -iE "Error|error:" "$log" | tail -5 >&2
    return 1
}

for nb in "${NOTEBOOKS[@]}"; do
    run_notebook "$nb"
done

cd state-bench-paper
changed=$(git status --short figures | awk '$1 ~ /^(M|A|\?\?)$/ {print $2}')
if [[ -z "$changed" ]]; then
    echo ">> No figure changed."
    exit 0
fi
echo ">> Changed figures:"
printf '   %s\n' $changed

if [[ "$PUSH" -eq 0 ]]; then
    echo ">> Not committing (pass --push to commit and push)."
    exit 0
fi

git fetch -q origin
git add $changed
git commit -q -m "Regenerate ${TARGET} figures"
git rebase -q origin/main
git push -q origin main
echo ">> state-bench-paper: $(git log --oneline -1)"

cd "$ROOT"
git add state-bench-paper
git commit -q -m "Bump state-bench-paper for regenerated ${TARGET} figures"
git push -q origin main
echo ">> $(basename "$ROOT"): $(git log --oneline -1)"
