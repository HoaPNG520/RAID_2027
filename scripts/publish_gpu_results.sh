#!/bin/sh
# Commit the outputs of a Kaggle job (bench/raw/<job_id>/), its run log and the GPU-hour
# ledger to the branch gpu-results, then push. Used by .github/workflows/kaggle.yml.
# Usage: sh scripts/publish_gpu_results.sh <job> <run.log>
set -eu
JOB=$1
LOG=$2
SHA=$(git rev-parse HEAD)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WT=$(mktemp -d)/wt
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
if git fetch --quiet origin gpu-results 2>/dev/null && git rev-parse -q --verify origin/gpu-results >/dev/null; then
  git worktree add -q -B gpu-results "$WT" origin/gpu-results
else
  git worktree add -q --detach "$WT"
  (cd "$WT" && git checkout -q --orphan gpu-results && git rm -rqf . && git clean -fdxq)
fi
RUN="$WT/runs/${JOB}-${STAMP}-$(echo "$SHA" | cut -c1-8)"
mkdir -p "$RUN"
[ -f "$LOG" ] && cp "$LOG" "$RUN/run.log"
if [ -d bench/raw ]; then cp -r bench/raw/. "$RUN/"; fi
[ -f bench/quota.json ] && cp bench/quota.json "$WT/quota.json"
echo "$JOB $STAMP $SHA $(basename "$RUN")" >> "$WT/INDEX.txt"
cd "$WT"
git add -A
git commit -q -m "results: $JOB on ${SHA}"
for i in 1 2 3; do
  git push -q origin gpu-results && exit 0
  git fetch -q origin gpu-results && git rebase -q origin/gpu-results || true
  sleep 5
done
echo "could not push results" >&2
exit 1
