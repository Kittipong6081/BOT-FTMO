#!/usr/bin/env bash
# ===============================================================================
# git_status_publisher.sh — bridge local pipeline state to GitHub
# ===============================================================================
# Pushes auto_train_pipeline_state.json + log tail to a `pipeline-status` branch
# on GitHub every 15 min. The cloud watchdog routine then reads this branch
# to know what's happening on the Mac.
#
# Pushes to a SEPARATE branch (`pipeline-status`) so it doesn't pollute `main`
# and never conflicts with regular development commits.
#
# Run in background:
#   nohup ./ftmo_trading_bot/scripts/git_status_publisher.sh > /dev/null 2>&1 &
#
# Stop:
#   kill $(pgrep -f git_status_publisher.sh)
# ===============================================================================

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

STATE_FILE="ftmo_trading_bot/logs/auto_train_pipeline_state.json"
LOG_FILE="ftmo_trading_bot/logs/auto_train_pipeline.log"
PUBLISH_DIR="ftmo_trading_bot/logs/_published"
BRANCH="pipeline-status"
INTERVAL_SEC=900   # 15 min

mkdir -p "$PUBLISH_DIR"

# Ensure remote auth works (skip silently if not — user can fix later)
if ! git ls-remote --exit-code origin > /dev/null 2>&1; then
    echo "[publisher] origin unreachable — exiting"
    exit 1
fi

# Setup orphan branch if needed (clean history, never merges to main)
git fetch origin "$BRANCH" 2>/dev/null || true
if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
        git branch "$BRANCH" "origin/$BRANCH" 2>/dev/null
    fi
fi

LAST_HASH=""

while true; do
    if [ -f "$STATE_FILE" ]; then
        # Hash content to skip no-op pushes
        CUR_HASH="$(shasum -a 256 "$STATE_FILE" | awk '{print $1}')"
        if [ "$CUR_HASH" != "$LAST_HASH" ]; then
            # Stage on a tmp dir so we don't disturb main working tree
            cp "$STATE_FILE" "$PUBLISH_DIR/state.json"
            tail -200 "$LOG_FILE" > "$PUBLISH_DIR/log_tail.txt" 2>/dev/null || true

            # Build status badge
            python3 -c "
import json, datetime
try:
    s = json.load(open('$PUBLISH_DIR/state.json'))
    best = s.get('best_metrics', {}) or {}
    n_iter = len(s.get('history', []))
    elapsed = s.get('elapsed_hours', 0)
    finished = s.get('finished', False)
    pr = best.get('pass_rate', 0) * 100
    dd = best.get('total_dd_max', 0) * 100
    pf = best.get('profitable_rate', 0) * 100
    br = best.get('breach_rate', 0) * 100
    open('$PUBLISH_DIR/STATUS.md', 'w').write(f'''# Pipeline Status

Last published (UTC): {datetime.datetime.utcnow().isoformat(timespec=\"seconds\")}Z
Iterations done: {n_iter}/10
Elapsed (hours): {elapsed:.2f}
Finished: {finished}

## Best metrics so far

- Pass Rate:        {pr:.2f}%   (target ≥ 8.0%)
- Breach Rate:      {br:.2f}%   (target ≤ 5.0%)
- Profitable Rate:  {pf:.2f}%   (target ≥ 55.0%)
- Total DD max:     {dd:.2f}%   (target ≤ 6.0%)
- Best iteration:   {s.get('best_iteration', '?')}

## Gates passed?

''' + ('✅ ALL GATES PASSED — model ready for live deployment' if finished and pr >= 8.0 and dd <= 6.0 and pf >= 55.0 else '⏳ Still iterating — see log_tail.txt for the latest decision'))
except Exception as e:
    open('$PUBLISH_DIR/STATUS.md', 'w').write(f'# Pipeline Status\\n\\nCould not parse state: {e}')
" 2>/dev/null

            # Commit only the published files to the status branch
            # Use git worktree to avoid disturbing main working tree
            WT_PATH="/tmp/ftmo-status-wt-$$"
            rm -rf "$WT_PATH"
            git worktree add -B "$BRANCH" "$WT_PATH" --no-checkout 2>/dev/null || \
                git worktree add "$WT_PATH" "$BRANCH" 2>/dev/null

            if [ -d "$WT_PATH" ]; then
                # Initialize empty branch if first time
                ( cd "$WT_PATH"
                  if [ ! -d .git ] || [ -z "$(ls -A .git 2>/dev/null)" ]; then
                      :  # already initialized by worktree
                  fi
                  rm -rf state.json log_tail.txt STATUS.md 2>/dev/null
                  cp "$ROOT/$PUBLISH_DIR/state.json" .
                  cp "$ROOT/$PUBLISH_DIR/log_tail.txt" . 2>/dev/null
                  cp "$ROOT/$PUBLISH_DIR/STATUS.md" .
                  git add -A
                  if ! git diff --staged --quiet; then
                      git commit -m "pipeline-status: $(date -u +%Y-%m-%dT%H:%M:%SZ)" --quiet
                      git push origin "$BRANCH" 2>&1 | tail -3
                  fi
                ) 2>/dev/null
                git worktree remove "$WT_PATH" --force 2>/dev/null
            fi

            LAST_HASH="$CUR_HASH"
        fi
    fi

    sleep "$INTERVAL_SEC"
done
