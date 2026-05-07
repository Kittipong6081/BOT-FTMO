#!/usr/bin/env bash
# ===============================================================================
# pipeline_status.sh — quick status check for the auto_train_pipeline
# ===============================================================================
# Shows: pipeline alive? current step? best-so-far metrics? recent log tail.
# Use:   ./ftmo_trading_bot/scripts/pipeline_status.sh
# ===============================================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
DATA_DIR="$ROOT/data"
MR_MODELS="$ROOT/models/mr"

echo "═══════════════════════════════════════════════════════════════════════"
echo " 🤖 Auto Train Pipeline — Status (`date '+%Y-%m-%d %H:%M:%S'`)"
echo "═══════════════════════════════════════════════════════════════════════"

# 1. Process state
if pgrep -f "auto_train_pipeline.py" > /dev/null; then
    PID=$(pgrep -f "auto_train_pipeline.py" | head -1)
    ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | xargs)
    echo "✅ Pipeline ALIVE  pid=$PID  elapsed=$ELAPSED"
    echo "   workers: $(pgrep -f 'multiprocessing.spawn' | wc -l | xargs) python child(ren)"
else
    echo "⏹  Pipeline NOT running (finished or stopped)"
fi
echo ""

# 2. Pool / GBM / Model artefacts
echo "📁 Artefacts:"
for f in "$DATA_DIR"/mr_signal_pool_*.pkl "$DATA_DIR"/mr_signal_quality_model.pkl \
         "$MR_MODELS"/ppo_mr_filter.zip "$MR_MODELS"/vec_normalize_mr.pkl \
         "$MR_MODELS"/best/best_meta.json; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        MTIME=$(stat -f '%Sm' -t '%H:%M:%S' "$f" 2>/dev/null || stat -c '%y' "$f" 2>/dev/null | cut -d. -f1)
        echo "   ✓ $(basename "$f"): $SIZE  ($MTIME)"
    fi
done
echo ""

# 3. Best-so-far summary
if [ -f "$LOG_DIR/auto_train_pipeline_state.json" ]; then
    echo "🏆 Best-so-far (from auto_train_pipeline_state.json):"
    python3 -c "
import json
with open('$LOG_DIR/auto_train_pipeline_state.json') as f:
    s = json.load(f)
best = s.get('best_metrics', {})
if best:
    print(f\"   Iteration:        {s.get('best_iteration', '?')}\")
    print(f\"   Pass Rate:        {best.get('pass_rate', 0)*100:.2f}%\")
    print(f\"   Breach Rate:      {best.get('breach_rate', 0)*100:.2f}%\")
    print(f\"   Profitable Rate:  {best.get('profitable_rate', 0)*100:.2f}%\")
    print(f\"   Win Rate (avg):   {best.get('win_rate', 0)*100:.2f}%\")
    print(f\"   Take Rate (avg):  {best.get('take_rate', 0)*100:.2f}%\")
    print(f\"   Total DD max:     {best.get('total_dd_max', 0)*100:.2f}%\")
    print(f\"   Daily DD max:     {best.get('daily_dd_max', 0)*100:.2f}%\")
    print(f\"   Profit avg:       \${best.get('profit_avg', 0):,.2f}\")
else:
    print('   (no completed iteration yet)')
print(f\"   Iterations done:  {len(s.get('history', []))}\")
print(f\"   Elapsed (h):      {s.get('elapsed_hours', 0):.2f}\")
" 2>/dev/null || echo "   (cannot parse state file)"
else
    echo "🏆 Best-so-far: no state file yet (still on iteration 1)"
fi
echo ""

# 4. Recent log tail (filtered for important lines)
echo "📋 Recent pipeline events (last 15 lines):"
if [ -f "$LOG_DIR/auto_train_pipeline.log" ]; then
    tail -15 "$LOG_DIR/auto_train_pipeline.log" | sed 's/^/   /'
else
    echo "   (no log yet)"
fi
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Run again any time:  ./ftmo_trading_bot/scripts/pipeline_status.sh"
echo " Full log:            tail -f $LOG_DIR/auto_train_pipeline.log"
echo " Console output:      tail -f $LOG_DIR/auto_pipeline_console.log"
echo " Stop pipeline:       kill \$(pgrep -f auto_train_pipeline.py)"
echo "═══════════════════════════════════════════════════════════════════════"
