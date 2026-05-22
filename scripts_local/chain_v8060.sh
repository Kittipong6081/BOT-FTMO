#!/bin/bash
# v8.0.60 chain: build_pool (RR 1.5) → retrain GBM → report
# Run with: caffeinate -dimsu nohup bash chain_v8060.sh > chain.log 2>&1 &

set -u
cd /Users/kittipong.n/Desktop/BOT/BOT-FTMO/ftmo_trading_bot || exit 1

TS=$(date +%s)
PY="../.venv/bin/python"
BUILD_LOG="logs/build_pool_v8060_${TS}.log"
GBM_LOG="logs/train_gbm_v8060_${TS}.log"
CHAIN_LOG="logs/chain_v8060_${TS}.log"

echo "=== CHAIN v8.0.60 START $(date '+%F %T') ===" | tee -a "$CHAIN_LOG"
echo "Build log: $BUILD_LOG" | tee -a "$CHAIN_LOG"
echo "GBM log:   $GBM_LOG"   | tee -a "$CHAIN_LOG"
echo ""

# --- Step 1: Build pool with new RR 1.5 ---
echo "--- Step 1: build_pool start $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"
$PY -u scripts/build_mr_signal_pool.py \
    --pool_size 5000 \
    --max_days 45 \
    --workers 8 \
    --seed 42 \
    > "$BUILD_LOG" 2>&1
BUILD_EXIT=$?
echo "--- Step 1: build_pool exit=$BUILD_EXIT $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"

if [ $BUILD_EXIT -ne 0 ]; then
    echo "❌ BUILD POOL FAILED — chain aborted" | tee -a "$CHAIN_LOG"
    tail -30 "$BUILD_LOG" | tee -a "$CHAIN_LOG"
    exit $BUILD_EXIT
fi

# Pool size sanity
ls -lh data/mr_signal_pool_5000.pkl | tee -a "$CHAIN_LOG"

# --- Step 2: Retrain GBM on new pool ---
echo "" | tee -a "$CHAIN_LOG"
echo "--- Step 2: train_gbm start $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"
$PY -u scripts/train_mr_signal_quality.py --seed 42 > "$GBM_LOG" 2>&1
GBM_EXIT=$?
echo "--- Step 2: train_gbm exit=$GBM_EXIT $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"

if [ $GBM_EXIT -ne 0 ]; then
    echo "❌ TRAIN GBM FAILED" | tee -a "$CHAIN_LOG"
    tail -30 "$GBM_LOG" | tee -a "$CHAIN_LOG"
    exit $GBM_EXIT
fi

# --- Report AUC vs baseline 0.6135 ---
echo "" | tee -a "$CHAIN_LOG"
echo "=== AUC Report (v8.0.60 RR 1.5 vs v8.0.52 baseline 0.6135) ===" | tee -a "$CHAIN_LOG"
grep -E "OOF AUC|Brier|Done" "$GBM_LOG" | tee -a "$CHAIN_LOG"

echo "" | tee -a "$CHAIN_LOG"
echo "=== CHAIN v8.0.60 END $(date '+%F %T') ===" | tee -a "$CHAIN_LOG"
