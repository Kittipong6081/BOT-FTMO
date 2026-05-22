#!/bin/bash
# v8.0.60 RL chain: train RL (RR 1.5 pool/GBM) → holdout eval → report
# Run with: caffeinate -dimsu nohup bash chain_v8060_rl.sh > /dev/null 2>&1 &

set -u
cd /Users/kittipong.n/Desktop/BOT/BOT-FTMO/ftmo_trading_bot || exit 1

TS=$(date +%s)
PY="../.venv/bin/python"
RL_LOG="logs/train_rl_v8060_${TS}.log"
HOLDOUT_LOG="logs/holdout_eval_v8060_${TS}.log"
CHAIN_LOG="logs/chain_v8060_rl_${TS}.log"

echo "=== CHAIN v8.0.60 RL START $(date '+%F %T') ===" | tee -a "$CHAIN_LOG"
echo "RL log:      $RL_LOG" | tee -a "$CHAIN_LOG"
echo "Holdout log: $HOLDOUT_LOG" | tee -a "$CHAIN_LOG"
echo ""

# --- Step 1: Train RL on RR 1.5 pool ---
echo "--- Step 1: train_rl start $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"
$PY -u scripts/train_mr_signal_filter.py \
    --fresh \
    --timesteps_p1 5000000 \
    --timesteps_p2 2000000 \
    --n_envs 4 \
    --pool_size 5000 \
    --outcome_noise 0.05 \
    --ml_threshold 0.30 \
    --risk_per_trade 0.0085 \
    > "$RL_LOG" 2>&1
RL_EXIT=$?
echo "--- Step 1: train_rl exit=$RL_EXIT $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"

if [ $RL_EXIT -ne 0 ]; then
    echo "❌ TRAIN RL FAILED — chain aborted" | tee -a "$CHAIN_LOG"
    tail -30 "$RL_LOG" | tee -a "$CHAIN_LOG"
    exit $RL_EXIT
fi

# --- Step 2: Holdout eval ---
echo "" | tee -a "$CHAIN_LOG"
echo "--- Step 2: holdout_eval start $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"
$PY -u scripts/holdout_eval.py > "$HOLDOUT_LOG" 2>&1
HOLDOUT_EXIT=$?
echo "--- Step 2: holdout_eval exit=$HOLDOUT_EXIT $(date '+%F %T') ---" | tee -a "$CHAIN_LOG"

# --- Report ---
echo "" | tee -a "$CHAIN_LOG"
echo "=== RL Eval Report (v8.0.60 vs v8.0.52 Pass 70.7%) ===" | tee -a "$CHAIN_LOG"
grep -E "Pass Rate|Profitable Rate|Breach|Total DD|Daily DD|Win Rate|Take Rate" "$RL_LOG" | tail -20 | tee -a "$CHAIN_LOG"

echo "" | tee -a "$CHAIN_LOG"
echo "=== Holdout Report ===" | tee -a "$CHAIN_LOG"
grep -E "Pass Rate|Profitable|Holdout|train|delta|Δ" "$HOLDOUT_LOG" | tail -20 | tee -a "$CHAIN_LOG"

echo "" | tee -a "$CHAIN_LOG"
echo "=== CHAIN v8.0.60 RL END $(date '+%F %T') ===" | tee -a "$CHAIN_LOG"
