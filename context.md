# SYSTEM_STATE_CONTEXT

> Technical reference for FTMO Trading Bot — Signal Filter Architecture  
> **Last Updated:** 2026-04-18

## 📌 Project Summary

**Name:** FTMO Signal Filter Agent  
**Goal:** Pass FTMO Challenge (10% profit, 4% daily DD, 8% total DD in 45 days)  
**Current Architecture:** Hybrid SMC + ML + RL (3-brain system)

### Headline Numbers

| Metric | Value |
|--------|-------|
| Pool baseline win rate | 32.4% |
| ML AUC (test split) | 0.58 |
| WR @ `ml_score > 0.40` | 48.3% (EV +0.265) |
| WR @ `ml_score > 0.45` | 57.5% (EV +0.527) |
| RL obs dims | 24 |
| Pool size | 2950 episodes / 158k signals |

---

## 🏗️ Architecture (Current)

```
OHLCV (M15/H1/H4) ─→ [SMC] ─→ signal + 16 features
                                    ↓
                              [GBM ML filter] ─→ ml_score ∈ [0,1]
                                    ↓
                         signal dict + ml_score (17 features)
                                    ↓
                         [RL Env] → obs 24 dims (+ portfolio 7)
                                    ↓
                              [PPO Agent] ─→ TAKE / SKIP
                                    ↓
                             [Trade Executor] ─→ MT5
```

### 3-Brain Breakdown

| # | Brain | File | Role |
|---|-------|------|------|
| 1 | **SMC Strategy** | [strategy/smc_strategy.py](ftmo_trading_bot/strategy/smc_strategy.py) | Hand-crafted rule-based signal generation (OB/FVG/BOS/Sweep) |
| 2 | **ML Quality** | [ml/signal_quality.py](ftmo_trading_bot/ml/signal_quality.py) | GBM classifier; data-driven P(win) prediction (AUC 0.58) |
| 3 | **RL Agent** | [ml/rl_agent.py](ftmo_trading_bot/ml/rl_agent.py) | PPO; TAKE/SKIP decision based on signal + ML + portfolio state |

---

## 📦 Core Modules

### `strategy/` — SMC Rule Engine

| File | Purpose |
|------|---------|
| `smc_strategy.py` | Main strategy; HTF→MTF→LTF bias chain; confluence scoring [0-100] |
| `indicators.py` | ATR, EMA, RSI, MACD, ADX, Stoch, BB, volatility filter |
| `order_blocks.py` | Fractal-based OB detection; impulse + mitigation scoring |
| `fair_value_gaps.py` | 3-candle imbalance detection; fill status tracking |
| `liquidity_sweeps.py` | Swing high/low sweep detection |
| `market_structure.py` | BOS/CHoCH detection (body-based, 5-bar lookback) |

**Key invariants**:
- HTF bias uses **5-bar window, ≥3 bars same side, <2 opposite** (fixed from unstable 2/3)
- Session config timezone = **UTC** (MT5 server = EET, converted in check)
- Confluence score capped at 100 (session multiplier × raw score)
- OB `strength_score ∈ [0, 100]`; cluster score = max of nearby OBs

### `ml/` — Machine Learning Layer

| File | Purpose |
|------|---------|
| `signal_quality.py` | `SignalQualityModel` wrapper (sklearn GBM) |
| `strategy_backtester.py` | Pool generation engine; auto-injects `ml_score` |
| `signal_filter_env.py` | Gymnasium env for RL training |
| `rl_agent.py` | `SelfLearningAgent` — PPO inference for live |

**Key invariants**:
- `StrategyBacktester._quality_model` auto-loads from `data/signal_quality_model.pkl` if exists
- Pool signals ALWAYS have `ml_score` field (if model available) — else defaults to 0.5
- `_resolve_trade` uses **bar color heuristic** for same-bar SL/TP hits (unbiased)
- Slippage/spread friction: **0.5% max** (reduced from 2% which caused systematic bias)
- Future window: **96 bars (24h)** M15 for resolution

### `scripts/` — Training Pipeline

| Script | Order | Purpose |
|--------|-------|---------|
| `build_signal_pool.py` | 1 | Parallel pool generation (multiprocessing.Pool) |
| `train_signal_quality.py` | 2 | GBM training + re-score pool in-place |
| `train_signal_filter.py` | 3 | PPO 2-phase curriculum (Alpha → Risk) |
| `fetch_mt5_data.py` | prep | MT5 OHLCV → CSV (Windows only) |

### `main.py` — Live Trading Entry

- Loads SMC strategy + RL agent + ML quality model
- Loops every `main_loop_interval` seconds (default 5s)
- For each signal → `_build_signal_observation(sig)` → `rl_agent.should_take_signal(obs)`
- Observation structure **must match** env obs (24 dims, same feature order)

---

## 🎯 Observation Space (24 dims)

**CRITICAL**: Must stay synchronized across 3 places:
1. [`ml/signal_filter_env.py` `_get_obs()`](ftmo_trading_bot/ml/signal_filter_env.py#L141)
2. [`main.py` `_build_signal_observation()`](ftmo_trading_bot/main.py#L407)
3. [`ml/rl_agent.py` `OBS_DIM`](ftmo_trading_bot/ml/rl_agent.py#L31)

### Layout

| Idx | Feature | Source | Norm |
|-----|---------|--------|------|
| **Signal Core [0-11]** |
| 0 | confluence_norm | `sig.confluence_score` | `(x-50)/50` |
| 1 | rr_ratio_norm | `sig.rr_ratio` | `(x-1)/4` |
| 2 | direction | BUY=+1, SELL=-1 | — |
| 3 | atr_norm | `sig.atr_pips` | `(x-15)/10` |
| 4 | ob_score_norm | `sig.ob_score` | `x/100` |
| 5 | bias_alignment | `direction × market_bias` | ±1 |
| 6 | sl_atr_ratio | `sl_distance / atr` | raw |
| 7 | rsi_norm | `sig.rsi_value` | `(x-50)/50` |
| 8 | macd_norm | `sig.macd_histogram / atr` | raw |
| 9 | trend_str | `sig.trend_strength` | `x/100` |
| 10 | ob_size_atr | OB body / ATR | raw |
| 11 | adx_norm | `sig.adx` | `x/100` |
| **Market Regime [12-15]** |
| 12 | stoch_norm | `sig.stoch_k` | `(x-50)/50` |
| 13 | bb_pctb | `sig.bb_pctb` | raw |
| 14 | atr_chg | `sig.atr_change_ratio` | raw |
| 15 | price_roc | `sig.price_roc` | raw |
| **ML Quality [16]** ⭐ |
| 16 | ml_score_norm | GBM `P(win)` | `(p-0.5)×2` |
| **Portfolio State [17-23]** |
| 17 | total_dd_norm | RiskMgr.total_dd | `-x/0.10` |
| 18 | daily_dd_norm | RiskMgr.daily_dd | `-x/0.05` |
| 19 | progress_norm | profit/target × 100 | `x/100` |
| 20 | day_progress | challenge_day/45 | raw |
| 21 | trades_today | open_positions/3 | raw |
| 22 | recent_wr_norm | last 10 WR | `(wr×2)-1` |
| 23 | consec_losses | min(consec, 5)/5 | raw |

---

## 🎓 Training Pipeline Details

### Pool Generation (`build_signal_pool.py`)

- **Parallel 8 workers** via `multiprocessing.Pool`
- Stratified sampling: `pool_size // n_symbols` episodes per symbol, evenly spaced start_bars with ±48 bar jitter
- Worker inits backtester ONCE at start (loads CSV + indicators)
- Each episode = 45 days × 4 scan_points = 180 potential scans → avg ~54 valid signals
- Output: `data/signal_pool_3000.pkl` (pickle list of episode lists)

### ML Training (`train_signal_quality.py`)

- **Model**: `GradientBoostingClassifier(max_depth=4, n_estimators=300, learning_rate=0.03)`
- **Features**: 17 (all signal fields except `outcome_pnl_ratio`, `ml_score`, metadata)
- **Target**: `wins = (outcome_pnl_ratio > 0).astype(int)`
- **Train flow**: 70/30 split → AUC on test → retrain on full → save + re-score pool in-place
- **Current AUC**: 0.5781 (test), moderate edge

### RL Training (`train_signal_filter.py`)

**2-Phase Curriculum**:

| Phase | enable_risk_penalty | Steps (default) | Learning | Output |
|-------|--------------------|-----------------|----------|--------|
| 1 (Alpha) | False | 10M | chart reading + oracle SKIP | `ppo_signal_filter_p1.zip` |
| 2 (Risk) | True | 5M | + DD penalties | `ppo_signal_filter.zip` |

**PPO Hyperparams**:
- `learning_rate=3e-4` (P1) → `1e-4` (P2)
- `gamma=0.99` (long-horizon credit)
- `n_steps=4096, batch_size=256, n_epochs=5`
- `ent_coef=0.05→0.005` (P1), `0.01→0.002` (P2)
- `policy_kwargs: net_arch=dict(pi=[256,128], vf=[256,128])`
- `clip_range=0.2, gae_lambda=0.95`

**VecNormalize**:
- `norm_obs=True, norm_reward=True`
- `clip_obs=10.0, clip_reward=20.0` (20 expanded from 10 for DD breach penalty)

**SubprocVecEnv**: Used when `n_envs > 1` → real multi-core parallelism

### Reward Structure

**Phase 1 (Alpha)** — Chart reading + profit:
```
TAKE: clip(pnl_norm, -1.5, 3.0)
    + chart_reading_bonus (confluence × outcome alignment)
    + 0.02 action nudge

SKIP: Oracle feedback
    outcome >= 0.5:  -0.50 (missed big win; -0.80 if conf >= 75)
    outcome >= 0.1:  -0.16 (small missed win)
    outcome <= -0.5: +0.30 (smart skip of big loss)
    outcome <= -0.1: +0.10 (smart skip of small loss)
    else: 0
```

**Phase 2 (Risk)** — Add DD management:
```
TAKE: Phase 1 reward
    + dd_penalty() = -0.5·(exp(3·ratio)-1) for total DD > 4%
    + -0.3·(exp(3·ratio)-1) for daily DD > 2.5%
    + -0.5 if risk guard clamped PnL

Activity floor (Phase 2 only):
    take_rate < 8%:  -5.0
    take_rate < 15%: -1.5
    progress < 30%:  -1.0
```

**Target bonus**: `+2.0` when `target_progress_pct >= 100%` (terminate episode)

### Outcome Perturbation

- `outcome_noise_std=0.02` (default) → 2% Gaussian noise on pool outcomes during env step
- Purpose: prevent overfitting to fixed pool outcomes; simulates live slippage variance
- Only applies when pool is loaded (fresh-generation mode doesn't need it)

---

## 🛠️ Key Files & Line References

### State Synchronization (when modifying)

If you change **observation space**, update ALL of:
- `ml/signal_filter_env.py:95` — `observation_space = Box(shape=(24,))`
- `ml/signal_filter_env.py:141` — `_get_obs()` returns 24-dim array
- `ml/rl_agent.py:31` — `OBS_DIM: int = 24`
- `main.py:461` — `obs = np.array([...])` must be 24 elements

If you change **reward structure**:
- Tune in `ml/signal_filter_env.py:316` (step function)
- Impact: Phase 1 and Phase 2 share same env, differ only via `enable_risk_penalty` flag

### Critical Paths

- Pool file: `data/signal_pool_3000.pkl` (auto-loaded by env if exists)
- ML model: `data/signal_quality_model.pkl` (auto-loaded by backtester)
- RL model: `models/ppo_signal_filter.zip` (loaded by `SelfLearningAgent`)
- VecNormalize stats: `models/vec_normalize_sf.pkl` (MUST match model's train obs)
- TensorBoard: `logs/tb_signal_filter/phaseN_*/`

---

## ⚙️ Configuration (`config/settings.py`)

### FTMO Rules
```python
DAILY_LOSS_HARD_STOP_PCT = 0.04
MAX_DRAWDOWN_HARD_STOP_PCT = 0.08
TARGET_PROFIT_PCT = 0.10
MIN_CONFLUENCE_SCORE = 70.0  # runtime override-able
```

### Symbols (9)
`EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY`

### Env Variables
- `SMC_QUIET=1` → silences strategy debug prints (auto-set by train scripts)

---

## 🧪 Current State (2026-04-18)

### What's Working ✅
- SMC signal generation (all 9 symbols, 3 TFs)
- ML GBM classifier (AUC 0.58, meaningful edge at p>0.40)
- RL PPO training with Signal Pool (14k+ it/s on 8 workers)
- Hybrid obs (24 dims with ML score)
- 2-phase curriculum pipeline
- Pool caching on disk (reuse across train runs)
- TensorBoard logging (7 metric groups)
- Graceful fallback if ML model missing

### Known Limitations ⚠️
- SMC hand-crafted confluence is weak predictor (near-zero correlation with outcomes)
- ML model edge is moderate (AUC 0.58) — supervisory upper bound on RL performance
- BUY direction systematically weaker than SELL (30.4% vs 33.6%) — possibly market regime bias in training data
- Pool outcomes fixed per entry — small `outcome_noise=0.02` added for regularization

### Recent Post-Review Patches (2026-04-18)

**Resolver & Data Quality**:
- Bar color heuristic for same-bar SL/TP ties (was distance-from-open bias)
- Slippage/spread 2% → 0.5% (realistic major pair)
- `_end_idx_at_or_before` timestamp-based window alignment (was ratio-based look-ahead bias)
- FVG `analyze()` precondition documented

**Architecture**:
- Added ML quality layer → 24-dim obs (was 23)
- SubprocVecEnv for real multi-core parallelism
- Signal pool system (250× training speedup)
- HTF bias 3/5 window instead of unstable 2/3

**Training**:
- Gamma 0.95 → 0.99 (long horizon)
- clip_reward 10 → 20 (DD penalty room)
- SKIP oracle reward ×2 (balance TAKE:SKIP gradients)
- Network [128,64] → [256,128] (value approximation)
- outcome_noise parameter (0.05 → 0.02 default)

### Migration Notes

**Model version incompatibility**:
- Old `ppo_signal_filter.zip` (23 obs dim) — incompatible, auto-backup as `.bak_*`
- Old `signal_pool_3000.pkl` without `ml_score` — rebuild required
- Old `signal_quality_model.pkl` on old pool — works but retrain recommended

**Recommended fresh setup**:
```bash
# 1. Clean old artifacts
rm data/signal_pool_*.pkl

# 2. Full pipeline
python scripts/build_signal_pool.py --pool_size 3000 --workers 8
python scripts/train_signal_quality.py
python scripts/train_signal_filter.py --fresh --timesteps_p1 10000000 \
    --timesteps_p2 5000000 --n_envs 8 --pool_size 3000 --outcome_noise 0.02
```

---

## 📚 Decision Log (Reverse Chronological)

### 2026-04-18: Hybrid ML+RL Architecture
- **Motivation**: SMC confluence near-zero edge; RL couldn't filter
- **Change**: Train GBM on pool outcomes → inject `ml_score` as obs[16]
- **Impact**: Expected 45-55% WR (vs 21-31% baseline)

### 2026-04-18: Signal Pool System
- **Motivation**: Reset time ~6-9s per episode made training take 10+ hrs
- **Change**: Pre-generate 3000 episodes, cache to disk, env samples from pool
- **Impact**: Training 10.5 hrs → 25-30 min (250× speedup)

### 2026-04-18: Resolver Bias Fix
- **Motivation**: Same-bar SL/TP always resolved to SL (dist-from-open bias)
- **Change**: Bar color heuristic (green→TP first for BUY)
- **Impact**: Win rate 31.8% → 32.5% (+0.7pp)

### 2026-04-18: Large Network + Long Horizon
- **Motivation**: explained_variance stuck at 0.08 (value func not learning)
- **Change**: Net [128,64]→[256,128], gamma 0.95→0.99, clip_reward 10→20
- **Impact**: explained_variance 0.08 → 0.30 (P2)

### 2026-04-17: 2-Phase Curriculum
- **Motivation**: Agent needs to learn chart reading before DD management
- **Change**: Phase 1 no DD penalty + oracle SKIP; Phase 2 adds DD
- **Impact**: Better policy foundation

---

## 🔑 Invariants (Must Preserve)

1. **Obs dim sync** — env / rl_agent / main.py must all be 24
2. **Feature order** — must match across pool / env obs / main.py obs
3. **MT5 deal matching** uses `position_id`, not `order`/`ticket`
4. **Pool pickle** is source of truth for training (regenerate if architecture changes)
5. **Risk guard denominators**: Total DD uses `INITIAL_BALANCE`, Daily DD uses `daily_start_balance` (FTMO rules)
6. **Session times in config are UTC**, MT5 server is EET — convert before comparing
7. **ML model path default**: `data/signal_quality_model.pkl` (loaded by backtester, env, main.py)
8. **Pool file default**: `data/signal_pool_3000.pkl` (loaded by env, retrainable by `train_signal_quality.py`)

---

## 📞 Development Workflow

**ถ้าเจอ bug**:
1. Check TB metrics first (`tensorboard --logdir logs/tb_signal_filter`)
2. Verify obs dim consistency across files
3. Check pool has `ml_score` field: `pickle.load() → first_sig.get('ml_score')`
4. Check ML model loads: `ls data/signal_quality_model.pkl`

**ถ้าต้องเพิ่ม feature**:
1. Add to signal dict in `strategy_backtester.py:generate_episode_signals`
2. Add to obs in `signal_filter_env.py:_get_obs` + `main.py:_build_signal_observation`
3. Increment OBS_DIM in `rl_agent.py`
4. Rebuild pool + retrain ML + retrain RL

**ถ้าต้อง retrain ML อย่างเดียว** (pool unchanged):
```bash
python scripts/train_signal_quality.py  # re-scores pool in-place
```

**ถ้าต้อง retrain RL อย่างเดียว** (pool + ML unchanged):
```bash
python scripts/train_signal_filter.py --fresh  # uses existing pool + ML
```
