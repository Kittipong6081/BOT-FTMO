# 05 — Invariants & Gotchas (Rules Not to Break)
> Last Updated: 2026-04-24 | Scope: red flags, version log, migration notes

## TL;DR (30-second scan)

- Do not touch: obs dim / order, risk anchors, position_id matching, timezone handling.
- ⛔ Changing obs without retraining → whole system breaks.
- ⛔ Deleting `bot_state.json` mid-challenge → FTMO anchor destroyed.
- Every invariant below has already broken production once. Do not skip.

---

## ⛔ Hard Invariants (broken before → leave alone)

### 1. Observation Space Sync (3 places)

Changing obs requires retraining the whole pipeline (pool → ML → RL):

- `SelfLearningAgent.OBS_DIM` must equal `FTMOSignalFilterEnv.observation_space.shape[0]`.
- `FTMOTradingBot._build_signal_observation` must produce obs matching `FTMOSignalFilterEnv._get_obs` in size, order, and scale.
- On size mismatch: `SelfLearningAgent._prepare_obs` raises `ValueError` (good — fail fast).
- On wrong order with correct size: **no error**, but the model returns nonsense (more dangerous than a crash).

### 2. FTMO Anchors

- `RiskManager._initial_balance` is the Total DD anchor — **never** change mid-challenge.
- `RiskManager._daily_start_balance` is the Daily DD anchor — only resets at broker day rollover.
- Do not delete `logs/bot_state.json` mid-challenge — losing the anchor invalidates every DD %.

### 3. Position ID Matching

- MT5 deal matching uses `position_id` — **not** `order` or `ticket`.
- `RiskManager` + `TradeManager` + `TradeExecutor` must all reference the same field.

### 4. Timezone (EET vs UTC)

- Broker time = **EET** (Europe/Bucharest).
- Config values (session windows, Friday cutoff) = **UTC** — convert before comparing.
- Daily reset in `RiskManager` must use `TimeManager.get_server_time().date()` — not `date.today()`.
- ⛔ Do not use `mt5.symbol_info_tick().time` directly — FTMO sends broker-local epoch → double-adds tz = +3 h drift.
- ✅ Use `datetime.now(Europe/Bucharest)` inside `TimeManager.get_server_time` + NTP-synced VPS.

### 5. Pip Size (JPY-aware)

- JPY pairs (price > 20): pip = `0.01`.
- Others: pip = `0.0001`.
- ⛔ Do not hardcode `× 10000` — auto-detect from `entry_price` or `symbol_info.digits`.

### 6. PositionSizer Pip Value (3 cases)

- Quote = account CCY (EURUSD, GBPUSD, ...) → raw pip value.
- Base = account CCY (USDJPY, USDCHF) → raw / symbol_price.
- Cross (EURJPY, GBPJPY) → use **USDJPY rate** (not the cross pair's own price).

### 7. Contract Size

- Use real `symbol_info.trade_contract_size` — do not hardcode `100_000`.
- XAUUSD = 100 oz (digits = 2) — `PositionSizer` must handle this.

### 8. Correlation Groups

Inside `TradeExecutor._check_correlation`:

- USD_WEAK, USD_STRONG, JPY_CROSS, EUR_PAIRS, GBP_PAIRS.
- `MAX_CORRELATED_POSITIONS` per group per direction (default 1).
- Duplicate symbols may not be opened twice.

### 9. FTMO Program Type

- Current = **2-step Standard** → `CONSISTENCY_RULE_THRESHOLD = 1.0` (check disabled).
- Swing/Pro = 0.45 (max day ≤ 50 % of total profit).
- ⚠️ Switching programs requires updating this value before starting the new challenge.

### 10. ATR Floor vs MIN_SL — separate mechanisms

- `SymbolConfig.symbol_overrides[X].atr_floor_pips` = **signal gate** inside `SMCStrategy.scan_signal`. If `atr_pips < floor` → drop signal. Does **not** touch SL.
- `SymbolConfig.symbol_overrides[X].min_sl_pips` = **SL clamp** inside `SMCStrategy` BUY/SELL branches (after OB override). Prevents spread from eating > ~15 % of SL.
- `bot_config.indicators.atr_sl_multiplier` = global ATR → SL base multiplier (1.5).
- ⛔ Do not merge these three into one. Lowering `atr_floor_pips` widens the accepted-signal population but does not narrow SL directly — SL shape is owned by `atr_sl_multiplier` + `min_sl_pips`.

---

## ⚠️ Soft Invariants (best practice)

- **Risk per trade**: train with `--risk_per_trade 0.007` → live `DEFAULT_RISK_PER_TRADE_PCT = 0.007` (must match).
- **Eval sample size**: 100 eps has ±5 pp variance — use ≥ 500 eps for true performance.
- **Pool + ML + RL dependency**: changing obs → rebuild pool → retrain ML → retrain RL (order matters).
- **VecNormalize stats**: `models/vec_normalize_sf.pkl` must match `models/ppo_signal_filter.zip` (otherwise obs is in the wrong scale).
- **Quality-first P1 reward**: Phase 1 relies on the oracle SKIP reward (×2) — do not dampen, or the agent will over-trade.

---

## 🔄 Migration Notes

### Changing Obs Space (e.g. 24 → 27)

1. Update `FTMOSignalFilterEnv.observation_space` (shape tuple).
2. Update `FTMOSignalFilterEnv._get_obs` (compute new features + return array).
3. Update `SelfLearningAgent.OBS_DIM` to match.
4. Update `FTMOTradingBot._build_signal_observation` (order + scale must match env).
5. Rebuild the pool (`build_signal_pool.py`) if the new features come from the signal dict.
6. Retrain ML (`train_signal_quality.py`) if the new features affect the GBM input.
7. Retrain RL `--fresh` (`train_signal_filter.py`).
8. Back up the old model: `mv models/ppo_signal_filter.zip models/ppo_signal_filter.zip.bak_<timestamp>`.
9. **Update wiki**: `03-rl-training.md` (obs table), `02-modules.md` (if any module signature changed), `context.md` (headline numbers).

### Starting a New FTMO Challenge

- Back up `logs/bot_state.json` → `logs/bot_state.json.bak_<timestamp>` (do not delete).
- Update `.env` if using a new account: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`.
- Start the bot — `RiskManager._load_state` creates a fresh state automatically.
- Log check: "🆕 [Risk Manager] เริ่ม Challenge ใหม่..." (log text may be Thai, code is shared).

### Switching FTMO Program

- Edit `FTMOConfig.CONSISTENCY_RULE_THRESHOLD` (Standard = 1.0, Swing/Pro = 0.45).
- Review `CONSISTENCY_MIN_PROFIT_PCT` (default 0.02) — adjust if needed.

---

## 📚 Version Log (reverse chronological)

### 2026-04-24 — v6.3 B1v2 mid-episode undertrading checks

Follow-up to B1 (Pass Rate 2.0 % → 3.7 %, but take rate ไม่ขยับ). Problem: terminal undertrading penalty fires too late for PPO credit assignment. B1v2 adds checks WITHIN the episode.

**RL reward changes (`ml/signal_filter_env.py`):**

- **Mid-episode check day 20 (P2, sticky):** progress < 40 % + takes < 6 → `-0.3`. Fires once per episode.
- **Mid-episode check day 35 (P2, sticky):** progress < 60 % + takes < 12 → `-0.7`. Fires once per episode.
- **Terminal threshold lowered:** `takes < 20` → `takes < 15`. Realistic for pool size (~16 signals at th0.36).
- **Progress shaping strengthened:** `0.02 × progress_delta` → `0.05 × progress_delta` (2.5×). Every 1 % toward target gives more reward.
- State flags added: `_mid_check_day20_fired`, `_mid_check_day35_fired` — reset per episode.

Expected: Pass Rate 3.7 % → 5-7 %, Take Rate 65 % → 72 %+, Orders/day 0.26 → 0.35+.

### 2026-04-24 — v6.3 B1 reward tuning (Pass Rate improvement)

Phase B1 targets low Pass Rate (2.0% at v6.3 baseline) — agent undertrades (10.4 orders/ep vs needed ~27 for 10% target). Only RL env changed; no pool/GBM rebuild needed.

**RL reward changes (`ml/signal_filter_env.py`):**

- **Milestone bonuses (sticky):** +0.5 at 30 % progress, +1.0 at 60 %, +1.5 at 90 %. Teaches agent that partial progress is rewarded, not just terminal target.
- **Target bonus raised:** first 100 % hit `+2.0 → +4.0`. Combined with milestones, hitting target gives total sticky `+7.0` vs prior `+2.0`.
- **Undertrading penalty (P2 only):** episode end + `_current_day ≥ 40` + `_total_takes < 20` + not passed → `-1.0`. Forces agent to use available trades instead of hoarding.
- **Passive SKIP cost reduced:** `-0.015 → -0.010` per step. Allows honest SKIPs on low-quality signals without excessive punishment.
- State flags added: `_milestone_30_given`, `_milestone_60_given`, `_milestone_90_given` — reset per episode via `_reset_state`.

**Retrain required (only RL, not pool/GBM):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected: Pass Rate 2 % → 5-8 %, Take Rate 65 % → 75 %+, Orders/day 0.26 → 0.40+.

### 2026-04-24 — v6.3 multi-brain audit fixes

Full 3-brain audit (SMC + ML + RL). 16 Critical fixes landed. Requires pool rebuild + ML retrain + RL retrain before live deployment.

**SMC (`smc_strategy.py`):**

- `_evaluate_sell_signal` now applies `atr_floor_pips` per-symbol override (was hardcoded 100/8, ignoring `SymbolConfig.symbol_overrides`).
- BUY signal `timestamp` uses `TimeManager.get_server_time(symbol)` (was `datetime.now()` — local machine offset).
- HTF bias anti-lookahead: `iloc[-6:-1]` instead of `iloc[-5:]` — skips currently-forming bar.
- EMA200 veto NaN guard (`pd.notna`) — prevents silent skip on fresh MTF data.
- `_get_d1_bias` cache invalidates on UTC day rollover.
- `min_sl_pips` clamp now prints debug warning.

**ML (`scripts/train_signal_quality.py`, `ml/strategy_backtester.py`):**

- `train_gbm` uses `GroupKFold` + `cross_val_predict` — episode-level OOF predictions, unbiased AUC, no in-sample leakage into RL.
- Pool's `ml_score` field re-scored with OOF probabilities.
- `_resolve_trade` gap handling — `bar_open` past SL/TP fills at `bar_open` with slippage.
- Pool `MIN_CONFLUENCE_SCORE` aligned with live (`bot_config.ftmo.MIN_CONFLUENCE_SCORE`) — was 60, now 70.

**RL (`ml/signal_filter_env.py`, `main.py`):**

- `FTMOSignalFilterEnv.reset` injects `spread_pips × uniform(0.7, 1.5)` — spread distribution shift fix.
- `spread_cost_R` clamped at 1.0 — prevents reward explosion on news-spike spreads.
- `obs_dim()` static method returns 27 (was stale 24).
- `main.py._has_opposite_recently_closed` logs warning when `RiskManager._flip_lock` missing — prevents silent obs[25]=0.

**Mandatory before live deploy:**

1. `python scripts/build_signal_pool.py --pool_size 3000` (pool has v6.2 min_sl_pips + v6.3 confluence=70 + gap handling)
2. `python scripts/train_signal_quality.py` (OOF AUC expected to drop vs prior ~0.59 — that's the leakage being removed)
3. `python scripts/train_signal_filter.py --fresh --timesteps_p1 10000000 --timesteps_p2 5000000 --n_envs 8 --pool_size 3000`
4. Back up old `models/ppo_signal_filter.zip` + `vec_normalize_sf.pkl` first.

### 2026-04-24 — ATR floor re-calibrated + MIN_SL guard (v6.2)

- Lowered `SymbolConfig.symbol_overrides[X].atr_floor_pips` from 8/15/20/100 → 3-8 pips FX (500 ticks XAUUSD). Prior floor blocked signals 68-100 % of the time.
- Added new per-symbol `min_sl_pips` (EURUSD 10, GBPUSD/USDCAD/USDCHF/NZDUSD 12, USDJPY 10, EURJPY 15, GBPJPY 20, XAUUSD 300) as an SL floor inside `SMCStrategy.scan_signal` BUY/SELL branches.
- Incident that prompted the guard: EURUSD SL collapsed to ~5 pips under low ATR, spread ate > 20 % of SL.
- Invariant added (Hard Invariant #10): `atr_floor_pips` (gate) and `min_sl_pips` (SL clamp) are distinct — do not conflate.
- Commit: `9b64f6c`.

### 2026-04-24 — LLM Wiki migration

- Migrated `context.md` from a 392-line monolith to Hub + Spoke (5 files under `wiki/`).
- Added `CLAUDE.md` at project root — Wiki Sync Protocol.
- Installed Stop hook in `.claude/settings.json` — warns if `.py` changed but wiki/context/readme did not.
- Switched source-reference style from line numbers to class / method / variable names.
- Set language policy: docs in English, `readme.md` in Thai, chat in Thai.

### 2026-04-22 — Obs Space v6 (24 → 27 dims)

- Added `spread_pct_of_atr` [24] (cost awareness, GBPJPY spread vs ATR).
- Added `has_opposite_recently_closed` [25] (flip-lock context, anti-whipsaw).
- Added `htf_trend_alignment` [26] (uses `bias_align` as proxy).
- Retrained the entire pipeline.

### 2026-04-20 — Risk 0.7 % verified (5000 eps)

- Changed `DEFAULT_RISK_PER_TRADE_PCT` from 0.006 → 0.007.
- Evaluation: Pass 12.5 %, Profit +2.59 %, DD max 8.50 %, Breach 0 %.
- Discovered MT5 FTMO `tick.time` quirk → switched to `datetime.now(Bucharest)`.

### 2026-04-19 — Added XAUUSD

- `SymbolConfig.symbols` went from 9 to 10 (added XAUUSD, Gold).
- PositionSizer supports contract size 100 oz.

### 2026-04-18 — Hybrid ML + RL + resolver fixes

- Added ML GBM quality layer → obs 23 → 24 dims.
- Resolver now uses bar-color heuristic (fixes distance-from-open bias).
- Slippage 2 % → 0.5 % (realistic on majors).
- Pool system: fresh-generate → 3000 pre-generated episodes (250× training speedup).
- HTF bias: unstable 2/3 → stable 5-bar ≥3 same side, <2 opposite.
- Network `[128,64]` → `[256,128]`, gamma 0.95 → 0.99.

### 2026-04-17 — 2-phase curriculum

- Phase 1 (Alpha): no DD penalty + oracle SKIP reward.
- Phase 2 (Risk): DD penalty + activity floor.

### 2026-04-13 — Code review patch

- Fixed timezone bugs (EET vs UTC).
- Fixed pip size to be JPY-aware.
- Fixed PositionSizer 3-case pip value.
- Fixed hardcoded contract size.

---

## Cross-links

- Architecture + loop priority → [01-architecture.md](01-architecture.md)
- Modules + symbols → [02-modules.md](02-modules.md)
- Obs 27 dims layout → [03-rl-training.md](03-rl-training.md)
- Live operations + state machine → [04-operations.md](04-operations.md)
