# 02 — Modules Map (30+ files)
> Last Updated: 2026-05-07 (v8.0.8 — SMC fully removed, MR live default) | Scope: every module + key class / method / variable
>
> **v8.0 new modules (MR pivot, code staged — not wired to live)**:
>
> - `strategy/mean_reversion_strategy.py` — `MeanReversionStrategy`, `MRSignal`, `MRSignalType`
> - `ml/mean_reversion_backtester.py` — `MeanReversionBacktester(StrategyBacktester)`
> - `ml/mean_reversion_env.py` — `MeanReversionFilterEnv(FTMOSignalFilterEnv)`
> - `scripts/build_mr_signal_pool.py`, `scripts/train_mr_signal_quality.py`, `scripts/train_mr_signal_filter.py`
> - `scripts/auto_train_pipeline.py` — autonomous orchestrator (Build pool → GBM → RL → Eval → Self-correct loop)
> - `config/settings.py::MeanReversionConfig` — `bot_config.mr.strategy_mode` (`"smc"` default; flip to `"mean_reversion"` after MR eval gates pass)
>
> **v7.1 new methods/symbols (awaiting retrain)**:
> - `RiskManager.get_unrealized_drawdown_pct`, `RiskManager.check_unrealized_circuit_breaker`
> - `TradeExecutor._populate_close_metadata`, `TradeExecutor.USD_THEME_DIR`, `TradeExecutor.MAX_USD_THEME_POSITIONS`
> - `SMCStrategy._is_session_warmup`, `_is_post_weekend_window`, `_check_spread_atr_ratio`, `_required_confluence`, `_compute_dynamic_sl_multiplier`
> - `TechnicalIndicators.classify_volatility_regime`, `TechnicalIndicators.compute_atr_zscore_30bars`
> - `SignalQualityModel.detect_drift`, `SignalQualityModel.record_live_signal`, `SignalQualityModel._train_dist`
> - `compute_temporal_features` (module-level in `ml/signal_quality.py`)
> - `FTMOTradingBot._check_gbm_drift`, `_compute_floating_pnl_norm`, `_compute_open_losing_count_norm`, `_compute_mins_since_session_norm`
> - `scripts/chronos_distribution_audit.py` — new audit script
> - `FTMOConfig.UNREALIZED_PAUSE_PCT`, `UNREALIZED_PAUSE_MIN_OPEN`, `MAX_USD_THEME_POSITIONS`, `SPREAD_ATR_RATIO_LIMIT`

## TL;DR (30-second scan)

- 7 layers: **strategy / ml / core / execution / config / analytics / scripts** + `main.py`.
- Finding code: look up the symbol in the tables below, then grep or jump to the file.
- Mission-critical modules: `RiskManager`, `MT5Connector`, `TradeExecutor`, `SMCStrategy`, `SelfLearningAgent`, `FTMOSignalFilterEnv`.
- Every config value and symbol is referenced by name — never by line number (line numbers rot quickly).

---

## 🎯 strategy/ — MR Brain (v8.0.6 cleaned)

| File | Key symbols | Role |
|------|-------------|------|
| `indicators.py` | `TechnicalIndicators.atr`, `.ema`, `.rsi`, `.macd`, `.adx`, `.stoch`, `.bb` | Indicator calculation on numpy arrays |
| `mean_reversion_strategy.py` (v8.0+) | `MeanReversionStrategy.analyze_with_data`, `MRSignal`, `MRSignalType`, `LiveMRScanner` (live entry), `TradeSignal=MRSignal` (legacy alias), `SignalType=MRSignalType` (legacy alias) | The whole strategy stack: BB %B extreme + RSI confirm + ADX H1 trend block + reversal-wick. RR 1:1, SL = 1.0×ATR (tight). `LiveMRScanner` is the drop-in replacement for `SMCStrategy` in main.py. |

**v8.0.6 cleanup (2026-05-07)** — SMC source files deleted: `smc_strategy.py` (and `.bak_v6.13`), `order_blocks.py`, `fair_value_gaps.py`, `liquidity_sweeps.py`, `inducement.py`, `market_structure.py`, plus `tests/test_order_blocks.py`. Live runtime uses MR exclusively. `TradeSignal` and `SignalType` are still importable from `mean_reversion_strategy` as legacy aliases (for `trade_executor` and historical test code).

### SL Formula & ATR Floor (4 separate knobs — do not confuse, v6.14)

SL is built inside `SMCStrategy.scan_signal` (BUY branch + SELL mirror) in four stacked steps:

1. **Base (v6.14 per-symbol)** — `sl_atr_mult = get_symbol_config(symbol, "sl_atr_multiplier", bot_config.indicators.atr_sl_multiplier)` then `sl_distance = atr_value × sl_atr_mult`. XAUUSD = 1.8×, FX default = 1.5×.
2. **OB override (v6.14 floored)** — if a recent Order Block is close and `ob_sl_floor < ob_sl_distance < sl_distance × 1.5`, swap in `ob_sl_distance`. `ob_sl_floor = atr_value × 0.5` prevents OB clamp from collapsing SL below 0.5×ATR.
3. **MIN_SL guard (v6.2 / raised v6.14 for XAU)** — if `sl_distance < min_sl_pips × pip_size`, bump up to the floor. Prevents spread from eating > ~15 % of SL.

Four knobs with distinct roles — **never conflate them**:

| Knob | Where | Role |
|------|-------|------|
| `SymbolConfig.symbol_overrides[X].atr_floor_pips` | gate at top of `SMCStrategy.scan_signal` | **Signal gate** — if `atr_pips < floor`, the signal is discarded (dead-market filter). Does **not** affect SL width of accepted trades. |
| `bot_config.indicators.atr_sl_multiplier` | base SL formula (fallback) | Global ATR-to-SL ratio (1.5 × ATR). Used only when symbol has no `sl_atr_multiplier` override. |
| `SymbolConfig.symbol_overrides[X].sl_atr_multiplier` (v6.13/v6.14-wired) | base SL formula (per-symbol) | Per-symbol override — XAUUSD 1.8×. Wired into SMCStrategy in v6.14 (was config-only before). |
| `SymbolConfig.symbol_overrides[X].min_sl_pips` | clamp after OB override | **SL floor** — per-symbol minimum to keep spread-to-SL ratio sane (EURUSD 10, GBPJPY 20, XAUUSD **1000 ticks** raised from 300 in v6.14). |

**Want SL narrower?** Lower `sl_atr_multiplier` (per-symbol) or `min_sl_pips` — lowering `atr_floor_pips` only widens the accepted-signal population (indirect, clamped by `min_sl_pips`).

### Live Demo Bug (v6.14 fix) — XAU SL collapsed to 0.28×ATR

Before v6.14, three layers compounded to let live XAUUSD SL drop to $3.11 (= 0.28×ATR) on ticket `437211678` (SL hit in 12 s):

- Layer A — `SMCStrategy` ignored per-symbol `sl_atr_multiplier` (used global 1.5× even for XAU).
- Layer B — OB clamp had no lower bound, so an OB very close to entry replaced base SL.
- Layer C — `min_sl_pips: 300` for XAUUSD = $3 floor — too low for Gold (ATR M15 8-15 USD).

v6.14 fixes all three layers; cross-link → [`wiki/05-invariants.md` v6.14 Version Log](05-invariants.md#-version-log-reverse-chronological).

### v6.7 Rollback Phase D (2026-04-25)

Both Phase D variants (full BE+partial+trail, and BE-only) reduced Pass Rate below the B1v2 baseline (3.7 %). Reverted to v6.3 B1v2 backtester (`_resolve_trade` uses only SL / TP / timeout / force-close / gap — no BE, no partial, no trailing inside training). **Live `TradeManager` still performs BE + partial + trail** — train-live gap is accepted as "train < live" direction.

### v6.5 Phase D — Train-Live Alignment (REVERTED 2026-04-25)

**Kept as historical reference only.** Evidence: BE-only dropped pool mean outcome from `−0.0645` → `−0.1051` (8.4 pp of winners became 0R vs 9.6 pp of losers saved — net EV negative). Phase D full caps tail (winners capped at 1.5R instead of 2R+) which is the wrong direction for FTMO's tail-probability objective.

- **`StrategyBacktester._resolve_trade`** was rewritten as bar-by-bar state machine (removed in v6.7).
- Class constants `_BE_TRIGGER_RR`, `_PARTIAL_CLOSE_PCT`, `_PARTIAL_TRIGGER_RR`, `_TRAIL_ACTIVATION_RR`, `_TRAIL_ATR_MULTIPLIER` — removed in v6.7.

### v6.4 SMC Phase C — 4 Professional Principles (REVERTED 2026-04-25)

**This block was rolled back** after empirical results (Pass Rate 3.7 % → 1.5 %). Kept here as historical reference only; none of these changes exist in the current codebase.

- **`SMCStrategy._get_h4_poi_zones` / `_is_near_h4_poi`** (removed) — H4 POI cache + gate.
- **`SMCStrategy._ob_detector_h4` / `_fvg_detector_h4` / `_h4_poi_cache`** (removed).
- **`MarketStructure.is_valid_pullback`** (removed) — 3-gate validator.
- **IDM sweep gate** (removed) — sweep + OB bonus / old OB penalty.
- **FVG + BOS conjunction** (removed).
- **ADX floor** reverted 25 → 20.

### v6.11 SMC Precision Overhaul (2026-04-29) — see `wiki/05-invariants.md` Version Log

- **`SMCStrategy._evaluate_buy_signal/_evaluate_sell_signal`** — pre-filter chain ตอนนี้: A (ATR floor) → B (RSI zone) → C (HTF+MTF align) → D (EMA200 H1) → E (ADX H1 ≥ 20) → **E2 (ADX H4 ≥ 22)** → **F (Counter-D1 hard veto)** → **F2 (Quiet-vol × off-overlap)** → **G (Recent sweep within 8 bars)** → **H (Fresh M15 BOS/CHoCH within 6 bars)**.
- **`TradeSignal`** dataclass — เพิ่ม fields: `htf_score, mtf_score, ob_pts, fvg_pts, sweep_pts, sweep_age_bars, htf_bias` (string label "BULLISH/BEARISH/RANGING"), `d1_bias`. Populate ใน BUY/SELL eval โดย track per-component contribution แยกจาก main `score` accumulator.
- **`SMCStrategy._idm_detector`** — instance ของ `InducementDetector(lookback=8)`. ใช้ใน confluence section (factor 3.7): IDM = +10, ไม่มี IDM = -5.
- **`OrderBlock.ob_grade: str`** — `"EXTREME"` / `"DECISIONAL"` / `"INTERNAL"`. classified ใน `OrderBlockDetector._classify_ob_grade(ob, df, avg_impulse)`.
- **`OrderBlockDetector._score_order_blocks`** — apply grade weight: EXTREME ×1.20, DECISIONAL ×1.00, INTERNAL ×0.60 (หลังคะแนน 4 ปัจจัยเดิม).
- **`TradeManager._manage_single_position`** — track `state.best_price` ทุก tick, BE/Partial trigger ใช้ `best_rr` (rolling MFE-based) แทน `current_rr`. Trailing ยังใช้ `current_rr` (เพราะต้อง confirm trend ต่อ).
- **`TradeManager._partial_close`** lot_min branch — mirror `trade.partial_closed_flag = True` (สอดคล้องกับ `partial_close_skipped`).
- **`FTMOTradingBot._build_live_context`** — อ่าน `htf_score/mtf_score/ob_pts/fvg_pts/sweep_pts/htf_bias/d1_bias/mtf_bias` จาก `signal` ตรงๆ (ไม่ใช่ hardcode 0).
- **`FTMOTradingBot._log_signal_scan`** — `htf_bias` field ใช้ `sig.htf_bias` (string) แทน `_strategy._htf_bias` (int).

### v6.11.1 Backtester parity fix (2026-04-29 evening)

- **`StrategyBacktester._init_strategy`** — เพิ่ม `self._strategy._idm_detector = InducementDetector(lookback=8)` (ตรงกับ `SMCStrategy.__init__`). กัน `AttributeError` ตอน `analyze_with_data` → `_evaluate_buy/sell_signal` เรียก IDM ใน factor 3.7. **CRITICAL** — ถ้าไม่แก้ `build_signal_pool.py` รันไม่ได้.

### v6.11.2 Partial rollback Tier 2.2 + 2.3 hard gates → soft bonuses (2026-04-29 evening)

หลัง rebuild pool ด้วย v6.11 hard gates → Pass Rate 0.0 % (pool หาย 99 %). ลด strict gate กลับเป็น scoring:

- **`SMCStrategy._evaluate_buy_signal/_evaluate_sell_signal`** — ลบ pre-filter G (Sweep within 8 bars) + pre-filter H (Fresh M15 BOS within 6 bars). Pre-filter chain เหลือ: A → B → C → D → E (ADX H1) → E2 (ADX H4) → F (Counter-D1) → F2 (Quiet-vol × off-overlap). Sweep ยังถูก score เป็น bonus ใน factor 3.6 (max +15)
- **`SMCStrategy._evaluate_buy/sell_signal` factor 2.5 (NEW)** — ถ้า `_structure_ltf.get_latest_event()` เป็น bullish BOS/CHoCH ภายใน 6 bars (BUY) หรือ bearish (SELL) → +5 confluence bonus. Rolled into `mtf_pts` สำหรับ logging

### v6.11.3 Mild relaxation tune (2026-04-29 evening)

หลัง v6.11.2 Pass Rate 2.7 % (math: 6.7 trades × 0.68 % = 4.6 % expected, ห่างเป้า 10 %). Tune 2 จุดเพื่อเพิ่ม signals + retrain:

- **`SMCStrategy._evaluate_buy/sell_signal` factor 3.7 IDM penalty** — `score -= 5` → `score -= 2` ถ้าไม่เจอ IDM rejection. Mild penalty ลด over-penalize ใน calm market
- **`SMCStrategy._evaluate_buy/sell_signal` pre-filter E2 ADX H4 floor** — `< 22.0` → `< 20.0` ให้สอดคล้องกับ ADX H1 floor

ผลหลัง rebuild pool (78k→90k signals) + retrain GBM (AUC 0.5915) + retrain RL (10M+5M, 23.6 min): Pass Rate 2.7 → **3.4 %** (+26 %), WR 65.6 → **68.8 %** (+3.2 pp), DD max 4.46 → **3.23 %** (-28 %, safer). Pure improvement ทุกมิติ.

### v6.3 SMC fixes (2026-04-24 audit)

- **SELL `atr_floor_pips` per-symbol override now applied** — previously `_evaluate_sell_signal` hardcoded 100/8, ignoring `SymbolConfig.symbol_overrides`. BUY + SELL now share the same gate.
- **BUY `timestamp` uses `TimeManager.get_server_time(symbol)`** — previously `datetime.now()` (local machine time). SELL was already correct.
- **HTF bias anti-lookahead** — `analyze_with_data` + `analyze` now use `htf_df["trend"].iloc[-6:-1]` (5 closed bars, excludes the currently-forming bar). Old `iloc[-5:]` leaked forming-bar data into backtests.
- **EMA200 NaN guard** — H1 EMA200 veto in BUY + SELL now wraps `pd.notna(ema200_h1) and ema200_h1 > 0 and ...`. NaN values (fresh MTF data < 200 bars) previously short-circuited the veto silently.
- **D1 bias cache invalidates on UTC day rollover** — `_get_d1_bias` no longer returns stale bias for up to 1 hour after midnight UTC (was previously caching by TTL alone).
- **`min_sl_pips` clamp now logs** — when SL width gets bumped up to the floor, `SMCStrategy` prints a warning (respects `bot_config.debug_mode`).

---

## 🔬🎓 ml/ — ML + RL Brain

| File | Key symbols | Role |
|------|-------------|------|
| `signal_quality.py` | `SignalQualityModel.score`, `SignalQualityModel.train_from_pool`, `SignalQualityModel._calibrate` | sklearn `GradientBoostingClassifier` + `IsotonicRegression` calibrator. `score()` returns calibrated probability. v8.0 uses `data/mr_signal_quality_model.pkl` (28-feature schema with MR extras: `bb_extreme`, `bb_band_width_atr`, `mr_setup_score`, `reversal_wick_ratio`). |
| `strategy_backtester.py` (v8.0.6 cleaned) | `StrategyBacktester` (data infra base class only — `_load_data`, `_precompute_indicators`, `_resolve_trade`, `_MockConnector`). `generate_episode_signals` raises `NotImplementedError` — subclass must implement. | SMC-specific code REMOVED in v8.0.6 (was `_init_strategy`/SMC `generate_episode_signals`/`simulate_day_*`/`_run_day_scan`). Now ~327 lines vs original 1055. Subclasses: `MeanReversionBacktester` only. |
| `signal_filter_env.py` | `FTMOSignalFilterEnv` (gym.Env, **shape=(32,)**), `._get_obs`, `.step`, `.reset`, `._is_correlation_blocked`, `.CORRELATION_GROUPS`, `.HOLD_SIGNALS_APPROX = 0`, **`DAILY_DD_GUARD = 0.030` (v8.0.4)**, **`TOTAL_DD_GUARD = 0.058` (v8.0.5)** | Gymnasium env. `step()` writes `info['aux_target'] = float(sig['outcome_pnl_ratio'])` for the aux head. `_get_obs` reads chronos features for obs[27,28]. v8.0.6: imports `MeanReversionBacktester as StrategyBacktester` (live env uses MR backtester). v8.0.4/5: tightened env guards under our gates so DD never pins at the ceiling. |
| `mean_reversion_env.py` (v8.0) | `MeanReversionFilterEnv(FTMOSignalFilterEnv)`, `.step` (overridden), `._get_obs` (overridden 3 slots), `QUICK_TP_BARS=5`, `QUICK_TP_BONUS=0.50`, `SLOW_WIN_BONUS=0.20`, `PROLONGED_LOSS_BARS=12`, `PROLONGED_LOSS_PENALTY=0.40`, `BASE_LOSS_PENALTY=0.10`, `DURATION_FINE_COEF=0.02`, `ADX_VIOLATION_PENALTY=0.30` | RL env for MR. Same 32-dim obs shape, reinterprets `obs[4]=bb_extreme`, `obs[10]=bb_band_width_atr/3`, `obs[26]=adx_inverse_norm`. Reward shaping per v8.0 spec: quick-TP bonus (≤5 bars +0.50R), slow-win +0.20R, base-loss -0.10R + per-bar duration fine 0.02R (cap -0.30R) + prolonged-loss extra -0.40R, ADX violation -0.30R. |
| `mean_reversion_backtester.py` (v8.0) | `MeanReversionBacktester(StrategyBacktester)`, `.generate_episode_signals`, `._bars_to_first_hit`, `MR_SCAN_POINTS_PER_DAY=48` (v8.0.2 = every 30min), `MR_FUTURE_BARS=32`, `DEDUP_BARS=4` | Pool builder for MR. Scans every 30min (48/day) — BB extremes are time-sensitive. `DEDUP_BARS=4` prevents same-direction signal flooding. Adds `bars_to_resolution` + `is_quick_tp` per signal for env reward shaping. Pool path: `data/mr_signal_pool_<N>.pkl`. |
| `aux_aware_policy.py` | `AuxAwareACPolicy(ActorCriticPolicy)`, `aux_head: nn.Linear(latent_dim_pi, 1)`, `predict_aux(obs)` | PPO policy extended with auxiliary regression head off the actor trunk. Predicts `outcome_pnl_ratio`. |
| `aux_aware_ppo.py` | `AuxAwarePPO(PPO)`, `aux_loss_weight=0.5` | PPO subclass. `collect_rollouts` extracts `info['aux_target']` into `AuxRolloutBuffer.aux_targets`; `train()` adds `MSE(aux_pred, aux_target) * 0.5` to the policy loss. |
| `aux_rollout_buffer.py` | `AuxRolloutBuffer(RolloutBuffer)`, `.aux_targets: np.ndarray` | RolloutBuffer extended with per-step `aux_targets` (parallels `rewards`/`values`). |
| `rl_agent.py` | `SelfLearningAgent.OBS_DIM = 32`, `.should_take_signal`, `.get_action_confidence`, `.initialize_model`, `._normalize_obs` | PPO inference wrapper. **v8.0 path-aware**: tries `models/mr/best/ppo_mr_filter.zip` (+ `vec_normalize_mr.pkl`) first, fallback to legacy `models/ppo_signal_filter.zip`. Uses `AuxAwareACPolicy` weights at inference (aux head ignored at predict). |
| `chronos_forecaster.py` | `ChronosForecaster.__init__`, `.forecast(symbol, df_m15)`, `.compute_features(forecast, direction, atr)`, `.forecast_features` (convenience), `.is_available` | Amazon Chronos 2 zero-shot wrapper (`amazon/chronos-bolt-small`). M15 × 8 ahead median + q10 + q90 → 2 obs features (`chronos_alignment`, `chronos_uncertainty_norm`). Cache key `(symbol, last_bar_ts)`, LRU evict @ 64 entries. Determinism via `torch.manual_seed(0)`. Disable: env `BOT_DISABLE_CHRONOS=1` or `bot_config.ml.CHRONOS_ENABLED = False`. v7.2 formula: un-flip alignment (`delta = median - close`). |

**Quality model loading**: `StrategyBacktester.__init__` auto-loads `data/mr_signal_quality_model.pkl` first, falls back to legacy `data/signal_quality_model.pkl` if MR not present. Payload format: `{"model": GBM, "calibrator": IsotonicRegression, "keys": [...], "train_dist": {...}, "strategy": "mean_reversion"}`.

### v6.3 ML + RL fixes (2026-04-24 audit)

- **Anti-leakage GBM training** — `train_signal_quality.py` now uses `GroupKFold` (group = episode index) with `cross_val_predict` to produce OOF probabilities. The pool's `ml_score` field is re-scored with **OOF predictions**, not in-sample. Quoted AUC is now an unbiased OOS estimate.
- **Gap handling in `_resolve_trade`** — when `bar_open` is already past SL or TP (weekend gap, news), outcome is filled at `bar_open` with slippage/spread cost, not at SL/TP price.
- **Pool confluence threshold aligned with live** — `generate_episode_signals` now uses `bot_config.ftmo.MIN_CONFLUENCE_SCORE` (live threshold) instead of hardcoded 60. Pool + live see the same obs[0] distribution.
- **Spread noise in training env** — `FTMOSignalFilterEnv.reset` jitters `spread_pips` by `uniform(0.7, 1.5)` once per episode. obs[24] `spread_pct_of_atr` + reward `spread_cost` both see realistic variance.
- **`spread_cost_R` clamp** — reward deduction now bounded at `min(spread/sl, 1.0)` × 0.5, capping spread penalty at -0.5R (was unbounded, could exceed -1R on news-spike spreads).
- **`FTMOSignalFilterEnv.obs_dim()` returns 27** — was stale `24`. Matches `observation_space.shape[0]`.
- **`_has_opposite_recently_closed` logs flip_lock missing** — `main.py` warns once if `RiskManager._flip_lock` is None, preventing silent obs[25]=0 forever.

---

## 🛡️ core/ — Risk, Connection, Notification

### `risk_manager.py` — FTMO compliance engine (CRITICAL)

| Symbol | Role |
|--------|------|
| `BotState` (Enum) | ACTIVE / DAILY_HALT / MAX_DRAWDOWN_HALT / MANUAL_HALT / DISCONNECTED |
| `RiskManager.check_state` | Pre-trade gate — returns False when halted |
| `RiskManager.get_risk_status` | Returns dict: `daily_loss_pct`, `overall_drawdown_pct`, `current_balance` |
| `RiskManager._initial_balance` | FTMO anchor — total DD is measured against this |
| `RiskManager._daily_start_balance` | Daily anchor — daily DD is measured against this |
| `RiskManager._save_state` / `._load_state` | Persistence via `logs/bot_state.json` (schema v4) |

**State file schema** (`logs/bot_state.json`):

- `initial_balance`, `state`, `highest_balance`, `current_day`, `daily_closed_pnl`, `consecutive_losses`, `halt_until`, `daily_pnl_history`, `mt5_login` (v4), `challenge_start_date` (v4), `schema_version` (v4).
- **Validation**: `mt5_login` mismatch → reset; balance diff > 20 % → warn only (never auto-reset).

### `mt5_connector.py` — MT5 API wrapper

| Symbol | Role |
|--------|------|
| `MT5Connector.connect` / `.disconnect` | MT5 login (mock fallback when library missing) |
| `MT5Connector.get_account_info` | Balance, equity, margin |
| `MT5Connector.get_ohlcv` | Fetch candles (M15 / H1 / H4) |
| `MT5Connector.place_order` | Market order with SL/TP |
| `MT5Connector.close_all_positions` | Emergency halt |
| `MT5Connector.get_open_positions` | List currently open positions |
| `MT5Connector.get_symbol_info` | Cached — reduces API calls |

⚠️ **Position ID vs ticket**: use `position_id` for deal matching (never `order` or `ticket`).

### `time_manager.py` — Broker time + sessions

| Symbol | Role |
|--------|------|
| `TimeManager.get_server_time` | Returns `datetime` in Europe/Bucharest (EET/EEST) |
| `TimeManager.is_trading_session` | Checks session window (London / NY) |
| `TimeManager.is_friday_close_time` | True when Friday ≥ 20:45 EET |

⚠️ Use `datetime.now(Europe/Bucharest)` — **never** `mt5.symbol_info_tick().time` (FTMO returns broker-local epoch, and `fromtimestamp(tz=Bucharest)` double-adds the offset → +3 h drift). Requires NTP-synced VPS.

### `position_sizer.py` — Lot calculation

| Symbol | Role |
|--------|------|
| `PositionSizer.calculate_lot_size` | Returns `{lot_size, risk_usd, risk_pct}` |
| `PositionSizer._pip_value` | Three cases: quote = USD, base = USD (USDJPY / USDCHF), cross (EURJPY / GBPJPY uses USDJPY rate) |

Always floor-round the lot size (never ceil) — safety margin against broker steps.

### `news_scheduler.py` — Auto-import calendar weekly

| Symbol | Role |
|--------|------|
| `NewsCalendarScheduler.check_and_run` | Called every tick; imports CSV when it is Sunday 23:30 EET and the import has not run yet |
| `NewsCalendarScheduler._import_latest_csv` | Parses newest CSV in `config/news_inbox/` → writes `config/news_calendar.json` → moves CSV into `processed/` |
| State file | `logs/news_scheduler_state.json` (prevents double-runs) |

### `notifier.py` — Discord webhook

| Symbol | Role |
|--------|------|
| `DiscordNotifier.send_message`, `.send_alert` | Sends webhook |
| Rate limit | 20 req/min sliding window + min 1 s interval + 429 Retry-After + `threading.Lock` |

---

## 💱 execution/ — Order placement + position lifecycle

### `trade_executor.py`

| Symbol | Role |
|--------|------|
| `TradeExecutor.execute_signal(sig, live_context=None)` | Final gate: re-check risk → calc lot → send order → log. Accepts `live_context` dict from `main.py` and applies fields to `ExecutedTrade`. v6.10: ตั้ง `self._last_reject_reason` ที่ทุก rejection point (correlation/lot/risk/spread/validation/order). |
| `TradeExecutor._last_reject_reason` (v6.10) | String key ของ rejection point ล่าสุด. Reset ต้น `execute_signal`. main.py อ่านหลัง None return → log ลง Signals sheet "Executor Reject" col. |
| `TradeExecutor.sync_with_mt5` | Cumulative profit accumulator: pulls **all deals** of a position via `MT5Connector.get_deals_by_position(ticket)`, sums `profit + swap + commission`. Partial-close + BE-SL = WIN if cumulative > 0 (see [05-invariants.md FAQ](05-invariants.md)). |
| `ExecutedTrade` (dataclass v3 — v6.10 enhanced, 60+ fields) | Schema carries everything for live analysis: ML features (cal/raw scores, agent decision), confluence breakdown (HTF/MTF/OB/FVG/Sweep pts), trade-mgmt state (`be_moved`, `partial_closed_flag`, `partial_close_skipped` v6.10, `trailing_active`, `final_sl_at_close`), bid/ask @entry/exit, market context (ADX H1/H4, MTF/D1 bias), account state, overtrading metrics, **`obs_27_json`** (full 27-dim obs at decision time → unlock retrain). |
| `TradeExecutor._check_correlation` | Groups: USD_WEAK / USD_STRONG / JPY_CROSS / EUR_PAIRS / GBP_PAIRS — `MAX_CORRELATED_POSITIONS` per group per direction |

**Hard rule**: every order must have SL — never send orderless.

### `trade_manager.py`

| Symbol | Role |
|--------|------|
| `TradeManager.update_positions` | Called every tick — syncs MT5 positions + trailing + partial close |
| `TrailingState` (dataclass) | Per-position state: `initial_sl`, `current_sl`, `best_price`, `trailing_active`, `partial_closed`, `breakeven_moved` |
| `TradeManager._move_to_breakeven` | At RR=1.0: moves SL to entry, mirrors to `trade.be_moved=True` and `trade.final_sl_at_close=new_sl` for logging |
| `TradeManager._partial_close` | At RR=1.0: closes 50%, mirrors to `trade.partial_closed_flag=True` and `trade.partial_close_count += 1` |
| Constants | `BE_TRIGGER_RR=1.0` (move SL → entry), `PARTIAL_CLOSE_PCT=0.5` (50 % at 1:1), `TRAIL_ACTIVATION_RR=1.5` |

---

## ⚙️ config/ — Settings + News

### `settings.py` — Master config

Key dataclasses:

| Class | Key fields |
|-------|------------|
| `MT5Config` | `terminal_path`, `login`, `password`, `server`, `timeout` (loaded from `.env`) |
| `FTMOConfig` | `DAILY_LOSS_HARD_STOP_PCT=0.04`, `MAX_DRAWDOWN_HARD_STOP_PCT=0.08`, `PROFIT_TARGET_PCT=0.10`, `MIN_RISK_PER_TRADE_PCT=0.005`, `MAX_RISK_PER_TRADE_PCT=0.008`, `DEFAULT_RISK_PER_TRADE_PCT=0.007` ⭐, `MIN_CONFLUENCE_SCORE=70.0`, `CONSISTENCY_RULE_THRESHOLD=1.0` (2-step Standard → check off), `COOLDOWN_AFTER_LOSS_MIN=60`, `POST_TP_LOCK_TTL_MIN=60` |
| `SymbolConfig.symbols` | 10 symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY, **XAUUSD** |
| `MLConfig` (v7) | `CHRONOS_MODEL_NAME="amazon/chronos-bolt-small"`, `CHRONOS_DEVICE="cpu"`, `CHRONOS_PREDICTION_LENGTH=8`, `CHRONOS_CONTEXT_LENGTH=512`, `CHRONOS_ENABLED=True`. Single source of truth สำหรับ pool builder + live. |
| `MeanReversionConfig` (v8.0) | `strategy_mode="smc"` (flip to `"mean_reversion"` after MR eval gates pass), `bb_period=20`, `bb_std=2.0`, `bb_oversold=0.10`, `bb_overbought=0.90`, `rsi_oversold=30.0`, `rsi_overbought=70.0`, `adx_trend_block=25.0`, `sl_atr_mult=1.0`, `rr_ratio=1.0`, `quick_tp_bars=5`, `quick_tp_bonus=0.50`, `slow_win_bonus=0.20`, `prolonged_loss_bars=12`, `prolonged_loss_penalty=0.40`, `base_loss_penalty=0.10`, `duration_fine_coef=0.02` |
| `bot_config` (singleton) | Aggregates every dataclass — import target for other modules. v7: + `bot_config.ml`. v8.0: + `bot_config.mr` |

### `news_events.py` — Hardcoded fallback news

| Symbol | Role |
|--------|------|
| `RecurringNewsEvent` (dataclass) | `name`, `currencies` (Set), `weekday`, `day_of_month`, `time_utc`, `.occurs_on()` |
| `HIGH_IMPACT_EVENTS` (list) | NFP, ECB, BoE, FOMC, CPI, etc. — used when `news_calendar.json` is missing or expired |

⚠️ All times UTC (compared after EET → UTC conversion).

### `news_csv_parser.py` — ForexFactory CSV importer

| Symbol | Role |
|--------|------|
| `parse_forexfactory_csv(csv_path, tz_offset=0)` | Returns `List[Dict]` of high-impact events |
| Filter | Only impact ∈ {"high", "holiday"} |

---

## 📊 analytics/

### `performance.py`

| Symbol | Role |
|--------|------|
| `TradeResult` (dataclass) | ticket, symbol, entry/close, lot, risk, profit, rr_achieved, times, confluence |
| `PerformanceAnalyzer.calculate_basic` | Win rate, profit factor |
| `PerformanceAnalyzer.calculate_advanced` | Sharpe (annualized 252), Sortino |
| `PerformanceAnalyzer.calculate_risk_metrics` | Max DD, Calmar |
| `PerformanceAnalyzer._trades` | List of `TradeResult` (used for `recent_wr_norm` in `_build_signal_observation`) |
| `PerformanceAnalyzer.load_from_excel(path)` (v6.14 wired) | Replays closed trades from `Trades` sheet → restores equity curve + Max DD + Sharpe across restarts. Called once at `FTMOTradingBot.initialize` step 3.6 after `set_initial_balance`. To reset history → delete `logs/ftmo_trades.xlsx` before run. |

### `trade_logger.py`

✅ **Re-enabled (v6.9 2026-04-25)** in `FTMOTradingBot.__init__` for live demo data capture.

| Symbol | Role |
|--------|------|
| `TradeLogger.__init__(log_dir)` | Auto-creates `logs/` folder; writes to single consolidated `ftmo_trades.xlsx` (no monthly split). |
| `TradeLogger.log_trade_opened(trade_data)` | Appends row to `Trades` sheet (64 cols). Color-codes BUY=green/SELL=red. |
| `TradeLogger.log_trade_closed(trade_data)` | Updates close columns (price, time, P/L, MAE/MFE, exit path) on existing row. **v6.14 fix**: column index off-by-one — เดิมเขียน 28-31 ทับ `DD@Entry % / MAE / MFE / Time-in-Trade`; แก้เป็น 29-32 ตรงกับ `TRADE_HEADERS` (`MAE=29, MFE=30, Time-in-Trade=31, Exit Path=32`). |
| `TradeLogger.log_signal_scan(scan_data)` | Per-scan event log → `Signals` sheet (21 cols, includes `AGENT_SKIP`/`AGENT_TAKE`/`REJECTED`/`NO_SIGNAL` results, color-coded). |
| `TradeLogger.log_daily_summary(balance, daily_dd, max_dd)` | Daily roll-up → `Daily` sheet (Date/Trades/Wins/WR%/PL/DD/Balance). Wins counter uses `profit > 0` on cumulative (not last deal). **v6.10 trigger:** `FTMOTradingBot.run` ตรวจ `broker_today != _last_logged_day` ตอนต้น loop → flush ของวันก่อนก่อน `check_risk()` reset state. + เรียกตอน `shutdown()` |
| `TradeLogger.update_stats_sheet(stats)` | Refreshes `Stats` sheet from `PerformanceAnalyzer` output. **v6.10 trigger:** ทุก 720 loops (~1 ชม. @ 5s interval) ใน main loop + ตอน day rollover + `shutdown()` |
| `TRADE_HEADERS` | **64 cols** (v6.10): Schema v3 + Partial Skipped (col 63) + Obs27 JSON (col 64). |
| `SIGNAL_HEADERS` | **21 cols** (v6.10): + Executor Reject (col 20) + Obs27 JSON (col 21). |

**Schema migration**: existing `ftmo_trades.xlsx` from before v6.9 has 62 cols / 19 cols — append will misalign. Rename or delete pre-v6.9 file before first start (`mv logs/ftmo_trades.xlsx logs/ftmo_trades_pre_obs27.xlsx`).

**Excel file**: `ftmo_trading_bot/logs/ftmo_trades.xlsx`. 4 sheets: Trades, Signals, Daily, Stats. Auto-creates on first scan/trade.

---

## 🛠️ scripts/ — Training + data fetch

| Script | Role |
|--------|------|
| **`build_mr_signal_pool.py`** (v8.0) | Multiprocessing 8 workers — calls `MeanReversionBacktester.generate_episode_signals` × N → `data/mr_signal_pool_<N>.pkl`. Replaces legacy `build_signal_pool.py` (deleted with SMC in v8.0.6). |
| **`train_mr_signal_quality.py`** (v8.0) | GBM + `GroupKFold cross_val_predict` + Isotonic calibrator on MR pool → `data/mr_signal_quality_model.pkl`. 28 features (`FEATURE_KEYS`): SMC-compat schema + MR extras (`mr_setup_score`, `bb_extreme`, `bb_band_width_atr`, `reversal_wick_ratio`). Saves drift baseline (`train_dist`) for live KS test. |
| **`train_mr_signal_filter.py`** (v8.0) | `AuxAwarePPO` + `AuxAwareACPolicy` 2-phase curriculum on `MeanReversionFilterEnv`. CLI flags forward MR shaping params (`--quick_tp_bonus`, `--prolonged_loss_penalty`, `--duration_fine_coef`, `--lr_p1`, `--lr_p2`, ...) → `models/mr/ppo_mr_filter.zip` + `vec_normalize_mr.pkl`. Default P1 5M / P2 2M, ml_threshold 0.30. |
| **`auto_train_pipeline.py`** (v8.0) | **Autonomous orchestrator** — Build pool → GBM → RL → 5000-eps Eval → check gates (Pass ≥ 8%, Total DD max ≤ 6%, Daily DD max ≤ 3.5%, Profitable ≥ 55%, Breach ≤ 5%) → if fail, `tune_hyperparams()` mutates `HyperParams` (smart 4-way: low-WR / high-WR-low-pass / over-trade / generic) and loops. Logs to `logs/auto_train_pipeline.log` + `.jsonl` + `state.json`. Snapshots best to `models/mr/best/`. v8.0.4: `rebuild_pool=False` after iter 0 (reuse pool/GBM). v8.0.5: detect pool/GBM exists on launch → skip rebuild. CLI: `--max_iterations`, `--max_hours`, `--target_*` gates, `--dry_run`. |
| **`leakage_audit.py`** (v8.0) | 4-step audit: (1) AST scan obs builders, (2) GBM features list + saved model `.keys`, (3) pool dict label sanity, (4) dynamic env step. Exit 0 = no leakage. v8.0.7: forces `encoding="utf-8"` for cross-platform (Windows cp1252 fix); Audit 4 graceful skip when OHLCV missing. |
| **`parity_audit.py`** (v8.0) | 7-step audit: strategy params, ml_threshold, risk_per_trade, obs dim 3-way sync, correlation groups, indicator parity (full vs rolling), VecNormalize availability. Exit 0 = train↔live aligned. v8.0.7: UTF-8 encoding + Audit 4/6 graceful skip on VPS without OHLCV. |
| **`pipeline_status.sh`** (v8.0) | Shell helper — pipeline alive? best metrics? recent log tail? Run anytime during pipeline run. |
| `fetch_mt5_data.py` | (Windows only) fetches OHLCV from MT5 → CSV under `data/ohlcv/` |
| `import_forexfactory_csv.py` | Manual CSV → `news_calendar.json` import (alternative to `NewsCalendarScheduler`) |
| `test_chronos_accuracy.py` (v7.0.7) | **Benchmark tool** — rolling-window backtest ของ `ChronosForecaster` บน historical CSV. คำนวณ 3 metrics: Direction Accuracy, Quantile Coverage, MAPE. Standalone (ไม่กระทบ live/train). v7.0.7 result: TOTAL Dir Acc 50.5%, Coverage 79.8% (calibrated), MAPE 0.12%. Run: `.venv/bin/python ftmo_trading_bot/scripts/test_chronos_accuracy.py` |
| `test_chronos_mtf.py` (v7.0.7) | **Multi-TF benchmark** — Chronos forecast บน H4 + H1 + M15 พร้อมกัน → consensus direction. Verdict: **ไม่ช่วย overall** (ALL_AGREE acc 45.3% < baseline M15 48.2%). บาง symbol ดีขึ้น (USDCHF +11pp, NZDUSD +4.4pp) แต่บาง symbol แย่ลงหนัก (USDCAD -19pp, GBPUSD -18pp). Conclusion: ไม่ integrate เข้า v7.0.7 (3× inference cost + ไม่ improve). |

---

## 🚀 main.py — Entry Point

| Symbol | Role |
|--------|------|
| `FTMOTradingBot.__init__` | Builds every subsystem (connector, risk, sizer, strategy, executor, manager, RL agent, ML model, news scheduler, notifier, **TradeLogger**). Initializes 5 announce-once flags: `_daily_halt_announced`, `_friday_announced`, `_weekend_announced`, `_daily_close_announced`, `_rollover_announced`. Initializes `_trade_open_history: list` (cap 200, for overtrading metrics). **v6.10:** เพิ่ม `_last_logged_day` (None) สำหรับ trigger Daily summary flush ตอนข้ามวัน. |
| `FTMOTradingBot.connect` | MT5 login + load state |
| `FTMOTradingBot.run` | Main loop (runs every `main_loop_interval` seconds, default 5 s). Idle-state guards use announce-once pattern (print on entry only, auto-reset in `else` branch). |
| `FTMOTradingBot._build_signal_observation` | Builds the **29-dim** obs (v7) from a `TradeSignal` + portfolio state — must match `FTMOSignalFilterEnv._get_obs`. Same path used to populate `live_context["obs_27_json"]` for retrain logging (key kept for backward compat, contains 29 dims since v7). v7: เรียก `self._chronos.forecast_features(sig.symbol, self._strategy._ltf_data, direction, atr_val)` สำหรับ obs[27,28]. |
| `FTMOTradingBot._chronos` (v7) | `ChronosForecaster` instance. Init จาก `bot_config.ml.CHRONOS_*`. ถ้า disable / load fail → obs[27,28] = 0.0 (graceful degrade). |
| `FTMOTradingBot._build_live_context(sig)` | Gathers `ml_score` (cal/raw), bid/ask snapshot, ADX H1/H4, MTF/D1 bias, balance, overtrading metrics (`trades_today`, `secs_since_last_*`), and JSON-encoded obs vector → passed to `TradeExecutor.execute_signal` and `TradeLogger.log_signal_scan`. **v6.12:** `ctx["ml_threshold_used"]` ดึงจาก `bot_config.ftmo.ML_FILTER_THRESHOLD` (single source of truth) — ไม่ใช่ `getattr(rl_agent, "ml_filter_threshold")` ที่ตกค่า 0.0 เพราะ attribute อยู่บน `FTMOSignalFilterEnv` ไม่ใช่ agent. |
| `FTMOTradingBot.run` (ML gate v6.12) | ใน loop ก่อนเรียก `_rl_agent.should_take_signal`: ถ้า `ml_score < bot_config.ftmo.ML_FILTER_THRESHOLD` → log `Result = "ML_FILTERED"` แล้ว `continue`. ทำให้ live distribution = training distribution (env กรองเดียวกันตอน train). |
| `FTMOTradingBot._log_signal_scan(sig, ctx, result)` | Wrapper: builds `scan_data` from signal + context + result label (`AGENT_TAKE`/`AGENT_SKIP`/`AGENT_TAKE_FAIL`/`REJECTED`/`NO_SIGNAL`/`ML_FILTERED` v6.12), calls `TradeLogger.log_signal_scan`. **v6.10d:** scan_data includes `executor_reject_reason` (col 20) + `obs_27_json` (col 21) จาก live_context — กัน Signals sheet col 20/21 ว่างเปล่า. |
| `FTMOTradingBot._build_spread_pct_of_atr` | Computes obs[24] (cost awareness) |
| `FTMOTradingBot._has_opposite_recently_closed` | Computes obs[25] (flip-lock context) |
| `FTMOTradingBot.shutdown` | Graceful shutdown — saves state, closes connector |

---

## Cross-links

- 3-brain pipeline + loop priority → [01-architecture.md](01-architecture.md)
- Obs 27 dims layout + reward + PPO → [03-rl-training.md](03-rl-training.md)
- FTMO state machine + news + sessions → [04-operations.md](04-operations.md)
- Red-flag rules + version log → [05-invariants.md](05-invariants.md)
