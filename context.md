# CONTEXT — FTMO Trading Bot (LLM Wiki Hub)
> Last Updated: 2026-05-05 (v7.1.4 staged — Combo C: keep threshold 0.40 + restore v7.0.x missed-winner -0.90, retrain pending) | Scope: Hub / Index — read this first, then drill into wiki/*

## TL;DR (LLM read first — 30-second scan)

- **Goal**: pass the FTMO 2-step Standard Challenge (10 % profit, 4 % daily DD, 8 % total DD).
- **3 brains + 1 forecaster**: `SMCStrategy` (rules) → `SignalQualityModel` (GBM + Isotonic calibrator) → **`ChronosForecaster`** (Amazon Chronos 2 zero-shot, v7) → `SelfLearningAgent` (PPO + Auxiliary Task — TAKE/SKIP).
- **Live entry**: `python main.py` → `FTMOTradingBot.run` loops every 5 s. Console runs in **quiet mode** (announce-once for idle states; per-signal SKIP/NO_AGENT logged to Excel `Signals` sheet, not console).
- **Obs = 29 dims** (v7, 2026-05-01 — adds `chronos_alignment` + `chronos_uncertainty_norm`). Must stay in sync across three places: `FTMOSignalFilterEnv._get_obs` / `FTMOTradingBot._build_signal_observation` / `SelfLearningAgent.OBS_DIM`.
- **Runs on**: macOS/Linux (train + backtest), Windows + MT5 (live).
- **Live logging**: `TradeLogger` (re-enabled v6.9, schema bumped v6.10) writes Excel — Trades 64 cols (incl. `Obs27 JSON` for retrain), Signals 21 cols (per-scan log), Daily, Stats.
- **v6.11 SMC overhaul (2026-04-29)**: Counter-D1 hard veto + Sweep within 8 bars + Fresh M15 BOS within 6 bars + ADX H4 ≥ 22 + Quiet-vol × off-overlap blocker + IDM detector + OB grading (Extreme/Decisional/Internal). BE trigger ใช้ `best_price` (rolling MFE). Per-component pts populate ใน TradeSignal → Trades sheet เห็น HTF/MTF/OB/FVG/Sweep pts จริง. ดู [`wiki/05-invariants.md` v6.11 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v6.11.1 (2026-04-29 evening)**: Post-impl audit fix — `StrategyBacktester._init_strategy` เพิ่ม `_idm_detector` init ให้ตรงกับ `SMCStrategy.__init__` (กัน AttributeError ตอน rebuild pool). Eval Pass Rate 11.2 % ที่ได้จาก cached pool (Apr 25, pre-v6.11) — **invalid สำหรับ v6.11**. ต้อง rebuild pool + GBM ก่อน eval ใหม่.
- **v6.11.2 (2026-04-29 evening)**: Partial rollback — Tier 2.2 (Sweep prereq) + Tier 2.3 (Fresh M15 BOS prereq) hard gates → soft bonuses. Pool หาย 99 % ภายใต้ v6.11 hard gates → 0.0 % Pass Rate. หย่อน 2 จุด → pool 78k signals, Pass Rate 2.7 %, WR 65.6 %.
- **v6.11.3 (2026-04-29 evening)**: Mild tune — IDM penalty 5→2 + ADX H4 floor 22→20. Pool 90k (+15 %), Pass Rate 2.7→**3.4 %**, WR 65.6→**68.8 %**, DD max 4.46→**3.23 %** (safer). Pure improvement ทุกมิติ. **KEEP + deploy demo**. Backups `*.bak_v6.11.2` พร้อมสำหรับ rollback.
- **v6.12 (2026-04-29 night)**: Live-vs-train sync fix — `FTMOConfig.ML_FILTER_THRESHOLD = 0.36` + ML gate ใน `FTMOTradingBot.run` ก่อน agent. ปิด silent regression จากการที่ live ไม่บังคับ ML threshold (training บังคับ). Logging fix — `ML Threshold` column ใน `Signals` sheet ตอนนี้แสดง 0.36 จริง (เดิม 0.0 เพราะ getattr ผิด attribute). ไม่ต้อง retrain.
- **v7.1 (2026-05-04) — Code staged, awaiting retrain ⚠️ ห้ามรัน live จนกว่าจะ retrain (OBS_DIM 29→32 mismatch)**:
  - **RCA**: 5 ออเดอร์โดน SL hit 100% ใน 3 ชม. (Daily DD 2.85% / Max DD 3.97%) แม้ HTF ตรง 4/5 + ML cal ≥ 0.44 + RL TAKE ทั้งหมด → 3 brains "ไม่รู้เวลา + ไม่รู้ portfolio risk" — ไม่ใช่ bug จุดเดียว
  - **Runtime guards (no retrain)**: `RiskManager.check_unrealized_circuit_breaker` (floating ≤ −1.5% × open ≥ 2 → pause), `MAX_USD_THEME_POSITIONS=2` (cross-group cap), `SPREAD_ATR_RATIO_LIMIT=0.20` (thin liquidity), Stats sheet limits จาก config (เลิก hardcode 10%/5%), close payload populate (Bid@Exit/Balance@Close/Equity Peak)
  - **SMC pre-filters ใหม่ (G/H/I)**: HTF=Neutral + ADX H1 < 25 hard veto, vol_regime ∈ {high, explosive} block, spread/ATR > 20% block, session warmup + post-weekend → required confluence +5/+10
  - **Volatility regime classifier**: `TechnicalIndicators.classify_volatility_regime` (quiet/normal/high/explosive) + `compute_atr_zscore_30bars`
  - **Dynamic SL multiplier**: `_compute_dynamic_sl_multiplier` (per-symbol base × clip(1+atr_z×0.3, 0.8, 1.8))
  - **GBM features 17→24**: เพิ่ม `hour_of_day_sin/cos`, `day_of_week`, `minutes_since_session_start`, `is_post_weekend_first_hour`, `volatility_regime_score`, `atr_zscore_30bars` + `compute_temporal_features` helper + `detect_drift` (KS test live vs train)
  - **Chronos formula**: linear `(q90-q10)/(atr×√8)` → `log1p(...)/2` (กัน saturation ที่ 3.0 ทุก signal)
  - **RL obs 29→32**: เพิ่ม `floating_pnl_norm`, `open_losing_count_norm`, `mins_since_session_norm`
  - **Reward shaping**: Chronos disagreement penalty (-0.40 ถ้า align<0 + ml<0.55, -0.15 อื่น), concurrent loss penalty (-0.25 ถ้า floating<-1%), missed-winner softened P2 -0.90→-0.65
  - **Pipeline**: ต้อง rebuild pool + retrain GBM + retrain RL ทั้งหมด ~36 ชม. compute. Backups `*.bak_v7.0.7` ก่อนเริ่ม
  - **Eval gate**: Pass Rate ≥ 9% + DD max ≤ 4.5% + Profitable ≥ 70% → keep. < 7% → restore. ดู [`wiki/05-invariants.md` v7.1 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.7 (2026-05-02) — Revert threshold 30 → 20 (v7.0.5 retrain after backup loss)**:
  - **Why** — v7.0.6 (threshold 30) eval Pass Rate **7.4%** (regression −3.3 pp จาก v7.0.5 = 10.7%). User skip backup step → v7.0.5 model หาย → ต้อง retrain ใหม่
  - **Lesson learned** — Early stop ที่ value_loss=20.45 = **inflection point detector** (ไม่ใช่ false positive). Phase 2 หลังจุดนี้ทำให้ agent over-tune toward safety: WR ขึ้น (65→68%), Profitable ขึ้น (86→88%), แต่ **passes น้อยลง** (selective เกินไม่ aggressive พอจะถึง 10% ใน 45 วัน)
  - **Fix** — single-line revert: `threshold=20.0` (กลับ v7.0.5 config) — engineered sweet spot @ Phase 2 ~70%
  - **No rebuild** pool/GBM. Retrain RL อย่างเดียว ~30 ชม.
  - **Risk**: stochastic — RNG อาจให้ Pass 9.5-11.5% (ไม่เป๊ะ 10.7%). v7.0.3 backup (10.0%) เป็น final safety net
  - **Backup discipline** — เพิ่ม emphasis: ต้อง execute backup commands ก่อนทุก retrain ห้ามข้าม
  - ดู [`wiki/05-invariants.md` v7.0.7 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.6 (2026-05-02) — Phase 2 EarlyStop threshold 20 → 30, Pass Rate 7.4% (regressed)**:
  - **Why** — v7.0.5 trigger ที่ `value_loss = 20.45` (just 1.02× threshold) = borderline. หลัง LR fix (5e-5) value_loss variance ต่ำลง → threshold 20 อาจ aggressive เกิน. Phase 2 รัน 70% (3.5M / 5M) ก่อนหยุด — มี potential ที่ยังไม่ได้
  - **Fix** — single-line: `EarlyStopOnValueLoss(threshold=30.0, ...)` (จาก 20.0). 30 = 1.5× = ตรงกับ "natural transient spike range" ของ v6.13/v7.0.3 historical (31)
  - **No rebuild** pool/GBM. Retrain RL อย่างเดียว ~30 ชม.
  - **Probability**: 40% Pass > 10.7%, 35% equivalent, 20% mild regress, 5% catastrophic. Backup `*.bak_v7.0.5` (10.7%) เป็น safety net
  - **Watch** — ถ้า value_loss > 30 = true divergence (early stop trigger). ถ้า Pass Rate trajectory ลด 2 snapshots ติด = kill + restore v7.0.5
  - **Gate** ≥ 11% = win, 10-11% = equivalent (keep), < 9% = restore v7.0.5
  - ดู [`wiki/05-invariants.md` v7.0.6 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.5 (2026-05-02) — Phase 2 LR schedule proper fix (latent bug จาก v6.x), Pass Rate 10.7%**:
  - **Why** — v7.0.4 retrain Phase 2 รันเต็ม 5M (warmup ปลด early-stop) → Pass Rate **2.8%** (regression −7.2 pp). Trajectory ลดลงตอนปลาย Phase 2 = over-training
  - **Root cause Layer 1 (latent bug)** — SB3 PPO `_setup_model()` wrap `lr_schedule` ตอน init เท่านั้น. การ set `model.learning_rate = 5e-5` หลัง `AuxAwarePPO.load()` ไม่ rebuild schedule → optimizer ใช้ Phase 1 default 3e-4 (= **6× สูงกว่า intended** ตลอด v6.x → v7.0.4)
  - **Root cause Layer 2** — High LR (3e-4) × Long Phase 2 (5M, ก่อนหน้านี้ early-stop @ ~32k) = over-train. v6.13/v7.0.3 "lucky" เพราะ early-stop ปกป้อง by accident
  - **Fix** — rebuild lr_schedule + update optimizer 3 ชั้น:
    ```python
    from stable_baselines3.common.utils import FloatSchedule
    model_p2.learning_rate = 5e-5
    model_p2.lr_schedule = FloatSchedule(5e-5)
    for _pg in model_p2.policy.optimizer.param_groups:
        _pg['lr'] = 5e-5
    ```
  - **No rebuild** pool/GBM. Retrain RL อย่างเดียว ~30 ชม.
  - **Watch** — Phase 2 step 0 TensorBoard `learning_rate = 0.00005` (ถ้าเป็น 0.0003 = fix fail)
  - **Gate** ≥ 10% = keep, < 7% = restore `*.bak_v7.0.3`
  - Backups `*.bak_v7.0.3` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.5 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.4 (2026-05-02) — Phase 2 EarlyStopOnValueLoss warmup grace, Pass Rate 2.8% (regressed)**:
  - **Why** — v7.0.3 retrain Phase 2 trigger early stop ที่ step 32k / 5M (0.6% ของ target). value_loss spike 31 > threshold 20 ตอน Phase 2 start เพราะ reward distribution shift (Phase 1 [-2,5] → Phase 2 [-15,5] หลังเปิด DD penalty + activity floor). Eval Pass Rate 10.0% (ผ่าน gate baseline) แต่ Phase 2 "เกือบไม่ได้ทำงาน" → potential ที่ขาด
  - **Fix** — เพิ่ม `warmup_steps` parameter ใน `EarlyStopOnValueLoss`:
    - Phase 2: `warmup_steps=50_000` (1% ของ 5M target) — ปล่อยให้ value head re-fit reward distribution ใหม่ก่อน enable check
    - Phase 1: `warmup_steps=0` (default) — backward compat, ไม่กระทบ
  - **No rebuild** pool / GBM. Retrain RL อย่างเดียว ~25-30 ชม. Backup `*.bak_v7.0.3` เผื่อ rollback
  - **Gate** > 10.0% = win (warmup ช่วย push), 9-10% = equivalent (no harm), < 9% = restore `*.bak_v7.0.3`
  - Backups พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.4 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.3 (2026-05-02) — Correlation simulator HOLD=0 (over-block fix), Pass Rate 10.0%**:
  - **Why** — v7.0.2 eval Pass Rate ตก 0.6% (จาก 9.7% baseline). Root cause: time-scale mismatch ระหว่าง pool (4 scans/day = 1 slot ≈ 6h) กับ live (avg position hold 75 นาที) → `HOLD_SIGNALS_APPROX = 1` block correlation 6h ใน live time = over-block 4-5×. Agent skip-all (Take Rate 50% → 32%, Orders/ep 7.7 → 4.8)
  - **Fix** — single-line: `HOLD_SIGNALS_APPROX = 1 → 0` ใน `ml/signal_filter_env.py`. Drop-stale logic clear `_open_positions` ทันทีทุก step → effective correlation simulator off, infrastructure คงไว้ (เผื่อ tune กลับ)
  - **No rebuild** pool/GBM (ไม่ขึ้นกับ HOLD). Retrain RL อย่างเดียว ~25-30 ชม.
  - **Gate** ≥ 9.7% = keep, < 4% = revert. Test isolation: ถ้าผ่าน = Chronos formula fix ใช้งานได้ จริง correlation simulator ต้อง parameter ที่ถูกต้อง. ถ้าไม่ผ่าน = revert + consider Trend Following (Plan B)
  - Backups `*.bak_v6.14` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.3 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.2 (2026-05-01) — Chronos formula fix + correlation training-live sync**:
  - **Why** — v7.0 retrain Pass Rate ตก 9.7% → 4.0%. Root cause analysis (pool diagnostics + ftmo_trades audit) พบ 3 จุด: (1) `chronos_uncertainty_norm` saturated 96.2% ที่ค่า max, (2) `chronos_alignment` corr ติดลบ (−0.0178 anti-signal), (3) live correlation block 96.3% ของ TAKE แต่ training env ไม่มี check
  - **Fix Chronos formula** (`ml/chronos_forecaster.py`): flip alignment sign (`direction × sign(close − median)`) + Brownian-scaled uncertainty (`/ atr × √8`) → distribution กระจาย ไม่ saturate, sign ตรงกับ outcome
  - **Add correlation simulator** (`ml/signal_filter_env.py`): mirror `TradeExecutor.CORRELATION_GROUPS` (USD_WEAK/STRONG, JPY_CROSS, EUR/GBP_PAIRS, SAFE_HAVEN), `_open_positions` virtual list, `_is_correlation_blocked()` method, forced SKIP ใน `step()` ถ้า block — training distribution ตรงกับ live
  - **No config change** — ห้ามแตะ Daily DD logic, OBS_DIM ยังคง 29, live source of truth (`TradeExecutor`) ไม่แก้
  - **Pipeline**: rebuild pool → retrain GBM → retrain RL ทั้งหมด. Eval gate ≥ 9.7% = keep, < 7% = revert จาก `*.bak_v6.14`
  - Backups `*.bak_v6.14` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.2 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0 (2026-05-01) — Amazon Chronos 2 zero-shot forecast features added (obs 27 → 29)**:
  - **New module** — `ftmo_trading_bot/ml/chronos_forecaster.py` (`ChronosForecaster`). โหลด `amazon/chronos-bolt-small` ผ่าน `BaseChronosPipeline.from_pretrained` ครั้งเดียว, cache forecast per `(symbol, last_bar_ts)`, deterministic via `torch.manual_seed(0)`.
  - **2 new obs features** — `obs[27] chronos_alignment` (`direction × sign(median_h+8 − close)`, ±1) + `obs[28] chronos_uncertainty_norm` (`(q90−q10)/atr`, [0, 3]). Sync ทั้ง 3 ที่ (`FTMOSignalFilterEnv._get_obs` / `FTMOTradingBot._build_signal_observation` / `SelfLearningAgent.OBS_DIM = 29`).
  - **Pool inject** — `StrategyBacktester.generate_episode_signals` คำนวณ chronos features จาก `ltf_slice` (closed bars only, ห้าม leak future) → เก็บใน signal dict สำหรับ `FTMOSignalFilterEnv._get_obs` อ่าน.
  - **Live path** — `FTMOTradingBot.__init__` instantiate `ChronosForecaster`. `_build_signal_observation` + `_build_live_context` ส่งค่าเข้า obs + Excel log.
  - **Excel schema bump** — `Signals` sheet 21 → 23 cols (+ `Chronos Align`, `Chronos Unc`); `Trades` sheet 64 → 66 cols (+ chronos @ entry). Pre-v7 Excel ต้อง `mv logs/ftmo_trades.xlsx logs/ftmo_trades_pre_v7.xlsx` ก่อน first run.
  - **Config single source of truth** — `bot_config.ml.CHRONOS_MODEL_NAME` (default `"amazon/chronos-bolt-small"`), `CHRONOS_PREDICTION_LENGTH=8`, `CHRONOS_CONTEXT_LENGTH=512`, `CHRONOS_ENABLED=True`. Disable via env `BOT_DISABLE_CHRONOS=1` (unit tests).
  - **ห้ามแตะ Daily DD** — `RiskManager.update_daily_pnl` / `_initial_balance` / `_daily_start_balance` ไม่ได้รับ touch ใดๆ. Chronos อยู่ที่ pre-trade gate เท่านั้น.
  - **Pipeline**: ต้อง rebuild pool + retrain GBM + retrain RL (Pass Rate gate ≥ 9.7 % vs v6.13 baseline = keep, ไม่ผ่าน = revert จาก `*.bak_v6.14`).
  - Backups `*.bak_v6.14` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v6.14 (2026-04-30) — Live demo log audit, 4 bug fixes** (silent regression vs v6.13 spec):
  - **Fix 1 — SL flow alignment (XAU)**: `SMCStrategy.scan_signal` BUY+SELL ใช้ per-symbol `sl_atr_multiplier` (XAU=1.8×, FX=1.5×) แทน global hardcoded 1.5× + เพิ่ม OB SL `ob_sl_floor = atr × 0.5` ป้องกัน clamp ลด SL ใกล้เกินไป
  - **Fix 1C — XAUUSD `min_sl_pips`**: 300 → **1000 ticks** ($3 → $10) — guard ชั้นสุดท้าย ไม่ให้ Gold SL ต่ำกว่า ~1.0×ATR
  - **Fix 2 — TradeLogger off-by-one**: close-row update path เขียน column 28-31 ทับ `DD@Entry % / MAE / MFE / Time-in-Trade` → แก้เป็น 29-32 ตรงกับ `TRADE_HEADERS`
  - **Fix 3 — PerformanceAnalyzer replay**: re-enable `load_from_excel(logs/ftmo_trades.xlsx)` ใน `FTMOTradingBot.initialize` ขั้นตอน 3.6 (เดิม `[DISABLED]` block) → Stats / Max DD / Sharpe ต่อเนื่องข้าม restart
  - **Symptom จาก live demo (04-29→04-30, 9 trades)**: XAU trade #0 [437211678] SL = 0.28×ATR → SL hit ใน 12s (-$103); MFE column ใน Excel เก็บค่า duration; Stats sheet "Total Trades 3" แทน 9
  - **ไม่ต้อง retrain**: train env (`StrategyBacktester`) ใช้ 1.8× อยู่แล้ว — Fix 1 ดึง live ให้ตามให้ทัน. หลัง deploy ต้องตรวจ live SL/ATR ratio ของ XAU trade ใหม่ ≈ 1.8 (ไม่ใช่ 0.28 หรือ 1.5)
  - Backups `*.bak_v6.13` พร้อม rollback. ดู [`wiki/05-invariants.md` v6.14 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v6.13 (2026-04-29 night) — VERIFIED Pass Rate 9.7 %** (5000 eps eval):
  - **L1 Pause** 2→3, **L2 defaults safety** (env const 0.003→0.007, outcome_noise 0.02→0.05), **L3 reward rebalance** (TAKE equalize @ ml ≥ 0.36, P2 SKIP-oracle missed-winner −0.70→−0.90, smart-skip +0.20→+0.35, day-10 early undertrading check), **L4 XAU SL 1.5×→1.8×**
  - **Eval result**: Pass Rate **9.7 %** (vs baseline 3.4 % = +185 %), WR 64.8 %, Orders/ep 7.7, DD max 4.40 % (ห่าง 8 % limit), Breach 0 %, Profit avg +3.89 %
  - Pass Rate เกือบถึง FTMO 10 % target — ทะลุเป้าทุกมิติ. KEEP + deploy demo
  - **No-leak audit ผ่าน**: 27 obs dims + GBM 17 features = signal-time only. Aux head ใช้ outcome เป็น MSE target. SKIP-oracle เป็น reward shaping (policy ไม่เห็น outcome)
  - Backups `*.bak_v6.12` พร้อม rollback. ดู [`wiki/05-invariants.md` v6.13 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **Wiki Sync Protocol**: editing `.py` files under `ftmo_trading_bot/` requires updating `wiki/` + `context.md` + `readme.md` (when user-facing) in the same turn. Stop hook enforces (`decision: block`). See `CLAUDE.md`.

## Headline Numbers

| Metric | Value | Source (symbol) |
|--------|-------|-----------------|
| Symbols | **10** (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY, XAUUSD) | `SymbolConfig.symbols` |
| Timeframes | M15 (entry) / H1 (structure) / H4 (HTF bias) | `SymbolConfig.primary/structure/higher_timeframe` |
| Profit target | 10 % | `FTMOConfig.PROFIT_TARGET_PCT` |
| Daily DD stop | 4 % | `FTMOConfig.DAILY_LOSS_HARD_STOP_PCT` |
| Total DD stop | 8 % | `FTMOConfig.MAX_DRAWDOWN_HARD_STOP_PCT` |
| Default risk per trade | **0.99 %** (v7.1.9 — FTMO 1% rule, sync กับ RL training) | `FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT` |
| Risk floor / cap | 0.5 % / 0.99 % | `FTMOConfig.MIN/MAX_RISK_PER_TRADE_PCT` |
| Min confluence | 70 | `FTMOConfig.MIN_CONFLUENCE_SCORE` |
| Max open positions | 3 | `FTMOConfig.MAX_OPEN_POSITIONS` |
| ATR floor (signal gate, per-symbol) | 3-8 pips FX, 500 ticks XAUUSD | `SymbolConfig.symbol_overrides[X].atr_floor_pips` |
| MIN_SL guard (per-symbol, v6.2 / v6.14 XAU raise) | 10-20 pips FX, **1000 ticks XAUUSD** ($10) | `SymbolConfig.symbol_overrides[X].min_sl_pips` |
| SL base multiplier (per-symbol, v6.14 wired) | FX 1.5×, **XAUUSD 1.8×** | `SymbolConfig.symbol_overrides[X].sl_atr_multiplier` (fallback `bot_config.indicators.atr_sl_multiplier`) |
| OB SL clamp lower bound (v6.14) | 0.5 × ATR | hard-coded in `SMCStrategy.scan_signal` |
| Obs dims | **32** (v7.1) — staged, retrain pending | `SelfLearningAgent.OBS_DIM` |
| GBM features | **24** (v7.1, 17→24 +temporal/regime) | `SignalQualityModel.FEATURES` |
| Unrealized DD breaker | −1.5% × open ≥ 2 | `FTMOConfig.UNREALIZED_PAUSE_PCT` |
| USD theme cap | 2 positions / theme | `FTMOConfig.MAX_USD_THEME_POSITIONS` |
| Spread/ATR limit | 0.20 (thin liquidity block) | `FTMOConfig.SPREAD_ATR_RATIO_LIMIT` |
| Chronos model (v7) | `amazon/chronos-bolt-small` | `bot_config.ml.CHRONOS_MODEL_NAME` |
| Chronos horizon (v7) | 8 M15 bars (~2 h) | `bot_config.ml.CHRONOS_PREDICTION_LENGTH` |
| RL model | `models/ppo_signal_filter.zip` + `models/vec_normalize_sf.pkl` | `SelfLearningAgent` |
| ML model | `data/signal_quality_model.pkl` | `SignalQualityModel` |
| Pool | `data/signal_pool_3000.pkl` (~158k signals) | `StrategyBacktester` |
| FTMO program | 2-step Standard (no Consistency Rule → threshold = 1.0) | `FTMOConfig.CONSISTENCY_RULE_THRESHOLD` |

## Verified Performance (v6.13 — Combined patch + obs no-leak audit, risk 0.7 %, 5000 eps, 2026-04-29)

| Metric | v6.11.3 baseline | **v6.13** | Δ |
|--------|------------------|-----------|---|
| Pass Rate | 3.4 % | **9.7 %** ⭐⭐⭐ | **+185 %** |
| Win Rate | 68.8 % | 64.8 % | -4 pp |
| Orders/ep | 6.1 | **7.7** | +26 % |
| Total DD max | 3.23 % | 4.40 % | +37 % (ห่าง 8 % limit) |
| Daily DD max | 2.12 % | 2.15 % | similar |
| Breach Rate | 0 % | **0 %** | same |
| Profit avg (5000 eps) | — | **+3.89 %** | — |

**Phase progression** (each = 5000-eps eval): leaky baseline 12.5 % → honest baseline 3.5 % → Phase C 1.5 % → Phase D 0.2 % → Phase E1 (calibration) 3.0 % → Phase E2 (aux task) 10.0 % → v6.11 SMC overhaul 0.0 % → v6.11.2 (rollback) 2.7 % → v6.11.3 (mild relax) 3.4 % → **v6.13 (combined patch) 9.7 %**. Details in [wiki/05-invariants.md § Version Log](wiki/05-invariants.md).

**Note**: the old "Option B 12.5 %" baseline was leaky (eval seeded with same pool used for GBM training). Honest baseline = 3.5 %. E2 is verified leak-free via runtime hook + obs feature audit.

---

## 🗺️ Wiki Navigation

| File | Read when you need to... |
|------|---------------------------|
| [wiki/01-architecture.md](wiki/01-architecture.md) | Understand the 3-brain pipeline, data flow, loop priority, and training pipeline overview |
| [wiki/02-modules.md](wiki/02-modules.md) | Find which class / method lives in which file (module map with symbol names) |
| [wiki/03-rl-training.md](wiki/03-rl-training.md) | Inspect the 27-dim obs layout, reward structure, PPO hyperparams, and curriculum |
| [wiki/04-operations.md](wiki/04-operations.md) | Understand main loop priority, FTMO state machine, news, sessions, cooldowns |
| [wiki/05-invariants.md](wiki/05-invariants.md) | ⛔ Rules that must not be broken, migration notes, and version log |

**Tip for LLM**: always start at `context.md`. Drill into the relevant wiki page only when needed. You do not need to read every file.

---

## Directory Layout (condensed)

```text
ftmo_trading_bot/
├── main.py                  ← FTMOTradingBot (live entry)
├── config/settings.py       ← MT5Config, FTMOConfig, SymbolConfig, bot_config
├── strategy/                ← SMCStrategy + 5 detectors (OB, FVG, Sweep, Structure, Indicators)
├── ml/                      ← SignalQualityModel, SelfLearningAgent, FTMOSignalFilterEnv, StrategyBacktester
├── core/                    ← RiskManager, MT5Connector, TimeManager, PositionSizer, NewsCalendarScheduler, DiscordNotifier
├── execution/               ← TradeExecutor, TradeManager
├── analytics/               ← PerformanceAnalyzer + TradeLogger (Excel: Trades 64 cols / Signals 21 cols / Daily / Stats)
├── scripts/                 ← build_signal_pool, train_signal_quality, train_signal_filter, fetch_mt5_data
├── data/                    ← OHLCV CSVs + signal_pool + ml_model pkl (with isotonic calibrator)
├── models/                  ← ppo_signal_filter.zip + vec_normalize_sf.pkl (aux-aware policy weights)
└── logs/                    ← bot_state.json + ftmo_trades.xlsx + tensorboard + news_scheduler_state
```

Full module details → [wiki/02-modules.md](wiki/02-modules.md).

---

## Quick Commands

**Training (3 steps, in order):**

```bash
python scripts/build_signal_pool.py --pool_size 3000 --workers 8
python scripts/train_signal_quality.py
python scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 4500 --outcome_noise 0.05 \
    --ml_threshold 0.40 --risk_per_trade 0.007
```

Phase E2 trainer uses `AuxAwarePPO` + `AuxAwareACPolicy` automatically (aux loss weight = 0.5).

**Evaluation:**

```bash
python scripts/train_signal_filter.py --eval_only \
    --pool_size 4500 --ml_threshold 0.40 --risk_per_trade 0.007
```

Default 5000 episodes. Use `.venv/bin/python` (not bare `python`) — version mismatch can shift Pass Rate.

**Live:**

```bash
python main.py
```

**TensorBoard:**

```bash
tensorboard --logdir logs/tb_signal_filter
```

---

## 🔴 Critical Invariants (must-read before editing)

- ⛔ **Obs 27 dims sync** — changing count or order of features requires retraining the whole pipeline.
- ⛔ **Position ID matching** — use `position_id`, never `ticket`.
- ⛔ **Do not delete `logs/bot_state.json`** mid-challenge.
- ⛔ **Timezone**: broker = EET, config = UTC — convert before comparing.
- ⛔ **Do not use `mt5.symbol_info_tick().time` directly** (FTMO quirk, +3 h drift). Use `datetime.now(Bucharest)` via `TimeManager`.
- ⛔ **Risk per trade** in live must match the training risk (`DEFAULT_RISK_PER_TRADE_PCT` ↔ `--risk_per_trade`).
- ⛔ **FTMO program type** = 2-step Standard → `CONSISTENCY_RULE_THRESHOLD = 1.0`.

Full list → [wiki/05-invariants.md](wiki/05-invariants.md).

---

## 📝 Wiki Maintenance Protocol

**When?** Every time `.py` files under `ftmo_trading_bot/` are edited.

**Update which files?**:

- Obs dim / feature / order → `wiki/03-rl-training.md` + `wiki/05-invariants.md` (version log)
- Config values (risk, symbols, DD thresholds) → `context.md` (Headline Numbers) + `wiki/04-operations.md` + `readme.md`
- Module signature / class name change → `wiki/02-modules.md`
- Loop / state machine change → `wiki/04-operations.md`
- User-facing change → `readme.md` (Thai)

**Last Updated** — bump the date on every file you touch (top of file).

**Source references** — always use class / method / variable names. Never line numbers (they rot quickly).

**Language** — docs in English; `readme.md` in Thai. See `CLAUDE.md`.

Details → [CLAUDE.md](CLAUDE.md).
