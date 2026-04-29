# 05 — Invariants & Gotchas (Rules Not to Break)
> Last Updated: 2026-04-29 | Scope: red flags, version log, migration notes (latest: v6.11 SMC precision overhaul — Counter-D1 hard veto, Sweep+IDM+M15 BOS hard gates, BE best_price trigger, OB grading)

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

## ❓ FAQ / Common Misunderstandings

### Q: Partial close + BE → ชน SL = WIN หรือ LOSS?

**A: WIN** (ถ้า partial profit > 0 และ accumulated net profit > 0).

`TradeExecutor.sync_with_mt5` ดึง **ทุก deals** ของ position ผ่าน `MT5Connector.get_deals_by_position(ticket)` แล้ว accumulate ทั้ง `profit + swap + commission` ของทุก deal. ดังนั้น partial close ($+50) + BE-SL remainder ($0) → `ExecutedTrade.profit = +$50` → WIN.

3 จุดที่ classify ใช้สูตรเดียวกัน (`profit > 0` บน cumulative value):

- `TradeLogger.log_daily_summary` — daily wins counter
- `TradeExecutor.get_stats` — overall win rate
- `RiskManager.update_daily_pnl` — `consecutive_losses` reset เมื่อ `pnl > 0`

ตัวอย่าง: risk = $100, RR target = 2.0
- Partial 50% @ 1R → realized +$50
- SL เลื่อนมา BE → remainder ชน BE-SL → realized $0
- `ExecutedTrade.profit = +$50` → WIN, `consecutive_losses = 0`, daily P/L +$50

⚠️ ถ้าโดน **swap/commission** ทำให้ accumulated < 0 → จะกลายเป็น LOSS ตามกฎเดียวกัน (cumulative-based, ไม่ใช่ remainder-only).

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

### 2026-04-29 — v6.11 SMC Precision Overhaul (post live demo audit)

Live demo Day-3 (2026-04-28) เจอ EV ติดลบ (PF 0.96, Net −$19.66, WR 46.2 %) แม้ไม่ผิด FTMO. Audit หาเจอว่า SMC entry gate **หลวมเกินไป** — Sweep, IDM, Fresh BOS, Counter-D1 ทั้งหมดเป็น *bonus* ไม่ใช่ *prerequisite*. รวมถึง TradeManager BE trigger พลาด trade ที่ MFE สูงแล้วย้อน, และ TradeLogger logging gap หลายฟิลด์.

**Tier 1 — Quick Wins (ไม่ต้อง retrain):**

- **`TradeManager._manage_single_position`** — BE/Partial trigger เปลี่ยนจาก `current_rr` เป็น `best_rr` (rolling MFE-based). track `state.best_price` ทุก tick (ไม่รอ trailing activate). Trade ที่ MFE สูงระหว่าง 5 s tick แล้ว revert จะ lock BE+Partial ทันที.
- **`TradeManager._partial_close`** lot_min branch — mirror `trade.partial_closed_flag = True` (เพิ่มจากเดิมที่ตั้งแค่ `partial_close_skipped=True`). log แสดงตรงกับ state จริง.
- **`SMCStrategy._evaluate_buy_signal/_evaluate_sell_signal`** — Counter-D1 เปลี่ยนจาก soft +15 confluence threshold bump → **hard reject**. BUY: `d1_bias == -1` → reject; SELL: `d1_bias == +1` → reject. Neutral (0) ผ่านได้ปกติ.
- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter F2 — เพิ่ม **Quiet-vol × off-overlap blocker**. ถ้า `atr_pips < 1.2 × atr_floor_pips` AND `_get_session_multiplier() < 1.0` → reject. ลด trade quiet vol นอก London-NY overlap ที่ pattern reliability ต่ำ.

**Tier 2 — Medium (ไม่ต้อง retrain):**

- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter E2 — **ADX(H4) ≥ 22 hard gate**. ดึงจาก `_htf_data["adx"]`. ลด whipsaw ใน H4 ranging.
- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter G — **Recent Sweep prerequisite within 8 bars**. ใช้ `LiquiditySweepDetector.get_recent_bullish/bearish_sweep(max_bars_ago=8)` เป็น hard gate (ก่อนหน้านี้แค่ confluence bonus). ทุก entry บังคับมี smart-money confirmation.
- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter H — **Fresh M15 BOS/CHoCH structural shift within 6 bars**. ใช้ `_structure_ltf.get_latest_event()` + ตรวจ `event.index >= len(ltf_df) - 6`. ตัด pullback-to-OB-without-break trades.
- **`TradeSignal` dataclass** — เพิ่ม fields: `htf_score, mtf_score, ob_pts, fvg_pts, sweep_pts, sweep_age_bars, htf_bias` (string label), `d1_bias`. populate ใน BUY/SELL eval.
- **`FTMOTradingBot._build_live_context`** — อ่าน per-component pts + `htf_bias` จาก `signal` ตรงๆ (ก่อนหน้านี้ hardcode 0). แก้ Trades sheet HTF/MTF/OB/FVG/Sweep pts ที่ว่างเปล่า.
- **`FTMOTradingBot._log_signal_scan`** — `htf_bias` field ใช้ `sig.htf_bias` (string "BULLISH/BEARISH/RANGING") แทน `_strategy._htf_bias` (int).

**Tier 3 — Strategic:**

- **NEW `strategy/inducement.py` — `InducementDetector` class.** ตรวจจับ rejection candle (wick failed) ภายใน 8 bars. API: `detect_idm(df, direction) -> Optional[InducementEvent]`. wired เข้า `SMCStrategy._evaluate_buy/sell_signal` หลัง Sweep block: IDM = +10 confluence; ไม่มี IDM = -5 (อาจเป็น obvious swing).
- **`OrderBlock`** dataclass — เพิ่ม field `ob_grade: str = "INTERNAL"`.
- **NEW `OrderBlockDetector._classify_ob_grade(ob, df, avg_impulse)`** — จัดประเภทเป็น `EXTREME` (ใกล้ swing extreme ของ window 50 bars), `DECISIONAL` (impulse ≥ 1.8 × avg), หรือ `INTERNAL`.
- **`OrderBlockDetector._score_order_blocks`** — apply grade weight: EXTREME ×1.20, DECISIONAL ×1.00, INTERNAL ×0.60. ลด false-positive จาก Internal OBs.

**Mandatory verification before next live deploy:**

1. **Schema migration** — ถ้า user เคย deploy ก่อน v6.11: lint error อาจไม่กระทบ แต่ field `obs_27_json` + per-component pts ต่าง → rename `logs/ftmo_trades.xlsx` ถ้าเปิดรอบใหม่.
2. **Smoke test**: รัน `python main.py` ใน demo MT5 ≥ 4 ชม. ใน London-NY overlap window
3. **ตรวจ Trades sheet**: `HTF Bias` (string), `MTF Bias` (int), `ADX H4` (float), `HTF pts/MTF pts/OB pts/FVG pts/Sweep pts` (int) — ทั้งหมดต้องมีค่าจริง ไม่ใช่ 0/null
4. **ตรวจ Signals sheet col 20** (Executor Reject) — TAKE_FAIL rows ต้องมี reason (verify v6.10d ทำงาน)
5. **ตรวจ BE Moved + Partial Closed**: ถ้า MFE > 1 R ต้อง True ทั้งคู่ หรือ `Partial Skipped=True` (lot น้อย)
6. **ตรวจไม่มี trade ที่ Counter-D1**: BUY ไม่ควรเข้าตอน D1 = -1; SELL ไม่ควรเข้าตอน D1 = +1

**Expected outcome (post Tier 1):**

- WR: 46 % → 60-65 %, PF: 0.96 → 1.3+, Expectancy: −$1.51 → +$3 to +$5
- Trades/day: 13 → 7-9 (volume ลด ~40 % แต่ quality ขึ้น)
- Counter-D1 trade %: 23 % → 0 %
- MFE-then-SL anomaly: 46 % → < 5 %

**ที่ไม่ทำ (out of scope):**

- ❌ Retrain RL/GBM — entry gate เปลี่ยนแล้ว → re-eval 5000 eps ก่อนตัดสิน. Pool distribution อาจต่าง แต่ fields obs_27 ไม่เปลี่ยน → existing model ยังใช้ได้
- ❌ Tighten `MIN_CONFLUENCE_SCORE` 70 → 75 — Tier 1.3 + 1.4 ตัด ~30-40 % volume แล้ว, ตึงเพิ่มเสี่ยง undertrade
- ❌ Reduce risk per trade — risk 0.7 % verified Pass Rate 10 % แล้ว, ปัญหาคือ entry quality ไม่ใช่ sizing

### 2026-04-25 — v6.4 SMC Phase C (4 professional principles)

Addresses root-cause gaps that Phase A (bug fixes) + Phase B (reward tuning) could not reach. All 5 sub-tasks landed in one pass. Requires **pool rebuild + GBM retrain + RL retrain** before deploy (strategy layer changed).

**SMC `smc_strategy.py`:**

- **C1 — ADX threshold raised 20 → 25** in BUY + SELL pre-filters. Industry standard for "actual trend vs ranging". Expected signal volume drop ~30 %.
- **C3 — H4 POI hard gate (new, Principle 2):** added `_get_h4_poi_zones` + `_is_near_h4_poi`. Before confluence score, signal is rejected if price is > 2 ATR away from an H4 bullish OB/FVG (for BUY) / bearish (for SELL). Cache per-symbol, invalidated when H4 bar timestamp changes. New state: `_h4_poi_cache`, `_ob_detector_h4`, `_fvg_detector_h4` (separate instances to avoid M15 state contamination).
- **C4 — IDM sweep soft gate (Principle 3):** in sweep scoring block, OB without recent IDM sweep now costs `-20` (old OB, age > 5 bars) or `-8` (fresh OB). Sweep + OB together adds `+10` bonus (ideal smart-money pattern).
- **C5 — FVG + BOS conjunction (Principle 4):** after MTF bias scoring, if LTF (M15) had recent BOS, check for active M15 FVG. BOS without FVG → `-15` (weak break). BOS + FVG → `+8`.

**SMC `market_structure.py`:**

- **C2 — `is_valid_pullback` helper (new, Principle 1):** added method with 3 gates:
  1. impulse size ≥ 1.0 × ATR (no tiny wobble)
  2. pullback retracement ≥ 30 % of impulse (deep enough)
  3. pullback depth ≥ 0.25 × ATR (absolute floor, guards wick-only "BOS")
- Wired into `detect_structure_breaks`: when close breaks active swing high/low, `is_valid_pullback` is called first. If invalid → swing marked broken but NO event raised (internal noise rejected).
- Uses `df['atr']` column — already populated by `TechnicalIndicators.calculate_all` upstream, no new param plumbing.

**Lookahead / correctness:**

- All checks operate on confirmed-closed bars (iloc slicing ≤ current bar index).
- H4 POI cache invalidates by bar_ts equality — new H4 close triggers recompute.
- Mirror BUY/SELL logic verified identical except direction.

**Pipeline impact:**

- Pool will shrink (stricter filters) — monitor signals-per-episode, raise `pool_size` if < 12 avg.
- VecNormalize stats from previous training are invalid — must retrain RL with `--fresh`.
- `obs[0] confluence_norm` distribution shifts (IDM/FVG penalties widen range).

**Retrain sequence (required):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/build_signal_pool.py --pool_size 3000 --workers 8
.venv/bin/python ftmo_trading_bot/scripts/train_signal_quality.py
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected: Pass Rate 3.7 % → 6-10 %, Win Rate 49.6 % → 53-57 % (quality-first), Orders/day 0.26 → 0.15-0.20 (fewer but better).

### 2026-04-25 — v6.9 Live Logging (Schema v3) — comprehensive demo data capture

Re-enabled `TradeLogger` for live demo deployment. Schema v3 = 58-col Trades sheet + 19-col Signals sheet to capture data needed for E1/E2/baseline comparison vs live behavior.

**`analytics/trade_logger.py`:**

- Extended `TRADE_HEADERS` from 31 → 58 cols. New cols (24 fields, grouped):
  - **ML / Agent decision** (5): `ML Score (cal)`, `ML Score (raw)`, `Agent Action`, `Agent Decision`, `ML Threshold`
  - **Confluence breakdown** (5): `HTF pts`, `MTF pts`, `OB pts`, `FVG pts`, `Sweep pts`
  - **Trade mgmt state** (4): `BE Moved`, `Partial Closed`, `Trailing`, `Final SL`
  - **Live execution** (5): `Bid@Entry`, `Ask@Entry`, `Spread (pips)`, `Bid@Exit`, `Ask@Exit`
  - **Market context** (4): `ADX H1`, `ADX H4`, `MTF Bias`, `D1 Bias`
  - **Account state** (3): `Balance@Entry`, `Balance@Close`, `Equity Peak`
- Added `log_signal_scan(scan_data)` method + `SIGNAL_HEADERS` (19 cols) — per-scan log including `AGENT_SKIP` / `AGENT_TAKE_FAIL` / `REJECTED` / `NO_SIGNAL` results. Color-coded by result.
- New `Signals` sheet auto-created on first scan log.

**`execution/trade_executor.py`:**

- Extended `ExecutedTrade` dataclass with 24 new fields matching the logger schema.
- `to_dict()` now exports all new fields.
- `execute_signal(signal, live_context=None)` accepts a context dict from main.py. Fields populated into `ExecutedTrade` if context provided.
- Close path captures `bid_at_exit`, `ask_at_exit`, `balance_at_close`, `equity_peak_during_trade`, `final_sl_at_close`.

**`execution/trade_manager.py`:**

- BE move (`_move_to_breakeven`) → mirrors `state.breakeven_moved` to `trade.be_moved` and `trade.final_sl_at_close`.
- Partial close (`_partial_close`) → mirrors `state.partial_closed` to `trade.partial_closed_flag`.
- Trail activation (`manage_position`) → mirrors `state.trailing_active` to `trade.trailing_active`.
- Trail SL update (BUY/SELL paths) → mirrors `new_sl` to `trade.final_sl_at_close`.

**`main.py`:**

- Re-enabled `TradeLogger` (was `None`); now `TradeLogger(log_dir=logs/)`.
- New `_build_live_context(sig)` method — computes `ml_score` (cal+raw), bid/ask snapshot, ADX H1/H4, MTF/D1 bias, balance at entry. Read from `_quality_model`, `_strategy._mtf_data`/`_htf_data`, `_connector`, `_risk_manager`.
- New `_log_signal_scan(sig, ctx, result)` — wrapper that builds `scan_data` from signal + context.
- Run loop (`scan_all_symbols` → for each sig) now logs every scan as `AGENT_SKIP` / `AGENT_TAKE` / `AGENT_TAKE_FAIL`. Passes `live_context` to `executor.execute_signal`.

**`requirements.txt`:**

- Added `openpyxl >= 3.1.0` — required for TradeLogger Excel output.
- Added `tqdm >= 4.65.0` and `rich >= 13.0.0` — required by `stable-baselines3` `model.learn(progress_bar=True)`.

**Overtrading detection (added 2026-04-25):**

- `ExecutedTrade` extended with 4 new fields (62 trade cols total):
  - `trades_today_at_open`: count of trades opened today before this one
  - `trades_last_hour_at_open`: count in trailing 60-min window
  - `secs_since_last_trade_open`: delta from last trade open (any symbol)
  - `secs_since_last_trade_same_symbol`: delta from last trade open (same symbol)
- `FTMOTradingBot` tracks `_trade_open_history: List[(datetime, symbol)]` (capped at 200 entries).
- `_build_live_context` computes the 4 metrics from history; passed to executor via `live_context`.
- Use case: filter Trades sheet by `Sec Since Last Open` < 60 → see clusters of fast-fire trades (overtrading symptom).

**Smoke test:** TradeLogger smoke test passed — 58 trade cols + 19 signal cols, all 4 sheets created (Trades, Daily, Stats, Signals).

**Output:** `ftmo_trading_bot/logs/ftmo_trades.xlsx` updated in real time during live run.

**Console quiet mode (added 2026-04-25):**

- Idle states (Daily Halt / Friday close / Weekend / Daily Close 23:30 / Rollover) ใช้ pattern **announce-once** แทน `if loop_count % N == 0` — print ครั้งเดียวตอน entry, silence จนกว่าออกแล้วเข้าใหม่.
- `FTMOTradingBot.__init__` มี 5 flags: `_daily_halt_announced`, `_friday_announced`, `_weekend_announced`, `_daily_close_announced`, `_rollover_announced` (init=False). Reset = False อัตโนมัติใน `else` branch ของแต่ละ state guard.
- Per-signal `AGENT_SKIP` print ลบทิ้ง — ข้อมูลครบใน `Signals` sheet (`AGENT_SKIP` row พร้อม ml_score, confidence, reasons).
- Per-signal `NO_AGENT` print ลบทิ้ง — fallback path เมื่อไม่มี RL agent loaded; logged ใน Signals sheet เช่นกัน.
- เก็บ `📡 [Agent] TAKE` print ไว้ — เป็น event สำคัญที่บอกว่าเทรดกำลังจะเปิด.

**v6.10d — Fix `_log_signal_scan` propagation bug (added 2026-04-28):**

Day-2 live demo analysis เจอว่า `Executor Reject` column ใน Signals sheet ทุกแถวเป็น None (104 AGENT_TAKE_FAIL events). Root cause: `FTMOTradingBot._log_signal_scan` สร้าง `scan_data` dict ครอบคลุมแค่ 19 keys (time, symbol, direction, ml_score, ฯลฯ) — **ไม่ copy `executor_reject_reason` และ `obs_27_json` จาก `live_context`** → `scan_data.get(...)` คืน "" → Excel cell empty.

**Fix:** เพิ่ม 2 keys ใน `_log_signal_scan()` scan_data:

```python
scan_data = {
    # ... existing 19 keys ...
    "executor_reject_reason": live_context.get("executor_reject_reason", ""),
    "obs_27_json": live_context.get("obs_27_json", ""),
}
```

ผลคือ col 20 (Executor Reject) + col 21 (Obs27 JSON) ของ Signals sheet จะมีค่าหลัง deploy. **Trade-level Obs27 JSON (col 64 ของ Trades sheet) ทำงานปกติอยู่แล้ว** เพราะ flow ผ่าน `ExecutedTrade.to_dict()` ที่ copy field จาก live_context ครบ.

**Verify after deploy:** หลัง bot รัน + เกิด AGENT_TAKE_FAIL → ตรวจ Signals sheet col 20 ต้องมี reject reason เช่น `spread_high:18>15` / `risk_manager:Cooldown active ...` / `correlation:USD pair max`. ถ้ายัง None → bug อื่น

**Day-2 fixes ที่ verified working:**

- ✅ Schema 64/21 cols deployed
- ✅ Stats sheet update hourly
- ✅ Timezone fix — DD@Entry แสดงค่าติดลบ (-0.32%) ตอน gain ✓ ตรงกับ MT5 reality
- ⚠️ Symbol coverage partial — GBPJPY scan ได้แล้ว (Day-1 = 0, Day-2 = 7) แต่ XAUUSD/USDJPY/USDCHF/EURJPY ยัง = 0 (อาจเพราะ broker rename — ต้องดู console output)

---

**v6.10c (Phase 1b) — Symbol coverage fix: pre-select Market Watch (added 2026-04-27):**

Live demo Day-1 analysis เจอว่า **5 ใน 10 symbols ไม่มี scan event เลย:** XAUUSD (0), USDCHF (0), USDJPY (2), EURJPY (0), GBPJPY (0).

**Root cause:** `MT5Connector.connect()` ไม่ pre-select symbols ใน Market Watch หลัง login. ผลคือ:

- `analyze()` เรียก `get_current_price(symbol)` → `mt5.symbol_info_tick(symbol)` คืน None ถ้า symbol ไม่อยู่ใน Market Watch
- `analyze()` return no_signal ทันที (line 343-344) — **ก่อน** จะถึง `get_symbol_info(symbol)` ที่จะ trigger `mt5.symbol_select()`
- บอท skip silently ตลอด — ไม่มี scan event ใน Excel

**Fix:** ใน `MT5Connector.connect()` หลัง login สำเร็จ → loop `bot_config.symbols.symbols` ทั้งหมดเรียก `mt5.symbol_select(sym, True)` ครั้งเดียว. Print summary `📌 Market Watch: enabled N/M symbols` + warn ถ้า broker ไม่รองรับ symbol บางตัว.

**Verify after deploy:** หลัง bot start ดู console message — ต้องเห็น `📌 Market Watch: enabled 10/10 symbols`. ถ้า < 10 → ดู warning ว่า broker ไม่รองรับ symbol อะไร (อาจเป็นชื่อต่าง เช่น `XAUUSD.r` แทน `XAUUSD`).

**v6.10c — Timezone fix: Excel timestamps ใช้ broker time (EEST) (added 2026-04-27):**

Live demo Day-1 analysis เปรียบเทียบ Excel `Open Time`/`Close Time` กับ MT5 history เจอว่า **Excel timestamps มี offset +4 ชม. จาก MT5** (Excel 16:39 vs MT5 12:39).

Root cause: production code หลายจุดใช้ `datetime.now()` (Python system local time = Bangkok VPS UTC+7) แทนที่จะใช้ broker time (EEST UTC+3). ทำให้ Excel timestamps คนละ wall-clock กับ MT5 history → debug ยาก + Friday/Daily close logic อาจ trigger ผิด.

Pattern: `TimeManager.get_server_time().replace(tzinfo=None)` — ได้ naive datetime ที่ display เป็น EEST wall clock (ตรงกับ MT5).

**Files patched:**

- `execution/trade_executor.py` — `execute_signal` capture entry time, `sync_with_mt5` close_time, `update_close` close_time. + import `TimeManager`.
- `main.py` — `_build_signal_observation` challenge_day calc, `_build_live_context` overtrading window, `_trade_open_history.append` (เก็บ EEST timestamp), Daily Halt print message.

**ที่ไม่แตะ (intentionally `datetime.now()` ยังใช้ได้):**

- `mt5_connector.py` `history_deals_get(today_start, datetime.now())` — MT5 API doc บอกใช้ naive UTC, library auto-convert
- mock data / test helpers / shutdown uptime calc — ไม่กระทบ FTMO logic
- `_signal_handler` / `__repr__` — diagnostic only

**Verify after deploy:** Excel `Open Time` ของ trade ใหม่ ต้อง match MT5 history mobile screen (วินาทีตรงกัน). ถ้ายังต่าง 4 ชม. → VPS NTP ไม่ sync หรือ timezone setting ผิด.

**v6.10b — Daily/Stats sheets fix (added 2026-04-27):**

Live demo Day-1 analysis เจอว่า Daily sheet + Stats sheet **ว่างเปล่า** ทั้งวัน ทั้งที่มี 6 trades. Root cause: `log_daily_summary()` + `update_stats_sheet()` ถูกเรียกแค่ใน `_run_phase4_tests()` (test function) — **ไม่เคยเรียกใน production loop** ตลอด.

**Fix:** เพิ่ม 3 hooks ใน `FTMOTradingBot`:

- `__init__`: `self._last_logged_day = None`
- `run()` ต้น loop iteration ก่อน `check_risk()`: ถ้า `broker_today != _last_logged_day` → flush ของวันก่อน (`log_daily_summary` + `update_stats_sheet`) → set `_last_logged_day = broker_today`. ครั้งแรกที่ loop รัน (None → today) ไม่ flush เพราะไม่มีวันก่อน
- `run()` ใน loop: ทุก 720 loops (~1 ชม. @ 5s) → `update_stats_sheet()` only (สำหรับ live monitor — user เปิด Excel ดูสถานะปัจจุบันได้)
- `shutdown()`: ทั้ง `log_daily_summary` + `update_stats_sheet` ก่อน save state — กัน user Ctrl+C แล้วข้อมูลวันสุดท้ายหาย

**Verify after deploy:** รัน bot ≥ 1 ชม. → ตรวจ `Stats` sheet ต้องมี data; รัน cross-day → ตรวจ `Daily` sheet มี row ของวันก่อน.

**v6.10 — Executor reject reason logging (added 2026-04-27, schema bump 63 → 64 trade cols, 20 → 21 signal cols):**

Live demo day 1 analysis revealed **62% of scans = AGENT_TAKE_FAIL** but reject reason ไม่ถูกบันทึก — Signals sheet's "Reject/Skip Reasons" column เก็บ SMC signal reasons แทน. Blind spot ใหญ่ เพราะไม่รู้ว่า cooldown / spread / correlation / DD halt / post-TP lock / order_send_failed อันไหน reject signal.

Changes:

- `TradeExecutor.execute_signal` — ตั้ง `self._last_reject_reason` ที่ทุก rejection point (signal_invalid / correlation:* / lot_calc_failed / risk_manager:* / price_fetch_failed / spread_high:* / final_validation:* / order_send_failed). Reset ที่ต้นของแต่ละ call.
- `FTMOTradingBot.run` — หลัง `executor.execute_signal()` คืน None → อ่าน `executor._last_reject_reason` → save เข้า `live_context["executor_reject_reason"]` → log ลง Signals sheet col 20 ใหม่ "Executor Reject".
- `FTMOTradingBot._build_live_context` — เพิ่ม raw account state (`balance_at_entry`, `equity_at_entry`, `floating_pnl_at_entry`, `daily_start_equity`) สำหรับ debug ตัวเลข `dd_at_entry_pct` ที่อาจ misleading (เช่น แสดง 11% ตอนที่ net P/L บวก).
- `ExecutedTrade.partial_close_skipped: bool` (col 63 ใหม่) — distinguish "Partial Closed = True (fired)" จาก "Partial Skipped = True (lot too small ข้ามไป)". `TradeManager._partial_close` set flag เมื่อ `remaining < lot_min` หรือ `close_volume < lot_min`.
- `TradeLogger.SIGNAL_HEADERS` 20 → 21 cols (เพิ่ม "Executor Reject" ก่อน "Obs27 JSON"). `TRADE_HEADERS` 63 → 64 cols (เพิ่ม "Partial Skipped" ก่อน "Obs27 JSON" — ไม่กระทบ hardcoded col index ของ `log_trade_closed`).

⚠️ **Schema migration:** VPS ต้อง **rename หรือลบ `logs/ftmo_trades.xlsx` เดิมก่อน restart** ไม่งั้น append ผิด column.

**Retrain unlock (added 2026-04-25, schema bump 62 → 63 trade cols, 19 → 20 signal cols):**

- `ExecutedTrade.obs_27_json: str` — JSON-encoded 27-dim obs vector at decision time. Lets us reconstruct full RL state for offline retrain / pool augmentation from live data.
- `FTMOTradingBot._build_live_context` calls `_build_signal_observation(sig)` (same path the live agent sees) and stores `json.dumps([round(float(x), 4) for x in obs.tolist()])` in `ctx["obs_27_json"]`.
- `TRADE_HEADERS[-1] = "Obs27 JSON"`, `SIGNAL_HEADERS[-1] = "Obs27 JSON"`. Both populated from `live_context` / `scan_data`. Cell capped at 600 chars.
- Round-trip verified: `json.dumps` (4-dec round) → `json.loads` → numpy float32. Max error ≈ 5e-5 (< 1e-3 invariant).
- Wrapped in `try/except` — JSON build failure leaves `obs_27_json=""` (no log break).
- File size impact: ~250 chars/row × ~1000 trades/month ≈ 250 KB/month (negligible).
- Use case: reconstruct exact obs the live agent saw → train next-gen agent on live distribution drift, or seed pool augmentation experiments.

### 2026-04-25 — v6.9 Phase E2 — Auxiliary Task on PPO

Research-backed (arXiv 2411.01456) — auxiliary regression head on policy network forces the trunk to learn signal-outcome-informative representations. Paper reports Sharpe lift -2.61 → 0.24 (Dataset 1) and -2.93 → 0.47 (Dataset 2) on forex DRL.

**3 new files:**

- `ml/aux_rollout_buffer.py` — `AuxRolloutBuffer` extends `RolloutBuffer` with per-step `aux_targets` field (shape `(buffer_size, n_envs)`). Adds `aux_target` kwarg to `add()`, includes `aux_targets` in `get()` swap_and_flatten loop, returns extended `AuxRolloutBufferSamples` NamedTuple.
- `ml/aux_aware_policy.py` — `AuxAwareACPolicy` extends `ActorCriticPolicy` with `aux_head: nn.Linear(latent_dim_pi, 1)` and `predict_aux(obs)` method that runs obs through actor trunk → aux head → squeezed scalar.
- `ml/aux_aware_ppo.py` — `AuxAwarePPO` extends `PPO`. Overrides:
  1. `__init__` — defaults `rollout_buffer_class = AuxRolloutBuffer`, accepts `aux_loss_weight=0.5`.
  2. `collect_rollouts()` — copy of `OnPolicyAlgorithm.collect_rollouts` with one extra line: `aux_targets = np.array([info.get('aux_target', 0.0) for info in infos])` then `rollout_buffer.add(..., aux_target=aux_targets)`.
  3. `train()` — copy of `PPO.train` with extra `aux_loss = F.mse_loss(policy.predict_aux(obs), aux_targets)` added to total loss as `+ aux_loss_weight × aux_loss`. Logs `train/aux_loss` to TensorBoard.

**Env modification (`ml/signal_filter_env.py`):**

- Added `info['aux_target'] = float(sig.get('outcome_pnl_ratio', 0.0))` in `step()` info dict. AuxAwarePPO reads this from `infos` returned by `env.step` (vectorized).

**Training script (`scripts/train_signal_filter.py`):**

- Replaced `PPO("MlpPolicy", ...)` with `AuxAwarePPO(AuxAwareACPolicy, ..., aux_loss_weight=0.5, ...)` in P1.
- Replaced `PPO.load(...)` with `AuxAwarePPO.load(...)` in P2 transition + final eval — preserves aux head + buffer class on reload.

**Smoke test (100 k P1 + 50 k P2, n_envs=4):**

- ✅ Pipeline runs end-to-end without errors.
- ✅ `train/aux_loss` logged: 1.39-1.45 (near regression baseline `var(outcome) ≈ 1.5`, stable not exploding).
- ✅ `train/value_loss`: 0.58-0.69 (healthy).
- ✅ `train/policy_gradient_loss`: -0.005 to 0 (healthy small values).
- ✅ Model save/load round-trip works (eval reloaded model successfully).

**Risks (still open):**

- Full 10M+5M training may diverge if `aux_loss_weight=0.5` is too high — fallback to 0.1 if value_loss or aux_loss explodes.
- VecNormalize wraps env — `info['aux_target']` is unmodified (VecNormalize only touches obs/reward).
- v6.8 P2 stability fix (LR 5e-5, ent 0.02, threshold 20) is preserved — paired with E2 aux loss.

**Retrain required (only RL — pool + GBM unchanged from v6.8 calibrated):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected (per arXiv paper extrapolation): Pass Rate 3.0-3.5 % (B1v2/E1 baseline) → 4.5-7 %, with stable P2 (no early-stop), DD safer due to better feature learning.

### 2026-04-25 — v6.8 Phase E1 — Isotonic Calibration on GBM

Research-backed improvement (Niculescu-Mizil & Caruana 2005, MQL5 financial-ML series). GBM `predict_proba` was uncalibrated — raw probabilities clustered around 0.30-0.45 regardless of true frequency. Calibration tightens probability semantics so `ml_score` is interpretable as actual win rate.

**Changes (`scripts/train_signal_quality.py`):**

- Added `IsotonicRegression(out_of_bounds='clip')` fitted on OOF probabilities (group-aware via existing `GroupKFold` setup → no leakage).
- Pool re-scored with **calibrated** OOF probabilities instead of raw.
- Production save bundle now includes both `model` (base GBM) and `calibrator` (isotonic mapping).
- Brier score logged before/after for verification.
- 5-bin reliability diagram printed (pred_avg vs true_avg per bin).

**Changes (`ml/signal_quality.py`):**

- `SignalQualityModel.__init__` loads optional `calibrator` from payload (None → backwards-compat with old uncalibrated models).
- `score` and `score_batch` apply `calibrator.transform` after `model.predict_proba`.

**Verification (Phase E1 first run):**

- Brier 0.2243 → 0.2234 (-0.4 %).
- Reliability bins: 5/5 ✅ — `pred_avg` matches `true_avg` exactly across [0,0.3), [0.3,0.4), [0.4,0.5), [0.5,0.6), [0.6,1.0).
- Distribution: mean 0.356, std grew 0.050 → 0.076 (calibration spreads probabilities to match true frequency).
- Threshold analysis shift:
  - 0.33: WR 38.8 % → 39.6 %, EV `−0.005` → `+0.010` (flip to positive).
  - 0.40: WR 46.8 % → 46.6 %, EV `+0.154` → `+0.149` (almost identical, larger n).
  - 0.45: 3.5 % kept → 8.3 % kept, EV `+0.393` → `+0.275` (more samples at sweet spot).

**Why isotonic, not Platt:**

- Tree models (GBM) have non-sigmoid miscalibration → Platt's log-linear assumption fails.
- 106 k samples ≫ 1 000 → isotonic strictly dominant per Niculescu-Mizil 2005.

**Live impact:** `ml_score >= 0.40` reward bonuses in `FTMOSignalFilterEnv.step` now trigger at the *true* 46 % WR threshold instead of an arbitrary raw-prob bucket. Threshold for `--ml_threshold` should typically use 0.40 (sweet spot) instead of the previous 0.36.

**Retrain required (only RL):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.40 --risk_per_trade 0.007
```

Expected: Pass Rate 3.5 % → 4-5 % (calibration improves position sizing per Niculescu-Mizil chain: probs → sizing → Kelly → equity smoothness).

### 2026-04-25 — v6.7 Rollback Phase D (BE-only tested + rejected)

Phase D full (partial + BE + trail) and Phase D BE-only both failed to beat the B1v2 baseline (Pass Rate 3.7 %). After two experiments the evidence is strong enough to lock in a decision: **trade management inside the training backtester hurts Pass Rate for FTMO challenges**, even though it improves WR in isolation.

**Why trade management loses for Pass Rate:**

- FTMO 10 % target in 45 days is a *tail* objective — it needs high variance, not low variance.
- Partial close caps winners at 1.5R (locks 0.5R early, remaining half to 2R net) → reduces tail events.
- BE-only at 1R trigger kills trades that reach 1R and minor-pullback back to entry: in the raw pool these are 8.4 pp of former winners that became 0R, versus 9.6 pp of losers saved. Net EV per trade worsens (mean outcome moved from `−0.0645` to `−0.1051`).
- Distribution confirms it: `TP 2R+` bucket 12.8 % (B1v2) → 8.6 % (BE-only). Tail got thinner.

**Rollback actions:**

- `ml/strategy_backtester.py` `_resolve_trade` — reverted to the v6.3 B1v2 version (no BE / partial / trail; SL or TP only, with bar-color heuristic for same-bar).
- `execution/trade_manager.py` constants — restored to live defaults (`PARTIAL_CLOSE_PCT=0.5`, `PARTIAL_TRIGGER_RR=1.0`, `TRAIL_ACTIVATION_RR=1.5`, `TRAIL_ATR_MULTIPLIER=1.0`). Live still uses trade management; only the training backtester is flat.
- Pool restored from `data/signal_pool_3000.pkl.bak_v6_2` (identical to the v6.3 B1v2 pool — 2 887 episodes, 106 454 signals, mean outcome `−0.0645`).
- GBM retrained on the restored pool (expected OOF AUC ≈ 0.5875).

**Note on train-live alignment:** with rollback, the training env now *under*-estimates live performance because live has BE + partial + trail while training does not. This is an *acceptable* direction of mismatch (live ≥ train) — the alternative (Phase D) produced the wrong direction (live worse than train in Pass Rate terms). Future work could revisit this gap with higher trigger points (e.g., BE at 1.5R with buffer) if empirical live data supports it.

**Retrain required (only RL — pool + GBM are restored/regenerated):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected to reproduce ≈ 3.7 % Pass Rate. Once confirmed, the bot can be deployed to demo for live data collection — further offline tuning hit diminishing returns.

### 2026-04-25 — v6.5 Phase D Train-Live Alignment (rollback Phase C)

Phase C (SMC 4 principles, 2026-04-25 earlier) reduced pool 44 % but Pass Rate dropped 3.7 % → 1.5 % — **rolled back fully** in same day. Root cause confirmed: Phase C filters removed signals proportionally without improving WR, and pool shrinkage caused PPO P2 to early-stop.

Phase D attacks the real gap: **backtester `_resolve_trade` did not simulate BE / partial-close / trailing** that live `TradeManager` performs. Training pool outcomes therefore misrepresented realized RR.

**Rollback (all Phase C changes reverted):**

- `smc_strategy.py` — removed `_get_h4_poi_zones`, `_is_near_h4_poi`, H4 POI gate (BUY+SELL), H4 POI soft scoring, IDM sweep penalty, FVG + BOS conjunction, ADX 25 threshold (back to 20). `_ob_detector_h4`, `_fvg_detector_h4`, `_h4_poi_cache` removed from `__init__`.
- `market_structure.py` — removed `is_valid_pullback`, wire-in dropped from `detect_structure_breaks`.
- `strategy_backtester.py` `_init_strategy` — removed `_ob_detector_h4`, `_fvg_detector_h4`, `_h4_poi_cache` setup.

**Phase D new (`ml/strategy_backtester.py`):**

- Added class constants mirroring `TradeManager`: `_BE_TRIGGER_RR=1.0`, `_PARTIAL_CLOSE_PCT=0.5`, `_PARTIAL_TRIGGER_RR=1.0`, `_TRAIL_ACTIVATION_RR=1.5`, `_TRAIL_ATR_MULTIPLIER=1.0`.
- Rewrote `_resolve_trade` as bar-by-bar state machine with fields: `effective_sl`, `partial_closed`, `partial_gain_R`, `trail_active`, `best_price`.
- On 1R hit: partial close 50 % (locks `+0.5R`) and moves SL to entry (BE).
- On 1.5R hit: activates trailing — SL = `best_price ± ATR × 1.0` (one-way).
- Gap / force-close logic preserved; applies to `effective_sl` so gap-SL below BE still counts as 0 R on remaining half.
- Outcome `total_R = partial_gain_R + remaining_pct × exit_R` — combines locked partial with remaining exit.

**Unit tests (6 scenarios) all pass:**

- TP direct hit → `+1.5R` (partial +0.5R + TP on remaining 50 %).
- Partial + BE stop (no trail) → `+0.5R` (half locked, half BE = 0R).
- Full SL before 1R → `−1R`.
- Partial + trail stop → `~+1R` (trail above entry, not full RR).
- SELL mirror of case 1 → `+1.5R` as expected.
- Timeout after partial → locked partial + remaining at last close.

**Pool v6.5 snapshot (after rebuild):**

- 2 887 episodes, 106 454 signals (same as v6.3 B1v2 baseline — rollback + Phase D preserves signal count).
- WR (outcome > 0) **35.56 % → 45.83 %** (+10.3 pp).
- New distribution buckets visible: `+0.1 to +0.5R` (18.0 %) and `+1.0 to +1.5R` (19.7 %) — partial-win outcomes that didn't exist before.
- Mean outcome −0.0887 (slightly more negative than v6.3's −0.0645) because partial-cap reduces winners from 2R → 1.5R on average.

**Retrain required:**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_quality.py
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected: Pass Rate 3.7 % → 6-10 %, WR 49.6 % → 55-60 %, DD max safer because BE caps downside of partial-winners.

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
