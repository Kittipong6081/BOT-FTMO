# SYSTEM_STATE_CONTEXT (End of Phase 5 + Post-Review Patches)

## 📌 Project Overview
**Name:** FTMO Algorithmic Forex Trading Bot
**Goal:** A fully autonomous trading system tailored specifically to pass the FTMO challenge and manage a funded account securely.
**Current Status:** All 5 phases finished. Discord Webhook integration active for real-time monitoring.
**Last Updated:** 2026-04-13 — Comprehensive code review + bug fix pass (13 issues)

## 🏗️ Architecture & Modules

### 1. Phase 1: Core Systems (`core/`)
- **`MT5Connector`**: Deals with MetaTrader 5 API connection, OHLCV data retrieval, position fetching, and emergency hard-closing. Currently supports a mock mode if `MetaTrader5` is unavailable. Mock data covers **all 9 configured symbols**. Symbol cache is **cleared on disconnect/reconnect** to avoid stale broker specs.
- **`RiskManager`**: Enforces strict FTMO rules (4% daily drawdown, 8% max total drawdown). Persists state locally via JSON to survive restarts. **Daily reset now uses broker EET time** (`TimeManager.get_server_time().date()`) instead of local VPS time — prevents incorrect reset if VPS timezone differs from broker.
- **`PositionSizer`**: Calculates trade lot size based on account equity, risk per trade, and ATR multiplier. **Pip value for cross pairs (EURJPY, GBPJPY) now uses the correct USD conversion rate** (USDJPY) instead of the cross pair's own price, which was numerically wrong.
- **`DiscordNotifier`**: A centralized notification engine leveraging Discord Webhooks. **Now includes a sliding-window rate limiter** (20 req/min + 1s min interval) with 429 back-off handling, so bursts of trades don't hit Discord's webhook limit.

### 2. Phase 2: Strategy (`strategy/`)
- **`TechnicalIndicators`**: Computes ATR, EMA (fast/med/slow), and RSI. **Volatility filter is now pip-aware**: autodetects pip size from price level (0.01 for JPY pairs where price > 20, else 0.0001) instead of hardcoding `× 10000`. Previously JPY pairs produced ATR-in-pips values that were 100× too large.
- **`MarketStructure`**: Evaluates H1 market bias (BOS, CHoCH) holding swing highs and lows.
- **`OrderBlockDetector`**: Detects bearish/bullish Order Blocks (OB) on M15 timeframe.
- **`SMCStrategy`**: The backbone. Generates trade signals by scoring setups based on HTF structure, MTF bias, RSI, volatility, and executing strict RR filtering. Enforces correct trading session windows by rigorously converting server time (EET) to UTC using `pytz` prior to session config comparisons to prevent timezone shifting bugs. Will only trigger if the Confluence Score is above `MIN_CONFLUENCE_SCORE` (auto-tuned by AI).

### 3. Phase 3: Trade Execution (`execution/`)
- **`TradeExecutor`**: Verifies signals across **8 gates**: Valid → **Duplicate Symbol + Correlation** → Lot Sizing → Risk Manager → Spread → Final Validation → Market Order → Notification. **New correlation guard** groups highly-correlated pairs (USD_WEAK, USD_STRONG, JPY_CROSS, EUR_PAIRS, GBP_PAIRS) and caps same-direction exposure at 2 positions per group to prevent over-concentration. **Duplicate symbol check** prevents opening a second position in an already-traded symbol. P/L calculation on `close_trade()` now uses **actual `trade_contract_size`** from the broker symbol info (not hardcoded 100000) and correctly converts from quote currency to account currency for cross pairs.
- **`TradeManager`**: Actively manages positions via ATR-Trailing Stop, Break-Even rules, and Session-end liquidation. **Session-close logic now converts server EET time to UTC** before comparing against UTC-defined config values (friday_cutoff, newyork_end). Previously mixed EET time with UTC config → sessions closed at wrong hour. Friday hard-close at 20:45 EET is still enforced via `TimeManager.is_friday_close_time()` (EET-native).

### 4. Phase 4: Analytics (`analytics/`)
- **`TradeLogger`**: Converts active and closed trade dictionaries into formatted lines inside an Excel file (`logs/trading_log.xlsx`). Tracks open vs closed states using colored cell formatting.
- **`PerformanceAnalyzer`**: Consumes trade dictionaries to output standard stats (Win rate, Expectancy), advanced stats (Profit factor), and risk stats (Sharpe, Sortino, Drawdowns vs Target).

### 5. Phase 5: Reinforcement Learning (`ml/`) — Rebuilt in Post-Review
- **`FTMORewardCalculator`**: A specialized algorithmic critic penalizing drawdowns exponentially to prevent FTMO breaches whilst rewarding smooth growth.
- **`FTMOOptimizationEnv`**: **Completely rewritten** as a real mini-backtest engine (previously was `np.random.uniform` with hardcoded if/else — PPO couldn't learn anything meaningful).
  - **1 Episode = 1 FTMO Challenge (30 trading days)**; **1 step = 1 trading day**.
  - Loads real OHLCV from `data/ohlcv/*.csv` if available (EURUSD_M15.csv, USDJPY_H1.csv, etc.); falls back to synthetic data tuned to realistic Forex characteristics (ATR ~12 ± 4 pips on M15).
  - Per-day simulation: (1) generates market regime (trending / ranging / volatile / quiet), (2) calculates expected setups from confluence threshold, (3) adjusts win-rate based on confluence/RR/ATR-mult parameters, (4) simulates each trade outcome with break-even protection + slippage model, (5) tracks intraday + daily + total drawdown, (6) calls reward function.
  - **Observation space expanded from 5 → 8 dims**: total_dd, daily_dd, progress, sortino, last_trade_result, volatility, day_progress, recent_win_rate.
  - `map_actions_to_parameters` is now a `@staticmethod` — no need to instantiate a dummy env for inference.
- **`SelfLearningAgent`**: A PPO RL implementation via Stable-Baselines3. **Refactored to call the static mapper directly** (no dummy env). Old models with 5-dim obs space are auto-rebuilt on load failure. Now trains 4096 timesteps default (was 2048) since each episode is a full 30-day challenge.
- **`main.py (Integration)`**: Phase 5 enables Auto-Tuning at the initial system boot. `FTMOTradingBot` connects to the `SelfLearningAgent` object and fetches parameters dynamically bounding the `bot_config`.

## ⚙️ Active Variables (`config/settings.py`)
- **Core Parameters**: Bound to `bot_config` Singleton. Holds dynamic variables overridden by Phase 5 AI Agent. Connective credentials and API webhooks are isolated via `python-dotenv` reading from the `.env` file for enhanced security.
  - `ftmo.DEFAULT_RISK_PER_TRADE_PCT` (Range: 0.5% - 1.0%)
  - `ftmo.PREFERRED_RISK_REWARD_RATIO` (Range: 1.5 - 3.0)
  - `indicators.atr_sl_multiplier` (Range: 1.0 - 2.5)
- **Timeframes**: H4 (HTF Trend), H1 (Structure), M15 (Primary Entry).
- **Symbols**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY (9 pairs — mock mode supports all).
- **Test Pipeline**: `python main.py --test-all` covers unit tests 1 through 22, verifying Core, Strategy, Execution, Analytics, and RL Environments sequentially.

## 🛠️ Post-Review Patches (2026-04-13)

Bug hunt and fixes applied after full code review:

### 🔴 CRITICAL
1. **RiskManager daily reset timezone**: `date.today()` → `TimeManager.get_server_time().date()` in 8 places. VPS local date ≠ broker EET date at rollover → could cause wrong daily reset.
2. **RL Environment rebuild**: Random simulator replaced with real backtest engine (OHLCV-driven or synthetic fallback with realistic market regimes). PPO can now learn meaningful policies.

### 🟠 HIGH
3. **Duplicate symbol + correlation guard in TradeExecutor**: Prevents opening multiple positions in same symbol or correlated group (EURUSD/GBPUSD/AUDUSD together in "USD_WEAK" etc.).
4. **Session-close timezone in TradeManager**: Server EET time now converted to UTC before comparing UTC config values; Friday hard-close still EET-native.
5. **Pip value for cross pairs in PositionSizer**: EURJPY/GBPJPY now use USDJPY rate for USD conversion (not the cross pair's own price).
6. **Volatility filter pip-awareness**: No more hardcoded `× 10000`; autodetects pip size from price level → JPY pairs now filter correctly.
7. **P/L calculation in close_trade**: Uses actual `trade_contract_size` from broker and converts quote→account currency for non-USD-quote pairs.

### 🟡 MEDIUM
8. **Discord rate limiter in Notifier**: Sliding window 20 req/min, 1s min interval, 429 back-off handling.
9. **Mock data completeness**: `MT5Connector` mock price/symbol dictionaries now cover all 9 configured symbols (not just 4).
10. **Symbol cache invalidation**: `_symbol_cache` is cleared on `disconnect()` to prevent stale broker specs after reconnect.
11. **RL Agent refactor**: `map_actions_to_parameters` is now `@staticmethod`; `SelfLearningAgent.get_optimized_parameters()` calls it directly instead of instantiating a dummy env.

### 📦 Dependencies
No new pip packages added; all existing `requirements.txt` deps already cover the additions (collections/threading are stdlib).

---

## 🛠️ Post-Review Patches v2 (2026-04-14)

รอบที่สองหลังเจอปัญหาจริงจาก live log: over-trading 45 เทรด/วัน, PnL=$0 ใน bot_state, Discord entry_price=0, ไม่มี cooldown, state persistence หลุด, ML ยังเทรนจาก mock. แก้ทั้งหมด 6 ปัญหา + hardening 5 จุดเพิ่ม

### 🔴 CRITICAL (Trading Safety)

1. **MT5 deal matching bug (P/L = $0 ทุกที่)** — `TradeExecutor.sync_with_mt5` เคย match deals ด้วย `deal.order/deal.ticket` → หา deal ที่ปิดไม่เจอ → profit = 0. แก้โดย `MT5Connector.get_trade_history` เพิ่ม field `position` (จาก `deal.position_id` / `deal.position`) และ sync_with_mt5 match ด้วย `h["position"] == ticket` + ขยาย lookback 7 วัน + warning ถ้าไม่เจอ deal. เป็น root cause ของทั้ง state PnL=$0 และ Discord PnL=$0

2. **State persistence atomic write** — `RiskManager._save_state` ใช้ tmp file + `os.fsync` + `os.replace` กันไฟล์พังถ้า crash ระหว่างเขียน. เพิ่ม public `save()` method. `main.shutdown()` เรียก `_risk_manager.save()` ก่อน disconnect. state file ตอนนี้เป็น source of truth ที่เชื่อถือได้

3. **Cooldown / Anti-Revenge Trading** — state machine เต็มตัวใน RiskManager:
   - `_last_loss_time_per_symbol: Dict[str, iso_str]` — cooldown 30 นาที/คู่เงินหลังโดน SL
   - `_consecutive_losses: int` + `_halt_until: iso_str`
   - 2 แพ้ติด → pause ทั้งระบบ 60 นาที; 3 แพ้ติด → DAILY HALT
   - ทุก field persist ใน state JSON (กัน restart = reset cooldown)
   - Integrated ใน `can_open_trade()` ผ่าน `is_symbol_in_cooldown()` + `is_global_halted()` (auto-clears expired)

### 🟠 HIGH (Anti-Overtrading)

4. **Guardrails ใหม่ใน `config.ftmo`**:
   - `MAX_TRADES_PER_DAY: 5` (ก่อนหน้าไม่มี cap → 45 trades/วัน)
   - `MAX_CORRELATED_POSITIONS: 1` (ลดจาก 2)
   - `MIN_CONFLUENCE_SCORE: 70.0` (ขึ้นจาก 60 — **config-driven**, SMCStrategy อ่านจาก `bot_config.ftmo.MIN_CONFLUENCE_SCORE` ปรับ runtime ได้)
   - `COOLDOWN_AFTER_LOSS_MIN: 30`, `CONSECUTIVE_LOSS_PAUSE_COUNT: 2`, `CONSECUTIVE_LOSS_HALT_COUNT: 3`

5. **Discord webhook hardening** (`DiscordNotifier`):
   - `_fmt_price()` — 0/None → "N/A" + trim 5-decimal zeros (ไม่โชว์ "0.00000")
   - `_fmt_num()` — safe null/exception handling
   - `send_trade_open`: เพิ่ม Confluence, Session, RR fields + ⚠️ ถ้า entry = N/A
   - `send_trade_close`: เพิ่ม PnL% ใน R-multiples, SL/TP/Close/Time-in-Trade, Reason
   - Entry fallback 3-tier ใน `trade_executor.send_market_order`: `result.price` → `get_open_positions.price_open` → `current market price`

### 🟡 MEDIUM (ML Pipeline Overhaul)

6. **Trade Schema v2** (`analytics/trade_logger.py`): ขยายจาก 19 → **31 columns** เพิ่ม ML features: `Session`, `DayOfWeek`, `HourOfDay`, `Spread@Entry`, `Slippage`, `HTF Bias`, `Volatility Regime`, `ConsecLoss Before`, `DD@Entry %`, `MAE`, `MFE`, `Time-in-Trade (s)`, `Exit Path`. File consolidated เป็น **`logs/ftmo_trades.xlsx`** (ไฟล์เดียว ไม่ split รายเดือนอีก)

7. **MAE/MFE per-tick tracking** (`TradeManager._manage_single_position`): คำนวณ Max Adverse / Favorable Excursion เป็น pips ทุก tick → store บน `ExecutedTrade.mae` / `.mfe` → logger บันทึกตอนปิดเทรด. ใช้ **cached `pip_size` ใน `TrailingState`** (แทนเรียก `get_symbol_info()` ทุก tick — ลด I/O)

8. **Entry context capture** (`TradeExecutor._capture_entry_context`): helper ดึง session bucket (LONDON / LONDON_NY_OVERLAP / NEW_YORK / ASIAN / OFF_HOURS), spread_pts, slippage, dd_pct, htf_bias, volatility_regime ตอนเปิดเทรด → feed เข้า Trade Schema v2

9. **PerformanceAnalyzer restore on restart** (`analytics/performance.py`):
   - `set_initial_balance(initial, peak)` — seed curve ด้วย balance จริงจาก broker + insert peak เพื่อรักษา Max DD history
   - `load_from_excel()` — replay เทรดที่ปิดแล้วจาก `ftmo_trades.xlsx` map ตาม header name (v1/v2 compatible, ข้ามแถว Close Time ว่าง)
   - `main.initialize()` เรียก `set_initial_balance` → `load_from_excel` หลัง `risk_manager.initialize` → equity curve / Sharpe / Sortino ต่อเนื่องข้าม restart

10. **RL Environment v2** (`ml/rl_environment.py`):
    - Action bounds แคบลงเพื่อบังคับ Agent เลือกคุณภาพ: risk `[0.3%, 0.7%]` (เดิม 0.5-1%), confluence `[65, 85]` (เดิม 50-80), atr_sl_mult `[1.2, 2.5]`, rr `[2.0, 4.0]` (เดิม 1.5-3)
    - `_get_stats()` ส่ง `trades_today` เข้า reward calculator
    - `FTMORewardCalculator` เพิ่ม **Over-trading penalty**: ถ้า `trades_today > 5` → penalty `excess × 2.0` (cap -20)
    - `SelfLearningAgent.excel_path` default → `"logs/ftmo_trades.xlsx"`
    - **Old `models/ppo_ftmo_agent.zip` rename เป็น `.v1bak`** — action bounds เปลี่ยน → บังคับ retrain รอบหน้า

### 🔵 LOW (Hardening)

11. **ExecutedTrade dataclass v2** — เพิ่ม 13 fields: `session`, `day_of_week`, `hour_of_day`, `spread_at_entry`, `slippage`, `htf_bias`, `volatility_regime`, `consec_loss_before`, `dd_at_entry_pct`, `mae`, `mfe`, `time_in_trade`, `exit_path`. `to_dict()` expose ครบ
12. **`update_daily_pnl(pnl, symbol=None)`** signature เพิ่ม symbol → `_record_trade_outcome()` track ชนะ/แพ้ per-symbol เพื่อเข้า cooldown logic
13. **`record_external_close()` finalize**: trade.time_in_trade = `(close_time - open_time).total_seconds()`, trade.exit_path = reason

### 🔑 Key Invariants (v2)

- **MT5 deal matching** ต้องใช้ `deal.position` (หรือ `deal.position_id`) **ห้ามใช้** `deal.order` / `deal.ticket` — เป็น id คนละชนิด
- **ftmo_trades.xlsx** ไฟล์เดียวเท่านั้น (consolidated ML dataset) — ห้ามแตกรายเดือน
- **ML ต้องใช้ข้อมูลจริงเท่านั้น** — ห้าม synthetic/mock เมื่อมีไฟล์นี้อยู่
- **RL model v1 → v2 incompatible** — ถ้าเจอ `ppo_ftmo_agent.zip` ที่ train บน bounds เก่า ต้อง rename เป็น `.v1bak` บังคับ retrain
- **State file เป็น source of truth** — ใช้ atomic write, persist cooldown + peak + consecutive_losses ครบ

### 📦 Dependencies (v2)
ยังไม่เพิ่ม pip package ใหม่ — `openpyxl` ที่มีอยู่แล้วรองรับ `load_from_excel`
