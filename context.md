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
