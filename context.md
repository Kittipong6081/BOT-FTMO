# SYSTEM_STATE_CONTEXT (End of Phase 5)

## 📌 Project Overview
**Name:** FTMO Algorithmic Forex Trading Bot
**Goal:** A fully autonomous trading system tailored specifically to pass the FTMO challenge and manage a funded account securely.
**Current Status:** Phase 5 (RL Self-Learning Automation) Completed. All 5 phases finished.

## 🏗️ Architecture & Modules

### 1. Phase 1: Core Systems (`core/`)
- **`MT5Connector`**: Deals with MetaTrader 5 API connection, OHLCV data retrieval, position fetching, and emergency hard-closing. Currently supports a mock mode if `MetaTrader5` is unavailable.
- **`RiskManager`**: Enforces strict FTMO rules (4% daily drawdown, 8% max total drawdown). Persists state locally via JSON to survive restarts.
- **`PositionSizer`**: Calculates trade lot size based on account equity, risk per trade, and ATR multiplier.

### 2. Phase 2: strategy (`strategy/`)
- **`TechnicalIndicators`**: Computes ATR, EMA (fast/med/slow), and RSI.
- **`MarketStructure`**: Evaluates H1 market bias (BOS, CHoCH) holding swing highs and lows.
- **`OrderBlockDetector`**: Detects bearish/bullish Order Blocks (OB) on M15 timeframe.
- **`SMCStrategy`**: The backbone. Generates trade signals by scoring setups based on HTF structure, MTF bias, RSI, volatility, and executing strict RR filtering. Will only trigger if the Confluence Score is above `MIN_CONFLUENCE_SCORE` (auto-tuned by AI).

### 3. Phase 3: Trade Execution (`execution/`)
- **`TradeExecutor`**: Verifies signals across 7 gates (Session, Hard Stop, Data, Spread, Risk Limit, Duplicate, Mocking). Executes orders into MT5 and logs opening/closing to `analytics/trade_logger`.
- **`TradeManager`**: Actively manages positions via ATR-Trailing Stop, Break-Even rules, and Session-end liquidation limits.

### 4. Phase 4: Analytics (`analytics/`)
- **`TradeLogger`**: Converts active and closed trade dictionaries into formatted lines inside an Excel file (`logs/trading_log.xlsx`). Tracks open vs closed states using colored cell formatting.
- **`PerformanceAnalyzer`**: Consumes trade dictionaries to output standard stats (Win rate, Expectancy), advanced stats (Profit factor), and risk stats (Sharpe, Sortino, Drawdowns vs Target).

### 5. Phase 5: Reinforcement Learning (`ml/`) [NEWLY ADDED]
- **`FTMORewardCalculator`**: A specialized algorithmic critic penalizing drawdowns exponentially to prevent FTMO breaches whilst rewarding smooth growth. 
- **`FTMOOptimizationEnv`**: A Gymnasium-based continuous observation state space interacting with Phase 4's mock-logs (Excel stats). Simulates the effect of 4 strategy parameters (Risk, Confluence, SL-Multiplier, RR-Ratio).
- **`SelfLearningAgent`**: A PPO RL implementation via Stable-Baselines3. Predicts safety bounds for execution parameters daily. Trained periodically (`python main.py --train-rl`). 
- **`main.py (Integration)`**: Phase 5 enables Auto-Tuning at the initial system boot. `FTMOTradingBot` connects to the `SelfLearningAgent` object and fetches parameters dynamically bounding the `bot_config`.

## ⚙️ Active Variables (`config/settings.py`)
- **Core Parameters**: Bound to `bot_config` Singleton. Holds dynamic variables overridden by Phase 5 AI Agent.
  - `ftmo.DEFAULT_RISK_PER_TRADE_PCT` (Range: 0.5% - 1.0%)
  - `ftmo.PREFERRED_RISK_REWARD_RATIO` (Range: 1.5 - 3.0)
  - `indicators.atr_sl_multiplier` (Range: 1.0 - 2.5)
- **Timeframes**: H4 (HTF Trend), H1 (Structure), M15 (Primary Entry).
- **Symbols**: EURUSD, GBPUSD, USDJPY, AUDUSD.
- **Test Pipeline**: `python main.py --test-all` covers unit tests 1 through 22, verifying Core, Strategy, Execution, Analytics, and RL Environments sequentially.
