# 🤖 FTMO Trading Bot

> ระบบเทรด Forex อัตโนมัติ ที่ออกแบบมาเพื่อผ่าน **FTMO Challenge** (10% profit, 4% daily DD, 8% total DD) โดยใช้ **3 สมอง** ทำงานร่วมกัน: SMC Strategy + ML Filter + RL Agent

---

## 📖 สารบัญ

- [โปรเจคนี้ทำอะไร](#-โปรเจคนี้ทำอะไร)
- [ทำงานยังไง (สำหรับผู้เริ่มต้น)](#-ทำงานยังไง-สำหรับผู้เริ่มต้น)
- [ติดตั้ง](#-ติดตั้ง)
- [เตรียมข้อมูล](#-เตรียมข้อมูล)
- [เทรน Model](#-เทรน-model)
- [เทรดจริง (Live)](#-เทรดจริง-live)
- [เข้าใจองค์ประกอบ](#-เข้าใจองค์ประกอบ)
- [โครงสร้างโปรเจค](#-โครงสร้างโปรเจค)
- [FAQ](#-faq)

---

## 🎯 โปรเจคนี้ทำอะไร

**FTMO Challenge** คืออะไร?
- สำนักลงทุนให้ทุน $100,000 ถ้าคุณพิสูจน์ได้ว่าเทรดเก่ง
- เงื่อนไข: ทำกำไร +10% ภายใน ~45 วัน โดย**ห้าม**ขาดทุนเกิน 4% ในวันเดียว หรือ 8% รวม
- **90%+ ของนักเทรดสอบไม่ผ่าน** เพราะความโลภ/เสียวินัย

**Bot นี้ช่วยอะไร**?
- ใช้กลยุทธ์ **Smart Money Concepts (SMC)** หา setup เทรด
- ใช้ **Machine Learning** กรอง signal ที่มีโอกาสชนะสูง
- ใช้ **Reinforcement Learning** ตัดสินใจว่าเทรดเมื่อไร โดยดูทั้งคุณภาพ signal และสถานะบัญชี (DD, progress)
- ทำงาน **24/5** อัตโนมัติ ไม่มีอารมณ์

---

## 🧠 ทำงานยังไง (สำหรับผู้เริ่มต้น)

Bot แบ่งเป็น **3 สมอง** ทำงานกันเป็นทีม:

### 1. 🎯 SMC Strategy (หาเซ็ทอัพ)

**Smart Money Concepts** คือทฤษฎีที่ว่า "เงินก้อนใหญ่ (ธนาคาร)" ทิ้งร่องรอยในกราฟ ให้เราเดินตามได้

SMC มองหา:
- **Order Blocks (OB)** — โซนที่สถาบันซื้อ/ขาย ปริมาณมาก → ราคามักกลับมาแตะ
- **Fair Value Gap (FVG)** — ช่องว่างราคาที่ต้องถูกเติม
- **Liquidity Sweep** — การเด้งกลับหลังราคาไล่ล่า stop loss
- **Market Structure (BOS/CHoCH)** — การเปลี่ยนแนวโน้ม

เมื่อพบ setup ที่ผ่านเกณฑ์ → ส่ง "signal" ต่อไปให้สมองที่ 2

```
H4 Trend → H1 Structure → M15 Entry (Order Block + Confluence)
```

### 2. 🔬 ML Quality Filter (กลั่นกรอง)

ปัญหาของ SMC เดิม: signal ส่วนใหญ่ **ไม่แม่น** (~32% win rate = เสมือนสุ่ม)

เราจึงเพิ่ม **Gradient Boosting Machine (GBM)** — AI ที่ดูจาก **history ของ signal เก่า** แล้วเรียนรู้ว่าแบบไหนมักจะชนะ

**ผลลัพธ์**: 
- `ml_score > 0.40` → win rate **48%** (จาก 32%) 
- `ml_score > 0.45` → win rate **57%**
- `ml_score > 0.50` → win rate **65%**

ML ให้**คะแนนคุณภาพ** (0-1) ของแต่ละ signal → ส่งให้สมองที่ 3

### 3. 🎓 RL Agent (ตัดสินใจเทรด)

**Reinforcement Learning Agent** คือ AI ที่ฝึกให้ตัดสินใจจาก **สถานการณ์** ไม่ใช่กฎตายตัว

Agent เห็นอะไรบ้าง (obs 24 มิติ):
- **Signal features** (17): confluence, RR, ML score, RSI, ADX, etc.
- **Portfolio state** (7): DD ปัจจุบัน, progress สู่เป้า, trades วันนี้, win rate ล่าสุด

Agent ตัดสินใจ:
- **TAKE** → เปิด order
- **SKIP** → ข้าม signal นี้ไป

**ทำไมต้องใช้ RL แทนกฎง่าย ๆ?**
- ML บอกแค่ "signal นี้น่าจะชนะ" แต่ไม่รู้**เวลา**
- RL เรียนรู้ว่า: ถ้า DD สูง → SKIP ไว้ก่อน, ถ้า progress ดี → เลือกเฉพาะ signal คุณภาพสูง

### 🔄 Flow รวม

```
┌─────────────────────────────────────────────────────────────┐
│  OHLCV Data (M15, H1, H4)                                   │
│                     ↓                                        │
│  ╔═══════════════════════════════════╗                      │
│  ║  1. SMC Strategy                  ║  → TradeSignal       │
│  ║   - Order Blocks, FVG, Sweeps     ║    + features        │
│  ║   - Confluence scoring (0-100)    ║                      │
│  ╚═══════════════════════════════════╝                      │
│                     ↓                                        │
│  ╔═══════════════════════════════════╗                      │
│  ║  2. ML Quality Filter (GBM)       ║  → ml_score ∈ [0,1]  │
│  ║   AUC ~0.58 บน historical data    ║                      │
│  ╚═══════════════════════════════════╝                      │
│                     ↓                                        │
│  ╔═══════════════════════════════════╗                      │
│  ║  3. RL Agent (PPO)                ║  → TAKE / SKIP       │
│  ║   Sees: signal + ML + portfolio   ║                      │
│  ║   Decides based on context        ║                      │
│  ╚═══════════════════════════════════╝                      │
│                     ↓                                        │
│  Trade Executor → MT5 Broker                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📥 ติดตั้ง

### ความต้องการ

- **Python 3.10+** (แนะนำ 3.11 หรือ 3.12)
- **RAM 8GB+** (16GB แนะนำถ้าเทรนพร้อมกัน 8 workers)
- **Disk 2GB+** สำหรับ model + data
- **OS**: macOS / Linux (เทรน), Windows (เทรดจริง with MT5)

### ขั้นตอน

```bash
# 1. Clone หรือ download โปรเจค
cd /path/to/project
cd ftmo_trading_bot

# 2. ติดตั้ง dependencies
pip install -r requirements.txt

# 3. (Windows เท่านั้น) ติดตั้ง MetaTrader5 library
pip install MetaTrader5
```

> 💡 บน macOS/Linux จะใช้ **Mock Mode** อัตโนมัติ (เทรนและทดสอบได้ แต่เทรดจริงไม่ได้)

### ตั้งค่า .env (สำหรับเทรดจริง)

สร้างไฟล์ `.env` ใน `ftmo_trading_bot/`:

```env
# MT5 Credentials
MT5_TERMINAL_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN=12345678
MT5_PASSWORD="your_password"
MT5_SERVER="FTMO-Server"

# Discord Notification (optional)
DISCORD_ENABLE=True
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

---

## 📊 เตรียมข้อมูล

ต้องมีข้อมูลราคา **3 timeframes** (M15, H1, H4) ของทุก symbol

### วิธีที่ 1: ดึงจาก MT5 อัตโนมัติ ⭐

```bash
# ดึงย้อนหลัง 3 ปี (default)
python scripts/fetch_mt5_data.py

# หรือระบุจำนวนปี
python scripts/fetch_mt5_data.py --years 5
```

Script จะสร้างไฟล์ใน `data/ohlcv/`:

```
data/ohlcv/
├── EURUSD_M15.csv   ← ~105,000 bars (3 ปี)
├── EURUSD_H1.csv
├── EURUSD_H4.csv
├── GBPUSD_M15.csv
├── ...
└── (9 symbols × 3 TFs = 27 ไฟล์)
```

> ⚠️ ต้องรันบน **Windows + MT5 login** เท่านั้น

### วิธีที่ 2: ใช้ CSV ของตัวเอง

ใส่ไฟล์ `{SYMBOL}_{TF}.csv` ใน `data/ohlcv/` มีคอลัมน์: `time, open, high, low, close, volume`

**Symbols ที่รองรับ**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY

---

## 🏋️ เทรน Model

การเทรนมี **3 ขั้นตอน** ทำตามลำดับ:

### Step 1: Build Signal Pool

Signal pool = ชุดข้อมูล signals พร้อม outcomes (win/lose) สำหรับสอน AI

```bash
python3 scripts/build_signal_pool.py --pool_size 3000 --workers 8
```

**เวลา**: ~8 นาที (parallel 8 cores)  
**Output**: `data/signal_pool_3000.pkl` (~60 MB, 158k signals)

### Step 2: Train ML Quality Model (GBM)

AI ที่ดูจาก history → ทำนายว่า signal ไหนจะชนะ

```bash
python3 scripts/train_signal_quality.py
```

**เวลา**: ~2-3 นาที  
**Output**: 
- `data/signal_quality_model.pkl` (0.7 MB)
- Pool signals ถูก update `ml_score` อัตโนมัติ

**คาดการณ์ผล**: AUC ~0.58 (มี edge) — จะเห็น threshold analysis ในคอนโซล

### Step 3: Train RL Agent (PPO)

AI ที่ตัดสินใจ TAKE/SKIP จาก signal + ML score + portfolio state

```bash
python3 scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 \
    --timesteps_p2 5000000 \
    --n_envs 8 \
    --pool_size 3000 \
    --outcome_noise 0.02 \
    --ml_threshold 0.33 \
    --risk_per_trade 0.006
```

**เวลา**: ~20-25 นาที
**Output**:
- `models/ppo_signal_filter.zip` — RL model สุดท้าย
- `models/vec_normalize_sf.pkl` — Observation normalization stats

**Parameters explained**:
- `--ml_threshold 0.33` — กรอง signals ที่ ml_score < 0.33 ออก (Hybrid ML+RL)
- `--risk_per_trade 0.006` — ความเสี่ยง 0.6% ต่อเทรด (balance profit/DD)
- `--outcome_noise 0.02` — Gaussian noise 2% กัน overfit pool

### ดู Training Progress

```bash
tensorboard --logdir logs/tb_signal_filter
```

เปิด browser ที่ http://localhost:6006

**ดูอะไร**:
- `train/explained_variance` ควรขึ้นไป **> 0.5** (value function เรียนรู้)
- `ftmo/pass_rate` ควรขึ้นเรื่อย ๆ (passing FTMO challenges)
- `4_Performance/Take_Rate` ควรอยู่ **60-70%** (หลัง ML filter)
- `4_Performance/Win_Rate` ควร **> 43%**

### ประเมินผลหลังเทรน

```bash
python3 scripts/train_signal_filter.py --eval_only \
    --pool_size 3000 \
    --ml_threshold 0.33 \
    --risk_per_trade 0.006
```

**เป้าหมายที่ realistic** (อิงจาก Run 11 benchmark):

| Metric | Good | Excellent | เหตุผล |
|--------|------|-----------|--------|
| Pass Rate | **5-10%** | > 10% | ถึงเป้า 10% FTMO (realistic ceiling ~10-15%) |
| Breach Rate | **< 2%** | 0% | ไม่ชน DD limit 8% |
| Profit avg/ep | **+2%** | +3% | กำไรคงที่ |
| Take Rate (filtered) | **60-70%** | - | หลัง ML filter — selective |
| Win Rate | **> 43%** | > 47% | agent filter ทำงาน |
| DD max | **< 6%** | < 4% | safe margin |

**Run 11 actual**: Pass 8.8%, Profit +2.03%, Win Rate 43.1%, DD max 5.8%, 0% breach ✅

---

## ⚙️ Sync Config ก่อน Deploy Live

### ⚠️ ต้องแก้ `config/settings.py` ให้ตรงกับ train

หลัง train ด้วย `--risk_per_trade 0.006` ต้องไปแก้ค่า **3 ตัว** ใน `FTMOConfig`:

```python
# config/settings.py line 71-73
MIN_RISK_PER_TRADE_PCT: float = 0.004      # 0.4% floor
MAX_RISK_PER_TRADE_PCT: float = 0.007      # 0.7% cap
DEFAULT_RISK_PER_TRADE_PCT: float = 0.006  # 0.6% ← ตรงกับ train
```

### 🧮 Rule of Thumb

```
DEFAULT = --risk_per_trade ตอน train
MIN     = DEFAULT − 0.002
MAX     = DEFAULT + 0.001
```

### 📊 ตาราง mapping

| Train risk | MIN | DEFAULT | MAX |
|-----------|-----|---------|-----|
| 0.003 (0.3%) | 0.002 | 0.003 | 0.004 |
| 0.005 (0.5%) | 0.003 | 0.005 | 0.006 |
| **0.006 (0.6%)** ⭐ | **0.004** | **0.006** | **0.007** |
| 0.007 (0.7%) | 0.005 | 0.007 | 0.008 |

### ❓ ทำไมต้องแก้?

- Bot live ใช้ `DEFAULT_RISK_PER_TRADE_PCT` ในการคำนวณ lot size
- ถ้าไม่ตรง train → **live sim-real mismatch** → performance ต่างจาก backtest
- `MIN/MAX` เป็น guardrail ใน RiskManager → ป้องกัน lot เกินตั้งใจ

---

## 💰 เทรดจริง (Live)

**ข้อกำหนด**:
- Windows + MT5 ติดตั้ง
- บัญชี FTMO login อยู่
- ไฟล์ `.env` ตั้งค่าครบ
- Model เทรนเสร็จแล้ว (ขั้นตอน 3 ข้อบน)

**รัน Bot**:

```bash
cd ftmo_trading_bot
python main.py
```

Bot จะ:
1. ✅ เชื่อมต่อ MT5
2. ✅ โหลด RL Agent + ML Quality Model
3. ✅ สแกน signal ทุก 5 วินาที (ปรับได้ใน `config/settings.py`)
4. ✅ ถาม AI ว่า TAKE หรือ SKIP
5. ✅ ส่งคำสั่งเทรดถ้า TAKE (มี SL, TP ตามกลยุทธ์)
6. ✅ บันทึก log + Discord notification

**ปิด Bot**: กด `Ctrl+C` (bot จะ save state + ปิด position ถ้าจำเป็น)

---

## 🔍 เข้าใจองค์ประกอบ

### 📁 Observation Space (24 dims)

สิ่งที่ RL Agent เห็นในแต่ละ signal:

| Index | Feature | ช่วงค่า | ความหมาย |
|-------|---------|---------|----------|
| **Signal Core (12)** |
| 0 | confluence_norm | [-1, 1] | คะแนน SMC |
| 1 | rr_ratio_norm | [0, 1] | Risk:Reward |
| 2 | direction | ±1 | BUY หรือ SELL |
| 3 | atr_norm | [-2, 2] | Volatility |
| 4 | ob_score_norm | [0, 1] | คะแนน Order Block |
| 5 | bias_alignment | [-1, 1] | ทิศทางตรงกับ H4 ไหม |
| 6 | sl_atr_ratio | [0, 2] | ระยะ SL กี่ ATR |
| 7-11 | rsi, macd, trend, ob_size, adx | varies | Momentum + Trend |
| **Market Regime (4)** |
| 12-15 | stoch, bb_pctb, atr_change, price_roc | varies | ลักษณะตลาด |
| **ML Quality (1)** ⭐ |
| 16 | ml_score_norm | [-1, 1] | GBM ทำนาย P(win) |
| **Portfolio State (7)** |
| 17 | total_dd_norm | [-5, 0] | DD รวม |
| 18 | daily_dd_norm | [-5, 0] | DD วันนี้ |
| 19 | progress_norm | [-1, 2] | % ใกล้เป้า 10% |
| 20 | day_progress | [0, 1] | วันที่ของ challenge |
| 21 | trades_today | [0, 1] | เทรดวันนี้ไปกี่ครั้ง |
| 22 | recent_wr_norm | [-1, 1] | Win rate 10 trades ล่าสุด |
| 23 | consec_losses | [0, 1] | แพ้ติดกันกี่ครั้ง |

### 🎓 2-Phase Curriculum Training

**ทำไมต้องแบ่ง 2 phase?**
- ถ้าสอนอะไรหลายอย่างพร้อมกัน → agent งง → เรียนช้า
- แบ่งให้เรียนทีละเรื่อง → เก่งขึ้นเร็วกว่า

**Phase 1 (Alpha)**: เรียน "อ่านกราฟ + ทำกำไร"
- `enable_risk_penalty=False` — ไม่มี DD penalty
- มี **Oracle SKIP reward** — agent รู้ outcome ล่วงหน้า (training only)
- **Bonus**: เทรดถูก + confluence สูง → +0.25 reward

**Phase 2 (Risk)**: เพิ่ม "จัดการ DD"
- `enable_risk_penalty=True` — DD penalty active
- Exponential penalty ยิ่งใกล้ 10% ยิ่งแรง
- Inherit weights จาก Phase 1 → ไม่ต้องเริ่มใหม่

### 🎯 FTMO Rules

| กฎ | ค่า | Bot ทำอะไร |
|----|------|-----------|
| เป้ากำไร | 10% ($10,000 จาก $100,000) | RL มี target bonus |
| Daily DD | 4% max ($4,000) | Risk guard + DD penalty |
| Total DD | 8% max ($8,000) | Risk guard + DD penalty |
| ระยะเวลา | 45 วัน | Episode length |
| วันเทรดขั้นต่ำ | 4 วัน | Auto-satisfied |

---

## 📂 โครงสร้างโปรเจค

```
ftmo_trading_bot/
├── main.py                          # 🎯 Bot loop หลัก (live trading)
├── requirements.txt                 # dependencies
├── config/
│   └── settings.py                  # ค่าตั้ง: symbols, risk, session
│
├── strategy/                        # 🎯 Brain 1: SMC Strategy
│   ├── smc_strategy.py              # Main strategy engine
│   ├── indicators.py                # RSI, MACD, ADX, etc.
│   ├── order_blocks.py              # OB detection
│   ├── fair_value_gaps.py           # FVG detection
│   ├── liquidity_sweeps.py          # Sweep detection
│   └── market_structure.py          # BOS/CHoCH
│
├── ml/                              # 🔬🎓 Brain 2+3: ML + RL
│   ├── signal_quality.py            # GBM wrapper (Brain 2)
│   ├── rl_agent.py                  # RL inference (Brain 3)
│   ├── signal_filter_env.py         # Gymnasium env สำหรับเทรน
│   └── strategy_backtester.py       # Generate signal pool
│
├── scripts/                         # 🛠️ Training scripts
│   ├── build_signal_pool.py         # Step 1: Pool generation
│   ├── train_signal_quality.py      # Step 2: Train GBM
│   ├── train_signal_filter.py       # Step 3: Train RL
│   └── fetch_mt5_data.py            # Data fetching
│
├── execution/                       # 💱 Trade execution
│   ├── mt5_connector.py             # MT5 API wrapper
│   └── trade_executor.py            # Order management
│
├── core/                            # 🛡️ Risk management
│   └── risk_manager.py              # DD tracking, cooldown
│
├── analytics/                       # 📊 Logging
│   ├── trade_logger.py              # Excel log
│   └── performance.py               # Win rate, Sharpe, etc.
│
├── models/                          # 🧠 Trained AI
│   ├── ppo_signal_filter.zip        # RL agent
│   ├── vec_normalize_sf.pkl         # Obs normalization
│   └── checkpoints_sf/              # Training checkpoints
│
├── data/                            # 💾 Data
│   ├── ohlcv/                       # ราคา (27 CSVs)
│   ├── signal_pool_3000.pkl         # Pre-generated signals
│   └── signal_quality_model.pkl     # GBM ML model
│
└── logs/                            # 📝 Logs
    ├── tb_signal_filter/            # TensorBoard
    └── ftmo_trades.xlsx             # Trade history
```

---

## ❓ FAQ

### Q: ต้องรัน macOS/Windows?

| งาน | macOS/Linux | Windows |
|------|-------------|---------|
| เทรน model | ✅ | ✅ |
| Backtest | ✅ | ✅ |
| เทรดจริงบน MT5 | ❌ (MT5 library ไม่มี) | ✅ |

### Q: ใช้เวลาเทรนนานแค่ไหน?

| ขั้นตอน | เวลา (M-series Mac, 8 workers) |
|---------|-------------------------------|
| Build signal pool | ~8 นาที |
| Train ML (GBM) | ~2-3 นาที |
| Train RL (15M steps) | ~20-25 นาที |
| **รวม** | **~30-35 นาที** |

### Q: Model เก่าใช้ต่อได้ไหม หลัง update?

⚠️ **ไม่ได้** เพราะ observation space เปลี่ยน (23→24 dims หลังเพิ่ม ml_score)  
→ ต้องเทรนใหม่ด้วย `--fresh`

### Q: ถ้า ML model พัง/หายไปล่ะ?

Bot มี **graceful fallback**:
- RL agent ใส่ `ml_score = 0.5` (neutral) → agent ยังทำงานได้
- แต่ performance ลดลงเพราะขาดข้อมูลคุณภาพ signal

### Q: ต้องการ GPU ไหม?

**ไม่จำเป็น** — network เล็ก ([256, 128]) รันบน CPU ได้สบาย  
PyTorch ใช้ multi-core อัตโนมัติ

### Q: เทรดจริงควรใช้ lot เท่าไร?

Bot train ด้วย `--risk_per_trade 0.006` (0.6% ต่อเทรด) — optimal balance  
→ แนะนำ **เริ่มที่ 0.3-0.4%** ก่อน (ต่ำกว่า sim) ดู performance 1-2 สัปดาห์  
→ ถ้าผล live ตรงกับ backtest → ขยับขึ้น 0.5-0.6% ตาม train settings  
→ ไม่ควรเกิน 0.7% (DD risk ใกล้ 8% limit)

**Risk vs Pass Rate trade-off** (จาก benchmark):

| Risk | Pass Rate | DD max | เหมาะกับ |
|------|-----------|--------|----------|
| 0.3% | 4% | 3.9% | Conservative (funded account) |
| 0.5% | 8% | 5.7% | Balanced (live demo) |
| **0.6%** | **8.8%** | **5.8%** | **Optimal (trained)** ⭐ |
| 0.7%+ | est 10% | ~7% | Aggressive (close to 8% limit) |

### Q: Bot รัน 24/7 บน VPS ยังไง?

1. เช่า Windows VPS (2 vCPU, 4GB RAM, SSD)
2. ติดตั้ง MT5 + Python + project
3. ใช้ `run_bot.bat` auto-restart:

```bat
:START
python main.py
echo Bot crashed! Restarting...
timeout /t 10
goto START
```

4. ปิด Windows Update automatic + Sleep mode
5. Remote Desktop ให้กด **Disconnect** (ไม่ใช่ Shut down/Sign out)

### Q: Performance จริงเทียบกับ backtest?

Backtest edge: ~48-57% win rate at ML threshold 0.40  
Live: อาจลดลง ~2-5% จาก slippage + spread จริง  
→ Expected live win rate: **45-55%**

---

## 🤝 Credits & License

- สร้างด้วย ❤️ โดย kittipong.n
- Framework: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [Gymnasium](https://gymnasium.farama.org/), [scikit-learn](https://scikit-learn.org/)
- SMC concepts inspired by Inner Circle Trader (ICT)

---

**⚠️ Disclaimer**: การเทรดมีความเสี่ยง past performance ไม่ garantee future results  
ใช้ bot ด้วยความรับผิดชอบของตัวเอง
