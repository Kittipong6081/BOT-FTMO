# 🎉 MR Bot — Ready for Live (v8.0.5)

> Locked-in 2026-05-07. All eval gates passed. Both audits clean.

## Final Eval Metrics (5000 episodes)

| Metric | Value | Gate | Margin |
|---|---:|---:|---:|
| **Pass Rate** | **59.30%** | ≥ 8.00% | **+51.30 pp** |
| Profitable Rate | 89.10% | ≥ 55.00% | +34.10 pp |
| Breach Rate | 0.00% | ≤ 5.00% | -5.00 pp |
| Total DD max | 5.80% | ≤ 6.00% | -0.20 pp |
| Daily DD max | 3.00% | ≤ 3.50% | -0.50 pp |
| Win Rate | 61.55% | (info) | — |
| Take Rate | 46.35% | (info) | — |
| Profit avg | +$7,229.59 (+7.23%) | (info) | — |

## Best Model

- `models/mr/best/ppo_mr_filter.zip` (1.0 MB)
- `models/mr/best/vec_normalize_mr.pkl` (3.2 KB)
- `models/mr/best/best_meta.json`

## Live Deploy

```bash
python ftmo_trading_bot/main.py
```

Live entry path:

1. `bot_config.mr.strategy_mode = "mean_reversion"` (default since v8.0)
2. `LiveMRScanner` scans 10 symbols on M15 every loop, gates on ADX H1 ≥ 30 (block), BB %B extreme + RSI confirm + reversal-wick
3. `SignalQualityModel` from `data/mr_signal_quality_model.pkl` (auto-loaded)
4. RL agent from `models/mr/best/ppo_mr_filter.zip` (auto-loaded — fallback to legacy if missing)
5. Obs[4]=`bb_extreme`, Obs[10]=`bb_band_width_atr/3`, Obs[26]=`adx_inverse_norm` reinterpreted to match training distribution
6. ML threshold = 0.30 (live ↔ training aligned)
7. Risk per trade = 0.99% (live ↔ training aligned)

## Hyperparams That Worked

| Param | Value |
|---|---:|
| timesteps_p1 | 5,000,000 |
| timesteps_p2 | 2,000,000 |
| n_envs | 8 |
| pool_size | 3,000 |
| ml_threshold | 0.30 |
| risk_per_trade | 0.0099 |
| quick_tp_bonus | 0.50 |
| slow_win_bonus | 0.20 |
| prolonged_loss_penalty | 0.40 |
| base_loss_penalty | 0.10 |
| duration_fine_coef | 0.02 |
| lr_p1 | 3e-4 |
| lr_p2 | 5e-5 |

## Strategy Params (bot_config.mr)

| Param | Value |
|---|---:|
| BB period | 20 |
| BB std | 2.0 |
| BB oversold | 0.30 |
| BB overbought | 0.70 |
| RSI period | 14 |
| RSI oversold | 40.0 |
| RSI overbought | 60.0 |
| ADX trend block | 30.0 |
| SL ATR mult | 1.0 |
| RR ratio | 1.0 (1:1 quick TP) |
| Min reversal-wick ratio | 0.4 |

## Env Guards (FTMOSignalFilterEnv)

| Guard | Value | Buffer to FTMO Limit |
|---|---:|---:|
| DAILY_DD_GUARD | 3.0% | 2.0 pp under 5% FTMO limit |
| TOTAL_DD_GUARD | 5.8% | 4.2 pp under 10% FTMO limit |

## Audit Certification

```bash
.venv/bin/python ftmo_trading_bot/scripts/leakage_audit.py   # exit 0 ✅
.venv/bin/python ftmo_trading_bot/scripts/parity_audit.py    # exit 0 ✅
```

Both passed at lock-in.

## Pre-Deploy Checklist

- [ ] Confirm Windows MT5 terminal installed + login credentials in `.env`
- [ ] Confirm broker server time is FTMO (EET)
- [ ] Confirm OHLCV data fresh (not stale)
- [ ] Run `python main.py` in dry mode first if available
- [ ] Set initial balance via `RiskManager.set_initial_balance($100,000)`
- [ ] Backup `logs/bot_state.json` before challenge starts
- [ ] Monitor first 24h closely

## Known Soft Gaps (not blockers)

- Iter 1 backup `models/mr/best_iter1_pass61pct_2026-05-06/` kept for forensic reference (trained with looser daily guard 4%; not for live)
- SMC files (`strategy/smc_strategy.py`, OB/FVG/Sweep/Structure detectors) preserved as deprecated reference. Safe to leave; can delete in future cleanup
- Auto pipeline rebuilds with timestamp-tagged backups; backups occupy ~5 MB in `models/mr/`

## How to Re-train if Needed

```bash
# Quick retrain (reuse pool + GBM)
.venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py \
    --max_iterations 10 --max_hours 60 \
    --pool_size 3000 --timesteps_p1 5000000 --timesteps_p2 2000000

# Full rebuild (delete pool + GBM first)
rm ftmo_trading_bot/data/mr_signal_pool_3000.pkl
rm ftmo_trading_bot/data/mr_signal_quality_model.pkl
.venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py [...]
```

Pipeline self-corrects via `tune_hyperparams()`. Logs land in:

- `logs/auto_train_pipeline.log` (human readable)
- `logs/auto_train_pipeline.jsonl` (events)
- `logs/auto_train_pipeline_state.json` (state + best metrics)

Status check any time:

```bash
./ftmo_trading_bot/scripts/pipeline_status.sh
```
