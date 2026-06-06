"""
Unit tests for the SMC engine foundation (swing / structure / fvg).

Runnable two ways:
    pytest ftmo_trading_bot/tests/test_smc_engines.py
    python ftmo_trading_bot/tests/test_smc_engines.py     # plain asserts, no pytest
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.smc import (  # noqa: E402
    BiasEngine,
    Direction,
    FVGEngine,
    LiquidityEngine,
    OrderBlockEngine,
    StructureEngine,
    StructureEvent,
    StructureKind,
    StructureState,
    SwingEngine,
    SwingLabel,
    SwingType,
)


def _candles(rows):
    """rows = list of (open, high, low, close)."""
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=len(rows), freq="15min"),
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [100] * len(rows),
    })


def _mk_df(pivots, e=0.0002):
    """Build OHLCV by linearly interpolating close between (index, price) pivots.
    high = close + e, low = close - e → each pivot bar is a clean fractal."""
    last = pivots[-1][0]
    idx = [p[0] for p in pivots]
    closes = []
    for i in range(last + 1):
        # find surrounding pivots
        for k in range(len(pivots) - 1):
            if idx[k] <= i <= idx[k + 1]:
                (i0, p0), (i1, p1) = pivots[k], pivots[k + 1]
                frac = (i - i0) / (i1 - i0) if i1 != i0 else 0.0
                closes.append(p0 + (p1 - p0) * frac)
                break
    opens = [closes[0]] + closes[:-1]
    rows = {
        "time": pd.date_range("2024-01-01", periods=len(closes), freq="15min"),
        "open": opens,
        "high": [c + e for c in closes],
        "low": [c - e for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
    }
    return pd.DataFrame(rows)


# lead-in high then: low, high, HL, HH, LL (reversal), LH
# 1.0020 ↓ 1.0000(L) ↑ 1.0100(H) ↓ 1.0050(HL) ↑ 1.0200(HH) ↓ 1.0030(LL) ↑ 1.0080(LH)
_PIVOTS = [(0, 1.0020), (2, 1.0000), (5, 1.0100), (8, 1.0050),
           (11, 1.0200), (14, 1.0030), (17, 1.0080), (20, 1.0000)]


def test_swing_detection_and_labels():
    df = _mk_df(_PIVOTS)
    swings = SwingEngine(fractal_n=1, external_n=2, atr_mult=0.0).analyze(df)
    # expect the 6 pivots, alternating
    assert len(swings) >= 5, f"expected >=5 swings, got {len(swings)}"
    types = [s.swing_type for s in swings]
    for a, b in zip(types, types[1:]):
        assert a is not b, "swings must alternate high/low"
    labels = {s.label for s in swings if s.label}
    assert SwingLabel.HH in labels and SwingLabel.HL in labels, labels
    assert SwingLabel.LL in labels and SwingLabel.LH in labels, labels
    # no-lookahead: confirm_index strictly after the swing bar
    for s in swings:
        assert s.confirm_index == s.index + 1


def test_structure_bos_then_choch():
    df = _mk_df(_PIVOTS)
    swings = SwingEngine(fractal_n=1, external_n=2, atr_mult=0.0).analyze(df)
    state = StructureEngine().analyze(df, swings)
    kinds = [(e.kind, e.direction) for e in state.events]
    assert (StructureKind.BOS, Direction.BULLISH) in kinds, kinds
    assert (StructureKind.CHOCH, Direction.BEARISH) in kinds, kinds
    # the bullish BOS must occur before the bearish CHOCH
    bos_i = next(e.index for e in state.events
                 if e.kind is StructureKind.BOS and e.direction is Direction.BULLISH)
    choch_i = next(e.index for e in state.events
                   if e.kind is StructureKind.CHOCH and e.direction is Direction.BEARISH)
    assert bos_i < choch_i
    assert state.bias is Direction.BEARISH


def test_structure_no_lookahead():
    """A break can only reference a swing already confirmed at the break bar."""
    df = _mk_df(_PIVOTS)
    swings = SwingEngine(fractal_n=1, external_n=2, atr_mult=0.0).analyze(df)
    state = StructureEngine().analyze(df, swings)
    for ev in state.events:
        ref = next(s for s in swings if s.index == ev.broken_swing_index)
        assert ref.confirm_index <= ev.index, "break referenced an unconfirmed swing (lookahead!)"


def test_fvg_bullish_partial_then_filled():
    # A(high=1.0000) , B(displacement up) , C(low=1.0010) → bullish gap [1.0000,1.0010]
    base = [
        # open, high, low, close
        (0.9990, 1.0000, 0.9985, 0.9995),   # A
        (0.9998, 1.0030, 0.9996, 1.0025),   # B displacement
        (1.0020, 1.0040, 1.0010, 1.0035),   # C  → low 1.0010 > A.high 1.0000
        (1.0035, 1.0036, 1.0005, 1.0008),   # taps to 1.0005 (partial ~0.5), close 1.0008 (not through)
    ]
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=len(base), freq="15min"),
        "open": [b[0] for b in base], "high": [b[1] for b in base],
        "low": [b[2] for b in base], "close": [b[3] for b in base],
        "volume": [100] * len(base),
    })
    fvgs = FVGEngine().detect(df)
    bull = [f for f in fvgs if f.direction is Direction.BULLISH]
    assert len(bull) == 1, fvgs
    f = bull[0]
    assert abs(f.bottom - 1.0000) < 1e-9 and abs(f.top - 1.0010) < 1e-9
    assert 0.0 < f.mitigation < 1.0 and not f.filled, (f.mitigation, f.filled)

    # extend: a candle closes below the gap bottom → filled
    df2 = pd.concat([df, pd.DataFrame({
        "time": [pd.Timestamp("2024-01-01 01:00")], "open": [1.0008],
        "high": [1.0009], "low": [0.9990], "close": [0.9992], "volume": [100],
    })], ignore_index=True)
    f2 = [x for x in FVGEngine().detect(df2) if x.direction is Direction.BULLISH][0]
    assert f2.filled and f2.mitigation == 1.0


def test_liquidity_sweep_bearish():
    # swing high at idx1 (1.0050); idx4 wicks to 1.0060 but closes 1.0030 inside → bearish sweep
    df = _candles([
        (1.0000, 1.0005, 0.9995, 1.0002),
        (1.0002, 1.0050, 1.0000, 1.0045),   # swing high here
        (1.0045, 1.0046, 1.0010, 1.0015),
        (1.0015, 1.0020, 1.0008, 1.0012),
        (1.0012, 1.0060, 1.0010, 1.0030),   # wick > 1.0050, close < 1.0050 → sweep
    ])
    swings = SwingEngine(fractal_n=1, external_n=2, atr_mult=0.0).analyze(df)
    sweeps = LiquidityEngine().sweeps(df, swings)
    bear = [s for s in sweeps if s.direction is Direction.BEARISH]
    assert len(bear) >= 1, sweeps
    assert abs(bear[0].level - 1.0050) < 1e-9 and bear[0].index == 4


def test_dealing_range_premium_discount():
    df = _mk_df(_PIVOTS)
    swings = SwingEngine(fractal_n=1, external_n=2, atr_mult=0.0).analyze(df)
    dr = LiquidityEngine().dealing_range(swings)
    assert dr is not None
    assert dr.low < dr.equilibrium < dr.high
    assert dr.in_discount(dr.low + 0.1 * dr.size)
    assert dr.in_premium(dr.high - 0.1 * dr.size)


def test_order_block_bullish():
    # idx1 = bearish OB candle, idx2 = bullish displacement causing a (manual) BOS up
    df = _candles([
        (1.0000, 1.0010, 0.9995, 1.0008),   # bullish
        (1.0008, 1.0009, 0.9980, 0.9985),   # bearish ← the OB
        (0.9985, 1.0060, 0.9984, 1.0055),   # displacement up → BOS
    ])
    state = StructureState(events=[
        StructureEvent(2, StructureKind.BOS, Direction.BULLISH, 1.0010, 0)
    ])
    obs = OrderBlockEngine().detect(df, state)
    assert len(obs) == 1, obs
    ob = obs[0]
    assert ob.direction is Direction.BULLISH and ob.index == 1
    assert abs(ob.bottom - 0.9980) < 1e-9 and abs(ob.top - 1.0009) < 1e-9
    assert ob.grade >= 0.75 and not ob.mitigated


def test_bias_engine_htf():
    df = _mk_df(_PIVOTS)
    mb = BiasEngine(SwingEngine(fractal_n=1, external_n=2, atr_mult=0.0)).analyze(df)
    assert mb.direction is Direction.BEARISH, mb.direction
    assert mb.dealing_range is not None
    assert mb.dealing_range.low < mb.dealing_range.high


def test_entry_engine_bullish_setup():
    from strategy.smc.entry import EntryEngine, SMCEntryConfig
    from strategy.smc.types import SMCSignalType

    # HTF (D1) clearly bullish; dealing range ~[1.0320,1.0450], LTF price in discount
    htf = _mk_df([(0, 1.0250), (3, 1.0380), (6, 1.0300), (9, 1.0420),
                  (12, 1.0320), (15, 1.0450), (18, 1.0340)], e=0.0003)

    # LTF (M15): swing low L1=1.0300 → sweep (wick 1.0288, close back up) → BOS up → OB → pullback into OB
    ltf = _candles([
        (1.0360, 1.0362, 1.0355, 1.0358),  # 0
        (1.0358, 1.0365, 1.0356, 1.0362),  # 1
        (1.0362, 1.0370, 1.0360, 1.0368),  # 2 swing high H0
        (1.0368, 1.0369, 1.0345, 1.0348),  # 3
        (1.0348, 1.0350, 1.0330, 1.0333),  # 4
        (1.0333, 1.0335, 1.0300, 1.0305),  # 5 swing low L1=1.0300
        (1.0305, 1.0325, 1.0303, 1.0322),  # 6 up (confirms L1)
        (1.0322, 1.0350, 1.0320, 1.0345),  # 7 swing high H1=1.0350
        (1.0345, 1.0335, 1.0315, 1.0318),  # 8 down (confirms H1)
        (1.0318, 1.0320, 1.0298, 1.0302),  # 9 bearish (future OB candle)
        (1.0302, 1.0306, 1.0288, 1.0306),  # 10 SWEEP L1 (low<1.0300, close>1.0300)
        (1.0306, 1.0330, 1.0304, 1.0328),  # 11 up displacement
        (1.0328, 1.0355, 1.0326, 1.0352),  # 12 close>H1 1.0350 → BOS up
        (1.0352, 1.0353, 1.0330, 1.0332),  # 13 pullback
        (1.0332, 1.0334, 1.0315, 1.0316),  # 14 into OB zone (last bar)
    ])

    eng = EntryEngine(
        SMCEntryConfig(min_rr=1.5, sweep_window=20, choch_window=20,
                       zone_tol_atr=1.5, sl_buffer_atr=0.1,
                       min_ltf_bars=12, min_htf_bars=12),
        swing=SwingEngine(fractal_n=1, external_n=2, atr_mult=0.0),
    )
    res = eng.evaluate("EURUSD", htf, ltf, pip_size=0.0001)
    assert res.signal is not None, f"no signal — stage={res.stage} reason={res.reason}"
    sig = res.signal
    assert sig.signal_type is SMCSignalType.BUY
    assert sig.sl_price < sig.entry_price < sig.tp_price
    assert sig.rr_ratio >= 1.5
    assert sig.swept_liquidity == 1.0300 or abs(sig.swept_liquidity - 1.0300) < 1e-6
    assert "sweep" in sig.reasons


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎯 {len(fns)}/{len(fns)} SMC engine tests passed")


if __name__ == "__main__":
    _run_all()
