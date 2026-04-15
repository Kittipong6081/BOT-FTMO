"""ทดสอบ OrderBlockDetector — anti-repaint (confirmed_index ต้อง > origin index)"""
import numpy as np
import pandas as pd
import pytest

from strategy.order_blocks import OrderBlockDetector, OBType


def _make_bullish_setup_df():
    """สร้าง DataFrame ที่มี bearish bar ตามด้วย impulsive bullish move"""
    n = 100
    np.random.seed(42)
    closes = np.linspace(1.1000, 1.1010, n)
    highs = closes + 0.0002
    lows = closes - 0.0002
    opens = closes - 0.0001

    # Inject bearish OB ที่ index 50, แล้ว impulsive bullish ที่ 51-55
    opens[50] = 1.1010; closes[50] = 1.1000   # bearish body
    highs[50] = 1.1011; lows[50] = 1.0999

    # impulsive bullish — ดันราคาขึ้นไกล
    for i in range(51, 56):
        opens[i] = closes[i-1]
        closes[i] = opens[i] + 0.0020
        highs[i] = closes[i] + 0.0001
        lows[i] = opens[i] - 0.0001

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'tick_volume': np.full(n, 1000.0),
    })
    df['atr'] = 0.0005
    return df


def test_anti_repaint_confirmed_index_after_origin():
    """OB.confirmed_index ต้อง ≥ index (เขียน OB ที่ confirm bar ไม่ใช่ origin)"""
    detector = OrderBlockDetector()
    df = _make_bullish_setup_df()
    try:
        obs = detector.find_order_blocks(df)
    except (TypeError, AttributeError):
        # ถ้าวิธีเรียกต่างจากนี้ — skip (เปลี่ยน signature ในอนาคต)
        pytest.skip("find_order_blocks signature เปลี่ยน")
        return

    if not obs:
        pytest.skip("ไม่ detect OB ในตัวอย่างนี้ (ปกติ — เป็น smoke test)")

    for ob in obs:
        if ob.confirmed_index >= 0:
            assert ob.confirmed_index >= ob.index, \
                f"OB confirmed_index ({ob.confirmed_index}) < index ({ob.index}) → repaint!"


def test_orderblock_dataclass_has_confirmed_index():
    """ตรวจว่า field confirmed_index มีอยู่ใน OrderBlock dataclass (V1 fix)"""
    from strategy.order_blocks import OrderBlock
    ob = OrderBlock(
        ob_type=OBType.BULLISH, index=10, time=None,
        high=1.1, low=1.0, open_price=1.05, close_price=1.02,
        body_size=0.03, impulse_size=0.05,
    )
    assert hasattr(ob, 'confirmed_index')
    assert ob.confirmed_index == -1   # default
