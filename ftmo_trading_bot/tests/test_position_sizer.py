"""ทดสอบ PositionSizer — spread buffer, lot calculation"""
from core.position_sizer import PositionSizer


def test_sl_distance_respects_spread_buffer(mock_mt5_connector):
    """ATR เล็กมาก → SL ต้องถูกขยายไปแตะ min_sl_distance"""
    mock_mt5_connector.get_symbol_info.return_value.update({
        "spread": 30,            # 30 points spread
        "trade_stops_level": 0,
    })
    sizer = PositionSizer(mock_mt5_connector)

    # ATR เล็ก → sl_distance เริ่มต้นจะต่ำกว่า min
    result = sizer.calculate_sl_tp_prices(
        symbol="EURUSD",
        order_type="BUY",
        entry_price=1.10000,
        atr_value=0.00001,        # 1 point only
        sl_multiplier=1.5,
        tp_multiplier=3.0,
    )
    assert result is not None
    # min_sl = max(0, 1.5×30×1e-5, 3×1e-5) = 4.5e-4
    assert result["sl_distance"] >= 1.5 * 30 * 1e-5 - 1e-9


def test_sl_below_entry_for_buy(mock_mt5_connector):
    sizer = PositionSizer(mock_mt5_connector)
    result = sizer.calculate_sl_tp_prices(
        symbol="EURUSD", order_type="BUY",
        entry_price=1.10000, atr_value=0.0010,
    )
    assert result["sl"] < 1.10000
    assert result["tp"] > 1.10000


def test_sl_above_entry_for_sell(mock_mt5_connector):
    sizer = PositionSizer(mock_mt5_connector)
    result = sizer.calculate_sl_tp_prices(
        symbol="EURUSD", order_type="SELL",
        entry_price=1.10000, atr_value=0.0010,
    )
    assert result["sl"] > 1.10000
    assert result["tp"] < 1.10000


def test_rr_ratio_at_least_minimum(mock_mt5_connector):
    sizer = PositionSizer(mock_mt5_connector)
    result = sizer.calculate_sl_tp_prices(
        symbol="EURUSD", order_type="BUY",
        entry_price=1.10000, atr_value=0.0010,
        sl_multiplier=2.0, tp_multiplier=2.0,    # RR = 1:1
    )
    # ต้องถูกบังคับขึ้นเป็น MIN_RISK_REWARD_RATIO (default ≥ 1.5)
    assert result["rr_ratio"] >= 1.5
