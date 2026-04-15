"""ทดสอบ RiskManager — DD limits, peak equity, give-back warning"""
from core.risk_manager import RiskManager, BotState


def test_initialize_sets_active(mock_mt5_connector):
    rm = RiskManager(mock_mt5_connector)
    assert rm.initialize() is True
    assert rm.state == BotState.ACTIVE


def test_peak_daily_equity_tracks_high_water(mock_mt5_connector):
    rm = RiskManager(mock_mt5_connector)
    rm.initialize()

    # Equity ขึ้นเป็น 102k → peak ต้อง update
    mock_mt5_connector.get_account_info.return_value['equity'] = 102_000.0
    rm.check_risk()
    assert rm._peak_daily_equity >= 102_000.0

    # Equity ตกกลับมา 101k → peak ไม่ลด
    mock_mt5_connector.get_account_info.return_value['equity'] = 101_000.0
    rm.check_risk()
    assert rm._peak_daily_equity >= 102_000.0


def test_total_dd_breach_triggers_halt(mock_mt5_connector):
    rm = RiskManager(mock_mt5_connector)
    rm.initialize()
    mock_mt5_connector.get_account_info.return_value['equity'] = 91_500.0  # -8.5%
    state = rm.check_risk()
    assert state == BotState.MAX_DRAWDOWN_HALT


def test_state_persistence_roundtrip(mock_mt5_connector, tmp_path):
    rm = RiskManager(mock_mt5_connector)
    rm.initialize()
    rm._peak_daily_equity = 105_000.0
    state_file = tmp_path / "risk_state.json"
    rm.save_state(str(state_file))

    rm2 = RiskManager(mock_mt5_connector)
    rm2.load_state(str(state_file))
    assert rm2._peak_daily_equity == 105_000.0
