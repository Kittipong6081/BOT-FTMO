"""ทดสอบ RiskManager — DD limits, peak equity, give-back warning"""
from core.risk_manager import RiskManager, BotState


def test_initialize_sets_active(mock_mt5_connector):
    rm = RiskManager(mock_mt5_connector)
    assert rm.initialize() is True
    assert rm.state == BotState.ACTIVE


def _set_equity(mock, value):
    """Helper: sync equity ทั้ง 2 ที่ (account dict + get_equity())"""
    mock.get_account_info.return_value['equity'] = value
    mock.get_equity.return_value = value


def test_peak_daily_equity_tracks_high_water(mock_mt5_connector):
    rm = RiskManager(mock_mt5_connector)
    rm.initialize()

    # Equity ขึ้นเป็น 102k → peak ต้อง update
    _set_equity(mock_mt5_connector, 102_000.0)
    rm.check_risk()
    assert rm._peak_daily_equity >= 102_000.0

    # Equity ตกกลับมา 101k → peak ไม่ลด
    _set_equity(mock_mt5_connector, 101_000.0)
    rm.check_risk()
    assert rm._peak_daily_equity >= 102_000.0


def test_total_dd_breach_triggers_halt(mock_mt5_connector, tmp_path, monkeypatch):
    """equity ตก -8.5% → MAX_DRAWDOWN_HALT (ใช้ tmp state file ป้องกัน leftover จาก test ก่อน)"""
    rm = RiskManager(mock_mt5_connector)
    monkeypatch.setattr(rm, '_state_file', str(tmp_path / "risk_state.json"))
    rm.initialize()
    assert rm._initial_balance == 100_000.0  # sanity
    _set_equity(mock_mt5_connector, 91_500.0)  # -8.5%
    state = rm.check_risk()
    assert state == BotState.MAX_DRAWDOWN_HALT


def test_state_persistence_roundtrip(mock_mt5_connector, tmp_path, monkeypatch):
    """ใช้ _save_state/_load_state internal (ไม่มี public API ใน V1)"""
    state_file = tmp_path / "risk_state.json"

    rm = RiskManager(mock_mt5_connector)
    rm.initialize()
    rm._peak_daily_equity = 105_000.0
    # บังคับ path ของ state file ไปยัง tmp
    monkeypatch.setattr(rm, '_state_file', str(state_file), raising=False)
    rm._save_state()

    rm2 = RiskManager(mock_mt5_connector)
    monkeypatch.setattr(rm2, '_state_file', str(state_file), raising=False)
    loaded = rm2._load_state()
    assert loaded is True
    assert rm2._peak_daily_equity == 105_000.0
