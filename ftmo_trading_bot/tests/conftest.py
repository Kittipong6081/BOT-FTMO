"""
Pytest fixtures + sys.path setup สำหรับ FTMO bot tests
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

# เพิ่ม project root เพื่อให้ import ftmo_trading_bot.* ได้
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def mock_mt5_connector():
    """Mock MT5Connector ที่ return ค่าสมจริง (EURUSD-like)"""
    conn = MagicMock()
    conn.is_connected = True
    conn.get_account_info.return_value = {
        "balance": 100_000.0,
        "equity": 100_000.0,
        "margin": 0.0,
        "free_margin": 100_000.0,
        "currency": "USD",
        "leverage": 100,
    }
    conn.get_symbol_info.return_value = {
        "digits": 5,
        "point": 1e-5,
        "spread": 10,             # 10 points = 1 pip
        "trade_stops_level": 0,
        "trade_contract_size": 100_000.0,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "currency_profit": "USD",
        "currency_base": "EUR",
    }
    conn.get_current_price.return_value = {"bid": 1.10000, "ask": 1.10010}
    conn.get_equity.return_value = 100_000.0
    conn.get_balance.return_value = 100_000.0
    conn.get_positions_count.return_value = 0
    conn.get_positions.return_value = []
    conn.close_all_positions.return_value = (0, 0)
    return conn


@pytest.fixture
def base_stats():
    """ค่าเริ่มต้นของ stats สำหรับ reward calculator"""
    return {
        'daily_dd_pct': 0.0,
        'total_dd_pct': 0.0,
        'sortino_ratio': 0.0,
        'balance': 100_000.0,
        'target_progress_pct': 0.0,
        'trades_today': 0,
        'trading_days': 0,
        'is_final_step': False,
        'consecutive_losses': 0,
        'intraday_excursion_pct': 0.0,
        'day_end_loss_pct': 0.0,
    }
