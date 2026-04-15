"""ทดสอบ FTMORewardCalculator — L1/L2/L3/L4/L5 + DD penalty"""
from ml.reward_function import FTMORewardCalculator


def test_breach_returns_minus_100(base_stats):
    calc = FTMORewardCalculator()
    s = {**base_stats, 'daily_dd_pct': 0.04}
    assert calc.calculate_reward(s, base_stats) == -100.0


def test_total_breach_returns_minus_100(base_stats):
    calc = FTMORewardCalculator()
    s = {**base_stats, 'total_dd_pct': 0.08}
    assert calc.calculate_reward(s, base_stats) == -100.0


def test_dd_penalty_grows_exponentially(base_stats):
    calc = FTMORewardCalculator()
    s_low = {**base_stats, 'daily_dd_pct': 0.022}    # ratio 0.55
    s_high = {**base_stats, 'daily_dd_pct': 0.036}   # ratio 0.90
    r_low = calc.calculate_reward(s_low, base_stats)
    r_high = calc.calculate_reward(s_high, base_stats)
    assert r_high < r_low < 0


def test_profit_reward_capped_at_8(base_stats):
    """L3: ป้องกัน reward-hack จาก micro-wins"""
    calc = FTMORewardCalculator()
    s = {**base_stats, 'target_progress_pct': 100.0, 'trading_days': 15}
    prev = {**base_stats, 'target_progress_pct': 0.0}
    r = calc.calculate_reward(s, prev)
    # delta = 100% × 0.5 = 50 → cap 8 + ถึงเป้า bonus 30 = ~38 (× consistency)
    assert r > 0
    # ส่วน profit อย่างเดียวต้อง ≤ 8 + 30 (bonus)
    assert r < 50  # ถ้าไม่ cap จะได้ ~80


def test_activity_floor_penalizes_idle(base_stats):
    """L1: ปิด policy 'do nothing'"""
    calc = FTMORewardCalculator()
    s = {**base_stats, 'is_final_step': True, 'trading_days': 0}
    r = calc.calculate_reward(s, base_stats)
    assert r <= -30.0  # ขาด 10 trading days = -30


def test_consecutive_loss_penalty(base_stats):
    calc = FTMORewardCalculator()
    s3 = {**base_stats, 'consecutive_losses': 3}
    s5 = {**base_stats, 'consecutive_losses': 5}
    r3 = calc.calculate_reward(s3, base_stats)
    r5 = calc.calculate_reward(s5, base_stats)
    assert r5 < r3 < 0


def test_intraday_swing_penalty_triggers(base_stats):
    """L2: hold loser แล้วเด้งกลับ = lucky → ลงโทษ"""
    calc = FTMORewardCalculator()
    s = {
        **base_stats,
        'intraday_excursion_pct': 0.025,    # เคยติดลบ 2.5% ระหว่างวัน
        'day_end_loss_pct': 0.005,          # แต่ปิดวันแค่ -0.5%
    }
    r = calc.calculate_reward(s, base_stats)
    assert r < 0


def test_intraday_swing_no_penalty_when_small_gap(base_stats):
    """ถ้า excursion ใกล้ day_end → ไม่ใช่ lucky bounce, ไม่ลงโทษ"""
    calc = FTMORewardCalculator()
    s = {
        **base_stats,
        'intraday_excursion_pct': 0.008,
        'day_end_loss_pct': 0.007,
    }
    r = calc.calculate_reward(s, base_stats)
    assert r >= -0.5  # อนุญาต DD penalty นิดหน่อย


def test_consistency_multiplier_reduces_positive(base_stats):
    """L4: agent เทรด 1 วัน vs 15 วัน — ตัวเดียวกันแต่ได้ reward น้อยกว่า"""
    calc = FTMORewardCalculator()
    s_few = {**base_stats, 'sortino_ratio': 2.0, 'trading_days': 1}
    s_many = {**base_stats, 'sortino_ratio': 2.0, 'trading_days': 15}
    r_few = calc.calculate_reward(s_few, base_stats)
    r_many = calc.calculate_reward(s_many, base_stats)
    assert 0 < r_few < r_many


def test_overtrading_penalty(base_stats):
    calc = FTMORewardCalculator()
    s = {**base_stats, 'trades_today': 10}   # MAX = 5
    r = calc.calculate_reward(s, base_stats)
    assert r < 0
