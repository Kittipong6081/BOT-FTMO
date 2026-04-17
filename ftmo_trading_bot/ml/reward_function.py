"""
===============================================================================
FTMO Trading Bot — Reward Function (สมการการให้คะแนนสำหรับ RL Agent)
===============================================================================
มุ่งเน้นเพื่อเอาตัวรอดจากการสอบ FTMO ประเมินผลงานจาก Logger 
เพื่อคืนค่า Reward สำหรับฝึกสอน PPO Model

กติกาและน้ำหนักการให้คะแนน:
1. การรอดชีวิตและความเสี่ยง (Drawdown) - น้ำหนักสูงสุด (Penalty แบบทวีคูณ)
2. ความเสถียร (Sortino/Sharpe) - ดันให้ Equity เรียบเนียน
3. ผลกำไร (Profit) - โบนัสสำหรับการเดินเข้าใกล้เป้า 10%
===============================================================================
"""

from typing import Dict, Any

class FTMORewardCalculator:
    """
    คลาสคำนวณ Reward สำรับ AI (RL Agent)
    โดยมุ่งเน้นที่การจำกัดความเสี่ยง (Drawdown) ให้อยู่ภายใต้กฎ FTMO เป็นหลัก
    """

    def __init__(self, daily_limit_pct: float = 0.04, total_limit_pct: float = 0.08, target_pct: float = 0.10):
        """
        กำหนดค่า Limit ที่จะใช้คำนวณ
        
        Args:
            daily_limit_pct: ขีดจำกัด Drawdown รายวัน (ค่าเริ่มต้นคือ 4%)
            total_limit_pct: ขีดจำกัด Drawdown ตลอดกาล (ค่าเริ่มต้นคือ 8%)
            target_pct: เป้าหมายกำไร (10%)
        """
        self.daily_limit = daily_limit_pct
        self.total_limit = total_limit_pct
        self.target = target_pct

    def calculate_reward(self, current_stats: Dict[str, Any], previous_stats: Dict[str, Any]) -> float:
        """
        รับสถิติก่อนและหลังการ Action ของ Agent มาเพื่อคำนวณคะแนน (+ / -)

        V3 — ปรับปรุงสำหรับข้อมูลจริง (500k+ timestamps):
        - ลด exponential DD penalty → ใช้ quadratic แทน (เรียนรู้ได้ smooth กว่า)
        - เพิ่ม survival bonus (รอดอีกวัน = ดี)
        - Cap reward per step ทั้งบวกและลบ (ป้องกัน value function ระเบิด)
        - ลด urgency penalty สำหรับวันที่ไม่มี signal (agent ควบคุมไม่ได้)
        """

        reward = 0.0

        daily_dd = current_stats.get('daily_dd_pct', 0.0)
        total_dd = current_stats.get('total_dd_pct', 0.0)
        sortino = current_stats.get('sortino_ratio', 0.0)
        progress = current_stats.get('target_progress_pct', 0.0)
        prev_progress = previous_stats.get('target_progress_pct', 0.0)
        trades_today = current_stats.get('trades_today', 0)

        trading_days = current_stats.get('trading_days', 0)
        is_final_step = current_stats.get('is_final_step', False)
        consecutive_losses = current_stats.get('consecutive_losses', 0)
        intraday_excursion = current_stats.get('intraday_excursion_pct', 0.0)
        day_end_loss = current_stats.get('day_end_loss_pct', 0.0)

        # ==========================================
        # 0. Survival Bonus
        # ==========================================
        reward += 0.1

        # ==========================================
        # 1. Drawdown Penalty — Quadratic + R1/R3 adjustments
        # ==========================================

        if daily_dd >= self.daily_limit or total_dd >= self.total_limit:
            return -15.0

        daily_danger_ratio = daily_dd / self.daily_limit  # 0 → 1.0
        if daily_danger_ratio > 0.15:
            reward -= 4.0 * (daily_danger_ratio ** 2)

        total_danger_ratio = total_dd / self.total_limit
        if total_danger_ratio > 0.15:
            reward -= 5.0 * (total_danger_ratio ** 2)

        # Safe zone bonus — reward การอยู่ในโซนปลอดภัย
        if daily_danger_ratio < 0.3 and total_danger_ratio < 0.3:
            reward += 0.3

        # ==========================================
        # 2. Stability Reward (Sortino) — เฉพาะหลัง 10 วัน
        # ==========================================
        if trading_days >= 10 and sortino > 0:
            reward += min(sortino * 0.15, 1.5)
        elif trading_days >= 10 and sortino < 0:
            reward += max(sortino * 0.1, -1.0)

        # ==========================================
        # 3. Progress Reward (กำไร)
        # ==========================================
        delta_progress = progress - prev_progress

        if delta_progress > 0:
            reward += min((delta_progress * 100) * 0.3, 4.0)
        elif delta_progress < 0:
            reward -= min(abs(delta_progress * 100) * 0.15, 3.0)

        # T1: โบนัสถึงเป้า 10% — ลดจาก +50 เป็น +15 + time-decay
        # ถึงช้า (วันหลังๆ) = bonus เต็ม, ถึงเร็ว (วันแรกๆ) = bonus น้อย
        # ป้องกัน agent เรียนรู้ "เสี่ยงเยอะ หวัง jackpot เร็ว"
        if progress >= 100.0:
            current_step = current_stats.get('current_step', 0)
            max_steps = current_stats.get('max_steps', 45)
            time_ratio = min(current_step / max(max_steps, 1), 1.0)
            reward += 15.0 * (0.5 + 0.5 * time_ratio)

        # ==========================================
        # 3.5. Trading Incentive
        # ==========================================
        if trades_today > 0 and trades_today <= 5:
            reward += 0.5

        # ==========================================
        # 4. Over-trading Penalty
        # ==========================================
        MAX_DAILY_TRADES = 5
        if trades_today > MAX_DAILY_TRADES:
            excess = trades_today - MAX_DAILY_TRADES
            reward -= min(excess * 1.0, 5.0)

        # ==========================================
        # 5. Activity Floor (L1) — จบ challenge โดยเทรดน้อย
        # ==========================================
        if is_final_step and trading_days < 10:
            missing = 10 - trading_days
            reward -= min(missing * 2.0, 15.0)

        # ==========================================
        # 5.5. Urgency — เฉพาะวันที่มีเทรด (ไม่ลงโทษวันไม่มี signal)
        # ==========================================
        current_step = current_stats.get('current_step', 0)
        max_steps = current_stats.get('max_steps', 45)
        if not is_final_step and current_step > 0:
            expected_progress = (current_step / max_steps) * 100.0
            if progress >= expected_progress * 0.5:
                reward += 1.0  # on-pace bonus
            elif trades_today > 0:
                # ลงโทษเฉพาะวันที่เทรดแล้วยังตามหลัง (ไม่โทษวันไม่มี signal)
                reward -= 0.5

        # ==========================================
        # 5.6. End-of-Challenge Penalty
        # ==========================================
        if is_final_step and progress < 100.0:
            if progress < 30.0:
                reward -= 5.0
            elif progress < 70.0:
                reward -= 3.0
            else:
                reward -= 1.0

        # ==========================================
        # 6. Consistency Multiplier
        # ==========================================
        if reward > 0 and trading_days >= 0:
            consistency = min(max(trading_days, 0), 30) / 30.0
            reward *= (0.7 + 0.3 * consistency)

        # ==========================================
        # 7. Consecutive-Loss Penalty (ลดลง)
        # ==========================================
        if consecutive_losses >= 3:
            reward -= min((consecutive_losses - 2) * 1.5, 6.0)

        # ==========================================
        # 7.5. Win Rate Quality Bonus (T12)
        # ==========================================
        recent_win_rate = current_stats.get('recent_win_rate', 0.5)
        if trading_days >= 5:
            if recent_win_rate > 0.55:
                reward += min((recent_win_rate - 0.5) * 5.0, 2.0)
            elif recent_win_rate < 0.35:
                reward -= min((0.35 - recent_win_rate) * 5.0, 2.0)

        # ==========================================
        # 8. Intraday Swing Penalty
        # ==========================================
        swing_gap = intraday_excursion - day_end_loss
        if swing_gap > 0.005 and intraday_excursion > 0.015:
            reward -= min(swing_gap * 100, 5.0)

        # ==========================================
        # 9. Per-Step Clip + Scale
        # ==========================================
        reward = max(min(reward, 20.0), -10.0)

        return float(reward) / 8.0
