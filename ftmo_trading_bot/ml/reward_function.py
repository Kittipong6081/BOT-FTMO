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

import math
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
        
        โครงสร้างสถิติที่คาดหวัง:
        - daily_dd_pct: Drawdown สูงสุดในวัน
        - total_dd_pct: Drawdown สูงสุดตลอดกาล
        - sortino_ratio: อัตราผลตอบแทนเทียบความเสี่ยงขาลง
        - balance: ยอดเงินจำลอง (Equity)
        - target_progress_pct: ความคืบหน้าของกำไร (%)
        - trades_today: จำนวนเทรดวันนี้ (สำหรับ over-trading penalty)
        """

        reward = 0.0

        daily_dd = current_stats.get('daily_dd_pct', 0.0)
        total_dd = current_stats.get('total_dd_pct', 0.0)
        sortino = current_stats.get('sortino_ratio', 0.0)
        progress = current_stats.get('target_progress_pct', 0.0)
        prev_progress = previous_stats.get('target_progress_pct', 0.0)
        trades_today = current_stats.get('trades_today', 0)

        # ==========================================
        # 1. บทลงโทษ (Drawdown Penalty) - แบบ Exponential (ยิ่งใกล้ 4% หรือ 8% ยิ่งติดลบหนัก)
        # ==========================================
        
        # ถ้าระบบผิดพลาดจนทำผิดกฎ (Fail) -> ลงโทษขั้นสูงสุด
        if daily_dd >= self.daily_limit or total_dd >= self.total_limit:
            return -100.0  # การฆ่าตัวตาย = พัง

        # ทวีคูณ Penalty ถ้ายิ่งใกล้ Limit ของ Daily
        daily_danger_ratio = daily_dd / self.daily_limit
        if daily_danger_ratio > 0.5:
            # ใช้ Exponential ฐาน e: (e^(ratio*3) - 1)
            # ตัวอย่าง ratio=0.9 -> e^2.7 - 1 ≈ 14.8 ลงโทษ
            reward -= (math.exp(daily_danger_ratio * 3.5) - 1.0)

        # ทวีคูณ Penalty ถ้ายิ่งใกล้ Limit ของ Total
        total_danger_ratio = total_dd / self.total_limit
        if total_danger_ratio > 0.5:
            reward -= (math.exp(total_danger_ratio * 4.0) - 1.0)

        # ==========================================
        # 2. รางวัลด้านความเรียบเนียน (Stability Reward)
        # ==========================================
        # ให้ความสำคัญกับ Sortino Ratio ที่มากกว่า 1.5 เพราะเรารู้ตัวว่าเสี่ยงขาลงต่ำ
        if sortino > 0:
            reward += min(sortino * 0.5, 5.0)  # Max bonus จาก Sortino = 5
        elif sortino < 0:
            reward += max(sortino * 0.5, -5.0)

        # ==========================================
        # 3. รางวัลผลกำไร (Progress Reward)
        # ==========================================
        # ให้รางวัลถ้า Progress (เข้าใกล้ 10%) ขยับเป็นบวกจาก Action ก่อนหน้า
        delta_progress = progress - prev_progress
        
        if delta_progress > 0:
            # ถ้ากำไรเพิ่ม ให้ 10 คะแนน ต่อทุก 1% ของพอร์ตที่โตขึ้น
            reward += (delta_progress * 100) * 0.5 
        elif delta_progress < 0:
            # กำไรหด ลงโทษเบาๆ เพราะเดี๋ยวไปเจอ DD penalty อยู่แล้ว
            reward -= abs(delta_progress * 100) * 0.2

        # โบนัสก้อนใหญ่ถ้าเข้าถึงเป้า 10% สำเร็จ
        if progress >= 100.0:
            reward += 50.0

        # ==========================================
        # 4. Over-trading Penalty — ลงโทษถ้าเทรดเกิน MAX_TRADES_PER_DAY
        # ==========================================
        # เป้าหมาย: บังคับ Agent ให้เลือก setup คุณภาพสูง (ลด 45/วัน → ≤5/วัน)
        # Penalty เริ่มที่เทรดที่ 6 และโตแบบ linear × จำนวนเกิน
        MAX_DAILY_TRADES = 5
        if trades_today > MAX_DAILY_TRADES:
            excess = trades_today - MAX_DAILY_TRADES
            reward -= min(excess * 2.0, 20.0)  # cap ที่ -20 กันเหวี่ยง

        return float(reward)
