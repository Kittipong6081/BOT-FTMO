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

        # New stats สำหรับปิดช่องโหว่ L1/L4/L5 (fallback 0 ถ้า env ยังไม่ส่งมา)
        trading_days = current_stats.get('trading_days', 0)         # วันที่เทรดอย่างน้อย 1 ไม้
        is_final_step = current_stats.get('is_final_step', False)   # step สุดท้ายของ challenge
        consecutive_losses = current_stats.get('consecutive_losses', 0)  # แพ้ติดกัน (recent)
        intraday_excursion = current_stats.get('intraday_excursion_pct', 0.0)
        day_end_loss = current_stats.get('day_end_loss_pct', 0.0)

        # ==========================================
        # 1. บทลงโทษ (Drawdown Penalty) - แบบ Exponential (ยิ่งใกล้ 4% หรือ 8% ยิ่งติดลบหนัก)
        # ==========================================
        
        # ถ้าระบบผิดพลาดจนทำผิดกฎ (Fail) -> ลงโทษขั้นสูงสุด
        if daily_dd >= self.daily_limit or total_dd >= self.total_limit:
            return -10.0  # การฆ่าตัวตาย = พัง (scaled 10×)

        # ทวีคูณ Penalty ถ้ายิ่งใกล้ Limit ของ Daily
        daily_danger_ratio = daily_dd / self.daily_limit
        if daily_danger_ratio > 0.5:
            # Exponential แต่ลดชันลง (3.5→2.5) — ไม่ให้ dominate positive signal
            # ratio=0.9 → e^2.25 - 1 ≈ 8.5 (เดิม 14.8)
            reward -= (math.exp(daily_danger_ratio * 2.5) - 1.0)

        # ทวีคูณ Penalty ถ้ายิ่งใกล้ Limit ของ Total (4.0→3.0)
        total_danger_ratio = total_dd / self.total_limit
        if total_danger_ratio > 0.5:
            reward -= (math.exp(total_danger_ratio * 3.0) - 1.0)

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
            # Cap profit reward per step ป้องกัน reward-hack จาก "ชนะเล็กๆ หลายครั้ง"
            # (L3: 100 micro-wins × 0.01% เดิมได้ 500 เต็มๆ; ตอนนี้ cap ที่ 8/step)
            reward += min((delta_progress * 100) * 0.8, 15.0)
        elif delta_progress < 0:
            # กำไรหด ลงโทษเบาๆ เพราะเดี๋ยวไปเจอ DD penalty อยู่แล้ว
            reward -= abs(delta_progress * 100) * 0.2

        # โบนัสถ้าถึงเป้า 10% — ต้องสูงพอให้ EV(aggressive) > EV(passive)
        # Passive agent: ~30 steps × -1.5 urgency + end penalty -20 = -65 → /10 = -6.5
        # Target agent: +150 target + progress + urgency - some DD = ต้อง net positive
        if progress >= 100.0:
            reward += 150.0

        # ==========================================
        # 3.5. Trading Incentive — ให้รางวัลเล็กน้อยสำหรับการเทรด (ต้านนั่งเฉย)
        # ==========================================
        if trades_today > 0 and trades_today <= 5:
            reward += 1.0

        # ==========================================
        # 4. Over-trading Penalty
        # ==========================================
        MAX_DAILY_TRADES = 5
        if trades_today > MAX_DAILY_TRADES:
            excess = trades_today - MAX_DAILY_TRADES
            reward -= min(excess * 2.0, 20.0)

        # ==========================================
        # 5. Activity Floor (L1) — บังคับให้เทรดจริง ไม่ใช่ policy "do nothing"
        # ==========================================
        # FTMO บังคับ ≥10 trading days. ถ้าจบ challenge โดยเทรดน้อยกว่านี้ → ปรับหนัก
        if is_final_step and trading_days < 10:
            missing = 10 - trading_days
            reward -= min(missing * 3.0, 30.0)  # ขาด 10 วัน = -30

        # ==========================================
        # 5.5. Urgency Reward — บังคับให้เดินหน้า ไม่ใช่นั่งเฉยรอจบ
        # ==========================================
        # expected_pace = วันที่ / 30 × 100 (ต้องถึง 100% ใน 30 วัน)
        # ถ้า progress >= 70% ของ expected → on-pace bonus
        # ถ้า progress < 70% ของ expected → behind-pace penalty
        current_step = current_stats.get('current_step', 0)
        max_steps = current_stats.get('max_steps', 45)
        if not is_final_step and current_step > 0:
            expected_progress = (current_step / max_steps) * 100.0
            if progress >= expected_progress * 0.7:
                reward += 2.0
            else:
                # passive policy: 45 steps × -1.0 = -45 → /10 = -4.5
                reward -= 1.0

        # ==========================================
        # 5.6. End-of-Challenge Penalty — ไม่ถึงเป้า = ลงโทษ
        # ==========================================
        if is_final_step and progress < 100.0:
            if progress < 30.0:
                # ไม่ถึง 30% = ปรับหนัก (passive agent ตกที่นี่)
                reward -= 20.0
            elif progress < 70.0:
                reward -= 10.0
            else:
                # 70-99% ใกล้แล้ว — penalty เบาๆ
                reward -= 5.0

        # ==========================================
        # 6. Consistency Multiplier (L4) — ให้ reward เชิงบวกลดลงถ้าเทรดน้อย
        # ==========================================
        # agent ที่เทรด 1-2 วันแล้วนั่ง idle จะได้ sortino/progress bonus ลดลง
        # สูตร: มี effect เฉพาะส่วน reward ที่เป็นบวก
        if reward > 0 and trading_days >= 0:
            consistency = min(max(trading_days, 0), 15) / 15.0  # 0 → 0, ≥15 → 1.0
            # blend: 30% × consistency + 70% flat → เบาลง ไม่ punish ช่วงต้น episode
            reward *= (0.7 + 0.3 * consistency)

        # ==========================================
        # 7. Consecutive-Loss Spiral Penalty (L4/L5)
        # ==========================================
        # แพ้ติดกัน ≥3 → ลดแรง revenge trading (ลดจาก 5→3 เพราะ 45 วันมีเวลาฟื้น)
        if consecutive_losses >= 3:
            reward -= min((consecutive_losses - 2) * 3.0, 15.0)

        # ==========================================
        # 8. Intraday Swing Penalty (L2-equivalent)
        # ==========================================
        # ปิดช่องโหว่ "เปิดไม้ลึกจนเกือบ breach แล้วโชคดีเด้งกลับ"
        # ถ้า intraday_excursion > 1.5× day_end_loss → swing แต่ปิดได้ = lucky, ลงโทษ
        # ป้องกันไม่ให้ agent เรียนรู้ว่า hold loser แล้วหวัง mean-revert คุ้ม
        swing_gap = intraday_excursion - day_end_loss
        if swing_gap > 0.005 and intraday_excursion > 0.015:
            # gap ≥ 0.5% และ excursion เคยถึง 1.5% ของ daily start equity
            reward -= min(swing_gap * 200, 12.0)  # cap -12

        # Scale down 10× เพื่อลด value_loss และป้องกัน policy diverge
        return float(reward) / 10.0
