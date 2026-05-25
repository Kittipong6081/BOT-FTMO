"""
===============================================================================
FTMO Trading Bot — Thai Explainer (v8.0.65)
===============================================================================
แปล signal/trade เป็นภาษาไทยเข้าใจง่าย ไม่มีศัพท์เทคนิค
ใช้สำหรับ Discord notification เท่านั้น — ไม่กระทบ trading logic

Safety:
- ทุก function wrap ใน try/except — ถ้า fail → return None / fallback string
- ไม่มี side effect, ไม่แก้ state ใด ๆ
- Pure formatting layer
===============================================================================
"""
from __future__ import annotations
import re
from typing import Dict, Optional, List


# ============================================================================
# Translation Dictionary — Technical → Plain Thai
# ============================================================================

def _explain_rsi(rsi: float) -> Optional[str]:
    """RSI = ความร้อน-เย็นของราคา (0-100)"""
    if rsi <= 0:
        return None
    if rsi >= 80:
        return f"🌡️ ความร้อนของราคา = {rsi:.0f}/100 → **ร้อนจัดมาก** (น่าจะเย็นลง)"
    if rsi >= 70:
        return f"🌡️ ความร้อนของราคา = {rsi:.0f}/100 → ร้อนเกิน (มักเย็นลง)"
    if rsi >= 60:
        return f"🌡️ ความร้อนของราคา = {rsi:.0f}/100 → อุ่น (กำลังเริ่มร้อน)"
    if rsi <= 20:
        return f"🌡️ ความร้อนของราคา = {rsi:.0f}/100 → **เย็นจัด** (น่าจะอุ่นขึ้น)"
    if rsi <= 30:
        return f"🌡️ ความร้อนของราคา = {rsi:.0f}/100 → เย็น (มักอุ่นขึ้น)"
    if rsi <= 40:
        return f"🌡️ ความร้อนของราคา = {rsi:.0f}/100 → เริ่มเย็น"
    return f"🌡️ ความร้อนของราคา = {rsi:.0f}/100 → ปกติ"


def _explain_bb(bb_pctb: float) -> Optional[str]:
    """BB %B = ตำแหน่งราคาในกรอบ (0=ขอบล่าง, 1=ขอบบน)"""
    if bb_pctb is None:
        return None
    if bb_pctb >= 1.0:
        return f"🎈 ตำแหน่งราคา = ทะลุกรอบบน ({bb_pctb:.2f}) → **ยืดสุดแล้ว เด้งกลับสูง**"
    if bb_pctb >= 0.85:
        return f"🎈 ตำแหน่งราคา = ใกล้ขอบบน ({bb_pctb:.2f}) → ยางยืดดึงสุด"
    if bb_pctb >= 0.70:
        return f"🎈 ตำแหน่งราคา = สูงกว่าเฉลี่ย ({bb_pctb:.2f}) → เริ่มยืด"
    if bb_pctb <= 0.0:
        return f"🎈 ตำแหน่งราคา = ทะลุกรอบล่าง ({bb_pctb:.2f}) → **ตกลึกสุด เด้งกลับสูง**"
    if bb_pctb <= 0.15:
        return f"🎈 ตำแหน่งราคา = ใกล้ขอบล่าง ({bb_pctb:.2f}) → ยางยืดดึงสุดด้านล่าง"
    if bb_pctb <= 0.30:
        return f"🎈 ตำแหน่งราคา = ต่ำกว่าเฉลี่ย ({bb_pctb:.2f}) → เริ่มยืดลง"
    return f"🎈 ตำแหน่งราคา = กลางกรอบ ({bb_pctb:.2f})"


def _explain_adx(adx: float) -> Optional[str]:
    """ADX = ความแรงของเทรนด์ (0-100)"""
    if adx <= 0:
        return None
    if adx >= 30:
        return f"💨 ความแรงเทรนด์ = {adx:.0f}/100 → **🔴 ตลาด trend แรง — ระวังสวนทาง!**"
    if adx >= 25:
        return f"💨 ความแรงเทรนด์ = {adx:.0f}/100 → 🟡 เริ่มมี trend ชัด"
    if adx >= 20:
        return f"💨 ความแรงเทรนด์ = {adx:.0f}/100 → ตลาดเริ่มทิศ (พอใช้)"
    return f"💨 ความแรงเทรนด์ = {adx:.0f}/100 → ✅ ตลาดเงียบ ไม่ trend (เหมาะ MR)"


def _explain_ml(ml_score: float) -> Optional[str]:
    """ML Score = AI ให้คะแนนความน่าจะชนะ (0-1)"""
    if ml_score is None or ml_score <= 0:
        return None
    pct = ml_score * 100
    if ml_score >= 0.80:
        return f"🤖 AI ให้คะแนน = {pct:.0f}% → **🌟 มั่นใจสูงมาก**"
    if ml_score >= 0.65:
        return f"🤖 AI ให้คะแนน = {pct:.0f}% → 🟢 มั่นใจสูง"
    if ml_score >= 0.50:
        return f"🤖 AI ให้คะแนน = {pct:.0f}% → 🟡 มั่นใจปานกลาง"
    if ml_score >= 0.35:
        return f"🤖 AI ให้คะแนน = {pct:.0f}% → 🟡 มั่นใจค่อนข้างต่ำ"
    return f"🤖 AI ให้คะแนน = {pct:.0f}% → ⚠️ มั่นใจต่ำ"


def _explain_confluence(conf: float) -> Optional[str]:
    """Confluence = คะแนนรวมจุดเทรด (0-100, เกณฑ์ ≥75)"""
    if conf is None or conf <= 0:
        return None
    if conf >= 90:
        return f"⭐ คะแนนรวมจุดเทรด = {conf:.0f}/100 → **เกรด A+ (ดีเยี่ยม)**"
    if conf >= 85:
        return f"⭐ คะแนนรวมจุดเทรด = {conf:.0f}/100 → เกรด A (ดี)"
    if conf >= 80:
        return f"⭐ คะแนนรวมจุดเทรด = {conf:.0f}/100 → เกรด B+ (ดี)"
    if conf >= 75:
        return f"⭐ คะแนนรวมจุดเทรด = {conf:.0f}/100 → เกรด B (ผ่านเกณฑ์)"
    return f"⭐ คะแนนรวมจุดเทรด = {conf:.0f}/100 → ต่ำกว่าเกณฑ์"


def _explain_volatility(regime: str) -> Optional[str]:
    """volatility_regime = สภาพคลื่นของราคา"""
    if not regime:
        return None
    mapping = {
        "quiet": "🌊 ความผันผวน = ตลาดเงียบนิ่ง",
        "normal": "🌊 ความผันผวน = ปกติ",
        "high": "🌊 ความผันผวน = สูง (ราคาแกว่งแรง)",
        "explosive": "🌊 ความผันผวน = 🔴 ระเบิด! (ระวัง — ตลาดบ้า)",
    }
    return mapping.get(str(regime).lower())


def _explain_chronos(align: float, unc: float) -> Optional[str]:
    """Chronos = AI ทำนายอนาคต 2 ชม. ข้างหน้า"""
    if align == 0 and unc == 0:
        return None
    direction = "เห็นด้วย ✅" if align > 0 else "ไม่เห็นด้วย ⚠️" if align < 0 else "กลาง ๆ"
    certainty = "มั่นใจ" if unc < 0.3 else "ไม่แน่ใจ" if unc > 0.7 else "ปานกลาง"
    return f"🔮 AI ทำนายอนาคต = {direction} ({certainty})"


def _explain_session(session: str) -> Optional[str]:
    """session = ช่วงเวลาเทรด"""
    if not session:
        return None
    mapping = {
        "LONDON": "🌅 ช่วงเวลา = London (volume สูง)",
        "NEW_YORK": "🌆 ช่วงเวลา = New York (volume สูง)",
        "LONDON_NY_OVERLAP": "🌇 ช่วงเวลา = London+NY ทับซ้อน (peak)",
        "ASIAN": "🌃 ช่วงเวลา = Asian (volume ต่ำ — ระวัง)",
        "OFF_HOURS": "🌙 ช่วงเวลา = นอกเวลา (low liquidity)",
    }
    return mapping.get(str(session).upper())


# ============================================================================
# Parser: ดึงค่าจาก signal_reasons string (e.g., "BB%B=0.95; RSI=75.8...")
# ============================================================================

def _parse_signal_reasons(reasons_str: str) -> Dict[str, float]:
    """Parse signal_reasons string → dict ของ feature values"""
    out: Dict[str, float] = {}
    if not reasons_str:
        return out
    # Patterns: "BB%B=0.95 (extreme=...)" "RSI=75.8 (...)" "ADX H1=12.5 (...)"
    patterns = {
        "bb_pctb": r"BB%B\s*=\s*([0-9.]+)",
        "rsi": r"RSI\s*=\s*([0-9.]+)",
        "wick_ratio": r"Wick ratio\s*=\s*([0-9.]+)",
        "adx_h1": r"ADX H1\s*=\s*([0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, reasons_str)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                continue
    return out


# ============================================================================
# Confidence Score (1-5 stars based on multiple factors)
# ============================================================================

def _compute_confidence_stars(
    ml_score: float, confluence: float, adx_h1: float
) -> str:
    """รวมหลายปัจจัย → ดาว 1-5"""
    score = 0
    # ML score (max 2 pts)
    if ml_score >= 0.75: score += 2
    elif ml_score >= 0.55: score += 1.5
    elif ml_score >= 0.40: score += 1
    elif ml_score >= 0.30: score += 0.5
    # Confluence (max 2 pts)
    if confluence >= 90: score += 2
    elif confluence >= 85: score += 1.5
    elif confluence >= 80: score += 1
    elif confluence >= 75: score += 0.5
    # ADX (penalty if trending)
    if adx_h1 >= 30: score -= 1
    elif adx_h1 >= 25: score -= 0.5
    elif adx_h1 < 20: score += 0.5
    # Clamp 0-5
    score = max(0, min(5, score))
    full = int(score)
    half = score - full >= 0.5
    return "⭐" * full + ("✨" if half else "")


# ============================================================================
# Main public function: format Thai explanation for trade_dict
# ============================================================================

def format_trade_open_explanation(trade_dict: Dict) -> Optional[str]:
    """
    สร้าง Discord-friendly Thai explanation จาก trade_dict.

    Args:
        trade_dict: จาก ExecutedTrade.to_dict()

    Returns:
        Markdown-formatted Thai string (Discord embed value)
        หรือ None ถ้า fail (caller จะข้าม field นี้ไป)
    """
    try:
        trade_type = (trade_dict.get("type") or "").upper()
        if trade_type not in ("BUY", "SELL"):
            return None

        # Direction explanation (MR context)
        if trade_type == "SELL":
            direction_th = (
                "**ขาย** — เก็งว่าราคาขึ้นมา**แรงเกินไป** "
                "เดี๋ยวจะกลับลง\n"
                "_(กลยุทธ์ยางยืด — ดึงสุดก็เด้งกลับ)_"
            )
        else:  # BUY
            direction_th = (
                "**ซื้อ** — เก็งว่าราคาลงมา**ลึกเกินไป** "
                "เดี๋ยวจะเด้งกลับขึ้น\n"
                "_(กลยุทธ์ยางยืด — ดึงสุดก็เด้งกลับ)_"
            )

        # Parse signal_reasons string for BB/RSI/Wick
        parsed = _parse_signal_reasons(trade_dict.get("reasons", "") or "")
        bb_pctb = parsed.get("bb_pctb")
        rsi = parsed.get("rsi", 0.0)

        # Direct fields
        adx_h1 = float(trade_dict.get("adx_h1") or 0)
        ml_score = float(trade_dict.get("ml_score") or 0)
        confluence = float(trade_dict.get("confluence") or 0)
        vol_regime = trade_dict.get("volatility_regime", "")
        session = trade_dict.get("session", "")
        chronos_align = float(trade_dict.get("chronos_align") or 0)
        chronos_unc = float(trade_dict.get("chronos_unc") or 0)
        risk_amount = float(trade_dict.get("risk_amount") or 0)
        risk_pct = float(trade_dict.get("risk_pct") or 0)
        rr_ratio = float(trade_dict.get("rr_ratio") or 1.0)

        # Build reason lines (skip None)
        reasons: List[str] = []
        for line in [
            _explain_bb(bb_pctb) if bb_pctb is not None else None,
            _explain_rsi(rsi),
            _explain_adx(adx_h1),
            _explain_ml(ml_score),
            _explain_confluence(confluence),
            _explain_volatility(vol_regime),
            _explain_chronos(chronos_align, chronos_unc),
            _explain_session(session),
        ]:
            if line:
                reasons.append(line)

        # Confidence stars
        stars = _compute_confidence_stars(ml_score, confluence, adx_h1)

        # Expected outcome (rough)
        max_loss = risk_amount
        max_profit = risk_amount * rr_ratio

        # Final composition
        parts = [
            f"💡 **ทิศทาง:** {direction_th}",
            "",
            f"💰 **เดิมพัน:** ${risk_amount:.0f} ({risk_pct:.1%} ของพอร์ต)",
            f"🎯 **เป้ากำไร:** +${max_profit:.0f}  |  🛡️ **เซฟสุด:** -${max_loss:.0f}",
            "",
            "📊 **ทำไมบอทถึงเข้า:**",
        ]
        parts.extend(reasons if reasons else ["• (ไม่มีข้อมูลรายละเอียด)"])
        parts.append("")
        parts.append(f"🎖️ **ระดับมั่นใจ:** {stars or '⭐'}")
        parts.append("")
        parts.append(
            "_หมายเหตุ: หากราคาเข้าทาง 0.3R → บอทเลื่อน SL ปกป้องอัตโนมัติ_"
        )

        return "\n".join(parts)

    except Exception:
        # Silent fail — ไม่ block notification เดิม
        return None
