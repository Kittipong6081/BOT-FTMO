"""
===============================================================================
SMC core types — enums + dataclasses shared by every SMC engine
===============================================================================
Strict Smart Money Concepts rebuild (branch rebuild/smc-v2).

These are pure value objects. Engines (swing/structure/liquidity/fvg/
order_block/bias/entry) produce and consume them. No pandas/IO here so the
types stay cheap to import and trivial to unit-test.

Sign convention: Direction.BULLISH = +1, Direction.BEARISH = -1 (matches the
project-wide int bias convention on signals: market_bias/d1_bias ∈ {-1,0,1}).
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Direction(Enum):
    BULLISH = 1
    BEARISH = -1

    @property
    def sign(self) -> int:
        return self.value

    def opposite(self) -> "Direction":
        return Direction.BEARISH if self is Direction.BULLISH else Direction.BULLISH


class SwingType(Enum):
    HIGH = "high"
    LOW = "low"


class SwingLabel(Enum):
    HH = "HH"   # higher high
    HL = "HL"   # higher low
    LH = "LH"   # lower high
    LL = "LL"   # lower low


class StructureKind(Enum):
    BOS = "BOS"      # break of structure — continuation (same direction as trend)
    CHOCH = "CHOCH"  # change of character — first break against the trend (reversal)


class Zone(Enum):
    PREMIUM = "premium"      # above equilibrium → shorts
    DISCOUNT = "discount"    # below equilibrium → longs
    EQUILIBRIUM = "equilibrium"


@dataclass
class SwingPoint:
    """A confirmed fractal swing.

    `confirm_index` is the bar at which the fractal becomes knowable (= index +
    right lookback). Downstream engines must NOT reference a swing before its
    `confirm_index` — that is the no-lookahead / leakage-safety contract.
    """
    index: int
    price: float
    swing_type: SwingType
    confirm_index: int
    is_external: bool = False
    label: Optional[SwingLabel] = None


@dataclass
class StructureEvent:
    index: int                 # bar where the break confirmed (candle close beyond level)
    kind: StructureKind
    direction: Direction       # direction of the break
    level: float               # the swing price that was broken
    broken_swing_index: int    # index of the swing whose level broke


@dataclass
class FVG:
    """3-candle fair value gap. `top`/`bottom` bound the imbalance zone."""
    direction: Direction       # BULLISH = bullish displacement (gap is a discount to revisit)
    top: float
    bottom: float
    index: int                 # index of candle 3 (completion) — when the gap is knowable
    mid: float = 0.0           # 50% (consequent encroachment)
    mitigation: float = 0.0    # [0,1] fraction of the gap price has retraced into
    filled: bool = False       # closed fully through the gap → edge gone

    def __post_init__(self) -> None:
        if self.top < self.bottom:
            self.top, self.bottom = self.bottom, self.top
        self.mid = (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class OrderBlock:
    """Last opposing candle before a displacement that caused a BOS."""
    direction: Direction       # BULLISH OB = demand zone (last down candle before up move)
    top: float
    bottom: float
    index: int                 # bar index of the OB candle
    confirm_index: int         # bar where the displacement/BOS confirmed it
    grade: float = 0.0         # [0,1] quality score
    mitigation: float = 0.0    # [0,1] fraction price has retraced into the zone
    mitigated: bool = False    # body closed through → spent

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class LiquidityLevel:
    """A resting-liquidity pool: a swing extreme or equal-high/low cluster."""
    direction: Direction       # BULLISH = buy-side (above highs), BEARISH = sell-side (below lows)
    price: float
    index: int                 # latest bar contributing to the level
    equal_count: int = 1       # how many swings cluster at this level (>=2 = equal highs/lows)
    swept: bool = False


@dataclass
class LiquiditySweep:
    """A wick beyond a liquidity level followed by a close back inside."""
    direction: Direction       # direction of the REVERSAL expected after the sweep
    level: float               # the liquidity price that was swept
    index: int                 # bar that did the sweep (wick beyond + close back in)
    penetration: float = 0.0   # how far beyond the level the wick reached (price units)


@dataclass
class DealingRange:
    """The range between the most recent external swing high and low.

    Equilibrium = 50%. OTE (optimal trade entry) = 62%–79% retracement zone.
    """
    high: float
    low: float
    high_index: int
    low_index: int

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def size(self) -> float:
        return self.high - self.low

    def position(self, price: float) -> float:
        """Where `price` sits in the range: 0.0 = low, 1.0 = high."""
        if self.size <= 0:
            return 0.5
        return (price - self.low) / self.size

    def zone(self, price: float) -> Zone:
        pos = self.position(price)
        if pos > 0.5:
            return Zone.PREMIUM
        if pos < 0.5:
            return Zone.DISCOUNT
        return Zone.EQUILIBRIUM

    def in_discount(self, price: float) -> bool:
        return self.position(price) <= 0.5

    def in_premium(self, price: float) -> bool:
        return self.position(price) >= 0.5

    def in_ote(self, price: float, direction: Direction) -> bool:
        """OTE = 0.62–0.79 retracement from the impulse origin.

        For a bullish setup the retrace is measured down from the high, so the
        OTE band sits at range position 0.21–0.38 (deep discount).
        """
        pos = self.position(price)
        if direction is Direction.BULLISH:
            return 0.21 <= pos <= 0.38
        return 0.62 <= pos <= 0.79


class SMCSignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def direction(self) -> Direction:
        return Direction.BULLISH if self is SMCSignalType.BUY else Direction.BEARISH


@dataclass
class SMCSignal:
    """The strategy → execution contract (mirrors the fields the live path and
    obs builder read off a signal; see wiki/02-modules infra audit)."""
    signal_type: SMCSignalType
    symbol: str
    entry_price: float
    sl_price: float
    tp_price: float
    confluence_score: float
    atr_value: float
    sl_distance: float
    rr_ratio: float

    # generic technicals (obs / GBM features / Excel log) — safe defaults
    market_bias: int = 0
    trend: int = 0
    rsi_value: float = 50.0
    trend_strength: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    stoch_k: float = 50.0
    bb_pctb: float = 0.5
    atr_change_ratio: float = 0.0
    price_roc: float = 0.0

    # SMC-specific telemetry (Brain-1 evidence → obs/log)
    htf_bias: int = 0                  # -1/0/1 from BiasEngine
    structure_event: str = ""          # "CHOCH" or "BOS" that armed the entry
    swept_liquidity: float = 0.0       # price level swept before the entry
    sweep_age_bars: int = 0
    ob_grade: float = 0.0              # [0,1] order-block quality (0 if FVG-only)
    fvg_mitigation: float = 0.0        # [0,1] mitigation of the entry FVG
    entry_zone_pos: float = 0.5        # position in HTF dealing range [0,1]
    has_ob: bool = False
    has_fvg: bool = False
    mss_confirmed: bool = False

    # meta
    strategy_id: str = "SMC"
    timestamp: Optional[object] = None
    ml_score: float = 0.5
    reasons: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.sl_distance > 0 and self.rr_ratio > 0
