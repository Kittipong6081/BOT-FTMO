"""
===============================================================================
FTMO Trading Bot — Strategy Abstraction Base (v8.1 multi-strategy pivot)
===============================================================================
Common interface for parallel strategies (Mean Reversion + Trend Following).

Design contract (CRITICAL — see wiki/05-invariants.md):
  • Each strategy OWNS its own per-symbol data caches (`_*_by_symbol` dicts) and
    NEVER reads another strategy's caches. The live loop (`FTMOTradingBot`)
    routes observation building, live-context, and RL inference by the
    signal's `strategy_id` → the matching `StrategyBase` instance.
  • `get_ltf_data/get_mtf_data/get_htf_data(symbol)` is the ONLY way the live
    path reads a strategy's cached OHLCV. The accessor name is shared but the
    timeframe is strategy-specific (MR → M15/H1/H4, TF → H1/H4/D1).
  • Each strategy carries a distinct `MAGIC_NUMBER` so its broker positions are
    attributable (per-strategy risk ledger + orphan recovery).

This keeps the v8.0.79/80 per-symbol cache contract intact while allowing a
second strategy to run in parallel without cross-strategy contamination.
===============================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Set

import pandas as pd


class StrategyBase(ABC):
    """Interface every live strategy scanner must implement.

    Subclasses set the three class attributes and implement the four abstract
    methods. `LiveMRScanner` (MR) and `TrendFollowingScanner` (TF) both derive
    from this.
    """

    #: Short strategy key, propagated onto every signal as `signal.strategy_id`.
    STRATEGY_ID: str = "BASE"
    #: Distinct MT5 magic number — makes broker positions strategy-attributable.
    MAGIC_NUMBER: int = 0
    #: Observation layout id — tells `_build_signal_observation` which obs vector
    #: (and which trained model) to use. MR → "mr_v8", TF → "tf_v1".
    OBS_LAYOUT_ID: str = "base"

    # ─── Scanning ─────────────────────────────────────────────────────────

    @abstractmethod
    def scan_all_symbols(self, allowed_symbols: Optional[Set[str]] = None) -> List:
        """Scan configured symbols and return a list of valid signals.

        Args:
            allowed_symbols: when provided, restrict scanning to this set
                (the StrategyRouter passes the symbols whose regime armed this
                strategy). `None` = scan every configured symbol (legacy
                behaviour — keeps single-strategy callers unchanged).
        """
        raise NotImplementedError

    # ─── Per-symbol cache accessors (own caches only — no cross-read) ──────

    @abstractmethod
    def get_ltf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Lower-timeframe (entry) dataframe for `symbol` from the latest scan."""
        raise NotImplementedError

    @abstractmethod
    def get_mtf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Mid-timeframe dataframe for `symbol` from the latest scan."""
        raise NotImplementedError

    @abstractmethod
    def get_htf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Higher-timeframe (trend) dataframe for `symbol` from the latest scan."""
        raise NotImplementedError

    # ─── Identity helpers (concrete) ──────────────────────────────────────

    def get_obs_layout_id(self) -> str:
        return self.OBS_LAYOUT_ID

    def get_strategy_id(self) -> str:
        return self.STRATEGY_ID

    def get_magic(self) -> int:
        return self.MAGIC_NUMBER
