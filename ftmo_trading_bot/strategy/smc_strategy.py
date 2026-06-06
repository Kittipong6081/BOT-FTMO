"""
===============================================================================
SMCScanner — live Smart Money Concepts scanner (StrategyBase)
===============================================================================
Drop-in for the live loop: `FTMOTradingBot` calls `scan_all_symbols()` and reads
`get_ltf_data/get_mtf_data/get_htf_data(symbol)`. Mirrors the LiveMRScanner
plumbing (per-symbol caches — no cross-symbol contamination) but the engine is
the strict-SMC `EntryEngine`.

Timeframe map (SMC MTF top-down):
  • HTF = D1  → BiasEngine (directional bias + liquidity target)
  • MTF = H4  → intermediate structure (cached for obs/context)
  • LTF = M15 → sweep + CHOCH + entry zone (EntryEngine)
===============================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import pandas as pd

from strategy.indicators import TechnicalIndicators
from strategy.strategy_base import StrategyBase
from strategy.smc.entry import EntryEngine, SMCEntryConfig
from strategy.smc.types import SMCSignal


def pip_size_for(symbol: str) -> float:
    s = symbol.upper()
    if "XAU" in s or "GOLD" in s:
        return 0.1
    if s.endswith("JPY"):
        return 0.01
    return 0.0001


class SMCScanner(StrategyBase):
    STRATEGY_ID: str = "SMC"
    MAGIC_NUMBER: int = 123458
    OBS_LAYOUT_ID: str = "smc_v1"

    SCAN_BARS_D1: int = 250
    SCAN_BARS_H4: int = 250
    SCAN_BARS_M15: int = 300

    def __init__(self, connector, config: Optional[SMCEntryConfig] = None):
        self._connector = connector
        self._indicators = TechnicalIndicators()
        self._entry = EntryEngine(config)
        self._ltf_by_symbol: Dict[str, pd.DataFrame] = {}   # M15
        self._mtf_by_symbol: Dict[str, pd.DataFrame] = {}   # H4
        self._htf_by_symbol: Dict[str, pd.DataFrame] = {}   # D1
        from config.settings import bot_config
        blocked = set(getattr(bot_config.symbols, "blocked_symbols", []))
        self._symbols = [s for s in bot_config.symbols.symbols if s not in blocked]
        if blocked:
            print(f"🚫 [SMCScanner] blocked_symbols: {sorted(blocked)} "
                  f"(trading {len(self._symbols)}/{len(bot_config.symbols.symbols)})")

    # ─── Public API ───────────────────────────────────────────────────
    def scan_all_symbols(self, allowed_symbols: Optional[Set[str]] = None) -> List[SMCSignal]:
        results: List[SMCSignal] = []
        for symbol in self._symbols:
            if allowed_symbols is not None and symbol not in allowed_symbols:
                continue
            sig = self._scan_one_symbol(symbol)
            if sig is not None and sig.is_valid:
                results.append(sig)
        results.sort(key=lambda s: s.confluence_score, reverse=True)
        return results

    def _scan_one_symbol(self, symbol: str) -> Optional[SMCSignal]:
        try:
            d1 = self._connector.get_ohlcv(symbol, "D1", self.SCAN_BARS_D1)
            h4 = self._connector.get_ohlcv(symbol, "H4", self.SCAN_BARS_H4)
            m15 = self._connector.get_ohlcv(symbol, "M15", self.SCAN_BARS_M15)
        except Exception:
            return None
        if m15 is None or d1 is None or len(m15) < 40 or len(d1) < 20:
            return None
        try:
            m15 = self._indicators.calculate_all(m15)
        except Exception:
            pass
        self._ltf_by_symbol[symbol] = m15
        self._mtf_by_symbol[symbol] = h4
        self._htf_by_symbol[symbol] = d1

        res = self._entry.evaluate(symbol, d1, m15, pip_size=pip_size_for(symbol))
        if res.signal is None:
            return None
        return self._enrich(res.signal, m15)

    def _enrich(self, sig: SMCSignal, m15: pd.DataFrame) -> SMCSignal:
        """Fill generic technical fields used by the obs builder / GBM / log."""
        try:
            last = m15.iloc[-1]
            sig.adx = float(last.get("adx", sig.adx))
            sig.rsi_value = float(last.get("rsi", sig.rsi_value))
            sig.atr_change_ratio = float(last.get("atr_change_ratio", 0.0))
            sig.price_roc = float(last.get("price_roc", 0.0))
            sig.stoch_k = float(last.get("stoch_k", 50.0))
            sig.bb_pctb = float(last.get("bb_pctb", 0.5))
            sig.macd_histogram = float(last.get("macd_histogram", 0.0))
        except Exception:
            pass
        return sig

    # ─── Per-symbol cache accessors ───────────────────────────────────
    def get_ltf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._ltf_by_symbol.get(symbol)

    def get_mtf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._mtf_by_symbol.get(symbol)

    def get_htf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._htf_by_symbol.get(symbol)
