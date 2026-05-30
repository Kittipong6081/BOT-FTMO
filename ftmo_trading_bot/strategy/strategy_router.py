"""
===============================================================================
FTMO Trading Bot — Strategy Router (v8.1 dual-strategy orchestration)
===============================================================================
Per scan: classify each symbol's regime, route it to the single strategy that
the regime arms (MR for RANGING, TF for TRENDING, none for AMBIGUOUS), run each
strategy ONLY on its allowed symbols, then merge + rank the signals.

Mutual exclusion is structural: `MarketRegimeClassifier.classify_all` returns
exactly one key (or None) per symbol, so a symbol can never be scanned by both
strategies in the same cycle. Symbols with an OPEN position are locked to their
owner (passed in as `open_owner`) so a regime flip mid-trade can't hand the
symbol to the other strategy — the executor's same-symbol block is the final
backstop.
===============================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from strategy.regime_classifier import MarketRegimeClassifier
from strategy.strategy_base import StrategyBase


class StrategyRouter:
    """Routes symbols to strategies by regime and merges their signals."""

    def __init__(
        self,
        strategies: Dict[str, StrategyBase],
        classifier: MarketRegimeClassifier,
    ):
        self._strategies = strategies
        self._classifier = classifier

    def scan(
        self,
        symbols: List[str],
        open_owner: Optional[Dict[str, str]] = None,
    ) -> Tuple[List, Dict[str, Optional[str]]]:
        """Classify regimes, scan the armed strategy per symbol, merge + rank.

        Args:
            symbols: all tradable symbols.
            open_owner: {symbol: strategy_id} for symbols with an open position
                (regime lock — held by their owner).

        Returns:
            (signals, regime_map) where `signals` is the merged, confluence-ranked
            list (each carrying `strategy_id`) and `regime_map` is
            {symbol: armed_strategy ("MR"/"TF"/None)} for logging/telemetry.
        """
        regime_map = self._classifier.classify_all(symbols, open_owner=open_owner)

        # Group symbols by the strategy that armed them.
        by_strategy: Dict[str, set] = {}
        for sym, sid in regime_map.items():
            if sid is None:
                continue
            by_strategy.setdefault(sid, set()).add(sym)

        signals: List = []
        for sid, scanner in self._strategies.items():
            allowed = by_strategy.get(sid)
            if not allowed:
                continue
            try:
                signals.extend(scanner.scan_all_symbols(allowed_symbols=allowed))
            except Exception as e:
                print(f"⚠️ [StrategyRouter] {sid} scan error: {e}")

        signals.sort(key=lambda s: s.confluence_score, reverse=True)
        return signals, regime_map
