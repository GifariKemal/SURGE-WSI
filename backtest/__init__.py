"""Backtest Module
=================

Historical validation for SURGE-WSI.

Usage:
    from backtest import Backtester

    bt = Backtester(symbol="GBPUSD", start_date="2024-01-01")
    results = bt.run()
    bt.report()

Author: SURIOTA Team
"""

from .backtester import Backtester, BacktestResult

__all__ = ["Backtester", "BacktestResult"]
