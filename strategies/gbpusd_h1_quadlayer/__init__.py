"""
GBPUSD H1 Quad-Layer Strategy Package
=====================================

Self-contained trading strategy for GBPUSD H1 timeframe with
4-layer quality filter for zero losing months.

Layers:
- Layer 1: Monthly Profile (tradeable percentage based on 2024 historical data)
- Layer 2: Technical (ATR stability, efficiency, EMA trend)
- Layer 3: Intra-Month Risk (IntraMonthRiskManager)
- Layer 4: Pattern-Based (PatternBasedFilter)

REMOVED (tested, made results worse):
- Choppiness Index (E.W. Dreiss 1993) - Only adjusted 10 trades, didn't prevent losses
- DirectionalMomentumFilter - Added May as losing month when combined with ADX
- ADX regime detection - Too strict, blocked good trades

Modules:
- executor: Live trading executor
- trading_filters: Shared filter classes
- strategy_config: All configurable parameters
- state_manager: Persistent state for circuit breakers
- main: Entry point with startup checks

Author: SURIOTA Team
"""

from .strategy_config import SYMBOL, TIMEFRAME, RISK, TECHNICAL
from .state_manager import StateManager
from .trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
)
from .executor import H1V64GBPUSDExecutor

__all__ = [
    'SYMBOL',
    'TIMEFRAME',
    'RISK',
    'TECHNICAL',
    'StateManager',
    'IntraMonthRiskManager',
    'PatternBasedFilter',
    'H1V64GBPUSDExecutor',
]

__version__ = '6.7.0'  # EMA Pullback as secondary entry signal (+62% trades)
