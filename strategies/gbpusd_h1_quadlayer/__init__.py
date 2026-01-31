"""
GBPUSD H1 Quad-Layer Strategy Package
=====================================

Self-contained trading strategy for GBPUSD H1 timeframe with
5-layer quality filter for zero losing months.

Layers:
- Layer 1: Monthly Profile (tradeable percentage)
- Layer 2: Technical (ATR stability, efficiency, trend)
- Layer 3: Intra-Month Risk (IntraMonthRiskManager)
- Layer 4: Pattern-Based (PatternBasedFilter)
- Layer 5: Choppiness Index (ChoppinessFilter) - NEW!

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
    ChoppinessFilter,
    calculate_choppiness_index,
    DirectionalMomentumFilter,
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
    'ChoppinessFilter',
    'calculate_choppiness_index',
    'DirectionalMomentumFilter',
    'H1V64GBPUSDExecutor',
]

__version__ = '6.5.1'
