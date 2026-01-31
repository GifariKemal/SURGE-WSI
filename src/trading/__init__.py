"""Trading Layer Module

Components:
- EntryTrigger: LTF confirmation for precise entries
- RiskManager: Dynamic position sizing by zone quality
- ExitManager: Partial TP strategy with trailing stop
- ExecutorH1V3: H1 v3 trade execution engine
- RSIMeanReversionV37: RSI Mean Reversion v3.7 executor (RECOMMENDED)
- TradeModeManager: Auto vs Signal-only mode management
"""

from .entry_trigger import EntryTrigger, LTFEntrySignal
from .risk_manager import RiskManager
from .exit_manager import ExitManager, PositionState
from .executor_h1_v3 import TradeExecutorH1V3
from .executor_rsi_v37 import RSIMeanReversionV37
from .trade_mode_manager import TradeModeManager, TradeMode, TradeModeConfig

__all__ = [
    "EntryTrigger",
    "LTFEntrySignal",
    "RiskManager",
    "ExitManager",
    "PositionState",
    "TradeExecutorH1V3",
    "RSIMeanReversionV37",
    "TradeModeManager",
    "TradeMode",
    "TradeModeConfig",
]
