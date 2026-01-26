"""Trading Layer Module

Components:
- EntryTrigger: LTF confirmation for precise entries
- RiskManager: Dynamic position sizing by zone quality
- ExitManager: Partial TP strategy with trailing stop
- Executor: Trade execution engine
- TradeModeManager: Auto vs Signal-only mode management
"""

from .entry_trigger import EntryTrigger, LTFEntrySignal
from .risk_manager import RiskManager
from .exit_manager import ExitManager, PositionState
from .executor import TradeExecutor
from .trade_mode_manager import TradeModeManager, TradeMode, TradeModeConfig

__all__ = [
    "EntryTrigger",
    "LTFEntrySignal",
    "RiskManager",
    "ExitManager",
    "PositionState",
    "TradeExecutor",
    "TradeModeManager",
    "TradeMode",
    "TradeModeConfig",
]
