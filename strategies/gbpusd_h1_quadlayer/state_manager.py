"""
State Manager - Persist critical trading state across restarts
=============================================================

Saves:
- Circuit breaker status (month stopped, day stopped)
- Monthly P&L tracking
- Consecutive loss counter
- Pattern filter state (trade history, halt status)

Author: SURIOTA Team
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class TradingState:
    """Persistent trading state"""
    # Timestamps
    last_updated: str = ""

    # Layer 3: Intra-month risk
    current_month: str = ""  # "2025-01"
    monthly_pnl: float = 0.0
    consecutive_losses: int = 0
    month_stopped: bool = False
    day_stopped: bool = False
    day_stopped_date: str = ""

    # Layer 4: Pattern filter
    is_halted: bool = False
    halt_reason: str = ""
    in_recovery: bool = False
    recovery_wins_needed: int = 0
    trade_count: int = 0

    # Trade history for pattern filter (last 50 trades)
    # Format: [{"direction": "BUY", "pnl": 100.0, "time": "2025-01-01T10:00:00"}, ...]
    trade_history: List[Dict] = None

    def __post_init__(self):
        if self.trade_history is None:
            self.trade_history = []


class StateManager:
    """
    Manages persistent state for trading bot.

    Saves state to JSON file in strategy folder.
    Automatically loads on init, saves on every update.
    """

    def __init__(self, state_file: str = None):
        """
        Initialize state manager.

        Args:
            state_file: Path to state file. Defaults to strategy/state.json
        """
        if state_file is None:
            strategy_dir = Path(__file__).parent
            state_file = strategy_dir / "state.json"

        self.state_file = Path(state_file)
        self.state = TradingState()

        # Load existing state if available
        self._load()

    def _load(self):
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                # Update state from loaded data
                for key, value in data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)

                logger.info(f"[StateManager] Loaded state from {self.state_file}")
                logger.info(f"[StateManager] Month: {self.state.current_month}, PnL: ${self.state.monthly_pnl:.2f}, Stopped: {self.state.month_stopped}")

            except Exception as e:
                logger.warning(f"[StateManager] Failed to load state: {e}")
                self.state = TradingState()
        else:
            logger.info(f"[StateManager] No existing state file, starting fresh")

    def _save(self):
        """Save state to file"""
        try:
            self.state.last_updated = datetime.now(timezone.utc).isoformat()

            # Convert to dict
            data = asdict(self.state)

            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"[StateManager] Failed to save state: {e}")

    # ============================================================
    # Layer 3: Intra-Month Risk State
    # ============================================================

    def check_month_reset(self, current_time: datetime) -> bool:
        """
        Check if we need to reset for new month.

        Returns:
            True if month was reset
        """
        current_month = current_time.strftime("%Y-%m")

        if self.state.current_month != current_month:
            logger.info(f"[StateManager] New month detected: {self.state.current_month} -> {current_month}")

            # Reset monthly state
            self.state.current_month = current_month
            self.state.monthly_pnl = 0.0
            self.state.consecutive_losses = 0
            self.state.month_stopped = False
            self.state.day_stopped = False
            self.state.day_stopped_date = ""

            # Reset pattern filter for new month
            self.state.is_halted = False
            self.state.halt_reason = ""
            self.state.in_recovery = False
            self.state.recovery_wins_needed = 0

            self._save()
            return True

        return False

    def check_day_reset(self, current_time: datetime) -> bool:
        """
        Check if we need to reset day stop.

        Returns:
            True if day was reset
        """
        current_date = current_time.strftime("%Y-%m-%d")

        if self.state.day_stopped and self.state.day_stopped_date != current_date:
            logger.info(f"[StateManager] New day - resetting day stop")
            self.state.day_stopped = False
            self.state.day_stopped_date = ""
            self._save()
            return True

        return False

    def record_trade(self, pnl: float, direction: str, current_time: datetime):
        """
        Record a trade result and update state.

        Args:
            pnl: Trade P&L in dollars
            direction: "BUY" or "SELL"
            current_time: Trade close time
        """
        # Ensure we're in the right month
        self.check_month_reset(current_time)

        # Update monthly P&L
        self.state.monthly_pnl += pnl

        # Update consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Add to trade history
        trade_record = {
            "direction": direction,
            "pnl": pnl,
            "time": current_time.isoformat()
        }
        self.state.trade_history.append(trade_record)

        # Keep only last 50 trades
        if len(self.state.trade_history) > 50:
            self.state.trade_history = self.state.trade_history[-50:]

        self.state.trade_count += 1

        self._save()

        logger.debug(f"[StateManager] Recorded trade: {direction} ${pnl:+.2f}, Monthly: ${self.state.monthly_pnl:.2f}, ConsecLoss: {self.state.consecutive_losses}")

    def set_month_stopped(self, reason: str = ""):
        """Set month circuit breaker"""
        self.state.month_stopped = True
        logger.warning(f"[StateManager] MONTH STOPPED: {reason}")
        self._save()

    def set_day_stopped(self, current_time: datetime, reason: str = ""):
        """Set day stop"""
        self.state.day_stopped = True
        self.state.day_stopped_date = current_time.strftime("%Y-%m-%d")
        logger.warning(f"[StateManager] DAY STOPPED: {reason}")
        self._save()

    def is_month_stopped(self) -> bool:
        """Check if month is stopped"""
        return self.state.month_stopped

    def is_day_stopped(self) -> bool:
        """Check if day is stopped"""
        return self.state.day_stopped

    def get_monthly_pnl(self) -> float:
        """Get current monthly P&L"""
        return self.state.monthly_pnl

    def get_consecutive_losses(self) -> int:
        """Get consecutive loss count"""
        return self.state.consecutive_losses

    # ============================================================
    # Layer 4: Pattern Filter State
    # ============================================================

    def set_halted(self, reason: str):
        """Set pattern filter halt"""
        self.state.is_halted = True
        self.state.halt_reason = reason
        logger.warning(f"[StateManager] PATTERN HALT: {reason}")
        self._save()

    def clear_halt(self):
        """Clear pattern filter halt"""
        self.state.is_halted = False
        self.state.halt_reason = ""
        self._save()

    def set_recovery_mode(self, wins_needed: int = 1):
        """Enter recovery mode"""
        self.state.in_recovery = True
        self.state.recovery_wins_needed = wins_needed
        self._save()

    def exit_recovery_mode(self):
        """Exit recovery mode"""
        self.state.in_recovery = False
        self.state.recovery_wins_needed = 0
        self._save()

    def is_halted(self) -> bool:
        """Check if pattern filter is halted"""
        return self.state.is_halted

    def is_in_recovery(self) -> bool:
        """Check if in recovery mode"""
        return self.state.in_recovery

    def get_trade_history(self) -> List[Tuple[str, float, datetime]]:
        """
        Get trade history as list of tuples.

        Returns:
            List of (direction, pnl, time) tuples
        """
        result = []
        for t in self.state.trade_history:
            try:
                time = datetime.fromisoformat(t['time'])
                result.append((t['direction'], t['pnl'], time))
            except:
                pass
        return result

    def get_trade_count(self) -> int:
        """Get total trade count"""
        return self.state.trade_count

    # ============================================================
    # Summary
    # ============================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get state summary for display"""
        return {
            "month": self.state.current_month,
            "monthly_pnl": self.state.monthly_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            "month_stopped": self.state.month_stopped,
            "day_stopped": self.state.day_stopped,
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "in_recovery": self.state.in_recovery,
            "trade_count": self.state.trade_count,
            "last_updated": self.state.last_updated
        }

    def reset_all(self):
        """Reset all state (use with caution!)"""
        logger.warning("[StateManager] FULL STATE RESET")
        self.state = TradingState()
        self._save()
