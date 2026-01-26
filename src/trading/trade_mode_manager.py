"""Trade Mode Manager - Auto vs Signal-Only Mode
================================================

Manages when to auto-trade vs when to only send signals.

Signal-Only Mode triggers:
1. December (holiday period)
2. Daily loss > 2%
3. Weekly loss > 5%
4. High volatility (ATR > 50 pips)
5. Regime instability (many changes)
6. After 3 consecutive losses

Author: SURIOTA Team
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum
from loguru import logger


class TradeMode(Enum):
    """Trading mode"""
    AUTO = "auto"           # Full auto execution
    RECOVERY = "recovery"   # Auto with reduced lot size
    SIGNAL_ONLY = "signal"  # Only send signals, no execution
    MONITORING = "monitor"  # Full pause - only observe, no signals


@dataclass
class TradeModeConfig:
    """Configuration for trade mode manager"""
    # December settings - Two tier system
    # Dec 1-14: SIGNAL_ONLY (send signals, no auto-execute)
    # Dec 15-31: MONITORING (full pause, no signals, just observe)
    december_signal_only_start: int = 1   # Signal-only from Dec 1
    december_monitoring_start: int = 15   # Full pause from Dec 15
    december_mode_early: TradeMode = TradeMode.SIGNAL_ONLY  # Dec 1-14
    december_mode_late: TradeMode = TradeMode.MONITORING    # Dec 15-31

    # Loss limits - Two tier system
    # Tier 1: Recovery mode (reduced lot)
    daily_loss_recovery_pct: float = 1.5    # Switch to RECOVERY after 1.5% daily loss
    weekly_loss_recovery_pct: float = 3.0   # Switch to RECOVERY after 3% weekly loss

    # Tier 2: Signal-only (stop trading)
    daily_loss_limit_pct: float = 2.5       # Switch to SIGNAL after 2.5% daily loss
    weekly_loss_limit_pct: float = 6.0      # Switch to SIGNAL after 6% weekly loss

    # Recovery mode settings
    recovery_lot_multiplier: float = 0.5    # Use 50% of normal lot in recovery mode

    # Volatility - RELAXED from 50 to 65 pips
    high_volatility_atr_pips: float = 65.0  # Signal-only if ATR > 65 pips (was 50)
    medium_volatility_atr_pips: float = 45.0  # Recovery mode if ATR > 45 pips

    # Consecutive losses - Two tier system
    consecutive_losses_recovery: int = 2    # Recovery mode after 2 losses
    max_consecutive_losses: int = 4         # Signal-only after 4 losses (was 3)

    # Regime instability
    max_regime_changes_per_day: int = 6     # Signal-only if > 6 changes/day (was 5)
    regime_changes_recovery: int = 4        # Recovery mode if > 4 changes/day

    # Summer/Low liquidity months (August)
    summer_months: tuple = (8,)             # August - reduced liquidity
    summer_mode: TradeMode = TradeMode.SIGNAL_ONLY


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: datetime = None
    starting_balance: float = 0.0
    current_balance: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    consecutive_losses: int = 0
    regime_changes: int = 0


class TradeModeManager:
    """Manages auto vs signal-only trading mode"""

    def __init__(self, config: TradeModeConfig = None):
        """Initialize Trade Mode Manager

        Args:
            config: Trade mode configuration
        """
        self.config = config or TradeModeConfig()

        # State tracking
        self.current_mode: TradeMode = TradeMode.AUTO
        self.mode_reason: str = "Normal trading"

        # Daily stats
        self.daily_stats: DailyStats = DailyStats()
        self.weekly_pnl: float = 0.0
        self.weekly_start_balance: float = 0.0

        # Trade history for consecutive loss tracking
        self.recent_results: List[bool] = []  # True = win, False = loss

        # Regime change tracking
        self.last_regime: str = None
        self.daily_regime_changes: int = 0
        self.last_regime_check_date: datetime = None

        # Force mode override
        self._forced_mode: TradeMode = None
        self._force_expires: datetime = None

    def reset_daily_stats(self, current_balance: float):
        """Reset daily statistics"""
        today = datetime.now().date()
        self.daily_stats = DailyStats(
            date=today,
            starting_balance=current_balance,
            current_balance=current_balance
        )
        self.daily_regime_changes = 0

    def reset_weekly_stats(self, current_balance: float):
        """Reset weekly statistics"""
        self.weekly_pnl = 0.0
        self.weekly_start_balance = current_balance

    def update_balance(self, new_balance: float):
        """Update current balance"""
        self.daily_stats.current_balance = new_balance
        self.daily_stats.pnl = new_balance - self.daily_stats.starting_balance
        self.weekly_pnl = new_balance - self.weekly_start_balance

    def record_trade_result(self, is_win: bool, pnl: float):
        """Record trade result

        Args:
            is_win: True if trade was profitable
            pnl: Profit/loss amount
        """
        self.daily_stats.trades += 1

        if is_win:
            self.daily_stats.wins += 1
            self.daily_stats.consecutive_losses = 0
        else:
            self.daily_stats.losses += 1
            self.daily_stats.consecutive_losses += 1

        self.daily_stats.pnl += pnl
        self.weekly_pnl += pnl

        # Track recent results
        self.recent_results.append(is_win)
        if len(self.recent_results) > 10:
            self.recent_results.pop(0)

    def record_regime_change(self, new_regime: str):
        """Record regime change

        Args:
            new_regime: New market regime
        """
        today = datetime.now().date()

        # Reset if new day
        if self.last_regime_check_date != today:
            self.daily_regime_changes = 0
            self.last_regime_check_date = today

        if self.last_regime is not None and self.last_regime != new_regime:
            self.daily_regime_changes += 1

        self.last_regime = new_regime

    def get_consecutive_losses(self) -> int:
        """Get current consecutive loss count"""
        count = 0
        for result in reversed(self.recent_results):
            if not result:
                count += 1
            else:
                break
        return count

    def force_mode(self, mode: TradeMode, duration_hours: int = 4):
        """Force a specific mode for a duration

        Args:
            mode: TradeMode to force
            duration_hours: How long to maintain force (default 4 hours)
        """
        self._forced_mode = mode
        self._force_expires = datetime.now() + timedelta(hours=duration_hours)
        self.current_mode = mode
        self.mode_reason = f"Forced {mode.value} until {self._force_expires.strftime('%H:%M')}"
        logger.info(f"Mode forced to {mode.value} for {duration_hours} hours")

    def clear_forced_mode(self):
        """Clear any forced mode"""
        self._forced_mode = None
        self._force_expires = None
        logger.info("Forced mode cleared")

    def evaluate_mode(
        self,
        current_time: datetime,
        current_balance: float,
        atr_pips: float = 0.0
    ) -> TradeMode:
        """Evaluate and return current trade mode

        Uses two-tier system:
        - SIGNAL_ONLY: Full stop for severe conditions
        - RECOVERY: Reduced lot size for moderate conditions
        - AUTO: Normal trading

        Args:
            current_time: Current datetime
            current_balance: Current account balance
            atr_pips: Current ATR in pips

        Returns:
            TradeMode (AUTO, RECOVERY, or SIGNAL_ONLY)
        """
        # Check for forced mode
        if self._forced_mode is not None:
            if self._force_expires and datetime.now() >= self._force_expires:
                # Force expired
                self.clear_forced_mode()
            else:
                # Still in forced mode
                return self._forced_mode

        monitoring_reasons = []  # Reasons for MONITORING (full pause)
        signal_reasons = []  # Reasons for SIGNAL_ONLY
        recovery_reasons = []  # Reasons for RECOVERY

        # Check 1: December - Two tier system
        # Dec 15-31: MONITORING (full pause, no signals)
        # Dec 1-14: SIGNAL_ONLY (send signals, no execute)
        if current_time.month == 12:
            if current_time.day >= self.config.december_monitoring_start:
                monitoring_reasons.append(f"December {current_time.day} (holiday pause - monitoring only)")
            elif current_time.day >= self.config.december_signal_only_start:
                signal_reasons.append(f"December {current_time.day} (holiday period - signals only)")

        # Check 1b: Summer months (SIGNAL_ONLY)
        if current_time.month in self.config.summer_months:
            signal_reasons.append(f"Summer month (low liquidity)")

        # Check 2: Daily loss (two-tier)
        if self.daily_stats.starting_balance > 0:
            daily_loss_pct = abs(min(0, self.daily_stats.pnl)) / self.daily_stats.starting_balance * 100
            if daily_loss_pct >= self.config.daily_loss_limit_pct:
                signal_reasons.append(f"Daily loss {daily_loss_pct:.1f}% >= {self.config.daily_loss_limit_pct}%")
            elif daily_loss_pct >= self.config.daily_loss_recovery_pct:
                recovery_reasons.append(f"Daily loss {daily_loss_pct:.1f}% >= {self.config.daily_loss_recovery_pct}%")

        # Check 3: Weekly loss (two-tier)
        if self.weekly_start_balance > 0:
            weekly_loss_pct = abs(min(0, self.weekly_pnl)) / self.weekly_start_balance * 100
            if weekly_loss_pct >= self.config.weekly_loss_limit_pct:
                signal_reasons.append(f"Weekly loss {weekly_loss_pct:.1f}% >= {self.config.weekly_loss_limit_pct}%")
            elif weekly_loss_pct >= self.config.weekly_loss_recovery_pct:
                recovery_reasons.append(f"Weekly loss {weekly_loss_pct:.1f}% >= {self.config.weekly_loss_recovery_pct}%")

        # Check 4: Volatility (two-tier)
        if atr_pips >= self.config.high_volatility_atr_pips:
            signal_reasons.append(f"High volatility ATR {atr_pips:.1f} >= {self.config.high_volatility_atr_pips} pips")
        elif atr_pips >= self.config.medium_volatility_atr_pips:
            recovery_reasons.append(f"Medium volatility ATR {atr_pips:.1f} >= {self.config.medium_volatility_atr_pips} pips")

        # Check 5: Consecutive losses (two-tier)
        consecutive_losses = self.get_consecutive_losses()
        if consecutive_losses >= self.config.max_consecutive_losses:
            signal_reasons.append(f"{consecutive_losses} consecutive losses")
        elif consecutive_losses >= self.config.consecutive_losses_recovery:
            recovery_reasons.append(f"{consecutive_losses} consecutive losses (recovery)")

        # Check 6: Regime instability (two-tier)
        if self.daily_regime_changes >= self.config.max_regime_changes_per_day:
            signal_reasons.append(f"{self.daily_regime_changes} regime changes today")
        elif self.daily_regime_changes >= self.config.regime_changes_recovery:
            recovery_reasons.append(f"{self.daily_regime_changes} regime changes (recovery)")

        # Determine mode (Priority: MONITORING > SIGNAL_ONLY > RECOVERY > AUTO)
        if monitoring_reasons:
            self.current_mode = TradeMode.MONITORING
            self.mode_reason = "; ".join(monitoring_reasons)
        elif signal_reasons:
            self.current_mode = TradeMode.SIGNAL_ONLY
            self.mode_reason = "; ".join(signal_reasons)
        elif recovery_reasons:
            self.current_mode = TradeMode.RECOVERY
            self.mode_reason = "; ".join(recovery_reasons)
        else:
            self.current_mode = TradeMode.AUTO
            self.mode_reason = "Normal trading conditions"

        return self.current_mode

    def should_auto_trade(self) -> bool:
        """Check if auto trading is enabled (AUTO or RECOVERY mode)"""
        return self.current_mode in (TradeMode.AUTO, TradeMode.RECOVERY)

    def is_recovery_mode(self) -> bool:
        """Check if in recovery mode (reduced lot size)"""
        return self.current_mode == TradeMode.RECOVERY

    def is_monitoring_mode(self) -> bool:
        """Check if in monitoring mode (full pause, no signals)"""
        return self.current_mode == TradeMode.MONITORING

    def get_lot_multiplier(self) -> float:
        """Get lot size multiplier based on mode

        Returns:
            1.0 for AUTO, recovery_lot_multiplier for RECOVERY, 0.0 for others
        """
        if self.current_mode == TradeMode.AUTO:
            return 1.0
        elif self.current_mode == TradeMode.RECOVERY:
            return self.config.recovery_lot_multiplier
        else:
            return 0.0

    def should_send_signal(self) -> bool:
        """Check if signals should be sent

        Returns:
            True for AUTO, RECOVERY, SIGNAL_ONLY
            False for MONITORING (full pause)
        """
        return self.current_mode != TradeMode.MONITORING

    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "mode": self.current_mode.value,
            "reason": self.mode_reason,
            "lot_multiplier": self.get_lot_multiplier(),
            "daily_pnl": self.daily_stats.pnl,
            "daily_trades": self.daily_stats.trades,
            "daily_win_rate": self.daily_stats.wins / self.daily_stats.trades * 100 if self.daily_stats.trades > 0 else 0,
            "consecutive_losses": self.get_consecutive_losses(),
            "regime_changes_today": self.daily_regime_changes,
            "weekly_pnl": self.weekly_pnl
        }

    def format_mode_message(self) -> str:
        """Format mode status for Telegram"""
        status = self.get_status()

        if self.current_mode == TradeMode.AUTO:
            emoji = "🟢"
            mode_text = "AUTO TRADE"
        elif self.current_mode == TradeMode.RECOVERY:
            emoji = "🟠"
            mode_text = f"RECOVERY ({int(self.config.recovery_lot_multiplier * 100)}% lot)"
        elif self.current_mode == TradeMode.MONITORING:
            emoji = "🔴"
            mode_text = "MONITORING ONLY (paused)"
        else:
            emoji = "🟡"
            mode_text = "SIGNAL ONLY"

        msg = [
            f"{emoji} <b>Mode: {mode_text}</b>",
            f"Reason: {self.mode_reason}",
            "",
            f"Daily P/L: ${status['daily_pnl']:+.2f}",
            f"Daily Trades: {status['daily_trades']} ({status['daily_win_rate']:.0f}% win)",
            f"Weekly P/L: ${status['weekly_pnl']:+.2f}",
        ]

        if self.current_mode == TradeMode.RECOVERY:
            msg.append(f"📉 Lot Multiplier: {status['lot_multiplier']:.0%}")

        if self.current_mode == TradeMode.MONITORING:
            msg.append(f"🛑 Trading paused - monitoring market only")

        if status['consecutive_losses'] > 0:
            msg.append(f"⚠️ Consecutive Losses: {status['consecutive_losses']}")

        return "\n".join(msg)
