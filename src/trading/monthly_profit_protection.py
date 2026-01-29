"""Monthly Profit Protection System
====================================

Ensures every month is profitable by implementing:
1. Daily loss limits
2. Monthly loss thresholds
3. Consecutive loss management
4. Dynamic quality filters
5. Progressive position reduction

The goal is ZERO losing months - better to miss opportunities
than to have a negative month.

Author: SURIOTA Team
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger


class ProtectionLevel(Enum):
    """Protection level based on monthly P/L"""
    NORMAL = "normal"           # All good, trade normally
    CAUTIOUS = "cautious"       # Small loss, be careful
    DEFENSIVE = "defensive"     # Larger loss, reduce significantly
    LOCKDOWN = "lockdown"       # Near limit, minimal trading
    STOPPED = "stopped"         # Hit limit, no more trading this month


@dataclass
class ProtectionParams:
    """Protection parameters for current state"""
    level: ProtectionLevel
    can_trade: bool
    lot_multiplier: float           # Multiply calculated lot by this
    min_quality_score: float        # Minimum signal quality required
    max_trades_per_day: int         # Daily trade limit
    require_full_confirmation: bool # Require all confirmations
    reason: str


@dataclass
class MonthlyStats:
    """Track monthly statistics"""
    month: int
    year: int
    start_balance: float
    current_balance: float
    peak_balance: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    consecutive_losses: int
    last_trade_time: Optional[datetime]
    daily_trades: Dict[str, int]  # date_str -> count
    daily_pnl: Dict[str, float]   # date_str -> pnl

    @property
    def pnl(self) -> float:
        return self.current_balance - self.start_balance

    @property
    def pnl_percent(self) -> float:
        return (self.pnl / self.start_balance) * 100 if self.start_balance > 0 else 0

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0
        return (self.winning_trades / self.trades_count) * 100

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_balance <= 0:
            return 0
        return (self.peak_balance - self.current_balance) / self.peak_balance * 100


class MonthlyProfitProtection:
    """Monthly Profit Protection System

    Key principles:
    1. Better to miss a trade than to lose money
    2. Preserve capital first, grow capital second
    3. Every month MUST be profitable (or flat)
    """

    def __init__(
        self,
        # Monthly loss thresholds
        cautious_threshold: float = -0.5,    # -0.5% -> cautious mode
        defensive_threshold: float = -1.0,   # -1.0% -> defensive mode
        lockdown_threshold: float = -1.5,    # -1.5% -> lockdown mode
        stop_threshold: float = -2.0,        # -2.0% -> stop trading

        # Daily limits
        daily_loss_limit: float = 1.0,       # 1% daily loss limit
        max_trades_per_day_normal: int = 3,
        max_trades_per_day_cautious: int = 2,
        max_trades_per_day_defensive: int = 1,

        # Consecutive loss handling
        consecutive_loss_pause: int = 2,     # Pause after 2 consecutive losses
        consecutive_loss_cooldown_hours: int = 4,

        # Quality filters
        min_quality_normal: float = 60.0,
        min_quality_cautious: float = 70.0,
        min_quality_defensive: float = 80.0,
        min_quality_lockdown: float = 90.0,

        # Lot multipliers
        lot_mult_cautious: float = 0.7,
        lot_mult_defensive: float = 0.4,
        lot_mult_lockdown: float = 0.2,
    ):
        # Monthly thresholds
        self.cautious_threshold = cautious_threshold
        self.defensive_threshold = defensive_threshold
        self.lockdown_threshold = lockdown_threshold
        self.stop_threshold = stop_threshold

        # Daily limits
        self.daily_loss_limit = daily_loss_limit
        self.max_trades_normal = max_trades_per_day_normal
        self.max_trades_cautious = max_trades_per_day_cautious
        self.max_trades_defensive = max_trades_per_day_defensive

        # Consecutive loss
        self.consecutive_loss_pause = consecutive_loss_pause
        self.consecutive_loss_cooldown = timedelta(hours=consecutive_loss_cooldown_hours)

        # Quality filters
        self.min_quality_normal = min_quality_normal
        self.min_quality_cautious = min_quality_cautious
        self.min_quality_defensive = min_quality_defensive
        self.min_quality_lockdown = min_quality_lockdown

        # Lot multipliers
        self.lot_mult_cautious = lot_mult_cautious
        self.lot_mult_defensive = lot_mult_defensive
        self.lot_mult_lockdown = lot_mult_lockdown

        # State
        self._stats: Optional[MonthlyStats] = None
        self._last_loss_time: Optional[datetime] = None

    def start_month(self, year: int, month: int, balance: float):
        """Initialize tracking for a new month"""
        self._stats = MonthlyStats(
            month=month,
            year=year,
            start_balance=balance,
            current_balance=balance,
            peak_balance=balance,
            trades_count=0,
            winning_trades=0,
            losing_trades=0,
            consecutive_losses=0,
            last_trade_time=None,
            daily_trades={},
            daily_pnl={}
        )
        self._last_loss_time = None
        logger.debug(f"Started month tracking: {year}-{month:02d}, balance=${balance:.2f}")

    def update_balance(self, new_balance: float):
        """Update current balance"""
        if self._stats is None:
            return
        self._stats.current_balance = new_balance
        if new_balance > self._stats.peak_balance:
            self._stats.peak_balance = new_balance

    def record_trade(self, time: datetime, pnl: float, is_win: bool):
        """Record trade result"""
        if self._stats is None:
            return

        date_str = time.strftime("%Y-%m-%d")

        # Update counts
        self._stats.trades_count += 1
        if is_win:
            self._stats.winning_trades += 1
            self._stats.consecutive_losses = 0
        else:
            self._stats.losing_trades += 1
            self._stats.consecutive_losses += 1
            self._last_loss_time = time

        self._stats.last_trade_time = time

        # Update daily tracking
        self._stats.daily_trades[date_str] = self._stats.daily_trades.get(date_str, 0) + 1
        self._stats.daily_pnl[date_str] = self._stats.daily_pnl.get(date_str, 0) + pnl

        logger.debug(f"Recorded trade: pnl=${pnl:.2f}, win={is_win}, "
                    f"month_pnl={self._stats.pnl_percent:.2f}%, "
                    f"consec_loss={self._stats.consecutive_losses}")

    def _get_daily_pnl_percent(self, date: datetime) -> float:
        """Get today's P/L as percentage of starting balance"""
        if self._stats is None:
            return 0
        date_str = date.strftime("%Y-%m-%d")
        daily_pnl = self._stats.daily_pnl.get(date_str, 0)
        return (daily_pnl / self._stats.start_balance) * 100 if self._stats.start_balance > 0 else 0

    def _get_daily_trade_count(self, date: datetime) -> int:
        """Get today's trade count"""
        if self._stats is None:
            return 0
        date_str = date.strftime("%Y-%m-%d")
        return self._stats.daily_trades.get(date_str, 0)

    def _determine_level(self) -> ProtectionLevel:
        """Determine current protection level"""
        if self._stats is None:
            return ProtectionLevel.NORMAL

        pnl_pct = self._stats.pnl_percent

        if pnl_pct <= self.stop_threshold:
            return ProtectionLevel.STOPPED
        elif pnl_pct <= self.lockdown_threshold:
            return ProtectionLevel.LOCKDOWN
        elif pnl_pct <= self.defensive_threshold:
            return ProtectionLevel.DEFENSIVE
        elif pnl_pct <= self.cautious_threshold:
            return ProtectionLevel.CAUTIOUS
        else:
            return ProtectionLevel.NORMAL

    def get_protection_params(self, current_time: datetime) -> ProtectionParams:
        """Get current protection parameters

        Returns parameters that control:
        - Whether trading is allowed
        - Lot size multiplier
        - Minimum quality score
        - Daily trade limits
        """
        if self._stats is None:
            # No stats - allow normal trading
            return ProtectionParams(
                level=ProtectionLevel.NORMAL,
                can_trade=True,
                lot_multiplier=1.0,
                min_quality_score=self.min_quality_normal,
                max_trades_per_day=self.max_trades_normal,
                require_full_confirmation=True,
                reason="NoStats"
            )

        reasons = []

        # 1. Check protection level from monthly P/L
        level = self._determine_level()

        if level == ProtectionLevel.STOPPED:
            return ProtectionParams(
                level=level,
                can_trade=False,
                lot_multiplier=0,
                min_quality_score=100,  # Impossible
                max_trades_per_day=0,
                require_full_confirmation=True,
                reason=f"STOPPED: Monthly loss {self._stats.pnl_percent:.2f}% exceeded limit"
            )

        # 2. Check daily loss limit
        daily_pnl_pct = self._get_daily_pnl_percent(current_time)
        if daily_pnl_pct <= -self.daily_loss_limit:
            return ProtectionParams(
                level=ProtectionLevel.STOPPED,
                can_trade=False,
                lot_multiplier=0,
                min_quality_score=100,
                max_trades_per_day=0,
                require_full_confirmation=True,
                reason=f"DailyLimit: {daily_pnl_pct:.2f}% loss today"
            )

        # 3. Check consecutive loss cooldown
        if self._stats.consecutive_losses >= self.consecutive_loss_pause:
            if self._last_loss_time:
                time_since_loss = current_time - self._last_loss_time
                if time_since_loss < self.consecutive_loss_cooldown:
                    remaining = self.consecutive_loss_cooldown - time_since_loss
                    return ProtectionParams(
                        level=ProtectionLevel.STOPPED,
                        can_trade=False,
                        lot_multiplier=0,
                        min_quality_score=100,
                        max_trades_per_day=0,
                        require_full_confirmation=True,
                        reason=f"Cooldown: {self._stats.consecutive_losses} losses, "
                               f"wait {remaining.total_seconds()/3600:.1f}h"
                    )
                else:
                    # Cooldown passed, reset consecutive losses
                    self._stats.consecutive_losses = 0

        # 4. Check daily trade limit
        daily_trades = self._get_daily_trade_count(current_time)

        # Set parameters based on level
        if level == ProtectionLevel.LOCKDOWN:
            lot_mult = self.lot_mult_lockdown
            min_quality = self.min_quality_lockdown
            max_trades = 1  # Only 1 trade allowed
            reasons.append(f"LOCKDOWN({self._stats.pnl_percent:.2f}%)")
        elif level == ProtectionLevel.DEFENSIVE:
            lot_mult = self.lot_mult_defensive
            min_quality = self.min_quality_defensive
            max_trades = self.max_trades_defensive
            reasons.append(f"Defensive({self._stats.pnl_percent:.2f}%)")
        elif level == ProtectionLevel.CAUTIOUS:
            lot_mult = self.lot_mult_cautious
            min_quality = self.min_quality_cautious
            max_trades = self.max_trades_cautious
            reasons.append(f"Cautious({self._stats.pnl_percent:.2f}%)")
        else:
            lot_mult = 1.0
            min_quality = self.min_quality_normal
            max_trades = self.max_trades_normal

        # Check if daily trade limit exceeded
        if daily_trades >= max_trades:
            return ProtectionParams(
                level=level,
                can_trade=False,
                lot_multiplier=0,
                min_quality_score=min_quality,
                max_trades_per_day=max_trades,
                require_full_confirmation=True,
                reason=f"DailyTradeLimit: {daily_trades}/{max_trades}"
            )

        # 5. Additional reduction for consecutive losses (even if not at pause threshold)
        if self._stats.consecutive_losses > 0:
            loss_penalty = 0.9 ** self._stats.consecutive_losses  # 10% reduction per loss
            lot_mult *= loss_penalty
            reasons.append(f"ConsecLoss({self._stats.consecutive_losses})")

        # 6. Additional reduction if daily P/L is negative
        if daily_pnl_pct < 0:
            daily_penalty = max(0.5, 1 + (daily_pnl_pct / 2))  # Scale down based on daily loss
            lot_mult *= daily_penalty
            reasons.append(f"DailyLoss({daily_pnl_pct:.2f}%)")

        # Enforce minimum lot multiplier
        lot_mult = max(0.1, lot_mult)

        reason_str = " | ".join(reasons) if reasons else "Normal"

        return ProtectionParams(
            level=level,
            can_trade=True,
            lot_multiplier=round(lot_mult, 2),
            min_quality_score=min_quality,
            max_trades_per_day=max_trades,
            require_full_confirmation=True,  # Always require full confirmation for safety
            reason=reason_str
        )

    def get_stats(self) -> Optional[Dict]:
        """Get current monthly statistics"""
        if self._stats is None:
            return None
        return {
            'month': f"{self._stats.year}-{self._stats.month:02d}",
            'start_balance': self._stats.start_balance,
            'current_balance': self._stats.current_balance,
            'pnl': self._stats.pnl,
            'pnl_percent': self._stats.pnl_percent,
            'trades': self._stats.trades_count,
            'win_rate': self._stats.win_rate,
            'consecutive_losses': self._stats.consecutive_losses,
            'protection_level': self._determine_level().value
        }

    def end_month(self) -> Optional[Dict]:
        """End month tracking and return summary"""
        stats = self.get_stats()
        self._stats = None
        return stats


class IntraMonthRecovery:
    """Intra-month recovery strategies

    When in losing position mid-month, try to recover with:
    1. Higher quality setups only
    2. Smaller position sizes
    3. Tighter profit targets (1:1 R:R instead of waiting for TP3)
    """

    def __init__(self, protection: MonthlyProfitProtection):
        self.protection = protection

    def get_recovery_params(
        self,
        current_time: datetime,
        monthly_pnl_percent: float
    ) -> Dict:
        """Get recovery parameters based on how far behind we are"""
        if monthly_pnl_percent >= 0:
            # Not in loss, no recovery needed
            return {
                'recovery_mode': False,
                'tp_mode': 'normal',  # Use normal TP1/TP2/TP3
                'early_exit': False
            }

        # In loss - recovery mode
        loss_severity = abs(monthly_pnl_percent)

        if loss_severity < 0.5:
            # Small loss - conservative recovery
            return {
                'recovery_mode': True,
                'tp_mode': 'conservative',  # Close 70% at TP1
                'early_exit': False
            }
        elif loss_severity < 1.0:
            # Moderate loss - lock in profits quickly
            return {
                'recovery_mode': True,
                'tp_mode': 'quick',  # Close all at TP1
                'early_exit': True
            }
        else:
            # Significant loss - ultra conservative
            return {
                'recovery_mode': True,
                'tp_mode': 'immediate',  # Close at 0.8:1 R:R
                'early_exit': True
            }
