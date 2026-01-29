"""Risk Manager - Dynamic Position Sizing
=========================================

Provides position sizing based on:
- Zone quality score
- Account balance
- Daily risk limits

Position sizing rules:
- High quality (>80): 1.5% risk
- Medium (60-80): 1.0% risk
- Low (<60): 0.5% risk

Author: SURIOTA Team
"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone
from loguru import logger


@dataclass
class RiskParams:
    """Risk calculation parameters"""
    lot_size: float
    risk_amount: float
    risk_percent: float
    sl_pips: float
    position_value: float


class RiskManager:
    """Dynamic position sizing based on zone quality

    Zero Losing Months Configuration:
    - Caps max lot at 0.5
    - Caps max loss per trade at 0.8% of balance
    - Stricter daily loss limits
    - December special handling
    """

    def __init__(
        self,
        high_quality_threshold: float = 80.0,
        high_quality_risk: float = 0.015,
        medium_quality_threshold: float = 60.0,
        medium_quality_risk: float = 0.01,
        low_quality_risk: float = 0.005,
        daily_profit_target: float = 100.0,
        daily_loss_limit: float = 80.0,  # Tighter: 0.8% of $10K = $80
        max_open_positions: int = 1,
        pip_value: float = 10.0,
        # Position size limits - Zero Losing Months config
        min_lot_size: float = 0.01,
        max_lot_size: float = 0.5,  # Conservative for zero-loss strategy
        min_sl_pips: float = 15.0,  # Minimum 15 pips SL
        max_sl_pips: float = 10.0,  # ZERO LOSING MONTHS: Maximum 10 pips SL (was 50)
        # ZERO LOSING MONTHS: Max loss per trade protection
        max_loss_per_trade_pct: float = 0.1,  # ZERO LOSING MONTHS: Maximum 0.1% loss per trade (was 0.8)
        monthly_loss_stop_pct: float = 2.0,  # Stop trading if monthly loss > 2%
    ):
        """Initialize Risk Manager

        Args:
            high_quality_threshold: Quality >= this for high risk
            high_quality_risk: Risk % for high quality (1.5%)
            medium_quality_threshold: Quality >= this for medium risk
            medium_quality_risk: Risk % for medium quality (1.0%)
            low_quality_risk: Risk % for low quality (0.5%)
            daily_profit_target: Stop trading after this profit
            daily_loss_limit: Stop trading after this loss
            max_open_positions: Max concurrent positions
            pip_value: Value per pip per lot
        """
        self.high_quality_threshold = high_quality_threshold
        self.high_quality_risk = high_quality_risk
        self.medium_quality_threshold = medium_quality_threshold
        self.medium_quality_risk = medium_quality_risk
        self.low_quality_risk = low_quality_risk
        self.daily_profit_target = daily_profit_target
        self.daily_loss_limit = daily_loss_limit
        self.max_open_positions = max_open_positions
        self.pip_value = pip_value

        # Position size limits - Zero Losing Months configuration
        self.min_lot_size = min_lot_size
        self.max_lot_size = max_lot_size
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

        # NEW: Max loss per trade protection (Zero Losing Months)
        self.max_loss_per_trade_pct = max_loss_per_trade_pct
        self.monthly_loss_stop_pct = monthly_loss_stop_pct

        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._current_date = datetime.now(timezone.utc).date()
        self._open_positions = 0

        # Monthly tracking (Zero Losing Months)
        self._monthly_pnl = 0.0
        self._monthly_trades = 0
        self._current_month = datetime.now(timezone.utc).month
        self._month_start_balance = 0.0

        # Drawdown tracking
        self._peak_balance = 0.0
        self._current_drawdown = 0.0

        # Drawdown-based position sizing thresholds (stricter for zero-loss)
        self.drawdown_reduction_1 = 0.08  # 8% drawdown -> reduce 25%
        self.drawdown_reduction_2 = 0.15  # 15% drawdown -> reduce 50%
        self.drawdown_reduction_3 = 0.25  # 25% drawdown -> reduce 75%

    def update_drawdown(self, current_balance: float):
        """Update drawdown tracking

        Args:
            current_balance: Current account balance
        """
        # Update peak balance
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        # Calculate current drawdown
        if self._peak_balance > 0:
            self._current_drawdown = (self._peak_balance - current_balance) / self._peak_balance
        else:
            self._current_drawdown = 0.0

    def get_drawdown_multiplier(self) -> float:
        """Get position size multiplier based on current drawdown

        Returns:
            Multiplier (0.25 - 1.0)
        """
        if self._current_drawdown >= self.drawdown_reduction_3:
            return 0.25  # 75% reduction at 30%+ drawdown
        elif self._current_drawdown >= self.drawdown_reduction_2:
            return 0.50  # 50% reduction at 20%+ drawdown
        elif self._current_drawdown >= self.drawdown_reduction_1:
            return 0.75  # 25% reduction at 10%+ drawdown
        else:
            return 1.0   # No reduction

    def get_risk_percent(self, quality_score: float) -> float:
        """Get risk percentage based on quality score

        Args:
            quality_score: Signal quality (0-100)

        Returns:
            Risk percentage (0.005 - 0.015)
        """
        if quality_score >= self.high_quality_threshold:
            return self.high_quality_risk
        elif quality_score >= self.medium_quality_threshold:
            return self.medium_quality_risk
        else:
            return self.low_quality_risk

    def calculate_lot_size(
        self,
        account_balance: float,
        quality_score: float,
        sl_pips: float
    ) -> RiskParams:
        """Calculate position size based on quality and risk

        Lot size is calculated automatically based on:
        - Account balance
        - Quality-based risk percentage
        - Stop loss distance
        - Current drawdown (reduces size during drawdown)

        Args:
            account_balance: Account balance in USD
            quality_score: Signal quality (0-100)
            sl_pips: Stop loss distance in pips

        Returns:
            RiskParams with lot size and risk details
        """
        # Update drawdown tracking
        self.update_drawdown(account_balance)

        # Get risk percent based on quality
        risk_percent = self.get_risk_percent(quality_score)

        # Apply drawdown-based reduction
        drawdown_multiplier = self.get_drawdown_multiplier()
        adjusted_risk_percent = risk_percent * drawdown_multiplier

        if drawdown_multiplier < 1.0:
            logger.debug(f"Drawdown {self._current_drawdown:.1%}: reducing risk {risk_percent:.1%} -> {adjusted_risk_percent:.1%}")

        # Calculate risk amount
        risk_amount = account_balance * adjusted_risk_percent

        # Calculate lot size automatically
        # Formula: Lot Size = Risk Amount / (SL Pips * Pip Value)

        # Enforce minimum SL to prevent over-leveraging
        effective_sl = max(sl_pips, self.min_sl_pips) if sl_pips > 0 else self.min_sl_pips

        # Cap SL at maximum to prevent overly wide stops
        effective_sl = min(effective_sl, self.max_sl_pips)

        if sl_pips < self.min_sl_pips:
            logger.debug(f"SL {sl_pips:.1f} pips < min {self.min_sl_pips:.1f}, using minimum")
        if sl_pips > self.max_sl_pips:
            logger.debug(f"SL {sl_pips:.1f} pips > max {self.max_sl_pips:.1f}, using maximum")

        lot_size = risk_amount / (effective_sl * self.pip_value)

        # Round to 2 decimal places
        lot_size = round(lot_size, 2)

        # Enforce minimum lot size (broker minimum)
        if lot_size < self.min_lot_size:
            lot_size = self.min_lot_size
            logger.debug(f"Lot size below minimum, using {self.min_lot_size}")

        # Enforce maximum lot size (prevents single large losses)
        if lot_size > self.max_lot_size:
            logger.warning(f"Lot size {lot_size} exceeds max {self.max_lot_size}, capping")
            lot_size = self.max_lot_size

        # Calculate position value
        position_value = lot_size * 100000  # Standard lot = 100k units

        return RiskParams(
            lot_size=lot_size,
            risk_amount=lot_size * sl_pips * self.pip_value,
            risk_percent=adjusted_risk_percent,
            sl_pips=sl_pips,
            position_value=position_value
        )

    def calculate_lot_with_loss_cap(
        self,
        account_balance: float,
        sl_pips: float,
        max_loss_pct: float = None
    ) -> float:
        """Calculate maximum lot size that respects max loss per trade

        Zero Losing Months Strategy:
        Ensures no single trade can lose more than max_loss_pct of balance.

        Args:
            account_balance: Current account balance
            sl_pips: Stop loss distance in pips
            max_loss_pct: Max loss as % of balance (default: self.max_loss_per_trade_pct)

        Returns:
            Maximum lot size that respects the loss cap
        """
        if max_loss_pct is None:
            max_loss_pct = self.max_loss_per_trade_pct

        # Calculate max loss in dollars
        max_loss_dollars = account_balance * max_loss_pct / 100

        # Enforce minimum SL
        effective_sl = max(sl_pips, self.min_sl_pips)
        effective_sl = min(effective_sl, self.max_sl_pips)

        # Calculate lot size: lot = max_loss / (sl_pips * pip_value)
        calculated_lot = max_loss_dollars / (effective_sl * self.pip_value)

        # Round and enforce limits
        calculated_lot = round(calculated_lot, 2)
        calculated_lot = max(self.min_lot_size, calculated_lot)
        calculated_lot = min(self.max_lot_size, calculated_lot)

        return calculated_lot

    def check_monthly_limit(self) -> Tuple[bool, str]:
        """Check if monthly loss limit has been reached

        Zero Losing Months Strategy:
        Stop trading if monthly loss exceeds threshold.

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        self._check_monthly_reset()

        if self._month_start_balance <= 0:
            return True, "OK"

        monthly_loss_pct = (self._monthly_pnl / self._month_start_balance) * 100

        if monthly_loss_pct <= -self.monthly_loss_stop_pct:
            return False, f"Monthly loss limit: {monthly_loss_pct:.2f}% exceeds -{self.monthly_loss_stop_pct}%"

        return True, "OK"

    def should_skip_december(self, current_time: datetime = None) -> Tuple[bool, str]:
        """Check if trading should be skipped (December anomaly)

        Zero Losing Months Strategy:
        December historically has very poor performance (12.5% win rate).
        Skip trading entirely to protect monthly P/L.

        Args:
            current_time: Current time (default: now)

        Returns:
            Tuple of (should_skip: bool, reason: str)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        if current_time.month == 12:
            return True, "December trading disabled (anomaly month)"

        return False, "OK"

    def _check_monthly_reset(self):
        """Reset monthly stats at start of new month"""
        today = datetime.now(timezone.utc)
        if today.month != self._current_month:
            logger.info(f"Monthly reset: month {self._current_month} -> {today.month}")
            self._monthly_pnl = 0.0
            self._monthly_trades = 0
            self._current_month = today.month

    def start_month(self, balance: float):
        """Initialize monthly tracking with starting balance

        Args:
            balance: Starting balance for the month
        """
        self._month_start_balance = balance
        self._monthly_pnl = 0.0
        self._monthly_trades = 0
        self._current_month = datetime.now(timezone.utc).month
        logger.info(f"Month started with balance ${balance:,.2f}")

    def register_trade_close_monthly(self, profit: float):
        """Register trade result for monthly tracking

        Args:
            profit: Trade profit/loss in USD
        """
        self._monthly_pnl += profit
        self._monthly_trades += 1
        logger.debug(f"Monthly P/L: ${self._monthly_pnl:+.2f} ({self._monthly_trades} trades)")

    def get_monthly_stats(self) -> Dict:
        """Get monthly trading statistics"""
        self._check_monthly_reset()

        monthly_loss_pct = 0.0
        if self._month_start_balance > 0:
            monthly_loss_pct = (self._monthly_pnl / self._month_start_balance) * 100

        return {
            "month": self._current_month,
            "trades": self._monthly_trades,
            "pnl": self._monthly_pnl,
            "pnl_percent": monthly_loss_pct,
            "start_balance": self._month_start_balance,
            "limit_reached": monthly_loss_pct <= -self.monthly_loss_stop_pct,
        }

    def can_open_trade(self, current_time: datetime = None) -> Tuple[bool, str]:
        """Check if new trade can be opened

        Zero Losing Months Strategy includes:
        - December skip
        - Monthly loss limit
        - Daily loss limit (stricter)

        Args:
            current_time: Current time for December check

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        # Check December skip (Zero Losing Months)
        skip_dec, reason = self.should_skip_december(current_time)
        if skip_dec:
            return False, reason

        # Check monthly limit (Zero Losing Months)
        can_monthly, reason = self.check_monthly_limit()
        if not can_monthly:
            return False, reason

        # Check date reset
        self._check_daily_reset()

        # Check position limit
        if self._open_positions >= self.max_open_positions:
            return False, f"Max positions reached ({self._open_positions}/{self.max_open_positions})"

        # Check daily profit target
        if self._daily_pnl >= self.daily_profit_target:
            return False, f"Daily profit target reached (${self._daily_pnl:+.2f})"

        # Check daily loss limit (stricter for zero-loss)
        if self._daily_pnl <= -self.daily_loss_limit:
            return False, f"Daily loss limit reached (${self._daily_pnl:+.2f})"

        return True, "OK"

    def _check_daily_reset(self):
        """Reset daily stats at midnight UTC"""
        today = datetime.now(timezone.utc).date()
        if today != self._current_date:
            logger.info(f"Daily reset: {self._current_date} -> {today}")
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._current_date = today

    def register_trade_open(self):
        """Register new trade opened"""
        self._open_positions += 1
        self._daily_trades += 1
        logger.debug(f"Trade opened: {self._open_positions} positions open")

    def register_trade_close(self, profit: float):
        """Register trade closed

        Updates both daily and monthly tracking.

        Args:
            profit: Trade profit/loss in USD
        """
        self._open_positions = max(0, self._open_positions - 1)
        self._daily_pnl += profit

        # Also track monthly (Zero Losing Months)
        self.register_trade_close_monthly(profit)

        logger.debug(f"Trade closed: P/L ${profit:+.2f}, Daily ${self._daily_pnl:+.2f}, Monthly ${self._monthly_pnl:+.2f}")

    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics"""
        self._check_daily_reset()

        return {
            "date": str(self._current_date),
            "trades": self._daily_trades,
            "pnl": self._daily_pnl,
            "open_positions": self._open_positions,
            "profit_target": self.daily_profit_target,
            "loss_limit": self.daily_loss_limit,
            "target_reached": self._daily_pnl >= self.daily_profit_target,
            "limit_reached": self._daily_pnl <= -self.daily_loss_limit,
        }

    def check_risk_warnings(self) -> Dict:
        """Check for risk warning conditions (approaching limits)

        Returns warnings when:
        - Daily loss is at 80% of limit
        - Drawdown exceeds threshold
        - Monthly loss approaching limit

        Returns:
            Dict with warning flags and messages
        """
        self._check_daily_reset()
        self._check_monthly_reset()

        warnings = {
            "has_warning": False,
            "messages": [],
            "daily_loss_warning": False,
            "drawdown_warning": False,
            "monthly_warning": False,
            "risk_reduced": False,
        }

        # Daily loss warning at 80% of limit
        daily_warning_threshold = self.daily_loss_limit * 0.8
        if self._daily_pnl <= -daily_warning_threshold:
            pct_used = abs(self._daily_pnl / self.daily_loss_limit) * 100
            warnings["has_warning"] = True
            warnings["daily_loss_warning"] = True
            warnings["messages"].append(
                f"⚠️ Daily loss at {pct_used:.0f}% of limit (${self._daily_pnl:+.2f} / -${self.daily_loss_limit:.0f})"
            )

        # Drawdown warning
        dd_multiplier = self.get_drawdown_multiplier()
        if dd_multiplier < 1.0:
            warnings["has_warning"] = True
            warnings["drawdown_warning"] = True
            warnings["risk_reduced"] = True
            reduction_pct = (1.0 - dd_multiplier) * 100
            warnings["messages"].append(
                f"⚠️ Drawdown at {self._current_drawdown:.1%} - Risk reduced by {reduction_pct:.0f}%"
            )

        # Monthly loss warning at 70% of limit
        if self._month_start_balance > 0:
            monthly_loss_pct = abs(self._monthly_pnl / self._month_start_balance) * 100
            monthly_warning_threshold = self.monthly_loss_stop_pct * 0.7
            if self._monthly_pnl < 0 and monthly_loss_pct >= monthly_warning_threshold:
                warnings["has_warning"] = True
                warnings["monthly_warning"] = True
                warnings["messages"].append(
                    f"⚠️ Monthly loss at {monthly_loss_pct:.1f}% (limit: {self.monthly_loss_stop_pct}%)"
                )

        return warnings

    def validate_risk(
        self,
        account_balance: float,
        lot_size: float,
        sl_pips: float
    ) -> Tuple[bool, str]:
        """Validate trade risk parameters

        Args:
            account_balance: Account balance
            lot_size: Proposed lot size
            sl_pips: Stop loss in pips

        Returns:
            Tuple of (valid: bool, reason: str)
        """
        # Calculate actual risk
        risk_amount = lot_size * sl_pips * self.pip_value
        risk_percent = risk_amount / account_balance

        # Check max risk per trade (allow up to 2x high quality risk as buffer)
        max_allowed_risk = self.high_quality_risk * 2.0
        if risk_percent > max_allowed_risk:
            return False, f"Risk too high: {risk_percent:.1%} > {max_allowed_risk:.1%}"

        # Check minimum broker lot size
        if lot_size < 0.01:
            return False, f"Lot size below broker minimum: {lot_size} < 0.01"

        # Check if risk would exceed daily limit
        potential_loss = risk_amount
        if self._daily_pnl - potential_loss < -self.daily_loss_limit:
            return False, f"Would exceed daily loss limit"

        return True, "OK"

    def calculate_partial_lots(
        self,
        total_lot: float,
        tp1_percent: float = 0.5,
        tp2_percent: float = 0.3,
        tp3_percent: float = 0.2
    ) -> Dict[str, float]:
        """Calculate lot sizes for partial TP

        Args:
            total_lot: Total position lot size
            tp1_percent: Percentage to close at TP1
            tp2_percent: Percentage to close at TP2
            tp3_percent: Percentage to close at TP3

        Returns:
            Dict with lot sizes for each TP level
        """
        min_broker_lot = 0.01  # Broker minimum

        # Round to 2 decimal places
        tp1_lot = round(total_lot * tp1_percent, 2)
        tp2_lot = round(total_lot * tp2_percent, 2)

        # Remaining for TP3 (ensure total matches)
        tp3_lot = round(total_lot - tp1_lot - tp2_lot, 2)

        # Ensure broker minimums
        tp1_lot = max(min_broker_lot, tp1_lot)
        tp2_lot = max(min_broker_lot, tp2_lot) if tp2_lot > 0 else 0
        tp3_lot = max(min_broker_lot, tp3_lot) if tp3_lot > 0 else 0

        return {
            "tp1_lot": tp1_lot,
            "tp2_lot": tp2_lot,
            "tp3_lot": tp3_lot,
            "total": tp1_lot + tp2_lot + tp3_lot
        }

    def reset_daily(self):
        """Manually reset daily stats"""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._current_date = datetime.now(timezone.utc).date()
        logger.info("Daily stats reset manually")

    @property
    def daily_pnl(self) -> float:
        """Get current daily P/L"""
        return self._daily_pnl

    @property
    def open_positions(self) -> int:
        """Get open position count"""
        return self._open_positions
