"""
Trading Filters - Shared filter classes for live and backtest
=============================================================

This module contains the shared trading filter logic used by both
executor.py (live trading) and backtest.py.

Active Classes:
- IntraMonthRiskManager: Layer 3 - Dynamic risk based on monthly performance
- PatternBasedFilter: Layer 4 - Choppy market detection (FIXED in v6.6.0)
- H4BiasFilter: Layer 7 - Higher Timeframe Bias (v6.8)
- MarketStructureFilter: Layer 8 - BOS/CHoCH detection (NEW in v6.9)

Deprecated Classes (kept for backwards compatibility):
- ChoppinessFilter: Layer 5 - Tested, made results worse
- DirectionalMomentumFilter: Layer 6 - Tested, made results worse

Key Fix (v6.6.0):
- PatternBasedFilter direction window bug: was using DIRECTION_TEST_WINDOW (8)
  instead of ROLLING_WINDOW (10) for direction-specific win rate calculation.
  Fixed by using same 'recent' variable for both rolling_wr and direction stats.

New Feature (v6.8):
- H4BiasFilter: Multi-Timeframe analysis using H4 EMA20/EMA50 crossover
  to filter H1 entries. Only allows trades aligned with H4 trend direction.

New Feature (v6.9):
- MarketStructureFilter: Break of Structure (BOS) and Change of Character
  (CHoCH) detection for market structure analysis.
  - BOS (Bullish): Price breaks above previous swing high in uptrend
  - BOS (Bearish): Price breaks below previous swing low in downtrend
  - CHoCH (Bullish): First higher low after downtrend (reversal signal)
  - CHoCH (Bearish): First lower high after uptrend (reversal signal)

Author: SURIOTA Team
"""

from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import math
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION - Layer 3 & 4
# ============================================================

# Layer 3: Intra-Month Dynamic Risk (original values)
MONTHLY_LOSS_THRESHOLD_1 = -150.0  # +5 quality
MONTHLY_LOSS_THRESHOLD_2 = -250.0  # +10 quality
MONTHLY_LOSS_THRESHOLD_3 = -350.0  # +15 quality
MONTHLY_LOSS_STOP = -400.0         # Stop trading
CONSECUTIVE_LOSS_THRESHOLD = 3     # +5 quality after 3 losses
CONSECUTIVE_LOSS_DAY_STOP = 6      # Stop for day after 6 losses

# Layer 4: Pattern-Based Filter
WARMUP_TRADES = 15                 # Trades before filter activates (same as original)
ROLLING_WINDOW = 10                # Window for rolling win rate
ROLLING_WR_HALT = 0.10             # Halt if WR < 10%
ROLLING_WR_CAUTION = 0.25          # Reduce size if WR < 25%
DIRECTION_TEST_WINDOW = 8          # Window for direction analysis
MIN_DIRECTION_WIN_RATE = 0.10      # Min WR per direction
BOTH_DIRECTIONS_FAIL_THRESHOLD = 4 # Halt if both directions fail 4+ times
RECOVERY_SIZE_MULT = 0.5           # Size multiplier in recovery
RECOVERY_WIN_REQUIRED = 1          # Wins needed to exit recovery
CAUTION_SIZE_MULT = 0.6            # Size multiplier in caution mode
PROBE_TRADE_SIZE = 0.4             # Size for probe trades
PROBE_QUALITY_EXTRA = 5            # Extra quality for probe trades

# Monthly tradeable percentage (historical data only - 2024)
# For future months, we use SEASONAL approach: same month from 2024 with safety margin
MONTHLY_TRADEABLE_PCT = {
    # 2024 - Actual historical data (used as seasonal template)
    (2024, 1): 67, (2024, 2): 55, (2024, 3): 70, (2024, 4): 80,
    (2024, 5): 65, (2024, 6): 72, (2024, 7): 78, (2024, 8): 60,
    (2024, 9): 75, (2024, 10): 58, (2024, 11): 68, (2024, 12): 45,
}

# Seasonal template: Use 2024 data with targeted adjustments
# Only apply safety margin to historically optimistic months (>70%)
# This balances conservatism with trade opportunities
SEASONAL_TEMPLATE = {
    1: 65,  # 67 - slight margin
    2: 55,  # 55 - already conservative, keep as is
    3: 70,  # 70 - keep as is
    4: 70,  # 80 - 10 (optimistic month, apply margin)
    5: 62,  # 65 - slight margin
    6: 68,  # 72 - slight margin
    7: 78,  # 78 - keep as is (good month)
    8: 60,  # 60 - keep as is (already cautious)
    9: 72,  # 75 - slight margin
    10: 58, # 58 - keep as is (already cautious)
    11: 66, # 68 - slight margin
    12: 45, # 45 - keep as is (worst month)
}


def get_monthly_quality_adjustment(dt: datetime) -> int:
    """
    Get quality adjustment based on month's historical tradeable percentage.

    Uses seasonal approach for future months: same month from 2024 template
    with 10% safety margin to account for year-to-year variation.

    Args:
        dt: Datetime to check

    Returns:
        Quality adjustment (higher = stricter filter)
    """
    key = (dt.year, dt.month)
    # First check exact match, then use conservative seasonal template
    tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, SEASONAL_TEMPLATE.get(dt.month, 55))

    if tradeable_pct < 30:
        return 50  # NO TRADE - extremely poor month
    elif tradeable_pct < 40:
        return 35  # HALT - very poor month
    elif tradeable_pct < 50:
        return 25  # Extreme caution
    elif tradeable_pct < 60:
        return 15  # Very poor month
    elif tradeable_pct < 70:
        return 10  # Below average
    elif tradeable_pct < 75:
        return 5   # Slight adjustment
    else:
        return 0   # Good month


# ============================================================
# Layer 3: Intra-Month Risk Manager
# ============================================================

class IntraMonthRiskManager:
    """
    Layer 3: Intra-Month Dynamic Risk Management

    Adjusts quality requirements and can halt trading based on:
    - Monthly P&L (circuit breaker if loss > $400)
    - Consecutive losses (halt for day if 6+ losses)
    - Dynamic quality adjustment based on current month performance

    State can be persisted via StateManager.
    """

    def __init__(self, state_manager=None):
        """
        Initialize risk manager.

        Args:
            state_manager: Optional StateManager for persistence
        """
        self.state_manager = state_manager

        # Load state if available
        if state_manager:
            self.current_month = state_manager.state.current_month
            self.monthly_pnl = state_manager.get_monthly_pnl()
            self.consecutive_losses = state_manager.get_consecutive_losses()
            self.month_stopped = state_manager.is_month_stopped()
            self.day_stopped = state_manager.is_day_stopped()
        else:
            self.current_month = None
            self.monthly_pnl = 0.0
            self.consecutive_losses = 0
            self.month_stopped = False
            self.day_stopped = False

        self.last_trade_date = None

    def _check_new_month(self, dt: datetime):
        """Check if we've entered a new month and reset if needed"""
        month_key = f"{dt.year}-{dt.month:02d}"

        if self.current_month != month_key:
            logger.info(f"[Layer3] New month: {self.current_month} -> {month_key}")
            self.current_month = month_key
            self.monthly_pnl = 0.0
            self.consecutive_losses = 0
            self.month_stopped = False
            self.day_stopped = False

            if self.state_manager:
                self.state_manager.check_month_reset(dt)

    def _check_new_day(self, dt: datetime):
        """Check if we've entered a new day and reset day stop if needed"""
        current_date = dt.strftime("%Y-%m-%d")

        if self.day_stopped and self.last_trade_date != current_date:
            logger.info(f"[Layer3] New day - resetting day stop")
            self.day_stopped = False

            if self.state_manager:
                self.state_manager.check_day_reset(dt)

        self.last_trade_date = current_date

    def new_trade_check(self, dt: datetime) -> Tuple[bool, int, str]:
        """
        Check if a new trade is allowed.

        Args:
            dt: Current datetime

        Returns:
            Tuple of (can_trade, quality_adjustment, reason_if_blocked)
        """
        self._check_new_month(dt)
        self._check_new_day(dt)

        # Check month circuit breaker
        if self.month_stopped:
            return False, 0, f"MONTH_STOPPED (loss: ${self.monthly_pnl:.0f})"

        # Check day stop
        if self.day_stopped:
            return False, 0, f"DAY_STOPPED (consec_loss: {self.consecutive_losses})"

        # Calculate dynamic quality adjustment
        quality_adj = 0

        # Adjustment based on monthly P&L
        if self.monthly_pnl <= MONTHLY_LOSS_STOP:
            self.month_stopped = True
            if self.state_manager:
                self.state_manager.set_month_stopped(f"Loss ${self.monthly_pnl:.0f}")
            return False, 0, f"MONTH_CIRCUIT_BREAKER (${self.monthly_pnl:.0f})"

        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_3:
            quality_adj += 15
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_2:
            quality_adj += 10
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_1:
            quality_adj += 5

        # Adjustment based on consecutive losses
        if self.consecutive_losses >= CONSECUTIVE_LOSS_DAY_STOP:
            self.day_stopped = True
            if self.state_manager:
                self.state_manager.set_day_stopped(dt, f"{self.consecutive_losses} losses")
            return False, 0, f"DAY_STOP ({self.consecutive_losses} consecutive losses)"

        elif self.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            quality_adj += 5

        return True, quality_adj, ""

    def record_trade(self, pnl: float, dt: datetime, direction: str = ""):
        """
        Record a completed trade.

        Args:
            pnl: Trade P&L in dollars
            dt: Trade close time
            direction: "BUY" or "SELL"
        """
        self._check_new_month(dt)

        self.monthly_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Persist if state manager available
        if self.state_manager:
            self.state_manager.record_trade(pnl, direction, dt)

        logger.debug(f"[Layer3] Trade recorded: ${pnl:+.2f}, Monthly: ${self.monthly_pnl:.2f}, ConsecLoss: {self.consecutive_losses}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current stats for display"""
        return {
            "month": self.current_month,
            "monthly_pnl": self.monthly_pnl,
            "consecutive_losses": self.consecutive_losses,
            "month_stopped": self.month_stopped,
            "day_stopped": self.day_stopped
        }


# ============================================================
# Layer 4: Pattern-Based Filter
# ============================================================

class PatternBasedFilter:
    """
    Layer 4: Pattern-Based Choppy Market Detector

    Detects choppy markets by analyzing recent trade patterns:
    - Rolling win rate (halt if < 10%)
    - Direction balance (halt if both BUY and SELL failing)
    - Recovery mode after halt

    State can be persisted via StateManager.
    """

    def __init__(self, state_manager=None):
        """
        Initialize pattern filter.

        Args:
            state_manager: Optional StateManager for persistence
        """
        self.state_manager = state_manager

        # Load state if available
        if state_manager:
            self.trade_history = state_manager.get_trade_history()
            self.is_halted = state_manager.is_halted()
            self.in_recovery = state_manager.is_in_recovery()
            self.trade_count = state_manager.get_trade_count()
        else:
            self.trade_history: List[Tuple[str, float, datetime]] = []
            self.is_halted = False
            self.in_recovery = False
            self.trade_count = 0

        self.halt_reason = ""
        self.recovery_wins = 0  # Track wins during recovery
        self.current_month = None
        self.probe_taken = False

    def record_trade(self, direction: str, pnl: float, dt: datetime):
        """
        Record a trade result.

        Args:
            direction: "BUY" or "SELL"
            pnl: Trade P&L
            dt: Trade close time
        """
        self.trade_history.append((direction, pnl, dt))

        # Keep only last 50 trades
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]

        self.trade_count += 1

        # Track recovery progress
        if self.in_recovery:
            if pnl > 0:
                self.recovery_wins += 1
                if self.recovery_wins >= RECOVERY_WIN_REQUIRED:
                    logger.info(f"[Layer4] Recovery complete ({self.recovery_wins} wins) - exiting recovery mode")
                    self.is_halted = False
                    self.in_recovery = False
                    self.recovery_wins = 0
                    if self.state_manager:
                        self.state_manager.exit_recovery_mode()
            else:
                self.recovery_wins = 0  # Reset on loss

    def check_trade_allowed(self) -> Tuple[bool, float, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (allowed, size_multiplier, reason_if_blocked)
        """
        # Warmup period - allow with full size (observe only, don't restrict)
        if self.trade_count < WARMUP_TRADES:
            return True, 1.0, f"WARMUP ({self.trade_count}/{WARMUP_TRADES})"

        # Halted - but allow if in recovery mode
        if self.is_halted and not self.in_recovery:
            return False, 0, f"HALTED: {self.halt_reason}"

        # Recovery mode - allow with reduced size (probe trade)
        if self.in_recovery:
            return True, RECOVERY_SIZE_MULT, "RECOVERY"

        # Check caution level
        if len(self.trade_history) >= ROLLING_WINDOW:
            recent = self.trade_history[-ROLLING_WINDOW:]
            wins = sum(1 for _, pnl, _ in recent if pnl > 0)
            rolling_wr = wins / len(recent)

            if rolling_wr < ROLLING_WR_CAUTION:
                return True, CAUTION_SIZE_MULT, f"CAUTION (WR={rolling_wr:.0%})"

        return True, 1.0, "OK"

    def get_quality_adjustment(self) -> int:
        """Get extra quality requirement based on pattern state"""
        # Warmup: no extra quality (just observe)
        if self.trade_count < WARMUP_TRADES:
            return 0

        # Recovery: extra quality for probe trades
        if self.in_recovery:
            return PROBE_QUALITY_EXTRA

        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get current stats for display"""
        recent = self.trade_history[-ROLLING_WINDOW:] if len(self.trade_history) >= ROLLING_WINDOW else self.trade_history
        rolling_wr = sum(1 for _, pnl, _ in recent if pnl > 0) / len(recent) if recent else 0

        return {
            "trade_count": self.trade_count,
            "warmup_complete": self.trade_count >= WARMUP_TRADES,
            "rolling_wr": rolling_wr,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "in_recovery": self.in_recovery,
            "history_size": len(self.trade_history)
        }

    # ============================================================
    # Backtest Compatibility Methods
    # ============================================================

    def reset_for_month(self, month: int):
        """
        Reset filter state for new month - fresh start each month.

        CRITICAL FIX: Always clear halt state on new month.
        Previous bug: halt could persist across months if not in recovery.
        Trade history is kept (capped) for pattern analysis continuity.
        """
        self.current_month = month
        self.probe_taken = False

        # ALWAYS reset halt state on new month - each month is a fresh start
        if self.is_halted:
            logger.info(f"[Layer4] New month {month}: clearing halt state (was: {self.halt_reason})")

        self.is_halted = False
        self.in_recovery = False
        self.recovery_wins = 0
        self.halt_reason = ""

        if self.state_manager:
            self.state_manager.clear_halt()
            self.state_manager.exit_recovery_mode()

        # Keep trade_history for learning but cap at last 30 trades
        if len(self.trade_history) > 30:
            self.trade_history = self.trade_history[-30:]

        logger.info(f"[Layer4] Reset for month {month}, history: {len(self.trade_history)} trades")

    def check_trade(self, direction: str) -> Tuple[bool, int, float, str]:
        """
        Check if trade is allowed (matches original check_trade exactly).

        Args:
            direction: "BUY" or "SELL"

        Returns:
            Tuple of (can_trade, extra_quality, size_multiplier, reason)
        """
        # During warmup, always allow trades (just observe) - RETURN EARLY
        if len(self.trade_history) < WARMUP_TRADES:
            return True, 0, 1.0, f"WARMUP ({len(self.trade_history)}/{WARMUP_TRADES})"

        # Check if halted (this never triggers since is_halted and in_recovery are set together)
        if self.is_halted and not self.in_recovery:
            return False, 0, 1.0, f"HALTED: {self.halt_reason}"

        # Compute rolling stats (matches original _get_rolling_stats)
        recent = self.trade_history[-ROLLING_WINDOW:]
        wins = sum(1 for _, pnl, _ in recent if pnl > 0)
        rolling_wr = wins / len(recent) if recent else 1.0

        # Check for choppy market (both directions failing)
        both_fail = False
        if len(self.trade_history) >= BOTH_DIRECTIONS_FAIL_THRESHOLD * 2:
            recent_window = self.trade_history[-BOTH_DIRECTIONS_FAIL_THRESHOLD*2:]
            buy_losses = sum(1 for d, p, _ in recent_window if d == 'BUY' and p < 0)
            sell_losses = sum(1 for d, p, _ in recent_window if d == 'SELL' and p < 0)
            if buy_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD and sell_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD:
                both_fail = True

        if both_fail:
            self.is_halted = True
            self.halt_reason = "BOTH_DIRECTIONS_FAIL"
            self.in_recovery = True
            self.recovery_wins = 0
            return False, 0, 1.0, "CHOPPY_MARKET_DETECTED"

        # Check rolling win rate
        if rolling_wr < ROLLING_WR_HALT:
            self.is_halted = True
            self.halt_reason = f"LOW_ROLLING_WR ({rolling_wr:.0%})"
            self.in_recovery = True
            self.recovery_wins = 0
            return False, 0, 1.0, f"LOW_WIN_RATE ({rolling_wr:.0%})"

        # Check direction-specific win rate (only if enough data)
        # Match original: use ROLLING_WINDOW for direction stats, same as rolling_wr
        if len(self.trade_history) >= 10:
            # Use same 'recent' window as rolling_wr calculation
            buy_trades = [(d, p) for d, p, _ in recent if d == 'BUY']
            sell_trades = [(d, p) for d, p, _ in recent if d == 'SELL']

            buy_wr = sum(1 for _, p in buy_trades if p > 0) / len(buy_trades) if buy_trades else 1.0
            sell_wr = sum(1 for _, p in sell_trades if p > 0) / len(sell_trades) if sell_trades else 1.0

            if direction == 'BUY' and buy_wr < MIN_DIRECTION_WIN_RATE:
                return False, 0, 1.0, f"BUY_DIRECTION_WEAK ({buy_wr:.0%})"
            if direction == 'SELL' and sell_wr < MIN_DIRECTION_WIN_RATE:
                return False, 0, 1.0, f"SELL_DIRECTION_WEAK ({sell_wr:.0%})"

        # Determine size and quality adjustments
        size_mult = 1.0
        extra_q = 0

        # Recovery mode - trade smaller with probe
        if self.in_recovery:
            size_mult = RECOVERY_SIZE_MULT
            extra_q = PROBE_QUALITY_EXTRA

        # Caution mode - rolling WR below threshold
        elif rolling_wr < ROLLING_WR_CAUTION:
            size_mult = CAUTION_SIZE_MULT
            extra_q = 3

        return True, extra_q, size_mult, "OK"


# ============================================================
# DEPRECATED: Layer 5 & 6 Filters
# ============================================================
# These filters were tested but MADE RESULTS WORSE:
# - ChoppinessFilter: Only adjusted 10 trades, didn't prevent losses
# - DirectionalMomentumFilter: Added May as losing month when combined with ADX
# Kept for backwards compatibility and research reference.
# ============================================================

# Choppiness Index thresholds (DEPRECATED)
CHOP_PERIOD = 14
CHOP_CHOPPY_THRESHOLD = 61.8
CHOP_TRENDING_THRESHOLD = 38.2
CHOP_EXTREME_CHOPPY = 70.0
CHOP_SIZE_REDUCTION_CHOPPY = 0.5
CHOP_SIZE_REDUCTION_EXTREME = 0.3
CHOP_QUALITY_ADD_CHOPPY = 10
CHOP_QUALITY_ADD_EXTREME = 20


def calculate_choppiness_index(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = CHOP_PERIOD
) -> float:
    """
    Calculate the Choppiness Index (CHOP).

    Formula: 100 * LOG10(SUM(ATR(1), n) / (MaxHi(n) - MinLo(n))) / LOG10(n)

    Args:
        highs: List of high prices (most recent last)
        lows: List of low prices (most recent last)
        closes: List of close prices (most recent last)
        period: Lookback period (default 14)

    Returns:
        Choppiness Index value (0-100)
        - High values (>61.8) = choppy/ranging market
        - Low values (<38.2) = trending market
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return 50.0  # Neutral if insufficient data

    # Get the most recent 'period' bars plus 1 for previous close
    highs = highs[-(period + 1):]
    lows = lows[-(period + 1):]
    closes = closes[-(period + 1):]

    # Calculate True Range for each bar
    atr_sum = 0.0
    for i in range(1, period + 1):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        true_range = max(tr1, tr2, tr3)
        atr_sum += true_range

    # Calculate highest high and lowest low over the period
    max_high = max(highs[1:])
    min_low = min(lows[1:])

    # Avoid division by zero
    range_diff = max_high - min_low
    if range_diff <= 0:
        return 50.0  # Neutral if no range

    # Calculate Choppiness Index
    # Formula: 100 * LOG10(SUM(TR, n) / (MaxHi - MinLo)) / LOG10(n)
    try:
        chop = 100 * math.log10(atr_sum / range_diff) / math.log10(period)
    except (ValueError, ZeroDivisionError):
        return 50.0

    # Clamp to 0-100 range
    return max(0.0, min(100.0, chop))


class ChoppinessFilter:
    """
    DEPRECATED: Layer 5 - Choppiness Index Based Market Filter

    WARNING: Tested in backtest - only adjusted 10 trades, didn't prevent losses.
    Kept for backwards compatibility only. DO NOT USE in production.

    Reference: E.W. Dreiss (1993), Chaos Theory applied to commodities
    """

    def __init__(self):
        """Initialize the Choppiness Filter"""
        self.current_chop = 50.0
        self.chop_history: List[Tuple[datetime, float]] = []
        self.is_choppy = False
        self.is_extreme_choppy = False
        self.is_trending = False

    def update(self, highs: List[float], lows: List[float],
               closes: List[float], timestamp: datetime) -> float:
        """
        Update the Choppiness Index with new price data.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            timestamp: Current bar timestamp

        Returns:
            Current Choppiness Index value
        """
        self.current_chop = calculate_choppiness_index(highs, lows, closes)

        # Update state flags
        self.is_extreme_choppy = self.current_chop >= CHOP_EXTREME_CHOPPY
        self.is_choppy = self.current_chop >= CHOP_CHOPPY_THRESHOLD
        self.is_trending = self.current_chop <= CHOP_TRENDING_THRESHOLD

        # Store history (keep last 50)
        self.chop_history.append((timestamp, self.current_chop))
        if len(self.chop_history) > 50:
            self.chop_history = self.chop_history[-50:]

        return self.current_chop

    def check_trade(self, direction: str) -> Tuple[bool, int, float, str]:
        """
        Check if trade is allowed and get adjustments based on choppiness.

        Unlike PatternBasedFilter, this NEVER blocks trades entirely.
        Instead, it adjusts size and quality requirements.

        Args:
            direction: "BUY" or "SELL"

        Returns:
            Tuple of (can_trade, extra_quality, size_multiplier, reason)
        """
        # Extreme choppy - very high risk, significant reduction
        if self.is_extreme_choppy:
            return True, CHOP_QUALITY_ADD_EXTREME, CHOP_SIZE_REDUCTION_EXTREME, \
                   f"EXTREME_CHOPPY (CHOP={self.current_chop:.1f})"

        # Choppy - moderate risk, reduce size and add quality
        if self.is_choppy:
            return True, CHOP_QUALITY_ADD_CHOPPY, CHOP_SIZE_REDUCTION_CHOPPY, \
                   f"CHOPPY (CHOP={self.current_chop:.1f})"

        # Trending - optimal conditions, full size
        if self.is_trending:
            return True, 0, 1.0, f"TRENDING (CHOP={self.current_chop:.1f})"

        # Normal - between thresholds
        return True, 0, 1.0, f"NORMAL (CHOP={self.current_chop:.1f})"

    def get_stats(self) -> Dict[str, Any]:
        """Get current filter statistics"""
        avg_chop = 50.0
        if self.chop_history:
            avg_chop = sum(c for _, c in self.chop_history) / len(self.chop_history)

        return {
            "current_chop": self.current_chop,
            "avg_chop": avg_chop,
            "is_choppy": self.is_choppy,
            "is_extreme_choppy": self.is_extreme_choppy,
            "is_trending": self.is_trending,
            "history_size": len(self.chop_history)
        }


# Directional Momentum thresholds (DEPRECATED)
DIR_CONSEC_LOSS_CAUTION = 2
DIR_CONSEC_LOSS_WARNING = 3
DIR_CONSEC_LOSS_EXTREME = 4
DIR_SIZE_CAUTION = 0.6
DIR_SIZE_WARNING = 0.4
DIR_SIZE_EXTREME = 0.25
DIR_QUALITY_CAUTION = 5
DIR_QUALITY_WARNING = 12
DIR_QUALITY_EXTREME = 20
DIR_RECOVERY_WINS = 1


class DirectionalMomentumFilter:
    """
    DEPRECATED: Layer 6 - Directional Momentum Filter

    WARNING: Tested in backtest - added May as losing month when combined with ADX.
    Kept for backwards compatibility only. DO NOT USE in production.
    """

    def __init__(self):
        """Initialize directional momentum tracker"""
        self.buy_consecutive_losses = 0
        self.sell_consecutive_losses = 0
        self.buy_recent_wins = 0
        self.sell_recent_wins = 0
        self.trade_history: List[Tuple[str, float, datetime]] = []

    def record_trade(self, direction: str, pnl: float, dt: datetime):
        """
        Record a trade result and update directional stats.

        Args:
            direction: "BUY" or "SELL"
            pnl: Trade P&L
            dt: Trade close time
        """
        self.trade_history.append((direction, pnl, dt))

        # Keep only last 30 trades
        if len(self.trade_history) > 30:
            self.trade_history = self.trade_history[-30:]

        if direction == 'BUY':
            if pnl < 0:
                self.buy_consecutive_losses += 1
                self.buy_recent_wins = 0
            else:
                self.buy_recent_wins += 1
                if self.buy_recent_wins >= DIR_RECOVERY_WINS:
                    # Recovery - reset consecutive losses
                    self.buy_consecutive_losses = 0
                    logger.debug(f"[Layer6] BUY direction recovered after {DIR_RECOVERY_WINS} win(s)")
        else:  # SELL
            if pnl < 0:
                self.sell_consecutive_losses += 1
                self.sell_recent_wins = 0
            else:
                self.sell_recent_wins += 1
                if self.sell_recent_wins >= DIR_RECOVERY_WINS:
                    # Recovery - reset consecutive losses
                    self.sell_consecutive_losses = 0
                    logger.debug(f"[Layer6] SELL direction recovered after {DIR_RECOVERY_WINS} win(s)")

        logger.debug(f"[Layer6] Trade recorded: {direction} ${pnl:+.2f}, "
                    f"BUY_consec={self.buy_consecutive_losses}, SELL_consec={self.sell_consecutive_losses}")

    def check_trade(self, direction: str) -> Tuple[bool, int, float, str]:
        """
        Check trade and return adjustments based on directional performance.

        This filter NEVER blocks trades, only adjusts size and quality.

        Args:
            direction: "BUY" or "SELL"

        Returns:
            Tuple of (can_trade, extra_quality, size_multiplier, reason)
        """
        # Get consecutive losses for this direction
        if direction == 'BUY':
            consec_losses = self.buy_consecutive_losses
        else:
            consec_losses = self.sell_consecutive_losses

        # Determine adjustments based on consecutive losses
        if consec_losses >= DIR_CONSEC_LOSS_EXTREME:
            return True, DIR_QUALITY_EXTREME, DIR_SIZE_EXTREME, \
                   f"{direction}_EXTREME_WEAK ({consec_losses} consec losses)"
        elif consec_losses >= DIR_CONSEC_LOSS_WARNING:
            return True, DIR_QUALITY_WARNING, DIR_SIZE_WARNING, \
                   f"{direction}_WARNING ({consec_losses} consec losses)"
        elif consec_losses >= DIR_CONSEC_LOSS_CAUTION:
            return True, DIR_QUALITY_CAUTION, DIR_SIZE_CAUTION, \
                   f"{direction}_CAUTION ({consec_losses} consec losses)"

        # Normal - no adjustment
        return True, 0, 1.0, "OK"

    def get_stats(self) -> Dict[str, Any]:
        """Get current filter statistics"""
        return {
            "buy_consecutive_losses": self.buy_consecutive_losses,
            "sell_consecutive_losses": self.sell_consecutive_losses,
            "buy_recent_wins": self.buy_recent_wins,
            "sell_recent_wins": self.sell_recent_wins,
            "history_size": len(self.trade_history)
        }

    def reset_for_month(self, month: int):
        """
        Soft reset for new month.

        Note: We don't fully reset consecutive losses because directional
        weakness can persist across month boundaries. We just reduce them.
        """
        # Decay consecutive losses slightly at month boundary
        self.buy_consecutive_losses = max(0, self.buy_consecutive_losses - 1)
        self.sell_consecutive_losses = max(0, self.sell_consecutive_losses - 1)
        # Keep trade history for continuity


# ============================================================
# Layer 7: H4 Multi-Timeframe Bias Filter (NEW in v6.8)
# ============================================================
# Uses H4 timeframe EMA crossover to determine higher timeframe bias.
# Only allows H1 trades that align with H4 trend direction.
# ============================================================

class H4Bias(Enum):
    """H4 Timeframe Bias states"""
    BULLISH = "bullish"    # EMA20 > EMA50, allow BUY only
    BEARISH = "bearish"    # EMA20 < EMA50, allow SELL only
    SIDEWAYS = "sideways"  # EMAs converging, skip trading


# H4 Bias Configuration
H4_EMA_FAST = 20           # Fast EMA period
H4_EMA_SLOW = 50           # Slow EMA period
H4_BARS_PER_CANDLE = 4     # H1 bars per H4 candle
H4_MIN_EMA_SEPARATION = 0.0003  # Minimum separation for trend confirmation (30 pips)
H4_LOOKBACK_BARS = 100     # Minimum H1 bars needed for H4 calculation


def resample_h1_to_h4(df_h1: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Resample H1 OHLCV data to H4 timeframe.

    Groups every 4 H1 bars into 1 H4 bar using standard OHLCV aggregation:
    - Open: first of the 4 bars
    - High: max of the 4 bars
    - Low: min of the 4 bars
    - Close: last of the 4 bars

    Args:
        df_h1: H1 DataFrame with datetime index
        col_map: Column mapping dict with 'open', 'high', 'low', 'close' keys

    Returns:
        H4 DataFrame with resampled OHLCV data
    """
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    # Ensure index is datetime
    if not isinstance(df_h1.index, pd.DatetimeIndex):
        df_h1 = df_h1.copy()
        df_h1.index = pd.to_datetime(df_h1.index)

    # Resample to 4H using standard OHLCV rules
    df_h4 = df_h1.resample('4H').agg({
        o: 'first',
        h: 'max',
        l: 'min',
        c: 'last'
    }).dropna()

    return df_h4


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def get_h4_bias(df_h1: pd.DataFrame, col_map: dict) -> Tuple[H4Bias, float, float, float]:
    """
    Calculate H4 timeframe bias from H1 data.

    Resamples H1 data to H4, then calculates EMA20/EMA50 crossover
    to determine the higher timeframe trend direction.

    Args:
        df_h1: H1 DataFrame (needs at least 200 bars for reliable H4 EMAs)
        col_map: Column mapping dict

    Returns:
        Tuple of (bias, ema20, ema50, separation):
        - bias: H4Bias enum (BULLISH, BEARISH, or SIDEWAYS)
        - ema20: Current H4 EMA20 value
        - ema50: Current H4 EMA50 value
        - separation: EMA separation in price units
    """
    # Need enough H1 bars for reliable H4 EMAs
    # For EMA50 on H4, we need at least 50 H4 bars = 200 H1 bars
    if len(df_h1) < H4_LOOKBACK_BARS:
        return H4Bias.SIDEWAYS, 0.0, 0.0, 0.0

    try:
        # Resample H1 to H4
        df_h4 = resample_h1_to_h4(df_h1, col_map)

        if len(df_h4) < H4_EMA_SLOW:
            return H4Bias.SIDEWAYS, 0.0, 0.0, 0.0

        c = col_map['close']

        # Calculate EMAs on H4 data
        ema20 = calculate_ema(df_h4[c], H4_EMA_FAST)
        ema50 = calculate_ema(df_h4[c], H4_EMA_SLOW)

        # Get current values
        current_ema20 = ema20.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_close = df_h4[c].iloc[-1]

        # Calculate separation
        separation = abs(current_ema20 - current_ema50)

        # Determine bias
        if current_ema20 > current_ema50 and separation >= H4_MIN_EMA_SEPARATION:
            # Strong bullish - EMA20 above EMA50 with clear separation
            if current_close > current_ema20:
                # Price above both EMAs - strong bullish
                return H4Bias.BULLISH, current_ema20, current_ema50, separation
            else:
                # Price between EMAs - weak bullish, treat as sideways
                return H4Bias.SIDEWAYS, current_ema20, current_ema50, separation

        elif current_ema20 < current_ema50 and separation >= H4_MIN_EMA_SEPARATION:
            # Strong bearish - EMA20 below EMA50 with clear separation
            if current_close < current_ema20:
                # Price below both EMAs - strong bearish
                return H4Bias.BEARISH, current_ema20, current_ema50, separation
            else:
                # Price between EMAs - weak bearish, treat as sideways
                return H4Bias.SIDEWAYS, current_ema20, current_ema50, separation
        else:
            # EMAs converging or too close - sideways/ranging market
            return H4Bias.SIDEWAYS, current_ema20, current_ema50, separation

    except Exception as e:
        logger.warning(f"[H4Bias] Error calculating H4 bias: {e}")
        return H4Bias.SIDEWAYS, 0.0, 0.0, 0.0


class H4BiasFilter:
    """
    Layer 7: H4 Multi-Timeframe Bias Filter

    Uses H4 timeframe EMA20/EMA50 crossover to filter H1 entries.
    Only allows trades that align with the higher timeframe trend.

    Rules:
    - H4 BULLISH (EMA20 > EMA50): Only allow BUY signals
    - H4 BEARISH (EMA20 < EMA50): Only allow SELL signals
    - H4 SIDEWAYS (EMAs converging): Skip all trading
    """

    def __init__(self):
        """Initialize H4 Bias Filter"""
        self.current_bias = H4Bias.SIDEWAYS
        self.ema20 = 0.0
        self.ema50 = 0.0
        self.separation = 0.0
        self.last_update = None

        # Statistics tracking
        self.stats = {
            'h4_aligned_trades': 0,
            'h4_contrary_blocked': 0,
            'h4_sideways_blocked': 0,
            'h4_bullish_periods': 0,
            'h4_bearish_periods': 0,
            'h4_sideways_periods': 0,
        }

    def update(self, df_h1: pd.DataFrame, col_map: dict) -> H4Bias:
        """
        Update H4 bias from latest H1 data.

        Args:
            df_h1: H1 DataFrame slice up to current bar
            col_map: Column mapping dict

        Returns:
            Current H4 bias
        """
        self.current_bias, self.ema20, self.ema50, self.separation = get_h4_bias(df_h1, col_map)

        # Track periods
        if self.current_bias == H4Bias.BULLISH:
            self.stats['h4_bullish_periods'] += 1
        elif self.current_bias == H4Bias.BEARISH:
            self.stats['h4_bearish_periods'] += 1
        else:
            self.stats['h4_sideways_periods'] += 1

        return self.current_bias

    def check_trade(self, direction: str) -> Tuple[bool, str]:
        """
        Check if trade direction aligns with H4 bias.

        Args:
            direction: "BUY" or "SELL"

        Returns:
            Tuple of (allowed, reason)
        """
        # SIDEWAYS - skip all trading
        if self.current_bias == H4Bias.SIDEWAYS:
            self.stats['h4_sideways_blocked'] += 1
            return False, f"H4_SIDEWAYS (EMA20={self.ema20:.5f}, EMA50={self.ema50:.5f}, sep={self.separation:.5f})"

        # BULLISH - only allow BUY
        if self.current_bias == H4Bias.BULLISH:
            if direction == 'BUY':
                self.stats['h4_aligned_trades'] += 1
                return True, f"H4_BULLISH_ALIGNED (EMA20={self.ema20:.5f} > EMA50={self.ema50:.5f})"
            else:
                self.stats['h4_contrary_blocked'] += 1
                return False, f"H4_BULLISH_CONTRARY (SELL blocked, H4 is bullish)"

        # BEARISH - only allow SELL
        if self.current_bias == H4Bias.BEARISH:
            if direction == 'SELL':
                self.stats['h4_aligned_trades'] += 1
                return True, f"H4_BEARISH_ALIGNED (EMA20={self.ema20:.5f} < EMA50={self.ema50:.5f})"
            else:
                self.stats['h4_contrary_blocked'] += 1
                return False, f"H4_BEARISH_CONTRARY (BUY blocked, H4 is bearish)"

        # Fallback - should not reach here
        return True, "H4_UNKNOWN"

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics"""
        return {
            "current_bias": self.current_bias.value,
            "ema20": self.ema20,
            "ema50": self.ema50,
            "separation": self.separation,
            **self.stats
        }

    def reset_stats(self):
        """Reset statistics for new backtest period"""
        self.stats = {
            'h4_aligned_trades': 0,
            'h4_contrary_blocked': 0,
            'h4_sideways_blocked': 0,
            'h4_bullish_periods': 0,
            'h4_bearish_periods': 0,
            'h4_sideways_periods': 0,
        }


# ============================================================
# Position Sizing Utilities
# ============================================================

def calculate_lot_size(
    balance: float,
    risk_percent: float,
    sl_pips: float,
    pip_value: float = 10.0,
    risk_multiplier: float = 1.0,
    pattern_size_mult: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 5.0,
    min_sl_pips: float = 5.0
) -> float:
    """
    Calculate position size with safety checks.

    Args:
        balance: Account balance
        risk_percent: Risk per trade (e.g., 1.0 for 1%)
        sl_pips: Stop loss in pips
        pip_value: Value per pip per lot (default 10 for GBPUSD)
        risk_multiplier: Quality-based multiplier
        pattern_size_mult: Pattern filter multiplier
        min_lot: Minimum lot size
        max_lot: Maximum lot size
        min_sl_pips: Minimum SL to prevent overleveraging

    Returns:
        Calculated lot size
    """
    # Enforce minimum SL to prevent lot size explosion
    sl_pips = max(sl_pips, min_sl_pips)

    # Calculate base risk amount
    risk_amount = balance * (risk_percent / 100.0) * risk_multiplier * pattern_size_mult

    # Calculate lot size
    lot_size = risk_amount / (sl_pips * pip_value)

    # Apply limits BEFORE rounding to prevent exceeding max
    lot_size = max(min_lot, min(max_lot - 0.01, lot_size))

    # Round to 2 decimal places
    lot_size = round(lot_size, 2)

    # Final safety check
    lot_size = max(min_lot, min(max_lot, lot_size))

    return lot_size


def calculate_sl_tp(
    entry_price: float,
    atr: float,
    direction: str,
    sl_atr_mult: float = 1.5,
    tp_ratio: float = 1.5,
    min_sl_pips: float = 5.0,
    max_sl_pips: float = 50.0,
    pip_size: float = 0.0001
) -> Tuple[float, float, float, float]:
    """
    Calculate stop loss and take profit levels.

    Args:
        entry_price: Entry price
        atr: Current ATR in price units
        direction: "BUY" or "SELL"
        sl_atr_mult: SL multiplier
        tp_ratio: TP/SL ratio
        min_sl_pips: Minimum SL in pips
        max_sl_pips: Maximum SL in pips
        pip_size: Pip size (0.0001 for GBPUSD)

    Returns:
        Tuple of (sl_price, tp_price, sl_pips, tp_pips)
    """
    # Calculate SL in pips
    sl_pips = (atr / pip_size) * sl_atr_mult

    # Enforce min/max
    sl_pips = max(min_sl_pips, min(max_sl_pips, sl_pips))

    # Calculate TP in pips
    tp_pips = sl_pips * tp_ratio

    # Calculate prices
    sl_distance = sl_pips * pip_size
    tp_distance = tp_pips * pip_size

    if direction == "BUY":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:  # SELL
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    return sl_price, tp_price, sl_pips, tp_pips


# ============================================================
# Layer 8: Market Structure Filter (BOS/CHoCH) - NEW in v6.9
# ============================================================
# Break of Structure (BOS) and Change of Character (CHoCH) detection
# for trend confirmation and reversal signals.
#
# Market Structure Concepts:
# - Uptrend: Higher Highs (HH) and Higher Lows (HL)
# - Downtrend: Lower Highs (LH) and Lower Lows (LL)
#
# BOS (Break of Structure) - Trend Continuation:
# - Bullish BOS: Price breaks above previous swing high in uptrend
# - Bearish BOS: Price breaks below previous swing low in downtrend
#
# CHoCH (Change of Character) - Trend Reversal:
# - Bullish CHoCH: First higher low forms after downtrend
# - Bearish CHoCH: First lower high forms after uptrend
# ============================================================

# Market Structure Configuration
STRUCTURE_SWING_LOOKBACK = 5       # Bars on each side for swing point detection
STRUCTURE_MIN_SWING_PIPS = 10      # Minimum pips between consecutive swings
STRUCTURE_BOS_CONFIDENCE = 0.85    # Confidence for BOS signal
STRUCTURE_CHOCH_CONFIDENCE = 0.70  # Confidence for CHoCH signal (reversal)
STRUCTURE_MAX_SWINGS = 20          # Maximum swing points to track
STRUCTURE_QUALITY_BOOST_BOS = 10   # Quality reduction (easier) after BOS confirmation - significant boost!
STRUCTURE_QUALITY_PENALTY_NO_BOS = 0  # No penalty for no structure - structure is informational
STRUCTURE_CHOCH_SIZE_MULT = 1.0    # No size reduction for CHoCH - just informational
STRUCTURE_COUNTER_QUALITY_PENALTY = 0   # No penalty for counter-structure - just informational
STRUCTURE_COUNTER_SIZE_MULT = 1.0    # No size reduction for counter-structure
STRUCTURE_BLOCK_COUNTER = False      # If True, block counter-structure trades; if False, penalize


class StructureType(Enum):
    """Types of market structure signals"""
    NONE = "none"
    BOS_BULLISH = "bos_bullish"      # Break of Structure - Bullish continuation
    BOS_BEARISH = "bos_bearish"      # Break of Structure - Bearish continuation
    CHOCH_BULLISH = "choch_bullish"  # Change of Character - Bearish to Bullish
    CHOCH_BEARISH = "choch_bearish"  # Change of Character - Bullish to Bearish


class TrendState(Enum):
    """Current trend state based on swing structure"""
    UNKNOWN = "unknown"
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class SwingPoint:
    """Represents a swing high or swing low"""
    index: int              # Bar index in dataframe
    price: float            # Price level
    is_high: bool           # True = swing high, False = swing low
    timestamp: datetime     # Time of the swing
    confirmed: bool = True  # Whether the swing is confirmed


@dataclass
class StructureSignal:
    """Market structure signal output"""
    structure_type: StructureType
    direction: str          # "BUY", "SELL", or "NONE"
    confidence: float       # 0.0 to 1.0
    trigger_price: float    # Price that triggered the signal
    trigger_index: int      # Bar index that triggered
    previous_swing: Optional[SwingPoint] = None


def detect_swing_points(
    df: pd.DataFrame,
    col_map: dict,
    lookback: int = STRUCTURE_SWING_LOOKBACK,
    min_distance_pips: float = STRUCTURE_MIN_SWING_PIPS,
    pip_size: float = 0.0001
) -> List[SwingPoint]:
    """
    Identify swing highs and swing lows in price data.

    A swing high is a bar whose high is higher than the highs of
    'lookback' bars on each side.
    A swing low is a bar whose low is lower than the lows of
    'lookback' bars on each side.

    Args:
        df: DataFrame with OHLCV data
        col_map: Column name mapping {'high': 'High', 'low': 'Low', ...}
        lookback: Number of bars on each side to confirm swing
        min_distance_pips: Minimum distance between consecutive swings
        pip_size: Pip size for the instrument

    Returns:
        List of SwingPoint objects sorted by index
    """
    if len(df) < lookback * 2 + 1:
        return []

    h = col_map['high']
    l = col_map['low']

    highs = df[h].values
    lows = df[l].values

    swing_points: List[SwingPoint] = []
    min_distance = min_distance_pips * pip_size

    # Get timestamps - handle both datetime index and column
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.to_list()
    else:
        timestamps = [df.index[i] for i in range(len(df))]

    # Detect swing highs and lows
    for i in range(lookback, len(df) - lookback):
        # Check swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break

        # Check swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break

        # Add swing point if valid
        if is_swing_high:
            # Check minimum distance from last swing high
            last_high = None
            for sp in reversed(swing_points):
                if sp.is_high:
                    last_high = sp
                    break

            if last_high is None or abs(highs[i] - last_high.price) >= min_distance:
                ts = timestamps[i]
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                swing_points.append(SwingPoint(
                    index=i,
                    price=float(highs[i]),
                    is_high=True,
                    timestamp=ts
                ))

        if is_swing_low:
            # Check minimum distance from last swing low
            last_low = None
            for sp in reversed(swing_points):
                if not sp.is_high:
                    last_low = sp
                    break

            if last_low is None or abs(lows[i] - last_low.price) >= min_distance:
                ts = timestamps[i]
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                swing_points.append(SwingPoint(
                    index=i,
                    price=float(lows[i]),
                    is_high=False,
                    timestamp=ts
                ))

    # Sort by index and limit to max swings
    swing_points.sort(key=lambda x: x.index)
    if len(swing_points) > STRUCTURE_MAX_SWINGS:
        swing_points = swing_points[-STRUCTURE_MAX_SWINGS:]

    return swing_points


def detect_market_structure(
    df: pd.DataFrame,
    col_map: dict,
    swing_points: Optional[List[SwingPoint]] = None,
    lookback: int = STRUCTURE_SWING_LOOKBACK,
    pip_size: float = 0.0001
) -> StructureSignal:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    Market Structure Concepts:
    - Uptrend: Higher Highs (HH) and Higher Lows (HL)
    - Downtrend: Lower Highs (LH) and Lower Lows (LL)

    BOS (Break of Structure):
    - Bullish BOS: Price breaks above previous swing high in uptrend
    - Bearish BOS: Price breaks below previous swing low in downtrend
    - Confirms trend continuation

    CHoCH (Change of Character):
    - Bullish CHoCH: First Higher Low after downtrend (trend reversal signal)
    - Bearish CHoCH: First Lower High after uptrend (trend reversal signal)
    - Early reversal signal, lower confidence than BOS

    Args:
        df: DataFrame with OHLCV data
        col_map: Column name mapping
        swing_points: Pre-computed swing points (optional)
        lookback: Swing detection lookback
        pip_size: Pip size for the instrument

    Returns:
        StructureSignal with detected structure type and details
    """
    # Detect swing points if not provided
    if swing_points is None:
        swing_points = detect_swing_points(df, col_map, lookback, pip_size=pip_size)

    # Need at least 4 swings to determine structure (2 highs + 2 lows minimum)
    if len(swing_points) < 4:
        return StructureSignal(
            structure_type=StructureType.NONE,
            direction="NONE",
            confidence=0.0,
            trigger_price=0.0,
            trigger_index=-1
        )

    h = col_map['high']
    l = col_map['low']
    c = col_map['close']

    current_idx = len(df) - 1
    current_high = float(df[h].iloc[-1])
    current_low = float(df[l].iloc[-1])
    current_close = float(df[c].iloc[-1])

    # Separate swing highs and lows
    swing_highs = [sp for sp in swing_points if sp.is_high]
    swing_lows = [sp for sp in swing_points if not sp.is_high]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return StructureSignal(
            structure_type=StructureType.NONE,
            direction="NONE",
            confidence=0.0,
            trigger_price=0.0,
            trigger_index=-1
        )

    # Get the last few swing points
    last_sh1 = swing_highs[-1]  # Most recent swing high
    last_sh2 = swing_highs[-2]  # Second most recent swing high
    last_sl1 = swing_lows[-1]   # Most recent swing low
    last_sl2 = swing_lows[-2]   # Second most recent swing low

    # Determine current trend state
    # Higher Highs + Higher Lows = Bullish
    # Lower Highs + Lower Lows = Bearish
    hh = last_sh1.price > last_sh2.price  # Higher High
    hl = last_sl1.price > last_sl2.price  # Higher Low
    lh = last_sh1.price < last_sh2.price  # Lower High
    ll = last_sl1.price < last_sl2.price  # Lower Low

    if hh and hl:
        trend_state = TrendState.BULLISH
    elif lh and ll:
        trend_state = TrendState.BEARISH
    else:
        trend_state = TrendState.UNKNOWN

    # ============================================================
    # Check for BOS (Break of Structure) - Trend Continuation
    # ============================================================

    # Bullish BOS: Current price breaks above most recent swing high in uptrend
    if trend_state == TrendState.BULLISH and current_high > last_sh1.price:
        return StructureSignal(
            structure_type=StructureType.BOS_BULLISH,
            direction="BUY",
            confidence=STRUCTURE_BOS_CONFIDENCE,
            trigger_price=last_sh1.price,
            trigger_index=current_idx,
            previous_swing=last_sh1
        )

    # Bearish BOS: Current price breaks below most recent swing low in downtrend
    if trend_state == TrendState.BEARISH and current_low < last_sl1.price:
        return StructureSignal(
            structure_type=StructureType.BOS_BEARISH,
            direction="SELL",
            confidence=STRUCTURE_BOS_CONFIDENCE,
            trigger_price=last_sl1.price,
            trigger_index=current_idx,
            previous_swing=last_sl1
        )

    # ============================================================
    # Check for CHoCH (Change of Character) - Trend Reversal
    # ============================================================

    # Bullish CHoCH: In downtrend, current bar forms a higher low than previous swing low
    # This signals potential trend reversal from bearish to bullish
    if trend_state == TrendState.BEARISH:
        # Check if current low is forming a higher low (above previous swing low)
        if current_low > last_sl1.price:
            return StructureSignal(
                structure_type=StructureType.CHOCH_BULLISH,
                direction="BUY",
                confidence=STRUCTURE_CHOCH_CONFIDENCE,
                trigger_price=last_sl1.price,
                trigger_index=current_idx,
                previous_swing=last_sl1
            )

    # Bearish CHoCH: In uptrend, current bar forms a lower high than previous swing high
    # This signals potential trend reversal from bullish to bearish
    if trend_state == TrendState.BULLISH:
        # Check if current high is forming a lower high (below previous swing high)
        if current_high < last_sh1.price:
            return StructureSignal(
                structure_type=StructureType.CHOCH_BEARISH,
                direction="SELL",
                confidence=STRUCTURE_CHOCH_CONFIDENCE,
                trigger_price=last_sh1.price,
                trigger_index=current_idx,
                previous_swing=last_sh1
            )

    # No structure signal
    return StructureSignal(
        structure_type=StructureType.NONE,
        direction="NONE",
        confidence=0.0,
        trigger_price=0.0,
        trigger_index=-1
    )


class MarketStructureFilter:
    """
    Layer 8: Market Structure Filter (BOS/CHoCH Detection)

    Filters trades based on market structure analysis:
    - BOS (Break of Structure): Trend continuation confirmation
    - CHoCH (Change of Character): Early trend reversal signal

    Usage:
    - Only take trades after BOS confirmation in trade direction
    - Or use CHoCH for early reversal entries (riskier but higher reward)

    This filter CAN block trades if structure doesn't support direction.
    """

    def __init__(self, pip_size: float = 0.0001):
        """
        Initialize market structure filter.

        Args:
            pip_size: Pip size for the instrument (0.0001 for GBPUSD)
        """
        self.pip_size = pip_size
        self.swing_points: List[SwingPoint] = []
        self.last_signal: Optional[StructureSignal] = None
        self.current_trend: TrendState = TrendState.UNKNOWN
        self.signal_history: List[Tuple[datetime, StructureSignal]] = []

        # Statistics
        self.bos_bullish_count = 0
        self.bos_bearish_count = 0
        self.choch_bullish_count = 0
        self.choch_bearish_count = 0
        self.no_signal_count = 0
        self.trades_allowed = 0
        self.trades_blocked = 0

    def update(self, df: pd.DataFrame, col_map: dict,
               timestamp: Optional[datetime] = None) -> StructureSignal:
        """
        Update market structure analysis with new price data.

        Args:
            df: DataFrame with OHLCV data (should include recent bars)
            col_map: Column name mapping
            timestamp: Current timestamp (optional)

        Returns:
            Current StructureSignal
        """
        # Detect swing points
        self.swing_points = detect_swing_points(
            df, col_map,
            lookback=STRUCTURE_SWING_LOOKBACK,
            pip_size=self.pip_size
        )

        # Detect market structure
        self.last_signal = detect_market_structure(
            df, col_map,
            swing_points=self.swing_points,
            pip_size=self.pip_size
        )

        # Update statistics
        if self.last_signal.structure_type == StructureType.BOS_BULLISH:
            self.bos_bullish_count += 1
        elif self.last_signal.structure_type == StructureType.BOS_BEARISH:
            self.bos_bearish_count += 1
        elif self.last_signal.structure_type == StructureType.CHOCH_BULLISH:
            self.choch_bullish_count += 1
        elif self.last_signal.structure_type == StructureType.CHOCH_BEARISH:
            self.choch_bearish_count += 1
        else:
            self.no_signal_count += 1

        # Update trend state
        if self.last_signal.structure_type in [StructureType.BOS_BULLISH, StructureType.CHOCH_BULLISH]:
            self.current_trend = TrendState.BULLISH
        elif self.last_signal.structure_type in [StructureType.BOS_BEARISH, StructureType.CHOCH_BEARISH]:
            self.current_trend = TrendState.BEARISH

        # Store in history
        if timestamp:
            self.signal_history.append((timestamp, self.last_signal))
            # Keep last 50 signals
            if len(self.signal_history) > 50:
                self.signal_history = self.signal_history[-50:]

        return self.last_signal

    def check_trade(self, direction: str,
                    allow_choch: bool = True) -> Tuple[bool, int, float, str]:
        """
        Check if trade is allowed based on market structure.

        Args:
            direction: "BUY" or "SELL"
            allow_choch: Whether to allow CHoCH signals (riskier reversals)

        Returns:
            Tuple of (can_trade, quality_adjustment, size_multiplier, reason)
        """
        if self.last_signal is None:
            self.trades_allowed += 1
            return True, 0, 1.0, "NO_STRUCTURE_DATA"

        signal = self.last_signal

        # ============================================================
        # BOS signals - Strong confirmation, reduce quality requirement
        # ============================================================

        # Bullish BOS + BUY = Strong confirmation
        if signal.structure_type == StructureType.BOS_BULLISH and direction == "BUY":
            self.trades_allowed += 1
            return True, -STRUCTURE_QUALITY_BOOST_BOS, 1.0, \
                   f"BOS_BULLISH_CONFIRMED (conf={signal.confidence:.0%})"

        # Bearish BOS + SELL = Strong confirmation
        if signal.structure_type == StructureType.BOS_BEARISH and direction == "SELL":
            self.trades_allowed += 1
            return True, -STRUCTURE_QUALITY_BOOST_BOS, 1.0, \
                   f"BOS_BEARISH_CONFIRMED (conf={signal.confidence:.0%})"

        # ============================================================
        # CHoCH signals - Reversal entries (if allowed)
        # ============================================================

        if allow_choch:
            # Bullish CHoCH + BUY = Early reversal entry
            if signal.structure_type == StructureType.CHOCH_BULLISH and direction == "BUY":
                self.trades_allowed += 1
                return True, 0, STRUCTURE_CHOCH_SIZE_MULT, \
                       f"CHOCH_BULLISH_REVERSAL (conf={signal.confidence:.0%})"

            # Bearish CHoCH + SELL = Early reversal entry
            if signal.structure_type == StructureType.CHOCH_BEARISH and direction == "SELL":
                self.trades_allowed += 1
                return True, 0, STRUCTURE_CHOCH_SIZE_MULT, \
                       f"CHOCH_BEARISH_REVERSAL (conf={signal.confidence:.0%})"

        # ============================================================
        # Counter-structure trades - Block or heavily penalize
        # ============================================================

        # BOS in opposite direction
        if signal.structure_type == StructureType.BOS_BULLISH and direction == "SELL":
            if STRUCTURE_BLOCK_COUNTER:
                self.trades_blocked += 1
                return False, 0, 0, "BOS_BULLISH_BLOCKS_SELL"
            else:
                self.trades_allowed += 1
                return True, STRUCTURE_COUNTER_QUALITY_PENALTY, STRUCTURE_COUNTER_SIZE_MULT, \
                       f"COUNTER_BOS_BULLISH (penalized)"

        if signal.structure_type == StructureType.BOS_BEARISH and direction == "BUY":
            if STRUCTURE_BLOCK_COUNTER:
                self.trades_blocked += 1
                return False, 0, 0, "BOS_BEARISH_BLOCKS_BUY"
            else:
                self.trades_allowed += 1
                return True, STRUCTURE_COUNTER_QUALITY_PENALTY, STRUCTURE_COUNTER_SIZE_MULT, \
                       f"COUNTER_BOS_BEARISH (penalized)"

        # CHoCH in opposite direction
        if signal.structure_type == StructureType.CHOCH_BULLISH and direction == "SELL":
            if STRUCTURE_BLOCK_COUNTER:
                self.trades_blocked += 1
                return False, 0, 0, "CHOCH_BULLISH_BLOCKS_SELL"
            else:
                self.trades_allowed += 1
                return True, STRUCTURE_COUNTER_QUALITY_PENALTY, STRUCTURE_COUNTER_SIZE_MULT, \
                       f"COUNTER_CHOCH_BULLISH (penalized)"

        if signal.structure_type == StructureType.CHOCH_BEARISH and direction == "BUY":
            if STRUCTURE_BLOCK_COUNTER:
                self.trades_blocked += 1
                return False, 0, 0, "CHOCH_BEARISH_BLOCKS_BUY"
            else:
                self.trades_allowed += 1
                return True, STRUCTURE_COUNTER_QUALITY_PENALTY, STRUCTURE_COUNTER_SIZE_MULT, \
                       f"COUNTER_CHOCH_BEARISH (penalized)"

        # ============================================================
        # No clear structure signal - Penalize but allow
        # ============================================================

        if signal.structure_type == StructureType.NONE:
            self.trades_allowed += 1
            return True, STRUCTURE_QUALITY_PENALTY_NO_BOS, 0.9, "NO_STRUCTURE_CONFIRMATION"

        # Default: Allow with slight penalty
        self.trades_allowed += 1
        return True, 5, 0.9, f"STRUCTURE_UNCLEAR ({signal.structure_type.value})"

    def get_stats(self) -> Dict[str, Any]:
        """Get current filter statistics"""
        total_signals = (self.bos_bullish_count + self.bos_bearish_count +
                        self.choch_bullish_count + self.choch_bearish_count)

        return {
            "current_trend": self.current_trend.value,
            "last_signal": self.last_signal.structure_type.value if self.last_signal else "none",
            "swing_count": len(self.swing_points),
            "bos_bullish": self.bos_bullish_count,
            "bos_bearish": self.bos_bearish_count,
            "choch_bullish": self.choch_bullish_count,
            "choch_bearish": self.choch_bearish_count,
            "no_signal": self.no_signal_count,
            "total_signals": total_signals,
            "trades_allowed": self.trades_allowed,
            "trades_blocked": self.trades_blocked,
            "history_size": len(self.signal_history)
        }

    def reset_for_month(self, month: int):
        """
        Soft reset for new month.

        Note: We keep swing points and trend state as market structure
        persists across month boundaries. Only reset monthly statistics.
        """
        # Reset monthly statistics only
        self.bos_bullish_count = 0
        self.bos_bearish_count = 0
        self.choch_bullish_count = 0
        self.choch_bearish_count = 0
        self.no_signal_count = 0
        self.trades_allowed = 0
        self.trades_blocked = 0
