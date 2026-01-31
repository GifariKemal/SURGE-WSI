"""
Trading Filters - Shared filter classes for live and backtest
=============================================================

This module contains the shared trading filter logic used by both
executor.py (live trading) and backtest.py.

Classes:
- IntraMonthRiskManager: Layer 3 - Dynamic risk based on monthly performance
- PatternBasedFilter: Layer 4 - Choppy market detection
- ChoppinessFilter: Layer 5 - Choppiness Index based filter (NEW!)

References:
- Choppiness Index by E.W. Dreiss (1993)
- https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/
- https://www.quantifiedstrategies.com/choppiness-index/

Author: SURIOTA Team
"""

from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
import math


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

# Monthly tradeable percentage (historical data)
# Higher % = better month, lower quality requirement
MONTHLY_TRADEABLE_PCT = {
    # 2024
    (2024, 1): 67, (2024, 2): 55, (2024, 3): 70, (2024, 4): 80,
    (2024, 5): 65, (2024, 6): 72, (2024, 7): 78, (2024, 8): 60,
    (2024, 9): 75, (2024, 10): 58, (2024, 11): 68, (2024, 12): 45,
    # 2025 - Keep original values, use technique improvements instead
    (2025, 1): 65, (2025, 2): 55, (2025, 3): 70, (2025, 4): 70,
    (2025, 5): 62, (2025, 6): 68, (2025, 7): 78, (2025, 8): 65,
    (2025, 9): 72, (2025, 10): 58, (2025, 11): 66, (2025, 12): 60,
    # 2026
    (2026, 1): 65, (2026, 2): 55, (2026, 3): 70, (2026, 4): 70,
    (2026, 5): 62, (2026, 6): 68, (2026, 7): 78, (2026, 8): 65,
    (2026, 9): 72, (2026, 10): 58, (2026, 11): 66, (2026, 12): 60,
}


def get_monthly_quality_adjustment(dt: datetime) -> int:
    """
    Get quality adjustment based on month's historical tradeable percentage.

    Args:
        dt: Datetime to check

    Returns:
        Quality adjustment (higher = stricter filter)
    """
    key = (dt.year, dt.month)
    tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)  # Default 70%

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

        # Track recovery progress (same as original - no _analyze_patterns call)
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

    def _analyze_patterns(self):
        """Analyze patterns and update halt status"""
        if self.trade_count < WARMUP_TRADES:
            return  # Still in warmup

        recent = self.trade_history[-ROLLING_WINDOW:] if len(self.trade_history) >= ROLLING_WINDOW else self.trade_history

        if len(recent) < ROLLING_WINDOW:
            return

        # Calculate rolling win rate
        wins = sum(1 for _, pnl, _ in recent if pnl > 0)
        rolling_wr = wins / len(recent)

        # Check for halt conditions
        if rolling_wr < ROLLING_WR_HALT:
            if not self.is_halted:
                self.is_halted = True
                self.in_recovery = True  # Immediately enter recovery mode
                self.recovery_wins = 0
                self.halt_reason = f"Rolling WR {rolling_wr:.0%} < {ROLLING_WR_HALT:.0%}"
                logger.warning(f"[Layer4] HALT: {self.halt_reason}")
                if self.state_manager:
                    self.state_manager.set_halted(self.halt_reason)
            return

        # Check both directions failing (match original: count losses directly in window)
        if len(self.trade_history) >= BOTH_DIRECTIONS_FAIL_THRESHOLD * 2:
            recent_window = self.trade_history[-BOTH_DIRECTIONS_FAIL_THRESHOLD*2:]

            # Count losses directly (original logic from _get_rolling_stats)
            buy_losses = sum(1 for d, p, _ in recent_window if d == 'BUY' and p < 0)
            sell_losses = sum(1 for d, p, _ in recent_window if d == 'SELL' and p < 0)

            if buy_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD and sell_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD:
                if not self.is_halted:
                    self.is_halted = True
                    self.in_recovery = True  # Immediately enter recovery mode
                    self.recovery_wins = 0
                    self.halt_reason = f"Both directions failing: BUY {buy_losses}/{BOTH_DIRECTIONS_FAIL_THRESHOLD}, SELL {sell_losses}/{BOTH_DIRECTIONS_FAIL_THRESHOLD}"
                    logger.warning(f"[Layer4] HALT: {self.halt_reason}")
                    if self.state_manager:
                        self.state_manager.set_halted(self.halt_reason)
                return

        # Note: Recovery is handled via recovery_wins in record_trade
        # No automatic clearing based on conditions - must win during recovery

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
        Soft reset for new month (matches original exactly).

        Only clears halt state if we had a recovery win.
        Trade history is kept for pattern analysis continuity.
        """
        self.current_month = month
        self.probe_taken = False
        # Don't reset trade_history - we want to learn across months!
        # Only reset halt if we had recovery
        if self.in_recovery and self.recovery_wins >= RECOVERY_WIN_REQUIRED:
            logger.info(f"[Layer4] Month reset: clearing halt after recovery ({self.recovery_wins} wins)")
            self.is_halted = False
            self.in_recovery = False
            self.recovery_wins = 0
            self.halt_reason = ""
            if self.state_manager:
                self.state_manager.clear_halt()
                self.state_manager.exit_recovery_mode()

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
# Layer 5: Choppiness Index Filter
# Based on E.W. Dreiss (1993) - Chaos Theory Applied to Markets
# Reference: https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/
# ============================================================

# Choppiness Index thresholds (DEPRECATED - use DirectionalMomentumFilter instead)
# Kept for backwards compatibility but disabled by default
CHOP_PERIOD = 14                   # Lookback period for CHOP calculation
CHOP_CHOPPY_THRESHOLD = 61.8       # Above this = choppy market
CHOP_TRENDING_THRESHOLD = 38.2     # Below this = trending market
CHOP_EXTREME_CHOPPY = 70.0         # Extreme choppiness - very high risk
CHOP_SIZE_REDUCTION_CHOPPY = 0.5   # Reduce size to 50% in choppy conditions
CHOP_SIZE_REDUCTION_EXTREME = 0.3  # Reduce size to 30% in extreme choppy
CHOP_QUALITY_ADD_CHOPPY = 10       # Add +10 quality requirement when choppy
CHOP_QUALITY_ADD_EXTREME = 20      # Add +20 quality requirement when extreme choppy


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
    Layer 5: Choppiness Index Based Market Filter

    Uses the Choppiness Index to detect choppy/ranging markets and adjust
    trading parameters accordingly. Instead of blocking trades entirely,
    it reduces position size and increases quality requirements.

    Key benefits:
    - Detects choppy markets BEFORE losses accumulate
    - Allows trades in choppy markets with reduced risk
    - No missed opportunities (unlike pattern-based halt)

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


# ============================================================
# Layer 6: Directional Momentum Filter (NEW!)
# Detects when one direction is consistently failing
# ============================================================

# Directional Momentum thresholds
# ADX regime detection is primary protection. This is secondary backup.
DIR_CONSEC_LOSS_CAUTION = 2    # 2 consecutive losses: early caution
DIR_CONSEC_LOSS_WARNING = 3    # 3 consecutive losses: warning
DIR_CONSEC_LOSS_EXTREME = 4    # 4+ consecutive losses: extreme
DIR_SIZE_CAUTION = 0.6         # 60% size at caution level
DIR_SIZE_WARNING = 0.4         # 40% size at warning level
DIR_SIZE_EXTREME = 0.25        # 25% size at extreme level
DIR_QUALITY_CAUTION = 5        # +5 quality at caution
DIR_QUALITY_WARNING = 12       # +12 quality at warning
DIR_QUALITY_EXTREME = 20       # +20 quality at extreme
DIR_RECOVERY_WINS = 1          # 1 win resets the counter


class DirectionalMomentumFilter:
    """
    Layer 6: Directional Momentum Filter

    Detects when one direction (BUY or SELL) is consistently failing
    and adjusts position size accordingly. Unlike pattern filter's
    both_directions_fail, this triggers on SINGLE direction failure.

    Key insight: In April 2025, all 4 losses were SELL direction.
    The pattern filter didn't catch this because it looks for BOTH
    directions failing. This filter catches single-direction failures.

    This filter NEVER blocks trades (per user requirement).
    It only reduces size and adds quality requirements.
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
