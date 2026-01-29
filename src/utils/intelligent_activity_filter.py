"""Intelligent Activity Filter (IAF)
=====================================

Replaces fixed Kill Zone with intelligent market activity detection.

Core Concept:
- Detect if market is MOVING using multiple indicators
- Trade when there's activity, skip when market is quiet
- No fixed time restrictions (except weekends)

Components:
1. Velocity Detection (via Kalman)
2. ATR-based Volatility
3. Range Analysis
4. Momentum Scoring

Author: SURIOTA Team
Date: 2026-01-29
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Dict
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class MarketActivity(Enum):
    """Market activity levels"""
    SURGING = "surging"      # Strong directional move - BEST
    ACTIVE = "active"        # Good activity - TRADE
    MODERATE = "moderate"    # Some activity - TRADE with caution
    QUIET = "quiet"          # Low activity - SKIP
    DEAD = "dead"            # No activity - SKIP


@dataclass
class ActivityResult:
    """Result from Intelligent Activity Filter"""
    should_trade: bool
    activity: MarketActivity
    score: float  # 0-100

    # Component scores
    velocity_score: float    # From Kalman velocity
    atr_score: float         # From ATR
    range_score: float       # From recent range
    momentum_score: float    # From price momentum

    # Metrics
    velocity: float          # Kalman velocity (pips/bar)
    atr_pips: float          # ATR in pips
    range_pips: float        # Current range in pips
    momentum: float          # Momentum indicator

    # Adaptive outputs
    quality_threshold: float  # Suggested quality threshold
    lot_multiplier: float     # Suggested lot adjustment

    reason: str

    def to_dict(self) -> Dict:
        return {
            'should_trade': self.should_trade,
            'activity': self.activity.value,
            'score': round(self.score, 1),
            'velocity_score': round(self.velocity_score, 1),
            'atr_score': round(self.atr_score, 1),
            'range_score': round(self.range_score, 1),
            'momentum_score': round(self.momentum_score, 1),
            'velocity': round(self.velocity, 2),
            'atr_pips': round(self.atr_pips, 1),
            'range_pips': round(self.range_pips, 1),
            'momentum': round(self.momentum, 4),
            'quality_threshold': self.quality_threshold,
            'lot_multiplier': self.lot_multiplier,
            'reason': self.reason
        }

    def get_emoji(self) -> str:
        return {
            MarketActivity.SURGING: "🚀",
            MarketActivity.ACTIVE: "🟢",
            MarketActivity.MODERATE: "🟡",
            MarketActivity.QUIET: "🟠",
            MarketActivity.DEAD: "🔴"
        }.get(self.activity, "⚪")


class IntelligentActivityFilter:
    """Intelligent Market Activity Filter

    Replaces Kill Zone with velocity-based activity detection.
    Trade when market MOVES, skip when QUIET.

    Key Features:
    - Uses Kalman velocity for momentum detection
    - ATR for volatility measurement
    - Adaptive quality thresholds based on activity
    - No fixed time restrictions
    """

    def __init__(
        self,
        # Velocity thresholds (pips per bar)
        min_velocity_pips: float = 2.0,       # Minimum velocity to consider active
        high_velocity_pips: float = 5.0,      # High velocity = surging

        # ATR thresholds (pips)
        min_atr_pips: float = 5.0,            # Minimum ATR
        high_atr_pips: float = 15.0,          # High ATR = volatile

        # Range thresholds (pips)
        min_range_pips: float = 3.0,          # Minimum bar range

        # Activity thresholds
        activity_threshold: float = 40.0,     # Minimum score to trade
        surging_threshold: float = 80.0,      # Score for SURGING
        active_threshold: float = 60.0,       # Score for ACTIVE
        moderate_threshold: float = 40.0,     # Score for MODERATE

        # Lookback periods
        atr_period: int = 14,
        velocity_period: int = 5,             # Bars to smooth velocity
        momentum_period: int = 10,            # Bars for momentum

        # Symbol settings
        pip_size: float = 0.0001,

        # Skip settings
        skip_weekends: bool = True,
        skip_friday_late: bool = True,
    ):
        self.min_velocity_pips = min_velocity_pips
        self.high_velocity_pips = high_velocity_pips
        self.min_atr_pips = min_atr_pips
        self.high_atr_pips = high_atr_pips
        self.min_range_pips = min_range_pips

        self.activity_threshold = activity_threshold
        self.surging_threshold = surging_threshold
        self.active_threshold = active_threshold
        self.moderate_threshold = moderate_threshold

        self.atr_period = atr_period
        self.velocity_period = velocity_period
        self.momentum_period = momentum_period

        self.pip_size = pip_size
        self.skip_weekends = skip_weekends
        self.skip_friday_late = skip_friday_late

        # History for calculations
        self._close_history: List[float] = []
        self._tr_history: List[float] = []
        self._velocity_history: List[float] = []

        # Last Kalman velocity (updated externally)
        self._kalman_velocity: float = 0.0

        logger.info(f"IntelligentActivityFilter initialized: "
                   f"threshold={activity_threshold}, min_vel={min_velocity_pips}p")

    def update_kalman_velocity(self, velocity: float):
        """Update with Kalman velocity from external Kalman filter

        Args:
            velocity: Velocity from KalmanState (in price units)
        """
        velocity_pips = velocity / self.pip_size
        self._kalman_velocity = velocity_pips
        self._velocity_history.append(velocity_pips)

        # Keep limited history
        if len(self._velocity_history) > self.velocity_period * 3:
            self._velocity_history = self._velocity_history[-self.velocity_period * 3:]

    def update(self, high: float, low: float, close: float):
        """Update with new bar data

        Args:
            high: Bar high
            low: Bar low
            close: Bar close
        """
        self._close_history.append(close)

        # Calculate True Range
        if len(self._close_history) > 1:
            prev_close = self._close_history[-2]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
        else:
            tr = high - low

        self._tr_history.append(tr)

        # Keep limited history
        max_history = max(self.atr_period, self.momentum_period) * 3
        if len(self._close_history) > max_history:
            self._close_history = self._close_history[-max_history:]
            self._tr_history = self._tr_history[-max_history:]

    def _calculate_atr(self) -> float:
        """Calculate ATR in pips"""
        if len(self._tr_history) < self.atr_period:
            return 0.0

        atr = np.mean(self._tr_history[-self.atr_period:])
        return atr / self.pip_size

    def _calculate_momentum(self) -> float:
        """Calculate price momentum

        Returns momentum as price change over period
        """
        if len(self._close_history) < self.momentum_period:
            return 0.0

        current = self._close_history[-1]
        past = self._close_history[-self.momentum_period]

        return (current - past) / self.pip_size

    def _get_avg_velocity(self) -> float:
        """Get smoothed velocity"""
        if not self._velocity_history:
            return 0.0

        n = min(len(self._velocity_history), self.velocity_period)
        return np.mean(np.abs(self._velocity_history[-n:]))

    def is_market_open(self, dt: datetime) -> Tuple[bool, str]:
        """Check if market is open (basic weekend check)"""
        weekday = dt.weekday()
        hour = dt.hour

        # Saturday
        if weekday == 5 and self.skip_weekends:
            return False, "Weekend (Saturday)"

        # Sunday before 22:00
        if weekday == 6 and hour < 22 and self.skip_weekends:
            return False, "Weekend (Sunday)"

        # Friday late
        if weekday == 4 and hour >= 20 and self.skip_friday_late:
            return False, "Friday late session"

        return True, "Market open"

    def check(
        self,
        dt: datetime,
        current_high: float,
        current_low: float,
        current_close: float,
        kalman_velocity: Optional[float] = None
    ) -> ActivityResult:
        """Check market activity

        Args:
            dt: Current datetime
            current_high: Current bar high
            current_low: Current bar low
            current_close: Current bar close
            kalman_velocity: Optional Kalman velocity (in price units)

        Returns:
            ActivityResult with detailed breakdown
        """
        # Update history
        self.update(current_high, current_low, current_close)

        # Update Kalman velocity if provided
        if kalman_velocity is not None:
            self.update_kalman_velocity(kalman_velocity)

        # Check market hours
        is_open, reason = self.is_market_open(dt)
        if not is_open:
            return ActivityResult(
                should_trade=False,
                activity=MarketActivity.DEAD,
                score=0,
                velocity_score=0,
                atr_score=0,
                range_score=0,
                momentum_score=0,
                velocity=0,
                atr_pips=0,
                range_pips=0,
                momentum=0,
                quality_threshold=100,
                lot_multiplier=0,
                reason=reason
            )

        # Calculate metrics
        current_range_pips = (current_high - current_low) / self.pip_size
        atr_pips = self._calculate_atr()
        momentum = self._calculate_momentum()
        avg_velocity = self._get_avg_velocity()

        # =================================
        # VELOCITY SCORE (0-30 points)
        # =================================
        # Velocity shows if market is MOVING
        if avg_velocity >= self.high_velocity_pips:
            velocity_score = 30  # Strong movement
        elif avg_velocity >= self.min_velocity_pips:
            velocity_score = 20  # Good movement
        elif avg_velocity >= self.min_velocity_pips * 0.5:
            velocity_score = 10  # Some movement
        else:
            velocity_score = 0   # No movement

        # =================================
        # ATR SCORE (0-30 points)
        # =================================
        # ATR shows volatility/range opportunity
        if atr_pips >= self.high_atr_pips:
            atr_score = 30  # High volatility
        elif atr_pips >= self.min_atr_pips:
            atr_score = 20  # Good volatility
        elif atr_pips >= self.min_atr_pips * 0.5:
            atr_score = 10  # Low volatility
        else:
            atr_score = 0   # Dead market

        # =================================
        # RANGE SCORE (0-20 points)
        # =================================
        # Current bar range shows immediate activity
        if current_range_pips >= self.min_range_pips * 2:
            range_score = 20  # Strong bar
        elif current_range_pips >= self.min_range_pips:
            range_score = 15  # Good bar
        elif current_range_pips >= self.min_range_pips * 0.5:
            range_score = 8   # Small bar
        else:
            range_score = 0   # Doji/no movement

        # =================================
        # MOMENTUM SCORE (0-20 points)
        # =================================
        # Momentum shows directional bias
        abs_momentum = abs(momentum)
        if abs_momentum >= 15:  # 15 pips in momentum_period bars
            momentum_score = 20  # Strong trend
        elif abs_momentum >= 10:
            momentum_score = 15  # Good trend
        elif abs_momentum >= 5:
            momentum_score = 10  # Some trend
        else:
            momentum_score = 5   # Ranging

        # =================================
        # TOTAL SCORE
        # =================================
        total_score = velocity_score + atr_score + range_score + momentum_score
        total_score = max(0, min(100, total_score))

        # Determine activity level
        if total_score >= self.surging_threshold:
            activity = MarketActivity.SURGING
        elif total_score >= self.active_threshold:
            activity = MarketActivity.ACTIVE
        elif total_score >= self.moderate_threshold:
            activity = MarketActivity.MODERATE
        elif total_score >= 20:
            activity = MarketActivity.QUIET
        else:
            activity = MarketActivity.DEAD

        # Should trade?
        should_trade = total_score >= self.activity_threshold

        # =================================
        # ADAPTIVE PARAMETERS
        # =================================
        # Quality threshold based on activity
        if activity == MarketActivity.SURGING:
            quality_threshold = 60.0  # Lower quality OK in strong moves
            lot_multiplier = 1.2      # Slightly larger lot
        elif activity == MarketActivity.ACTIVE:
            quality_threshold = 65.0  # Standard quality
            lot_multiplier = 1.0      # Normal lot
        elif activity == MarketActivity.MODERATE:
            quality_threshold = 70.0  # Higher quality needed
            lot_multiplier = 0.8      # Smaller lot
        else:
            quality_threshold = 75.0  # Very high quality only
            lot_multiplier = 0.5      # Minimal lot

        # Build reason
        if should_trade:
            components = []
            if velocity_score >= 20:
                components.append(f"vel={avg_velocity:.1f}p")
            if atr_score >= 20:
                components.append(f"atr={atr_pips:.1f}p")
            if momentum_score >= 15:
                components.append(f"mom={momentum:.1f}p")
            reason = f"ACTIVE ({', '.join(components) if components else 'score=' + str(int(total_score))})"
        else:
            issues = []
            if velocity_score < 10:
                issues.append("low velocity")
            if atr_score < 10:
                issues.append("low ATR")
            if range_score < 8:
                issues.append("small range")
            reason = f"SKIP: {', '.join(issues) if issues else 'below threshold'}"

        return ActivityResult(
            should_trade=should_trade,
            activity=activity,
            score=total_score,
            velocity_score=velocity_score,
            atr_score=atr_score,
            range_score=range_score,
            momentum_score=momentum_score,
            velocity=avg_velocity,
            atr_pips=atr_pips,
            range_pips=current_range_pips,
            momentum=momentum,
            quality_threshold=quality_threshold,
            lot_multiplier=lot_multiplier,
            reason=reason
        )

    def warmup(self, df: pd.DataFrame, kalman_states: Optional[List] = None):
        """Warmup filter with historical data

        Args:
            df: Historical OHLCV DataFrame
            kalman_states: Optional list of KalmanState objects with velocity
        """
        for i, (_, row) in enumerate(df.iterrows()):
            h = row.get('High', row.get('high', 0))
            l = row.get('Low', row.get('low', 0))
            c = row.get('Close', row.get('close', 0))
            self.update(h, l, c)

            # Use Kalman velocity if available
            if kalman_states and i < len(kalman_states):
                self.update_kalman_velocity(kalman_states[i].velocity)

    def reset(self):
        """Reset filter state"""
        self._close_history = []
        self._tr_history = []
        self._velocity_history = []
        self._kalman_velocity = 0.0

    def format_status(self, result: ActivityResult) -> str:
        """Format activity status for display"""
        emoji = result.get_emoji()
        lines = [
            f"{emoji} Activity: {result.activity.value.upper()} ({result.score:.0f}/100)",
            f"   Velocity: {result.velocity_score:.0f}/30 ({result.velocity:.1f} pips/bar)",
            f"   ATR: {result.atr_score:.0f}/30 ({result.atr_pips:.1f} pips)",
            f"   Range: {result.range_score:.0f}/20 ({result.range_pips:.1f} pips)",
            f"   Momentum: {result.momentum_score:.0f}/20 ({result.momentum:.1f} pips)",
            f"   → Quality threshold: {result.quality_threshold:.0f}",
            f"   → Lot multiplier: {result.lot_multiplier:.1f}x",
        ]
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_intelligent_filter(
    mode: str = "balanced"
) -> IntelligentActivityFilter:
    """Create filter with preset configurations

    Args:
        mode: 'conservative', 'balanced', 'aggressive'

    Returns:
        Configured IntelligentActivityFilter
    """
    if mode == "conservative":
        return IntelligentActivityFilter(
            min_velocity_pips=3.0,
            min_atr_pips=7.0,
            activity_threshold=50.0,
        )
    elif mode == "aggressive":
        return IntelligentActivityFilter(
            min_velocity_pips=1.5,
            min_atr_pips=4.0,
            activity_threshold=35.0,
        )
    else:  # balanced
        return IntelligentActivityFilter(
            min_velocity_pips=2.0,
            min_atr_pips=5.0,
            activity_threshold=40.0,
        )
