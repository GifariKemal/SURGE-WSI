"""Dynamic Activity Filter
===========================

A more adaptive filter than fixed Kill Zones.
Detects market activity dynamically based on:
- Volatility (ATR-based)
- Price movement
- Spread conditions
- Market hours (weekend check only)

HYBRID MODE:
- Trade during Kill Zones (traditional) OR
- Trade when market shows high activity outside KZ

References:
- https://www.luxalgo.com/blog/volatility-strategies-in-algo-trading/
- https://www.quantifiedstrategies.com/atr-based-trading-strategy-with-python/

Author: SURIOTA Team
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Optional, List
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class ActivityLevel(Enum):
    """Market activity level"""
    HIGH = "high"           # Very active - optimal for trading
    MODERATE = "moderate"   # Active enough - tradeable
    LOW = "low"            # Low activity - caution
    DEAD = "dead"          # No activity - avoid


@dataclass
class ActivityScore:
    """Activity detection result"""
    is_active: bool
    score: float  # 0-100
    volatility_score: float
    range_score: float
    time_score: float
    reason: str
    level: ActivityLevel = ActivityLevel.LOW
    spread_pips: float = 0.0
    atr_pips: float = 0.0

    def to_dict(self):
        return {
            'is_active': self.is_active,
            'score': round(self.score, 1),
            'level': self.level.value,
            'volatility_score': round(self.volatility_score, 1),
            'range_score': round(self.range_score, 1),
            'time_score': round(self.time_score, 1),
            'atr_pips': round(self.atr_pips, 1),
            'spread_pips': round(self.spread_pips, 1),
            'reason': self.reason
        }

    def get_emoji(self) -> str:
        """Get emoji for activity level"""
        return {
            ActivityLevel.HIGH: "🟢",
            ActivityLevel.MODERATE: "🟡",
            ActivityLevel.LOW: "🟠",
            ActivityLevel.DEAD: "🔴"
        }.get(self.level, "⚪")


class DynamicActivityFilter:
    """Dynamic market activity filter
    
    Instead of fixed Kill Zones, this filter detects:
    1. Is the market actually moving? (volatility check)
    2. Is there enough range to trade? (range check)
    3. Is it a valid trading day? (weekend/holiday check)
    
    Benefits:
    - Captures opportunities in ANY session if volatility exists
    - Automatically avoids dead markets during "active" hours
    - More adaptive and suitable for automation
    """
    
    def __init__(
        self,
        min_atr_pips: float = 5.0,           # Minimum ATR in pips
        min_bar_range_pips: float = 3.0,      # Minimum bar range in pips
        volatility_lookback: int = 14,        # ATR lookback period
        activity_threshold: float = 40.0,     # Minimum score to trade
        pip_size: float = 0.0001,             # Pip size for pair
        skip_weekends: bool = True,           # Skip Saturday-Sunday
        skip_friday_late: bool = True,        # Skip Friday after 20:00 UTC
        boost_overlap_hours: bool = True      # Boost score during overlap hours
    ):
        """Initialize Dynamic Activity Filter
        
        Args:
            min_atr_pips: Minimum ATR in pips to consider market active
            min_bar_range_pips: Minimum single bar range in pips
            volatility_lookback: Bars to look back for ATR
            activity_threshold: Minimum activity score (0-100) to trade
            pip_size: Pip size for the currency pair
            skip_weekends: Don't trade on weekends
            skip_friday_late: Avoid late Friday trading
            boost_overlap_hours: Give bonus score during London-NY overlap
        """
        self.min_atr_pips = min_atr_pips
        self.min_bar_range_pips = min_bar_range_pips
        self.volatility_lookback = volatility_lookback
        self.activity_threshold = activity_threshold
        self.pip_size = pip_size
        self.skip_weekends = skip_weekends
        self.skip_friday_late = skip_friday_late
        self.boost_overlap_hours = boost_overlap_hours

        # State
        self._atr_history: List[float] = []
        self._range_history: List[float] = []

        # Hybrid mode settings
        self.hybrid_mode = True  # Kill Zone + Dynamic
        self.outside_kz_min_score = 70.0  # Higher threshold outside KZ

        # Spread settings
        self.max_spread_pips = 3.0
        self.ideal_spread_pips = 1.5

        logger.info(f"DynamicActivityFilter initialized: threshold={activity_threshold}, "
                   f"min_atr={min_atr_pips} pips, hybrid_mode=True")
        
    def update(self, high: float, low: float, close: float) -> None:
        """Update filter with new bar data
        
        Args:
            high: Bar high
            low: Bar low
            close: Bar close
        """
        bar_range = high - low
        self._range_history.append(bar_range)
        
        # Keep limited history
        if len(self._range_history) > self.volatility_lookback * 2:
            self._range_history = self._range_history[-self.volatility_lookback * 2:]
    
    def calculate_atr(self) -> float:
        """Calculate Average True Range from history
        
        Returns:
            ATR in price units
        """
        if len(self._range_history) < 2:
            return 0.0
        
        # Simplified ATR using just range (no previous close gaps for now)
        lookback = min(len(self._range_history), self.volatility_lookback)
        recent_ranges = self._range_history[-lookback:]
        return np.mean(recent_ranges)
    
    def is_market_open(self, dt: datetime) -> Tuple[bool, str]:
        """Check if market is open (not weekend)
        
        Args:
            dt: Datetime to check
            
        Returns:
            Tuple of (is_open, reason)
        """
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        hour = dt.hour
        
        # Saturday - always closed
        if weekday == 5:
            return False, "Weekend (Saturday)"
        
        # Sunday before 22:00 UTC - closed
        if weekday == 6 and hour < 22:
            return False, "Weekend (Sunday before market open)"
        
        # Friday after 20:00 UTC - high spread, avoid
        if self.skip_friday_late and weekday == 4 and hour >= 20:
            return False, "Friday late session (high spread risk)"
        
        return True, "Market open"
    
    def check_activity(
        self,
        dt: datetime,
        current_high: float,
        current_low: float,
        recent_df: Optional[pd.DataFrame] = None
    ) -> ActivityScore:
        """Check if market is active enough for trading
        
        Args:
            dt: Current datetime
            current_high: Current bar high
            current_low: Current bar low
            recent_df: Recent OHLCV data for volatility calculation
            
        Returns:
            ActivityScore with detailed breakdown
        """
        # Check market open first
        is_open, reason = self.is_market_open(dt)
        if not is_open:
            return ActivityScore(
                is_active=False,
                score=0,
                volatility_score=0,
                range_score=0,
                time_score=0,
                reason=reason
            )
        
        # Calculate current bar range
        current_range = current_high - current_low
        current_range_pips = current_range / self.pip_size
        
        # Update history if we have data
        if recent_df is not None and len(recent_df) > 0:
            for _, row in recent_df.tail(self.volatility_lookback).iterrows():
                bar_range = row.get('high', 0) - row.get('low', 0)
                if bar_range > 0:
                    self._range_history.append(bar_range)
            # Dedupe to avoid double-counting
            self._range_history = self._range_history[-self.volatility_lookback * 2:]
        
        # Calculate ATR
        atr = self.calculate_atr()
        atr_pips = atr / self.pip_size
        
        # ============================
        # VOLATILITY SCORE (0-40 points)
        # ============================
        if atr_pips >= self.min_atr_pips * 2:
            volatility_score = 40  # Very active
        elif atr_pips >= self.min_atr_pips:
            volatility_score = 30  # Adequately active
        elif atr_pips >= self.min_atr_pips * 0.5:
            volatility_score = 15  # Low activity
        else:
            volatility_score = 5   # Dead market
        
        # ============================
        # RANGE SCORE (0-30 points)
        # ============================
        if current_range_pips >= self.min_bar_range_pips * 2:
            range_score = 30  # Good movement
        elif current_range_pips >= self.min_bar_range_pips:
            range_score = 20  # Adequate movement
        elif current_range_pips >= self.min_bar_range_pips * 0.5:
            range_score = 10  # Low movement
        else:
            range_score = 0   # No movement
        
        # ============================
        # TIME SCORE (0-30 points)
        # ============================
        hour = dt.hour
        weekday = dt.weekday()
        
        # Base time score
        time_score = 15  # Default
        
        # Overlap hours bonus (12:00-16:00 UTC)
        if self.boost_overlap_hours and 12 <= hour < 16:
            time_score = 30
        # Primary sessions (07:00-17:00 UTC)
        elif 7 <= hour < 17:
            time_score = 25
        # Asian session (22:00-07:00 UTC)
        elif hour >= 22 or hour < 7:
            time_score = 15
        # Off-peak
        else:
            time_score = 10
        
        # Weekday adjustments
        if weekday in [1, 2, 3]:  # Tue-Wed-Thu
            time_score = min(30, time_score + 5)
        elif weekday == 0:  # Monday
            time_score = max(0, time_score - 5)  # Gap risk
        elif weekday == 4:  # Friday
            if hour >= 15:
                time_score = max(0, time_score - 10)  # Weekend risk
        
        # ============================
        # TOTAL SCORE
        # ============================
        total_score = volatility_score + range_score + time_score
        total_score = max(0, min(100, total_score))
        
        # Determine activity level
        if total_score >= 80:
            level = ActivityLevel.HIGH
        elif total_score >= self.activity_threshold:
            level = ActivityLevel.MODERATE
        elif total_score >= 25:
            level = ActivityLevel.LOW
        else:
            level = ActivityLevel.DEAD

        # Determine if active
        is_active = total_score >= self.activity_threshold

        # Build reason
        if is_active:
            reason = f"Active (score={total_score:.0f}, ATR={atr_pips:.1f} pips)"
        else:
            components = []
            if volatility_score < 20:
                components.append(f"low volatility ({atr_pips:.1f} pips)")
            if range_score < 10:
                components.append(f"low movement ({current_range_pips:.1f} pips)")
            if time_score < 15:
                components.append("off-peak hours")
            reason = f"Inactive: {', '.join(components) if components else 'below threshold'}"

        return ActivityScore(
            is_active=is_active,
            score=total_score,
            volatility_score=volatility_score,
            range_score=range_score,
            time_score=time_score,
            reason=reason,
            level=level,
            atr_pips=atr_pips,
            spread_pips=0.0  # Will be set by spread check
        )
    
    def reset(self):
        """Reset filter state"""
        self._atr_history = []
        self._range_history = []

    def should_trade_hybrid(
        self,
        dt: datetime,
        current_high: float,
        current_low: float,
        in_killzone: bool,
        session_name: str,
        recent_df: Optional[pd.DataFrame] = None,
        current_spread: Optional[float] = None
    ) -> Tuple[bool, str, ActivityScore]:
        """Hybrid mode: Trade in Kill Zone OR when market is highly active

        This is the main entry point for hybrid mode filtering.

        Logic:
        - If in Kill Zone AND activity OK -> TRADE
        - If NOT in Kill Zone BUT activity HIGH (>70) -> TRADE
        - Otherwise -> NO TRADE

        Args:
            dt: Current datetime
            current_high: Current bar high
            current_low: Current bar low
            in_killzone: Whether currently in a Kill Zone
            session_name: Name of current session
            recent_df: Recent OHLCV data
            current_spread: Current spread (optional, in price units)

        Returns:
            Tuple of (should_trade, reason, ActivityScore)
        """
        # Get activity score
        activity = self.check_activity(dt, current_high, current_low, recent_df)

        # Check spread if provided
        spread_ok = True
        if current_spread is not None:
            spread_pips = current_spread / self.pip_size
            activity.spread_pips = spread_pips
            if spread_pips > self.max_spread_pips:
                spread_ok = False
                activity.reason += f" | Spread too wide: {spread_pips:.1f} pips"

        # Hybrid logic
        if in_killzone:
            # In Kill Zone - lower activity threshold OK
            if activity.level == ActivityLevel.DEAD:
                return False, f"Kill Zone ({session_name}) but DEAD market", activity
            elif not spread_ok:
                return False, f"Kill Zone ({session_name}) but spread too wide", activity
            else:
                emoji = activity.get_emoji()
                return True, f"{emoji} Kill Zone ({session_name}) + Activity {activity.score:.0f}", activity
        else:
            # Outside Kill Zone - need high activity
            if activity.score >= self.outside_kz_min_score and spread_ok:
                emoji = activity.get_emoji()
                return True, f"{emoji} High activity outside KZ ({activity.score:.0f})", activity
            elif activity.level == ActivityLevel.HIGH and spread_ok:
                # Even if score slightly below threshold, HIGH level is OK
                emoji = activity.get_emoji()
                return True, f"{emoji} Market active outside KZ ({activity.score:.0f})", activity
            else:
                return False, f"Outside KZ, activity too low ({activity.score:.0f})", activity

    def format_status(self, activity: ActivityScore, in_kz: bool, session: str) -> str:
        """Format activity status for Telegram/logging"""
        emoji = activity.get_emoji()
        kz_status = f"✅ {session}" if in_kz else "❌ Outside KZ"

        lines = [
            f"{emoji} Activity: {activity.level.value.upper()} ({activity.score:.0f}/100)",
            f"   Kill Zone: {kz_status}",
            f"   Volatility: {activity.volatility_score:.0f}/40 (ATR: {activity.atr_pips:.1f} pips)",
            f"   Range: {activity.range_score:.0f}/30",
            f"   Time: {activity.time_score:.0f}/30",
        ]

        if activity.spread_pips > 0:
            lines.append(f"   Spread: {activity.spread_pips:.1f} pips")

        return "\n".join(lines)


# Convenience function for backtesting
def should_trade_dynamic(
    dt: datetime,
    high: float,
    low: float,
    recent_df: Optional[pd.DataFrame] = None,
    filter_instance: Optional[DynamicActivityFilter] = None
) -> Tuple[bool, str]:
    """Quick check if we should trade
    
    Args:
        dt: Current datetime
        high: Current bar high
        low: Current bar low
        recent_df: Recent OHLCV data
        filter_instance: DynamicActivityFilter instance (creates new if None)
        
    Returns:
        Tuple of (should_trade, reason)
    """
    if filter_instance is None:
        filter_instance = DynamicActivityFilter()
    
    result = filter_instance.check_activity(dt, high, low, recent_df)
    return result.is_active, result.reason
