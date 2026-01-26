"""Market Filter - Enhanced Entry Filters
=========================================

Solves problem months by adding:
1. Trend Filter - Avoid choppy markets
2. Volatility-Adjusted SL - Dynamic SL based on ATR
3. Trend Direction Check - Avoid counter-trend entries

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum
from loguru import logger


class MarketCondition(Enum):
    """Market condition classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOPPY = "choppy"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class MarketAnalysis:
    """Market analysis result"""
    condition: MarketCondition
    trend_strength: float  # 0-100
    volatility_level: float  # ATR in pips
    recommended_sl_multiplier: float  # 1.0 = normal, 1.5 = wider
    can_trade: bool
    reason: str


class MarketFilter:
    """Enhanced market filter for better entry quality"""

    def __init__(
        self,
        trend_threshold: float = 0.6,  # 60% directional bars = trending
        min_volatility_pips: float = 15.0,  # Min ATR to trade
        max_volatility_pips: float = 80.0,  # Max ATR (too risky)
        atr_period: int = 14,
        lookback_bars: int = 20
    ):
        """Initialize Market Filter

        Args:
            trend_threshold: Ratio of directional bars to confirm trend
            min_volatility_pips: Minimum ATR in pips to trade
            max_volatility_pips: Maximum ATR in pips (too volatile)
            atr_period: Period for ATR calculation
            lookback_bars: Bars to analyze for trend
        """
        self.trend_threshold = trend_threshold
        self.min_volatility_pips = min_volatility_pips
        self.max_volatility_pips = max_volatility_pips
        self.atr_period = atr_period
        self.lookback_bars = lookback_bars

    def calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range in pips"""
        if len(df) < self.atr_period:
            return 0.0

        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        close_col = 'close' if 'close' in df.columns else 'Close'

        high = df[high_col].values
        low = df[low_col].values
        close = df[close_col].values

        tr_list = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        if len(tr_list) < self.atr_period:
            return np.mean(tr_list) / 0.0001 if tr_list else 0.0

        atr = np.mean(tr_list[-self.atr_period:])
        return atr / 0.0001  # Convert to pips

    def analyze_trend(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze trend strength and direction

        Returns:
            Tuple of (trend_strength 0-100, direction 'up'/'down'/'none')
        """
        if len(df) < self.lookback_bars:
            return 0.0, 'none'

        close_col = 'close' if 'close' in df.columns else 'Close'
        recent = df[close_col].iloc[-self.lookback_bars:]

        # Count up vs down bars
        returns = recent.pct_change().dropna()
        up_bars = (returns > 0).sum()
        down_bars = (returns < 0).sum()
        total = up_bars + down_bars

        if total == 0:
            return 0.0, 'none'

        # Calculate directional ratio
        up_ratio = up_bars / total
        down_ratio = down_bars / total

        if up_ratio >= self.trend_threshold:
            return up_ratio * 100, 'up'
        elif down_ratio >= self.trend_threshold:
            return down_ratio * 100, 'down'
        else:
            # Choppy - calculate how choppy (closer to 50% = more choppy)
            choppiness = 1 - abs(up_ratio - 0.5) * 2  # 0-1 scale
            return choppiness * 100, 'none'

    def analyze_market(self, df: pd.DataFrame) -> MarketAnalysis:
        """Comprehensive market analysis

        Args:
            df: OHLCV DataFrame (H4 recommended)

        Returns:
            MarketAnalysis with trading recommendation
        """
        # Calculate ATR
        atr_pips = self.calculate_atr(df)

        # Analyze trend
        trend_strength, trend_dir = self.analyze_trend(df)

        # Determine condition and trading permission
        can_trade = True
        reason = "OK"
        sl_multiplier = 1.0

        # Check volatility
        if atr_pips < self.min_volatility_pips:
            condition = MarketCondition.LOW_VOLATILITY
            can_trade = True  # Allow but note it
            reason = f"Low volatility ({atr_pips:.1f} pips)"
            sl_multiplier = 0.8  # Tighter SL for low vol

        elif atr_pips > self.max_volatility_pips:
            condition = MarketCondition.HIGH_VOLATILITY
            can_trade = False  # Too risky
            reason = f"High volatility ({atr_pips:.1f} pips) - Skip"
            sl_multiplier = 1.5

        # Check trend
        elif trend_dir == 'up':
            condition = MarketCondition.TRENDING_UP
            can_trade = True
            reason = f"Trending UP ({trend_strength:.0f}% strength)"
            sl_multiplier = 1.0

        elif trend_dir == 'down':
            condition = MarketCondition.TRENDING_DOWN
            can_trade = True
            reason = f"Trending DOWN ({trend_strength:.0f}% strength)"
            sl_multiplier = 1.0

        else:
            condition = MarketCondition.CHOPPY
            # Allow trading but with stricter filters
            if trend_strength > 70:  # Very choppy
                can_trade = False
                reason = f"Very choppy market ({trend_strength:.0f}%) - Skip"
            else:
                can_trade = True
                reason = f"Mildly choppy ({trend_strength:.0f}%) - Trade with caution"
                sl_multiplier = 1.2  # Slightly wider SL

        return MarketAnalysis(
            condition=condition,
            trend_strength=trend_strength,
            volatility_level=atr_pips,
            recommended_sl_multiplier=sl_multiplier,
            can_trade=can_trade,
            reason=reason
        )

    def check_trend_alignment(
        self,
        df: pd.DataFrame,
        signal_direction: str
    ) -> Tuple[bool, str]:
        """Check if signal aligns with trend

        Args:
            df: OHLCV DataFrame
            signal_direction: 'BUY' or 'SELL'

        Returns:
            Tuple of (is_aligned, reason)
        """
        _, trend_dir = self.analyze_trend(df)

        if trend_dir == 'none':
            # No clear trend - allow with caution
            return True, "No clear trend - proceed with caution"

        if signal_direction == 'BUY' and trend_dir == 'up':
            return True, "BUY aligned with uptrend"
        elif signal_direction == 'SELL' and trend_dir == 'down':
            return True, "SELL aligned with downtrend"
        elif signal_direction == 'BUY' and trend_dir == 'down':
            return False, "BUY against downtrend - SKIP"
        elif signal_direction == 'SELL' and trend_dir == 'up':
            return False, "SELL against uptrend - SKIP"

        return True, "OK"

    def get_dynamic_sl(
        self,
        df: pd.DataFrame,
        base_sl_pips: float
    ) -> float:
        """Get volatility-adjusted stop loss

        Args:
            df: OHLCV DataFrame
            base_sl_pips: Original SL in pips

        Returns:
            Adjusted SL in pips
        """
        analysis = self.analyze_market(df)

        # Apply multiplier
        adjusted_sl = base_sl_pips * analysis.recommended_sl_multiplier

        # Ensure minimum SL
        min_sl = 10.0
        adjusted_sl = max(adjusted_sl, min_sl)

        # Cap maximum SL at 50 pips to limit risk
        max_sl = 50.0
        adjusted_sl = min(adjusted_sl, max_sl)

        return adjusted_sl


class RelaxedEntryFilter:
    """Relaxed entry filter for low-activity periods"""

    def __init__(
        self,
        min_quality_normal: float = 60.0,
        min_quality_relaxed: float = 50.0,
        require_full_confirmation_normal: bool = True,
        require_full_confirmation_relaxed: bool = False
    ):
        self.min_quality_normal = min_quality_normal
        self.min_quality_relaxed = min_quality_relaxed
        self.require_full_confirmation_normal = require_full_confirmation_normal
        self.require_full_confirmation_relaxed = require_full_confirmation_relaxed

    def get_entry_params(
        self,
        recent_trade_count: int,
        lookback_days: int = 7
    ) -> Tuple[float, bool]:
        """Get entry parameters based on recent activity

        If no trades in recent period, relax filters.

        Args:
            recent_trade_count: Trades in last N days
            lookback_days: Days to look back

        Returns:
            Tuple of (min_quality, require_full_confirmation)
        """
        if recent_trade_count == 0:
            # No recent trades - relax filters
            logger.debug("Relaxing entry filters due to low activity")
            return self.min_quality_relaxed, self.require_full_confirmation_relaxed
        else:
            return self.min_quality_normal, self.require_full_confirmation_normal
