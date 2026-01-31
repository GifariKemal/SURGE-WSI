"""Volatility Regime Filter - Dynamic Low-Quality Market Detection
=================================================================

Instead of hardcoding "skip June", this filter DYNAMICALLY detects
market conditions that are unfavorable for trading:

1. LOW VOLATILITY REGIME - ATR significantly below average
2. RANGING/CHOPPY MARKET - High choppiness, low directional movement
3. LOW ACTIVITY PERIODS - Volume and range compression
4. TREND EXHAUSTION - Declining momentum after extended moves

These conditions often occur during:
- Summer months (June-August) - "Summer lull"
- Holiday periods (Dec-Jan)
- Pre-major news events
- Post-trend exhaustion

The filter outputs a "Market Quality Score" (0-100):
- 80-100: Excellent conditions, full risk
- 60-79: Good conditions, normal risk
- 40-59: Marginal conditions, reduced risk
- 20-39: Poor conditions, minimal risk or skip
- 0-19: Very poor conditions, SKIP

Author: SURIOTA Team
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
from loguru import logger


@dataclass
class MarketQualityResult:
    """Result of market quality analysis"""
    score: float                    # 0-100 overall quality score
    can_trade: bool                 # Whether trading is recommended
    risk_multiplier: float          # Suggested risk adjustment (0.0-1.2)

    # Component scores
    volatility_score: float         # ATR relative to average
    trend_score: float              # Directional movement strength
    choppiness_score: float         # Inverse of choppiness
    activity_score: float           # Range and volume activity

    # Flags
    is_low_volatility: bool
    is_ranging: bool
    is_trend_exhaustion: bool

    reason: str                     # Human readable reason


class VolatilityRegimeFilter:
    """
    Dynamic filter that detects poor market quality conditions.

    This replaces hardcoded "skip June" logic with intelligent
    detection of June-LIKE conditions (low vol, ranging, etc.)
    """

    def __init__(
        self,
        # Volatility thresholds
        atr_lookback: int = 50,              # Bars to calculate average ATR
        low_vol_threshold: float = 0.7,       # ATR < 70% of avg = low vol
        very_low_vol_threshold: float = 0.5,  # ATR < 50% of avg = very low

        # Choppiness thresholds
        chop_lookback: int = 14,
        high_chop_threshold: float = 62.0,    # > 62 = ranging
        very_high_chop_threshold: float = 70.0,

        # Trend strength
        adx_lookback: int = 14,
        weak_trend_threshold: float = 20.0,   # ADX < 20 = weak trend

        # Activity thresholds
        min_daily_range_pips: float = 30.0,   # Minimum daily range
        range_compression_threshold: float = 0.6,  # Range < 60% of avg

        # Score thresholds
        min_quality_score: float = 35.0,      # Below this = skip
        reduced_risk_threshold: float = 55.0,  # Below this = reduce risk
    ):
        self.atr_lookback = atr_lookback
        self.low_vol_threshold = low_vol_threshold
        self.very_low_vol_threshold = very_low_vol_threshold

        self.chop_lookback = chop_lookback
        self.high_chop_threshold = high_chop_threshold
        self.very_high_chop_threshold = very_high_chop_threshold

        self.adx_lookback = adx_lookback
        self.weak_trend_threshold = weak_trend_threshold

        self.min_daily_range_pips = min_daily_range_pips
        self.range_compression_threshold = range_compression_threshold

        self.min_quality_score = min_quality_score
        self.reduced_risk_threshold = reduced_risk_threshold

        # Cache for historical averages
        self._avg_atr = None
        self._avg_range = None

        logger.info(
            f"VolatilityRegimeFilter initialized: "
            f"low_vol<{low_vol_threshold:.0%}, high_chop>{high_chop_threshold}, "
            f"min_score>{min_quality_score}"
        )

    def analyze(
        self,
        df: pd.DataFrame,
        current_time: datetime = None
    ) -> MarketQualityResult:
        """
        Analyze current market quality.

        Args:
            df: DataFrame with OHLCV data (needs at least atr_lookback bars)
            current_time: Current timestamp (for logging)

        Returns:
            MarketQualityResult with quality score and recommendations
        """
        if len(df) < self.atr_lookback:
            return self._default_result("Insufficient data")

        # Detect column names
        col_map = self._get_column_map(df)

        # Calculate indicators
        atr_current, atr_avg = self._calculate_atr_regime(df, col_map)
        choppiness = self._calculate_choppiness(df, col_map)
        adx = self._calculate_adx(df, col_map)
        range_ratio = self._calculate_range_compression(df, col_map)

        # Calculate component scores (0-100)
        volatility_score = self._score_volatility(atr_current, atr_avg)
        trend_score = self._score_trend(adx)
        choppiness_score = self._score_choppiness(choppiness)
        activity_score = self._score_activity(range_ratio, atr_current)

        # Detect flags
        is_low_volatility = (atr_current / atr_avg) < self.low_vol_threshold if atr_avg > 0 else False
        is_ranging = choppiness > self.high_chop_threshold
        is_trend_exhaustion = adx < self.weak_trend_threshold and is_ranging

        # Calculate overall score (weighted average)
        weights = {
            'volatility': 0.30,
            'trend': 0.25,
            'choppiness': 0.25,
            'activity': 0.20
        }

        overall_score = (
            volatility_score * weights['volatility'] +
            trend_score * weights['trend'] +
            choppiness_score * weights['choppiness'] +
            activity_score * weights['activity']
        )

        # Apply penalties for severe conditions
        if (atr_current / atr_avg) < self.very_low_vol_threshold if atr_avg > 0 else False:
            overall_score *= 0.7  # 30% penalty for very low volatility

        if choppiness > self.very_high_chop_threshold:
            overall_score *= 0.8  # 20% penalty for very high choppiness

        # Determine trading recommendation
        can_trade = overall_score >= self.min_quality_score

        # Calculate risk multiplier
        if overall_score >= 80:
            risk_multiplier = 1.1  # Boost for excellent conditions
        elif overall_score >= self.reduced_risk_threshold:
            risk_multiplier = 1.0  # Normal
        elif overall_score >= self.min_quality_score:
            # Linear reduction from 1.0 to 0.5
            ratio = (overall_score - self.min_quality_score) / (self.reduced_risk_threshold - self.min_quality_score)
            risk_multiplier = 0.5 + (0.5 * ratio)
        else:
            risk_multiplier = 0.0  # Skip

        # Generate reason
        reason = self._generate_reason(
            overall_score, is_low_volatility, is_ranging, is_trend_exhaustion,
            atr_current, atr_avg, choppiness, adx
        )

        return MarketQualityResult(
            score=overall_score,
            can_trade=can_trade,
            risk_multiplier=risk_multiplier,
            volatility_score=volatility_score,
            trend_score=trend_score,
            choppiness_score=choppiness_score,
            activity_score=activity_score,
            is_low_volatility=is_low_volatility,
            is_ranging=is_ranging,
            is_trend_exhaustion=is_trend_exhaustion,
            reason=reason
        )

    def _get_column_map(self, df: pd.DataFrame) -> dict:
        """Get column name mapping"""
        return {
            'open': 'open' if 'open' in df.columns else 'Open',
            'high': 'high' if 'high' in df.columns else 'High',
            'low': 'low' if 'low' in df.columns else 'Low',
            'close': 'close' if 'close' in df.columns else 'Close',
        }

    def _calculate_atr_regime(self, df: pd.DataFrame, col_map: dict) -> Tuple[float, float]:
        """Calculate current ATR and long-term average ATR"""
        high = df[col_map['high']]
        low = df[col_map['low']]
        close = df[col_map['close']]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Current ATR (14-period)
        atr_current = tr.rolling(14).mean().iloc[-1]

        # Long-term average ATR
        atr_avg = tr.rolling(self.atr_lookback).mean().iloc[-1]

        return atr_current, atr_avg

    def _calculate_choppiness(self, df: pd.DataFrame, col_map: dict) -> float:
        """Calculate Choppiness Index"""
        high = df[col_map['high']]
        low = df[col_map['low']]
        close = df[col_map['close']]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_sum = tr.rolling(self.chop_lookback).sum()
        highest_high = high.rolling(self.chop_lookback).max()
        lowest_low = low.rolling(self.chop_lookback).min()

        price_range = highest_high - lowest_low
        price_range = price_range.replace(0, np.nan)

        choppiness = 100 * np.log10(atr_sum / price_range) / np.log10(self.chop_lookback)

        return choppiness.iloc[-1] if not pd.isna(choppiness.iloc[-1]) else 50.0

    def _calculate_adx(self, df: pd.DataFrame, col_map: dict) -> float:
        """Calculate ADX (Average Directional Index)"""
        high = df[col_map['high']]
        low = df[col_map['low']]
        close = df[col_map['close']]

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed
        atr = tr.rolling(self.adx_lookback).mean()
        plus_di = 100 * (plus_dm.rolling(self.adx_lookback).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_lookback).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(self.adx_lookback).mean()

        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0

    def _calculate_range_compression(self, df: pd.DataFrame, col_map: dict) -> float:
        """Calculate range compression ratio"""
        high = df[col_map['high']]
        low = df[col_map['low']]

        # Recent range (last 5 bars)
        recent_range = (high.tail(5).max() - low.tail(5).min())

        # Average range (lookback period)
        rolling_range = (high.rolling(20).max() - low.rolling(20).min()).mean()

        if rolling_range > 0:
            return recent_range / rolling_range
        return 1.0

    def _score_volatility(self, atr_current: float, atr_avg: float) -> float:
        """Score volatility (higher = better)"""
        if atr_avg <= 0:
            return 50.0

        ratio = atr_current / atr_avg

        if ratio >= 1.2:
            return 100.0  # High volatility = great
        elif ratio >= 1.0:
            return 80 + (ratio - 1.0) * 100  # 80-100
        elif ratio >= 0.7:
            return 50 + (ratio - 0.7) * 100  # 50-80
        elif ratio >= 0.5:
            return 20 + (ratio - 0.5) * 150  # 20-50
        else:
            return max(0, ratio * 40)  # 0-20

    def _score_trend(self, adx: float) -> float:
        """Score trend strength (higher ADX = better)"""
        if adx >= 40:
            return 100.0
        elif adx >= 25:
            return 70 + (adx - 25) * 2  # 70-100
        elif adx >= 20:
            return 50 + (adx - 20) * 4  # 50-70
        elif adx >= 15:
            return 30 + (adx - 15) * 4  # 30-50
        else:
            return max(0, adx * 2)  # 0-30

    def _score_choppiness(self, choppiness: float) -> float:
        """Score choppiness (lower = better)"""
        if choppiness <= 38.2:
            return 100.0  # Very trending
        elif choppiness <= 50:
            return 80 + (50 - choppiness) * 1.7  # 80-100
        elif choppiness <= 61.8:
            return 50 + (61.8 - choppiness) * 2.5  # 50-80
        elif choppiness <= 70:
            return 20 + (70 - choppiness) * 3.7  # 20-50
        else:
            return max(0, (80 - choppiness) * 2)  # 0-20

    def _score_activity(self, range_ratio: float, atr_current: float) -> float:
        """Score market activity"""
        # Convert ATR to pips (assuming 4-digit pairs)
        atr_pips = atr_current * 10000

        # Range ratio score
        if range_ratio >= 1.0:
            range_score = 100
        elif range_ratio >= 0.7:
            range_score = 60 + (range_ratio - 0.7) * 133
        else:
            range_score = max(0, range_ratio * 85)

        # ATR pips score
        if atr_pips >= 15:
            atr_score = 100
        elif atr_pips >= 10:
            atr_score = 70 + (atr_pips - 10) * 6
        elif atr_pips >= 5:
            atr_score = 30 + (atr_pips - 5) * 8
        else:
            atr_score = max(0, atr_pips * 6)

        return (range_score + atr_score) / 2

    def _generate_reason(
        self, score: float, is_low_vol: bool, is_ranging: bool,
        is_exhaustion: bool, atr_curr: float, atr_avg: float,
        chop: float, adx: float
    ) -> str:
        """Generate human-readable reason"""
        reasons = []

        if score >= 80:
            reasons.append("Excellent market conditions")
        elif score >= 60:
            reasons.append("Good market conditions")
        elif score >= 40:
            reasons.append("Marginal conditions")
        else:
            reasons.append("Poor conditions")

        if is_low_vol:
            vol_ratio = (atr_curr / atr_avg * 100) if atr_avg > 0 else 0
            reasons.append(f"Low volatility ({vol_ratio:.0f}% of avg)")

        if is_ranging:
            reasons.append(f"Ranging market (chop={chop:.1f})")

        if is_exhaustion:
            reasons.append(f"Trend exhaustion (ADX={adx:.1f})")

        return "; ".join(reasons)

    def _default_result(self, reason: str) -> MarketQualityResult:
        """Return default result when analysis not possible"""
        return MarketQualityResult(
            score=50.0,
            can_trade=True,
            risk_multiplier=0.8,
            volatility_score=50.0,
            trend_score=50.0,
            choppiness_score=50.0,
            activity_score=50.0,
            is_low_volatility=False,
            is_ranging=False,
            is_trend_exhaustion=False,
            reason=reason
        )


# Convenience function
def check_market_quality(df: pd.DataFrame) -> MarketQualityResult:
    """Quick function to check market quality"""
    filter = VolatilityRegimeFilter()
    return filter.analyze(df)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VOLATILITY REGIME FILTER TEST")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='H')

    # Simulate normal volatility
    normal_data = pd.DataFrame({
        'open': 1.2500 + np.cumsum(np.random.randn(100) * 0.001),
        'high': 1.2500 + np.cumsum(np.random.randn(100) * 0.001) + 0.002,
        'low': 1.2500 + np.cumsum(np.random.randn(100) * 0.001) - 0.002,
        'close': 1.2500 + np.cumsum(np.random.randn(100) * 0.001) + 0.0005,
    }, index=dates)

    filter = VolatilityRegimeFilter()
    result = filter.analyze(normal_data)

    print("\nNormal Market Conditions:")
    print(f"  Score: {result.score:.1f}/100")
    print(f"  Can Trade: {result.can_trade}")
    print(f"  Risk Mult: {result.risk_multiplier:.2f}")
    print(f"  Reason: {result.reason}")
    print(f"  Components:")
    print(f"    - Volatility: {result.volatility_score:.1f}")
    print(f"    - Trend: {result.trend_score:.1f}")
    print(f"    - Choppiness: {result.choppiness_score:.1f}")
    print(f"    - Activity: {result.activity_score:.1f}")

    # Simulate low volatility (summer lull)
    low_vol_data = normal_data.copy()
    low_vol_data['high'] = low_vol_data['open'] + 0.0005  # Much smaller range
    low_vol_data['low'] = low_vol_data['open'] - 0.0005
    low_vol_data['close'] = low_vol_data['open'] + 0.0002

    result_low = filter.analyze(low_vol_data)

    print("\n" + "-" * 40)
    print("Low Volatility (Summer Lull) Conditions:")
    print(f"  Score: {result_low.score:.1f}/100")
    print(f"  Can Trade: {result_low.can_trade}")
    print(f"  Risk Mult: {result_low.risk_multiplier:.2f}")
    print(f"  Reason: {result_low.reason}")
    print(f"  Flags:")
    print(f"    - Low Volatility: {result_low.is_low_volatility}")
    print(f"    - Ranging: {result_low.is_ranging}")
    print(f"    - Trend Exhaustion: {result_low.is_trend_exhaustion}")

    print("\n" + "=" * 60)
