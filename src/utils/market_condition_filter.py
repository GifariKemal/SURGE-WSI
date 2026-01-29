"""Market Condition Filter (Confluence-Based)
=============================================

Combine multiple indicators to determine market tradability:
1. Choppiness Index - Detect ranging markets
2. ADX - Measure trend strength
3. Directional Alignment - Price vs EMA

Only trade when multiple conditions align (confluence).

References:
- Choppiness Index: https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58
- ADX: Average Directional Index by Welles Wilder
- EMA Alignment: Trend following confirmation

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class MarketConditionResult:
    """Result of market condition analysis"""
    # Individual scores (0-100)
    choppiness_score: float      # Lower = more trending
    adx_score: float             # Higher = stronger trend
    alignment_score: float       # Higher = better price-EMA alignment

    # Overall
    confluence_score: float      # Combined score (0-100)
    is_tradeable: bool           # True if conditions favorable
    market_condition: str        # 'STRONG_TREND', 'WEAK_TREND', 'CHOPPY', 'UNCERTAIN'
    recommendation: str          # 'TRADE', 'CAUTION', 'SKIP'
    reasons: list                # List of reasons

    def __str__(self):
        return (f"Confluence={self.confluence_score:.0f} ({self.market_condition}) "
                f"-> {self.recommendation}")


class MarketConditionFilter:
    """Combined market condition filter using multiple indicators"""

    def __init__(
        self,
        # Choppiness settings
        chop_period: int = 14,
        chop_choppy_threshold: float = 61.8,
        chop_trending_threshold: float = 38.2,

        # ADX settings
        adx_period: int = 14,
        adx_weak_threshold: float = 20.0,
        adx_strong_threshold: float = 25.0,

        # EMA settings
        ema_period: int = 21,

        # Confluence settings
        min_confluence_score: float = 60.0,  # Minimum to trade
        pip_size: float = 0.0001
    ):
        """Initialize Market Condition Filter

        Args:
            chop_period: Choppiness Index period
            chop_choppy_threshold: CHOP > this = choppy
            chop_trending_threshold: CHOP < this = trending
            adx_period: ADX period
            adx_weak_threshold: ADX < this = weak trend
            adx_strong_threshold: ADX > this = strong trend
            ema_period: EMA period for alignment check
            min_confluence_score: Minimum score to allow trading
        """
        self.chop_period = chop_period
        self.chop_choppy_threshold = chop_choppy_threshold
        self.chop_trending_threshold = chop_trending_threshold

        self.adx_period = adx_period
        self.adx_weak_threshold = adx_weak_threshold
        self.adx_strong_threshold = adx_strong_threshold

        self.ema_period = ema_period
        self.min_confluence_score = min_confluence_score
        self.pip_size = pip_size

        logger.info(f"MarketConditionFilter: CHOP<{chop_choppy_threshold}, "
                   f"ADX>{adx_weak_threshold}, EMA{ema_period}, "
                   f"min_confluence={min_confluence_score}")

    def calculate_choppiness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Choppiness Index"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_sum = tr.rolling(window=self.chop_period).sum()
        highest_high = high.rolling(window=self.chop_period).max()
        lowest_low = low.rolling(window=self.chop_period).min()
        price_range = highest_high - lowest_low
        price_range = price_range.replace(0, np.nan)

        chop = 100 * np.log10(atr_sum / price_range) / np.log10(self.chop_period)
        return chop

    def calculate_adx(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, -DI"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth using Wilder's method (EMA with alpha = 1/period)
        atr = pd.Series(tr).ewm(alpha=1/self.adx_period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/self.adx_period, adjust=False).mean() / atr

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/self.adx_period, adjust=False).mean()

        return adx, plus_di, minus_di

    def calculate_ema(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EMA"""
        close = df['close'] if 'close' in df.columns else df['Close']
        return close.ewm(span=self.ema_period, adjust=False).mean()

    def analyze(self, df: pd.DataFrame, direction: str = None) -> Optional[MarketConditionResult]:
        """Analyze current market conditions

        Args:
            df: DataFrame with OHLC data
            direction: Expected trade direction ('BUY' or 'SELL')

        Returns:
            MarketConditionResult
        """
        if len(df) < max(self.chop_period, self.adx_period, self.ema_period) + 5:
            return None

        reasons = []

        # 1. Choppiness Index Score (inverse - lower CHOP = higher score)
        chop = self.calculate_choppiness(df)
        current_chop = chop.iloc[-1]

        if pd.isna(current_chop):
            return None

        # Convert CHOP to score (0-100): CHOP 38.2->100, CHOP 61.8->0
        if current_chop <= self.chop_trending_threshold:
            chop_score = 100.0
            reasons.append(f"Strong trend (CHOP={current_chop:.1f})")
        elif current_chop >= self.chop_choppy_threshold:
            chop_score = 0.0
            reasons.append(f"Choppy market (CHOP={current_chop:.1f})")
        else:
            # Linear interpolation between thresholds
            chop_range = self.chop_choppy_threshold - self.chop_trending_threshold
            chop_score = 100 * (self.chop_choppy_threshold - current_chop) / chop_range
            if chop_score < 50:
                reasons.append(f"Moderate choppiness (CHOP={current_chop:.1f})")

        # 2. ADX Score
        adx, plus_di, minus_di = self.calculate_adx(df)
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        if pd.isna(current_adx):
            return None

        # Convert ADX to score (0-100): ADX 20->0, ADX 40->100
        if current_adx <= self.adx_weak_threshold:
            adx_score = 0.0
            reasons.append(f"Weak trend (ADX={current_adx:.1f})")
        elif current_adx >= 40:
            adx_score = 100.0
            reasons.append(f"Very strong trend (ADX={current_adx:.1f})")
        else:
            # Linear interpolation
            adx_score = 100 * (current_adx - self.adx_weak_threshold) / (40 - self.adx_weak_threshold)
            if adx_score >= 50:
                reasons.append(f"Good trend strength (ADX={current_adx:.1f})")

        # 3. EMA Alignment Score
        close = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
        ema = self.calculate_ema(df)
        current_ema = ema.iloc[-1]

        # Check alignment with direction
        alignment_score = 50.0  # Neutral if no direction specified

        if direction:
            price_vs_ema = (close - current_ema) / current_ema * 100  # % above/below EMA

            if direction == "BUY":
                # For BUY, price should be above EMA, +DI > -DI
                if close > current_ema and current_plus_di > current_minus_di:
                    alignment_score = min(100, 50 + abs(price_vs_ema) * 10)
                    reasons.append(f"Price aligned for BUY (+{price_vs_ema:.2f}% above EMA)")
                elif close > current_ema or current_plus_di > current_minus_di:
                    alignment_score = 50.0
                    reasons.append("Partial BUY alignment")
                else:
                    alignment_score = max(0, 50 - abs(price_vs_ema) * 10)
                    reasons.append(f"Price misaligned for BUY ({price_vs_ema:.2f}% vs EMA)")

            elif direction == "SELL":
                # For SELL, price should be below EMA, -DI > +DI
                if close < current_ema and current_minus_di > current_plus_di:
                    alignment_score = min(100, 50 + abs(price_vs_ema) * 10)
                    reasons.append(f"Price aligned for SELL ({price_vs_ema:.2f}% below EMA)")
                elif close < current_ema or current_minus_di > current_plus_di:
                    alignment_score = 50.0
                    reasons.append("Partial SELL alignment")
                else:
                    alignment_score = max(0, 50 - abs(price_vs_ema) * 10)
                    reasons.append(f"Price misaligned for SELL (+{price_vs_ema:.2f}% vs EMA)")

        # 4. Calculate Confluence Score (weighted average)
        # Choppiness: 40%, ADX: 40%, Alignment: 20%
        confluence_score = (
            chop_score * 0.40 +
            adx_score * 0.40 +
            alignment_score * 0.20
        )

        # Determine market condition and recommendation
        if confluence_score >= 70:
            market_condition = 'STRONG_TREND'
            recommendation = 'TRADE'
            is_tradeable = True
        elif confluence_score >= self.min_confluence_score:
            market_condition = 'WEAK_TREND'
            recommendation = 'CAUTION'
            is_tradeable = True
        elif confluence_score >= 40:
            market_condition = 'UNCERTAIN'
            recommendation = 'SKIP'
            is_tradeable = False
        else:
            market_condition = 'CHOPPY'
            recommendation = 'SKIP'
            is_tradeable = False

        return MarketConditionResult(
            choppiness_score=chop_score,
            adx_score=adx_score,
            alignment_score=alignment_score,
            confluence_score=confluence_score,
            is_tradeable=is_tradeable,
            market_condition=market_condition,
            recommendation=recommendation,
            reasons=reasons
        )

    def should_skip_trading(self, df: pd.DataFrame, direction: str = None) -> Tuple[bool, str]:
        """Check if trading should be skipped

        Args:
            df: DataFrame with OHLC data
            direction: Trade direction

        Returns:
            (should_skip, reason)
        """
        result = self.analyze(df, direction)

        if result is None:
            return False, "Insufficient data for market condition analysis"

        if not result.is_tradeable:
            return True, f"Unfavorable conditions: {result.market_condition} (score={result.confluence_score:.0f})"

        return False, f"Market OK: {result.market_condition} (score={result.confluence_score:.0f})"


# Convenience function
def check_market_condition(df: pd.DataFrame, direction: str = None,
                          min_score: float = 60.0) -> Tuple[bool, float, str]:
    """Quick check of market conditions

    Args:
        df: DataFrame with OHLC data
        direction: Trade direction
        min_score: Minimum confluence score to trade

    Returns:
        (is_tradeable, score, condition)
    """
    filter = MarketConditionFilter(min_confluence_score=min_score)
    result = filter.analyze(df, direction)

    if result is None:
        return True, 50.0, "UNKNOWN"

    return result.is_tradeable, result.confluence_score, result.market_condition


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 100

    # Create sample trending data
    trend = np.linspace(0, 0.02, n)
    noise = np.random.randn(n) * 0.001
    close = 1.25 + trend + np.cumsum(noise)

    df = pd.DataFrame({
        'open': close - 0.0005,
        'high': close + np.random.rand(n) * 0.002,
        'low': close - np.random.rand(n) * 0.002,
        'close': close
    })

    filter = MarketConditionFilter()

    print("Testing Market Condition Filter")
    print("=" * 50)

    result = filter.analyze(df, "BUY")
    print(f"\nResult: {result}")
    print(f"Choppiness Score: {result.choppiness_score:.1f}")
    print(f"ADX Score: {result.adx_score:.1f}")
    print(f"Alignment Score: {result.alignment_score:.1f}")
    print(f"Confluence: {result.confluence_score:.1f}")
    print(f"Reasons: {result.reasons}")
