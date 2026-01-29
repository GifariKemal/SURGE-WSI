"""Choppiness Index Filter
==========================

Detect choppy/ranging vs trending market conditions.

Choppiness Index (CHOP) measures market "choppiness":
- High values (> 61.8): Choppy, ranging market - AVOID trading
- Low values (< 38.2): Trending market - GOOD for trading
- Middle values: Transitional

Formula:
CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)

References:
- https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58
- TradingView Choppiness Index

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ChoppinessResult:
    """Result of choppiness analysis"""
    value: float                    # CHOP value (0-100)
    is_choppy: bool                 # True if market is choppy
    is_trending: bool               # True if market is trending
    market_state: str               # 'CHOPPY', 'TRENDING', 'TRANSITIONAL'
    recommendation: str             # 'SKIP', 'TRADE', 'CAUTION'

    def __str__(self):
        return f"CHOP={self.value:.1f} ({self.market_state}) -> {self.recommendation}"


class ChoppinessFilter:
    """Filter for detecting choppy market conditions"""

    def __init__(
        self,
        period: int = 14,
        choppy_threshold: float = 61.8,     # Above this = choppy
        trending_threshold: float = 38.2,   # Below this = trending
        pip_size: float = 0.0001
    ):
        """Initialize Choppiness Filter

        Args:
            period: Lookback period for calculation (default 14)
            choppy_threshold: CHOP > this = choppy market (default 61.8 Fibonacci)
            trending_threshold: CHOP < this = trending market (default 38.2 Fibonacci)
            pip_size: Pip size for the instrument
        """
        self.period = period
        self.choppy_threshold = choppy_threshold
        self.trending_threshold = trending_threshold
        self.pip_size = pip_size

        # History for analysis
        self._chop_history: list = []
        self._max_history = 100

        logger.info(f"ChoppinessFilter initialized: period={period}, "
                   f"choppy>{choppy_threshold}, trending<{trending_threshold}")

    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range

        Args:
            df: DataFrame with high, low, close columns
            period: ATR period (default: self.period)

        Returns:
            Series of ATR values
        """
        if period is None:
            period = self.period

        # Normalize column names
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        # True Range is max of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is SMA of True Range
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_choppiness(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Choppiness Index

        Formula:
        CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)

        Args:
            df: DataFrame with OHLC data
            period: Calculation period

        Returns:
            Series of CHOP values (0-100)
        """
        if period is None:
            period = self.period

        # Normalize column names
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Sum of ATR over period
        atr_sum = tr.rolling(window=period).sum()

        # Highest High and Lowest Low over period
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        # Price range
        price_range = highest_high - lowest_low

        # Avoid division by zero
        price_range = price_range.replace(0, np.nan)

        # Choppiness Index formula
        # CHOP = 100 * LOG10(SUM(ATR, n) / (HH - LL)) / LOG10(n)
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)

        return chop

    def get_current_choppiness(self, df: pd.DataFrame) -> Optional[float]:
        """Get current (latest) choppiness value

        Args:
            df: DataFrame with OHLC data (needs at least period+1 bars)

        Returns:
            Current CHOP value or None if insufficient data
        """
        if len(df) < self.period + 1:
            return None

        chop_series = self.calculate_choppiness(df)
        current_chop = chop_series.iloc[-1]

        if pd.isna(current_chop):
            return None

        return float(current_chop)

    def analyze(self, df: pd.DataFrame) -> Optional[ChoppinessResult]:
        """Analyze current market choppiness

        Args:
            df: DataFrame with OHLC data

        Returns:
            ChoppinessResult with analysis
        """
        chop = self.get_current_choppiness(df)

        if chop is None:
            return None

        # Determine market state
        if chop > self.choppy_threshold:
            is_choppy = True
            is_trending = False
            market_state = 'CHOPPY'
            recommendation = 'SKIP'
        elif chop < self.trending_threshold:
            is_choppy = False
            is_trending = True
            market_state = 'TRENDING'
            recommendation = 'TRADE'
        else:
            is_choppy = False
            is_trending = False
            market_state = 'TRANSITIONAL'
            recommendation = 'CAUTION'

        result = ChoppinessResult(
            value=chop,
            is_choppy=is_choppy,
            is_trending=is_trending,
            market_state=market_state,
            recommendation=recommendation
        )

        # Store history
        self._chop_history.append(chop)
        if len(self._chop_history) > self._max_history:
            self._chop_history.pop(0)

        return result

    def should_skip_trading(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if trading should be skipped due to choppy conditions

        Args:
            df: DataFrame with OHLC data

        Returns:
            (should_skip, reason)
        """
        result = self.analyze(df)

        if result is None:
            return False, "Insufficient data for CHOP calculation"

        if result.is_choppy:
            return True, f"Choppy market detected (CHOP={result.value:.1f} > {self.choppy_threshold})"

        return False, f"Market OK (CHOP={result.value:.1f})"

    def get_chop_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get full CHOP series with signals

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with CHOP values and signals
        """
        chop = self.calculate_choppiness(df)

        result_df = pd.DataFrame(index=df.index)
        result_df['chop'] = chop
        result_df['is_choppy'] = chop > self.choppy_threshold
        result_df['is_trending'] = chop < self.trending_threshold
        result_df['signal'] = 'TRANSITIONAL'
        result_df.loc[result_df['is_choppy'], 'signal'] = 'CHOPPY'
        result_df.loc[result_df['is_trending'], 'signal'] = 'TRENDING'

        return result_df

    def get_statistics(self) -> dict:
        """Get statistics from history

        Returns:
            Dictionary with CHOP statistics
        """
        if not self._chop_history:
            return {}

        history = np.array(self._chop_history)

        return {
            'current': history[-1],
            'mean': np.mean(history),
            'std': np.std(history),
            'min': np.min(history),
            'max': np.max(history),
            'pct_choppy': np.sum(history > self.choppy_threshold) / len(history) * 100,
            'pct_trending': np.sum(history < self.trending_threshold) / len(history) * 100
        }


# Convenience function
def is_market_choppy(df: pd.DataFrame, period: int = 14, threshold: float = 61.8) -> bool:
    """Quick check if market is choppy

    Args:
        df: DataFrame with OHLC data
        period: CHOP calculation period
        threshold: Choppy threshold

    Returns:
        True if market is choppy
    """
    filter = ChoppinessFilter(period=period, choppy_threshold=threshold)
    should_skip, _ = filter.should_skip_trading(df)
    return should_skip


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Create sample data
    np.random.seed(42)
    n = 100

    # Trending data
    trending_close = 1.25 + np.cumsum(np.random.randn(n) * 0.001 + 0.0002)
    trending_high = trending_close + np.random.rand(n) * 0.002
    trending_low = trending_close - np.random.rand(n) * 0.002

    df_trending = pd.DataFrame({
        'open': trending_close - 0.0005,
        'high': trending_high,
        'low': trending_low,
        'close': trending_close
    })

    # Choppy data
    choppy_close = 1.25 + np.cumsum(np.random.randn(n) * 0.002)  # No drift
    choppy_high = choppy_close + np.random.rand(n) * 0.003
    choppy_low = choppy_close - np.random.rand(n) * 0.003

    df_choppy = pd.DataFrame({
        'open': choppy_close - 0.0005,
        'high': choppy_high,
        'low': choppy_low,
        'close': choppy_close
    })

    # Test
    filter = ChoppinessFilter()

    print("Testing Choppiness Filter")
    print("=" * 40)

    result_trending = filter.analyze(df_trending)
    print(f"Trending data: {result_trending}")

    result_choppy = filter.analyze(df_choppy)
    print(f"Choppy data: {result_choppy}")
