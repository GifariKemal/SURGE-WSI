"""Regime Validation Filter
============================

Validate HMM regime detection against actual price action.
Skip trades when detected regime doesn't match price momentum.

Problem: HMM detects BEARISH but price is going UP -> SELL trades lose
Solution: Validate regime with short-term momentum before trading

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class RegimeValidationResult:
    """Result of regime validation"""
    detected_regime: str           # From HMM: 'BULLISH', 'BEARISH', 'SIDEWAYS'
    price_momentum: str            # From price: 'UP', 'DOWN', 'FLAT'
    momentum_strength: float       # 0-100 (how strong the momentum)
    is_aligned: bool               # True if regime matches momentum
    confidence: float              # 0-100 confidence in validation
    recommendation: str            # 'TRADE', 'SKIP', 'CAUTION'
    reason: str

    def __str__(self):
        return (f"Regime={self.detected_regime}, Momentum={self.price_momentum} "
                f"({self.momentum_strength:.0f}%), Aligned={self.is_aligned} "
                f"-> {self.recommendation}")


class RegimeValidator:
    """Validate regime detection against price action"""

    def __init__(
        self,
        # Momentum calculation
        momentum_period: int = 10,            # Bars to calculate momentum
        momentum_threshold: float = 0.3,      # % move to consider directional

        # EMA for trend confirmation
        fast_ema: int = 8,
        slow_ema: int = 21,

        # Validation thresholds
        min_momentum_strength: float = 30.0,  # Min momentum to validate
        require_ema_alignment: bool = True,   # Also check EMA crossover

        pip_size: float = 0.0001
    ):
        """Initialize Regime Validator

        Args:
            momentum_period: Bars for momentum calculation
            momentum_threshold: % price change to consider directional
            fast_ema: Fast EMA period
            slow_ema: Slow EMA period
            min_momentum_strength: Minimum momentum strength to validate
            require_ema_alignment: Also require EMA alignment
        """
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.min_momentum_strength = min_momentum_strength
        self.require_ema_alignment = require_ema_alignment
        self.pip_size = pip_size

        logger.info(f"RegimeValidator: momentum_period={momentum_period}, "
                   f"threshold={momentum_threshold}%, EMA={fast_ema}/{slow_ema}")

    def calculate_momentum(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Calculate price momentum

        Args:
            df: DataFrame with OHLC data

        Returns:
            (direction, strength): ('UP'/'DOWN'/'FLAT', 0-100)
        """
        close = df['close'] if 'close' in df.columns else df['Close']

        if len(close) < self.momentum_period:
            return 'FLAT', 0.0

        # Price change over momentum period
        start_price = close.iloc[-self.momentum_period]
        end_price = close.iloc[-1]
        pct_change = (end_price - start_price) / start_price * 100

        # Also check how many bars were up vs down
        returns = close.pct_change().tail(self.momentum_period)
        up_bars = (returns > 0).sum()
        down_bars = (returns < 0).sum()
        total_bars = len(returns.dropna())

        # Directional strength (0-100)
        if total_bars > 0:
            directional_strength = abs(up_bars - down_bars) / total_bars * 100
        else:
            directional_strength = 0

        # Combined strength (price change + directional consistency)
        strength = (abs(pct_change) * 20 + directional_strength) / 2
        strength = min(100, max(0, strength))

        # Determine direction
        if pct_change > self.momentum_threshold and up_bars > down_bars:
            return 'UP', strength
        elif pct_change < -self.momentum_threshold and down_bars > up_bars:
            return 'DOWN', strength
        else:
            return 'FLAT', strength

    def calculate_ema_alignment(self, df: pd.DataFrame) -> Tuple[str, bool]:
        """Check EMA alignment

        Args:
            df: DataFrame with OHLC data

        Returns:
            (trend_direction, is_crossed): ('UP'/'DOWN'/'FLAT', recently_crossed)
        """
        close = df['close'] if 'close' in df.columns else df['Close']

        if len(close) < self.slow_ema + 5:
            return 'FLAT', False

        fast_ema = close.ewm(span=self.fast_ema, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_ema, adjust=False).mean()

        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-3]
        prev_slow = slow_ema.iloc[-3]

        # Current trend
        if current_fast > current_slow:
            trend = 'UP'
        elif current_fast < current_slow:
            trend = 'DOWN'
        else:
            trend = 'FLAT'

        # Check for recent crossover (within last 3 bars)
        recently_crossed = (
            (prev_fast <= prev_slow and current_fast > current_slow) or
            (prev_fast >= prev_slow and current_fast < current_slow)
        )

        return trend, recently_crossed

    def validate(self, df: pd.DataFrame, detected_regime: str) -> RegimeValidationResult:
        """Validate detected regime against price action

        Args:
            df: DataFrame with OHLC data
            detected_regime: HMM detected regime ('BULLISH', 'BEARISH', 'SIDEWAYS')

        Returns:
            RegimeValidationResult
        """
        # Get price momentum
        momentum_dir, momentum_strength = self.calculate_momentum(df)

        # Get EMA alignment
        ema_trend, ema_crossed = self.calculate_ema_alignment(df)

        # Map regime to expected momentum
        regime_expected = {
            'BULLISH': 'UP',
            'bullish': 'UP',
            'BEARISH': 'DOWN',
            'bearish': 'DOWN',
            'SIDEWAYS': 'FLAT',
            'sideways': 'FLAT'
        }

        expected_momentum = regime_expected.get(detected_regime, 'FLAT')

        # Check alignment
        is_momentum_aligned = (
            (expected_momentum == 'UP' and momentum_dir == 'UP') or
            (expected_momentum == 'DOWN' and momentum_dir == 'DOWN') or
            (expected_momentum == 'FLAT')  # Sideways always "aligned"
        )

        is_ema_aligned = (
            (expected_momentum == 'UP' and ema_trend == 'UP') or
            (expected_momentum == 'DOWN' and ema_trend == 'DOWN') or
            (expected_momentum == 'FLAT')
        )

        # Overall alignment check
        if self.require_ema_alignment:
            is_aligned = is_momentum_aligned and is_ema_aligned
        else:
            is_aligned = is_momentum_aligned

        # Calculate confidence
        if is_aligned:
            confidence = momentum_strength
            if is_ema_aligned:
                confidence = min(100, confidence + 20)
        else:
            confidence = 100 - momentum_strength

        # Determine recommendation
        if detected_regime.upper() == 'SIDEWAYS':
            recommendation = 'SKIP'
            reason = "Sideways regime - no clear direction"
        elif is_aligned and momentum_strength >= self.min_momentum_strength:
            recommendation = 'TRADE'
            reason = f"Regime {detected_regime} confirmed by {momentum_dir} momentum"
        elif is_aligned and momentum_strength < self.min_momentum_strength:
            recommendation = 'CAUTION'
            reason = f"Regime aligned but weak momentum ({momentum_strength:.0f}%)"
        elif not is_aligned and momentum_strength >= self.min_momentum_strength:
            recommendation = 'SKIP'
            reason = f"CONFLICT: {detected_regime} regime vs {momentum_dir} momentum ({momentum_strength:.0f}%)"
        else:
            recommendation = 'CAUTION'
            reason = "Unclear market direction"

        return RegimeValidationResult(
            detected_regime=detected_regime,
            price_momentum=momentum_dir,
            momentum_strength=momentum_strength,
            is_aligned=is_aligned,
            confidence=confidence,
            recommendation=recommendation,
            reason=reason
        )

    def should_skip_trade(self, df: pd.DataFrame, detected_regime: str,
                          direction: str) -> Tuple[bool, str]:
        """Check if trade should be skipped due to regime mismatch

        Args:
            df: DataFrame with OHLC data
            detected_regime: HMM detected regime
            direction: Trade direction ('BUY' or 'SELL')

        Returns:
            (should_skip, reason)
        """
        result = self.validate(df, detected_regime)

        # Additional check: direction vs momentum
        direction_matches_momentum = (
            (direction == 'BUY' and result.price_momentum == 'UP') or
            (direction == 'SELL' and result.price_momentum == 'DOWN')
        )

        if result.recommendation == 'SKIP':
            return True, result.reason

        if not direction_matches_momentum and result.momentum_strength >= self.min_momentum_strength:
            return True, f"{direction} signal vs {result.price_momentum} momentum ({result.momentum_strength:.0f}%)"

        return False, result.reason


# Convenience function
def validate_regime(df: pd.DataFrame, regime: str, direction: str = None,
                   momentum_period: int = 10) -> Tuple[bool, str, float]:
    """Quick regime validation

    Args:
        df: DataFrame with OHLC data
        regime: Detected regime
        direction: Trade direction
        momentum_period: Momentum calculation period

    Returns:
        (is_valid, reason, confidence)
    """
    validator = RegimeValidator(momentum_period=momentum_period)
    result = validator.validate(df, regime)

    is_valid = result.recommendation != 'SKIP'
    return is_valid, result.reason, result.confidence


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 50

    # Create upward trending data
    trend = np.linspace(0, 0.01, n)
    close = 1.25 + trend + np.random.randn(n) * 0.001

    df = pd.DataFrame({
        'high': close + np.random.rand(n) * 0.002,
        'low': close - np.random.rand(n) * 0.002,
        'close': close
    })

    validator = RegimeValidator()

    print("Testing Regime Validator")
    print("=" * 50)

    # Test: BULLISH regime on uptrending data (should pass)
    result = validator.validate(df, 'BULLISH')
    print(f"\nBULLISH regime on UP data: {result}")

    # Test: BEARISH regime on uptrending data (should fail)
    result = validator.validate(df, 'BEARISH')
    print(f"BEARISH regime on UP data: {result}")

    # Test should_skip_trade
    skip, reason = validator.should_skip_trade(df, 'BEARISH', 'SELL')
    print(f"\nSELL with BEARISH on UP data: skip={skip}, {reason}")

    skip, reason = validator.should_skip_trade(df, 'BULLISH', 'BUY')
    print(f"BUY with BULLISH on UP data: skip={skip}, {reason}")
