"""Confluence Validator v1 - Multi-Factor Trade Validation
===========================================================

Professional-grade validation system that requires multiple
factors to align before taking a trade.

Research basis:
- "When multiple timeframes align, probability of success increases dramatically"
- Strong validation: Trend, Liquidity, Volatility, Momentum, Location
- From: moneyminiblog.com/investing/technical-analysis-for-professional-traders

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from loguru import logger


class ValidationFactor(Enum):
    """Individual validation factors"""
    TREND_ALIGNMENT = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    POI_QUALITY = "poi_quality"
    SESSION_OPTIMAL = "session"
    REGIME_CONFIDENCE = "regime"
    NO_DRIFT = "no_drift"
    DAY_OF_WEEK = "day_ok"


@dataclass
class FactorResult:
    """Result of single factor validation"""
    factor: ValidationFactor
    passed: bool
    score: float  # 0-100
    weight: float  # Factor weight
    reason: str


@dataclass
class ConfluenceResult:
    """Result of confluence validation"""
    # Overall
    passed: bool
    total_score: float  # Weighted score 0-100
    factors_passed: int
    factors_total: int

    # Details
    factor_results: List[FactorResult]

    # Recommendations
    can_trade: bool
    risk_multiplier: float  # 1.0 = normal, <1 = reduce
    confidence_level: str  # 'high', 'medium', 'low'

    # Message
    message: str

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} | Score: {self.total_score:.0f}/100 | "
            f"Factors: {self.factors_passed}/{self.factors_total} | "
            f"{self.confidence_level.upper()}"
        )

    def get_failed_factors(self) -> List[str]:
        """Get list of failed factors."""
        return [f.factor.value for f in self.factor_results if not f.passed]

    def get_summary(self) -> str:
        """Get detailed summary."""
        lines = [f"Confluence: {self}"]
        for fr in self.factor_results:
            status = "✅" if fr.passed else "❌"
            lines.append(f"  {status} {fr.factor.value}: {fr.score:.0f} - {fr.reason}")
        return "\n".join(lines)


class ConfluenceValidator:
    """
    Multi-factor confluence validator.

    Validates trades against multiple factors:
    1. Trend alignment (price vs EMA)
    2. Momentum (RSI/velocity direction)
    3. Volatility (ATR within acceptable range)
    4. POI Quality (order block quality)
    5. Session optimal (trading session)
    6. Regime confidence (HMM confidence)
    7. No drift (market stability)
    8. Day of week (avoid Thursday in weak markets)
    """

    def __init__(
        self,
        min_total_score: float = 60.0,
        min_factors_passed: int = 5,
        weights: Dict[ValidationFactor, float] = None
    ):
        """
        Initialize Confluence Validator.

        Args:
            min_total_score: Minimum weighted score to pass (0-100)
            min_factors_passed: Minimum number of factors that must pass
            weights: Custom weights for each factor (must sum to 1.0)
        """
        self.min_total_score = min_total_score
        self.min_factors_passed = min_factors_passed

        # Default weights (sum to 1.0)
        self.weights = weights or {
            ValidationFactor.TREND_ALIGNMENT: 0.15,
            ValidationFactor.MOMENTUM: 0.15,
            ValidationFactor.VOLATILITY: 0.10,
            ValidationFactor.POI_QUALITY: 0.20,
            ValidationFactor.SESSION_OPTIMAL: 0.10,
            ValidationFactor.REGIME_CONFIDENCE: 0.15,
            ValidationFactor.NO_DRIFT: 0.10,
            ValidationFactor.DAY_OF_WEEK: 0.05,
        }

        # Ensure weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}

        logger.info(
            f"ConfluenceValidator initialized: min_score={min_total_score}, "
            f"min_factors={min_factors_passed}"
        )

    def validate(
        self,
        direction: str,
        df: pd.DataFrame,
        poi_quality: float,
        regime_confidence: float,
        regime_bias: str,
        session_name: str,
        is_optimal_session: bool,
        drift_detected: bool,
        current_time: datetime = None,
        kalman_velocity: float = None
    ) -> ConfluenceResult:
        """
        Validate trade against all confluence factors.

        Args:
            direction: Trade direction ('BUY' or 'SELL')
            df: OHLCV DataFrame (at least 50 bars)
            poi_quality: POI quality score (0-100)
            regime_confidence: HMM regime confidence (0-100)
            regime_bias: Regime bias ('BUY', 'SELL', 'NONE')
            session_name: Current session name
            is_optimal_session: Whether in optimal trading time
            drift_detected: Whether drift was detected
            current_time: Current datetime
            kalman_velocity: Kalman filter velocity (optional)

        Returns:
            ConfluenceResult with validation details
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        col_map = self._get_col_map(df)
        factor_results = []

        # 1. Trend Alignment
        trend_result = self._check_trend_alignment(df, direction, col_map)
        factor_results.append(trend_result)

        # 2. Momentum
        momentum_result = self._check_momentum(df, direction, col_map, kalman_velocity)
        factor_results.append(momentum_result)

        # 3. Volatility
        volatility_result = self._check_volatility(df, col_map)
        factor_results.append(volatility_result)

        # 4. POI Quality
        poi_result = self._check_poi_quality(poi_quality)
        factor_results.append(poi_result)

        # 5. Session Optimal
        session_result = self._check_session(session_name, is_optimal_session)
        factor_results.append(session_result)

        # 6. Regime Confidence
        regime_result = self._check_regime(regime_confidence, regime_bias, direction)
        factor_results.append(regime_result)

        # 7. No Drift
        drift_result = self._check_drift(drift_detected)
        factor_results.append(drift_result)

        # 8. Day of Week
        day_result = self._check_day_of_week(current_time, regime_confidence)
        factor_results.append(day_result)

        # Calculate total score
        total_score = sum(
            fr.score * self.weights[fr.factor]
            for fr in factor_results
        )

        factors_passed = sum(1 for fr in factor_results if fr.passed)
        factors_total = len(factor_results)

        # Determine if passed
        passed = (
            total_score >= self.min_total_score and
            factors_passed >= self.min_factors_passed
        )

        # Determine confidence level and risk multiplier
        if total_score >= 80 and factors_passed >= 7:
            confidence_level = 'high'
            risk_multiplier = 1.1  # Can increase risk slightly
        elif total_score >= 65 and factors_passed >= 6:
            confidence_level = 'medium'
            risk_multiplier = 1.0
        elif passed:
            confidence_level = 'low'
            risk_multiplier = 0.8  # Reduce risk
        else:
            confidence_level = 'insufficient'
            risk_multiplier = 0.5

        # Build message
        if passed:
            message = f"Confluence OK: {factors_passed}/{factors_total} factors, score={total_score:.0f}"
        else:
            failed = [fr.factor.value for fr in factor_results if not fr.passed]
            message = f"Confluence FAILED: {failed}"

        return ConfluenceResult(
            passed=passed,
            total_score=total_score,
            factors_passed=factors_passed,
            factors_total=factors_total,
            factor_results=factor_results,
            can_trade=passed,
            risk_multiplier=risk_multiplier,
            confidence_level=confidence_level,
            message=message
        )

    def _get_col_map(self, df: pd.DataFrame) -> dict:
        """Get column name mapping."""
        return {
            'close': 'close' if 'close' in df.columns else 'Close',
            'open': 'open' if 'open' in df.columns else 'Open',
            'high': 'high' if 'high' in df.columns else 'High',
            'low': 'low' if 'low' in df.columns else 'Low',
        }

    def _check_trend_alignment(
        self,
        df: pd.DataFrame,
        direction: str,
        col_map: dict
    ) -> FactorResult:
        """Check if price aligns with trend (EMA)."""
        close = df[col_map['close']]

        # Calculate EMAs
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()

        current_price = close.iloc[-1]
        current_ema20 = ema_20.iloc[-1]
        current_ema50 = ema_50.iloc[-1]

        # Check alignment
        if direction == 'BUY':
            # For BUY: Price > EMA20 > EMA50 is ideal
            price_above_ema20 = current_price > current_ema20
            ema20_above_ema50 = current_ema20 > current_ema50

            if price_above_ema20 and ema20_above_ema50:
                score = 100
                reason = "Strong uptrend: Price > EMA20 > EMA50"
                passed = True
            elif price_above_ema20:
                score = 70
                reason = "Moderate uptrend: Price > EMA20"
                passed = True
            elif current_price > current_ema50:
                score = 50
                reason = "Weak uptrend: Price > EMA50 only"
                passed = True
            else:
                score = 20
                reason = "No uptrend alignment"
                passed = False
        else:  # SELL
            price_below_ema20 = current_price < current_ema20
            ema20_below_ema50 = current_ema20 < current_ema50

            if price_below_ema20 and ema20_below_ema50:
                score = 100
                reason = "Strong downtrend: Price < EMA20 < EMA50"
                passed = True
            elif price_below_ema20:
                score = 70
                reason = "Moderate downtrend: Price < EMA20"
                passed = True
            elif current_price < current_ema50:
                score = 50
                reason = "Weak downtrend: Price < EMA50 only"
                passed = True
            else:
                score = 20
                reason = "No downtrend alignment"
                passed = False

        return FactorResult(
            factor=ValidationFactor.TREND_ALIGNMENT,
            passed=passed,
            score=score,
            weight=self.weights[ValidationFactor.TREND_ALIGNMENT],
            reason=reason
        )

    def _check_momentum(
        self,
        df: pd.DataFrame,
        direction: str,
        col_map: dict,
        kalman_velocity: float = None
    ) -> FactorResult:
        """Check momentum alignment."""
        close = df[col_map['close']]

        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Also check recent price change
        pct_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

        if direction == 'BUY':
            # For BUY: RSI 40-70 is good, momentum positive
            rsi_ok = 35 <= current_rsi <= 75
            momentum_ok = pct_change > 0 or (kalman_velocity and kalman_velocity > 0)

            if rsi_ok and momentum_ok:
                score = 100 if current_rsi < 60 else 80  # Penalize overbought
                reason = f"Good buying momentum: RSI={current_rsi:.0f}"
                passed = True
            elif rsi_ok:
                score = 60
                reason = f"RSI OK but momentum weak: RSI={current_rsi:.0f}"
                passed = True
            elif current_rsi < 30:
                score = 70  # Oversold can be good for reversal
                reason = f"Oversold potential reversal: RSI={current_rsi:.0f}"
                passed = True
            else:
                score = 30
                reason = f"Weak buying momentum: RSI={current_rsi:.0f}"
                passed = False
        else:  # SELL
            rsi_ok = 25 <= current_rsi <= 65
            momentum_ok = pct_change < 0 or (kalman_velocity and kalman_velocity < 0)

            if rsi_ok and momentum_ok:
                score = 100 if current_rsi > 40 else 80
                reason = f"Good selling momentum: RSI={current_rsi:.0f}"
                passed = True
            elif rsi_ok:
                score = 60
                reason = f"RSI OK but momentum weak: RSI={current_rsi:.0f}"
                passed = True
            elif current_rsi > 70:
                score = 70  # Overbought can be good for reversal
                reason = f"Overbought potential reversal: RSI={current_rsi:.0f}"
                passed = True
            else:
                score = 30
                reason = f"Weak selling momentum: RSI={current_rsi:.0f}"
                passed = False

        return FactorResult(
            factor=ValidationFactor.MOMENTUM,
            passed=passed,
            score=score,
            weight=self.weights[ValidationFactor.MOMENTUM],
            reason=reason
        )

    def _check_volatility(self, df: pd.DataFrame, col_map: dict) -> FactorResult:
        """Check if volatility is acceptable."""
        high = df[col_map['high']]
        low = df[col_map['low']]
        close = df[col_map['close']]

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-50:].mean() if len(atr) >= 50 else atr.mean()

        # ATR ratio
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        if 0.7 <= atr_ratio <= 1.5:
            score = 100
            reason = f"Normal volatility: ATR ratio={atr_ratio:.2f}"
            passed = True
        elif 0.5 <= atr_ratio <= 2.0:
            score = 70
            reason = f"Acceptable volatility: ATR ratio={atr_ratio:.2f}"
            passed = True
        elif atr_ratio < 0.5:
            score = 40
            reason = f"Low volatility: ATR ratio={atr_ratio:.2f}"
            passed = False
        else:
            score = 40
            reason = f"High volatility: ATR ratio={atr_ratio:.2f}"
            passed = False

        return FactorResult(
            factor=ValidationFactor.VOLATILITY,
            passed=passed,
            score=score,
            weight=self.weights[ValidationFactor.VOLATILITY],
            reason=reason
        )

    def _check_poi_quality(self, poi_quality: float) -> FactorResult:
        """Check POI quality score."""
        if poi_quality >= 70:
            score = 100
            reason = f"High quality POI: {poi_quality:.0f}"
            passed = True
        elif poi_quality >= 50:
            score = 75
            reason = f"Good quality POI: {poi_quality:.0f}"
            passed = True
        elif poi_quality >= 35:
            score = 50
            reason = f"Acceptable POI: {poi_quality:.0f}"
            passed = True
        else:
            score = 25
            reason = f"Low quality POI: {poi_quality:.0f}"
            passed = False

        return FactorResult(
            factor=ValidationFactor.POI_QUALITY,
            passed=passed,
            score=score,
            weight=self.weights[ValidationFactor.POI_QUALITY],
            reason=reason
        )

    def _check_session(self, session_name: str, is_optimal: bool) -> FactorResult:
        """Check trading session."""
        if 'overlap' in session_name.lower():
            score = 100
            reason = "Optimal: London/NY Overlap"
            passed = True
        elif 'london' in session_name.lower() or 'new york' in session_name.lower():
            score = 85
            reason = f"Good session: {session_name}"
            passed = True
        elif is_optimal:
            score = 60
            reason = f"Acceptable: {session_name}"
            passed = True
        else:
            score = 30
            reason = f"Suboptimal session: {session_name}"
            passed = False

        return FactorResult(
            factor=ValidationFactor.SESSION_OPTIMAL,
            passed=passed,
            score=score,
            weight=self.weights[ValidationFactor.SESSION_OPTIMAL],
            reason=reason
        )

    def _check_regime(
        self,
        confidence: float,
        regime_bias: str,
        direction: str
    ) -> FactorResult:
        """Check regime confidence and alignment."""
        # Check alignment
        aligned = (
            (direction == 'BUY' and regime_bias == 'BUY') or
            (direction == 'SELL' and regime_bias == 'SELL')
        )

        if not aligned:
            return FactorResult(
                factor=ValidationFactor.REGIME_CONFIDENCE,
                passed=False,
                score=0,
                weight=self.weights[ValidationFactor.REGIME_CONFIDENCE],
                reason=f"Regime mismatch: {direction} vs {regime_bias}"
            )

        # Check confidence
        if confidence >= 75:
            score = 100
            reason = f"High regime confidence: {confidence:.0f}%"
            passed = True
        elif confidence >= 60:
            score = 75
            reason = f"Good regime confidence: {confidence:.0f}%"
            passed = True
        elif confidence >= 50:
            score = 50
            reason = f"Low regime confidence: {confidence:.0f}%"
            passed = True
        else:
            score = 25
            reason = f"Very low confidence: {confidence:.0f}%"
            passed = False

        return FactorResult(
            factor=ValidationFactor.REGIME_CONFIDENCE,
            passed=passed,
            score=score,
            weight=self.weights[ValidationFactor.REGIME_CONFIDENCE],
            reason=reason
        )

    def _check_drift(self, drift_detected: bool) -> FactorResult:
        """Check if drift was detected."""
        if not drift_detected:
            return FactorResult(
                factor=ValidationFactor.NO_DRIFT,
                passed=True,
                score=100,
                weight=self.weights[ValidationFactor.NO_DRIFT],
                reason="No drift detected - market stable"
            )
        else:
            return FactorResult(
                factor=ValidationFactor.NO_DRIFT,
                passed=False,
                score=20,
                weight=self.weights[ValidationFactor.NO_DRIFT],
                reason="Drift detected - market changing"
            )

    def _check_day_of_week(
        self,
        current_time: datetime,
        regime_confidence: float
    ) -> FactorResult:
        """Check day of week (Thursday caution)."""
        day = current_time.weekday()

        if day == 4:  # Friday
            return FactorResult(
                factor=ValidationFactor.DAY_OF_WEEK,
                passed=True,
                score=70,
                weight=self.weights[ValidationFactor.DAY_OF_WEEK],
                reason="Friday: Caution near close"
            )
        elif day == 3:  # Thursday
            # Thursday only ok if regime is strong
            if regime_confidence >= 70:
                return FactorResult(
                    factor=ValidationFactor.DAY_OF_WEEK,
                    passed=True,
                    score=60,
                    weight=self.weights[ValidationFactor.DAY_OF_WEEK],
                    reason="Thursday: OK with strong regime"
                )
            else:
                return FactorResult(
                    factor=ValidationFactor.DAY_OF_WEEK,
                    passed=False,
                    score=30,
                    weight=self.weights[ValidationFactor.DAY_OF_WEEK],
                    reason="Thursday: Weak regime - skip"
                )
        elif day == 0:  # Monday
            return FactorResult(
                factor=ValidationFactor.DAY_OF_WEEK,
                passed=True,
                score=80,
                weight=self.weights[ValidationFactor.DAY_OF_WEEK],
                reason="Monday: Wait for direction"
            )
        else:  # Tue, Wed
            return FactorResult(
                factor=ValidationFactor.DAY_OF_WEEK,
                passed=True,
                score=100,
                weight=self.weights[ValidationFactor.DAY_OF_WEEK],
                reason="Mid-week: Optimal"
            )


if __name__ == "__main__":
    import numpy as np

    print("\n" + "=" * 70)
    print("CONFLUENCE VALIDATOR TEST")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n = 100
    trend = np.linspace(0, 0.01, n)
    noise = np.random.randn(n) * 0.001
    close = 1.25 + trend + np.cumsum(noise)

    df = pd.DataFrame({
        'open': close - 0.0005,
        'high': close + np.random.rand(n) * 0.002,
        'low': close - np.random.rand(n) * 0.002,
        'close': close
    })

    validator = ConfluenceValidator(min_total_score=60, min_factors_passed=5)

    # Test BUY signal
    result = validator.validate(
        direction='BUY',
        df=df,
        poi_quality=65,
        regime_confidence=75,
        regime_bias='BUY',
        session_name='London/NY Overlap',
        is_optimal_session=True,
        drift_detected=False,
        current_time=datetime(2025, 1, 28, 14, 0, tzinfo=timezone.utc)  # Tuesday
    )

    print(f"\nBUY Signal Validation:")
    print(result.get_summary())

    # Test with drift
    result2 = validator.validate(
        direction='BUY',
        df=df,
        poi_quality=65,
        regime_confidence=75,
        regime_bias='BUY',
        session_name='London/NY Overlap',
        is_optimal_session=True,
        drift_detected=True,  # Drift!
        current_time=datetime(2025, 1, 30, 14, 0, tzinfo=timezone.utc)  # Thursday
    )

    print(f"\n\nBUY Signal with Drift + Thursday:")
    print(result2.get_summary())

    print("\n" + "=" * 70)
