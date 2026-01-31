"""Drift Detector v1 - ADWIN-based Market Change Detection
==========================================================

Detects when market conditions change significantly (concept drift).
Based on River library's ADWIN algorithm.

Research basis:
- ADWIN (ADaptive WINdowing) efficiently detects distribution changes
- Detects regime changes hours/days before losses compound
- From: https://riverml.xyz/dev/api/drift/ADWIN/

Author: SURIOTA Team
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from loguru import logger

try:
    from river.drift import ADWIN
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logger.warning("River library not available, using fallback drift detection")


@dataclass
class DriftResult:
    """Result of drift detection analysis"""
    drift_detected: bool
    drift_type: str  # 'none', 'volatility', 'trend', 'regime'
    confidence: float  # 0-100

    # Metrics
    current_mean: float
    historical_mean: float
    deviation: float

    # Recommendations
    should_reduce_position: bool
    should_skip_trading: bool

    # Details
    message: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def __str__(self):
        status = "🚨 DRIFT" if self.drift_detected else "✅ STABLE"
        return f"{status} | {self.drift_type} | conf={self.confidence:.0f}% | dev={self.deviation:.2f}"


class ADWINDriftDetector:
    """
    ADWIN-based drift detector for trading.

    Monitors multiple streams:
    1. Price returns - detect trend changes
    2. Volatility - detect volatility regime shifts
    3. Win rate - detect strategy degradation
    """

    def __init__(
        self,
        delta: float = 0.002,  # Sensitivity (lower = more sensitive)
        min_samples: int = 30,
        volatility_window: int = 20,
        return_threshold: float = 0.5,  # std devs for significant change
        volatility_threshold: float = 0.3,
    ):
        """
        Initialize ADWIN Drift Detector.

        Args:
            delta: ADWIN sensitivity (0.002 default, lower = more sensitive)
            min_samples: Minimum samples before drift detection active
            volatility_window: Window for volatility calculation
            return_threshold: Threshold for return drift (in std devs)
            volatility_threshold: Threshold for volatility drift
        """
        self.delta = delta
        self.min_samples = min_samples
        self.volatility_window = volatility_window
        self.return_threshold = return_threshold
        self.volatility_threshold = volatility_threshold

        # Initialize ADWIN detectors (or fallback)
        if RIVER_AVAILABLE:
            self._return_detector = ADWIN(delta=delta)
            self._volatility_detector = ADWIN(delta=delta)
            self._winrate_detector = ADWIN(delta=delta * 10)  # Less sensitive for win rate
        else:
            self._return_detector = None
            self._volatility_detector = None
            self._winrate_detector = None

        # Historical data for fallback and analysis
        self._returns: deque = deque(maxlen=200)
        self._volatilities: deque = deque(maxlen=200)
        self._win_rates: deque = deque(maxlen=50)

        # State
        self._sample_count = 0
        self._last_drift_time: Optional[datetime] = None
        self._drift_cooldown_hours = 4  # Don't flag drift again within cooldown

        # Stats
        self._total_drifts_detected = 0
        self._return_drifts = 0
        self._volatility_drifts = 0

        logger.info(
            f"ADWINDriftDetector initialized: delta={delta}, "
            f"min_samples={min_samples}, river={'yes' if RIVER_AVAILABLE else 'no'}"
        )

    def update_price(self, price: float, prev_price: float) -> None:
        """Update with new price data."""
        if prev_price <= 0:
            return

        # Calculate return
        ret = (price - prev_price) / prev_price
        self._returns.append(ret)
        self._sample_count += 1

        # Update ADWIN detector
        if RIVER_AVAILABLE and self._return_detector:
            self._return_detector.update(ret)

        # Update volatility
        if len(self._returns) >= self.volatility_window:
            recent_returns = list(self._returns)[-self.volatility_window:]
            vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
            self._volatilities.append(vol)

            if RIVER_AVAILABLE and self._volatility_detector:
                self._volatility_detector.update(vol)

    def update_trade_result(self, is_win: bool) -> None:
        """Update with trade result for win rate drift detection."""
        self._win_rates.append(1.0 if is_win else 0.0)

        if RIVER_AVAILABLE and self._winrate_detector:
            self._winrate_detector.update(1.0 if is_win else 0.0)

    def detect(self, current_time: datetime = None) -> DriftResult:
        """
        Detect if drift has occurred.

        Returns:
            DriftResult with drift status and recommendations
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check cooldown
        if self._last_drift_time:
            hours_since_drift = (current_time - self._last_drift_time).total_seconds() / 3600
            if hours_since_drift < self._drift_cooldown_hours:
                return DriftResult(
                    drift_detected=False,
                    drift_type='cooldown',
                    confidence=0,
                    current_mean=0,
                    historical_mean=0,
                    deviation=0,
                    should_reduce_position=True,  # Still cautious during cooldown
                    should_skip_trading=False,
                    message=f"In drift cooldown ({hours_since_drift:.1f}h/{self._drift_cooldown_hours}h)"
                )

        # Not enough data
        if self._sample_count < self.min_samples:
            return DriftResult(
                drift_detected=False,
                drift_type='none',
                confidence=0,
                current_mean=0,
                historical_mean=0,
                deviation=0,
                should_reduce_position=False,
                should_skip_trading=False,
                message=f"Warming up ({self._sample_count}/{self.min_samples} samples)"
            )

        # Check for drift using ADWIN or fallback
        return_drift = self._check_return_drift()
        volatility_drift = self._check_volatility_drift()
        winrate_drift = self._check_winrate_drift()

        # Determine overall drift status
        drift_detected = return_drift[0] or volatility_drift[0] or winrate_drift[0]

        if drift_detected:
            self._total_drifts_detected += 1
            self._last_drift_time = current_time

            # Determine drift type (prioritize most impactful)
            if winrate_drift[0]:
                drift_type = 'strategy'
                confidence = winrate_drift[1]
                deviation = winrate_drift[2]
                message = f"Strategy performance degradation detected: WR changed by {deviation:.1%}"
                should_skip = True
            elif volatility_drift[0]:
                drift_type = 'volatility'
                confidence = volatility_drift[1]
                deviation = volatility_drift[2]
                message = f"Volatility regime change: {deviation:+.1%} shift"
                should_skip = False
                self._volatility_drifts += 1
            else:
                drift_type = 'trend'
                confidence = return_drift[1]
                deviation = return_drift[2]
                message = f"Trend/momentum shift detected: {deviation:+.2%}"
                should_skip = False
                self._return_drifts += 1

            return DriftResult(
                drift_detected=True,
                drift_type=drift_type,
                confidence=confidence,
                current_mean=self._get_current_mean(),
                historical_mean=self._get_historical_mean(),
                deviation=deviation,
                should_reduce_position=True,
                should_skip_trading=should_skip,
                message=message,
                timestamp=current_time
            )

        return DriftResult(
            drift_detected=False,
            drift_type='none',
            confidence=100 - max(return_drift[1], volatility_drift[1]),
            current_mean=self._get_current_mean(),
            historical_mean=self._get_historical_mean(),
            deviation=0,
            should_reduce_position=False,
            should_skip_trading=False,
            message="Market conditions stable"
        )

    def _check_return_drift(self) -> Tuple[bool, float, float]:
        """Check for return/trend drift."""
        if RIVER_AVAILABLE and self._return_detector:
            if self._return_detector.drift_detected:
                # Calculate deviation
                recent = list(self._returns)[-20:]
                older = list(self._returns)[-100:-20] if len(self._returns) > 20 else []

                if recent and older:
                    recent_mean = np.mean(recent)
                    older_mean = np.mean(older)
                    older_std = np.std(older) if np.std(older) > 0 else 0.0001
                    deviation = (recent_mean - older_mean) / older_std
                    confidence = min(100, abs(deviation) * 30)
                    return (abs(deviation) > self.return_threshold, confidence, recent_mean - older_mean)

        # Fallback: simple statistical test
        if len(self._returns) >= 50:
            recent = list(self._returns)[-20:]
            older = list(self._returns)[-100:-20]

            if recent and older:
                recent_mean = np.mean(recent)
                older_mean = np.mean(older)
                older_std = np.std(older) if np.std(older) > 0 else 0.0001
                z_score = abs(recent_mean - older_mean) / (older_std / np.sqrt(len(recent)))

                if z_score > 2.0:  # ~95% confidence
                    confidence = min(100, z_score * 25)
                    return (True, confidence, recent_mean - older_mean)

        return (False, 0, 0)

    def _check_volatility_drift(self) -> Tuple[bool, float, float]:
        """Check for volatility regime drift."""
        if len(self._volatilities) < 30:
            return (False, 0, 0)

        if RIVER_AVAILABLE and self._volatility_detector:
            if self._volatility_detector.drift_detected:
                recent = list(self._volatilities)[-10:]
                older = list(self._volatilities)[-50:-10]

                if recent and older:
                    recent_vol = np.mean(recent)
                    older_vol = np.mean(older)
                    change_pct = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
                    confidence = min(100, abs(change_pct) * 200)
                    return (abs(change_pct) > self.volatility_threshold, confidence, change_pct)

        # Fallback
        recent = list(self._volatilities)[-10:]
        older = list(self._volatilities)[-50:-10]

        if recent and older:
            recent_vol = np.mean(recent)
            older_vol = np.mean(older)
            change_pct = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0

            if abs(change_pct) > self.volatility_threshold:
                confidence = min(100, abs(change_pct) * 200)
                return (True, confidence, change_pct)

        return (False, 0, 0)

    def _check_winrate_drift(self) -> Tuple[bool, float, float]:
        """Check for win rate degradation."""
        if len(self._win_rates) < 20:
            return (False, 0, 0)

        recent = list(self._win_rates)[-10:]
        older = list(self._win_rates)[-30:-10] if len(self._win_rates) >= 30 else list(self._win_rates)[:-10]

        if recent and older:
            recent_wr = np.mean(recent)
            older_wr = np.mean(older)
            change = recent_wr - older_wr

            # Significant degradation: WR dropped by >15%
            if change < -0.15:
                confidence = min(100, abs(change) * 300)
                return (True, confidence, change)

        return (False, 0, 0)

    def _get_current_mean(self) -> float:
        """Get current return mean."""
        if len(self._returns) >= 20:
            return np.mean(list(self._returns)[-20:])
        return 0.0

    def _get_historical_mean(self) -> float:
        """Get historical return mean."""
        if len(self._returns) >= 50:
            return np.mean(list(self._returns)[-100:-20])
        return 0.0

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            'sample_count': self._sample_count,
            'total_drifts': self._total_drifts_detected,
            'return_drifts': self._return_drifts,
            'volatility_drifts': self._volatility_drifts,
            'current_volatility': self._volatilities[-1] if self._volatilities else 0,
            'avg_volatility': np.mean(list(self._volatilities)) if self._volatilities else 0,
            'river_available': RIVER_AVAILABLE
        }

    def reset(self) -> None:
        """Reset detector state."""
        if RIVER_AVAILABLE:
            self._return_detector = ADWIN(delta=self.delta)
            self._volatility_detector = ADWIN(delta=self.delta)
            self._winrate_detector = ADWIN(delta=self.delta * 10)

        self._returns.clear()
        self._volatilities.clear()
        self._win_rates.clear()
        self._sample_count = 0
        self._last_drift_time = None

        logger.info("Drift detector reset")


class MultiStreamDriftDetector:
    """
    Multi-stream drift detector that monitors multiple data streams.
    Useful for monitoring different aspects of market simultaneously.
    """

    def __init__(self):
        """Initialize multi-stream detector."""
        self.detectors = {
            'price': ADWINDriftDetector(delta=0.002),
            'volume': ADWINDriftDetector(delta=0.005),  # Less sensitive
        }

    def update(self, stream: str, value: float, prev_value: float = None) -> None:
        """Update specific stream."""
        if stream in self.detectors:
            if prev_value is not None:
                self.detectors[stream].update_price(value, prev_value)

    def detect_any(self, current_time: datetime = None) -> DriftResult:
        """Detect drift in any stream."""
        for name, detector in self.detectors.items():
            result = detector.detect(current_time)
            if result.drift_detected:
                result.message = f"[{name}] {result.message}"
                return result

        return DriftResult(
            drift_detected=False,
            drift_type='none',
            confidence=100,
            current_mean=0,
            historical_mean=0,
            deviation=0,
            should_reduce_position=False,
            should_skip_trading=False,
            message="All streams stable"
        )


if __name__ == "__main__":
    # Test
    import random

    print("\n" + "=" * 60)
    print("ADWIN DRIFT DETECTOR TEST")
    print("=" * 60)

    detector = ADWINDriftDetector(delta=0.002, min_samples=30)

    # Simulate stable market
    print("\nPhase 1: Stable market (50 samples)")
    base_price = 1.2500
    for i in range(50):
        noise = random.gauss(0, 0.0005)  # Small noise
        new_price = base_price + noise
        detector.update_price(new_price, base_price)
        base_price = new_price

    result = detector.detect()
    print(f"  Result: {result}")

    # Simulate trend change
    print("\nPhase 2: Trend change (30 samples with upward drift)")
    for i in range(30):
        noise = random.gauss(0.001, 0.0005)  # Upward bias
        new_price = base_price + noise
        detector.update_price(new_price, base_price)
        base_price = new_price

    result = detector.detect()
    print(f"  Result: {result}")

    # Simulate volatility spike
    print("\nPhase 3: Volatility spike (20 samples)")
    for i in range(20):
        noise = random.gauss(0, 0.003)  # Higher volatility
        new_price = base_price + noise
        detector.update_price(new_price, base_price)
        base_price = new_price

    result = detector.detect()
    print(f"  Result: {result}")

    print(f"\nStats: {detector.get_stats()}")
    print("=" * 60)
