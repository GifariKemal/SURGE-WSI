"""Kalman Filter for SURGE-WSI - NOISE REDUCTION
================================================

Layer 1 of 6-Layer Architecture

Function: Clean noise from price data for downstream analysis.
NOT for signal generation - that's the job of HMM + SMC.

Output:
- Smoothed price (clean price)
- Velocity (rate of change)
- Residual (deviation from smooth - for anomaly detection)

Author: SURIOTA Team
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None
    logger.warning("filterpy not installed, Kalman features disabled")


@dataclass
class KalmanState:
    """Output from Kalman Filter - clean data for further analysis"""
    raw_price: float           # Raw price (input)
    smoothed_price: float      # Price after noise reduction
    velocity: float            # Rate of change (for trend detection)
    acceleration: float        # Velocity change
    residual: float            # raw - smoothed (for anomaly detection)
    uncertainty: float         # Estimation uncertainty level


class KalmanNoiseReducer:
    """Single Kalman Filter for noise reduction

    Uses constant acceleration model for tracking:
    - State: [price, velocity, acceleration]
    - Output: cleaned price data for HMM/Regime detector input
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        name: str = "default"
    ):
        if KalmanFilter is None:
            raise ImportError("filterpy required for Kalman filtering")

        self.name = name
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State: [price, velocity, acceleration]
        self.kf = KalmanFilter(dim_x=3, dim_z=1)

        # State transition matrix (constant acceleration model)
        dt = 1.0
        self.kf.F = np.array([
            [1., dt, 0.5 * dt**2],
            [0., 1., dt],
            [0., 0., 1.]
        ])

        # Measurement function (observe price only)
        self.kf.H = np.array([[1., 0., 0.]])

        # Process noise covariance
        q = process_noise
        self.kf.Q = np.array([
            [q * 0.25, q * 0.5, q * 0.5],
            [q * 0.5, q, q],
            [q * 0.5, q, q]
        ])

        # Measurement noise covariance
        self.kf.R = np.array([[measurement_noise]])

        # Initial state
        self.kf.x = np.array([[0.], [0.], [0.]])
        self.kf.P *= 1000.

        self._initialized = False
        self._history: List[KalmanState] = []

    def update(self, price: float) -> KalmanState:
        """Update filter with new price, return cleaned data

        Args:
            price: Raw price from market

        Returns:
            KalmanState with smoothed data
        """
        # Initialize on first update
        if not self._initialized:
            self.kf.x = np.array([[price], [0.], [0.]])
            self._initialized = True

        # Kalman predict & update
        self.kf.predict()
        self.kf.update(np.array([[price]]))

        # Extract state
        smoothed = float(self.kf.x[0, 0])
        velocity = float(self.kf.x[1, 0])
        acceleration = float(self.kf.x[2, 0])
        uncertainty = float(np.sqrt(self.kf.P[0, 0]))

        # Calculate residual (deviation from smooth)
        residual = price - smoothed

        state = KalmanState(
            raw_price=price,
            smoothed_price=smoothed,
            velocity=velocity,
            acceleration=acceleration,
            residual=residual,
            uncertainty=uncertainty
        )

        self._history.append(state)
        if len(self._history) > 500:
            self._history = self._history[-500:]

        return state

    def get_smoothed_series(self, n: int = 50) -> np.ndarray:
        """Get array of recent smoothed prices"""
        if len(self._history) < n:
            return np.array([s.smoothed_price for s in self._history])
        return np.array([s.smoothed_price for s in self._history[-n:]])

    def get_velocity_series(self, n: int = 50) -> np.ndarray:
        """Get array of recent velocities"""
        if len(self._history) < n:
            return np.array([s.velocity for s in self._history])
        return np.array([s.velocity for s in self._history[-n:]])

    def get_residual_series(self, n: int = 50) -> np.ndarray:
        """Get array of recent residuals"""
        if len(self._history) < n:
            return np.array([s.residual for s in self._history])
        return np.array([s.residual for s in self._history[-n:]])

    def reset(self, initial_price: float = None):
        """Reset filter state"""
        if initial_price:
            self.kf.x = np.array([[initial_price], [0.], [0.]])
            self._initialized = True
        else:
            self.kf.x = np.array([[0.], [0.], [0.]])
            self._initialized = False
        self.kf.P *= 1000.
        self._history.clear()


class MultiScaleKalman:
    """Multi-scale Kalman for analysis across different timeframes

    Combines 3 filters with different sensitivities:
    - Fast: Sensitive to quick changes (scalping)
    - Medium: Balanced (intraday)
    - Slow: Smooth (swing trading)

    Output: Clean data for HMM regime detection
    """

    def __init__(
        self,
        fast_process_noise: float = 0.05,
        fast_measurement_noise: float = 0.05,
        medium_process_noise: float = 0.01,
        medium_measurement_noise: float = 0.1,
        slow_process_noise: float = 0.001,
        slow_measurement_noise: float = 0.2
    ):
        # Fast filter (responsive to quick changes)
        self.fast = KalmanNoiseReducer(
            process_noise=fast_process_noise,
            measurement_noise=fast_measurement_noise,
            name="fast"
        )

        # Medium filter (balanced)
        self.medium = KalmanNoiseReducer(
            process_noise=medium_process_noise,
            measurement_noise=medium_measurement_noise,
            name="medium"
        )

        # Slow filter (smooth trend)
        self.slow = KalmanNoiseReducer(
            process_noise=slow_process_noise,
            measurement_noise=slow_measurement_noise,
            name="slow"
        )

        self._last_state: Optional[Dict] = None

    def update(self, price: float) -> Dict:
        """Update all filters, return cleaned data for HMM

        Args:
            price: Raw market price

        Returns:
            Dict with smoothed data from all timeframes
        """
        fast_state = self.fast.update(price)
        medium_state = self.medium.update(price)
        slow_state = self.slow.update(price)

        # Primary smoothed price (medium filter)
        primary_smoothed = medium_state.smoothed_price

        # Trend direction from slow filter
        trend_velocity = slow_state.velocity

        # Volatility indicator from residual std
        residuals = self.medium.get_residual_series(20)
        volatility = float(np.std(residuals)) if len(residuals) > 5 else 0.0

        # Returns for HMM input (smoothed)
        smoothed_prices = self.medium.get_smoothed_series(50)
        if len(smoothed_prices) >= 2:
            returns = np.diff(smoothed_prices) / smoothed_prices[:-1]
            current_return = float(returns[-1]) if len(returns) > 0 else 0.0
        else:
            returns = np.array([0.0])
            current_return = 0.0

        self._last_state = {
            # Primary output for downstream
            'smoothed_price': primary_smoothed,
            'raw_price': price,
            'velocity': medium_state.velocity,
            'acceleration': medium_state.acceleration,

            # For HMM input
            'returns': returns,
            'current_return': current_return,
            'volatility': volatility,

            # Multi-scale analysis
            'fast': {
                'smoothed': fast_state.smoothed_price,
                'velocity': fast_state.velocity,
                'residual': fast_state.residual
            },
            'medium': {
                'smoothed': medium_state.smoothed_price,
                'velocity': medium_state.velocity,
                'residual': medium_state.residual
            },
            'slow': {
                'smoothed': slow_state.smoothed_price,
                'velocity': slow_state.velocity,
                'residual': slow_state.residual
            },

            # Trend analysis (from slow filter)
            'trend_velocity': trend_velocity,
            'trend_direction': 'UP' if trend_velocity > 0 else 'DOWN' if trend_velocity < 0 else 'FLAT',

            # Noise level
            'noise_level': volatility,
        }

        return self._last_state

    def get_hmm_features(self, lookback: int = 50) -> np.ndarray:
        """Get feature array for HMM input

        Returns:
            2D array: [returns, volatility] for each timestep
        """
        returns = self.medium.get_velocity_series(lookback)
        residuals = self.medium.get_residual_series(lookback)

        if len(returns) < lookback:
            # Pad with zeros if not enough data
            pad_size = lookback - len(returns)
            returns = np.pad(returns, (pad_size, 0), mode='constant')
            residuals = np.pad(residuals, (pad_size, 0), mode='constant')

        # Calculate rolling volatility
        volatility = np.zeros(len(residuals))
        for i in range(5, len(residuals)):
            volatility[i] = np.std(residuals[i-5:i])

        return np.column_stack([returns, volatility])

    def reset(self, initial_price: float = None):
        """Reset all filters"""
        self.fast.reset(initial_price)
        self.medium.reset(initial_price)
        self.slow.reset(initial_price)
        self._last_state = None

    @property
    def last_state(self) -> Optional[Dict]:
        """Get last computed state"""
        return self._last_state
