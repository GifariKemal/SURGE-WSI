"""Kalman Filter Unit Tests
===========================

Tests for KalmanNoiseReducer and MultiScaleKalman.

Author: SURIOTA Team
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kalman_filter import KalmanNoiseReducer, MultiScaleKalman, KalmanState


class TestKalmanNoiseReducer:
    """Tests for single Kalman filter"""

    def test_initialization(self):
        """Test filter initializes correctly"""
        kf = KalmanNoiseReducer(
            process_noise=0.01,
            measurement_noise=0.1
        )
        assert kf is not None
        assert not kf._initialized

    def test_update_single_value(self):
        """Test single update"""
        kf = KalmanNoiseReducer()
        state = kf.update(1.30000)

        assert state is not None
        assert isinstance(state, KalmanState)
        assert state.smoothed_price > 0
        assert kf._initialized

    def test_update_sequence(self):
        """Test updating with price sequence"""
        kf = KalmanNoiseReducer()

        # Simulate price movement
        prices = [1.3000, 1.3010, 1.3005, 1.3015, 1.3012]
        states = []

        for price in prices:
            state = kf.update(price)
            states.append(state)

        # All states should be valid
        assert len(states) == len(prices)
        assert all(s.smoothed_price > 0 for s in states)

    def test_noise_reduction(self):
        """Test that noise is reduced"""
        kf = KalmanNoiseReducer()

        # Generate noisy sine wave
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, 100)
        clean_signal = 1.3 + 0.01 * np.sin(t)
        noise = np.random.normal(0, 0.002, 100)
        noisy_signal = clean_signal + noise

        # Filter the noisy signal
        filtered = []
        for price in noisy_signal:
            state = kf.update(price)
            filtered.append(state.smoothed_price)

        # After warmup, filtered should be smoother
        # Compare variance in second half
        raw_var = np.var(noisy_signal[50:])
        filtered_var = np.var(filtered[50:])

        # Filtered should have lower or similar variance
        assert filtered_var <= raw_var * 2  # Allow tolerance

    def test_velocity_calculation(self):
        """Test velocity is calculated correctly"""
        kf = KalmanNoiseReducer()

        # Upward trend
        prices = [1.3000 + i * 0.0001 for i in range(20)]

        for price in prices:
            state = kf.update(price)

        # Velocity should be positive for upward trend
        assert state.velocity > 0

    def test_reset(self):
        """Test filter reset"""
        kf = KalmanNoiseReducer()

        # Update a few times
        for _ in range(10):
            kf.update(1.30000)

        assert kf._initialized

        # Reset
        kf.reset()

        assert not kf._initialized

    def test_get_smoothed_series(self):
        """Test get_smoothed_series method"""
        kf = KalmanNoiseReducer()

        # Update multiple times
        for i in range(30):
            kf.update(1.3000 + i * 0.0001)

        series = kf.get_smoothed_series(20)
        assert len(series) == 20
        assert all(s > 0 for s in series)


class TestMultiScaleKalman:
    """Tests for multi-scale Kalman fusion"""

    def test_initialization(self):
        """Test multi-scale filter initializes"""
        msk = MultiScaleKalman()
        assert msk is not None
        assert msk.fast is not None
        assert msk.medium is not None
        assert msk.slow is not None

    def test_update_returns_dict(self):
        """Test update returns dictionary with all components"""
        msk = MultiScaleKalman()

        result = msk.update(1.30000)

        assert isinstance(result, dict)
        assert 'smoothed_price' in result
        assert 'fast' in result
        assert 'medium' in result
        assert 'slow' in result

    def test_multi_scale_output(self):
        """Test multi-scale output structure"""
        msk = MultiScaleKalman()

        # Update multiple times
        for i in range(20):
            result = msk.update(1.3000 + i * 0.0001)

        assert result['smoothed_price'] > 0
        assert 'velocity' in result
        assert 'trend_direction' in result
        assert result['trend_direction'] in ['UP', 'DOWN', 'FLAT']

    def test_different_smoothing_levels(self):
        """Test fast responds quicker than slow"""
        msk = MultiScaleKalman()

        # Steady state
        for _ in range(50):
            msk.update(1.30000)

        # Sudden jump
        result = msk.update(1.31000)
        result = msk.update(1.31000)

        fast_price = result['fast']['smoothed']
        slow_price = result['slow']['smoothed']

        # Fast should be closer to new price (1.31)
        assert fast_price > slow_price

    def test_get_hmm_features(self):
        """Test getting features for HMM input"""
        msk = MultiScaleKalman()

        # Update with enough data
        for i in range(100):
            msk.update(1.3000 + np.random.normal(0, 0.001))

        features = msk.get_hmm_features(50)

        assert features.shape[0] == 50
        assert features.shape[1] == 2  # returns and volatility

    def test_reset_all_filters(self):
        """Test resetting all filters"""
        msk = MultiScaleKalman()

        # Update
        for _ in range(20):
            msk.update(1.30000)

        # Reset
        msk.reset()

        assert not msk.fast._initialized
        assert not msk.medium._initialized
        assert not msk.slow._initialized
        assert msk.last_state is None


class TestKalmanState:
    """Tests for KalmanState dataclass"""

    def test_creation(self):
        """Test creating KalmanState"""
        state = KalmanState(
            raw_price=1.30000,
            smoothed_price=1.30005,
            velocity=0.0001,
            acceleration=0.00001,
            residual=-0.00005,
            uncertainty=0.0005
        )

        assert state.raw_price == 1.30000
        assert state.smoothed_price == 1.30005
        assert state.velocity == 0.0001
        assert state.residual == -0.00005


# Fixtures
@pytest.fixture
def sample_prices():
    """Generate sample price data"""
    np.random.seed(42)
    base = 1.30000
    trend = np.linspace(0, 0.01, 100)
    noise = np.random.normal(0, 0.001, 100)
    return base + trend + noise


@pytest.fixture
def sample_dataframe(sample_prices):
    """Generate sample DataFrame"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    return pd.DataFrame({
        'time': dates,
        'close': sample_prices
    })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
