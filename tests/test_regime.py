"""Regime Detector Unit Tests
=============================

Tests for HMMRegimeDetector.

Author: SURIOTA Team
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.regime_detector import (
    HMMRegimeDetector,
    MarketRegime,
    RegimeInfo
)


class TestMarketRegime:
    """Tests for MarketRegime enum"""

    def test_regime_values(self):
        """Test regime enum values"""
        assert MarketRegime.BULLISH.value == "BULLISH"
        assert MarketRegime.BEARISH.value == "BEARISH"
        assert MarketRegime.SIDEWAYS.value == "SIDEWAYS"
        assert MarketRegime.UNKNOWN.value == "UNKNOWN"

    def test_all_regimes(self):
        """Test all regimes exist"""
        regimes = list(MarketRegime)
        assert len(regimes) == 4  # BULLISH, BEARISH, SIDEWAYS, UNKNOWN


class TestRegimeInfo:
    """Tests for RegimeInfo dataclass"""

    def test_creation(self):
        """Test creating RegimeInfo"""
        info = RegimeInfo(
            regime=MarketRegime.BULLISH,
            probability=0.85,
            probabilities={"BULLISH": 0.85, "BEARISH": 0.10, "SIDEWAYS": 0.05},
            bias="BUY"
        )

        assert info.regime == MarketRegime.BULLISH
        assert info.probability == 0.85
        assert info.bias == "BUY"

    def test_is_tradeable_bullish(self):
        """Test is_tradeable for bullish regime"""
        info = RegimeInfo(
            regime=MarketRegime.BULLISH,
            probability=0.80,
            probabilities={"BULLISH": 0.80},
            bias="BUY"
        )

        assert info.is_tradeable is True
        assert info.should_buy_only is True
        assert info.should_sell_only is False

    def test_is_tradeable_bearish(self):
        """Test is_tradeable for bearish regime"""
        info = RegimeInfo(
            regime=MarketRegime.BEARISH,
            probability=0.75,
            probabilities={"BEARISH": 0.75},
            bias="SELL"
        )

        assert info.is_tradeable is True
        assert info.should_buy_only is False
        assert info.should_sell_only is True

    def test_is_tradeable_sideways(self):
        """Test is_tradeable for sideways regime"""
        info = RegimeInfo(
            regime=MarketRegime.SIDEWAYS,
            probability=0.70,
            probabilities={"SIDEWAYS": 0.70},
            bias="NONE"
        )

        assert info.is_tradeable is False

    def test_is_tradeable_low_probability(self):
        """Test is_tradeable with low probability"""
        info = RegimeInfo(
            regime=MarketRegime.BULLISH,
            probability=0.50,  # Below threshold
            probabilities={"BULLISH": 0.50},
            bias="NONE"
        )

        assert info.is_tradeable is False

    def test_to_dict(self):
        """Test conversion to dictionary"""
        info = RegimeInfo(
            regime=MarketRegime.BULLISH,
            probability=0.85,
            probabilities={"BULLISH": 0.85, "BEARISH": 0.10, "SIDEWAYS": 0.05},
            bias="BUY"
        )

        d = info.to_dict()

        assert d['regime'] == "BULLISH"
        assert d['probability'] == 0.85
        assert d['bias'] == "BUY"
        assert d['is_tradeable'] is True


class TestHMMRegimeDetector:
    """Tests for HMM Regime Detector"""

    def test_initialization(self):
        """Test detector initializes correctly"""
        detector = HMMRegimeDetector(
            n_states=3,
            lookback=50,
            min_probability=0.65
        )

        assert detector is not None
        assert detector.n_states == 3
        assert detector.lookback == 50

    def test_update_returns_regime_info(self):
        """Test update returns RegimeInfo"""
        detector = HMMRegimeDetector(lookback=50)

        # Update with some prices
        for i in range(30):
            result = detector.update(1.3000 + i * 0.0001)

        assert result is not None
        assert isinstance(result, RegimeInfo)
        assert result.regime in list(MarketRegime)

    def test_update_with_trend(self):
        """Test detection with upward trend"""
        detector = HMMRegimeDetector(lookback=50)

        # Strong uptrend
        for i in range(100):
            result = detector.update(1.3000 + i * 0.001)

        # Should have a result
        assert result is not None
        assert result.regime in list(MarketRegime)

    def test_get_trading_bias(self):
        """Test get_trading_bias method"""
        detector = HMMRegimeDetector(lookback=50)

        for i in range(100):
            detector.update(1.3000 + i * 0.0005)

        bias, confidence = detector.get_trading_bias()

        assert bias in ["BUY", "SELL", "NONE"]
        assert 0 <= confidence <= 1

    def test_should_trade(self):
        """Test should_trade method"""
        detector = HMMRegimeDetector(lookback=50)

        for i in range(100):
            detector.update(1.3000 + i * 0.001)

        result = detector.should_trade()
        assert isinstance(result, bool)

    def test_reset(self):
        """Test detector reset"""
        detector = HMMRegimeDetector(lookback=50)

        # Train
        for i in range(100):
            detector.update(1.3000 + i * 0.0001)

        # Reset
        detector.reset()

        # last_info should be None
        assert detector.last_info is None
        assert len(detector.price_history) == 0


class TestRegimeDetectorEdgeCases:
    """Edge case tests"""

    def test_single_price(self):
        """Test with single price update"""
        detector = HMMRegimeDetector()

        result = detector.update(1.30000)

        # Should return unknown with low data
        assert result is not None
        assert result.regime == MarketRegime.UNKNOWN

    def test_constant_prices(self):
        """Test with constant prices"""
        detector = HMMRegimeDetector(lookback=50)

        # All same price
        for _ in range(100):
            result = detector.update(1.30000)

        # Should handle gracefully (likely sideways)
        assert result is not None

    def test_extreme_volatility(self):
        """Test with extreme volatility"""
        detector = HMMRegimeDetector(lookback=50)

        np.random.seed(42)
        for _ in range(100):
            price = 1.3000 + np.random.normal(0, 0.05)
            result = detector.update(price)

        # Should handle without crashing
        assert result is not None


# Fixtures
@pytest.fixture
def uptrend_prices():
    """Generate uptrend price data"""
    return [1.3000 + i * 0.0005 for i in range(100)]


@pytest.fixture
def downtrend_prices():
    """Generate downtrend price data"""
    return [1.3500 - i * 0.0005 for i in range(100)]


@pytest.fixture
def sideways_prices():
    """Generate sideways price data"""
    np.random.seed(42)
    return [1.3000 + np.random.normal(0, 0.001) for _ in range(100)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
