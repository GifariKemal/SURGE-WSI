"""POI Detector Unit Tests
==========================

Tests for POIDetector (Order Blocks + FVG).

Author: SURIOTA Team
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.poi_detector import (
    POIDetector,
    OrderBlock,
    FairValueGap,
    POIResult,
    POIType
)


class TestOrderBlock:
    """Tests for OrderBlock dataclass"""

    def test_creation(self):
        """Test creating OrderBlock"""
        ob = OrderBlock(
            id="ob_1",
            poi_type=POIType.ORDER_BLOCK_BULLISH,
            top=1.3010,
            bottom=1.3000,
            volume=1000,
            strength=75.0
        )

        assert ob.poi_type == POIType.ORDER_BLOCK_BULLISH
        assert ob.top == 1.3010
        assert ob.bottom == 1.3000
        assert ob.strength == 75.0

    def test_mid_property(self):
        """Test mid calculation"""
        ob = OrderBlock(
            id="ob_1",
            poi_type=POIType.ORDER_BLOCK_BULLISH,
            top=1.3010,
            bottom=1.3000,
            volume=1000,
            strength=75.0
        )

        assert ob.mid == pytest.approx(1.3005, rel=1e-5)

    def test_size_pips(self):
        """Test size_pips calculation"""
        ob = OrderBlock(
            id="ob_1",
            poi_type=POIType.ORDER_BLOCK_BEARISH,
            top=1.3020,
            bottom=1.3000,
            volume=1000,
            strength=80.0
        )

        assert ob.size_pips == pytest.approx(20.0, rel=1e-3)

    def test_direction(self):
        """Test direction property"""
        bull_ob = OrderBlock(
            id="ob_1",
            poi_type=POIType.ORDER_BLOCK_BULLISH,
            top=1.3010,
            bottom=1.3000
        )
        bear_ob = OrderBlock(
            id="ob_2",
            poi_type=POIType.ORDER_BLOCK_BEARISH,
            top=1.3010,
            bottom=1.3000
        )

        assert bull_ob.direction == "BUY"
        assert bear_ob.direction == "SELL"

    def test_contains_price(self):
        """Test contains_price method"""
        ob = OrderBlock(
            id="ob_1",
            poi_type=POIType.ORDER_BLOCK_BULLISH,
            top=1.3010,
            bottom=1.3000
        )

        assert ob.contains_price(1.3005) is True
        assert ob.contains_price(1.3020) is False

    def test_to_dict(self):
        """Test conversion to dictionary"""
        ob = OrderBlock(
            id="ob_1",
            poi_type=POIType.ORDER_BLOCK_BULLISH,
            top=1.3010,
            bottom=1.3000,
            volume=1000,
            strength=75.0
        )

        d = ob.to_dict()

        assert d['type'] == "OB_BULL"
        assert d['top'] == 1.3010
        assert d['bottom'] == 1.3000
        assert 'mid' in d
        assert 'size_pips' in d


class TestFairValueGap:
    """Tests for FairValueGap dataclass"""

    def test_creation(self):
        """Test creating FairValueGap"""
        fvg = FairValueGap(
            id="fvg_1",
            poi_type=POIType.FVG_BULLISH,
            high=1.3015,
            low=1.3005
        )

        assert fvg.poi_type == POIType.FVG_BULLISH
        assert fvg.high == 1.3015
        assert fvg.low == 1.3005

    def test_size_pips(self):
        """Test FVG size calculation"""
        fvg = FairValueGap(
            id="fvg_1",
            poi_type=POIType.FVG_BULLISH,
            high=1.3015,
            low=1.3005
        )

        assert fvg.size_pips == pytest.approx(10.0, rel=1e-3)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        fvg = FairValueGap(
            id="fvg_1",
            poi_type=POIType.FVG_BEARISH,
            high=1.3020,
            low=1.3010,
            fill_percentage=25.0
        )

        d = fvg.to_dict()

        assert d['type'] == "FVG_BEAR"
        assert d['fill_percentage'] == 25.0
        assert 'size_pips' in d


class TestPOIResult:
    """Tests for POIResult dataclass"""

    def test_creation(self):
        """Test creating POIResult"""
        result = POIResult(
            order_blocks=[],
            fvgs=[],
            swing_highs=[],
            swing_lows=[],
            bos_choch=[]
        )

        assert result.order_blocks == []
        assert result.fvgs == []

    def test_with_order_blocks(self):
        """Test POIResult with order blocks"""
        ob = OrderBlock(
            id="ob_1",
            poi_type=POIType.ORDER_BLOCK_BULLISH,
            top=1.3010,
            bottom=1.3000,
            volume=1000,
            strength=75.0
        )

        result = POIResult(
            order_blocks=[ob],
            fvgs=[]
        )

        assert len(result.order_blocks) == 1
        assert result.order_blocks[0] == ob


class TestPOIDetector:
    """Tests for POI Detector"""

    @pytest.fixture
    def sample_df(self):
        """Generate sample OHLCV DataFrame"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')

        opens = 1.3000 + np.cumsum(np.random.normal(0, 0.0005, n))
        closes = opens + np.random.normal(0, 0.001, n)
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.0005, n))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.0005, n))

        return pd.DataFrame({
            'time': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(100, 1000, n)
        })

    def test_initialization(self):
        """Test detector initializes correctly"""
        detector = POIDetector(
            swing_length=10,
            ob_min_strength=0.6,
            max_poi_age_bars=100
        )

        assert detector is not None
        assert detector.swing_length == 10
        assert detector.ob_min_strength == 0.6

    def test_detect_returns_poi_result(self, sample_df):
        """Test detect returns POIResult"""
        detector = POIDetector()

        result = detector.detect(sample_df)

        assert result is not None
        assert isinstance(result, POIResult)

    def test_detect_order_blocks(self, sample_df):
        """Test order block detection"""
        detector = POIDetector(ob_min_strength=50.0)

        result = detector.detect(sample_df)

        # Should have order_blocks list (may be empty depending on data)
        assert hasattr(result, 'order_blocks')
        assert isinstance(result.order_blocks, list)

    def test_detect_fvgs(self, sample_df):
        """Test FVG detection"""
        detector = POIDetector()

        result = detector.detect(sample_df)

        assert hasattr(result, 'fvgs')
        assert isinstance(result.fvgs, list)

    def test_reset(self, sample_df):
        """Test detector reset"""
        detector = POIDetector()

        detector.detect(sample_df)

        detector.reset()

        assert detector.last_result is None


class TestPOIDetectorEdgeCases:
    """Edge case tests"""

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        detector = POIDetector()

        df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

        result = detector.detect(df)

        # Should return empty result or None
        assert result is None or (len(result.order_blocks) == 0 and len(result.fvgs) == 0)

    def test_small_dataframe(self):
        """Test with very small DataFrame"""
        detector = POIDetector(swing_length=10)

        df = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=5, freq='1H'),
            'open': [1.3000, 1.3005, 1.3010, 1.3008, 1.3012],
            'high': [1.3010, 1.3015, 1.3020, 1.3018, 1.3022],
            'low': [1.2995, 1.3000, 1.3005, 1.3003, 1.3007],
            'close': [1.3005, 1.3010, 1.3008, 1.3012, 1.3015],
            'volume': [100, 150, 120, 130, 140]
        })

        # Should handle gracefully without crashing
        result = detector.detect(df)

    def test_constant_prices(self):
        """Test with constant prices"""
        detector = POIDetector()

        n = 100
        df = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=n, freq='1H'),
            'open': [1.3000] * n,
            'high': [1.3000] * n,
            'low': [1.3000] * n,
            'close': [1.3000] * n,
            'volume': [100] * n
        })

        # Should handle without crashing
        result = detector.detect(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
