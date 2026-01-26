"""Trade Executor Unit Tests
=============================

Tests for TradeExecutor integration.

Author: SURIOTA Team
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.executor import TradeExecutor, ExecutorState, TradeResult
from src.trading.risk_manager import RiskManager
from src.trading.exit_manager import ExitManager
from src.analysis.kalman_filter import KalmanState
from src.analysis.regime_detector import MarketRegime, RegimeInfo


class TestExecutorState:
    """Tests for ExecutorState enum"""

    def test_state_values(self):
        """Test executor state values"""
        assert ExecutorState.IDLE.value == "idle"
        assert ExecutorState.WARMING_UP.value == "warming_up"
        assert ExecutorState.MONITORING.value == "monitoring"
        assert ExecutorState.TRADING.value == "trading"
        assert ExecutorState.PAUSED.value == "paused"
        assert ExecutorState.STOPPED.value == "stopped"
        assert ExecutorState.ERROR.value == "error"


class TestTradeResult:
    """Tests for TradeResult dataclass"""

    def test_successful_trade(self):
        """Test successful trade result"""
        result = TradeResult(
            success=True,
            direction="BUY",
            entry_price=1.30000,
            stop_loss=1.29900,
            volume=0.10,
            ticket=12345,
            message="Trade executed"
        )

        assert result.success is True
        assert result.direction == "BUY"
        assert result.ticket == 12345
        assert result.stop_loss == 1.29900
        assert result.volume == 0.10

    def test_failed_trade(self):
        """Test failed trade result"""
        result = TradeResult(
            success=False,
            message="Insufficient margin"
        )

        assert result.success is False
        assert result.ticket == 0


class TestTradeExecutor:
    """Tests for TradeExecutor"""

    @pytest.fixture
    def executor(self):
        """Create executor instance"""
        return TradeExecutor(
            symbol="GBPUSD",
            timeframe_htf="H1",
            timeframe_ltf="M5",
            warmup_bars=50,
            magic_number=123456
        )

    @pytest.fixture
    def sample_htf_data(self):
        """Generate sample HTF data"""
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

    @pytest.fixture
    def sample_ltf_data(self):
        """Generate sample LTF data"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')

        opens = 1.3000 + np.cumsum(np.random.normal(0, 0.0002, n))
        closes = opens + np.random.normal(0, 0.0003, n)
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.0002, n))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.0002, n))

        return pd.DataFrame({
            'time': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(50, 500, n)
        })

    def test_initialization(self, executor):
        """Test executor initializes correctly"""
        assert executor is not None
        assert executor.symbol == "GBPUSD"
        assert executor.state == ExecutorState.IDLE

    def test_components_initialized(self, executor):
        """Test all components are initialized"""
        assert executor.kalman is not None
        assert executor.regime_detector is not None
        assert executor.poi_detector is not None
        assert executor.entry_trigger is not None
        assert executor.risk_manager is not None
        assert executor.exit_manager is not None

    @pytest.mark.asyncio
    async def test_warmup(self, executor, sample_htf_data, sample_ltf_data):
        """Test warmup with historical data"""
        result = await executor.warmup(sample_htf_data, sample_ltf_data)

        assert result is True
        assert executor.state == ExecutorState.MONITORING

    def test_pause(self, executor):
        """Test pausing executor"""
        executor.state = ExecutorState.MONITORING

        executor.pause()

        assert executor.state == ExecutorState.PAUSED

    def test_resume(self, executor):
        """Test resuming executor"""
        executor.state = ExecutorState.PAUSED
        executor._warmup_count = 100  # Simulate warmup complete

        executor.resume()

        assert executor.state == ExecutorState.MONITORING

    def test_get_status(self, executor):
        """Test getting executor status"""
        status = executor.get_status()

        assert 'state' in status
        assert 'warmup_complete' in status
        assert 'regime' in status
        assert 'open_positions' in status
        assert 'daily_pnl' in status
        assert 'stats' in status

    @pytest.mark.asyncio
    async def test_process_tick_when_paused(self, executor):
        """Test process_tick returns None when paused"""
        executor.state = ExecutorState.PAUSED

        result = await executor.process_tick(1.30000, 10000.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_process_tick_before_warmup(self, executor):
        """Test process_tick returns None before warmup"""
        executor.state = ExecutorState.IDLE

        result = await executor.process_tick(1.30000, 10000.0)

        assert result is None


class TestRiskManager:
    """Tests for RiskManager"""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance"""
        return RiskManager()

    def test_initialization(self, risk_manager):
        """Test risk manager initializes correctly"""
        assert risk_manager is not None
        assert risk_manager.high_quality_threshold == 80
        assert risk_manager.high_quality_risk == 0.015

    def test_calculate_lot_size_high_quality(self, risk_manager):
        """Test position sizing for high quality zone"""
        params = risk_manager.calculate_lot_size(
            account_balance=10000.0,
            quality_score=85,  # High quality
            sl_pips=20.0
        )

        assert params.lot_size > 0
        assert params.risk_percent == risk_manager.high_quality_risk

    def test_calculate_lot_size_low_quality(self, risk_manager):
        """Test position sizing for low quality zone"""
        params = risk_manager.calculate_lot_size(
            account_balance=10000.0,
            quality_score=50,  # Low quality
            sl_pips=20.0
        )

        assert params.lot_size > 0
        assert params.risk_percent == risk_manager.low_quality_risk

    def test_quality_affects_position_size(self, risk_manager):
        """Test higher quality gives larger position"""
        # Use very large sl_pips to avoid hitting max_lot_size cap
        # High: 10000 * 0.015 / (200 * 10) = 0.075
        # Low: 10000 * 0.005 / (200 * 10) = 0.025
        high_params = risk_manager.calculate_lot_size(
            account_balance=10000.0,
            quality_score=90,
            sl_pips=200.0
        )

        low_params = risk_manager.calculate_lot_size(
            account_balance=10000.0,
            quality_score=50,
            sl_pips=200.0
        )

        # High quality (1.5% risk) should give larger lot than low quality (0.5% risk)
        assert high_params.lot_size > low_params.lot_size
        assert high_params.risk_percent == 0.015
        assert low_params.risk_percent == 0.005

    def test_can_open_trade_within_limits(self, risk_manager):
        """Test can_open_trade when within limits"""
        can_trade, reason = risk_manager.can_open_trade()

        assert can_trade is True
        assert reason == "OK"

    def test_can_open_trade_max_positions(self, risk_manager):
        """Test can_open_trade at max positions"""
        risk_manager._open_positions = risk_manager.max_open_positions

        can_trade, reason = risk_manager.can_open_trade()

        assert can_trade is False
        assert "Max positions" in reason

    def test_can_open_trade_profit_target(self, risk_manager):
        """Test can_open_trade at profit target"""
        risk_manager._daily_pnl = risk_manager.daily_profit_target + 10

        can_trade, reason = risk_manager.can_open_trade()

        assert can_trade is False
        assert "profit target" in reason.lower()

    def test_can_open_trade_loss_limit(self, risk_manager):
        """Test can_open_trade at loss limit"""
        risk_manager._daily_pnl = -(risk_manager.daily_loss_limit + 10)

        can_trade, reason = risk_manager.can_open_trade()

        assert can_trade is False
        assert "loss limit" in reason.lower()


class TestExitManager:
    """Tests for ExitManager"""

    @pytest.fixture
    def exit_manager(self):
        """Create exit manager instance"""
        return ExitManager()

    def test_initialization(self, exit_manager):
        """Test exit manager initializes correctly"""
        assert exit_manager is not None
        assert exit_manager.tp1_rr == 1.0
        assert exit_manager.tp1_percent == 0.5
        assert exit_manager.trailing_enabled is True

    def test_create_position_buy(self, exit_manager):
        """Test creating BUY position with TP levels"""
        pos = exit_manager.create_position(
            ticket=12345,
            symbol="GBPUSD",
            direction="BUY",
            entry_price=1.30000,
            stop_loss=1.29900,
            volume=0.10
        )

        assert pos is not None
        assert pos.tp1.price > pos.entry_price
        assert pos.tp2.price > pos.tp1.price
        assert pos.tp3.price > pos.tp2.price

    def test_create_position_sell(self, exit_manager):
        """Test creating SELL position with TP levels"""
        pos = exit_manager.create_position(
            ticket=12346,
            symbol="GBPUSD",
            direction="SELL",
            entry_price=1.30000,
            stop_loss=1.30100,
            volume=0.10
        )

        assert pos is not None
        assert pos.tp1.price < pos.entry_price
        assert pos.tp2.price < pos.tp1.price
        assert pos.tp3.price < pos.tp2.price

    def test_update_position_tp1_hit(self, exit_manager):
        """Test TP1 hit detection"""
        pos = exit_manager.create_position(
            ticket=12345,
            symbol="GBPUSD",
            direction="BUY",
            entry_price=1.30000,
            stop_loss=1.29900,
            volume=0.10
        )

        # Price reaches TP1
        action, details = exit_manager.update_position(12345, pos.tp1.price + 0.0001)

        assert action == 'CLOSE_PARTIAL_TP1'
        assert details['move_sl_to_be'] is True

    def test_set_breakeven(self, exit_manager):
        """Test breakeven setting"""
        exit_manager.create_position(
            ticket=12345,
            symbol="GBPUSD",
            direction="BUY",
            entry_price=1.30000,
            stop_loss=1.29900,
            volume=0.10
        )

        exit_manager.set_breakeven(12345)

        pos = exit_manager.get_position(12345)
        assert pos.breakeven_set is True
        assert pos.current_sl >= pos.entry_price


class TestExecutorIntegration:
    """Integration tests for executor"""

    @pytest.fixture
    def executor_with_callbacks(self):
        """Create executor with mock callbacks"""
        executor = TradeExecutor(
            symbol="GBPUSD",
            timeframe_htf="H1",
            timeframe_ltf="M5"
        )

        # Setup mock callbacks
        executor.set_callbacks(
            get_account_info=AsyncMock(return_value={
                'balance': 10000.0,
                'equity': 10000.0,
                'profit': 0.0
            }),
            get_tick=AsyncMock(return_value={
                'bid': 1.30000,
                'ask': 1.30002
            }),
            get_ohlcv_htf=AsyncMock(),
            get_ohlcv_ltf=AsyncMock(),
            place_market_order=AsyncMock(return_value={'ticket': 12345}),
            modify_position=AsyncMock(return_value=True),
            close_position=AsyncMock(return_value=True),
            close_partial=AsyncMock(return_value=True),
            get_positions=AsyncMock(return_value=[]),
            send_telegram=AsyncMock()
        )

        return executor

    @pytest.mark.asyncio
    async def test_full_analysis_cycle(self, executor_with_callbacks):
        """Test full analysis cycle"""
        executor = executor_with_callbacks

        # Manually set state for testing
        executor.state = ExecutorState.MONITORING

        # This would normally be called with real data
        # Just verify the structure works
        status = executor.get_status()
        assert status['state'] == 'monitoring'


# Fixtures for common test data
@pytest.fixture
def mock_regime_info():
    """Create mock regime info"""
    return RegimeInfo(
        regime=MarketRegime.BULLISH,
        probability=0.80,
        state_probs=[0.80, 0.15, 0.05],
        volatility=0.002,
        bias="long",
        timestamp=datetime.now()
    )


@pytest.fixture
def mock_kalman_state():
    """Create mock Kalman state"""
    return KalmanState(
        filtered_price=1.30000,
        velocity=0.0001,
        acceleration=0.00001,
        uncertainty=0.0005,
        timestamp=datetime.now()
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
