"""SURGE-WSI Backtester
======================

Historical validation for SURGE-WSI 6-Layer Trading System.

Features:
- Walk-forward backtesting
- Partial TP simulation
- Detailed statistics
- Trade-by-trade analysis
- Equity curve plotting

Author: SURIOTA Team
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector, MarketRegime
from src.analysis.poi_detector import POIDetector
from src.analysis.market_filter import MarketFilter, RelaxedEntryFilter
from src.trading.entry_trigger import EntryTrigger
from src.trading.risk_manager import RiskManager
from src.trading.exit_manager import ExitManager
from src.trading.trade_mode_manager import TradeModeManager, TradeMode, TradeModeConfig
from src.utils.killzone import KillZone


class TradeStatus(Enum):
    """Trade status"""
    OPEN = "open"
    TP1 = "tp1"
    TP2 = "tp2"
    TP3 = "tp3"
    SL = "sl"
    BE = "breakeven"
    CLOSED = "closed"


@dataclass
class BacktestTrade:
    """Single backtest trade"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    initial_volume: float = 0.0
    pnl: float = 0.0
    pips: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    regime: str = ""
    quality_score: float = 0.0

    # Partial close tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    remaining_volume: float = 0.0
    partial_pnl: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'sl_price': self.sl_price,
            'tp1_price': self.tp1_price,
            'tp2_price': self.tp2_price,
            'tp3_price': self.tp3_price,
            'initial_volume': self.initial_volume,
            'pnl': self.pnl,
            'pips': self.pips,
            'status': self.status.value,
            'regime': self.regime,
            'quality_score': self.quality_score,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'tp3_hit': self.tp3_hit
        }


@dataclass
class BacktestResult:
    """Backtest results summary"""
    # Basic info
    symbol: str = ""
    start_date: datetime = None
    end_date: datetime = None
    initial_balance: float = 10000.0

    # Performance
    final_balance: float = 0.0
    net_profit: float = 0.0
    net_profit_percent: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Time
    avg_trade_duration: timedelta = None
    trades_per_month: float = 0.0

    # Partial TP stats
    tp1_hit_rate: float = 0.0
    tp2_hit_rate: float = 0.0
    tp3_hit_rate: float = 0.0

    # By regime
    bullish_trades: int = 0
    bearish_trades: int = 0
    bullish_win_rate: float = 0.0
    bearish_win_rate: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    trade_list: List[BacktestTrade] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'net_profit': self.net_profit,
            'net_profit_percent': self.net_profit_percent,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'max_drawdown_percent': self.max_drawdown_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'tp1_hit_rate': self.tp1_hit_rate,
            'tp2_hit_rate': self.tp2_hit_rate,
            'tp3_hit_rate': self.tp3_hit_rate,
            'trades_per_month': self.trades_per_month
        }


class Backtester:
    """SURGE-WSI Backtester"""

    def __init__(
        self,
        symbol: str = "GBPUSD",
        start_date: str = "2024-01-01",
        end_date: str = None,
        initial_balance: float = 10000.0,
        pip_value: float = 10.0,  # Per standard lot
        spread_pips: float = 1.5,
        use_killzone: bool = True,
        use_trend_filter: bool = True,  # Enable trend alignment filter
        use_relaxed_filter: bool = True  # NEW: Relax filters in low-activity periods
    ):
        """Initialize backtester

        Args:
            symbol: Trading symbol
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (default: now)
            initial_balance: Starting balance
            pip_value: Value per pip per standard lot
            spread_pips: Spread in pips
            use_killzone: Filter by kill zones
        """
        self.symbol = symbol
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_date else datetime.now(timezone.utc)
        self.initial_balance = initial_balance
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.use_killzone = use_killzone
        self.use_trend_filter = use_trend_filter
        self.use_relaxed_filter = use_relaxed_filter

        # Regime stability settings
        self.min_regime_stability_bars = 5  # Regime must be stable for N bars
        self._regime_history: List[str] = []  # Track recent regimes
        self._debug_regime_unstable = 0  # Counter for regime stability filter

        # Components
        self.kalman = MultiScaleKalman()
        self.regime_detector = HMMRegimeDetector()
        self.poi_detector = POIDetector()
        self.entry_trigger = EntryTrigger()
        self.risk_manager = RiskManager(
            pip_value=pip_value
            # Lot size calculated automatically based on risk %
        )
        self.exit_manager = ExitManager()
        self.killzone = KillZone()
        self.market_filter = MarketFilter()  # Trend alignment filter
        self.relaxed_filter = RelaxedEntryFilter()  # NEW: Relax filters in low-activity
        self.trade_mode_manager = TradeModeManager(TradeModeConfig())  # Trade mode manager

        # State
        self.balance = initial_balance
        self._daily_start_balance = initial_balance
        self._last_day: datetime = None
        self.equity_curve = [initial_balance]
        self.trades: List[BacktestTrade] = []
        self.open_trade: Optional[BacktestTrade] = None

        # Data
        self.htf_data: Optional[pd.DataFrame] = None
        self.ltf_data: Optional[pd.DataFrame] = None

        # Debug counters
        self._debug_checks = 0
        self._debug_regime_fail = 0
        self._debug_poi_none = 0
        self._debug_regime_sideways = 0
        self._debug_no_pois = 0
        self._debug_not_in_poi = 0
        self._debug_in_poi = 0
        self._debug_entry_fail = 0
        self._debug_entry_ok = 0
        self._debug_trend_filtered = 0  # Trades filtered by trend alignment
        self._debug_relaxed_entries = 0  # NEW: Entries with relaxed filters
        self._debug_signal_only = 0  # NEW: Trades in signal-only mode

        # Recent trade tracking for relaxed filter
        self._recent_trades: List[datetime] = []  # Track trade times

        # Signal-only trade tracking
        self.signal_only_trades: List[Dict] = []  # Trades that would be signal-only

    def load_data(
        self,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame
    ):
        """Load historical data

        Args:
            htf_data: Higher timeframe OHLCV
            ltf_data: Lower timeframe OHLCV
        """
        self.htf_data = htf_data.copy()
        self.ltf_data = ltf_data.copy()

        # Normalize column names to lowercase
        self.htf_data.columns = [c.lower() for c in self.htf_data.columns]
        self.ltf_data.columns = [c.lower() for c in self.ltf_data.columns]

        # Ensure time column is datetime
        if 'time' in self.htf_data.columns:
            self.htf_data['time'] = pd.to_datetime(self.htf_data['time'])
        if 'time' in self.ltf_data.columns:
            self.ltf_data['time'] = pd.to_datetime(self.ltf_data['time'])

        logger.info(f"Loaded HTF data: {len(self.htf_data)} bars")
        logger.info(f"Loaded LTF data: {len(self.ltf_data)} bars")

    def run(self) -> BacktestResult:
        """Run backtest

        Returns:
            BacktestResult with statistics
        """
        if self.htf_data is None or self.ltf_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info(f"Starting backtest: {self.symbol}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")

        # Filter data by date range
        htf = self.htf_data[
            (self.htf_data['time'] >= self.start_date) &
            (self.htf_data['time'] <= self.end_date)
        ].copy()

        ltf = self.ltf_data[
            (self.ltf_data['time'] >= self.start_date) &
            (self.ltf_data['time'] <= self.end_date)
        ].copy()

        # Warmup period
        warmup_htf = self.htf_data[self.htf_data['time'] < self.start_date].tail(100)
        warmup_ltf = self.ltf_data[self.ltf_data['time'] < self.start_date].tail(500)

        # Process warmup
        self._warmup(warmup_htf, warmup_ltf)

        # Initialize trade mode manager
        self.trade_mode_manager.reset_daily_stats(self.balance)
        self.trade_mode_manager.reset_weekly_stats(self.balance)

        # Main backtest loop
        htf_idx = 0
        ltf_pos = 0  # Positional counter for ltf slicing
        for _, ltf_row in ltf.iterrows():
            current_time = ltf_row['time']
            current_price = ltf_row['close']
            current_high = ltf_row['high']
            current_low = ltf_row['low']

            # Daily stats reset
            if self._last_day is None or current_time.date() != self._last_day:
                if self._last_day is not None:
                    # New day - reset daily stats
                    self.trade_mode_manager.reset_daily_stats(self.balance)
                    # Check for Monday (new week)
                    if current_time.weekday() == 0:
                        self.trade_mode_manager.reset_weekly_stats(self.balance)
                self._last_day = current_time.date()
                self._daily_start_balance = self.balance

            # Update balance in mode manager
            self.trade_mode_manager.update_balance(self.balance)

            # Update HTF analysis when new HTF bar forms
            if htf_idx < len(htf):
                htf_row = htf.iloc[htf_idx]
                if current_time >= htf_row['time']:
                    self._process_htf_bar(htf.iloc[:htf_idx+1])
                    htf_idx += 1

            # Check kill zone
            if self.use_killzone:
                in_kz, _ = self.killzone.is_in_killzone(current_time)
                if not in_kz:
                    ltf_pos += 1
                    continue

            # Manage open position
            if self.open_trade is not None:
                self._manage_position(current_time, current_high, current_low, current_price)

            # Check for new entry (if no open position)
            if self.open_trade is None:
                # Use positional index for slicing
                ltf_window = ltf.iloc[max(0, ltf_pos-50):ltf_pos+1]
                # Pass HTF data for trend filter (use current htf_idx)
                htf_window = htf.iloc[max(0, htf_idx-20):htf_idx+1] if htf_idx > 0 else None
                self._check_entry(current_time, current_price, ltf_window, htf_window)

            ltf_pos += 1

            # Update equity curve
            if self.open_trade is not None:
                unrealized = self._calculate_unrealized_pnl(current_price)
                self.equity_curve.append(self.balance + unrealized)
            else:
                self.equity_curve.append(self.balance)

        # Close any open trade at end
        if self.open_trade is not None:
            self._close_trade(ltf.iloc[-1]['time'], ltf.iloc[-1]['close'], TradeStatus.CLOSED)

        # Calculate results
        return self._calculate_results()

    def _warmup(self, htf_data: pd.DataFrame, ltf_data: pd.DataFrame):
        """Warmup components with historical data"""
        # Warmup Kalman
        for _, row in ltf_data.iterrows():
            self.kalman.update(row['close'])

        # Warmup regime detector - update() takes single price at a time
        if len(htf_data) > 0:
            for _, row in htf_data.iterrows():
                self.regime_detector.update(row['close'])

        # Warmup POI detector
        if len(htf_data) >= 50:
            self.poi_detector.detect(htf_data)

        logger.info("Warmup complete")

    def _process_htf_bar(self, htf_data: pd.DataFrame):
        """Process new HTF bar"""
        # Update regime with latest close price
        if len(htf_data) > 0:
            latest_close = htf_data['close'].iloc[-1]
            self.regime_detector.update(latest_close)

            # Track regime history for stability filter
            regime_info = self.regime_detector.last_info
            if regime_info is not None:
                self._regime_history.append(regime_info.regime.value)
                # Keep only last 10 regimes
                if len(self._regime_history) > 10:
                    self._regime_history.pop(0)

        # Update POIs
        self.poi_detector.detect(htf_data)

    def _check_entry(
        self,
        current_time: datetime,
        current_price: float,
        ltf_window: pd.DataFrame,
        htf_window: Optional[pd.DataFrame] = None
    ):
        """Check for entry signal"""
        # Debug counters
        self._debug_checks += 1

        # Get regime
        regime_info = self.regime_detector.last_info
        if regime_info is None or not regime_info.is_tradeable:
            self._debug_regime_fail += 1
            return

        # Get POI
        poi_result = self.poi_detector.last_result
        if poi_result is None:
            self._debug_poi_none += 1
            return

        # Determine direction based on regime
        if regime_info.regime == MarketRegime.BULLISH:
            direction = "BUY"
            pois = poi_result.bullish_pois
        elif regime_info.regime == MarketRegime.BEARISH:
            direction = "SELL"
            pois = poi_result.bearish_pois
        else:
            self._debug_regime_sideways += 1
            return

        # December trading restriction - reduce exposure during year-end volatility
        # December historically has poor performance due to low liquidity and holiday trading
        # Dec 1-14: SIGNAL_ONLY (tracked separately)
        # Dec 15-31: MONITORING (full skip - don't even track)
        if current_time.month == 12 and current_time.day >= 15:
            # Skip trading in late December (holiday period) - MONITORING mode
            self._debug_regime_unstable += 1
            logger.debug("Late December (Dec 15+) - MONITORING mode, skipping trade")
            return

        if not pois:
            self._debug_no_pois += 1
            return

        # Check if price is in POI (using POIResult.price_at_poi method)
        in_poi, poi = poi_result.price_at_poi(current_price, direction)
        if not in_poi:
            self._debug_not_in_poi += 1
            return

        self._debug_in_poi += 1

        # Check trend alignment before entry
        if self.use_trend_filter and htf_window is not None and len(htf_window) >= 10:
            aligned, reason = self.market_filter.check_trend_alignment(htf_window, direction)
            if not aligned:
                self._debug_trend_filtered += 1
                logger.debug(f"Trend filter: {reason}")
                return

        # NEW: Determine if we should use relaxed entry filters
        require_full_confirmation = True
        min_quality = 60.0

        if self.use_relaxed_filter:
            # Count recent trades in last 7 days
            cutoff_time = current_time - timedelta(days=7)
            recent_trade_count = sum(1 for t in self._recent_trades if t > cutoff_time)

            # Get relaxed parameters if no recent trades
            min_quality, require_full_confirmation = self.relaxed_filter.get_entry_params(
                recent_trade_count=recent_trade_count,
                lookback_days=7
            )

            # Temporarily adjust entry trigger's min_quality_score
            original_min_quality = self.entry_trigger.min_quality_score
            self.entry_trigger.min_quality_score = min_quality

        # Check LTF confirmation (poi is a Dict from to_dict())
        # Build poi_info dict for entry trigger
        poi_info = {
            'high': poi.get('top', poi.get('high', poi.get('mid', 0) + 0.001)),
            'low': poi.get('bottom', poi.get('low', poi.get('mid', 0) - 0.001)),
            'mid': poi.get('mid', current_price),
            'strength': poi.get('strength', 70.0)
        }

        has_entry, signal = self.entry_trigger.check_for_entry(
            ltf_df=ltf_window,
            direction=direction,
            poi_info=poi_info,
            current_price=current_price,
            require_full_confirmation=require_full_confirmation
        )

        # Restore original min_quality if relaxed filter was used
        if self.use_relaxed_filter:
            self.entry_trigger.min_quality_score = original_min_quality

        if not has_entry or signal is None:
            self._debug_entry_fail += 1
            return

        self._debug_entry_ok += 1

        # Track relaxed entries
        if self.use_relaxed_filter and not require_full_confirmation:
            self._debug_relaxed_entries += 1
            logger.debug(f"Relaxed entry: quality={min_quality}, no full confirmation required")

        # Track this trade time for relaxed filter
        self._recent_trades.append(current_time)

        # Calculate position size (poi is a Dict)
        quality_score = poi.get('strength', 70.0)
        sl_pips = signal.sl_pips

        risk_params = self.risk_manager.calculate_lot_size(
            account_balance=self.balance,
            quality_score=quality_score,
            sl_pips=sl_pips
        )
        lot_size = risk_params.lot_size

        # Calculate TP levels using R:R ratios
        sl_distance = abs(current_price - signal.stop_loss)
        if direction == "BUY":
            tp1 = current_price + sl_distance * self.exit_manager.tp1_rr
            tp2 = current_price + sl_distance * self.exit_manager.tp2_rr
            tp3 = current_price + sl_distance * self.exit_manager.tp3_rr
        else:
            tp1 = current_price - sl_distance * self.exit_manager.tp1_rr
            tp2 = current_price - sl_distance * self.exit_manager.tp2_rr
            tp3 = current_price - sl_distance * self.exit_manager.tp3_rr

        # Evaluate trade mode - check if this would be signal-only
        atr_pips = sl_pips * 1.5  # Approximate ATR from SL
        trade_mode = self.trade_mode_manager.evaluate_mode(
            current_time=current_time,
            current_balance=self.balance,
            atr_pips=atr_pips
        )

        # Track regime change
        self.trade_mode_manager.record_regime_change(regime_info.regime.value)

        # Track if this trade would be signal-only or recovery (for comparison)
        is_signal_only = trade_mode == TradeMode.SIGNAL_ONLY
        is_recovery = trade_mode == TradeMode.RECOVERY

        # Apply lot multiplier for recovery mode
        original_lot = lot_size
        if is_recovery:
            lot_multiplier = self.trade_mode_manager.get_lot_multiplier()
            lot_size = round(lot_size * lot_multiplier, 2)
            lot_size = max(0.01, lot_size)  # Minimum broker lot
            logger.debug(f"Recovery mode: lot {original_lot} -> {lot_size} (x{lot_multiplier})")

        # Get session quality score and adjust lot accordingly
        session_quality = self.killzone.get_session_quality_score(current_time)
        quality_lot_adj = session_quality.get_lot_adjustment()
        if quality_lot_adj != 1.0:
            lot_size = round(lot_size * quality_lot_adj, 2)
            lot_size = max(0.01, lot_size)
            logger.debug(f"Session quality {session_quality.total_score:.0f}: lot adjusted x{quality_lot_adj}")

        if is_signal_only:
            self._debug_signal_only += 1
            # Store signal-only trade for analysis
            self.signal_only_trades.append({
                'time': current_time,
                'direction': direction,
                'entry': current_price,
                'sl': signal.stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'lot': original_lot,
                'adjusted_lot': lot_size,
                'reason': self.trade_mode_manager.mode_reason,
                'session_quality': session_quality.total_score
            })
            logger.debug(f"Signal-only mode: {self.trade_mode_manager.mode_reason}")
            # In signal-only mode, we don't execute the trade in real life
            # But for backtesting comparison, we still execute to see the outcome
            # This will help compare AUTO vs SIGNAL_ONLY performance

        # Open trade
        self.open_trade = BacktestTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=current_price + (self.spread_pips * 0.0001 if direction == "BUY" else 0),
            sl_price=signal.stop_loss,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            initial_volume=lot_size,
            remaining_volume=lot_size,
            regime=regime_info.regime.value,
            quality_score=quality_score
        )

        logger.debug(f"Entry: {direction} @ {current_price:.5f}, SL: {signal.stop_loss:.5f}")

    def _manage_position(
        self,
        current_time: datetime,
        high: float,
        low: float,
        close: float
    ):
        """Manage open position"""
        trade = self.open_trade

        if trade.direction == "BUY":
            # Check SL
            if low <= trade.sl_price:
                self._close_trade(current_time, trade.sl_price, TradeStatus.SL)
                return

            # Check TP levels
            if not trade.tp1_hit and high >= trade.tp1_price:
                self._partial_close(current_time, trade.tp1_price, 0.5, "TP1")
                trade.tp1_hit = True
                trade.sl_price = trade.entry_price  # Move to BE

            if trade.tp1_hit and not trade.tp2_hit and high >= trade.tp2_price:
                self._partial_close(current_time, trade.tp2_price, 0.6, "TP2")  # 60% of remaining
                trade.tp2_hit = True

            if trade.tp2_hit and not trade.tp3_hit and high >= trade.tp3_price:
                self._close_trade(current_time, trade.tp3_price, TradeStatus.TP3)
                return

        else:  # SELL
            # Check SL
            if high >= trade.sl_price:
                self._close_trade(current_time, trade.sl_price, TradeStatus.SL)
                return

            # Check TP levels
            if not trade.tp1_hit and low <= trade.tp1_price:
                self._partial_close(current_time, trade.tp1_price, 0.5, "TP1")
                trade.tp1_hit = True
                trade.sl_price = trade.entry_price

            if trade.tp1_hit and not trade.tp2_hit and low <= trade.tp2_price:
                self._partial_close(current_time, trade.tp2_price, 0.6, "TP2")
                trade.tp2_hit = True

            if trade.tp2_hit and not trade.tp3_hit and low <= trade.tp3_price:
                self._close_trade(current_time, trade.tp3_price, TradeStatus.TP3)
                return

    def _partial_close(
        self,
        time: datetime,
        price: float,
        percent: float,
        reason: str
    ):
        """Close partial position"""
        trade = self.open_trade
        close_volume = trade.remaining_volume * percent

        # Calculate PnL for closed portion
        if trade.direction == "BUY":
            pips = (price - trade.entry_price) / 0.0001
        else:
            pips = (trade.entry_price - price) / 0.0001

        partial_pnl = pips * self.pip_value * close_volume  # pip_value per lot * lot_size
        trade.partial_pnl += partial_pnl
        trade.remaining_volume -= close_volume
        self.balance += partial_pnl

        logger.debug(f"{reason} hit: +{partial_pnl:.2f} ({pips:.1f} pips)")

    def _close_trade(
        self,
        time: datetime,
        price: float,
        status: TradeStatus
    ):
        """Close trade completely"""
        trade = self.open_trade

        # Calculate final PnL for remaining volume
        if trade.direction == "BUY":
            pips = (price - trade.entry_price) / 0.0001
        else:
            pips = (trade.entry_price - price) / 0.0001

        final_pnl = pips * self.pip_value * trade.remaining_volume  # pip_value per lot * lot_size
        trade.pnl = trade.partial_pnl + final_pnl
        trade.pips = pips
        trade.exit_time = time
        trade.exit_price = price
        trade.status = status

        self.balance += final_pnl
        self.trades.append(trade)
        self.open_trade = None

        # Record trade result for mode manager (tracks consecutive losses)
        is_win = trade.pnl > 0
        self.trade_mode_manager.record_trade_result(is_win, trade.pnl)

        logger.debug(f"Closed: {status.value}, Total P/L: {trade.pnl:.2f}")

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P/L"""
        if self.open_trade is None:
            return 0.0

        trade = self.open_trade
        if trade.direction == "BUY":
            pips = (current_price - trade.entry_price) / 0.0001
        else:
            pips = (trade.entry_price - current_price) / 0.0001

        return trade.partial_pnl + (pips * self.pip_value * trade.remaining_volume)

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest statistics"""
        result = BacktestResult(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            equity_curve=self.equity_curve,
            trade_list=self.trades
        )

        # Debug output
        if self._debug_checks > 0:
            logger.debug(f"DEBUG STATS: checks={self._debug_checks}, regime_fail={self._debug_regime_fail}, "
                        f"poi_none={self._debug_poi_none}, sideways={self._debug_regime_sideways}, "
                        f"regime_unstable={self._debug_regime_unstable}, no_pois={self._debug_no_pois}, "
                        f"not_in_poi={self._debug_not_in_poi}, in_poi={self._debug_in_poi}, "
                        f"entry_fail={self._debug_entry_fail}, entry_ok={self._debug_entry_ok}, "
                        f"trend_filtered={self._debug_trend_filtered}, relaxed_entries={self._debug_relaxed_entries}, "
                        f"signal_only={self._debug_signal_only}")

        if not self.trades:
            return result

        # Basic metrics
        result.net_profit = self.balance - self.initial_balance
        result.net_profit_percent = result.net_profit / self.initial_balance * 100
        result.total_trades = len(self.trades)

        # Win/Loss
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        result.gross_profit = sum(t.pnl for t in wins)
        result.gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = result.gross_profit / result.gross_loss if result.gross_loss > 0 else 0

        result.avg_win = result.gross_profit / len(wins) if wins else 0
        result.avg_loss = result.gross_loss / len(losses) if losses else 0

        # Drawdown
        peak = self.initial_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        result.max_drawdown = max_dd
        result.max_drawdown_percent = max_dd / self.initial_balance * 100

        # Time metrics
        durations = [(t.exit_time - t.entry_time) for t in self.trades if t.exit_time]
        if durations:
            result.avg_trade_duration = sum(durations, timedelta()) / len(durations)

        months = (self.end_date - self.start_date).days / 30.44
        result.trades_per_month = len(self.trades) / months if months > 0 else 0

        # TP hit rates
        result.tp1_hit_rate = sum(1 for t in self.trades if t.tp1_hit) / len(self.trades) * 100
        result.tp2_hit_rate = sum(1 for t in self.trades if t.tp2_hit) / len(self.trades) * 100
        result.tp3_hit_rate = sum(1 for t in self.trades if t.tp3_hit) / len(self.trades) * 100

        # By regime
        bullish_trades = [t for t in self.trades if t.regime == "bullish"]
        bearish_trades = [t for t in self.trades if t.regime == "bearish"]

        result.bullish_trades = len(bullish_trades)
        result.bearish_trades = len(bearish_trades)

        if bullish_trades:
            result.bullish_win_rate = sum(1 for t in bullish_trades if t.pnl > 0) / len(bullish_trades) * 100
        if bearish_trades:
            result.bearish_win_rate = sum(1 for t in bearish_trades if t.pnl > 0) / len(bearish_trades) * 100

        # Sharpe/Sortino (simplified)
        if len(self.trades) > 1:
            returns = [t.pnl / self.initial_balance for t in self.trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return > 0:
                result.sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized

            neg_returns = [r for r in returns if r < 0]
            if neg_returns:
                downside_std = np.std(neg_returns)
                if downside_std > 0:
                    result.sortino_ratio = avg_return / downside_std * np.sqrt(252)

        return result

    def report(self, result: BacktestResult = None) -> str:
        """Generate text report

        Args:
            result: BacktestResult (if None, uses last run)

        Returns:
            Formatted report string
        """
        if result is None:
            result = self._calculate_results()

        report = []
        report.append("=" * 60)
        report.append("SURGE-WSI BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Symbol: {result.symbol}")
        report.append(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        report.append(f"Initial Balance: ${result.initial_balance:,.2f}")
        report.append("")
        report.append("-" * 40)
        report.append("PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Final Balance: ${result.final_balance:,.2f}")
        report.append(f"Net Profit: ${result.net_profit:,.2f} ({result.net_profit_percent:+.2f}%)")
        report.append(f"Profit Factor: {result.profit_factor:.2f}")
        report.append(f"Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_percent:.2f}%)")
        report.append("")
        report.append("-" * 40)
        report.append("TRADES")
        report.append("-" * 40)
        report.append(f"Total Trades: {result.total_trades}")
        report.append(f"Winning: {result.winning_trades} ({result.win_rate:.1f}%)")
        report.append(f"Losing: {result.losing_trades}")
        report.append(f"Avg Win: ${result.avg_win:.2f}")
        report.append(f"Avg Loss: ${result.avg_loss:.2f}")
        report.append(f"Trades/Month: {result.trades_per_month:.1f}")
        report.append("")
        report.append("-" * 40)
        report.append("PARTIAL TP PERFORMANCE")
        report.append("-" * 40)
        report.append(f"TP1 Hit Rate: {result.tp1_hit_rate:.1f}%")
        report.append(f"TP2 Hit Rate: {result.tp2_hit_rate:.1f}%")
        report.append(f"TP3 Hit Rate: {result.tp3_hit_rate:.1f}%")
        report.append("")
        report.append("-" * 40)
        report.append("BY REGIME")
        report.append("-" * 40)
        report.append(f"Bullish Trades: {result.bullish_trades} ({result.bullish_win_rate:.1f}% win)")
        report.append(f"Bearish Trades: {result.bearish_trades} ({result.bearish_win_rate:.1f}% win)")
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame

        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self.trades])


def run_backtest(
    htf_file: str,
    ltf_file: str,
    symbol: str = "GBPUSD",
    start_date: str = "2024-01-01",
    end_date: str = None
) -> BacktestResult:
    """Convenience function to run backtest from CSV files

    Args:
        htf_file: Path to HTF CSV file
        ltf_file: Path to LTF CSV file
        symbol: Trading symbol
        start_date: Start date
        end_date: End date

    Returns:
        BacktestResult
    """
    # Load data
    htf_data = pd.read_csv(htf_file, parse_dates=['time'])
    ltf_data = pd.read_csv(ltf_file, parse_dates=['time'])

    # Create backtester
    bt = Backtester(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

    # Load and run
    bt.load_data(htf_data, ltf_data)
    result = bt.run()

    # Print report
    print(bt.report(result))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SURGE-WSI Backtester")
    parser.add_argument("--htf", required=True, help="HTF CSV file path")
    parser.add_argument("--ltf", required=True, help="LTF CSV file path")
    parser.add_argument("--symbol", default="GBPUSD", help="Symbol")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")

    args = parser.parse_args()

    result = run_backtest(
        htf_file=args.htf,
        ltf_file=args.ltf,
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end
    )
