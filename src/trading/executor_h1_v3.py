"""Trade Executor H1 v3 FINAL - Production Version
===================================================

Based on backtest results:
- 100 trades, 51.0% WR, +$2,669, PF 1.50
- Return: +26.7% in 13 months
- Only 3 losing months

Workflow:
1. Kill Zone + Hybrid Mode (Dynamic Activity Filter)
2. HMM Regime Detection
3. Enhanced OB Quality + FVG Detection
4. Entry Triggers: REJECTION, MOMENTUM, HIGHER_LOW, ENGULF
5. Quality-adjusted position sizing (0.8%-1.2%)
6. Fixed SL/TP (25 pips SL, 1.5R TP)

Author: SURIOTA Team
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import pandas as pd
import numpy as np

from ..analysis.kalman_filter import MultiScaleKalman
from ..analysis.regime_detector import HMMRegimeDetector
from ..utils.killzone import KillZone
from ..utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel


class ExecutorState(Enum):
    """Executor state"""
    IDLE = "idle"
    WARMING_UP = "warming_up"
    MONITORING = "monitoring"
    TRADING = "trading"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    ticket: int = 0
    direction: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    volume: float = 0.0
    message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class ExecutorStats:
    """Executor statistics"""
    start_time: datetime = None
    total_signals: int = 0
    signals_filtered: int = 0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0

    # Daily tracking for circuit breaker
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_daily_reset: datetime = None

    @property
    def win_rate(self) -> float:
        if self.trades_executed == 0:
            return 0.0
        return self.trades_won / self.trades_executed * 100

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss


@dataclass
class OpenPosition:
    """Track open position"""
    ticket: int
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    entry_time: datetime
    poi_type: str = ""
    entry_type: str = ""
    quality_score: float = 0.0


class TradeExecutorH1V3:
    """H1 Trading Executor v3 FINAL - Production Version"""

    def __init__(
        self,
        symbol: str = "GBPUSD",
        warmup_bars: int = 100,
        magic_number: int = 20250129
    ):
        """Initialize H1 v3 FINAL Executor"""
        self.symbol = symbol
        self.warmup_bars = warmup_bars
        self.magic_number = magic_number

        # Core components
        self.kalman = MultiScaleKalman()
        self.regime_detector = HMMRegimeDetector()
        self.killzone = KillZone()

        # Dynamic Activity Filter for Hybrid Mode
        self.activity_filter = DynamicActivityFilter(
            min_atr_pips=5.0,
            min_bar_range_pips=3.0,
            activity_threshold=35.0,
            pip_size=0.0001
        )
        self.activity_filter.outside_kz_min_score = 60.0

        # v3 FINAL parameters
        self.sl_pips = 25.0
        self.tp_rr = 1.5  # 1.5R
        self.base_risk = 0.01  # 1%
        self.min_quality = 0  # No quality filter (like v2)

        # Cooldown
        self.cooldown_after_sl = timedelta(hours=1)
        self.cooldown_after_tp = timedelta(minutes=30)
        self._cooldown_until: Optional[datetime] = None

        # Circuit Breaker (max daily loss protection)
        self.max_daily_loss_pct = 0.03  # 3% max daily loss
        self._circuit_breaker_triggered = False
        self._starting_balance: float = 0.0

        # State
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(
            start_time=datetime.now(timezone.utc),
            last_daily_reset=datetime.now(timezone.utc).date()
        )
        self._warmup_count = 0
        self._position: Optional[OpenPosition] = None

        # Async locks for thread safety
        self._position_lock = asyncio.Lock()
        self._cooldown_lock = asyncio.Lock()

        # Data buffers for POI detection
        self._h1_data: Optional[pd.DataFrame] = None
        self._last_h1_bar_time: Optional[datetime] = None

        # MT5 callbacks
        self.get_account_info: Optional[Callable] = None
        self.get_tick: Optional[Callable] = None
        self.get_ohlcv: Optional[Callable] = None
        self.get_symbol_info: Optional[Callable] = None
        self.place_market_order: Optional[Callable] = None
        self.modify_position: Optional[Callable] = None
        self.close_position: Optional[Callable] = None
        self.get_positions: Optional[Callable] = None
        self.get_deal_history: Optional[Callable] = None
        self.send_telegram: Optional[Callable] = None

        # Symbol info (fetched from MT5)
        self._contract_size: float = 100000.0  # Default for forex
        self._pip_value: float = 0.0001

        logger.info("H1 v3 FINAL Executor initialized")

    def set_callbacks(
        self,
        get_account_info: Callable = None,
        get_tick: Callable = None,
        get_ohlcv: Callable = None,
        get_symbol_info: Callable = None,
        place_market_order: Callable = None,
        modify_position: Callable = None,
        close_position: Callable = None,
        get_positions: Callable = None,
        get_deal_history: Callable = None,
        send_telegram: Callable = None
    ):
        """Set execution callbacks"""
        self.get_account_info = get_account_info
        self.get_tick = get_tick
        self.get_ohlcv = get_ohlcv
        self.get_symbol_info = get_symbol_info
        self.place_market_order = place_market_order
        self.modify_position = modify_position
        self.close_position = close_position
        self.get_positions = get_positions
        self.get_deal_history = get_deal_history
        self.send_telegram = send_telegram

    async def fetch_symbol_info(self) -> bool:
        """Fetch and cache symbol info from MT5"""
        if not self.get_symbol_info:
            logger.warning("No get_symbol_info callback, using defaults")
            return True

        try:
            info = await self.get_symbol_info(self.symbol)
            if info:
                self._contract_size = info.get('contract_size', 100000.0)
                self._pip_value = info.get('point', 0.0001)
                logger.info(f"Symbol info: contract_size={self._contract_size}, pip_value={self._pip_value}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to fetch symbol info: {e}")
            return False

    async def recover_positions(self) -> bool:
        """Recover existing positions with matching magic number on startup"""
        if not self.get_positions:
            logger.warning("No get_positions callback, skipping position recovery")
            return True

        try:
            positions = await self.get_positions()
            if not positions:
                logger.info("No existing positions to recover")
                return True

            # Find positions with matching magic number
            my_positions = [p for p in positions if p.get('magic') == self.magic_number]

            if not my_positions:
                logger.info(f"No positions with magic {self.magic_number} found")
                return True

            # We only track one position at a time, take the first one
            pos = my_positions[0]
            logger.info(f"Recovering position: {pos}")

            async with self._position_lock:
                self._position = OpenPosition(
                    ticket=pos['ticket'],
                    direction='BUY' if pos['type'] == 0 else 'SELL',
                    entry_price=pos['price_open'],
                    stop_loss=pos.get('sl', 0),
                    take_profit=pos.get('tp', 0),
                    lot_size=pos['volume'],
                    entry_time=datetime.now(timezone.utc),  # Approximate
                    poi_type="RECOVERED",
                    entry_type="RECOVERED",
                    quality_score=50.0  # Default for recovered positions
                )

            logger.info(
                f"Position recovered: {self._position.direction} {self._position.lot_size} "
                f"@ {self._position.entry_price:.5f}, Ticket: {self._position.ticket}"
            )

            # Notify via Telegram
            if self.send_telegram:
                msg = f"🔄 <b>Position Recovered</b>\n\n"
                msg += f"├ Ticket: {self._position.ticket}\n"
                msg += f"├ Direction: {self._position.direction}\n"
                msg += f"├ Entry: {self._position.entry_price:.5f}\n"
                msg += f"├ SL: {self._position.stop_loss:.5f}\n"
                msg += f"├ TP: {self._position.take_profit:.5f}\n"
                msg += f"└ Lot: {self._position.lot_size}"
                await self.send_telegram(msg)

            if len(my_positions) > 1:
                logger.warning(f"Multiple positions found ({len(my_positions)}), only tracking first one")

            return True

        except Exception as e:
            logger.error(f"Position recovery failed: {e}")
            return False

    async def warmup(self, h1_data: pd.DataFrame) -> bool:
        """Warmup with H1 historical data"""
        self.state = ExecutorState.WARMING_UP

        try:
            close_col = 'close' if 'close' in h1_data.columns else 'Close'

            # Warmup Kalman and regime detector
            for price in h1_data[close_col].values:
                self.kalman.update(price)
                self.regime_detector.update(price)
                self._warmup_count += 1

            # Store H1 data for POI detection
            self._h1_data = h1_data.copy()

            logger.info(f"Warmup complete with {self._warmup_count} H1 bars")
            self.state = ExecutorState.MONITORING
            return True

        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self.state = ExecutorState.ERROR
            return False

    # =========================================================================
    # CIRCUIT BREAKER (Daily Loss Protection)
    # =========================================================================

    async def _check_daily_reset(self, now: datetime, current_balance: float):
        """Check and reset daily stats at midnight UTC"""
        today = now.date()

        if self.stats.last_daily_reset is None or self.stats.last_daily_reset != today:
            logger.info(f"Daily reset: Previous daily P/L: ${self.stats.daily_pnl:+.2f}")

            # Reset daily stats
            self.stats.daily_pnl = 0.0
            self.stats.daily_trades = 0
            self.stats.last_daily_reset = today

            # Update starting balance for new day
            self._starting_balance = current_balance
            self._circuit_breaker_triggered = False

            logger.info(f"Daily reset complete. Starting balance: ${self._starting_balance:,.2f}")

            # Notify via Telegram
            if self.send_telegram:
                msg = f"🔄 <b>Daily Reset</b>\n\n"
                msg += f"├ New trading day started\n"
                msg += f"├ Balance: ${current_balance:,.2f}\n"
                msg += f"└ Max daily loss: ${current_balance * self.max_daily_loss_pct:,.2f}"
                await self.send_telegram(msg)

    async def _check_circuit_breaker(self, current_balance: float) -> bool:
        """Check if circuit breaker should be triggered"""
        if self._starting_balance == 0:
            return False

        daily_loss_pct = (self._starting_balance - current_balance) / self._starting_balance

        if daily_loss_pct >= self.max_daily_loss_pct:
            if not self._circuit_breaker_triggered:
                await self._trigger_circuit_breaker(daily_loss_pct, current_balance)
            return True

        return False

    async def _trigger_circuit_breaker(self, loss_pct: float, current_balance: float):
        """Trigger circuit breaker - pause trading"""
        self._circuit_breaker_triggered = True
        self.state = ExecutorState.PAUSED

        logger.warning(
            f"CIRCUIT BREAKER TRIGGERED! Daily loss: {loss_pct:.2%} "
            f"(${self._starting_balance - current_balance:,.2f})"
        )

        if self.send_telegram:
            msg = f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n\n"
            msg += f"├ Daily loss: {loss_pct:.2%}\n"
            msg += f"├ Loss amount: ${self._starting_balance - current_balance:,.2f}\n"
            msg += f"├ Starting balance: ${self._starting_balance:,.2f}\n"
            msg += f"├ Current balance: ${current_balance:,.2f}\n"
            msg += f"└ Trading PAUSED until tomorrow\n\n"
            msg += f"⚠️ Max daily loss ({self.max_daily_loss_pct:.1%}) exceeded!"
            await self.send_telegram(msg)

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (use with caution)"""
        self._circuit_breaker_triggered = False
        self.state = ExecutorState.MONITORING
        logger.info("Circuit breaker manually reset")

    # =========================================================================
    # ENHANCED OB QUALITY (from v3 FINAL)
    # =========================================================================

    def _calculate_enhanced_ob_quality(
        self,
        df: pd.DataFrame,
        ob_idx: int,
        direction: str
    ) -> float:
        """Enhanced OB quality scoring"""
        quality = 0.0

        if ob_idx < 5 or ob_idx >= len(df) - 3:
            return 50

        col_map = self._get_col_map(df)
        ob_bar = df.iloc[ob_idx]
        next_bars = df.iloc[ob_idx+1:ob_idx+4]

        # 1. Base quality from impulse move (0-50 pts)
        if direction == 'BUY':
            impulse = next_bars[col_map['close']].max() - ob_bar[col_map['low']]
        else:
            impulse = ob_bar[col_map['high']] - next_bars[col_map['close']].min()

        impulse_pips = impulse * 10000
        quality += min(50, impulse_pips * 2.5)

        # 2. Wick analysis (0-25 pts)
        ob_range = ob_bar[col_map['high']] - ob_bar[col_map['low']]
        if ob_range > 0:
            if direction == 'BUY':
                upper_wick = ob_bar[col_map['high']] - max(ob_bar[col_map['open']], ob_bar[col_map['close']])
                wick_ratio = upper_wick / ob_range
            else:
                lower_wick = min(ob_bar[col_map['open']], ob_bar[col_map['close']]) - ob_bar[col_map['low']]
                wick_ratio = lower_wick / ob_range

            if wick_ratio > 0.3:
                quality += 25
            elif wick_ratio > 0.2:
                quality += 15
            elif wick_ratio > 0.1:
                quality += 10

        # 3. Fresh zone bonus (0-25 pts)
        zone_high = ob_bar[col_map['high']]
        zone_low = ob_bar[col_map['low']]
        touched = False

        for i in range(ob_idx + 4, min(ob_idx + 15, len(df))):
            bar = df.iloc[i]
            if direction == 'BUY':
                if bar[col_map['low']] <= zone_high:
                    touched = True
                    break
            else:
                if bar[col_map['high']] >= zone_low:
                    touched = True
                    break

        if not touched:
            quality += 25

        return min(100, quality)

    def _get_col_map(self, df: pd.DataFrame) -> dict:
        """Get column name mapping"""
        return {
            'close': 'close' if 'close' in df.columns else 'Close',
            'open': 'open' if 'open' in df.columns else 'Open',
            'high': 'high' if 'high' in df.columns else 'High',
            'low': 'low' if 'low' in df.columns else 'Low',
        }

    # =========================================================================
    # POI DETECTION (Order Block + FVG)
    # =========================================================================

    def _detect_order_block(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str,
        lookback: int = 15
    ) -> Optional[dict]:
        """Detect Order Block with enhanced quality"""
        if idx < lookback + 3:
            return None

        col_map = self._get_col_map(df)
        recent = df.iloc[idx-lookback:idx]

        for i in range(len(recent) - 3):
            bar = recent.iloc[i]
            next_bars = recent.iloc[i+1:i+4]
            actual_idx = idx - lookback + i

            if direction == 'BUY':
                if bar[col_map['close']] < bar[col_map['open']]:  # Bearish candle
                    move_up = next_bars[col_map['close']].max() - bar[col_map['low']]
                    if move_up > 0.0010:  # 10 pips
                        quality = self._calculate_enhanced_ob_quality(df, actual_idx, direction)
                        return {
                            'type': 'OB',
                            'direction': 'BUY',
                            'zone_high': bar[col_map['high']],
                            'zone_low': bar[col_map['low']],
                            'quality': quality
                        }
            else:
                if bar[col_map['close']] > bar[col_map['open']]:  # Bullish candle
                    move_down = bar[col_map['high']] - next_bars[col_map['close']].min()
                    if move_down > 0.0010:
                        quality = self._calculate_enhanced_ob_quality(df, actual_idx, direction)
                        return {
                            'type': 'OB',
                            'direction': 'SELL',
                            'zone_high': bar[col_map['high']],
                            'zone_low': bar[col_map['low']],
                            'quality': quality
                        }

        return None

    def _detect_fvg(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str,
        lookback: int = 8
    ) -> Optional[dict]:
        """Detect Fair Value Gap"""
        if idx < lookback + 3:
            return None

        col_map = self._get_col_map(df)
        recent = df.iloc[idx-lookback:idx]

        for i in range(len(recent) - 2):
            bar1 = recent.iloc[i]
            bar3 = recent.iloc[i+2]

            if direction == 'BUY':
                gap = bar3[col_map['low']] - bar1[col_map['high']]
                if gap > 0.0003:  # 3 pips
                    return {
                        'type': 'FVG',
                        'direction': 'BUY',
                        'zone_high': bar3[col_map['low']],
                        'zone_low': bar1[col_map['high']],
                        'quality': min(100, gap * 10000 * 2)
                    }
            else:
                gap = bar1[col_map['low']] - bar3[col_map['high']]
                if gap > 0.0003:
                    return {
                        'type': 'FVG',
                        'direction': 'SELL',
                        'zone_high': bar1[col_map['low']],
                        'zone_low': bar3[col_map['high']],
                        'quality': min(100, gap * 10000 * 2)
                    }

        return None

    # =========================================================================
    # ENTRY TRIGGER (v3 FINAL - optimized types)
    # =========================================================================

    def _check_entry_trigger(
        self,
        bar: pd.Series,
        prev_bar: pd.Series,
        direction: str
    ) -> Optional[str]:
        """
        Entry types (only profitable ones from v3 FINAL):
        - REJECTION (50% WR, +$1,063)
        - MOMENTUM (50% WR, +$973)
        - HIGHER_LOW (60% WR, +$634)
        - ENGULF
        - REMOVED: LOWER_HIGH (33% WR, negative)
        """
        col_map = self._get_col_map(self._h1_data)

        o = bar[col_map['open']]
        h = bar[col_map['high']]
        l = bar[col_map['low']]
        c = bar[col_map['close']]

        total_range = h - l
        if total_range < 0.0003:  # Min 3 pips
            return None

        body = abs(c - o)
        is_bullish = c > o
        is_bearish = c < o

        po = prev_bar[col_map['open']]
        ph = prev_bar[col_map['high']]
        pl = prev_bar[col_map['low']]
        pc = prev_bar[col_map['close']]

        if direction == 'BUY':
            # 1. Rejection candle
            lower_wick = min(o, c) - l
            if lower_wick > body and lower_wick > total_range * 0.5:
                return "REJECTION"

            # 2. Bullish momentum candle
            if is_bullish and body > total_range * 0.6:
                return "MOMENTUM"

            # 3. Bullish engulfing
            if is_bullish and c > ph and o <= pl:
                return "ENGULF"

            # 4. Higher low + bullish close
            if l > pl and is_bullish:
                return "HIGHER_LOW"

        else:  # SELL
            # 1. Rejection candle
            upper_wick = h - max(o, c)
            if upper_wick > body and upper_wick > total_range * 0.5:
                return "REJECTION"

            # 2. Bearish momentum candle
            if is_bearish and body > total_range * 0.6:
                return "MOMENTUM"

            # 3. Bearish engulfing
            if is_bearish and c < pl and o >= ph:
                return "ENGULF"

            # LOWER_HIGH removed (negative P/L in backtest)

        return None

    # =========================================================================
    # MAIN TRADING LOGIC
    # =========================================================================

    async def process_tick(
        self,
        price: float,
        account_balance: float
    ) -> Optional[TradeResult]:
        """Process new tick"""
        if self.state not in [ExecutorState.MONITORING, ExecutorState.TRADING]:
            return None

        now = datetime.now(timezone.utc)

        # Initialize starting balance for circuit breaker
        if self._starting_balance == 0:
            self._starting_balance = account_balance

        # Check for daily reset (at midnight UTC)
        await self._check_daily_reset(now, account_balance)

        # Check circuit breaker
        if self._circuit_breaker_triggered:
            return None

        # Update Kalman
        kalman_state = self.kalman.update(price)
        smoothed_price = kalman_state['smoothed_price']

        # Update regime
        regime_info = self.regime_detector.update(smoothed_price)

        # Manage existing position
        await self._manage_position(price)

        # Check cooldown (thread-safe)
        async with self._cooldown_lock:
            if self._cooldown_until and now < self._cooldown_until:
                return None

        # Skip if already in position (thread-safe)
        async with self._position_lock:
            if self._position is not None:
                return None

        # Get H1 data (with caching - only fetch when new bar forms)
        should_fetch = False
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        if self._h1_data is None:
            should_fetch = True
        elif self._last_h1_bar_time is None:
            should_fetch = True
        elif current_hour > self._last_h1_bar_time:
            should_fetch = True
            logger.debug(f"New H1 bar detected: {current_hour}")

        if should_fetch and self.get_ohlcv:
            h1_data = await self.get_ohlcv(self.symbol, "H1", 200)
            if h1_data is not None and len(h1_data) > 0:
                self._h1_data = h1_data
                self._last_h1_bar_time = current_hour
                logger.debug(f"H1 data refreshed: {len(h1_data)} bars")

        if self._h1_data is None or len(self._h1_data) < 100:
            return None

        # LAYER 2: TIME FILTER (Kill Zone + Hybrid)
        in_kz, session = self.killzone.is_in_killzone(now)

        can_trade_outside = False
        activity_score = 0.0
        if not in_kz:
            # Hybrid mode check
            idx = len(self._h1_data) - 1
            col_map = self._get_col_map(self._h1_data)
            bar = self._h1_data.iloc[-1]
            high = bar[col_map['high']]
            low = bar[col_map['low']]
            recent_df = self._h1_data.iloc[max(0, idx-20):idx+1]

            activity = self.activity_filter.check_activity(now, high, low, recent_df)
            activity_score = activity.score

            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE] and activity.score >= 60:
                can_trade_outside = True
                session = "Hybrid"

        should_trade = in_kz or can_trade_outside
        if not should_trade:
            return None

        # LAYER 3: REGIME CHECK
        if not regime_info or not regime_info.is_tradeable:
            return None
        if regime_info.bias == 'NONE':
            return None

        direction = regime_info.bias

        # LAYER 4: POI DETECTION
        idx = len(self._h1_data) - 1
        col_map = self._get_col_map(self._h1_data)

        poi = self._detect_order_block(self._h1_data, idx, direction)
        if not poi:
            poi = self._detect_fvg(self._h1_data, idx, direction)

        if not poi:
            return None

        # Check price near POI zone
        poi_tolerance = 0.0015  # 15 pips
        if direction == 'BUY':
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                return None
        else:
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                return None

        # LAYER 5: ENTRY TRIGGER
        bar = self._h1_data.iloc[-1]
        prev_bar = self._h1_data.iloc[-2]
        entry_type = self._check_entry_trigger(bar, prev_bar, direction)

        if not entry_type:
            return None

        # LAYER 6: RISK MANAGEMENT (Quality-adjusted sizing)
        quality_multiplier = 0.8 + (poi['quality'] / 100) * 0.4  # 0.8 to 1.2
        risk_pct = self.base_risk * quality_multiplier

        risk_amount = account_balance * risk_pct
        lot_size = risk_amount / (self.sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        tp_pips = self.sl_pips * self.tp_rr

        if direction == 'BUY':
            sl_price = price - self.sl_pips * 0.0001
            tp_price = price + tp_pips * 0.0001
        else:
            sl_price = price + self.sl_pips * 0.0001
            tp_price = price - tp_pips * 0.0001

        # Execute trade
        result = await self._execute_trade(
            direction=direction,
            entry_price=price,
            stop_loss=sl_price,
            take_profit=tp_price,
            volume=lot_size,
            poi_type=poi['type'],
            entry_type=entry_type,
            quality_score=poi['quality']
        )

        if result.success:
            self.stats.trades_executed += 1

            # Send Telegram notification
            if self.send_telegram:
                await self._send_trade_notification(result, session, regime_info.regime.value)

        return result

    async def _execute_trade(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volume: float,
        poi_type: str,
        entry_type: str,
        quality_score: float
    ) -> TradeResult:
        """Execute trade via MT5"""
        if not self.place_market_order:
            return TradeResult(success=False, message="No execution callback")

        try:
            result = await self.place_market_order(
                symbol=self.symbol,
                order_type=direction,
                volume=volume,
                sl=stop_loss,
                tp=take_profit,
                comment=f"H1v3_{entry_type}",
                magic=self.magic_number
            )

            if result and result.get('ticket'):
                # Track position
                self._position = OpenPosition(
                    ticket=result['ticket'],
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=volume,
                    entry_time=datetime.now(timezone.utc),
                    poi_type=poi_type,
                    entry_type=entry_type,
                    quality_score=quality_score
                )

                logger.info(
                    f"Trade executed: {direction} {volume} {self.symbol} "
                    f"@ {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}"
                )

                return TradeResult(
                    success=True,
                    ticket=result['ticket'],
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    message="Trade executed"
                )
            else:
                return TradeResult(success=False, message=f"Order rejected: {result}")

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResult(success=False, message=str(e))

    async def _manage_position(self, current_price: float):
        """Manage open position with thread-safe lock"""
        async with self._position_lock:
            if self._position is None:
                return

            # Check if position still exists
            if self.get_positions:
                positions = await self.get_positions()
                pos_exists = any(p['ticket'] == self._position.ticket for p in (positions or []))

                if not pos_exists:
                    # Position closed - try to get actual close info from deal history
                    close_reason = "UNKNOWN"
                    exit_price = current_price
                    actual_pnl = None

                    if self.get_deal_history:
                        try:
                            deal_info = await self.get_deal_history(self._position.ticket)
                            if deal_info:
                                close_reason = deal_info.get('close_reason', 'UNKNOWN')
                                exit_price = deal_info.get('close_price', current_price)
                                actual_pnl = deal_info.get('profit', 0) + deal_info.get('swap', 0) + deal_info.get('commission', 0)
                                logger.info(f"Deal history: {deal_info}")
                        except Exception as e:
                            logger.warning(f"Failed to get deal history: {e}")

                    # Fallback: determine close reason from price if deal history not available
                    if close_reason == "UNKNOWN":
                        if self._position.direction == 'BUY':
                            if current_price >= self._position.take_profit:
                                close_reason = "TP"
                            else:
                                close_reason = "SL"
                        else:
                            if current_price <= self._position.take_profit:
                                close_reason = "TP"
                            else:
                                close_reason = "SL"

                    await self._handle_close(close_reason, exit_price, actual_pnl)

    async def _handle_close(self, reason: str, exit_price: float, actual_pnl: float = None):
        """Handle position close with thread-safe cooldown update"""
        if self._position is None:
            return

        # Use actual P/L from MT5 if available, otherwise calculate
        if actual_pnl is not None:
            pnl = actual_pnl
        else:
            # Calculate P/L using contract size from symbol info
            if self._position.direction == 'BUY':
                pnl = (exit_price - self._position.entry_price) * self._position.lot_size * self._contract_size
            else:
                pnl = (self._position.entry_price - exit_price) * self._position.lot_size * self._contract_size

        is_win = pnl > 0

        async with self._cooldown_lock:
            if is_win:
                self.stats.trades_won += 1
                self.stats.total_profit += pnl
                self._cooldown_until = datetime.now(timezone.utc) + self.cooldown_after_tp
            else:
                self.stats.trades_lost += 1
                self.stats.total_loss += abs(pnl)
                self._cooldown_until = datetime.now(timezone.utc) + self.cooldown_after_sl

            # Update daily stats
            self.stats.daily_pnl += pnl
            self.stats.daily_trades += 1

        logger.info(f"Position closed: {reason}, P/L: ${pnl:+.2f}, Daily P/L: ${self.stats.daily_pnl:+.2f}")

        # Send notification
        if self.send_telegram:
            asyncio.create_task(self._send_close_notification(reason, pnl))

        self._position = None

    async def _send_trade_notification(self, result: TradeResult, session: str, regime: str):
        """Send trade notification to Telegram"""
        sl_pips = self.sl_pips
        tp_pips = sl_pips * self.tp_rr

        msg = f"🚀 <b>H1 v3 FINAL - {result.direction}</b>\n\n"
        msg += f"├ Symbol: {self.symbol}\n"
        msg += f"├ Entry: <code>{result.entry_price:.5f}</code>\n"
        msg += f"├ SL: <code>{result.stop_loss:.5f}</code> ({sl_pips:.0f} pips)\n"
        msg += f"├ TP: <code>{result.take_profit:.5f}</code> ({tp_pips:.0f} pips)\n"
        msg += f"├ Lot: {result.volume}\n"
        msg += f"├ Session: {session}\n"
        msg += f"└ Regime: {regime}\n"
        msg += f"\n⏳ Target: 1.5R"

        try:
            await self.send_telegram(msg)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    async def _send_close_notification(self, reason: str, pnl: float):
        """Send close notification to Telegram"""
        emoji = "✅" if pnl > 0 else "❌"
        msg = f"{emoji} <b>Position Closed - {reason}</b>\n\n"
        msg += f"├ P/L: ${pnl:+.2f}\n"
        msg += f"├ Win Rate: {self.stats.win_rate:.1f}%\n"
        msg += f"└ Net P/L: ${self.stats.net_pnl:+.2f}"

        try:
            await self.send_telegram(msg)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    # =========================================================================
    # CONTROL METHODS
    # =========================================================================

    def pause(self):
        """Pause trading"""
        self.state = ExecutorState.PAUSED
        logger.info("Executor paused")

    def resume(self):
        """Resume trading"""
        self.state = ExecutorState.MONITORING
        logger.info("Executor resumed")

    def stop(self):
        """Stop executor"""
        self.state = ExecutorState.STOPPED
        logger.info("Executor stopped")

    def get_status(self) -> Dict:
        """Get executor status"""
        regime = self.regime_detector.last_info
        in_kz, session = self.killzone.is_in_killzone(datetime.now(timezone.utc))

        return {
            'state': self.state.value,
            'strategy': 'H1 v3 FINAL',
            'warmup_complete': self._warmup_count >= self.warmup_bars,
            'warmup_count': self._warmup_count,
            'regime': regime.regime.value if regime else 'UNKNOWN',
            'bias': regime.bias if regime else 'NONE',
            'in_killzone': in_kz,
            'session': session,
            'has_position': self._position is not None,
            'position': {
                'direction': self._position.direction if self._position else None,
                'entry': self._position.entry_price if self._position else None,
                'sl': self._position.stop_loss if self._position else None,
                'tp': self._position.take_profit if self._position else None,
            } if self._position else None,
            'stats': {
                'trades': self.stats.trades_executed,
                'wins': self.stats.trades_won,
                'losses': self.stats.trades_lost,
                'win_rate': self.stats.win_rate,
                'net_pnl': self.stats.net_pnl,
            },
            'circuit_breaker': {
                'triggered': self._circuit_breaker_triggered,
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'daily_pnl': self.stats.daily_pnl,
                'daily_trades': self.stats.daily_trades,
            }
        }

    def reset(self):
        """Reset executor"""
        self.kalman.reset()
        self.regime_detector.reset()
        self._warmup_count = 0
        self._position = None
        self._cooldown_until = None
        self._h1_data = None
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(start_time=datetime.now(timezone.utc))
        logger.info("Executor reset")
