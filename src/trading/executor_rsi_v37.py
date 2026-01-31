"""
RSI Mean Reversion Strategy v3.7 - Production Executor
=======================================================

Backtest Results (2020-2026):
- Total Return: +618.2%
- Max Drawdown: 30.7%
- Win Rate: 37.7%
- Profit Factor: 1.22
- Risk:Reward: 1:2.02
- Total Trades: 2,654
- Profitable Months: 74%

Strategy Parameters:
- RSI(10) with 42/58 thresholds
- SL: 1.5x ATR
- TP: 2.4/3.0/3.6x ATR (dynamic by volatility)
- Time TP Bonus: +0.35x during 12:00-16:00 UTC
- ATR Filter: 20-80 percentile
- Trading Hours: 07:00-22:00 UTC, skip 12:00
- Max Holding: 46 hours
- Risk: 1% per trade

Author: SURIOTA Team
Version: 3.7 FINAL
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import pandas as pd
import numpy as np


class ExecutorState(Enum):
    """Executor state"""
    IDLE = "idle"
    WARMING_UP = "warming_up"
    MONITORING = "monitoring"
    IN_POSITION = "in_position"
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
    rsi_value: float = 0.0
    atr_pct: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class ExecutorStats:
    """Executor statistics"""
    start_time: Optional[datetime] = None
    total_signals: int = 0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0

    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_daily_reset: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return (self.trades_won / total * 100) if total > 0 else 0.0

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss


@dataclass
class OpenPosition:
    """Track open position"""
    ticket: int
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    entry_time: datetime
    entry_bar_idx: int
    rsi_entry: float
    atr_pct: float


class RSIMeanReversionV37:
    """
    RSI Mean Reversion Strategy v3.7 - Production Executor

    Simple, robust mean reversion strategy using RSI(10).
    """

    # =========================================================================
    # STRATEGY PARAMETERS (DO NOT MODIFY - Optimized from 61+ tests)
    # =========================================================================

    # RSI Parameters
    RSI_PERIOD = 10
    RSI_OVERSOLD = 42
    RSI_OVERBOUGHT = 58

    # ATR Filter
    ATR_PERIOD = 14
    ATR_LOOKBACK = 100  # For percentile calculation
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    # SL/TP Parameters
    SL_MULT = 1.5
    TP_LOW = 2.4   # ATR < 40 percentile
    TP_MED = 3.0   # ATR 40-60 percentile
    TP_HIGH = 3.6  # ATR > 60 percentile
    TIME_TP_BONUS = 0.35  # Bonus during London/NY overlap

    # Time Filters
    TRADING_START_HOUR = 7   # 07:00 UTC
    TRADING_END_HOUR = 22    # 22:00 UTC
    SKIP_HOURS = [12]        # Skip 12:00 UTC (London lunch)
    TP_BONUS_START = 12      # 12:00 UTC
    TP_BONUS_END = 16        # 16:00 UTC

    # Position Management
    MAX_HOLDING_HOURS = 46
    RISK_PER_TRADE = 0.01    # 1%

    # Circuit Breaker
    MAX_DAILY_LOSS_PCT = 0.03  # 3%

    def __init__(
        self,
        symbol: str = "GBPUSD",
        magic_number: int = 20250131,
        warmup_bars: int = 150
    ):
        """Initialize RSI Mean Reversion v3.7 Executor"""
        self.symbol = symbol
        self.magic_number = magic_number
        self.warmup_bars = warmup_bars

        # State
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(
            start_time=datetime.now(timezone.utc),
            last_daily_reset=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        )

        # Position tracking
        self._position: Optional[OpenPosition] = None
        self._position_lock = asyncio.Lock()

        # Data buffers
        self._h1_data: Optional[pd.DataFrame] = None
        self._last_bar_time: Optional[datetime] = None
        self._current_bar_idx: int = 0

        # Calculated indicators
        self._rsi: Optional[pd.Series] = None
        self._atr: Optional[pd.Series] = None
        self._atr_pct: Optional[pd.Series] = None

        # Circuit breaker
        self._circuit_breaker_triggered = False
        self._starting_balance: float = 0.0

        # MT5 Callbacks
        self.get_account_info: Optional[Callable] = None
        self.get_tick: Optional[Callable] = None
        self.get_ohlcv: Optional[Callable] = None
        self.get_symbol_info: Optional[Callable] = None
        self.place_market_order: Optional[Callable] = None
        self.close_position: Optional[Callable] = None
        self.get_positions: Optional[Callable] = None
        self.get_deal_history: Optional[Callable] = None
        self.send_telegram: Optional[Callable] = None

        # Symbol info
        self._contract_size: float = 100000.0
        self._pip_value: float = 0.0001
        self._digits: int = 5

        logger.info(f"RSI Mean Reversion v3.7 Executor initialized for {symbol}")

    def set_callbacks(
        self,
        get_account_info: Callable = None,
        get_tick: Callable = None,
        get_ohlcv: Callable = None,
        get_symbol_info: Callable = None,
        place_market_order: Callable = None,
        close_position: Callable = None,
        get_positions: Callable = None,
        get_deal_history: Callable = None,
        send_telegram: Callable = None
    ):
        """Set MT5 execution callbacks"""
        self.get_account_info = get_account_info
        self.get_tick = get_tick
        self.get_ohlcv = get_ohlcv
        self.get_symbol_info = get_symbol_info
        self.place_market_order = place_market_order
        self.close_position = close_position
        self.get_positions = get_positions
        self.get_deal_history = get_deal_history
        self.send_telegram = send_telegram

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI using SMA method (faster reaction, better for mean reversion)"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.RSI_PERIOD).mean()

        # Safe division
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill initial NaN with neutral value

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR(14)"""
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )
        return tr.rolling(self.ATR_PERIOD).mean()

    def _calculate_atr_percentile(self, atr: pd.Series) -> pd.Series:
        """Calculate ATR percentile: % of historical values below current"""
        def atr_percentile(x):
            if len(x) <= 1:
                return 50.0
            current = x[-1]
            count_below = (x[:-1] < current).sum()  # Compare against historical only
            return (count_below / (len(x) - 1)) * 100

        return atr.rolling(self.ATR_LOOKBACK).apply(atr_percentile, raw=True)

    def _update_indicators(self, df: pd.DataFrame):
        """Update all indicators from OHLCV data"""
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        self._rsi = self._calculate_rsi(close)
        self._atr = self._calculate_atr(high, low, close)
        self._atr_pct = self._calculate_atr_percentile(self._atr)

    # =========================================================================
    # TIME FILTERS
    # =========================================================================

    def _is_trading_hour(self, dt: datetime) -> bool:
        """Check if current time is within trading hours"""
        hour = dt.hour
        weekday = dt.weekday()

        # Skip weekends
        if weekday >= 5:
            return False

        # Check trading hours
        if hour < self.TRADING_START_HOUR or hour >= self.TRADING_END_HOUR:
            return False

        # Skip lunch hour
        if hour in self.SKIP_HOURS:
            return False

        return True

    def _get_tp_multiplier(self, atr_pct: float, hour: int) -> float:
        """Get TP multiplier based on volatility and time"""
        # Base TP by volatility regime
        if atr_pct < 40:
            base_tp = self.TP_LOW
        elif atr_pct > 60:
            base_tp = self.TP_HIGH
        else:
            base_tp = self.TP_MED

        # Add time bonus during London/NY overlap
        if self.TP_BONUS_START <= hour < self.TP_BONUS_END:
            return base_tp + self.TIME_TP_BONUS

        return base_tp

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _check_signal(self, idx: int, dt: datetime) -> Optional[int]:
        """
        Check for trade signal.
        Returns: 1 for BUY, -1 for SELL, None for no signal
        """
        # Time filter
        if not self._is_trading_hour(dt):
            return None

        # Get current indicator values
        rsi = self._rsi.iloc[idx]
        atr_pct = self._atr_pct.iloc[idx]

        # Check for NaN
        if pd.isna(rsi) or pd.isna(atr_pct):
            return None

        # ATR filter
        if atr_pct < self.MIN_ATR_PCT or atr_pct > self.MAX_ATR_PCT:
            return None

        # RSI signal
        if rsi < self.RSI_OVERSOLD:
            return 1  # BUY
        elif rsi > self.RSI_OVERBOUGHT:
            return -1  # SELL

        return None

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def _calculate_position_size(self, balance: float, entry: float, sl: float) -> Optional[float]:
        """Calculate position size based on risk. Returns None if invalid."""
        risk_amount = balance * self.RISK_PER_TRADE
        sl_distance = abs(entry - sl)

        if sl_distance <= 0:
            logger.warning(f"Invalid SL distance: {sl_distance}")
            return None

        # Lot size = Risk / (SL distance in price * contract size)
        # For GBPUSD: contract_size = 100,000, sl_distance in price units
        # Example: $100 risk, 0.0015 SL distance, 100k contract
        #          lot_size = 100 / (0.0015 * 100000) = 0.67 lots
        lot_size = risk_amount / (sl_distance * self._contract_size)

        # Clamp to valid range (0.01 to 10.0 lots)
        original_lot_size = lot_size
        lot_size = max(0.01, min(10.0, round(lot_size, 2)))

        if lot_size != round(original_lot_size, 2):
            logger.warning(f"Lot size clamped: {original_lot_size:.4f} -> {lot_size}")

        return lot_size

    async def _check_max_holding(self, current_time: datetime) -> bool:
        """Check if position exceeded max holding time"""
        if self._position is None:
            return False

        holding_hours = (current_time - self._position.entry_time).total_seconds() / 3600
        return holding_hours >= self.MAX_HOLDING_HOURS

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    async def _check_daily_reset(self, now: datetime, balance: float):
        """Reset daily stats at midnight UTC"""
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self.stats.last_daily_reset is None or self.stats.last_daily_reset < today_midnight:
            logger.info(f"Daily reset: Yesterday P/L: ${self.stats.daily_pnl:+.2f}")

            self.stats.daily_pnl = 0.0
            self.stats.daily_trades = 0
            self.stats.last_daily_reset = today_midnight
            self._starting_balance = balance
            self._circuit_breaker_triggered = False

            if self.send_telegram:
                msg = f"🔄 <b>Daily Reset - RSI v3.7</b>\n\n"
                msg += f"├ Balance: ${balance:,.2f}\n"
                msg += f"└ Max daily loss: ${balance * self.MAX_DAILY_LOSS_PCT:,.2f}"
                await self.send_telegram(msg)

    async def _check_circuit_breaker(self, balance: float) -> bool:
        """Check if daily loss limit exceeded"""
        if self._starting_balance <= 0:
            return False

        daily_loss = (self._starting_balance - balance) / self._starting_balance

        if daily_loss >= self.MAX_DAILY_LOSS_PCT:
            if not self._circuit_breaker_triggered:
                self._circuit_breaker_triggered = True
                self.state = ExecutorState.PAUSED

                logger.warning(f"CIRCUIT BREAKER: Daily loss {daily_loss:.2%}")

                if self.send_telegram:
                    msg = f"🚨 <b>CIRCUIT BREAKER - RSI v3.7</b>\n\n"
                    msg += f"├ Daily loss: {daily_loss:.2%}\n"
                    msg += f"├ Loss: ${self._starting_balance - balance:,.2f}\n"
                    msg += f"└ Trading PAUSED until tomorrow"
                    await self.send_telegram(msg)

            return True

        return False

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    async def fetch_symbol_info(self) -> bool:
        """Fetch symbol info from MT5"""
        if not self.get_symbol_info:
            logger.warning("No get_symbol_info callback")
            return True

        try:
            info = await self.get_symbol_info(self.symbol)
            if info:
                self._contract_size = info.get('contract_size', 100000.0)
                self._pip_value = info.get('point', 0.0001)
                self._digits = info.get('digits', 5)
                logger.info(f"Symbol info: contract={self._contract_size}, pip={self._pip_value}")
            return True
        except Exception as e:
            logger.error(f"Failed to fetch symbol info: {e}")
            return False

    async def recover_positions(self) -> bool:
        """Recover existing positions on startup"""
        if not self.get_positions:
            return True

        try:
            positions = await self.get_positions()
            if not positions:
                return True

            my_positions = [p for p in positions if p.get('magic') == self.magic_number]

            if not my_positions:
                logger.info("No positions to recover")
                return True

            pos = my_positions[0]

            # Get entry time from position or use current time
            entry_time = pos.get('time')
            if entry_time is None:
                entry_time = datetime.now(timezone.utc)
            elif isinstance(entry_time, (int, float)):
                entry_time = datetime.fromtimestamp(entry_time, tz=timezone.utc)

            async with self._position_lock:
                self._position = OpenPosition(
                    ticket=pos['ticket'],
                    direction='BUY' if pos['type'] == 0 else 'SELL',
                    entry_price=pos['price_open'],
                    stop_loss=pos.get('sl', 0),
                    take_profit=pos.get('tp', 0),
                    lot_size=pos['volume'],
                    entry_time=entry_time,
                    entry_bar_idx=0,
                    rsi_entry=50.0,
                    atr_pct=50.0
                )
                self.state = ExecutorState.IN_POSITION

            logger.info(f"Position recovered: {self._position.direction} @ {self._position.entry_price}")

            if self.send_telegram:
                msg = f"🔄 <b>Position Recovered - RSI v3.7</b>\n\n"
                msg += f"├ Direction: {self._position.direction}\n"
                msg += f"├ Entry: {self._position.entry_price:.5f}\n"
                msg += f"├ SL: {self._position.stop_loss:.5f}\n"
                msg += f"└ TP: {self._position.take_profit:.5f}"
                await self.send_telegram(msg)

            return True

        except Exception as e:
            logger.error(f"Position recovery failed: {e}")
            return False

    async def warmup(self, h1_data: pd.DataFrame = None) -> bool:
        """Warmup with historical data"""
        self.state = ExecutorState.WARMING_UP

        try:
            if h1_data is None and self.get_ohlcv:
                h1_data = await self.get_ohlcv(self.symbol, "H1", self.warmup_bars + 50)

            if h1_data is None or len(h1_data) < self.warmup_bars:
                logger.error("Insufficient warmup data")
                self.state = ExecutorState.ERROR
                return False

            self._h1_data = h1_data.copy()
            self._update_indicators(self._h1_data)
            self._current_bar_idx = len(self._h1_data) - 1

            logger.info(f"Warmup complete: {len(self._h1_data)} bars")
            self.state = ExecutorState.MONITORING
            return True

        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self.state = ExecutorState.ERROR
            return False

    # =========================================================================
    # MAIN TRADING LOOP
    # =========================================================================

    async def on_new_bar(self, bar_time: datetime, balance: float) -> Optional[TradeResult]:
        """
        Called when a new H1 bar forms.
        This is the main entry point for the strategy.
        """
        if self.state not in [ExecutorState.MONITORING, ExecutorState.IN_POSITION]:
            return None

        now = datetime.now(timezone.utc)

        # Initialize starting balance
        if self._starting_balance <= 0:
            self._starting_balance = balance

        # Daily reset check
        await self._check_daily_reset(now, balance)

        # Circuit breaker check
        if await self._check_circuit_breaker(balance):
            return None

        # Fetch latest H1 data
        if self.get_ohlcv:
            new_data = await self.get_ohlcv(self.symbol, "H1", self.warmup_bars + 50)
            if new_data is not None and len(new_data) > 0:
                self._h1_data = new_data
                self._update_indicators(self._h1_data)
                self._current_bar_idx = len(self._h1_data) - 1

        if self._h1_data is None or self._rsi is None:
            return None

        # Check if in position - manage existing position
        async with self._position_lock:
            if self._position is not None:
                return await self._manage_position(now, balance)

        # Check for new signal
        signal = self._check_signal(self._current_bar_idx, bar_time)

        if signal is None:
            return None

        self.stats.total_signals += 1

        # Get current values
        close_col = 'close' if 'close' in self._h1_data.columns else 'Close'
        entry_price = self._h1_data[close_col].iloc[-1]
        atr = self._atr.iloc[-1]
        atr_pct = self._atr_pct.iloc[-1]
        rsi = self._rsi.iloc[-1]

        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.002  # Fallback

        # Calculate SL/TP
        direction = 'BUY' if signal == 1 else 'SELL'
        tp_mult = self._get_tp_multiplier(atr_pct, bar_time.hour)

        if direction == 'BUY':
            sl = entry_price - atr * self.SL_MULT
            tp = entry_price + atr * tp_mult
        else:
            sl = entry_price + atr * self.SL_MULT
            tp = entry_price - atr * tp_mult

        # Calculate position size
        lot_size = self._calculate_position_size(balance, entry_price, sl)

        if lot_size is None:
            logger.warning("Invalid position size, skipping trade")
            return None

        # Execute trade
        return await self._execute_trade(
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            volume=lot_size,
            rsi_value=rsi,
            atr_pct=atr_pct,
            bar_time=bar_time
        )

    async def _execute_trade(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volume: float,
        rsi_value: float,
        atr_pct: float,
        bar_time: datetime
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
                comment=f"RSIv37_{direction[:1]}",
                magic=self.magic_number
            )

            if result and result.get('ticket'):
                async with self._position_lock:
                    self._position = OpenPosition(
                        ticket=result['ticket'],
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        lot_size=volume,
                        entry_time=datetime.now(timezone.utc),
                        entry_bar_idx=self._current_bar_idx,
                        rsi_entry=rsi_value,
                        atr_pct=atr_pct
                    )
                    self.state = ExecutorState.IN_POSITION

                self.stats.trades_executed += 1
                self.stats.daily_trades += 1

                logger.info(
                    f"Trade executed: {direction} {volume} @ {entry_price:.5f} "
                    f"SL:{stop_loss:.5f} TP:{take_profit:.5f} RSI:{rsi_value:.1f}"
                )

                # Send notification
                if self.send_telegram:
                    await self._send_trade_notification(
                        direction, entry_price, stop_loss, take_profit,
                        volume, rsi_value, atr_pct
                    )

                return TradeResult(
                    success=True,
                    ticket=result['ticket'],
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    rsi_value=rsi_value,
                    atr_pct=atr_pct,
                    message="Trade executed"
                )
            else:
                return TradeResult(success=False, message=f"Order rejected: {result}")

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResult(success=False, message=str(e))

    async def _manage_position(self, current_time: datetime, balance: float) -> Optional[TradeResult]:
        """Manage existing position"""
        if self._position is None:
            return None

        # Check max holding time
        if await self._check_max_holding(current_time):
            logger.info("Max holding time reached - closing position")
            return await self._close_position_timeout()

        # Check if position still exists in MT5
        if self.get_positions:
            positions = await self.get_positions()
            pos_exists = any(
                p['ticket'] == self._position.ticket
                for p in (positions or [])
            )

            if not pos_exists:
                # Position was closed by SL/TP
                await self._handle_position_closed()

        return None

    async def _close_position_timeout(self) -> Optional[TradeResult]:
        """Close position due to timeout"""
        if not self.close_position or self._position is None:
            return None

        try:
            result = await self.close_position(self._position.ticket)
            if result:
                await self._handle_position_closed(close_reason="TIMEOUT")
            return None
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None

    async def _handle_position_closed(self, close_reason: str = "SL/TP"):
        """Handle position closure"""
        if self._position is None:
            return

        # Try to get actual P/L from deal history
        actual_pnl = 0.0

        if self.get_deal_history:
            try:
                deal_info = await self.get_deal_history(self._position.ticket)
                if deal_info:
                    actual_pnl = deal_info.get('profit', 0)
                    actual_pnl += deal_info.get('swap', 0)
                    actual_pnl += deal_info.get('commission', 0)
                    close_reason = deal_info.get('close_reason', close_reason)
            except Exception as e:
                logger.warning(f"Failed to get deal history: {e}")

        is_win = actual_pnl > 0

        if is_win:
            self.stats.trades_won += 1
            self.stats.total_profit += actual_pnl
        else:
            self.stats.trades_lost += 1
            self.stats.total_loss += abs(actual_pnl)

        self.stats.daily_pnl += actual_pnl

        logger.info(
            f"Position closed: {close_reason}, P/L: ${actual_pnl:+.2f}, "
            f"Daily: ${self.stats.daily_pnl:+.2f}"
        )

        # Send notification
        if self.send_telegram:
            await self._send_close_notification(close_reason, actual_pnl)

        self._position = None
        self.state = ExecutorState.MONITORING

    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================

    async def _send_trade_notification(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        volume: float,
        rsi: float,
        atr_pct: float
    ):
        """Send trade notification"""
        sl_pips = abs(entry - sl) / self._pip_value
        tp_pips = abs(tp - entry) / self._pip_value
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        emoji = "🟢" if direction == "BUY" else "🔴"

        msg = f"{emoji} <b>RSI v3.7 - {direction}</b>\n\n"
        msg += f"├ Symbol: {self.symbol}\n"
        msg += f"├ Entry: <code>{entry:.5f}</code>\n"
        msg += f"├ SL: <code>{sl:.5f}</code> ({sl_pips:.0f} pips)\n"
        msg += f"├ TP: <code>{tp:.5f}</code> ({tp_pips:.0f} pips)\n"
        msg += f"├ R:R: 1:{rr:.1f}\n"
        msg += f"├ Lot: {volume}\n"
        msg += f"├ RSI: {rsi:.1f}\n"
        msg += f"└ ATR%: {atr_pct:.0f}\n"
        msg += f"\n📊 WR: {self.stats.win_rate:.1f}% | Net: ${self.stats.net_pnl:+.0f}"

        try:
            await self.send_telegram(msg)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    async def _send_close_notification(self, reason: str, pnl: float):
        """Send close notification"""
        emoji = "✅" if pnl > 0 else "❌"

        msg = f"{emoji} <b>Position Closed - {reason}</b>\n\n"
        msg += f"├ P/L: ${pnl:+.2f}\n"
        msg += f"├ Daily P/L: ${self.stats.daily_pnl:+.2f}\n"
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
        logger.info("RSI v3.7 Executor paused")

    def resume(self):
        """Resume trading"""
        if self._position:
            self.state = ExecutorState.IN_POSITION
        else:
            self.state = ExecutorState.MONITORING
        logger.info("RSI v3.7 Executor resumed")

    def stop(self):
        """Stop executor"""
        self.state = ExecutorState.STOPPED
        logger.info("RSI v3.7 Executor stopped")

    def get_status(self) -> Dict:
        """Get executor status"""
        return {
            'strategy': 'RSI Mean Reversion v3.7',
            'symbol': self.symbol,
            'state': self.state.value,
            'has_position': self._position is not None,
            'position': {
                'direction': self._position.direction,
                'entry': self._position.entry_price,
                'sl': self._position.stop_loss,
                'tp': self._position.take_profit,
                'lot': self._position.lot_size,
                'rsi': self._position.rsi_entry,
            } if self._position else None,
            'stats': {
                'signals': self.stats.total_signals,
                'trades': self.stats.trades_executed,
                'wins': self.stats.trades_won,
                'losses': self.stats.trades_lost,
                'win_rate': f"{self.stats.win_rate:.1f}%",
                'net_pnl': f"${self.stats.net_pnl:+.2f}",
                'daily_pnl': f"${self.stats.daily_pnl:+.2f}",
            },
            'circuit_breaker': {
                'triggered': self._circuit_breaker_triggered,
                'daily_loss_limit': f"{self.MAX_DAILY_LOSS_PCT:.1%}",
            },
            'indicators': {
                'rsi': f"{self._rsi.iloc[-1]:.1f}" if self._rsi is not None else None,
                'atr_pct': f"{self._atr_pct.iloc[-1]:.0f}" if self._atr_pct is not None else None,
            }
        }

    def reset(self):
        """Reset executor"""
        self._position = None
        self._h1_data = None
        self._rsi = None
        self._atr = None
        self._atr_pct = None
        self._circuit_breaker_triggered = False
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(start_time=datetime.now(timezone.utc))
        logger.info("RSI v3.7 Executor reset")
