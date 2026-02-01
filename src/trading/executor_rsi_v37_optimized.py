"""
RSI Mean Reversion Strategy v3.7 OPTIMIZED - Production Executor
=================================================================

Optimized with SIDEWAYS Regime + Consecutive Loss Filter

Backtest Results (Oct 2024 - Jan 2026):
- Total Return: +72.7%
- Max Drawdown: 14.4% (improved from 19.7%)
- Win Rate: 37.6%
- Losing Months: 2 (reduced from 3)
- Total Trades: 519

Filters Applied:
1. SIDEWAYS Regime Only (SMA20/50 crossover + slope)
2. Consecutive Loss Pause (after 3 consecutive losses)

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
Version: 3.7 OPTIMIZED
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import pandas as pd
import numpy as np


class ExecutorState(Enum):
    IDLE = "idle"
    WARMING_UP = "warming_up"
    MONITORING = "monitoring"
    IN_POSITION = "in_position"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TradeResult:
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
    regime: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class ExecutorStats:
    start_time: Optional[datetime] = None
    total_signals: int = 0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    trades_filtered: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    consecutive_losses: int = 0

    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_daily_reset: Optional[datetime] = None

    # Monthly tracking
    monthly_pnl: float = 0.0
    current_month: str = ""

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return (self.trades_won / total * 100) if total > 0 else 0.0

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss


@dataclass
class OpenPosition:
    ticket: int
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    entry_time: datetime
    entry_bar_idx: int
    rsi_entry: float
    atr_pct: float
    regime: str = "SIDEWAYS"


class RSIMeanReversionV37Optimized:
    """
    RSI Mean Reversion Strategy v3.7 OPTIMIZED

    Includes:
    - SIDEWAYS regime filter
    - Consecutive loss filter (pause after 3 losses)
    """

    # =========================================================================
    # STRATEGY PARAMETERS
    # =========================================================================

    # RSI Parameters
    RSI_PERIOD = 10
    RSI_OVERSOLD = 42
    RSI_OVERBOUGHT = 58

    # ATR Filter
    ATR_PERIOD = 14
    ATR_LOOKBACK = 100
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    # SL/TP Parameters
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    TIME_TP_BONUS = 0.35

    # Time Filters
    TRADING_START_HOUR = 7
    TRADING_END_HOUR = 22
    SKIP_HOURS = [12]
    TP_BONUS_START = 12
    TP_BONUS_END = 16

    # Position Management
    MAX_HOLDING_HOURS = 46
    RISK_PER_TRADE = 0.01

    # Circuit Breaker
    MAX_DAILY_LOSS_PCT = 0.03

    # =========================================================================
    # OPTIMIZATION FILTERS
    # =========================================================================

    # Regime Filter
    USE_REGIME_FILTER = True
    ALLOWED_REGIMES = ["SIDEWAYS"]  # Only trade in SIDEWAYS market

    # Consecutive Loss Filter
    USE_CONSEC_LOSS_FILTER = True
    CONSEC_LOSS_LIMIT = 3  # Pause after 3 consecutive losses

    def __init__(
        self,
        symbol: str = "GBPUSD",
        magic_number: int = 20250201,
        warmup_bars: int = 150
    ):
        self.symbol = symbol
        self.magic_number = magic_number
        self.warmup_bars = warmup_bars

        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(
            start_time=datetime.now(timezone.utc),
            last_daily_reset=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        )

        self._position: Optional[OpenPosition] = None
        self._position_lock = asyncio.Lock()

        self._h1_data: Optional[pd.DataFrame] = None
        self._last_bar_time: Optional[datetime] = None
        self._current_bar_idx: int = 0

        # Indicators
        self._rsi: Optional[pd.Series] = None
        self._atr: Optional[pd.Series] = None
        self._atr_pct: Optional[pd.Series] = None
        self._regime: Optional[pd.Series] = None

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

        self._contract_size: float = 100000.0
        self._pip_value: float = 0.0001
        self._digits: int = 5

        logger.info(f"RSI v3.7 OPTIMIZED initialized for {symbol}")
        logger.info(f"Filters: Regime={self.ALLOWED_REGIMES}, ConsecLoss={self.CONSEC_LOSS_LIMIT}")

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
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.RSI_PERIOD).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )
        return pd.Series(tr, index=close.index).rolling(self.ATR_PERIOD).mean()

    def _calculate_atr_percentile(self, atr: pd.Series) -> pd.Series:
        def atr_percentile(x):
            if len(x) <= 1:
                return 50.0
            current = x[-1]
            count_below = (x[:-1] < current).sum()
            return (count_below / (len(x) - 1)) * 100
        return atr.rolling(self.ATR_LOOKBACK).apply(atr_percentile, raw=True)

    def _calculate_regime(self, close: pd.Series) -> pd.Series:
        """Calculate market regime: BULL, BEAR, or SIDEWAYS"""
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_slope = (sma_20 / sma_20.shift(10) - 1) * 100

        conditions = [
            (sma_20 > sma_50) & (sma_slope > 0.5),
            (sma_20 < sma_50) & (sma_slope < -0.5),
        ]
        choices = ['BULL', 'BEAR']
        regime = np.select(conditions, choices, default='SIDEWAYS')
        return pd.Series(regime, index=close.index)

    def _update_indicators(self, df: pd.DataFrame):
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']

        self._rsi = self._calculate_rsi(close)
        self._atr = self._calculate_atr(high, low, close)
        self._atr_pct = self._calculate_atr_percentile(self._atr)
        self._regime = self._calculate_regime(close)

    # =========================================================================
    # TIME FILTERS
    # =========================================================================

    def _is_trading_hour(self, dt: datetime) -> bool:
        hour = dt.hour
        weekday = dt.weekday()

        if weekday >= 5:
            return False
        if hour < self.TRADING_START_HOUR or hour >= self.TRADING_END_HOUR:
            return False
        if hour in self.SKIP_HOURS:
            return False

        return True

    def _get_tp_multiplier(self, atr_pct: float, hour: int) -> float:
        if atr_pct < 40:
            base_tp = self.TP_LOW
        elif atr_pct > 60:
            base_tp = self.TP_HIGH
        else:
            base_tp = self.TP_MED

        if self.TP_BONUS_START <= hour < self.TP_BONUS_END:
            return base_tp + self.TIME_TP_BONUS

        return base_tp

    # =========================================================================
    # OPTIMIZATION FILTERS
    # =========================================================================

    def _check_regime_filter(self, regime: str) -> bool:
        """Check if current regime is allowed for trading"""
        if not self.USE_REGIME_FILTER:
            return True
        return regime in self.ALLOWED_REGIMES

    def _check_consec_loss_filter(self) -> bool:
        """Check if we should pause due to consecutive losses"""
        if not self.USE_CONSEC_LOSS_FILTER:
            return True

        if self.stats.consecutive_losses >= self.CONSEC_LOSS_LIMIT:
            logger.info(f"ConsecLoss filter: {self.stats.consecutive_losses} consecutive losses, skipping")
            self.stats.consecutive_losses = 0  # Reset after skip
            return False

        return True

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _check_signal(self, idx: int, dt: datetime) -> Optional[int]:
        if not self._is_trading_hour(dt):
            return None

        rsi = self._rsi.iloc[idx]
        atr_pct = self._atr_pct.iloc[idx]
        regime = self._regime.iloc[idx]

        if pd.isna(rsi) or pd.isna(atr_pct):
            return None

        # ATR filter
        if atr_pct < self.MIN_ATR_PCT or atr_pct > self.MAX_ATR_PCT:
            return None

        # Regime filter
        if not self._check_regime_filter(regime):
            self.stats.trades_filtered += 1
            return None

        # Consecutive loss filter
        if not self._check_consec_loss_filter():
            self.stats.trades_filtered += 1
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
        risk_amount = balance * self.RISK_PER_TRADE
        sl_distance = abs(entry - sl)

        if sl_distance <= 0:
            return None

        lot_size = risk_amount / (sl_distance * self._contract_size)
        lot_size = max(0.01, min(10.0, round(lot_size, 2)))
        return lot_size

    async def _check_max_holding(self, current_time: datetime) -> bool:
        if self._position is None:
            return False
        holding_hours = (current_time - self._position.entry_time).total_seconds() / 3600
        return holding_hours >= self.MAX_HOLDING_HOURS

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    async def _check_daily_reset(self, now: datetime, balance: float):
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self.stats.last_daily_reset is None or self.stats.last_daily_reset < today_midnight:
            logger.info(f"Daily reset: Yesterday P/L: ${self.stats.daily_pnl:+.2f}")

            self.stats.daily_pnl = 0.0
            self.stats.daily_trades = 0
            self.stats.last_daily_reset = today_midnight
            self._starting_balance = balance
            self._circuit_breaker_triggered = False

            if self.send_telegram:
                msg = f"🔄 <b>Daily Reset - RSI v3.7 OPT</b>\n\n"
                msg += f"├ Balance: ${balance:,.2f}\n"
                msg += f"├ Consec Losses: {self.stats.consecutive_losses}\n"
                msg += f"└ Max daily loss: ${balance * self.MAX_DAILY_LOSS_PCT:,.2f}"
                await self.send_telegram(msg)

    async def _check_circuit_breaker(self, balance: float) -> bool:
        if self._starting_balance <= 0:
            return False

        daily_loss = (self._starting_balance - balance) / self._starting_balance

        if daily_loss >= self.MAX_DAILY_LOSS_PCT:
            if not self._circuit_breaker_triggered:
                self._circuit_breaker_triggered = True
                self.state = ExecutorState.PAUSED

                logger.warning(f"CIRCUIT BREAKER: Daily loss {daily_loss:.2%}")

                if self.send_telegram:
                    msg = f"🚨 <b>CIRCUIT BREAKER</b>\n\n"
                    msg += f"├ Daily loss: {daily_loss:.2%}\n"
                    msg += f"└ Trading PAUSED"
                    await self.send_telegram(msg)

            return True

        return False

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    async def fetch_symbol_info(self) -> bool:
        if not self.get_symbol_info:
            return True

        try:
            info = await self.get_symbol_info(self.symbol)
            if info:
                self._contract_size = info.get('contract_size', 100000.0)
                self._pip_value = info.get('point', 0.0001)
                self._digits = info.get('digits', 5)
            return True
        except Exception as e:
            logger.error(f"Failed to fetch symbol info: {e}")
            return False

    async def recover_positions(self) -> bool:
        if not self.get_positions:
            return True

        try:
            positions = await self.get_positions()
            if not positions:
                return True

            my_positions = [p for p in positions if p.get('magic') == self.magic_number]

            if not my_positions:
                return True

            pos = my_positions[0]
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
                    atr_pct=50.0,
                    regime="SIDEWAYS"
                )
                self.state = ExecutorState.IN_POSITION

            logger.info(f"Position recovered: {self._position.direction} @ {self._position.entry_price}")
            return True

        except Exception as e:
            logger.error(f"Position recovery failed: {e}")
            return False

    async def warmup(self, h1_data: pd.DataFrame = None) -> bool:
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

            # Log current regime
            current_regime = self._regime.iloc[-1] if self._regime is not None else "UNKNOWN"
            logger.info(f"Warmup complete: {len(self._h1_data)} bars, Regime: {current_regime}")

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
        if self.state not in [ExecutorState.MONITORING, ExecutorState.IN_POSITION]:
            return None

        now = datetime.now(timezone.utc)

        if self._starting_balance <= 0:
            self._starting_balance = balance

        await self._check_daily_reset(now, balance)

        if await self._check_circuit_breaker(balance):
            return None

        # Fetch latest data
        if self.get_ohlcv:
            new_data = await self.get_ohlcv(self.symbol, "H1", self.warmup_bars + 50)
            if new_data is not None and len(new_data) > 0:
                self._h1_data = new_data
                self._update_indicators(self._h1_data)
                self._current_bar_idx = len(self._h1_data) - 1

        if self._h1_data is None or self._rsi is None:
            return None

        # Manage existing position
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
        regime = self._regime.iloc[-1]

        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.002

        # Calculate SL/TP
        direction = 'BUY' if signal == 1 else 'SELL'
        tp_mult = self._get_tp_multiplier(atr_pct, bar_time.hour)

        if direction == 'BUY':
            sl = entry_price - atr * self.SL_MULT
            tp = entry_price + atr * tp_mult
        else:
            sl = entry_price + atr * self.SL_MULT
            tp = entry_price - atr * tp_mult

        lot_size = self._calculate_position_size(balance, entry_price, sl)

        if lot_size is None:
            return None

        return await self._execute_trade(
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            volume=lot_size,
            rsi_value=rsi,
            atr_pct=atr_pct,
            regime=regime,
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
        regime: str,
        bar_time: datetime
    ) -> TradeResult:
        if not self.place_market_order:
            return TradeResult(success=False, message="No execution callback")

        try:
            result = await self.place_market_order(
                symbol=self.symbol,
                order_type=direction,
                volume=volume,
                sl=stop_loss,
                tp=take_profit,
                comment=f"RSIv37O_{direction[:1]}",
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
                        atr_pct=atr_pct,
                        regime=regime
                    )
                    self.state = ExecutorState.IN_POSITION

                self.stats.trades_executed += 1
                self.stats.daily_trades += 1

                logger.info(
                    f"Trade: {direction} {volume} @ {entry_price:.5f} "
                    f"SL:{stop_loss:.5f} TP:{take_profit:.5f} RSI:{rsi_value:.1f} Regime:{regime}"
                )

                if self.send_telegram:
                    await self._send_trade_notification(
                        direction, entry_price, stop_loss, take_profit,
                        volume, rsi_value, atr_pct, regime
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
                    regime=regime,
                    message="Trade executed"
                )
            else:
                return TradeResult(success=False, message=f"Order rejected: {result}")

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResult(success=False, message=str(e))

    async def _manage_position(self, current_time: datetime, balance: float) -> Optional[TradeResult]:
        if self._position is None:
            return None

        if await self._check_max_holding(current_time):
            logger.info("Max holding time - closing")
            return await self._close_position_timeout()

        if self.get_positions:
            positions = await self.get_positions()
            pos_exists = any(
                p['ticket'] == self._position.ticket
                for p in (positions or [])
            )

            if not pos_exists:
                await self._handle_position_closed()

        return None

    async def _close_position_timeout(self) -> Optional[TradeResult]:
        if not self.close_position or self._position is None:
            return None

        try:
            result = await self.close_position(self._position.ticket)
            if result:
                await self._handle_position_closed(close_reason="TIMEOUT")
            return None
        except Exception as e:
            logger.error(f"Failed to close: {e}")
            return None

    async def _handle_position_closed(self, close_reason: str = "SL/TP"):
        if self._position is None:
            return

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
            self.stats.consecutive_losses = 0  # Reset on win
        else:
            self.stats.trades_lost += 1
            self.stats.total_loss += abs(actual_pnl)
            self.stats.consecutive_losses += 1  # Increment on loss

        self.stats.daily_pnl += actual_pnl

        logger.info(
            f"Closed: {close_reason}, P/L: ${actual_pnl:+.2f}, "
            f"ConsecLoss: {self.stats.consecutive_losses}"
        )

        if self.send_telegram:
            await self._send_close_notification(close_reason, actual_pnl)

        self._position = None
        self.state = ExecutorState.MONITORING

    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================

    async def _send_trade_notification(
        self, direction: str, entry: float, sl: float, tp: float,
        volume: float, rsi: float, atr_pct: float, regime: str
    ):
        sl_pips = abs(entry - sl) / self._pip_value
        tp_pips = abs(tp - entry) / self._pip_value
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        emoji = "🟢" if direction == "BUY" else "🔴"

        msg = f"{emoji} <b>RSI v3.7 OPT - {direction}</b>\n\n"
        msg += f"├ Symbol: {self.symbol}\n"
        msg += f"├ Entry: <code>{entry:.5f}</code>\n"
        msg += f"├ SL: <code>{sl:.5f}</code> ({sl_pips:.0f} pips)\n"
        msg += f"├ TP: <code>{tp:.5f}</code> ({tp_pips:.0f} pips)\n"
        msg += f"├ R:R: 1:{rr:.1f}\n"
        msg += f"├ Lot: {volume}\n"
        msg += f"├ RSI: {rsi:.1f}\n"
        msg += f"├ ATR%: {atr_pct:.0f}\n"
        msg += f"├ Regime: {regime}\n"
        msg += f"└ ConsecLoss: {self.stats.consecutive_losses}\n"
        msg += f"\n📊 WR: {self.stats.win_rate:.1f}% | Net: ${self.stats.net_pnl:+.0f}"

        try:
            await self.send_telegram(msg)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    async def _send_close_notification(self, reason: str, pnl: float):
        emoji = "✅" if pnl > 0 else "❌"

        msg = f"{emoji} <b>Closed - {reason}</b>\n\n"
        msg += f"├ P/L: ${pnl:+.2f}\n"
        msg += f"├ Daily: ${self.stats.daily_pnl:+.2f}\n"
        msg += f"├ ConsecLoss: {self.stats.consecutive_losses}\n"
        msg += f"├ WR: {self.stats.win_rate:.1f}%\n"
        msg += f"└ Net: ${self.stats.net_pnl:+.2f}"

        try:
            await self.send_telegram(msg)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    # =========================================================================
    # CONTROL METHODS
    # =========================================================================

    def pause(self):
        self.state = ExecutorState.PAUSED
        logger.info("Executor paused")

    def resume(self):
        if self._position:
            self.state = ExecutorState.IN_POSITION
        else:
            self.state = ExecutorState.MONITORING
        logger.info("Executor resumed")

    def stop(self):
        self.state = ExecutorState.STOPPED
        logger.info("Executor stopped")

    def get_status(self) -> Dict:
        current_regime = self._regime.iloc[-1] if self._regime is not None else "UNKNOWN"

        return {
            'strategy': 'RSI v3.7 OPTIMIZED',
            'symbol': self.symbol,
            'state': self.state.value,
            'current_regime': current_regime,
            'regime_allowed': current_regime in self.ALLOWED_REGIMES,
            'consecutive_losses': self.stats.consecutive_losses,
            'has_position': self._position is not None,
            'position': {
                'direction': self._position.direction,
                'entry': self._position.entry_price,
                'sl': self._position.stop_loss,
                'tp': self._position.take_profit,
                'lot': self._position.lot_size,
                'regime': self._position.regime,
            } if self._position else None,
            'stats': {
                'signals': self.stats.total_signals,
                'trades': self.stats.trades_executed,
                'filtered': self.stats.trades_filtered,
                'wins': self.stats.trades_won,
                'losses': self.stats.trades_lost,
                'win_rate': f"{self.stats.win_rate:.1f}%",
                'net_pnl': f"${self.stats.net_pnl:+.2f}",
                'daily_pnl': f"${self.stats.daily_pnl:+.2f}",
            },
            'filters': {
                'regime_filter': self.USE_REGIME_FILTER,
                'allowed_regimes': self.ALLOWED_REGIMES,
                'consec_loss_filter': self.USE_CONSEC_LOSS_FILTER,
                'consec_loss_limit': self.CONSEC_LOSS_LIMIT,
            }
        }

    def reset(self):
        self._position = None
        self._h1_data = None
        self._rsi = None
        self._atr = None
        self._atr_pct = None
        self._regime = None
        self._circuit_breaker_triggered = False
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(start_time=datetime.now(timezone.utc))
        logger.info("Executor reset")
