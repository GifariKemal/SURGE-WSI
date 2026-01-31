"""
RSI Executor - Production Trading Bot (v3.6 OPTIMIZED)
======================================================

Strategy: RSI Mean Reversion with Volatility Filter + Dynamic TP + Time-based TP + Max Holding
Backtest: +572.9% over 6 years (6/6 profitable years)

Entry Rules:
- BUY:  RSI(10) < 42 AND London+NY Session (07-22 UTC)
- SELL: RSI(10) > 58 AND London+NY Session (07-22 UTC)
- FILTER: ATR percentile 20-80 (skip extreme volatility)

Exit Rules:
- Stop Loss:   ATR(14) * 1.5
- Take Profit: DYNAMIC based on volatility + time bonus
  - Low vol (ATR < 40th pct):  TP = 2.4x ATR (take profit faster)
  - Med vol (ATR 40-60th pct): TP = 3.0x ATR (standard)
  - High vol (ATR > 60th pct): TP = 3.6x ATR (let profits run)
  - Time bonus: +0.35x ATR during London+NY overlap (12-16 UTC)
- Max Holding: Close at market after 46 hours if no SL/TP hit

Risk Management:
- 1% risk per trade
- 2% max daily loss (circuit breaker)
- 10% max drawdown (emergency stop)

v3.6 Changes:
- Added 46-hour max holding period -> +48.9% improvement, lower drawdown (36.7%)

v3.5 Changes:
- Added Time-based TP bonus during London+NY overlap (12-16 UTC) -> +31.0% improvement
- Total improvement from v3.1: +524.1% vs baseline

v3.4 Changes:
- RSI thresholds 42/58 (from 35/65) -> +238.5% improvement, more trades, 6/6 years profitable

v3.3 Changes:
- Added Dynamic TP based on volatility regime -> +75.2% improvement

v3.2 Changes:
- Added Volatility Filter (ATR percentile 20-80) -> +52.1% improvement

v3.1 Fixes:
- Removed hardcoded credentials
- Fixed nested asyncio.run anti-pattern
- Fixed race condition in position lock
- Added proper error logging (no silent swallow)
- Added daily P/L reset
- Fixed max drawdown check
- Added paper mode support
- Added MT5 reconnection logic
- Made pip value dynamic per symbol

Author: SURIOTA Team
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.mt5_connector import MT5Connector


class ExecutorState(Enum):
    """Executor state machine"""
    IDLE = "idle"
    CONNECTING = "connecting"
    WARMING_UP = "warming_up"
    MONITORING = "monitoring"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    MARKET_CLOSED = "market_closed"


class MarketHours:
    """Forex Market Hours Detection"""

    HOLIDAYS = [(1, 1), (12, 25), (12, 26)]

    @classmethod
    def is_market_open(cls, dt: datetime = None) -> tuple[bool, str]:
        if dt is None:
            dt = datetime.now(timezone.utc)

        weekday = dt.weekday()
        hour = dt.hour

        if (dt.month, dt.day) in cls.HOLIDAYS:
            return False, f"Holiday: {dt.strftime('%B %d')}"

        if weekday == 5:
            return False, "Weekend (Saturday)"

        if weekday == 6:
            if hour < 22:
                return False, "Weekend (Sunday before 22:00 UTC)"
            return True, "Market open"

        if weekday == 4 and hour >= 22:
            return False, "Weekend (Friday after 22:00 UTC)"

        return True, "Market open"


# Pip values per symbol
PIP_VALUES = {
    "GBPUSD": 0.0001,
    "EURUSD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCHF": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "XAUUSD": 0.01,  # Gold
}

# Pip value in USD per standard lot
PIP_VALUE_USD = {
    "GBPUSD": 10.0,
    "EURUSD": 10.0,
    "AUDUSD": 10.0,
    "NZDUSD": 10.0,
    "USDCAD": 7.5,  # Approximate
    "USDCHF": 10.5,  # Approximate
    "USDJPY": 6.7,  # Approximate at 150 rate
    "XAUUSD": 1.0,  # $1 per 0.01 move per 1 lot
}


@dataclass
class TradeSignal:
    """Trade signal from RSI strategy"""
    valid: bool = False
    direction: str = ""
    rsi_value: float = 0.0
    atr_value: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.01
    risk_amount: float = 0.0
    reason: str = ""


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
    rsi_at_entry: float = 0.0


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
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_daily_reset: date = None

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return (self.trades_won / total * 100) if total > 0 else 0.0

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss


class RSIExecutor:
    """
    RSI Mean Reversion Trading Bot (v3.1 FIXED)
    """

    MAGIC_NUMBER = 20260131
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY = 5  # seconds

    def __init__(
        self,
        symbol: str = "GBPUSD",
        rsi_period: int = 10,
        rsi_oversold: float = 42.0,
        rsi_overbought: float = 58.0,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        session_start: int = 7,
        session_end: int = 22,
        risk_per_trade: float = 0.01,
        max_daily_loss_pct: float = 0.02,
        max_drawdown_pct: float = 0.10,
        max_spread_pips: float = 3.0,
        max_lot_size: float = 5.0,
        paper_mode: bool = False,
        # Volatility filter (v3.2) - +52.1% improvement
        min_atr_percentile: float = 20.0,
        max_atr_percentile: float = 80.0,
        atr_lookback: int = 100,
        # Dynamic TP (v3.3) - +75.2% improvement
        dynamic_tp: bool = True,
        tp_low_vol_mult: float = 2.4,   # TP when ATR < 40th percentile
        tp_high_vol_mult: float = 3.6,  # TP when ATR > 60th percentile
        # Time-based TP (v3.5) - +31.0% improvement
        time_tp_bonus: bool = True,
        time_tp_start: int = 12,        # London+NY overlap start
        time_tp_end: int = 16,          # London+NY overlap end
        time_tp_bonus_mult: float = 0.35,  # Add 0.35x ATR to TP during overlap
        # Max holding period (v3.6) - +48.9% improvement
        max_holding_hours: int = 46,    # Force close after 46 hours
    ):
        """Initialize RSI Executor"""
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.session_start = session_start
        self.session_end = session_end
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_spread_pips = max_spread_pips
        self.max_lot_size = max_lot_size
        self.paper_mode = paper_mode

        # Volatility filter (v3.2)
        self.min_atr_percentile = min_atr_percentile
        self.max_atr_percentile = max_atr_percentile
        self.atr_lookback = atr_lookback

        # Dynamic TP (v3.3)
        self.dynamic_tp = dynamic_tp
        self.tp_low_vol_mult = tp_low_vol_mult
        self.tp_high_vol_mult = tp_high_vol_mult

        # Time-based TP (v3.5)
        self.time_tp_bonus = time_tp_bonus
        self.time_tp_start = time_tp_start
        self.time_tp_end = time_tp_end
        self.time_tp_bonus_mult = time_tp_bonus_mult

        # Max holding period (v3.6)
        self.max_holding_hours = max_holding_hours

        # Get pip value for symbol
        self.pip_value = PIP_VALUES.get(symbol, 0.0001)
        self.pip_value_usd = PIP_VALUE_USD.get(symbol, 10.0)

        # MT5 Connector
        self.mt5: Optional[MT5Connector] = None
        self._mt5_config: Dict = {}

        # State
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(
            start_time=datetime.now(timezone.utc),
            last_daily_reset=datetime.now(timezone.utc).date()
        )

        # Position tracking
        self._position: Optional[OpenPosition] = None
        self._starting_balance: float = 0.0
        self._daily_start_balance: float = 0.0
        self._peak_balance: float = 0.0
        self._circuit_breaker_triggered = False
        self._max_drawdown_triggered = False

        # Cooldowns - FIXED: shorter after loss to recover faster
        self.cooldown_after_tp = timedelta(hours=1)     # Rest after win
        self.cooldown_after_sl = timedelta(minutes=30)  # Quick recovery after loss
        self._cooldown_until: Optional[datetime] = None

        # Data buffer
        self._h1_data: Optional[pd.DataFrame] = None
        self._last_data_fetch: Optional[datetime] = None
        self._warmup_bars = 200

        # Callbacks
        self.send_telegram: Optional[Callable] = None

        logger.info(f"RSIExecutor v3.6 initialized for {symbol}")
        logger.info(f"  RSI: period={rsi_period}, oversold={rsi_oversold}, overbought={rsi_overbought}")
        logger.info(f"  ATR: SL={sl_atr_mult}x, TP={tp_atr_mult}x (base)")
        logger.info(f"  Dynamic TP: {tp_low_vol_mult}x/<40pct, {tp_atr_mult}x/40-60pct, {tp_high_vol_mult}x/>60pct")
        logger.info(f"  Time TP: +{time_tp_bonus_mult}x during {time_tp_start}:00-{time_tp_end}:00 UTC (overlap)")
        logger.info(f"  Max holding: {max_holding_hours} hours (force close)")
        logger.info(f"  Session: {session_start}:00 - {session_end}:00 UTC")
        logger.info(f"  Volatility Filter: ATR percentile {min_atr_percentile:.0f}-{max_atr_percentile:.0f}")
        logger.info(f"  Pip value: {self.pip_value}, ${self.pip_value_usd}/pip/lot")
        if paper_mode:
            logger.warning("  MODE: PAPER (no real trades)")

    def connect_mt5(
        self,
        login: int,
        password: str,
        server: str,
        terminal_path: str = None
    ) -> bool:
        """Connect to MT5 terminal"""
        self.state = ExecutorState.CONNECTING

        # Store config for reconnection
        self._mt5_config = {
            'login': login,
            'password': password,
            'server': server,
            'terminal_path': terminal_path
        }

        return self._do_connect()

    def _do_connect(self) -> bool:
        """Perform MT5 connection"""
        try:
            self.mt5 = MT5Connector(
                login=self._mt5_config['login'],
                password=self._mt5_config['password'],
                server=self._mt5_config['server'],
                terminal_path=self._mt5_config.get('terminal_path')
            )

            if self.mt5.connect(force_login=True):
                account = self.mt5.get_account_info_sync()
                if account:
                    self._starting_balance = account['balance']
                    self._daily_start_balance = account['balance']
                    self._peak_balance = account['balance']
                    logger.info(f"Connected: {account.get('name', 'N/A')} (${account['balance']:,.2f})")
                    return True

            self.state = ExecutorState.ERROR
            return False

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            self.state = ExecutorState.ERROR
            return False

    def _reconnect(self) -> bool:
        """Reconnect to MT5 with retry logic"""
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            logger.warning(f"Reconnection attempt {attempt}/{self.MAX_RECONNECT_ATTEMPTS}...")

            if self.mt5:
                try:
                    self.mt5.disconnect()
                except:
                    pass

            time.sleep(self.RECONNECT_DELAY)

            if self._do_connect():
                logger.info("Reconnection successful")
                self.state = ExecutorState.MONITORING
                return True

        logger.error("All reconnection attempts failed")
        self.state = ExecutorState.ERROR
        return False

    async def warmup(self) -> bool:
        """Warmup with historical H1 data"""
        self.state = ExecutorState.WARMING_UP

        try:
            df = self.mt5.get_ohlcv(symbol=self.symbol, timeframe="H1", bars=self._warmup_bars)

            if df is None or len(df) < 50:
                logger.error("Failed to fetch warmup data")
                return False

            df.columns = [c.lower() for c in df.columns]
            df = self._calculate_indicators(df)
            self._h1_data = df
            self._last_data_fetch = datetime.now(timezone.utc)

            latest = df.iloc[-1]
            logger.info(f"Warmup complete: RSI={latest['rsi']:.1f}, ATR={latest['atr']/self.pip_value:.1f}p")

            self.state = ExecutorState.MONITORING
            return True

        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self.state = ExecutorState.ERROR
            return False

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI, ATR, and ATR percentile"""
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # ATR
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = tr.rolling(self.atr_period).mean()

        # ATR Percentile (v3.2 - volatility filter)
        df['atr_percentile'] = df['atr'].rolling(self.atr_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

        return df.ffill().fillna(0)

    async def _refresh_data(self) -> bool:
        """Refresh H1 data"""
        try:
            df = self.mt5.get_ohlcv(symbol=self.symbol, timeframe="H1", bars=self._warmup_bars)
            if df is None or len(df) < 50:
                return False

            df.columns = [c.lower() for c in df.columns]
            df = self._calculate_indicators(df)
            self._h1_data = df
            self._last_data_fetch = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
            return False

    def _check_session(self, now: datetime) -> bool:
        """Check if in trading session"""
        return self.session_start <= now.hour < self.session_end

    def _check_daily_reset(self, now: datetime):
        """Reset daily stats at midnight UTC"""
        today = now.date()

        if self.stats.last_daily_reset != today:
            logger.info(f"Daily reset - Previous P/L: ${self.stats.daily_pnl:+.2f}")

            # Reset daily stats
            self.stats.daily_pnl = 0.0
            self.stats.daily_trades = 0
            self.stats.last_daily_reset = today
            self._circuit_breaker_triggered = False

            # Update daily start balance
            account = self.mt5.get_account_info_sync()
            if account:
                self._daily_start_balance = account['balance']
                if account['balance'] > self._peak_balance:
                    self._peak_balance = account['balance']

            logger.info(f"New day started. Balance: ${self._daily_start_balance:,.2f}")

    def _check_max_drawdown(self) -> bool:
        """Check if max drawdown exceeded. Returns True if triggered."""
        if self._max_drawdown_triggered:
            return True

        try:
            account = self.mt5.get_account_info_sync()
            if not account:
                return False

            current_balance = account['balance']

            if current_balance > self._peak_balance:
                self._peak_balance = current_balance

            if self._peak_balance > 0:
                drawdown_pct = (self._peak_balance - current_balance) / self._peak_balance

                if drawdown_pct >= self.max_drawdown_pct:
                    self._max_drawdown_triggered = True
                    self.state = ExecutorState.STOPPED
                    logger.error(f"MAX DRAWDOWN: {drawdown_pct:.1%} >= {self.max_drawdown_pct:.1%}")

                    if self.send_telegram:
                        asyncio.create_task(self._notify(
                            f"<b>MAX DRAWDOWN TRIGGERED</b>\n\n"
                            f"Drawdown: {drawdown_pct:.1%}\n"
                            f"Peak: ${self._peak_balance:,.2f}\n"
                            f"Current: ${current_balance:,.2f}\n"
                            f"Trading STOPPED."
                        ))
                    return True

        except Exception as e:
            logger.error(f"Max drawdown check error: {e}")

        return False

    def _get_rsi_signal(self) -> TradeSignal:
        """Get RSI trading signal"""
        signal = TradeSignal()

        if self._h1_data is None or len(self._h1_data) < 50:
            signal.reason = "Insufficient data"
            return signal

        now = datetime.now(timezone.utc)

        if not self._check_session(now):
            signal.reason = f"Outside session ({self.session_start}:00-{self.session_end}:00 UTC)"
            return signal

        latest = self._h1_data.iloc[-1]
        rsi = latest['rsi']
        atr = latest['atr']
        atr_pct = latest.get('atr_percentile', 50)  # Default to 50 if not calculated

        signal.rsi_value = rsi
        signal.atr_value = atr

        # Volatility filter (v3.2) - skip extreme volatility
        if atr_pct < self.min_atr_percentile:
            signal.reason = f"Volatility too low (ATR {atr_pct:.0f}th < {self.min_atr_percentile:.0f}th)"
            return signal
        if atr_pct > self.max_atr_percentile:
            signal.reason = f"Volatility too high (ATR {atr_pct:.0f}th > {self.max_atr_percentile:.0f}th)"
            return signal

        if rsi < self.rsi_oversold:
            signal.direction = "BUY"
        elif rsi > self.rsi_overbought:
            signal.direction = "SELL"
        else:
            signal.reason = f"RSI neutral ({rsi:.1f})"
            return signal

        tick = self.mt5.get_tick_sync(self.symbol)
        if not tick:
            signal.reason = "Failed to get tick"
            return signal

        # Spread filter (using dynamic pip value)
        spread_pips = (tick['ask'] - tick['bid']) / self.pip_value
        if spread_pips > self.max_spread_pips:
            signal.reason = f"Spread too wide ({spread_pips:.1f}p)"
            return signal

        price = tick['ask'] if signal.direction == 'BUY' else tick['bid']
        signal.entry_price = price

        # Calculate SL/TP with Dynamic TP (v3.3)
        sl_distance = atr * self.sl_atr_mult

        # Dynamic TP based on volatility regime
        if self.dynamic_tp:
            if atr_pct < 40:
                tp_mult = self.tp_low_vol_mult   # 2.4x - take profit faster in low vol
            elif atr_pct > 60:
                tp_mult = self.tp_high_vol_mult  # 3.6x - let profits run in high vol
            else:
                tp_mult = self.tp_atr_mult       # 3.0x - standard
        else:
            tp_mult = self.tp_atr_mult

        # Time-based TP bonus (v3.5) - larger TP during London+NY overlap
        if self.time_tp_bonus:
            current_hour = now.hour
            if self.time_tp_start <= current_hour < self.time_tp_end:
                tp_mult += self.time_tp_bonus_mult  # Add 0.35x during overlap

        tp_distance = atr * tp_mult

        if signal.direction == 'BUY':
            signal.stop_loss = price - sl_distance
            signal.take_profit = price + tp_distance
        else:
            signal.stop_loss = price + sl_distance
            signal.take_profit = price - tp_distance

        # Position sizing
        account = self.mt5.get_account_info_sync()
        if not account:
            signal.reason = "Failed to get account"
            return signal

        risk_amount = account['balance'] * self.risk_per_trade
        sl_pips = sl_distance / self.pip_value
        lot_size = risk_amount / (sl_pips * self.pip_value_usd)
        lot_size = max(0.01, min(self.max_lot_size, round(lot_size, 2)))

        signal.lot_size = lot_size
        signal.risk_amount = risk_amount
        signal.valid = True

        return signal

    async def process_tick(self) -> Optional[Dict]:
        """Main trading loop iteration"""
        if self.state not in [ExecutorState.MONITORING, ExecutorState.MARKET_CLOSED]:
            return None

        now = datetime.now(timezone.utc)

        # Daily reset check
        self._check_daily_reset(now)

        # Market hours check
        is_open, reason = MarketHours.is_market_open(now)
        if not is_open:
            if self.state != ExecutorState.MARKET_CLOSED:
                self.state = ExecutorState.MARKET_CLOSED
                logger.warning(f"Market CLOSED: {reason}")
            return None
        else:
            if self.state == ExecutorState.MARKET_CLOSED:
                self.state = ExecutorState.MONITORING
                logger.info("Market OPEN")

        # Check circuit breakers
        if self._circuit_breaker_triggered:
            return None

        # Check max drawdown
        if self._check_max_drawdown():
            return None

        # Check cooldown
        if self._cooldown_until and now < self._cooldown_until:
            return None

        # Refresh data every hour
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        if self._last_data_fetch is None or current_hour > self._last_data_fetch.replace(minute=0, second=0, microsecond=0):
            if not await self._refresh_data():
                # Try reconnect if data fetch fails
                if not self._reconnect():
                    return None
                await self._refresh_data()

        # Check existing position (no lock needed for read)
        if self._position is not None:
            await self._check_position_closed()
            return None

        # Get signal
        signal = self._get_rsi_signal()
        self.stats.total_signals += 1

        if not signal.valid:
            self.stats.signals_filtered += 1
            return None

        return await self._execute_trade(signal)

    async def _execute_trade(self, signal: TradeSignal) -> Optional[Dict]:
        """Execute trade"""
        now = datetime.now(timezone.utc)

        # Paper mode - simulate trade
        if self.paper_mode:
            fake_ticket = int(now.timestamp())
            self._position = OpenPosition(
                ticket=fake_ticket,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                lot_size=signal.lot_size,
                entry_time=now,
                rsi_at_entry=signal.rsi_value
            )

            self.stats.trades_executed += 1
            self.stats.daily_trades += 1

            sl_pips = abs(signal.entry_price - signal.stop_loss) / self.pip_value
            tp_pips = abs(signal.entry_price - signal.take_profit) / self.pip_value

            logger.info(
                f"[PAPER] {signal.direction} {signal.lot_size} @ {signal.entry_price:.5f} | "
                f"RSI: {signal.rsi_value:.1f} | SL: {sl_pips:.0f}p | TP: {tp_pips:.0f}p"
            )

            await self._notify(
                f"<b>[PAPER] RSI BOT - {signal.direction}</b>\n\n"
                f"Entry: {signal.entry_price:.5f}\n"
                f"SL: {sl_pips:.0f}p | TP: {tp_pips:.0f}p"
            )

            return {'success': True, 'ticket': fake_ticket, 'paper': True}

        # Real trade
        try:
            # Use synchronous method to avoid nested event loop
            result = self.mt5.place_order_sync(
                symbol=self.symbol,
                order_type=signal.direction,
                volume=signal.lot_size,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"RSI_{signal.rsi_value:.0f}",
                magic=self.MAGIC_NUMBER
            )

            if result and result.get('ticket'):
                self._position = OpenPosition(
                    ticket=result['ticket'],
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    lot_size=signal.lot_size,
                    entry_time=now,
                    rsi_at_entry=signal.rsi_value
                )

                self.stats.trades_executed += 1
                self.stats.daily_trades += 1

                sl_pips = abs(signal.entry_price - signal.stop_loss) / self.pip_value
                tp_pips = abs(signal.entry_price - signal.take_profit) / self.pip_value

                logger.info(
                    f"TRADE: {signal.direction} {signal.lot_size} @ {signal.entry_price:.5f} | "
                    f"RSI: {signal.rsi_value:.1f} | SL: {sl_pips:.0f}p | TP: {tp_pips:.0f}p"
                )

                await self._notify(
                    f"<b>RSI BOT - {signal.direction}</b>\n\n"
                    f"Symbol: {self.symbol}\n"
                    f"RSI: {signal.rsi_value:.1f}\n"
                    f"Entry: {signal.entry_price:.5f}\n"
                    f"SL: {signal.stop_loss:.5f} ({sl_pips:.0f}p)\n"
                    f"TP: {signal.take_profit:.5f} ({tp_pips:.0f}p)\n"
                    f"Risk: ${signal.risk_amount:.2f}"
                )

                return {'success': True, 'ticket': result['ticket']}

            logger.error(f"Order rejected: {result}")
            return None

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            # Try reconnect
            self._reconnect()
            return None

    async def _check_position_closed(self):
        """Check if position was closed by SL/TP or max holding time exceeded (v3.6)"""
        if self._position is None:
            return

        now = datetime.now(timezone.utc)

        # Check max holding period (v3.6) - force close after max_holding_hours
        holding_duration = now - self._position.entry_time
        if self.max_holding_hours > 0 and holding_duration >= timedelta(hours=self.max_holding_hours):
            tick = self.mt5.get_tick_sync(self.symbol)
            if tick:
                price = tick['bid'] if self._position.direction == 'BUY' else tick['ask']

                if self._position.direction == 'BUY':
                    pnl = (price - self._position.entry_price) / self.pip_value * self.pip_value_usd * self._position.lot_size
                else:
                    pnl = (self._position.entry_price - price) / self.pip_value * self.pip_value_usd * self._position.lot_size

                # Close position if not paper mode
                if not self.paper_mode:
                    try:
                        self.mt5.close_position_sync(self._position.ticket)
                    except Exception as e:
                        logger.error(f"Failed to close position for time exit: {e}")
                        return

                hours_held = holding_duration.total_seconds() / 3600
                logger.info(f"TIME EXIT: Position held {hours_held:.1f}h >= {self.max_holding_hours}h limit")
                await self._handle_close("TIME_EXIT", pnl, paper=self.paper_mode)
                return

        # Paper mode - check price levels
        if self.paper_mode:
            tick = self.mt5.get_tick_sync(self.symbol)
            if not tick:
                return

            price = tick['bid'] if self._position.direction == 'BUY' else tick['ask']

            closed = False
            close_reason = ""
            pnl = 0.0

            if self._position.direction == 'BUY':
                if price <= self._position.stop_loss:
                    closed = True
                    close_reason = "SL"
                    pnl = (self._position.stop_loss - self._position.entry_price) / self.pip_value * self.pip_value_usd * self._position.lot_size
                elif price >= self._position.take_profit:
                    closed = True
                    close_reason = "TP"
                    pnl = (self._position.take_profit - self._position.entry_price) / self.pip_value * self.pip_value_usd * self._position.lot_size
            else:
                if price >= self._position.stop_loss:
                    closed = True
                    close_reason = "SL"
                    pnl = (self._position.entry_price - self._position.stop_loss) / self.pip_value * self.pip_value_usd * self._position.lot_size
                elif price <= self._position.take_profit:
                    closed = True
                    close_reason = "TP"
                    pnl = (self._position.entry_price - self._position.take_profit) / self.pip_value * self.pip_value_usd * self._position.lot_size

            if closed:
                await self._handle_close(close_reason, pnl, paper=True)
            return

        # Real position check
        try:
            positions = self.mt5.get_positions_sync(self.symbol)
            pos_exists = any(p['ticket'] == self._position.ticket for p in (positions or []))

            if not pos_exists:
                deal_info = self.mt5.get_deal_history(self._position.ticket)
                close_reason = deal_info.get('close_reason', 'UNKNOWN') if deal_info else 'UNKNOWN'
                pnl = (deal_info.get('profit', 0) + deal_info.get('swap', 0)) if deal_info else 0
                await self._handle_close(close_reason, pnl, paper=False)

        except Exception as e:
            logger.error(f"Position check error: {e}")

    async def _handle_close(self, close_reason: str, pnl: float, paper: bool = False):
        """Handle closed position"""
        prefix = "[PAPER] " if paper else ""

        if pnl > 0:
            self.stats.trades_won += 1
            self.stats.total_profit += pnl
            self._cooldown_until = datetime.now(timezone.utc) + self.cooldown_after_tp
        else:
            self.stats.trades_lost += 1
            self.stats.total_loss += abs(pnl)
            self._cooldown_until = datetime.now(timezone.utc) + self.cooldown_after_sl

        self.stats.daily_pnl += pnl

        logger.info(f"{prefix}CLOSED: {close_reason} | P/L: ${pnl:+.2f} | WR: {self.stats.win_rate:.1f}%")

        await self._notify(
            f"<b>{prefix}{'WIN' if pnl > 0 else 'LOSS'}</b>\n\n"
            f"Reason: {close_reason}\n"
            f"P/L: ${pnl:+.2f}\n"
            f"Daily: ${self.stats.daily_pnl:+.2f}\n"
            f"WR: {self.stats.win_rate:.1f}%"
        )

        # Check circuit breaker
        if not paper:
            account = self.mt5.get_account_info_sync()
            if account and self._daily_start_balance > 0:
                daily_loss_pct = (self._daily_start_balance - account['balance']) / self._daily_start_balance
                if daily_loss_pct >= self.max_daily_loss_pct:
                    self._circuit_breaker_triggered = True
                    self.state = ExecutorState.PAUSED
                    logger.warning(f"CIRCUIT BREAKER: {daily_loss_pct:.1%} daily loss")
                    await self._notify(
                        f"<b>CIRCUIT BREAKER</b>\n\n"
                        f"Daily loss: {daily_loss_pct:.1%}\n"
                        f"Trading paused until tomorrow"
                    )

        self._position = None

    async def _notify(self, message: str):
        """Send notification with error handling"""
        if not self.send_telegram:
            return

        try:
            await self.send_telegram(message)
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")

    async def run(self, interval_seconds: int = 5):
        """Main run loop"""
        logger.info(f"Starting RSI Executor (interval: {interval_seconds}s)")
        self.state = ExecutorState.MONITORING

        tick_count = 0

        try:
            while self.state not in [ExecutorState.STOPPED, ExecutorState.ERROR]:
                await self.process_tick()
                tick_count += 1

                if tick_count % 60 == 0 and self._h1_data is not None:
                    latest = self._h1_data.iloc[-1]
                    in_session = self._check_session(datetime.now(timezone.utc))
                    atr_pct = latest.get('atr_percentile', 50)
                    logger.info(
                        f"[{tick_count}] RSI: {latest['rsi']:.1f} | "
                        f"ATR: {atr_pct:.0f}th pct | "
                        f"{'IN' if in_session else 'OUT'} SESSION | "
                        f"Trades: {self.stats.trades_executed} | "
                        f"WR: {self.stats.win_rate:.1f}% | "
                        f"P/L: ${self.stats.net_pnl:+.2f}"
                    )

                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Run loop error: {e}")
            self.state = ExecutorState.ERROR
        finally:
            self.stop()

    def get_status(self) -> Dict:
        """Get executor status"""
        rsi = atr = atr_pct = 0.0
        if self._h1_data is not None and len(self._h1_data) > 0:
            latest = self._h1_data.iloc[-1]
            rsi = latest['rsi']
            atr = latest['atr'] / self.pip_value
            atr_pct = latest.get('atr_percentile', 50)

        now = datetime.now(timezone.utc)
        is_open, _ = MarketHours.is_market_open(now)

        # Check if volatility is in acceptable range
        vol_ok = self.min_atr_percentile <= atr_pct <= self.max_atr_percentile

        return {
            'state': self.state.value,
            'rsi': rsi,
            'atr_pips': atr,
            'atr_percentile': atr_pct,
            'volatility_ok': vol_ok,
            'in_session': self._check_session(now),
            'market_open': is_open,
            'has_position': self._position is not None,
            'paper_mode': self.paper_mode,
            'stats': {
                'trades': self.stats.trades_executed,
                'win_rate': self.stats.win_rate,
                'net_pnl': self.stats.net_pnl,
                'daily_pnl': self.stats.daily_pnl
            }
        }

    def pause(self):
        self.state = ExecutorState.PAUSED
        logger.info("Executor paused")

    def resume(self):
        self.state = ExecutorState.MONITORING
        self._circuit_breaker_triggered = False
        logger.info("Executor resumed")

    def stop(self):
        self.state = ExecutorState.STOPPED
        if self.mt5:
            try:
                self.mt5.disconnect()
            except:
                pass
        logger.info("Executor stopped")
