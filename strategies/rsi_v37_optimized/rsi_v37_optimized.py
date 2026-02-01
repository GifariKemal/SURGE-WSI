"""
RSI v3.7 OPTIMIZED - Complete Live Trading Bot
===============================================

All-in-one file: Executor + Launcher

Strategy: RSI Mean Reversion with SIDEWAYS Regime + ConsecLoss3 filters

Backtest Results (Oct 2024 - Jan 2026):
- Return: +72.7%
- Drawdown: 14.4%
- Win Rate: 37.6%
- Losing Months: 2/16 (reduced from 3)

Improvements over baseline:
- Oct 2024: FIXED (was -$668, now +$47)
- Drawdown: 19.7% -> 14.4%
- Return: +71.9% -> +72.7%

Usage:
    python rsi_v37_optimized.py

Author: SURIOTA Team
Version: 3.7 OPTIMIZED
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

# Logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# MetaTrader 5
import MetaTrader5 as mt5


# =============================================================================
# CONFIGURATION - Use environment variables or edit defaults below
# =============================================================================

# MT5 Credentials (use environment variables for security)
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '61045904'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')  # Set via: set MT5_PASSWORD=your_password
MT5_SERVER = os.getenv('MT5_SERVER', 'FinexBisnisSolusi-Demo')
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe")

SYMBOL = "GBPUSD"
MAGIC_NUMBER = 20250201  # Date format: 2025-02-01

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


# =============================================================================
# TELEGRAM FORMATTER (SURIOTA Style)
# =============================================================================

class TelegramFormatter:
    """Message formatter with SURIOTA styling"""

    # Emojis
    EAGLE = "🦅"        # SURIOTA brand
    ROCKET = "🚀"       # Big profit
    CHECK = "✅"        # Profit/success
    CROSS = "❌"        # Loss/failure
    WARNING = "⚠️"      # Warning
    BELL = "🔔"         # Alert
    UP = "📈"           # BUY
    DOWN = "📉"         # SELL
    TARGET = "🎯"       # Target
    SHIELD = "🛡"       # Stop loss
    CHART = "📊"        # Stats
    MONEY = "💰"        # Money
    GEAR = "⚙️"         # Config
    CLOCK = "⏰"        # Time
    GREEN = "🟢"        # Active
    RED = "🔴"          # Inactive
    YELLOW = "🟡"       # Warning
    BRANCH = "├"
    LAST = "└"

    @classmethod
    def startup(cls, login: int, balance: float, server: str, symbol: str) -> str:
        """Bot startup message"""
        msg = f"{cls.EAGLE} <b>RSI v3.7 OPTIMIZED Started</b>\n\n"
        msg += f"{cls.BRANCH} Account: {login}\n"
        msg += f"{cls.BRANCH} Balance: ${balance:,.2f}\n"
        msg += f"{cls.BRANCH} Server: {server}\n"
        msg += f"{cls.BRANCH} Symbol: {symbol}\n"
        msg += f"{cls.LAST} Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
        msg += f"\n{cls.GEAR} Filters: SIDEWAYS + ConsecLoss3"
        return msg

    @classmethod
    def trade_executed(cls, direction: str, symbol: str, entry: float, sl: float,
                       tp: float, lot: float, ticket: int, rsi: float, regime: str) -> str:
        """Trade execution message"""
        pip_mult = 10000
        sl_pips = abs(entry - sl) * pip_mult
        tp_pips = abs(tp - entry) * pip_mult
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        dir_emoji = cls.GREEN if direction == "BUY" else cls.RED

        msg = f"{cls.TARGET} <b>TRADE EXECUTED</b>\n\n"
        msg += f"{dir_emoji} <b>{symbol}</b> • <b>{direction}</b>\n"
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f"SL:    {sl:.5f} ({sl_pips:.0f}p)\n"
        msg += f"TP:    {tp:.5f} ({tp_pips:.0f}p)\n"
        msg += f"Lot:   {lot:.2f}\n"
        msg += "</pre>"
        msg += f"\n{cls.CHART} RSI: {rsi:.1f} | Regime: {regime}\n"
        msg += f"R:R 1:{rr:.1f} • <code>#{ticket}</code>"
        return msg

    @classmethod
    def position_closed(cls, direction: str, symbol: str, entry: float,
                        exit_price: float, pnl: float, reason: str,
                        consec_losses: int, daily_pnl: float) -> str:
        """Position close message"""
        pips = (exit_price - entry) / 0.0001 if direction == "BUY" else (entry - exit_price) / 0.0001

        if pnl >= 50:
            emoji = cls.ROCKET
        elif pnl > 0:
            emoji = cls.CHECK
        else:
            emoji = cls.CROSS

        msg = f"{emoji} <b>POSITION CLOSED</b>\n\n"
        msg += f"<b>{symbol}</b> • {direction}\n"
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f"Exit:  {exit_price:.5f}\n"
        msg += f"Pips:  {pips:+.1f}\n"
        msg += f"P/L:   ${pnl:+.2f}\n"
        msg += "</pre>"
        msg += f"Result: <b>{reason}</b>\n"
        msg += f"\n{cls.CHART} ConsecLoss: {consec_losses} | Daily: ${daily_pnl:+.2f}"
        return msg

    @classmethod
    def circuit_breaker(cls, loss_pct: float, daily_pnl: float, balance: float) -> str:
        """Circuit breaker alert"""
        msg = f"{cls.WARNING} <b>CIRCUIT BREAKER</b>\n\n"
        msg += f"{cls.BRANCH} Daily Loss: {loss_pct:.2%}\n"
        msg += f"{cls.BRANCH} P/L Today: ${daily_pnl:+.2f}\n"
        msg += f"{cls.LAST} Balance: ${balance:,.2f}\n"
        msg += f"\n{cls.RED} Trading PAUSED until tomorrow"
        return msg

    @classmethod
    def cooldown_triggered(cls, consec_losses: int, cooldown_hours: int) -> str:
        """Consecutive loss cooldown message"""
        msg = f"{cls.YELLOW} <b>COOLDOWN TRIGGERED</b>\n\n"
        msg += f"{cls.BRANCH} Consecutive Losses: {consec_losses}\n"
        msg += f"{cls.LAST} Pause Duration: {cooldown_hours}h\n"
        msg += f"\n{cls.CLOCK} Trading will resume automatically"
        return msg

    @classmethod
    def cooldown_expired(cls) -> str:
        """Cooldown expired message"""
        return f"{cls.GREEN} <b>Cooldown Expired</b>\nTrading resumed"

    @classmethod
    def status(cls, state: str, regime: str, consec_losses: int, cooldown: str,
               trades: int, wins: int, losses: int, net_pnl: float, daily_pnl: float) -> str:
        """Status report"""
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        state_emoji = cls.GREEN if state == "monitoring" else cls.YELLOW

        msg = f"{cls.CHART} <b>RSI v3.7 Status</b>\n\n"
        msg += f"{cls.BRANCH} State: {state_emoji} {state.upper()}\n"
        msg += f"{cls.BRANCH} Regime: {regime}\n"
        msg += f"{cls.BRANCH} ConsecLoss: {consec_losses}/3\n"
        if cooldown:
            msg += f"{cls.BRANCH} Cooldown: {cooldown}\n"
        msg += f"\n{cls.TARGET} <b>Stats</b>\n"
        msg += f"{cls.BRANCH} Trades: {trades}\n"
        msg += f"{cls.BRANCH} Win Rate: {win_rate:.1f}%\n"
        msg += f"{cls.BRANCH} Net P/L: ${net_pnl:+.2f}\n"
        msg += f"{cls.LAST} Daily P/L: ${daily_pnl:+.2f}"
        return msg

    @classmethod
    def daily_summary(cls, date: str, trades: int, wins: int, losses: int,
                      net_pnl: float, balance: float) -> str:
        """Daily summary"""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        emoji = cls.ROCKET if net_pnl >= 50 else (cls.CHECK if net_pnl > 0 else cls.CROSS)

        msg = f"{emoji} <b>Daily Summary - {date}</b>\n\n"
        msg += f"{cls.TARGET} <b>Trades</b>\n"
        msg += f"{cls.BRANCH} Total: {trades}\n"
        msg += f"{cls.BRANCH} Winners: {wins} ({win_rate:.0f}%)\n"
        msg += f"{cls.LAST} Losers: {losses}\n"
        msg += f"\n{cls.MONEY} Net P/L: <b>${net_pnl:+.2f}</b>\n"
        msg += f"Balance: ${balance:,.2f}"
        return msg

    @classmethod
    def error(cls, error_msg: str) -> str:
        """Error message"""
        return f"{cls.CROSS} <b>ERROR</b>\n<code>{error_msg}</code>"


# =============================================================================
# EXECUTOR CLASS
# =============================================================================

class ExecutorState(Enum):
    IDLE = "idle"
    WARMING_UP = "warming_up"
    MONITORING = "monitoring"
    IN_POSITION = "in_position"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


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
    consec_loss_pause_until: Optional[datetime] = None  # Cooldown timestamp
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


class RSIv37Optimized:
    """
    RSI Mean Reversion v3.7 OPTIMIZED

    Filters:
    1. SIDEWAYS Regime Only
    2. Consecutive Loss Pause (after 3 losses)
    """

    # Strategy Parameters (DO NOT MODIFY)
    RSI_PERIOD = 10
    RSI_OVERSOLD = 42
    RSI_OVERBOUGHT = 58

    ATR_PERIOD = 14
    ATR_LOOKBACK = 100
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    TIME_TP_BONUS = 0.35

    TRADING_START_HOUR = 7
    TRADING_END_HOUR = 22
    SKIP_HOURS = [12]
    TP_BONUS_START = 12
    TP_BONUS_END = 16

    MAX_HOLDING_HOURS = 46
    RISK_PER_TRADE = 0.01
    MAX_DAILY_LOSS_PCT = 0.03

    # Optimization Filters
    USE_REGIME_FILTER = True
    ALLOWED_REGIMES = ["SIDEWAYS"]
    USE_CONSEC_LOSS_FILTER = True
    CONSEC_LOSS_LIMIT = 3
    CONSEC_LOSS_COOLDOWN_HOURS = 2  # Hours to pause after consecutive losses

    def __init__(self, symbol: str = "GBPUSD", magic_number: int = 20250201):
        self.symbol = symbol
        self.magic_number = magic_number
        self.warmup_bars = 150

        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(
            start_time=datetime.now(timezone.utc),
            last_daily_reset=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        )

        self._position: Optional[OpenPosition] = None
        self._position_lock = asyncio.Lock()

        self._h1_data: Optional[pd.DataFrame] = None
        self._current_bar_idx: int = 0

        self._rsi: Optional[pd.Series] = None
        self._atr: Optional[pd.Series] = None
        self._atr_pct: Optional[pd.Series] = None
        self._regime: Optional[pd.Series] = None

        self._circuit_breaker_triggered = False
        self._starting_balance: float = 0.0

        # Notification flags
        self._notify_cooldown_triggered = False
        self._notify_cooldown_expired = False
        self._notify_circuit_breaker = False

        self._contract_size: float = 100000.0
        self._pip_value: float = 0.0001
        self._digits: int = 5

        logger.info(f"RSI v3.7 OPTIMIZED initialized: {symbol}")
        logger.info(f"Filters: Regime={self.ALLOWED_REGIMES}, ConsecLoss={self.CONSEC_LOSS_LIMIT}")

    # Indicators
    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.RSI_PERIOD).mean()
        rs = gain / (loss + 1e-10)
        return (100 - (100 / (1 + rs))).fillna(50)

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        return pd.Series(tr, index=close.index).rolling(self.ATR_PERIOD).mean()

    def _calculate_atr_percentile(self, atr: pd.Series) -> pd.Series:
        def pct(x):
            if len(x) <= 1: return 50.0
            return ((x[:-1] < x[-1]).sum() / (len(x) - 1)) * 100
        return atr.rolling(self.ATR_LOOKBACK).apply(pct, raw=True)

    def _calculate_regime(self, close: pd.Series) -> pd.Series:
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        slope = (sma_20 / sma_20.shift(10) - 1) * 100
        conditions = [(sma_20 > sma_50) & (slope > 0.5), (sma_20 < sma_50) & (slope < -0.5)]
        return pd.Series(np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS'), index=close.index)

    def _update_indicators(self, df: pd.DataFrame):
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        self._rsi = self._calculate_rsi(close)
        self._atr = self._calculate_atr(high, low, close)
        self._atr_pct = self._calculate_atr_percentile(self._atr)
        self._regime = self._calculate_regime(close)

    # Filters
    def _is_trading_hour(self, dt: datetime) -> bool:
        if dt.weekday() >= 5: return False
        if dt.hour < self.TRADING_START_HOUR or dt.hour >= self.TRADING_END_HOUR: return False
        if dt.hour in self.SKIP_HOURS: return False
        return True

    def _get_tp_multiplier(self, atr_pct: float, hour: int) -> float:
        base = self.TP_LOW if atr_pct < 40 else (self.TP_HIGH if atr_pct > 60 else self.TP_MED)
        return base + self.TIME_TP_BONUS if self.TP_BONUS_START <= hour < self.TP_BONUS_END else base

    def _check_signal(self, idx: int, dt: datetime) -> Optional[int]:
        if not self._is_trading_hour(dt): return None

        rsi = self._rsi.iloc[idx]
        atr_pct = self._atr_pct.iloc[idx]
        regime = self._regime.iloc[idx]

        if pd.isna(rsi) or pd.isna(atr_pct): return None
        if atr_pct < self.MIN_ATR_PCT or atr_pct > self.MAX_ATR_PCT: return None

        # Regime filter
        if self.USE_REGIME_FILTER and regime not in self.ALLOWED_REGIMES:
            self.stats.trades_filtered += 1
            return None

        # Consecutive loss filter with cooldown
        if self.USE_CONSEC_LOSS_FILTER:
            now = datetime.now(timezone.utc)
            # Check if in cooldown period
            if self.stats.consec_loss_pause_until and now < self.stats.consec_loss_pause_until:
                self.stats.trades_filtered += 1
                remaining = (self.stats.consec_loss_pause_until - now).total_seconds() / 60
                logger.debug(f"Cooldown active: {remaining:.0f}min remaining")
                return None
            # Cooldown expired - reset counter and allow trading
            if self.stats.consec_loss_pause_until and now >= self.stats.consec_loss_pause_until:
                logger.info(f"Cooldown expired. Resetting consecutive losses from {self.stats.consecutive_losses}")
                self.stats.consecutive_losses = 0
                self.stats.consec_loss_pause_until = None
                self._notify_cooldown_expired = True  # Flag for notification
            # Check if consecutive losses reached limit
            if self.stats.consecutive_losses >= self.CONSEC_LOSS_LIMIT:
                self.stats.trades_filtered += 1
                self.stats.consec_loss_pause_until = now + timedelta(hours=self.CONSEC_LOSS_COOLDOWN_HOURS)
                logger.warning(f"ConsecLoss={self.stats.consecutive_losses} - Pausing for {self.CONSEC_LOSS_COOLDOWN_HOURS}h")
                self._notify_cooldown_triggered = True  # Flag for notification
                return None

        if rsi < self.RSI_OVERSOLD: return 1
        if rsi > self.RSI_OVERBOUGHT: return -1
        return None

    def _calculate_lot_size(self, balance: float, entry: float, sl: float) -> Optional[float]:
        risk = balance * self.RISK_PER_TRADE
        sl_dist = abs(entry - sl)
        if sl_dist <= 0: return None
        lot = risk / (sl_dist * self._contract_size)
        return max(0.01, min(10.0, round(lot, 2)))

    async def warmup(self, df: pd.DataFrame = None) -> bool:
        self.state = ExecutorState.WARMING_UP
        try:
            if df is None:
                df = get_ohlcv_sync(self.symbol, "H1", self.warmup_bars + 50)
            if df is None or len(df) < self.warmup_bars:
                self.state = ExecutorState.ERROR
                return False
            self._h1_data = df.copy()
            self._update_indicators(self._h1_data)
            self._current_bar_idx = len(self._h1_data) - 1
            regime = self._regime.iloc[-1] if self._regime is not None else "?"
            logger.info(f"Warmup OK: {len(self._h1_data)} bars, Regime: {regime}")
            self.state = ExecutorState.MONITORING
            return True
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self.state = ExecutorState.ERROR
            return False

    async def on_new_bar(self, bar_time: datetime, balance: float) -> Optional[Dict]:
        if self.state not in [ExecutorState.MONITORING, ExecutorState.IN_POSITION]:
            return None

        now = datetime.now(timezone.utc)
        if self._starting_balance <= 0:
            self._starting_balance = balance

        # Daily reset
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.stats.last_daily_reset is None or self.stats.last_daily_reset < today:
            logger.info(f"Daily reset. Yesterday: ${self.stats.daily_pnl:+.2f}")
            self.stats.daily_pnl = 0
            self.stats.daily_trades = 0
            self.stats.last_daily_reset = today
            self._starting_balance = balance
            self._circuit_breaker_triggered = False

        # Circuit breaker
        if self._starting_balance > 0:
            loss = (self._starting_balance - balance) / self._starting_balance
            if loss >= self.MAX_DAILY_LOSS_PCT:
                if not self._circuit_breaker_triggered:
                    self._circuit_breaker_triggered = True
                    self._notify_circuit_breaker = True  # Flag for notification
                    self.state = ExecutorState.PAUSED
                    logger.warning(f"CIRCUIT BREAKER: {loss:.2%}")
                return None

        # Process pending notifications
        if self._notify_cooldown_triggered:
            self._notify_cooldown_triggered = False
            await notify_cooldown_triggered(self.stats.consecutive_losses, self.CONSEC_LOSS_COOLDOWN_HOURS)
        if self._notify_cooldown_expired:
            self._notify_cooldown_expired = False
            await notify_cooldown_expired()
        if self._notify_circuit_breaker:
            self._notify_circuit_breaker = False
            await notify_circuit_breaker(loss if self._starting_balance > 0 else 0, self.stats.daily_pnl, balance)

        # Update data
        new_data = get_ohlcv_sync(self.symbol, "H1", self.warmup_bars + 50)
        if new_data is not None:
            self._h1_data = new_data
            self._update_indicators(self._h1_data)
            self._current_bar_idx = len(self._h1_data) - 1

        if self._h1_data is None or self._rsi is None:
            return None

        # Manage position
        async with self._position_lock:
            if self._position is not None:
                return await self._manage_position(now)

        # Check signal
        signal = self._check_signal(self._current_bar_idx, bar_time)
        if signal is None:
            return None

        self.stats.total_signals += 1

        close_col = 'close' if 'close' in self._h1_data.columns else 'Close'
        entry = self._h1_data[close_col].iloc[-1]
        atr = self._atr.iloc[-1]
        atr_pct = self._atr_pct.iloc[-1]
        rsi = self._rsi.iloc[-1]
        regime = self._regime.iloc[-1]

        if pd.isna(atr) or atr <= 0:
            atr = entry * 0.002

        direction = 'BUY' if signal == 1 else 'SELL'
        tp_mult = self._get_tp_multiplier(atr_pct, bar_time.hour)

        if direction == 'BUY':
            sl = entry - atr * self.SL_MULT
            tp = entry + atr * tp_mult
        else:
            sl = entry + atr * self.SL_MULT
            tp = entry - atr * tp_mult

        lot = self._calculate_lot_size(balance, entry, sl)
        if lot is None:
            return None

        return await self._execute_trade(direction, entry, sl, tp, lot, rsi, atr_pct, regime, bar_time)

    async def _execute_trade(self, direction, entry, sl, tp, lot, rsi, atr_pct, regime, bar_time):
        result = place_order_sync(self.symbol, direction, lot, sl, tp, f"RSI37O_{direction[0]}", self.magic_number)

        if result and result.get('ticket'):
            async with self._position_lock:
                self._position = OpenPosition(
                    ticket=result['ticket'], direction=direction, entry_price=entry,
                    stop_loss=sl, take_profit=tp, lot_size=lot,
                    entry_time=datetime.now(timezone.utc), entry_bar_idx=self._current_bar_idx,
                    rsi_entry=rsi, atr_pct=atr_pct, regime=regime
                )
                self.state = ExecutorState.IN_POSITION

            self.stats.trades_executed += 1
            self.stats.daily_trades += 1

            logger.info(f"TRADE: {direction} {lot} @ {entry:.5f} SL:{sl:.5f} TP:{tp:.5f} RSI:{rsi:.1f} [{regime}]")

            # Send Telegram notification
            await notify_trade(direction, entry, sl, tp, lot, result['ticket'], rsi, regime)

            return {'success': True, 'direction': direction, 'entry': entry, 'ticket': result['ticket']}

        return None

    async def _manage_position(self, now: datetime):
        if self._position is None:
            return None

        # Max holding
        hours = (now - self._position.entry_time).total_seconds() / 3600
        if hours >= self.MAX_HOLDING_HOURS:
            logger.info("Max holding - closing")
            close_position_sync(self._position.ticket)
            await self._handle_closed("TIMEOUT")
            return None

        # Check if still open
        positions = get_positions_sync()
        exists = any(p['ticket'] == self._position.ticket for p in positions)
        if not exists:
            await self._handle_closed("SL/TP")

        return None

    async def _handle_closed(self, reason: str):
        if self._position is None:
            return

        # Save position data before clearing
        pos_direction = self._position.direction
        pos_entry = self._position.entry_price
        pos_ticket = self._position.ticket

        # Get exit price from last deal
        pnl = get_deal_pnl_sync(pos_ticket)
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(now - timedelta(days=7), now, position=pos_ticket)
        exit_price = deals[-1].price if deals and len(deals) >= 2 else pos_entry

        if pnl > 0:
            self.stats.trades_won += 1
            self.stats.total_profit += pnl
            self.stats.consecutive_losses = 0
        else:
            self.stats.trades_lost += 1
            self.stats.total_loss += abs(pnl)
            self.stats.consecutive_losses += 1

        self.stats.daily_pnl += pnl

        logger.info(f"CLOSED #{pos_ticket}: {reason} P/L=${pnl:+.2f} ConsecLoss={self.stats.consecutive_losses}")

        # Send Telegram notification
        await notify_close(pos_direction, pos_entry, exit_price, pnl, reason,
                          self.stats.consecutive_losses, self.stats.daily_pnl)

        self._position = None
        self.state = ExecutorState.MONITORING

    def get_status(self) -> Dict:
        regime = self._regime.iloc[-1] if self._regime is not None else "?"
        now = datetime.now(timezone.utc)
        cooldown_active = self.stats.consec_loss_pause_until and now < self.stats.consec_loss_pause_until
        cooldown_remaining = ""
        if cooldown_active:
            mins = (self.stats.consec_loss_pause_until - now).total_seconds() / 60
            cooldown_remaining = f"{mins:.0f}min"
        return {
            'state': self.state.value,
            'regime': regime,
            'regime_ok': regime in self.ALLOWED_REGIMES,
            'consec_losses': self.stats.consecutive_losses,
            'cooldown': cooldown_remaining if cooldown_active else None,
            'position': self._position is not None,
            'trades': self.stats.trades_executed,
            'wins': self.stats.trades_won,
            'losses': self.stats.trades_lost,
            'filtered': self.stats.trades_filtered,
            'win_rate': f"{self.stats.win_rate:.1f}%",
            'net_pnl': f"${self.stats.net_pnl:+.2f}",
            'daily_pnl': f"${self.stats.daily_pnl:+.2f}",
        }


# =============================================================================
# MT5 FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration before starting"""
    errors = []
    if not MT5_PASSWORD:
        errors.append("MT5_PASSWORD not set. Use: set MT5_PASSWORD=your_password")
    if not os.path.exists(MT5_PATH):
        errors.append(f"MT5 path not found: {MT5_PATH}")
    if errors:
        for e in errors:
            logger.error(e)
        return False
    return True


def connect_mt5():
    if not mt5.initialize(path=MT5_PATH):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        return False
    acc = mt5.account_info()
    logger.info(f"Connected: {acc.login} | Balance: ${acc.balance:,.2f}")
    return True


def get_ohlcv_sync(symbol: str, timeframe: str, count: int):
    tf_map = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
    rates = mt5.copy_rates_from_pos(symbol, tf_map.get(timeframe, mt5.TIMEFRAME_H1), 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df


def get_positions_sync():
    positions = mt5.positions_get()
    if not positions:
        return []
    return [{'ticket': p.ticket, 'type': p.type, 'volume': p.volume, 'price_open': p.price_open,
             'sl': p.sl, 'tp': p.tp, 'profit': p.profit, 'magic': p.magic, 'time': p.time} for p in positions]


def place_order_sync(symbol, direction, volume, sl, tp, comment, magic):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None

    price = tick.ask if direction == 'BUY' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
        "type": order_type, "price": price, "sl": sl, "tp": tp,
        "deviation": 20, "magic": magic, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {'ticket': result.order, 'price': price}
    logger.error(f"Order failed: {result}")
    return None


def close_position_sync(ticket):
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False

    pos = positions[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    if not tick:
        return False

    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
        "type": close_type, "position": ticket, "price": price, "deviation": 20,
        "magic": pos.magic, "comment": "Close", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return result and result.retcode == mt5.TRADE_RETCODE_DONE


def get_deal_pnl_sync(ticket):
    now = datetime.now(timezone.utc)
    deals = mt5.history_deals_get(now - timedelta(days=7), now, position=ticket)
    if not deals or len(deals) < 2:
        return 0.0
    d = deals[-1]
    return d.profit + d.swap + d.commission


# =============================================================================
# TELEGRAM FUNCTIONS
# =============================================================================

async def send_telegram(message: str):
    """Send Telegram notification"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram failed: {resp.status}")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


async def notify_startup(login: int, balance: float, server: str):
    """Send startup notification"""
    msg = TelegramFormatter.startup(login, balance, server, SYMBOL)
    await send_telegram(msg)


async def notify_trade(direction: str, entry: float, sl: float, tp: float,
                       lot: float, ticket: int, rsi: float, regime: str):
    """Send trade execution notification"""
    msg = TelegramFormatter.trade_executed(direction, SYMBOL, entry, sl, tp, lot, ticket, rsi, regime)
    await send_telegram(msg)


async def notify_close(direction: str, entry: float, exit_price: float, pnl: float,
                       reason: str, consec_losses: int, daily_pnl: float):
    """Send position close notification"""
    msg = TelegramFormatter.position_closed(direction, SYMBOL, entry, exit_price, pnl, reason, consec_losses, daily_pnl)
    await send_telegram(msg)


async def notify_circuit_breaker(loss_pct: float, daily_pnl: float, balance: float):
    """Send circuit breaker notification"""
    msg = TelegramFormatter.circuit_breaker(loss_pct, daily_pnl, balance)
    await send_telegram(msg)


async def notify_cooldown_triggered(consec_losses: int, cooldown_hours: int):
    """Send cooldown triggered notification"""
    msg = TelegramFormatter.cooldown_triggered(consec_losses, cooldown_hours)
    await send_telegram(msg)


async def notify_cooldown_expired():
    """Send cooldown expired notification"""
    msg = TelegramFormatter.cooldown_expired()
    await send_telegram(msg)


async def notify_daily_summary(date: str, trades: int, wins: int, losses: int,
                               net_pnl: float, balance: float):
    """Send daily summary notification"""
    msg = TelegramFormatter.daily_summary(date, trades, wins, losses, net_pnl, balance)
    await send_telegram(msg)


async def notify_error(error_msg: str):
    """Send error notification"""
    msg = TelegramFormatter.error(error_msg)
    await send_telegram(msg)


# =============================================================================
# MAIN LOOP
# =============================================================================

async def main():
    print("=" * 60)
    print("RSI v3.7 OPTIMIZED - Live Trading")
    print("SIDEWAYS Regime + ConsecLoss3 Filter (2h cooldown)")
    print("=" * 60)
    print()

    if not validate_config():
        print("\nConfiguration error. Please check settings.")
        return

    if not connect_mt5():
        return

    try:
        executor = RSIv37Optimized(symbol=SYMBOL, magic_number=MAGIC_NUMBER)

        logger.info("Warming up...")
        if not await executor.warmup():
            logger.error("Warmup failed")
            return

        # Send startup notification
        acc = mt5.account_info()
        if acc:
            await notify_startup(acc.login, acc.balance, MT5_SERVER)

        # Recover positions
        positions = get_positions_sync()
        my_pos = [p for p in positions if p['magic'] == MAGIC_NUMBER]
        if my_pos:
            pos = my_pos[0]
            direction = 'BUY' if pos['type'] == 0 else 'SELL'
            # Reconstruct position object
            executor._position = OpenPosition(
                ticket=pos['ticket'],
                direction=direction,
                entry_price=pos['price_open'],
                stop_loss=pos['sl'],
                take_profit=pos['tp'],
                lot_size=pos['volume'],
                entry_time=datetime.fromtimestamp(pos.get('time', 0), tz=timezone.utc) if pos.get('time') else datetime.now(timezone.utc),
                entry_bar_idx=executor._current_bar_idx,
                rsi_entry=50.0,  # Unknown, use neutral
                atr_pct=50.0,    # Unknown, use neutral
                regime="SIDEWAYS"
            )
            executor.state = ExecutorState.IN_POSITION
            logger.info(f"Recovered position #{pos['ticket']}: {direction} {pos['volume']} @ {pos['price_open']:.5f}")

        status = executor.get_status()
        logger.info(f"Status: {status['state']} | Regime: {status['regime']} ({status['regime_ok']})")

        logger.info("\nRunning... Press Ctrl+C to stop\n")

        last_bar = None

        reconnect_attempts = 0
        max_reconnect_attempts = 3

        while True:
            try:
                acc = mt5.account_info()
                if not acc:
                    reconnect_attempts += 1
                    logger.warning(f"MT5 disconnected. Reconnect attempt {reconnect_attempts}/{max_reconnect_attempts}")
                    if reconnect_attempts <= max_reconnect_attempts:
                        mt5.shutdown()
                        await asyncio.sleep(5)
                        if connect_mt5():
                            logger.info("Reconnected successfully")
                            reconnect_attempts = 0
                        else:
                            await asyncio.sleep(30)
                    else:
                        logger.error("Max reconnect attempts reached. Stopping.")
                        break
                    continue
                reconnect_attempts = 0  # Reset on successful connection

                df = get_ohlcv_sync(SYMBOL, "H1", 2)
                if df is None or len(df) < 2:
                    await asyncio.sleep(5)
                    continue

                current_bar = df.index[-1]

                if last_bar is None or current_bar > last_bar:
                    last_bar = current_bar
                    result = await executor.on_new_bar(current_bar, acc.balance)

                    if result:
                        logger.info(f"Trade executed: {result}")

                    status = executor.get_status()
                    logger.info(f"[{current_bar}] {status['regime']} | Trades:{status['trades']} | {status['net_pnl']}")

                await asyncio.sleep(10)

            except KeyboardInterrupt:
                logger.info("Stopping...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(30)

    finally:
        mt5.shutdown()
        logger.info("Disconnected")


if __name__ == "__main__":
    # Setup logging
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if hasattr(logger, 'remove'):
        logger.remove()
        logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
        logger.add(os.path.join(log_dir, "rsi_v37_opt_{time:YYYY-MM-DD}.log"), rotation="1 day", retention="30 days", level="DEBUG")

    asyncio.run(main())
