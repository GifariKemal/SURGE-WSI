"""
SURGE-WSI MULTI-PAIR EXECUTOR
=============================
Single terminal running 2 pairs simultaneously:
- GBPUSD: 60% allocation with full Quad-Layer filter
- GBPJPY: 40% allocation with full Quad-Layer filter

Each pair has INDEPENDENT:
- Risk management (Layer 3)
- Pattern filter (Layer 4)
- Trade history
- Position tracking

Usage:
    python main_multi_pair.py --demo    # Demo account
    python main_multi_pair.py --live    # Live account (careful!)
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.telegram import TelegramNotifier
from src.config.mt5_config import MT5Config
from strategies.gbpusd_h1_quadlayer.trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    get_monthly_quality_adjustment
)

# ============================================================
# MULTI-PAIR CONFIGURATION
# ============================================================
@dataclass
class PairConfig:
    """Configuration for each trading pair"""
    symbol: str
    allocation: float       # Percentage of total capital
    pip_size: float
    pip_value: float        # USD per pip per lot
    magic_number: int       # Unique ID for MT5 orders

    # Risk parameters (from proven backtest config)
    risk_percent: float = 0.5
    sl_atr_mult: float = 1.5   # SL = ATR * 1.5
    tp_ratio: float = 1.5      # TP = SL * 1.5 (1.5:1 R:R)
    max_lot: float = 5.0

# Define pairs - NO ALLOCATION SPLIT, each uses full balance for risk calc
PAIRS = {
    'GBPUSD': PairConfig(
        symbol='GBPUSD',
        allocation=1.0,  # 100% - uses full balance
        pip_size=0.0001,
        pip_value=10.0,
        magic_number=64001,
    ),
    'GBPJPY': PairConfig(
        symbol='GBPJPY',
        allocation=1.0,  # 100% - uses full balance
        pip_size=0.01,
        pip_value=6.5,
        magic_number=64002,
    )
}
# Note: With 0.5% risk per pair, max exposure is 1% if both trade simultaneously

# Trading hours (UTC)
TRADING_HOURS = list(range(8, 18))  # 08:00-17:59

# Day multipliers (same for both pairs)
DAY_MULTIPLIERS = {
    0: 1.0,   # Monday
    1: 0.9,   # Tuesday
    2: 1.0,   # Wednesday
    3: 0.8,   # Thursday
    4: 0.3,   # Friday (reduced)
    5: 0.0,   # Saturday (no trade)
    6: 0.0,   # Sunday (no trade)
}

# ============================================================
# PAIR STATE TRACKER
# ============================================================
@dataclass
class PairState:
    """State for each trading pair"""
    config: PairConfig
    risk_manager: IntraMonthRiskManager = field(default_factory=IntraMonthRiskManager)
    pattern_filter: PatternBasedFilter = field(default_factory=PatternBasedFilter)

    # Trading state
    current_position: Optional[dict] = None
    last_signal_time: Optional[datetime] = None
    daily_trades: int = 0
    last_trade_date: Optional[str] = None

    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0

# ============================================================
# MULTI-PAIR EXECUTOR
# ============================================================
class MultiPairExecutor:
    """Executor for multiple pairs in single terminal"""

    def __init__(self, demo: bool = True):
        self.demo = demo
        self.running = False
        self.telegram: Optional[TelegramNotifier] = None

        # Initialize state for each pair
        self.pair_states: Dict[str, PairState] = {}
        for symbol, config in PAIRS.items():
            self.pair_states[symbol] = PairState(config=config)

        # Get account balance
        self.initial_balance = 0.0

    async def initialize(self):
        """Initialize MT5 and Telegram"""

        # Initialize MT5
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False

        # Get account info
        account = mt5.account_info()
        if account is None:
            logger.error("Failed to get account info")
            return False

        self.initial_balance = account.balance

        logger.info(f"MT5 connected: {account.server}")
        logger.info(f"Account: {account.login} ({'Demo' if self.demo else 'LIVE'})")
        logger.info(f"Balance: ${account.balance:,.2f}")

        # Initialize Telegram
        try:
            mt5_config = MT5Config()
            self.telegram = TelegramNotifier(
                mt5_config.telegram_token,
                mt5_config.telegram_chat_id
            )
            await self.telegram.initialize()
            logger.info("Telegram initialized")
        except Exception as e:
            logger.warning(f"Telegram init failed: {e}")

        # Log pairs
        for symbol in self.pair_states.keys():
            logger.info(f"{symbol}: Active (0.5% risk per trade)")

        return True

    def get_pair_balance(self, symbol: str) -> float:
        """Get allocated balance for a pair"""
        account = mt5.account_info()
        if account:
            return account.balance * self.pair_states[symbol].config.allocation
        return 0.0

    async def fetch_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Fetch H1 data for a symbol"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Calculate indicators
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def detect_signal(self, df: pd.DataFrame, config: PairConfig) -> Optional[dict]:
        """Detect trading signal for a pair"""
        if len(df) < 50:
            return None

        bar = df.iloc[-1]
        prev = df.iloc[-2]
        hour = datetime.now().hour

        # Hour filter
        if hour not in TRADING_HOURS:
            return None

        # Day filter
        day = datetime.now().weekday()
        if DAY_MULTIPLIERS.get(day, 0) == 0:
            return None

        # ATR check
        atr = bar['atr']
        if pd.isna(atr) or atr <= 0:
            return None

        atr_pips = atr / config.pip_size
        if atr_pips < 10 or atr_pips > 50:
            return None

        # EMA alignment
        bullish = bar['ema_9'] > bar['ema_21'] > bar['ema_50']
        bearish = bar['ema_9'] < bar['ema_21'] < bar['ema_50']

        # RSI
        rsi = bar['rsi']
        rsi_bull = 40 < rsi < 70
        rsi_bear = 30 < rsi < 60

        signal = None

        # Order Block (Engulfing)
        if bullish and rsi_bull:
            if prev['close'] < prev['open'] and bar['close'] > bar['open'] and bar['close'] > prev['high']:
                signal = {'direction': 'BUY', 'type': 'ORDER_BLOCK'}

        if bearish and rsi_bear and signal is None:
            if prev['close'] > prev['open'] and bar['close'] < bar['open'] and bar['close'] < prev['low']:
                signal = {'direction': 'SELL', 'type': 'ORDER_BLOCK'}

        # EMA Pullback
        if bullish and rsi_bull and signal is None:
            if bar['low'] <= bar['ema_21'] * 1.002 and bar['close'] > bar['ema_9']:
                signal = {'direction': 'BUY', 'type': 'EMA_PULLBACK'}

        if bearish and rsi_bear and signal is None:
            if bar['high'] >= bar['ema_21'] * 0.998 and bar['close'] < bar['ema_9']:
                signal = {'direction': 'SELL', 'type': 'EMA_PULLBACK'}

        if signal:
            signal['price'] = bar['close']
            signal['atr_pips'] = atr_pips
            signal['time'] = bar.name

        return signal

    def check_filters(self, symbol: str, signal: dict) -> tuple[bool, float, str]:
        """Check all filters for a pair"""
        state = self.pair_states[symbol]
        now = datetime.now()

        # Day multiplier
        day_mult = DAY_MULTIPLIERS.get(now.weekday(), 1.0)
        if day_mult == 0:
            return False, 0, "DAY_BLOCKED"

        # Risk manager (Layer 3)
        can_trade, quality_adj, reason = state.risk_manager.new_trade_check(now)
        if not can_trade:
            return False, 0, f"RISK: {reason}"

        # Pattern filter (Layer 4)
        pattern_ok, size_mult, pattern_reason = state.pattern_filter.check_trade_allowed()
        if not pattern_ok:
            return False, 0, f"PATTERN: {pattern_reason}"

        # Quality threshold
        quality = 70 + quality_adj
        if quality >= 100:
            return False, 0, f"QUALITY: {quality}"

        # Calculate final size multiplier
        final_mult = day_mult * size_mult

        return True, final_mult, "OK"

    def calculate_lot_size(self, symbol: str, atr_pips: float, size_mult: float) -> float:
        """Calculate position size for a pair"""
        state = self.pair_states[symbol]
        config = state.config

        balance = self.get_pair_balance(symbol)
        sl_pips = atr_pips * config.sl_atr_mult

        risk_amount = balance * (config.risk_percent / 100)
        lot_size = risk_amount / (sl_pips * config.pip_value)
        lot_size = min(lot_size, config.max_lot)
        lot_size = round(lot_size * size_mult, 2)

        return max(0.01, lot_size)

    async def open_position(self, symbol: str, signal: dict, lot_size: float) -> bool:
        """Open position on MT5"""
        config = self.pair_states[symbol].config

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        direction = signal['direction']
        atr_pips = signal['atr_pips']

        sl_pips = atr_pips * config.sl_atr_mult
        tp_pips = sl_pips * config.tp_ratio

        if direction == 'BUY':
            price = tick.ask
            sl = price - (sl_pips * config.pip_size)
            tp = price + (tp_pips * config.pip_size)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + (sl_pips * config.pip_size)
            tp = price - (tp_pips * config.pip_size)
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": config.magic_number,
            "comment": f"SURGE_{signal['type']}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[{symbol}] Position opened: {direction} {lot_size} @ {price:.5f}")

            # Update state
            state = self.pair_states[symbol]
            state.current_position = {
                'ticket': result.order,
                'direction': direction,
                'entry_price': price,
                'lot_size': lot_size,
                'sl': sl,
                'tp': tp,
                'entry_time': datetime.now()
            }
            state.total_trades += 1

            # Telegram notification
            if self.telegram:
                await self.telegram.send_message(
                    f"🟢 *{symbol} {direction}*\n"
                    f"Entry: {price:.5f}\n"
                    f"Lot: {lot_size}\n"
                    f"SL: {sl:.5f}\n"
                    f"TP: {tp:.5f}\n"
                    f"Type: {signal['type']}"
                )

            return True
        else:
            logger.error(f"[{symbol}] Order failed: {result.retcode}")
            return False

    def check_position(self, symbol: str) -> Optional[dict]:
        """Check if there's an open position for this pair"""
        config = self.pair_states[symbol].config
        positions = mt5.positions_get(symbol=symbol)

        if positions:
            for pos in positions:
                if pos.magic == config.magic_number:
                    return {
                        'ticket': pos.ticket,
                        'direction': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'entry_price': pos.price_open,
                        'current_price': pos.price_current,
                        'lot_size': pos.volume,
                        'profit': pos.profit,
                        'sl': pos.sl,
                        'tp': pos.tp
                    }
        return None

    async def process_pair(self, symbol: str):
        """Process a single pair - check signals and manage positions"""
        state = self.pair_states[symbol]
        config = state.config

        # Check existing position
        position = self.check_position(symbol)

        if position:
            # Already have position - update state
            state.current_position = position
            return
        else:
            # Position closed - record if was open
            if state.current_position:
                # Position was closed (SL/TP hit)
                old_pos = state.current_position

                # Get trade history to find the closed trade
                deals = mt5.history_deals_get(
                    datetime.now() - timedelta(hours=1),
                    datetime.now()
                )

                pnl = 0
                if deals:
                    for deal in deals:
                        if deal.magic == config.magic_number and deal.position_id == old_pos.get('ticket'):
                            pnl = deal.profit
                            break

                # Update filters
                state.risk_manager.record_trade(pnl, datetime.now(), old_pos['direction'])
                state.pattern_filter.record_trade(old_pos['direction'], pnl, datetime.now())

                state.total_pnl += pnl
                if pnl > 0:
                    state.winning_trades += 1

                logger.info(f"[{symbol}] Position closed: ${pnl:+.2f}")

                if self.telegram:
                    emoji = "🟢" if pnl > 0 else "🔴"
                    await self.telegram.send_message(
                        f"{emoji} *{symbol} Closed*\n"
                        f"P/L: ${pnl:+.2f}\n"
                        f"Total: ${state.total_pnl:+.2f}"
                    )

                state.current_position = None

        # No position - look for new signal
        # Cooldown check
        if state.last_signal_time:
            if (datetime.now() - state.last_signal_time).total_seconds() < 2 * 3600:
                return

        # Fetch data and detect signal
        df = await self.fetch_data(symbol)
        if df.empty:
            return

        signal = self.detect_signal(df, config)
        if not signal:
            return

        # Check filters
        can_trade, size_mult, reason = self.check_filters(symbol, signal)
        if not can_trade:
            logger.debug(f"[{symbol}] Signal blocked: {reason}")
            return

        # Calculate lot size
        lot_size = self.calculate_lot_size(symbol, signal['atr_pips'], size_mult)

        # Open position
        success = await self.open_position(symbol, signal, lot_size)
        if success:
            state.last_signal_time = datetime.now()

    async def run(self):
        """Main loop"""
        if not await self.initialize():
            return

        self.running = True

        # Startup message
        pairs_list = ", ".join(self.pair_states.keys())

        startup_msg = (
            f"🚀 *MULTI-PAIR EXECUTOR STARTED*\n\n"
            f"Account: {'Demo' if self.demo else 'LIVE'}\n"
            f"Balance: ${self.initial_balance:,.0f}\n\n"
            f"*Pairs:* {pairs_list}\n"
            f"*Risk:* 0.5% per trade per pair\n"
            f"*Max Risk:* 1% (if both pairs trade)\n\n"
            f"Trading Hours: 08:00-18:00 UTC"
        )

        if self.telegram:
            await self.telegram.send_message(startup_msg)

        logger.info("Multi-pair executor started")
        logger.info(f"Pairs: {list(PAIRS.keys())}")

        try:
            while self.running:
                # Process each pair
                for symbol in PAIRS.keys():
                    try:
                        await self.process_pair(symbol)
                    except Exception as e:
                        logger.error(f"[{symbol}] Error: {e}")

                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            mt5.shutdown()

    async def get_status(self) -> str:
        """Get current status of all pairs"""
        account = mt5.account_info()
        balance = account.balance if account else 0

        lines = ["📊 *MULTI-PAIR STATUS*\n"]
        lines.append(f"Balance: ${balance:,.0f}\n")

        for symbol, state in self.pair_states.items():
            pos = self.check_position(symbol)

            lines.append(f"\n*{symbol}*")
            lines.append(f"  Trades: {state.total_trades}")

            if state.total_trades > 0:
                wr = (state.winning_trades / state.total_trades) * 100
                lines.append(f"  Win Rate: {wr:.0f}%")

            lines.append(f"  P/L: ${state.total_pnl:+,.2f}")

            if pos:
                lines.append(f"  Position: {pos['direction']} {pos['lot_size']}")
                lines.append(f"  Profit: ${pos['profit']:+,.2f}")
            else:
                lines.append(f"  Position: None")

        return "\n".join(lines)


# ============================================================
# TELEGRAM COMMANDS
# ============================================================
async def setup_telegram_commands(executor: MultiPairExecutor):
    """Setup Telegram command handlers"""
    if not executor.telegram:
        return

    @executor.telegram.command("/status")
    async def cmd_status(msg):
        status = await executor.get_status()
        await executor.telegram.send_message(status)

    @executor.telegram.command("/pairs")
    async def cmd_pairs(msg):
        lines = ["📈 *ACTIVE PAIRS*\n"]
        for symbol, state in executor.pair_states.items():
            pos = executor.check_position(symbol)
            status = f"📍 {pos['direction']}" if pos else "⏸ No position"
            lines.append(f"{symbol}: {status}")
        await executor.telegram.send_message("\n".join(lines))

    @executor.telegram.command("/stop")
    async def cmd_stop(msg):
        executor.running = False
        await executor.telegram.send_message("⏹ Executor stopping...")


# ============================================================
# MAIN
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="SURGE-WSI Multi-Pair Executor")
    parser.add_argument("--demo", action="store_true", help="Run on demo account")
    parser.add_argument("--live", action="store_true", help="Run on live account")
    args = parser.parse_args()

    if not args.demo and not args.live:
        print("Please specify --demo or --live")
        return

    if args.live:
        print("⚠️  WARNING: LIVE TRADING MODE")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

    executor = MultiPairExecutor(demo=args.demo)
    await setup_telegram_commands(executor)
    await executor.run()


if __name__ == "__main__":
    asyncio.run(main())
