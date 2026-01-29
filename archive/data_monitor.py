"""SURGE-WSI Data Monitor
=========================

Live monitor for database and MT5 data updates.
Shows multi-timeframe OHLCV data in real-time.

Usage:
    python data_monitor.py

Author: SURIOTA Team
"""
import sys
import asyncio
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import MetaTrader5 as mt5
from loguru import logger
from config import config
from src.data.db_handler import DBHandler

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")


def clear_screen():
    """Clear terminal screen"""
    print("\033[2J\033[H", end="")


def format_price(price: float, digits: int = 5) -> str:
    """Format price with color based on change"""
    return f"{price:.{digits}f}"


async def get_db_latest(db: DBHandler, symbol: str, timeframe: str, limit: int = 5):
    """Get latest data from database"""
    try:
        df = await db.get_ohlcv(symbol, timeframe, limit)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.error(f"DB error: {e}")
    return None


def get_mt5_latest(symbol: str, timeframe_str: str, limit: int = 5):
    """Get latest data from MT5"""
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }
    tf = timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M15)

    try:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
        if rates is not None and len(rates) > 0:
            return rates
    except Exception as e:
        logger.error(f"MT5 error: {e}")
    return None


def print_header():
    """Print monitor header"""
    now = datetime.now()
    print("=" * 70)
    print(f"  SURGE-WSI DATA MONITOR - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_mt5_data(symbol: str, timeframes: list):
    """Print MT5 live data for multiple timeframes"""
    print(f"\n📊 MT5 LIVE DATA - {symbol}")
    print("-" * 70)

    for tf in timeframes:
        rates = get_mt5_latest(symbol, tf, 3)
        if rates is not None:
            latest = rates[-1]
            time_str = datetime.fromtimestamp(latest['time']).strftime('%H:%M')
            o, h, l, c = latest['open'], latest['high'], latest['low'], latest['close']
            spread = (h - l) * 10000  # in pips
            change = (c - o) * 10000  # in pips
            direction = "▲" if c >= o else "▼"

            print(f"  {tf:>4} | {time_str} | O:{o:.5f} H:{h:.5f} L:{l:.5f} C:{c:.5f} | {direction} {abs(change):.1f}p | Range:{spread:.1f}p")
        else:
            print(f"  {tf:>4} | No data")


async def print_db_data(db: DBHandler, symbol: str, timeframes: list):
    """Print database data for multiple timeframes"""
    print(f"\n🗄️  DATABASE DATA - {symbol}")
    print("-" * 70)

    for tf in timeframes:
        df = await get_db_latest(db, symbol, tf, 3)
        if df is not None and not df.empty:
            # Get column names
            close_col = 'close' if 'close' in df.columns else 'Close'
            open_col = 'open' if 'open' in df.columns else 'Open'
            high_col = 'high' if 'high' in df.columns else 'High'
            low_col = 'low' if 'low' in df.columns else 'Low'

            latest = df.iloc[-1]
            time_idx = df.index[-1]
            time_str = time_idx.strftime('%H:%M') if hasattr(time_idx, 'strftime') else str(time_idx)[-8:-3]

            o = latest[open_col]
            h = latest[high_col]
            l = latest[low_col]
            c = latest[close_col]

            bars_count = len(df)

            print(f"  {tf:>4} | {time_str} | O:{o:.5f} H:{h:.5f} L:{l:.5f} C:{c:.5f} | {bars_count} bars in DB")
        else:
            print(f"  {tf:>4} | No data in database")


def print_tick_info(symbol: str):
    """Print current tick info"""
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        spread = (tick.ask - tick.bid) * 10000
        print(f"\n💹 CURRENT TICK")
        print("-" * 70)
        print(f"  Bid: {tick.bid:.5f}  |  Ask: {tick.ask:.5f}  |  Spread: {spread:.1f} pips")
        print(f"  Last: {tick.last:.5f}  |  Time: {datetime.fromtimestamp(tick.time).strftime('%H:%M:%S')}")


def print_account_info():
    """Print account info"""
    account = mt5.account_info()
    if account:
        print(f"\n💰 ACCOUNT INFO")
        print("-" * 70)
        print(f"  Balance: ${account.balance:,.2f}  |  Equity: ${account.equity:,.2f}  |  Margin: ${account.margin:,.2f}")
        print(f"  Free Margin: ${account.margin_free:,.2f}  |  Profit: ${account.profit:+,.2f}")


async def main():
    """Main monitor loop"""
    symbol = config.trading.symbol
    timeframes = ['M5', 'M15', 'H1', 'H4']

    # Initialize MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return

    logger.info(f"MT5 connected: {mt5.terminal_info().name}")

    # Initialize database
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    if not await db.connect():
        logger.error("Failed to connect to database")
        mt5.shutdown()
        return

    logger.info("Database connected")
    print("\nStarting data monitor... Press Ctrl+C to stop.\n")
    time.sleep(2)

    try:
        while True:
            clear_screen()
            print_header()

            # MT5 Data
            print_mt5_data(symbol, timeframes)

            # Database Data
            await print_db_data(db, symbol, timeframes)

            # Current Tick
            print_tick_info(symbol)

            # Account Info
            print_account_info()

            print("\n" + "=" * 70)
            print("  Refreshing every 10 seconds... Press Ctrl+C to stop")
            print("=" * 70)

            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user.")
    finally:
        await db.disconnect()
        mt5.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
