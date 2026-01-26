"""MT5 Data Sync Script
========================

Fetches OHLCV data from MT5 and stores in TimescaleDB.

Supports:
- Multiple symbols (GBPUSD, EURUSD, etc.)
- Multiple timeframes (M5, M15, H1, H4, D1)
- Incremental sync (only new bars)
- Full historical sync

Usage:
    python sync_mt5_data.py [--full] [--days 365] [--symbol GBPUSD]

Author: SURIOTA Team
"""
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from config import config
from src.data.db_handler import DBHandler


# MT5 Timeframe mapping
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# Default sync configuration - GBPUSD Only
DEFAULT_SYMBOLS = ["GBPUSD"]
DEFAULT_TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
DEFAULT_BARS = 10000  # For incremental sync
FULL_SYNC_DAYS = 365  # 1 year for full sync


class MT5DataSync:
    """MT5 to TimescaleDB data synchronizer"""

    def __init__(
        self,
        db_handler: DBHandler,
        symbols: List[str] = None,
        timeframes: List[str] = None
    ):
        """Initialize data sync

        Args:
            db_handler: Database handler instance
            symbols: List of symbols to sync
            timeframes: List of timeframes to sync
        """
        self.db = db_handler
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.mt5_connected = False

    def connect_mt5(self) -> bool:
        """Connect to MT5 terminal

        Returns:
            True if successful
        """
        if mt5.initialize():
            self.mt5_connected = True
            terminal_info = mt5.terminal_info()
            account_info = mt5.account_info()
            logger.info(f"MT5 connected: {terminal_info.name}")
            logger.info(f"Account: {account_info.login} ({account_info.server})")
            return True

        # Try with terminal path
        if config.mt5.terminal_path:
            if mt5.initialize(path=config.mt5.terminal_path):
                self.mt5_connected = True
                terminal_info = mt5.terminal_info()
                logger.info(f"MT5 connected: {terminal_info.name}")
                return True

        logger.error(f"MT5 connection failed: {mt5.last_error()}")
        return False

    def disconnect_mt5(self):
        """Disconnect from MT5"""
        if self.mt5_connected:
            mt5.shutdown()
            self.mt5_connected = False
            logger.info("MT5 disconnected")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        bars: int = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from MT5

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            bars: Number of bars to fetch
            start_time: Start time for range fetch
            end_time: End time for range fetch

        Returns:
            DataFrame with OHLCV data or None
        """
        if not self.mt5_connected:
            logger.error("MT5 not connected")
            return None

        tf = TIMEFRAMES.get(timeframe)
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        try:
            if start_time and end_time:
                # Fetch by date range
                rates = mt5.copy_rates_range(symbol, tf, start_time, end_time)
            elif bars:
                # Fetch last N bars
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            else:
                # Default: fetch 1000 bars
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 1000)

            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol} {timeframe}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume',
                'spread': 'Spread'
            }, inplace=True)

            # Keep only needed columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']]

            logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return None

    async def sync_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        bars: int = None,
        full_sync: bool = False,
        days: int = None
    ) -> int:
        """Sync a single symbol/timeframe combination

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            bars: Number of bars for incremental sync
            full_sync: If True, sync full history
            days: Number of days for full sync

        Returns:
            Number of bars synced
        """
        if full_sync:
            # Full historical sync
            sync_days = days or FULL_SYNC_DAYS
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=sync_days)
            df = self.fetch_ohlcv(symbol, timeframe, start_time=start_time, end_time=end_time)
        else:
            # Incremental sync
            df = self.fetch_ohlcv(symbol, timeframe, bars=bars or DEFAULT_BARS)

        if df is None or len(df) == 0:
            return 0

        # Save to database
        try:
            count = await self.db.save_ohlcv(symbol, timeframe, df)
            logger.info(f"Synced {count} bars for {symbol} {timeframe}")

            # Log sync
            await self._log_sync(symbol, timeframe, count, df.index.min(), df.index.max(), "SUCCESS")

            return count
        except Exception as e:
            logger.error(f"Error saving {symbol} {timeframe}: {e}")
            await self._log_sync(symbol, timeframe, 0, None, None, "ERROR", str(e))
            return 0

    async def _log_sync(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
        start_time: datetime,
        end_time: datetime,
        status: str,
        message: str = None
    ):
        """Log sync operation to database"""
        if not self.db._pool:
            return

        try:
            async with self.db._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sync_log (symbol, timeframe, bars_synced, start_time, end_time, status, message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, symbol, timeframe, bars, start_time, end_time, status, message)
        except Exception as e:
            logger.debug(f"Could not log sync: {e}")

    async def sync_all(
        self,
        full_sync: bool = False,
        days: int = None,
        bars: int = None
    ) -> Dict[str, int]:
        """Sync all configured symbols and timeframes

        Args:
            full_sync: If True, sync full history
            days: Number of days for full sync
            bars: Number of bars for incremental sync

        Returns:
            Dict with sync counts per symbol/timeframe
        """
        results = {}
        total_bars = 0

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                key = f"{symbol}_{timeframe}"
                count = await self.sync_symbol_timeframe(
                    symbol, timeframe,
                    bars=bars,
                    full_sync=full_sync,
                    days=days
                )
                results[key] = count
                total_bars += count

                # Small delay to avoid overwhelming MT5
                await asyncio.sleep(0.1)

        logger.info(f"Total synced: {total_bars} bars across {len(results)} symbol/timeframe pairs")
        return results

    async def get_sync_status(self) -> Dict[str, Dict]:
        """Get sync status for all symbols/timeframes

        Returns:
            Dict with last sync info per symbol/timeframe
        """
        status = {}

        if not self.db._pool:
            return status

        async with self.db._pool.acquire() as conn:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    # Get latest bar time
                    row = await conn.fetchrow("""
                        SELECT MAX(time) as last_time, COUNT(*) as bar_count
                        FROM ohlcv
                        WHERE symbol = $1 AND timeframe = $2
                    """, symbol, timeframe)

                    # Get last sync info
                    sync_row = await conn.fetchrow("""
                        SELECT sync_time, bars_synced, status
                        FROM sync_log
                        WHERE symbol = $1 AND timeframe = $2
                        ORDER BY sync_time DESC
                        LIMIT 1
                    """, symbol, timeframe)

                    key = f"{symbol}_{timeframe}"
                    status[key] = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "last_bar_time": row['last_time'] if row else None,
                        "bar_count": row['bar_count'] if row else 0,
                        "last_sync": sync_row['sync_time'] if sync_row else None,
                        "last_sync_bars": sync_row['bars_synced'] if sync_row else 0,
                        "last_sync_status": sync_row['status'] if sync_row else None,
                    }

        return status


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Sync MT5 data to TimescaleDB")
    parser.add_argument("--full", action="store_true", help="Full historical sync")
    parser.add_argument("--days", type=int, default=365, help="Days to sync for full sync")
    parser.add_argument("--bars", type=int, default=10000, help="Bars for incremental sync")
    parser.add_argument("--symbol", type=str, help="Sync specific symbol only")
    parser.add_argument("--timeframe", type=str, help="Sync specific timeframe only")
    parser.add_argument("--status", action="store_true", help="Show sync status")

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add("logs/sync_mt5.log", rotation="10 MB", retention="7 days")

    # Initialize database
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    # Connect to database
    if not await db.connect():
        logger.error("Failed to connect to database")
        logger.info("Make sure Docker containers are running: docker-compose up -d")
        return

    # Initialize tables
    await db.initialize_tables()

    # Determine symbols and timeframes
    symbols = [args.symbol] if args.symbol else DEFAULT_SYMBOLS
    timeframes = [args.timeframe] if args.timeframe else DEFAULT_TIMEFRAMES

    # Initialize syncer
    syncer = MT5DataSync(db, symbols=symbols, timeframes=timeframes)

    # Show status if requested
    if args.status:
        status = await syncer.get_sync_status()
        print("\n" + "=" * 70)
        print("SURGE-WSI Data Sync Status")
        print("=" * 70)
        for key, info in status.items():
            print(f"\n{info['symbol']} {info['timeframe']}:")
            print(f"  Bars in DB: {info['bar_count']:,}")
            print(f"  Last bar: {info['last_bar_time']}")
            print(f"  Last sync: {info['last_sync']}")
            print(f"  Last sync status: {info['last_sync_status']}")
        print("=" * 70)
        await db.disconnect()
        return

    # Connect to MT5
    if not syncer.connect_mt5():
        logger.error("Failed to connect to MT5. Make sure terminal is running.")
        await db.disconnect()
        return

    try:
        print("\n" + "=" * 70)
        print("SURGE-WSI MT5 Data Sync")
        print("=" * 70)
        print(f"Symbols: {symbols}")
        print(f"Timeframes: {timeframes}")
        print(f"Mode: {'Full Sync' if args.full else 'Incremental'}")
        if args.full:
            print(f"Days: {args.days}")
        else:
            print(f"Bars: {args.bars}")
        print("=" * 70 + "\n")

        # Run sync
        results = await syncer.sync_all(
            full_sync=args.full,
            days=args.days,
            bars=args.bars
        )

        # Print results
        print("\n" + "=" * 70)
        print("Sync Results")
        print("=" * 70)
        total = 0
        for key, count in sorted(results.items()):
            print(f"  {key}: {count:,} bars")
            total += count
        print("-" * 70)
        print(f"  TOTAL: {total:,} bars")
        print("=" * 70)

    finally:
        syncer.disconnect_mt5()
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
