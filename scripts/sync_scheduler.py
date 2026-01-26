"""MT5 Data Sync Scheduler
===========================

Runs continuous data sync from MT5 to TimescaleDB.
Syncs on configurable intervals when market is open.

Features:
- Auto-sync when market is active
- Skips sync when market is closed (weekends)
- Stores all data to TimescaleDB for AI Trading

Usage:
    python sync_scheduler.py

Author: SURIOTA Team
"""
import sys
import asyncio
import signal
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config import config
from src.data.db_handler import DBHandler
from sync_mt5_data import MT5DataSync

# GBPUSD Only
SYNC_SYMBOLS = ["GBPUSD"]
SYNC_TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

# Sync intervals (in seconds)
SYNC_INTERVALS = {
    "M1": 60,       # Every 1 minute
    "M5": 300,      # Every 5 minutes
    "M15": 900,     # Every 15 minutes
    "M30": 1800,    # Every 30 minutes
    "H1": 3600,     # Every 1 hour
    "H4": 14400,    # Every 4 hours
    "D1": 86400,    # Every 24 hours
}

# Bars to fetch per sync
SYNC_BARS = {
    "M1": 100,
    "M5": 50,
    "M15": 30,
    "M30": 20,
    "H1": 10,
    "H4": 5,
    "D1": 3,
}


def is_market_open() -> bool:
    """Check if Forex market is open

    Forex market hours (UTC):
    - Opens: Sunday 22:00 UTC (Sydney)
    - Closes: Friday 22:00 UTC (New York)

    Returns:
        True if market is open
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour

    # Market closed on Saturday
    if weekday == 5:
        return False

    # Market closed on Sunday before 22:00 UTC
    if weekday == 6 and hour < 22:
        return False

    # Market closed on Friday after 22:00 UTC
    if weekday == 4 and hour >= 22:
        return False

    return True


def get_next_market_open() -> datetime:
    """Get next market open time

    Returns:
        Next market open datetime
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()

    if weekday == 5:  # Saturday
        # Next open is Sunday 22:00 UTC
        days_until = 1
    elif weekday == 6:  # Sunday
        if now.hour < 22:
            days_until = 0
        else:
            days_until = 7  # Next Sunday
    elif weekday == 4 and now.hour >= 22:  # Friday after close
        days_until = 2  # Sunday
    else:
        return now  # Market is open

    next_open = now.replace(hour=22, minute=0, second=0, microsecond=0)
    next_open += timedelta(days=days_until)
    return next_open


class SyncScheduler:
    """Scheduled MT5 data sync with market hours awareness"""

    def __init__(
        self,
        db_handler: DBHandler,
        symbols: list = None,
        timeframes: list = None
    ):
        """Initialize scheduler

        Args:
            db_handler: Database handler
            symbols: Symbols to sync
            timeframes: Timeframes to sync
        """
        self.db = db_handler
        self.symbols = symbols or SYNC_SYMBOLS
        self.timeframes = timeframes or SYNC_TIMEFRAMES
        self.syncer = None
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}
        self._last_sync: Dict[str, datetime] = {}
        self._sync_count: int = 0
        self._market_status: str = "unknown"

    async def start(self):
        """Start the sync scheduler"""
        logger.info("Starting sync scheduler...")

        # Initialize syncer
        self.syncer = MT5DataSync(self.db, self.symbols, self.timeframes)

        # Connect to MT5
        if not self.syncer.connect_mt5():
            logger.error("Failed to connect to MT5")
            return

        self._running = True

        # Start sync tasks for each timeframe
        for tf in self.timeframes:
            self._tasks[tf] = asyncio.create_task(self._sync_loop(tf))
            logger.info(f"Started sync task for {tf} (interval: {SYNC_INTERVALS.get(tf, 3600)}s)")

        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks.values())
        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")

    async def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping sync scheduler...")
        self._running = False

        # Cancel all tasks
        for task in self._tasks.values():
            task.cancel()

        # Disconnect
        if self.syncer:
            self.syncer.disconnect_mt5()

    async def _sync_loop(self, timeframe: str):
        """Sync loop for a single timeframe with market hours check

        Args:
            timeframe: Timeframe to sync
        """
        interval = SYNC_INTERVALS.get(timeframe, 3600)
        bars = SYNC_BARS.get(timeframe, 100)

        while self._running:
            try:
                # Check if market is open
                if not is_market_open():
                    if self._market_status != "closed":
                        self._market_status = "closed"
                        next_open = get_next_market_open()
                        logger.warning(f"Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M UTC')}")
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue

                if self._market_status != "open":
                    self._market_status = "open"
                    logger.info("Market is OPEN - syncing data...")

                # Sync all symbols for this timeframe
                for symbol in self.symbols:
                    try:
                        count = await self.syncer.sync_symbol_timeframe(
                            symbol, timeframe, bars=bars
                        )
                        if count > 0:
                            self._last_sync[f"{symbol}_{timeframe}"] = datetime.now(timezone.utc)
                            self._sync_count += count
                            logger.debug(f"Synced {count} bars for {symbol} {timeframe}")
                    except Exception as e:
                        logger.error(f"Error syncing {symbol} {timeframe}: {e}")

                # Wait for next interval
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error for {timeframe}: {e}")
                await asyncio.sleep(60)  # Wait before retry

    def get_status(self) -> Dict:
        """Get scheduler status

        Returns:
            Status dict
        """
        return {
            "running": self._running,
            "market_status": self._market_status,
            "market_open": is_market_open(),
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "active_tasks": len([t for t in self._tasks.values() if not t.done()]),
            "total_synced": self._sync_count,
            "last_sync": self._last_sync,
        }


async def main():
    """Main entry point"""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add("logs/sync_scheduler.log", rotation="10 MB", retention="7 days")

    print("\n" + "=" * 70)
    print("SURGE-WSI MT5 Data Sync Scheduler")
    print("=" * 70)
    print(f"Symbols: {SYNC_SYMBOLS}")
    print(f"Timeframes: {SYNC_TIMEFRAMES}")
    print(f"Market Open: {is_market_open()}")
    if not is_market_open():
        next_open = get_next_market_open()
        print(f"Next Open: {next_open.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    print("Auto-sync enabled: Data will be stored to TimescaleDB")
    print("Press Ctrl+C to stop")
    print("=" * 70 + "\n")

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

    # Initialize scheduler with GBPUSD only
    scheduler = SyncScheduler(db, symbols=SYNC_SYMBOLS, timeframes=SYNC_TIMEFRAMES)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(scheduler.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        # Start scheduler
        await scheduler.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await scheduler.stop()
        await db.disconnect()
        logger.info(f"Total bars synced this session: {scheduler._sync_count}")


if __name__ == "__main__":
    asyncio.run(main())
