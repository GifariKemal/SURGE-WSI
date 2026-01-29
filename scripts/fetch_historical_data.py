"""Fetch Historical Data from MT5
=================================

Fetch M15 data for 2023 from MT5 and save to TimescaleDB.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
from datetime import datetime, timezone
from loguru import logger

from config import config
from src.data.mt5_connector import MT5Connector
from src.data.db_handler import DBHandler


async def fetch_and_save(symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
    """Fetch data from MT5 and save to database"""

    print(f"\n{'='*60}")
    print(f"Fetching {symbol} {timeframe} data")
    print(f"From: {start_date}")
    print(f"To:   {end_date}")
    print(f"{'='*60}\n")

    # Connect to MT5
    print("[1/4] Connecting to MT5...")
    mt5 = MT5Connector()
    if not mt5.connect():
        print("ERROR: Failed to connect to MT5")
        print("Make sure MT5 terminal is running and logged in")
        return False
    print("OK - MT5 connected")

    # Fetch data
    print(f"\n[2/4] Fetching {timeframe} data from MT5...")
    df = mt5.get_ohlcv_range(symbol, timeframe, start_date, end_date)
    mt5.disconnect()

    if df is None or df.empty:
        print("ERROR: No data returned from MT5")
        print("This could be because:")
        print("  - The symbol doesn't have that much history")
        print("  - MT5 terminal needs to download the data first")
        print("  - Try opening a chart with the timeframe in MT5 first")
        return False

    print(f"OK - Fetched {len(df)} candles")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Connect to database
    print(f"\n[3/4] Connecting to TimescaleDB...")
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    if not await db.connect():
        print("ERROR: Failed to connect to database")
        return False
    print("OK - Database connected")

    # Save data using existing save_ohlcv method
    print(f"\n[4/4] Saving to TimescaleDB...")
    try:
        # Make sure index is timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

        rows_saved = await db.save_ohlcv(symbol, timeframe, df)
        print(f"OK - Saved {rows_saved} candles to database")

    except Exception as e:
        print(f"ERROR saving data: {e}")
        await db.disconnect()
        return False

    await db.disconnect()
    return True


async def verify_data(symbol: str, timeframe: str):
    """Verify data in database"""
    print(f"\nVerifying {symbol} {timeframe} data in database...")

    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    await db.connect()
    df = await db.get_ohlcv(symbol, timeframe, limit=1000000)
    await db.disconnect()

    if df.empty:
        print("  No data found")
        return

    print(f"  Total candles: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Check by year
    for year in [2022, 2023, 2024, 2025, 2026]:
        year_data = df[df.index.year == year]
        if not year_data.empty:
            print(f"  {year}: {len(year_data)} candles ({year_data.index.min().strftime('%b')} - {year_data.index.max().strftime('%b')})")


async def main():
    print("\n" + "="*60)
    print("FETCH HISTORICAL DATA FROM MT5")
    print("="*60)

    symbol = "GBPUSD"

    # Check current data first
    print("\n[0] Current data status:")
    await verify_data(symbol, "M15")
    await verify_data(symbol, "H4")

    # Fetch 2023 M15 data
    print("\n" + "="*60)
    print("FETCHING 2023 M15 DATA")
    print("="*60)

    start_2023 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_2023 = datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc)

    success = await fetch_and_save(symbol, "M15", start_2023, end_2023)

    if success:
        print("\n[SUCCESS] 2023 M15 data fetched and saved!")
    else:
        print("\n[FAILED] Could not fetch 2023 M15 data")
        print("\nTroubleshooting:")
        print("1. Open MT5 terminal")
        print("2. Open a GBPUSD M15 chart")
        print("3. Press Ctrl+End to scroll to oldest data")
        print("4. Let MT5 download historical data")
        print("5. Run this script again")
        return

    # Also fetch 2022 for warmup data
    print("\n" + "="*60)
    print("FETCHING 2022 M15 DATA (for warmup)")
    print("="*60)

    start_2022 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_2022 = datetime(2022, 12, 31, 23, 59, tzinfo=timezone.utc)

    await fetch_and_save(symbol, "M15", start_2022, end_2022)

    # Verify final data
    print("\n" + "="*60)
    print("FINAL DATA STATUS")
    print("="*60)
    await verify_data(symbol, "M15")
    await verify_data(symbol, "H4")

    print("\n" + "="*60)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main())
