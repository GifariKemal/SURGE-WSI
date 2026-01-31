#!/usr/bin/env python3
"""
Fetch 10-Year Historical Data from MT5
======================================

Fetches GBPUSD H1 data from 2015-2019 from MT5 and saves to TimescaleDB.
This completes the 10-year dataset (2015-2026) for AI/Bot training.

Key events in 2015-2019:
- 2015: Start of Fed rate hike cycle
- 2016 June: BREXIT VOTE (GBP crash -10%)
- 2017: Article 50 triggered
- 2018: Trade war volatility
- 2019: Brexit uncertainty, multiple deadline extensions

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


async def fetch_year_data(mt5: MT5Connector, symbol: str, timeframe: str, year: int) -> pd.DataFrame:
    """Fetch data for a specific year"""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    print(f"  Fetching {year}...")
    df = mt5.get_ohlcv_range(symbol, timeframe, start, end)

    if df is not None and not df.empty:
        print(f"    Got {len(df)} bars ({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})")
        return df
    else:
        print(f"    WARNING: No data for {year}")
        return pd.DataFrame()


async def save_to_db(db: DBHandler, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Save dataframe to database"""
    if df.empty:
        return 0

    # Ensure timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    rows_saved = await db.save_ohlcv(symbol, timeframe, df)
    return rows_saved


async def verify_db_data(db: DBHandler, symbol: str, timeframe: str):
    """Check what data exists in DB"""
    query = f"""
    SELECT
        EXTRACT(YEAR FROM time) as year,
        COUNT(*) as bars,
        MIN(time) as first_bar,
        MAX(time) as last_bar
    FROM ohlcv
    WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
    GROUP BY EXTRACT(YEAR FROM time)
    ORDER BY year
    """
    results = await db._pool.fetch(query)

    print(f"\n{symbol} {timeframe} data in database:")
    total = 0
    for r in results:
        year = int(r['year'])
        bars = r['bars']
        total += bars
        first = r['first_bar'].strftime('%Y-%m-%d')
        last = r['last_bar'].strftime('%Y-%m-%d')
        print(f"  {year}: {bars:,} bars ({first} to {last})")
    print(f"  TOTAL: {total:,} bars")
    return results


async def main():
    print("=" * 70)
    print("FETCH 10-YEAR HISTORICAL DATA FROM MT5")
    print("Fetching 2015-2019 GBPUSD H1 data for complete 10-year dataset")
    print("=" * 70)

    symbol = "GBPUSD"
    timeframe = "H1"
    years_to_fetch = [2015, 2016, 2017, 2018, 2019]

    # Step 1: Connect to MT5
    print("\n[1/4] Connecting to MT5...")
    mt5 = MT5Connector()
    if not mt5.connect():
        print("ERROR: Failed to connect to MT5")
        print("\nTroubleshooting:")
        print("1. Make sure MT5 terminal is running")
        print("2. Make sure you're logged in to your broker account")
        print("3. Try opening a GBPUSD H1 chart and scroll to 2015")
        print("4. Let MT5 download the historical data")
        print("5. Run this script again")
        return False

    account_info = mt5.get_account_info_sync()
    if account_info:
        print(f"OK - Connected to {account_info.get('server', 'unknown')} "
              f"(Account: {account_info.get('login', 'unknown')})")
    else:
        print("OK - MT5 connected")

    # Step 2: Fetch historical data year by year
    print(f"\n[2/4] Fetching historical {timeframe} data...")
    all_data = []

    for year in years_to_fetch:
        df = await fetch_year_data(mt5, symbol, timeframe, year)
        if not df.empty:
            all_data.append(df)

    mt5.disconnect()

    if not all_data:
        print("\nERROR: No data fetched from MT5!")
        print("\nMake sure MT5 has historical data available:")
        print("1. Open MT5 terminal")
        print("2. Go to Tools -> Options -> Charts")
        print("3. Set 'Max bars in chart' to 'Unlimited'")
        print("4. Open GBPUSD H1 chart")
        print("5. Press Ctrl+End and let it load historical data")
        print("6. Wait for data to download (may take several minutes)")
        print("7. Run this script again")
        return False

    # Combine all years
    combined_df = pd.concat(all_data)
    combined_df = combined_df.sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    print(f"\nTotal fetched: {len(combined_df)} bars")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

    # Step 3: Save to database
    print(f"\n[3/4] Saving to TimescaleDB...")
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

    try:
        rows_saved = await save_to_db(db, symbol, timeframe, combined_df)
        print(f"OK - Saved {rows_saved} bars to database")
    except Exception as e:
        print(f"ERROR saving: {e}")
        await db.disconnect()
        return False

    # Step 4: Verify data
    print(f"\n[4/4] Verifying database data...")
    await verify_db_data(db, symbol, timeframe)

    await db.disconnect()

    print("\n" + "=" * 70)
    print("FETCH COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run profile generation for 2015-2019:")
    print("  python scripts/generate_2015_2019_profiles.py")

    return True


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main())
