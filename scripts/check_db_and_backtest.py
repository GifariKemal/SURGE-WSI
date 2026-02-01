"""
Check database data source and run backtest with Finex data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timezone
import MetaTrader5 as mt5

# Check MT5 connection first
print("="*60)
print("CHECKING MT5 CONNECTION")
print("="*60)

if mt5.initialize():
    info = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Terminal: {info.name}")
    print(f"Company: {info.company}")
    print(f"Account: {account.login}")
    print(f"Server: {account.server}")
    print(f"Balance: ${account.balance}")

    # Check data
    print("\n" + "="*60)
    print("CHECKING GBPUSD H1 DATA FROM MT5")
    print("="*60)

    rates = mt5.copy_rates_range(
        "GBPUSD",
        mt5.TIMEFRAME_H1,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 10, tzinfo=timezone.utc)
    )

    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(f"Data count: {len(df)} bars")
        print(f"First bar: {df.iloc[0]['time']}")
        print(f"Last bar: {df.iloc[-1]['time']}")
        print("\nSample data (first 5 bars):")
        print(df[['time', 'open', 'high', 'low', 'close']].head())
    else:
        print("No data available!")

    mt5.shutdown()
else:
    print(f"MT5 initialization failed: {mt5.last_error()}")

# Now check database
print("\n" + "="*60)
print("CHECKING DATABASE")
print("="*60)

try:
    import psycopg2
    from config import config

    conn = psycopg2.connect(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    cursor = conn.cursor()

    # Check if ohlcv table exists and has data
    cursor.execute("""
        SELECT COUNT(*), MIN(time), MAX(time)
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
    """)
    result = cursor.fetchone()

    if result and result[0] > 0:
        print(f"Database has {result[0]} GBPUSD H1 bars")
        print(f"Date range: {result[1]} to {result[2]}")

        # Check 2025 data specifically
        cursor.execute("""
            SELECT COUNT(*)
            FROM ohlcv
            WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
            AND time >= '2025-01-01' AND time < '2026-01-01'
        """)
        count_2025 = cursor.fetchone()[0]
        print(f"2025 data: {count_2025} bars")
    else:
        print("No GBPUSD H1 data in database!")

    conn.close()

except Exception as e:
    print(f"Database error: {e}")

print("\n" + "="*60)
print("DONE")
print("="*60)
