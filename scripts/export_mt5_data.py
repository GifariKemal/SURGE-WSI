"""
Export MT5 Historical Data to CSV
=================================
Exports OHLCV data from MT5 for use in Python backtest.
This ensures both MQL5 EA and Python use identical data.

Usage:
    python scripts/export_mt5_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import pandas as pd
from src.data.mt5_connector import MT5Connector

# Configuration
SYMBOL = "GBPUSD"
TIMEFRAME = "H1"
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 1, 31, 23, 59, 59)
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "mt5_export"


def main():
    print("=" * 60)
    print("MT5 Historical Data Export")
    print("=" * 60)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to MT5
    connector = MT5Connector()
    if not connector.connect():
        print("ERROR: Failed to connect to MT5")
        print("Make sure MT5 terminal is running and logged in")
        return False

    print("Connected to MT5")

    # Fetch data
    print(f"Fetching {SYMBOL} {TIMEFRAME} data...")
    df = connector.get_ohlcv_range(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE
    )

    if df is None or df.empty:
        print("ERROR: No data received from MT5")
        connector.disconnect()
        return False

    print(f"Received {len(df)} bars")

    # Show data info
    print(f"\nData range:")
    print(f"  First bar: {df.index[0]}")
    print(f"  Last bar: {df.index[-1]}")
    print(f"\nSample data (first 5 bars):")
    print(df.head().to_string())

    # Prepare for CSV export
    # Reset index to make timestamp a column
    df_export = df.reset_index()

    # Get actual column names and rename appropriately
    print(f"\nOriginal columns: {list(df_export.columns)}")

    # Standard column mapping
    df_export = df_export.rename(columns={
        'time': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Spread': 'spread'
    })

    # Keep only needed columns
    df_export = df_export[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

    # Convert timestamp to string for CSV
    df_export['timestamp'] = df_export['timestamp'].astype(str)

    # Save to CSV
    output_file = OUTPUT_DIR / f"{SYMBOL}_{TIMEFRAME}_{START_DATE.strftime('%Y%m%d')}_{END_DATE.strftime('%Y%m%d')}.csv"
    df_export.to_csv(output_file, index=False)
    print(f"\nData exported to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Also save a simple version for quick reference
    simple_file = OUTPUT_DIR / f"{SYMBOL}_{TIMEFRAME}_latest.csv"
    df_export.to_csv(simple_file, index=False)
    print(f"Also saved to: {simple_file}")

    # Disconnect
    connector.disconnect()
    print("\nExport complete!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
