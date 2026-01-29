"""Step 1: Test Kalman Filter (Layer 1 - Data Pipeline)
========================================================

Test noise reduction dan velocity/acceleration extraction.

Usage:
    python step1_kalman.py

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import MultiScaleKalman, KalmanState


async def fetch_data(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    if not await db.connect():
        logger.error("Failed to connect to database")
        return pd.DataFrame()

    df = await db.get_ohlcv(symbol, timeframe, limit=limit)
    await db.disconnect()

    return df


def test_kalman_filter(df: pd.DataFrame, month_label: str):
    """Test Kalman Filter on data"""
    print(f"\n{'='*60}")
    print(f"KALMAN FILTER TEST - {month_label}")
    print(f"{'='*60}")

    if df.empty:
        print("No data available!")
        return None

    # Initialize Kalman
    kalman = MultiScaleKalman()

    # Process each price
    states = []
    for i, (idx, row) in enumerate(df.iterrows()):
        price = row['Close']
        state = kalman.update(price)
        # MultiScaleKalman.update() returns a Dict, not KalmanState
        states.append({
            'time': idx,
            'raw_price': price,
            'smoothed': state['smoothed_price'],
            'velocity': state['velocity'],
            'acceleration': state['acceleration'],
            'noise_level': state['noise_level']
        })

    states_df = pd.DataFrame(states)

    # Statistics
    print(f"\nData points: {len(states_df)}")
    print(f"Date range: {states_df['time'].iloc[0]} to {states_df['time'].iloc[-1]}")

    print(f"\n{'-'*40}")
    print("PRICE STATISTICS")
    print(f"{'-'*40}")
    print(f"Raw Price  - Min: {states_df['raw_price'].min():.5f}, Max: {states_df['raw_price'].max():.5f}")
    print(f"Smoothed   - Min: {states_df['smoothed'].min():.5f}, Max: {states_df['smoothed'].max():.5f}")

    # Noise reduction effectiveness
    raw_std = states_df['raw_price'].diff().std()
    smooth_std = states_df['smoothed'].diff().std()
    noise_reduction = (1 - smooth_std/raw_std) * 100 if raw_std > 0 else 0

    print(f"\n{'-'*40}")
    print("NOISE REDUCTION")
    print(f"{'-'*40}")
    print(f"Raw price volatility:      {raw_std:.6f}")
    print(f"Smoothed price volatility: {smooth_std:.6f}")
    print(f"Noise reduction:           {noise_reduction:.1f}%")

    print(f"\n{'-'*40}")
    print("VELOCITY STATISTICS")
    print(f"{'-'*40}")
    print(f"Mean velocity:     {states_df['velocity'].mean():.6f}")
    print(f"Velocity std:      {states_df['velocity'].std():.6f}")
    print(f"Max velocity:      {states_df['velocity'].max():.6f}")
    print(f"Min velocity:      {states_df['velocity'].min():.6f}")

    # Direction analysis
    positive_velocity = (states_df['velocity'] > 0).sum()
    negative_velocity = (states_df['velocity'] < 0).sum()

    print(f"\n{'-'*40}")
    print("DIRECTION ANALYSIS")
    print(f"{'-'*40}")
    print(f"Bullish periods (v > 0): {positive_velocity} ({positive_velocity/len(states_df)*100:.1f}%)")
    print(f"Bearish periods (v < 0): {negative_velocity} ({negative_velocity/len(states_df)*100:.1f}%)")

    print(f"\n{'-'*40}")
    print("ACCELERATION STATISTICS")
    print(f"{'-'*40}")
    print(f"Mean acceleration: {states_df['acceleration'].mean():.8f}")
    print(f"Acceleration std:  {states_df['acceleration'].std():.8f}")

    print(f"\n{'-'*40}")
    print("NOISE LEVEL")
    print(f"{'-'*40}")
    print(f"Avg noise level:   {states_df['noise_level'].mean():.6f}")
    print(f"Max noise level:   {states_df['noise_level'].max():.6f}")

    return states_df


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("\n" + "=" * 70)
    print("SURGE-WSI STEP 1: KALMAN FILTER TEST")
    print("Layer 1 - Data Pipeline / Noise Reduction")
    print("=" * 70)

    symbol = "GBPUSD"
    timeframe = "H1"  # Test on H1 for speed

    # Fetch data
    print(f"\nFetching {symbol} {timeframe} data from database...")
    df = await fetch_data(symbol, timeframe, limit=5000)

    if df.empty:
        print("ERROR: No data available. Run sync_mt5_data.py first.")
        return

    print(f"Loaded {len(df)} bars")

    # Monthly breakdown - test Jan 2025 to Jan 2026
    months = [
        ("2025-01", "January 2025"),
        ("2025-02", "February 2025"),
        ("2025-03", "March 2025"),
        ("2025-04", "April 2025"),
        ("2025-05", "May 2025"),
        ("2025-06", "June 2025"),
        ("2025-07", "July 2025"),
        ("2025-08", "August 2025"),
        ("2025-09", "September 2025"),
        ("2025-10", "October 2025"),
        ("2025-11", "November 2025"),
        ("2025-12", "December 2025"),
        ("2026-01", "January 2026"),
    ]

    all_results = []

    for month_prefix, month_label in months:
        # Filter data for month
        month_df = df[df.index.strftime('%Y-%m') == month_prefix]

        if month_df.empty:
            print(f"\n[SKIP] {month_label} - No data")
            continue

        result = test_kalman_filter(month_df, month_label)
        if result is not None:
            all_results.append({
                'month': month_label,
                'bars': len(result),
                'noise_reduction': (1 - result['smoothed'].diff().std() / result['raw_price'].diff().std()) * 100,
                'avg_velocity': result['velocity'].mean(),
                'bullish_pct': (result['velocity'] > 0).mean() * 100
            })

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("MONTHLY SUMMARY")
        print("=" * 70)

        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))

        print(f"\n{'-'*40}")
        print("OVERALL")
        print(f"{'-'*40}")
        print(f"Avg Noise Reduction: {summary_df['noise_reduction'].mean():.1f}%")
        print(f"Avg Bullish %:       {summary_df['bullish_pct'].mean():.1f}%")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE - Kalman Filter Working")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
