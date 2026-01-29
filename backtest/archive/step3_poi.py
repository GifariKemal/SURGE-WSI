"""Step 3: Test POI Detection (Layer 3)
========================================

Test Order Blocks dan Fair Value Gaps detection.

Usage:
    python step3_poi.py

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
from src.analysis.poi_detector import POIDetector, POIResult


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


def test_poi_detection(df: pd.DataFrame, month_label: str):
    """Test POI Detection"""
    print(f"\n{'='*60}")
    print(f"POI DETECTION TEST - {month_label}")
    print(f"{'='*60}")

    if df.empty or len(df) < 50:
        print("Insufficient data for POI detection!")
        return None

    # Initialize detector
    detector = POIDetector(
        swing_length=10,
        ob_min_strength=0.6,
        fvg_min_pips=3.0,
        max_poi_age_bars=100
    )

    # Convert to format expected by detector
    df_reset = df.reset_index()
    df_reset.rename(columns={'index': 'time'}, inplace=True)

    # Detect POIs
    result = detector.detect(df_reset)

    if result is None:
        print("No POI detection result!")
        return None

    # Analyze results
    print(f"\nData analyzed: {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Get raw OBs and FVGs from result
    bullish_obs = [ob for ob in result.order_blocks if 'BULL' in ob.poi_type.value]
    bearish_obs = [ob for ob in result.order_blocks if 'BEAR' in ob.poi_type.value]
    bullish_fvgs = [fvg for fvg in result.fvgs if 'BULL' in fvg.poi_type.value]
    bearish_fvgs = [fvg for fvg in result.fvgs if 'BEAR' in fvg.poi_type.value]

    print(f"\n{'-'*40}")
    print("ORDER BLOCKS DETECTED")
    print(f"{'-'*40}")
    print(f"Bullish OBs: {len(bullish_obs)}")
    print(f"Bearish OBs: {len(bearish_obs)}")

    if bullish_obs:
        print("\n  Top 3 Bullish OBs:")
        for i, ob in enumerate(bullish_obs[:3]):
            print(f"    {i+1}. Price: {ob.mid:.5f}, Strength: {ob.strength:.2f}")

    if bearish_obs:
        print("\n  Top 3 Bearish OBs:")
        for i, ob in enumerate(bearish_obs[:3]):
            print(f"    {i+1}. Price: {ob.mid:.5f}, Strength: {ob.strength:.2f}")

    print(f"\n{'-'*40}")
    print("FAIR VALUE GAPS DETECTED")
    print(f"{'-'*40}")
    print(f"Bullish FVGs: {len(bullish_fvgs)}")
    print(f"Bearish FVGs: {len(bearish_fvgs)}")

    if bullish_fvgs:
        print("\n  Top 3 Bullish FVGs:")
        for i, fvg in enumerate(bullish_fvgs[:3]):
            print(f"    {i+1}. Price: {fvg.mid:.5f}, Size: {fvg.size_pips:.1f} pips")

    if bearish_fvgs:
        print("\n  Top 3 Bearish FVGs:")
        for i, fvg in enumerate(bearish_fvgs[:3]):
            print(f"    {i+1}. Price: {fvg.mid:.5f}, Size: {fvg.size_pips:.1f} pips")

    print(f"\n{'-'*40}")
    print("QUALITY ANALYSIS")
    print(f"{'-'*40}")

    all_obs = bullish_obs + bearish_obs
    if all_obs:
        strengths = [ob.strength for ob in all_obs]
        print(f"OB Strength - Mean: {np.mean(strengths):.2f}, Max: {np.max(strengths):.2f}")
        high_quality = sum(1 for s in strengths if s >= 0.7)
        print(f"High quality OBs (>70%): {high_quality}")

    all_fvgs = bullish_fvgs + bearish_fvgs
    if all_fvgs:
        sizes = [fvg.size_pips for fvg in all_fvgs]
        print(f"FVG Size - Mean: {np.mean(sizes):.1f} pips, Max: {np.max(sizes):.1f} pips")

    # Calculate potential entries
    current_price = df['Close'].iloc[-1]
    print(f"\n{'-'*40}")
    print(f"POTENTIAL ZONES (Current price: {current_price:.5f})")
    print(f"{'-'*40}")

    nearby_buy = [ob for ob in bullish_obs if abs(ob.mid - current_price) < 0.005]
    nearby_sell = [ob for ob in bearish_obs if abs(ob.mid - current_price) < 0.005]

    print(f"Nearby BUY zones (<50 pips): {len(nearby_buy)}")
    print(f"Nearby SELL zones (<50 pips): {len(nearby_sell)}")

    return {
        'month': month_label,
        'bars': len(df),
        'bullish_obs': len(bullish_obs),
        'bearish_obs': len(bearish_obs),
        'bullish_fvgs': len(bullish_fvgs),
        'bearish_fvgs': len(bearish_fvgs),
        'total_pois': len(all_obs) + len(all_fvgs),
        'avg_ob_strength': np.mean([ob.strength for ob in all_obs]) if all_obs else 0
    }


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("\n" + "=" * 70)
    print("SURGE-WSI STEP 3: POI DETECTION TEST")
    print("Layer 3 - Where to Trade (Order Blocks + FVG)")
    print("=" * 70)

    symbol = "GBPUSD"

    # Test on H4 (main POI timeframe)
    print(f"\nFetching {symbol} H4 data from database...")
    h4_df = await fetch_data(symbol, "H4", limit=5000)

    if h4_df.empty:
        print("ERROR: No H4 data available.")
        return

    print(f"Loaded {len(h4_df)} H4 bars")

    # Also test H1 for comparison
    h1_df = await fetch_data(symbol, "H1", limit=10000)
    print(f"Loaded {len(h1_df)} H1 bars")

    # Monthly breakdown
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

    print("\n" + "=" * 70)
    print("H4 POI DETECTION (Major Zones)")
    print("=" * 70)

    h4_results = []
    for month_prefix, month_label in months:
        month_df = h4_df[h4_df.index.strftime('%Y-%m') == month_prefix]

        if month_df.empty or len(month_df) < 50:
            print(f"\n[SKIP] {month_label} - Insufficient data")
            continue

        result = test_poi_detection(month_df, f"H4 {month_label}")
        if result:
            h4_results.append(result)

    print("\n" + "=" * 70)
    print("H1 POI DETECTION (Swing Zones)")
    print("=" * 70)

    h1_results = []
    for month_prefix, month_label in months:
        month_df = h1_df[h1_df.index.strftime('%Y-%m') == month_prefix]

        if month_df.empty or len(month_df) < 50:
            continue

        result = test_poi_detection(month_df, f"H1 {month_label}")
        if result:
            h1_results.append(result)

    # Summary
    if h4_results:
        print("\n" + "=" * 70)
        print("H4 MONTHLY SUMMARY")
        print("=" * 70)
        summary_df = pd.DataFrame(h4_results)
        print(summary_df.to_string(index=False))

        print(f"\nTotal H4 POIs detected: {summary_df['total_pois'].sum()}")

    if h1_results:
        print("\n" + "=" * 70)
        print("H1 MONTHLY SUMMARY")
        print("=" * 70)
        summary_df = pd.DataFrame(h1_results)
        print(summary_df.to_string(index=False))

        print(f"\nTotal H1 POIs detected: {summary_df['total_pois'].sum()}")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE - POI Detection Working")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
