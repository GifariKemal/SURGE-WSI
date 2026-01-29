"""Step 2: Test HMM Regime Detection + Kill Zone (Layer 2)
==========================================================

Test regime detection (Bullish/Bearish/Sideways) dan Kill Zone filter.

Usage:
    python step2_regime.py

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector, MarketRegime
from src.utils.killzone import KillZone


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


def test_regime_detection(df: pd.DataFrame, month_label: str):
    """Test HMM Regime Detection"""
    print(f"\n{'='*60}")
    print(f"REGIME DETECTION TEST - {month_label}")
    print(f"{'='*60}")

    if df.empty or len(df) < 50:
        print("Insufficient data for regime detection!")
        return None

    # Initialize components
    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()
    killzone = KillZone()

    # Process data through Kalman + Regime detector
    results = []
    for i, (idx, row) in enumerate(df.iterrows()):
        price = row['Close']

        # Update Kalman filter
        kalman_state = kalman.update(price)
        smoothed_price = kalman_state['smoothed_price']

        # Update regime detector with smoothed price
        regime_info = regime_detector.update(smoothed_price)

        # Check kill zone
        in_kz, kz_name = killzone.is_in_killzone(idx)

        # Only record after warmup
        if i >= 50:
            results.append({
                'time': idx,
                'price': price,
                'regime': regime_info.regime.value if regime_info else 'unknown',
                'confidence': regime_info.confidence if regime_info else 0,
                'is_tradeable': regime_info.is_tradeable if regime_info else False,
                'in_killzone': in_kz,
                'killzone': kz_name if in_kz else 'none',
                'bias': regime_info.bias if regime_info else 'none'
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No results generated!")
        return None

    # Statistics
    print(f"\nData points analyzed: {len(results_df)}")

    print(f"\n{'-'*40}")
    print("REGIME DISTRIBUTION")
    print(f"{'-'*40}")
    regime_counts = results_df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(results_df) * 100
        print(f"{regime:12s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n{'-'*40}")
    print("TRADEABLE PERIODS")
    print(f"{'-'*40}")
    tradeable = results_df['is_tradeable'].sum()
    print(f"Tradeable:     {tradeable} ({tradeable/len(results_df)*100:.1f}%)")
    print(f"Not Tradeable: {len(results_df) - tradeable}")

    print(f"\n{'-'*40}")
    print("KILL ZONE FILTER")
    print(f"{'-'*40}")
    in_kz = results_df['in_killzone'].sum()
    print(f"In Kill Zone:      {in_kz} ({in_kz/len(results_df)*100:.1f}%)")
    print(f"Outside Kill Zone: {len(results_df) - in_kz}")

    kz_dist = results_df[results_df['in_killzone']]['killzone'].value_counts()
    for kz, count in kz_dist.items():
        print(f"  {kz}: {count}")

    print(f"\n{'-'*40}")
    print("CONFIDENCE STATISTICS")
    print(f"{'-'*40}")
    print(f"Mean confidence:   {results_df['confidence'].mean()*100:.1f}%")
    print(f"High conf (>70%):  {(results_df['confidence'] > 0.7).sum()}")
    print(f"Low conf (<50%):   {(results_df['confidence'] < 0.5).sum()}")

    print(f"\n{'-'*40}")
    print("COMBINED FILTER (Tradeable + In Kill Zone)")
    print(f"{'-'*40}")
    both = ((results_df['is_tradeable']) & (results_df['in_killzone'])).sum()
    print(f"Trading opportunities: {both} ({both/len(results_df)*100:.1f}%)")

    return {
        'month': month_label,
        'bars': len(results_df),
        'bullish_pct': (results_df['regime'] == 'bullish').mean() * 100,
        'bearish_pct': (results_df['regime'] == 'bearish').mean() * 100,
        'sideways_pct': (results_df['regime'] == 'sideways').mean() * 100,
        'tradeable_pct': results_df['is_tradeable'].mean() * 100,
        'in_kz_pct': results_df['in_killzone'].mean() * 100,
        'opportunities': both
    }


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("\n" + "=" * 70)
    print("SURGE-WSI STEP 2: REGIME + KILL ZONE TEST")
    print("Layer 2 - When to Trade")
    print("=" * 70)

    symbol = "GBPUSD"
    timeframe = "H1"

    # Fetch data
    print(f"\nFetching {symbol} {timeframe} data from database...")
    df = await fetch_data(symbol, timeframe, limit=10000)

    if df.empty:
        print("ERROR: No data available. Run sync_mt5_data.py first.")
        return

    print(f"Loaded {len(df)} bars")

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

    all_results = []

    for month_prefix, month_label in months:
        month_df = df[df.index.strftime('%Y-%m') == month_prefix]

        if month_df.empty:
            print(f"\n[SKIP] {month_label} - No data")
            continue

        result = test_regime_detection(month_df, month_label)
        if result:
            all_results.append(result)

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("MONTHLY SUMMARY")
        print("=" * 70)

        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))

        print(f"\n{'-'*40}")
        print("OVERALL STATISTICS")
        print(f"{'-'*40}")
        print(f"Avg Bullish %:     {summary_df['bullish_pct'].mean():.1f}%")
        print(f"Avg Bearish %:     {summary_df['bearish_pct'].mean():.1f}%")
        print(f"Avg Sideways %:    {summary_df['sideways_pct'].mean():.1f}%")
        print(f"Avg Tradeable %:   {summary_df['tradeable_pct'].mean():.1f}%")
        print(f"Total Opportunities: {summary_df['opportunities'].sum()}")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE - Regime Detection Working")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
