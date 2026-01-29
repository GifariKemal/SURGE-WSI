"""Diagnose November 2025: Why 0 Trades?
========================================

Deep investigation into why November has no trades.

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector, MarketRegime
from src.analysis.poi_detector import POIDetector
from src.utils.killzone import KillZone


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    if not await db.connect():
        return pd.DataFrame()

    df = await db.get_ohlcv(symbol, timeframe, 50000, start, end)
    await db.disconnect()
    return df


async def main():
    """Main diagnostic function"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("\n" + "=" * 70)
    print("NOVEMBER 2025 DIAGNOSTIC")
    print("Why are there 0 trades?")
    print("=" * 70)

    symbol = "GBPUSD"

    # November 2025 date range
    start_date = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 11, 30, tzinfo=timezone.utc)
    warmup_start = start_date - timedelta(days=60)

    # Fetch data
    print("\nFetching data...")
    htf_df = await fetch_data(symbol, "H4", warmup_start, end_date)
    ltf_df = await fetch_data(symbol, "M15", warmup_start, end_date)

    if htf_df.empty:
        print("ERROR: No H4 data available for November!")
        return

    print(f"H4 bars: {len(htf_df)}, M15 bars: {len(ltf_df)}")

    # Filter to November only for analysis
    htf_nov = htf_df[htf_df.index >= start_date]
    ltf_nov = ltf_df[ltf_df.index >= start_date]

    print(f"November H4 bars: {len(htf_nov)}, M15 bars: {len(ltf_nov)}")

    # 1. CHECK DATA AVAILABILITY
    print("\n" + "-" * 50)
    print("1. DATA CHECK")
    print("-" * 50)

    if htf_nov.empty:
        print("   [PROBLEM] NO H4 DATA FOR NOVEMBER!")
        return
    else:
        print(f"   [OK] H4 data available: {htf_nov.index[0]} to {htf_nov.index[-1]}")
        print(f"   [OK] M15 data available: {ltf_nov.index[0]} to {ltf_nov.index[-1]}")

    # 2. CHECK REGIME DETECTION
    print("\n" + "-" * 50)
    print("2. REGIME DETECTION")
    print("-" * 50)

    # Initialize and warm up regime detector
    regime_detector = HMMRegimeDetector()
    kalman = MultiScaleKalman()

    # Warmup with pre-November data
    warmup_htf = htf_df[htf_df.index < start_date]
    warmup_ltf = ltf_df[ltf_df.index < start_date]

    close_col = 'close' if 'close' in warmup_ltf.columns else 'Close'

    for _, row in warmup_ltf.tail(500).iterrows():
        kalman.update(row[close_col])

    for _, row in warmup_htf.tail(100).iterrows():
        regime_detector.update(row[close_col])

    # Track regimes throughout November
    regime_counts = {'BULLISH': 0, 'BEARISH': 0, 'SIDEWAYS': 0}
    tradeable_count = 0
    non_tradeable_count = 0

    close_col_htf = 'close' if 'close' in htf_nov.columns else 'Close'
    for _, row in htf_nov.iterrows():
        regime_detector.update(row[close_col_htf])
        info = regime_detector.last_info

        if info is not None:
            regime_key = info.regime.value.upper() if hasattr(info.regime, 'value') else str(info.regime).upper()
            if regime_key in regime_counts:
                regime_counts[regime_key] += 1
            if info.is_tradeable:
                tradeable_count += 1
            else:
                non_tradeable_count += 1

    print(f"   Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(htf_nov) * 100 if len(htf_nov) > 0 else 0
        print(f"   - {regime}: {count} bars ({pct:.1f}%)")

    print(f"\n   Tradeable: {tradeable_count} bars")
    print(f"   Non-Tradeable: {non_tradeable_count} bars")

    if non_tradeable_count > tradeable_count:
        print("   [PROBLEM] Most of November is marked non-tradeable!")

    # 3. CHECK POI DETECTION
    print("\n" + "-" * 50)
    print("3. POI DETECTION")
    print("-" * 50)

    poi_detector = POIDetector()

    # Process HTF data to detect POIs
    htf_reset = htf_df.reset_index()
    htf_reset.rename(columns={'index': 'time'}, inplace=True)

    # Only up to November end
    htf_reset = htf_reset[htf_reset['time'] <= end_date]

    poi_detector.detect(htf_reset.iloc[:len(htf_reset)-1])
    poi_result = poi_detector.last_result

    if poi_result is None:
        print("   [PROBLEM] No POIs detected!")
    else:
        print(f"   Bullish POIs: {len(poi_result.bullish_pois)}")
        print(f"   Bearish POIs: {len(poi_result.bearish_pois)}")

        if poi_result.bullish_pois:
            print("   \n   Recent Bullish POIs:")
            for poi in poi_result.bullish_pois[-3:]:
                print(f"   - {poi.get('type', 'unknown')}: mid={poi.get('mid', 0):.5f}, strength={poi.get('strength', 0):.1f}")

        if poi_result.bearish_pois:
            print("   \n   Recent Bearish POIs:")
            for poi in poi_result.bearish_pois[-3:]:
                print(f"   - {poi.get('type', 'unknown')}: mid={poi.get('mid', 0):.5f}, strength={poi.get('strength', 0):.1f}")

    # 4. CHECK KILL ZONE COVERAGE
    print("\n" + "-" * 50)
    print("4. KILL ZONE COVERAGE")
    print("-" * 50)

    kz = KillZone()
    kz_count = 0
    non_kz_count = 0

    for idx in ltf_nov.index:
        in_kz, name = kz.is_in_killzone(idx)
        if in_kz:
            kz_count += 1
        else:
            non_kz_count += 1

    print(f"   Bars in Kill Zone: {kz_count} ({kz_count/(kz_count+non_kz_count)*100:.1f}%)")
    print(f"   Bars outside KZ: {non_kz_count}")

    # 5. CHECK PRICE AT POI
    print("\n" + "-" * 50)
    print("5. PRICE AT POI CHECK")
    print("-" * 50)

    close_col_ltf = 'close' if 'close' in ltf_nov.columns else 'Close'
    if poi_result:
        price_at_poi_count = 0

        for _, row in ltf_nov.iterrows():
            current_price = row[close_col_ltf]

            # Check if price is at any POI
            in_bullish_poi, _ = poi_result.price_at_poi(current_price, "BUY")
            in_bearish_poi, _ = poi_result.price_at_poi(current_price, "SELL")

            if in_bullish_poi or in_bearish_poi:
                price_at_poi_count += 1

        print(f"   M15 bars with price at POI: {price_at_poi_count}")
        if price_at_poi_count == 0:
            print("   [PROBLEM] Price never reached any detected POI during November!")

    # 6. PRICE RANGE ANALYSIS
    print("\n" + "-" * 50)
    print("6. PRICE RANGE ANALYSIS")
    print("-" * 50)

    high_col = 'high' if 'high' in htf_nov.columns else 'High'
    low_col = 'low' if 'low' in htf_nov.columns else 'Low'
    nov_high = htf_nov[high_col].max()
    nov_low = htf_nov[low_col].min()
    nov_range = (nov_high - nov_low) / 0.0001

    print(f"   November High: {nov_high:.5f}")
    print(f"   November Low: {nov_low:.5f}")
    print(f"   Total Range: {nov_range:.0f} pips")

    # Check if POIs are within November's range
    if poi_result:
        print("\n   POI vs November Range:")

        for poi in (poi_result.bullish_pois + poi_result.bearish_pois)[-5:]:
            poi_mid = poi.get('mid', 0)
            if nov_low <= poi_mid <= nov_high:
                print(f"   - {poi.get('type', 'unknown')} @ {poi_mid:.5f} [IN RANGE]")
            else:
                print(f"   - {poi.get('type', 'unknown')} @ {poi_mid:.5f} [OUT OF RANGE]")

    # 7. SUMMARY & RECOMMENDATIONS
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if non_tradeable_count > tradeable_count:
        issues.append("Most of November marked as non-tradeable by regime detector")

    if poi_result and len(poi_result.bullish_pois) + len(poi_result.bearish_pois) == 0:
        issues.append("No POIs detected")

    if poi_result:
        poi_in_range = 0
        for poi in (poi_result.bullish_pois + poi_result.bearish_pois):
            poi_mid = poi.get('mid', 0)
            if nov_low <= poi_mid <= nov_high:
                poi_in_range += 1
        if poi_in_range == 0:
            issues.append("No POIs within November's price range")

    if issues:
        print("\nISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\nRECOMMENDATIONS:")
        if "non-tradeable" in str(issues):
            print("   - Consider making regime detector less strict")
            print("   - Allow trading in sideways market with tighter stops")
        if "POI" in str(issues):
            print("   - Refresh POI detection more frequently")
            print("   - Lower POI quality threshold")
            print("   - Consider counter-trend POIs")
    else:
        print("\n   No obvious issues found. Further investigation needed.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
