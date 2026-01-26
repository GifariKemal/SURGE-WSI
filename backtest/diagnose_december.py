"""Diagnose December 2025: Why -$1907?
========================================

December has 18 trades but only 38.9% win rate.
Let's find out why.

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
from src.analysis.market_filter import MarketFilter


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
    print("DECEMBER 2025 DIAGNOSTIC")
    print("18 trades, 38.9% win rate, -$1907")
    print("=" * 70)

    symbol = "GBPUSD"
    start_date = datetime(2025, 12, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 12, 31, tzinfo=timezone.utc)
    warmup_start = start_date - timedelta(days=60)

    # Fetch data
    print("\nFetching data...")
    htf_df = await fetch_data(symbol, "H4", warmup_start, end_date)
    ltf_df = await fetch_data(symbol, "M15", warmup_start, end_date)

    if htf_df.empty:
        print("ERROR: No H4 data!")
        return

    print(f"H4 bars: {len(htf_df)}, M15 bars: {len(ltf_df)}")

    # Filter to December only
    htf_dec = htf_df[htf_df.index >= start_date]
    ltf_dec = ltf_df[ltf_df.index >= start_date]

    high_col = 'High' if 'High' in htf_dec.columns else 'high'
    low_col = 'Low' if 'Low' in htf_dec.columns else 'low'
    close_col = 'Close' if 'Close' in htf_dec.columns else 'close'

    print(f"December H4 bars: {len(htf_dec)}, M15 bars: {len(ltf_dec)}")

    # 1. CHECK MARKET CONDITIONS
    print("\n" + "-" * 50)
    print("1. MARKET CONDITIONS")
    print("-" * 50)

    dec_high = htf_dec[high_col].max()
    dec_low = htf_dec[low_col].min()
    dec_range = (dec_high - dec_low) / 0.0001
    start_price = htf_dec[close_col].iloc[0]
    end_price = htf_dec[close_col].iloc[-1]
    net_move = (end_price - start_price) / 0.0001

    print(f"   December High: {dec_high:.5f}")
    print(f"   December Low: {dec_low:.5f}")
    print(f"   Total Range: {dec_range:.0f} pips")
    print(f"   Start: {start_price:.5f}")
    print(f"   End: {end_price:.5f}")
    print(f"   Net Move: {net_move:+.0f} pips")

    # 2. CHECK VOLATILITY (ATR)
    print("\n" + "-" * 50)
    print("2. VOLATILITY ANALYSIS")
    print("-" * 50)

    market_filter = MarketFilter()
    htf_reset = htf_dec.reset_index()
    htf_reset.columns = [c.lower() for c in htf_reset.columns]
    htf_reset.rename(columns={'index': 'time'}, inplace=True)

    atr_pips = market_filter.calculate_atr(htf_reset)
    print(f"   ATR (14 period): {atr_pips:.1f} pips")

    if atr_pips > 50:
        print("   [WARNING] High volatility period!")
    elif atr_pips < 15:
        print("   [INFO] Low volatility period")
    else:
        print("   [OK] Normal volatility")

    # 3. CHECK TREND ANALYSIS
    print("\n" + "-" * 50)
    print("3. TREND ANALYSIS")
    print("-" * 50)

    trend_strength, trend_dir = market_filter.analyze_trend(htf_reset)
    print(f"   Trend Direction: {trend_dir}")
    print(f"   Trend Strength: {trend_strength:.1f}%")

    if trend_dir == 'none':
        print("   [WARNING] Choppy/sideways market!")

    # Check up/down bars distribution
    returns = htf_dec[close_col].pct_change().dropna()
    up_bars = (returns > 0).sum()
    down_bars = (returns < 0).sum()
    print(f"   Up bars: {up_bars}, Down bars: {down_bars}")

    # 4. CHECK REGIME DETECTION
    print("\n" + "-" * 50)
    print("4. REGIME DETECTION")
    print("-" * 50)

    regime_detector = HMMRegimeDetector()
    kalman = MultiScaleKalman()

    # Warmup
    warmup_htf = htf_df[htf_df.index < start_date]
    warmup_ltf = ltf_df[ltf_df.index < start_date]

    for _, row in warmup_ltf.tail(500).iterrows():
        kalman.update(row[close_col])

    for _, row in warmup_htf.tail(100).iterrows():
        regime_detector.update(row[close_col])

    # Track regimes
    regime_counts = {'BULLISH': 0, 'BEARISH': 0, 'SIDEWAYS': 0}
    tradeable_count = 0
    regime_changes = 0
    prev_regime = None

    for _, row in htf_dec.iterrows():
        regime_detector.update(row[close_col])
        info = regime_detector.last_info

        if info is not None:
            regime_key = info.regime.value.upper()
            if regime_key in regime_counts:
                regime_counts[regime_key] += 1
            if info.is_tradeable:
                tradeable_count += 1

            # Count regime changes
            if prev_regime is not None and prev_regime != info.regime:
                regime_changes += 1
            prev_regime = info.regime

    print(f"   Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(htf_dec) * 100 if len(htf_dec) > 0 else 0
        print(f"   - {regime}: {count} bars ({pct:.1f}%)")

    print(f"\n   Tradeable bars: {tradeable_count}")
    print(f"   Regime changes: {regime_changes}")

    if regime_changes > 5:
        print("   [WARNING] Many regime changes = whipsaw conditions!")

    # 5. DIAGNOSIS SUMMARY
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if trend_dir == 'none':
        issues.append("Choppy/sideways market - many false breakouts expected")

    if regime_changes > 5:
        issues.append(f"{regime_changes} regime changes - whipsaw conditions")

    if atr_pips > 50:
        issues.append(f"High volatility ({atr_pips:.0f} pips ATR) - wider stops needed")

    if abs(net_move) < 100 and dec_range > 300:
        issues.append("Large range but small net move - ranging market")

    if issues:
        print("\nISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\nRECOMMENDATIONS:")
        print("   - Skip trading in choppy conditions (trend_dir == 'none')")
        print("   - Reduce position size during high volatility")
        print("   - Wait for clearer trend confirmation")
        print("   - Consider adding a 'confidence threshold' for regime")
    else:
        print("\n   No obvious issues found.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
