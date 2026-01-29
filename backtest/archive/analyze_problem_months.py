"""Analyze Problem Months in Backtest
=====================================

Detailed analysis of why certain months performed poorly:
- March 2025: -$447 (12 trades, 58% win, PF 0.41)
- May 2025: -$258 (3 trades, 67% win, PF 0.81)
- November 2025: $0 (0 trades)
- December 2025: -$1544 (6 trades, 33% win, PF 0.20)

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
from src.analysis.regime_detector import HMMRegimeDetector
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


def analyze_market_conditions(df: pd.DataFrame, month_name: str):
    """Analyze market conditions for a month"""
    print(f"\n{'='*60}")
    print(f"MARKET CONDITIONS: {month_name}")
    print(f"{'='*60}")

    if df.empty:
        print("No data available")
        return

    # Basic stats
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'
    close_col = 'close' if 'close' in df.columns else 'Close'

    # Daily range (volatility)
    df['range'] = df[high_col] - df[low_col]
    avg_range_pips = df['range'].mean() / 0.0001

    # Price movement
    start_price = df[close_col].iloc[0]
    end_price = df[close_col].iloc[-1]
    net_move_pips = (end_price - start_price) / 0.0001

    # Trend strength (simple)
    returns = df[close_col].pct_change().dropna()
    up_days = (returns > 0).sum()
    down_days = (returns < 0).sum()

    print(f"\n1. VOLATILITY")
    print(f"   Avg H4 Range: {avg_range_pips:.1f} pips")
    print(f"   Max Range: {df['range'].max()/0.0001:.1f} pips")
    print(f"   Min Range: {df['range'].min()/0.0001:.1f} pips")

    print(f"\n2. TREND")
    print(f"   Start: {start_price:.5f}")
    print(f"   End: {end_price:.5f}")
    print(f"   Net Move: {net_move_pips:+.0f} pips")
    print(f"   Up bars: {up_days}, Down bars: {down_days}")

    # Regime analysis
    print(f"\n3. REGIME ANALYSIS")
    try:
        regime_detector = HMMRegimeDetector()
        kalman = MultiScaleKalman()

        # Apply Kalman filter
        smoothed = []
        for i in range(len(df)):
            row = df.iloc[i]
            result = kalman.update(row[close_col])
            smoothed.append(result['fused'])

        df['smoothed'] = smoothed

        # Detect regime
        regime_info = regime_detector.detect_regime(df['smoothed'].values)
        print(f"   Current Regime: {regime_info.regime.value}")
        print(f"   Confidence: {regime_info.confidence:.1%}")
        print(f"   Volatility State: {regime_info.volatility_state}")

    except Exception as e:
        print(f"   Regime detection error: {e}")

    return df


def analyze_killzones(df: pd.DataFrame, month_name: str):
    """Analyze kill zone coverage"""
    print(f"\n4. KILL ZONE ANALYSIS")

    if df.empty:
        print("   No data")
        return

    kz = KillZone()

    in_killzone = 0
    london_count = 0
    ny_count = 0

    for idx in df.index:
        if hasattr(idx, 'hour'):
            time = idx
        else:
            continue

        result = kz.is_in_killzone(time)
        if result[0]:
            in_killzone += 1
            if 'London' in result[1]:
                london_count += 1
            elif 'New York' in result[1]:
                ny_count += 1

    total_bars = len(df)
    print(f"   Total H4 bars: {total_bars}")
    print(f"   Bars in Kill Zone: {in_killzone} ({in_killzone/total_bars*100:.1f}%)")
    print(f"   London session bars: {london_count}")
    print(f"   NY session bars: {ny_count}")


def analyze_pois(htf_df: pd.DataFrame, month_name: str):
    """Analyze POI detection for the month"""
    print(f"\n5. POI ANALYSIS")

    if htf_df.empty:
        print("   No data")
        return 0

    poi_detector = POIDetector()

    # Detect POIs
    htf_reset = htf_df.reset_index()
    htf_reset.rename(columns={'index': 'time'}, inplace=True)

    try:
        pois = poi_detector.detect_all_pois(htf_reset)

        ob_count = len([p for p in pois if p.poi_type.value == 'order_block'])
        fvg_count = len([p for p in pois if p.poi_type.value == 'fvg'])

        # Quality distribution
        high_quality = len([p for p in pois if p.quality_score >= 80])
        medium_quality = len([p for p in pois if 60 <= p.quality_score < 80])
        low_quality = len([p for p in pois if p.quality_score < 60])

        print(f"   Total POIs: {len(pois)}")
        print(f"   Order Blocks: {ob_count}")
        print(f"   FVGs: {fvg_count}")
        print(f"\n   Quality Distribution:")
        print(f"   - High (>=80): {high_quality}")
        print(f"   - Medium (60-80): {medium_quality}")
        print(f"   - Low (<60): {low_quality}")

        return len(pois)

    except Exception as e:
        print(f"   POI detection error: {e}")
        return 0


def generate_diagnosis(month_name: str, trades: int, win_rate: float, pf: float,
                      net_pnl: float, poi_count: int, avg_range: float):
    """Generate diagnosis for why the month performed poorly"""
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: {month_name}")
    print(f"{'='*60}")

    issues = []

    # Check for no trades
    if trades == 0:
        issues.append("NO TRADES - Possible causes:")
        issues.append("  - No valid POIs detected")
        issues.append("  - No Kill Zone overlap with POI touches")
        issues.append("  - Regime filter blocked all entries")
        issues.append("  - LTF confirmation not triggered")

    # Check low win rate
    elif win_rate < 50:
        issues.append(f"LOW WIN RATE ({win_rate:.1f}%) - Possible causes:")
        issues.append("  - Choppy/sideways market (false breakouts)")
        issues.append("  - POI quality too low")
        issues.append("  - Entry timing off")

    # Check low profit factor
    if pf > 0 and pf < 1.0:
        issues.append(f"LOW PROFIT FACTOR ({pf:.2f}) - Possible causes:")
        issues.append("  - Losses larger than wins")
        issues.append("  - Stop losses too tight")
        issues.append("  - TP levels not reached")

    # Check high volatility
    if avg_range > 50:
        issues.append(f"HIGH VOLATILITY ({avg_range:.0f} pips avg) - Impact:")
        issues.append("  - Wider stop losses needed")
        issues.append("  - More false signals")

    # Check low volatility
    if avg_range < 20:
        issues.append(f"LOW VOLATILITY ({avg_range:.0f} pips avg) - Impact:")
        issues.append("  - Fewer trading opportunities")
        issues.append("  - TP levels harder to reach")

    # Check low POI count
    if poi_count < 10:
        issues.append(f"LOW POI COUNT ({poi_count}) - Impact:")
        issues.append("  - Fewer entry opportunities")
        issues.append("  - System may be too selective")

    for issue in issues:
        print(f"   {issue}")

    # Recommendations
    print(f"\n   RECOMMENDATIONS:")
    if trades == 0:
        print("   - Check if data exists for this period")
        print("   - Consider relaxing entry filters")
        print("   - Verify Kill Zone times are correct")
    elif win_rate < 50 or pf < 1.0:
        print("   - Consider adding trend filter")
        print("   - Increase POI quality threshold")
        print("   - Review SL/TP ratio")


async def main():
    """Main analysis function"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("\n" + "="*70)
    print("PROBLEM MONTHS ANALYSIS")
    print("Analyzing: March, May, November, December 2025")
    print("="*70)

    symbol = "GBPUSD"

    # Problem months to analyze
    problem_months = [
        (datetime(2025, 3, 1, tzinfo=timezone.utc),
         datetime(2025, 3, 31, tzinfo=timezone.utc),
         "March 2025", 12, 58.3, 0.41, -447),

        (datetime(2025, 5, 1, tzinfo=timezone.utc),
         datetime(2025, 5, 31, tzinfo=timezone.utc),
         "May 2025", 3, 66.7, 0.81, -258),

        (datetime(2025, 11, 1, tzinfo=timezone.utc),
         datetime(2025, 11, 30, tzinfo=timezone.utc),
         "November 2025", 0, 0, 0, 0),

        (datetime(2025, 12, 1, tzinfo=timezone.utc),
         datetime(2025, 12, 31, tzinfo=timezone.utc),
         "December 2025", 6, 33.3, 0.20, -1544),
    ]

    for start, end, name, trades, win_rate, pf, net_pnl in problem_months:
        # Fetch data with warmup
        warmup_start = start - timedelta(days=30)

        htf_df = await fetch_data(symbol, "H4", warmup_start, end)
        ltf_df = await fetch_data(symbol, "M15", warmup_start, end)

        if htf_df.empty:
            print(f"\n[SKIP] {name} - No data available")
            continue

        # Filter to actual month
        htf_month = htf_df[htf_df.index >= start]

        # Analyze
        df = analyze_market_conditions(htf_month, name)

        if df is not None and not df.empty:
            avg_range = df['range'].mean() / 0.0001 if 'range' in df.columns else 0
        else:
            avg_range = 0

        analyze_killzones(htf_month, name)
        poi_count = analyze_pois(htf_month, name)

        # Generate diagnosis
        generate_diagnosis(name, trades, win_rate, pf, net_pnl, poi_count, avg_range)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
