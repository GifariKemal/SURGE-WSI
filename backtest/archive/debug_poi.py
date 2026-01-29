"""Debug POI Detection
======================

Debug what the POI detector and smartmoneyconcepts library returns.

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
    """Main function"""
    print("\n" + "=" * 70)
    print("POI DEBUG")
    print("=" * 70)

    symbol = "GBPUSD"
    start_date = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 11, 30, tzinfo=timezone.utc)
    warmup_start = start_date - timedelta(days=30)

    # Fetch data
    print("\nFetching H4 data...")
    htf_df = await fetch_data(symbol, "H4", warmup_start, end_date)

    if htf_df.empty:
        print("ERROR: No data!")
        return

    print(f"Total bars: {len(htf_df)}")
    print(f"Date range: {htf_df.index[0]} to {htf_df.index[-1]}")
    print(f"Price range: {htf_df['Low'].min():.5f} to {htf_df['High'].max():.5f}")

    # Normalize columns
    htf = htf_df.copy()
    htf.columns = [c.lower() for c in htf.columns]
    htf = htf.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)

    print(f"\nNormalized columns: {list(htf.columns)}")
    print(f"Sample data:\n{htf.tail(5)}")

    # Try smartmoneyconcepts
    print("\n" + "-" * 50)
    print("TESTING SMARTMONEYCONCEPTS")
    print("-" * 50)

    try:
        import io
        _original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        import smartmoneyconcepts as smc_module
        sys.stdout = _original_stdout
        smc = smc_module.smc
        print("[OK] smartmoneyconcepts imported successfully")

        # Test swing detection
        print("\n1. Swing Detection:")
        swing_hl = smc.swing_highs_lows(htf, 10)
        print(f"   Result type: {type(swing_hl)}")
        print(f"   Shape: {swing_hl.shape}")
        print(f"   Columns: {list(swing_hl.columns)}")

        # Count swings
        if 'HighLow' in swing_hl.columns:
            highs = (swing_hl['HighLow'] == 1).sum()
            lows = (swing_hl['HighLow'] == -1).sum()
            print(f"   Swing Highs: {highs}, Swing Lows: {lows}")

        # Test Order Block detection
        print("\n2. Order Block Detection:")
        ob_result = smc.ob(htf, swing_hl)
        print(f"   Result type: {type(ob_result)}")
        print(f"   Shape: {ob_result.shape}")
        print(f"   Columns: {list(ob_result.columns)}")

        if 'OB' in ob_result.columns:
            bullish_obs = (ob_result['OB'] == 1).sum()
            bearish_obs = (ob_result['OB'] == -1).sum()
            print(f"   Bullish OBs: {bullish_obs}, Bearish OBs: {bearish_obs}")

            # Show actual OB values
            ob_rows = ob_result[ob_result['OB'] != 0]
            if len(ob_rows) > 0:
                print(f"\n   Sample OB values:")
                for i, row in ob_rows.tail(5).iterrows():
                    print(f"   - idx={i}, OB={row.get('OB',0)}, "
                          f"Top={row.get('OBTop',0)}, Bottom={row.get('OBBottom',0)}")
            else:
                print("   [PROBLEM] No OBs have non-zero OB column value")

        # Test FVG detection
        print("\n3. FVG Detection:")
        fvg_result = smc.fvg(htf)
        print(f"   Result type: {type(fvg_result)}")
        print(f"   Shape: {fvg_result.shape}")
        print(f"   Columns: {list(fvg_result.columns)}")

        if 'FVG' in fvg_result.columns:
            bullish_fvgs = (fvg_result['FVG'] == 1).sum()
            bearish_fvgs = (fvg_result['FVG'] == -1).sum()
            print(f"   Bullish FVGs: {bullish_fvgs}, Bearish FVGs: {bearish_fvgs}")

            # Show actual FVG values
            fvg_rows = fvg_result[fvg_result['FVG'] != 0]
            if len(fvg_rows) > 0:
                print(f"\n   Sample FVG values:")
                for i, row in fvg_rows.tail(5).iterrows():
                    top_col = 'FVGTop' if 'FVGTop' in fvg_result.columns else 'Top'
                    bot_col = 'FVGBottom' if 'FVGBottom' in fvg_result.columns else 'Bottom'
                    print(f"   - idx={i}, FVG={row.get('FVG',0)}, "
                          f"Top={row.get(top_col,0):.5f}, Bottom={row.get(bot_col,0):.5f}")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()

    # Test fallback detection
    print("\n" + "-" * 50)
    print("TESTING FALLBACK DETECTION")
    print("-" * 50)

    from src.analysis.poi_detector import POIDetector

    # Force fallback by testing with fallback method directly
    detector = POIDetector()
    result = detector._fallback_detection(htf)

    print(f"Order Blocks: {len(result.order_blocks)}")
    print(f"FVGs: {len(result.fvgs)}")
    print(f"Swing Highs: {len(result.swing_highs)}")
    print(f"Swing Lows: {len(result.swing_lows)}")

    if result.order_blocks:
        print("\nSample OBs from fallback:")
        for ob in result.order_blocks[-5:]:
            print(f"  - {ob.poi_type.value}: top={ob.top:.5f}, bottom={ob.bottom:.5f}, "
                  f"mid={ob.mid:.5f}, mitigated={ob.mitigated}")

    if result.fvgs:
        print("\nSample FVGs from fallback:")
        for fvg in result.fvgs[-5:]:
            print(f"  - {fvg.poi_type.value}: high={fvg.high:.5f}, low={fvg.low:.5f}, "
                  f"mid={fvg.mid:.5f}")

    # Check price range
    nov_htf = htf[htf['time'] >= start_date]
    nov_high = nov_htf['high'].max()
    nov_low = nov_htf['low'].min()

    print(f"\nNovember price range: {nov_low:.5f} to {nov_high:.5f}")

    # Check if any POIs are in range
    in_range = 0
    for ob in result.order_blocks:
        if nov_low <= ob.mid <= nov_high:
            in_range += 1
    for fvg in result.fvgs:
        if nov_low <= fvg.mid <= nov_high:
            in_range += 1

    print(f"POIs in November range: {in_range}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
