"""Detailed Debug Backtest: January 26-28, 2026
================================================

Investigates WHY no trades were triggered with verbose output for each layer.

Usage:
    python -m backtest.debug_jan26_28_detailed

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding issues on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

# Output file path
OUTPUT_FILE = Path(__file__).parent / "results" / "debug_jan26_28_detailed.txt"


async def fetch_data(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Fetch data from database for date range"""
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

    df = await db.get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=50000,
        start_time=start_date,
        end_time=end_date
    )
    await db.disconnect()

    return df


def log_output(msg: str, file_handle=None):
    """Print and optionally log to file"""
    print(msg)
    if file_handle:
        file_handle.write(msg + "\n")
        file_handle.flush()


async def investigate_period(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    output_file=None
):
    """Investigate each layer of the trading system"""
    
    log_output(f"\n{'='*70}", output_file)
    log_output("DETAILED INVESTIGATION: Why No Trades?", output_file)
    log_output(f"Period: {start_date.date()} to {end_date.date()}", output_file)
    log_output(f"{'='*70}", output_file)

    # Prepare data with 'time' column
    htf = htf_df.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)
    htf.columns = [c.lower() for c in htf.columns]

    ltf = ltf_df.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)
    ltf.columns = [c.lower() for c in ltf.columns]

    # Filter data for backtest period only
    htf_period = htf[htf['time'] >= start_date].copy()
    ltf_period = ltf[ltf['time'] >= start_date].copy()
    
    log_output(f"\nData available in backtest period:", output_file)
    log_output(f"  HTF (H4) bars: {len(htf_period)}", output_file)
    log_output(f"  LTF (M15) bars: {len(ltf_period)}", output_file)
    
    if htf_period.empty or ltf_period.empty:
        log_output("\nNO DATA in backtest period! Check database.", output_file)
        return
    
    # ============================================================
    # LAYER 1: CHECK KILL ZONES
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("LAYER 1: KILL ZONE CHECK", output_file)
    log_output(f"{'='*70}", output_file)
    
    killzone = KillZone()
    kz_stats = {"london": 0, "new_york": 0, "overlap": 0, "outside": 0}
    
    # Check each M15 bar for killzone
    kz_bars = []
    for idx, row in ltf_period.iterrows():
        bar_time = row['time']
        is_in_kz, session = killzone.is_in_killzone(bar_time)
        if is_in_kz:
            kz_bars.append((bar_time, session))
            session_lower = session.lower() if session else ""
            if "london" in session_lower:
                kz_stats["london"] += 1
            if "new_york" in session_lower or "ny" in session_lower:
                kz_stats["new_york"] += 1
            if "overlap" in session_lower:
                kz_stats["overlap"] += 1
        else:
            kz_stats["outside"] += 1
    
    log_output(f"\nKill Zone Distribution (M15 bars):", output_file)
    log_output(f"  London Session:   {kz_stats['london']} bars", output_file)
    log_output(f"  New York Session: {kz_stats['new_york']} bars", output_file)
    log_output(f"  Overlap:          {kz_stats['overlap']} bars", output_file)
    log_output(f"  Outside KZ:       {kz_stats['outside']} bars", output_file)
    
    total_kz = kz_stats['london'] + kz_stats['new_york']
    log_output(f"\n  Total in Kill Zones: {total_kz} bars", output_file)
    
    if kz_bars:
        log_output(f"\n  First KZ bar: {kz_bars[0][0]} ({kz_bars[0][1]})", output_file)
        log_output(f"  Last KZ bar:  {kz_bars[-1][0]} ({kz_bars[-1][1]})", output_file)
    
    # ============================================================
    # LAYER 2: CHECK REGIME DETECTION
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("LAYER 2: REGIME DETECTION (HMM)", output_file)
    log_output(f"{'='*70}", output_file)
    
    # Initialize components
    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()
    
    # Warmup with all HTF data before start_date
    warmup_htf = htf[htf['time'] < start_date]
    for _, row in warmup_htf.iterrows():
        kalman.update(row['close'])
        regime_detector.update(row['close'])
    
    log_output(f"  Warmed up with {len(warmup_htf)} HTF bars", output_file)
    
    # Check regime for each HTF bar in period
    regime_history = []
    for _, row in htf_period.iterrows():
        kalman_result = kalman.update(row['close'])
        regime_info = regime_detector.update(row['close'])
        
        if regime_info:
            regime_history.append({
                'time': row['time'],
                'close': row['close'],
                'velocity': kalman_result.get('velocity', 0) if isinstance(kalman_result, dict) else 0,
                'regime': regime_info.regime.value if hasattr(regime_info, 'regime') else str(regime_info),
                'confidence': regime_info.probability if hasattr(regime_info, 'probability') else 0,
                'tradeable': regime_info.is_tradeable if hasattr(regime_info, 'is_tradeable') else False
            })
    
    log_output(f"\nRegime Analysis for {len(regime_history)} HTF bars:", output_file)
    
    regime_counts = {"BULLISH": 0, "BEARISH": 0, "SIDEWAYS": 0}
    tradeable_bars = 0
    for r in regime_history:
        regime_label = r['regime']
        if regime_label in regime_counts:
            regime_counts[regime_label] += 1
        else:
            regime_counts["SIDEWAYS"] += 1
        if r['tradeable']:
            tradeable_bars += 1
    
    log_output(f"\n  Regime Distribution:", output_file)
    log_output(f"    BULLISH:  {regime_counts['BULLISH']} bars", output_file)
    log_output(f"    BEARISH:  {regime_counts['BEARISH']} bars", output_file)
    log_output(f"    SIDEWAYS: {regime_counts['SIDEWAYS']} bars", output_file)
    log_output(f"    Tradeable: {tradeable_bars} bars", output_file)
    
    log_output(f"\n  Individual Bar Regimes:", output_file)
    for r in regime_history[:20]:  # Show first 20
        log_output(f"    {r['time']} | Close: {r['close']:.5f} | Regime: {r['regime']} | Conf: {r['confidence']:.2%} | Trade: {r['tradeable']}", output_file)
    
    # ============================================================
    # LAYER 3: POI DETECTION
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("LAYER 3: POI DETECTION (Order Blocks & FVG)", output_file)
    log_output(f"{'='*70}", output_file)
    
    poi_detector = POIDetector()
    
    try:
        # Detect POIs on full HTF data (needs history for pattern detection)
        poi_result = poi_detector.detect(htf)
        
        if poi_result:
            obs = poi_result.order_blocks if hasattr(poi_result, 'order_blocks') else []
            fvgs = poi_result.fvgs if hasattr(poi_result, 'fvgs') else []
            
            log_output(f"\n  Total Order Blocks: {len(obs)}", output_file)
            log_output(f"  Total FVGs: {len(fvgs)}", output_file)
            
            # Show recent order blocks
            if obs:
                log_output(f"\n  Recent Order Blocks:", output_file)
                for ob in obs[-5:]:  # Last 5
                    ob_dict = ob.to_dict() if hasattr(ob, 'to_dict') else ob
                    log_output(f"    Type: {ob_dict.get('poi_type', 'N/A')} | "
                              f"Top: {ob_dict.get('top', 0):.5f} | "
                              f"Bottom: {ob_dict.get('bottom', 0):.5f} | "
                              f"Strength: {ob_dict.get('strength', 0):.2f}", output_file)
            
            # Show recent FVGs
            if fvgs:
                log_output(f"\n  Recent FVGs:", output_file)
                for fvg in fvgs[-5:]:  # Last 5
                    fvg_dict = fvg.to_dict() if hasattr(fvg, 'to_dict') else fvg
                    log_output(f"    Type: {fvg_dict.get('poi_type', 'N/A')} | "
                              f"High: {fvg_dict.get('high', 0):.5f} | "
                              f"Low: {fvg_dict.get('low', 0):.5f}", output_file)
            
            # Check if current price is at any POI
            if ltf_period is not None and len(ltf_period) > 0:
                current_price = ltf_period['close'].iloc[-1]
                
                # Check for BUY POI (bullish)
                bullish_pois = poi_result.bullish_pois if hasattr(poi_result, 'bullish_pois') else []
                bearish_pois = poi_result.bearish_pois if hasattr(poi_result, 'bearish_pois') else []
                
                log_output(f"\n  Current price: {current_price:.5f}", output_file)
                log_output(f"  Active Bullish POIs: {len(bullish_pois)}", output_file)
                log_output(f"  Active Bearish POIs: {len(bearish_pois)}", output_file)
                
                at_bullish, bullish_poi = poi_result.price_at_poi(current_price, "BUY")
                at_bearish, bearish_poi = poi_result.price_at_poi(current_price, "SELL")
                
                log_output(f"  Price at Bullish POI: {at_bullish}", output_file)
                log_output(f"  Price at Bearish POI: {at_bearish}", output_file)
                
        else:
            log_output(f"\n  No POIs detected - this could be the reason for no trades!", output_file)
            
    except Exception as e:
        log_output(f"  ERROR detecting POIs: {e}", output_file)
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # LAYER 4: ENTRY TRIGGER CHECK (Rejection Candles)
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("LAYER 4: ENTRY TRIGGER (Rejection Candles)", output_file)
    log_output(f"{'='*70}", output_file)
    
    # Check for rejection candles in LTF data during KZ
    rejection_candles = []
    for bar_time, session in kz_bars:
        # Get candle data
        ltf_bar = ltf_period[ltf_period['time'] == bar_time]
        if ltf_bar.empty:
            continue
        
        bar = ltf_bar.iloc[0]
        open_price = bar['open']
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        # Calculate rejection metrics
        body = abs(close - open_price)
        total_range = high - low
        
        if total_range > 0:
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low
            
            # Check for rejection (wick > 50% of range)
            if upper_wick > 0.5 * total_range:
                rejection_candles.append({
                    'time': bar_time,
                    'type': 'BEARISH_REJECTION',
                    'wick_ratio': upper_wick / total_range,
                    'session': session
                })
            elif lower_wick > 0.5 * total_range:
                rejection_candles.append({
                    'time': bar_time,
                    'type': 'BULLISH_REJECTION',
                    'wick_ratio': lower_wick / total_range,
                    'session': session
                })
    
    log_output(f"\n  Rejection candles found in Kill Zones: {len(rejection_candles)}", output_file)
    
    if rejection_candles:
        log_output(f"\n  Sample rejection candles:", output_file)
        for rc in rejection_candles[:15]:  # Show first 15
            log_output(f"    {rc['time']} | {rc['type']} | Wick: {rc['wick_ratio']:.2%} | {rc['session']}", output_file)
    else:
        log_output(f"\n  No rejection candles found - this could be the reason for no trades!", output_file)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("INVESTIGATION SUMMARY", output_file)
    log_output(f"{'='*70}", output_file)
    
    issues = []
    
    if total_kz == 0:
        issues.append("- NO bars in Kill Zones (trading hours)")
    elif total_kz < 10:
        issues.append(f"- Only {total_kz} bars in Kill Zones (may be limited opportunity)")
    
    if tradeable_bars == 0:
        issues.append("- NO bars with tradeable regime (need BULLISH or BEARISH with high confidence)")
    
    if regime_counts['SIDEWAYS'] > (regime_counts['BULLISH'] + regime_counts['BEARISH']):
        issues.append("- Market primarily in SIDEWAYS regime (no clear trend)")
    
    if poi_result and len(obs) == 0 and len(fvgs) == 0:
        issues.append("- NO POIs (Order Blocks/FVG) detected")
    
    if len(rejection_candles) == 0:
        issues.append("- NO rejection candles found in Kill Zones")
    
    if issues:
        log_output(f"\nPotential reasons for no trades:", output_file)
        for issue in issues:
            log_output(f"  {issue}", output_file)
    else:
        log_output(f"\nAll conditions seem favorable, but entry criteria may not have aligned.", output_file)
        log_output(f"Check that:", output_file)
        log_output(f"  1. POIs coincided with regime direction", output_file)
        log_output(f"  2. Rejection candles formed AT the POI zones", output_file)
        log_output(f"  3. Entry quality score was above threshold", output_file)
    
    log_output(f"\n{'='*70}", output_file)


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="{time:HH:mm:ss} | {level: <8} | {message}")

    # Create output dir
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        log_output("\n" + "=" * 70, f)
        log_output("SURGE-WSI DETAILED INVESTIGATION", f)
        log_output("Why no trades on January 26-28, 2026?", f)
        log_output("=" * 70, f)

        symbol = "GBPUSD"
        
        # Define date range (timezone-aware)
        start_date = datetime(2026, 1, 26, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 28, 23, 59, 59, tzinfo=timezone.utc)
        
        # Get warmup data (extra 60 days before for HMM training)
        warmup_start = start_date - timedelta(days=60)

        log_output(f"\nFetching data...", f)
        log_output(f"  Warmup Start: {warmup_start.date()}", f)
        log_output(f"  Backtest Start: {start_date.date()}", f)
        log_output(f"  Backtest End: {end_date.date()}", f)

        htf_df = await fetch_data(symbol, "H4", warmup_start, end_date)
        ltf_df = await fetch_data(symbol, "M15", warmup_start, end_date)

        if htf_df.empty or ltf_df.empty:
            log_output("\nERROR: No data available in database.", f)
            return

        log_output(f"\nLoaded H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars", f)

        # Run investigation
        await investigate_period(
            htf_df, ltf_df,
            start_date, end_date,
            output_file=f
        )

        log_output(f"\nResults saved to: {OUTPUT_FILE}", f)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
