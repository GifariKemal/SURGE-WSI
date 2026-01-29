"""Test Backtest: January 26-28, 2026 - No Kill Zone
====================================================

Test what happens when we disable Kill Zone filter.

Usage:
    python -m backtest.test_no_killzone

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
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester

# Output file path
OUTPUT_FILE = Path(__file__).parent / "results" / "test_no_killzone.txt"


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


def run_backtest_comparison(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    output_file=None
):
    """Run backtest with and without Kill Zone"""
    
    # Prepare data
    htf = htf_df.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)
    
    ltf = ltf_df.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)
    
    # ============================================================
    # TEST 1: WITH Kill Zone (Default)
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("TEST 1: WITH Kill Zone Filter (Default)", output_file)
    log_output(f"{'='*70}", output_file)
    
    bt_with_kz = Backtester(
        symbol="GBPUSD",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_balance=10000.0,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=True,    # WITH Kill Zone
        use_trend_filter=True,
        use_relaxed_filter=True
    )
    bt_with_kz.load_data(htf, ltf)
    result_with_kz = bt_with_kz.run()
    
    log_output(f"\nResults WITH Kill Zone:", output_file)
    log_output(f"  Total Trades:   {result_with_kz.total_trades}", output_file)
    log_output(f"  Net Profit:     ${result_with_kz.net_profit:,.2f}", output_file)
    log_output(f"  Win Rate:       {result_with_kz.win_rate:.1f}%", output_file)
    
    # Debug stats
    log_output(f"\n  Debug Stats:", output_file)
    log_output(f"    Entry checks:      {bt_with_kz._debug_checks}", output_file)
    log_output(f"    Regime failures:   {bt_with_kz._debug_regime_fail}", output_file)
    log_output(f"    Regime sideways:   {bt_with_kz._debug_regime_sideways}", output_file)
    log_output(f"    POI none:          {bt_with_kz._debug_poi_none}", output_file)
    log_output(f"    No POIs:           {bt_with_kz._debug_no_pois}", output_file)
    log_output(f"    Not in POI:        {bt_with_kz._debug_not_in_poi}", output_file)
    log_output(f"    In POI:            {bt_with_kz._debug_in_poi}", output_file)
    log_output(f"    Entry OK:          {bt_with_kz._debug_entry_ok}", output_file)
    
    # ============================================================
    # TEST 2: WITHOUT Kill Zone
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("TEST 2: WITHOUT Kill Zone Filter", output_file)
    log_output(f"{'='*70}", output_file)
    
    bt_no_kz = Backtester(
        symbol="GBPUSD",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_balance=10000.0,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=False,   # WITHOUT Kill Zone
        use_trend_filter=True,
        use_relaxed_filter=True
    )
    bt_no_kz.load_data(htf, ltf)
    result_no_kz = bt_no_kz.run()
    
    log_output(f"\nResults WITHOUT Kill Zone:", output_file)
    log_output(f"  Total Trades:   {result_no_kz.total_trades}", output_file)
    log_output(f"  Net Profit:     ${result_no_kz.net_profit:,.2f}", output_file)
    log_output(f"  Win Rate:       {result_no_kz.win_rate:.1f}%", output_file)
    
    # Debug stats
    log_output(f"\n  Debug Stats:", output_file)
    log_output(f"    Entry checks:      {bt_no_kz._debug_checks}", output_file)
    log_output(f"    Regime failures:   {bt_no_kz._debug_regime_fail}", output_file)
    log_output(f"    Regime sideways:   {bt_no_kz._debug_regime_sideways}", output_file)
    log_output(f"    POI none:          {bt_no_kz._debug_poi_none}", output_file)
    log_output(f"    No POIs:           {bt_no_kz._debug_no_pois}", output_file)
    log_output(f"    Not in POI:        {bt_no_kz._debug_not_in_poi}", output_file)
    log_output(f"    In POI:            {bt_no_kz._debug_in_poi}", output_file)
    log_output(f"    Entry OK:          {bt_no_kz._debug_entry_ok}", output_file)
    
    # ============================================================
    # COMPARISON
    # ============================================================
    log_output(f"\n{'='*70}", output_file)
    log_output("COMPARISON SUMMARY", output_file)
    log_output(f"{'='*70}", output_file)
    
    log_output(f"\n{'Metric':<25} {'WITH KZ':>15} {'WITHOUT KZ':>15}", output_file)
    log_output(f"{'-'*55}", output_file)
    log_output(f"{'Entry Checks':<25} {bt_with_kz._debug_checks:>15} {bt_no_kz._debug_checks:>15}", output_file)
    log_output(f"{'Regime Failures':<25} {bt_with_kz._debug_regime_fail:>15} {bt_no_kz._debug_regime_fail:>15}", output_file)
    log_output(f"{'Regime Sideways':<25} {bt_with_kz._debug_regime_sideways:>15} {bt_no_kz._debug_regime_sideways:>15}", output_file)
    log_output(f"{'Total Trades':<25} {result_with_kz.total_trades:>15} {result_no_kz.total_trades:>15}", output_file)
    log_output(f"{'Net Profit':<25} {'$'+f'{result_with_kz.net_profit:.2f}':>15} {'$'+f'{result_no_kz.net_profit:.2f}':>15}", output_file)
    
    # Analysis
    log_output(f"\n{'='*70}", output_file)
    log_output("ANALYSIS", output_file)
    log_output(f"{'='*70}", output_file)
    
    checks_diff = bt_no_kz._debug_checks - bt_with_kz._debug_checks
    log_output(f"\nWithout Kill Zone:", output_file)
    log_output(f"  - {checks_diff} MORE entry checks performed", output_file)
    
    if bt_no_kz._debug_regime_fail > 0 or bt_no_kz._debug_regime_sideways > 0:
        regime_total = bt_no_kz._debug_regime_fail + bt_no_kz._debug_regime_sideways
        log_output(f"  - {regime_total} checks still blocked by REGIME filter", output_file)
        log_output(f"\n  CONCLUSION: Kill Zone is NOT the bottleneck!", output_file)
        log_output(f"  The REGIME filter (SIDEWAYS market) is preventing trades.", output_file)
    
    if result_no_kz.total_trades == result_with_kz.total_trades:
        log_output(f"\n  Same number of trades with or without Kill Zone.", output_file)
        log_output(f"  This confirms the REGIME filter is the main gatekeeper.", output_file)


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="{time:HH:mm:ss} | {level: <8} | {message}")

    # Create output dir
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        log_output("\n" + "=" * 70, f)
        log_output("KILL ZONE COMPARISON TEST", f)
        log_output("Testing: January 26-28, 2026", f)
        log_output("=" * 70, f)

        symbol = "GBPUSD"
        
        # Define date range
        start_date = datetime(2026, 1, 26, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 28, 23, 59, 59, tzinfo=timezone.utc)
        warmup_start = start_date - timedelta(days=60)

        log_output(f"\nFetching data...", f)

        htf_df = await fetch_data(symbol, "H4", warmup_start, end_date)
        ltf_df = await fetch_data(symbol, "M15", warmup_start, end_date)

        if htf_df.empty or ltf_df.empty:
            log_output("\nERROR: No data available.", f)
            return

        log_output(f"Loaded H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars", f)

        # Run comparison
        run_backtest_comparison(
            htf_df, ltf_df,
            start_date, end_date,
            output_file=f
        )

        log_output(f"\n" + "=" * 70, f)
        log_output(f"Results saved to: {OUTPUT_FILE}", f)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
