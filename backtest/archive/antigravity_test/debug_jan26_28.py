"""Debug Backtest: January 26-28, 2026
=======================================

Run detailed backtest for specific date range to verify trading signals.

Usage:
    python -m backtest.debug_jan26_28

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
from backtest.backtester import Backtester, BacktestResult

# Output file path
OUTPUT_FILE = Path(__file__).parent / "results" / "debug_jan26_28_output.txt"


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


def run_detailed_backtest(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float = 10000.0,
    output_file=None
):
    """Run detailed backtest with verbose output"""
    
    log_output(f"\n{'='*70}", output_file)
    log_output("SURGE-WSI DEBUG BACKTEST", output_file)
    log_output(f"Period: {start_date.date()} to {end_date.date()}", output_file)
    log_output(f"{'='*70}", output_file)

    if htf_df.empty or ltf_df.empty:
        log_output("Insufficient data for this period!", output_file)
        return None

    # Prepare data with 'time' column
    htf = htf_df.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)

    ltf = ltf_df.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)

    log_output(f"\nHTF (H4) bars: {len(htf)}", output_file)
    log_output(f"LTF (M15) bars: {len(ltf)}", output_file)
    
    # Show data range
    if not htf.empty:
        log_output(f"\nHTF Data Range:", output_file)
        log_output(f"  First: {htf['time'].min()}", output_file)
        log_output(f"  Last:  {htf['time'].max()}", output_file)
    
    if not ltf.empty:
        log_output(f"\nLTF Data Range:", output_file)
        log_output(f"  First: {ltf['time'].min()}", output_file)
        log_output(f"  Last:  {ltf['time'].max()}", output_file)

    # Create backtester with verbose settings
    bt = Backtester(
        symbol="GBPUSD",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_balance=initial_balance,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=True,
        use_trend_filter=True,
        use_relaxed_filter=True
    )

    # Load data
    bt.load_data(htf, ltf)

    try:
        # Run backtest
        result = bt.run()

        # Print detailed summary
        log_output(f"\n{'='*70}", output_file)
        log_output("BACKTEST RESULTS", output_file)
        log_output(f"{'='*70}", output_file)
        log_output(f"\nFinal Balance:  ${result.final_balance:,.2f}", output_file)
        log_output(f"Net Profit:     ${result.net_profit:,.2f} ({result.net_profit_percent:+.2f}%)", output_file)
        log_output(f"Total Trades:   {result.total_trades}", output_file)
        log_output(f"Win Rate:       {result.win_rate:.1f}%", output_file)
        log_output(f"Profit Factor:  {result.profit_factor:.2f}", output_file)
        log_output(f"Max Drawdown:   {result.max_drawdown_percent:.2f}%", output_file)

        if result.total_trades > 0:
            log_output(f"\n{'-'*40}", output_file)
            log_output("PARTIAL TP STATS", output_file)
            log_output(f"{'-'*40}", output_file)
            log_output(f"TP1 Hit Rate:   {result.tp1_hit_rate:.1f}%", output_file)
            log_output(f"TP2 Hit Rate:   {result.tp2_hit_rate:.1f}%", output_file)
            log_output(f"TP3 Hit Rate:   {result.tp3_hit_rate:.1f}%", output_file)

            # Show individual trades
            log_output(f"\n{'-'*40}", output_file)
            log_output("INDIVIDUAL TRADES", output_file)
            log_output(f"{'-'*40}", output_file)
            
            trades_df = bt.get_trades_df()
            if not trades_df.empty:
                for i, trade in trades_df.iterrows():
                    log_output(f"\n  Trade #{i+1}:", output_file)
                    log_output(f"    Direction:  {trade.get('direction', 'N/A')}", output_file)
                    log_output(f"    Entry Time: {trade.get('entry_time', 'N/A')}", output_file)
                    log_output(f"    Exit Time:  {trade.get('exit_time', 'N/A')}", output_file)
                    log_output(f"    Entry:      {trade.get('entry_price', 0):.5f}", output_file)
                    log_output(f"    Exit:       {trade.get('exit_price', 0):.5f}", output_file)
                    log_output(f"    SL:         {trade.get('sl_price', 0):.5f}", output_file)
                    log_output(f"    TP1:        {trade.get('tp1_price', 0):.5f}", output_file)
                    log_output(f"    Status:     {trade.get('status', 'N/A')}", output_file)
                    log_output(f"    PnL:        ${trade.get('pnl', 0):+.2f}", output_file)
                    log_output(f"    Regime:     {trade.get('regime', 'N/A')}", output_file)
        else:
            log_output("\n[WARNING] NO TRADES EXECUTED during this period", output_file)
            log_output("\nPossible reasons:", output_file)
            log_output("  1. Market was outside Kill Zone hours", output_file)
            log_output("  2. Regime was SIDEWAYS (no clear trend)", output_file)
            log_output("  3. No valid POI (Order Block/FVG) detected", output_file)
            log_output("  4. No rejection candle triggered entry", output_file)
            
        return result

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="{time:HH:mm:ss} | {level: <8} | {message}")

    # Create output dir
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        log_output("\n" + "=" * 70, f)
        log_output("SURGE-WSI DEBUG BACKTEST", f)
        log_output("Checking for trades: January 26-28, 2026", f)
        log_output("=" * 70, f)

        symbol = "GBPUSD"
        
        # Define date range (timezone-aware)
        start_date = datetime(2026, 1, 26, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 28, 23, 59, 59, tzinfo=timezone.utc)
        
        # Get warmup data (extra 30 days before for HMM training)
        warmup_start = start_date - timedelta(days=30)

        log_output(f"\nFetching data...", f)
        log_output(f"  Warmup Start: {warmup_start.date()}", f)
        log_output(f"  Backtest Start: {start_date.date()}", f)
        log_output(f"  Backtest End: {end_date.date()}", f)

        htf_df = await fetch_data(symbol, "H4", warmup_start, end_date)
        ltf_df = await fetch_data(symbol, "M15", warmup_start, end_date)

        if htf_df.empty or ltf_df.empty:
            log_output("\nERROR: No data available in database.", f)
            log_output("Please ensure MT5 data has been synced to TimescaleDB.", f)
            return

        log_output(f"\nLoaded H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars", f)

        # Run backtest
        result = run_detailed_backtest(
            htf_df, ltf_df,
            start_date, end_date,
            initial_balance=10000.0,
            output_file=f
        )

        log_output("\n" + "=" * 70, f)
        log_output("DEBUG BACKTEST COMPLETE", f)
        log_output("=" * 70, f)
        
        log_output(f"\nResults saved to: {OUTPUT_FILE}", f)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
