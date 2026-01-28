"""Monthly Backtest: Full Integration Test
==========================================

Run full 6-layer backtest per month from Jan 2025 to Jan 2026.

Usage:
    python monthly_backtest.py

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester, BacktestResult


@dataclass
class MonthlyResult:
    """Monthly backtest result"""
    month: str
    year: int
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    net_profit: float
    net_profit_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    trades_per_week: float
    tp1_hit_rate: float
    tp2_hit_rate: float


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


def run_month_backtest(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    month_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float
) -> Optional[MonthlyResult]:
    """Run backtest for a single month"""

    print(f"\n{'='*60}")
    print(f"BACKTEST: {month_name}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"{'='*60}")

    if htf_df.empty or ltf_df.empty:
        print("Insufficient data for this month!")
        return None

    # Prepare data with 'time' column
    htf = htf_df.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)

    ltf = ltf_df.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)

    print(f"HTF bars: {len(htf)}, LTF bars: {len(ltf)}")

    # Create backtester with trend filter and relaxed entry filter
    # NOTE: Hybrid Mode disabled - backtest shows KZ Only performs better (+52% vs +18%)
    bt = Backtester(
        symbol="GBPUSD",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_balance=initial_balance,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=True,
        use_trend_filter=True,      # Enable trend alignment filter
        use_relaxed_filter=True,    # Enable relaxed filters in low-activity periods
        use_hybrid_mode=False       # Disabled: KZ Only performs better in backtest
    )

    # Load data
    bt.load_data(htf, ltf)

    try:
        # Run backtest
        result = bt.run()

        # Print summary
        print(f"\n{'-'*40}")
        print("RESULTS")
        print(f"{'-'*40}")
        print(f"Final Balance:  ${result.final_balance:,.2f}")
        print(f"Net Profit:     ${result.net_profit:,.2f} ({result.net_profit_percent:+.2f}%)")
        print(f"Total Trades:   {result.total_trades}")
        print(f"Win Rate:       {result.win_rate:.1f}%")
        print(f"Profit Factor:  {result.profit_factor:.2f}")
        print(f"Max Drawdown:   {result.max_drawdown_percent:.2f}%")

        if result.total_trades > 0:
            print(f"\n{'-'*40}")
            print("PARTIAL TP STATS")
            print(f"{'-'*40}")
            print(f"TP1 Hit Rate:   {result.tp1_hit_rate:.1f}%")
            print(f"TP2 Hit Rate:   {result.tp2_hit_rate:.1f}%")
            print(f"TP3 Hit Rate:   {result.tp3_hit_rate:.1f}%")

        # Calculate weeks
        weeks = (end_date - start_date).days / 7

        return MonthlyResult(
            month=start_date.strftime("%B"),
            year=start_date.year,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            final_balance=result.final_balance,
            net_profit=result.net_profit,
            net_profit_pct=result.net_profit_percent,
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            losing_trades=result.losing_trades,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            max_drawdown_pct=result.max_drawdown_percent,
            trades_per_week=result.total_trades / weeks if weeks > 0 else 0,
            tp1_hit_rate=result.tp1_hit_rate,
            tp2_hit_rate=result.tp2_hit_rate
        )

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI MONTHLY BACKTEST")
    print("Full 6-Layer Integration Test")
    print("Period: January 2025 - January 2026 (13 Months)")
    print("=" * 70)

    symbol = "GBPUSD"
    initial_balance = 10000.0

    # Define months to backtest (timezone-aware for database compatibility)
    months = []
    for year in [2025, 2026]:
        for month in range(1, 13):
            if year == 2025 or (year == 2026 and month == 1):
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                months.append((start, end, f"{start.strftime('%B')} {year}"))

    # Fetch all data upfront
    print("\nFetching historical data...")
    overall_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    overall_end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    # Get warmup data (extra 3 months before)
    warmup_start = datetime(2024, 10, 1, tzinfo=timezone.utc)

    htf_df = await fetch_data(symbol, "H4", warmup_start, overall_end)
    ltf_df = await fetch_data(symbol, "M15", warmup_start, overall_end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available. Run sync_mt5_data.py first.")
        return

    print(f"Loaded H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Run monthly backtests
    results: List[MonthlyResult] = []
    running_balance = initial_balance

    for start_date, end_date, month_name in months:
        # Filter data for this month (with warmup)
        warmup_days = 30
        month_start_with_warmup = start_date - timedelta(days=warmup_days)

        htf_month = htf_df[(htf_df.index >= month_start_with_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_with_warmup) & (ltf_df.index <= end_date)]

        if htf_month.empty or ltf_month.empty:
            print(f"\n[SKIP] {month_name} - No data available")
            continue

        result = run_month_backtest(
            htf_month, ltf_month,
            month_name, start_date, end_date,
            running_balance
        )

        if result:
            results.append(result)
            running_balance = result.final_balance

    # Generate summary report
    if results:
        print("\n" + "=" * 70)
        print("MONTHLY SUMMARY TABLE")
        print("=" * 70)

        summary_data = []
        for r in results:
            summary_data.append({
                'Month': f"{r.month[:3]} {r.year}",
                'Trades': r.total_trades,
                'Win%': f"{r.win_rate:.1f}%",
                'PF': f"{r.profit_factor:.2f}",
                'P/L': f"${r.net_profit:+.0f}",
                'P/L%': f"{r.net_profit_pct:+.1f}%",
                'DD%': f"{r.max_drawdown_pct:.1f}%",
                'Balance': f"${r.final_balance:,.0f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Overall statistics
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS (13 Months)")
        print("=" * 70)

        total_trades = sum(r.total_trades for r in results)
        total_wins = sum(r.winning_trades for r in results)
        total_pnl = sum(r.net_profit for r in results)
        final_balance = results[-1].final_balance if results else initial_balance

        print(f"\n{'-'*40}")
        print("PERFORMANCE")
        print(f"{'-'*40}")
        print(f"Starting Balance:  ${initial_balance:,.2f}")
        print(f"Final Balance:     ${final_balance:,.2f}")
        print(f"Total Net Profit:  ${total_pnl:,.2f}")
        print(f"Total Return:      {(final_balance/initial_balance - 1)*100:+.2f}%")
        print(f"Monthly Avg:       {total_pnl/len(results):+.2f}$" if results else "")

        print(f"\n{'-'*40}")
        print("TRADES")
        print(f"{'-'*40}")
        print(f"Total Trades:      {total_trades}")
        print(f"Winning Trades:    {total_wins}")
        print(f"Overall Win Rate:  {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
        print(f"Trades per Month:  {total_trades/len(results):.1f}" if results else "")

        print(f"\n{'-'*40}")
        print("RISK METRICS")
        print(f"{'-'*40}")
        max_dd = max(r.max_drawdown_pct for r in results) if results else 0
        avg_pf = np.mean([r.profit_factor for r in results if r.profit_factor > 0])
        print(f"Max Monthly DD:    {max_dd:.2f}%")
        print(f"Avg Profit Factor: {avg_pf:.2f}")

        print(f"\n{'-'*40}")
        print("PARTIAL TP EFFECTIVENESS")
        print(f"{'-'*40}")
        avg_tp1 = np.mean([r.tp1_hit_rate for r in results])
        avg_tp2 = np.mean([r.tp2_hit_rate for r in results])
        print(f"Avg TP1 Hit Rate:  {avg_tp1:.1f}%")
        print(f"Avg TP2 Hit Rate:  {avg_tp2:.1f}%")

        # Monthly performance chart (text-based)
        print("\n" + "=" * 70)
        print("MONTHLY P/L CHART")
        print("=" * 70)

        max_pnl = max(abs(r.net_profit) for r in results) if results else 1
        if max_pnl == 0:
            max_pnl = 1  # Prevent division by zero
        for r in results:
            bar_length = int(abs(r.net_profit) / max_pnl * 30)
            if r.net_profit >= 0:
                bar = "#" * bar_length
                print(f"{r.month[:3]} {r.year}: {bar:>30} +${r.net_profit:.0f}")
            else:
                bar = "-" * bar_length
                print(f"{r.month[:3]} {r.year}: {bar:>30} -${abs(r.net_profit):.0f}")

        # Save results to CSV
        results_path = Path(__file__).parent / "results" / "monthly_backtest_results.csv"
        results_path.parent.mkdir(exist_ok=True)

        results_df = pd.DataFrame([{
            'month': r.month,
            'year': r.year,
            'trades': r.total_trades,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'net_profit': r.net_profit,
            'net_profit_pct': r.net_profit_pct,
            'max_drawdown_pct': r.max_drawdown_pct,
            'final_balance': r.final_balance
        } for r in results])

        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 70)
    print("MONTHLY BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
