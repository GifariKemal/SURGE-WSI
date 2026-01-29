"""Test Choppiness Filter Impact on 2024 Losing Months
======================================================

Compare backtest results with and without Choppiness Filter
to see if it reduces losing months in 2024.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester


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
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=100000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def run_monthly_backtest(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                         year: int, use_choppiness: bool, chop_threshold: float = 61.8) -> dict:
    """Run backtest for a year with or without choppiness filter"""

    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'min_sl_pips': 15.0,
        'max_sl_pips': 50.0,
        'skip_december': True,
    }

    months = []
    for month in range(1, 13):
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        months.append((start, end))

    running_balance = 10000.0
    monthly_results = []
    total_trades = 0
    total_wins = 0
    total_chop_filtered = 0

    for start_date, end_date in months:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        if is_december and CONFIG.get('skip_december', True):
            monthly_results.append({
                'month': month_name,
                'pnl': 0,
                'trades': 0,
                'wins': 0,
                'balance': running_balance,
                'skipped': True,
                'chop_filtered': 0
            })
            continue

        warmup_days = 30
        month_start_warmup = start_date - timedelta(days=warmup_days)

        htf_month = htf_df[(htf_df.index >= month_start_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_warmup) & (ltf_df.index <= end_date)]

        if htf_month.empty or ltf_month.empty:
            monthly_results.append({
                'month': month_name,
                'pnl': 0,
                'trades': 0,
                'wins': 0,
                'balance': running_balance,
                'skipped': True,
                'chop_filtered': 0
            })
            continue

        htf = htf_month.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        ltf = ltf_month.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)

        bt = Backtester(
            symbol="GBPUSD",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_balance=running_balance,
            pip_value=10.0,
            spread_pips=1.5,
            use_killzone=True,
            use_trend_filter=True,
            use_relaxed_filter=False,
            use_hybrid_mode=False,
            use_choppiness_filter=use_choppiness,
            choppiness_threshold=chop_threshold
        )

        bt.risk_manager.max_lot_size = CONFIG['max_lot']
        bt.risk_manager.min_sl_pips = CONFIG['min_sl_pips']
        bt.risk_manager.max_sl_pips = CONFIG['max_sl_pips']
        bt.entry_trigger.min_quality_score = CONFIG['min_quality']

        bt.load_data(htf, ltf)
        result = bt.run()

        # Get choppiness filtered count
        chop_filtered = bt._debug_choppiness_filtered

        # Process trades with loss capping
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        simulated_balance = running_balance

        max_loss_dollars = simulated_balance * CONFIG['max_loss_per_trade_pct'] / 100

        for trade in result.trade_list:
            if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
                adjusted_pnl = -max_loss_dollars
            else:
                adjusted_pnl = trade.pnl

            simulated_balance += adjusted_pnl
            month_pnl += adjusted_pnl
            month_trades += 1
            if adjusted_pnl > 0:
                month_wins += 1

            max_loss_dollars = simulated_balance * CONFIG['max_loss_per_trade_pct'] / 100

        running_balance = simulated_balance
        total_trades += month_trades
        total_wins += month_wins
        total_chop_filtered += chop_filtered

        monthly_results.append({
            'month': month_name,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
            'balance': running_balance,
            'skipped': False,
            'chop_filtered': chop_filtered
        })

    losing_months = [m for m in monthly_results if m['pnl'] < 0]

    return {
        'year': year,
        'use_choppiness': use_choppiness,
        'chop_threshold': chop_threshold,
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'losing_month_names': [m['month'] for m in losing_months],
        'total_chop_filtered': total_chop_filtered
    }


async def main():
    print("\n" + "=" * 70)
    print("TEST: CHOPPINESS FILTER IMPACT ON 2024")
    print("=" * 70)

    # Fetch data
    print("\nFetching data...")
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Test different configurations
    configs = [
        (False, 61.8, "WITHOUT Choppiness Filter"),
        (True, 61.8, "WITH Choppiness Filter (CHOP > 61.8)"),
        (True, 55.0, "WITH Choppiness Filter (CHOP > 55.0) - Stricter"),
        (True, 50.0, "WITH Choppiness Filter (CHOP > 50.0) - Very Strict"),
    ]

    results = []
    for use_chop, threshold, name in configs:
        print(f"\nTesting: {name}...")
        result = run_monthly_backtest(htf_df, ltf_df, 2024, use_chop, threshold)
        result['config_name'] = name
        results.append(result)
        print(f"  -> {result['losing_months']} losing months, +{result['total_return']:.1f}%, "
              f"{result['total_trades']} trades")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n{:<45} {:>8} {:>8} {:>8} {:>10}".format(
        "Configuration", "Losing", "Trades", "WinRate", "Return"
    ))
    print("-" * 80)

    for r in results:
        print("{:<45} {:>8} {:>8} {:>7.1f}% {:>9.1f}%".format(
            r['config_name'][:45],
            r['losing_months'],
            r['total_trades'],
            r['win_rate'],
            r['total_return']
        ))

    # Find best configuration
    best = min(results, key=lambda x: (x['losing_months'], -x['total_return']))

    print("\n" + "=" * 70)
    print(f"BEST CONFIGURATION: {best['config_name']}")
    print("=" * 70)

    print(f"\nLosing Months: {best['losing_months']}")
    print(f"Total Return: +{best['total_return']:.1f}%")
    print(f"Total Trades: {best['total_trades']}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Signals Filtered by Choppiness: {best['total_chop_filtered']}")

    if best['losing_months'] > 0:
        print(f"\nRemaining Losing Months: {best['losing_month_names']}")

    # Detailed monthly breakdown for best config
    print("\n" + "-" * 70)
    print("MONTHLY BREAKDOWN (Best Config)")
    print("-" * 70)
    print("{:<10} {:>10} {:>8} {:>8} {:>12} {:>10}".format(
        "Month", "P/L", "Trades", "WinRate", "Balance", "ChopSkip"
    ))
    print("-" * 60)

    for m in best['monthly']:
        if m.get('skipped', False):
            print("{:<10} {:>10} {:>8} {:>8} {:>11,.0f}$ {:>10}".format(
                m['month'], "SKIP", "-", "-", m['balance'], "-"
            ))
        else:
            wr = m['wins'] / m['trades'] * 100 if m['trades'] > 0 else 0
            status = " X" if m['pnl'] < 0 else ""
            print("{:<10} {:>+9,.0f}$ {:>8} {:>7.0f}% {:>11,.0f}$ {:>10}{}".format(
                m['month'], m['pnl'], m['trades'], wr, m['balance'],
                m['chop_filtered'], status
            ))

    # Compare with without filter
    without = results[0]
    with_default = results[1]

    print("\n" + "=" * 70)
    print("IMPACT ANALYSIS")
    print("=" * 70)

    print(f"\nWITHOUT Choppiness Filter:")
    print(f"  Losing Months: {without['losing_months']} ({without['losing_month_names']})")
    print(f"  Total Trades: {without['total_trades']}")
    print(f"  Return: +{without['total_return']:.1f}%")

    print(f"\nWITH Choppiness Filter (61.8):")
    print(f"  Losing Months: {with_default['losing_months']} ({with_default['losing_month_names']})")
    print(f"  Total Trades: {with_default['total_trades']}")
    print(f"  Return: +{with_default['total_return']:.1f}%")
    print(f"  Signals Filtered: {with_default['total_chop_filtered']}")

    trade_reduction = without['total_trades'] - with_default['total_trades']
    losing_reduction = without['losing_months'] - with_default['losing_months']

    print(f"\nIMPACT:")
    print(f"  Trade Reduction: {trade_reduction} trades ({trade_reduction/without['total_trades']*100:.1f}%)")
    print(f"  Losing Month Reduction: {losing_reduction} months")

    if with_default['losing_months'] == 0:
        print("\n*** SUCCESS: ZERO LOSING MONTHS WITH CHOPPINESS FILTER! ***")

    print("\nTest complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
