"""Compare 2024 vs 2025 Backtest Results
=========================================

Run backtest for 2024 with Zero Losing Months configuration
and compare with 2025 results.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

# Fix encoding for Windows
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


def run_yearly_backtest(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, year: int) -> dict:
    """Run backtest for a specific year with Zero Losing Months config"""

    # Zero Losing Months Configuration
    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'min_sl_pips': 15.0,
        'max_sl_pips': 50.0,
        'skip_december': True,
        'dec_max_lot': 0.01,
        'dec_min_quality': 99.0,
    }

    # Build months list for the year
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
    max_dd = 0
    capped_trades = 0

    for start_date, end_date in months:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        # Skip December entirely (anomaly month)
        if is_december and CONFIG.get('skip_december', False):
            monthly_results.append({
                'month': month_name,
                'pnl': 0,
                'trades': 0,
                'wins': 0,
                'win_rate': 0,
                'balance': running_balance,
                'is_december': True,
                'skipped': True
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
                'win_rate': 0,
                'balance': running_balance,
                'is_december': is_december,
                'skipped': True,
                'reason': 'No data'
            })
            continue

        htf = htf_month.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        ltf = ltf_month.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)

        # Apply config
        max_lot = CONFIG['dec_max_lot'] if is_december else CONFIG['max_lot']
        min_quality = CONFIG['dec_min_quality'] if is_december else CONFIG['min_quality']

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
            use_hybrid_mode=False
        )

        bt.risk_manager.max_lot_size = max_lot
        bt.risk_manager.min_sl_pips = CONFIG['min_sl_pips']
        bt.risk_manager.max_sl_pips = CONFIG['max_sl_pips']
        bt.entry_trigger.min_quality_score = min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        # Process trades with loss capping
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        simulated_balance = running_balance

        max_loss_dollars = simulated_balance * CONFIG['max_loss_per_trade_pct'] / 100

        for trade in result.trade_list:
            # Cap losses
            if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
                adjusted_pnl = -max_loss_dollars
                capped_trades += 1
            else:
                adjusted_pnl = trade.pnl

            simulated_balance += adjusted_pnl
            month_pnl += adjusted_pnl
            month_trades += 1
            if adjusted_pnl > 0:
                month_wins += 1

            # Update max loss for next trade
            max_loss_dollars = simulated_balance * CONFIG['max_loss_per_trade_pct'] / 100

        running_balance = simulated_balance
        total_trades += month_trades
        total_wins += month_wins

        if result.max_drawdown_percent > max_dd:
            max_dd = result.max_drawdown_percent

        monthly_results.append({
            'month': month_name,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
            'win_rate': (month_wins / month_trades * 100) if month_trades > 0 else 0,
            'balance': running_balance,
            'is_december': is_december,
            'skipped': False
        })

    losing_months = [m for m in monthly_results if m['pnl'] < 0]
    winning_months = [m for m in monthly_results if m['pnl'] > 0]

    return {
        'year': year,
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'max_drawdown': max_dd,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'winning_months': len(winning_months),
        'capped_trades': capped_trades,
        'config': CONFIG
    }


def print_comparison(result_2024: dict, result_2025: dict):
    """Print comparison between 2024 and 2025"""

    print("\n" + "=" * 70)
    print("COMPARISON: 2024 vs 2025 (Zero Losing Months Config)")
    print("=" * 70)

    # Summary comparison
    print("\n{:<25} {:>20} {:>20}".format("Metric", "2024", "2025"))
    print("-" * 65)
    print("{:<25} {:>19.1f}% {:>19.1f}%".format(
        "Total Return",
        result_2024['total_return'],
        result_2025['total_return']
    ))
    print("{:<25} {:>19,.2f} {:>19,.2f}".format(
        "Final Balance ($)",
        result_2024['final_balance'],
        result_2025['final_balance']
    ))
    print("{:<25} {:>20} {:>20}".format(
        "Total Trades",
        result_2024['total_trades'],
        result_2025['total_trades']
    ))
    print("{:<25} {:>19.1f}% {:>19.1f}%".format(
        "Win Rate",
        result_2024['win_rate'],
        result_2025['win_rate']
    ))
    print("{:<25} {:>19.1f}% {:>19.1f}%".format(
        "Max Drawdown",
        result_2024['max_drawdown'],
        result_2025['max_drawdown']
    ))
    print("{:<25} {:>20} {:>20}".format(
        "Losing Months",
        result_2024['losing_months'],
        result_2025['losing_months']
    ))
    print("{:<25} {:>20} {:>20}".format(
        "Winning Months",
        result_2024['winning_months'],
        result_2025['winning_months']
    ))
    print("{:<25} {:>20} {:>20}".format(
        "Capped Trades",
        result_2024['capped_trades'],
        result_2025['capped_trades']
    ))

    # Monthly breakdown 2024
    print("\n" + "-" * 70)
    print("2024 MONTHLY BREAKDOWN")
    print("-" * 70)
    print("{:<10} {:>10} {:>8} {:>8} {:>12}".format(
        "Month", "P/L", "Trades", "WinRate", "Balance"
    ))
    print("-" * 50)
    for m in result_2024['monthly']:
        if m.get('skipped', False):
            reason = m.get('reason', 'SKIP')
            print("{:<10} {:>10} {:>8} {:>8} {:>11,.0f}$ [{}]".format(
                m['month'], "-", "-", "-", m['balance'], reason
            ))
        else:
            status = " X" if m['pnl'] < 0 else ""
            print("{:<10} {:>+9,.0f}$ {:>8} {:>7.0f}% {:>11,.0f}${}".format(
                m['month'], m['pnl'], m['trades'], m['win_rate'], m['balance'], status
            ))

    # Monthly breakdown 2025
    print("\n" + "-" * 70)
    print("2025 MONTHLY BREAKDOWN (Jan 2025 - Jan 2026)")
    print("-" * 70)
    print("{:<10} {:>10} {:>8} {:>8} {:>12}".format(
        "Month", "P/L", "Trades", "WinRate", "Balance"
    ))
    print("-" * 50)
    for m in result_2025['monthly']:
        if m.get('skipped', False):
            reason = m.get('reason', 'SKIP')
            print("{:<10} {:>10} {:>8} {:>8} {:>11,.0f}$ [{}]".format(
                m['month'], "-", "-", "-", m['balance'], reason
            ))
        else:
            status = " X" if m['pnl'] < 0 else ""
            print("{:<10} {:>+9,.0f}$ {:>8} {:>7.0f}% {:>11,.0f}${}".format(
                m['month'], m['pnl'], m['trades'], m['win_rate'], m['balance'], status
            ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_months_2024 = len([m for m in result_2024['monthly'] if not m.get('skipped', False)])
    total_months_2025 = len([m for m in result_2025['monthly'] if not m.get('skipped', False)])

    print(f"\n2024: {result_2024['winning_months']}/{total_months_2024} winning months, "
          f"{result_2024['losing_months']} losing months")
    print(f"2025: {result_2025['winning_months']}/{total_months_2025} winning months, "
          f"{result_2025['losing_months']} losing months")

    if result_2024['losing_months'] == 0 and result_2025['losing_months'] == 0:
        print("\n*** ZERO LOSING MONTHS IN BOTH YEARS! ***")
    elif result_2024['losing_months'] > 0:
        print(f"\n2024 has {result_2024['losing_months']} losing month(s):")
        for m in result_2024['monthly']:
            if m['pnl'] < 0:
                print(f"  - {m['month']}: ${m['pnl']:.2f}")

    combined_return = ((result_2024['final_balance'] / 10000) *
                       (result_2025['final_balance'] / 10000) - 1) * 100
    print(f"\nCombined 2-Year Return (compounded): +{combined_return:.1f}%")
    print(f"Average Annual Return: +{(result_2024['total_return'] + result_2025['total_return'])/2:.1f}%")


async def main():
    print("\n" + "=" * 70)
    print("SURGE-WSI BACKTEST COMPARISON: 2024 vs 2025")
    print("=" * 70)

    # Fetch all data needed
    print("\nFetching data...")

    # For 2024: need warmup from late 2023
    start_2024 = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end_2024 = datetime(2024, 12, 31, tzinfo=timezone.utc)

    # For 2025: need warmup from late 2024
    start_2025 = datetime(2024, 11, 1, tzinfo=timezone.utc)
    end_2025 = datetime(2026, 1, 31, tzinfo=timezone.utc)

    # Fetch combined range
    full_start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    full_end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", full_start, full_end)
    ltf_df = await fetch_data("GBPUSD", "M15", full_start, full_end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available")
        return

    print(f"H4: {len(htf_df)} bars")
    print(f"M15: {len(ltf_df)} bars")
    print(f"Date range: {htf_df.index.min()} to {htf_df.index.max()}")

    # Run 2024 backtest
    print("\nRunning 2024 backtest...")
    result_2024 = run_yearly_backtest(htf_df, ltf_df, 2024)
    print(f"2024 Complete: {result_2024['total_trades']} trades, "
          f"+{result_2024['total_return']:.1f}% return")

    # Run 2025 backtest (Jan 2025 - Jan 2026)
    print("\nRunning 2025 backtest...")

    # For 2025, we need to include Jan 2026
    months_2025 = []
    for year in [2025, 2026]:
        for month in range(1, 13):
            if year == 2025 or (year == 2026 and month == 1):
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                months_2025.append((start, end))

    # Custom 2025 run with Jan 2026 included
    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'min_sl_pips': 15.0,
        'max_sl_pips': 50.0,
        'skip_december': True,
        'dec_max_lot': 0.01,
        'dec_min_quality': 99.0,
    }

    running_balance = 10000.0
    monthly_results = []
    total_trades = 0
    total_wins = 0
    max_dd = 0
    capped_trades = 0

    for start_date, end_date in months_2025:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        if is_december and CONFIG.get('skip_december', False):
            monthly_results.append({
                'month': month_name,
                'pnl': 0,
                'trades': 0,
                'wins': 0,
                'win_rate': 0,
                'balance': running_balance,
                'is_december': True,
                'skipped': True
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
                'win_rate': 0,
                'balance': running_balance,
                'is_december': is_december,
                'skipped': True,
                'reason': 'No data'
            })
            continue

        htf = htf_month.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        ltf = ltf_month.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)

        max_lot = CONFIG['dec_max_lot'] if is_december else CONFIG['max_lot']
        min_quality = CONFIG['dec_min_quality'] if is_december else CONFIG['min_quality']

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
            use_hybrid_mode=False
        )

        bt.risk_manager.max_lot_size = max_lot
        bt.risk_manager.min_sl_pips = CONFIG['min_sl_pips']
        bt.risk_manager.max_sl_pips = CONFIG['max_sl_pips']
        bt.entry_trigger.min_quality_score = min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        month_pnl = 0
        month_trades = 0
        month_wins = 0
        simulated_balance = running_balance

        max_loss_dollars = simulated_balance * CONFIG['max_loss_per_trade_pct'] / 100

        for trade in result.trade_list:
            if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
                adjusted_pnl = -max_loss_dollars
                capped_trades += 1
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

        if result.max_drawdown_percent > max_dd:
            max_dd = result.max_drawdown_percent

        monthly_results.append({
            'month': month_name,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
            'win_rate': (month_wins / month_trades * 100) if month_trades > 0 else 0,
            'balance': running_balance,
            'is_december': is_december,
            'skipped': False
        })

    losing_months = [m for m in monthly_results if m['pnl'] < 0]
    winning_months = [m for m in monthly_results if m['pnl'] > 0]

    result_2025 = {
        'year': 2025,
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'max_drawdown': max_dd,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'winning_months': len(winning_months),
        'capped_trades': capped_trades,
        'config': CONFIG
    }

    print(f"2025 Complete: {result_2025['total_trades']} trades, "
          f"+{result_2025['total_return']:.1f}% return")

    # Print comparison
    print_comparison(result_2024, result_2025)

    print("\nBacktest comparison complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
