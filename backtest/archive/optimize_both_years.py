"""Optimize for Zero Losing Months in Both 2024 and 2025
=======================================================

Find configuration that achieves zero losing months across both years.

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
from itertools import product

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


def run_backtest_with_config(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                              months_list: list, config_params: dict) -> dict:
    """Run backtest for a list of months with specific config"""

    running_balance = 10000.0
    monthly_results = []
    total_trades = 0
    total_wins = 0
    max_dd = 0
    capped_trades = 0

    for start_date, end_date in months_list:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        # Skip December entirely
        if is_december and config_params.get('skip_december', True):
            monthly_results.append({
                'month': month_name,
                'pnl': 0,
                'trades': 0,
                'wins': 0,
                'balance': running_balance,
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
                'balance': running_balance,
                'skipped': True
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
            use_hybrid_mode=False
        )

        bt.risk_manager.max_lot_size = config_params['max_lot']
        bt.risk_manager.min_sl_pips = config_params.get('min_sl_pips', 15.0)
        bt.risk_manager.max_sl_pips = config_params.get('max_sl_pips', 50.0)
        bt.entry_trigger.min_quality_score = config_params['min_quality']

        bt.load_data(htf, ltf)
        result = bt.run()

        # Process trades with loss capping
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        simulated_balance = running_balance

        max_loss_pct = config_params['max_loss_per_trade_pct']
        max_loss_dollars = simulated_balance * max_loss_pct / 100

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

            max_loss_dollars = simulated_balance * max_loss_pct / 100

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
            'balance': running_balance,
            'skipped': False
        })

    losing_months = [m for m in monthly_results if m['pnl'] < 0]

    return {
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'max_drawdown': max_dd,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'losing_month_names': [m['month'] for m in losing_months],
        'capped_trades': capped_trades
    }


async def main():
    print("\n" + "=" * 70)
    print("OPTIMIZE FOR ZERO LOSING MONTHS (2024 + 2025)")
    print("=" * 70)

    # Fetch all data
    print("\nFetching data...")
    full_start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    full_end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", full_start, full_end)
    ltf_df = await fetch_data("GBPUSD", "M15", full_start, full_end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Build months for 2024
    months_2024 = []
    for month in range(1, 13):
        start = datetime(2024, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(2025, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end = datetime(2024, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        months_2024.append((start, end))

    # Build months for 2025 (including Jan 2026)
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

    # Parameters to test
    max_lots = [0.3, 0.4, 0.5]
    max_loss_pcts = [0.5, 0.6, 0.7, 0.8]
    min_qualities = [65, 70, 75, 80]

    total_configs = len(max_lots) * len(max_loss_pcts) * len(min_qualities)
    print(f"\nTesting {total_configs} configurations...")
    print("-" * 70)

    best_config = None
    best_return = -999
    zero_loss_configs = []

    config_num = 0
    for max_lot, max_loss_pct, min_quality in product(max_lots, max_loss_pcts, min_qualities):
        config_num += 1

        config_params = {
            'max_lot': max_lot,
            'max_loss_per_trade_pct': max_loss_pct,
            'min_quality': min_quality,
            'min_sl_pips': 15.0,
            'max_sl_pips': 50.0,
            'skip_december': True
        }

        # Run 2024
        result_2024 = run_backtest_with_config(htf_df, ltf_df, months_2024, config_params)

        # Run 2025
        result_2025 = run_backtest_with_config(htf_df, ltf_df, months_2025, config_params)

        total_losing = result_2024['losing_months'] + result_2025['losing_months']
        combined_return = result_2024['total_return'] + result_2025['total_return']
        total_trades = result_2024['total_trades'] + result_2025['total_trades']

        # Progress
        status = "ZERO!" if total_losing == 0 else f"{total_losing} losing"
        print(f"[{config_num:2d}/{total_configs}] lot={max_lot}, loss={max_loss_pct}%, qual={min_quality} "
              f"-> 2024: {result_2024['losing_months']}L, 2025: {result_2025['losing_months']}L, "
              f"Return: +{combined_return:.1f}%, Trades: {total_trades} [{status}]")

        if total_losing == 0:
            zero_loss_configs.append({
                'config': config_params.copy(),
                'result_2024': result_2024,
                'result_2025': result_2025,
                'combined_return': combined_return,
                'total_trades': total_trades
            })

            if combined_return > best_return:
                best_return = combined_return
                best_config = {
                    'config': config_params.copy(),
                    'result_2024': result_2024,
                    'result_2025': result_2025
                }

    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    if not zero_loss_configs:
        print("\nNo configuration achieved zero losing months in both years.")
        print("Finding best compromise...")

        # Find config with minimum losing months
        best_compromise = None
        min_losing = 999

        for max_lot, max_loss_pct, min_quality in product(max_lots, max_loss_pcts, min_qualities):
            config_params = {
                'max_lot': max_lot,
                'max_loss_per_trade_pct': max_loss_pct,
                'min_quality': min_quality,
                'min_sl_pips': 15.0,
                'max_sl_pips': 50.0,
                'skip_december': True
            }

            result_2024 = run_backtest_with_config(htf_df, ltf_df, months_2024, config_params)
            result_2025 = run_backtest_with_config(htf_df, ltf_df, months_2025, config_params)

            total_losing = result_2024['losing_months'] + result_2025['losing_months']
            combined_return = result_2024['total_return'] + result_2025['total_return']

            if total_losing < min_losing or (total_losing == min_losing and combined_return > best_return):
                min_losing = total_losing
                best_return = combined_return
                best_compromise = {
                    'config': config_params.copy(),
                    'result_2024': result_2024,
                    'result_2025': result_2025
                }

        if best_compromise:
            print(f"\nBest Compromise Configuration:")
            print(f"  max_lot: {best_compromise['config']['max_lot']}")
            print(f"  max_loss_per_trade_pct: {best_compromise['config']['max_loss_per_trade_pct']}%")
            print(f"  min_quality: {best_compromise['config']['min_quality']}")
            print(f"\n2024: {best_compromise['result_2024']['losing_months']} losing months, "
                  f"+{best_compromise['result_2024']['total_return']:.1f}%")
            print(f"2025: {best_compromise['result_2025']['losing_months']} losing months, "
                  f"+{best_compromise['result_2025']['total_return']:.1f}%")

            # Show losing months
            if best_compromise['result_2024']['losing_months'] > 0:
                print(f"\n2024 Losing Months: {best_compromise['result_2024']['losing_month_names']}")
            if best_compromise['result_2025']['losing_months'] > 0:
                print(f"2025 Losing Months: {best_compromise['result_2025']['losing_month_names']}")

            best_config = best_compromise
    else:
        print(f"\nFound {len(zero_loss_configs)} configurations with ZERO losing months!")
        print("\nAll zero-loss configurations (sorted by return):")
        print("-" * 70)

        zero_loss_configs.sort(key=lambda x: x['combined_return'], reverse=True)

        for i, cfg in enumerate(zero_loss_configs[:10], 1):
            print(f"{i}. lot={cfg['config']['max_lot']}, loss={cfg['config']['max_loss_per_trade_pct']}%, "
                  f"qual={cfg['config']['min_quality']} -> +{cfg['combined_return']:.1f}% "
                  f"({cfg['total_trades']} trades)")

    # Show best configuration details
    if best_config:
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION DETAILS")
        print("=" * 70)

        cfg = best_config['config']
        r24 = best_config['result_2024']
        r25 = best_config['result_2025']

        print(f"\nConfiguration:")
        print(f"  max_lot: {cfg['max_lot']}")
        print(f"  max_loss_per_trade_pct: {cfg['max_loss_per_trade_pct']}%")
        print(f"  min_quality: {cfg['min_quality']}")
        print(f"  skip_december: True")

        print(f"\n2024 Results:")
        print(f"  Return: +{r24['total_return']:.1f}%")
        print(f"  Final Balance: ${r24['final_balance']:,.2f}")
        print(f"  Trades: {r24['total_trades']}")
        print(f"  Win Rate: {r24['win_rate']:.1f}%")
        print(f"  Losing Months: {r24['losing_months']}")

        print(f"\n2025 Results:")
        print(f"  Return: +{r25['total_return']:.1f}%")
        print(f"  Final Balance: ${r25['final_balance']:,.2f}")
        print(f"  Trades: {r25['total_trades']}")
        print(f"  Win Rate: {r25['win_rate']:.1f}%")
        print(f"  Losing Months: {r25['losing_months']}")

        combined = r24['total_return'] + r25['total_return']
        print(f"\nCombined 2-Year Return: +{combined:.1f}%")

        # Monthly breakdown
        print("\n" + "-" * 70)
        print("2024 MONTHLY P/L")
        print("-" * 70)
        for m in r24['monthly']:
            if m.get('skipped'):
                print(f"  {m['month']}: SKIP")
            else:
                status = " X" if m['pnl'] < 0 else ""
                print(f"  {m['month']}: {m['pnl']:>+8,.0f}$ ({m['trades']:>2} trades){status}")

        print("\n" + "-" * 70)
        print("2025 MONTHLY P/L")
        print("-" * 70)
        for m in r25['monthly']:
            if m.get('skipped'):
                print(f"  {m['month']}: SKIP")
            else:
                status = " X" if m['pnl'] < 0 else ""
                print(f"  {m['month']}: {m['pnl']:>+8,.0f}$ ({m['trades']:>2} trades){status}")

    print("\n" + "=" * 70)
    print("Optimization complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
