"""Test Regime Validator on 2025 Data
====================================

Verify that Regime Validator doesn't break the zero losing months in 2025.

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
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.utils.regime_validator import RegimeValidator
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


def run_backtest_with_validator(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                                 year: int, use_validator: bool = True) -> dict:
    """Run backtest with optional regime validator"""

    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'skip_december': True,
    }

    validator = RegimeValidator(
        momentum_period=10,
        momentum_threshold=0.3,
        fast_ema=8,
        slow_ema=21,
        min_momentum_strength=0,  # Check any misalignment
        require_ema_alignment=False
    ) if use_validator else None

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
    total_filtered = 0

    for start_date, end_date in months:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        if is_december and CONFIG.get('skip_december', True):
            monthly_results.append({
                'month': month_name,
                'pnl': 0,
                'trades': 0,
                'filtered': 0,
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
                'filtered': 0,
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
            use_hybrid_mode=False,
            use_choppiness_filter=False
        )

        bt.risk_manager.max_lot_size = CONFIG['max_lot']
        bt.entry_trigger.min_quality_score = CONFIG['min_quality']

        bt.load_data(htf, ltf)
        result = bt.run()

        # Process trades with optional validator
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_filtered = 0
        simulated_balance = running_balance
        max_loss_dollars = simulated_balance * CONFIG['max_loss_per_trade_pct'] / 100

        for trade in result.trade_list:
            # Apply regime validator if enabled
            if validator:
                trade_time = trade.entry_time
                htf_before = htf_month[htf_month.index <= trade_time].tail(30)

                if len(htf_before) >= 15:
                    should_skip, reason = validator.should_skip_trade(
                        htf_before, trade.regime, trade.direction
                    )

                    if should_skip:
                        month_filtered += 1
                        continue

            # Process trade
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
        total_filtered += month_filtered

        monthly_results.append({
            'month': month_name,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
            'filtered': month_filtered,
            'original_trades': len(result.trade_list),
            'balance': running_balance,
            'skipped': False
        })

    losing_months = [m for m in monthly_results if m['pnl'] < 0]

    return {
        'year': year,
        'use_validator': use_validator,
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'losing_month_names': [m['month'] for m in losing_months],
        'total_filtered': total_filtered
    }


async def main():
    print("\n" + "=" * 70)
    print("VERIFY: REGIME VALIDATOR ON 2025 DATA")
    print("Ensure zero losing months is preserved")
    print("=" * 70)

    print("\nFetching data...")
    # Fetch full range covering both 2024 and 2025 with warmup
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Test both 2024 and 2025
    print("\n" + "=" * 70)
    print("2024 COMPARISON")
    print("=" * 70)

    # 2024 without validator
    print("\n2024 WITHOUT Regime Validator...")
    result_2024_no_val = run_backtest_with_validator(htf_df, ltf_df, 2024, use_validator=False)
    print(f"  -> {result_2024_no_val['losing_months']} losing months, +{result_2024_no_val['total_return']:.1f}%, "
          f"{result_2024_no_val['total_trades']} trades")

    # 2024 with validator
    print("\n2024 WITH Regime Validator...")
    result_2024_val = run_backtest_with_validator(htf_df, ltf_df, 2024, use_validator=True)
    print(f"  -> {result_2024_val['losing_months']} losing months, +{result_2024_val['total_return']:.1f}%, "
          f"{result_2024_val['total_trades']} trades (filtered {result_2024_val['total_filtered']})")

    print("\n" + "=" * 70)
    print("2025 COMPARISON")
    print("=" * 70)

    # 2025 without validator
    print("\n2025 WITHOUT Regime Validator...")
    result_2025_no_val = run_backtest_with_validator(htf_df, ltf_df, 2025, use_validator=False)
    print(f"  -> {result_2025_no_val['losing_months']} losing months, +{result_2025_no_val['total_return']:.1f}%, "
          f"{result_2025_no_val['total_trades']} trades")

    # 2025 with validator
    print("\n2025 WITH Regime Validator...")
    result_2025_val = run_backtest_with_validator(htf_df, ltf_df, 2025, use_validator=True)
    print(f"  -> {result_2025_val['losing_months']} losing months, +{result_2025_val['total_return']:.1f}%, "
          f"{result_2025_val['total_trades']} trades (filtered {result_2025_val['total_filtered']})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n{:<30} {:>10} {:>10} {:>10} {:>10}".format(
        "Configuration", "Losing", "Trades", "WinRate", "Return"
    ))
    print("-" * 75)

    configs = [
        ("2024 WITHOUT Validator", result_2024_no_val),
        ("2024 WITH Validator", result_2024_val),
        ("2025 WITHOUT Validator", result_2025_no_val),
        ("2025 WITH Validator", result_2025_val),
    ]

    for name, r in configs:
        print("{:<30} {:>10} {:>10} {:>9.1f}% {:>9.1f}%".format(
            name,
            r['losing_months'],
            r['total_trades'],
            r['win_rate'],
            r['total_return']
        ))

    # Monthly breakdown for 2025 with validator
    print("\n" + "-" * 70)
    print("2025 MONTHLY BREAKDOWN (WITH Validator)")
    print("-" * 70)
    print("{:<10} {:>10} {:>12} {:>10} {:>12}".format(
        "Month", "P/L", "Trades", "Filtered", "Balance"
    ))
    print("-" * 55)

    for m in result_2025_val['monthly']:
        if m.get('skipped', False):
            print("{:<10} {:>10} {:>12} {:>10} {:>11,.0f}$".format(
                m['month'], "SKIP", "-", "-", m['balance']
            ))
        else:
            status = " X" if m['pnl'] < 0 else ""
            orig = m.get('original_trades', m['trades'])
            print("{:<10} {:>+9,.0f}$ {:>12} {:>10} {:>11,.0f}${}".format(
                m['month'], m['pnl'], f"{m['trades']}/{orig}",
                m['filtered'], m['balance'], status
            ))

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if result_2025_val['losing_months'] == 0:
        print("\n[OK] 2025 maintains ZERO LOSING MONTHS with Regime Validator!")
        print(f"     Return: +{result_2025_val['total_return']:.1f}%")
    else:
        print(f"\n[WARNING] 2025 has {result_2025_val['losing_months']} losing months with Regime Validator")
        print(f"          Losing: {result_2025_val['losing_month_names']}")

    if result_2024_val['losing_months'] < result_2024_no_val['losing_months']:
        improvement = result_2024_no_val['losing_months'] - result_2024_val['losing_months']
        print(f"\n[OK] 2024 improved by {improvement} fewer losing months with Regime Validator!")
        print(f"     {result_2024_no_val['losing_months']} -> {result_2024_val['losing_months']} losing months")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if result_2025_val['losing_months'] == 0 and result_2024_val['losing_months'] < result_2024_no_val['losing_months']:
        print("\n*** REGIME VALIDATOR APPROVED FOR INTEGRATION ***")
        print("\nBenefits:")
        print(f"  - 2024: {result_2024_no_val['losing_months']} -> {result_2024_val['losing_months']} losing months")
        print(f"  - 2025: Maintains {result_2025_val['losing_months']} losing months")
        print(f"  - Trade quality improvement through regime validation")
    else:
        print("\n*** NEEDS FURTHER INVESTIGATION ***")

    print("\nTest complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
