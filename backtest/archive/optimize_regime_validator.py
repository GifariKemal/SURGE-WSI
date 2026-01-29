"""Optimize Regime Validator Parameters
======================================

Find parameters that improve 2024 without breaking 2025.

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


def run_backtest_with_config(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                             year: int, momentum_period: int,
                             min_momentum_strength: float,
                             require_ema: bool) -> dict:
    """Run backtest with specific validator config"""

    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'skip_december': True,
    }

    # Create validator if min_momentum_strength > 0 (0 means no validator)
    validator = None
    if min_momentum_strength > 0:
        validator = RegimeValidator(
            momentum_period=momentum_period,
            momentum_threshold=0.3,
            fast_ema=8,
            slow_ema=21,
            min_momentum_strength=min_momentum_strength,
            require_ema_alignment=require_ema
        )

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
    print("OPTIMIZE: REGIME VALIDATOR PARAMETERS")
    print("Find balance between 2024 improvement and 2025 preservation")
    print("=" * 70)

    print("\nFetching data...")
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Test configurations: (momentum_period, min_strength, require_ema, name)
    configs = [
        (10, 0, False, "No Validator (Baseline)"),
        (10, 20, False, "Momentum 20%"),
        (10, 30, False, "Momentum 30%"),
        (10, 40, False, "Momentum 40%"),
        (10, 50, False, "Momentum 50%"),
        (10, 60, False, "Momentum 60%"),
        (10, 70, False, "Momentum 70%"),
        (10, 40, True, "Momentum 40% + EMA"),
        (10, 50, True, "Momentum 50% + EMA"),
        (10, 60, True, "Momentum 60% + EMA"),
    ]

    results_2024 = []
    results_2025 = []

    for momentum_period, min_strength, require_ema, name in configs:
        print(f"\nTesting: {name}...")

        # Test 2024
        r2024 = run_backtest_with_config(htf_df, ltf_df, 2024, momentum_period, min_strength, require_ema)
        r2024['config_name'] = name
        results_2024.append(r2024)

        # Test 2025
        r2025 = run_backtest_with_config(htf_df, ltf_df, 2025, momentum_period, min_strength, require_ema)
        r2025['config_name'] = name
        results_2025.append(r2025)

        print(f"  2024: {r2024['losing_months']} losing, +{r2024['total_return']:.1f}%, {r2024['total_trades']} trades")
        print(f"  2025: {r2025['losing_months']} losing, +{r2025['total_return']:.1f}%, {r2025['total_trades']} trades")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON: ALL CONFIGURATIONS")
    print("=" * 70)

    print("\n{:<25} | {:^25} | {:^25}".format(
        "Configuration", "2024", "2025"
    ))
    print("{:<25} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7}".format(
        "", "Losing", "Trades", "Return", "Losing", "Trades", "Return"
    ))
    print("-" * 80)

    for i, name in enumerate([c[3] for c in configs]):
        r2024 = results_2024[i]
        r2025 = results_2025[i]
        print("{:<25} | {:>8} {:>8} {:>6.1f}% | {:>8} {:>8} {:>6.1f}%".format(
            name[:25],
            r2024['losing_months'],
            r2024['total_trades'],
            r2024['total_return'],
            r2025['losing_months'],
            r2025['total_trades'],
            r2025['total_return']
        ))

    # Find configurations that:
    # 1. Reduce 2024 losing months
    # 2. Maintain 2025 at 0 losing months
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATIONS")
    print("Goal: Improve 2024 while keeping 2025 at 0 losing months")
    print("=" * 70)

    baseline_2024 = results_2024[0]['losing_months']
    optimal = []

    for i, name in enumerate([c[3] for c in configs]):
        r2024 = results_2024[i]
        r2025 = results_2025[i]

        # Check if configuration improves 2024 without hurting 2025
        if r2024['losing_months'] < baseline_2024 and r2025['losing_months'] == 0:
            optimal.append({
                'name': name,
                '2024_losing': r2024['losing_months'],
                '2024_return': r2024['total_return'],
                '2024_trades': r2024['total_trades'],
                '2025_losing': r2025['losing_months'],
                '2025_return': r2025['total_return'],
                '2025_trades': r2025['total_trades'],
                'combined_return': r2024['total_return'] + r2025['total_return']
            })

    if optimal:
        print("\nFound optimal configurations:")
        for opt in sorted(optimal, key=lambda x: (-x['combined_return'], x['2024_losing'])):
            print(f"\n  {opt['name']}:")
            print(f"    2024: {opt['2024_losing']} losing months, +{opt['2024_return']:.1f}%")
            print(f"    2025: {opt['2025_losing']} losing months, +{opt['2025_return']:.1f}%")
            print(f"    Combined Return: +{opt['combined_return']:.1f}%")
    else:
        print("\nNo configuration found that improves 2024 while keeping 2025 at 0.")
        print("Consider: The optimal strategy may be to NOT use the validator")
        print("since 2025 already achieves zero losing months.")

        # Find best trade-off
        print("\nBest trade-off configurations (accept some 2025 degradation):")
        tradeoffs = []
        for i, name in enumerate([c[3] for c in configs]):
            r2024 = results_2024[i]
            r2025 = results_2025[i]

            if r2024['losing_months'] < baseline_2024:
                improvement = baseline_2024 - r2024['losing_months']
                degradation = r2025['losing_months']
                score = improvement - degradation * 2  # Weight degradation more heavily
                tradeoffs.append({
                    'name': name,
                    '2024_losing': r2024['losing_months'],
                    '2024_return': r2024['total_return'],
                    '2025_losing': r2025['losing_months'],
                    '2025_return': r2025['total_return'],
                    'improvement': improvement,
                    'degradation': degradation,
                    'score': score
                })

        for t in sorted(tradeoffs, key=lambda x: -x['score'])[:3]:
            print(f"\n  {t['name']}:")
            print(f"    2024: {baseline_2024} -> {t['2024_losing']} ({t['improvement']} fewer losing months)")
            print(f"    2025: 0 -> {t['2025_losing']} ({t['degradation']} more losing months)")
            print(f"    Net score: {t['score']}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Baseline comparison
    print(f"\nBaseline (No Validator):")
    print(f"  2024: {results_2024[0]['losing_months']} losing months, +{results_2024[0]['total_return']:.1f}%")
    print(f"  2025: {results_2025[0]['losing_months']} losing months, +{results_2025[0]['total_return']:.1f}%")
    print(f"  Combined: +{results_2024[0]['total_return'] + results_2025[0]['total_return']:.1f}%")

    print("\nTest complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
