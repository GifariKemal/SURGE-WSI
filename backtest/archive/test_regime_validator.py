"""Test Regime Validator Impact on 2024 Losing Months
====================================================

Test if validating regime against price momentum can reduce losses.

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


def run_with_regime_validator(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                               year: int, momentum_period: int,
                               min_momentum_strength: float,
                               require_ema: bool = True) -> dict:
    """Run backtest with regime validator"""

    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'skip_december': True,
    }

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

        # Filter trades using regime validator
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_filtered = 0
        simulated_balance = running_balance
        max_loss_dollars = simulated_balance * CONFIG['max_loss_per_trade_pct'] / 100

        for trade in result.trade_list:
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
        'momentum_period': momentum_period,
        'min_momentum_strength': min_momentum_strength,
        'require_ema': require_ema,
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
    print("TEST: REGIME VALIDATOR")
    print("Validate HMM regime against price momentum")
    print("=" * 70)

    print("\nFetching data...")
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Test configurations
    configs = [
        # (momentum_period, min_strength, require_ema, name)
        (10, 0, False, "Baseline (no filter)"),
        (10, 30, False, "Momentum only (30%)"),
        (10, 30, True, "Momentum + EMA (30%)"),
        (10, 40, True, "Momentum + EMA (40%)"),
        (10, 50, True, "Momentum + EMA (50%)"),
        (5, 30, True, "Short momentum (5 bars)"),
        (15, 30, True, "Long momentum (15 bars)"),
    ]

    results = []

    # First run baseline
    print("\nTesting: Baseline (no filter)...")
    baseline = run_with_regime_validator(htf_df, ltf_df, 2024, 10, 0, False)
    baseline['config_name'] = "Baseline (no filter)"
    results.append(baseline)
    print(f"  -> {baseline['losing_months']} losing, +{baseline['total_return']:.1f}%, {baseline['total_trades']} trades")

    for momentum_period, min_strength, require_ema, name in configs[1:]:
        print(f"\nTesting: {name}...")
        result = run_with_regime_validator(htf_df, ltf_df, 2024,
                                           momentum_period, min_strength, require_ema)
        result['config_name'] = name
        results.append(result)
        print(f"  -> {result['losing_months']} losing, +{result['total_return']:.1f}%, "
              f"{result['total_trades']} trades (filtered {result['total_filtered']})")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n{:<30} {:>7} {:>7} {:>7} {:>9} {:>8}".format(
        "Configuration", "Losing", "Trades", "WinRate", "Return", "Filtered"
    ))
    print("-" * 75)

    for r in results:
        print("{:<30} {:>7} {:>7} {:>6.1f}% {:>8.1f}% {:>8}".format(
            r['config_name'][:30],
            r['losing_months'],
            r['total_trades'],
            r['win_rate'],
            r['total_return'],
            r.get('total_filtered', 0)
        ))

    # Find best
    best = min(results, key=lambda x: (x['losing_months'], -x['total_return']))

    print("\n" + "=" * 70)
    print(f"BEST: {best['config_name']}")
    print("=" * 70)

    print(f"\nLosing Months: {best['losing_months']}")
    if best['losing_months'] > 0:
        print(f"  -> {best['losing_month_names']}")
    print(f"Return: +{best['total_return']:.1f}%")
    print(f"Trades: {best['total_trades']} (filtered {best.get('total_filtered', 0)})")
    print(f"Win Rate: {best['win_rate']:.1f}%")

    # Monthly breakdown
    print("\n" + "-" * 70)
    print("MONTHLY BREAKDOWN (Best Config)")
    print("-" * 70)
    print("{:<10} {:>10} {:>12} {:>10} {:>12}".format(
        "Month", "P/L", "Trades", "Filtered", "Balance"
    ))
    print("-" * 55)

    for m in best['monthly']:
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

    # Impact analysis
    print("\n" + "=" * 70)
    print("IMPACT ANALYSIS")
    print("=" * 70)

    baseline = results[0]
    print(f"\nBaseline: {baseline['losing_months']} losing months, +{baseline['total_return']:.1f}%")
    print(f"Best: {best['losing_months']} losing months, +{best['total_return']:.1f}%")

    if best['losing_months'] < baseline['losing_months']:
        improvement = baseline['losing_months'] - best['losing_months']
        print(f"\nImprovement: {improvement} fewer losing months!")
    elif best['losing_months'] == 0:
        print("\n*** ZERO LOSING MONTHS! ***")

    print("\nTest complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
