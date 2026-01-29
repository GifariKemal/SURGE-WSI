"""Analyze Filter Impact on Trade Count
======================================

Find out which filter causes the most trade reduction.
Goal: Find balance between safety and trade frequency.

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
from dataclasses import dataclass

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester


@dataclass
class TestConfig:
    name: str
    max_sl_pips: float = 50.0
    min_quality: float = 65.0
    max_loss_pct: float = 0.8


async def fetch_data(symbol, timeframe, start, end):
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=200000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def run_test(htf_df, ltf_df, year, cfg: TestConfig):
    """Run backtest with specific config"""
    running_balance = 10000.0
    total_trades = 0
    total_wins = 0
    total_pnl = 0
    filtered_by_sl = 0
    filtered_by_quality = 0
    losing_months = 0

    for month in range(1, 12):
        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 11:
            end_date = datetime(year, 12, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)

        warmup = start_date - timedelta(days=30)
        htf_m = htf_df[(htf_df.index >= warmup) & (htf_df.index <= end_date)]
        ltf_m = ltf_df[(ltf_df.index >= warmup) & (ltf_df.index <= end_date)]

        if htf_m.empty or ltf_m.empty or len(ltf_m) < 100:
            continue

        htf = htf_m.reset_index().rename(columns={'index': 'time'})
        ltf = ltf_m.reset_index().rename(columns={'index': 'time'})

        # Use original settings to get ALL potential trades
        bt = Backtester(
            symbol='GBPUSD',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_balance=running_balance,
            pip_value=10.0, spread_pips=1.5,
            use_killzone=True, use_trend_filter=True,
            use_relaxed_filter=False, use_hybrid_mode=False,
            use_choppiness_filter=False
        )
        bt.risk_manager.max_lot_size = 0.5
        bt.risk_manager.max_sl_pips = cfg.max_sl_pips
        bt.entry_trigger.min_quality_score = cfg.min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        month_pnl = 0
        month_trades = 0

        for trade in result.trade_list:
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001

            # Apply SL filter
            if risk_pips > cfg.max_sl_pips:
                filtered_by_sl += 1
                continue

            # Apply loss cap
            max_loss = running_balance * cfg.max_loss_pct / 100
            if trade.pnl < 0 and abs(trade.pnl) > max_loss:
                adj_pnl = -max_loss
            else:
                adj_pnl = trade.pnl

            running_balance += adj_pnl
            month_pnl += adj_pnl
            month_trades += 1
            total_trades += 1
            total_pnl += adj_pnl

            if adj_pnl > 0:
                total_wins += 1

        if month_pnl < 0:
            losing_months += 1

    return {
        'config': cfg.name,
        'trades': total_trades,
        'wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'return': (running_balance - 10000) / 100,
        'losing_months': losing_months,
        'filtered_sl': filtered_by_sl,
    }


async def main():
    print("\n" + "="*70)
    print("ANALYZE FILTER IMPACT ON TRADE COUNT")
    print("="*70)

    print("\nFetching data...")
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data('GBPUSD', 'H4', start, end)
    ltf_df = await fetch_data('GBPUSD', 'M15', start, end)

    print(f"H4={len(htf_df)}, M15={len(ltf_df)}")

    # Test different configurations
    configs = [
        # Baseline (Original)
        TestConfig("ORIGINAL (SL50/Q65/L0.8)", 50.0, 65.0, 0.8),

        # Only change SL
        TestConfig("SL40 only", 40.0, 65.0, 0.8),
        TestConfig("SL30 only", 30.0, 65.0, 0.8),
        TestConfig("SL25 only", 25.0, 65.0, 0.8),
        TestConfig("SL20 only", 20.0, 65.0, 0.8),
        TestConfig("SL15 only", 15.0, 65.0, 0.8),
        TestConfig("SL10 only", 10.0, 65.0, 0.8),

        # Only change Quality
        TestConfig("Q70 only", 50.0, 70.0, 0.8),
        TestConfig("Q75 only", 50.0, 75.0, 0.8),
        TestConfig("Q80 only", 50.0, 80.0, 0.8),

        # Only change Loss Cap
        TestConfig("L0.5% only", 50.0, 65.0, 0.5),
        TestConfig("L0.3% only", 50.0, 65.0, 0.3),
        TestConfig("L0.2% only", 50.0, 65.0, 0.2),
        TestConfig("L0.1% only", 50.0, 65.0, 0.1),

        # Combinations - Balanced
        TestConfig("SL30/Q70/L0.5", 30.0, 70.0, 0.5),
        TestConfig("SL25/Q70/L0.3", 25.0, 70.0, 0.3),
        TestConfig("SL20/Q70/L0.2", 20.0, 70.0, 0.2),

        # ZERO_LOSS (current)
        TestConfig("ZERO_LOSS (SL10/Q75/L0.1)", 10.0, 75.0, 0.1),

        # Alternative balanced configs
        TestConfig("BALANCED-A (SL25/Q70/L0.2)", 25.0, 70.0, 0.2),
        TestConfig("BALANCED-B (SL20/Q68/L0.25)", 20.0, 68.0, 0.25),
        TestConfig("BALANCED-C (SL30/Q68/L0.3)", 30.0, 68.0, 0.3),
    ]

    print("\nTesting configurations for 2024 & 2025...")
    print("-" * 70)

    results = []
    for cfg in configs:
        r24 = run_test(htf_df, ltf_df, 2024, cfg)
        r25 = run_test(htf_df, ltf_df, 2025, cfg)

        combined = {
            'config': cfg.name,
            'trades_24': r24['trades'],
            'trades_25': r25['trades'],
            'total_trades': r24['trades'] + r25['trades'],
            'losing_24': r24['losing_months'],
            'losing_25': r25['losing_months'],
            'total_losing': r24['losing_months'] + r25['losing_months'],
            'return_24': r24['return'],
            'return_25': r25['return'],
            'total_return': r24['return'] + r25['return'],
        }
        results.append(combined)

        losing_mark = "✓" if combined['total_losing'] == 0 else f"✗{combined['total_losing']}"
        print(f"{cfg.name:<35} | Trades: {combined['total_trades']:>4} | "
              f"Losing: {losing_mark:<3} | Return: {combined['total_return']:>+6.1f}%")

    # Sort by trades (descending) then by losing months (ascending)
    print("\n" + "="*70)
    print("BEST CONFIGS (sorted by trades, then safety)")
    print("="*70)

    # Filter configs with 0-1 losing months and sort by trades
    safe_configs = [r for r in results if r['total_losing'] <= 1]
    safe_configs.sort(key=lambda x: (-x['total_trades'], x['total_losing']))

    print(f"\n{'Config':<35} | {'Trades':>6} | {'Lose':>4} | {'Return':>8}")
    print("-" * 70)
    for r in safe_configs[:10]:
        print(f"{r['config']:<35} | {r['total_trades']:>6} | {r['total_losing']:>4} | {r['total_return']:>+7.1f}%")

    # Find optimal balance
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Best with 0 losing
    zero_losing = [r for r in results if r['total_losing'] == 0]
    if zero_losing:
        best_zero = max(zero_losing, key=lambda x: x['total_trades'])
        print(f"\nBest with 0 losing months:")
        print(f"  {best_zero['config']}")
        print(f"  Trades: {best_zero['total_trades']}, Return: {best_zero['total_return']:+.1f}%")

    # Best with max 1 losing
    one_losing = [r for r in results if r['total_losing'] <= 1]
    if one_losing:
        best_one = max(one_losing, key=lambda x: x['total_trades'])
        print(f"\nBest with ≤1 losing months:")
        print(f"  {best_one['config']}")
        print(f"  Trades: {best_one['total_trades']}, Return: {best_one['total_return']:+.1f}%")

    # Balanced (most trades with good safety)
    balanced = [r for r in results if r['total_losing'] <= 2 and r['total_trades'] >= 50]
    if balanced:
        best_bal = max(balanced, key=lambda x: (x['total_trades'], -x['total_losing']))
        print(f"\nBest balanced (≤2 losing, ≥50 trades):")
        print(f"  {best_bal['config']}")
        print(f"  Trades: {best_bal['total_trades']}, Losing: {best_bal['total_losing']}, Return: {best_bal['total_return']:+.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
