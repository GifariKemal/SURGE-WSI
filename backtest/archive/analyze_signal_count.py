"""Analyze Signal Count
======================

Find out what's limiting the number of trades.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester


async def main():
    print("\n" + "="*60)
    print("ANALYZE SIGNAL COUNT")
    print("="*60)

    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()

    # Get March 2024 data (a month with decent trades)
    start = datetime(2024, 2, 15, tzinfo=timezone.utc)
    end = datetime(2024, 3, 31, tzinfo=timezone.utc)

    htf_df = await db.get_ohlcv('GBPUSD', 'H4', 100000, start, end)
    ltf_df = await db.get_ohlcv('GBPUSD', 'M15', 100000, start, end)
    await db.disconnect()

    print(f"\nMarch 2024 data: H4={len(htf_df)}, M15={len(ltf_df)}")

    htf = htf_df.reset_index().rename(columns={'index': 'time'})
    ltf = ltf_df.reset_index().rename(columns={'index': 'time'})

    # Test different filter combinations
    tests = [
        {"name": "ALL FILTERS ON", "killzone": True, "trend": True, "quality": 65.0},
        {"name": "NO KILLZONE", "killzone": False, "trend": True, "quality": 65.0},
        {"name": "NO TREND FILTER", "killzone": True, "trend": False, "quality": 65.0},
        {"name": "QUALITY 50", "killzone": True, "trend": True, "quality": 50.0},
        {"name": "QUALITY 40", "killzone": True, "trend": True, "quality": 40.0},
        {"name": "MINIMAL FILTERS", "killzone": False, "trend": False, "quality": 40.0},
    ]

    print("\n" + "-"*60)
    print(f"{'Test':<25} | {'Trades':>6} | {'P/L':>10}")
    print("-"*60)

    for test in tests:
        bt = Backtester(
            symbol='GBPUSD',
            start_date='2024-03-01',
            end_date='2024-03-31',
            initial_balance=10000,
            pip_value=10.0, spread_pips=1.5,
            use_killzone=test['killzone'],
            use_trend_filter=test['trend'],
            use_relaxed_filter=False,
            use_hybrid_mode=False,
            use_choppiness_filter=False
        )
        bt.risk_manager.max_sl_pips = 50.0
        bt.entry_trigger.min_quality_score = test['quality']

        bt.load_data(htf, ltf)
        result = bt.run()

        pnl = result.final_balance - 10000
        print(f"{test['name']:<25} | {len(result.trade_list):>6} | ${pnl:>+9.2f}")

    # Now test full year with minimal filters
    print("\n" + "="*60)
    print("FULL YEAR TEST (2024) - Minimal Filters")
    print("="*60)

    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    htf_df = await db.get_ohlcv('GBPUSD', 'H4', 100000, start, end)
    ltf_df = await db.get_ohlcv('GBPUSD', 'M15', 100000, start, end)
    await db.disconnect()

    # Run full year with minimal filters
    monthly_trades = []
    for month in range(1, 12):
        start_date = datetime(2024, month, 1, tzinfo=timezone.utc)
        if month == 11:
            end_date = datetime(2024, 12, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end_date = datetime(2024, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)

        warmup = start_date - timedelta(days=30)
        htf_m = htf_df[(htf_df.index >= warmup) & (htf_df.index <= end_date)]
        ltf_m = ltf_df[(ltf_df.index >= warmup) & (ltf_df.index <= end_date)]

        if htf_m.empty or ltf_m.empty:
            monthly_trades.append(0)
            continue

        htf = htf_m.reset_index().rename(columns={'index': 'time'})
        ltf = ltf_m.reset_index().rename(columns={'index': 'time'})

        bt = Backtester(
            symbol='GBPUSD',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_balance=10000,
            pip_value=10.0, spread_pips=1.5,
            use_killzone=False,  # OFF
            use_trend_filter=False,  # OFF
            use_relaxed_filter=False,
            use_hybrid_mode=False,
            use_choppiness_filter=False
        )
        bt.risk_manager.max_sl_pips = 50.0
        bt.entry_trigger.min_quality_score = 40.0  # Very low

        bt.load_data(htf, ltf)
        result = bt.run()

        monthly_trades.append(len(result.trade_list))
        print(f"  2024-{month:02d}: {len(result.trade_list)} trades")

    total = sum(monthly_trades)
    print(f"\nTotal 2024 with MINIMAL FILTERS: {total} trades")
    print(f"Average per month: {total/11:.1f} trades")

    print("\n" + "="*60)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
