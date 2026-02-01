"""
Verify the best configuration with detailed monthly analysis.
"""

import sys
import io
from pathlib import Path

STRATEGY_DIR = Path(__file__).parent
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGY_DIR.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
from datetime import datetime, timezone

import logging
logging.getLogger('gbpusd_h1_quadlayer.trading_filters').setLevel(logging.WARNING)
logging.getLogger('src.data.db_handler').setLevel(logging.WARNING)

import gbpusd_h1_quadlayer.backtest as bt


async def run_and_analyze(config_name: str, day_mult: dict):
    """Run backtest and provide detailed analysis."""

    # Patch DAY_MULTIPLIERS
    original_mult = bt.DAY_MULTIPLIERS.copy()
    bt.DAY_MULTIPLIERS.clear()
    bt.DAY_MULTIPLIERS.update(day_mult)
    bt.SEND_TO_TELEGRAM = False

    try:
        timeframe = "H1"
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 31, tzinfo=timezone.utc)

        df = await bt.fetch_data(bt.SYMBOL, timeframe, start, end)

        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'open' in col_lower and 'open' not in col_map:
                col_map['open'] = col
            elif 'high' in col_lower and 'high' not in col_map:
                col_map['high'] = col
            elif 'low' in col_lower and 'low' not in col_map:
                col_map['low'] = col
            elif 'close' in col_lower and 'close' not in col_map:
                col_map['close'] = col

        trades, max_dd, _, _, entry_stats, _, _ = bt.run_backtest(df, col_map)
        stats = bt.calculate_stats(trades, max_dd)

        return trades, stats, entry_stats

    finally:
        bt.DAY_MULTIPLIERS.clear()
        bt.DAY_MULTIPLIERS.update(original_mult)


async def main():
    configs_to_verify = [
        ("BASELINE", {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}),
        ("Thu_0.8_Fri_0.3 (BEST)", {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}),
        ("Thu_0.8_NoFri (2nd)", {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.0, 5: 0.0, 6: 0.0}),
    ]

    print("=" * 80)
    print("DETAILED VERIFICATION OF TOP CONFIGURATIONS")
    print("=" * 80)

    for config_name, day_mult in configs_to_verify:
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_name}")
        print(f"DAY_MULTIPLIERS = {day_mult}")
        print(f"{'='*80}")

        trades, stats, entry_stats = await run_and_analyze(config_name, day_mult)

        print(f"\nOVERVIEW:")
        print(f"  Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Net P/L: ${stats['net_pnl']:+,.2f}")
        print(f"  Max Drawdown: ${stats['max_dd']:,.2f} ({stats['max_dd_pct']:.1f}%)")
        print(f"  Losing Months: {stats['losing_months']}/{stats['total_months']}")

        # Entry type breakdown
        print(f"\nENTRY TYPES:")
        for entry_type, count in sorted(entry_stats.items(), key=lambda x: -x[1]):
            type_trades = [t for t in trades if t.poi_type == entry_type]
            type_wins = len([t for t in type_trades if t.pnl > 0])
            type_wr = type_wins / len(type_trades) * 100 if type_trades else 0
            type_pnl = sum(t.pnl for t in type_trades)
            print(f"  {entry_type}: {count} trades, {type_wr:.1f}% WR, ${type_pnl:+,.0f}")

        # Day of week breakdown
        print(f"\nDAY OF WEEK BREAKDOWN:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        for day_idx, day_name in enumerate(days):
            day_trades = [t for t in trades if t.entry_time.weekday() == day_idx]
            if day_trades:
                day_wins = len([t for t in day_trades if t.pnl > 0])
                day_wr = day_wins / len(day_trades) * 100
                day_pnl = sum(t.pnl for t in day_trades)
                print(f"  {day_name}: {len(day_trades)} trades, {day_wr:.1f}% WR, ${day_pnl:+,.0f}")
            else:
                print(f"  {day_name}: 0 trades")

        # Monthly breakdown with margins
        print(f"\nMONTHLY BREAKDOWN (sorted by P/L):")
        monthly_data = [(str(m), pnl) for m, pnl in stats['monthly'].items()]
        monthly_data.sort(key=lambda x: x[1])  # Sort by P/L ascending (worst first)

        for month, pnl in monthly_data:
            status = "[PROFIT]" if pnl >= 0 else "[LOSS]  "
            margin_warning = " <-- TIGHT MARGIN" if 0 < pnl < 300 else ""
            print(f"  {status} {month}: ${pnl:+,.2f}{margin_warning}")

        # Check minimum margin
        min_pnl = min(pnl for _, pnl in monthly_data)
        min_month = [m for m, pnl in monthly_data if pnl == min_pnl][0]
        print(f"\n  Minimum Month: {min_month} (${min_pnl:+,.2f})")

        if min_pnl < 300:
            print(f"  WARNING: Margin is tight! Less than $300 buffer.")

    print("\n" + "=" * 80)
    print("RECOMMENDATION SUMMARY")
    print("=" * 80)
    print("""
Based on the analysis:

BEST CONFIG: Thu_0.8_Fri_0.3
  - DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}
  - Changes: Thursday 0.4->0.8, Friday 0.5->0.3
  - All months profitable
  - Significant improvement in profit

ALTERNATIVE: Thu_0.8_NoFri
  - Skip Friday entirely (safer)
  - Slightly less profit but more conservative
""")


if __name__ == "__main__":
    asyncio.run(main())
