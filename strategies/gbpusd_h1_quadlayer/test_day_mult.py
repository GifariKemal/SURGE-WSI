"""
Quick test script for DAY_MULTIPLIERS optimization.
Patches the DAY_MULTIPLIERS in backtest.py and runs the test.
"""

import sys
import io
from pathlib import Path

# Strategy directory (where this file is located)
STRATEGY_DIR = Path(__file__).parent
# Project root is 2 levels up
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add strategies folder to allow package imports
sys.path.insert(0, str(STRATEGY_DIR.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
from datetime import datetime, timezone

# Suppress logging
import logging
logging.getLogger('gbpusd_h1_quadlayer.trading_filters').setLevel(logging.WARNING)
logging.getLogger('src.data.db_handler').setLevel(logging.WARNING)

# Import backtest module
import gbpusd_h1_quadlayer.backtest as bt

# Test configurations
TEST_CONFIGS = {
    "BASELINE":         {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0},
    "Thu_0.5":          {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.0, 6: 0.0},
    "Thu_0.6":          {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.6, 4: 0.5, 5: 0.0, 6: 0.0},
    "Thu_0.7":          {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.7, 4: 0.5, 5: 0.0, 6: 0.0},
    "Thu_0.8":          {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.5, 5: 0.0, 6: 0.0},
    "Thu_0.6_Fri_0.4":  {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.6, 4: 0.4, 5: 0.0, 6: 0.0},
    "Thu_0.7_Fri_0.3":  {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.7, 4: 0.3, 5: 0.0, 6: 0.0},
    "Thu_0.8_Fri_0.3":  {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0},
    "NoFri":            {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.0, 5: 0.0, 6: 0.0},
    "Thu_0.6_NoFri":    {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.6, 4: 0.0, 5: 0.0, 6: 0.0},
    "Thu_0.8_NoFri":    {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.0, 5: 0.0, 6: 0.0},
    "Tue_1.0":          {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0},
    "Tue_1.0_Thu_0.6":  {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.6, 4: 0.5, 5: 0.0, 6: 0.0},
}


async def run_test(config_name: str, day_mult: dict) -> dict:
    """Run a single backtest with given day multipliers."""
    # Patch the DAY_MULTIPLIERS in the backtest module
    original_mult = bt.DAY_MULTIPLIERS.copy()
    bt.DAY_MULTIPLIERS.clear()
    bt.DAY_MULTIPLIERS.update(day_mult)

    try:
        # Fetch data
        timeframe = "H1"
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 31, tzinfo=timezone.utc)

        df = await bt.fetch_data(bt.SYMBOL, timeframe, start, end)
        if df.empty:
            return {"error": "No data"}

        # Build col_map
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

        # Run backtest
        trades, max_dd, condition_stats, skip_stats, entry_stats, h4_stats, structure_stats = bt.run_backtest(df, col_map)

        if not trades:
            return {"error": "No trades"}

        # Calculate stats
        stats = bt.calculate_stats(trades, max_dd)

        return {
            'config': config_name,
            'day_mult': day_mult,
            'trades': stats['total_trades'],
            'win_rate': stats['win_rate'],
            'pf': stats['profit_factor'],
            'net_pnl': stats['net_pnl'],
            'losing_months': stats['losing_months'],
            'monthly': stats['monthly']
        }
    finally:
        # Restore original multipliers
        bt.DAY_MULTIPLIERS.clear()
        bt.DAY_MULTIPLIERS.update(original_mult)


async def main():
    print("=" * 80)
    print("SURGE-WSI H1 GBPUSD - DAY MULTIPLIER OPTIMIZATION")
    print("=" * 80)
    print(f"\nTesting {len(TEST_CONFIGS)} configurations...")
    print("This will take a few minutes...\n")

    # Disable telegram during tests
    original_telegram = bt.SEND_TO_TELEGRAM
    bt.SEND_TO_TELEGRAM = False

    results = []

    for config_name, day_mult in TEST_CONFIGS.items():
        print(f"Testing: {config_name}...", end=" ", flush=True)

        result = await run_test(config_name, day_mult)

        if 'error' in result:
            print(f"ERROR: {result['error']}")
            continue

        status = "[OK]" if result['losing_months'] == 0 else "[FAIL]"
        print(f"{status} {result['trades']} trades, {result['win_rate']:.1f}% WR, "
              f"PF {result['pf']:.2f}, ${result['net_pnl']:+,.0f}, "
              f"LM={result['losing_months']}")

        results.append(result)

    # Restore telegram setting
    bt.SEND_TO_TELEGRAM = original_telegram

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Filter only 0 losing months
    valid_results = [r for r in results if r['losing_months'] == 0]

    print(f"\nConfigurations with 0 losing months: {len(valid_results)}/{len(results)}")

    if valid_results:
        # Sort by profit
        valid_results.sort(key=lambda x: x['net_pnl'], reverse=True)

        print("\n" + "-" * 80)
        print(f"{'Config':<20} {'Trades':>7} {'WR':>6} {'PF':>6} {'Net P/L':>12}")
        print("-" * 80)

        for r in valid_results:
            print(f"{r['config']:<20} {r['trades']:>7} {r['win_rate']:>5.1f}% "
                  f"{r['pf']:>6.2f} ${r['net_pnl']:>10,.0f}")

        # Best configuration
        best = valid_results[0]

        print("\n" + "=" * 80)
        print("BEST CONFIGURATION (0 losing months)")
        print("=" * 80)
        print(f"\nConfig: {best['config']}")
        print(f"DAY_MULTIPLIERS = {best['day_mult']}")
        print(f"\nPerformance:")
        print(f"  Trades: {best['trades']}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Profit Factor: {best['pf']:.2f}")
        print(f"  Net P/L: ${best['net_pnl']:+,.2f}")

        print(f"\nMonthly Breakdown:")
        for month, pnl in best['monthly'].items():
            status = "WIN " if pnl >= 0 else "LOSS"
            print(f"  [{status}] {month}: ${pnl:+,.2f}")

        # Improvement over baseline
        baseline = next((r for r in results if r['config'] == 'BASELINE'), None)
        if baseline:
            improvement = best['net_pnl'] - baseline['net_pnl']
            print(f"\nImprovement vs BASELINE: ${improvement:+,.2f}")
    else:
        print("\nNo configuration found with 0 losing months!")

        # Show all results sorted
        results.sort(key=lambda x: (x['losing_months'], -x['net_pnl']))
        print("\nAll results (sorted by losing months):")
        for r in results[:5]:
            print(f"  {r['config']}: LM={r['losing_months']}, ${r['net_pnl']:+,.0f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
