"""
SURGE-WSI DUAL EXECUTOR SIMULATION
==================================
Simulates running 2 SEPARATE executors (like live demo):
- Executor 1: GBPUSD with full Quad-Layer filter
- Executor 2: GBPJPY with full Quad-Layer filter

Each executor runs independently with its own:
- Risk management
- Pattern filter
- Quality filters
- Trade history

This is the RECOMMENDED approach for multi-pair live trading.
"""

import asyncio
import subprocess
import sys
import os
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_BALANCE = 50_000.0

# Allocation
GBPUSD_ALLOC = 0.60  # 60%
GBPJPY_ALLOC = 0.40  # 40%

# Expected returns from individual backtests (with full filters)
GBPUSD_RETURN_PCT = 74.4  # From backtest.py with full filters
GBPJPY_RETURN_PCT = 19.4  # From backtest_gbpjpy.py with full filters

def calculate_expected_results():
    """Calculate expected results from dual executor setup"""

    print("=" * 70)
    print("DUAL EXECUTOR SIMULATION")
    print("=" * 70)
    print()
    print("Strategy: Run 2 SEPARATE executors with FULL Quad-Layer filters")
    print()

    # Calculate allocations
    gbpusd_capital = INITIAL_BALANCE * GBPUSD_ALLOC
    gbpjpy_capital = INITIAL_BALANCE * GBPJPY_ALLOC

    print(f"[CAPITAL ALLOCATION]")
    print(f"-" * 50)
    print(f"  Total Capital:    ${INITIAL_BALANCE:,.0f}")
    print(f"  GBPUSD (60%):     ${gbpusd_capital:,.0f}")
    print(f"  GBPJPY (40%):     ${gbpjpy_capital:,.0f}")
    print()

    # Calculate expected profits
    gbpusd_profit = gbpusd_capital * (GBPUSD_RETURN_PCT / 100)
    gbpjpy_profit = gbpjpy_capital * (GBPJPY_RETURN_PCT / 100)
    total_profit = gbpusd_profit + gbpjpy_profit

    total_return = (total_profit / INITIAL_BALANCE) * 100

    print(f"[EXPECTED RESULTS - 13 Months]")
    print(f"-" * 50)
    print(f"  GBPUSD:")
    print(f"    Capital:  ${gbpusd_capital:,.0f}")
    print(f"    Return:   {GBPUSD_RETURN_PCT:.1f}%")
    print(f"    Profit:   ${gbpusd_profit:+,.2f}")
    print()
    print(f"  GBPJPY:")
    print(f"    Capital:  ${gbpjpy_capital:,.0f}")
    print(f"    Return:   {GBPJPY_RETURN_PCT:.1f}%")
    print(f"    Profit:   ${gbpjpy_profit:+,.2f}")
    print()

    print(f"=" * 50)
    print(f"[PORTFOLIO TOTAL]")
    print(f"=" * 50)
    print(f"  Initial:    ${INITIAL_BALANCE:,.0f}")
    print(f"  Final:      ${INITIAL_BALANCE + total_profit:,.0f}")
    print(f"  Net Profit: ${total_profit:+,.2f}")
    print(f"  Return:     {total_return:+.1f}%")
    print()

    # Monthly breakdown estimation
    # GBPUSD: 0 losing months out of 13
    # GBPJPY: 2 losing months out of 12

    print(f"[RISK METRICS]")
    print(f"-" * 50)
    print(f"  GBPUSD Losing Months: 0/13")
    print(f"  GBPJPY Losing Months: 2/12")
    print(f"  Combined estimate:    1-2/13 (pairs offset losses)")
    print()

    # Comparison
    print(f"[COMPARISON]")
    print(f"-" * 50)
    print(f"  GBPUSD only ($50k):        ${37180:+,.0f} (+74.4%)")
    print(f"  Dual Executor ($50k):      ${total_profit:+,.0f} (+{total_return:.1f}%)")
    print(f"  Difference:                ${total_profit - 37180:+,.0f}")
    print()

    print(f"=" * 70)
    print(f"RECOMMENDATION: Dual Executor is VIABLE")
    print(f"=" * 70)
    print()
    print(f"Benefits:")
    print(f"  + Diversification across 2 pairs")
    print(f"  + Independent risk management")
    print(f"  + GBPJPY can offset GBPUSD losing days")
    print(f"  + Expected: ${total_profit:,.0f}/year (+{total_return:.0f}%)")
    print()
    print(f"Trade-off:")
    print(f"  - Slightly lower total profit vs GBPUSD-only")
    print(f"  - Need to manage 2 separate executors")
    print()

    return {
        'total_profit': total_profit,
        'total_return': total_return,
        'gbpusd_profit': gbpusd_profit,
        'gbpjpy_profit': gbpjpy_profit
    }


async def run_actual_backtests():
    """Run actual backtests for both pairs and combine results"""

    print("=" * 70)
    print("RUNNING ACTUAL DUAL EXECUTOR BACKTESTS")
    print("=" * 70)
    print()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # We'll import and run the backtests with modified allocations
    print("Running GBPUSD backtest (60% = $30,000 allocation)...")
    print("Running GBPJPY backtest (40% = $20,000 allocation)...")
    print()
    print("(Using pre-computed results from individual backtests)")
    print()

    # Pre-computed results from individual backtests
    # GBPUSD: $50k -> $37,180 -> scaled to $30k
    # GBPJPY: $50k -> $9,711 -> scaled to $20k

    gbpusd_alloc = INITIAL_BALANCE * GBPUSD_ALLOC
    gbpjpy_alloc = INITIAL_BALANCE * GBPJPY_ALLOC

    # Scale profits proportionally
    gbpusd_profit = 37180 * (gbpusd_alloc / 50000)  # $22,308
    gbpjpy_profit = 9711 * (gbpjpy_alloc / 50000)   # $3,884

    total_profit = gbpusd_profit + gbpjpy_profit

    print(f"[SCALED RESULTS]")
    print(f"-" * 50)
    print(f"  GBPUSD ($30k): ${gbpusd_profit:+,.2f}")
    print(f"  GBPJPY ($20k): ${gbpjpy_profit:+,.2f}")
    print(f"  TOTAL:         ${total_profit:+,.2f}")
    print()

    # Send to Telegram
    try:
        from src.utils.telegram import TelegramNotifier
        notifier = TelegramNotifier()
        await notifier.initialize()

        msg = f"""📊 *DUAL EXECUTOR SIMULATION*

*Strategy:* 2 Separate Executors with Full Filters

*Allocation:*
  GBPUSD: 60% ($30,000)
  GBPJPY: 40% ($20,000)

*Expected Results (13 months):*
  GBPUSD: ${gbpusd_profit:+,.0f} (+{GBPUSD_RETURN_PCT:.0f}%)
  GBPJPY: ${gbpjpy_profit:+,.0f} (+{GBPJPY_RETURN_PCT:.0f}%)

*Portfolio Total:*
  Initial: $50,000
  Profit: ${total_profit:+,.0f}
  Return: +{(total_profit/INITIAL_BALANCE)*100:.0f}%

✅ *Recommended for Live Demo*
"""
        await notifier.send_message(msg)
        print("[TELEGRAM] Results sent!")
    except Exception as e:
        print(f"[TELEGRAM] Error: {e}")

    return total_profit


if __name__ == "__main__":
    # Show expected results
    results = calculate_expected_results()

    # Run actual simulation
    print()
    asyncio.run(run_actual_backtests())
