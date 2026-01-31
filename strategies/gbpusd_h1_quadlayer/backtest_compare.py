"""
SURGE-WSI H1 v6.4 GBPUSD - PARTIAL TP & TRAILING STOP COMPARISON
================================================================

Compares two exit strategies:
1. Fixed TP (Original - 2:1 RR)
2. Partial TP + Trailing Stop (50% at 1.5:1, trail at 1x ATR)

Usage:
    python backtest_compare.py

Author: SURIOTA Team
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

# Import from main backtest module in current directory
import importlib.util
spec = importlib.util.spec_from_file_location("backtest_module", STRATEGY_DIR / "backtest.py")
backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest)

# Get functions and constants
fetch_data = backtest.fetch_data
run_backtest = backtest.run_backtest
calculate_stats = backtest.calculate_stats
print_results = backtest.print_results
send_telegram_report = backtest.send_telegram_report
SYMBOL = backtest.SYMBOL
USE_ORDER_BLOCK = backtest.USE_ORDER_BLOCK
USE_EMA_PULLBACK = backtest.USE_EMA_PULLBACK
USE_VECTOR_FEATURES = backtest.USE_VECTOR_FEATURES
VECTOR_AVAILABLE = backtest.VECTOR_AVAILABLE
PARTIAL_TP_PERCENT = backtest.PARTIAL_TP_PERCENT
PARTIAL_TP_RR = backtest.PARTIAL_TP_RR
TRAIL_ATR_MULT = backtest.TRAIL_ATR_MULT

# Try to get H4 bias config
USE_H4_BIAS = getattr(backtest, 'USE_H4_BIAS', False)


def run_comparison_backtest(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Run both Fixed TP and Partial TP + Trailing Stop backtests for comparison.
    Returns a dict with both results.
    """
    results = {}

    # ============================================================
    # RUN 1: FIXED TP (Original strategy)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"RUNNING FIXED TP BACKTEST (Original - 2:1 RR)")
    print(f"{'='*70}")

    # Disable partial TP and trailing stop
    backtest.USE_PARTIAL_TP = False
    backtest.USE_TRAILING_STOP = False
    backtest.MOVE_SL_TO_BE = False

    result_fixed = run_backtest(df, col_map)
    # Handle variable return values (5, 6, or 7)
    if len(result_fixed) == 7:
        trades_fixed, max_dd_fixed, cond_fixed, skip_fixed, entry_fixed, h4_fixed, struct_fixed = result_fixed
    elif len(result_fixed) == 6:
        trades_fixed, max_dd_fixed, cond_fixed, skip_fixed, entry_fixed, h4_fixed = result_fixed
        struct_fixed = {}
    else:
        trades_fixed, max_dd_fixed, cond_fixed, skip_fixed, entry_fixed = result_fixed
        h4_fixed = {}
        struct_fixed = {}

    stats_fixed = calculate_stats(trades_fixed, max_dd_fixed) if trades_fixed else None
    results['fixed'] = {
        'trades': trades_fixed,
        'stats': stats_fixed,
        'max_dd': max_dd_fixed,
        'condition_stats': cond_fixed,
        'skip_stats': skip_fixed,
        'entry_stats': entry_fixed,
        'h4_stats': h4_fixed,
        'structure_stats': struct_fixed
    }

    # ============================================================
    # RUN 2: PARTIAL TP + TRAILING STOP
    # ============================================================
    print(f"\n{'='*70}")
    print(f"RUNNING PARTIAL TP + TRAILING STOP BACKTEST")
    print(f"  - {PARTIAL_TP_PERCENT*100:.0f}% close at {PARTIAL_TP_RR}:1 RR")
    print(f"  - SL moved to breakeven")
    print(f"  - Trailing stop at {TRAIL_ATR_MULT}x ATR")
    print(f"{'='*70}")

    # Enable partial TP and trailing stop
    backtest.USE_PARTIAL_TP = True
    backtest.USE_TRAILING_STOP = True
    backtest.MOVE_SL_TO_BE = True

    result_partial = run_backtest(df, col_map)
    # Handle variable return values (5, 6, or 7)
    if len(result_partial) == 7:
        trades_partial, max_dd_partial, cond_partial, skip_partial, entry_partial, h4_partial, struct_partial = result_partial
    elif len(result_partial) == 6:
        trades_partial, max_dd_partial, cond_partial, skip_partial, entry_partial, h4_partial = result_partial
        struct_partial = {}
    else:
        trades_partial, max_dd_partial, cond_partial, skip_partial, entry_partial = result_partial
        h4_partial = {}
        struct_partial = {}

    stats_partial = calculate_stats(trades_partial, max_dd_partial) if trades_partial else None
    results['partial'] = {
        'trades': trades_partial,
        'stats': stats_partial,
        'max_dd': max_dd_partial,
        'condition_stats': cond_partial,
        'skip_stats': skip_partial,
        'entry_stats': entry_partial,
        'h4_stats': h4_partial,
        'structure_stats': struct_partial
    }

    return results


def print_comparison_results(results: dict):
    """Print side-by-side comparison of Fixed TP vs Partial TP + Trailing"""
    fixed = results['fixed']['stats']
    partial = results['partial']['stats']

    if not fixed or not partial:
        print("Error: Missing results for comparison")
        return

    print(f"\n{'='*80}")
    print(f"{'COMPARISON: FIXED TP vs PARTIAL TP + TRAILING STOP':^80}")
    print(f"{'='*80}")

    print(f"\n{'Metric':<25} {'Fixed TP':>20} {'Partial+Trail':>20} {'Diff':>12}")
    print(f"{'-'*25} {'-'*20} {'-'*20} {'-'*12}")

    # Trade counts
    print(f"{'Total Trades':<25} {fixed['total_trades']:>20} {partial['total_trades']:>20} {partial['total_trades']-fixed['total_trades']:>+12}")
    print(f"{'Winners':<25} {fixed['winners']:>20} {partial['winners']:>20} {partial['winners']-fixed['winners']:>+12}")
    print(f"{'Losers':<25} {fixed['losers']:>20} {partial['losers']:>20} {partial['losers']-fixed['losers']:>+12}")

    # Win rate
    wr_diff = partial['win_rate'] - fixed['win_rate']
    print(f"{'Win Rate':<25} {fixed['win_rate']:>19.1f}% {partial['win_rate']:>19.1f}% {wr_diff:>+11.1f}%")

    # P&L
    pnl_diff = partial['net_pnl'] - fixed['net_pnl']
    print(f"{'Net P/L':<25} ${fixed['net_pnl']:>18,.0f} ${partial['net_pnl']:>18,.0f} ${pnl_diff:>+10,.0f}")

    # Profit Factor
    pf_diff = partial['profit_factor'] - fixed['profit_factor']
    print(f"{'Profit Factor':<25} {fixed['profit_factor']:>20.2f} {partial['profit_factor']:>20.2f} {pf_diff:>+12.2f}")

    # Average Win/Loss
    print(f"{'Avg Win':<25} ${fixed['avg_win']:>18,.0f} ${partial['avg_win']:>18,.0f} ${partial['avg_win']-fixed['avg_win']:>+10,.0f}")
    print(f"{'Avg Loss':<25} ${fixed['avg_loss']:>18,.0f} ${partial['avg_loss']:>18,.0f} ${partial['avg_loss']-fixed['avg_loss']:>+10,.0f}")

    # Drawdown
    dd_diff = partial['max_dd'] - fixed['max_dd']
    print(f"{'Max Drawdown':<25} ${fixed['max_dd']:>18,.0f} ${partial['max_dd']:>18,.0f} ${dd_diff:>+10,.0f}")
    print(f"{'Max DD %':<25} {fixed['max_dd_pct']:>19.1f}% {partial['max_dd_pct']:>19.1f}% {partial['max_dd_pct']-fixed['max_dd_pct']:>+11.1f}%")

    # Losing months
    print(f"{'Losing Months':<25} {fixed['losing_months']:>20} {partial['losing_months']:>20} {partial['losing_months']-fixed['losing_months']:>+12}")

    # Final balance
    print(f"{'Final Balance':<25} ${fixed['final_balance']:>18,.0f} ${partial['final_balance']:>18,.0f} ${partial['final_balance']-fixed['final_balance']:>+10,.0f}")

    print(f"\n{'='*80}")

    # Partial TP specific stats
    partial_trades = results['partial']['trades']
    if partial_trades:
        partial_closed_trades = [t for t in partial_trades if t.partial_closed]
        trail_exits = [t for t in partial_trades if 'TRAIL' in t.exit_reason]
        be_exits = [t for t in partial_trades if t.exit_reason == 'BE_SL']
        full_tp = [t for t in partial_trades if t.exit_reason == 'TP']

        print(f"\n[PARTIAL TP + TRAILING DETAILS]")
        print(f"{'-'*50}")
        print(f"  Trades hit partial TP: {len(partial_closed_trades)}/{len(partial_trades)} ({len(partial_closed_trades)/len(partial_trades)*100:.1f}%)")
        print(f"  Exit by trailing stop: {len(trail_exits)}")
        print(f"  Exit at breakeven: {len(be_exits)}")
        print(f"  Exit at full TP: {len(full_tp)}")

        if partial_closed_trades:
            partial_pnl_sum = sum(t.partial_pnl for t in partial_closed_trades)
            remaining_pnl_sum = sum(t.pnl - t.partial_pnl for t in partial_closed_trades)
            print(f"  Partial TP contribution: ${partial_pnl_sum:+,.0f}")
            print(f"  Remaining pos contribution: ${remaining_pnl_sum:+,.0f}")

    # Recommendation
    print(f"\n[RECOMMENDATION]")
    if partial['net_pnl'] > fixed['net_pnl'] and partial['max_dd'] <= fixed['max_dd']:
        print(f"  --> PARTIAL TP + TRAILING is BETTER")
        print(f"      Higher profit (${pnl_diff:+,.0f}) with same or lower drawdown")
    elif partial['net_pnl'] > fixed['net_pnl'] * 1.1:
        print(f"  --> PARTIAL TP + TRAILING is BETTER")
        print(f"      Significantly higher profit (${pnl_diff:+,.0f})")
    elif fixed['net_pnl'] > partial['net_pnl'] and fixed['max_dd'] <= partial['max_dd']:
        print(f"  --> FIXED TP is BETTER")
        print(f"      Higher profit with same or lower drawdown")
    else:
        print(f"  --> MIXED RESULTS - consider risk tolerance")
        if partial['max_dd'] < fixed['max_dd']:
            print(f"      Partial TP has lower drawdown (${dd_diff:,.0f})")
        if partial['profit_factor'] > fixed['profit_factor']:
            print(f"      Partial TP has better profit factor ({pf_diff:+.2f})")

    print(f"\n{'='*80}")


async def main():
    """Run comparison backtest between Fixed TP and Partial TP + Trailing"""
    timeframe = "H1"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    print(f"SURGE-WSI H1 v6.4 GBPUSD - PARTIAL TP & TRAILING STOP COMPARISON")
    print(f"{'='*70}")
    print(f"Comparing:")
    print(f"  1. Fixed TP (Original - 2:1 RR)")
    print(f"  2. Partial TP + Trailing Stop ({PARTIAL_TP_PERCENT*100:.0f}% at {PARTIAL_TP_RR}:1, trail at {TRAIL_ATR_MULT}x ATR)")
    print(f"{'='*70}")
    print(f"Vector Features: {'ENABLED' if USE_VECTOR_FEATURES and VECTOR_AVAILABLE else 'DISABLED'}")

    print(f"\nFetching {SYMBOL} {timeframe} data...")

    df = await fetch_data(SYMBOL, timeframe, start, end)

    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")

    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_map['open'] = col
        elif 'high' in col_lower:
            col_map['high'] = col
        elif 'low' in col_lower:
            col_map['low'] = col
        elif 'close' in col_lower:
            col_map['close'] = col

    print(f"\nRunning COMPARISON backtest...")
    print(f"Entry Signals: Order Block={'ON' if USE_ORDER_BLOCK else 'OFF'}, EMA Pullback={'ON' if USE_EMA_PULLBACK else 'OFF'}")
    if USE_H4_BIAS:
        print(f"H4 Bias Filter: ON")

    # Run comparison
    results = run_comparison_backtest(df, col_map)

    # Print individual results
    print(f"\n{'#'*70}")
    print(f"FIXED TP RESULTS")
    print(f"{'#'*70}")
    if results['fixed']['stats']:
        h4_stats = results['fixed'].get('h4_stats', {})
        print_results(results['fixed']['stats'], results['fixed']['trades'],
                     results['fixed']['condition_stats'], results['fixed']['skip_stats'],
                     results['fixed']['entry_stats'], h4_stats)

    print(f"\n{'#'*70}")
    print(f"PARTIAL TP + TRAILING STOP RESULTS")
    print(f"{'#'*70}")
    if results['partial']['stats']:
        h4_stats = results['partial'].get('h4_stats', {})
        print_results(results['partial']['stats'], results['partial']['trades'],
                     results['partial']['condition_stats'], results['partial']['skip_stats'],
                     results['partial']['entry_stats'], h4_stats)

    # Print comparison
    print_comparison_results(results)

    # Send Telegram report for the better performing strategy
    fixed_pnl = results['fixed']['stats']['net_pnl'] if results['fixed']['stats'] else 0
    partial_pnl = results['partial']['stats']['net_pnl'] if results['partial']['stats'] else 0

    if partial_pnl >= fixed_pnl:
        await send_telegram_report(results['partial']['stats'], results['partial']['trades'],
                                  results['partial']['condition_stats'], start, end)
        best_label = "partial"
    else:
        await send_telegram_report(results['fixed']['stats'], results['fixed']['trades'],
                                  results['fixed']['condition_stats'], start, end)
        best_label = "fixed"

    # Save trades for both strategies
    for label, result in results.items():
        if result['trades']:
            trades_df = pd.DataFrame([{
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'lot_size': t.lot_size,
                'original_lot_size': t.original_lot_size,
                'atr_pips': t.atr_pips,
                'pnl': t.pnl,
                'partial_pnl': t.partial_pnl,
                'partial_closed': t.partial_closed,
                'trailing_stop': t.trailing_stop,
                'exit_reason': t.exit_reason,
                'quality_score': t.quality_score,
                'entry_type': t.entry_type,
                'poi_type': t.poi_type,
                'session': t.session,
                'dynamic_quality': t.dynamic_quality,
                'market_condition': t.market_condition,
                'monthly_adj': t.monthly_adj,
                'h4_bias': getattr(t, 'h4_bias', ''),
                'h4_aligned': getattr(t, 'h4_aligned', True)
            } for t in result['trades']])

            output_path = STRATEGY_DIR / "reports" / f"quadlayer_trades_{label}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            trades_df.to_csv(output_path, index=False)
            print(f"\nTrades saved to: {output_path}")

    print(f"\nBest performing: {best_label.upper()}")


if __name__ == "__main__":
    asyncio.run(main())
