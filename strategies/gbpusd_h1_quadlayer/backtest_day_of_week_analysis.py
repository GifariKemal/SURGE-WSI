"""
SURGE-WSI H1 GBPUSD - Day-of-Week Pattern Analysis
===================================================

Analyzes trade performance by day of week to identify:
1. Best and worst performing days
2. Whether skipping certain days improves results
3. Day-specific lot size multipliers

Period: 2024-02-01 to 2026-01-30
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
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from config import config
from src.data.db_handler import DBHandler

# Import from main backtest
from gbpusd_h1_quadlayer.backtest import (
    SYMBOL, PIP_SIZE, PIP_VALUE,
    INITIAL_BALANCE, RISK_PERCENT, SL_ATR_MULT, TP_RATIO,
    MAX_LOT, MAX_LOSS_PER_TRADE_PCT,
    MIN_ATR, MAX_ATR,
    BASE_QUALITY, MIN_QUALITY_GOOD, MAX_QUALITY_BAD,
    HOUR_MULTIPLIERS, MONTHLY_RISK, ENTRY_MULTIPLIERS,
    ATR_STABILITY_THRESHOLD, EFFICIENCY_THRESHOLD, TREND_STRENGTH_THRESHOLD,
    USE_PATTERN_FILTER, USE_ORDER_BLOCK, USE_EMA_PULLBACK, USE_SESSION_POI_FILTER,
    SKIP_HOURS, SKIP_ORDER_BLOCK_HOURS, SKIP_EMA_PULLBACK_HOURS,
    Trade, Regime, MarketCondition,
    fetch_data, calculate_atr, calculate_ema, calculate_rsi, calculate_adx,
    assess_market_condition, detect_regime, detect_order_blocks, detect_ema_pullback,
    check_entry_trigger, should_skip_by_session,
)
from gbpusd_h1_quadlayer.trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    calculate_lot_size,
    get_monthly_quality_adjustment,
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DAY OF WEEK ANALYSIS CONFIGURATION
# ============================================================

DAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
}

# Current production multipliers
BASELINE_DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}


def run_backtest_with_day_filter(
    df: pd.DataFrame,
    col_map: dict,
    day_multipliers: Dict[int, float] = None,
    skip_days: List[int] = None
) -> Tuple[List[Trade], float, dict]:
    """
    Run backtest with custom day filtering.

    Args:
        df: Price data DataFrame
        col_map: Column mapping
        day_multipliers: Custom day multipliers (0=Mon to 4=Fri)
        skip_days: List of days to skip completely (0=Mon to 4=Fri)

    Returns:
        Tuple of (trades, max_dd, stats)
    """
    if day_multipliers is None:
        day_multipliers = BASELINE_DAY_MULTIPLIERS.copy()
    if skip_days is None:
        skip_days = []

    # Merge skip_days into multipliers
    for day in skip_days:
        day_multipliers[day] = 0.0

    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Filters
    risk_manager = IntraMonthRiskManager()
    pattern_filter = PatternBasedFilter() if USE_PATTERN_FILTER else None
    current_month_key = None

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]

        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Handle open position
        if position:
            high = current_bar[col_map['high']]
            low = current_bar[col_map['low']]
            exit_price = None
            exit_reason = ""

            if position.direction == 'BUY':
                if low <= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = "SL"
                elif high >= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "TP"
            else:
                if high >= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = "SL"
                elif low <= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "TP"

            if exit_price:
                if position.direction == 'BUY':
                    pips = (exit_price - position.entry_price) / PIP_SIZE
                else:
                    pips = (position.entry_price - exit_price) / PIP_SIZE

                pnl = pips * position.lot_size * PIP_VALUE
                max_loss = balance * (MAX_LOSS_PER_TRADE_PCT / 100)

                if pnl < 0 and abs(pnl) > max_loss:
                    pnl = -max_loss
                    exit_reason = "SL_CAPPED"

                position.exit_time = current_time
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pips = pips
                position.exit_reason = exit_reason
                balance += pnl
                trades.append(position)

                risk_manager.record_trade(pnl, current_time)
                if USE_PATTERN_FILTER and pattern_filter:
                    pattern_filter.record_trade(position.direction, pnl, current_time)

                position = None
            continue

        # Skip weekends
        if current_time.weekday() >= 5:
            continue

        # DAY FILTER - Apply custom day multiplier
        day_of_week = current_time.weekday()
        day_mult = day_multipliers.get(day_of_week, 1.0)
        if day_mult == 0.0:
            continue

        # ATR filter
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        # Hour filter
        hour = current_time.hour
        if not (8 <= hour <= 11 or 13 <= hour <= 17):
            continue
        session = "london" if 8 <= hour <= 11 else "newyork"

        # Intra-month risk check
        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            continue

        # Reset pattern filter on month change
        month_key = (current_time.year, current_time.month)
        if month_key != current_month_key:
            current_month_key = month_key
            if USE_PATTERN_FILTER and pattern_filter:
                pattern_filter.reset_for_month(current_time.month)

        # Regime detection
        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # Market condition assessment
        condition = assess_market_condition(df, col_map, i, atr_series, current_time)
        dynamic_quality = condition.final_quality + intra_month_adj

        # Pattern filter check
        pattern_adj = 0
        pattern_size_mult = 1.0
        if USE_PATTERN_FILTER and pattern_filter:
            allowed, pattern_size_mult, reason = pattern_filter.check_trade_allowed()
            if not allowed:
                continue
            pattern_adj = pattern_filter.get_quality_adjustment()
            dynamic_quality += pattern_adj

        # Detect entry signals
        pois = []
        if USE_ORDER_BLOCK:
            pois.extend(detect_order_blocks(current_slice, col_map, dynamic_quality))
        if USE_EMA_PULLBACK:
            pois.extend(detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality))

        if not pois:
            continue

        # Sort by quality
        pois.sort(key=lambda x: x['quality'], reverse=True)
        best_poi = pois[0]

        # Session filter
        if USE_SESSION_POI_FILTER:
            should_skip, skip_reason = should_skip_by_session(hour, best_poi.get('type', ''))
            if should_skip:
                continue

        # Check entry trigger
        if i < 2:
            continue
        prev_bar = df.iloc[i-1]
        triggered, entry_type = check_entry_trigger(current_bar, prev_bar, best_poi['direction'], col_map)

        if triggered and best_poi['quality'] >= dynamic_quality:
            # Calculate position sizing with day multiplier
            entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 0.8)
            hour_mult = HOUR_MULTIPLIERS.get(hour, 0.5)
            month_mult = MONTHLY_RISK.get(current_time.month, 0.8)
            quality_mult = best_poi['quality'] / 100.0

            combined_mult = day_mult * hour_mult * entry_mult * quality_mult * month_mult * pattern_size_mult
            if combined_mult < 0.25:
                continue

            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO

            if best_poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            lot_size = calculate_lot_size(
                balance, RISK_PERCENT, current_atr,
                SL_ATR_MULT, PIP_SIZE, PIP_VALUE, combined_mult
            )
            lot_size = min(lot_size, MAX_LOT)

            if lot_size < 0.01:
                continue

            position = Trade(
                entry_time=current_time,
                direction=best_poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                risk_amount=balance * RISK_PERCENT / 100 * combined_mult,
                atr_pips=current_atr,
                quality_score=best_poi['quality'],
                entry_type=entry_type,
                poi_type=best_poi.get('type', 'ORDER_BLOCK'),
                session=session,
                dynamic_quality=dynamic_quality,
                market_condition=condition.label,
                monthly_adj=condition.monthly_adjustment,
                original_lot_size=lot_size,
            )

    return trades, max_dd, {}


def analyze_trades_by_day(trades: List[Trade]) -> Dict:
    """Analyze trades grouped by day of week."""
    day_stats = {}

    for day in range(5):
        day_trades = [t for t in trades if t.entry_time.weekday() == day]

        if not day_trades:
            day_stats[day] = {
                'name': DAY_NAMES[day],
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
            }
            continue

        wins = [t for t in day_trades if t.pnl > 0]
        losses = [t for t in day_trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in day_trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        day_stats[day] = {
            'name': DAY_NAMES[day],
            'trades': len(day_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(day_trades) * 100 if day_trades else 0,
            'pnl': total_pnl,
            'avg_pnl': total_pnl / len(day_trades) if day_trades else 0,
            'avg_win': gross_profit / len(wins) if wins else 0,
            'avg_loss': gross_loss / len(losses) if losses else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        }

    return day_stats


def calculate_overall_stats(trades: List[Trade]) -> Dict:
    """Calculate overall backtest statistics."""
    if not trades:
        return {'total': 0, 'win_rate': 0, 'pnl': 0, 'profit_factor': 0}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

    # Monthly breakdown
    monthly_pnl = {}
    for t in trades:
        month_key = t.entry_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month_key not in monthly_pnl:
            monthly_pnl[month_key] = 0
        monthly_pnl[month_key] += t.pnl

    losing_months = sum(1 for pnl in monthly_pnl.values() if pnl < 0)

    return {
        'total': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'pnl': sum(t.pnl for t in trades),
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'avg_win': gross_profit / len(wins) if wins else 0,
        'avg_loss': gross_loss / len(losses) if losses else 0,
        'losing_months': losing_months,
        'total_months': len(monthly_pnl),
        'monthly_pnl': monthly_pnl,
    }


def print_day_analysis(day_stats: Dict, overall: Dict, title: str = "BASELINE"):
    """Print day-of-week analysis results."""
    print(f"\n{'='*80}")
    print(f"DAY-OF-WEEK ANALYSIS: {title}")
    print(f"{'='*80}")

    print(f"\n{'Day':<12} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'WR%':>8} {'P/L':>12} {'Avg P/L':>10} {'PF':>8}")
    print(f"{'-'*80}")

    for day in range(5):
        stats = day_stats[day]
        pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
        print(f"{stats['name']:<12} {stats['trades']:>8} {stats['wins']:>6} {stats['losses']:>8} "
              f"{stats['win_rate']:>7.1f}% ${stats['pnl']:>+10,.0f} ${stats['avg_pnl']:>+8,.0f} {pf_str:>8}")

    print(f"{'-'*80}")
    pf_str = f"{overall['profit_factor']:.2f}" if overall['profit_factor'] != float('inf') else "INF"
    print(f"{'TOTAL':<12} {overall['total']:>8} {overall['wins']:>6} {overall['losses']:>8} "
          f"{overall['win_rate']:>7.1f}% ${overall['pnl']:>+10,.0f} ${overall['pnl']/overall['total'] if overall['total'] else 0:>+8,.0f} {pf_str:>8}")

    print(f"\nLosing Months: {overall['losing_months']}/{overall['total_months']}")


async def main():
    """Main analysis function."""
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 30, tzinfo=timezone.utc)
    timeframe = "H1"

    print(f"\n{'#'*80}")
    print(f"# SURGE-WSI H1 GBPUSD - DAY-OF-WEEK PATTERN ANALYSIS")
    print(f"# Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"{'#'*80}")

    print(f"\nFetching {SYMBOL} {timeframe} data...")
    df = await fetch_data(SYMBOL, timeframe, start, end)

    if df.empty:
        print("Error: No data fetched")
        return

    print(f"Fetched {len(df)} bars")

    # Build column map
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

    # ================================================================
    # STEP 1: BASELINE ANALYSIS (Current Production Settings)
    # ================================================================
    print(f"\n\n{'='*80}")
    print("STEP 1: BASELINE ANALYSIS (Current Production Settings)")
    print(f"{'='*80}")
    print(f"Current DAY_MULTIPLIERS: {BASELINE_DAY_MULTIPLIERS}")

    baseline_trades, baseline_max_dd, _ = run_backtest_with_day_filter(
        df, col_map, BASELINE_DAY_MULTIPLIERS.copy(), skip_days=[]
    )

    baseline_day_stats = analyze_trades_by_day(baseline_trades)
    baseline_overall = calculate_overall_stats(baseline_trades)
    print_day_analysis(baseline_day_stats, baseline_overall, "BASELINE (Current Production)")

    # ================================================================
    # STEP 2: NO DAY FILTER (All Days Equal)
    # ================================================================
    print(f"\n\n{'='*80}")
    print("STEP 2: NO DAY FILTER (All Days Equal Weight = 1.0)")
    print(f"{'='*80}")

    equal_day_mults = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0}
    equal_trades, _, _ = run_backtest_with_day_filter(df, col_map, equal_day_mults, skip_days=[])

    equal_day_stats = analyze_trades_by_day(equal_trades)
    equal_overall = calculate_overall_stats(equal_trades)
    print_day_analysis(equal_day_stats, equal_overall, "ALL DAYS EQUAL (1.0)")

    # ================================================================
    # STEP 3: IDENTIFY WORST DAY(S)
    # ================================================================
    print(f"\n\n{'='*80}")
    print("STEP 3: IDENTIFY BEST/WORST DAYS")
    print(f"{'='*80}")

    # Sort days by performance (P/L)
    sorted_by_pnl = sorted(equal_day_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    sorted_by_wr = sorted(equal_day_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    sorted_by_pf = sorted(equal_day_stats.items(), key=lambda x: x[1]['profit_factor'] if x[1]['profit_factor'] != float('inf') else 0, reverse=True)

    print(f"\nRanking by P/L:")
    for rank, (day, stats) in enumerate(sorted_by_pnl, 1):
        print(f"  {rank}. {stats['name']:<12} ${stats['pnl']:>+10,.0f} ({stats['trades']} trades, {stats['win_rate']:.1f}% WR)")

    print(f"\nRanking by Win Rate:")
    for rank, (day, stats) in enumerate(sorted_by_wr, 1):
        print(f"  {rank}. {stats['name']:<12} {stats['win_rate']:>6.1f}% ({stats['trades']} trades, ${stats['pnl']:>+,.0f})")

    print(f"\nRanking by Profit Factor:")
    for rank, (day, stats) in enumerate(sorted_by_pf, 1):
        pf = stats['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "INF"
        print(f"  {rank}. {stats['name']:<12} PF={pf_str:>6} ({stats['trades']} trades, ${stats['pnl']:>+,.0f})")

    # Identify worst day (by P/L)
    worst_day = sorted_by_pnl[-1][0]
    worst_name = sorted_by_pnl[-1][1]['name']
    worst_pnl = sorted_by_pnl[-1][1]['pnl']

    # Identify second worst day
    second_worst_day = sorted_by_pnl[-2][0] if len(sorted_by_pnl) > 1 else None
    second_worst_name = sorted_by_pnl[-2][1]['name'] if second_worst_day is not None else ""

    print(f"\n>>> WORST DAY: {worst_name} (${worst_pnl:+,.0f})")
    if second_worst_day is not None:
        print(f">>> SECOND WORST: {second_worst_name} (${sorted_by_pnl[-2][1]['pnl']:+,.0f})")

    # ================================================================
    # STEP 4: TEST SKIPPING WORST DAY
    # ================================================================
    print(f"\n\n{'='*80}")
    print(f"STEP 4: TEST SKIPPING WORST DAY ({worst_name})")
    print(f"{'='*80}")

    skip_worst_trades, _, _ = run_backtest_with_day_filter(
        df, col_map, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
        skip_days=[worst_day]
    )
    skip_worst_stats = analyze_trades_by_day(skip_worst_trades)
    skip_worst_overall = calculate_overall_stats(skip_worst_trades)
    print_day_analysis(skip_worst_stats, skip_worst_overall, f"SKIP {worst_name.upper()}")

    # Compare
    print(f"\n>>> COMPARISON vs EQUAL WEIGHTS:")
    print(f"    Trades: {skip_worst_overall['total']} vs {equal_overall['total']} ({skip_worst_overall['total'] - equal_overall['total']:+d})")
    print(f"    Win Rate: {skip_worst_overall['win_rate']:.1f}% vs {equal_overall['win_rate']:.1f}% ({skip_worst_overall['win_rate'] - equal_overall['win_rate']:+.1f}%)")
    print(f"    P/L: ${skip_worst_overall['pnl']:+,.0f} vs ${equal_overall['pnl']:+,.0f} (${skip_worst_overall['pnl'] - equal_overall['pnl']:+,.0f})")
    print(f"    Losing Months: {skip_worst_overall['losing_months']} vs {equal_overall['losing_months']}")

    # ================================================================
    # STEP 5: TEST SKIPPING TWO WORST DAYS
    # ================================================================
    if second_worst_day is not None:
        print(f"\n\n{'='*80}")
        print(f"STEP 5: TEST SKIPPING TWO WORST DAYS ({worst_name} + {second_worst_name})")
        print(f"{'='*80}")

        skip_two_trades, _, _ = run_backtest_with_day_filter(
            df, col_map, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
            skip_days=[worst_day, second_worst_day]
        )
        skip_two_stats = analyze_trades_by_day(skip_two_trades)
        skip_two_overall = calculate_overall_stats(skip_two_trades)
        print_day_analysis(skip_two_stats, skip_two_overall, f"SKIP {worst_name.upper()} + {second_worst_name.upper()}")

        print(f"\n>>> COMPARISON vs EQUAL WEIGHTS:")
        print(f"    Trades: {skip_two_overall['total']} vs {equal_overall['total']} ({skip_two_overall['total'] - equal_overall['total']:+d})")
        print(f"    Win Rate: {skip_two_overall['win_rate']:.1f}% vs {equal_overall['win_rate']:.1f}% ({skip_two_overall['win_rate'] - equal_overall['win_rate']:+.1f}%)")
        print(f"    P/L: ${skip_two_overall['pnl']:+,.0f} vs ${equal_overall['pnl']:+,.0f} (${skip_two_overall['pnl'] - equal_overall['pnl']:+,.0f})")
        print(f"    Losing Months: {skip_two_overall['losing_months']} vs {equal_overall['losing_months']}")

    # ================================================================
    # STEP 6: TEST ONLY BEST DAYS
    # ================================================================
    print(f"\n\n{'='*80}")
    print("STEP 6: TEST ONLY BEST 3 DAYS")
    print(f"{'='*80}")

    best_3_days = [d[0] for d in sorted_by_pnl[:3]]
    skip_worst_2 = [d for d in range(5) if d not in best_3_days]

    print(f"Best 3 days by P/L: {[DAY_NAMES[d] for d in best_3_days]}")
    print(f"Skipping: {[DAY_NAMES[d] for d in skip_worst_2]}")

    best3_trades, _, _ = run_backtest_with_day_filter(
        df, col_map, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
        skip_days=skip_worst_2
    )
    best3_stats = analyze_trades_by_day(best3_trades)
    best3_overall = calculate_overall_stats(best3_trades)
    print_day_analysis(best3_stats, best3_overall, f"ONLY BEST 3 DAYS")

    # ================================================================
    # STEP 7: TEST REDUCED SIZE ON WORST DAYS
    # ================================================================
    print(f"\n\n{'='*80}")
    print("STEP 7: TEST REDUCED SIZE ON WORST DAYS (0.3x multiplier)")
    print(f"{'='*80}")

    reduced_mults = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0}
    reduced_mults[worst_day] = 0.3
    if second_worst_day is not None:
        reduced_mults[second_worst_day] = 0.5

    print(f"Multipliers: {reduced_mults}")

    reduced_trades, _, _ = run_backtest_with_day_filter(df, col_map, reduced_mults, skip_days=[])
    reduced_stats = analyze_trades_by_day(reduced_trades)
    reduced_overall = calculate_overall_stats(reduced_trades)
    print_day_analysis(reduced_stats, reduced_overall, "REDUCED SIZE WORST DAYS")

    # ================================================================
    # STEP 8: TEST SPECIFIC PATTERNS
    # ================================================================
    print(f"\n\n{'='*80}")
    print("STEP 8: TEST COMMON DAY PATTERNS")
    print(f"{'='*80}")

    patterns = {
        "Monday Light (0.5)": {0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
        "Friday Light (0.5)": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0, 6: 0.0},
        "Mon+Fri Light (0.5)": {0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0, 6: 0.0},
        "Mid-Week Focus": {0: 0.3, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.3, 5: 0.0, 6: 0.0},
        "Tue-Wed-Thu Only": {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0},
    }

    results_summary = []
    results_summary.append(("BASELINE", baseline_overall['total'], baseline_overall['win_rate'],
                           baseline_overall['pnl'], baseline_overall['profit_factor'], baseline_overall['losing_months']))
    results_summary.append(("ALL EQUAL", equal_overall['total'], equal_overall['win_rate'],
                           equal_overall['pnl'], equal_overall['profit_factor'], equal_overall['losing_months']))

    for pattern_name, mults in patterns.items():
        pattern_trades, _, _ = run_backtest_with_day_filter(df, col_map, mults, skip_days=[])
        pattern_overall = calculate_overall_stats(pattern_trades)

        print(f"\n{pattern_name}:")
        print(f"  Multipliers: {mults}")
        print(f"  Trades: {pattern_overall['total']}, WR: {pattern_overall['win_rate']:.1f}%, "
              f"P/L: ${pattern_overall['pnl']:+,.0f}, PF: {pattern_overall['profit_factor']:.2f}, "
              f"Losing Months: {pattern_overall['losing_months']}")

        results_summary.append((pattern_name, pattern_overall['total'], pattern_overall['win_rate'],
                               pattern_overall['pnl'], pattern_overall['profit_factor'], pattern_overall['losing_months']))

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - ALL SCENARIOS COMPARED")
    print(f"{'='*80}")

    print(f"\n{'Scenario':<25} {'Trades':>8} {'WR%':>8} {'P/L':>12} {'PF':>8} {'Loss Mo':>8}")
    print(f"{'-'*80}")

    for name, trades, wr, pnl, pf, loss_mo in results_summary:
        pf_str = f"{pf:.2f}" if pf != float('inf') else "INF"
        print(f"{name:<25} {trades:>8} {wr:>7.1f}% ${pnl:>+10,.0f} {pf_str:>8} {loss_mo:>8}")

    # Find best scenario
    # Criteria: Highest P/L with 0 losing months, or highest P/L if all have losing months
    zero_loss_scenarios = [(n, t, w, p, pf, l) for n, t, w, p, pf, l in results_summary if l == 0]

    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    if zero_loss_scenarios:
        best = max(zero_loss_scenarios, key=lambda x: x[3])  # Highest P/L with 0 losing months
        print(f"\nBEST SCENARIO (0 losing months): {best[0]}")
        print(f"  Trades: {best[1]}, WR: {best[2]:.1f}%, P/L: ${best[3]:+,.0f}, PF: {best[4]:.2f}")
    else:
        best = max(results_summary, key=lambda x: x[3])  # Highest P/L overall
        print(f"\nBEST SCENARIO (by P/L): {best[0]}")
        print(f"  Trades: {best[1]}, WR: {best[2]:.1f}%, P/L: ${best[3]:+,.0f}, PF: {best[4]:.2f}, Losing Months: {best[5]}")

    # Show baseline comparison
    baseline_result = results_summary[0]
    if best[0] != "BASELINE":
        print(f"\nCOMPARISON vs BASELINE:")
        print(f"  Trades: {best[1] - baseline_result[1]:+d}")
        print(f"  Win Rate: {best[2] - baseline_result[2]:+.1f}%")
        print(f"  P/L: ${best[3] - baseline_result[3]:+,.0f}")
        print(f"  Losing Months: {best[5] - baseline_result[5]:+d}")
    else:
        print(f"\n>>> RECOMMENDATION: KEEP current BASELINE settings (no change needed)")


if __name__ == "__main__":
    asyncio.run(main())
