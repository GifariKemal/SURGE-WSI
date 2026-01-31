"""
SURGE-WSI H1 GBPUSD - Day Multiplier Optimization
==================================================

Tests different DAY_MULTIPLIER configurations to find the best
combination that maintains EXACTLY 0 losing months.

CRITICAL: 0 losing months is MANDATORY - no exceptions.
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
from dataclasses import dataclass
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
    calculate_sl_tp,
    get_monthly_quality_adjustment,
)

import warnings
warnings.filterwarnings('ignore')

# Suppress logging during optimization
import logging
logging.getLogger('gbpusd_h1_quadlayer.trading_filters').setLevel(logging.WARNING)
logging.getLogger('src.data.db_handler').setLevel(logging.WARNING)

# ============================================================
# CONFIGURATIONS TO TEST
# ============================================================

# Current production configuration (BASELINE)
BASELINE_DAY_MULT = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}

# Test configurations - conservative changes to Thursday and Friday
TEST_CONFIGS = {
    "BASELINE": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0},  # Current v6.8.0

    # Option A: Only increase Thursday slightly
    "Thu_0.5": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.0, 6: 0.0},  # Thu: 0.4 -> 0.5
    "Thu_0.6": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.6, 4: 0.5, 5: 0.0, 6: 0.0},  # Thu: 0.4 -> 0.6
    "Thu_0.7": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.7, 4: 0.5, 5: 0.0, 6: 0.0},  # Thu: 0.4 -> 0.7
    "Thu_0.8": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.5, 5: 0.0, 6: 0.0},  # Thu: 0.4 -> 0.8

    # Option B: Increase Thursday, decrease Friday
    "Thu_0.6_Fri_0.4": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.6, 4: 0.4, 5: 0.0, 6: 0.0},
    "Thu_0.7_Fri_0.3": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.7, 4: 0.3, 5: 0.0, 6: 0.0},
    "Thu_0.8_Fri_0.3": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0},

    # Option C: Skip Friday entirely
    "NoFri": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.0, 5: 0.0, 6: 0.0},  # No Friday
    "Thu_0.6_NoFri": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.6, 4: 0.0, 5: 0.0, 6: 0.0},
    "Thu_0.8_NoFri": {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.0, 5: 0.0, 6: 0.0},

    # Option D: Slightly increase Tuesday
    "Tue_1.0": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0},  # Tue: 0.9 -> 1.0
    "Tue_1.0_Thu_0.6": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.6, 4: 0.5, 5: 0.0, 6: 0.0},

    # Option E: Try original "fix" with safety measures (reference only)
    "ORIGINAL_FIX": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.9, 4: 0.6, 5: 0.0, 6: 0.0},  # RISKY
}


def run_backtest_with_day_mult(
    df: pd.DataFrame,
    col_map: dict,
    day_multipliers: Dict[int, float]
) -> Tuple[List[Trade], float, dict]:
    """
    Run backtest with specific day multipliers.
    Minimal version without debug output for speed.
    """
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Filters
    risk_manager = IntraMonthRiskManager()
    pattern_filter = PatternBasedFilter() if USE_PATTERN_FILTER else None

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

        # DAY FILTER - Apply day multiplier
        day_of_week = current_time.weekday()
        day_mult = day_multipliers.get(day_of_week, 1.0)
        if day_mult == 0.0:
            continue

        # ATR filter
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        # Get market condition
        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)

        # Check Layer 3 (Intra-Month Risk)
        can_trade, intra_month_adj, block_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            continue

        # Dynamic quality requirement
        dynamic_quality = market_cond.final_quality + intra_month_adj

        # Pattern filter
        pattern_adj = 0
        size_mult = 1.0
        if USE_PATTERN_FILTER and pattern_filter:
            allowed, size_mult, reason = pattern_filter.check_trade_allowed()
            if not allowed:
                continue
            pattern_adj = pattern_filter.get_quality_adjustment()

        dynamic_quality += pattern_adj

        # Get regime
        regime, regime_strength = detect_regime(current_slice, col_map)

        # Detect entry signals (ORDER_BLOCK and EMA_PULLBACK)
        pois = []
        if USE_ORDER_BLOCK:
            ob_pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            pois.extend(ob_pois)
        if USE_EMA_PULLBACK:
            ema_pois = detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality)
            pois.extend(ema_pois)

        if not pois:
            continue

        # Sort by quality
        pois.sort(key=lambda x: x['quality'], reverse=True)

        hour = current_time.hour
        month = current_time.month

        for poi in pois:
            poi_type = poi.get('poi_type', 'ORDER_BLOCK')

            # Session-POI filter
            if USE_SESSION_POI_FILTER:
                skip, reason = should_skip_by_session(hour, poi_type)
                if skip:
                    continue

            # Quality check
            if poi['quality'] < dynamic_quality:
                continue

            # Entry trigger check
            entry_type = check_entry_trigger(df, col_map, i, poi)
            if not entry_type:
                continue

            # Calculate position
            hour_mult = HOUR_MULTIPLIERS.get(hour, 0.0)
            if hour_mult == 0.0:
                continue

            month_mult = MONTHLY_RISK.get(month, 1.0)
            entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 1.0)

            combined_mult = day_mult * hour_mult * month_mult * entry_mult * size_mult

            sl_price, tp_price = calculate_sl_tp(
                current_price, poi['direction'], current_atr,
                SL_ATR_MULT, TP_RATIO, PIP_SIZE
            )

            lot_size, risk_amount = calculate_lot_size(
                balance, current_price, sl_price,
                RISK_PERCENT * combined_mult, MAX_LOT,
                PIP_SIZE, PIP_VALUE
            )

            if lot_size < 0.01:
                continue

            # Get session
            if hour < 7:
                session = "ASIAN"
            elif hour < 12:
                session = "LONDON"
            elif hour < 17:
                session = "NY"
            else:
                session = "LATE_NY"

            position = Trade(
                entry_time=current_time,
                direction=poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                risk_amount=risk_amount,
                atr_pips=current_atr,
                quality_score=poi['quality'],
                entry_type=entry_type,
                poi_type=poi_type,
                session=session,
                dynamic_quality=dynamic_quality,
                market_condition=market_cond.label,
                monthly_adj=market_cond.monthly_adjustment + intra_month_adj,
            )
            break

    return trades, max_dd, {}


def calculate_stats_with_monthly(trades: List[Trade]) -> dict:
    """Calculate stats with monthly P/L breakdown."""
    if not trades:
        return {"error": "No trades"}

    total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total * 100) if total > 0 else 0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    net_pnl = gross_profit - gross_loss

    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Monthly breakdown
    trade_df = pd.DataFrame([{'time': t.entry_time, 'pnl': t.pnl} for t in trades])
    trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
    monthly = trade_df.groupby('month')['pnl'].sum()
    losing_months = (monthly < 0).sum()

    # List losing months
    losing_month_list = [str(m) for m, pnl in monthly.items() if pnl < 0]

    # Find minimum monthly P/L
    min_monthly = monthly.min() if len(monthly) > 0 else 0
    min_month = monthly.idxmin() if len(monthly) > 0 else "N/A"

    return {
        'total_trades': total,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'profit_factor': pf,
        'losing_months': losing_months,
        'total_months': len(monthly),
        'losing_month_list': losing_month_list,
        'min_monthly_pnl': min_monthly,
        'min_month': str(min_month),
        'monthly': monthly
    }


async def run_optimization():
    """Run optimization tests for different DAY_MULTIPLIER configurations."""

    timeframe = "H1"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    print("=" * 80)
    print("SURGE-WSI H1 GBPUSD - DAY MULTIPLIER OPTIMIZATION")
    print("=" * 80)
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Testing {len(TEST_CONFIGS)} configurations")
    print("\nFetching data...")

    # Fetch data
    df = await fetch_data(SYMBOL, timeframe, start, end)
    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")

    # Column mapping
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

    print(f"Column map: {col_map}")

    print("\n" + "=" * 80)
    print("RUNNING TESTS...")
    print("=" * 80)

    results = []

    for config_name, day_mult in TEST_CONFIGS.items():
        print(f"\nTesting: {config_name}...")
        print(f"  Day multipliers: {day_mult}")

        trades, max_dd, _ = run_backtest_with_day_mult(df, col_map, day_mult)

        if not trades:
            print(f"  Result: No trades")
            continue

        stats = calculate_stats_with_monthly(trades)

        result = {
            'config': config_name,
            'day_mult': day_mult,
            'trades': stats['total_trades'],
            'win_rate': stats['win_rate'],
            'pf': stats['profit_factor'],
            'net_pnl': stats['net_pnl'],
            'losing_months': stats['losing_months'],
            'losing_month_list': stats['losing_month_list'],
            'min_monthly_pnl': stats['min_monthly_pnl'],
            'min_month': stats['min_month'],
            'monthly': stats['monthly']
        }
        results.append(result)

        status = "OK" if stats['losing_months'] == 0 else "FAIL"
        print(f"  Result: [{status}] {stats['total_trades']} trades, {stats['win_rate']:.1f}% WR, PF {stats['profit_factor']:.2f}, ${stats['net_pnl']:+,.0f}")
        print(f"          Losing months: {stats['losing_months']}, Min month: {stats['min_month']} (${stats['min_monthly_pnl']:+,.0f})")
        if stats['losing_months'] > 0:
            print(f"          Losing: {', '.join(stats['losing_month_list'])}")

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
        print(f"{'Config':<20} {'Trades':>7} {'WR':>6} {'PF':>6} {'Net P/L':>12} {'Min Month':>10}")
        print("-" * 80)

        for r in valid_results:
            print(f"{r['config']:<20} {r['trades']:>7} {r['win_rate']:>5.1f}% {r['pf']:>6.2f} ${r['net_pnl']:>10,.0f} ${r['min_monthly_pnl']:>9,.0f}")

        # Best configuration
        best = valid_results[0]

        print("\n" + "=" * 80)
        print("RECOMMENDED CONFIGURATION (Best with 0 losing months)")
        print("=" * 80)
        print(f"\nConfig: {best['config']}")
        print(f"DAY_MULTIPLIERS = {best['day_mult']}")
        print(f"\nPerformance:")
        print(f"  Trades: {best['trades']}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Profit Factor: {best['pf']:.2f}")
        print(f"  Net P/L: ${best['net_pnl']:+,.2f}")
        print(f"  Losing Months: {best['losing_months']}")

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
        print("All configurations had at least 1 losing month.")

        # Show configs sorted by losing months then profit
        results.sort(key=lambda x: (x['losing_months'], -x['net_pnl']))

        print("\nTop configurations by losing months:")
        for r in results[:5]:
            print(f"  {r['config']}: {r['losing_months']} losing months, ${r['net_pnl']:+,.0f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(run_optimization())
