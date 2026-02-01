"""
SURGE-WSI Position Sizing Methods Comparison Test
==================================================

Compare different position sizing methods:
1. Fixed Fractional (Current/Baseline) - Risk 1% per trade
2. Kelly Criterion - Optimal f based on WR and avg win/loss
3. Volatility-Based ATR Scaling - Inverse ATR sizing
4. Equity Curve Trading - Reduce size when below equity SMA

All methods use the SAME trade signals (v6.8 baseline with 115 trades).
Only position size calculation differs.

Author: SURIOTA Team
Date: 2026-01-31
"""

import sys
import io
from pathlib import Path

# Strategy directory (where this file is located)
STRATEGY_DIR = Path(__file__).parent
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGY_DIR.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from config import config
from src.data.db_handler import DBHandler

# Import from main backtest
from gbpusd_h1_quadlayer.backtest import (
    Trade, Regime, MarketCondition,
    INITIAL_BALANCE, RISK_PERCENT, SL_ATR_MULT, TP_RATIO, MAX_LOT,
    MAX_LOSS_PER_TRADE_PCT, MIN_ATR, MAX_ATR,
    BASE_QUALITY, MIN_QUALITY_GOOD, MAX_QUALITY_BAD,
    DAY_MULTIPLIERS, HOUR_MULTIPLIERS, MONTHLY_RISK, ENTRY_MULTIPLIERS,
    USE_ORDER_BLOCK, USE_EMA_PULLBACK, USE_SESSION_POI_FILTER,
    SKIP_HOURS, SKIP_ORDER_BLOCK_HOURS, SKIP_EMA_PULLBACK_HOURS,
    fetch_data, calculate_atr, calculate_ema,
    assess_market_condition, detect_regime, detect_order_blocks,
    detect_ema_pullback, check_entry_trigger, calculate_risk_multiplier,
    should_skip_by_session, PIP_SIZE, PIP_VALUE,
)
from gbpusd_h1_quadlayer.trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    get_monthly_quality_adjustment,
    MONTHLY_TRADEABLE_PCT,
    SEASONAL_TEMPLATE,
)
from gbpusd_h1_quadlayer.strategy_config import SYMBOL

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# POSITION SIZING METHODS
# ============================================================

@dataclass
class SizingConfig:
    """Configuration for a position sizing method"""
    name: str
    description: str
    # Kelly parameters
    kelly_fraction: float = 0.25  # Use 25% Kelly by default
    # ATR scaling parameters
    base_risk: float = 0.01      # 1% base risk
    avg_atr: float = 15.0        # Average ATR in pips (calculated from data)
    # Equity curve parameters
    equity_sma_period: int = 20  # SMA period for equity curve
    equity_reduce_mult: float = 0.5  # Reduce to 50% when below SMA


def calculate_fixed_fractional_size(
    balance: float,
    risk_percent: float,
    sl_pips: float,
    risk_mult: float,
    combined_size_mult: float,
    **kwargs
) -> float:
    """
    Fixed Fractional - Current baseline method
    Risk a fixed percentage per trade
    """
    risk_amount = balance * (risk_percent / 100.0) * risk_mult * combined_size_mult
    lot_size = risk_amount / (sl_pips * PIP_VALUE)
    return max(0.01, min(MAX_LOT, round(lot_size, 2)))


def calculate_kelly_size(
    balance: float,
    risk_percent: float,
    sl_pips: float,
    risk_mult: float,
    combined_size_mult: float,
    win_rate: float = 0.496,      # Current strategy WR
    avg_win: float = 390.0,        # Avg win in $
    avg_loss: float = 75.0,        # Avg loss in $
    kelly_fraction: float = 0.25,  # Use fractional Kelly
    **kwargs
) -> float:
    """
    Kelly Criterion - Optimal betting based on edge
    Kelly % = (WR * avg_win - (1-WR) * avg_loss) / avg_win
    Use fractional Kelly (25-50%) for safety
    """
    if avg_win <= 0 or win_rate <= 0:
        # Fallback to fixed fractional if no history
        return calculate_fixed_fractional_size(
            balance, risk_percent, sl_pips, risk_mult, combined_size_mult
        )

    # Calculate full Kelly percentage
    kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

    # Apply fractional Kelly and clamp to reasonable range
    kelly_pct = max(0.001, min(0.05, kelly_pct * kelly_fraction))

    # Calculate position size
    risk_amount = balance * kelly_pct * risk_mult * combined_size_mult
    lot_size = risk_amount / (sl_pips * PIP_VALUE)
    return max(0.01, min(MAX_LOT, round(lot_size, 2)))


def calculate_volatility_size(
    balance: float,
    risk_percent: float,
    sl_pips: float,
    risk_mult: float,
    combined_size_mult: float,
    current_atr: float = 15.0,
    avg_atr: float = 15.0,
    base_risk: float = 0.01,
    **kwargs
) -> float:
    """
    Volatility-Based ATR Scaling
    Lower position size when ATR is high, higher when ATR is low
    """
    if current_atr <= 0:
        current_atr = avg_atr

    # ATR multiplier: inverse relationship
    # If current ATR > avg, reduce size; if < avg, increase size
    atr_mult = avg_atr / current_atr if current_atr > 0 else 1.0

    # Clamp ATR multiplier to reasonable range (0.5x to 1.5x)
    atr_mult = max(0.5, min(1.5, atr_mult))

    # Apply base risk with ATR scaling
    risk_amount = balance * base_risk * atr_mult * risk_mult * combined_size_mult
    lot_size = risk_amount / (sl_pips * PIP_VALUE)
    return max(0.01, min(MAX_LOT, round(lot_size, 2)))


def calculate_equity_curve_size(
    balance: float,
    risk_percent: float,
    sl_pips: float,
    risk_mult: float,
    combined_size_mult: float,
    equity_history: List[float] = None,
    equity_sma_period: int = 20,
    equity_reduce_mult: float = 0.5,
    **kwargs
) -> float:
    """
    Equity Curve Trading
    Trade smaller after losses (when equity below SMA), larger after wins
    """
    equity_mult = 1.0

    if equity_history and len(equity_history) >= equity_sma_period:
        # Calculate equity SMA
        recent_equity = equity_history[-equity_sma_period:]
        equity_sma = sum(recent_equity) / len(recent_equity)

        # Current equity vs SMA
        if balance < equity_sma:
            equity_mult = equity_reduce_mult  # Reduce size when below SMA
        else:
            equity_mult = 1.0  # Normal size when above SMA

    risk_amount = balance * (risk_percent / 100.0) * risk_mult * combined_size_mult * equity_mult
    lot_size = risk_amount / (sl_pips * PIP_VALUE)
    return max(0.01, min(MAX_LOT, round(lot_size, 2)))


# ============================================================
# BACKTEST WITH CONFIGURABLE SIZING
# ============================================================

@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    method_name: str
    trades: List[Trade]
    net_pnl: float
    max_dd: float
    max_dd_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winners: int
    losers: int
    avg_win: float
    avg_loss: float
    final_balance: float
    return_pct: float
    losing_months: int
    return_dd_ratio: float
    monthly_pnl: Dict  # Monthly breakdown


def run_backtest_with_sizing(
    df: pd.DataFrame,
    col_map: dict,
    sizing_func: Callable,
    sizing_config: SizingConfig,
) -> BacktestResult:
    """
    Run backtest with a specific position sizing method.

    Uses same signal generation as v6.8 baseline.
    Only position sizing calculation differs.
    """
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Calculate average ATR for volatility-based sizing
    avg_atr = atr_series.dropna().mean()

    # Equity history for equity curve sizing
    equity_history = [INITIAL_BALANCE]

    # Kelly stats (updated after warmup period)
    warmup_trades = []
    kelly_win_rate = 0.496  # Start with baseline
    kelly_avg_win = 390.0
    kelly_avg_loss = 75.0

    # Layer 3: Intra-month risk manager
    risk_manager = IntraMonthRiskManager()

    # Layer 4: Pattern-based filter
    pattern_filter = PatternBasedFilter()
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

                # Update equity history
                equity_history.append(balance)

                # Update Kelly stats (after warmup)
                warmup_trades.append(position)
                if len(warmup_trades) >= 20:
                    wins = [t for t in warmup_trades if t.pnl > 0]
                    losses = [t for t in warmup_trades if t.pnl <= 0]
                    if wins and losses:
                        kelly_win_rate = len(wins) / len(warmup_trades)
                        kelly_avg_win = sum(t.pnl for t in wins) / len(wins)
                        kelly_avg_loss = abs(sum(t.pnl for t in losses)) / len(losses)

                risk_manager.record_trade(pnl, current_time)
                pattern_filter.record_trade(position.direction, pnl, current_time)

                position = None
            continue

        # Skip weekends
        if current_time.weekday() >= 5:
            continue
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        hour = current_time.hour
        if not (8 <= hour <= 11 or 13 <= hour <= 17):
            continue
        session = "london" if 8 <= hour <= 11 else "newyork"

        # Layer 3: Check intra-month risk
        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            continue

        # Reset filters on month change
        month_key = (current_time.year, current_time.month)
        if month_key != current_month_key:
            current_month_key = month_key
            pattern_filter.reset_for_month(current_time.month)

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        # Detect POIs
        pois = []
        if USE_ORDER_BLOCK:
            ob_pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            pois.extend(ob_pois)

        if USE_EMA_PULLBACK:
            ema_pois = detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality)
            existing_indices = {p['idx'] for p in pois}
            for ep in ema_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        if not pois:
            continue

        for poi in pois:
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            poi_type = poi.get('type', 'ORDER_BLOCK')

            # Session filter
            session_skip, session_reason = should_skip_by_session(hour, poi_type)
            if session_skip:
                continue

            if poi_type == 'EMA_PULLBACK':
                entry_type = 'MOMENTUM'
            else:
                zone_size = abs(current_bar[col_map['high']] - current_bar[col_map['low']]) * 2
                if abs(current_price - poi['price']) > zone_size:
                    continue

                prev_bar = df.iloc[i-1]
                has_trigger, entry_type = check_entry_trigger(current_bar, prev_bar, poi['direction'], col_map)
                if not has_trigger:
                    continue

            risk_mult, should_skip = calculate_risk_multiplier(current_time, entry_type, poi['quality'])
            if should_skip:
                continue

            # Layer 4: Pattern filter
            pattern_size_mult = 1.0
            pattern_extra_q = 0
            pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(poi['direction'])
            if not pattern_can_trade:
                continue

            total_extra_q = pattern_extra_q
            if total_extra_q > 0:
                effective_quality = dynamic_quality + total_extra_q
                if poi['quality'] < effective_quality:
                    continue

            combined_size_mult = pattern_size_mult

            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO

            # POSITION SIZING - Use configured method
            lot_size = sizing_func(
                balance=balance,
                risk_percent=RISK_PERCENT,
                sl_pips=sl_pips,
                risk_mult=risk_mult,
                combined_size_mult=combined_size_mult,
                # Kelly parameters
                win_rate=kelly_win_rate,
                avg_win=kelly_avg_win,
                avg_loss=kelly_avg_loss,
                kelly_fraction=sizing_config.kelly_fraction,
                # ATR parameters
                current_atr=current_atr,
                avg_atr=avg_atr,
                base_risk=sizing_config.base_risk,
                # Equity curve parameters
                equity_history=equity_history,
                equity_sma_period=sizing_config.equity_sma_period,
                equity_reduce_mult=sizing_config.equity_reduce_mult,
            )

            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            position = Trade(
                entry_time=current_time,
                direction=poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                risk_amount=lot_size * sl_pips * PIP_VALUE,
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

    # Calculate statistics
    if not trades:
        return BacktestResult(
            method_name=sizing_config.name,
            trades=[],
            net_pnl=0,
            max_dd=0,
            max_dd_pct=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            winners=0,
            losers=0,
            avg_win=0,
            avg_loss=0,
            final_balance=INITIAL_BALANCE,
            return_pct=0,
            losing_months=0,
            return_dd_ratio=0,
            monthly_pnl={},
        )

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

    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0

    # Monthly breakdown
    trade_df = pd.DataFrame([{'time': t.entry_time, 'pnl': t.pnl} for t in trades])
    trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
    monthly = trade_df.groupby('month')['pnl'].sum()
    losing_months = (monthly < 0).sum()

    return_pct = (net_pnl / INITIAL_BALANCE) * 100
    max_dd_pct = (max_dd / INITIAL_BALANCE) * 100
    return_dd_ratio = return_pct / max_dd_pct if max_dd_pct > 0 else float('inf')

    return BacktestResult(
        method_name=sizing_config.name,
        trades=trades,
        net_pnl=net_pnl,
        max_dd=max_dd,
        max_dd_pct=max_dd_pct,
        win_rate=win_rate,
        profit_factor=pf,
        total_trades=total,
        winners=win_count,
        losers=loss_count,
        avg_win=avg_win,
        avg_loss=avg_loss,
        final_balance=INITIAL_BALANCE + net_pnl,
        return_pct=return_pct,
        losing_months=losing_months,
        return_dd_ratio=return_dd_ratio,
        monthly_pnl=monthly.to_dict(),
    )


# ============================================================
# COMPARISON FUNCTIONS
# ============================================================

def print_comparison_results(results: List[BacktestResult]):
    """Print side-by-side comparison of all sizing methods"""

    print("\n" + "=" * 100)
    print("POSITION SIZING METHODS COMPARISON")
    print("=" * 100)

    # Header
    print(f"\n{'Method':<25} {'Net P/L':>12} {'Return':>10} {'Max DD':>10} {'Ret/DD':>8} {'WR':>8} {'PF':>8} {'Trades':>8} {'Lose Mo':>8}")
    print("-" * 100)

    # Sort by Return/DD ratio (risk-adjusted return)
    sorted_results = sorted(results, key=lambda r: r.return_dd_ratio, reverse=True)

    for r in sorted_results:
        print(f"{r.method_name:<25} "
              f"${r.net_pnl:>+10,.0f} "
              f"{r.return_pct:>+9.1f}% "
              f"{r.max_dd_pct:>9.1f}% "
              f"{r.return_dd_ratio:>7.2f} "
              f"{r.win_rate:>7.1f}% "
              f"{r.profit_factor:>7.2f} "
              f"{r.total_trades:>7} "
              f"{r.losing_months:>7}")

    print("-" * 100)

    # Find best method
    best = sorted_results[0]
    baseline = next((r for r in results if "Fixed" in r.method_name), results[0])

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    print(f"\nBEST METHOD by Risk-Adjusted Return: {best.method_name}")
    print(f"  - Return/Drawdown Ratio: {best.return_dd_ratio:.2f}")
    print(f"  - Net P/L: ${best.net_pnl:+,.0f}")
    print(f"  - Max Drawdown: {best.max_dd_pct:.1f}%")

    if best.method_name != baseline.method_name:
        pnl_diff = best.net_pnl - baseline.net_pnl
        dd_diff = best.max_dd_pct - baseline.max_dd_pct
        print(f"\nCompared to baseline ({baseline.method_name}):")
        print(f"  - P/L Difference: ${pnl_diff:+,.0f}")
        print(f"  - DD Difference: {dd_diff:+.1f}%")

    # Monthly comparison
    print(f"\n{'='*60}")
    print("MONTHLY P/L BREAKDOWN")
    print(f"{'='*60}")

    # Get all months
    all_months = set()
    for r in results:
        all_months.update(r.monthly_pnl.keys())
    all_months = sorted(all_months)

    # Print header
    header = f"{'Month':<12}"
    for r in results:
        header += f"{r.method_name[:12]:>12}"
    print(header)
    print("-" * (12 + len(results) * 12))

    # Print monthly data
    for month in all_months:
        row = f"{str(month):<12}"
        for r in results:
            pnl = r.monthly_pnl.get(month, 0)
            row += f"${pnl:>+10,.0f}"
        print(row)

    print("-" * (12 + len(results) * 12))

    # Totals row
    totals = f"{'TOTAL':<12}"
    for r in results:
        totals += f"${r.net_pnl:>+10,.0f}"
    print(totals)

    # Final recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    # Check if any method beats baseline significantly
    improvements = []
    for r in sorted_results:
        if r.method_name != baseline.method_name:
            pnl_imp = (r.net_pnl - baseline.net_pnl) / baseline.net_pnl * 100 if baseline.net_pnl > 0 else 0
            dd_imp = baseline.max_dd_pct - r.max_dd_pct
            if pnl_imp > 5 or dd_imp > 2:  # Significant improvement
                improvements.append((r.method_name, pnl_imp, dd_imp, r.return_dd_ratio))

    if improvements:
        print(f"\nMethods that significantly improve on baseline:")
        for name, pnl_imp, dd_imp, ratio in improvements:
            print(f"  - {name}: P/L {pnl_imp:+.1f}%, DD {dd_imp:+.1f}%, Ret/DD {ratio:.2f}")

        best_imp = improvements[0]
        print(f"\n>>> SWITCH to {best_imp[0]} <<<")
        print(f"    Reason: Better risk-adjusted returns (Ret/DD = {best_imp[3]:.2f})")
    else:
        print(f"\n>>> KEEP current method ({baseline.method_name}) <<<")
        print(f"    Reason: No significant improvement from alternative methods")
        print(f"    Current Ret/DD ratio: {baseline.return_dd_ratio:.2f}")


async def main():
    """Run position sizing comparison test"""

    print("=" * 70)
    print("SURGE-WSI POSITION SIZING COMPARISON TEST")
    print("=" * 70)
    print(f"Period: 2024-02-01 to 2026-01-30")
    print(f"Strategy: v6.8 GBPUSD H1 with Session Filter")
    print("=" * 70)

    # Fetch data
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 30, tzinfo=timezone.utc)

    print(f"\nFetching {SYMBOL} H1 data...")
    df = await fetch_data(SYMBOL, "H1", start, end)

    if df.empty:
        print("Error: No data fetched")
        return

    print(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Column mapping
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

    # Define sizing methods to test
    sizing_methods = [
        (
            calculate_fixed_fractional_size,
            SizingConfig(
                name="Fixed Fractional (Baseline)",
                description="Risk 1% per trade based on SL",
            )
        ),
        (
            calculate_kelly_size,
            SizingConfig(
                name="Kelly 25%",
                description="25% of full Kelly criterion",
                kelly_fraction=0.25,
            )
        ),
        (
            calculate_kelly_size,
            SizingConfig(
                name="Kelly 50%",
                description="50% of full Kelly criterion",
                kelly_fraction=0.50,
            )
        ),
        (
            calculate_volatility_size,
            SizingConfig(
                name="Volatility ATR Scaling",
                description="Inverse ATR-based position sizing",
                base_risk=0.01,
            )
        ),
        (
            calculate_equity_curve_size,
            SizingConfig(
                name="Equity Curve Trading",
                description="Reduce size when below equity SMA",
                equity_sma_period=20,
                equity_reduce_mult=0.5,
            )
        ),
    ]

    # Run backtests
    results = []
    for sizing_func, sizing_config in sizing_methods:
        print(f"\nTesting: {sizing_config.name}...")
        result = run_backtest_with_sizing(df, col_map, sizing_func, sizing_config)
        results.append(result)
        print(f"  -> {result.total_trades} trades, ${result.net_pnl:+,.0f}, "
              f"DD {result.max_dd_pct:.1f}%, WR {result.win_rate:.1f}%")

    # Print comparison
    print_comparison_results(results)

    # Save results to CSV
    output_dir = STRATEGY_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for r in results:
        summary_data.append({
            'method': r.method_name,
            'net_pnl': r.net_pnl,
            'return_pct': r.return_pct,
            'max_dd': r.max_dd,
            'max_dd_pct': r.max_dd_pct,
            'return_dd_ratio': r.return_dd_ratio,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'total_trades': r.total_trades,
            'winners': r.winners,
            'losers': r.losers,
            'avg_win': r.avg_win,
            'avg_loss': r.avg_loss,
            'losing_months': r.losing_months,
            'final_balance': r.final_balance,
        })

    summary_df = pd.DataFrame(summary_data)
    output_path = output_dir / "position_sizing_comparison.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
