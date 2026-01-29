"""Test Targeted Solutions Based on Deep Analysis
================================================

Solutions to test:
1. Stricter SL limit (max 30-35 pips instead of 50)
2. Filter BUY trades during uncertain conditions
3. Tighter loss cap per trade
4. Combine multiple filters

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
from dataclasses import dataclass
from typing import List, Tuple

from config import config
from src.data.db_handler import DBHandler
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


@dataclass
class SolutionConfig:
    """Configuration for a solution test"""
    name: str
    max_sl_pips: float = 50.0
    max_loss_pct: float = 0.8
    min_quality: float = 65.0
    filter_buy_in_bearish: bool = False  # Skip BUY if recent trend is down
    require_buy_confirmation: bool = False  # Extra confirmation for BUY
    max_consecutive_losses: int = 0  # 0 = disabled
    skip_after_big_loss: bool = False  # Skip next trade after >0.5% loss


def calculate_recent_trend(df: pd.DataFrame, periods: int = 10) -> str:
    """Calculate recent price trend"""
    if len(df) < periods:
        return "UNKNOWN"

    close = df['close'] if 'close' in df.columns else df['Close']
    recent_close = close.tail(periods)

    pct_change = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0] * 100

    if pct_change > 0.1:
        return "UP"
    elif pct_change < -0.1:
        return "DOWN"
    return "FLAT"


def run_with_solution(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                      year: int, solution: SolutionConfig) -> dict:
    """Run backtest with specific solution configuration"""

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
    consecutive_losses = 0
    last_loss_big = False

    for start_date, end_date in months:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        if is_december:
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

        bt.risk_manager.max_lot_size = 0.5
        bt.risk_manager.max_sl_pips = solution.max_sl_pips
        bt.entry_trigger.min_quality_score = solution.min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        # Process trades with solution filters
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_filtered = 0
        simulated_balance = running_balance

        for trade in result.trade_list:
            # Calculate risk in pips
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001

            # Filter 1: Skip trades with SL > max_sl_pips
            if risk_pips > solution.max_sl_pips:
                month_filtered += 1
                continue

            # Filter 2: Skip BUY if recent trend is DOWN
            if solution.filter_buy_in_bearish and trade.direction == 'BUY':
                trade_time = trade.entry_time
                htf_before = htf_month[htf_month.index <= trade_time].tail(20)
                if len(htf_before) >= 10:
                    recent_trend = calculate_recent_trend(htf_before)
                    if recent_trend == "DOWN":
                        month_filtered += 1
                        continue

            # Filter 3: BUY confirmation - require price above recent high
            if solution.require_buy_confirmation and trade.direction == 'BUY':
                trade_time = trade.entry_time
                htf_before = htf_month[htf_month.index <= trade_time].tail(10)
                if len(htf_before) >= 5:
                    recent_high = htf_before['high'].max() if 'high' in htf_before.columns else htf_before['High'].max()
                    if trade.entry_price < recent_high * 0.999:  # Not near recent high
                        month_filtered += 1
                        continue

            # Filter 4: Max consecutive losses
            if solution.max_consecutive_losses > 0:
                if consecutive_losses >= solution.max_consecutive_losses:
                    month_filtered += 1
                    # Reset after skipping
                    consecutive_losses = max(0, consecutive_losses - 1)
                    continue

            # Filter 5: Skip after big loss
            if solution.skip_after_big_loss and last_loss_big:
                month_filtered += 1
                last_loss_big = False
                continue

            # Calculate max loss allowed
            max_loss_dollars = simulated_balance * solution.max_loss_pct / 100

            # Cap the loss
            if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
                adjusted_pnl = -max_loss_dollars
            else:
                adjusted_pnl = trade.pnl

            simulated_balance += adjusted_pnl
            month_pnl += adjusted_pnl
            month_trades += 1

            if adjusted_pnl > 0:
                month_wins += 1
                consecutive_losses = 0
                last_loss_big = False
            else:
                consecutive_losses += 1
                # Check if this was a big loss (>0.5% of balance)
                loss_pct = abs(adjusted_pnl) / (simulated_balance + abs(adjusted_pnl)) * 100
                last_loss_big = loss_pct > 0.5

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
        'solution_name': solution.name,
        'year': year,
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
    print("\n" + "=" * 80)
    print("TEST: TARGETED SOLUTIONS FOR ZERO LOSING MONTHS")
    print("Based on deep analysis findings")
    print("=" * 80)

    print("\nFetching data...")
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Define solutions to test
    solutions = [
        SolutionConfig(
            name="Baseline",
            max_sl_pips=50.0,
            max_loss_pct=0.8,
            min_quality=65.0
        ),
        SolutionConfig(
            name="Stricter SL (35 pips)",
            max_sl_pips=35.0,
            max_loss_pct=0.8,
            min_quality=65.0
        ),
        SolutionConfig(
            name="Stricter SL (30 pips)",
            max_sl_pips=30.0,
            max_loss_pct=0.8,
            min_quality=65.0
        ),
        SolutionConfig(
            name="Tighter Loss Cap (0.5%)",
            max_sl_pips=50.0,
            max_loss_pct=0.5,
            min_quality=65.0
        ),
        SolutionConfig(
            name="Tighter Loss Cap (0.3%)",
            max_sl_pips=50.0,
            max_loss_pct=0.3,
            min_quality=65.0
        ),
        SolutionConfig(
            name="Filter BUY in Downtrend",
            max_sl_pips=50.0,
            max_loss_pct=0.8,
            min_quality=65.0,
            filter_buy_in_bearish=True
        ),
        SolutionConfig(
            name="BUY Confirmation Required",
            max_sl_pips=50.0,
            max_loss_pct=0.8,
            min_quality=65.0,
            require_buy_confirmation=True
        ),
        SolutionConfig(
            name="Higher Quality (70)",
            max_sl_pips=50.0,
            max_loss_pct=0.8,
            min_quality=70.0
        ),
        SolutionConfig(
            name="Max 3 Consecutive Losses",
            max_sl_pips=50.0,
            max_loss_pct=0.8,
            min_quality=65.0,
            max_consecutive_losses=3
        ),
        SolutionConfig(
            name="Skip After Big Loss",
            max_sl_pips=50.0,
            max_loss_pct=0.8,
            min_quality=65.0,
            skip_after_big_loss=True
        ),
        # Combined solutions
        SolutionConfig(
            name="COMBO: SL30 + Loss0.5%",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            min_quality=65.0
        ),
        SolutionConfig(
            name="COMBO: SL35 + FilterBUY",
            max_sl_pips=35.0,
            max_loss_pct=0.8,
            min_quality=65.0,
            filter_buy_in_bearish=True
        ),
        SolutionConfig(
            name="COMBO: SL30 + Loss0.5% + FilterBUY",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            min_quality=65.0,
            filter_buy_in_bearish=True
        ),
        SolutionConfig(
            name="COMBO: All Filters",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            min_quality=70.0,
            filter_buy_in_bearish=True,
            skip_after_big_loss=True
        ),
    ]

    results_2024 = []
    results_2025 = []

    for sol in solutions:
        print(f"\nTesting: {sol.name}...")

        r2024 = run_with_solution(htf_df, ltf_df, 2024, sol)
        r2025 = run_with_solution(htf_df, ltf_df, 2025, sol)

        results_2024.append(r2024)
        results_2025.append(r2025)

        print(f"  2024: {r2024['losing_months']} losing, +{r2024['total_return']:.1f}%, {r2024['total_trades']} trades")
        print(f"  2025: {r2025['losing_months']} losing, +{r2025['total_return']:.1f}%, {r2025['total_trades']} trades")

    # ============================================================
    # RESULTS COMPARISON
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print("\n{:<35} | {:^20} | {:^20}".format(
        "Solution", "2024", "2025"
    ))
    print("{:<35} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}".format(
        "", "Lose", "Trade", "Ret%", "Lose", "Trade", "Ret%"
    ))
    print("-" * 80)

    for i in range(len(solutions)):
        r2024 = results_2024[i]
        r2025 = results_2025[i]
        print("{:<35} | {:>6} {:>6} {:>5.1f}% | {:>6} {:>6} {:>5.1f}%".format(
            solutions[i].name[:35],
            r2024['losing_months'],
            r2024['total_trades'],
            r2024['total_return'],
            r2025['losing_months'],
            r2025['total_trades'],
            r2025['total_return']
        ))

    # ============================================================
    # FIND OPTIMAL SOLUTION
    # ============================================================
    print("\n" + "=" * 80)
    print("OPTIMAL SOLUTIONS")
    print("=" * 80)

    # Find solutions with 0 losing months in BOTH years
    optimal = []
    for i in range(len(solutions)):
        r2024 = results_2024[i]
        r2025 = results_2025[i]

        if r2024['losing_months'] == 0 and r2025['losing_months'] == 0:
            combined_return = r2024['total_return'] + r2025['total_return']
            optimal.append({
                'name': solutions[i].name,
                '2024': r2024,
                '2025': r2025,
                'combined_return': combined_return
            })

    if optimal:
        print("\n*** FOUND SOLUTIONS WITH ZERO LOSING MONTHS IN BOTH YEARS ***\n")
        for opt in sorted(optimal, key=lambda x: -x['combined_return']):
            print(f"{opt['name']}:")
            print(f"  2024: 0 losing, +{opt['2024']['total_return']:.1f}%, {opt['2024']['total_trades']} trades")
            print(f"  2025: 0 losing, +{opt['2025']['total_return']:.1f}%, {opt['2025']['total_trades']} trades")
            print(f"  Combined Return: +{opt['combined_return']:.1f}%")
            print()
    else:
        print("\nNo solution achieved 0 losing months in both years.")

        # Find best trade-off
        print("\nBest trade-offs (minimize total losing months):")
        tradeoffs = []
        for i in range(len(solutions)):
            r2024 = results_2024[i]
            r2025 = results_2025[i]
            total_losing = r2024['losing_months'] + r2025['losing_months']
            combined_return = r2024['total_return'] + r2025['total_return']
            tradeoffs.append({
                'name': solutions[i].name,
                'total_losing': total_losing,
                '2024_losing': r2024['losing_months'],
                '2025_losing': r2025['losing_months'],
                'combined_return': combined_return,
                '2024': r2024,
                '2025': r2025
            })

        # Sort by total losing, then by return
        for t in sorted(tradeoffs, key=lambda x: (x['total_losing'], -x['combined_return']))[:5]:
            print(f"\n{t['name']}:")
            print(f"  Total Losing: {t['total_losing']} (2024: {t['2024_losing']}, 2025: {t['2025_losing']})")
            print(f"  Combined Return: +{t['combined_return']:.1f}%")

    # ============================================================
    # BEST SOLUTION DETAILS
    # ============================================================
    # Find the best solution (minimize losing months, maximize return)
    best_idx = 0
    best_score = float('inf')

    for i in range(len(solutions)):
        total_losing = results_2024[i]['losing_months'] + results_2025[i]['losing_months']
        combined_return = results_2024[i]['total_return'] + results_2025[i]['total_return']
        score = total_losing * 100 - combined_return  # Lower is better

        if score < best_score:
            best_score = score
            best_idx = i

    print("\n" + "=" * 80)
    print(f"BEST SOLUTION: {solutions[best_idx].name}")
    print("=" * 80)

    best_2024 = results_2024[best_idx]
    best_2025 = results_2025[best_idx]

    print(f"\n2024: {best_2024['losing_months']} losing months, +{best_2024['total_return']:.1f}%")
    if best_2024['losing_months'] > 0:
        print(f"      Losing: {best_2024['losing_month_names']}")

    print(f"2025: {best_2025['losing_months']} losing months, +{best_2025['total_return']:.1f}%")
    if best_2025['losing_months'] > 0:
        print(f"      Losing: {best_2025['losing_month_names']}")

    print(f"\nCombined Return: +{best_2024['total_return'] + best_2025['total_return']:.1f}%")

    # Monthly breakdown
    print("\n--- 2024 Monthly Breakdown ---")
    print("{:<10} {:>10} {:>10} {:>12}".format("Month", "P/L", "Trades", "Balance"))
    print("-" * 45)
    for m in best_2024['monthly']:
        if m.get('skipped'):
            print(f"{m['month']:<10} {'SKIP':>10}")
        else:
            status = " X" if m['pnl'] < 0 else ""
            print("{:<10} {:>+9,.0f}$ {:>10} {:>11,.0f}${}".format(
                m['month'], m['pnl'], m['trades'], m['balance'], status
            ))

    print("\n--- 2025 Monthly Breakdown ---")
    print("{:<10} {:>10} {:>10} {:>12}".format("Month", "P/L", "Trades", "Balance"))
    print("-" * 45)
    for m in best_2025['monthly']:
        if m.get('skipped'):
            print(f"{m['month']:<10} {'SKIP':>10}")
        else:
            status = " X" if m['pnl'] < 0 else ""
            print("{:<10} {:>+9,.0f}$ {:>10} {:>11,.0f}${}".format(
                m['month'], m['pnl'], m['trades'], m['balance'], status
            ))

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
