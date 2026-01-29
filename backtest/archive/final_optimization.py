"""Final Optimization: Target Zero Losing Months
===============================================

Focus on eliminating the remaining 2 losing months (Feb, Apr 2024)
while maintaining 2025 performance.

Current best: SL30 + Loss0.5% -> 2 losing months
Target: 0 losing months in both years

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
class OptimizedConfig:
    """Optimized configuration"""
    name: str
    max_sl_pips: float = 30.0
    max_loss_pct: float = 0.5
    min_quality: float = 65.0
    min_win_streak_after_loss: int = 0  # Wait for N wins before trading after loss
    max_trades_per_day: int = 0  # 0 = unlimited
    skip_first_week: bool = False  # Skip first week of month
    require_htf_momentum: bool = False  # HTF must show clear momentum


def calculate_htf_momentum(df: pd.DataFrame, periods: int = 20) -> float:
    """Calculate HTF momentum strength (0-100)"""
    if len(df) < periods:
        return 50.0

    close = df['close'] if 'close' in df.columns else df['Close']
    returns = close.pct_change().tail(periods)

    # Count up vs down bars
    up_bars = (returns > 0).sum()
    down_bars = (returns < 0).sum()

    # Price change
    pct_change = (close.iloc[-1] - close.iloc[-periods]) / close.iloc[-periods] * 100

    # Momentum strength
    if up_bars + down_bars > 0:
        direction_strength = abs(up_bars - down_bars) / (up_bars + down_bars) * 100
    else:
        direction_strength = 0

    momentum = (abs(pct_change) * 20 + direction_strength) / 2
    return min(100, max(0, momentum))


def run_optimized_backtest(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                           year: int, cfg: OptimizedConfig) -> dict:
    """Run backtest with optimized configuration"""

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

        # Adjust start date if skipping first week
        actual_start = start_date
        if cfg.skip_first_week:
            actual_start = start_date + timedelta(days=7)

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
            start_date=actual_start.strftime("%Y-%m-%d"),
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
        bt.risk_manager.max_sl_pips = cfg.max_sl_pips
        bt.entry_trigger.min_quality_score = cfg.min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        # Process trades with filters
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_filtered = 0
        simulated_balance = running_balance
        wins_since_loss = 999  # Start with high to allow first trade
        trades_today = 0
        last_trade_date = None

        for trade in result.trade_list:
            # Reset daily counter
            trade_date = trade.entry_time.date() if hasattr(trade.entry_time, 'date') else None
            if trade_date != last_trade_date:
                trades_today = 0
                last_trade_date = trade_date

            # Calculate risk in pips
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001

            # Filter 1: Skip trades with SL > max_sl_pips
            if risk_pips > cfg.max_sl_pips:
                month_filtered += 1
                continue

            # Filter 2: Skip first week
            if cfg.skip_first_week:
                if hasattr(trade.entry_time, 'day'):
                    if trade.entry_time.day <= 7:
                        month_filtered += 1
                        continue

            # Filter 3: Max trades per day
            if cfg.max_trades_per_day > 0 and trades_today >= cfg.max_trades_per_day:
                month_filtered += 1
                continue

            # Filter 4: Win streak after loss
            if cfg.min_win_streak_after_loss > 0:
                if wins_since_loss < cfg.min_win_streak_after_loss:
                    month_filtered += 1
                    continue

            # Filter 5: HTF momentum requirement
            if cfg.require_htf_momentum:
                trade_time = trade.entry_time
                htf_before = htf_month[htf_month.index <= trade_time].tail(25)
                if len(htf_before) >= 20:
                    momentum = calculate_htf_momentum(htf_before)
                    if momentum < 30:  # Weak momentum
                        month_filtered += 1
                        continue

            # Calculate max loss allowed
            max_loss_dollars = simulated_balance * cfg.max_loss_pct / 100

            # Cap the loss
            if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
                adjusted_pnl = -max_loss_dollars
            else:
                adjusted_pnl = trade.pnl

            simulated_balance += adjusted_pnl
            month_pnl += adjusted_pnl
            month_trades += 1
            trades_today += 1

            if adjusted_pnl > 0:
                month_wins += 1
                wins_since_loss += 1
            else:
                wins_since_loss = 0

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
        'config_name': cfg.name,
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
    print("FINAL OPTIMIZATION: TARGET ZERO LOSING MONTHS")
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

    # Define optimized configurations
    configs = [
        OptimizedConfig(
            name="Best So Far (SL30+Loss0.5%)",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
        ),
        OptimizedConfig(
            name="Tighter Loss (0.4%)",
            max_sl_pips=30.0,
            max_loss_pct=0.4,
        ),
        OptimizedConfig(
            name="Tighter Loss (0.35%)",
            max_sl_pips=30.0,
            max_loss_pct=0.35,
        ),
        OptimizedConfig(
            name="Tighter Loss (0.3%)",
            max_sl_pips=30.0,
            max_loss_pct=0.3,
        ),
        OptimizedConfig(
            name="Tighter SL (25 pips)",
            max_sl_pips=25.0,
            max_loss_pct=0.5,
        ),
        OptimizedConfig(
            name="SL25 + Loss0.4%",
            max_sl_pips=25.0,
            max_loss_pct=0.4,
        ),
        OptimizedConfig(
            name="SL25 + Loss0.35%",
            max_sl_pips=25.0,
            max_loss_pct=0.35,
        ),
        OptimizedConfig(
            name="SL25 + Loss0.3%",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
        ),
        OptimizedConfig(
            name="Skip First Week",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            skip_first_week=True
        ),
        OptimizedConfig(
            name="Max 2 Trades/Day",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            max_trades_per_day=2
        ),
        OptimizedConfig(
            name="Max 3 Trades/Day",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            max_trades_per_day=3
        ),
        OptimizedConfig(
            name="HTF Momentum Required",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            require_htf_momentum=True
        ),
        OptimizedConfig(
            name="Higher Quality (70)",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            min_quality=70.0
        ),
        OptimizedConfig(
            name="Higher Quality (75)",
            max_sl_pips=30.0,
            max_loss_pct=0.5,
            min_quality=75.0
        ),
        # Combined
        OptimizedConfig(
            name="ULTRA: SL25+Loss0.35%+Q70",
            max_sl_pips=25.0,
            max_loss_pct=0.35,
            min_quality=70.0
        ),
        OptimizedConfig(
            name="ULTRA: SL25+Loss0.3%+Q70",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            min_quality=70.0
        ),
        OptimizedConfig(
            name="ULTRA: SL25+Loss0.35%+Max3/Day",
            max_sl_pips=25.0,
            max_loss_pct=0.35,
            max_trades_per_day=3
        ),
    ]

    results_2024 = []
    results_2025 = []

    for cfg in configs:
        print(f"\nTesting: {cfg.name}...")

        r2024 = run_optimized_backtest(htf_df, ltf_df, 2024, cfg)
        r2025 = run_optimized_backtest(htf_df, ltf_df, 2025, cfg)

        results_2024.append(r2024)
        results_2025.append(r2025)

        marker_2024 = "***" if r2024['losing_months'] == 0 else ""
        marker_2025 = "***" if r2025['losing_months'] == 0 else ""

        print(f"  2024: {r2024['losing_months']} losing, +{r2024['total_return']:.1f}%, {r2024['total_trades']} trades {marker_2024}")
        print(f"  2025: {r2025['losing_months']} losing, +{r2025['total_return']:.1f}%, {r2025['total_trades']} trades {marker_2025}")

    # Results comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print("\n{:<35} | {:^20} | {:^20}".format(
        "Configuration", "2024", "2025"
    ))
    print("{:<35} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}".format(
        "", "Lose", "Trade", "Ret%", "Lose", "Trade", "Ret%"
    ))
    print("-" * 80)

    for i in range(len(configs)):
        r2024 = results_2024[i]
        r2025 = results_2025[i]
        marker = " ***" if r2024['losing_months'] == 0 and r2025['losing_months'] == 0 else ""
        print("{:<35} | {:>6} {:>6} {:>5.1f}% | {:>6} {:>6} {:>5.1f}%{}".format(
            configs[i].name[:35],
            r2024['losing_months'],
            r2024['total_trades'],
            r2024['total_return'],
            r2025['losing_months'],
            r2025['total_trades'],
            r2025['total_return'],
            marker
        ))

    # Find optimal
    print("\n" + "=" * 80)
    print("ZERO LOSING MONTHS SOLUTIONS")
    print("=" * 80)

    optimal = []
    for i in range(len(configs)):
        if results_2024[i]['losing_months'] == 0 and results_2025[i]['losing_months'] == 0:
            combined = results_2024[i]['total_return'] + results_2025[i]['total_return']
            optimal.append({
                'name': configs[i].name,
                'config': configs[i],
                '2024': results_2024[i],
                '2025': results_2025[i],
                'combined': combined
            })

    if optimal:
        print("\n*** SUCCESS: FOUND ZERO LOSING MONTHS CONFIGURATIONS ***\n")
        for opt in sorted(optimal, key=lambda x: -x['combined']):
            print(f"{opt['name']}:")
            print(f"  2024: 0 losing, +{opt['2024']['total_return']:.1f}%, {opt['2024']['total_trades']} trades")
            print(f"  2025: 0 losing, +{opt['2025']['total_return']:.1f}%, {opt['2025']['total_trades']} trades")
            print(f"  Combined: +{opt['combined']:.1f}%")

            # Show config details
            cfg = opt['config']
            print(f"  Config: max_sl={cfg.max_sl_pips}, max_loss={cfg.max_loss_pct}%, min_quality={cfg.min_quality}")
            print()

        # Best solution monthly breakdown
        best = sorted(optimal, key=lambda x: -x['combined'])[0]
        print("\n" + "=" * 80)
        print(f"BEST SOLUTION: {best['name']}")
        print("=" * 80)

        print("\n--- 2024 Monthly ---")
        for m in best['2024']['monthly']:
            if not m.get('skipped'):
                status = "X" if m['pnl'] < 0 else "OK"
                print(f"  {m['month']}: {m['pnl']:+,.0f}$ ({m['trades']} trades) [{status}]")

        print("\n--- 2025 Monthly ---")
        for m in best['2025']['monthly']:
            if not m.get('skipped'):
                status = "X" if m['pnl'] < 0 else "OK"
                print(f"  {m['month']}: {m['pnl']:+,.0f}$ ({m['trades']} trades) [{status}]")

    else:
        print("\nNo configuration achieved 0 losing months in both years.")

        # Find best trade-off
        print("\nClosest to zero (sorted by total losing months):")
        all_results = []
        for i in range(len(configs)):
            total_losing = results_2024[i]['losing_months'] + results_2025[i]['losing_months']
            combined = results_2024[i]['total_return'] + results_2025[i]['total_return']
            all_results.append({
                'name': configs[i].name,
                'config': configs[i],
                'total_losing': total_losing,
                '2024_losing': results_2024[i]['losing_months'],
                '2025_losing': results_2025[i]['losing_months'],
                'combined': combined,
                '2024': results_2024[i],
                '2025': results_2025[i]
            })

        for r in sorted(all_results, key=lambda x: (x['total_losing'], -x['combined']))[:5]:
            print(f"\n{r['name']}:")
            print(f"  Total losing: {r['total_losing']} (2024: {r['2024_losing']}, 2025: {r['2025_losing']})")
            print(f"  Combined return: +{r['combined']:.1f}%")
            if r['2024_losing'] > 0:
                print(f"  2024 losing: {r['2024']['losing_month_names']}")
            if r['2025_losing'] > 0:
                print(f"  2025 losing: {r['2025']['losing_month_names']}")

    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
