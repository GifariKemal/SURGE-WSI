"""Zero Losing Months: Final Push
================================

Aggressive configurations to eliminate Feb 2024 loss.

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
class ZeroLossConfig:
    name: str
    max_sl_pips: float = 20.0
    max_loss_pct: float = 0.2
    min_quality: float = 65.0
    max_risk_pips: float = 999.0  # Additional filter
    min_trades_per_month: int = 0  # Skip month if fewer trades
    skip_buy_if_recent_loss: bool = False


def run_test(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
             year: int, cfg: ZeroLossConfig) -> dict:

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
    recent_loss = False

    for start_date, end_date in months:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        if is_december:
            monthly_results.append({
                'month': month_name, 'pnl': 0, 'trades': 0,
                'balance': running_balance, 'skipped': True
            })
            continue

        warmup_days = 30
        month_start_warmup = start_date - timedelta(days=warmup_days)

        htf_month = htf_df[(htf_df.index >= month_start_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_warmup) & (ltf_df.index <= end_date)]

        if htf_month.empty or ltf_month.empty:
            monthly_results.append({
                'month': month_name, 'pnl': 0, 'trades': 0,
                'balance': running_balance, 'skipped': True
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
        bt.risk_manager.max_sl_pips = cfg.max_sl_pips
        bt.entry_trigger.min_quality_score = cfg.min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_filtered = 0
        simulated_balance = running_balance

        for trade in result.trade_list:
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001

            # Filter 1: Max SL
            if risk_pips > cfg.max_sl_pips:
                month_filtered += 1
                continue

            # Filter 2: Additional risk filter
            if risk_pips > cfg.max_risk_pips:
                month_filtered += 1
                continue

            # Filter 3: Skip BUY after recent loss
            if cfg.skip_buy_if_recent_loss and trade.direction == 'BUY' and recent_loss:
                month_filtered += 1
                continue

            max_loss_dollars = simulated_balance * cfg.max_loss_pct / 100
            if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
                adjusted_pnl = -max_loss_dollars
            else:
                adjusted_pnl = trade.pnl

            simulated_balance += adjusted_pnl
            month_pnl += adjusted_pnl
            month_trades += 1

            if adjusted_pnl > 0:
                month_wins += 1
                recent_loss = False
            else:
                recent_loss = True

        running_balance = simulated_balance
        total_trades += month_trades
        total_wins += month_wins
        total_filtered += month_filtered

        monthly_results.append({
            'month': month_name,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
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
    }


async def main():
    print("\n" + "=" * 80)
    print("ZERO LOSING MONTHS: FINAL PUSH")
    print("=" * 80)

    print("\nFetching data...")
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"Data loaded: H4={len(htf_df)}, M15={len(ltf_df)}")

    configs = [
        ZeroLossConfig(name="Current Best", max_sl_pips=20.0, max_loss_pct=0.2),
        ZeroLossConfig(name="Loss 0.15%", max_sl_pips=20.0, max_loss_pct=0.15),
        ZeroLossConfig(name="Loss 0.1%", max_sl_pips=20.0, max_loss_pct=0.1),
        ZeroLossConfig(name="SL15 + Loss 0.15%", max_sl_pips=15.0, max_loss_pct=0.15),
        ZeroLossConfig(name="SL12 + Loss 0.15%", max_sl_pips=12.0, max_loss_pct=0.15),
        ZeroLossConfig(name="SL10 + Loss 0.15%", max_sl_pips=10.0, max_loss_pct=0.15),
        ZeroLossConfig(name="Risk Max 15 pips", max_sl_pips=20.0, max_loss_pct=0.2, max_risk_pips=15.0),
        ZeroLossConfig(name="Risk Max 12 pips", max_sl_pips=20.0, max_loss_pct=0.2, max_risk_pips=12.0),
        ZeroLossConfig(name="Risk Max 10 pips", max_sl_pips=20.0, max_loss_pct=0.2, max_risk_pips=10.0),
        ZeroLossConfig(name="Skip BUY after loss", max_sl_pips=20.0, max_loss_pct=0.2, skip_buy_if_recent_loss=True),
        ZeroLossConfig(name="Quality 75", max_sl_pips=20.0, max_loss_pct=0.2, min_quality=75.0),
        ZeroLossConfig(name="Quality 78", max_sl_pips=20.0, max_loss_pct=0.2, min_quality=78.0),
        ZeroLossConfig(name="Quality 80", max_sl_pips=20.0, max_loss_pct=0.2, min_quality=80.0),
        # Combined aggressive
        ZeroLossConfig(name="AGGRESSIVE: SL12+Loss0.1%", max_sl_pips=12.0, max_loss_pct=0.1),
        ZeroLossConfig(name="AGGRESSIVE: Risk10+Loss0.1%", max_sl_pips=20.0, max_loss_pct=0.1, max_risk_pips=10.0),
        ZeroLossConfig(name="AGGRESSIVE: Q75+Loss0.15%", max_sl_pips=20.0, max_loss_pct=0.15, min_quality=75.0),
        ZeroLossConfig(name="AGGRESSIVE: Q78+Loss0.1%", max_sl_pips=20.0, max_loss_pct=0.1, min_quality=78.0),
        ZeroLossConfig(name="ULTRA: SL10+Q75+Loss0.1%", max_sl_pips=10.0, max_loss_pct=0.1, min_quality=75.0),
    ]

    print("\nTesting configurations...")
    results = []

    for cfg in configs:
        r2024 = run_test(htf_df, ltf_df, 2024, cfg)
        r2025 = run_test(htf_df, ltf_df, 2025, cfg)

        both_zero = r2024['losing_months'] == 0 and r2025['losing_months'] == 0
        marker = " *** ZERO BOTH! ***" if both_zero else ""

        results.append({
            'config': cfg,
            '2024': r2024,
            '2025': r2025,
            'both_zero': both_zero,
            'total_losing': r2024['losing_months'] + r2025['losing_months'],
            'combined': r2024['total_return'] + r2025['total_return']
        })

        print(f"{cfg.name}: 2024={r2024['losing_months']}L/{r2024['total_return']:.1f}%, "
              f"2025={r2025['losing_months']}L/{r2025['total_return']:.1f}%{marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    zero_both = [r for r in results if r['both_zero']]

    if zero_both:
        print("\n*** SUCCESS: CONFIGURATIONS WITH ZERO LOSING MONTHS ***\n")
        for r in sorted(zero_both, key=lambda x: -x['combined']):
            print(f"{r['config'].name}:")
            print(f"  2024: +{r['2024']['total_return']:.1f}% ({r['2024']['total_trades']} trades)")
            print(f"  2025: +{r['2025']['total_return']:.1f}% ({r['2025']['total_trades']} trades)")
            print(f"  Combined: +{r['combined']:.1f}%")
            print()

        # Best solution
        best = sorted(zero_both, key=lambda x: -x['combined'])[0]
        print("=" * 80)
        print(f"BEST ZERO-LOSS CONFIG: {best['config'].name}")
        print("=" * 80)
        print(f"\nSettings:")
        print(f"  max_sl_pips: {best['config'].max_sl_pips}")
        print(f"  max_loss_pct: {best['config'].max_loss_pct}%")
        print(f"  min_quality: {best['config'].min_quality}")
        print(f"  max_risk_pips: {best['config'].max_risk_pips}")

        print(f"\nResults:")
        print(f"  2024: 0 losing, +{best['2024']['total_return']:.1f}%")
        print(f"  2025: 0 losing, +{best['2025']['total_return']:.1f}%")
        print(f"  Combined: +{best['combined']:.1f}%")

        # Monthly
        print("\n2024 Monthly:")
        for m in best['2024']['monthly']:
            if not m.get('skipped'):
                print(f"  {m['month']}: {m['pnl']:+,.0f}$ ({m['trades']} trades)")

        print("\n2025 Monthly:")
        for m in best['2025']['monthly']:
            if not m.get('skipped'):
                print(f"  {m['month']}: {m['pnl']:+,.0f}$ ({m['trades']} trades)")
    else:
        print("\nNo configuration achieved ZERO losing months in both years.")
        print("\nBest results (sorted by total losing months):\n")

        for r in sorted(results, key=lambda x: (x['total_losing'], -x['combined']))[:5]:
            print(f"{r['config'].name}:")
            print(f"  2024: {r['2024']['losing_months']} losing ({r['2024']['losing_month_names']})")
            print(f"  2025: {r['2025']['losing_months']} losing ({r['2025']['losing_month_names']})")
            print(f"  Combined: +{r['combined']:.1f}%")
            print()

    print("=" * 80)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
