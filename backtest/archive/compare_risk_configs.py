"""Compare Risk Configurations
==============================

Simple comparison of different risk settings using the same
monthly compounding approach as the original backtest.

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple
from loguru import logger

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
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=50000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def run_monthly_backtest_with_config(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    max_lot: float,
    min_sl: float,
    max_sl: float,
    config_name: str
) -> Dict:
    """Run 13-month backtest with specific config (with compounding)"""

    months = []
    for year in [2025, 2026]:
        for month in range(1, 13):
            if year == 2025 or (year == 2026 and month == 1):
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                months.append((start, end))

    running_balance = 10000.0
    total_trades = 0
    total_wins = 0
    monthly_results = []
    max_dd = 0

    for start_date, end_date in months:
        # Filter data
        warmup_days = 30
        month_start_warmup = start_date - timedelta(days=warmup_days)

        htf_month = htf_df[(htf_df.index >= month_start_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_warmup) & (ltf_df.index <= end_date)]

        if htf_month.empty or ltf_month.empty:
            continue

        # Prepare data
        htf = htf_month.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        ltf = ltf_month.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)

        # Run backtest
        bt = Backtester(
            symbol="GBPUSD",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_balance=running_balance,
            pip_value=10.0,
            spread_pips=1.5,
            use_killzone=True,
            use_trend_filter=True,
            use_relaxed_filter=True,
            use_hybrid_mode=False
        )

        # Apply custom risk settings
        bt.risk_manager.max_lot_size = max_lot
        bt.risk_manager.min_sl_pips = min_sl
        bt.risk_manager.max_sl_pips = max_sl

        bt.load_data(htf, ltf)

        try:
            result = bt.run()
            monthly_results.append({
                'month': start_date.strftime("%b %Y"),
                'pnl': result.net_profit,
                'trades': result.total_trades,
                'wins': result.winning_trades,
                'dd': result.max_drawdown_percent
            })
            running_balance = result.final_balance
            total_trades += result.total_trades
            total_wins += result.winning_trades
            if result.max_drawdown_percent > max_dd:
                max_dd = result.max_drawdown_percent
        except Exception as e:
            pass

    total_return = (running_balance - 10000) / 10000 * 100
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'config': config_name,
        'max_lot': max_lot,
        'min_sl': min_sl,
        'max_sl': max_sl,
        'final_balance': running_balance,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'monthly': monthly_results
    }


async def main():
    """Compare configurations"""

    print("\n" + "=" * 70)
    print("RISK CONFIGURATION COMPARISON")
    print("13-Month Backtest with Compounding")
    print("=" * 70)

    # Fetch data
    print("\nFetching data...")
    warmup_start = datetime(2024, 10, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", warmup_start, end_date)
    ltf_df = await fetch_data("GBPUSD", "M15", warmup_start, end_date)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Configurations to test
    configs = [
        ("Conservative", 0.3, 20.0, 40.0),
        ("Balanced", 0.5, 15.0, 50.0),
        ("Moderate", 0.6, 12.0, 50.0),
        ("Aggressive", 0.75, 10.0, 50.0),
        ("No Cap", 5.0, 10.0, 100.0),
    ]

    results = []
    for name, max_lot, min_sl, max_sl in configs:
        print(f"\nTesting: {name}...")
        result = run_monthly_backtest_with_config(
            htf_df, ltf_df, max_lot, min_sl, max_sl, name
        )
        results.append(result)
        print(f"  Return: {result['total_return']:+.1f}%, DD: {result['max_drawdown']:.1f}%")

    # Display comparison
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    print("\n{:<15} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
        "Config", "MaxLot", "Return%", "MaxDD%", "Trades", "WinRate%", "Risk-Adj"
    ))
    print("-" * 75)

    for r in results:
        risk_adj = r['total_return'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0
        print("{:<15} {:>8.2f} {:>+7.1f}% {:>7.1f}% {:>8} {:>9.1f}% {:>9.2f}".format(
            r['config'],
            r['max_lot'],
            r['total_return'],
            r['max_drawdown'],
            r['total_trades'],
            r['win_rate'],
            risk_adj
        ))

    # Find best by risk-adjusted return
    best = max(results, key=lambda x: x['total_return'] / x['max_drawdown'] if x['max_drawdown'] > 0 else 0)

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Monthly comparison for top configs
    print("\nMonthly P/L Comparison (Top 3):")
    top_configs = sorted(results, key=lambda x: x['total_return'], reverse=True)[:3]

    print("\n{:<10}".format("Month"), end="")
    for r in top_configs:
        print("{:>12}".format(r['config'][:10]), end="")
    print()
    print("-" * 50)

    for i in range(13):
        month_name = top_configs[0]['monthly'][i]['month'] if i < len(top_configs[0]['monthly']) else "N/A"
        print("{:<10}".format(month_name), end="")
        for r in top_configs:
            if i < len(r['monthly']):
                pnl = r['monthly'][i]['pnl']
                print("{:>+11.0f}$".format(pnl), end="")
            else:
                print("{:>12}".format("N/A"), end="")
        print()

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Score configs
    print("\nScoring (Return 40%, Risk-Adj 40%, Low DD 20%):")
    scored = []
    max_ret = max(r['total_return'] for r in results)
    max_risk_adj = max(r['total_return'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0 for r in results)
    min_dd = min(r['max_drawdown'] for r in results)
    max_dd_val = max(r['max_drawdown'] for r in results)

    for r in results:
        risk_adj = r['total_return'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0
        # Normalize (handle edge cases)
        ret_score = r['total_return'] / max_ret * 40 if max_ret > 0 else 0
        risk_score = risk_adj / max_risk_adj * 40 if max_risk_adj > 0 else 0
        dd_score = (1 - (r['max_drawdown'] - min_dd) / (max_dd_val - min_dd)) * 20 if max_dd_val > min_dd else 20

        total_score = ret_score + risk_score + dd_score
        scored.append((r, total_score))
        print(f"  {r['config']}: {total_score:.1f} pts (Ret:{ret_score:.1f} + Risk:{risk_score:.1f} + DD:{dd_score:.1f})")

    winner = max(scored, key=lambda x: x[1])[0]

    print(f"\n*** RECOMMENDED: {winner['config']} ***")
    print(f"    Max Lot: {winner['max_lot']}")
    print(f"    Min SL:  {winner['min_sl']} pips")
    print(f"    Max SL:  {winner['max_sl']} pips")
    print(f"    Expected Return: {winner['total_return']:+.1f}%")
    print(f"    Max Drawdown:    {winner['max_drawdown']:.1f}%")

    # Save
    output_path = Path(__file__).parent / "results" / "config_comparison.csv"
    pd.DataFrame([{
        'config': r['config'],
        'max_lot': r['max_lot'],
        'min_sl': r['min_sl'],
        'max_sl': r['max_sl'],
        'return': r['total_return'],
        'max_dd': r['max_drawdown'],
        'trades': r['total_trades'],
        'win_rate': r['win_rate']
    } for r in results]).to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
