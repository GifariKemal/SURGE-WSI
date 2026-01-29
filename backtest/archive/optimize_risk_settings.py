"""Optimize Risk Settings
========================

Test multiple risk configurations to find optimal settings.
Also implements Adaptive Risk Management based on market conditions.

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester


@dataclass
class ConfigResult:
    """Result for a single configuration"""
    config_name: str
    max_lot: float
    min_sl: float
    max_sl: float
    final_balance: float
    total_return: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    avg_monthly_return: float
    worst_month: float
    best_month: float


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

    df = await db.get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=50000,
        start_time=start,
        end_time=end
    )
    await db.disconnect()
    return df


def run_backtest_with_config(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    max_lot: float,
    min_sl: float,
    max_sl: float,
    start_date: str,
    end_date: str
) -> Tuple[float, int, float, float, float, float, List[float]]:
    """Run backtest with specific config"""

    # Prepare data
    htf = htf_df.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)
    ltf = ltf_df.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)

    # Create backtester with custom config
    bt = Backtester(
        symbol="GBPUSD",
        start_date=start_date,
        end_date=end_date,
        initial_balance=10000.0,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=True,
        use_trend_filter=True,
        use_relaxed_filter=True,
        use_hybrid_mode=False
    )

    # Override risk manager settings
    bt.risk_manager.max_lot_size = max_lot
    bt.risk_manager.min_sl_pips = min_sl
    bt.risk_manager.max_sl_pips = max_sl

    bt.load_data(htf, ltf)
    result = bt.run()

    # Calculate monthly returns for Sharpe
    monthly_returns = []
    if result.trade_list:
        trades_df = bt.get_trades_df()
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        monthly_returns = (monthly_pnl / 10000 * 100).tolist()

    return (
        result.final_balance,
        result.total_trades,
        result.win_rate,
        result.profit_factor,
        result.max_drawdown_percent,
        result.sharpe_ratio,
        monthly_returns
    )


async def test_configurations():
    """Test multiple risk configurations"""

    print("\n" + "=" * 70)
    print("RISK SETTINGS OPTIMIZATION")
    print("Testing multiple configurations to find optimal settings")
    print("=" * 70)

    # Fetch data
    print("\nFetching data...")
    warmup_start = datetime(2024, 10, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", warmup_start, end_date)
    ltf_df = await fetch_data("GBPUSD", "M15", warmup_start, end_date)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Define configurations to test
    configs = [
        # (name, max_lot, min_sl, max_sl)
        ("Conservative", 0.3, 20.0, 40.0),
        ("Moderate-Low", 0.4, 15.0, 45.0),
        ("Moderate", 0.5, 15.0, 50.0),
        ("Moderate-High", 0.6, 12.0, 50.0),
        ("Aggressive", 0.75, 10.0, 50.0),
        ("Very Aggressive", 1.0, 10.0, 50.0),
        ("Original (No Cap)", 5.0, 10.0, 100.0),  # Effectively no cap
    ]

    results: List[ConfigResult] = []

    for name, max_lot, min_sl, max_sl in configs:
        print(f"\nTesting: {name} (max_lot={max_lot}, min_sl={min_sl}, max_sl={max_sl})...")

        try:
            final_bal, trades, win_rate, pf, max_dd, sharpe, monthly_rets = run_backtest_with_config(
                htf_df, ltf_df,
                max_lot, min_sl, max_sl,
                "2025-01-01", "2026-01-31"
            )

            total_return = (final_bal - 10000) / 10000 * 100
            avg_monthly = np.mean(monthly_rets) if monthly_rets else 0
            worst_month = min(monthly_rets) if monthly_rets else 0
            best_month = max(monthly_rets) if monthly_rets else 0

            results.append(ConfigResult(
                config_name=name,
                max_lot=max_lot,
                min_sl=min_sl,
                max_sl=max_sl,
                final_balance=final_bal,
                total_return=total_return,
                total_trades=trades,
                win_rate=win_rate,
                profit_factor=pf,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                avg_monthly_return=avg_monthly,
                worst_month=worst_month,
                best_month=best_month
            ))

            print(f"  -> Return: {total_return:+.1f}%, DD: {max_dd:.1f}%, Sharpe: {sharpe:.2f}")

        except Exception as e:
            print(f"  -> ERROR: {e}")

    # Display results
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPARISON")
    print("=" * 70)

    print("\n{:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
        "Config", "Return%", "MaxDD%", "WinRate", "PF", "Sharpe", "Risk-Adj"
    ))
    print("-" * 70)

    for r in results:
        # Risk-adjusted return = Return / MaxDD
        risk_adj = r.total_return / r.max_drawdown if r.max_drawdown > 0 else 0
        print("{:<18} {:>+7.1f}% {:>7.1f}% {:>7.1f}% {:>7.2f} {:>7.2f} {:>9.2f}".format(
            r.config_name,
            r.total_return,
            r.max_drawdown,
            r.win_rate,
            r.profit_factor,
            r.sharpe_ratio,
            risk_adj
        ))

    # Find best configurations
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS BY METRIC")
    print("=" * 70)

    best_return = max(results, key=lambda x: x.total_return)
    best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
    best_risk_adj = max(results, key=lambda x: x.total_return / x.max_drawdown if x.max_drawdown > 0 else 0)
    lowest_dd = min(results, key=lambda x: x.max_drawdown)

    print(f"\nHighest Return:      {best_return.config_name} ({best_return.total_return:+.1f}%)")
    print(f"Best Sharpe Ratio:   {best_sharpe.config_name} ({best_sharpe.sharpe_ratio:.2f})")
    print(f"Best Risk-Adjusted:  {best_risk_adj.config_name} (Return/DD = {best_risk_adj.total_return/best_risk_adj.max_drawdown:.2f})")
    print(f"Lowest Drawdown:     {lowest_dd.config_name} ({lowest_dd.max_drawdown:.1f}%)")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Score each config (weighted)
    # 40% return, 30% risk-adjusted, 20% sharpe, 10% win rate
    scored_results = []
    for r in results:
        risk_adj = r.total_return / r.max_drawdown if r.max_drawdown > 0 else 0

        # Normalize scores (0-100)
        max_ret = max(x.total_return for x in results)
        max_risk_adj = max(x.total_return / x.max_drawdown if x.max_drawdown > 0 else 0 for x in results)
        max_sharpe = max(x.sharpe_ratio for x in results)
        max_wr = max(x.win_rate for x in results)

        score = (
            (r.total_return / max_ret * 40 if max_ret > 0 else 0) +
            (risk_adj / max_risk_adj * 30 if max_risk_adj > 0 else 0) +
            (r.sharpe_ratio / max_sharpe * 20 if max_sharpe > 0 else 0) +
            (r.win_rate / max_wr * 10 if max_wr > 0 else 0)
        )
        scored_results.append((r, score))

    scored_results.sort(key=lambda x: x[1], reverse=True)

    print("\nRanking by Weighted Score (40% Return, 30% Risk-Adj, 20% Sharpe, 10% WinRate):")
    for i, (r, score) in enumerate(scored_results, 1):
        print(f"  {i}. {r.config_name}: {score:.1f} pts")

    winner = scored_results[0][0]
    print(f"\n*** RECOMMENDED CONFIG: {winner.config_name} ***")
    print(f"    Max Lot Size: {winner.max_lot}")
    print(f"    Min SL Pips:  {winner.min_sl}")
    print(f"    Max SL Pips:  {winner.max_sl}")
    print(f"    Expected Return: {winner.total_return:+.1f}%")
    print(f"    Max Drawdown:    {winner.max_drawdown:.1f}%")

    # Save results
    results_df = pd.DataFrame([{
        'config': r.config_name,
        'max_lot': r.max_lot,
        'min_sl': r.min_sl,
        'max_sl': r.max_sl,
        'final_balance': r.final_balance,
        'return_pct': r.total_return,
        'trades': r.total_trades,
        'win_rate': r.win_rate,
        'profit_factor': r.profit_factor,
        'max_drawdown': r.max_drawdown,
        'sharpe_ratio': r.sharpe_ratio
    } for r in results])

    output_path = Path(__file__).parent / "results" / "risk_optimization_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return winner


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(test_configurations())
