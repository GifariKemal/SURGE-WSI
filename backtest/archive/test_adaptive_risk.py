"""Test Adaptive Risk Manager
=============================

Compare fixed risk settings vs adaptive risk management.

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.trading.adaptive_risk import AdaptiveRiskManager, calculate_atr
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


def run_adaptive_backtest(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    use_adaptive: bool = True
) -> dict:
    """Run 13-month backtest with adaptive or fixed risk"""

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

    # Initialize adaptive risk manager - tuned for GBPUSD
    adaptive_risk = AdaptiveRiskManager(
        base_max_lot=0.75,     # Higher base for better returns
        base_min_sl=15.0,
        base_max_sl=50.0,
        base_risk_percent=0.01,
        low_volatility_atr=12.0,       # GBPUSD rarely below 12
        high_volatility_atr=40.0,      # Only reduce above 40
        extreme_volatility_atr=55.0,   # Extreme above 55
        consecutive_loss_threshold=3,
        drawdown_threshold=0.10
    )

    for start_date, end_date in months:
        warmup_days = 30
        month_start_warmup = start_date - timedelta(days=warmup_days)

        htf_month = htf_df[(htf_df.index >= month_start_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_warmup) & (ltf_df.index <= end_date)]

        if htf_month.empty or ltf_month.empty:
            continue

        htf = htf_month.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        ltf = ltf_month.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)

        # Calculate ATR for this month
        atr_pips = calculate_atr(htf, period=14)

        # Get adaptive parameters
        if use_adaptive:
            params = adaptive_risk.get_adaptive_params(
                current_balance=running_balance,
                atr_pips=atr_pips,
                regime_confidence=0.7,
                current_time=start_date
            )
            max_lot = params.max_lot_size
            min_sl = params.min_sl_pips
            max_sl = params.max_sl_pips
            reason = params.reason
        else:
            # Fixed aggressive settings (for fair comparison)
            max_lot = 0.75
            min_sl = 15.0
            max_sl = 50.0
            reason = "Fixed"

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

        bt.risk_manager.max_lot_size = max_lot
        bt.risk_manager.min_sl_pips = min_sl
        bt.risk_manager.max_sl_pips = max_sl

        bt.load_data(htf, ltf)

        try:
            result = bt.run()

            # Update adaptive risk manager with results
            if use_adaptive:
                for trade in result.trade_list:
                    adaptive_risk.record_trade_result(trade.pnl > 0, trade.pnl)

            monthly_results.append({
                'month': start_date.strftime("%b %Y"),
                'pnl': result.net_profit,
                'trades': result.total_trades,
                'wins': result.winning_trades,
                'dd': result.max_drawdown_percent,
                'max_lot': max_lot,
                'reason': reason,
                'atr': atr_pips
            })

            running_balance = result.final_balance
            total_trades += result.total_trades
            total_wins += result.winning_trades
            if result.max_drawdown_percent > max_dd:
                max_dd = result.max_drawdown_percent

        except Exception as e:
            logger.error(f"Error: {e}")

    return {
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'max_drawdown': max_dd,
        'monthly': monthly_results
    }


async def main():
    """Compare fixed vs adaptive risk"""

    print("\n" + "=" * 70)
    print("ADAPTIVE RISK MANAGER TEST")
    print("Fixed Settings vs Adaptive Settings")
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

    # Test Fixed
    print("\n" + "-" * 40)
    print("Testing FIXED settings (max_lot=0.6)...")
    print("-" * 40)
    fixed_result = run_adaptive_backtest(htf_df, ltf_df, use_adaptive=False)

    # Test Adaptive
    print("\n" + "-" * 40)
    print("Testing ADAPTIVE settings...")
    print("-" * 40)
    adaptive_result = run_adaptive_backtest(htf_df, ltf_df, use_adaptive=True)

    # Comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n{:<20} {:>15} {:>15}".format("Metric", "FIXED", "ADAPTIVE"))
    print("-" * 50)
    print("{:<20} {:>14.1f}% {:>14.1f}%".format("Total Return",
        fixed_result['total_return'], adaptive_result['total_return']))
    print("{:<20} {:>14.1f}% {:>14.1f}%".format("Max Drawdown",
        fixed_result['max_drawdown'], adaptive_result['max_drawdown']))
    print("{:<20} {:>15} {:>15}".format("Total Trades",
        fixed_result['total_trades'], adaptive_result['total_trades']))
    print("{:<20} {:>14.1f}% {:>14.1f}%".format("Win Rate",
        fixed_result['win_rate'], adaptive_result['win_rate']))

    # Risk-adjusted
    fixed_risk_adj = fixed_result['total_return'] / fixed_result['max_drawdown'] if fixed_result['max_drawdown'] > 0 else 0
    adaptive_risk_adj = adaptive_result['total_return'] / adaptive_result['max_drawdown'] if adaptive_result['max_drawdown'] > 0 else 0
    print("{:<20} {:>15.2f} {:>15.2f}".format("Risk-Adjusted", fixed_risk_adj, adaptive_risk_adj))

    # Monthly comparison
    print("\n" + "-" * 70)
    print("MONTHLY COMPARISON")
    print("-" * 70)
    print("{:<10} {:>10} {:>10} {:>15} {:>8}".format(
        "Month", "Fixed$", "Adapt$", "Adapt Reason", "ATR"
    ))
    print("-" * 70)

    for i, (f, a) in enumerate(zip(fixed_result['monthly'], adaptive_result['monthly'])):
        print("{:<10} {:>+9.0f}$ {:>+9.0f}$ {:>15} {:>7.1f}".format(
            f['month'], f['pnl'], a['pnl'], a['reason'][:15], a['atr']
        ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    improvement = adaptive_result['total_return'] - fixed_result['total_return']
    dd_improvement = fixed_result['max_drawdown'] - adaptive_result['max_drawdown']

    if improvement > 0:
        print(f"\nAdaptive is BETTER: +{improvement:.1f}% more return")
    else:
        print(f"\nFixed is BETTER: {-improvement:.1f}% more return")

    if dd_improvement > 0:
        print(f"Adaptive has LOWER drawdown: -{dd_improvement:.1f}%")
    else:
        print(f"Fixed has LOWER drawdown: {-dd_improvement:.1f}%")

    print(f"\nAdaptive Risk-Adjusted: {adaptive_risk_adj:.2f}")
    print(f"Fixed Risk-Adjusted:    {fixed_risk_adj:.2f}")

    if adaptive_risk_adj > fixed_risk_adj:
        print("\n*** ADAPTIVE has better risk-adjusted returns ***")
    else:
        print("\n*** FIXED has better risk-adjusted returns ***")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
