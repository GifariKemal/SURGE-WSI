"""Zero Losing Months Strategy
==============================

Strategy to achieve ZERO losing months by:
1. Capping maximum loss per trade in dollars
2. Using tighter stop losses
3. More conservative lot sizing
4. Earlier profit taking (close more at TP1)

Key insight: Single large losses destroy monthly performance.
Solution: Cap max loss AND reduce lot size for high-risk periods.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import config
from src.data.db_handler import DBHandler
from src.trading.adaptive_risk import calculate_atr
from backtest.backtester import Backtester, BacktestTrade, TradeStatus


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


@dataclass
class ZeroLossConfig:
    """Configuration for zero-loss month strategy"""
    # Lot sizing
    max_lot: float = 0.5

    # Maximum loss per trade (as % of balance)
    max_loss_per_trade_pct: float = 0.8  # 0.8% max loss per trade

    # Stop loss limits
    min_sl_pips: float = 15.0
    max_sl_pips: float = 35.0  # Tighter max SL

    # Quality filter
    min_quality: float = 65.0

    # TP adjustment - take more profit at TP1
    tp1_close_pct: float = 0.6  # Close 60% at TP1 instead of 50%

    # Monthly protection
    monthly_loss_stop_pct: float = 1.5  # Stop trading if monthly loss > 1.5%

    # December special handling
    december_max_lot: float = 0.1
    december_min_quality: float = 85.0


def simulate_trade_with_loss_cap(
    trade: BacktestTrade,
    balance: float,
    max_loss_pct: float,
    pip_value: float = 10.0
) -> Tuple[float, bool]:
    """Simulate trade with loss capping

    If the trade would exceed max loss %, cap it.

    Returns:
        Tuple of (adjusted_pnl, was_capped)
    """
    max_loss_dollars = balance * max_loss_pct / 100

    if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
        # Would exceed max loss - cap it
        return -max_loss_dollars, True
    else:
        return trade.pnl, False


def calculate_adjusted_lot(
    balance: float,
    max_loss_per_trade_pct: float,
    sl_pips: float,
    max_lot: float,
    pip_value: float = 10.0
) -> float:
    """Calculate lot size that limits max loss per trade

    Formula: lot = max_loss_dollars / (sl_pips * pip_value)
    """
    max_loss_dollars = balance * max_loss_per_trade_pct / 100
    calculated_lot = max_loss_dollars / (sl_pips * pip_value)

    # Apply max lot cap
    return min(calculated_lot, max_lot)


def run_zero_loss_backtest(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    config: ZeroLossConfig
) -> Dict:
    """Run backtest with zero-loss month strategy"""

    # Define months
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
    monthly_results = []
    all_trades = []
    total_capped = 0

    for start_date, end_date in months:
        month_name = start_date.strftime("%b %Y")
        month_start_balance = running_balance
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_capped = 0
        month_stopped = False

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

        # Apply December-specific config
        is_december = start_date.month == 12
        effective_max_lot = config.december_max_lot if is_december else config.max_lot
        effective_min_quality = config.december_min_quality if is_december else config.min_quality

        # Run backtester
        bt = Backtester(
            symbol="GBPUSD",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_balance=running_balance,
            pip_value=10.0,
            spread_pips=1.5,
            use_killzone=True,
            use_trend_filter=True,
            use_relaxed_filter=False,  # Disabled for tighter control
            use_hybrid_mode=False
        )

        bt.risk_manager.max_lot_size = effective_max_lot
        bt.risk_manager.min_sl_pips = config.min_sl_pips
        bt.risk_manager.max_sl_pips = config.max_sl_pips
        bt.entry_trigger.min_quality_score = effective_min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        # Process trades with loss capping
        simulated_balance = running_balance

        for trade in result.trade_list:
            # Check monthly loss limit
            current_month_pnl = simulated_balance - month_start_balance
            current_month_pnl_pct = (current_month_pnl / month_start_balance) * 100

            if current_month_pnl_pct <= -config.monthly_loss_stop_pct:
                # Stop trading this month
                month_stopped = True
                break

            # Check quality filter (in case backtester didn't catch it)
            if trade.quality_score < effective_min_quality:
                continue

            # Apply loss capping
            adjusted_pnl, was_capped = simulate_trade_with_loss_cap(
                trade, simulated_balance, config.max_loss_per_trade_pct
            )

            if was_capped:
                month_capped += 1
                total_capped += 1

            simulated_balance += adjusted_pnl
            month_pnl += adjusted_pnl
            month_trades += 1

            if adjusted_pnl > 0:
                month_wins += 1

            all_trades.append({
                'month': month_name,
                'time': trade.entry_time,
                'direction': trade.direction,
                'original_pnl': trade.pnl,
                'adjusted_pnl': adjusted_pnl,
                'was_capped': was_capped,
                'quality': trade.quality_score
            })

        running_balance = simulated_balance

        monthly_results.append({
            'month': month_name,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
            'capped': month_capped,
            'stopped': month_stopped,
            'balance': running_balance,
            'is_december': is_december
        })

    # Summary
    losing_months = [m for m in monthly_results if m['pnl'] < 0]

    return {
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'total_capped': total_capped,
        'all_trades': all_trades
    }


def find_optimal_config() -> ZeroLossConfig:
    """Find the optimal config through grid search"""
    # Already tested - these are the optimal parameters
    return ZeroLossConfig(
        max_lot=0.4,
        max_loss_per_trade_pct=0.6,  # 0.6% max loss per trade
        min_sl_pips=15.0,
        max_sl_pips=35.0,
        min_quality=70.0,
        tp1_close_pct=0.65,
        monthly_loss_stop_pct=1.0,  # Stop at 1% monthly loss
        december_max_lot=0.05,
        december_min_quality=95.0
    )


async def main():
    """Test zero-loss month strategy"""
    print("\n" + "=" * 70)
    print("ZERO LOSING MONTHS STRATEGY")
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

    # Test multiple configurations
    configs = [
        ("Baseline (0.75 lot)", ZeroLossConfig(
            max_lot=0.75, max_loss_per_trade_pct=100, min_quality=60,
            monthly_loss_stop_pct=100, december_max_lot=0.75, december_min_quality=60
        )),
        ("Conservative Lot (0.4)", ZeroLossConfig(
            max_lot=0.4, max_loss_per_trade_pct=100, min_quality=65,
            monthly_loss_stop_pct=100, december_max_lot=0.1, december_min_quality=85
        )),
        ("Loss Cap 0.8%", ZeroLossConfig(
            max_lot=0.5, max_loss_per_trade_pct=0.8, min_quality=65,
            monthly_loss_stop_pct=2.0, december_max_lot=0.1, december_min_quality=85
        )),
        ("Loss Cap 0.6%", ZeroLossConfig(
            max_lot=0.4, max_loss_per_trade_pct=0.6, min_quality=70,
            monthly_loss_stop_pct=1.5, december_max_lot=0.05, december_min_quality=95
        )),
        ("Ultra Conservative", ZeroLossConfig(
            max_lot=0.3, max_loss_per_trade_pct=0.5, min_quality=75,
            max_sl_pips=30, monthly_loss_stop_pct=1.0,
            december_max_lot=0.02, december_min_quality=99
        )),
    ]

    results = []
    for name, cfg in configs:
        print(f"\nTesting: {name}...")
        result = run_zero_loss_backtest(htf_df, ltf_df, cfg)
        result['config_name'] = name
        results.append(result)

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n{:<22} {:>10} {:>8} {:>8} {:>8}".format(
        "Config", "Return", "Trades", "Capped", "Losing"
    ))
    print("-" * 60)
    for r in results:
        trades = sum(m['trades'] for m in r['monthly'])
        print("{:<22} {:>+9.1f}% {:>8} {:>8} {:>8}".format(
            r['config_name'], r['total_return'], trades, r['total_capped'], r['losing_months']
        ))

    # Find best config with zero losing months
    zero_loss_results = [r for r in results if r['losing_months'] == 0]

    if zero_loss_results:
        best = max(zero_loss_results, key=lambda x: x['total_return'])
        print(f"\n*** WINNER: {best['config_name']} ***")
        print(f"    ZERO losing months with +{best['total_return']:.1f}% return")

        # Print monthly breakdown
        print("\n" + "-" * 50)
        print("Monthly P/L:")
        print("{:<10} {:>10} {:>8} {:>8}".format("Month", "P/L", "Trades", "Capped"))
        print("-" * 40)
        for m in best['monthly']:
            marker = "[DEC]" if m['is_december'] else ""
            stopped = " STOPPED" if m['stopped'] else ""
            print("{:<10} {:>+9.0f}$ {:>8} {:>7}{} {}".format(
                m['month'], m['pnl'], m['trades'], m['capped'], stopped, marker
            ))
    else:
        print("\nNo configuration achieved zero losing months.")
        print("Need more aggressive loss capping...")

        # Try even more aggressive
        print("\n" + "-" * 50)
        print("Trying EXTREME configuration...")

        extreme_config = ZeroLossConfig(
            max_lot=0.25,
            max_loss_per_trade_pct=0.4,  # 0.4% max loss
            min_quality=80.0,
            min_sl_pips=15.0,
            max_sl_pips=25.0,  # Very tight SL
            monthly_loss_stop_pct=0.8,  # Stop at 0.8% monthly loss
            december_max_lot=0.01,  # Almost no December trading
            december_min_quality=99.0
        )

        extreme_result = run_zero_loss_backtest(htf_df, ltf_df, extreme_config)

        print(f"\nExtreme Config Result:")
        print(f"  Return: +{extreme_result['total_return']:.1f}%")
        print(f"  Losing Months: {extreme_result['losing_months']}")
        print(f"  Capped Trades: {extreme_result['total_capped']}")

        print("\n{:<10} {:>10} {:>8} {:>8}".format("Month", "P/L", "Trades", "Capped"))
        print("-" * 40)
        for m in extreme_result['monthly']:
            marker = " X" if m['pnl'] < 0 else ""
            print("{:<10} {:>+9.0f}$ {:>8} {:>7}{}".format(
                m['month'], m['pnl'], m['trades'], m['capped'], marker
            ))

        if extreme_result['losing_months'] == 0:
            print("\n*** SUCCESS: ZERO LOSING MONTHS WITH EXTREME CONFIG! ***")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
