"""Test Monthly Profit Protection
=================================

Test if Monthly Profit Protection can achieve ZERO losing months.

Key strategies:
1. Stricter quality filters when in loss
2. Daily loss limits
3. Consecutive loss cooldown
4. Progressive position reduction
5. Early profit taking during recovery

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
from typing import List, Dict

from config import config
from src.data.db_handler import DBHandler
from src.trading.adaptive_risk import AdaptiveRiskManager, calculate_atr
from src.trading.monthly_profit_protection import (
    MonthlyProfitProtection, ProtectionLevel, IntraMonthRecovery
)
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


def run_protected_backtest(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    use_protection: bool = True,
    protection_config: Dict = None
) -> Dict:
    """Run backtest with Monthly Profit Protection

    Args:
        htf_df: Higher timeframe data
        ltf_df: Lower timeframe data
        use_protection: Enable protection system
        protection_config: Custom protection parameters

    Returns:
        Results dictionary with monthly breakdown
    """
    # Default protection config (tuned for zero losing months)
    default_config = {
        'cautious_threshold': -0.3,      # -0.3% -> cautious
        'defensive_threshold': -0.7,     # -0.7% -> defensive
        'lockdown_threshold': -1.2,      # -1.2% -> lockdown
        'stop_threshold': -2.0,          # -2.0% -> stop
        'daily_loss_limit': 0.8,         # 0.8% daily loss limit
        'max_trades_per_day_normal': 2,
        'max_trades_per_day_cautious': 1,
        'max_trades_per_day_defensive': 1,
        'consecutive_loss_pause': 2,     # Pause after 2 losses
        'consecutive_loss_cooldown_hours': 6,
        'min_quality_normal': 65.0,      # Higher base quality
        'min_quality_cautious': 75.0,
        'min_quality_defensive': 85.0,
        'min_quality_lockdown': 95.0,
        'lot_mult_cautious': 0.6,
        'lot_mult_defensive': 0.3,
        'lot_mult_lockdown': 0.15,
    }

    if protection_config:
        default_config.update(protection_config)

    # Initialize protection system
    protection = MonthlyProfitProtection(**default_config) if use_protection else None
    recovery = IntraMonthRecovery(protection) if protection else None

    # Initialize adaptive risk
    adaptive_risk = AdaptiveRiskManager(
        base_max_lot=0.75,
        low_volatility_atr=12.0,
        high_volatility_atr=40.0,
        extreme_volatility_atr=55.0,
        consecutive_loss_threshold=3,
        drawdown_threshold=0.10
    )

    # Define months to test
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
    skipped_by_protection = 0

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

        # Calculate ATR for adaptive risk
        atr_pips = calculate_atr(htf, period=14)

        # Get adaptive parameters
        adaptive_params = adaptive_risk.get_adaptive_params(
            current_balance=running_balance,
            atr_pips=atr_pips,
            regime_confidence=0.7,
            current_time=start_date
        )

        # Start month tracking in protection system
        if protection:
            protection.start_month(start_date.year, start_date.month, running_balance)

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
            use_relaxed_filter=True,
            use_hybrid_mode=False
        )

        # Apply adaptive risk settings
        bt.risk_manager.max_lot_size = adaptive_params.max_lot_size
        bt.risk_manager.min_sl_pips = adaptive_params.min_sl_pips
        bt.risk_manager.max_sl_pips = adaptive_params.max_sl_pips

        # If using protection, apply stricter quality filter
        if protection:
            # Get initial protection params
            prot_params = protection.get_protection_params(start_date)
            bt.entry_trigger.min_quality_score = prot_params.min_quality_score

        bt.load_data(htf, ltf)

        try:
            result = bt.run()

            # Process trades with protection system
            month_pnl = 0
            month_trades = 0
            month_wins = 0
            month_skipped = 0

            if protection and result.trade_list:
                # Re-evaluate trades with protection system
                # This simulates what would have happened with protection
                simulated_balance = running_balance
                simulated_trades = []

                for trade in result.trade_list:
                    # Check protection before each trade
                    prot_params = protection.get_protection_params(trade.entry_time)

                    if not prot_params.can_trade:
                        # Would skip this trade
                        month_skipped += 1
                        skipped_by_protection += 1
                        logger.debug(f"Skipped trade at {trade.entry_time}: {prot_params.reason}")
                        continue

                    # Check if trade meets quality requirement
                    if trade.quality_score < prot_params.min_quality_score:
                        month_skipped += 1
                        skipped_by_protection += 1
                        logger.debug(f"Quality filter: {trade.quality_score:.0f} < {prot_params.min_quality_score:.0f}")
                        continue

                    # Apply lot multiplier
                    adjusted_pnl = trade.pnl * prot_params.lot_multiplier

                    # Record trade
                    is_win = adjusted_pnl > 0
                    protection.record_trade(trade.entry_time, adjusted_pnl, is_win)
                    protection.update_balance(simulated_balance + adjusted_pnl)

                    # Update adaptive risk
                    adaptive_risk.record_trade_result(is_win, adjusted_pnl)

                    simulated_balance += adjusted_pnl
                    month_pnl += adjusted_pnl
                    month_trades += 1
                    if is_win:
                        month_wins += 1

                    simulated_trades.append({
                        'time': trade.entry_time,
                        'direction': trade.direction,
                        'original_pnl': trade.pnl,
                        'adjusted_pnl': adjusted_pnl,
                        'lot_mult': prot_params.lot_multiplier,
                        'protection_level': prot_params.level.value
                    })

                # End month in protection system
                protection.end_month()

                # Update running balance with protected result
                running_balance = simulated_balance

            else:
                # No protection - use original results
                for trade in result.trade_list:
                    adaptive_risk.record_trade_result(trade.pnl > 0, trade.pnl)

                month_pnl = result.net_profit
                month_trades = result.total_trades
                month_wins = result.winning_trades
                running_balance = result.final_balance

            # Track results
            monthly_results.append({
                'month': start_date.strftime("%b %Y"),
                'pnl': month_pnl,
                'trades': month_trades,
                'wins': month_wins,
                'skipped': month_skipped,
                'balance': running_balance,
                'atr': atr_pips,
                'max_lot': adaptive_params.max_lot_size
            })

            total_trades += month_trades
            total_wins += month_wins
            if result.max_drawdown_percent > max_dd:
                max_dd = result.max_drawdown_percent

        except Exception as e:
            logger.error(f"Error in {start_date.strftime('%b %Y')}: {e}")

    # Calculate summary
    losing_months = [m for m in monthly_results if m['pnl'] < 0]
    profitable_months = [m for m in monthly_results if m['pnl'] >= 0]

    return {
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'max_drawdown': max_dd,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'profitable_months': len(profitable_months),
        'skipped_by_protection': skipped_by_protection
    }


async def main():
    """Compare protected vs unprotected backtest"""
    print("\n" + "=" * 70)
    print("MONTHLY PROFIT PROTECTION TEST")
    print("Goal: ZERO Losing Months")
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

    # Test 1: Without protection (baseline)
    print("\n" + "-" * 50)
    print("TEST 1: WITHOUT PROTECTION (Baseline)")
    print("-" * 50)
    baseline = run_protected_backtest(htf_df, ltf_df, use_protection=False)

    # Test 2: With protection (default settings)
    print("\n" + "-" * 50)
    print("TEST 2: WITH PROTECTION (Default)")
    print("-" * 50)
    protected = run_protected_backtest(htf_df, ltf_df, use_protection=True)

    # Test 3: With aggressive protection (stricter)
    print("\n" + "-" * 50)
    print("TEST 3: WITH AGGRESSIVE PROTECTION")
    print("-" * 50)
    aggressive_config = {
        'cautious_threshold': -0.2,      # Very early cautious
        'defensive_threshold': -0.5,     # Earlier defensive
        'lockdown_threshold': -1.0,      # Earlier lockdown
        'stop_threshold': -1.5,          # Earlier stop
        'daily_loss_limit': 0.5,         # Tighter daily limit
        'consecutive_loss_pause': 1,     # Pause after 1 loss!
        'consecutive_loss_cooldown_hours': 8,
        'min_quality_normal': 70.0,      # Higher base quality
        'min_quality_cautious': 80.0,
        'min_quality_defensive': 90.0,
        'lot_mult_cautious': 0.5,
        'lot_mult_defensive': 0.25,
        'lot_mult_lockdown': 0.1,
    }
    aggressive = run_protected_backtest(htf_df, ltf_df, use_protection=True,
                                        protection_config=aggressive_config)

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n{:<20} {:>12} {:>12} {:>12}".format(
        "Metric", "No Protect", "Protected", "Aggressive"
    ))
    print("-" * 56)
    print("{:<20} {:>+11.1f}% {:>+11.1f}% {:>+11.1f}%".format(
        "Total Return",
        baseline['total_return'],
        protected['total_return'],
        aggressive['total_return']
    ))
    print("{:<20} {:>12} {:>12} {:>12}".format(
        "Total Trades",
        baseline['total_trades'],
        protected['total_trades'],
        aggressive['total_trades']
    ))
    print("{:<20} {:>11.1f}% {:>11.1f}% {:>11.1f}%".format(
        "Win Rate",
        baseline['win_rate'],
        protected['win_rate'],
        aggressive['win_rate']
    ))
    print("{:<20} {:>11.1f}% {:>11.1f}% {:>11.1f}%".format(
        "Max Drawdown",
        baseline['max_drawdown'],
        protected['max_drawdown'],
        aggressive['max_drawdown']
    ))
    print("{:<20} {:>12} {:>12} {:>12}".format(
        "Losing Months",
        baseline['losing_months'],
        protected['losing_months'],
        aggressive['losing_months']
    ))
    print("{:<20} {:>12} {:>12} {:>12}".format(
        "Skipped Trades",
        0,
        protected['skipped_by_protection'],
        aggressive['skipped_by_protection']
    ))

    # Monthly breakdown
    print("\n" + "-" * 70)
    print("MONTHLY P/L COMPARISON")
    print("-" * 70)
    print("{:<10} {:>12} {:>12} {:>12}".format(
        "Month", "No Protect", "Protected", "Aggressive"
    ))
    print("-" * 46)

    for i, (b, p, a) in enumerate(zip(baseline['monthly'], protected['monthly'], aggressive['monthly'])):
        marker_b = " ❌" if b['pnl'] < 0 else ""
        marker_p = " ❌" if p['pnl'] < 0 else ""
        marker_a = " ❌" if a['pnl'] < 0 else ""
        print("{:<10} {:>+10.0f}${:<2} {:>+10.0f}${:<2} {:>+10.0f}${:<2}".format(
            b['month'], b['pnl'], marker_b, p['pnl'], marker_p, a['pnl'], marker_a
        ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check which achieved zero losing months
    best = None
    for name, result in [("No Protection", baseline), ("Protected", protected), ("Aggressive", aggressive)]:
        status = "✅ ZERO LOSING MONTHS" if result['losing_months'] == 0 else f"❌ {result['losing_months']} losing months"
        print(f"\n{name}: {status}")
        print(f"  Return: {result['total_return']:+.1f}%")
        print(f"  Trades: {result['total_trades']}")

        if result['losing_months'] == 0:
            if best is None or result['total_return'] > best[1]['total_return']:
                best = (name, result)

    if best:
        print(f"\n*** WINNER: {best[0]} ***")
        print(f"    Achieved ZERO losing months with {best[1]['total_return']:+.1f}% return")
    else:
        print("\n*** No configuration achieved zero losing months ***")
        print("Need to further tune protection parameters...")

        # Show which months are still losing
        print("\nStill losing in Aggressive mode:")
        for m in aggressive['monthly']:
            if m['pnl'] < 0:
                print(f"  - {m['month']}: ${m['pnl']:.2f} ({m['trades']} trades, {m['skipped']} skipped)")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
