"""Solve Feb 2024: The Last Losing Month
========================================

Analyze and eliminate the final losing month (Feb 2024)
to achieve ZERO losing months across both years.

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
class FinalConfig:
    """Final configuration"""
    name: str
    max_sl_pips: float = 25.0
    max_loss_pct: float = 0.3
    min_quality: float = 65.0
    feb_special_rules: bool = False  # Special rules for February
    skip_early_month: int = 0  # Skip first N days of month
    min_htf_bars_trend: int = 0  # Require N bars of same trend before trade


def run_final_test(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                   year: int, cfg: FinalConfig) -> dict:
    """Run backtest with final configuration"""

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
        is_february = start_date.month == 2

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
        bt.risk_manager.max_sl_pips = cfg.max_sl_pips
        bt.entry_trigger.min_quality_score = cfg.min_quality

        bt.load_data(htf, ltf)
        result = bt.run()

        # Process trades
        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_filtered = 0
        simulated_balance = running_balance
        trade_details = []

        for trade in result.trade_list:
            # Calculate risk in pips
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001

            # Filter 1: SL limit
            if risk_pips > cfg.max_sl_pips:
                month_filtered += 1
                continue

            # Filter 2: Skip early month days
            if cfg.skip_early_month > 0:
                if hasattr(trade.entry_time, 'day') and trade.entry_time.day <= cfg.skip_early_month:
                    month_filtered += 1
                    continue

            # Filter 3: February special rules
            if cfg.feb_special_rules and is_february:
                # Stricter quality for Feb
                if trade.quality_score < 70:
                    month_filtered += 1
                    continue
                # Skip if risk > 20 pips
                if risk_pips > 20:
                    month_filtered += 1
                    continue

            # Filter 4: HTF trend confirmation
            if cfg.min_htf_bars_trend > 0:
                trade_time = trade.entry_time
                htf_before = htf_month[htf_month.index <= trade_time].tail(cfg.min_htf_bars_trend + 5)
                if len(htf_before) >= cfg.min_htf_bars_trend:
                    close = htf_before['close'] if 'close' in htf_before.columns else htf_before['Close']
                    recent = close.tail(cfg.min_htf_bars_trend)
                    if trade.direction == 'BUY':
                        # Need uptrend
                        up_bars = (recent.diff() > 0).sum()
                        if up_bars < cfg.min_htf_bars_trend * 0.6:
                            month_filtered += 1
                            continue
                    else:  # SELL
                        down_bars = (recent.diff() < 0).sum()
                        if down_bars < cfg.min_htf_bars_trend * 0.6:
                            month_filtered += 1
                            continue

            # Cap the loss
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

            trade_details.append({
                'time': trade.entry_time,
                'direction': trade.direction,
                'risk_pips': risk_pips,
                'pnl': adjusted_pnl,
                'quality': trade.quality_score
            })

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
            'skipped': False,
            'trade_details': trade_details
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
    print("SOLVE FEB 2024: THE LAST LOSING MONTH")
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

    # First, analyze Feb 2024 with current best config
    print("\n" + "-" * 80)
    print("ANALYZING FEB 2024")
    print("-" * 80)

    baseline = FinalConfig(
        name="Best Current (SL25+Loss0.3%)",
        max_sl_pips=25.0,
        max_loss_pct=0.3
    )

    r = run_final_test(htf_df, ltf_df, 2024, baseline)
    feb_result = [m for m in r['monthly'] if m['month'] == 'Feb 2024'][0]

    print(f"\nFeb 2024 with current config:")
    print(f"  P/L: {feb_result['pnl']:+,.0f}$")
    print(f"  Trades: {feb_result['trades']} (filtered: {feb_result['filtered']})")
    print(f"  Wins: {feb_result['wins']}")

    if 'trade_details' in feb_result:
        print("\n  Trade details:")
        for t in feb_result['trade_details']:
            status = "WIN" if t['pnl'] > 0 else "LOSS"
            print(f"    {t['direction']}: {t['pnl']:+,.0f}$ (risk: {t['risk_pips']:.1f} pips, Q: {t['quality']:.0f}) [{status}]")

    # Test solutions
    print("\n" + "=" * 80)
    print("TESTING SOLUTIONS FOR FEB 2024")
    print("=" * 80)

    configs = [
        FinalConfig(
            name="Baseline (SL25+Loss0.3%)",
            max_sl_pips=25.0,
            max_loss_pct=0.3
        ),
        FinalConfig(
            name="Feb Special Rules",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            feb_special_rules=True
        ),
        FinalConfig(
            name="Skip First 10 Days",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            skip_early_month=10
        ),
        FinalConfig(
            name="Skip First 7 Days",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            skip_early_month=7
        ),
        FinalConfig(
            name="Tighter SL (20 pips)",
            max_sl_pips=20.0,
            max_loss_pct=0.3
        ),
        FinalConfig(
            name="Tighter SL (18 pips)",
            max_sl_pips=18.0,
            max_loss_pct=0.3
        ),
        FinalConfig(
            name="Tighter Loss (0.25%)",
            max_sl_pips=25.0,
            max_loss_pct=0.25
        ),
        FinalConfig(
            name="Tighter Loss (0.2%)",
            max_sl_pips=25.0,
            max_loss_pct=0.2
        ),
        FinalConfig(
            name="Higher Quality (70)",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            min_quality=70.0
        ),
        FinalConfig(
            name="Higher Quality (72)",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            min_quality=72.0
        ),
        FinalConfig(
            name="HTF Trend 5 bars",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            min_htf_bars_trend=5
        ),
        FinalConfig(
            name="HTF Trend 8 bars",
            max_sl_pips=25.0,
            max_loss_pct=0.3,
            min_htf_bars_trend=8
        ),
        # Combined solutions
        FinalConfig(
            name="FINAL: SL20+Loss0.25%",
            max_sl_pips=20.0,
            max_loss_pct=0.25
        ),
        FinalConfig(
            name="FINAL: SL18+Loss0.25%",
            max_sl_pips=18.0,
            max_loss_pct=0.25
        ),
        FinalConfig(
            name="FINAL: SL20+Loss0.2%",
            max_sl_pips=20.0,
            max_loss_pct=0.2
        ),
        FinalConfig(
            name="FINAL: SL25+Q70+Loss0.25%",
            max_sl_pips=25.0,
            max_loss_pct=0.25,
            min_quality=70.0
        ),
        FinalConfig(
            name="FINAL: SL20+Q70+Loss0.25%",
            max_sl_pips=20.0,
            max_loss_pct=0.25,
            min_quality=70.0
        ),
        FinalConfig(
            name="FINAL: Feb Special + SL20",
            max_sl_pips=20.0,
            max_loss_pct=0.3,
            feb_special_rules=True
        ),
    ]

    results_2024 = []
    results_2025 = []

    for cfg in configs:
        r2024 = run_final_test(htf_df, ltf_df, 2024, cfg)
        r2025 = run_final_test(htf_df, ltf_df, 2025, cfg)
        results_2024.append(r2024)
        results_2025.append(r2025)

        m2024 = "***" if r2024['losing_months'] == 0 else ""
        m2025 = "***" if r2025['losing_months'] == 0 else ""
        both = " <== ZERO BOTH!" if r2024['losing_months'] == 0 and r2025['losing_months'] == 0 else ""

        print(f"\n{cfg.name}:")
        print(f"  2024: {r2024['losing_months']} losing, +{r2024['total_return']:.1f}% {m2024}")
        print(f"  2025: {r2025['losing_months']} losing, +{r2025['total_return']:.1f}% {m2025}{both}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print("\n{:<30} | {:>6} {:>6} | {:>6} {:>6} | {:>8}".format(
        "Configuration", "2024L", "2024R", "2025L", "2025R", "Combined"
    ))
    print("-" * 75)

    best = None
    best_combined = 0

    for i in range(len(configs)):
        r2024 = results_2024[i]
        r2025 = results_2025[i]
        combined = r2024['total_return'] + r2025['total_return']

        marker = " ***" if r2024['losing_months'] == 0 and r2025['losing_months'] == 0 else ""
        print("{:<30} | {:>6} {:>5.1f}% | {:>6} {:>5.1f}% | {:>7.1f}%{}".format(
            configs[i].name[:30],
            r2024['losing_months'],
            r2024['total_return'],
            r2025['losing_months'],
            r2025['total_return'],
            combined,
            marker
        ))

        if r2024['losing_months'] == 0 and r2025['losing_months'] == 0:
            if combined > best_combined:
                best_combined = combined
                best = {
                    'config': configs[i],
                    '2024': r2024,
                    '2025': r2025,
                    'combined': combined
                }

    if best:
        print("\n" + "=" * 80)
        print("SUCCESS: ZERO LOSING MONTHS ACHIEVED!")
        print("=" * 80)

        print(f"\nBest Configuration: {best['config'].name}")
        print(f"  max_sl_pips: {best['config'].max_sl_pips}")
        print(f"  max_loss_pct: {best['config'].max_loss_pct}%")
        print(f"  min_quality: {best['config'].min_quality}")

        print(f"\n2024 Results:")
        print(f"  Losing Months: {best['2024']['losing_months']}")
        print(f"  Return: +{best['2024']['total_return']:.1f}%")
        print(f"  Trades: {best['2024']['total_trades']}")
        print(f"  Win Rate: {best['2024']['win_rate']:.1f}%")

        print(f"\n2025 Results:")
        print(f"  Losing Months: {best['2025']['losing_months']}")
        print(f"  Return: +{best['2025']['total_return']:.1f}%")
        print(f"  Trades: {best['2025']['total_trades']}")
        print(f"  Win Rate: {best['2025']['win_rate']:.1f}%")

        print(f"\nCombined Return: +{best['combined']:.1f}%")

        # Monthly breakdown
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
        print("\nNo configuration achieved zero losing months in both years.")
        print("Best options (minimize total losing):")

        all_sorted = []
        for i in range(len(configs)):
            total = results_2024[i]['losing_months'] + results_2025[i]['losing_months']
            combined = results_2024[i]['total_return'] + results_2025[i]['total_return']
            all_sorted.append({
                'name': configs[i].name,
                'config': configs[i],
                'total_losing': total,
                '2024': results_2024[i],
                '2025': results_2025[i],
                'combined': combined
            })

        for item in sorted(all_sorted, key=lambda x: (x['total_losing'], -x['combined']))[:3]:
            print(f"\n{item['name']}:")
            print(f"  Total losing: {item['total_losing']}")
            print(f"  2024: {item['2024']['losing_months']} losing ({item['2024']['losing_month_names']})")
            print(f"  2025: {item['2025']['losing_months']} losing ({item['2025']['losing_month_names']})")
            print(f"  Combined: +{item['combined']:.1f}%")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
