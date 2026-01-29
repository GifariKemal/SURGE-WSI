"""Backtest 13 Months (Jan 2025 - Jan 2026)
==========================================

Compare:
1. Kill Zone ON (traditional)
2. Intelligent Filter (velocity-based)

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
from datetime import datetime, timedelta, timezone
from loguru import logger
from dataclasses import dataclass
from typing import List, Dict

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


@dataclass
class TestConfig:
    name: str
    use_killzone: bool
    use_intelligent: bool
    intelligent_threshold: float = 50.0
    use_trend_filter: bool = True
    use_choppiness_filter: bool = False


CONFIGS = [
    TestConfig("KZ_ON", use_killzone=True, use_intelligent=False),
    TestConfig("KZ_OFF", use_killzone=False, use_intelligent=False),
    TestConfig("INTEL_40", use_killzone=False, use_intelligent=True, intelligent_threshold=40.0),
    TestConfig("INTEL_50", use_killzone=False, use_intelligent=True, intelligent_threshold=50.0),
    TestConfig("INTEL_60", use_killzone=False, use_intelligent=True, intelligent_threshold=60.0),
]


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime):
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=200000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def run_backtest_month(htf_df, ltf_df, year, month, cfg: TestConfig, initial_balance=10000.0):
    """Run backtest for a single month"""
    # Date range
    start_date = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)

    # Warmup period
    warmup = start_date - timedelta(days=30)

    # Filter data
    htf_m = htf_df[(htf_df.index >= warmup) & (htf_df.index <= end_date)]
    ltf_m = ltf_df[(ltf_df.index >= warmup) & (ltf_df.index <= end_date)]

    if htf_m.empty or ltf_m.empty or len(ltf_m) < 100:
        return {'month': month, 'year': year, 'pnl': 0, 'trades': 0, 'wins': 0, 'skipped': True}

    htf = htf_m.reset_index().rename(columns={'index': 'time'})
    ltf = ltf_m.reset_index().rename(columns={'index': 'time'})

    # Create backtester
    bt = Backtester(
        symbol='GBPUSD',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_balance=initial_balance,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=cfg.use_killzone,
        use_trend_filter=cfg.use_trend_filter,
        use_relaxed_filter=False,
        use_hybrid_mode=False,
        use_choppiness_filter=cfg.use_choppiness_filter,
        use_intelligent_filter=cfg.use_intelligent,
        intelligent_threshold=cfg.intelligent_threshold
    )

    bt.load_data(htf, ltf)
    result = bt.run()

    # Calculate stats
    month_pnl = result.net_profit
    month_trades = result.total_trades
    month_wins = result.winning_trades

    return {
        'month': month,
        'year': year,
        'pnl': month_pnl,
        'trades': month_trades,
        'wins': month_wins,
        'win_rate': result.win_rate,
        'skipped': False,
        'final_balance': result.final_balance
    }


def run_13month_backtest(htf_df, ltf_df, cfg: TestConfig):
    """Run 13-month backtest from Jan 2025 to Jan 2026"""
    monthly_results = []
    running_balance = 10000.0

    # Months: Jan 2025 to Jan 2026 (13 months)
    months = [
        (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2025, 5), (2025, 6),
        (2025, 7), (2025, 8), (2025, 9), (2025, 10), (2025, 11), (2025, 12),
        (2026, 1)
    ]

    for year, month in months:
        result = run_backtest_month(htf_df, ltf_df, year, month, cfg, running_balance)
        monthly_results.append(result)

        if not result.get('skipped'):
            running_balance += result['pnl']

    # Calculate summary
    total_trades = sum(m['trades'] for m in monthly_results if not m.get('skipped'))
    total_wins = sum(m['wins'] for m in monthly_results if not m.get('skipped'))
    total_pnl = sum(m['pnl'] for m in monthly_results if not m.get('skipped'))
    losing_months = [m for m in monthly_results if m['pnl'] < 0 and not m.get('skipped')]

    return {
        'config': cfg.name,
        'monthly': monthly_results,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_pnl': total_pnl,
        'total_return': total_pnl / 100,  # Percentage
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'losing_months': len(losing_months),
        'final_balance': running_balance,
        'trades_per_month': total_trades / 13,
    }


async def main():
    print("\n" + "="*70)
    print("BACKTEST 13 MONTHS (JAN 2025 - JAN 2026)")
    print("Compare: Kill Zone vs Intelligent Filter")
    print("="*70)

    print("\nFetching data...")
    # Need warmup data from Nov 2024
    start = datetime(2024, 11, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data('GBPUSD', 'H4', start, end)
    ltf_df = await fetch_data('GBPUSD', 'M15', start, end)

    print(f"H4={len(htf_df)}, M15={len(ltf_df)}")

    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing: {cfg.name}")
        if cfg.use_intelligent:
            print(f"  Intelligent Threshold: {cfg.intelligent_threshold}")
        print("="*70)

        result = run_13month_backtest(htf_df, ltf_df, cfg)
        all_results.append(result)

        # Print monthly details
        print(f"\n{'Month':<10} | {'Trades':>6} | {'Wins':>5} | {'WR':>5} | {'P/L':>10}")
        print("-" * 50)

        for m in result['monthly']:
            if m.get('skipped'):
                print(f"{m['year']}-{m['month']:02d}    | {'SKIP':>6} | {'':>5} | {'':>5} | {'':>10}")
            else:
                status = "+" if m['pnl'] >= 0 else ""
                wr = m['wins'] / m['trades'] * 100 if m['trades'] > 0 else 0
                print(f"{m['year']}-{m['month']:02d}    | {m['trades']:>6} | {m['wins']:>5} | {wr:>4.0f}% | {status}${m['pnl']:>8.0f}")

        print("-" * 50)
        print(f"{'TOTAL':<10} | {result['total_trades']:>6} | {result['total_wins']:>5} | "
              f"{result['win_rate']:>4.0f}% | +${result['total_pnl']:>7.0f}")
        print(f"\nLosing months: {result['losing_months']}")
        print(f"Return: +{result['total_return']:.1f}%")
        print(f"Trades/month: {result['trades_per_month']:.1f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - 13 MONTHS (JAN 2025 - JAN 2026)")
    print("="*70)
    print(f"\n{'Config':<12} | {'Trades':>6} | {'/Mon':>5} | {'WR':>5} | {'Lose':>4} | {'Return':>10}")
    print("-" * 60)

    for r in all_results:
        print(f"{r['config']:<12} | {r['total_trades']:>6} | {r['trades_per_month']:>5.1f} | "
              f"{r['win_rate']:>4.0f}% | {r['losing_months']:>4} | +{r['total_return']:>8.1f}%")

    # Find best config
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Best trades
    best_trades = max(all_results, key=lambda x: x['total_trades'])
    print(f"\nMost Trades: {best_trades['config']}")
    print(f"  {best_trades['total_trades']} trades ({best_trades['trades_per_month']:.1f}/month)")

    # Best safety
    best_safe = min(all_results, key=lambda x: x['losing_months'])
    print(f"\nFewest Losing Months: {best_safe['config']}")
    print(f"  {best_safe['losing_months']} losing months")

    # Best balance (trades * return / (losing + 1))
    for r in all_results:
        r['score'] = (r['total_trades'] * max(0, r['total_return'])) / (r['losing_months'] + 1)

    best_balanced = max(all_results, key=lambda x: x['score'])
    print(f"\nBest Balanced: {best_balanced['config']}")
    print(f"  {best_balanced['total_trades']} trades, {best_balanced['losing_months']} losing, +{best_balanced['total_return']:.1f}%")

    # Compare KZ_ON vs best intelligent
    kz_on = next((r for r in all_results if r['config'] == "KZ_ON"), None)
    intel_results = [r for r in all_results if "INTEL" in r['config']]

    if kz_on and intel_results:
        # Best intel with <= 3 losing months
        good_intel = [r for r in intel_results if r['losing_months'] <= 3]
        if good_intel:
            best_intel = max(good_intel, key=lambda x: x['total_trades'])
        else:
            best_intel = min(intel_results, key=lambda x: x['losing_months'])

        print(f"\n" + "="*70)
        print("KZ_ON vs INTELLIGENT COMPARISON")
        print("="*70)
        print(f"\n{'Metric':<20} | {'KZ_ON':>12} | {best_intel['config']:>12}")
        print("-" * 50)
        print(f"{'Trades':<20} | {kz_on['total_trades']:>12} | {best_intel['total_trades']:>12}")
        print(f"{'Trades/Month':<20} | {kz_on['trades_per_month']:>12.1f} | {best_intel['trades_per_month']:>12.1f}")
        print(f"{'Win Rate':<20} | {kz_on['win_rate']:>11.0f}% | {best_intel['win_rate']:>11.0f}%")
        print(f"{'Losing Months':<20} | {kz_on['losing_months']:>12} | {best_intel['losing_months']:>12}")
        print(f"{'Return':<20} | {kz_on['total_return']:>+11.1f}% | {best_intel['total_return']:>+11.1f}%")

        if kz_on['total_trades'] > 0:
            diff = best_intel['total_trades'] - kz_on['total_trades']
            pct = diff / kz_on['total_trades'] * 100
            print(f"\nIntelligent vs KZ_ON: {diff:+d} trades ({pct:+.0f}%)")

    # Send to Telegram
    if TELEGRAM_AVAILABLE:
        print("\n" + "="*70)
        print("Sending to Telegram...")

        msg = "🧠 <b>INTELLIGENT FILTER - 13 MONTHS</b>\n"
        msg += "<i>Jan 2025 - Jan 2026</i>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        msg += "📊 <b>Results:</b>\n"
        msg += "<pre>\n"
        msg += f"{'Config':<12}|Trds|/Mo |WR% |Los|Return\n"
        msg += "-" * 42 + "\n"

        for r in all_results:
            emoji = "✓" if r['losing_months'] <= 2 else "△" if r['losing_months'] <= 4 else "✗"
            msg += f"{r['config']:<12}|{r['total_trades']:>4}|{r['trades_per_month']:>4.1f}|{r['win_rate']:>3.0f}%| {r['losing_months']}{emoji}|{r['total_return']:>+5.0f}%\n"

        msg += "</pre>\n"

        # Best recommendation
        msg += f"\n🎯 <b>Recommended: {best_balanced['config']}</b>\n"
        msg += f"├ Trades: {best_balanced['total_trades']} ({best_balanced['trades_per_month']:.1f}/mo)\n"
        msg += f"├ Win Rate: {best_balanced['win_rate']:.0f}%\n"
        msg += f"├ Losing: {best_balanced['losing_months']} months\n"
        msg += f"└ Return: +{best_balanced['total_return']:.1f}%\n"

        # Monthly breakdown for best
        msg += f"\n📅 <b>Monthly ({best_balanced['config']}):</b>\n"
        msg += "<pre>\n"
        for m in next(r for r in all_results if r['config'] == best_balanced['config'])['monthly']:
            if not m.get('skipped'):
                status = "+" if m['pnl'] >= 0 else "-"
                msg += f"{m['year']}-{m['month']:02d}: {status}${abs(m['pnl']):>4.0f} ({m['trades']}t)\n"
        msg += "</pre>\n"

        msg += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</i>"

        try:
            bot = Bot(token=config.telegram.bot_token)
            await bot.send_message(chat_id=config.telegram.chat_id, text=msg, parse_mode='HTML')
            print("Sent!")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
