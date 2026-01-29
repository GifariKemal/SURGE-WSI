"""Test Balanced Configurations
===============================

Compare ORIGINAL, ZERO_LOSS, and BALANCED configs.

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

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester

# Telegram
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


@dataclass
class TestConfig:
    name: str
    max_sl_pips: float
    min_quality: float
    max_loss_pct: float


# Configs to test
CONFIGS = [
    TestConfig("ORIGINAL", 50.0, 65.0, 0.8),
    TestConfig("BALANCED_CONS", 25.0, 68.0, 0.3),
    TestConfig("BALANCED_MOD", 30.0, 65.0, 0.4),
    TestConfig("BALANCED_ACT", 40.0, 60.0, 0.5),
    TestConfig("ZERO_LOSS", 10.0, 75.0, 0.1),
]


async def fetch_data(symbol, timeframe, start, end):
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


def run_backtest_year(htf_df, ltf_df, year, cfg: TestConfig):
    running_balance = 10000.0
    monthly_results = []
    total_trades = 0
    total_wins = 0

    for month in range(1, 12):  # Skip December
        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 11:
            end_date = datetime(year, 12, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)

        warmup = start_date - timedelta(days=30)
        htf_m = htf_df[(htf_df.index >= warmup) & (htf_df.index <= end_date)]
        ltf_m = ltf_df[(ltf_df.index >= warmup) & (ltf_df.index <= end_date)]

        if htf_m.empty or ltf_m.empty or len(ltf_m) < 100:
            monthly_results.append({'month': month, 'pnl': 0, 'trades': 0, 'skipped': True})
            continue

        htf = htf_m.reset_index().rename(columns={'index': 'time'})
        ltf = ltf_m.reset_index().rename(columns={'index': 'time'})

        bt = Backtester(
            symbol='GBPUSD',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_balance=running_balance,
            pip_value=10.0, spread_pips=1.5,
            use_killzone=True, use_trend_filter=True,
            use_relaxed_filter=False, use_hybrid_mode=False,
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

        for trade in result.trade_list:
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001

            if risk_pips > cfg.max_sl_pips:
                continue

            max_loss = running_balance * cfg.max_loss_pct / 100
            if trade.pnl < 0 and abs(trade.pnl) > max_loss:
                adj_pnl = -max_loss
            else:
                adj_pnl = trade.pnl

            running_balance += adj_pnl
            month_pnl += adj_pnl
            month_trades += 1

            if adj_pnl > 0:
                month_wins += 1

        total_trades += month_trades
        total_wins += month_wins

        monthly_results.append({
            'month': month,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
            'skipped': False
        })

    losing = [m for m in monthly_results if m['pnl'] < 0 and not m.get('skipped')]
    return {
        'year': year,
        'config': cfg.name,
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'monthly': monthly_results,
        'losing_months': len(losing),
        'losing_names': [f"M{m['month']:02d}" for m in losing],
    }


async def main():
    print("\n" + "="*70)
    print("TEST BALANCED CONFIGURATIONS")
    print("="*70)

    print("\nFetching data...")
    start = datetime(2021, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data('GBPUSD', 'H4', start, end)
    ltf_df = await fetch_data('GBPUSD', 'M15', start, end)

    print(f"H4={len(htf_df)}, M15={len(ltf_df)}")

    # Results storage
    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing: {cfg.name} (SL={cfg.max_sl_pips}, Q={cfg.min_quality}, L={cfg.max_loss_pct}%)")
        print("="*70)

        years_results = []
        for year in [2024, 2025]:
            r = run_backtest_year(htf_df, ltf_df, year, cfg)
            years_results.append(r)
            print(f"  {year}: {r['losing_months']} losing, {r['total_trades']} trades, +{r['total_return']:.1f}%")

        total_losing = sum(r['losing_months'] for r in years_results)
        total_trades = sum(r['total_trades'] for r in years_results)
        total_return = sum(r['total_return'] for r in years_results)

        all_results.append({
            'config': cfg,
            'years': years_results,
            'total_losing': total_losing,
            'total_trades': total_trades,
            'total_return': total_return,
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - 2024 + 2025 Combined")
    print("="*70)
    print(f"\n{'Config':<20} | {'SL':>4} | {'Q':>4} | {'L%':>4} | {'Trades':>6} | {'Lose':>4} | {'Return':>8}")
    print("-" * 70)

    for r in all_results:
        cfg = r['config']
        print(f"{cfg.name:<20} | {cfg.max_sl_pips:>4.0f} | {cfg.min_quality:>4.0f} | {cfg.max_loss_pct:>4.1f} | "
              f"{r['total_trades']:>6} | {r['total_losing']:>4} | {r['total_return']:>+7.1f}%")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Find best balanced (most trades with <=2 losing months)
    balanced_options = [r for r in all_results if r['total_losing'] <= 2]
    if balanced_options:
        best = max(balanced_options, key=lambda x: x['total_trades'])
        print(f"\nBest Balanced (≤2 losing, max trades):")
        print(f"  Config: {best['config'].name}")
        print(f"  Settings: SL={best['config'].max_sl_pips}, Quality={best['config'].min_quality}, Loss={best['config'].max_loss_pct}%")
        print(f"  Results: {best['total_trades']} trades, {best['total_losing']} losing, +{best['total_return']:.1f}%")

    # Send to Telegram
    if TELEGRAM_AVAILABLE:
        print("\nSending to Telegram...")
        msg = "📊 <b>CONFIG COMPARISON</b>\n"
        msg += "<i>2024 + 2025 Combined</i>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        msg += "<pre>\n"
        msg += f"{'Config':<15}|Trds|Lose|Return\n"
        msg += "-" * 35 + "\n"
        for r in all_results:
            emoji = "✓" if r['total_losing'] <= 1 else ("△" if r['total_losing'] <= 2 else "✗")
            msg += f"{r['config'].name:<15}|{r['total_trades']:>4}| {r['total_losing']}{emoji} |{r['total_return']:>+6.1f}%\n"
        msg += "</pre>\n"

        if balanced_options:
            best = max(balanced_options, key=lambda x: x['total_trades'])
            msg += f"\n🎯 <b>Recommended:</b> {best['config'].name}\n"
            msg += f"SL={best['config'].max_sl_pips} Q={best['config'].min_quality} L={best['config'].max_loss_pct}%"

        try:
            bot = Bot(token=config.telegram.bot_token)
            await bot.send_message(chat_id=config.telegram.chat_id, text=msg, parse_mode='HTML')
            print("Sent!")
        except Exception as e:
            print(f"Telegram error: {e}")

    print("\n" + "="*70)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
