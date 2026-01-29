"""Backtest All Years with ZERO_LOSS_CONFIG
==========================================

Run backtest for 2022-2025 and send report to Telegram.

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

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester

# Telegram
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# ZERO_LOSS_CONFIG
MAX_SL_PIPS = 10.0
MAX_LOSS_PCT = 0.1
MIN_QUALITY = 75.0


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


def run_backtest_year(htf_df, ltf_df, year):
    running_balance = 10000.0
    monthly_results = []

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
        bt.risk_manager.max_sl_pips = MAX_SL_PIPS
        bt.entry_trigger.min_quality_score = MIN_QUALITY

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

            if risk_pips > MAX_SL_PIPS:
                continue

            max_loss = running_balance * MAX_LOSS_PCT / 100
            if trade.pnl < 0 and abs(trade.pnl) > max_loss:
                adj_pnl = -max_loss
            else:
                adj_pnl = trade.pnl

            running_balance += adj_pnl
            month_pnl += adj_pnl
            month_trades += 1
            if adj_pnl > 0:
                month_wins += 1

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
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 100,
        'monthly': monthly_results,
        'losing_months': len(losing),
        'losing_names': [f"M{m['month']:02d}" for m in losing],
        'total_trades': sum(m['trades'] for m in monthly_results)
    }


async def main():
    print("\n" + "="*60)
    print("BACKTEST ALL YEARS - ZERO LOSS CONFIG")
    print("="*60)

    print("\nFetching data...")
    start = datetime(2021, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data('GBPUSD', 'H4', start, end)
    ltf_df = await fetch_data('GBPUSD', 'M15', start, end)

    print(f"H4={len(htf_df)}, M15={len(ltf_df)}")

    results = []
    for year in [2022, 2023, 2024, 2025]:
        print(f"\nRunning {year} backtest...")
        r = run_backtest_year(htf_df, ltf_df, year)
        results.append(r)
        print(f"  {year}: {r['losing_months']} losing, +{r['total_return']:.1f}%, {r['total_trades']} trades")
        if r['losing_names']:
            print(f"       Losing: {r['losing_names']}")

        # Print monthly
        for m in r['monthly']:
            if not m.get('skipped'):
                status = "+" if m['pnl'] >= 0 else "-"
                print(f"    M{m['month']:02d}: {status}${abs(m['pnl']):.0f} ({m['trades']}t)")

    # Summary
    total_losing = sum(r['losing_months'] for r in results)
    total_return = sum(r['total_return'] for r in results)
    total_trades = sum(r['total_trades'] for r in results)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Losing Months: {total_losing}")
    print(f"Total Return: +{total_return:.1f}%")
    print(f"Total Trades: {total_trades}")

    # Build Telegram message
    msg = "🦅 <b>BACKTEST REPORT - ZERO LOSS CONFIG</b>\n"
    msg += "<i>SL10 + Quality75 + Loss0.1%</i>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    msg += "📊 <b>Results by Year:</b>\n"
    msg += "<pre>\n"
    msg += "Year |Loss|Return|Trades\n"
    msg += "-----+----+------+------\n"

    for r in results:
        emoji = "✓" if r['losing_months'] == 0 else "✗"
        msg += f"{r['year']} | {r['losing_months']} {emoji}|{r['total_return']:+5.1f}%|  {r['total_trades']:>3}\n"

    msg += "-----+----+------+------\n"
    msg += f"TOTAL| {total_losing}  |{total_return:+5.1f}%|  {total_trades}\n"
    msg += "</pre>\n"

    # Monthly details
    for r in results:
        msg += f"\n<b>{r['year']}:</b> "
        parts = []
        for m in r['monthly']:
            if m.get('skipped'):
                continue
            emoji = "+" if m['pnl'] >= 0 else "-"
            parts.append(f"{emoji}${abs(m['pnl']):.0f}")
        msg += " ".join(parts)

    if total_losing == 0:
        msg += "\n\n🏆 <b>ZERO LOSING MONTHS!</b>"
    else:
        all_losing = []
        for r in results:
            if r['losing_names']:
                all_losing.extend([f"{r['year']}-{n}" for n in r['losing_names']])
        msg += f"\n\n⚠️ Losing: {', '.join(all_losing)}"

    msg += f"\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</i>"

    # Send to Telegram
    print("\n" + "="*60)
    print("Sending to Telegram...")

    if TELEGRAM_AVAILABLE:
        try:
            bot = Bot(token=config.telegram.bot_token)
            await bot.send_message(chat_id=config.telegram.chat_id, text=msg, parse_mode='HTML')
            print("Report sent to Telegram!")
        except Exception as e:
            print(f"Telegram error: {e}")
    else:
        print("Telegram not available")
        print("\nReport content:")
        import re
        print(re.sub('<[^<]+?>', '', msg))

    print("\n" + "="*60)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
