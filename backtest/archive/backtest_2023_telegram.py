"""Backtest 2023 with ZERO LOSING MONTHS Config
===============================================

Test ZERO_LOSS_CONFIG on 2023 data and send report to Telegram.

Config: SL10 + Quality75 + Loss0.1%

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

# Telegram
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("WARNING: python-telegram-bot not installed")


# =============================================================================
# ZERO LOSING MONTHS CONFIG
# =============================================================================
ZERO_LOSS_CONFIG = {
    'max_sl_pips': 10.0,
    'max_loss_pct': 0.1,
    'min_quality': 75.0,
}


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data from database"""
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


async def send_telegram(message: str):
    """Send message to Telegram"""
    if not TELEGRAM_AVAILABLE:
        print("Telegram not available")
        return False

    try:
        token = config.telegram.bot_token
        chat_id = config.telegram.chat_id

        if not token or not chat_id:
            print("Telegram token/chat_id not configured")
            return False

        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
        print("Telegram message sent!")
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def run_backtest_year(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, year: int) -> dict:
    """Run backtest for a specific year with ZERO_LOSS_CONFIG"""

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

        # Skip December
        if is_december:
            monthly_results.append({
                'month': month_name, 'pnl': 0, 'trades': 0,
                'wins': 0, 'balance': running_balance, 'skipped': True
            })
            continue

        warmup_days = 30
        month_start_warmup = start_date - timedelta(days=warmup_days)

        htf_month = htf_df[(htf_df.index >= month_start_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_warmup) & (ltf_df.index <= end_date)]

        if htf_month.empty or ltf_month.empty:
            monthly_results.append({
                'month': month_name, 'pnl': 0, 'trades': 0,
                'wins': 0, 'balance': running_balance, 'skipped': True
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

        # Apply ZERO_LOSS_CONFIG
        bt.risk_manager.max_lot_size = 0.5
        bt.risk_manager.max_sl_pips = ZERO_LOSS_CONFIG['max_sl_pips']
        bt.entry_trigger.min_quality_score = ZERO_LOSS_CONFIG['min_quality']

        bt.load_data(htf, ltf)
        result = bt.run()

        month_pnl = 0
        month_trades = 0
        month_wins = 0
        month_filtered = 0
        simulated_balance = running_balance

        for trade in result.trade_list:
            # Calculate risk in pips
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001

            # Filter: Max SL
            if risk_pips > ZERO_LOSS_CONFIG['max_sl_pips']:
                month_filtered += 1
                continue

            # Apply loss cap
            max_loss_dollars = simulated_balance * ZERO_LOSS_CONFIG['max_loss_pct'] / 100
            if trade.pnl < 0 and abs(trade.pnl) > max_loss_dollars:
                adjusted_pnl = -max_loss_dollars
            else:
                adjusted_pnl = trade.pnl

            simulated_balance += adjusted_pnl
            month_pnl += adjusted_pnl
            month_trades += 1

            if adjusted_pnl > 0:
                month_wins += 1

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

    losing_months = [m for m in monthly_results if m['pnl'] < 0 and not m.get('skipped')]
    winning_months = [m for m in monthly_results if m['pnl'] > 0 and not m.get('skipped')]

    return {
        'year': year,
        'final_balance': running_balance,
        'total_return': (running_balance - 10000) / 10000 * 100,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'monthly': monthly_results,
        'losing_months': len(losing_months),
        'winning_months': len(winning_months),
        'losing_month_names': [m['month'] for m in losing_months],
        'total_filtered': total_filtered,
    }


def format_telegram_report(result: dict, comparison_2024: dict = None, comparison_2025: dict = None) -> str:
    """Format backtest result as Telegram message"""

    year = result['year']
    losing = result['losing_months']
    winning = result['winning_months']
    total_return = result['total_return']
    trades = result['total_trades']
    win_rate = result['win_rate']

    # Status emoji
    if losing == 0:
        status_emoji = "🏆"
        status_text = "ZERO LOSING MONTHS!"
    elif losing == 1:
        status_emoji = "🟡"
        status_text = f"{losing} losing month"
    else:
        status_emoji = "🔴"
        status_text = f"{losing} losing months"

    msg = f"📊 <b>BACKTEST REPORT {year}</b>\n"
    msg += f"<i>ZERO LOSING MONTHS CONFIG</i>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # Config
    msg += "⚙️ <b>Configuration:</b>\n"
    msg += f"├ Max SL: {ZERO_LOSS_CONFIG['max_sl_pips']} pips\n"
    msg += f"├ Min Quality: {ZERO_LOSS_CONFIG['min_quality']}\n"
    msg += f"└ Max Loss/Trade: {ZERO_LOSS_CONFIG['max_loss_pct']}%\n\n"

    # Results
    msg += f"{status_emoji} <b>Result: {status_text}</b>\n\n"

    msg += "📈 <b>Performance:</b>\n"
    msg += f"├ Return: <b>{total_return:+.1f}%</b>\n"
    msg += f"├ Final Balance: ${result['final_balance']:,.0f}\n"
    msg += f"├ Total Trades: {trades}\n"
    msg += f"├ Win Rate: {win_rate:.1f}%\n"
    msg += f"├ Winning Months: {winning}\n"
    msg += f"└ Losing Months: {losing}\n\n"

    # Monthly breakdown
    msg += "📅 <b>Monthly P/L:</b>\n"
    msg += "<pre>"
    for m in result['monthly']:
        if m.get('skipped'):
            msg += f"{m['month']}: SKIP\n"
        else:
            emoji = "✅" if m['pnl'] >= 0 else "❌"
            msg += f"{m['month']}: {emoji} ${m['pnl']:+,.0f} ({m['trades']}t)\n"
    msg += "</pre>\n"

    # Comparison if available
    if comparison_2024 or comparison_2025:
        msg += "📊 <b>Year Comparison:</b>\n"
        msg += "<pre>"
        msg += f"Year  | Lose | Return\n"
        msg += f"------+------+--------\n"
        msg += f"{year}  |  {losing}   | {total_return:+.1f}%\n"
        if comparison_2024:
            msg += f"2024  |  {comparison_2024['losing_months']}   | {comparison_2024['total_return']:+.1f}%\n"
        if comparison_2025:
            msg += f"2025  |  {comparison_2025['losing_months']}   | {comparison_2025['total_return']:+.1f}%\n"
        msg += "</pre>\n"

    # Losing months detail
    if result['losing_month_names']:
        msg += f"\n⚠️ <b>Losing Months:</b> {', '.join(result['losing_month_names'])}\n"

    msg += "\n━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"<i>Tested: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"

    return msg


async def main():
    print("\n" + "=" * 60)
    print("BACKTEST 2023 - ZERO LOSING MONTHS CONFIG")
    print("=" * 60)

    print("\nFetching data...")
    start = datetime(2022, 11, 1, tzinfo=timezone.utc)  # Warmup for Jan 2023
    end = datetime(2023, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available for 2023")
        return

    print(f"Data loaded: H4={len(htf_df)}, M15={len(ltf_df)}")

    # Run 2023 backtest
    print("\nRunning 2023 backtest...")
    result_2023 = run_backtest_year(htf_df, ltf_df, 2023)

    # Print to console
    print("\n" + "=" * 60)
    print(f"2023 RESULTS")
    print("=" * 60)
    print(f"Losing Months: {result_2023['losing_months']}")
    print(f"Return: {result_2023['total_return']:+.1f}%")
    print(f"Trades: {result_2023['total_trades']}")
    print(f"Win Rate: {result_2023['win_rate']:.1f}%")
    print(f"Filtered: {result_2023['total_filtered']}")

    print("\nMonthly:")
    for m in result_2023['monthly']:
        if not m.get('skipped'):
            status = "✓" if m['pnl'] >= 0 else "✗"
            print(f"  {m['month']}: {status} ${m['pnl']:+,.0f} ({m['trades']} trades)")

    # Also run 2024 and 2025 for comparison
    print("\nRunning 2024 backtest for comparison...")
    start_24 = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end_24 = datetime(2024, 12, 31, tzinfo=timezone.utc)
    htf_24 = await fetch_data("GBPUSD", "H4", start_24, end_24)
    ltf_24 = await fetch_data("GBPUSD", "M15", start_24, end_24)
    result_2024 = run_backtest_year(htf_24, ltf_24, 2024) if not htf_24.empty else None

    print("Running 2025 backtest for comparison...")
    start_25 = datetime(2024, 11, 1, tzinfo=timezone.utc)
    end_25 = datetime(2025, 12, 31, tzinfo=timezone.utc)
    htf_25 = await fetch_data("GBPUSD", "H4", start_25, end_25)
    ltf_25 = await fetch_data("GBPUSD", "M15", start_25, end_25)
    result_2025 = run_backtest_year(htf_25, ltf_25, 2025) if not htf_25.empty else None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - ZERO LOSING MONTHS CONFIG")
    print("=" * 60)
    print(f"{'Year':<6} | {'Lose':>4} | {'Return':>8} | {'Trades':>6}")
    print("-" * 35)
    print(f"{'2023':<6} | {result_2023['losing_months']:>4} | {result_2023['total_return']:>+7.1f}% | {result_2023['total_trades']:>6}")
    if result_2024:
        print(f"{'2024':<6} | {result_2024['losing_months']:>4} | {result_2024['total_return']:>+7.1f}% | {result_2024['total_trades']:>6}")
    if result_2025:
        print(f"{'2025':<6} | {result_2025['losing_months']:>4} | {result_2025['total_return']:>+7.1f}% | {result_2025['total_trades']:>6}")

    total_losing = result_2023['losing_months']
    total_return = result_2023['total_return']
    if result_2024:
        total_losing += result_2024['losing_months']
        total_return += result_2024['total_return']
    if result_2025:
        total_losing += result_2025['losing_months']
        total_return += result_2025['total_return']

    print("-" * 35)
    print(f"{'TOTAL':<6} | {total_losing:>4} | {total_return:>+7.1f}%")

    # Send to Telegram
    print("\n" + "=" * 60)
    print("Sending report to Telegram...")

    telegram_msg = format_telegram_report(result_2023, result_2024, result_2025)
    success = await send_telegram(telegram_msg)

    if success:
        print("Report sent to Telegram!")
    else:
        print("Failed to send to Telegram (check config)")
        print("\nTelegram message content:")
        print("-" * 40)
        # Strip HTML tags for console
        import re
        clean_msg = re.sub('<[^<]+?>', '', telegram_msg)
        print(clean_msg)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
