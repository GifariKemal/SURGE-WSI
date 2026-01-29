"""Full Backtest Report - ZERO LOSING MONTHS CONFIG
===================================================

Comprehensive backtest for available data (2024-2025)
Send detailed report to Telegram.

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
    all_trades = []

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

        if htf_month.empty or ltf_month.empty or len(ltf_month) < 100:
            monthly_results.append({
                'month': month_name, 'pnl': 0, 'trades': 0,
                'wins': 0, 'balance': running_balance, 'skipped': True, 'reason': 'no_data'
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

            all_trades.append({
                'month': month_name,
                'direction': trade.direction,
                'pnl': adjusted_pnl,
                'risk_pips': risk_pips
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
            'balance': running_balance,
            'skipped': False
        })

    losing_months = [m for m in monthly_results if m['pnl'] < 0 and not m.get('skipped')]
    winning_months = [m for m in monthly_results if m['pnl'] > 0 and not m.get('skipped')]
    break_even = [m for m in monthly_results if m['pnl'] == 0 and not m.get('skipped') and m['trades'] > 0]

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
        'break_even_months': len(break_even),
        'losing_month_names': [m['month'] for m in losing_months],
        'total_filtered': total_filtered,
        'all_trades': all_trades,
    }


def format_full_report(results: list) -> str:
    """Format comprehensive Telegram report"""

    msg = "🦅 <b>SURGE-WSI BACKTEST REPORT</b>\n"
    msg += "<i>ZERO LOSING MONTHS Configuration</i>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # Config
    msg += "⚙️ <b>Config:</b>\n"
    msg += f"├ Max SL: <b>{ZERO_LOSS_CONFIG['max_sl_pips']}</b> pips\n"
    msg += f"├ Min Quality: <b>{ZERO_LOSS_CONFIG['min_quality']}</b>\n"
    msg += f"└ Max Loss: <b>{ZERO_LOSS_CONFIG['max_loss_pct']}%</b>/trade\n\n"

    # Summary table
    total_losing = sum(r['losing_months'] for r in results)
    total_return = sum(r['total_return'] for r in results)
    total_trades = sum(r['total_trades'] for r in results)

    if total_losing == 0:
        msg += "🏆 <b>STATUS: ZERO LOSING MONTHS!</b>\n\n"
    else:
        msg += f"⚠️ <b>STATUS: {total_losing} losing month(s)</b>\n\n"

    msg += "📊 <b>Results by Year:</b>\n"
    msg += "<pre>"
    msg += f"Year  |Loss|Return |Trades\n"
    msg += f"------+----+-------+------\n"
    for r in results:
        loss_str = "0 ✓" if r['losing_months'] == 0 else f"{r['losing_months']} ✗"
        msg += f"{r['year']}  | {loss_str:>2} |{r['total_return']:+6.1f}%| {r['total_trades']:>4}\n"
    msg += f"------+----+-------+------\n"
    total_emoji = "✓" if total_losing == 0 else "✗"
    msg += f"TOTAL | {total_losing} {total_emoji}|{total_return:+6.1f}%| {total_trades:>4}\n"
    msg += "</pre>\n"

    # Monthly details for each year
    for r in results:
        msg += f"\n📅 <b>{r['year']} Monthly:</b>\n"
        msg += "<pre>"
        for m in r['monthly']:
            if m.get('skipped'):
                if m.get('reason') == 'no_data':
                    msg += f"{m['month'][:3]}: -- (no data)\n"
                else:
                    msg += f"{m['month'][:3]}: -- (skip Dec)\n"
            else:
                emoji = "+" if m['pnl'] >= 0 else "-"
                wr = m['wins']/m['trades']*100 if m['trades'] > 0 else 0
                msg += f"{m['month'][:3]}: {emoji}${abs(m['pnl']):>5,.0f} {m['trades']:>2}t {wr:>4.0f}%\n"
        msg += "</pre>"

    # Losing months detail
    all_losing = []
    for r in results:
        all_losing.extend(r['losing_month_names'])
    if all_losing:
        msg += f"\n❌ <b>Losing:</b> {', '.join(all_losing)}\n"

    msg += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</i>"

    return msg


async def main():
    print("\n" + "=" * 60)
    print("FULL BACKTEST REPORT - ZERO LOSING MONTHS CONFIG")
    print("=" * 60)

    print("\nFetching all available data...")

    # Data range: Dec 2023 onwards (M15 available from Dec 18, 2023)
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"Data loaded: H4={len(htf_df)}, M15={len(ltf_df)}")
    print(f"M15 range: {ltf_df.index.min()} to {ltf_df.index.max()}")

    # Run backtests
    results = []

    print("\nRunning 2024 backtest...")
    result_2024 = run_backtest_year(htf_df, ltf_df, 2024)
    results.append(result_2024)
    print(f"  -> {result_2024['losing_months']} losing, {result_2024['total_return']:+.1f}%, {result_2024['total_trades']} trades")

    print("Running 2025 backtest...")
    result_2025 = run_backtest_year(htf_df, ltf_df, 2025)
    results.append(result_2025)
    print(f"  -> {result_2025['losing_months']} losing, {result_2025['total_return']:+.1f}%, {result_2025['total_trades']} trades")

    # Print summary
    total_losing = result_2024['losing_months'] + result_2025['losing_months']
    total_return = result_2024['total_return'] + result_2025['total_return']

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Losing Months: {total_losing}")
    print(f"Total Return: {total_return:+.1f}%")
    print(f"Total Trades: {result_2024['total_trades'] + result_2025['total_trades']}")

    if total_losing == 0:
        print("\n🏆 ZERO LOSING MONTHS ACHIEVED!")
    else:
        print(f"\n⚠️ Losing months: {result_2024['losing_month_names'] + result_2025['losing_month_names']}")

    # Format and send Telegram
    print("\n" + "=" * 60)
    print("Sending to Telegram...")

    telegram_msg = format_full_report(results)

    success = await send_telegram(telegram_msg)
    if success:
        print("✓ Report sent to Telegram!")
    else:
        print("✗ Failed to send to Telegram")
        print("\nReport content:")
        print("-" * 40)
        import re
        print(re.sub('<[^<]+?>', '', telegram_msg))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
