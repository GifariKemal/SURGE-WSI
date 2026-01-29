"""Run Gold 13-Month Backtest
============================

Run backtest for XAUUSD from Jan 2025 to Jan 2026
and send detailed report to Telegram.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from config import config
from src.data.db_handler import DBHandler
from gold.config.gold_settings import get_gold_config, print_config
from gold.backtest.gold_backtester import GoldBacktester, GoldBacktestResult, print_result

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


# Backtest period
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 31, tzinfo=timezone.utc)
WARMUP_DAYS = 60


async def load_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load data from database"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=500000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def run_monthly_backtest(
    backtester: GoldBacktester,
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    year: int,
    month: int
) -> Dict:
    """Run backtest for a single month"""

    # Define month range
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)

    # Add warmup period
    warmup_start = start - timedelta(days=30)

    # Filter data
    htf_m = htf_df[(htf_df.index >= warmup_start) & (htf_df.index <= end)].copy()
    ltf_m = ltf_df[(ltf_df.index >= warmup_start) & (ltf_df.index <= end)].copy()

    if htf_m.empty or ltf_m.empty or len(htf_m) < 50:
        return {
            'year': year,
            'month': month,
            'skipped': True,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'pnl': 0,
            'pnl_pips': 0
        }

    # Run backtest
    result = backtester.run(htf_m, ltf_m, start_date=start, end_date=end)

    return {
        'year': year,
        'month': month,
        'skipped': False,
        'total_trades': result.total_trades,
        'wins': result.wins,
        'losses': result.losses,
        'win_rate': result.win_rate,
        'pnl': result.total_pnl,
        'pnl_pips': result.total_pnl_pips,
        'max_dd': result.max_drawdown,
        'profit_factor': result.profit_factor,
        'trades': result.trades
    }


async def send_telegram_report(results: List[Dict], gold_config, htf_count: int, ltf_count: int):
    """Send detailed report to Telegram"""
    if not TELEGRAM_AVAILABLE:
        print("Telegram not available")
        return

    bot_token = config.telegram.bot_token
    chat_id = config.telegram.chat_id

    if not bot_token or not chat_id:
        print("Telegram not configured")
        return

    bot = Bot(token=bot_token)

    # Calculate totals
    valid_results = [r for r in results if not r.get('skipped')]
    total_trades = sum(r['total_trades'] for r in valid_results)
    total_wins = sum(r['wins'] for r in valid_results)
    total_pnl = sum(r['pnl'] for r in valid_results)
    total_pnl_pips = sum(r['pnl_pips'] for r in valid_results)

    monthly_pnls = [r['pnl'] for r in valid_results]
    losing_months = sum(1 for p in monthly_pnls if p < 0)
    profitable_months = sum(1 for p in monthly_pnls if p > 0)

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_monthly_pnl = np.mean(monthly_pnls) if monthly_pnls else 0
    best_month = max(monthly_pnls) if monthly_pnls else 0
    worst_month = min(monthly_pnls) if monthly_pnls else 0

    # Build message
    msg = f"""<b>📊 XAUUSD (GOLD) BACKTEST REPORT</b>
<b>Period: Jan 2025 - Jan 2026 (13 months)</b>

━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>📈 DATA & CONFIG</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
  Symbol: XAUUSD (Gold)
  HTF Bars: {htf_count:,}
  LTF Bars: {ltf_count:,}
  Pip Size: {gold_config.symbol.pip_size}
  Spread: {gold_config.symbol.spread_pips} pips
  Activity Threshold: {gold_config.intel_filter.activity_threshold}
  Min Velocity: {gold_config.intel_filter.min_velocity_pips} pips
  Max SL: {gold_config.risk.max_sl_pips} pips

━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>💰 OVERALL PERFORMANCE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Trades: {total_trades}
  Win Rate: {win_rate:.1f}%
  Total P&L: ${total_pnl:,.2f}
  Total Pips: {total_pnl_pips:,.1f}
  Profitable Months: {profitable_months}/13
  Losing Months: {losing_months}/13

━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>📅 MONTHLY BREAKDOWN</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

    for i, r in enumerate(results):
        year = r['year']
        month_name = months[i] if i < 12 else 'Jan'

        if r.get('skipped'):
            msg += f"\n  ⚪ {month_name} {year}: <i>No data</i>"
        else:
            pnl = r['pnl']
            trades = r['total_trades']
            wr = r['win_rate']
            emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            msg += f"\n  {emoji} {month_name} {year}: ${pnl:+,.2f} ({trades}T, {wr:.0f}%WR)"

    msg += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>📊 STATISTICS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
  Avg Monthly P&L: ${avg_monthly_pnl:,.2f}
  Best Month: ${best_month:,.2f}
  Worst Month: ${worst_month:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>🔧 GOLD-SPECIFIC SETTINGS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
  Based on analysis of 2 years data:
  • ATR: ~187 pips (5.8x GBPUSD)
  • Velocity: ~93 pips (5.7x GBPUSD)
  • Best Session: New York (12:00-20:00 UTC)

━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>
<i>Config: Gold-Optimized INTEL_50</i>
"""

    try:
        await bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML')
        print("Report sent to Telegram!")
    except Exception as e:
        print(f"Failed to send Telegram: {e}")


async def main():
    print("=" * 70)
    print("XAUUSD (GOLD) 13-MONTH BACKTEST")
    print("=" * 70)
    print()

    # Print configuration
    gold_config = get_gold_config()
    print_config()
    print()

    # Load data
    print("[1/3] Loading data from database...")
    fetch_start = START_DATE - timedelta(days=WARMUP_DAYS)

    htf_df = await load_data("XAUUSD", "H4", fetch_start, END_DATE)
    ltf_df = await load_data("XAUUSD", "M15", fetch_start, END_DATE)

    if htf_df is None or htf_df.empty:
        print("ERROR: No H4 data found. Run gold/analysis/fetch_and_analyze.py first.")
        return

    if ltf_df is None or ltf_df.empty:
        print("ERROR: No M15 data found. Run gold/analysis/fetch_and_analyze.py first.")
        return

    print(f"  H4: {len(htf_df)} bars ({htf_df.index.min()} to {htf_df.index.max()})")
    print(f"  M15: {len(ltf_df)} bars")
    print()

    # Run backtest
    print("[2/3] Running monthly backtests...")
    print()

    results = []
    backtester = GoldBacktester(config=gold_config, initial_balance=10000.0)

    # Jan 2025 - Dec 2025
    for month in range(1, 13):
        result = run_monthly_backtest(backtester, htf_df, ltf_df, 2025, month)
        results.append(result)

        if not result.get('skipped'):
            pnl = result['pnl']
            trades = result['total_trades']
            emoji = "+" if pnl > 0 else ""
            print(f"  2025-{month:02d}: {emoji}${pnl:.2f} ({trades} trades)")
        else:
            print(f"  2025-{month:02d}: SKIPPED (insufficient data)")

    # Jan 2026
    result = run_monthly_backtest(backtester, htf_df, ltf_df, 2026, 1)
    results.append(result)
    if not result.get('skipped'):
        pnl = result['pnl']
        trades = result['total_trades']
        emoji = "+" if pnl > 0 else ""
        print(f"  2026-01: {emoji}${pnl:.2f} ({trades} trades)")
    else:
        print(f"  2026-01: SKIPPED (insufficient data)")

    print()

    # Summary
    valid_results = [r for r in results if not r.get('skipped')]
    total_pnl = sum(r['pnl'] for r in valid_results)
    total_trades = sum(r['total_trades'] for r in valid_results)
    total_pnl_pips = sum(r['pnl_pips'] for r in valid_results)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total P&L: ${total_pnl:,.2f} ({total_pnl_pips:,.1f} pips)")
    print(f"  Total Trades: {total_trades}")
    print(f"  Losing Months: {sum(1 for r in valid_results if r['pnl'] < 0)}/13")
    print()

    # Send to Telegram
    print("[3/3] Sending report to Telegram...")
    await send_telegram_report(results, gold_config, len(htf_df), len(ltf_df))

    print()
    print("DONE!")


if __name__ == "__main__":
    asyncio.run(main())
