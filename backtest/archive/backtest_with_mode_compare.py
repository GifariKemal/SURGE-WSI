"""Backtest with Trade Mode Comparison
=====================================

Runs 13-month backtest (Jan 2025 - Jan 2026) and compares:
- All trades (as if AUTO mode always)
- Which trades would have been SIGNAL_ONLY mode

Sends results to Telegram.

Author: SURIOTA Team
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester, BacktestResult
from src.trading.trade_mode_manager import TradeMode


async def fetch_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    if not await db.connect():
        logger.error("Failed to connect to database")
        return pd.DataFrame()

    df = await db.get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=100000,
        start_time=start_date,
        end_time=end_date
    )
    await db.disconnect()

    return df


def calculate_signal_only_impact(bt: Backtester, result: BacktestResult) -> dict:
    """Calculate the impact if signal-only trades were skipped

    Returns dict with comparison stats
    """
    all_trades = result.trade_list
    signal_only_times = {t['time'] for t in bt.signal_only_trades}

    # Categorize trades
    auto_trades = [t for t in all_trades if t.entry_time not in signal_only_times]
    signal_trades = [t for t in all_trades if t.entry_time in signal_only_times]

    # Calculate PnL
    auto_pnl = sum(t.pnl for t in auto_trades)
    signal_pnl = sum(t.pnl for t in signal_trades)

    # Win rates
    auto_wins = sum(1 for t in auto_trades if t.pnl > 0)
    signal_wins = sum(1 for t in signal_trades if t.pnl > 0)

    auto_win_rate = (auto_wins / len(auto_trades) * 100) if auto_trades else 0
    signal_win_rate = (signal_wins / len(signal_trades) * 100) if signal_trades else 0

    return {
        'total_trades': len(all_trades),
        'auto_trades': len(auto_trades),
        'signal_trades': len(signal_trades),
        'total_pnl': result.net_profit,
        'auto_pnl': auto_pnl,
        'signal_pnl': signal_pnl,
        'auto_win_rate': auto_win_rate,
        'signal_win_rate': signal_win_rate,
        'signal_trades_list': bt.signal_only_trades
    }


def format_comparison_report(result: BacktestResult, comparison: dict) -> str:
    """Format comparison report for Telegram"""
    report = []
    report.append("=" * 50)
    report.append("SURGE-WSI BACKTEST WITH MODE COMPARISON")
    report.append("=" * 50)
    report.append("")
    report.append(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    report.append(f"Initial Balance: ${result.initial_balance:,.2f}")
    report.append("")

    report.append("-" * 40)
    report.append("OVERALL PERFORMANCE (All Trades Executed)")
    report.append("-" * 40)
    report.append(f"Final Balance: ${result.final_balance:,.2f}")
    report.append(f"Net Profit: ${result.net_profit:,.2f} ({result.net_profit_percent:+.2f}%)")
    report.append(f"Total Trades: {result.total_trades}")
    report.append(f"Win Rate: {result.win_rate:.1f}%")
    report.append(f"Profit Factor: {result.profit_factor:.2f}")
    report.append(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")
    report.append("")

    report.append("-" * 40)
    report.append("TRADE MODE COMPARISON")
    report.append("-" * 40)
    report.append(f"AUTO Mode Trades: {comparison['auto_trades']}")
    report.append(f"SIGNAL-ONLY Trades: {comparison['signal_trades']}")
    report.append("")
    report.append(f"AUTO Mode P/L: ${comparison['auto_pnl']:+,.2f}")
    report.append(f"SIGNAL-ONLY P/L: ${comparison['signal_pnl']:+,.2f}")
    report.append("")
    report.append(f"AUTO Win Rate: {comparison['auto_win_rate']:.1f}%")
    report.append(f"SIGNAL-ONLY Win Rate: {comparison['signal_win_rate']:.1f}%")
    report.append("")

    # If signal-only trades were not executed
    if comparison['signal_trades'] > 0:
        hypothetical_balance = result.initial_balance + comparison['auto_pnl']
        hypothetical_profit_pct = comparison['auto_pnl'] / result.initial_balance * 100
        avoided_loss = comparison['signal_pnl'] if comparison['signal_pnl'] < 0 else 0

        report.append("-" * 40)
        report.append("HYPOTHETICAL: If Signal-Only Trades Skipped")
        report.append("-" * 40)
        report.append(f"Balance would be: ${hypothetical_balance:,.2f}")
        report.append(f"Profit would be: ${comparison['auto_pnl']:+,.2f} ({hypothetical_profit_pct:+.2f}%)")
        if comparison['signal_pnl'] < 0:
            report.append(f"Losses Avoided: ${abs(comparison['signal_pnl']):,.2f}")

    report.append("")
    report.append("=" * 50)

    return "\n".join(report)


def format_telegram_message(result: BacktestResult, comparison: dict) -> str:
    """Format for Telegram (HTML)"""
    msg = "<b>SURGE-WSI Backtest with Mode Comparison</b>\n"
    msg += f"<i>{result.start_date.date()} to {result.end_date.date()}</i>\n\n"

    msg += "<b>OVERALL (All Trades):</b>\n"
    msg += f"  Final: ${result.final_balance:,.2f}\n"
    msg += f"  Profit: ${result.net_profit:,.2f} ({result.net_profit_percent:+.1f}%)\n"
    msg += f"  Trades: {result.total_trades} | Win: {result.win_rate:.0f}%\n"
    msg += f"  PF: {result.profit_factor:.2f} | DD: {result.max_drawdown_percent:.1f}%\n\n"

    msg += "<b>MODE COMPARISON:</b>\n"
    msg += f"  AUTO: {comparison['auto_trades']} trades | ${comparison['auto_pnl']:+,.2f}\n"
    msg += f"  SIGNAL: {comparison['signal_trades']} trades | ${comparison['signal_pnl']:+,.2f}\n\n"

    if comparison['signal_trades'] > 0:
        hypothetical = result.initial_balance + comparison['auto_pnl']
        msg += "<b>IF Signal-Only Skipped:</b>\n"
        msg += f"  Balance: ${hypothetical:,.2f}\n"
        if comparison['signal_pnl'] < 0:
            msg += f"  Losses Avoided: ${abs(comparison['signal_pnl']):,.2f}\n"
        else:
            msg += f"  Gains Missed: ${comparison['signal_pnl']:,.2f}\n"

    # Signal-only trade reasons breakdown
    if comparison['signal_trades_list']:
        msg += "\n<b>Signal-Only Reasons:</b>\n"
        reasons = {}
        for t in comparison['signal_trades_list']:
            reason = t.get('reason', 'Unknown')
            # Shorten reason
            if 'December' in reason:
                key = 'December (holiday)'
            elif 'Daily loss' in reason:
                key = 'Daily loss limit'
            elif 'Weekly loss' in reason:
                key = 'Weekly loss limit'
            elif 'consecutive' in reason:
                key = 'Consecutive losses'
            elif 'volatility' in reason:
                key = 'High volatility'
            elif 'regime' in reason:
                key = 'Regime instability'
            else:
                key = reason[:20]
            reasons[key] = reasons.get(key, 0) + 1

        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            msg += f"  - {reason}: {count}\n"

    return msg


async def send_to_telegram(message: str, bot_token: str, chat_id: str):
    """Send message to Telegram"""
    try:
        from telegram import Bot
        bot = Bot(token=bot_token)
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='HTML'
        )
        logger.info("Backtest results sent to Telegram")
    except Exception as e:
        logger.error(f"Failed to send to Telegram: {e}")


def run_monthly_breakdown(htf_data: pd.DataFrame, ltf_data: pd.DataFrame) -> pd.DataFrame:
    """Run backtest broken down by month"""
    months_results = []

    # Generate monthly periods
    for i in range(13):  # 13 months
        year = 2025 if i < 12 else 2026
        month = (i % 12) + 1

        month_start = datetime(year, month, 1, tzinfo=timezone.utc)

        if i == 12:  # Last month (Jan 2026)
            month_end = datetime(2026, 1, 26, tzinfo=timezone.utc)
        else:
            next_month = month + 1 if month < 12 else 1
            next_year = year if month < 12 else year + 1
            month_end = datetime(next_year, next_month, 1, tzinfo=timezone.utc) - timedelta(days=1)

        # Filter data for this month with warmup
        warmup_start = month_start - timedelta(days=30)
        htf_month = htf_data[(htf_data.index >= warmup_start) & (htf_data.index <= month_end)]
        ltf_month = ltf_data[(ltf_data.index >= warmup_start) & (ltf_data.index <= month_end)]

        if htf_month.empty or ltf_month.empty:
            logger.warning(f"No data for {month_start.strftime('%B %Y')}")
            continue

        # Prepare data with 'time' column
        htf = htf_month.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        ltf = ltf_month.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)

        # Run backtest for this month
        bt = Backtester(
            symbol="GBPUSD",
            start_date=month_start.strftime("%Y-%m-%d"),
            end_date=month_end.strftime("%Y-%m-%d"),
            initial_balance=10000.0
        )
        bt.load_data(htf, ltf)
        result = bt.run()

        # Get comparison
        comparison = calculate_signal_only_impact(bt, result)

        months_results.append({
            'month': month_start.strftime("%B"),
            'year': month_start.year,
            'trades': result.total_trades,
            'auto_trades': comparison['auto_trades'],
            'signal_trades': comparison['signal_trades'],
            'win_rate': result.win_rate,
            'net_profit': result.net_profit,
            'auto_pnl': comparison['auto_pnl'],
            'signal_pnl': comparison['signal_pnl'],
            'max_dd': result.max_drawdown_percent
        })

    return pd.DataFrame(months_results)


async def async_main():
    """Async main function"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Fetch data from database
    logger.info("Fetching data from database...")

    symbol = "GBPUSD"
    warmup_start = datetime(2024, 10, 1, tzinfo=timezone.utc)
    overall_end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    htf_data = await fetch_data(symbol, "H4", warmup_start, overall_end)
    ltf_data = await fetch_data(symbol, "M15", warmup_start, overall_end)

    if htf_data.empty or ltf_data.empty:
        logger.error("No data available. Run sync_mt5_data.py first.")
        return

    logger.info(f"Loaded H4: {len(htf_data)} bars, M15: {len(ltf_data)} bars")

    # Prepare data with 'time' column for full backtest
    htf = htf_data.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)
    ltf = ltf_data.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)

    # Run full 13-month backtest
    logger.info("Running 13-month backtest (Jan 2025 - Jan 2026)...")

    bt = Backtester(
        symbol="GBPUSD",
        start_date="2025-01-01",
        end_date="2026-01-26",
        initial_balance=10000.0
    )
    bt.load_data(htf, ltf)
    result = bt.run()

    # Calculate comparison
    comparison = calculate_signal_only_impact(bt, result)

    # Print console report
    console_report = format_comparison_report(result, comparison)
    print(console_report)

    # Run monthly breakdown
    logger.info("Running monthly breakdown...")
    monthly_df = run_monthly_breakdown(htf_data, ltf_data)

    print("\n" + "=" * 60)
    print("MONTHLY BREAKDOWN")
    print("=" * 60)
    print(monthly_df.to_string(index=False))

    # Save monthly results
    results_dir = project_root / "backtest" / "results"
    results_dir.mkdir(exist_ok=True)
    monthly_df.to_csv(results_dir / "monthly_mode_comparison.csv", index=False)
    logger.info(f"Saved to {results_dir / 'monthly_mode_comparison.csv'}")

    # Send to Telegram
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        logger.info("Sending to Telegram...")
        telegram_msg = format_telegram_message(result, comparison)

        # Add monthly summary
        telegram_msg += "\n<b>MONTHLY P/L:</b>\n<pre>"
        for _, row in monthly_df.iterrows():
            telegram_msg += f"{row['month'][:3]} {row['year']}: ${row['net_profit']:+7.0f} (Auto: ${row['auto_pnl']:+7.0f}, Sig: ${row['signal_pnl']:+7.0f})\n"
        telegram_msg += "</pre>"

        await send_to_telegram(telegram_msg, bot_token, chat_id)
    else:
        logger.warning("Telegram credentials not found in .env")


def main():
    """Main function wrapper"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
