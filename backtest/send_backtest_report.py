"""Send Detailed Backtest Report to Telegram
=============================================

Generate and send detailed monthly backtest report with SURIOTA styling.

Usage:
    python send_backtest_report.py

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows console encoding for emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
from datetime import datetime
from telegram import Bot
from telegram.constants import ParseMode
from config import config


# ============================================================
# SURIOTA STYLE FORMATTERS
# ============================================================

# Emojis
EAGLE = "🦅"
ROCKET = "🚀"
CHECK = "✅"
CROSS = "❌"
CHART = "📊"
MONEY = "💰"
TARGET = "🎯"
TROPHY = "🏆"
WARNING = "⚠️"
CALENDAR = "📅"

# Tree connectors
BRANCH = "├"
LAST = "└"


def progress_bar(percent: float, width: int = 10) -> str:
    """Create progress bar: ████████░░"""
    filled = int(percent / 100 * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def month_emoji(pnl: float) -> str:
    """Get emoji based on monthly P/L"""
    if pnl >= 500:
        return ROCKET
    elif pnl > 0:
        return CHECK
    elif pnl == 0:
        return "➖"
    return CROSS


def generate_detailed_report() -> str:
    """Generate detailed monthly backtest report (console)"""

    results_path = Path(__file__).parent / "results" / "monthly_backtest_results.csv"
    if not results_path.exists():
        return "ERROR: Backtest results not found. Run monthly_backtest.py first."

    df = pd.read_csv(results_path)

    report = []
    report.append("=" * 60)
    report.append(f"{EAGLE} SURGE-WSI BACKTEST REPORT")
    report.append("6-Layer Trading System")
    report.append("=" * 60)
    report.append("")

    # Per-month details
    report.append(f"{CALENDAR} MONTHLY BREAKDOWN")
    report.append("-" * 60)

    total_trades = 0
    total_wins = 0
    total_pnl = 0

    for _, row in df.iterrows():
        month = row['month'][:3]
        year = int(row['year'])
        trades = int(row['trades'])
        win_rate = row['win_rate']
        pf = row['profit_factor']
        pnl = row['net_profit']
        dd = row['max_drawdown_pct']
        balance = row['final_balance']

        total_trades += trades
        total_wins += int(trades * win_rate / 100)
        total_pnl += pnl

        emoji = month_emoji(pnl)

        report.append(f"\n{emoji} {month} {year}")
        report.append(f"   {BRANCH} Trades: {trades}")
        if trades > 0:
            report.append(f"   {BRANCH} Win Rate: {win_rate:.1f}%")
            report.append(f"   {BRANCH} Profit Factor: {pf:.2f}")
            report.append(f"   {BRANCH} P/L: ${pnl:+.2f}")
            report.append(f"   {BRANCH} Max DD: {dd:.2f}%")
        report.append(f"   {LAST} Balance: ${balance:,.2f}")

    # Overall summary
    report.append("")
    report.append("=" * 60)
    report.append(f"{TARGET} OVERALL PERFORMANCE")
    report.append("=" * 60)

    initial_balance = 10000.0
    final_balance = df['final_balance'].iloc[-1]
    total_return = (final_balance / initial_balance - 1) * 100

    report.append(f"\n{BRANCH} Starting Balance: ${initial_balance:,.2f}")
    report.append(f"{BRANCH} Final Balance: ${final_balance:,.2f}")
    report.append(f"{BRANCH} Total Net Profit: ${total_pnl:,.2f}")
    report.append(f"{LAST} Total Return: {total_return:+.2f}%")
    report.append("")
    report.append(f"{BRANCH} Total Trades: {total_trades}")
    report.append(f"{BRANCH} Winning Trades: {total_wins}")
    report.append(f"{BRANCH} Win Rate: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
    report.append(f"{LAST} Trades/Month: {total_trades/len(df):.1f}")

    # Monthly P/L distribution
    profitable_months = len(df[df['net_profit'] > 0])
    losing_months = len(df[df['net_profit'] < 0])

    report.append("")
    report.append(f"{CHART} MONTHLY DISTRIBUTION")
    report.append("-" * 60)
    report.append(f"{BRANCH} Profitable Months: {profitable_months}")
    report.append(f"{LAST} Losing Months: {losing_months}")

    # Best/Worst month
    best_month = df.loc[df['net_profit'].idxmax()]
    worst_month = df.loc[df['net_profit'].idxmin()]

    report.append("")
    report.append(f"{TROPHY} Best Month: {best_month['month'][:3]} {int(best_month['year'])} (+${best_month['net_profit']:.2f})")
    report.append(f"{WARNING} Worst Month: {worst_month['month'][:3]} {int(worst_month['year'])} (${worst_month['net_profit']:.2f})")

    # Risk metrics
    max_dd = df['max_drawdown_pct'].max()
    avg_pf = df[df['profit_factor'] > 0]['profit_factor'].mean()

    report.append("")
    report.append(f"{BRANCH} Max Monthly Drawdown: {max_dd:.2f}%")
    report.append(f"{LAST} Avg Profit Factor: {avg_pf:.2f}")

    # Timestamp
    report.append("")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)

    return "\n".join(report)


def generate_telegram_report() -> str:
    """Generate formatted report for Telegram with SURIOTA styling"""

    results_path = Path(__file__).parent / "results" / "monthly_backtest_results.csv"
    if not results_path.exists():
        return "ERROR: Backtest results not found."

    df = pd.read_csv(results_path)

    # Calculate totals
    initial_balance = 10000.0
    final_balance = df['final_balance'].iloc[-1]
    total_pnl = df['net_profit'].sum()
    total_return = (final_balance / initial_balance - 1) * 100
    total_trades = int(df['trades'].sum())
    total_wins = sum(int(row['trades'] * row['win_rate'] / 100) for _, row in df.iterrows())
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Header
    msg = []
    msg.append(f"{EAGLE} <b>SURGE-WSI BACKTEST REPORT</b>")
    msg.append("<i>6-Layer Trading System</i>")
    msg.append("")

    # Overall Performance (Tree-style)
    msg.append(f"{TARGET} <b>Performance</b>")
    msg.append(f"{BRANCH} Balance: ${initial_balance:,.0f} → ${final_balance:,.2f}")

    pnl_emoji = ROCKET if total_pnl >= 1000 else (CHECK if total_pnl > 0 else CROSS)
    msg.append(f"{BRANCH} Net P/L: {pnl_emoji} <b>${total_pnl:+,.2f}</b>")
    msg.append(f"{BRANCH} Return: <b>{total_return:+.2f}%</b>")
    msg.append(f"{LAST} Win Rate: {win_rate:.1f}% ({total_wins}/{total_trades})")
    msg.append("")

    # Monthly breakdown table
    msg.append(f"{CALENDAR} <b>Monthly Detail</b>")
    msg.append("<pre>")
    msg.append("Month   │ Tr │ Win% │  PF  │    P/L")
    msg.append("────────┼────┼──────┼──────┼────────")

    for _, row in df.iterrows():
        month = f"{row['month'][:3]}{int(row['year']) % 100:02d}"
        trades = int(row['trades'])
        win = row['win_rate']
        pf = row['profit_factor']
        pnl = row['net_profit']

        if trades == 0:
            msg.append(f"{month}  │  0 │   -  │  -   │    $0")
        else:
            emoji = "+" if pnl >= 0 else ""
            msg.append(f"{month}  │{trades:3d} │{win:5.1f}%│{pf:5.2f} │{emoji}${pnl:>6.0f}")

    msg.append("</pre>")

    # Stats section
    profitable = len(df[df['net_profit'] > 0])
    losing = len(df[df['net_profit'] < 0])
    best = df.loc[df['net_profit'].idxmax()]
    worst = df.loc[df['net_profit'].idxmin()]
    max_dd = df['max_drawdown_pct'].max()
    avg_pf = df[df['profit_factor'] > 0]['profit_factor'].mean()

    msg.append("")
    msg.append(f"{CHART} <b>Statistics</b>")
    msg.append(f"{BRANCH} Profitable: {profitable}/13 months")
    msg.append(f"{BRANCH} Max DD: {max_dd:.2f}%")
    msg.append(f"{LAST} Avg PF: {avg_pf:.2f}")
    msg.append("")

    # Best/Worst with progress bars
    best_bar = progress_bar(min(100, best['net_profit'] / 30), 8)
    worst_bar = progress_bar(min(100, abs(worst['net_profit']) / 30), 8)

    msg.append(f"{TROPHY} Best: {best['month'][:3]} {int(best['year'])}")
    msg.append(f"   {best_bar} +${best['net_profit']:.0f}")
    msg.append(f"{WARNING} Worst: {worst['month'][:3]} {int(worst['year'])}")
    msg.append(f"   {worst_bar} ${worst['net_profit']:.0f}")

    # Footer
    msg.append("")
    msg.append(f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>")

    return "\n".join(msg)


def generate_compact_summary() -> str:
    """Generate compact summary for quick view"""

    results_path = Path(__file__).parent / "results" / "monthly_backtest_results.csv"
    if not results_path.exists():
        return "ERROR: No results"

    df = pd.read_csv(results_path)

    initial_balance = 10000.0
    final_balance = df['final_balance'].iloc[-1]
    total_pnl = df['net_profit'].sum()
    total_return = (final_balance / initial_balance - 1) * 100
    total_trades = int(df['trades'].sum())
    total_wins = sum(int(row['trades'] * row['win_rate'] / 100) for _, row in df.iterrows())
    profitable = len(df[df['net_profit'] > 0])

    pnl_emoji = ROCKET if total_pnl >= 1000 else (CHECK if total_pnl > 0 else CROSS)
    bar = progress_bar(min(100, total_return), 10)

    msg = f"{EAGLE} <b>SURGE-WSI</b> • 13mo Backtest\n"
    msg += f"{bar} <b>{total_return:+.1f}%</b>\n"
    msg += f"\n"
    msg += f"P/L: {pnl_emoji} <b>${total_pnl:+,.0f}</b>\n"
    msg += f"Trades: {total_trades} • Win: {total_wins/total_trades*100:.0f}%\n"
    msg += f"Profitable: {profitable}/13 months"

    return msg


async def send_to_telegram(message: str, compact: bool = False):
    """Send message to Telegram"""
    bot_token = config.telegram.bot_token
    chat_id = config.telegram.chat_id

    if not bot_token or not chat_id:
        print("ERROR: Telegram credentials not configured")
        return False

    try:
        bot = Bot(token=bot_token)

        # Split message if too long (Telegram limit is 4096 chars)
        if len(message) > 4000:
            parts = []
            current = ""
            for line in message.split("\n"):
                if len(current) + len(line) + 1 > 4000:
                    parts.append(current)
                    current = line
                else:
                    current += "\n" + line if current else line
            if current:
                parts.append(current)

            for part in parts:
                await bot.send_message(
                    chat_id=chat_id,
                    text=part,
                    parse_mode=ParseMode.HTML
                )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.HTML
            )

        print("Report sent to Telegram successfully!")
        return True

    except Exception as e:
        print(f"ERROR sending to Telegram: {e}")
        return False


async def main():
    """Main function"""
    print("\n" + "=" * 60)
    print(f"{EAGLE} SURGE-WSI BACKTEST REPORT GENERATOR")
    print("=" * 60)

    # Generate detailed console report
    print("\nGenerating detailed report...")
    detailed_report = generate_detailed_report()
    print("\n" + detailed_report)

    # Generate and send Telegram report
    print("\nSending to Telegram...")
    telegram_report = generate_telegram_report()
    await send_to_telegram(telegram_report)

    # Also send compact summary
    print("\nSending compact summary...")
    compact_report = generate_compact_summary()
    await send_to_telegram(compact_report)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
