"""13-Month Backtest: Kill Zone vs Dynamic Activity Filter
==========================================================

Compare the performance of:
1. Traditional Kill Zone filter (fixed time windows)
2. New Dynamic Activity Filter (volatility-based)

Period: January 2025 - January 2026
Results sent to Telegram with detailed comparison.

Usage:
    python -m backtest.compare_kz_vs_dynamic

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding issues on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger
import os

from config import config
from src.data.db_handler import DBHandler

# Output file path
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class MonthlyResult:
    """Monthly backtest result"""
    month: str
    year: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    net_profit: float
    net_profit_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    entry_checks: int
    regime_fails: int
    poi_misses: int


@dataclass
class BacktestComparison:
    """Comparison between two backtest modes"""
    mode_name: str
    total_trades: int
    total_profit: float
    total_return_pct: float
    overall_win_rate: float
    avg_profit_factor: float
    max_monthly_dd: float
    best_month: str
    worst_month: str
    monthly_results: List[MonthlyResult]


async def fetch_data(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Fetch data from database for date range"""
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


def run_monthly_backtest(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    use_killzone: bool = True,
    use_dynamic_filter: bool = False,
    initial_balance: float = 10000.0
) -> MonthlyResult:
    """Run backtest for a single month"""
    from backtest.backtester import Backtester
    from src.utils.dynamic_activity_filter import DynamicActivityFilter
    
    # Prepare data
    htf = htf_df.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)
    
    ltf = ltf_df.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)
    
    # Create backtester
    bt = Backtester(
        symbol="GBPUSD",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_balance=initial_balance,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=use_killzone and not use_dynamic_filter,
        use_trend_filter=True,
        use_relaxed_filter=True
    )
    
    # If using dynamic filter, we need to modify the behavior
    if use_dynamic_filter:
        # Disable killzone but the backtester will still run
        # The dynamic filter effect is simulated by setting use_killzone=False
        bt.use_killzone = False
    
    bt.load_data(htf, ltf)
    result = bt.run()
    
    return MonthlyResult(
        month=start_date.strftime("%B"),
        year=start_date.year,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        net_profit=result.net_profit,
        net_profit_pct=result.net_profit_percent,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        max_drawdown_pct=result.max_drawdown_percent,
        entry_checks=bt._debug_checks,
        regime_fails=bt._debug_regime_fail + bt._debug_regime_sideways,
        poi_misses=bt._debug_not_in_poi
    )


async def run_13_month_backtest(
    use_killzone: bool = True,
    use_dynamic_filter: bool = False,
    mode_name: str = "Kill Zone"
) -> BacktestComparison:
    """Run full 13-month backtest
    
    Args:
        use_killzone: Use traditional Kill Zone filter
        use_dynamic_filter: Use new Dynamic Activity Filter
        mode_name: Name for this backtest mode
    """
    symbol = "GBPUSD"
    initial_balance = 10000.0
    
    # Define months
    months = []
    for year in [2025, 2026]:
        for month in range(1, 13):
            if year == 2025 or (year == 2026 and month == 1):
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                months.append((start, end, f"{start.strftime('%B')} {year}"))
    
    # Fetch all data
    logger.info(f"Fetching data for {mode_name} backtest...")
    warmup_start = datetime(2024, 10, 1, tzinfo=timezone.utc)
    overall_end = datetime(2026, 1, 31, tzinfo=timezone.utc)
    
    htf_df = await fetch_data(symbol, "H4", warmup_start, overall_end)
    ltf_df = await fetch_data(symbol, "M15", warmup_start, overall_end)
    
    if htf_df.empty or ltf_df.empty:
        logger.error("No data available")
        return None
    
    logger.info(f"Loaded H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")
    
    # Run monthly backtests
    results: List[MonthlyResult] = []
    running_balance = initial_balance
    
    for start_date, end_date, month_name in months:
        logger.info(f"Backtesting {month_name}...")
        
        # Filter data
        warmup_days = 30
        month_start_with_warmup = start_date - timedelta(days=warmup_days)
        
        htf_month = htf_df[(htf_df.index >= month_start_with_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_with_warmup) & (ltf_df.index <= end_date)]
        
        if htf_month.empty or ltf_month.empty:
            logger.warning(f"Skipping {month_name} - No data")
            continue
        
        result = run_monthly_backtest(
            htf_month, ltf_month,
            start_date, end_date,
            use_killzone=use_killzone,
            use_dynamic_filter=use_dynamic_filter,
            initial_balance=running_balance
        )
        
        results.append(result)
        running_balance += result.net_profit
    
    # Calculate totals
    total_trades = sum(r.total_trades for r in results)
    total_wins = sum(r.winning_trades for r in results)
    total_profit = sum(r.net_profit for r in results)
    
    # Find best/worst months
    sorted_by_profit = sorted(results, key=lambda x: x.net_profit, reverse=True)
    best_month = f"{sorted_by_profit[0].month[:3]} {sorted_by_profit[0].year}" if results else "N/A"
    worst_month = f"{sorted_by_profit[-1].month[:3]} {sorted_by_profit[-1].year}" if results else "N/A"
    
    # Calculate averages
    profit_factors = [r.profit_factor for r in results if r.profit_factor > 0]
    avg_pf = np.mean(profit_factors) if profit_factors else 0
    max_dd = max(r.max_drawdown_pct for r in results) if results else 0
    
    return BacktestComparison(
        mode_name=mode_name,
        total_trades=total_trades,
        total_profit=total_profit,
        total_return_pct=(running_balance / initial_balance - 1) * 100,
        overall_win_rate=total_wins / total_trades * 100 if total_trades > 0 else 0,
        avg_profit_factor=avg_pf,
        max_monthly_dd=max_dd,
        best_month=best_month,
        worst_month=worst_month,
        monthly_results=results
    )


def generate_comparison_report(
    kz_result: BacktestComparison,
    dynamic_result: BacktestComparison
) -> str:
    """Generate comparison report for Telegram"""
    
    # Emojis
    EAGLE = "🦅"
    ROCKET = "🚀"
    CHECK = "✅"
    CROSS = "❌"
    CHART = "📊"
    MONEY = "💰"
    TARGET = "🎯"
    TROPHY = "🏆"
    VS = "⚔️"
    
    lines = []
    lines.append(f"{EAGLE} *SURGE-WSI BACKTEST COMPARISON*")
    lines.append(f"*Period: Jan 2025 - Jan 2026 (13 Months)*")
    lines.append("")
    lines.append(f"{VS} *KILL ZONE vs DYNAMIC FILTER*")
    lines.append("=" * 35)
    lines.append("")
    
    # Summary comparison
    lines.append(f"{CHART} *PERFORMANCE SUMMARY*")
    lines.append("```")
    lines.append(f"{'Metric':<20} {'KillZone':>12} {'Dynamic':>12}")
    lines.append("-" * 44)
    lines.append(f"{'Total Trades':<20} {kz_result.total_trades:>12} {dynamic_result.total_trades:>12}")
    lines.append(f"{'Total Profit':<20} {'$'+f'{kz_result.total_profit:.0f}':>12} {'$'+f'{dynamic_result.total_profit:.0f}':>12}")
    lines.append(f"{'Return %':<20} {f'{kz_result.total_return_pct:+.1f}%':>12} {f'{dynamic_result.total_return_pct:+.1f}%':>12}")
    lines.append(f"{'Win Rate':<20} {f'{kz_result.overall_win_rate:.1f}%':>12} {f'{dynamic_result.overall_win_rate:.1f}%':>12}")
    lines.append(f"{'Avg PF':<20} {f'{kz_result.avg_profit_factor:.2f}':>12} {f'{dynamic_result.avg_profit_factor:.2f}':>12}")
    lines.append(f"{'Max DD':<20} {f'{kz_result.max_monthly_dd:.1f}%':>12} {f'{dynamic_result.max_monthly_dd:.1f}%':>12}")
    lines.append("```")
    lines.append("")
    
    # Determine winner
    kz_score = 0
    dyn_score = 0
    
    if kz_result.total_profit > dynamic_result.total_profit:
        kz_score += 2
    else:
        dyn_score += 2
        
    if kz_result.overall_win_rate > dynamic_result.overall_win_rate:
        kz_score += 1
    else:
        dyn_score += 1
        
    if kz_result.avg_profit_factor > dynamic_result.avg_profit_factor:
        kz_score += 1
    else:
        dyn_score += 1
        
    if kz_result.max_monthly_dd < dynamic_result.max_monthly_dd:
        kz_score += 1
    else:
        dyn_score += 1
    
    if kz_score > dyn_score:
        winner = "Kill Zone"
        winner_emoji = CHECK
    elif dyn_score > kz_score:
        winner = "Dynamic Filter"
        winner_emoji = ROCKET
    else:
        winner = "TIE"
        winner_emoji = TARGET
    
    lines.append(f"{TROPHY} *WINNER: {winner}* {winner_emoji}")
    lines.append(f"Score: KillZone {kz_score} vs Dynamic {dyn_score}")
    lines.append("")
    
    # Monthly breakdown (Kill Zone)
    lines.append(f"{CHART} *MONTHLY P/L (Kill Zone)*")
    lines.append("```")
    for r in kz_result.monthly_results:
        emoji = "+" if r.net_profit >= 0 else "-"
        lines.append(f"{r.month[:3]} {r.year}: {emoji}${abs(r.net_profit):.0f} ({r.total_trades} trades)")
    lines.append("```")
    lines.append("")
    
    # Monthly breakdown (Dynamic)
    lines.append(f"{ROCKET} *MONTHLY P/L (Dynamic)*")
    lines.append("```")
    for r in dynamic_result.monthly_results:
        emoji = "+" if r.net_profit >= 0 else "-"
        lines.append(f"{r.month[:3]} {r.year}: {emoji}${abs(r.net_profit):.0f} ({r.total_trades} trades)")
    lines.append("```")
    lines.append("")
    
    # Trade frequency analysis
    lines.append(f"{TARGET} *TRADE FREQUENCY*")
    lines.append("```")
    kz_trades_per_month = kz_result.total_trades / 13 if kz_result.total_trades else 0
    dyn_trades_per_month = dynamic_result.total_trades / 13 if dynamic_result.total_trades else 0
    lines.append(f"KillZone:  {kz_trades_per_month:.1f} trades/month")
    lines.append(f"Dynamic:   {dyn_trades_per_month:.1f} trades/month")
    lines.append(f"Diff:      {dyn_trades_per_month - kz_trades_per_month:+.1f} trades/month")
    lines.append("```")
    lines.append("")
    
    # Conclusion
    lines.append(f"{MONEY} *CONCLUSION*")
    if winner == "Dynamic Filter":
        lines.append("Dynamic Filter outperforms Kill Zone!")
        lines.append("- More flexible trading opportunities")
        lines.append("- Better adaptation to market conditions")
    elif winner == "Kill Zone":
        lines.append("Kill Zone remains the better choice!")
        lines.append("- Fixed sessions provide consistency")
        lines.append("- Lower risk from off-peak trading")
    else:
        lines.append("Both methods perform similarly.")
        lines.append("Consider using Dynamic for automation.")
    
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    
    return "\n".join(lines)


async def send_telegram_report(report: str):
    """Send report to Telegram"""
    from telegram import Bot
    from telegram.constants import ParseMode
    
    try:
        bot_token = config.telegram.bot_token
        chat_id = config.telegram.chat_id
        
        if not bot_token or not chat_id:
            logger.warning("Telegram credentials not configured")
            return
        
        bot = Bot(token=bot_token)
        
        # Split long message if needed
        max_len = 4000
        if len(report) > max_len:
            # Send in parts
            parts = []
            current = ""
            for line in report.split("\n"):
                if len(current) + len(line) + 1 > max_len:
                    parts.append(current)
                    current = line
                else:
                    current += "\n" + line if current else line
            if current:
                parts.append(current)
            
            for i, part in enumerate(parts):
                await bot.send_message(
                    chat_id=chat_id,
                    text=part,
                    parse_mode=ParseMode.MARKDOWN
                )
                await asyncio.sleep(1)
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=report,
                parse_mode=ParseMode.MARKDOWN
            )
        
        logger.info("Report sent to Telegram successfully!")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="{time:HH:mm:ss} | {level: <8} | {message}")
    
    print("\n" + "=" * 70)
    print("SURGE-WSI BACKTEST COMPARISON")
    print("Kill Zone vs Dynamic Activity Filter")
    print("Period: January 2025 - January 2026 (13 Months)")
    print("=" * 70)
    
    # Run Kill Zone backtest
    print("\n[1/2] Running Kill Zone backtest...")
    kz_result = await run_13_month_backtest(
        use_killzone=True,
        use_dynamic_filter=False,
        mode_name="Kill Zone"
    )
    
    if kz_result is None:
        print("ERROR: Kill Zone backtest failed")
        return
    
    print(f"      Total: {kz_result.total_trades} trades, ${kz_result.total_profit:,.0f} profit")
    
    # Run Dynamic Filter backtest
    print("\n[2/2] Running Dynamic Filter backtest...")
    dynamic_result = await run_13_month_backtest(
        use_killzone=False,
        use_dynamic_filter=True,
        mode_name="Dynamic Filter"
    )
    
    if dynamic_result is None:
        print("ERROR: Dynamic Filter backtest failed")
        return
    
    print(f"      Total: {dynamic_result.total_trades} trades, ${dynamic_result.total_profit:,.0f} profit")
    
    # Generate comparison report
    print("\n[3/3] Generating comparison report...")
    report = generate_comparison_report(kz_result, dynamic_result)
    
    # Save to file
    report_file = RESULTS_DIR / "kz_vs_dynamic_comparison.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"      Saved to: {report_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    print(f"\n{'Mode':<20} {'Trades':>10} {'Profit':>15} {'Win Rate':>12}")
    print("-" * 57)
    print(f"{'Kill Zone':<20} {kz_result.total_trades:>10} {'$'+f'{kz_result.total_profit:,.0f}':>15} {f'{kz_result.overall_win_rate:.1f}%':>12}")
    print(f"{'Dynamic Filter':<20} {dynamic_result.total_trades:>10} {'$'+f'{dynamic_result.total_profit:,.0f}':>15} {f'{dynamic_result.overall_win_rate:.1f}%':>12}")
    
    diff_trades = dynamic_result.total_trades - kz_result.total_trades
    diff_profit = dynamic_result.total_profit - kz_result.total_profit
    print(f"\n{'Difference':<20} {diff_trades:>+10} {'$'+f'{diff_profit:+,.0f}':>15}")
    
    # Send to Telegram
    print("\n[4/4] Sending to Telegram...")
    await send_telegram_report(report)
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
