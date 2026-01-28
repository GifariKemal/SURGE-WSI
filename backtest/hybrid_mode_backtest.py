"""Hybrid Mode Backtest Comparison
==================================

Compare performance between:
1. Traditional Kill Zone Only
2. Hybrid Mode (Kill Zone + Dynamic Activity)

This properly simulates Hybrid Mode by checking if market activity
is high enough to trade outside Kill Zones.

Period: January 2025 - January 2026 (13 months)
Results sent to Telegram.

Usage:
    python -m backtest.hybrid_mode_backtest

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
from typing import List, Optional
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.utils.killzone import KillZone
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel


@dataclass
class MonthResult:
    """Monthly result"""
    month: str
    year: int
    trades: int
    wins: int
    losses: int
    profit: float
    win_rate: float
    kz_trades: int      # Trades in Kill Zone
    outside_kz: int     # Trades outside KZ (Hybrid only)


@dataclass
class BacktestSummary:
    """Full backtest summary"""
    mode: str
    total_trades: int
    total_profit: float
    return_pct: float
    win_rate: float
    avg_profit_per_trade: float
    monthly_results: List[MonthResult]
    kz_trades: int
    outside_kz_trades: int


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

    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def simulate_trades(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    use_hybrid: bool = False,
    initial_balance: float = 10000.0
) -> BacktestSummary:
    """
    Simplified trade simulation

    Logic:
    - Check Kill Zone status
    - If Hybrid mode: also check Dynamic Activity outside KZ
    - Simulate entry when conditions met
    - Use fixed RR targets
    """
    from src.analysis.kalman_filter import MultiScaleKalman
    from src.analysis.regime_detector import HMMRegimeDetector
    from src.analysis.poi_detector import POIDetector

    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=5.0,
        min_bar_range_pips=3.0,
        activity_threshold=40.0,
        pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 70.0

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()
    poi_detector = POIDetector()

    # Detect column names (handle both lowercase and capitalized)
    close_col = 'close' if 'close' in ltf_df.columns else 'Close'
    high_col = 'high' if 'high' in ltf_df.columns else 'High'
    low_col = 'low' if 'low' in ltf_df.columns else 'Low'

    # Warmup
    for _, row in ltf_df.head(100).iterrows():
        kalman.update(row[close_col])
        regime_detector.update(row[close_col])

    # State
    balance = initial_balance
    trades = []
    current_position = None
    cooldown_until = None

    # Monthly tracking
    monthly_results = {}

    # Stats
    total_kz_trades = 0
    total_outside_kz = 0

    # Process bars
    for idx in range(100, len(ltf_df)):
        bar = ltf_df.iloc[idx]
        current_time = bar.name if isinstance(bar.name, datetime) else bar.get('time', bar.name)
        price = bar[close_col]
        high = bar[high_col]
        low = bar[low_col]

        # Update indicators
        kalman.update(price)
        regime_info = regime_detector.update(price)

        # Manage existing position
        if current_position:
            direction = current_position['direction']
            entry = current_position['entry']
            sl = current_position['sl']
            tp1 = current_position['tp1']
            tp2 = current_position['tp2']

            # Check SL hit
            if (direction == 'BUY' and low <= sl) or (direction == 'SELL' and high >= sl):
                pnl = -abs(entry - sl) / 0.0001 * 0.1  # Simplified
                balance += pnl
                trades.append({
                    'time': current_time,
                    'direction': direction,
                    'pnl': pnl,
                    'result': 'loss',
                    'in_kz': current_position.get('in_kz', True)
                })
                current_position = None
                cooldown_until = current_time + timedelta(hours=4)
                continue

            # Check TP1 (1:1 RR)
            if not current_position.get('tp1_hit'):
                if (direction == 'BUY' and high >= tp1) or (direction == 'SELL' and low <= tp1):
                    current_position['tp1_hit'] = True
                    current_position['sl'] = entry  # Move to breakeven

            # Check TP2 (2:1 RR)
            if (direction == 'BUY' and high >= tp2) or (direction == 'SELL' and low <= tp2):
                pnl = abs(tp2 - entry) / 0.0001 * 0.1 * 0.8  # Partial closed earlier
                balance += pnl
                trades.append({
                    'time': current_time,
                    'direction': direction,
                    'pnl': pnl,
                    'result': 'win',
                    'in_kz': current_position.get('in_kz', True)
                })
                current_position = None
                cooldown_until = current_time + timedelta(hours=4)
                continue

        # Skip if in cooldown
        if cooldown_until and current_time < cooldown_until:
            continue

        # Skip if already have position
        if current_position:
            continue

        # Check trading conditions
        in_kz, session = killzone.is_in_killzone(current_time)

        # Activity check for hybrid mode
        can_trade_outside_kz = False
        if use_hybrid and not in_kz:
            # Get recent data for activity calculation
            recent_idx = max(0, idx - 20)
            recent_df = ltf_df.iloc[recent_idx:idx+1]

            activity = activity_filter.check_activity(
                current_time, high, low, recent_df
            )

            # Only trade outside KZ if activity is HIGH
            if activity.level == ActivityLevel.HIGH:
                can_trade_outside_kz = True

        # Decide if we can trade
        should_trade = in_kz or (use_hybrid and can_trade_outside_kz)
        if not should_trade:
            continue

        # Check regime
        if not regime_info or not regime_info.is_tradeable:
            continue

        direction = regime_info.bias
        if direction == 'NONE':
            continue

        # Update POI detector (simplified - every 100 bars)
        if idx % 100 == 0:
            htf_subset = htf_df[htf_df.index <= current_time].tail(200)
            if len(htf_subset) > 50:
                poi_detector.detect(htf_subset)

        # Check if at POI (simplified)
        poi_result = poi_detector.last_result
        if poi_result is None:
            continue

        at_poi, poi_info = poi_result.price_at_poi(price, direction)
        if not at_poi or poi_info is None:
            continue

        # Entry signal - open position
        sl_pips = 25.0
        if direction == 'BUY':
            entry_price = price
            sl_price = price - sl_pips * 0.0001
            tp1_price = price + sl_pips * 0.0001
            tp2_price = price + sl_pips * 2 * 0.0001
        else:
            entry_price = price
            sl_price = price + sl_pips * 0.0001
            tp1_price = price - sl_pips * 0.0001
            tp2_price = price - sl_pips * 2 * 0.0001

        current_position = {
            'direction': direction,
            'entry': entry_price,
            'sl': sl_price,
            'tp1': tp1_price,
            'tp2': tp2_price,
            'tp1_hit': False,
            'in_kz': in_kz
        }

        # Track KZ vs outside
        if in_kz:
            total_kz_trades += 1
        else:
            total_outside_kz += 1

    # Calculate results
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'win')
    total_profit = sum(t['pnl'] for t in trades)

    # Group by month
    for trade in trades:
        t = trade['time']
        if isinstance(t, pd.Timestamp):
            t = t.to_pydatetime()
        key = f"{t.year}-{t.month:02d}"
        if key not in monthly_results:
            monthly_results[key] = {
                'trades': 0, 'wins': 0, 'losses': 0,
                'profit': 0.0, 'kz': 0, 'outside': 0
            }
        monthly_results[key]['trades'] += 1
        if trade['result'] == 'win':
            monthly_results[key]['wins'] += 1
        else:
            monthly_results[key]['losses'] += 1
        monthly_results[key]['profit'] += trade['pnl']
        if trade.get('in_kz', True):
            monthly_results[key]['kz'] += 1
        else:
            monthly_results[key]['outside'] += 1

    # Convert to MonthResult list
    monthly_list = []
    for key in sorted(monthly_results.keys()):
        year, month = map(int, key.split('-'))
        m = monthly_results[key]
        monthly_list.append(MonthResult(
            month=datetime(year, month, 1).strftime('%B'),
            year=year,
            trades=m['trades'],
            wins=m['wins'],
            losses=m['losses'],
            profit=m['profit'],
            win_rate=m['wins']/m['trades']*100 if m['trades'] > 0 else 0,
            kz_trades=m['kz'],
            outside_kz=m['outside']
        ))

    return BacktestSummary(
        mode="Hybrid" if use_hybrid else "Kill Zone Only",
        total_trades=total_trades,
        total_profit=total_profit,
        return_pct=(balance / initial_balance - 1) * 100,
        win_rate=wins/total_trades*100 if total_trades > 0 else 0,
        avg_profit_per_trade=total_profit/total_trades if total_trades > 0 else 0,
        monthly_results=monthly_list,
        kz_trades=total_kz_trades,
        outside_kz_trades=total_outside_kz
    )


def generate_report(kz_result: BacktestSummary, hybrid_result: BacktestSummary) -> str:
    """Generate comparison report for Telegram"""
    lines = []

    lines.append("🦅 <b>SURGE-WSI HYBRID MODE BACKTEST</b>")
    lines.append("<i>Period: Jan 2025 - Jan 2026 (13 months)</i>")
    lines.append("")
    lines.append("⚔️ <b>KILL ZONE vs HYBRID MODE</b>")
    lines.append("═" * 35)
    lines.append("")

    # Performance comparison
    lines.append("📊 <b>PERFORMANCE SUMMARY</b>")
    lines.append("<pre>")
    lines.append(f"{'Metric':<18} {'KZ Only':>12} {'Hybrid':>12}")
    lines.append("-" * 42)
    lines.append(f"{'Total Trades':<18} {kz_result.total_trades:>12} {hybrid_result.total_trades:>12}")
    lines.append(f"{'Total Profit':<18} {'$'+f'{kz_result.total_profit:.0f}':>12} {'$'+f'{hybrid_result.total_profit:.0f}':>12}")
    lines.append(f"{'Return %':<18} {f'{kz_result.return_pct:+.1f}%':>12} {f'{hybrid_result.return_pct:+.1f}%':>12}")
    lines.append(f"{'Win Rate':<18} {f'{kz_result.win_rate:.1f}%':>12} {f'{hybrid_result.win_rate:.1f}%':>12}")
    lines.append(f"{'Avg/Trade':<18} {'$'+f'{kz_result.avg_profit_per_trade:.0f}':>12} {'$'+f'{hybrid_result.avg_profit_per_trade:.0f}':>12}")
    lines.append("</pre>")
    lines.append("")

    # Hybrid mode breakdown
    lines.append("🔄 <b>HYBRID MODE BREAKDOWN</b>")
    lines.append("<pre>")
    lines.append(f"Trades in KZ:      {hybrid_result.kz_trades}")
    lines.append(f"Trades outside KZ: {hybrid_result.outside_kz_trades}")
    pct_outside = (hybrid_result.outside_kz_trades / hybrid_result.total_trades * 100) if hybrid_result.total_trades > 0 else 0
    lines.append(f"Extra opportunity: +{pct_outside:.1f}%")
    lines.append("</pre>")
    lines.append("")

    # Determine winner
    diff_trades = hybrid_result.total_trades - kz_result.total_trades
    diff_profit = hybrid_result.total_profit - kz_result.total_profit

    if hybrid_result.total_profit > kz_result.total_profit:
        lines.append("🏆 <b>WINNER: HYBRID MODE</b> ✅")
        lines.append(f"   +{diff_trades} more trades")
        lines.append(f"   +${diff_profit:.0f} extra profit")
    else:
        lines.append("🏆 <b>WINNER: KILL ZONE ONLY</b> ✅")
        lines.append(f"   Safer with fewer trades")

    lines.append("")

    # Monthly P/L comparison
    lines.append("📈 <b>MONTHLY P/L</b>")
    lines.append("<pre>")
    lines.append(f"{'Month':<10} {'KZ Only':>10} {'Hybrid':>10}")
    lines.append("-" * 30)

    # Align monthly results
    kz_months = {f"{r.month[:3]} {r.year}": r for r in kz_result.monthly_results}
    hybrid_months = {f"{r.month[:3]} {r.year}": r for r in hybrid_result.monthly_results}
    all_months = sorted(set(list(kz_months.keys()) + list(hybrid_months.keys())))

    for month in all_months:
        kz_r = kz_months.get(month)
        hyb_r = hybrid_months.get(month)
        kz_pnl = f"${kz_r.profit:+.0f}" if kz_r else "-"
        hyb_pnl = f"${hyb_r.profit:+.0f}" if hyb_r else "-"
        lines.append(f"{month:<10} {kz_pnl:>10} {hyb_pnl:>10}")

    lines.append("</pre>")
    lines.append("")

    # Conclusion
    lines.append("💡 <b>CONCLUSION</b>")
    if hybrid_result.total_profit > kz_result.total_profit and hybrid_result.outside_kz_trades > 0:
        lines.append("Hybrid Mode adds value by:")
        lines.append(f"• {hybrid_result.outside_kz_trades} extra trades outside KZ")
        lines.append("• Captures high-activity opportunities")
        lines.append("• Recommended for active markets")
    else:
        lines.append("Kill Zone Only is sufficient:")
        lines.append("• Lower trade frequency = lower risk")
        lines.append("• Hybrid adds minimal benefit")

    lines.append("")
    lines.append(f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>")

    return "\n".join(lines)


async def send_telegram(message: str):
    """Send message to Telegram"""
    from telegram import Bot

    try:
        bot = Bot(token=config.telegram.bot_token)

        # Split if too long
        if len(message) > 4000:
            parts = message.split("\n\n")
            current = ""
            for part in parts:
                if len(current) + len(part) + 2 > 4000:
                    await bot.send_message(chat_id=config.telegram.chat_id, text=current, parse_mode='HTML')
                    current = part
                    await asyncio.sleep(1)
                else:
                    current = current + "\n\n" + part if current else part
            if current:
                await bot.send_message(chat_id=config.telegram.chat_id, text=current, parse_mode='HTML')
        else:
            await bot.send_message(chat_id=config.telegram.chat_id, text=message, parse_mode='HTML')

        logger.info("Report sent to Telegram!")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 60)
    print("SURGE-WSI HYBRID MODE BACKTEST")
    print("Kill Zone Only vs Hybrid Mode (KZ + Dynamic Activity)")
    print("Period: January 2025 - January 2026")
    print("=" * 60)

    # Fetch data
    print("\n[1/4] Fetching historical data...")
    symbol = "GBPUSD"
    start = datetime(2024, 11, 1, tzinfo=timezone.utc)  # Warmup
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data(symbol, "H4", start, end)
    ltf_df = await fetch_data(symbol, "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available. Make sure database has data.")
        return

    print(f"      Loaded H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Run Kill Zone Only backtest
    print("\n[2/4] Running Kill Zone Only backtest...")
    kz_result = simulate_trades(htf_df, ltf_df, use_hybrid=False)
    print(f"      {kz_result.total_trades} trades, ${kz_result.total_profit:,.0f} profit")

    # Run Hybrid Mode backtest
    print("\n[3/4] Running Hybrid Mode backtest...")
    hybrid_result = simulate_trades(htf_df, ltf_df, use_hybrid=True)
    print(f"      {hybrid_result.total_trades} trades, ${hybrid_result.total_profit:,.0f} profit")
    print(f"      (KZ: {hybrid_result.kz_trades}, Outside KZ: {hybrid_result.outside_kz_trades})")

    # Generate and send report
    print("\n[4/4] Generating report and sending to Telegram...")
    report = generate_report(kz_result, hybrid_result)

    # Save to file
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    report_file = results_dir / "hybrid_mode_comparison.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        # Remove HTML tags for file
        import re
        clean = re.sub(r'<[^>]+>', '', report)
        f.write(clean)
    print(f"      Saved to: {report_file}")

    # Send to Telegram
    await send_telegram(report)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Mode':<20} {'Trades':>8} {'Profit':>12} {'Win Rate':>10}")
    print("-" * 50)
    print(f"{'Kill Zone Only':<20} {kz_result.total_trades:>8} {'$'+f'{kz_result.total_profit:,.0f}':>12} {f'{kz_result.win_rate:.1f}%':>10}")
    print(f"{'Hybrid Mode':<20} {hybrid_result.total_trades:>8} {'$'+f'{hybrid_result.total_profit:,.0f}':>12} {f'{hybrid_result.win_rate:.1f}%':>10}")

    diff = hybrid_result.total_profit - kz_result.total_profit
    print(f"\n{'Difference':<20} {hybrid_result.total_trades - kz_result.total_trades:>+8} {'$'+f'{diff:+,.0f}':>12}")

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE - Report sent to Telegram!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
