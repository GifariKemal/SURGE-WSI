"""Quick Hybrid Mode Backtest
==============================

Fast comparison test for Hybrid Mode (3 months)
Results sent to Telegram.

Usage:
    python -m backtest.quick_hybrid_test

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.utils.killzone import KillZone
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        return pd.DataFrame()
    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def run_backtest(ltf_df: pd.DataFrame, use_hybrid: bool = False) -> dict:
    """Run simplified backtest"""
    from src.analysis.kalman_filter import MultiScaleKalman
    from src.analysis.regime_detector import HMMRegimeDetector

    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=6.0, min_bar_range_pips=4.0,  # CONSERVATIVE thresholds
        activity_threshold=40.0, pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 80.0  # CONSERVATIVE: very high activity only

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    # Detect columns
    close_col = 'close' if 'close' in ltf_df.columns else 'Close'
    high_col = 'high' if 'high' in ltf_df.columns else 'High'
    low_col = 'low' if 'low' in ltf_df.columns else 'Low'

    # Warmup (first 100 bars)
    for _, row in ltf_df.head(100).iterrows():
        kalman.update(row[close_col])
        regime_detector.update(row[close_col])

    # Tracking
    trades = []
    kz_entries = 0
    outside_kz_entries = 0
    balance = 10000.0
    position = None
    cooldown_until = None

    # Process (skip to every 4th bar for speed)
    for idx in range(100, len(ltf_df), 4):
        bar = ltf_df.iloc[idx]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name)
        price = bar[close_col]
        high = bar[high_col]
        low = bar[low_col]

        kalman.update(price)
        regime_info = regime_detector.update(price)

        # Manage position
        if position:
            d = position['direction']
            entry = position['entry']
            sl = position['sl']
            tp = position['tp']

            # Check SL
            if (d == 'BUY' and low <= sl) or (d == 'SELL' and high >= sl):
                pnl = -20.0  # Fixed loss
                balance += pnl
                trades.append({'result': 'loss', 'pnl': pnl, 'in_kz': position.get('in_kz', True)})
                position = None
                cooldown_until = current_time + timedelta(hours=2)
                continue

            # Check TP
            if (d == 'BUY' and high >= tp) or (d == 'SELL' and low <= tp):
                pnl = 40.0  # Fixed win (2R)
                balance += pnl
                trades.append({'result': 'win', 'pnl': pnl, 'in_kz': position.get('in_kz', True)})
                position = None
                cooldown_until = current_time + timedelta(hours=2)
                continue

        # Cooldown
        if cooldown_until and current_time < cooldown_until:
            continue

        if position:
            continue

        # Check KZ
        in_kz, _ = killzone.is_in_killzone(current_time)

        # Hybrid check - CONSERVATIVE: only HIGH activity outside KZ
        can_trade_outside = False
        if use_hybrid and not in_kz:
            recent_df = ltf_df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            # CONSERVATIVE: Only HIGH level AND score >= 80
            if activity.level == ActivityLevel.HIGH and activity.score >= 80:
                can_trade_outside = True

        should_trade = in_kz or (use_hybrid and can_trade_outside)
        if not should_trade:
            continue

        # Regime check
        if not regime_info or not regime_info.is_tradeable:
            continue
        if regime_info.bias == 'NONE':
            continue

        # Simple entry (every qualified bar)
        direction = regime_info.bias
        sl_pips = 20.0
        tp_pips = 40.0

        if direction == 'BUY':
            sl_price = price - sl_pips * 0.0001
            tp_price = price + tp_pips * 0.0001
        else:
            sl_price = price + sl_pips * 0.0001
            tp_price = price - tp_pips * 0.0001

        position = {
            'direction': direction,
            'entry': price,
            'sl': sl_price,
            'tp': tp_price,
            'in_kz': in_kz
        }

        if in_kz:
            kz_entries += 1
        else:
            outside_kz_entries += 1

    # Results
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'win')
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        'mode': 'Hybrid' if use_hybrid else 'KZ Only',
        'trades': total_trades,
        'wins': wins,
        'losses': total_trades - wins,
        'win_rate': wins / total_trades * 100 if total_trades > 0 else 0,
        'total_pnl': total_pnl,
        'return_pct': (balance / 10000 - 1) * 100,
        'kz_entries': kz_entries,
        'outside_kz': outside_kz_entries
    }


async def send_telegram(message: str):
    """Send to Telegram"""
    from telegram import Bot
    try:
        bot = Bot(token=config.telegram.bot_token)
        await bot.send_message(chat_id=config.telegram.chat_id, text=message, parse_mode='HTML')
        logger.info("Report sent to Telegram!")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 50)
    print("HYBRID MODE BACKTEST (CONSERVATIVE)")
    print("Period: Jul 2025 - Jan 2026 (7 months)")
    print("High threshold: score >= 80 + HIGH activity only")
    print("=" * 50)

    # Fetch data (7 months)
    print("\n[1/3] Fetching data...")
    symbol = "GBPUSD"
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    ltf_df = await fetch_data(symbol, "M15", start, end)
    if ltf_df.empty:
        print("ERROR: No data available")
        return

    print(f"      Loaded {len(ltf_df)} M15 bars")

    # Run backtests
    print("\n[2/3] Running backtests...")
    print("      Testing Kill Zone Only...")
    kz_result = run_backtest(ltf_df, use_hybrid=False)
    print(f"      -> {kz_result['trades']} trades, ${kz_result['total_pnl']:.0f}")

    print("      Testing Hybrid Mode...")
    hybrid_result = run_backtest(ltf_df, use_hybrid=True)
    print(f"      -> {hybrid_result['trades']} trades, ${hybrid_result['total_pnl']:.0f}")
    print(f"      -> KZ: {hybrid_result['kz_entries']}, Outside: {hybrid_result['outside_kz']}")

    # Generate report
    print("\n[3/3] Sending report to Telegram...")

    diff_trades = hybrid_result['trades'] - kz_result['trades']
    diff_profit = hybrid_result['total_pnl'] - kz_result['total_pnl']

    winner = "HYBRID" if hybrid_result['total_pnl'] > kz_result['total_pnl'] else "KZ ONLY"
    winner_emoji = "🔄" if winner == "HYBRID" else "⏰"

    msg = f"""🦅 <b>SURGE-WSI HYBRID MODE TEST</b>
<i>Conservative Backtest: Jul 2025 - Jan 2026</i>
<i>High threshold: Score ≥80 + HIGH activity only</i>

⚔️ <b>KILL ZONE vs HYBRID</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Results:</b>
<pre>
Metric          KZ Only    Hybrid
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trades          {kz_result['trades']:>7}    {hybrid_result['trades']:>7}
Wins            {kz_result['wins']:>7}    {hybrid_result['wins']:>7}
Win Rate       {kz_result['win_rate']:>6.1f}%   {hybrid_result['win_rate']:>6.1f}%
P/L            ${kz_result['total_pnl']:>6.0f}   ${hybrid_result['total_pnl']:>6.0f}
Return         {kz_result['return_pct']:>+6.1f}%  {hybrid_result['return_pct']:>+6.1f}%
</pre>

🔄 <b>Hybrid Breakdown:</b>
├ Trades in KZ: {hybrid_result['kz_entries']}
├ Trades outside KZ: {hybrid_result['outside_kz']}
└ Extra trades: {diff_trades:+d}

{winner_emoji} <b>WINNER: {winner}</b>
├ P/L Diff: ${diff_profit:+.0f}
└ Trade Diff: {diff_trades:+d}

💡 <b>Insight:</b>
{"Hybrid Mode captures more opportunities when market is active outside traditional Kill Zones!" if winner == "HYBRID" else "Kill Zone filtering remains effective. Hybrid adds minimal extra value."}

<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"""

    await send_telegram(msg)

    # Print summary
    print("\n" + "=" * 50)
    print(f"WINNER: {winner}")
    print(f"Hybrid trades: {hybrid_result['trades']} (KZ: {hybrid_result['kz_entries']}, Outside: {hybrid_result['outside_kz']})")
    print(f"P/L Difference: ${diff_profit:+.0f}")
    print("=" * 50)
    print("Report sent to Telegram!")


if __name__ == "__main__":
    asyncio.run(main())
