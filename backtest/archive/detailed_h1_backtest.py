"""Detailed H1 Backtest - 13 Months
===================================

Comprehensive backtest testing all SURGE-WSI features:
- Kalman Filter (noise reduction)
- HMM Regime Detection (BULLISH/BEARISH/SIDEWAYS)
- Kill Zones (London/New York sessions)
- Hybrid Mode (Dynamic Activity Filter)
- POI Detection (Order Blocks, FVG)
- Entry Triggers (Rejection Candles)
- Risk Management (Position Sizing)
- Exit Management (Partial TP, Trailing SL)

Usage:
    python -m backtest.detailed_h1_backtest

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
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.utils.killzone import KillZone
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector


@dataclass
class BacktestTrade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""  # BUY or SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    lot_size: float = 0.1
    pnl: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""  # WIN, LOSS, BE
    exit_reason: str = ""  # TP1, TP2, SL, TRAILING, REGIME_FLIP
    in_killzone: bool = True
    session: str = ""
    regime: str = ""
    activity_score: float = 0.0
    poi_type: str = ""  # OB, FVG
    quality_score: float = 0.0


@dataclass
class BacktestStats:
    """Backtest statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0  # in hours

    # By session
    london_trades: int = 0
    london_pnl: float = 0.0
    newyork_trades: int = 0
    newyork_pnl: float = 0.0
    hybrid_trades: int = 0
    hybrid_pnl: float = 0.0

    # By regime
    bullish_trades: int = 0
    bullish_pnl: float = 0.0
    bearish_trades: int = 0
    bearish_pnl: float = 0.0

    # By exit reason
    tp1_exits: int = 0
    tp2_exits: int = 0
    sl_exits: int = 0
    trailing_exits: int = 0
    regime_flip_exits: int = 0

    # Monthly breakdown
    monthly_stats: Dict = field(default_factory=dict)


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
        logger.error("Failed to connect to database")
        return pd.DataFrame()

    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def detect_order_block(df: pd.DataFrame, idx: int, direction: str, lookback: int = 10) -> Optional[dict]:
    """Detect Order Block at current index"""
    if idx < lookback + 3:
        return None

    close_col = 'close' if 'close' in df.columns else 'Close'
    open_col = 'open' if 'open' in df.columns else 'Open'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'

    # Look for impulse move followed by mitigation
    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 3):
        bar = recent.iloc[i]
        next_bars = recent.iloc[i+1:i+4]

        if direction == 'BUY':
            # Bullish OB: last bearish candle before strong bullish move
            if bar[close_col] < bar[open_col]:  # Bearish candle
                move_up = next_bars[close_col].max() - bar[low_col]
                if move_up > 0.0015:  # 15 pips move
                    return {
                        'type': 'OB',
                        'direction': 'BUY',
                        'zone_high': bar[high_col],
                        'zone_low': bar[low_col],
                        'quality': min(100, move_up * 10000)
                    }
        else:
            # Bearish OB: last bullish candle before strong bearish move
            if bar[close_col] > bar[open_col]:  # Bullish candle
                move_down = bar[high_col] - next_bars[close_col].min()
                if move_down > 0.0015:  # 15 pips move
                    return {
                        'type': 'OB',
                        'direction': 'SELL',
                        'zone_high': bar[high_col],
                        'zone_low': bar[low_col],
                        'quality': min(100, move_down * 10000)
                    }

    return None


def detect_fvg(df: pd.DataFrame, idx: int, direction: str, lookback: int = 5) -> Optional[dict]:
    """Detect Fair Value Gap at current index"""
    if idx < lookback + 3:
        return None

    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'

    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 2):
        bar1 = recent.iloc[i]
        bar2 = recent.iloc[i+1]
        bar3 = recent.iloc[i+2]

        if direction == 'BUY':
            # Bullish FVG: gap between bar1 high and bar3 low
            gap = bar3[low_col] - bar1[high_col]
            if gap > 0.0005:  # 5 pips gap
                return {
                    'type': 'FVG',
                    'direction': 'BUY',
                    'zone_high': bar3[low_col],
                    'zone_low': bar1[high_col],
                    'quality': min(100, gap * 10000 * 2)
                }
        else:
            # Bearish FVG: gap between bar3 high and bar1 low
            gap = bar1[low_col] - bar3[high_col]
            if gap > 0.0005:  # 5 pips gap
                return {
                    'type': 'FVG',
                    'direction': 'SELL',
                    'zone_high': bar1[low_col],
                    'zone_low': bar3[high_col],
                    'quality': min(100, gap * 10000 * 2)
                }

    return None


def is_rejection_candle(bar: pd.Series, direction: str, col_map: dict) -> bool:
    """Check if bar is a rejection candle"""
    o = bar[col_map['open']]
    h = bar[col_map['high']]
    l = bar[col_map['low']]
    c = bar[col_map['close']]

    total_range = h - l
    if total_range < 0.0003:  # Min 3 pips range
        return False

    body = abs(c - o)

    if direction == 'BUY':
        lower_wick = min(o, c) - l
        return lower_wick > body and lower_wick > total_range * 0.5
    else:
        upper_wick = h - max(o, c)
        return upper_wick > body and upper_wick > total_range * 0.5


def run_backtest(df: pd.DataFrame, use_hybrid: bool = True) -> tuple:
    """Run comprehensive backtest"""

    # Initialize components
    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=6.0,
        min_bar_range_pips=4.0,
        activity_threshold=40.0,
        pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 80.0

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    # Column detection
    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Warmup (first 100 bars)
    print("      Warming up indicators...")
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    # Trading state
    trades: List[BacktestTrade] = []
    position: Optional[BacktestTrade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    print("      Processing bars...")
    total_bars = len(df) - 100

    # Process bars
    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        # Update indicators
        kalman.update(price)
        regime_info = regime_detector.update(price)

        # Progress update
        if (idx - 100) % 500 == 0:
            pct = (idx - 100) / total_bars * 100
            print(f"      Progress: {pct:.0f}% ({idx-100}/{total_bars} bars)")

        # Manage open position
        if position:
            # Check SL
            if position.direction == 'BUY' and low <= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.sl - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'LOSS'
                position.exit_reason = 'SL'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + timedelta(hours=2)
                continue

            elif position.direction == 'SELL' and high >= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.entry_price - position.sl) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'LOSS'
                position.exit_reason = 'SL'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + timedelta(hours=2)
                continue

            # Check TP1 (1.5R)
            if position.direction == 'BUY' and high >= position.tp1:
                position.exit_time = current_time
                position.exit_price = position.tp1
                position.pnl_pips = (position.tp1 - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN'
                position.exit_reason = 'TP1'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + timedelta(hours=1)
                continue

            elif position.direction == 'SELL' and low <= position.tp1:
                position.exit_time = current_time
                position.exit_price = position.tp1
                position.pnl_pips = (position.entry_price - position.tp1) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN'
                position.exit_reason = 'TP1'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + timedelta(hours=1)
                continue

            # Check regime flip exit
            if regime_info:
                if position.direction == 'BUY' and regime_info.bias == 'SELL':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (price - position.entry_price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=1)
                    continue

                elif position.direction == 'SELL' and regime_info.bias == 'BUY':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (position.entry_price - price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=1)
                    continue

        # Update drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Skip if cooldown active
        if cooldown_until and current_time < cooldown_until:
            continue

        # Skip if already in position
        if position:
            continue

        # Check Kill Zone
        in_kz, session = killzone.is_in_killzone(current_time)

        # Check hybrid mode
        can_trade_outside = False
        activity_score = 0.0
        if use_hybrid and not in_kz:
            recent_df = df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            activity_score = activity.score
            if activity.level == ActivityLevel.HIGH and activity.score >= 80:
                can_trade_outside = True

        should_trade = in_kz or can_trade_outside
        if not should_trade:
            continue

        # Check regime
        if not regime_info or not regime_info.is_tradeable:
            continue
        if regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias

        # Detect POI
        poi = detect_order_block(df, idx, direction)
        if not poi:
            poi = detect_fvg(df, idx, direction)

        if not poi:
            continue

        # Check rejection candle
        if not is_rejection_candle(bar, direction, col_map):
            continue

        # Check price in POI zone
        if direction == 'BUY':
            if not (poi['zone_low'] <= price <= poi['zone_high'] + 0.001):
                continue
        else:
            if not (poi['zone_low'] - 0.001 <= price <= poi['zone_high']):
                continue

        # Calculate position size and levels
        sl_pips = 25.0
        tp1_pips = sl_pips * 1.5  # 1.5R
        tp2_pips = sl_pips * 2.5  # 2.5R

        # Risk-based lot size (1% risk)
        risk_amount = balance * 0.01
        lot_size = risk_amount / (sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            sl_price = price - sl_pips * 0.0001
            tp1_price = price + tp1_pips * 0.0001
            tp2_price = price + tp2_pips * 0.0001
        else:
            sl_price = price + sl_pips * 0.0001
            tp1_price = price - tp1_pips * 0.0001
            tp2_price = price - tp2_pips * 0.0001

        # Open position
        position = BacktestTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=price,
            sl=sl_price,
            tp1=tp1_price,
            tp2=tp2_price,
            lot_size=lot_size,
            in_killzone=in_kz,
            session=session if in_kz else "Hybrid",
            regime=regime_info.regime.value,
            activity_score=activity_score,
            poi_type=poi['type'],
            quality_score=poi['quality']
        )

    # Close any remaining position at last bar
    if position:
        last_bar = df.iloc[-1]
        position.exit_time = last_bar.name if isinstance(last_bar.name, datetime) else pd.Timestamp(last_bar.name).to_pydatetime()
        position.exit_price = last_bar[col_map['close']]
        if position.direction == 'BUY':
            position.pnl_pips = (position.exit_price - position.entry_price) * 10000
        else:
            position.pnl_pips = (position.entry_price - position.exit_price) * 10000
        position.pnl = position.pnl_pips * position.lot_size * 10
        position.result = 'WIN' if position.pnl > 0 else 'LOSS'
        position.exit_reason = 'END_OF_TEST'
        balance += position.pnl
        trades.append(position)

    # Calculate statistics
    stats = calculate_stats(trades, balance, max_dd)

    return trades, stats, balance


def calculate_stats(trades: List[BacktestTrade], final_balance: float, max_dd: float) -> BacktestStats:
    """Calculate comprehensive statistics"""
    stats = BacktestStats()

    if not trades:
        return stats

    stats.total_trades = len(trades)
    stats.wins = sum(1 for t in trades if t.result == 'WIN')
    stats.losses = sum(1 for t in trades if t.result == 'LOSS')
    stats.breakeven = sum(1 for t in trades if t.result == 'BE')
    stats.win_rate = stats.wins / stats.total_trades * 100 if stats.total_trades > 0 else 0

    stats.total_pnl = sum(t.pnl for t in trades)
    stats.total_pips = sum(t.pnl_pips for t in trades)
    stats.max_drawdown = max_dd
    stats.max_drawdown_pct = max_dd / 10000 * 100

    winning_trades = [t.pnl for t in trades if t.pnl > 0]
    losing_trades = [abs(t.pnl) for t in trades if t.pnl < 0]

    stats.avg_win = np.mean(winning_trades) if winning_trades else 0
    stats.avg_loss = np.mean(losing_trades) if losing_trades else 0
    stats.largest_win = max(winning_trades) if winning_trades else 0
    stats.largest_loss = max(losing_trades) if losing_trades else 0

    total_wins = sum(winning_trades)
    total_losses = sum(losing_trades)
    stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Average trade duration
    durations = []
    for t in trades:
        if t.exit_time and t.entry_time:
            duration = (t.exit_time - t.entry_time).total_seconds() / 3600
            durations.append(duration)
    stats.avg_trade_duration = np.mean(durations) if durations else 0

    # By session
    london_trades = [t for t in trades if t.session == 'London']
    newyork_trades = [t for t in trades if t.session == 'New York']
    hybrid_trades = [t for t in trades if t.session == 'Hybrid']

    stats.london_trades = len(london_trades)
    stats.london_pnl = sum(t.pnl for t in london_trades)
    stats.newyork_trades = len(newyork_trades)
    stats.newyork_pnl = sum(t.pnl for t in newyork_trades)
    stats.hybrid_trades = len(hybrid_trades)
    stats.hybrid_pnl = sum(t.pnl for t in hybrid_trades)

    # By regime
    bullish_trades = [t for t in trades if t.regime == 'BULLISH']
    bearish_trades = [t for t in trades if t.regime == 'BEARISH']

    stats.bullish_trades = len(bullish_trades)
    stats.bullish_pnl = sum(t.pnl for t in bullish_trades)
    stats.bearish_trades = len(bearish_trades)
    stats.bearish_pnl = sum(t.pnl for t in bearish_trades)

    # By exit reason
    stats.tp1_exits = sum(1 for t in trades if t.exit_reason == 'TP1')
    stats.tp2_exits = sum(1 for t in trades if t.exit_reason == 'TP2')
    stats.sl_exits = sum(1 for t in trades if t.exit_reason == 'SL')
    stats.trailing_exits = sum(1 for t in trades if t.exit_reason == 'TRAILING')
    stats.regime_flip_exits = sum(1 for t in trades if t.exit_reason == 'REGIME_FLIP')

    # Monthly breakdown
    for t in trades:
        month_key = t.entry_time.strftime('%Y-%m')
        if month_key not in stats.monthly_stats:
            stats.monthly_stats[month_key] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        stats.monthly_stats[month_key]['trades'] += 1
        if t.result == 'WIN':
            stats.monthly_stats[month_key]['wins'] += 1
        stats.monthly_stats[month_key]['pnl'] += t.pnl

    return stats


async def send_telegram_report(stats: BacktestStats, trades: List[BacktestTrade], final_balance: float, period: str):
    """Send detailed report to Telegram"""
    from telegram import Bot

    try:
        bot = Bot(token=config.telegram.bot_token)

        # Main report
        msg1 = f"""🦅 <b>SURGE-WSI DETAILED BACKTEST REPORT</b>
<i>Period: {period}</i>
<i>Timeframe: H1 | Symbol: GBPUSD</i>

📊 <b>OVERALL PERFORMANCE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
<pre>
Initial Balance:    $10,000.00
Final Balance:      ${final_balance:,.2f}
Net P/L:            ${stats.total_pnl:+,.2f}
Return:             {(final_balance/10000-1)*100:+.1f}%
Total Pips:         {stats.total_pips:+.1f}
</pre>

📈 <b>TRADE STATISTICS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
<pre>
Total Trades:       {stats.total_trades}
Wins:               {stats.wins}
Losses:             {stats.losses}
Win Rate:           {stats.win_rate:.1f}%
Profit Factor:      {stats.profit_factor:.2f}
</pre>

💰 <b>PROFIT/LOSS ANALYSIS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
<pre>
Average Win:        ${stats.avg_win:,.2f}
Average Loss:       ${stats.avg_loss:,.2f}
Largest Win:        ${stats.largest_win:,.2f}
Largest Loss:       ${stats.largest_loss:,.2f}
Max Drawdown:       ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)
Avg Duration:       {stats.avg_trade_duration:.1f} hours
</pre>
"""
        await bot.send_message(chat_id=config.telegram.chat_id, text=msg1, parse_mode='HTML')

        # Session breakdown
        msg2 = f"""⏰ <b>SESSION BREAKDOWN</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
<pre>
Session      Trades    P/L
━━━━━━━━━━━━━━━━━━━━━━━━━━
London       {stats.london_trades:>6}    ${stats.london_pnl:>+8.2f}
New York     {stats.newyork_trades:>6}    ${stats.newyork_pnl:>+8.2f}
Hybrid       {stats.hybrid_trades:>6}    ${stats.hybrid_pnl:>+8.2f}
</pre>

📊 <b>REGIME BREAKDOWN</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
<pre>
Regime       Trades    P/L
━━━━━━━━━━━━━━━━━━━━━━━━━━
BULLISH      {stats.bullish_trades:>6}    ${stats.bullish_pnl:>+8.2f}
BEARISH      {stats.bearish_trades:>6}    ${stats.bearish_pnl:>+8.2f}
</pre>

🚪 <b>EXIT REASONS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
<pre>
TP1 (1.5R):         {stats.tp1_exits}
TP2 (2.5R):         {stats.tp2_exits}
Stop Loss:          {stats.sl_exits}
Regime Flip:        {stats.regime_flip_exits}
Trailing:           {stats.trailing_exits}
</pre>
"""
        await bot.send_message(chat_id=config.telegram.chat_id, text=msg2, parse_mode='HTML')

        # Monthly breakdown
        msg3 = """📅 <b>MONTHLY PERFORMANCE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
<pre>
Month      Trades  Wins   WR%     P/L
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for month, data in sorted(stats.monthly_stats.items()):
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            msg3 += f"\n{month}   {data['trades']:>5}  {data['wins']:>4}  {wr:>5.1f}%  ${data['pnl']:>+7.2f}"

        msg3 += """
</pre>
"""
        await bot.send_message(chat_id=config.telegram.chat_id, text=msg3, parse_mode='HTML')

        # Feature analysis
        kz_trades = [t for t in trades if t.in_killzone]
        hybrid_only = [t for t in trades if not t.in_killzone]
        ob_trades = [t for t in trades if t.poi_type == 'OB']
        fvg_trades = [t for t in trades if t.poi_type == 'FVG']

        kz_wr = sum(1 for t in kz_trades if t.result == 'WIN') / len(kz_trades) * 100 if kz_trades else 0
        hybrid_wr = sum(1 for t in hybrid_only if t.result == 'WIN') / len(hybrid_only) * 100 if hybrid_only else 0
        ob_wr = sum(1 for t in ob_trades if t.result == 'WIN') / len(ob_trades) * 100 if ob_trades else 0
        fvg_wr = sum(1 for t in fvg_trades if t.result == 'WIN') / len(fvg_trades) * 100 if fvg_trades else 0

        msg4 = f"""🔬 <b>FEATURE ANALYSIS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Kill Zone vs Hybrid:</b>
<pre>
Type         Trades  WinRate    P/L
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kill Zone    {len(kz_trades):>6}   {kz_wr:>5.1f}%   ${sum(t.pnl for t in kz_trades):>+8.2f}
Hybrid       {len(hybrid_only):>6}   {hybrid_wr:>5.1f}%   ${sum(t.pnl for t in hybrid_only):>+8.2f}
</pre>

<b>POI Type Performance:</b>
<pre>
POI Type     Trades  WinRate    P/L
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Order Block  {len(ob_trades):>6}   {ob_wr:>5.1f}%   ${sum(t.pnl for t in ob_trades):>+8.2f}
FVG          {len(fvg_trades):>6}   {fvg_wr:>5.1f}%   ${sum(t.pnl for t in fvg_trades):>+8.2f}
</pre>

💡 <b>KEY INSIGHTS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Generate insights
        insights = []
        if stats.win_rate >= 50:
            insights.append(f"✅ Win rate {stats.win_rate:.1f}% is above breakeven")
        else:
            insights.append(f"⚠️ Win rate {stats.win_rate:.1f}% needs improvement")

        if stats.profit_factor >= 1.5:
            insights.append(f"✅ Profit factor {stats.profit_factor:.2f} is healthy")
        elif stats.profit_factor >= 1.0:
            insights.append(f"⚠️ Profit factor {stats.profit_factor:.2f} is marginal")
        else:
            insights.append(f"❌ Profit factor {stats.profit_factor:.2f} is negative")

        if stats.newyork_pnl > stats.london_pnl:
            insights.append("📍 New York session more profitable")
        else:
            insights.append("📍 London session more profitable")

        if len(ob_trades) > 0 and len(fvg_trades) > 0:
            if ob_wr > fvg_wr:
                insights.append("📦 Order Blocks outperform FVGs")
            else:
                insights.append("📐 FVGs outperform Order Blocks")

        if stats.hybrid_trades > 0:
            if stats.hybrid_pnl > 0:
                insights.append(f"🔄 Hybrid mode added ${stats.hybrid_pnl:+.2f}")
            else:
                insights.append(f"🔄 Hybrid mode cost ${stats.hybrid_pnl:.2f}")

        for insight in insights:
            msg4 += f"{insight}\n"

        msg4 += f"""
<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>
<i>SURGE-WSI v1.0.0</i>
"""
        await bot.send_message(chat_id=config.telegram.chat_id, text=msg4, parse_mode='HTML')

        logger.info("Detailed report sent to Telegram!")

    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 60)
    print("SURGE-WSI DETAILED H1 BACKTEST")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("Features: All (Kalman, HMM, KZ, Hybrid, POI, Exit)")
    print("=" * 60)

    # Fetch data
    print("\n[1/3] Fetching H1 data...")
    symbol = "GBPUSD"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    df = await fetch_data(symbol, "H1", start, end)
    if df.empty:
        print("ERROR: No data available")
        return

    print(f"      Loaded {len(df)} H1 bars")
    print(f"      Period: {df.index[0]} to {df.index[-1]}")

    # Run backtest
    print("\n[2/3] Running backtest...")
    trades, stats, final_balance = run_backtest(df, use_hybrid=True)

    print(f"\n      Results:")
    print(f"      - Trades: {stats.total_trades}")
    print(f"      - Win Rate: {stats.win_rate:.1f}%")
    print(f"      - Net P/L: ${stats.total_pnl:+,.2f}")
    print(f"      - Final Balance: ${final_balance:,.2f}")

    # Send report
    print("\n[3/3] Sending report to Telegram...")
    period = "Jan 2025 - Jan 2026 (13 months)"
    await send_telegram_report(stats, trades, final_balance, period)

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE - Report sent to Telegram!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
