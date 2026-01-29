"""Detailed H1 Backtest v2 - Optimized for More Signals
======================================================

Based on the profitable detailed_h1_backtest.py but with relaxed filters:
1. Lower Hybrid threshold (60 instead of 80)
2. Shorter cooldown (1 hour instead of 2)
3. Multiple entry types (not just rejection)
4. Wider POI tolerance

Target: 50-80 trades (vs 31 original) with 55%+ win rate

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
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    lot_size: float = 0.1
    pnl: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""
    exit_reason: str = ""
    in_killzone: bool = True
    session: str = ""
    regime: str = ""
    activity_score: float = 0.0
    poi_type: str = ""
    entry_type: str = ""
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
    avg_trade_duration: float = 0.0
    trades_per_day: float = 0.0

    london_trades: int = 0
    london_pnl: float = 0.0
    newyork_trades: int = 0
    newyork_pnl: float = 0.0
    hybrid_trades: int = 0
    hybrid_pnl: float = 0.0

    bullish_trades: int = 0
    bullish_pnl: float = 0.0
    bearish_trades: int = 0
    bearish_pnl: float = 0.0

    tp1_exits: int = 0
    tp2_exits: int = 0
    sl_exits: int = 0
    trailing_exits: int = 0
    regime_flip_exits: int = 0

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


def detect_order_block(df: pd.DataFrame, idx: int, direction: str, lookback: int = 15) -> Optional[dict]:
    """Detect Order Block - relaxed version"""
    if idx < lookback + 3:
        return None

    close_col = 'close' if 'close' in df.columns else 'Close'
    open_col = 'open' if 'open' in df.columns else 'Open'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'

    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 3):
        bar = recent.iloc[i]
        next_bars = recent.iloc[i+1:i+4]

        if direction == 'BUY':
            if bar[close_col] < bar[open_col]:  # Bearish candle
                move_up = next_bars[close_col].max() - bar[low_col]
                if move_up > 0.0010:  # 10 pips (relaxed from 15)
                    return {
                        'type': 'OB',
                        'direction': 'BUY',
                        'zone_high': bar[high_col],
                        'zone_low': bar[low_col],
                        'quality': min(100, move_up * 10000)
                    }
        else:
            if bar[close_col] > bar[open_col]:  # Bullish candle
                move_down = bar[high_col] - next_bars[close_col].min()
                if move_down > 0.0010:  # 10 pips (relaxed from 15)
                    return {
                        'type': 'OB',
                        'direction': 'SELL',
                        'zone_high': bar[high_col],
                        'zone_low': bar[low_col],
                        'quality': min(100, move_down * 10000)
                    }

    return None


def detect_fvg(df: pd.DataFrame, idx: int, direction: str, lookback: int = 8) -> Optional[dict]:
    """Detect Fair Value Gap - relaxed version"""
    if idx < lookback + 3:
        return None

    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'

    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 2):
        bar1 = recent.iloc[i]
        bar3 = recent.iloc[i+2]

        if direction == 'BUY':
            gap = bar3[low_col] - bar1[high_col]
            if gap > 0.0003:  # 3 pips (relaxed from 5)
                return {
                    'type': 'FVG',
                    'direction': 'BUY',
                    'zone_high': bar3[low_col],
                    'zone_low': bar1[high_col],
                    'quality': min(100, gap * 10000 * 2)
                }
        else:
            gap = bar1[low_col] - bar3[high_col]
            if gap > 0.0003:  # 3 pips (relaxed from 5)
                return {
                    'type': 'FVG',
                    'direction': 'SELL',
                    'zone_high': bar1[low_col],
                    'zone_low': bar3[high_col],
                    'quality': min(100, gap * 10000 * 2)
                }

    return None


def check_entry_trigger(bar: pd.Series, prev_bar: pd.Series, direction: str, col_map: dict) -> Optional[str]:
    """
    Multiple entry types - not just rejection candle

    Returns: entry_type or None
    """
    o = bar[col_map['open']]
    h = bar[col_map['high']]
    l = bar[col_map['low']]
    c = bar[col_map['close']]

    total_range = h - l
    if total_range < 0.0003:  # Min 3 pips range
        return None

    body = abs(c - o)
    is_bullish = c > o
    is_bearish = c < o

    # Previous bar
    po = prev_bar[col_map['open']]
    ph = prev_bar[col_map['high']]
    pl = prev_bar[col_map['low']]
    pc = prev_bar[col_map['close']]

    if direction == 'BUY':
        # 1. Rejection candle (original)
        lower_wick = min(o, c) - l
        if lower_wick > body and lower_wick > total_range * 0.5:
            return "REJECTION"

        # 2. Bullish momentum candle
        if is_bullish and body > total_range * 0.6:
            return "MOMENTUM"

        # 3. Bullish engulfing
        if is_bullish and c > ph and o <= pl:
            return "ENGULF"

        # 4. Higher low + bullish close
        if l > pl and is_bullish:
            return "HIGHER_LOW"

    else:  # SELL
        # 1. Rejection candle (original)
        upper_wick = h - max(o, c)
        if upper_wick > body and upper_wick > total_range * 0.5:
            return "REJECTION"

        # 2. Bearish momentum candle
        if is_bearish and body > total_range * 0.6:
            return "MOMENTUM"

        # 3. Bearish engulfing
        if is_bearish and c < pl and o >= ph:
            return "ENGULF"

        # 4. Lower high + bearish close
        if h < ph and is_bearish:
            return "LOWER_HIGH"

    return None


def run_backtest(df: pd.DataFrame, use_hybrid: bool = True) -> tuple:
    """Run backtest with relaxed parameters"""

    # Initialize components
    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=5.0,  # Relaxed from 6.0
        min_bar_range_pips=3.0,  # Relaxed from 4.0
        activity_threshold=35.0,  # Relaxed from 40.0
        pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 60.0  # Relaxed from 80.0

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Warmup
    print("      Warming up indicators...")
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    trades: List[BacktestTrade] = []
    position: Optional[BacktestTrade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    # Relaxed cooldowns
    cooldown_after_sl = timedelta(hours=1)  # Reduced from 2
    cooldown_after_tp = timedelta(minutes=30)  # Reduced from 1 hour

    print("      Processing bars...")
    total_bars = len(df) - 100

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1] if idx > 0 else bar
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        kalman.update(price)
        regime_info = regime_detector.update(price)

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
                cooldown_until = current_time + cooldown_after_sl
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
                cooldown_until = current_time + cooldown_after_sl
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
                cooldown_until = current_time + cooldown_after_tp
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
                cooldown_until = current_time + cooldown_after_tp
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
                    cooldown_until = current_time + cooldown_after_tp
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
                    cooldown_until = current_time + cooldown_after_tp
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

        # Check hybrid mode (relaxed threshold)
        can_trade_outside = False
        activity_score = 0.0
        if use_hybrid and not in_kz:
            recent_df = df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            activity_score = activity.score
            # Relaxed: 60 instead of 80, and accept MODERATE level too
            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE] and activity.score >= 60:
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

        # Detect POI (try OB first, then FVG)
        poi = detect_order_block(df, idx, direction)
        if not poi:
            poi = detect_fvg(df, idx, direction)

        if not poi:
            continue

        # Check price near POI zone (wider tolerance: 15 pips)
        poi_tolerance = 0.0015
        if direction == 'BUY':
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue
        else:
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue

        # Check entry trigger (multiple types now)
        entry_type = check_entry_trigger(bar, prev_bar, direction, col_map)
        if not entry_type:
            continue

        # Calculate position size and levels
        sl_pips = 25.0
        tp1_pips = sl_pips * 1.5  # 1.5R
        tp2_pips = sl_pips * 2.5  # 2.5R

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
            regime=regime_info.regime.value if hasattr(regime_info.regime, 'value') else str(regime_info.regime),
            activity_score=activity_score,
            poi_type=poi['type'],
            entry_type=entry_type,
            quality_score=poi['quality']
        )

    # Close remaining position
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

    stats = calculate_stats(trades, balance, max_dd, df)
    return trades, stats, balance


def calculate_stats(trades: List[BacktestTrade], final_balance: float, max_dd: float, df: pd.DataFrame) -> BacktestStats:
    """Calculate statistics"""
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

    durations = []
    for t in trades:
        if t.exit_time and t.entry_time:
            duration = (t.exit_time - t.entry_time).total_seconds() / 3600
            durations.append(duration)
    stats.avg_trade_duration = np.mean(durations) if durations else 0

    # Trades per day
    if len(df) > 0:
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            stats.trades_per_day = stats.total_trades / days

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

    # Monthly
    for t in trades:
        month_key = t.entry_time.strftime('%Y-%m')
        if month_key not in stats.monthly_stats:
            stats.monthly_stats[month_key] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pips': 0.0}
        stats.monthly_stats[month_key]['trades'] += 1
        if t.result == 'WIN':
            stats.monthly_stats[month_key]['wins'] += 1
        stats.monthly_stats[month_key]['pnl'] += t.pnl
        stats.monthly_stats[month_key]['pips'] += t.pnl_pips

    return stats


def print_report(stats: BacktestStats, final_balance: float, trades: List[BacktestTrade]):
    """Print detailed report"""
    print()
    print("=" * 70)
    print("H1 DETAILED BACKTEST v2 RESULTS")
    print("(Relaxed parameters for more signals)")
    print("=" * 70)
    print()

    print("OVERALL PERFORMANCE")
    print("-" * 50)
    print(f"Initial Balance:     $10,000.00")
    print(f"Final Balance:       ${final_balance:,.2f}")
    print(f"Net P/L:             ${stats.total_pnl:+,.2f}")
    print(f"Return:              {(final_balance/10000-1)*100:+.1f}%")
    print(f"Total Pips:          {stats.total_pips:+.1f}")
    print()

    print("TRADE STATISTICS")
    print("-" * 50)
    print(f"Total Trades:        {stats.total_trades} ({stats.trades_per_day:.2f}/day)")
    print(f"Wins:                {stats.wins}")
    print(f"Losses:              {stats.losses}")
    print(f"Win Rate:            {stats.win_rate:.1f}%")
    print(f"Profit Factor:       {stats.profit_factor:.2f}")
    print()

    print("P/L ANALYSIS")
    print("-" * 50)
    print(f"Average Win:         ${stats.avg_win:,.2f}")
    print(f"Average Loss:        ${stats.avg_loss:,.2f}")
    print(f"Largest Win:         ${stats.largest_win:,.2f}")
    print(f"Largest Loss:        ${stats.largest_loss:,.2f}")
    print(f"Max Drawdown:        ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)")
    print(f"Avg Duration:        {stats.avg_trade_duration:.1f} hours")
    print()

    # Entry type breakdown
    entry_types = {}
    for t in trades:
        et = t.entry_type
        if et not in entry_types:
            entry_types[et] = {'count': 0, 'wins': 0, 'pnl': 0}
        entry_types[et]['count'] += 1
        if t.result == 'WIN':
            entry_types[et]['wins'] += 1
        entry_types[et]['pnl'] += t.pnl

    print("ENTRY TYPE BREAKDOWN")
    print("-" * 60)
    print(f"{'Type':<15} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for et, data in sorted(entry_types.items(), key=lambda x: -x[1]['pnl']):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"{et:<15} {data['count']:>8} {data['wins']:>6} {wr:>7.1f}% ${data['pnl']:>+10.2f}")
    print()

    print("SESSION BREAKDOWN")
    print("-" * 50)
    print(f"{'Session':<15} {'Trades':>8} {'P/L':>12}")
    print(f"{'London':<15} {stats.london_trades:>8} ${stats.london_pnl:>+10.2f}")
    print(f"{'New York':<15} {stats.newyork_trades:>8} ${stats.newyork_pnl:>+10.2f}")
    print(f"{'Hybrid':<15} {stats.hybrid_trades:>8} ${stats.hybrid_pnl:>+10.2f}")
    print()

    print("EXIT REASONS")
    print("-" * 50)
    print(f"TP1 (1.5R):          {stats.tp1_exits}")
    print(f"TP2 (2.5R):          {stats.tp2_exits}")
    print(f"Stop Loss:           {stats.sl_exits}")
    print(f"Regime Flip:         {stats.regime_flip_exits}")
    print()

    print("MONTHLY PERFORMANCE")
    print("-" * 70)
    print(f"{'Month':<10} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Pips':>10} {'P/L':>12}")
    print("-" * 70)

    losing_months = 0
    for month, data in sorted(stats.monthly_stats.items()):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        status = "[-]" if data['pnl'] < 0 else "[+]"
        if data['pnl'] < 0:
            losing_months += 1
        print(f"{month:<10} {data['trades']:>8} {data['wins']:>6} {wr:>7.1f}% {data['pips']:>+9.1f} ${data['pnl']:>+10.2f} {status}")

    print("-" * 70)
    print(f"Losing months: {losing_months}/{len(stats.monthly_stats)}")
    print()


async def send_telegram_report(stats: BacktestStats, trades: List[BacktestTrade], final_balance: float):
    """Send report to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        # Entry breakdown
        entry_types = {}
        for t in trades:
            et = t.entry_type
            if et not in entry_types:
                entry_types[et] = {'count': 0, 'pnl': 0}
            entry_types[et]['count'] += 1
            entry_types[et]['pnl'] += t.pnl

        entry_str = "\n".join([f"• {k}: {v['count']} trades, ${v['pnl']:+.2f}" for k, v in sorted(entry_types.items(), key=lambda x: -x[1]['pnl'])])

        msg = f"""*H1 BACKTEST v2 (Relaxed)*
Period: Jan 2025 - Jan 2026 (13 months)

*PERFORMANCE*
• Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
• Win Rate: {stats.win_rate:.1f}%
• Net P/L: ${stats.total_pnl:+,.2f} ({(final_balance/10000-1)*100:+.1f}%)
• Profit Factor: {stats.profit_factor:.2f}
• Max DD: ${stats.max_drawdown:,.2f}

*ENTRY TYPES*
{entry_str}

*SESSION*
• London: {stats.london_trades} trades, ${stats.london_pnl:+.2f}
• New York: {stats.newyork_trades} trades, ${stats.newyork_pnl:+.2f}
• Hybrid: {stats.hybrid_trades} trades, ${stats.hybrid_pnl:+.2f}

*vs Original*
Original: 31 trades, 61% WR, +$1,799
This v2: {stats.total_trades} trades, {stats.win_rate:.0f}% WR, ${stats.total_pnl:+,.0f}

Final Balance: ${final_balance:,.2f}
"""
        await telegram.send(msg)
        logger.info("Report sent to Telegram!")

    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI DETAILED H1 BACKTEST v2")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("Strategy: Original + Relaxed Filters for More Signals")
    print("=" * 70)

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

    print("\n[2/3] Running backtest...")
    trades, stats, final_balance = run_backtest(df, use_hybrid=True)

    print_report(stats, final_balance, trades)

    print("[3/3] Sending report to Telegram...")
    await send_telegram_report(stats, trades, final_balance)

    print("=" * 70)
    print("BACKTEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
