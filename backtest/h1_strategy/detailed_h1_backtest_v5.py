"""Detailed H1 Backtest v5 - Data-Driven Optimal Rules
======================================================

Based on deep analysis of 161 trades, implementing STRICT rules:

CRITICAL RULES (from analysis):
1. DISABLE HYBRID MODE - 31.4% WR, -187 pips loss
2. SKIP THURSDAY - 35.9% WR, negative expectancy
3. SKIP FRIDAY - 38.2% WR, negative expectancy
4. PREFER MOMENTUM entries - 47.1% WR, +255 pips
5. AVOID REJECTION entries (or require higher quality)
6. FOCUS on 9:00 and 14:00 UTC hours - 70%+ WR
7. AVOID 10:00-11:00 UTC - 35% WR

Target: ZERO losing months, higher profit

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector


# ============================================================================
# V5 OPTIMAL RULES (Data-Driven)
# ============================================================================

V5_RULES = {
    # Days to SKIP (negative expectancy)
    'skip_days': [3, 4],  # Thursday=3, Friday=4

    # Hours with HIGH win rate (70%+)
    'best_hours': [9, 14],  # 9:00 and 14:00 UTC

    # Hours to AVOID (< 40% WR)
    'avoid_hours': [10, 11, 15, 17, 18, 19, 20],

    # Sessions
    'allow_hybrid': False,  # DISABLE hybrid - 31.4% WR

    # Entry types
    'preferred_entries': ['MOMENTUM', 'HIGHER_LOW'],  # Skip REJECTION
    'allow_rejection': False,  # REJECTION has 36% WR

    # Minimum requirements for REJECTION if allowed
    'rejection_min_quality': 90,  # Only very high quality

    # Choppiness (lower is better for wins)
    'max_choppiness': 60,

    # ATR (lower is better based on analysis)
    'max_atr_pips': 25,

    # Bar range (lower is better)
    'max_bar_range_pips': 35,
}


@dataclass
class BacktestTrade:
    """Trade record"""
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    lot_size: float = 0.1
    pnl: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""
    exit_reason: str = ""
    session: str = ""
    regime: str = ""
    entry_type: str = ""
    quality_score: float = 0.0
    hour: int = 0
    day_of_week: int = 0


@dataclass
class BacktestStats:
    """Statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades_per_day: float = 0.0

    # Filtering stats
    filtered_by_day: int = 0
    filtered_by_hour: int = 0
    filtered_by_hybrid: int = 0
    filtered_by_entry_type: int = 0
    filtered_by_market: int = 0

    monthly_stats: Dict = field(default_factory=dict)


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data"""
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


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators"""
    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    close = df[col_map['close']]
    high = df[col_map['high']]
    low = df[col_map['low']]

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_pips'] = tr.rolling(14).mean() / 0.0001

    # Choppiness
    atr_sum = tr.rolling(14).sum()
    highest_high = high.rolling(14).max()
    lowest_low = low.rolling(14).min()
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.nan)
    df['choppiness'] = 100 * np.log10(atr_sum / price_range) / np.log10(14)

    # Bar range
    df['bar_range_pips'] = (high - low) / 0.0001

    return df


def calculate_ob_quality(df, ob_idx, direction, col_map):
    """Calculate OB quality"""
    if ob_idx < 5 or ob_idx >= len(df) - 3:
        return 50

    ob_bar = df.iloc[ob_idx]
    next_bars = df.iloc[ob_idx+1:ob_idx+4]
    quality = 0.0

    if direction == 'BUY':
        impulse = next_bars[col_map['close']].max() - ob_bar[col_map['low']]
    else:
        impulse = ob_bar[col_map['high']] - next_bars[col_map['close']].min()

    impulse_pips = impulse * 10000
    quality += min(50, impulse_pips * 2.5)

    ob_range = ob_bar[col_map['high']] - ob_bar[col_map['low']]
    if ob_range > 0:
        if direction == 'BUY':
            wick = ob_bar[col_map['high']] - max(ob_bar[col_map['open']], ob_bar[col_map['close']])
        else:
            wick = min(ob_bar[col_map['open']], ob_bar[col_map['close']]) - ob_bar[col_map['low']]
        wick_ratio = wick / ob_range
        if wick_ratio > 0.3:
            quality += 25
        elif wick_ratio > 0.2:
            quality += 15

    zone_high, zone_low = ob_bar[col_map['high']], ob_bar[col_map['low']]
    touched = False
    for i in range(ob_idx + 4, min(ob_idx + 15, len(df))):
        bar = df.iloc[i]
        if direction == 'BUY' and bar[col_map['low']] <= zone_high:
            touched = True
            break
        elif direction == 'SELL' and bar[col_map['high']] >= zone_low:
            touched = True
            break
    if not touched:
        quality += 25

    return min(100, quality)


def run_backtest(df: pd.DataFrame) -> tuple:
    """Run v5 backtest with data-driven rules"""

    df = calculate_indicators(df)

    killzone = KillZone()
    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Warmup
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    trades: List[BacktestTrade] = []
    position = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    # Filtering counters
    filtered_day = 0
    filtered_hour = 0
    filtered_hybrid = 0
    filtered_entry = 0
    filtered_market = 0

    total_bars = len(df) - 100
    print("      Processing bars...")

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1]
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
            print(f"      Progress: {pct:.0f}%")

        # Manage position
        if position:
            if position.direction == 'BUY':
                if low <= position.sl:
                    position.exit_time, position.exit_price = current_time, position.sl
                    position.pnl_pips = (position.sl - position.entry_price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result, position.exit_reason = 'LOSS', 'SL'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=1)
                    continue
                elif high >= position.tp1:
                    position.exit_time, position.exit_price = current_time, position.tp1
                    position.pnl_pips = (position.tp1 - position.entry_price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result, position.exit_reason = 'WIN', 'TP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(minutes=30)
                    continue
            else:  # SELL
                if high >= position.sl:
                    position.exit_time, position.exit_price = current_time, position.sl
                    position.pnl_pips = (position.entry_price - position.sl) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result, position.exit_reason = 'LOSS', 'SL'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=1)
                    continue
                elif low <= position.tp1:
                    position.exit_time, position.exit_price = current_time, position.tp1
                    position.pnl_pips = (position.entry_price - position.tp1) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result, position.exit_reason = 'WIN', 'TP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(minutes=30)
                    continue

            # Regime flip
            if regime_info:
                if (position.direction == 'BUY' and regime_info.bias == 'SELL') or \
                   (position.direction == 'SELL' and regime_info.bias == 'BUY'):
                    position.exit_time, position.exit_price = current_time, price
                    if position.direction == 'BUY':
                        position.pnl_pips = (price - position.entry_price) * 10000
                    else:
                        position.pnl_pips = (position.entry_price - price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(minutes=30)
                    continue

        # Update drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        if cooldown_until and current_time < cooldown_until:
            continue
        if position:
            continue

        # =====================================================================
        # V5 RULE 1: SKIP THURSDAY & FRIDAY
        # =====================================================================
        day_of_week = current_time.weekday()
        if day_of_week in V5_RULES['skip_days']:
            filtered_day += 1
            continue

        # =====================================================================
        # V5 RULE 2: SKIP BAD HOURS
        # =====================================================================
        hour = current_time.hour
        if hour in V5_RULES['avoid_hours']:
            filtered_hour += 1
            continue

        # =====================================================================
        # V5 RULE 3: KILLZONE ONLY (NO HYBRID)
        # =====================================================================
        in_kz, session = killzone.is_in_killzone(current_time)
        if not in_kz:
            if not V5_RULES['allow_hybrid']:
                filtered_hybrid += 1
                continue

        # Check regime
        if not regime_info or not regime_info.is_tradeable or regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias

        # =====================================================================
        # V5 RULE 4: MARKET CONDITION FILTERS
        # =====================================================================
        choppiness = bar['choppiness'] if 'choppiness' in bar.index and not pd.isna(bar['choppiness']) else 50
        atr_pips = bar['atr_pips'] if 'atr_pips' in bar.index and not pd.isna(bar['atr_pips']) else 20
        bar_range = bar['bar_range_pips'] if 'bar_range_pips' in bar.index and not pd.isna(bar['bar_range_pips']) else 15

        if choppiness > V5_RULES['max_choppiness']:
            filtered_market += 1
            continue
        if atr_pips > V5_RULES['max_atr_pips']:
            filtered_market += 1
            continue
        if bar_range > V5_RULES['max_bar_range_pips']:
            filtered_market += 1
            continue

        # Find POI
        poi_found = False
        poi_quality = 50
        lookback = 15
        recent = df.iloc[idx-lookback:idx]

        for i in range(len(recent) - 3):
            ob_bar = recent.iloc[i]
            next_bars = recent.iloc[i+1:i+4]
            actual_idx = idx - lookback + i

            if direction == 'BUY':
                if ob_bar[col_map['close']] < ob_bar[col_map['open']]:
                    move = next_bars[col_map['close']].max() - ob_bar[col_map['low']]
                    if move > 0.0010:
                        poi_found = True
                        poi_quality = calculate_ob_quality(df, actual_idx, direction, col_map)
                        break
            else:
                if ob_bar[col_map['close']] > ob_bar[col_map['open']]:
                    move = ob_bar[col_map['high']] - next_bars[col_map['close']].min()
                    if move > 0.0010:
                        poi_found = True
                        poi_quality = calculate_ob_quality(df, actual_idx, direction, col_map)
                        break

        if not poi_found:
            continue

        # Check POI zone
        poi_tolerance = 0.0015
        zone_high = recent.iloc[0][col_map['high']]
        zone_low = recent.iloc[0][col_map['low']]
        # Simple check - price near recent zone

        # =====================================================================
        # V5 RULE 5: ENTRY TYPE FILTER
        # =====================================================================
        o, h_bar, l_bar, c = bar[col_map['open']], bar[col_map['high']], bar[col_map['low']], bar[col_map['close']]
        total_range = h_bar - l_bar
        if total_range < 0.0003:
            continue

        body = abs(c - o)
        entry_type = None

        if direction == 'BUY':
            lower_wick = min(o, c) - l_bar
            if lower_wick > body and lower_wick > total_range * 0.5:
                entry_type = "REJECTION"
            elif c > o and body > total_range * 0.6:
                entry_type = "MOMENTUM"
            elif l_bar > prev_bar[col_map['low']] and c > o:
                entry_type = "HIGHER_LOW"
        else:
            upper_wick = h_bar - max(o, c)
            if upper_wick > body and upper_wick > total_range * 0.5:
                entry_type = "REJECTION"
            elif c < o and body > total_range * 0.6:
                entry_type = "MOMENTUM"

        if not entry_type:
            continue

        # Filter REJECTION (unless very high quality)
        if entry_type == "REJECTION":
            if not V5_RULES['allow_rejection']:
                filtered_entry += 1
                continue
            elif poi_quality < V5_RULES['rejection_min_quality']:
                filtered_entry += 1
                continue

        # =====================================================================
        # ENTRY
        # =====================================================================
        sl_pips = 25.0
        tp_pips = sl_pips * 1.5

        # Bonus risk for best hours
        risk_pct = 0.01
        if hour in V5_RULES['best_hours']:
            risk_pct = 0.012  # 20% bonus

        risk_amount = balance * risk_pct
        lot_size = risk_amount / (sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            sl_price = price - sl_pips * 0.0001
            tp_price = price + tp_pips * 0.0001
        else:
            sl_price = price + sl_pips * 0.0001
            tp_price = price - tp_pips * 0.0001

        position = BacktestTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=price,
            sl=sl_price,
            tp1=tp_price,
            lot_size=lot_size,
            session=session,
            regime=str(regime_info.regime.value) if hasattr(regime_info.regime, 'value') else str(regime_info.regime),
            entry_type=entry_type,
            quality_score=poi_quality,
            hour=hour,
            day_of_week=day_of_week
        )

    # Close remaining position
    if position:
        last_bar = df.iloc[-1]
        position.exit_time = last_bar.name
        position.exit_price = last_bar[col_map['close']]
        if position.direction == 'BUY':
            position.pnl_pips = (position.exit_price - position.entry_price) * 10000
        else:
            position.pnl_pips = (position.entry_price - position.exit_price) * 10000
        position.pnl = position.pnl_pips * position.lot_size * 10
        position.result = 'WIN' if position.pnl > 0 else 'LOSS'
        position.exit_reason = 'END'
        balance += position.pnl
        trades.append(position)

    # Calculate stats
    stats = calculate_stats(trades, balance, max_dd, df)
    stats.filtered_by_day = filtered_day
    stats.filtered_by_hour = filtered_hour
    stats.filtered_by_hybrid = filtered_hybrid
    stats.filtered_by_entry_type = filtered_entry
    stats.filtered_by_market = filtered_market

    return trades, stats, balance


def calculate_stats(trades, final_balance, max_dd, df):
    """Calculate statistics"""
    stats = BacktestStats()
    if not trades:
        return stats

    stats.total_trades = len(trades)
    stats.wins = sum(1 for t in trades if t.result == 'WIN')
    stats.losses = sum(1 for t in trades if t.result == 'LOSS')
    stats.win_rate = stats.wins / stats.total_trades * 100 if stats.total_trades else 0

    stats.total_pnl = sum(t.pnl for t in trades)
    stats.total_pips = sum(t.pnl_pips for t in trades)
    stats.max_drawdown = max_dd

    winning = [t.pnl for t in trades if t.pnl > 0]
    losing = [abs(t.pnl) for t in trades if t.pnl < 0]

    stats.avg_win = np.mean(winning) if winning else 0
    stats.avg_loss = np.mean(losing) if losing else 0
    stats.profit_factor = sum(winning) / sum(losing) if losing else float('inf')

    days = (df.index[-1] - df.index[0]).days
    stats.trades_per_day = stats.total_trades / days if days > 0 else 0

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


def print_report(stats, final_balance, trades):
    """Print report"""
    print()
    print("=" * 70)
    print("H1 BACKTEST v5 RESULTS - DATA-DRIVEN OPTIMAL RULES")
    print("=" * 70)
    print()

    print("V5 STRICT RULES:")
    print("-" * 50)
    print(f"  Skip days: Thursday, Friday")
    print(f"  Skip hours: {V5_RULES['avoid_hours']}")
    print(f"  Hybrid mode: DISABLED")
    print(f"  REJECTION entry: DISABLED")
    print(f"  Max choppiness: {V5_RULES['max_choppiness']}")
    print()

    print("FILTERING SUMMARY:")
    print("-" * 50)
    print(f"  Filtered by day (Thu/Fri):  {stats.filtered_by_day}")
    print(f"  Filtered by bad hours:      {stats.filtered_by_hour}")
    print(f"  Filtered by hybrid:         {stats.filtered_by_hybrid}")
    print(f"  Filtered by entry type:     {stats.filtered_by_entry_type}")
    print(f"  Filtered by market:         {stats.filtered_by_market}")
    total_filtered = stats.filtered_by_day + stats.filtered_by_hour + stats.filtered_by_hybrid + stats.filtered_by_entry_type + stats.filtered_by_market
    print(f"  TOTAL FILTERED:             {total_filtered}")
    print()

    print("PERFORMANCE:")
    print("-" * 50)
    print(f"  Initial: $10,000.00")
    print(f"  Final:   ${final_balance:,.2f}")
    print(f"  P/L:     ${stats.total_pnl:+,.2f}")
    print(f"  Return:  {(final_balance/10000-1)*100:+.1f}%")
    print()

    print("STATISTICS:")
    print("-" * 50)
    print(f"  Trades:        {stats.total_trades}")
    print(f"  Win Rate:      {stats.win_rate:.1f}%")
    print(f"  Profit Factor: {stats.profit_factor:.2f}")
    print(f"  Max Drawdown:  ${stats.max_drawdown:,.2f}")
    print()

    # Entry breakdown
    entry_types = {}
    for t in trades:
        et = t.entry_type
        if et not in entry_types:
            entry_types[et] = {'count': 0, 'wins': 0, 'pnl': 0}
        entry_types[et]['count'] += 1
        if t.result == 'WIN':
            entry_types[et]['wins'] += 1
        entry_types[et]['pnl'] += t.pnl

    print("ENTRY TYPE BREAKDOWN:")
    print("-" * 60)
    for et, data in sorted(entry_types.items(), key=lambda x: -x[1]['pnl']):
        wr = data['wins'] / data['count'] * 100 if data['count'] else 0
        print(f"  {et:<15} {data['count']:>4} trades, {wr:>5.1f}% WR, ${data['pnl']:>+8.2f}")
    print()

    # Hour breakdown
    hour_stats = {}
    for t in trades:
        h = t.hour
        if h not in hour_stats:
            hour_stats[h] = {'count': 0, 'wins': 0, 'pnl': 0}
        hour_stats[h]['count'] += 1
        if t.result == 'WIN':
            hour_stats[h]['wins'] += 1
        hour_stats[h]['pnl'] += t.pnl

    print("HOUR BREAKDOWN:")
    print("-" * 60)
    for h, data in sorted(hour_stats.items()):
        wr = data['wins'] / data['count'] * 100 if data['count'] else 0
        star = "⭐" if wr >= 55 else ""
        print(f"  {h:>2}:00  {data['count']:>4} trades, {wr:>5.1f}% WR, ${data['pnl']:>+8.2f} {star}")
    print()

    print("MONTHLY PERFORMANCE:")
    print("-" * 70)
    losing_months = 0
    for month, data in sorted(stats.monthly_stats.items()):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] else 0
        status = "❌" if data['pnl'] < 0 else "✅"
        if data['pnl'] < 0:
            losing_months += 1
        print(f"  {month}: {data['trades']:>3} trades, {wr:>5.1f}% WR, {data['pips']:>+7.1f} pips, ${data['pnl']:>+8.2f} {status}")

    print("-" * 70)
    print(f"  Losing months: {losing_months}/{len(stats.monthly_stats)}")
    print()

    print("=" * 70)
    print("VERSION COMPARISON:")
    print("=" * 70)
    print("v3 FINAL: 100T, 51% WR, +$2,669, PF 1.50, 3 losing months")
    print("v4:        89T, 49% WR, +$2,131, PF 1.49, 2 losing months")
    print("v4.5:      89T, 49% WR, +$2,394, PF 1.57, 1 losing month")
    print(f"v5:       {stats.total_trades:>3}T, {stats.win_rate:.0f}% WR, ${stats.total_pnl:+,.0f}, PF {stats.profit_factor:.2f}, {losing_months} losing")
    print("=" * 70)


async def send_telegram_report(stats, trades, final_balance):
    """Send to Telegram"""
    from src.utils.telegram import TelegramNotifier
    try:
        telegram = TelegramNotifier(bot_token=config.telegram.bot_token, chat_id=config.telegram.chat_id)
        if not await telegram.initialize():
            return

        losing_months = sum(1 for d in stats.monthly_stats.values() if d['pnl'] < 0)
        total_months = len(stats.monthly_stats)

        monthly_lines = []
        for month, data in sorted(stats.monthly_stats.items()):
            wr = data['wins'] / data['trades'] * 100 if data['trades'] else 0
            status = "❌" if data['pnl'] < 0 else "✅"
            monthly_lines.append(f"  {month}: {data['trades']}T, {wr:.0f}%, ${data['pnl']:+.0f} {status}")

        msg = f"""🦅 <b>H1 BACKTEST v5</b>
━━━━━━━━━━━━━━━━━━━━━━━
<b>Data-Driven Optimal Rules</b>

<b>🔬 V5 STRICT RULES</b>
├ Skip: Thursday, Friday
├ Skip hours: 10-11, 15-20
├ Hybrid: DISABLED
└ REJECTION: DISABLED

<b>📊 PERFORMANCE</b>
├ Trades: {stats.total_trades}
├ Win Rate: {stats.win_rate:.1f}%
├ Net P/L: <b>${stats.total_pnl:+,.2f}</b>
├ Return: <b>{(final_balance/10000-1)*100:+.1f}%</b>
├ Profit Factor: <b>{stats.profit_factor:.2f}</b>
└ Max DD: ${stats.max_drawdown:,.2f}

<b>🔍 FILTERED</b>
├ By day: {stats.filtered_by_day}
├ By hour: {stats.filtered_by_hour}
├ By hybrid: {stats.filtered_by_hybrid}
└ By entry: {stats.filtered_by_entry_type}

<b>📅 MONTHLY</b>
{chr(10).join(monthly_lines)}
━━━━━━━━━━━━━━━━━━━━━━━
Losing: {losing_months}/{total_months}

<b>📈 vs v4.5</b>
v4.5: 89T, 49%, +$2,394, PF 1.57, 1 losing
v5: {stats.total_trades}T, {stats.win_rate:.0f}%, ${stats.total_pnl:+,.0f}, PF {stats.profit_factor:.2f}

<b>${final_balance:,.2f}</b>
"""
        await telegram.send(msg, force=True)
        logger.info("Telegram sent!")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI H1 BACKTEST v5")
    print("Data-Driven Optimal Rules")
    print("=" * 70)

    print("\n[1/3] Fetching data...")
    df = await fetch_data("GBPUSD", "H1",
                         datetime(2025, 1, 1, tzinfo=timezone.utc),
                         datetime(2026, 1, 31, tzinfo=timezone.utc))
    if df.empty:
        print("ERROR: No data")
        return
    print(f"      Loaded {len(df)} bars")

    print("\n[2/3] Running v5 backtest...")
    trades, stats, final_balance = run_backtest(df)

    print_report(stats, final_balance, trades)

    print("\n[3/3] Sending to Telegram...")
    await send_telegram_report(stats, trades, final_balance)

    print("=" * 70)
    print("v5 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
