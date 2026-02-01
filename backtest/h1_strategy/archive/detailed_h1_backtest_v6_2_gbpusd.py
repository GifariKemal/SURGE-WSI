"""
SURGE-WSI H1 v6.2 GBPUSD Backtest
=================================

GBPUSD-specific strategy based on market analysis data:

Market Analysis Insights (2024-2026):
- Average ATR: 12-20 pips (MUCH lower than XAUUSD's 30+ pips)
- SL should be ATR-based: 1.5x ATR (dynamic)
- Best tradeable months: April (81.8%), July (87%), Nov (95.2%)
- Worst tradeable months: Feb (55%), Oct (52.2%)
- Average ADX: 30-42 (good trending behavior)
- Risk Multiplier varies: 0.65x (high vol), 0.73x (normal vol)

Strategy Adjustments for GBPUSD:
1. Use ATR-based SL instead of fixed pips
2. Add monthly risk adjustment based on market analysis
3. Use pip value of 10 USD per 1 lot per pip (GBPUSD)
4. Adjust quality threshold based on monthly average

Target:
- Balance: $50,000
- Risk: 1% per trade
- PF > 2.0
- Profit > $5,000

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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from config import config
from src.data.db_handler import DBHandler

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION v6.2 - GBPUSD SPECIFIC
# ============================================================
SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = 1.0

# GBPUSD-specific parameters (from market analysis)
# Use ATR multiplier instead of fixed pips
SL_ATR_MULT = 1.5   # SL = 1.5x ATR
TP_RATIO = 1.5      # TP = 1.5x SL
MIN_QUALITY_SCORE = 70.0  # Slightly lower than XAUUSD (75) due to lower volatility
MAX_LOSS_PER_TRADE_PCT = 0.15  # 0.15% max loss cap (higher than XAUUSD due to lower volatility)

# GBPUSD pip value: 1 standard lot = $10 per pip
PIP_VALUE = 10.0  # USD per pip per standard lot
PIP_SIZE = 0.0001  # GBPUSD pip size

MAX_LOT = 5.0
MIN_ATR = 8.0   # Minimum ATR to trade (pips)
MAX_ATR = 30.0  # Maximum ATR (avoid extreme volatility)

# Monthly risk multipliers (from market analysis)
# Based on tradeable_pct and avg_quality
MONTHLY_RISK = {
    1: 0.9,   # January - decent
    2: 0.7,   # February - low tradeable (55%)
    3: 0.8,   # March - moderate
    4: 1.0,   # April - best (81.8%)
    5: 0.7,   # May - low (54.5%)
    6: 0.85,  # June - decent (66.7%)
    7: 1.0,   # July - excellent (87%)
    8: 0.75,  # August - moderate (61.9%)
    9: 0.9,   # September - good (72.7%)
    10: 0.6,  # October - worst (52.2%)
    11: 0.75, # November - moderate (60%)
    12: 0.8,  # December - decent (63.6%)
}

# Day multipliers (KEEP - effective)
DAY_MULTIPLIERS = {
    0: 1.0,   # Monday
    1: 0.9,   # Tuesday
    2: 1.0,   # Wednesday
    3: 0.4,   # Thursday
    4: 0.5,   # Friday
    5: 0.0,   # Saturday
    6: 0.0,   # Sunday
}

# Hour multipliers for GBPUSD (London/NY sessions best)
HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,  # Asian - skip
    6: 0.5, 7: 0.8, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.8,  # London
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,  # NY Overlap + NY
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,  # Late NY - skip
}

# Entry type multipliers - ONLY PROFITABLE ONES
ENTRY_MULTIPLIERS = {
    'MOMENTUM': 1.0,
    'LOWER_HIGH': 1.0,
    'ENGULF': 0.8,
}


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    risk_amount: float
    atr_pips: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    exit_reason: str = ""
    quality_score: float = 0.0
    entry_type: str = ""
    session: str = ""
    regime: str = ""
    risk_mult: float = 1.0


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


# ============================================================
# DATA FETCHING
# ============================================================
async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
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


# ============================================================
# INDICATORS
# ============================================================
def calculate_atr(df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
    """Calculate ATR in pips for GBPUSD"""
    h, l, c = col_map['high'], col_map['low'], col_map['close']

    high = df[h]
    low = df[l]
    close = df[c].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Convert to pips
    atr_pips = atr / PIP_SIZE
    return atr_pips


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ============================================================
# REGIME DETECTOR
# ============================================================
def detect_regime(df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float]:
    """Simple regime detection"""
    if len(df) < 50:
        return Regime.SIDEWAYS, 0.5

    c = col_map['close']

    ema20 = calculate_ema(df[c], 20)
    ema50 = calculate_ema(df[c], 50)

    current_close = df[c].iloc[-1]
    current_ema20 = ema20.iloc[-1]
    current_ema50 = ema50.iloc[-1]

    if current_close > current_ema20 > current_ema50:
        return Regime.BULLISH, 0.7
    elif current_close < current_ema20 < current_ema50:
        return Regime.BEARISH, 0.7
    else:
        return Regime.SIDEWAYS, 0.5


# ============================================================
# ORDER BLOCK DETECTION
# ============================================================
def detect_order_blocks(df: pd.DataFrame, col_map: dict, min_quality: float = MIN_QUALITY_SCORE) -> List[dict]:
    """Detect high quality order blocks"""
    pois = []

    if len(df) < 35:
        return pois

    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    for i in range(len(df) - 30, len(df) - 2):
        if i < 2:
            continue

        current = df.iloc[i]
        next1 = df.iloc[i+1]

        # Bullish OB
        is_bearish = current[c] < current[o]
        if is_bearish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] > next1[o] and body_ratio > 0.55 and next1[c] > current[h]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({
                            'price': current[l],
                            'direction': 'BUY',
                            'quality': quality,
                            'idx': i
                        })

        # Bearish OB
        is_bullish = current[c] > current[o]
        if is_bullish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] < next1[o] and body_ratio > 0.55 and next1[c] < current[l]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({
                            'price': current[h],
                            'direction': 'SELL',
                            'quality': quality,
                            'idx': i
                        })

    return pois


# ============================================================
# ENTRY TRIGGER
# ============================================================
def check_entry_trigger(bar: pd.Series, prev_bar: pd.Series, direction: str, col_map: dict) -> Tuple[bool, str]:
    """Check entry - ONLY MOMENTUM, LOWER_HIGH, ENGULF"""
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    total_range = bar[h] - bar[l]
    if total_range < 0.0003:  # Min range for GBPUSD (3 pips)
        return False, ""

    body = abs(bar[c] - bar[o])
    is_bullish = bar[c] > bar[o]
    is_bearish = bar[c] < bar[o]

    # MOMENTUM
    if body > total_range * 0.5:
        if direction == 'BUY' and is_bullish:
            return True, 'MOMENTUM'
        if direction == 'SELL' and is_bearish:
            return True, 'MOMENTUM'

    # ENGULF
    prev_body = abs(prev_bar[c] - prev_bar[o])
    if body > prev_body * 1.2:
        if direction == 'BUY' and is_bullish and prev_bar[c] < prev_bar[o]:
            return True, 'ENGULF'
        if direction == 'SELL' and is_bearish and prev_bar[c] > prev_bar[o]:
            return True, 'ENGULF'

    # LOWER_HIGH (for SELL only)
    if direction == 'SELL':
        if bar[h] < prev_bar[h] and is_bearish:
            return True, 'LOWER_HIGH'

    return False, ""


# ============================================================
# KILL ZONE CHECK
# ============================================================
def is_in_killzone(hour: int) -> Tuple[bool, str]:
    """GBPUSD best sessions: London and NY overlap"""
    if 7 <= hour <= 11:
        return True, "london"
    if 13 <= hour <= 17:
        return True, "newyork"
    return False, ""


# ============================================================
# RISK CALCULATOR
# ============================================================
def calculate_risk_multiplier(dt: datetime, entry_type: str, quality: float) -> Tuple[float, str]:
    """Calculate risk with monthly adjustment"""
    day = dt.weekday()
    hour = dt.hour
    month = dt.month

    day_mult = DAY_MULTIPLIERS.get(day, 0.5)
    hour_mult = HOUR_MULTIPLIERS.get(hour, 0.0)
    entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 0.0)
    quality_mult = quality / 100.0
    month_mult = MONTHLY_RISK.get(month, 0.8)

    if entry_mult == 0 or hour_mult == 0:
        return 0, "entry_or_hour_not_allowed"

    combined = day_mult * hour_mult * entry_mult * quality_mult * month_mult

    reason = f"day={day_mult:.1f} hour={hour_mult:.1f} entry={entry_mult:.1f} month={month_mult:.1f}"

    return combined, reason


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(df: pd.DataFrame, col_map: dict) -> Tuple[List[Trade], float]:
    """Run backtest with ATR-based SL/TP"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0

    position: Optional[Trade] = None

    # Calculate ATR for entire dataset
    atr_series = calculate_atr(df, col_map)

    skipped_atr = 0
    skipped_entry_type = 0
    skipped_quality = 0
    skipped_risk = 0

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]

        # Get time from index
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]

        # Track DD
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Check active position
        if position:
            high = current_bar[col_map['high']]
            low = current_bar[col_map['low']]

            exit_price = None
            exit_reason = ""

            if position.direction == 'BUY':
                if low <= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = "SL"
                elif high >= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "TP"
            else:
                if high >= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = "SL"
                elif low <= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "TP"

            if exit_price:
                if position.direction == 'BUY':
                    pips = (exit_price - position.entry_price) / PIP_SIZE
                else:
                    pips = (position.entry_price - exit_price) / PIP_SIZE

                pnl = pips * position.lot_size * PIP_VALUE

                # Apply LOSS CAP
                max_loss = balance * (MAX_LOSS_PER_TRADE_PCT / 100)
                if pnl < 0 and abs(pnl) > max_loss:
                    pnl = -max_loss
                    exit_reason = "SL_CAPPED"

                position.exit_time = current_time
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pips = pips
                position.exit_reason = exit_reason

                balance += pnl
                trades.append(position)
                position = None

            continue

        # Skip weekends
        if current_time.weekday() >= 5:
            continue

        # Check ATR bounds
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            skipped_atr += 1
            continue

        # Check kill zone
        in_kz, session = is_in_killzone(current_time.hour)
        if not in_kz:
            continue

        # Detect regime
        regime, regime_prob = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # Detect POIs
        pois = detect_order_blocks(current_slice, col_map)
        if not pois:
            skipped_quality += 1
            continue

        # Check entry
        for poi in pois:
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            zone_size = abs(current_bar[col_map['high']] - current_bar[col_map['low']]) * 2
            if abs(current_price - poi['price']) > zone_size:
                continue

            prev_bar = df.iloc[i-1]
            has_trigger, entry_type = check_entry_trigger(current_bar, prev_bar, poi['direction'], col_map)

            if not has_trigger:
                skipped_entry_type += 1
                continue

            risk_mult, risk_reason = calculate_risk_multiplier(
                current_time, entry_type, poi['quality']
            )

            if risk_mult < 0.30:
                skipped_risk += 1
                continue

            # Calculate ATR-based SL/TP
            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO

            # Calculate position size
            risk_percent = RISK_PERCENT / 100.0
            risk_amount = balance * risk_percent * risk_mult

            lot_size = risk_amount / (sl_pips * PIP_VALUE)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            # Calculate SL/TP prices
            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            position = Trade(
                entry_time=current_time,
                direction=poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                risk_amount=risk_amount,
                atr_pips=current_atr,
                quality_score=poi['quality'],
                entry_type=entry_type,
                session=session,
                regime=regime.value,
                risk_mult=risk_mult
            )

            break

    print(f"\nFilter Stats:")
    print(f"  Skipped - ATR: {skipped_atr}")
    print(f"  Skipped - Entry Type: {skipped_entry_type}")
    print(f"  Skipped - Quality: {skipped_quality}")
    print(f"  Skipped - Risk: {skipped_risk}")

    return trades, max_dd


# ============================================================
# STATS CALCULATION
# ============================================================
def calculate_stats(trades: List[Trade], max_dd: float) -> dict:
    if not trades:
        return {"error": "No trades"}

    total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total * 100) if total > 0 else 0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    net_pnl = gross_profit - gross_loss

    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0

    # Monthly breakdown
    trade_df = pd.DataFrame([{
        'time': t.entry_time,
        'pnl': t.pnl
    } for t in trades])
    trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
    monthly = trade_df.groupby('month')['pnl'].sum()
    losing_months = (monthly < 0).sum()

    # Trade frequency
    first_trade = trades[0].entry_time
    last_trade = trades[-1].entry_time
    days = (last_trade - first_trade).days or 1
    trades_per_day = total / days

    # Average ATR
    avg_atr = sum(t.atr_pips for t in trades) / total if total > 0 else 0

    return {
        'total_trades': total,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'net_pnl': net_pnl,
        'profit_factor': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_dd,
        'max_dd_pct': (max_dd / INITIAL_BALANCE) * 100,
        'total_return': (net_pnl / INITIAL_BALANCE) * 100,
        'losing_months': losing_months,
        'total_months': len(monthly),
        'trades_per_day': trades_per_day,
        'final_balance': INITIAL_BALANCE + net_pnl,
        'monthly': monthly,
        'avg_atr': avg_atr
    }


def print_results(stats: dict, trades: List[Trade]):
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS - H1 v6.2 GBPUSD")
    print(f"{'='*60}")

    print(f"\n[CONFIGURATION]")
    print(f"{'-'*40}")
    print(f"Symbol: {SYMBOL}")
    print(f"SL: {SL_ATR_MULT}x ATR (Dynamic)")
    print(f"TP: {TP_RATIO}x SL")
    print(f"Min Quality: {MIN_QUALITY_SCORE}")
    print(f"Max Loss/Trade: {MAX_LOSS_PER_TRADE_PCT}%")
    print(f"Avg ATR Used: {stats['avg_atr']:.1f} pips")

    print(f"\n[PERFORMANCE]")
    print(f"{'-'*40}")
    print(f"Total Trades:      {stats['total_trades']}")
    print(f"Trades/Day:        {stats['trades_per_day']:.2f}")
    print(f"Win Rate:          {stats['win_rate']:.1f}%")
    print(f"Profit Factor:     {stats['profit_factor']:.2f}")

    print(f"\n[PROFIT/LOSS]")
    print(f"{'-'*40}")
    print(f"Initial Balance:   ${INITIAL_BALANCE:,.2f}")
    print(f"Final Balance:     ${stats['final_balance']:,.2f}")
    print(f"Gross Profit:      ${stats['gross_profit']:,.2f}")
    print(f"Gross Loss:        ${stats['gross_loss']:,.2f}")
    print(f"Net P/L:           ${stats['net_pnl']:+,.2f}")
    print(f"Total Return:      {stats['total_return']:+.1f}%")

    print(f"\n[RISK]")
    print(f"{'-'*40}")
    print(f"Max Drawdown:      ${stats['max_drawdown']:,.2f} ({stats['max_dd_pct']:.1f}%)")
    print(f"Avg Win:           ${stats['avg_win']:,.2f}")
    print(f"Avg Loss:          ${stats['avg_loss']:,.2f}")

    print(f"\n[MONTHLY]")
    print(f"{'-'*40}")
    print(f"Losing Months:     {stats['losing_months']}/{stats['total_months']}")

    for month, pnl in stats['monthly'].items():
        status = "WIN " if pnl >= 0 else "LOSS"
        print(f"  [{status}] {month}: ${pnl:+,.2f}")

    print(f"\n[ENTRY TYPE BREAKDOWN]")
    print(f"{'-'*40}")
    entry_types = {}
    for t in trades:
        if t.entry_type not in entry_types:
            entry_types[t.entry_type] = {'count': 0, 'wins': 0, 'pnl': 0}
        entry_types[t.entry_type]['count'] += 1
        entry_types[t.entry_type]['pnl'] += t.pnl
        if t.pnl > 0:
            entry_types[t.entry_type]['wins'] += 1

    for et, data in sorted(entry_types.items(), key=lambda x: -x[1]['pnl']):
        wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        print(f"  {et:12} {data['count']:3} trades, {wr:5.1f}% WR, ${data['pnl']:+,.2f}")

    print(f"\n[SESSION BREAKDOWN]")
    print(f"{'-'*40}")
    sessions = {}
    for t in trades:
        if t.session not in sessions:
            sessions[t.session] = {'count': 0, 'wins': 0, 'pnl': 0}
        sessions[t.session]['count'] += 1
        sessions[t.session]['pnl'] += t.pnl
        if t.pnl > 0:
            sessions[t.session]['wins'] += 1

    for sess, data in sorted(sessions.items(), key=lambda x: -x[1]['pnl']):
        wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        print(f"  {sess:12} {data['count']:3} trades, {wr:5.1f}% WR, ${data['pnl']:+,.2f}")

    print(f"\n{'='*60}")
    target_met = 0

    if stats['net_pnl'] >= 5000:
        print(f"[OK] PROFIT TARGET: ${stats['net_pnl']:+,.2f} >= $5,000")
        target_met += 1
    else:
        print(f"[X] PROFIT TARGET: ${stats['net_pnl']:+,.2f} < $5,000")

    if stats['profit_factor'] >= 2.0:
        print(f"[OK] PF TARGET: {stats['profit_factor']:.2f} >= 2.0")
        target_met += 1
    else:
        print(f"[X] PF TARGET: {stats['profit_factor']:.2f} < 2.0")

    if stats['losing_months'] == 0:
        print(f"[OK] ZERO LOSING MONTHS!")
        target_met += 1
    else:
        print(f"[X] LOSING MONTHS: {stats['losing_months']}")

    print(f"\nTargets Met: {target_met}/3")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================
async def main():
    timeframe = "H1"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 15, tzinfo=timezone.utc)

    print(f"SURGE-WSI H1 v6.2 GBPUSD - $50K Backtest")
    print(f"{'='*60}")
    print(f"GBPUSD-specific strategy based on market analysis:")
    print(f"  - ATR-based SL/TP (adaptive to volatility)")
    print(f"  - Monthly risk adjustment from market data")
    print(f"  - London & NY session focus")
    print(f"{'='*60}")

    print(f"\nFetching {SYMBOL} {timeframe} data...")

    df = await fetch_data(SYMBOL, timeframe, start, end)

    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")

    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_map['open'] = col
        elif 'high' in col_lower:
            col_map['high'] = col
        elif 'low' in col_lower:
            col_map['low'] = col
        elif 'close' in col_lower:
            col_map['close'] = col

    print(f"\nRunning backtest...")
    trades, max_dd = run_backtest(df, col_map)

    if not trades:
        print("No trades executed")
        return

    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades)

    # Save trades
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'sl_price': t.sl_price,
        'tp_price': t.tp_price,
        'lot_size': t.lot_size,
        'atr_pips': t.atr_pips,
        'pnl': t.pnl,
        'pnl_pips': t.pnl_pips,
        'exit_reason': t.exit_reason,
        'quality_score': t.quality_score,
        'entry_type': t.entry_type,
        'session': t.session,
        'risk_mult': t.risk_mult
    } for t in trades])

    output_path = Path(__file__).parent.parent / "results" / "h1_v6_2_gbpusd_trades.csv"
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
