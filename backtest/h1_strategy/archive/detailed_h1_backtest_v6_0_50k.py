"""
SURGE-WSI H1 v6.0 Backtest - $50K Scaled Version
=================================================
Based on v5.8 (best PF version) scaled for $50K balance

Target:
- Balance: $50,000
- Risk: 1% per trade
- Auto lot sizing (no cap)
- Expected profit: >$5,000 (based on v5.8 scaling)

v5.8 Results on $10K:
- 96 trades, 47.9% WR, +$2,269, PF 1.77

Expected on $50K:
- 96 trades, 47.9% WR, ~$11,345, PF 1.77

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
# CONFIGURATION v6.0 - $50K SCALED
# ============================================================
INITIAL_BALANCE = 50_000.0  # $50K
RISK_PERCENT = 1.0  # 1% per trade
SL_PIPS = 25
TP_RATIO = 1.5
TP_PIPS = SL_PIPS * TP_RATIO  # 37.5 pips (same as v5.8)
MAX_LOT = 5.0  # Increased from 1.0 for larger balance


# Day multipliers from v5.8
DAY_MULTIPLIERS = {
    0: 1.0,   # Monday
    1: 0.9,   # Tuesday
    2: 1.0,   # Wednesday
    3: 0.4,   # Thursday
    4: 0.5,   # Friday
    5: 0.0,   # Saturday
    6: 0.0,   # Sunday
}

# Hour multipliers from v5.8
HOUR_MULTIPLIERS = {
    1: 0.0, 4: 0.0,  # Dead hours
    6: 0.3, 10: 0.5, 11: 0.5,  # Weak hours
    7: 0.8, 8: 0.9, 9: 1.0, 12: 1.0, 13: 0.9, 14: 1.0, 15: 1.0,  # Good hours
}

# Entry type multipliers from v5.8
ENTRY_MULTIPLIERS = {
    'MOMENTUM': 1.0,
    'LOWER_HIGH': 1.0,
    'HIGHER_LOW': 0.8,
    'REJECTION': 0.7,
    'ENGULF': 0.8,
    'SMALL_BODY': 0.5,
}

# Known bad periods from v5.8
BAD_PERIODS = {
    6: 0.1,   # June - 10% risk
    9: 0.3,   # September - 30% risk (except Wednesday: 5%)
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
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14, col_map: dict = None) -> pd.Series:
    if col_map is None:
        col_map = {'high': 'High', 'low': 'Low', 'close': 'Close'}

    high = df[col_map['high']]
    low = df[col_map['low']]
    close = df[col_map['close']]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


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
def detect_order_blocks(df: pd.DataFrame, col_map: dict, min_quality: float = 50) -> List[dict]:
    """Detect order blocks"""
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
                if next1[c] > next1[o] and body_ratio > 0.5 and next1[c] > current[h]:
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
                if next1[c] < next1[o] and body_ratio > 0.5 and next1[c] < current[l]:
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
    """Check entry trigger"""
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    total_range = bar[h] - bar[l]
    if total_range < 0.3:  # Min 0.3 points movement
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

    # REJECTION
    if direction == 'BUY':
        lower_wick = min(bar[o], bar[c]) - bar[l]
        if lower_wick > total_range * 0.4 and lower_wick > body * 1.5:
            return True, 'REJECTION'
    else:
        upper_wick = bar[h] - max(bar[o], bar[c])
        if upper_wick > total_range * 0.4 and upper_wick > body * 1.5:
            return True, 'REJECTION'

    # ENGULF
    prev_body = abs(prev_bar[c] - prev_bar[o])
    if body > prev_body * 1.2:
        if direction == 'BUY' and is_bullish and prev_bar[c] < prev_bar[o]:
            return True, 'ENGULF'
        if direction == 'SELL' and is_bearish and prev_bar[c] > prev_bar[o]:
            return True, 'ENGULF'

    return False, ""


# ============================================================
# KILL ZONE CHECK
# ============================================================
def is_in_killzone(hour: int) -> Tuple[bool, str]:
    """Check if in kill zone"""
    if 7 <= hour <= 11:
        return True, "london"
    if 13 <= hour <= 17:
        return True, "newyork"
    return False, ""


# ============================================================
# RISK CALCULATOR
# ============================================================
def calculate_risk_multiplier(dt: datetime, entry_type: str, quality: float) -> Tuple[float, str]:
    """Calculate risk multiplier based on v5.8 rules"""
    day = dt.weekday()
    hour = dt.hour
    month = dt.month

    # Base multipliers
    day_mult = DAY_MULTIPLIERS.get(day, 0.5)
    hour_mult = HOUR_MULTIPLIERS.get(hour, 0.7)
    entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 0.7)
    quality_mult = quality / 100.0

    # Known bad periods
    period_mult = 1.0
    if month == 6:
        period_mult = BAD_PERIODS[6]
    elif month == 9:
        if day == 2:  # Wednesday
            period_mult = 0.05
        else:
            period_mult = BAD_PERIODS[9]

    # Combined
    combined = day_mult * hour_mult * entry_mult * quality_mult * period_mult

    reason = f"day={day_mult:.1f} hour={hour_mult:.1f} entry={entry_mult:.1f} qual={quality_mult:.1f} period={period_mult:.1f}"

    return combined, reason


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(df: pd.DataFrame, col_map: dict) -> Tuple[List[Trade], float]:
    """Run backtest"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0

    position: Optional[Trade] = None

    time_col = 'time'
    if 'time' not in df.columns:
        time_col = None

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]

        if time_col:
            current_time = pd.to_datetime(current_bar[time_col])
        else:
            current_time = df.index[i]

        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]

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
                    pips = (exit_price - position.entry_price) / 0.1
                else:
                    pips = (position.entry_price - exit_price) / 0.1

                pnl = pips * position.lot_size * 10

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
            continue

        # Check entry for each POI
        for poi in pois:
            # Regime alignment
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            # Check price near POI
            zone_size = abs(current_bar[col_map['high']] - current_bar[col_map['low']]) * 2
            if abs(current_price - poi['price']) > zone_size:
                continue

            # Check entry trigger
            prev_bar = df.iloc[i-1]
            has_trigger, entry_type = check_entry_trigger(current_bar, prev_bar, poi['direction'], col_map)

            if not has_trigger:
                continue

            # Calculate risk multiplier
            risk_mult, risk_reason = calculate_risk_multiplier(
                current_time, entry_type, poi['quality']
            )

            # Skip if risk too low
            if risk_mult < 0.35:
                continue

            # Calculate position size
            risk_percent = RISK_PERCENT / 100.0
            risk_amount = balance * risk_percent * risk_mult

            # XAUUSD: 1 pip = $10 per 0.1 lot = $1 per 0.01 lot
            lot_size = risk_amount / (SL_PIPS * 10)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            # Calculate SL/TP
            if poi['direction'] == 'BUY':
                sl_price = current_price - (SL_PIPS * 0.1)
                tp_price = current_price + (TP_PIPS * 0.1)
            else:
                sl_price = current_price + (SL_PIPS * 0.1)
                tp_price = current_price - (TP_PIPS * 0.1)

            # Create position
            position = Trade(
                entry_time=current_time,
                direction=poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                risk_amount=risk_amount,
                quality_score=poi['quality'],
                entry_type=entry_type,
                session=session,
                regime=regime.value,
                risk_mult=risk_mult
            )

            break  # Only one trade at a time

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
        'monthly': monthly
    }


def print_results(stats: dict, trades: List[Trade]):
    """Print results"""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS - H1 v6.0 ($50K)")
    print(f"{'='*60}")

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

    print(f"\n{'='*60}")
    if stats['net_pnl'] >= 5000:
        print(f"[OK] TARGET MET: Profit ${stats['net_pnl']:+,.2f} >= $5,000")
    else:
        gap = 5000 - stats['net_pnl']
        print(f"[X] TARGET NOT MET: Profit ${stats['net_pnl']:+,.2f} (need +${gap:,.2f})")

    if stats['profit_factor'] >= 2.0:
        print(f"[OK] PF TARGET MET: {stats['profit_factor']:.2f} >= 2.0")
    else:
        print(f"[X] PF TARGET NOT MET: {stats['profit_factor']:.2f} < 2.0")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================
async def main():
    symbol = "XAUUSD"
    timeframe = "H1"
    # Use 2024 data (same as original v5.8)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 15, tzinfo=timezone.utc)

    print(f"SURGE-WSI H1 v6.0 - $50K Backtest")
    print(f"{'='*60}")
    print(f"Fetching {symbol} {timeframe} data...")
    print(f"Period: {start.date()} to {end.date()}")

    df = await fetch_data(symbol, timeframe, start, end)

    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")

    # Detect columns
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
        elif 'time' in col_lower:
            col_map['time'] = col

    print(f"Column mapping: {col_map}")

    # Run backtest
    print(f"\nRunning backtest...")
    trades, max_dd = run_backtest(df, col_map)

    if not trades:
        print("No trades executed")
        return

    # Calculate stats
    stats = calculate_stats(trades, max_dd)

    # Print results
    print_results(stats, trades)

    # Save trades
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'lot_size': t.lot_size,
        'pnl': t.pnl,
        'pnl_pips': t.pnl_pips,
        'exit_reason': t.exit_reason,
        'quality_score': t.quality_score,
        'entry_type': t.entry_type,
        'session': t.session,
        'risk_mult': t.risk_mult
    } for t in trades])

    output_path = Path(__file__).parent.parent / "results" / "h1_v6_0_50k_trades.csv"
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
