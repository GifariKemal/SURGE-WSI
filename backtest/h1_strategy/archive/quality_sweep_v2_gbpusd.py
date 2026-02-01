"""
Quality Threshold Sweep V2 - GBPUSD
===================================
Test different quality levels with PROPER loss cap logic

Key insight from v6.2:
- Loss cap at 0.15% = $75 per trade on $50K
- Avg Win ~$493, Avg Loss ~$90
- This asymmetric R:R creates high PF even with 40% WR

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


SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = 1.0

SL_ATR_MULT = 1.5
TP_RATIO = 1.5
MAX_LOSS_PER_TRADE_PCT = 0.15  # KEY: 0.15% loss cap = $75 on $50K

PIP_VALUE = 10.0
PIP_SIZE = 0.0001
MAX_LOT = 5.0
MIN_ATR = 8.0
MAX_ATR = 30.0

MONTHLY_RISK = {
    1: 0.9, 2: 0.7, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,
    7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
}

DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}

HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.8, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.8,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}


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


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


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


def calculate_atr(df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
    h, l, c = col_map['high'], col_map['low'], col_map['close']
    high = df[h]
    low = df[l]
    close = df[c].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr / PIP_SIZE


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def detect_regime(df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float]:
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


def detect_order_blocks(df: pd.DataFrame, col_map: dict, min_quality: float) -> List[dict]:
    """Detect OBs with specified quality threshold - same logic as v6.2"""
    pois = []
    if len(df) < 35:
        return pois

    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    for i in range(len(df) - 30, len(df) - 2):
        if i < 2:
            continue
        current = df.iloc[i]
        next1 = df.iloc[i+1]

        # Bullish OB: bearish candle followed by strong bullish
        is_bearish = current[c] < current[o]
        if is_bearish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                # v6.2 uses 0.55 threshold
                if next1[c] > next1[o] and body_ratio > 0.55 and next1[c] > current[h]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[l], 'direction': 'BUY', 'quality': quality, 'idx': i})

        # Bearish OB: bullish candle followed by strong bearish
        is_bullish = current[c] > current[o]
        if is_bullish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] < next1[o] and body_ratio > 0.55 and next1[c] < current[l]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[h], 'direction': 'SELL', 'quality': quality, 'idx': i})

    return pois


def check_entry_trigger(bar: pd.Series, prev_bar: pd.Series, direction: str, col_map: dict) -> Tuple[bool, str]:
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    total_range = bar[h] - bar[l]
    if total_range < 0.0003:
        return False, ""

    body = abs(bar[c] - bar[o])
    is_bullish = bar[c] > bar[o]
    is_bearish = bar[c] < bar[o]
    prev_body = abs(prev_bar[c] - prev_bar[o])

    # MOMENTUM
    if body > total_range * 0.5:
        if direction == 'BUY' and is_bullish:
            return True, 'MOMENTUM'
        if direction == 'SELL' and is_bearish:
            return True, 'MOMENTUM'

    # ENGULF
    if body > prev_body * 1.2:
        if direction == 'BUY' and is_bullish and prev_bar[c] < prev_bar[o]:
            return True, 'ENGULF'
        if direction == 'SELL' and is_bearish and prev_bar[c] > prev_bar[o]:
            return True, 'ENGULF'

    # LOWER_HIGH (SELL only)
    if direction == 'SELL':
        if bar[h] < prev_bar[h] and is_bearish:
            return True, 'LOWER_HIGH'

    return False, ""


def calculate_risk_multiplier(dt: datetime, entry_type: str, quality: float) -> Tuple[float, bool]:
    day = dt.weekday()
    hour = dt.hour
    month = dt.month

    day_mult = DAY_MULTIPLIERS.get(day, 0.5)
    hour_mult = HOUR_MULTIPLIERS.get(hour, 0.0)
    entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 0.0)
    quality_mult = quality / 100.0
    month_mult = MONTHLY_RISK.get(month, 0.8)

    if day_mult == 0.0 or hour_mult == 0.0 or entry_mult == 0.0:
        return 0.0, True

    combined = day_mult * hour_mult * entry_mult * quality_mult * month_mult
    if combined < 0.30:
        return combined, True

    return max(0.30, min(1.2, combined)), False


def run_backtest(df: pd.DataFrame, col_map: dict, min_quality: float) -> Tuple[List[Trade], float]:
    """Run backtest with EXACT same logic as v6.2"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]

        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Manage position
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

                # KEY: Apply LOSS CAP
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

        # ATR filter
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        # Kill zone check
        hour = current_time.hour
        if not (7 <= hour <= 11 or 13 <= hour <= 17):
            continue

        session = "london" if 7 <= hour <= 11 else "newyork"

        # Regime check
        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # POI detection
        pois = detect_order_blocks(current_slice, col_map, min_quality)
        if not pois:
            continue

        # Entry check
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
                continue

            risk_mult, should_skip = calculate_risk_multiplier(current_time, entry_type, poi['quality'])
            if should_skip:
                continue

            # ATR-based SL/TP
            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO

            # Position sizing - IMPORTANT: DON'T cap lot size, only cap loss at exit!
            risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult

            # Calculate lot size from full risk amount (not capped)
            lot_size = risk_amount / (sl_pips * PIP_VALUE)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

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
            )
            break

    return trades, max_dd


def calculate_stats(trades: List[Trade], max_dd: float) -> dict:
    if not trades:
        return None

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

    trade_df = pd.DataFrame([{'time': t.entry_time, 'pnl': t.pnl} for t in trades])
    trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
    monthly = trade_df.groupby('month')['pnl'].sum()
    losing_months = (monthly < 0).sum()

    first_trade = trades[0].entry_time
    last_trade = trades[-1].entry_time
    days = (last_trade - first_trade).days or 1
    trades_per_day = total / days

    return {
        'total_trades': total,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'profit_factor': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_dd_pct': (max_dd / INITIAL_BALANCE) * 100,
        'losing_months': losing_months,
        'total_months': len(monthly),
        'trades_per_day': trades_per_day,
    }


async def main():
    timeframe = "H1"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 15, tzinfo=timezone.utc)

    print(f"GBPUSD Quality Threshold Sweep V2")
    print(f"{'='*90}")
    print(f"Testing different MIN_QUALITY with EXACT v6.2 logic (including 0.15% loss cap)")
    print(f"{'='*90}")

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

    # Test different quality thresholds
    quality_levels = [50, 55, 60, 65, 70, 75, 80]

    print(f"\n{'Quality':<10} {'Trades':<8} {'T/Day':<8} {'WR%':<8} {'PF':<8} {'AvgWin':<10} {'AvgLoss':<10} {'Net P/L':<12} {'Loss Mo':<10}")
    print("-" * 90)

    results = []

    for quality in quality_levels:
        trades, max_dd = run_backtest(df, col_map, quality)

        if trades:
            stats = calculate_stats(trades, max_dd)
            if stats:
                results.append({
                    'quality': quality,
                    'trades': stats['total_trades'],
                    'trades_per_day': stats['trades_per_day'],
                    'win_rate': stats['win_rate'],
                    'pf': stats['profit_factor'],
                    'avg_win': stats['avg_win'],
                    'avg_loss': stats['avg_loss'],
                    'net_pnl': stats['net_pnl'],
                    'losing_months': stats['losing_months'],
                    'total_months': stats['total_months']
                })

                losing_str = f"{stats['losing_months']}/{stats['total_months']}"
                pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 100 else "INF"

                print(f"{quality:<10} {stats['total_trades']:<8} {stats['trades_per_day']:<8.2f} {stats['win_rate']:<8.1f} {pf_str:<8} ${stats['avg_win']:<9,.0f} ${stats['avg_loss']:<9,.0f} ${stats['net_pnl']:<11,.0f} {losing_str:<10}")
        else:
            print(f"{quality:<10} {'0':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<10} {'-':<10} {'-':<12} {'-':<10}")

    print("-" * 90)

    # Analysis
    print(f"\n{'='*90}")
    print("ANALYSIS")
    print(f"{'='*90}")

    # Filter for zero losing months
    zero_loss = [r for r in results if r['losing_months'] == 0]

    if zero_loss:
        print("\nConfigurations with ZERO losing months:")
        for r in zero_loss:
            print(f"  Quality {r['quality']}: {r['trades']} trades, {r['trades_per_day']:.2f}/day, PF {r['pf']:.2f}, ${r['net_pnl']:+,.0f}")

        best_trades = max(zero_loss, key=lambda x: x['trades_per_day'])
        print(f"\n  >>> RECOMMENDED (Most Trades + Zero Loss): Quality = {best_trades['quality']}")
        print(f"      {best_trades['trades']} trades, {best_trades['trades_per_day']:.2f}/day")
        print(f"      PF {best_trades['pf']:.2f}, ${best_trades['net_pnl']:+,.0f}")
    else:
        print("\nNo configuration with zero losing months!")
        # Find configuration with least losing months
        if results:
            best = min(results, key=lambda x: (x['losing_months'], -x['net_pnl']))
            print(f"  Best option: Quality = {best['quality']}")
            print(f"  {best['trades']} trades, {best['losing_months']} losing months, ${best['net_pnl']:+,.0f}")

    print(f"\n{'='*90}")

    # DETAILED LOSING MONTH ANALYSIS
    print(f"\n\n{'='*90}")
    print("LOSING MONTH DEEP DIVE - Quality 60 vs Quality 70")
    print(f"{'='*90}")

    for quality in [60, 70]:
        trades, _ = run_backtest(df, col_map, quality)
        if not trades:
            continue

        trade_df = pd.DataFrame([{
            'time': t.entry_time,
            'pnl': t.pnl,
            'quality': t.quality_score,
            'entry_type': t.entry_type
        } for t in trades])
        trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
        monthly = trade_df.groupby('month')['pnl'].sum()

        print(f"\n[Quality {quality}] Monthly P/L:")
        for month, pnl in monthly.items():
            status = "LOSS" if pnl < 0 else "WIN"
            mark = "<<<" if pnl < 0 else ""
            print(f"  {month}: ${pnl:+,.0f} [{status}] {mark}")

        # Find losing months
        losing = monthly[monthly < 0]
        if not losing.empty:
            print(f"\n  LOSING MONTH BREAKDOWN:")
            for month in losing.index:
                month_trades = [t for t in trades if t.entry_time.strftime('%Y-%m') == str(month)[:7]]
                low_q = [t for t in month_trades if t.quality_score < 70]
                high_q = [t for t in month_trades if t.quality_score >= 70]

                low_pnl = sum(t.pnl for t in low_q)
                high_pnl = sum(t.pnl for t in high_q)

                print(f"\n  {month}:")
                print(f"    Total trades: {len(month_trades)}")
                print(f"    Low Quality (Q<70): {len(low_q)} trades, ${low_pnl:+,.0f}")
                print(f"    High Quality (Q>=70): {len(high_q)} trades, ${high_pnl:+,.0f}")
                print(f"    >>> Low quality trades cause: ${low_pnl:+,.0f} of the loss!")


if __name__ == "__main__":
    asyncio.run(main())
