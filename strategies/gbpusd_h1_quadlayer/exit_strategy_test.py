"""
EXIT STRATEGY COMPARISON TEST
=============================
Tests different exit strategies for GBPUSD H1 strategy.

Strategies tested:
1. Fixed R:R Ratios (2:1, 3:1, 4:1, 5:1)
2. Dynamic TP based on ATR volatility
3. Partial Take Profit (50% at 2R, 50% at 4R)
4. Time-Based Exit (close if not hit TP/SL within N bars)

Test Period: 2024-02-01 to 2026-01-30
Baseline: Current strategy v6.8.0

Author: SURIOTA Team
"""

import sys
import io
from pathlib import Path

# Strategy directory (where this file is located)
STRATEGY_DIR = Path(__file__).parent
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGY_DIR.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from config import config
from src.data.db_handler import DBHandler

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
SYMBOL = "GBPUSD"
TIMEFRAME = "H1"
PIP_SIZE = 0.0001
PIP_VALUE = 10.0

INITIAL_BALANCE = 50_000.0
RISK_PERCENT = 1.0
MAX_LOT = 5.0
MAX_LOSS_PER_TRADE_PCT = 0.15

MIN_ATR = 8.0
MAX_ATR = 25.0

BASE_QUALITY = 65
SL_ATR_MULT = 1.5  # Base SL = 1.5 ATR (this is the constant for all tests)


# ============================================================
# EXIT STRATEGY CONFIGURATIONS
# ============================================================
@dataclass
class ExitStrategy:
    """Configuration for a single exit strategy"""
    name: str
    description: str

    # R:R configuration
    tp_ratio: float = 3.0  # TP = SL * tp_ratio

    # Dynamic TP configuration
    use_dynamic_tp: bool = False
    dynamic_tp_low_vol: float = 2.0   # R:R for low volatility
    dynamic_tp_normal: float = 3.0    # R:R for normal volatility
    dynamic_tp_high_vol: float = 4.0  # R:R for high volatility
    atr_low_threshold: float = 0.8    # ATR < avg * this = low vol
    atr_high_threshold: float = 1.2   # ATR > avg * this = high vol

    # Partial TP configuration
    use_partial_tp: bool = False
    partial_tp_1_ratio: float = 2.0   # First partial at 2R
    partial_tp_1_percent: float = 0.5 # Close 50%
    partial_tp_2_ratio: float = 4.0   # Second partial at 4R (remaining)
    move_sl_to_be: bool = True        # Move SL to breakeven after partial

    # Time-based exit configuration
    use_time_exit: bool = False
    max_bars_in_trade: int = 24       # Close if in trade > N bars


# Define all exit strategies to test
EXIT_STRATEGIES = [
    ExitStrategy(
        name="Fixed_1.5R",
        description="Fixed 1.5:1 R:R (TP = 1.5x SL) - CURRENT CONFIG",
        tp_ratio=1.5
    ),
    ExitStrategy(
        name="Fixed_2R",
        description="Fixed 2:1 R:R (TP = 2x SL)",
        tp_ratio=2.0
    ),
    ExitStrategy(
        name="Fixed_2.5R",
        description="Fixed 2.5:1 R:R (TP = 2.5x SL)",
        tp_ratio=2.5
    ),
    ExitStrategy(
        name="Fixed_3R",
        description="Fixed 3:1 R:R (TP = 3x SL)",
        tp_ratio=3.0
    ),
    ExitStrategy(
        name="Fixed_4R",
        description="Fixed 4:1 R:R (TP = 4x SL)",
        tp_ratio=4.0
    ),
    ExitStrategy(
        name="Fixed_5R",
        description="Fixed 5:1 R:R (TP = 5x SL)",
        tp_ratio=5.0
    ),
    ExitStrategy(
        name="Dynamic_ATR",
        description="Dynamic TP based on ATR volatility",
        use_dynamic_tp=True,
        dynamic_tp_low_vol=2.0,
        dynamic_tp_normal=3.0,
        dynamic_tp_high_vol=4.0
    ),
    ExitStrategy(
        name="Partial_TP",
        description="Partial TP: 50% at 2R, 50% at 4R",
        use_partial_tp=True,
        partial_tp_1_ratio=2.0,
        partial_tp_1_percent=0.5,
        partial_tp_2_ratio=4.0,
        move_sl_to_be=True
    ),
    ExitStrategy(
        name="Time_Exit_24h",
        description="Time-based: Close if not hit TP/SL in 24 bars",
        tp_ratio=3.0,
        use_time_exit=True,
        max_bars_in_trade=24
    ),
    ExitStrategy(
        name="Time_Exit_48h",
        description="Time-based: Close if not hit TP/SL in 48 bars",
        tp_ratio=3.0,
        use_time_exit=True,
        max_bars_in_trade=48
    ),
    ExitStrategy(
        name="Combo_Dynamic_Partial",
        description="Combo: Dynamic TP + Partial (50% at 1.5R)",
        use_dynamic_tp=True,
        use_partial_tp=True,
        partial_tp_1_ratio=1.5,
        partial_tp_1_percent=0.5,
        partial_tp_2_ratio=3.0
    ),
    # Additional hybrid strategies
    ExitStrategy(
        name="Fixed_4R_TimeExit",
        description="4:1 R:R with 48h time exit",
        tp_ratio=4.0,
        use_time_exit=True,
        max_bars_in_trade=48
    ),
    ExitStrategy(
        name="Partial_3R_4R",
        description="Partial: 50% at 3R, 50% at 4R",
        use_partial_tp=True,
        partial_tp_1_ratio=3.0,
        partial_tp_1_percent=0.5,
        partial_tp_2_ratio=4.0,
        move_sl_to_be=True
    ),
    ExitStrategy(
        name="Partial_2R_5R",
        description="Partial: 50% at 2R, 50% at 5R",
        use_partial_tp=True,
        partial_tp_1_ratio=2.0,
        partial_tp_1_percent=0.5,
        partial_tp_2_ratio=5.0,
        move_sl_to_be=True
    ),
    ExitStrategy(
        name="Fixed_3.5R",
        description="Fixed 3.5:1 R:R (TP = 3.5x SL)",
        tp_ratio=3.5
    ),
]


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
    # Exit strategy specific
    partial_closed: bool = False
    partial_pnl: float = 0.0
    original_lot_size: float = 0.0
    partial_tp_price: float = 0.0
    bars_in_trade: int = 0


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data from database"""
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
    """Calculate Average True Range in pips"""
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
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def detect_regime(df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float]:
    """Detect market regime using EMA crossover"""
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
    """Detect order block POIs"""
    pois = []
    if len(df) < 35:
        return pois
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    for i in range(len(df) - 30, len(df) - 2):
        if i < 2:
            continue
        current = df.iloc[i]
        next1 = df.iloc[i+1]

        # Bearish OB -> BUY signal
        is_bearish = current[c] < current[o]
        if is_bearish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] > next1[o] and body_ratio > 0.55 and next1[c] > current[h]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[l], 'direction': 'BUY',
                                   'quality': quality, 'idx': i, 'type': 'ORDER_BLOCK'})

        # Bullish OB -> SELL signal
        is_bullish = current[c] > current[o]
        if is_bullish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] < next1[o] and body_ratio > 0.55 and next1[c] < current[l]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[h], 'direction': 'SELL',
                                   'quality': quality, 'idx': i, 'type': 'ORDER_BLOCK'})
    return pois


def check_entry_trigger(bar: pd.Series, prev_bar: pd.Series, direction: str, col_map: dict) -> Tuple[bool, str]:
    """Check for entry trigger confirmation"""
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    total_range = bar[h] - bar[l]
    if total_range < 0.0003:
        return False, ""
    body = abs(bar[c] - bar[o])
    is_bullish = bar[c] > bar[o]
    is_bearish = bar[c] < bar[o]
    prev_body = abs(prev_bar[c] - prev_bar[o])

    if body > total_range * 0.5:
        if direction == 'BUY' and is_bullish:
            return True, 'MOMENTUM'
        if direction == 'SELL' and is_bearish:
            return True, 'MOMENTUM'

    if body > prev_body * 1.2:
        if direction == 'BUY' and is_bullish and prev_bar[c] < prev_bar[o]:
            return True, 'ENGULF'
        if direction == 'SELL' and is_bearish and prev_bar[c] > prev_bar[o]:
            return True, 'ENGULF'

    if direction == 'SELL':
        if bar[h] < prev_bar[h] and is_bearish:
            return True, 'LOWER_HIGH'

    return False, ""


def calculate_risk_multiplier(dt: datetime, entry_type: str, quality: float) -> Tuple[float, bool]:
    """Calculate combined risk multiplier"""
    day = dt.weekday()
    hour = dt.hour
    month = dt.month

    DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}
    HOUR_MULTIPLIERS = {
        0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
        6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
        12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
        18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
    }
    ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}
    MONTHLY_RISK = {
        1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,
        7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
    }

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


def get_dynamic_tp_ratio(current_atr: float, atr_avg: float, strategy: ExitStrategy) -> float:
    """Calculate dynamic TP ratio based on ATR volatility"""
    if not strategy.use_dynamic_tp:
        return strategy.tp_ratio

    if current_atr > atr_avg * strategy.atr_high_threshold:
        return strategy.dynamic_tp_high_vol
    elif current_atr < atr_avg * strategy.atr_low_threshold:
        return strategy.dynamic_tp_low_vol
    else:
        return strategy.dynamic_tp_normal


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest_with_strategy(df: pd.DataFrame, col_map: dict, strategy: ExitStrategy) -> Tuple[List[Trade], float, dict]:
    """Run backtest with a specific exit strategy"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Calculate ATR average for dynamic TP
    atr_avg = atr_series.rolling(window=50).mean()

    stats = {'entries': 0, 'exits': {'TP': 0, 'SL': 0, 'TIME': 0, 'PARTIAL': 0, 'BE': 0}}

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]
        current_atr_avg = atr_avg.iloc[i] if not pd.isna(atr_avg.iloc[i]) else current_atr

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # ============================================================
        # POSITION MANAGEMENT
        # ============================================================
        if position:
            high = current_bar[col_map['high']]
            low = current_bar[col_map['low']]
            exit_price = None
            exit_reason = ""

            position.bars_in_trade += 1

            # ============================================================
            # PARTIAL TP LOGIC
            # ============================================================
            if strategy.use_partial_tp and not position.partial_closed:
                if position.direction == 'BUY':
                    if high >= position.partial_tp_price:
                        partial_exit_price = position.partial_tp_price
                        partial_pips = (partial_exit_price - position.entry_price) / PIP_SIZE
                        partial_lot = position.original_lot_size * strategy.partial_tp_1_percent
                        position.partial_pnl = partial_pips * partial_lot * PIP_VALUE
                        position.partial_closed = True
                        position.lot_size = position.original_lot_size * (1 - strategy.partial_tp_1_percent)
                        stats['exits']['PARTIAL'] += 1

                        if strategy.move_sl_to_be:
                            position.sl_price = position.entry_price
                else:  # SELL
                    if low <= position.partial_tp_price:
                        partial_exit_price = position.partial_tp_price
                        partial_pips = (position.entry_price - partial_exit_price) / PIP_SIZE
                        partial_lot = position.original_lot_size * strategy.partial_tp_1_percent
                        position.partial_pnl = partial_pips * partial_lot * PIP_VALUE
                        position.partial_closed = True
                        position.lot_size = position.original_lot_size * (1 - strategy.partial_tp_1_percent)
                        stats['exits']['PARTIAL'] += 1

                        if strategy.move_sl_to_be:
                            position.sl_price = position.entry_price

            # ============================================================
            # TIME-BASED EXIT CHECK
            # ============================================================
            if strategy.use_time_exit and position.bars_in_trade >= strategy.max_bars_in_trade:
                exit_price = current_price
                exit_reason = "TIME"
                stats['exits']['TIME'] += 1

            # ============================================================
            # SL/TP EXIT CHECK
            # ============================================================
            if not exit_price:  # Not already exiting by time
                if position.direction == 'BUY':
                    if low <= position.sl_price:
                        exit_price = position.sl_price
                        if position.partial_closed:
                            exit_reason = "BE_SL" if position.sl_price >= position.entry_price else "SL"
                            if position.sl_price >= position.entry_price:
                                stats['exits']['BE'] += 1
                            else:
                                stats['exits']['SL'] += 1
                        else:
                            exit_reason = "SL"
                            stats['exits']['SL'] += 1
                    elif high >= position.tp_price:
                        exit_price = position.tp_price
                        exit_reason = "TP"
                        stats['exits']['TP'] += 1
                else:  # SELL
                    if high >= position.sl_price:
                        exit_price = position.sl_price
                        if position.partial_closed:
                            exit_reason = "BE_SL" if position.sl_price <= position.entry_price else "SL"
                            if position.sl_price <= position.entry_price:
                                stats['exits']['BE'] += 1
                            else:
                                stats['exits']['SL'] += 1
                        else:
                            exit_reason = "SL"
                            stats['exits']['SL'] += 1
                    elif low <= position.tp_price:
                        exit_price = position.tp_price
                        exit_reason = "TP"
                        stats['exits']['TP'] += 1

            # ============================================================
            # CLOSE POSITION
            # ============================================================
            if exit_price:
                if position.direction == 'BUY':
                    pips = (exit_price - position.entry_price) / PIP_SIZE
                else:
                    pips = (position.entry_price - exit_price) / PIP_SIZE

                remaining_pnl = pips * position.lot_size * PIP_VALUE
                max_loss = balance * (MAX_LOSS_PER_TRADE_PCT / 100)

                total_pnl = position.partial_pnl + remaining_pnl

                if total_pnl < 0 and abs(total_pnl) > max_loss:
                    total_pnl = -max_loss
                    exit_reason = "SL_CAPPED"

                position.exit_time = current_time
                position.exit_price = exit_price
                position.pnl = total_pnl
                position.pnl_pips = pips
                position.exit_reason = exit_reason
                balance += total_pnl
                trades.append(position)
                position = None
            continue

        # ============================================================
        # ENTRY LOGIC (Simplified - same for all strategies)
        # ============================================================
        if current_time.weekday() >= 5:
            continue
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        hour = current_time.hour
        if not (8 <= hour <= 11 or 13 <= hour <= 17):
            continue
        if hour in [7, 11]:  # Skip bad hours
            continue

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # Detect POIs
        pois = detect_order_blocks(current_slice, col_map, BASE_QUALITY)
        if not pois:
            continue

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

            # ============================================================
            # CALCULATE SL/TP BASED ON EXIT STRATEGY
            # ============================================================
            sl_pips = current_atr * SL_ATR_MULT

            # Get TP ratio (may be dynamic)
            tp_ratio = get_dynamic_tp_ratio(current_atr, current_atr_avg, strategy)
            tp_pips = sl_pips * tp_ratio

            risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult
            lot_size = risk_amount / (sl_pips * PIP_VALUE)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
                # Partial TP price
                partial_tp_pips = sl_pips * strategy.partial_tp_1_ratio
                partial_tp_price = current_price + (partial_tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)
                # Partial TP price
                partial_tp_pips = sl_pips * strategy.partial_tp_1_ratio
                partial_tp_price = current_price - (partial_tp_pips * PIP_SIZE)

            stats['entries'] += 1

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
                partial_closed=False,
                partial_pnl=0.0,
                original_lot_size=lot_size,
                partial_tp_price=partial_tp_price,
                bars_in_trade=0
            )
            break

    return trades, max_dd, stats


def calculate_stats(trades: List[Trade], max_dd: float) -> dict:
    """Calculate performance statistics"""
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

    trade_df = pd.DataFrame([{'time': t.entry_time, 'pnl': t.pnl} for t in trades])
    trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
    monthly = trade_df.groupby('month')['pnl'].sum()
    losing_months = (monthly < 0).sum()

    return {
        'total_trades': total,
        'winners': win_count,
        'losers': loss_count,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'profit_factor': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_dd': max_dd,
        'max_dd_pct': (max_dd / INITIAL_BALANCE) * 100,
        'losing_months': losing_months,
        'total_months': len(monthly),
        'final_balance': INITIAL_BALANCE + net_pnl,
        'monthly': monthly
    }


# ============================================================
# MAIN TEST RUNNER
# ============================================================
async def run_exit_strategy_comparison():
    """Run backtest for all exit strategies and compare results"""

    print("=" * 80)
    print("EXIT STRATEGY COMPARISON TEST")
    print("=" * 80)
    print(f"Symbol: {SYMBOL} | Timeframe: {TIMEFRAME}")
    print(f"Test Period: 2024-02-01 to 2026-01-30")
    print(f"Initial Balance: ${INITIAL_BALANCE:,.0f}")
    print(f"Base SL: {SL_ATR_MULT}x ATR")
    print("=" * 80)

    # Fetch data
    print("\n[1] Fetching data...")
    start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 30, tzinfo=timezone.utc)

    df = await fetch_data(SYMBOL, TIMEFRAME, start_date, end_date)

    if df.empty:
        print("ERROR: No data fetched!")
        return

    # Detect column names
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

    print(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Run backtest for each strategy
    print("\n[2] Running backtests...")
    results = []

    for strategy in EXIT_STRATEGIES:
        print(f"\n  Testing: {strategy.name} - {strategy.description}")

        trades, max_dd, exit_stats = run_backtest_with_strategy(df, col_map, strategy)
        stats = calculate_stats(trades, max_dd)

        if 'error' in stats:
            print(f"    ERROR: {stats['error']}")
            continue

        results.append({
            'strategy': strategy,
            'trades': trades,
            'stats': stats,
            'exit_stats': exit_stats
        })

        print(f"    Trades: {stats['total_trades']} | WR: {stats['win_rate']:.1f}% | "
              f"PF: {stats['profit_factor']:.2f} | Net P/L: ${stats['net_pnl']:+,.0f} | "
              f"Losing Months: {stats['losing_months']}")

    # ============================================================
    # RESULTS COMPARISON TABLE
    # ============================================================
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)

    # Header
    print(f"{'Strategy':<25} {'Trades':>8} {'Win Rate':>10} {'PF':>8} {'Net P/L':>12} "
          f"{'Avg Win':>10} {'Avg Loss':>10} {'Max DD':>10} {'Losing Mo':>10}")
    print("-" * 100)

    # Sort by Net P/L
    results_sorted = sorted(results, key=lambda x: x['stats']['net_pnl'], reverse=True)

    for r in results_sorted:
        s = r['stats']
        strategy = r['strategy']

        # Highlight best performer
        prefix = "*" if r == results_sorted[0] else " "

        print(f"{prefix}{strategy.name:<24} {s['total_trades']:>8} {s['win_rate']:>9.1f}% "
              f"{s['profit_factor']:>8.2f} ${s['net_pnl']:>11,.0f} "
              f"${s['avg_win']:>9,.0f} ${s['avg_loss']:>9,.0f} "
              f"${s['max_dd']:>9,.0f} {s['losing_months']:>10}")

    print("-" * 100)
    print("* = Best performing strategy by Net P/L")

    # ============================================================
    # DETAILED ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    for r in results_sorted[:3]:  # Top 3
        s = r['stats']
        strategy = r['strategy']
        exit_stats = r['exit_stats']

        print(f"\n--- {strategy.name}: {strategy.description} ---")
        print(f"  Total Trades: {s['total_trades']}")
        print(f"  Win Rate: {s['win_rate']:.1f}% ({s['winners']}W / {s['losers']}L)")
        print(f"  Profit Factor: {s['profit_factor']:.2f}")
        print(f"  Net P/L: ${s['net_pnl']:+,.0f} ({(s['net_pnl']/INITIAL_BALANCE)*100:+.1f}%)")
        print(f"  Avg Win: ${s['avg_win']:,.0f} | Avg Loss: ${s['avg_loss']:,.0f}")
        print(f"  Max Drawdown: ${s['max_dd']:,.0f} ({s['max_dd_pct']:.1f}%)")
        print(f"  Losing Months: {s['losing_months']}/{s['total_months']}")
        print(f"  Exit Stats: TP={exit_stats['exits']['TP']} SL={exit_stats['exits']['SL']} "
              f"TIME={exit_stats['exits']['TIME']} PARTIAL={exit_stats['exits']['PARTIAL']} "
              f"BE={exit_stats['exits']['BE']}")

        # Monthly breakdown
        print(f"  Monthly P/L:")
        monthly = s['monthly']
        for period, pnl in monthly.items():
            indicator = "+" if pnl >= 0 else ""
            loss_marker = " *LOSS*" if pnl < 0 else ""
            print(f"    {period}: ${indicator}{pnl:,.0f}{loss_marker}")

    # ============================================================
    # RECOMMENDATION
    # ============================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best = results_sorted[0]
    best_strategy = best['strategy']
    best_stats = best['stats']

    # Find strategy with 0 losing months (if any)
    zero_loss_months = [r for r in results_sorted if r['stats']['losing_months'] == 0]

    print(f"\nBest by Net P/L: {best_strategy.name}")
    print(f"  - Net P/L: ${best_stats['net_pnl']:+,.0f}")
    print(f"  - Win Rate: {best_stats['win_rate']:.1f}%")
    print(f"  - Profit Factor: {best_stats['profit_factor']:.2f}")
    print(f"  - Losing Months: {best_stats['losing_months']}")

    if zero_loss_months:
        print(f"\nStrategies with 0 Losing Months:")
        for r in zero_loss_months:
            s = r['stats']
            print(f"  - {r['strategy'].name}: ${s['net_pnl']:+,.0f} (PF: {s['profit_factor']:.2f})")

    # Balanced recommendation
    print(f"\n{'='*40}")
    print("FINAL RECOMMENDATION:")
    print(f"{'='*40}")

    # Score each strategy (weighted)
    for r in results:
        s = r['stats']
        # Weighted score: 40% Net P/L, 30% PF, 20% WR, 10% Losing Months penalty
        pnl_score = min(100, max(0, (s['net_pnl'] / 30000) * 100))  # Cap at $30K
        pf_score = min(100, max(0, (s['profit_factor'] / 5) * 100))  # Cap at PF 5
        wr_score = s['win_rate']
        loss_month_penalty = s['losing_months'] * 10  # -10 per losing month

        r['total_score'] = (pnl_score * 0.4) + (pf_score * 0.3) + (wr_score * 0.2) - loss_month_penalty

    results_by_score = sorted(results, key=lambda x: x['total_score'], reverse=True)
    winner = results_by_score[0]

    print(f"\nWINNER: {winner['strategy'].name}")
    print(f"Description: {winner['strategy'].description}")
    print(f"\nParameters:")
    if winner['strategy'].use_dynamic_tp:
        print(f"  - Dynamic TP: Low Vol={winner['strategy'].dynamic_tp_low_vol}R, "
              f"Normal={winner['strategy'].dynamic_tp_normal}R, "
              f"High Vol={winner['strategy'].dynamic_tp_high_vol}R")
    else:
        print(f"  - Fixed TP Ratio: {winner['strategy'].tp_ratio}:1 R:R")
    if winner['strategy'].use_partial_tp:
        print(f"  - Partial TP: {winner['strategy'].partial_tp_1_percent*100:.0f}% at "
              f"{winner['strategy'].partial_tp_1_ratio}R, rest at {winner['strategy'].partial_tp_2_ratio}R")
        print(f"  - Move SL to BE: {winner['strategy'].move_sl_to_be}")
    if winner['strategy'].use_time_exit:
        print(f"  - Time Exit: {winner['strategy'].max_bars_in_trade} bars")

    ws = winner['stats']
    print(f"\nPerformance:")
    print(f"  - Total Trades: {ws['total_trades']}")
    print(f"  - Win Rate: {ws['win_rate']:.1f}%")
    print(f"  - Profit Factor: {ws['profit_factor']:.2f}")
    print(f"  - Net P/L: ${ws['net_pnl']:+,.0f} ({(ws['net_pnl']/INITIAL_BALANCE)*100:+.1f}%)")
    print(f"  - Losing Months: {ws['losing_months']}/{ws['total_months']}")

    return results


if __name__ == "__main__":
    asyncio.run(run_exit_strategy_comparison())
