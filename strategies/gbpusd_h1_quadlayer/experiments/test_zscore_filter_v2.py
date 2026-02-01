"""
Z-SCORE FILTER TEST V2 - TREND CONFIRMATION MODE
=================================================

The mean reversion approach (v1) blocked ALL trades because:
- Strategy is trend-following (BUY in BULLISH, SELL in BEARISH)
- Mean reversion wants BUY when oversold (Z < -1.5) and SELL when overbought (Z > 1.5)
- These two approaches are fundamentally opposite!

V2: Z-Score as TREND CONFIRMATION (not mean reversion)
- BUY signals: Allow when Z-Score > 0 (price above SMA = bullish momentum)
- SELL signals: Allow when Z-Score < 0 (price below SMA = bearish momentum)
- This aligns with trend-following: trade in direction of current momentum

Also testing extreme exhaustion filter:
- Block BUY when Z > 2.5 (extremely overbought, may reverse)
- Block SELL when Z < -2.5 (extremely oversold, may reverse)

Author: SURIOTA Team
"""

import sys
import io
from pathlib import Path

# Strategy directory (where this file is located)
STRATEGY_DIR = Path(__file__).parent
# Project root is 2 levels up
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add strategies folder to allow package imports
sys.path.insert(0, str(STRATEGY_DIR.parent))

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

# Import shared trading filters from strategy package
from gbpusd_h1_quadlayer.trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    get_monthly_quality_adjustment,
    MONTHLY_TRADEABLE_PCT,
    SEASONAL_TEMPLATE,
)
from gbpusd_h1_quadlayer.strategy_config import (
    SYMBOL, PIP_SIZE, PIP_VALUE,
    RISK, TECHNICAL, INTRA_MONTH,
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ZSCORE FILTER CONFIG - V2 TREND CONFIRMATION
# ============================================================
USE_ZSCORE_FILTER = True
ZSCORE_PERIOD = 20

# V2: Trend confirmation thresholds
ZSCORE_TREND_CONFIRM = 0.0       # Basic: Z > 0 for BUY, Z < 0 for SELL
ZSCORE_EXTREME_BLOCK = 2.5       # Block if Z exceeds this (exhaustion)

# ============================================================
# CONFIGURATION (from v6.8)
# ============================================================
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = RISK.risk_percent
SL_ATR_MULT = RISK.sl_atr_mult
TP_RATIO = RISK.tp_ratio
MAX_LOT = RISK.max_lot
MAX_LOSS_PER_TRADE_PCT = RISK.max_loss_per_trade_pct
MIN_ATR = TECHNICAL.min_atr
MAX_ATR = TECHNICAL.max_atr

BASE_QUALITY = TECHNICAL.base_quality_normal
MIN_QUALITY_GOOD = TECHNICAL.base_quality_good
MAX_QUALITY_BAD = TECHNICAL.base_quality_bad

# Layer 3: Intra-Month Risk Thresholds
MONTHLY_LOSS_THRESHOLD_1 = INTRA_MONTH.loss_threshold_1
MONTHLY_LOSS_THRESHOLD_2 = INTRA_MONTH.loss_threshold_2
MONTHLY_LOSS_THRESHOLD_3 = INTRA_MONTH.loss_threshold_3
MONTHLY_LOSS_STOP = INTRA_MONTH.loss_stop
CONSECUTIVE_LOSS_THRESHOLD = INTRA_MONTH.consec_loss_quality
CONSECUTIVE_LOSS_MAX = INTRA_MONTH.consec_loss_day_stop

# Layer 4: Pattern filter enabled
USE_PATTERN_FILTER = True

# Layer 2: Technical thresholds
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

# Entry signal configs
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True

# Session-POI filter (v6.8)
USE_SESSION_POI_FILTER = True
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]

# Risk Multipliers
MONTHLY_RISK = {
    1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,
    7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
}

DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}

HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}


# ============================================================
# Z-SCORE CALCULATION FUNCTIONS
# ============================================================
def calculate_zscore(df: pd.DataFrame, col_map: dict, period: int = 20) -> pd.Series:
    """Calculate Z-Score for price."""
    close = df[col_map['close']]
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    zscore = (close - sma) / (std + 1e-10)
    return zscore


def check_zscore_trend_confirmation(zscore_value: float, direction: str,
                                    trend_threshold: float = 0.0,
                                    extreme_threshold: float = 2.5) -> Tuple[bool, str]:
    """
    Check if Z-Score confirms trend direction.

    V2: Trend Confirmation Logic:
    - BUY: Allow when Z > trend_threshold (bullish momentum)
    - SELL: Allow when Z < -trend_threshold (bearish momentum)
    - Block if Z is extreme (potential exhaustion/reversal)
    """
    if pd.isna(zscore_value):
        return True, "ZSCORE_NA"

    if direction == 'BUY':
        # Block if extremely overbought (exhaustion)
        if zscore_value > extreme_threshold:
            return False, f"EXTREME_OB_Z={zscore_value:.2f}"
        # Require bullish momentum (Z > threshold)
        if zscore_value > trend_threshold:
            return True, f"BULLISH_Z={zscore_value:.2f}"
        else:
            return False, f"NO_BULLISH_Z={zscore_value:.2f}"
    else:  # SELL
        # Block if extremely oversold (exhaustion)
        if zscore_value < -extreme_threshold:
            return False, f"EXTREME_OS_Z={zscore_value:.2f}"
        # Require bearish momentum (Z < -threshold)
        if zscore_value < -trend_threshold:
            return True, f"BEARISH_Z={zscore_value:.2f}"
        else:
            return False, f"NO_BEARISH_Z={zscore_value:.2f}"


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
    poi_type: str = ""
    session: str = ""
    dynamic_quality: float = 0.0
    market_condition: str = ""
    monthly_adj: int = 0
    zscore_at_entry: float = 0.0


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class MarketCondition:
    atr_stability: float
    efficiency: float
    trend_strength: float
    technical_quality: float
    monthly_adjustment: int
    final_quality: float
    label: str


# ============================================================
# HELPER FUNCTIONS (from original backtest)
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


def assess_market_condition(df: pd.DataFrame, col_map: dict, idx: int,
                           atr_series: pd.Series, current_time: datetime) -> MarketCondition:
    lookback = 20
    start_idx = max(0, idx - lookback)
    h, l, c = col_map['high'], col_map['low'], col_map['close']

    recent_atr = atr_series.iloc[start_idx:idx+1]
    if len(recent_atr) > 5 and recent_atr.mean() > 0:
        atr_cv = recent_atr.std() / recent_atr.mean()
    else:
        atr_cv = 0.5

    if idx >= lookback:
        net_move = abs(df[c].iloc[idx] - df[c].iloc[start_idx])
        total_move = sum(abs(df[c].iloc[i] - df[c].iloc[i-1]) for i in range(start_idx+1, idx+1))
        efficiency = net_move / total_move if total_move > 0 else 0
    else:
        efficiency = 0.1

    if idx >= lookback:
        highs = df[h].iloc[start_idx:idx+1]
        lows = df[l].iloc[start_idx:idx+1]
        closes = df[c].iloc[start_idx:idx+1]

        plus_dm = (highs - highs.shift(1)).clip(lower=0)
        minus_dm = (lows.shift(1) - lows).clip(lower=0)

        tr = pd.concat([
            highs - lows,
            abs(highs - closes.shift(1)),
            abs(lows - closes.shift(1))
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(14).mean()

        trend_strength = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
    else:
        trend_strength = 25

    score = 0
    if atr_cv < ATR_STABILITY_THRESHOLD:
        score += 33
    elif atr_cv < ATR_STABILITY_THRESHOLD * 1.5:
        score += 20

    if efficiency > EFFICIENCY_THRESHOLD:
        score += 33
    elif efficiency > EFFICIENCY_THRESHOLD * 0.5:
        score += 20

    if trend_strength > TREND_STRENGTH_THRESHOLD:
        score += 34
    elif trend_strength > TREND_STRENGTH_THRESHOLD * 0.7:
        score += 20

    if score >= 80:
        technical_quality = MIN_QUALITY_GOOD
    elif score >= 40:
        technical_quality = BASE_QUALITY
    else:
        technical_quality = MAX_QUALITY_BAD

    monthly_adj = get_monthly_quality_adjustment(current_time)
    final_quality = technical_quality + monthly_adj

    if monthly_adj >= 15:
        label = "POOR_MONTH"
    elif monthly_adj >= 10:
        label = "CAUTION"
    elif score >= 80:
        label = "GOOD"
    elif score >= 40:
        label = "NORMAL"
    else:
        label = "BAD"

    return MarketCondition(
        atr_stability=atr_cv,
        efficiency=efficiency,
        trend_strength=trend_strength,
        technical_quality=technical_quality,
        monthly_adjustment=monthly_adj,
        final_quality=final_quality,
        label=label
    )


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
    pois = []
    if len(df) < 35:
        return pois
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    for i in range(len(df) - 30, len(df) - 2):
        if i < 2:
            continue
        current = df.iloc[i]
        next1 = df.iloc[i+1]

        is_bearish = current[c] < current[o]
        if is_bearish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] > next1[o] and body_ratio > 0.55 and next1[c] > current[h]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[l], 'direction': 'BUY', 'quality': quality, 'idx': i, 'type': 'ORDER_BLOCK'})

        is_bullish = current[c] > current[o]
        if is_bullish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] < next1[o] and body_ratio > 0.55 and next1[c] < current[l]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[h], 'direction': 'SELL', 'quality': quality, 'idx': i, 'type': 'ORDER_BLOCK'})
    return pois


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_adx(df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
    h, l, c = col_map['high'], col_map['low'], col_map['close']
    highs = df[h]
    lows = df[l]
    closes = df[c]

    plus_dm = (highs - highs.shift(1)).clip(lower=0)
    minus_dm = (lows.shift(1) - lows).clip(lower=0)

    tr = pd.concat([
        highs - lows,
        abs(highs - closes.shift(1)),
        abs(lows - closes.shift(1))
    ], axis=1).max(axis=1)

    atr_14 = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr_14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr_14 + 1e-10))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return adx


def detect_ema_pullback(df: pd.DataFrame, col_map: dict, atr_series: pd.Series,
                        min_quality: float) -> List[dict]:
    pois = []
    if len(df) < 50:
        return pois

    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    ema20 = calculate_ema(df[c], 20)
    ema50 = calculate_ema(df[c], 50)
    rsi = calculate_rsi(df[c], 14)
    adx = calculate_adx(df, col_map, 14)

    i = len(df) - 1
    bar = df.iloc[i]

    current_close = bar[c]
    current_open = bar[o]
    current_high = bar[h]
    current_low = bar[l]
    current_ema20 = ema20.iloc[i]
    current_ema50 = ema50.iloc[i]
    current_rsi = rsi.iloc[i]
    current_adx = adx.iloc[i]
    current_atr = atr_series.iloc[i] if i < len(atr_series) else 0

    if pd.isna(current_ema20) or pd.isna(current_ema50) or pd.isna(current_rsi) or pd.isna(current_adx):
        return pois

    total_range = current_high - current_low
    if total_range < 0.0003:
        return pois
    body = abs(current_close - current_open)
    body_ratio = body / total_range

    if body_ratio < 0.4:
        return pois
    if current_adx < 20:
        return pois
    if not (30 <= current_rsi <= 70):
        return pois
    if current_atr < MIN_ATR or current_atr > MAX_ATR:
        return pois

    atr_distance = current_atr * PIP_SIZE * 1.5

    is_bullish = current_close > current_open
    if is_bullish and current_close > current_ema20 > current_ema50:
        distance_to_ema = current_low - current_ema20
        if distance_to_ema <= atr_distance:
            touch_quality = max(0, 30 - (distance_to_ema / PIP_SIZE))
            adx_quality = min(25, (current_adx - 15) * 1.5)
            rsi_quality = min(25, abs(50 - current_rsi) < 20 and 25 or 15)
            body_quality = min(20, body_ratio * 30)

            quality = touch_quality + adx_quality + rsi_quality + body_quality
            quality = min(100, max(55, quality))

            if quality >= min_quality:
                pois.append({
                    'price': current_close,
                    'direction': 'BUY',
                    'quality': quality,
                    'idx': i,
                    'type': 'EMA_PULLBACK',
                })

    is_bearish = current_close < current_open
    if is_bearish and current_close < current_ema20 < current_ema50:
        distance_to_ema = current_ema20 - current_high
        if distance_to_ema <= atr_distance:
            touch_quality = max(0, 30 - (distance_to_ema / PIP_SIZE))
            adx_quality = min(25, (current_adx - 15) * 1.5)
            rsi_quality = min(25, abs(50 - current_rsi) < 20 and 25 or 15)
            body_quality = min(20, body_ratio * 30)

            quality = touch_quality + adx_quality + rsi_quality + body_quality
            quality = min(100, max(55, quality))

            if quality >= min_quality:
                pois.append({
                    'price': current_close,
                    'direction': 'SELL',
                    'quality': quality,
                    'idx': i,
                    'type': 'EMA_PULLBACK',
                })

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


def should_skip_by_session(hour: int, poi_type: str) -> tuple:
    if not USE_SESSION_POI_FILTER:
        return False, ""

    if hour in SKIP_HOURS:
        return True, f"HOUR_{hour}_SKIP"

    if poi_type == "ORDER_BLOCK" and hour in SKIP_ORDER_BLOCK_HOURS:
        return True, f"OB_HOUR_{hour}_SKIP"

    if poi_type == "EMA_PULLBACK" and hour in SKIP_EMA_PULLBACK_HOURS:
        return True, f"EMA_HOUR_{hour}_SKIP"

    return False, ""


# ============================================================
# BACKTEST WITH ZSCORE TREND CONFIRMATION FILTER
# ============================================================
def run_backtest_zscore_v2(df: pd.DataFrame, col_map: dict,
                           trend_threshold: float = 0.0,
                           extreme_threshold: float = 2.5,
                           use_filter: bool = True) -> Tuple[List[Trade], float, dict, dict]:
    """Run backtest with Z-Score trend confirmation filter"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Pre-calculate Z-Score for entire dataframe
    zscore_series = calculate_zscore(df, col_map, ZSCORE_PERIOD)

    # Layer 3: Intra-month risk manager
    risk_manager = IntraMonthRiskManager()

    # Layer 4: Pattern-based filter
    pattern_filter = PatternBasedFilter()
    current_month_key = None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0, 'ZSCORE_FILTERED': 0, 'ZSCORE_EXTREME': 0}
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0}

    zscore_stats = {
        'total_signals': 0,
        'zscore_passed': 0,
        'zscore_blocked': 0,
        'zscore_extreme_blocked': 0,
        'zscore_at_entry': [],
        'blocked_zscores': [],
    }

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]
        current_zscore = zscore_series.iloc[i]

        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

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

                risk_manager.record_trade(pnl, current_time)

                if USE_PATTERN_FILTER:
                    pattern_filter.record_trade(position.direction, pnl, current_time)

                position = None
            continue

        if current_time.weekday() >= 5:
            continue
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        hour = current_time.hour
        if not (8 <= hour <= 11 or 13 <= hour <= 17):
            continue
        session = "london" if 8 <= hour <= 11 else "newyork"

        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            if 'MONTH' in skip_reason:
                skip_stats['MONTH_STOPPED'] += 1
            elif 'DAY' in skip_reason:
                skip_stats['DAY_STOPPED'] += 1
            continue

        month_key = (current_time.year, current_time.month)
        if month_key != current_month_key:
            current_month_key = month_key
            if USE_PATTERN_FILTER:
                pattern_filter.reset_for_month(current_time.month)

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        if intra_month_adj > 0:
            skip_stats['DYNAMIC_ADJ'] += 1

        condition_stats[market_cond.label] = condition_stats.get(market_cond.label, 0) + 1

        pois = []

        if USE_ORDER_BLOCK:
            ob_pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            pois.extend(ob_pois)

        if USE_EMA_PULLBACK:
            ema_pois = detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality)
            existing_indices = {p['idx'] for p in pois}
            for ep in ema_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        if not pois:
            continue

        for poi in pois:
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            poi_type = poi.get('type', 'ORDER_BLOCK')

            session_skip, session_reason = should_skip_by_session(hour, poi_type)
            if session_skip:
                skip_stats[session_reason] = skip_stats.get(session_reason, 0) + 1
                continue

            if poi_type == 'EMA_PULLBACK':
                entry_type = 'MOMENTUM'
            else:
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

            pattern_size_mult = 1.0
            pattern_extra_q = 0
            if USE_PATTERN_FILTER:
                pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(poi['direction'])
                if not pattern_can_trade:
                    skip_stats['PATTERN_STOPPED'] += 1
                    continue

            # ============================================================
            # Z-SCORE TREND CONFIRMATION FILTER (V2)
            # ============================================================
            zscore_stats['total_signals'] += 1

            if use_filter:
                zscore_allowed, zscore_reason = check_zscore_trend_confirmation(
                    current_zscore, poi['direction'], trend_threshold, extreme_threshold
                )

                if not zscore_allowed:
                    skip_stats['ZSCORE_FILTERED'] += 1
                    zscore_stats['zscore_blocked'] += 1
                    zscore_stats['blocked_zscores'].append(current_zscore)
                    if 'EXTREME' in zscore_reason:
                        zscore_stats['zscore_extreme_blocked'] += 1
                        skip_stats['ZSCORE_EXTREME'] += 1
                    continue

                zscore_stats['zscore_passed'] += 1

            if pattern_extra_q > 0:
                effective_quality = dynamic_quality + pattern_extra_q
                if poi['quality'] < effective_quality:
                    continue

            combined_size_mult = pattern_size_mult

            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO
            risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult * combined_size_mult
            lot_size = risk_amount / (sl_pips * PIP_VALUE)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            entry_stats[poi_type] = entry_stats.get(poi_type, 0) + 1
            zscore_stats['zscore_at_entry'].append(current_zscore)

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
                poi_type=poi_type,
                session=session,
                dynamic_quality=dynamic_quality,
                market_condition=market_cond.label,
                monthly_adj=market_cond.monthly_adjustment + intra_month_adj,
                zscore_at_entry=current_zscore
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats, zscore_stats


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
        'trades_per_day': trades_per_day,
        'final_balance': INITIAL_BALANCE + net_pnl,
        'monthly': monthly
    }


def print_results(stats: dict, trades: List[Trade], zscore_stats: dict,
                  trend_threshold: float, extreme_threshold: float, baseline_stats: dict = None):
    print(f"\n{'='*70}")
    print(f"Z-SCORE V2: TREND CONFIRMATION FILTER")
    print(f"Trend Threshold: {trend_threshold}, Extreme Block: {extreme_threshold}")
    print(f"{'='*70}")

    print(f"\n[Z-SCORE FILTER CONFIGURATION]")
    print(f"{'-'*50}")
    print(f"  Period: {ZSCORE_PERIOD}")
    print(f"  Trend Threshold: {trend_threshold}")
    print(f"  Extreme Block Threshold: {extreme_threshold}")
    print(f"  Filter Logic (Trend Confirmation):")
    print(f"    BUY: Allow when Z > {trend_threshold} (bullish momentum)")
    print(f"    SELL: Allow when Z < {-trend_threshold} (bearish momentum)")
    print(f"    Block if Z > {extreme_threshold} or Z < {-extreme_threshold} (exhaustion)")

    print(f"\n[Z-SCORE FILTER STATISTICS]")
    print(f"{'-'*50}")
    print(f"  Total signals before filter: {zscore_stats['total_signals']}")
    print(f"  Passed Z-Score filter: {zscore_stats['zscore_passed']}")
    print(f"  Blocked by Z-Score filter: {zscore_stats['zscore_blocked']}")
    print(f"    - Blocked by extreme exhaustion: {zscore_stats['zscore_extreme_blocked']}")

    if zscore_stats['zscore_at_entry']:
        entry_zscores = zscore_stats['zscore_at_entry']
        print(f"\n  Z-Score Distribution at Entry:")
        print(f"    Min: {min(entry_zscores):.2f}")
        print(f"    Max: {max(entry_zscores):.2f}")
        print(f"    Mean: {np.mean(entry_zscores):.2f}")
        print(f"    Std: {np.std(entry_zscores):.2f}")

        # Histogram
        bins = [(-float('inf'), -2), (-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0),
                (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, float('inf'))]
        print(f"\n  Z-Score Histogram (at entry):")
        for low, high in bins:
            count = sum(1 for z in entry_zscores if low <= z < high)
            pct = count / len(entry_zscores) * 100 if entry_zscores else 0
            bar = '#' * int(pct / 2)
            if low == -float('inf'):
                label = f"Z < -2.0"
            elif high == float('inf'):
                label = f"Z > 2.0"
            else:
                label = f"{low:.1f} <= Z < {high:.1f}"
            print(f"    {label:18s}: {count:3d} ({pct:5.1f}%) {bar}")

        # Win rate by Z-Score bucket
        print(f"\n  Win Rate by Z-Score at Entry:")
        for low, high in bins:
            bucket_trades = [t for t in trades if low <= t.zscore_at_entry < high]
            if bucket_trades:
                bucket_wins = len([t for t in bucket_trades if t.pnl > 0])
                bucket_wr = bucket_wins / len(bucket_trades) * 100
                bucket_pnl = sum(t.pnl for t in bucket_trades)
                if low == -float('inf'):
                    label = f"Z < -2.0"
                elif high == float('inf'):
                    label = f"Z > 2.0"
                else:
                    label = f"{low:.1f} <= Z < {high:.1f}"
                print(f"    {label:18s}: {len(bucket_trades):3d} trades, {bucket_wr:5.1f}% WR, ${bucket_pnl:+,.0f}")

    print(f"\n[PERFORMANCE]")
    print(f"{'-'*50}")
    print(f"Total Trades:      {stats['total_trades']}")
    print(f"Win Rate:          {stats['win_rate']:.1f}%")
    print(f"Profit Factor:     {stats['profit_factor']:.2f}")
    print(f"Net P/L:           ${stats['net_pnl']:+,.2f}")
    print(f"Losing Months:     {stats['losing_months']}/{stats['total_months']}")

    print(f"\n[MONTHLY BREAKDOWN]")
    print(f"{'-'*50}")
    for month, pnl in stats['monthly'].items():
        status = "WIN " if pnl >= 0 else "LOSS"
        print(f"  [{status}] {month}: ${pnl:+,.2f}")

    if baseline_stats:
        print(f"\n{'='*70}")
        print(f"COMPARISON VS BASELINE (v6.8.0)")
        print(f"{'='*70}")

        trade_diff = stats['total_trades'] - baseline_stats['total_trades']
        wr_diff = stats['win_rate'] - baseline_stats['win_rate']
        pf_diff = stats['profit_factor'] - baseline_stats['profit_factor']
        pnl_diff = stats['net_pnl'] - baseline_stats['net_pnl']
        lm_diff = stats['losing_months'] - baseline_stats['losing_months']

        print(f"\n  Metric               Baseline    Z-Score      Diff")
        print(f"  {'-'*55}")
        print(f"  Trades:              {baseline_stats['total_trades']:>8}    {stats['total_trades']:>8}    {trade_diff:>+8}")
        print(f"  Win Rate:            {baseline_stats['win_rate']:>7.1f}%   {stats['win_rate']:>7.1f}%   {wr_diff:>+7.1f}%")
        print(f"  Profit Factor:       {baseline_stats['profit_factor']:>8.2f}    {stats['profit_factor']:>8.2f}    {pf_diff:>+8.2f}")
        print(f"  Net P/L:           ${baseline_stats['net_pnl']:>+9,.0f}  ${stats['net_pnl']:>+9,.0f}  ${pnl_diff:>+9,.0f}")
        print(f"  Losing Months:       {baseline_stats['losing_months']:>8}    {stats['losing_months']:>8}    {lm_diff:>+8}")


async def main():
    timeframe = "H1"
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 30, tzinfo=timezone.utc)

    print(f"Z-SCORE TREND CONFIRMATION FILTER TEST (V2)")
    print(f"{'='*70}")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

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

    # Baseline stats
    baseline_stats = {
        'total_trades': 115,
        'win_rate': 49.6,
        'profit_factor': 5.09,
        'net_pnl': 22346,
        'losing_months': 0
    }

    # ============================================================
    # FIRST: Run baseline without Z-Score filter
    # ============================================================
    print(f"\n{'='*70}")
    print(f"Running BASELINE (no Z-Score filter)")
    print(f"{'='*70}")

    baseline_trades, baseline_max_dd, _, _, _, baseline_zscore_stats = run_backtest_zscore_v2(
        df, col_map, trend_threshold=0.0, extreme_threshold=999.0, use_filter=False
    )

    if baseline_trades:
        actual_baseline = calculate_stats(baseline_trades, baseline_max_dd)
        print(f"\nActual Baseline Results:")
        print(f"  Trades: {actual_baseline['total_trades']}")
        print(f"  Win Rate: {actual_baseline['win_rate']:.1f}%")
        print(f"  Profit Factor: {actual_baseline['profit_factor']:.2f}")
        print(f"  Net P/L: ${actual_baseline['net_pnl']:+,.2f}")
        print(f"  Losing Months: {actual_baseline['losing_months']}/{actual_baseline['total_months']}")

        # Z-Score distribution from baseline trades
        if baseline_zscore_stats['zscore_at_entry']:
            entry_zscores = baseline_zscore_stats['zscore_at_entry']
            print(f"\n  Z-Score Distribution at Entry (Baseline):")
            print(f"    Min: {min(entry_zscores):.2f}")
            print(f"    Max: {max(entry_zscores):.2f}")
            print(f"    Mean: {np.mean(entry_zscores):.2f}")
            print(f"    Std: {np.std(entry_zscores):.2f}")

            # Check how many would pass trend confirmation
            buy_trades = [t for t in baseline_trades if t.direction == 'BUY']
            sell_trades = [t for t in baseline_trades if t.direction == 'SELL']

            buy_pass_z0 = len([t for t in buy_trades if t.zscore_at_entry > 0])
            sell_pass_z0 = len([t for t in sell_trades if t.zscore_at_entry < 0])

            print(f"\n  Trend Confirmation Analysis:")
            print(f"    BUY trades: {len(buy_trades)}")
            print(f"    BUY with Z > 0 (bullish): {buy_pass_z0} ({buy_pass_z0/len(buy_trades)*100:.1f}% would pass)")
            print(f"    SELL trades: {len(sell_trades)}")
            print(f"    SELL with Z < 0 (bearish): {sell_pass_z0} ({sell_pass_z0/len(sell_trades)*100:.1f}% would pass)")

        # Use actual baseline for comparison
        baseline_stats = actual_baseline

    # ============================================================
    # TEST CONFIGURATIONS
    # ============================================================
    test_configs = [
        {'trend': 0.0, 'extreme': 2.5, 'name': 'Z>0 trend, block extreme 2.5'},
        {'trend': 0.5, 'extreme': 2.5, 'name': 'Z>0.5 trend, block extreme 2.5'},
        {'trend': 0.0, 'extreme': 2.0, 'name': 'Z>0 trend, block extreme 2.0'},
        {'trend': -0.5, 'extreme': 2.5, 'name': 'Z>-0.5 trend (relaxed), block 2.5'},
        {'trend': 0.0, 'extreme': 999, 'name': 'Z>0 trend only (no extreme block)'},
    ]

    results = []

    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")

        trades, max_dd, condition_stats, skip_stats, entry_stats, zscore_stats = run_backtest_zscore_v2(
            df, col_map,
            trend_threshold=config['trend'],
            extreme_threshold=config['extreme'],
            use_filter=True
        )

        if not trades:
            print(f"No trades for config: {config['name']}")
            results.append({
                'name': config['name'],
                'trend': config['trend'],
                'extreme': config['extreme'],
                'trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'net_pnl': 0,
                'losing_months': 0,
                'blocked': zscore_stats['zscore_blocked'],
                'extreme_blocked': zscore_stats['zscore_extreme_blocked'],
            })
            continue

        stats = calculate_stats(trades, max_dd)
        print_results(stats, trades, zscore_stats, config['trend'], config['extreme'], baseline_stats)

        results.append({
            'name': config['name'],
            'trend': config['trend'],
            'extreme': config['extreme'],
            'trades': stats['total_trades'],
            'win_rate': stats['win_rate'],
            'profit_factor': stats['profit_factor'],
            'net_pnl': stats['net_pnl'],
            'losing_months': stats['losing_months'],
            'blocked': zscore_stats['zscore_blocked'],
            'extreme_blocked': zscore_stats['zscore_extreme_blocked'],
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"Z-SCORE V2 FILTER TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\nBaseline (without Z-Score filter):")
    print(f"  Trades: {baseline_stats['total_trades']}, WR: {baseline_stats['win_rate']:.1f}%, PF: {baseline_stats['profit_factor']:.2f}, P/L: ${baseline_stats['net_pnl']:+,.0f}, Losing: {baseline_stats['losing_months']}")

    print(f"\n{'Config':<40} {'Trades':>7} {'WR':>7} {'PF':>7} {'P/L':>11} {'LM':>3} {'Blocked':>8}")
    print(f"{'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*11} {'-'*3} {'-'*8}")

    for r in results:
        print(f"{r['name']:<40} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['profit_factor']:>7.2f} ${r['net_pnl']:>+9,.0f} {r['losing_months']:>3} {r['blocked']:>8}")

    # Best result
    if results:
        valid_results = [r for r in results if r['trades'] > 0 and r['losing_months'] == baseline_stats['losing_months']]
        if valid_results:
            best = max(valid_results, key=lambda x: x['net_pnl'])
            print(f"\n{'='*70}")
            print(f"FINAL RECOMMENDATION")
            print(f"{'='*70}")

            if best['net_pnl'] > baseline_stats['net_pnl'] and best['losing_months'] <= baseline_stats['losing_months']:
                print(f"[POTENTIAL KEEP] Best config: {best['name']}")
                print(f"         P/L improvement: ${best['net_pnl'] - baseline_stats['net_pnl']:+,.0f}")
                print(f"         Losing months: {best['losing_months']} (baseline: {baseline_stats['losing_months']})")
            elif best['net_pnl'] >= baseline_stats['net_pnl'] * 0.95:
                print(f"[NEUTRAL] Config '{best['name']}' maintains similar performance")
                print(f"         May help filter out some bad trades")
            else:
                print(f"[REJECT] Z-Score trend confirmation filter reduces profit")
                print(f"         Best result still ${baseline_stats['net_pnl'] - best['net_pnl']:,.0f} less than baseline")
        else:
            print(f"\n[REJECT] All configurations either blocked all trades or added losing months")


if __name__ == "__main__":
    asyncio.run(main())
