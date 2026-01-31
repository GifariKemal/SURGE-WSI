"""
SURGE-WSI H1 v6.8.0 GBPUSD - RSI DIVERGENCE FILTER TEST
=========================================================

Testing RSI Divergence as an additional filter for the v6.8.0 strategy.

RSI Divergence Detection:
- Bullish Divergence: Price makes LOWER LOW but RSI makes HIGHER LOW -> BUY signal
- Bearish Divergence: Price makes HIGHER HIGH but RSI makes LOWER HIGH -> SELL signal

This is a TEST to see if RSI Divergence improves or hurts results.
We're testing with different lookback periods: 10, 20, 30 bars.

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
    calculate_lot_size,
    calculate_sl_tp,
    get_monthly_quality_adjustment,
    MONTHLY_TRADEABLE_PCT,
    SEASONAL_TEMPLATE,
    WARMUP_TRADES,
    ROLLING_WINDOW,
    ROLLING_WR_HALT,
    ROLLING_WR_CAUTION,
    RECOVERY_SIZE_MULT,
    CAUTION_SIZE_MULT,
)
from gbpusd_h1_quadlayer.strategy_config import (
    SYMBOL, PIP_SIZE, PIP_VALUE,
    RISK, TECHNICAL, INTRA_MONTH, PATTERN,
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# RSI DIVERGENCE FILTER CONFIG
# ============================================================
USE_RSI_DIVERGENCE = True        # Enable RSI Divergence filter
RSI_PERIOD = 14                  # RSI calculation period
RSI_LOOKBACK = 20                # Default lookback for divergence detection (will test 10, 20, 30)
RSI_SWING_TOLERANCE = 3          # Minimum bars between swing points


# ============================================================
# CONFIGURATION - Same as v6.8.0
# ============================================================
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = RISK.risk_percent
SL_ATR_MULT = RISK.sl_atr_mult
TP_RATIO = RISK.tp_ratio
MAX_LOT = RISK.max_lot
MAX_LOSS_PER_TRADE_PCT = RISK.max_loss_per_trade_pct
MIN_ATR = TECHNICAL.min_atr
MAX_ATR = TECHNICAL.max_atr

# Base Quality Thresholds
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

# Entry signals
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True

# Session filter (from v6.8.0)
USE_SESSION_POI_FILTER = True
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]


# ==========================================================
# LAYER 2: REAL-TIME TECHNICAL THRESHOLDS
# ==========================================================
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

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
    # RSI Divergence fields
    rsi_divergence: str = ""        # BULLISH, BEARISH, or ""
    rsi_value: float = 0.0          # RSI at entry


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


@dataclass
class RSIDivergence:
    """RSI Divergence signal"""
    divergence_type: str  # "BULLISH" or "BEARISH"
    price_swing1: float   # First swing point price
    price_swing2: float   # Second swing point price
    rsi_swing1: float     # First swing point RSI
    rsi_swing2: float     # Second swing point RSI
    bars_apart: int       # Bars between swing points
    strength: float       # Divergence strength (0-1)


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


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def find_swing_lows(prices: pd.Series, tolerance: int = 3) -> List[Tuple[int, float]]:
    """Find swing low points in price series"""
    swing_lows = []
    for i in range(tolerance, len(prices) - tolerance):
        is_swing_low = True
        for j in range(1, tolerance + 1):
            if prices.iloc[i] >= prices.iloc[i - j] or prices.iloc[i] >= prices.iloc[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append((i, prices.iloc[i]))
    return swing_lows


def find_swing_highs(prices: pd.Series, tolerance: int = 3) -> List[Tuple[int, float]]:
    """Find swing high points in price series"""
    swing_highs = []
    for i in range(tolerance, len(prices) - tolerance):
        is_swing_high = True
        for j in range(1, tolerance + 1):
            if prices.iloc[i] <= prices.iloc[i - j] or prices.iloc[i] <= prices.iloc[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((i, prices.iloc[i]))
    return swing_highs


def detect_rsi_divergence(df: pd.DataFrame, col_map: dict, idx: int,
                          lookback: int = 20, rsi_period: int = 14,
                          swing_tolerance: int = 3) -> Tuple[Optional[RSIDivergence], Optional[RSIDivergence]]:
    """
    Detect RSI divergence in the given price data.

    Returns:
        Tuple of (bullish_divergence, bearish_divergence)
        Each can be None if no divergence detected
    """
    if idx < lookback + rsi_period + swing_tolerance:
        return None, None

    c = col_map['close']
    l = col_map['low']
    h = col_map['high']

    # Get the slice of data we're looking at
    start_idx = max(0, idx - lookback)
    price_slice = df[c].iloc[start_idx:idx + 1]
    low_slice = df[l].iloc[start_idx:idx + 1]
    high_slice = df[h].iloc[start_idx:idx + 1]

    # Calculate RSI for the slice
    rsi = calculate_rsi(df[c], rsi_period)
    rsi_slice = rsi.iloc[start_idx:idx + 1]

    bullish_div = None
    bearish_div = None

    # Find swing lows for bullish divergence
    price_swing_lows = find_swing_lows(low_slice, swing_tolerance)
    rsi_swing_lows = find_swing_lows(rsi_slice, swing_tolerance)

    # Check for BULLISH divergence: Price LL + RSI HL
    if len(price_swing_lows) >= 2 and len(rsi_swing_lows) >= 2:
        # Get the two most recent swing lows
        price_low1_idx, price_low1 = price_swing_lows[-2]
        price_low2_idx, price_low2 = price_swing_lows[-1]

        # Find corresponding RSI swing lows
        rsi_low1_idx, rsi_low1 = rsi_swing_lows[-2]
        rsi_low2_idx, rsi_low2 = rsi_swing_lows[-1]

        # Check: Price makes LOWER LOW but RSI makes HIGHER LOW
        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
            bars_apart = price_low2_idx - price_low1_idx
            if bars_apart >= swing_tolerance:
                # Calculate divergence strength
                price_diff = abs(price_low1 - price_low2) / price_low1
                rsi_diff = (rsi_low2 - rsi_low1) / 100
                strength = min(1.0, (price_diff * 100 + rsi_diff) / 2)

                bullish_div = RSIDivergence(
                    divergence_type="BULLISH",
                    price_swing1=price_low1,
                    price_swing2=price_low2,
                    rsi_swing1=rsi_low1,
                    rsi_swing2=rsi_low2,
                    bars_apart=bars_apart,
                    strength=strength
                )

    # Find swing highs for bearish divergence
    price_swing_highs = find_swing_highs(high_slice, swing_tolerance)
    rsi_swing_highs = find_swing_highs(rsi_slice, swing_tolerance)

    # Check for BEARISH divergence: Price HH + RSI LH
    if len(price_swing_highs) >= 2 and len(rsi_swing_highs) >= 2:
        # Get the two most recent swing highs
        price_high1_idx, price_high1 = price_swing_highs[-2]
        price_high2_idx, price_high2 = price_swing_highs[-1]

        # Find corresponding RSI swing highs
        rsi_high1_idx, rsi_high1 = rsi_swing_highs[-2]
        rsi_high2_idx, rsi_high2 = rsi_swing_highs[-1]

        # Check: Price makes HIGHER HIGH but RSI makes LOWER HIGH
        if price_high2 > price_high1 and rsi_high2 < rsi_high1:
            bars_apart = price_high2_idx - price_high1_idx
            if bars_apart >= swing_tolerance:
                # Calculate divergence strength
                price_diff = abs(price_high2 - price_high1) / price_high1
                rsi_diff = (rsi_high1 - rsi_high2) / 100
                strength = min(1.0, (price_diff * 100 + rsi_diff) / 2)

                bearish_div = RSIDivergence(
                    divergence_type="BEARISH",
                    price_swing1=price_high1,
                    price_swing2=price_high2,
                    rsi_swing1=rsi_high1,
                    rsi_swing2=rsi_high2,
                    bars_apart=bars_apart,
                    strength=strength
                )

    return bullish_div, bearish_div


def check_rsi_divergence_filter(direction: str, bullish_div: Optional[RSIDivergence],
                                 bearish_div: Optional[RSIDivergence]) -> Tuple[bool, str]:
    """
    Check if trade should be allowed based on RSI divergence.

    Returns:
        Tuple of (allowed, divergence_type)
    """
    if direction == "BUY":
        if bullish_div is not None:
            return True, "BULLISH"
        return False, ""
    else:  # SELL
        if bearish_div is not None:
            return True, "BEARISH"
        return False, ""


def should_skip_by_session(hour: int, poi_type: str) -> tuple[bool, str]:
    """Check if trade should be skipped based on session analysis."""
    if not USE_SESSION_POI_FILTER:
        return False, ""

    if hour in SKIP_HOURS:
        return True, f"HOUR_{hour}_SKIP"

    if poi_type == "ORDER_BLOCK" and hour in SKIP_ORDER_BLOCK_HOURS:
        return True, f"OB_HOUR_{hour}_SKIP"

    if poi_type == "EMA_PULLBACK" and hour in SKIP_EMA_PULLBACK_HOURS:
        return True, f"EMA_HOUR_{hour}_SKIP"

    return False, ""


def assess_market_condition(df: pd.DataFrame, col_map: dict, idx: int,
                           atr_series: pd.Series, current_time: datetime) -> MarketCondition:
    """Dual-layer market condition assessment"""
    lookback = 20
    start_idx = max(0, idx - lookback)

    h, l, c = col_map['high'], col_map['low'], col_map['close']

    # ATR Stability
    recent_atr = atr_series.iloc[start_idx:idx+1]
    if len(recent_atr) > 5 and recent_atr.mean() > 0:
        atr_cv = recent_atr.std() / recent_atr.mean()
    else:
        atr_cv = 0.5

    # Price Efficiency
    if idx >= lookback:
        net_move = abs(df[c].iloc[idx] - df[c].iloc[start_idx])
        total_move = sum(abs(df[c].iloc[i] - df[c].iloc[i-1]) for i in range(start_idx+1, idx+1))
        efficiency = net_move / total_move if total_move > 0 else 0
    else:
        efficiency = 0.1

    # Trend Strength (ADX)
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

    # Calculate technical score
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
        tech_label = "TECH_GOOD"
    elif score >= 40:
        technical_quality = BASE_QUALITY
        tech_label = "TECH_NORMAL"
    else:
        technical_quality = MAX_QUALITY_BAD
        tech_label = "TECH_BAD"

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
    """Standard regime detection using EMA crossover."""
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


def calculate_adx(df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
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
    """Detect EMA Pullback entry signals"""
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
                    'adx': current_adx,
                    'rsi': current_rsi,
                    'body_ratio': body_ratio
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
                    'adx': current_adx,
                    'rsi': current_rsi,
                    'body_ratio': body_ratio
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


def run_backtest(df: pd.DataFrame, col_map: dict, rsi_lookback: int = 20) -> Tuple[List[Trade], float, dict, dict]:
    """Run backtest with RSI Divergence filter"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Layer 3: Intra-month risk manager
    risk_manager = IntraMonthRiskManager()

    # Layer 4: Pattern-based filter
    pattern_filter = PatternBasedFilter()
    current_month_key = None

    # Calculate RSI for the entire series
    c = col_map['close']
    rsi_series = calculate_rsi(df[c], RSI_PERIOD)

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0, 'RSI_DIV_BLOCKED': 0}
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0}
    rsi_div_stats = {'bullish_detected': 0, 'bearish_detected': 0, 'bullish_used': 0, 'bearish_used': 0}

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]
        current_rsi = rsi_series.iloc[i] if i < len(rsi_series) else 50

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

                total_pnl = pips * position.lot_size * PIP_VALUE
                max_loss = balance * (MAX_LOSS_PER_TRADE_PCT / 100)

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

                risk_manager.record_trade(total_pnl, current_time)

                if USE_PATTERN_FILTER:
                    pattern_filter.record_trade(position.direction, total_pnl, current_time)

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

        # LAYER 3: Check intra-month risk manager
        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            if 'MONTH' in skip_reason:
                skip_stats['MONTH_STOPPED'] += 1
            elif 'DAY' in skip_reason:
                skip_stats['DAY_STOPPED'] += 1
            continue

        # Reset filters if month changed
        month_key = (current_time.year, current_time.month)
        if month_key != current_month_key:
            current_month_key = month_key
            if USE_PATTERN_FILTER:
                pattern_filter.reset_for_month(current_time.month)

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # RSI DIVERGENCE DETECTION
        bullish_div, bearish_div = detect_rsi_divergence(
            df, col_map, i,
            lookback=rsi_lookback,
            rsi_period=RSI_PERIOD,
            swing_tolerance=RSI_SWING_TOLERANCE
        )

        if bullish_div:
            rsi_div_stats['bullish_detected'] += 1
        if bearish_div:
            rsi_div_stats['bearish_detected'] += 1

        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        if intra_month_adj > 0:
            skip_stats['DYNAMIC_ADJ'] += 1

        condition_stats[market_cond.label] = condition_stats.get(market_cond.label, 0) + 1

        # Detect POIs
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

            # SESSION-BASED FILTER
            session_skip, session_reason = should_skip_by_session(hour, poi_type)
            if session_skip:
                skip_stats[session_reason] = skip_stats.get(session_reason, 0) + 1
                continue

            # RSI DIVERGENCE FILTER - Only allow trades with divergence confirmation
            if USE_RSI_DIVERGENCE:
                div_allowed, div_type = check_rsi_divergence_filter(poi['direction'], bullish_div, bearish_div)
                if not div_allowed:
                    skip_stats['RSI_DIV_BLOCKED'] += 1
                    continue

                # Track divergence usage
                if div_type == "BULLISH":
                    rsi_div_stats['bullish_used'] += 1
                elif div_type == "BEARISH":
                    rsi_div_stats['bearish_used'] += 1
            else:
                div_type = ""

            # Entry trigger for ORDER_BLOCK
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

            # LAYER 4: Pattern-based filter
            pattern_size_mult = 1.0
            pattern_extra_q = 0
            if USE_PATTERN_FILTER:
                pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(poi['direction'])
                if not pattern_can_trade:
                    skip_stats['PATTERN_STOPPED'] += 1
                    continue

            if pattern_extra_q > 0:
                effective_quality = dynamic_quality + pattern_extra_q
                if poi['quality'] < effective_quality:
                    continue

            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO
            risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult * pattern_size_mult
            lot_size = risk_amount / (sl_pips * PIP_VALUE)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            entry_stats[poi_type] = entry_stats.get(poi_type, 0) + 1

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
                rsi_divergence=div_type,
                rsi_value=current_rsi
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats, rsi_div_stats


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


def print_results(stats: dict, trades: List[Trade], condition_stats: dict, skip_stats: dict,
                  entry_stats: dict, rsi_div_stats: dict, lookback: int):
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS - H1 v6.8.0 + RSI DIVERGENCE FILTER (Lookback={lookback})")
    print(f"{'='*70}")

    print(f"\n[RSI DIVERGENCE FILTER CONFIG]")
    print(f"{'-'*50}")
    print(f"  RSI Period: {RSI_PERIOD}")
    print(f"  Lookback Bars: {lookback}")
    print(f"  Swing Tolerance: {RSI_SWING_TOLERANCE}")
    print(f"  Filter Mode: REQUIRE divergence for entry")

    print(f"\n[RSI DIVERGENCE STATS]")
    print(f"{'-'*50}")
    print(f"  Bullish Divergences Detected: {rsi_div_stats['bullish_detected']}")
    print(f"  Bearish Divergences Detected: {rsi_div_stats['bearish_detected']}")
    print(f"  Bullish Used for Trades: {rsi_div_stats['bullish_used']}")
    print(f"  Bearish Used for Trades: {rsi_div_stats['bearish_used']}")
    print(f"  Trades Blocked (no divergence): {skip_stats.get('RSI_DIV_BLOCKED', 0)}")

    if skip_stats:
        print(f"\n[PROTECTION STATS]")
        print(f"{'-'*50}")
        print(f"  Month circuit breaker: {skip_stats.get('MONTH_STOPPED', 0)}")
        print(f"  Day stop: {skip_stats.get('DAY_STOPPED', 0)}")
        print(f"  Dynamic quality adj: {skip_stats.get('DYNAMIC_ADJ', 0)}")
        print(f"  Pattern filter stops: {skip_stats.get('PATTERN_STOPPED', 0)}")
        print(f"  RSI Divergence blocked: {skip_stats.get('RSI_DIV_BLOCKED', 0)}")

    if entry_stats:
        print(f"\n[ENTRY SIGNALS USED]")
        print(f"{'-'*50}")
        for sig_type, count in sorted(entry_stats.items(), key=lambda x: -x[1]):
            sig_trades = [t for t in trades if t.poi_type == sig_type]
            sig_wins = len([t for t in sig_trades if t.pnl > 0])
            sig_wr = (sig_wins / len(sig_trades) * 100) if sig_trades else 0
            sig_pnl = sum(t.pnl for t in sig_trades)
            print(f"  {sig_type}: {count} trades, {sig_wr:.1f}% WR, ${sig_pnl:+,.0f}")

    # RSI Divergence trade analysis
    print(f"\n[TRADES BY RSI DIVERGENCE TYPE]")
    print(f"{'-'*50}")
    for div_type in ['BULLISH', 'BEARISH']:
        div_trades = [t for t in trades if t.rsi_divergence == div_type]
        if div_trades:
            wins = len([t for t in div_trades if t.pnl > 0])
            total = len(div_trades)
            wr = wins / total * 100
            net = sum(t.pnl for t in div_trades)
            avg_rsi = sum(t.rsi_value for t in div_trades) / total
            print(f"  {div_type:12} {total:>3} trades, {wr:>5.1f}% WR, RSI={avg_rsi:.1f}, ${net:>+10,.0f}")

    print(f"\n[MARKET CONDITIONS]")
    print(f"{'-'*50}")
    for cond, count in sorted(condition_stats.items(), key=lambda x: -x[1]):
        print(f"  {cond}: {count} bars")

    print(f"\n[PERFORMANCE]")
    print(f"{'-'*50}")
    print(f"Total Trades:      {stats['total_trades']}")
    print(f"Trades/Day:        {stats['trades_per_day']:.2f}")
    print(f"Win Rate:          {stats['win_rate']:.1f}%")
    print(f"Profit Factor:     {stats['profit_factor']:.2f}")

    print(f"\n[PROFIT/LOSS]")
    print(f"{'-'*50}")
    print(f"Initial Balance:   ${INITIAL_BALANCE:,.2f}")
    print(f"Final Balance:     ${stats['final_balance']:,.2f}")
    print(f"Net P/L:           ${stats['net_pnl']:+,.2f}")
    print(f"Total Return:      {(stats['net_pnl']/INITIAL_BALANCE)*100:+.1f}%")
    print(f"Avg Win:           ${stats['avg_win']:,.2f}")
    print(f"Avg Loss:          ${stats['avg_loss']:,.2f}")

    print(f"\n[MONTHLY BREAKDOWN]")
    print(f"{'-'*50}")
    print(f"Losing Months:     {stats['losing_months']}/{stats['total_months']}")

    for month, pnl in stats['monthly'].items():
        status = "WIN " if pnl >= 0 else "LOSS"
        year = month.year
        mon = month.month
        tradeable = MONTHLY_TRADEABLE_PCT.get((year, mon), SEASONAL_TEMPLATE.get(mon, 65))
        adj = get_monthly_quality_adjustment(datetime(year, mon, 1))
        print(f"  [{status}] {month}: ${pnl:+,.2f} (tradeable={tradeable}%, adj=+{adj})")

    print(f"\n{'='*70}")
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
    print(f"{'='*70}")

    return target_met


async def main():
    timeframe = "H1"
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 30, tzinfo=timezone.utc)

    print(f"="*70)
    print(f"SURGE-WSI H1 v6.8.0 GBPUSD - RSI DIVERGENCE FILTER TEST")
    print(f"="*70)
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"\nRSI Divergence Filter:")
    print(f"  - BULLISH: Price LL + RSI HL -> BUY only")
    print(f"  - BEARISH: Price HH + RSI LH -> SELL only")
    print(f"  - Testing lookbacks: 10, 20, 30 bars")
    print(f"="*70)

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

    # Store results for comparison
    results = {}

    # Test different lookback periods
    for lookback in [10, 20, 30]:
        print(f"\n{'#'*70}")
        print(f"# Testing RSI Divergence Lookback = {lookback} bars")
        print(f"{'#'*70}")

        trades, max_dd, condition_stats, skip_stats, entry_stats, rsi_div_stats = run_backtest(df, col_map, lookback)

        if not trades:
            print(f"No trades executed with lookback={lookback}")
            results[lookback] = None
            continue

        stats = calculate_stats(trades, max_dd)
        targets_met = print_results(stats, trades, condition_stats, skip_stats, entry_stats, rsi_div_stats, lookback)

        results[lookback] = {
            'stats': stats,
            'trades': len(trades),
            'targets_met': targets_met,
            'rsi_div_stats': rsi_div_stats,
            'skip_stats': skip_stats
        }

    # Final comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON: RSI DIVERGENCE FILTER TEST RESULTS")
    print(f"{'='*70}")
    print(f"\nBaseline v6.8.0: 115 trades, 49.6% WR, PF 5.09, $22,346 profit, 0 losing months")
    print(f"\n{'Lookback':<10} {'Trades':<8} {'WR':<8} {'PF':<8} {'Net P/L':<12} {'Losing Mo':<10} {'Filtered':<10}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")

    for lookback, result in results.items():
        if result is None:
            print(f"{lookback:<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
        else:
            stats = result['stats']
            filtered = result['skip_stats'].get('RSI_DIV_BLOCKED', 0)
            print(f"{lookback:<10} {stats['total_trades']:<8} {stats['win_rate']:.1f}%{'':<3} {stats['profit_factor']:.2f}{'':<4} ${stats['net_pnl']:>+10,.0f} {stats['losing_months']:<10} {filtered:<10}")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION:")
    print(f"{'='*70}")

    # Analyze results
    best_result = None
    best_lookback = None
    baseline_pnl = 22346
    baseline_trades = 115
    baseline_wr = 49.6
    baseline_pf = 5.09

    for lookback, result in results.items():
        if result and result['stats']['net_pnl'] > 0:
            if best_result is None or result['stats']['net_pnl'] > best_result['stats']['net_pnl']:
                best_result = result
                best_lookback = lookback

    if best_result:
        stats = best_result['stats']

        # Compare to baseline
        pnl_diff = stats['net_pnl'] - baseline_pnl
        pnl_pct = (pnl_diff / baseline_pnl) * 100
        trade_diff = stats['total_trades'] - baseline_trades
        wr_diff = stats['win_rate'] - baseline_wr
        pf_diff = stats['profit_factor'] - baseline_pf

        print(f"\nBest Lookback: {best_lookback}")
        print(f"  Net P/L Change: ${pnl_diff:+,.0f} ({pnl_pct:+.1f}%)")
        print(f"  Trade Count Change: {trade_diff:+d}")
        print(f"  Win Rate Change: {wr_diff:+.1f}%")
        print(f"  PF Change: {pf_diff:+.2f}")
        print(f"  Losing Months: {stats['losing_months']}")

        # Final recommendation
        if stats['net_pnl'] > baseline_pnl and stats['losing_months'] == 0:
            print(f"\n>>> KEEP RSI DIVERGENCE FILTER (Lookback={best_lookback})")
            print(f"    Improved profit while maintaining 0 losing months!")
        elif stats['net_pnl'] > baseline_pnl * 0.9 and stats['win_rate'] > baseline_wr:
            print(f"\n>>> CONSIDER RSI DIVERGENCE (Lookback={best_lookback})")
            print(f"    Higher WR but slightly lower profit.")
        else:
            print(f"\n>>> REJECT RSI DIVERGENCE FILTER")
            print(f"    Did not improve baseline results.")
    else:
        print(f"\n>>> REJECT RSI DIVERGENCE FILTER")
        print(f"    All tests produced negative or no results.")

    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
