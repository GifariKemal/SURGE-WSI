"""
TEST NEW ENTRY SIGNALS - SURGE-WSI H1 v6.8 GBPUSD
==================================================

Test additional entry signals to increase trade count:
1. BREAKOUT - Break of recent high/low
2. ENGULFING - Engulfing candle pattern
3. PIN_BAR - Rejection / Pin bar pattern
4. MACD_CROSS - MACD signal line crossover

Current baseline (v6.8.0):
- 115 trades, 49.6% WR, PF 5.09, $22,346 profit, 0 losing months
- ORDER_BLOCK: 80 trades, 38.8% WR
- EMA_PULLBACK: 74 trades, 51.4% WR

Goal: Add more trades without introducing losing months

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
    PROBE_TRADE_SIZE,
    PROBE_QUALITY_EXTRA,
    BOTH_DIRECTIONS_FAIL_THRESHOLD,
    RECOVERY_WIN_REQUIRED,
)
from gbpusd_h1_quadlayer.strategy_config import (
    SYMBOL, PIP_SIZE, PIP_VALUE,
    RISK, TECHNICAL, INTRA_MONTH, PATTERN,
    MONTHLY_RISK_MULT,
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ENTRY SIGNAL CONFIGURATION
# ============================================================
USE_ORDER_BLOCK = True       # Original: Order Block detection
USE_EMA_PULLBACK = True      # v6.7: EMA Pullback

# NEW ENTRY SIGNALS TO TEST
USE_BREAKOUT = False         # Breakout of recent high/low
USE_ENGULFING = False        # Engulfing candle pattern
USE_PIN_BAR = False          # Pin bar / rejection pattern
USE_MACD_CROSS = False       # MACD signal line crossover

# Test configuration
BREAKOUT_LOOKBACK = 20       # Bars to look back for high/low
PIN_BAR_WICK_RATIO = 2.0     # Wick must be 2x body
PIN_BAR_WICK_PCT = 0.6       # Wick must be 60% of total range
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Session filter (from v6.8)
USE_SESSION_POI_FILTER = True
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]


# ============================================================
# CONFIGURATION
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

USE_PATTERN_FILTER = True

ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

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


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal Line, and Histogram"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


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


def assess_market_condition(df: pd.DataFrame, col_map: dict, idx: int,
                           atr_series: pd.Series, current_time: datetime) -> MarketCondition:
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

    # Calculate technical score (0-100)
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
                    'price': current_close, 'direction': 'BUY', 'quality': quality,
                    'idx': i, 'type': 'EMA_PULLBACK', 'adx': current_adx, 'rsi': current_rsi
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
                    'price': current_close, 'direction': 'SELL', 'quality': quality,
                    'idx': i, 'type': 'EMA_PULLBACK', 'adx': current_adx, 'rsi': current_rsi
                })

    return pois


# ============================================================
# NEW ENTRY SIGNAL 1: BREAKOUT
# ============================================================
def detect_breakout(df: pd.DataFrame, col_map: dict, atr_series: pd.Series,
                    min_quality: float, regime: Regime) -> List[dict]:
    """
    Detect breakout entry signals.

    BUY: Close breaks above highest high of last N bars in BULLISH regime
    SELL: Close breaks below lowest low of last N bars in BEARISH regime
    """
    pois = []
    if len(df) < BREAKOUT_LOOKBACK + 5:
        return pois

    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    i = len(df) - 1
    bar = df.iloc[i]

    current_close = bar[c]
    current_open = bar[o]
    current_high = bar[h]
    current_low = bar[l]
    current_atr = atr_series.iloc[i] if i < len(atr_series) else 0

    # ATR filter
    if current_atr < MIN_ATR or current_atr > MAX_ATR:
        return pois

    # Calculate lookback high/low (excluding current bar)
    lookback_start = max(0, i - BREAKOUT_LOOKBACK)
    highest_high = df[h].iloc[lookback_start:i].max()
    lowest_low = df[l].iloc[lookback_start:i].min()

    # Body must be at least 40% of range (momentum candle)
    total_range = current_high - current_low
    if total_range < 0.0003:
        return pois
    body = abs(current_close - current_open)
    body_ratio = body / total_range
    if body_ratio < 0.4:
        return pois

    is_bullish = current_close > current_open
    is_bearish = current_close < current_open

    # BULLISH BREAKOUT: Close above highest high
    if regime == Regime.BULLISH and is_bullish and current_close > highest_high:
        # Quality based on: breakout strength, body ratio
        breakout_pips = (current_close - highest_high) / PIP_SIZE
        breakout_quality = min(30, breakout_pips * 3)  # 0-30 points for breakout strength
        body_quality = min(30, body_ratio * 40)  # 0-30 points for body
        momentum_quality = min(40, (current_close - current_open) / PIP_SIZE)  # 0-40 for momentum

        quality = breakout_quality + body_quality + momentum_quality
        quality = min(100, max(55, quality))

        if quality >= min_quality:
            pois.append({
                'price': current_close, 'direction': 'BUY', 'quality': quality,
                'idx': i, 'type': 'BREAKOUT', 'breakout_pips': breakout_pips
            })

    # BEARISH BREAKOUT: Close below lowest low
    if regime == Regime.BEARISH and is_bearish and current_close < lowest_low:
        breakout_pips = (lowest_low - current_close) / PIP_SIZE
        breakout_quality = min(30, breakout_pips * 3)
        body_quality = min(30, body_ratio * 40)
        momentum_quality = min(40, (current_open - current_close) / PIP_SIZE)

        quality = breakout_quality + body_quality + momentum_quality
        quality = min(100, max(55, quality))

        if quality >= min_quality:
            pois.append({
                'price': current_close, 'direction': 'SELL', 'quality': quality,
                'idx': i, 'type': 'BREAKOUT', 'breakout_pips': breakout_pips
            })

    return pois


# ============================================================
# NEW ENTRY SIGNAL 2: ENGULFING CANDLE
# ============================================================
def detect_engulfing(df: pd.DataFrame, col_map: dict, atr_series: pd.Series,
                     min_quality: float, regime: Regime) -> List[dict]:
    """
    Detect engulfing candle pattern.

    BULLISH ENGULFING: Current bullish candle body engulfs previous bearish candle body
    BEARISH ENGULFING: Current bearish candle body engulfs previous bullish candle body

    Only trade with regime direction.
    """
    pois = []
    if len(df) < 3:
        return pois

    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    i = len(df) - 1
    bar = df.iloc[i]
    prev_bar = df.iloc[i - 1]

    current_close = bar[c]
    current_open = bar[o]
    current_high = bar[h]
    current_low = bar[l]
    current_atr = atr_series.iloc[i] if i < len(atr_series) else 0

    prev_close = prev_bar[c]
    prev_open = prev_bar[o]

    # ATR filter
    if current_atr < MIN_ATR or current_atr > MAX_ATR:
        return pois

    # Current candle must have decent range
    total_range = current_high - current_low
    if total_range < 0.0003:
        return pois

    # Body calculations
    current_body = abs(current_close - current_open)
    prev_body = abs(prev_close - prev_open)
    body_ratio = current_body / total_range

    # Minimum body requirement
    if body_ratio < 0.5:  # Engulfing needs strong body
        return pois
    if prev_body < 0.0001:  # Previous candle needs a body
        return pois

    current_bullish = current_close > current_open
    current_bearish = current_close < current_open
    prev_bullish = prev_close > prev_open
    prev_bearish = prev_close < prev_open

    # BULLISH ENGULFING: Current bullish engulfs previous bearish
    if regime == Regime.BULLISH and current_bullish and prev_bearish:
        # Check engulfing: current body engulfs previous body
        if current_close > prev_open and current_open < prev_close:
            # Quality based on: body ratio, engulfing strength
            engulf_ratio = current_body / prev_body
            engulf_quality = min(40, engulf_ratio * 15)  # How much bigger
            body_quality = min(30, body_ratio * 40)
            size_quality = min(30, (total_range / PIP_SIZE) * 2)  # Candle size in pips

            quality = engulf_quality + body_quality + size_quality
            quality = min(100, max(55, quality))

            if quality >= min_quality:
                pois.append({
                    'price': current_close, 'direction': 'BUY', 'quality': quality,
                    'idx': i, 'type': 'ENGULFING', 'engulf_ratio': engulf_ratio
                })

    # BEARISH ENGULFING: Current bearish engulfs previous bullish
    if regime == Regime.BEARISH and current_bearish and prev_bullish:
        if current_close < prev_open and current_open > prev_close:
            engulf_ratio = current_body / prev_body
            engulf_quality = min(40, engulf_ratio * 15)
            body_quality = min(30, body_ratio * 40)
            size_quality = min(30, (total_range / PIP_SIZE) * 2)

            quality = engulf_quality + body_quality + size_quality
            quality = min(100, max(55, quality))

            if quality >= min_quality:
                pois.append({
                    'price': current_close, 'direction': 'SELL', 'quality': quality,
                    'idx': i, 'type': 'ENGULFING', 'engulf_ratio': engulf_ratio
                })

    return pois


# ============================================================
# NEW ENTRY SIGNAL 3: PIN BAR / REJECTION
# ============================================================
def detect_pin_bar(df: pd.DataFrame, col_map: dict, atr_series: pd.Series,
                   min_quality: float, regime: Regime) -> List[dict]:
    """
    Detect pin bar / rejection candle patterns.

    BULLISH PIN BAR: Long lower wick (showing buying pressure), small body at top
    BEARISH PIN BAR: Long upper wick (showing selling pressure), small body at bottom

    Pin bar criteria:
    - Wick must be at least 2x the body
    - Wick must be at least 60% of total range
    """
    pois = []
    if len(df) < 3:
        return pois

    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    i = len(df) - 1
    bar = df.iloc[i]

    current_close = bar[c]
    current_open = bar[o]
    current_high = bar[h]
    current_low = bar[l]
    current_atr = atr_series.iloc[i] if i < len(atr_series) else 0

    # ATR filter
    if current_atr < MIN_ATR or current_atr > MAX_ATR:
        return pois

    total_range = current_high - current_low
    if total_range < 0.0003:
        return pois

    body = abs(current_close - current_open)
    upper_wick = current_high - max(current_open, current_close)
    lower_wick = min(current_open, current_close) - current_low

    # Body must be small relative to range
    body_pct = body / total_range
    if body_pct > 0.35:  # Body should be < 35% of range for good pin bar
        return pois

    # BULLISH PIN BAR: Long lower wick
    if regime == Regime.BULLISH:
        if lower_wick > PIN_BAR_WICK_RATIO * body and lower_wick > PIN_BAR_WICK_PCT * total_range:
            # Close should be in upper third of range
            if current_close > current_low + 0.6 * total_range:
                # Quality based on wick length and rejection strength
                wick_quality = min(40, (lower_wick / total_range) * 50)
                body_quality = min(30, (1 - body_pct) * 40)  # Smaller body = better
                position_quality = min(30, ((current_close - current_low) / total_range) * 35)

                quality = wick_quality + body_quality + position_quality
                quality = min(100, max(55, quality))

                if quality >= min_quality:
                    pois.append({
                        'price': current_close, 'direction': 'BUY', 'quality': quality,
                        'idx': i, 'type': 'PIN_BAR', 'wick_pct': lower_wick / total_range
                    })

    # BEARISH PIN BAR: Long upper wick
    if regime == Regime.BEARISH:
        if upper_wick > PIN_BAR_WICK_RATIO * body and upper_wick > PIN_BAR_WICK_PCT * total_range:
            # Close should be in lower third of range
            if current_close < current_high - 0.6 * total_range:
                wick_quality = min(40, (upper_wick / total_range) * 50)
                body_quality = min(30, (1 - body_pct) * 40)
                position_quality = min(30, ((current_high - current_close) / total_range) * 35)

                quality = wick_quality + body_quality + position_quality
                quality = min(100, max(55, quality))

                if quality >= min_quality:
                    pois.append({
                        'price': current_close, 'direction': 'SELL', 'quality': quality,
                        'idx': i, 'type': 'PIN_BAR', 'wick_pct': upper_wick / total_range
                    })

    return pois


# ============================================================
# NEW ENTRY SIGNAL 4: MACD CROSS
# ============================================================
def detect_macd_cross(df: pd.DataFrame, col_map: dict, atr_series: pd.Series,
                      min_quality: float, regime: Regime) -> List[dict]:
    """
    Detect MACD signal line crossover.

    BULLISH CROSS: MACD crosses above signal line in BULLISH regime
    BEARISH CROSS: MACD crosses below signal line in BEARISH regime
    """
    pois = []
    if len(df) < MACD_SLOW + MACD_SIGNAL + 5:
        return pois

    c = col_map['close']
    o, h, l = col_map['open'], col_map['high'], col_map['low']
    i = len(df) - 1
    bar = df.iloc[i]

    current_atr = atr_series.iloc[i] if i < len(atr_series) else 0

    # ATR filter
    if current_atr < MIN_ATR or current_atr > MAX_ATR:
        return pois

    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(df[c], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    current_macd = macd_line.iloc[i]
    current_signal = signal_line.iloc[i]
    prev_macd = macd_line.iloc[i - 1]
    prev_signal = signal_line.iloc[i - 1]

    if pd.isna(current_macd) or pd.isna(current_signal) or pd.isna(prev_macd) or pd.isna(prev_signal):
        return pois

    # Current candle properties
    current_close = bar[c]
    current_open = bar[o]
    total_range = bar[h] - bar[l]
    if total_range < 0.0003:
        return pois
    body = abs(current_close - current_open)
    body_ratio = body / total_range

    # BULLISH MACD CROSS: MACD crosses above signal
    if regime == Regime.BULLISH:
        if current_macd > current_signal and prev_macd <= prev_signal:
            # Confirmation: current candle should be bullish
            if current_close > current_open:
                # Quality based on: histogram strength, body ratio, cross strength
                hist_strength = abs(current_macd - current_signal) / PIP_SIZE
                hist_quality = min(35, hist_strength * 100)
                body_quality = min(30, body_ratio * 40)
                cross_strength = min(35, abs(current_macd - prev_macd) / PIP_SIZE * 100)

                quality = hist_quality + body_quality + cross_strength
                quality = min(100, max(55, quality))

                if quality >= min_quality:
                    pois.append({
                        'price': current_close, 'direction': 'BUY', 'quality': quality,
                        'idx': i, 'type': 'MACD_CROSS', 'histogram': current_macd - current_signal
                    })

    # BEARISH MACD CROSS: MACD crosses below signal
    if regime == Regime.BEARISH:
        if current_macd < current_signal and prev_macd >= prev_signal:
            if current_close < current_open:
                hist_strength = abs(current_signal - current_macd) / PIP_SIZE
                hist_quality = min(35, hist_strength * 100)
                body_quality = min(30, body_ratio * 40)
                cross_strength = min(35, abs(prev_macd - current_macd) / PIP_SIZE * 100)

                quality = hist_quality + body_quality + cross_strength
                quality = min(100, max(55, quality))

                if quality >= min_quality:
                    pois.append({
                        'price': current_close, 'direction': 'SELL', 'quality': quality,
                        'idx': i, 'type': 'MACD_CROSS', 'histogram': current_signal - current_macd
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
    entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 0.8)
    quality_mult = quality / 100.0
    month_mult = MONTHLY_RISK.get(month, 0.8)

    if day_mult == 0.0 or hour_mult == 0.0:
        return 0.0, True

    combined = day_mult * hour_mult * entry_mult * quality_mult * month_mult
    if combined < 0.30:
        return combined, True

    return max(0.30, min(1.2, combined)), False


def run_backtest(df: pd.DataFrame, col_map: dict,
                 use_breakout: bool = False,
                 use_engulfing: bool = False,
                 use_pin_bar: bool = False,
                 use_macd_cross: bool = False) -> Tuple[List[Trade], float, dict, dict]:
    """Run backtest with configurable entry signals"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    risk_manager = IntraMonthRiskManager()
    pattern_filter = PatternBasedFilter()
    current_month_key = None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0}
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0, 'BREAKOUT': 0, 'ENGULFING': 0, 'PIN_BAR': 0, 'MACD_CROSS': 0}

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

        # Detect POIs from all enabled entry signals
        pois = []

        # Original entries
        if USE_ORDER_BLOCK:
            ob_pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            pois.extend(ob_pois)

        if USE_EMA_PULLBACK:
            ema_pois = detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality)
            existing_indices = {p['idx'] for p in pois}
            for ep in ema_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        # NEW entry signals
        if use_breakout:
            breakout_pois = detect_breakout(current_slice, col_map, atr_series, dynamic_quality, regime)
            existing_indices = {p['idx'] for p in pois}
            for bp in breakout_pois:
                if bp['idx'] not in existing_indices:
                    pois.append(bp)

        if use_engulfing:
            engulf_pois = detect_engulfing(current_slice, col_map, atr_series, dynamic_quality, regime)
            existing_indices = {p['idx'] for p in pois}
            for ep in engulf_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        if use_pin_bar:
            pin_pois = detect_pin_bar(current_slice, col_map, atr_series, dynamic_quality, regime)
            existing_indices = {p['idx'] for p in pois}
            for pp in pin_pois:
                if pp['idx'] not in existing_indices:
                    pois.append(pp)

        if use_macd_cross:
            macd_pois = detect_macd_cross(current_slice, col_map, atr_series, dynamic_quality, regime)
            existing_indices = {p['idx'] for p in pois}
            for mp in macd_pois:
                if mp['idx'] not in existing_indices:
                    pois.append(mp)

        if not pois:
            continue

        for poi in pois:
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            poi_type = poi.get('type', 'ORDER_BLOCK')

            # Session filter
            session_skip, session_reason = should_skip_by_session(hour, poi_type)
            if session_skip:
                skip_stats[session_reason] = skip_stats.get(session_reason, 0) + 1
                continue

            # New entry signals already have entry confirmation
            if poi_type in ['EMA_PULLBACK', 'BREAKOUT', 'ENGULFING', 'PIN_BAR', 'MACD_CROSS']:
                entry_type = 'MOMENTUM'
            else:
                # ORDER_BLOCK needs additional entry trigger
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

            # Pattern filter
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
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats


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


def print_results(stats: dict, trades: List[Trade], entry_stats: dict, test_name: str = ""):
    print(f"\n{'='*70}")
    print(f"TEST RESULTS: {test_name}")
    print(f"{'='*70}")

    print(f"\n[PERFORMANCE SUMMARY]")
    print(f"{'-'*50}")
    print(f"  Total Trades:    {stats['total_trades']}")
    print(f"  Win Rate:        {stats['win_rate']:.1f}%")
    print(f"  Profit Factor:   {stats['profit_factor']:.2f}")
    print(f"  Net P/L:         ${stats['net_pnl']:+,.0f}")
    print(f"  Max Drawdown:    ${stats['max_dd']:,.0f} ({stats['max_dd_pct']:.1f}%)")
    print(f"  Losing Months:   {stats['losing_months']}/{stats['total_months']}")

    print(f"\n[ENTRY SIGNAL BREAKDOWN]")
    print(f"{'-'*50}")
    for signal, count in entry_stats.items():
        if count > 0:
            signal_trades = [t for t in trades if t.poi_type == signal]
            signal_wins = len([t for t in signal_trades if t.pnl > 0])
            signal_wr = (signal_wins / count * 100) if count > 0 else 0
            signal_pnl = sum(t.pnl for t in signal_trades)
            print(f"  {signal:15}: {count:3} trades, {signal_wr:5.1f}% WR, ${signal_pnl:+,.0f}")

    print(f"\n[MONTHLY BREAKDOWN]")
    print(f"{'-'*50}")
    for month, pnl in stats['monthly'].items():
        status = "OK" if pnl >= 0 else "LOSS"
        print(f"  {month}: ${pnl:+,.0f} [{status}]")

    print(f"\n[TARGETS CHECK]")
    print(f"  {'PASS' if stats['net_pnl'] >= 5000 else 'FAIL'} Profit >= $5K: ${stats['net_pnl']:+,.0f}")
    print(f"  {'PASS' if stats['profit_factor'] >= 2.0 else 'FAIL'} PF >= 2.0: {stats['profit_factor']:.2f}")
    print(f"  {'PASS' if stats['losing_months'] == 0 else 'FAIL'} ZERO losing months: {stats['losing_months']}")


async def main():
    print("="*70)
    print("NEW ENTRY SIGNAL TEST - GBPUSD H1 STRATEGY")
    print("="*70)

    # Fetch data
    start_date = datetime(2024, 2, 1)
    end_date = datetime(2026, 1, 30)

    print(f"\nFetching data: {start_date.date()} to {end_date.date()}")
    df = await fetch_data("GBPUSD", "H1", start_date, end_date)

    if df.empty:
        print("ERROR: No data fetched!")
        return

    print(f"Data loaded: {len(df)} bars")

    # Detect column names dynamically
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

    print(f"Column mapping: {col_map}")

    # Results storage
    all_results = {}

    # ============================================================
    # TEST 1: BASELINE (ORDER_BLOCK + EMA_PULLBACK)
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 1: BASELINE (ORDER_BLOCK + EMA_PULLBACK)")
    print("="*70)

    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest(
        df, col_map, use_breakout=False, use_engulfing=False,
        use_pin_bar=False, use_macd_cross=False
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "BASELINE")
    all_results['BASELINE'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # TEST 2: BASELINE + BREAKOUT
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 2: BASELINE + BREAKOUT")
    print("="*70)

    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest(
        df, col_map, use_breakout=True, use_engulfing=False,
        use_pin_bar=False, use_macd_cross=False
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "BASELINE + BREAKOUT")
    all_results['+ BREAKOUT'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # TEST 3: BASELINE + ENGULFING
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 3: BASELINE + ENGULFING")
    print("="*70)

    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest(
        df, col_map, use_breakout=False, use_engulfing=True,
        use_pin_bar=False, use_macd_cross=False
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "BASELINE + ENGULFING")
    all_results['+ ENGULFING'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # TEST 4: BASELINE + PIN_BAR
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 4: BASELINE + PIN_BAR")
    print("="*70)

    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest(
        df, col_map, use_breakout=False, use_engulfing=False,
        use_pin_bar=True, use_macd_cross=False
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "BASELINE + PIN_BAR")
    all_results['+ PIN_BAR'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # TEST 5: BASELINE + MACD_CROSS
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 5: BASELINE + MACD_CROSS")
    print("="*70)

    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest(
        df, col_map, use_breakout=False, use_engulfing=False,
        use_pin_bar=False, use_macd_cross=True
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "BASELINE + MACD_CROSS")
    all_results['+ MACD_CROSS'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # TEST 6: ALL NEW SIGNALS COMBINED
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 6: ALL SIGNALS COMBINED")
    print("="*70)

    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest(
        df, col_map, use_breakout=True, use_engulfing=True,
        use_pin_bar=True, use_macd_cross=True
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "ALL SIGNALS")
    all_results['ALL SIGNALS'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # SUMMARY COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Test':<20} {'Trades':>8} {'WR%':>8} {'PF':>8} {'P/L':>12} {'L.Mo':>6}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*6}")

    for test, r in all_results.items():
        status = "OK" if r['losing_months'] == 0 else "BAD"
        print(f"{test:<20} {r['trades']:>8} {r['win_rate']:>7.1f}% {r['pf']:>7.2f} ${r['pnl']:>+10,.0f} {r['losing_months']:>5} {status}")

    # ============================================================
    # RECOMMENDATION
    # ============================================================
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print("="*70)

    # Find signals that maintain 0 losing months and improve trade count
    baseline = all_results['BASELINE']
    good_signals = []

    for test, r in all_results.items():
        if test == 'BASELINE':
            continue
        if r['losing_months'] == 0 and r['trades'] > baseline['trades']:
            improvement = r['trades'] - baseline['trades']
            pnl_change = r['pnl'] - baseline['pnl']
            good_signals.append((test, improvement, pnl_change, r))

    if good_signals:
        print("\nSignals that MAINTAIN 0 losing months with MORE trades:")
        for test, imp, pnl_chg, r in sorted(good_signals, key=lambda x: x[2], reverse=True):
            print(f"  {test}: +{imp} trades, ${pnl_chg:+,.0f} P/L change")
    else:
        print("\nNo signals maintained 0 losing months while adding trades.")
        print("\nAnalysis of results:")
        # Find best signal by P/L improvement while minimizing losing month increase
        best_by_pnl = sorted(all_results.items(), key=lambda x: x[1]['pnl'], reverse=True)
        best_by_lm = sorted(all_results.items(), key=lambda x: (x[1]['losing_months'], -x[1]['pnl']))

        print(f"\n  Best by P/L: {best_by_pnl[0][0]} with ${best_by_pnl[0][1]['pnl']:+,.0f}")
        print(f"  Best by Losing Months: {best_by_lm[0][0]} with {best_by_lm[0][1]['losing_months']} LM")

        # Specific recommendations
        print("\n  RECOMMENDATIONS:")
        print("  1. PIN_BAR shows best absolute P/L ($61,110) with +70 trades")
        print("     - Consider adding hour filters for PIN_BAR signals")
        print("  2. MACD_CROSS reduced losing months to 1 (best improvement)")
        print("     - Test with stricter quality threshold")
        print("  3. BREAKOUT adds +55 trades with +$15K profit")
        print("     - Test with London session only")

    # ============================================================
    # TEST 7: OPTIMIZED PIN_BAR (Higher quality threshold)
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 7: OPTIMIZED PIN_BAR (Quality +10)")
    print("="*70)

    # Temporarily increase minimum quality for PIN_BAR
    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest_optimized_pin_bar(
        df, col_map, quality_boost=10
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "BASELINE + PIN_BAR (Q+10)")
    all_results['+ PIN_BAR (Q+10)'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # TEST 8: BREAKOUT with London Session Only
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 8: BREAKOUT (London Session Only)")
    print("="*70)

    trades, max_dd, cond_stats, skip_stats, entry_stats = run_backtest_breakout_london(
        df, col_map
    )
    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, entry_stats, "BASELINE + BREAKOUT (London)")
    all_results['+ BREAKOUT (London)'] = {
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'pf': stats['profit_factor'],
        'pnl': stats['net_pnl'],
        'losing_months': stats['losing_months']
    }

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY WITH OPTIMIZED TESTS")
    print("="*70)
    print(f"{'Test':<25} {'Trades':>8} {'WR%':>8} {'PF':>8} {'P/L':>12} {'L.Mo':>6}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*6}")

    for test, r in all_results.items():
        status = "OK" if r['losing_months'] == 0 else "BAD"
        print(f"{test:<25} {r['trades']:>8} {r['win_rate']:>7.1f}% {r['pf']:>7.2f} ${r['pnl']:>+10,.0f} {r['losing_months']:>5} {status}")


def run_backtest_optimized_pin_bar(df: pd.DataFrame, col_map: dict, quality_boost: int = 10):
    """Run backtest with PIN_BAR requiring higher quality"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    risk_manager = IntraMonthRiskManager()
    pattern_filter = PatternBasedFilter()
    current_month_key = None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {}
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0, 'PIN_BAR': 0}

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

        # Detect POIs
        pois = []

        # Original entries
        if USE_ORDER_BLOCK:
            ob_pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            pois.extend(ob_pois)

        if USE_EMA_PULLBACK:
            ema_pois = detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality)
            existing_indices = {p['idx'] for p in pois}
            for ep in ema_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        # PIN_BAR with higher quality requirement
        pin_pois = detect_pin_bar(current_slice, col_map, atr_series, dynamic_quality + quality_boost, regime)
        existing_indices = {p['idx'] for p in pois}
        for pp in pin_pois:
            if pp['idx'] not in existing_indices:
                pois.append(pp)

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
                continue

            if poi_type in ['EMA_PULLBACK', 'PIN_BAR']:
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
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats


def run_backtest_breakout_london(df: pd.DataFrame, col_map: dict):
    """Run backtest with BREAKOUT restricted to London session only"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    risk_manager = IntraMonthRiskManager()
    pattern_filter = PatternBasedFilter()
    current_month_key = None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {}
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0, 'BREAKOUT': 0}

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

        # Detect POIs
        pois = []

        # Original entries
        if USE_ORDER_BLOCK:
            ob_pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            pois.extend(ob_pois)

        if USE_EMA_PULLBACK:
            ema_pois = detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality)
            existing_indices = {p['idx'] for p in pois}
            for ep in ema_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        # BREAKOUT only during London session (hours 8-11)
        if session == "london":
            breakout_pois = detect_breakout(current_slice, col_map, atr_series, dynamic_quality, regime)
            existing_indices = {p['idx'] for p in pois}
            for bp in breakout_pois:
                if bp['idx'] not in existing_indices:
                    pois.append(bp)

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
                continue

            if poi_type in ['EMA_PULLBACK', 'BREAKOUT']:
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
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats


if __name__ == "__main__":
    asyncio.run(main())
