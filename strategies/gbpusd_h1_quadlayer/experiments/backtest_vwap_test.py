"""
SURGE-WSI H1 GBPUSD - VWAP FILTER TEST
========================================

Testing VWAP (Volume-Weighted Average Price) as a directional filter.

Baseline (v6.8.0): 115 trades, 49.6% WR, PF 5.09, $22,346 profit, 0 losing months

VWAP Filter Logic:
- BUY only when price > VWAP (bullish bias)
- SELL only when price < VWAP (bearish bias)
- VWAP resets daily at 00:00 UTC
- Uses tick volume from MT5 data

Expected outcome: Fewer trades with potentially higher win rate by filtering
against-trend trades.

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
from dataclasses import dataclass
from enum import Enum

from config import config
from src.data.db_handler import DBHandler
from src.utils.telegram import TelegramNotifier, TelegramFormatter

# Import VWAP filter
from gbpusd_h1_quadlayer.vwap_filter import (
    VWAPFilter,
    VWAPState,
    calculate_vwap_series,
    get_vwap_signal,
    VWAP_RESET_HOUR,
    VWAP_MIN_BARS,
    VWAP_STD_BANDS,
    VWAP_BUFFER_PIPS,
)

# Import shared trading filters
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
# VWAP FILTER TEST CONFIGURATION
# ============================================================
USE_VWAP_FILTER = True          # Main switch for VWAP filter
VWAP_RESET_HOUR_CONFIG = 0      # Reset at 00:00 UTC
VWAP_MIN_BARS_CONFIG = 3        # Minimum bars before VWAP is valid
VWAP_STD_BANDS_CONFIG = 0       # 0 = disabled, 2.0 = use bands
VWAP_BUFFER_PIPS_CONFIG = 5.0   # Buffer zone around VWAP

# Telegram notifications
SEND_TO_TELEGRAM = True


# ============================================================
# STANDARD CONFIGURATION (same as v6.8)
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

USE_PATTERN_FILTER = True
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True
USE_SESSION_POI_FILTER = True

# Session filter (v6.8)
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]

# Layer 2 thresholds
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

# Risk multipliers
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
    # VWAP fields
    vwap_value: float = 0.0
    vwap_distance_pips: float = 0.0
    vwap_signal: str = ""


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
    """Assess market condition with dual-layer quality filter"""
    lookback = 20
    start_idx = max(0, idx - lookback)
    h, l, c = col_map['high'], col_map['low'], col_map['close']

    # Layer 2: Technical indicators
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
                    'type': 'EMA_PULLBACK'
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
                    'type': 'EMA_PULLBACK'
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


def should_skip_by_session(hour: int, poi_type: str) -> tuple[bool, str]:
    if not USE_SESSION_POI_FILTER:
        return False, ""
    if hour in SKIP_HOURS:
        return True, f"HOUR_{hour}_SKIP"
    if poi_type == "ORDER_BLOCK" and hour in SKIP_ORDER_BLOCK_HOURS:
        return True, f"OB_HOUR_{hour}_SKIP"
    if poi_type == "EMA_PULLBACK" and hour in SKIP_EMA_PULLBACK_HOURS:
        return True, f"EMA_HOUR_{hour}_SKIP"
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


def run_backtest(df: pd.DataFrame, col_map: dict) -> Tuple[List[Trade], float, dict, dict, dict, dict]:
    """Run backtest with VWAP filter"""
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

    # VWAP Filter (NEW!)
    vwap_filter = VWAPFilter(
        reset_hour=VWAP_RESET_HOUR_CONFIG,
        min_bars=VWAP_MIN_BARS_CONFIG,
        std_bands=VWAP_STD_BANDS_CONFIG,
        buffer_pips=VWAP_BUFFER_PIPS_CONFIG,
        pip_size=PIP_SIZE
    ) if USE_VWAP_FILTER else None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0}
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0}
    vwap_stats = {
        'vwap_blocked': 0,
        'vwap_aligned': 0,
        'vwap_warmup': 0,
        'vwap_buffer': 0,
        'vwap_bearish_bias': 0,
        'vwap_bullish_bias': 0,
        'vwap_above_upper': 0,
        'vwap_below_lower': 0,
    }

    # Get volume column
    vol_col = col_map.get('volume', 'Volume')
    has_volume = vol_col in df.columns

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]

        # Update VWAP with each bar
        if USE_VWAP_FILTER and vwap_filter:
            volume = current_bar.get(vol_col, 1) if has_volume else 1
            if pd.isna(volume) or volume <= 0:
                volume = 1
            vwap_filter.update(
                high=current_bar[col_map['high']],
                low=current_bar[col_map['low']],
                close=current_bar[col_map['close']],
                volume=volume,
                current_time=current_time
            )

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

        # Layer 3: Check intra-month risk
        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            if 'MONTH' in skip_reason:
                skip_stats['MONTH_STOPPED'] += 1
            elif 'DAY' in skip_reason:
                skip_stats['DAY_STOPPED'] += 1
            continue

        # Reset pattern filter for new month
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

            # Session filter
            session_skip, session_reason = should_skip_by_session(hour, poi_type)
            if session_skip:
                skip_stats[session_reason] = skip_stats.get(session_reason, 0) + 1
                continue

            # ============================================================
            # VWAP FILTER CHECK (NEW!)
            # ============================================================
            vwap_value = 0.0
            vwap_distance = 0.0
            vwap_signal = ""

            if USE_VWAP_FILTER and vwap_filter:
                vwap_can_trade, vwap_reason = vwap_filter.check_trade(poi['direction'], current_price)

                # Track VWAP stats
                if 'WARMUP' in vwap_reason:
                    vwap_stats['vwap_warmup'] += 1
                elif 'BUFFER' in vwap_reason:
                    vwap_stats['vwap_buffer'] += 1
                    vwap_stats['vwap_blocked'] += 1
                elif 'BEARISH_BIAS' in vwap_reason:
                    vwap_stats['vwap_bearish_bias'] += 1
                    vwap_stats['vwap_blocked'] += 1
                elif 'BULLISH_BIAS' in vwap_reason:
                    vwap_stats['vwap_bullish_bias'] += 1
                    vwap_stats['vwap_blocked'] += 1
                elif 'ABOVE_UPPER' in vwap_reason:
                    vwap_stats['vwap_above_upper'] += 1
                    vwap_stats['vwap_blocked'] += 1
                elif 'BELOW_LOWER' in vwap_reason:
                    vwap_stats['vwap_below_lower'] += 1
                    vwap_stats['vwap_blocked'] += 1
                elif 'ALIGNED' in vwap_reason:
                    vwap_stats['vwap_aligned'] += 1

                if not vwap_can_trade:
                    skip_stats['VWAP_BLOCKED'] = skip_stats.get('VWAP_BLOCKED', 0) + 1
                    continue

                # Record VWAP values for trade
                vwap_value = vwap_filter.state.vwap
                vwap_distance = (current_price - vwap_value) / PIP_SIZE
                vwap_signal = vwap_reason

            # EMA_PULLBACK entry confirmation
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

            # Layer 4: Pattern filter
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
                vwap_value=vwap_value,
                vwap_distance_pips=vwap_distance,
                vwap_signal=vwap_signal
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats, vwap_stats


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


def print_results(stats: dict, trades: List[Trade], condition_stats: dict,
                  skip_stats: dict, entry_stats: dict, vwap_stats: dict):
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS - VWAP FILTER TEST")
    print(f"{'='*70}")

    # VWAP Configuration
    print(f"\n[VWAP FILTER CONFIGURATION]")
    print(f"{'-'*50}")
    print(f"  VWAP Filter: {'ENABLED' if USE_VWAP_FILTER else 'DISABLED'}")
    print(f"  Reset Hour: {VWAP_RESET_HOUR_CONFIG}:00 UTC")
    print(f"  Min Bars for Valid VWAP: {VWAP_MIN_BARS_CONFIG}")
    print(f"  Buffer Zone: {VWAP_BUFFER_PIPS_CONFIG} pips")
    print(f"  Std Dev Bands: {VWAP_STD_BANDS_CONFIG if VWAP_STD_BANDS_CONFIG > 0 else 'DISABLED'}")
    print(f"  Filter Logic:")
    print(f"    - BUY only when price > VWAP (bullish bias)")
    print(f"    - SELL only when price < VWAP (bearish bias)")

    # VWAP Stats
    if USE_VWAP_FILTER:
        print(f"\n[VWAP FILTER STATS]")
        print(f"{'-'*50}")
        print(f"  VWAP-aligned trades: {vwap_stats.get('vwap_aligned', 0)}")
        print(f"  VWAP-blocked trades: {vwap_stats.get('vwap_blocked', 0)}")
        print(f"    - Buffer zone: {vwap_stats.get('vwap_buffer', 0)}")
        print(f"    - Bearish bias (BUY blocked): {vwap_stats.get('vwap_bearish_bias', 0)}")
        print(f"    - Bullish bias (SELL blocked): {vwap_stats.get('vwap_bullish_bias', 0)}")
        if VWAP_STD_BANDS_CONFIG > 0:
            print(f"    - Above upper band: {vwap_stats.get('vwap_above_upper', 0)}")
            print(f"    - Below lower band: {vwap_stats.get('vwap_below_lower', 0)}")
        print(f"  Warmup (first {VWAP_MIN_BARS_CONFIG} bars): {vwap_stats.get('vwap_warmup', 0)}")

        # VWAP performance breakdown
        if trades:
            # Analyze VWAP distance correlation with win rate
            vwap_trades = [t for t in trades if t.vwap_value > 0]
            if vwap_trades:
                vwap_wins = [t for t in vwap_trades if t.pnl > 0]
                vwap_wr = len(vwap_wins) / len(vwap_trades) * 100 if vwap_trades else 0
                vwap_pnl = sum(t.pnl for t in vwap_trades)
                avg_vwap_distance = sum(abs(t.vwap_distance_pips) for t in vwap_trades) / len(vwap_trades)

                print(f"\n[VWAP-ALIGNED TRADE PERFORMANCE]")
                print(f"{'-'*50}")
                print(f"  Trades with VWAP: {len(vwap_trades)}")
                print(f"  Win Rate: {vwap_wr:.1f}%")
                print(f"  Net P/L: ${vwap_pnl:+,.0f}")
                print(f"  Avg VWAP Distance: {avg_vwap_distance:.1f} pips")

                # By direction
                buy_trades = [t for t in vwap_trades if t.direction == 'BUY']
                sell_trades = [t for t in vwap_trades if t.direction == 'SELL']

                if buy_trades:
                    buy_wins = len([t for t in buy_trades if t.pnl > 0])
                    buy_wr = buy_wins / len(buy_trades) * 100
                    buy_pnl = sum(t.pnl for t in buy_trades)
                    avg_buy_dist = sum(t.vwap_distance_pips for t in buy_trades) / len(buy_trades)
                    print(f"  BUY (above VWAP): {len(buy_trades)} trades, {buy_wr:.1f}% WR, ${buy_pnl:+,.0f}, avg +{avg_buy_dist:.1f} pips")

                if sell_trades:
                    sell_wins = len([t for t in sell_trades if t.pnl > 0])
                    sell_wr = sell_wins / len(sell_trades) * 100
                    sell_pnl = sum(t.pnl for t in sell_trades)
                    avg_sell_dist = sum(t.vwap_distance_pips for t in sell_trades) / len(sell_trades)
                    print(f"  SELL (below VWAP): {len(sell_trades)} trades, {sell_wr:.1f}% WR, ${sell_pnl:+,.0f}, avg {avg_sell_dist:.1f} pips")

    # Protection stats
    if skip_stats:
        print(f"\n[PROTECTION STATS]")
        print(f"{'-'*50}")
        print(f"  VWAP blocked: {skip_stats.get('VWAP_BLOCKED', 0)}")
        print(f"  Month circuit breaker: {skip_stats.get('MONTH_STOPPED', 0)}")
        print(f"  Day stop: {skip_stats.get('DAY_STOPPED', 0)}")
        print(f"  Dynamic quality adj: {skip_stats.get('DYNAMIC_ADJ', 0)}")
        print(f"  Pattern filter stops: {skip_stats.get('PATTERN_STOPPED', 0)}")
        # Session filter stats
        session_skips = sum(v for k, v in skip_stats.items() if '_SKIP' in k)
        if session_skips > 0:
            print(f"  Session filter skips: {session_skips}")

    # Entry stats
    if entry_stats:
        print(f"\n[ENTRY SIGNALS USED]")
        print(f"{'-'*50}")
        for sig_type, count in sorted(entry_stats.items(), key=lambda x: -x[1]):
            sig_trades = [t for t in trades if t.poi_type == sig_type]
            sig_wins = len([t for t in sig_trades if t.pnl > 0])
            sig_wr = (sig_wins / len(sig_trades) * 100) if sig_trades else 0
            sig_pnl = sum(t.pnl for t in sig_trades)
            print(f"  {sig_type}: {count} trades, {sig_wr:.1f}% WR, ${sig_pnl:+,.0f}")

    # Market conditions
    print(f"\n[MARKET CONDITIONS OBSERVED]")
    print(f"{'-'*50}")
    for cond, count in sorted(condition_stats.items(), key=lambda x: -x[1]):
        print(f"  {cond}: {count} bars")

    # Performance summary
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

    # Comparison with baseline
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH ACTUAL BASELINE (same period, no VWAP)")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'No VWAP':>15} {'With VWAP':>15} {'Change':>15}")
    print(f"{'-'*70}")

    # Actual baseline for period 2024-02-01 to 2026-01-31 (without VWAP)
    baseline = {
        'trades': 217,
        'win_rate': 44.2,
        'profit_factor': 4.27,
        'net_pnl': 39012,
        'losing_months': 2
    }

    trades_diff = stats['total_trades'] - baseline['trades']
    wr_diff = stats['win_rate'] - baseline['win_rate']
    pf_diff = stats['profit_factor'] - baseline['profit_factor']
    pnl_diff = stats['net_pnl'] - baseline['net_pnl']
    lm_diff = stats['losing_months'] - baseline['losing_months']

    print(f"{'Trades':<20} {baseline['trades']:>15} {stats['total_trades']:>15} {trades_diff:>+15}")
    print(f"{'Win Rate':<20} {baseline['win_rate']:>14.1f}% {stats['win_rate']:>14.1f}% {wr_diff:>+14.1f}%")
    print(f"{'Profit Factor':<20} {baseline['profit_factor']:>15.2f} {stats['profit_factor']:>15.2f} {pf_diff:>+15.2f}")
    print(f"{'Net P/L':<20} ${baseline['net_pnl']:>14,} ${stats['net_pnl']:>14,.0f} ${pnl_diff:>+14,.0f}")
    print(f"{'Losing Months':<20} {baseline['losing_months']:>15} {stats['losing_months']:>15} {lm_diff:>+15}")

    # Trades filtered by VWAP
    vwap_blocked = skip_stats.get('VWAP_BLOCKED', 0)
    print(f"\nTrades filtered by VWAP: {vwap_blocked}")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    score = 0
    if stats['net_pnl'] >= baseline['net_pnl']:
        print(f"[OK] Net P/L improved or maintained: ${stats['net_pnl']:+,.0f} vs ${baseline['net_pnl']:+,}")
        score += 1
    else:
        print(f"[X] Net P/L decreased: ${stats['net_pnl']:+,.0f} vs ${baseline['net_pnl']:+,}")

    if stats['profit_factor'] >= baseline['profit_factor']:
        print(f"[OK] PF improved or maintained: {stats['profit_factor']:.2f} vs {baseline['profit_factor']:.2f}")
        score += 1
    else:
        print(f"[X] PF decreased: {stats['profit_factor']:.2f} vs {baseline['profit_factor']:.2f}")

    if stats['losing_months'] <= baseline['losing_months']:
        print(f"[OK] Losing months same or better: {stats['losing_months']} vs {baseline['losing_months']}")
        score += 1
    else:
        print(f"[X] Losing months increased: {stats['losing_months']} vs {baseline['losing_months']}")

    if stats['win_rate'] >= baseline['win_rate']:
        print(f"[OK] Win rate improved: {stats['win_rate']:.1f}% vs {baseline['win_rate']:.1f}%")
        score += 1
    else:
        print(f"[X] Win rate decreased: {stats['win_rate']:.1f}% vs {baseline['win_rate']:.1f}%")

    print(f"\nScore: {score}/4")

    # Primary concern: Does it maintain profit while improving quality?
    pnl_retention = stats['net_pnl'] / baseline['net_pnl'] * 100 if baseline['net_pnl'] > 0 else 0
    print(f"\nP/L Retention: {pnl_retention:.1f}% (vs baseline)")

    if score >= 3 and stats['net_pnl'] >= baseline['net_pnl'] * 0.85:
        print(f"\n>>> RECOMMENDATION: CONSIDER - Good quality improvement with acceptable P/L retention")
    elif stats['net_pnl'] < baseline['net_pnl'] * 0.80:
        print(f"\n>>> RECOMMENDATION: REJECT - VWAP filter reduces profit too much ({pnl_retention:.1f}% retention)")
    elif score >= 2:
        print(f"\n>>> RECOMMENDATION: NEUTRAL - Mixed results, needs further analysis")
    else:
        print(f"\n>>> RECOMMENDATION: REJECT - VWAP filter hurts performance")

    print(f"{'='*70}")


async def main():
    timeframe = "H1"
    # Use same period as original backtest
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    print(f"SURGE-WSI VWAP FILTER TEST - GBPUSD H1")
    print(f"{'='*70}")
    print(f"Testing VWAP as directional filter")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Baseline (v6.8.0): 115 trades, 49.6% WR, PF 5.09, $22,346 profit")
    print(f"{'='*70}")

    print(f"\nFetching {SYMBOL} {timeframe} data...")
    df = await fetch_data(SYMBOL, timeframe, start, end)

    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")

    # Check for volume column
    has_volume = 'Volume' in df.columns or 'volume' in df.columns
    if has_volume:
        vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
        valid_volume = df[vol_col].notna().sum()
        zero_volume = (df[vol_col] == 0).sum()
        print(f"Volume data available: {valid_volume} bars, {zero_volume} zero volume bars")
    else:
        print("WARNING: No volume column found, using equal weights for VWAP")

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
        elif 'volume' in col_lower:
            col_map['volume'] = col

    print(f"\nRunning backtest with VWAP filter...")
    trades, max_dd, condition_stats, skip_stats, entry_stats, vwap_stats = run_backtest(df, col_map)

    if not trades:
        print("No trades executed")
        return

    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, condition_stats, skip_stats, entry_stats, vwap_stats)

    # Save trades
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'lot_size': t.lot_size,
        'atr_pips': t.atr_pips,
        'pnl': t.pnl,
        'exit_reason': t.exit_reason,
        'quality_score': t.quality_score,
        'entry_type': t.entry_type,
        'poi_type': t.poi_type,
        'session': t.session,
        'dynamic_quality': t.dynamic_quality,
        'market_condition': t.market_condition,
        'monthly_adj': t.monthly_adj,
        'vwap_value': t.vwap_value,
        'vwap_distance_pips': t.vwap_distance_pips,
        'vwap_signal': t.vwap_signal
    } for t in trades])

    output_file = STRATEGY_DIR / "backtest_vwap_trades.csv"
    trades_df.to_csv(output_file, index=False)
    print(f"\nTrades saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
