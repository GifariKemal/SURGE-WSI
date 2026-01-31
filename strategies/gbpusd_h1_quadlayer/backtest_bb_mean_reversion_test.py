"""
SURGE-WSI H1 GBPUSD - BOLLINGER BAND MEAN REVERSION FILTER TEST
================================================================

Testing Bollinger Band Mean Reversion as a filter for entry signals.

Baseline (v6.8.0): 115 trades, 49.6% WR, PF 5.09, $22,346 profit, 0 losing months

Bollinger Band Mean Reversion Filter Logic:
- BUY only when price touched/crossed LOWER band recently (oversold -> expect reversion up)
- SELL only when price touched/crossed UPPER band recently (overbought -> expect reversion down)
- Uses BB with period 20, std dev 2.0
- Lookback configurable: tests 3, 5, 10 bars

Expected outcome: Filter trades to only take mean reversion setups,
potentially improving win rate by entering at extremes.

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
# BOLLINGER BAND MEAN REVERSION FILTER CONFIGURATION
# ============================================================
USE_BB_FILTER = True            # Main switch for BB filter
BB_PERIOD = 20                  # Bollinger Band period
BB_STD_DEV = 2.0                # Standard deviations for bands
BB_LOOKBACK = 5                 # How many bars to look back for band touch (test: 3, 5, 10)

# %B indicator thresholds (optional - position within bands)
USE_PERCENT_B = False           # Use %B indicator for additional filtering
PERCENT_B_BUY_THRESHOLD = 0.2   # Buy when %B < 0.2 (near lower band)
PERCENT_B_SELL_THRESHOLD = 0.8  # Sell when %B > 0.8 (near upper band)

# Telegram notifications
SEND_TO_TELEGRAM = False


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


# ============================================================
# BOLLINGER BAND FUNCTIONS
# ============================================================
def calculate_bollinger_bands(df: pd.DataFrame, col_map: dict, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    close = df[col_map['close']]

    # Middle band = SMA of close
    middle = close.rolling(window=period).mean()

    # Standard deviation
    std = close.rolling(window=period).std()

    # Upper and lower bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return middle, upper, lower


def calculate_percent_b(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """
    Calculate %B indicator (position within Bollinger Bands).

    %B = (Close - Lower Band) / (Upper Band - Lower Band)

    Values:
    - %B > 1: Price above upper band
    - %B = 1: Price at upper band
    - %B = 0.5: Price at middle band
    - %B = 0: Price at lower band
    - %B < 0: Price below lower band
    """
    band_width = upper - lower
    # Avoid division by zero
    band_width = band_width.replace(0, np.nan)
    percent_b = (close - lower) / band_width
    return percent_b


def check_bb_mean_reversion(df: pd.DataFrame, col_map: dict, direction: str,
                            lookback: int = 5, period: int = 20, std_dev: float = 2.0) -> Tuple[bool, dict]:
    """
    Check if price touched/crossed Bollinger Band recently for mean reversion setup.

    For BUY: Check if price touched LOWER band in last N bars (oversold -> reversion up)
    For SELL: Check if price touched UPPER band in last N bars (overbought -> reversion down)

    Returns:
        Tuple of (is_valid, info_dict)
    """
    if len(df) < period + lookback:
        return False, {}

    close = df[col_map['close']]
    high = df[col_map['high']]
    low = df[col_map['low']]

    # Calculate Bollinger Bands
    middle, upper, lower = calculate_bollinger_bands(df, col_map, period, std_dev)

    # Get last N bars
    recent_lows = low.iloc[-lookback:]
    recent_highs = high.iloc[-lookback:]
    recent_lower_band = lower.iloc[-lookback:]
    recent_upper_band = upper.iloc[-lookback:]
    recent_close = close.iloc[-lookback:]

    # Calculate %B for current bar
    current_percent_b = calculate_percent_b(close, upper, lower).iloc[-1]
    current_bb_middle = middle.iloc[-1]
    current_bb_upper = upper.iloc[-1]
    current_bb_lower = lower.iloc[-1]

    info = {
        'bb_middle': current_bb_middle,
        'bb_upper': current_bb_upper,
        'bb_lower': current_bb_lower,
        'percent_b': current_percent_b,
        'touched_band': False,
        'bars_since_touch': 0,
    }

    if direction == 'BUY':
        # Check if price touched/crossed lower band recently
        touched = (recent_lows <= recent_lower_band).any()
        if touched:
            # Find how many bars ago
            touch_indices = recent_lows <= recent_lower_band
            bars_since = lookback - touch_indices[::-1].argmax() - 1 if touch_indices.any() else lookback
            info['touched_band'] = True
            info['bars_since_touch'] = bars_since
            return True, info
        return False, info

    else:  # SELL
        # Check if price touched/crossed upper band recently
        touched = (recent_highs >= recent_upper_band).any()
        if touched:
            # Find how many bars ago
            touch_indices = recent_highs >= recent_upper_band
            bars_since = lookback - touch_indices[::-1].argmax() - 1 if touch_indices.any() else lookback
            info['touched_band'] = True
            info['bars_since_touch'] = bars_since
            return True, info
        return False, info


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
    # BB Mean Reversion fields
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    percent_b: float = 0.0
    bars_since_touch: int = 0


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
    """DUAL-LAYER market condition assessment"""
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


def run_backtest(df: pd.DataFrame, col_map: dict, bb_lookback: int = 5) -> Tuple[List[Trade], float, dict, dict]:
    """Run backtest with Bollinger Band Mean Reversion filter"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Layer 3: Intra-month risk manager
    risk_manager = IntraMonthRiskManager()

    # Layer 4: Pattern-based choppy market filter
    pattern_filter = PatternBasedFilter()
    current_month_key = None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {
        'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0,
        'BB_NO_TOUCH': 0,  # BB filter blocked
    }
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0}
    bb_stats = {'buy_touches': 0, 'sell_touches': 0, 'total_blocked': 0}

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

            # ============================================================
            # BOLLINGER BAND MEAN REVERSION FILTER
            # ============================================================
            if USE_BB_FILTER:
                bb_valid, bb_info = check_bb_mean_reversion(
                    current_slice, col_map, poi['direction'],
                    lookback=bb_lookback, period=BB_PERIOD, std_dev=BB_STD_DEV
                )

                if not bb_valid:
                    skip_stats['BB_NO_TOUCH'] += 1
                    bb_stats['total_blocked'] += 1
                    continue

                # Track BB touches
                if poi['direction'] == 'BUY':
                    bb_stats['buy_touches'] += 1
                else:
                    bb_stats['sell_touches'] += 1

                # Optional: %B check
                if USE_PERCENT_B:
                    percent_b = bb_info.get('percent_b', 0.5)
                    if poi['direction'] == 'BUY' and percent_b > PERCENT_B_BUY_THRESHOLD:
                        skip_stats['BB_NO_TOUCH'] += 1
                        continue
                    if poi['direction'] == 'SELL' and percent_b < PERCENT_B_SELL_THRESHOLD:
                        skip_stats['BB_NO_TOUCH'] += 1
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

            total_extra_q = pattern_extra_q
            if total_extra_q > 0:
                effective_quality = dynamic_quality + total_extra_q
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

            # Get BB info for trade record
            _, bb_info = check_bb_mean_reversion(
                current_slice, col_map, poi['direction'],
                lookback=bb_lookback, period=BB_PERIOD, std_dev=BB_STD_DEV
            )

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
                bb_upper=bb_info.get('bb_upper', 0),
                bb_lower=bb_info.get('bb_lower', 0),
                bb_middle=bb_info.get('bb_middle', 0),
                percent_b=bb_info.get('percent_b', 0),
                bars_since_touch=bb_info.get('bars_since_touch', 0)
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats, bb_stats


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
                  skip_stats: dict, entry_stats: dict, bb_stats: dict,
                  bb_lookback: int, baseline_stats: dict = None):
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS - BB MEAN REVERSION FILTER (Lookback={bb_lookback})")
    print(f"{'='*70}")

    print(f"\n[BB MEAN REVERSION FILTER CONFIG]")
    print(f"{'-'*50}")
    print(f"  BB Period: {BB_PERIOD}")
    print(f"  BB Std Dev: {BB_STD_DEV}")
    print(f"  Lookback for band touch: {bb_lookback} bars")
    print(f"  Use %B: {USE_PERCENT_B}")
    if USE_PERCENT_B:
        print(f"    Buy threshold: %B < {PERCENT_B_BUY_THRESHOLD}")
        print(f"    Sell threshold: %B > {PERCENT_B_SELL_THRESHOLD}")

    print(f"\n[BB FILTER STATS]")
    print(f"{'-'*50}")
    print(f"  Buy signals (lower band touch): {bb_stats.get('buy_touches', 0)}")
    print(f"  Sell signals (upper band touch): {bb_stats.get('sell_touches', 0)}")
    print(f"  Total blocked (no band touch): {bb_stats.get('total_blocked', 0)}")

    if skip_stats:
        print(f"\n[PROTECTION STATS]")
        print(f"{'-'*50}")
        print(f"  BB no band touch: {skip_stats.get('BB_NO_TOUCH', 0)}")
        print(f"  Month circuit breaker: {skip_stats.get('MONTH_STOPPED', 0)}")
        print(f"  Day stop: {skip_stats.get('DAY_STOPPED', 0)}")
        print(f"  Pattern filter stops: {skip_stats.get('PATTERN_STOPPED', 0)}")

    if entry_stats:
        print(f"\n[ENTRY SIGNALS USED]")
        print(f"{'-'*50}")
        for sig_type, count in sorted(entry_stats.items(), key=lambda x: -x[1]):
            sig_trades = [t for t in trades if t.poi_type == sig_type]
            sig_wins = len([t for t in sig_trades if t.pnl > 0])
            sig_wr = (sig_wins / len(sig_trades) * 100) if sig_trades else 0
            sig_pnl = sum(t.pnl for t in sig_trades)
            print(f"  {sig_type}: {count} trades, {sig_wr:.1f}% WR, ${sig_pnl:+,.0f}")

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
        print(f"  [{status}] {month}: ${pnl:+,.2f}")

    # Comparison with baseline
    if baseline_stats:
        print(f"\n{'='*70}")
        print(f"COMPARISON VS BASELINE (v6.8.0)")
        print(f"{'='*70}")
        print(f"{'Metric':<20} {'Baseline':>15} {'BB Filter':>15} {'Change':>15}")
        print(f"{'-'*65}")

        # Trades
        trade_diff = stats['total_trades'] - baseline_stats['trades']
        trade_pct = (trade_diff / baseline_stats['trades']) * 100 if baseline_stats['trades'] > 0 else 0
        print(f"{'Trades':<20} {baseline_stats['trades']:>15} {stats['total_trades']:>15} {trade_diff:>+10} ({trade_pct:>+.1f}%)")

        # Win Rate
        wr_diff = stats['win_rate'] - baseline_stats['win_rate']
        print(f"{'Win Rate':<20} {baseline_stats['win_rate']:>14.1f}% {stats['win_rate']:>14.1f}% {wr_diff:>+14.1f}%")

        # Profit Factor
        pf_diff = stats['profit_factor'] - baseline_stats['pf']
        print(f"{'Profit Factor':<20} {baseline_stats['pf']:>15.2f} {stats['profit_factor']:>15.2f} {pf_diff:>+15.2f}")

        # Net P/L
        pnl_diff = stats['net_pnl'] - baseline_stats['profit']
        pnl_pct = (pnl_diff / baseline_stats['profit']) * 100 if baseline_stats['profit'] > 0 else 0
        print(f"{'Net P/L':<20} ${baseline_stats['profit']:>13,.0f} ${stats['net_pnl']:>13,.0f} ${pnl_diff:>+12,.0f} ({pnl_pct:>+.1f}%)")

        # Losing Months
        lm_diff = stats['losing_months'] - baseline_stats['losing_months']
        print(f"{'Losing Months':<20} {baseline_stats['losing_months']:>15} {stats['losing_months']:>15} {lm_diff:>+15}")

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


async def main():
    """Main entry point - test different BB lookback periods"""
    timeframe = "H1"
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 30, tzinfo=timezone.utc)

    # Baseline v6.8.0 stats for comparison
    baseline_stats = {
        'trades': 115,
        'win_rate': 49.6,
        'pf': 5.09,
        'profit': 22346,
        'losing_months': 0
    }

    print(f"{'='*70}")
    print(f"SURGE-WSI H1 GBPUSD - BOLLINGER BAND MEAN REVERSION FILTER TEST")
    print(f"{'='*70}")
    print(f"Baseline v6.8.0: {baseline_stats['trades']} trades, {baseline_stats['win_rate']:.1f}% WR, ")
    print(f"                 PF {baseline_stats['pf']:.2f}, ${baseline_stats['profit']:,} profit, {baseline_stats['losing_months']} losing months")
    print(f"{'='*70}")
    print(f"\nBB Mean Reversion Logic:")
    print(f"  - BUY: Price must have touched LOWER band recently (oversold)")
    print(f"  - SELL: Price must have touched UPPER band recently (overbought)")
    print(f"  - Testing lookback periods: 3, 5, 10 bars")
    print(f"{'='*70}")

    print(f"\nFetching {SYMBOL} {timeframe} data...")

    df = await fetch_data(SYMBOL, timeframe, start, end)

    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

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

    # Test different lookback periods
    lookback_periods = [3, 5, 10]
    all_results = []

    for lookback in lookback_periods:
        print(f"\n{'#'*70}")
        print(f"# Testing BB Lookback = {lookback} bars")
        print(f"{'#'*70}")

        trades, max_dd, condition_stats, skip_stats, entry_stats, bb_stats = run_backtest(df, col_map, bb_lookback=lookback)

        if not trades:
            print(f"No trades executed with lookback={lookback}")
            all_results.append({
                'lookback': lookback,
                'trades': 0,
                'win_rate': 0,
                'pf': 0,
                'profit': 0,
                'losing_months': 0,
                'blocked': skip_stats.get('BB_NO_TOUCH', 0)
            })
            continue

        stats = calculate_stats(trades, max_dd)
        print_results(stats, trades, condition_stats, skip_stats, entry_stats, bb_stats, lookback, baseline_stats)

        all_results.append({
            'lookback': lookback,
            'trades': stats['total_trades'],
            'win_rate': stats['win_rate'],
            'pf': stats['profit_factor'],
            'profit': stats['net_pnl'],
            'losing_months': stats['losing_months'],
            'blocked': skip_stats.get('BB_NO_TOUCH', 0)
        })

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"SUMMARY: BB MEAN REVERSION FILTER TEST RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Trades':>8} {'Blocked':>8} {'WR%':>8} {'PF':>8} {'Profit':>12} {'LM':>4}")
    print(f"{'-'*65}")
    print(f"{'Baseline v6.8':<15} {baseline_stats['trades']:>8} {'-':>8} {baseline_stats['win_rate']:>7.1f}% {baseline_stats['pf']:>8.2f} ${baseline_stats['profit']:>10,} {baseline_stats['losing_months']:>4}")

    for result in all_results:
        print(f"{'BB LB=' + str(result['lookback']):<15} {result['trades']:>8} {result['blocked']:>8} {result['win_rate']:>7.1f}% {result['pf']:>8.2f} ${result['profit']:>10,.0f} {result['losing_months']:>4}")

    # Recommendation
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")

    best_result = None
    for result in all_results:
        if result['trades'] > 0:
            # Score: prioritize zero losing months, then PF, then profit
            score = 0
            if result['losing_months'] == 0:
                score += 1000
            score += result['pf'] * 100
            score += result['profit'] / 100

            if best_result is None or score > best_result.get('score', 0):
                result['score'] = score
                best_result = result

    if best_result and best_result['profit'] >= baseline_stats['profit'] * 0.8 and best_result['losing_months'] <= baseline_stats['losing_months']:
        print(f"\n[KEEP] BB Mean Reversion Filter (Lookback={best_result['lookback']}) shows promise:")
        print(f"  - Trades: {best_result['trades']} (filtered {best_result['blocked']} signals)")
        print(f"  - Win Rate: {best_result['win_rate']:.1f}%")
        print(f"  - Profit Factor: {best_result['pf']:.2f}")
        print(f"  - Net P/L: ${best_result['profit']:,.0f}")
        print(f"  - Losing Months: {best_result['losing_months']}")
    else:
        print(f"\n[REJECT] BB Mean Reversion Filter does not improve results:")
        if all_results:
            best = max(all_results, key=lambda x: x['profit'] if x['trades'] > 0 else -999999)
            print(f"  - Best result (LB={best['lookback']}): ${best['profit']:,.0f} profit, {best['losing_months']} losing months")
            print(f"  - Baseline: ${baseline_stats['profit']:,} profit, {baseline_stats['losing_months']} losing months")
            if best['profit'] < baseline_stats['profit']:
                print(f"  - Reason: Reduced profit by ${baseline_stats['profit'] - best['profit']:,.0f}")
            if best['losing_months'] > baseline_stats['losing_months']:
                print(f"  - Reason: Added {best['losing_months'] - baseline_stats['losing_months']} losing month(s)")


if __name__ == "__main__":
    asyncio.run(main())
