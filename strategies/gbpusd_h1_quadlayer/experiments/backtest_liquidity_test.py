"""
SURGE-WSI H1 v6.8.0 GBPUSD - LIQUIDITY SWEEP FILTER TEST
=========================================================

Testing smartmoneyconcepts Liquidity Sweep detection as a trade filter.

Concept:
- Detect liquidity zones (clusters of swing highs/lows where stop losses sit)
- Only take BUY trades AFTER price swept below a swing low and reversed
- Only take SELL trades AFTER price swept above a swing high and reversed
- This filters for "smart money" entries after retail stops are hunted

Baseline: v6.8.0 with 115 trades, 49.6% WR, PF 5.09, $22,346 profit

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

# Import smartmoneyconcepts for liquidity detection
try:
    from smartmoneyconcepts.smc import smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    print("[WARNING] smartmoneyconcepts not available - install with: pip install smartmoneyconcepts")

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
# LIQUIDITY SWEEP FILTER CONFIG
# ============================================================
USE_LIQUIDITY_FILTER = True       # Main toggle for this test
LIQUIDITY_SWING_LENGTH = 5        # Bars to detect swing highs/lows (reduced for H1)
LIQUIDITY_RANGE_PERCENT = 0.005   # 0.5% range for liquidity zone clustering (more sensitive)
LIQUIDITY_LOOKBACK = 100          # Look back this many bars for liquidity sweeps (extended)
LIQUIDITY_RECENCY = 50            # Sweep must have occurred within this many bars (more lenient)

# Debug mode
DEBUG_LIQUIDITY = True


# ============================================================
# CONFIGURATION v6.8.0 BASELINE
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
PATTERN_FILTER_ENABLED = True

# Entry signal toggles (same as v6.8.0)
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True

# Session filter (same as v6.8.0)
USE_SESSION_POI_FILTER = True
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]

# Partial TP disabled (same as v6.8.0)
USE_PARTIAL_TP = False

# ==========================================================
# LAYER 2: REAL-TIME TECHNICAL THRESHOLDS
# ==========================================================
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

# Risk Multipliers (same as v6.8.0)
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


def should_skip_by_session(hour: int, poi_type: str) -> tuple:
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
    # Liquidity filter info
    liquidity_swept: bool = False
    sweep_direction: str = ""
    sweep_level: float = 0.0
    bars_since_sweep: int = 0


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
class LiquiditySweep:
    """Represents a detected liquidity sweep event"""
    sweep_type: str           # 'bullish' (swept lows) or 'bearish' (swept highs)
    sweep_index: int          # Bar index where sweep occurred
    level: float              # The liquidity level that was swept
    current_bar: int          # Current bar index (for recency calculation)

    @property
    def bars_ago(self) -> int:
        return self.current_bar - self.sweep_index


class LiquidityFilter:
    """
    Liquidity Sweep Filter - SIMPLIFIED VERSION.

    Instead of using the clustered liquidity detection (which is too strict),
    this version detects simple swing point sweeps:

    - Detect recent swing highs/lows
    - Check if price swept (broke) past the swing point
    - Check if price then reversed (closed back inside)

    Trading logic:
    - BUY: Price swept below a swing low and reversed up (stop hunt complete)
    - SELL: Price swept above a swing high and reversed down (stop hunt complete)
    """

    def __init__(self, swing_length: int = 5, range_percent: float = 0.005,
                 lookback: int = 100, recency: int = 50):
        self.swing_length = swing_length
        self.range_percent = range_percent  # Not used in simplified version
        self.lookback = lookback
        self.recency = recency

        # Stats
        self.stats = {
            'total_sweeps_detected': 0,
            'bullish_sweeps': 0,
            'bearish_sweeps': 0,
            'trades_allowed': 0,
            'trades_blocked': 0,
        }

    def _find_swing_points(self, df: pd.DataFrame, col_map: dict) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Find swing highs and lows in the dataframe.

        Returns: (swing_highs, swing_lows) as lists of (index, price) tuples
        """
        h, l = col_map['high'], col_map['low']
        swing_highs = []
        swing_lows = []

        lookback = self.swing_length

        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df[h].iloc[j] >= df[h].iloc[i]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append((i, df[h].iloc[i]))

            # Check for swing low
            is_swing_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df[l].iloc[j] <= df[l].iloc[i]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append((i, df[l].iloc[i]))

        return swing_highs, swing_lows

    def update(self, df: pd.DataFrame, col_map: dict) -> Tuple[Optional[LiquiditySweep], Optional[LiquiditySweep]]:
        """
        Update liquidity detection with new data.

        Returns: (bullish_sweep, bearish_sweep) - most recent sweeps if any
        """
        if len(df) < self.lookback + self.swing_length * 2:
            return None, None

        h, l, c = col_map['high'], col_map['low'], col_map['close']
        current_idx = len(df) - 1
        current_low = df[l].iloc[current_idx]
        current_high = df[h].iloc[current_idx]
        current_close = df[c].iloc[current_idx]

        # Find swing points in recent history
        lookback_start = max(0, current_idx - self.lookback)
        recent_df = df.iloc[lookback_start:current_idx]

        swing_highs, swing_lows = self._find_swing_points(df.iloc[:current_idx], col_map)

        bullish_sweep = None
        bearish_sweep = None

        # Check for BULLISH sweep (price went below swing low, then reversed)
        # This means lows were swept = bullish signal for BUY
        for swing_idx, swing_price in reversed(swing_lows):
            if swing_idx < lookback_start:
                continue

            bars_since_swing = current_idx - swing_idx
            if bars_since_swing > self.recency:
                continue

            # Check if price swept below this swing low at some point after the swing
            swept = False
            swept_idx = None
            for i in range(swing_idx + 1, current_idx + 1):
                if df[l].iloc[i] < swing_price:
                    swept = True
                    swept_idx = i
                    break

            if swept and swept_idx:
                # Check if price reversed (current close above the sweep low)
                if current_close > swing_price:
                    bars_since = current_idx - swept_idx
                    if bars_since <= self.recency:
                        sweep = LiquiditySweep(
                            sweep_type='bullish',
                            sweep_index=swept_idx,
                            level=swing_price,
                            current_bar=current_idx
                        )
                        if bullish_sweep is None or sweep.bars_ago < bullish_sweep.bars_ago:
                            bullish_sweep = sweep
                            self.stats['bullish_sweeps'] += 1
                        break  # Take the most recent valid sweep

        # Check for BEARISH sweep (price went above swing high, then reversed)
        # This means highs were swept = bearish signal for SELL
        for swing_idx, swing_price in reversed(swing_highs):
            if swing_idx < lookback_start:
                continue

            bars_since_swing = current_idx - swing_idx
            if bars_since_swing > self.recency:
                continue

            # Check if price swept above this swing high at some point after the swing
            swept = False
            swept_idx = None
            for i in range(swing_idx + 1, current_idx + 1):
                if df[h].iloc[i] > swing_price:
                    swept = True
                    swept_idx = i
                    break

            if swept and swept_idx:
                # Check if price reversed (current close below the sweep high)
                if current_close < swing_price:
                    bars_since = current_idx - swept_idx
                    if bars_since <= self.recency:
                        sweep = LiquiditySweep(
                            sweep_type='bearish',
                            sweep_index=swept_idx,
                            level=swing_price,
                            current_bar=current_idx
                        )
                        if bearish_sweep is None or sweep.bars_ago < bearish_sweep.bars_ago:
                            bearish_sweep = sweep
                            self.stats['bearish_sweeps'] += 1
                        break  # Take the most recent valid sweep

        return bullish_sweep, bearish_sweep

    def check_trade(self, direction: str, bullish_sweep: Optional[LiquiditySweep],
                    bearish_sweep: Optional[LiquiditySweep]) -> Tuple[bool, str, Optional[LiquiditySweep]]:
        """
        Check if a trade is allowed based on liquidity sweep.

        Returns: (can_trade, reason, sweep_used)
        """
        if direction == 'BUY':
            # BUY requires prior swing low sweep (price dipped below, then reversed up)
            if bullish_sweep is not None:
                self.stats['trades_allowed'] += 1
                return True, f"BULLISH_SWEEP_{bullish_sweep.bars_ago}bars", bullish_sweep
            else:
                self.stats['trades_blocked'] += 1
                return False, "NO_BULLISH_SWEEP", None

        elif direction == 'SELL':
            # SELL requires prior swing high sweep (price spiked above, then reversed down)
            if bearish_sweep is not None:
                self.stats['trades_allowed'] += 1
                return True, f"BEARISH_SWEEP_{bearish_sweep.bars_ago}bars", bearish_sweep
            else:
                self.stats['trades_blocked'] += 1
                return False, "NO_BEARISH_SWEEP", None

        return False, "INVALID_DIRECTION", None


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
    """DUAL-LAYER market condition assessment (same as v6.8.0)"""
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

    # Technical quality threshold
    if score >= 80:
        technical_quality = MIN_QUALITY_GOOD
        tech_label = "TECH_GOOD"
    elif score >= 40:
        technical_quality = BASE_QUALITY
        tech_label = "TECH_NORMAL"
    else:
        technical_quality = MAX_QUALITY_BAD
        tech_label = "TECH_BAD"

    # Monthly profile adjustment
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
    """Detect EMA Pullback entry signals (v6.7)"""
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

    # BUY: Uptrend pullback
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

    # SELL: Downtrend pullback
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


def run_backtest(df: pd.DataFrame, col_map: dict) -> Tuple[List[Trade], float, dict, dict]:
    """Run backtest with LIQUIDITY SWEEP FILTER"""
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

    # LIQUIDITY FILTER
    liquidity_filter = LiquidityFilter(
        swing_length=LIQUIDITY_SWING_LENGTH,
        range_percent=LIQUIDITY_RANGE_PERCENT,
        lookback=LIQUIDITY_LOOKBACK,
        recency=LIQUIDITY_RECENCY
    ) if USE_LIQUIDITY_FILTER and SMC_AVAILABLE else None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0, 'LIQUIDITY_BLOCKED': 0}
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0}
    liquidity_stats = {'bullish_sweeps': 0, 'bearish_sweeps': 0, 'allowed': 0, 'blocked': 0}

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

        # Handle existing position
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
            pattern_filter.reset_for_month(current_time.month)

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # UPDATE LIQUIDITY FILTER
        bullish_sweep = None
        bearish_sweep = None
        if liquidity_filter:
            bullish_sweep, bearish_sweep = liquidity_filter.update(current_slice, col_map)
            if bullish_sweep:
                liquidity_stats['bullish_sweeps'] += 1
            if bearish_sweep:
                liquidity_stats['bearish_sweeps'] += 1

        # Market condition assessment
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
            # LIQUIDITY SWEEP FILTER (NEW!)
            # ============================================================
            sweep_info = None
            if USE_LIQUIDITY_FILTER and liquidity_filter:
                can_trade_liq, liq_reason, sweep_info = liquidity_filter.check_trade(
                    poi['direction'], bullish_sweep, bearish_sweep
                )
                if not can_trade_liq:
                    skip_stats['LIQUIDITY_BLOCKED'] += 1
                    liquidity_stats['blocked'] += 1
                    if DEBUG_LIQUIDITY:
                        print(f"[LIQUIDITY] {current_time} {poi['direction']} blocked: {liq_reason}")
                    continue
                else:
                    liquidity_stats['allowed'] += 1
                    if DEBUG_LIQUIDITY:
                        print(f"[LIQUIDITY] {current_time} {poi['direction']} allowed: {liq_reason}")

            # EMA_PULLBACK already has entry confirmation
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
            if PATTERN_FILTER_ENABLED:
                pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(poi['direction'])
                if not pattern_can_trade:
                    skip_stats['PATTERN_STOPPED'] += 1
                    continue

            # Extra quality check
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
                liquidity_swept=sweep_info is not None,
                sweep_direction=sweep_info.sweep_type if sweep_info else "",
                sweep_level=sweep_info.level if sweep_info else 0.0,
                bars_since_sweep=sweep_info.bars_ago if sweep_info else 0,
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats, liquidity_stats


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
                  skip_stats: dict, entry_stats: dict, liquidity_stats: dict):
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS - H1 v6.8.0 + LIQUIDITY SWEEP FILTER TEST")
    print(f"{'='*70}")

    print(f"\n[LIQUIDITY SWEEP FILTER CONFIG]")
    print(f"{'-'*50}")
    print(f"  Filter Status: {'ENABLED' if USE_LIQUIDITY_FILTER else 'DISABLED'}")
    print(f"  SMC Library: {'AVAILABLE' if SMC_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Swing Length: {LIQUIDITY_SWING_LENGTH} (bars each side)")
    print(f"  Range Percent: {LIQUIDITY_RANGE_PERCENT*100:.1f}%")
    print(f"  Lookback: {LIQUIDITY_LOOKBACK} bars")
    print(f"  Recency: {LIQUIDITY_RECENCY} bars (max bars since sweep)")
    print(f"\n  Logic:")
    print(f"    - BUY: Only after bearish liquidity (lows) swept + reversal")
    print(f"    - SELL: Only after bullish liquidity (highs) swept + reversal")

    print(f"\n[LIQUIDITY FILTER STATS]")
    print(f"{'-'*50}")
    print(f"  Bullish sweeps detected: {liquidity_stats['bullish_sweeps']}")
    print(f"  Bearish sweeps detected: {liquidity_stats['bearish_sweeps']}")
    print(f"  Trades allowed: {liquidity_stats['allowed']}")
    print(f"  Trades blocked: {liquidity_stats['blocked']}")

    if skip_stats:
        print(f"\n[PROTECTION STATS]")
        print(f"{'-'*50}")
        print(f"  Month circuit breaker: {skip_stats.get('MONTH_STOPPED', 0)}")
        print(f"  Day stop: {skip_stats.get('DAY_STOPPED', 0)}")
        print(f"  Dynamic quality adj: {skip_stats.get('DYNAMIC_ADJ', 0)}")
        print(f"  Pattern filter stops: {skip_stats.get('PATTERN_STOPPED', 0)}")
        print(f"  LIQUIDITY BLOCKED: {skip_stats.get('LIQUIDITY_BLOCKED', 0)}")
        session_skips = sum(v for k, v in skip_stats.items() if '_SKIP' in k)
        if session_skips > 0:
            print(f"  Session filter skips: {session_skips}")

    if entry_stats:
        print(f"\n[ENTRY SIGNALS USED]")
        print(f"{'-'*50}")
        for sig_type, count in sorted(entry_stats.items(), key=lambda x: -x[1]):
            sig_trades = [t for t in trades if t.poi_type == sig_type]
            sig_wins = len([t for t in sig_trades if t.pnl > 0])
            sig_wr = (sig_wins / len(sig_trades) * 100) if sig_trades else 0
            sig_pnl = sum(t.pnl for t in sig_trades)
            print(f"  {sig_type}: {count} trades, {sig_wr:.1f}% WR, ${sig_pnl:+,.0f}")

    # Liquidity sweep analysis
    if USE_LIQUIDITY_FILTER:
        print(f"\n[LIQUIDITY SWEEP TRADE ANALYSIS]")
        print(f"{'-'*50}")
        swept_trades = [t for t in trades if t.liquidity_swept]
        if swept_trades:
            by_sweep_type = {}
            for t in swept_trades:
                st = t.sweep_direction
                if st not in by_sweep_type:
                    by_sweep_type[st] = []
                by_sweep_type[st].append(t)

            for sweep_type, st_trades in by_sweep_type.items():
                wins = len([t for t in st_trades if t.pnl > 0])
                wr = (wins / len(st_trades) * 100) if st_trades else 0
                pnl = sum(t.pnl for t in st_trades)
                avg_bars = sum(t.bars_since_sweep for t in st_trades) / len(st_trades)
                print(f"  {sweep_type.upper()} sweep trades: {len(st_trades)}")
                print(f"    - Win Rate: {wr:.1f}%")
                print(f"    - P/L: ${pnl:+,.0f}")
                print(f"    - Avg bars since sweep: {avg_bars:.1f}")
        else:
            print(f"  No trades with liquidity sweep confirmation")

    print(f"\n[MARKET CONDITIONS OBSERVED]")
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
    print(f"COMPARISON vs v6.8.0 BASELINE")
    print(f"{'='*70}")

    baseline_trades = 115
    baseline_wr = 49.6
    baseline_pf = 5.09
    baseline_pnl = 22346
    baseline_losing = 0

    trade_diff = stats['total_trades'] - baseline_trades
    wr_diff = stats['win_rate'] - baseline_wr
    pf_diff = stats['profit_factor'] - baseline_pf
    pnl_diff = stats['net_pnl'] - baseline_pnl
    losing_diff = stats['losing_months'] - baseline_losing

    print(f"\n  Metric          v6.8.0      With Filter    Change")
    print(f"  {'-'*50}")
    print(f"  Trades:         {baseline_trades:>6}        {stats['total_trades']:>6}        {trade_diff:+d}")
    print(f"  Win Rate:       {baseline_wr:>6.1f}%       {stats['win_rate']:>6.1f}%      {wr_diff:+.1f}%")
    print(f"  Profit Factor:  {baseline_pf:>6.2f}        {stats['profit_factor']:>6.2f}       {pf_diff:+.2f}")
    print(f"  Net P/L:        ${baseline_pnl:>+,}     ${stats['net_pnl']:>+,.0f}    ${pnl_diff:+,.0f}")
    print(f"  Losing Months:  {baseline_losing:>6}        {stats['losing_months']:>6}        {losing_diff:+d}")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")

    # Determine recommendation
    is_better = True
    reasons = []

    if stats['losing_months'] > baseline_losing:
        is_better = False
        reasons.append(f"Added {losing_diff} losing month(s)")

    if stats['net_pnl'] < baseline_pnl * 0.9:  # More than 10% worse
        is_better = False
        reasons.append(f"Profit reduced by ${-pnl_diff:,.0f}")

    if stats['profit_factor'] < baseline_pf * 0.9:  # More than 10% worse
        is_better = False
        reasons.append(f"PF reduced by {-pf_diff:.2f}")

    if stats['total_trades'] < baseline_trades * 0.5:  # More than 50% fewer trades
        is_better = False
        reasons.append(f"Trade count reduced by {-trade_diff} ({-trade_diff/baseline_trades*100:.0f}%)")

    if is_better:
        print(f"\n  [KEEP] Liquidity Sweep Filter IMPROVES results")
        print(f"\n  Benefits:")
        if wr_diff > 0:
            print(f"    + Win rate improved by {wr_diff:.1f}%")
        if pf_diff > 0:
            print(f"    + Profit factor improved by {pf_diff:.2f}")
        if pnl_diff > 0:
            print(f"    + Net profit improved by ${pnl_diff:,.0f}")
        if losing_diff <= 0 and baseline_losing == 0:
            print(f"    + Maintained ZERO losing months")
    else:
        print(f"\n  [REJECT] Liquidity Sweep Filter HURTS results")
        print(f"\n  Issues:")
        for reason in reasons:
            print(f"    - {reason}")

    print(f"\n{'='*70}")


async def main():
    timeframe = "H1"
    # Same period as baseline: 2024-02-01 to 2026-01-30
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 30, tzinfo=timezone.utc)

    print(f"SURGE-WSI H1 v6.8.0 GBPUSD - LIQUIDITY SWEEP FILTER TEST")
    print(f"{'='*70}")
    print(f"Testing smartmoneyconcepts liquidity detection as trade filter")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Baseline: v6.8.0 = 115 trades, 49.6% WR, PF 5.09, $22,346 profit, 0 losing months")
    print(f"{'='*70}")

    if not SMC_AVAILABLE:
        print("\n[ERROR] smartmoneyconcepts not available!")
        print("Install with: pip install smartmoneyconcepts")
        return

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

    print(f"\nRunning backtest with LIQUIDITY SWEEP FILTER...")
    trades, max_dd, condition_stats, skip_stats, entry_stats, liquidity_stats = run_backtest(df, col_map)

    if not trades:
        print("No trades executed")
        return

    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, condition_stats, skip_stats, entry_stats, liquidity_stats)

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
        'liquidity_swept': t.liquidity_swept,
        'sweep_direction': t.sweep_direction,
        'sweep_level': t.sweep_level,
        'bars_since_sweep': t.bars_since_sweep,
    } for t in trades])

    output_path = STRATEGY_DIR / "reports" / "liquidity_filter_test_trades.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
