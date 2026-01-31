"""
SURGE-WSI H1 v6.9.0 GBPUSD - ROUND 4 COMBINED OPTIMIZATIONS TEST
=================================================================

Testing ALL Round 4 optimizations combined:
1. Kelly 25% Position Sizing
2. Fixed 4:1 R:R Exit
3. Fixed Day Multipliers (Thursday = 1.0)
4. PIN_BAR Entry Signal

Base: v6.8.0 (115 trades, 49.6% WR, PF 5.09, $22,346 profit, 0 losing months)

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
# ROUND 4 OPTIMIZATION FLAGS
# ============================================================
USE_KELLY_SIZING = False         # Optimization 1: Kelly 25% Position Sizing - REJECTED (adds variance)
USE_4_TO_1_RR = False            # Optimization 2: Fixed 4:1 R:R Exit - REJECTED (adds losing months)
USE_FIXED_DAY_MULTS = True       # Optimization 3: Fixed Day Multipliers - APPROVED (reduces losing months)
USE_PIN_BAR_ENTRY = False        # Optimization 4: PIN_BAR Entry Signal - REJECTED (adds losing months)


# ============================================================
# CONFIGURATION - ROUND 4 COMBINED
# ============================================================
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = RISK.risk_percent
SL_ATR_MULT = RISK.sl_atr_mult   # 1.5 ATR

# Optimization 2: Fixed 4:1 R:R (REJECTED)
if USE_4_TO_1_RR:
    TP_RATIO = 4.0               # TP = SL * 4 - not used
else:
    TP_RATIO = RISK.tp_ratio     # Default 1.5

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


# ============================================================
# LAYER 2: REAL-TIME TECHNICAL THRESHOLDS
# ============================================================
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

# Risk Multipliers
MONTHLY_RISK = {
    1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,
    7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
}

# Optimization 3: Fixed Day Multipliers (based on actual WR analysis)
if USE_FIXED_DAY_MULTS:
    # FIXED based on actual win rate analysis:
    # Mon: 43.0% WR, Tue: 44.4% WR, Wed: 40.6% WR, Thu: 51.8% WR (BEST!), Fri: 33.8% WR
    DAY_MULTIPLIERS = {
        0: 1.0,   # Monday - 43.0% WR
        1: 1.0,   # Tuesday - 44.4% WR
        2: 1.0,   # Wednesday - 40.6% WR
        3: 1.0,   # Thursday - 51.8% WR (BEST - was wrongly at 0.4!)
        4: 0.5,   # Friday - 33.8% WR (WORST - keep reduced)
        5: 0.0,   # Saturday
        6: 0.0,   # Sunday
    }
else:
    # Original (WRONG) day multipliers
    DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}

HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8, 'PIN_BAR': 1.0}

# Session-based filters
USE_SESSION_POI_FILTER = True
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True
USE_PATTERN_FILTER = True


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
    kelly_pct: float = 0.0  # Store Kelly % used


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
# KELLY 25% POSITION SIZING (Optimization 1)
# ============================================================
class KellyPositionSizer:
    """
    Kelly Criterion Position Sizing with 25% fraction.

    Formula: kelly_pct = (WR * avg_win - (1-WR) * avg_loss) / avg_win
    Use 25% of Kelly for safety, capped at 5% max risk.
    """

    def __init__(self, min_trades: int = 20):
        self.trades: List[Trade] = []
        self.min_trades = min_trades

    def record_trade(self, trade: Trade):
        """Record a completed trade for Kelly calculation."""
        self.trades.append(trade)

    def calculate_kelly(self) -> Tuple[float, float, float, float]:
        """
        Calculate Kelly percentage.

        Returns: (kelly_pct, win_rate, avg_win, avg_loss)
        """
        if len(self.trades) < self.min_trades:
            return 0.01, 0, 0, 0  # Default to 1% if insufficient data

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 1

        if avg_win <= 0:
            return 0.01, win_rate, avg_win, avg_loss

        # Kelly formula
        kelly_full = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Use 25% of Kelly for safety, cap at 5% max
        kelly_25 = max(0.005, min(kelly_full * 0.25, 0.05))  # Min 0.5%, max 5%

        return kelly_25, win_rate, avg_win, avg_loss

    def calculate_lot_size(self, balance: float, sl_pips: float, base_risk_mult: float = 1.0) -> Tuple[float, float]:
        """
        Calculate lot size using Kelly 25%.

        Args:
            balance: Account balance
            sl_pips: Stop loss in pips
            base_risk_mult: Additional risk multiplier from filters

        Returns: (lot_size, kelly_pct_used)
        """
        kelly_pct, _, _, _ = self.calculate_kelly()

        # Apply base risk multiplier
        effective_risk = kelly_pct * base_risk_mult

        # Calculate position risk in dollars
        risk_amount = balance * effective_risk

        # Calculate lot size
        pip_value_per_lot = PIP_VALUE  # $10 per pip per lot for GBPUSD
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

        return lot_size, kelly_pct


# ============================================================
# PIN BAR DETECTION (Optimization 4)
# ============================================================
def detect_pin_bar(df: pd.DataFrame, col_map: dict, i: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect pin bar / rejection candle.

    Bullish pin bar: long lower wick, small body at top
    Bearish pin bar: long upper wick, small body at bottom

    Returns: (entry_type, direction) or (None, None)
    """
    if i >= len(df):
        return None, None

    row = df.iloc[i]
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    body = abs(row[c] - row[o])
    total_range = row[h] - row[l]

    if total_range < 0.0003:  # Minimum range
        return None, None

    upper_wick = row[h] - max(row[c], row[o])
    lower_wick = min(row[c], row[o]) - row[l]
    body_ratio = body / total_range

    # Bullish pin bar: long lower wick, small body at top
    if lower_wick > 2 * body and lower_wick > 0.6 * total_range and body_ratio < 0.3:
        return 'PIN_BAR', 'BUY'

    # Bearish pin bar: long upper wick, small body at bottom
    if upper_wick > 2 * body and upper_wick > 0.6 * total_range and body_ratio < 0.3:
        return 'PIN_BAR', 'SELL'

    return None, None


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

    # Determine overall label
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


def run_backtest(df: pd.DataFrame, col_map: dict) -> Tuple[List[Trade], float, dict, dict, dict]:
    """Run backtest with Round 4 combined optimizations."""
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

    # Optimization 1: Kelly Position Sizer
    kelly_sizer = KellyPositionSizer(min_trades=20) if USE_KELLY_SIZING else None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0}
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

            # Check exit conditions (SL/TP)
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

                # Record trade for intra-month tracking
                risk_manager.record_trade(pnl, current_time)

                # Record trade for pattern filter
                if USE_PATTERN_FILTER:
                    pattern_filter.record_trade(position.direction, pnl, current_time)

                # Record trade for Kelly sizer
                if kelly_sizer:
                    kelly_sizer.record_trade(position)

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

        # Check intra-month risk manager
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

        # Assess market condition
        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        if intra_month_adj > 0:
            skip_stats['DYNAMIC_ADJ'] += 1

        condition_stats[market_cond.label] = condition_stats.get(market_cond.label, 0) + 1

        # Detect POIs
        pois = []

        # Entry Signal 1: Order Block detection
        if USE_ORDER_BLOCK:
            ob_pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            pois.extend(ob_pois)

        # Entry Signal 2: EMA Pullback detection
        if USE_EMA_PULLBACK:
            ema_pois = detect_ema_pullback(current_slice, col_map, atr_series, dynamic_quality)
            existing_indices = {p['idx'] for p in pois}
            for ep in ema_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        # Entry Signal 3: PIN BAR detection (Optimization 4)
        if USE_PIN_BAR_ENTRY:
            pin_type, pin_direction = detect_pin_bar(df, col_map, i)
            if pin_type and pin_direction:
                # Calculate quality based on wick ratio
                row = df.iloc[i]
                o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
                total_range = row[h] - row[l]
                lower_wick = min(row[c], row[o]) - row[l]
                upper_wick = row[h] - max(row[c], row[o])

                if pin_direction == 'BUY':
                    wick_ratio = lower_wick / total_range if total_range > 0 else 0
                else:
                    wick_ratio = upper_wick / total_range if total_range > 0 else 0

                quality = min(100, max(55, wick_ratio * 120))  # Scale 0.6+ to 72+

                if quality >= dynamic_quality:
                    # Check if this index already has a POI
                    if i not in {p['idx'] for p in pois}:
                        pois.append({
                            'price': current_price,
                            'direction': pin_direction,
                            'quality': quality,
                            'idx': i,
                            'type': 'PIN_BAR'
                        })

        if not pois:
            continue

        for poi in pois:
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            poi_type = poi.get('type', 'ORDER_BLOCK')

            # Session-based filter
            session_skip, session_reason = should_skip_by_session(hour, poi_type)
            if session_skip:
                skip_stats[session_reason] = skip_stats.get(session_reason, 0) + 1
                continue

            # Get entry type
            if poi_type == 'EMA_PULLBACK':
                entry_type = 'MOMENTUM'
            elif poi_type == 'PIN_BAR':
                entry_type = 'PIN_BAR'
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

            # Pattern-based filter check
            pattern_size_mult = 1.0
            pattern_extra_q = 0
            if USE_PATTERN_FILTER:
                pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(poi['direction'])
                if not pattern_can_trade:
                    skip_stats['PATTERN_STOPPED'] += 1
                    continue

            # Check stricter quality if needed
            if pattern_extra_q > 0:
                if poi['quality'] < dynamic_quality + pattern_extra_q:
                    continue

            # Calculate SL/TP
            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO  # 4:1 if USE_4_TO_1_RR is True

            # Calculate lot size
            kelly_pct_used = 0.0
            if USE_KELLY_SIZING and kelly_sizer:
                lot_size, kelly_pct_used = kelly_sizer.calculate_lot_size(
                    balance, sl_pips, risk_mult * pattern_size_mult
                )
            else:
                risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult * pattern_size_mult
                lot_size = risk_amount / (sl_pips * PIP_VALUE)
                lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            risk_amount = lot_size * sl_pips * PIP_VALUE

            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            # Track entry signal type
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
                kelly_pct=kelly_pct_used
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


def print_results(stats: dict, trades: List[Trade], condition_stats: dict,
                  skip_stats: dict = None, entry_stats: dict = None):
    print(f"\n{'='*70}")
    print(f"ROUND 4 COMBINED OPTIMIZATIONS - BACKTEST RESULTS")
    print(f"{'='*70}")

    print(f"\n[OPTIMIZATION FLAGS]")
    print(f"{'-'*50}")
    print(f"  1. Kelly 25% Sizing:    {'ENABLED' if USE_KELLY_SIZING else 'DISABLED'}")
    print(f"  2. 4:1 R:R Exit:        {'ENABLED' if USE_4_TO_1_RR else 'DISABLED'}")
    print(f"  3. Fixed Day Mults:     {'ENABLED' if USE_FIXED_DAY_MULTS else 'DISABLED'}")
    print(f"  4. PIN_BAR Entry:       {'ENABLED' if USE_PIN_BAR_ENTRY else 'DISABLED'}")

    print(f"\n[PERFORMANCE SUMMARY]")
    print(f"{'-'*50}")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Net P/L: ${stats['net_pnl']:+,.2f}")
    print(f"  Max Drawdown: ${stats['max_dd']:,.2f} ({stats['max_dd_pct']:.2f}%)")
    print(f"  Losing Months: {stats['losing_months']}/{stats['total_months']}")
    print(f"  Final Balance: ${stats['final_balance']:,.2f}")

    print(f"\n[COMPARISON VS v6.8.0 BASELINE]")
    print(f"{'-'*50}")
    baseline = {
        'trades': 115, 'win_rate': 49.6, 'pf': 5.09,
        'profit': 22346, 'losing_months': 0
    }
    print(f"  {'Metric':<20} {'v6.8.0':>12} {'Round4':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Trades':<20} {baseline['trades']:>12} {stats['total_trades']:>12} {stats['total_trades'] - baseline['trades']:>+12}")
    print(f"  {'Win Rate':<20} {baseline['win_rate']:>11.1f}% {stats['win_rate']:>11.1f}% {stats['win_rate'] - baseline['win_rate']:>+11.1f}%")
    print(f"  {'Profit Factor':<20} {baseline['pf']:>12.2f} {stats['profit_factor']:>12.2f} {stats['profit_factor'] - baseline['pf']:>+12.2f}")
    print(f"  {'Net Profit':<20} ${baseline['profit']:>10,} ${stats['net_pnl']:>10,.0f} ${stats['net_pnl'] - baseline['profit']:>+10,.0f}")
    print(f"  {'Losing Months':<20} {baseline['losing_months']:>12} {stats['losing_months']:>12} {stats['losing_months'] - baseline['losing_months']:>+12}")

    # Entry stats
    if entry_stats:
        print(f"\n[ENTRY SIGNAL BREAKDOWN]")
        print(f"{'-'*50}")
        for entry_type, count in sorted(entry_stats.items()):
            pct = count / stats['total_trades'] * 100 if stats['total_trades'] > 0 else 0
            print(f"  {entry_type:<20}: {count:>5} ({pct:>5.1f}%)")

    # Monthly breakdown
    print(f"\n[MONTHLY P/L BREAKDOWN]")
    print(f"{'-'*50}")
    print(f"  {'Month':<12} {'P/L':>12} {'Status':>10}")
    print(f"  {'-'*34}")
    for month, pnl in stats['monthly'].items():
        status = "WIN" if pnl >= 0 else "LOSS"
        print(f"  {str(month):<12} ${pnl:>+10,.0f} {status:>10}")

    # Skip stats
    if skip_stats:
        print(f"\n[FILTER STATISTICS]")
        print(f"{'-'*50}")
        for reason, count in sorted(skip_stats.items()):
            print(f"  {reason:<25}: {count:>5}")

    # Day of week analysis
    print(f"\n[DAY OF WEEK ANALYSIS]")
    print(f"{'-'*50}")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_trades = {i: [] for i in range(5)}
    for t in trades:
        day = t.entry_time.weekday()
        if day < 5:
            day_trades[day].append(t)

    print(f"  {'Day':<12} {'Trades':>8} {'Wins':>8} {'WR':>8} {'P/L':>12}")
    print(f"  {'-'*48}")
    for day in range(5):
        day_t = day_trades[day]
        if day_t:
            wins = len([t for t in day_t if t.pnl > 0])
            wr = wins / len(day_t) * 100
            pnl = sum(t.pnl for t in day_t)
            print(f"  {day_names[day]:<12} {len(day_t):>8} {wins:>8} {wr:>7.1f}% ${pnl:>+10,.0f}")
        else:
            print(f"  {day_names[day]:<12} {0:>8} {0:>8} {'N/A':>8} ${0:>+10,.0f}")

    # Final recommendation
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")

    # Check criteria
    profit_ok = stats['net_pnl'] >= baseline['profit'] * 0.8  # Allow 20% variance
    pf_ok = stats['profit_factor'] >= 1.5
    losing_months_ok = stats['losing_months'] <= 1  # Allow max 1 losing month

    if profit_ok and pf_ok and losing_months_ok:
        print(f"\n  STATUS: READY FOR v6.9.0")
        print(f"\n  All criteria met:")
        print(f"  - Profit maintained (>= 80% of baseline)")
        print(f"  - PF >= 1.5")
        print(f"  - Losing months <= 1")
    else:
        print(f"\n  STATUS: NEEDS ADJUSTMENT")
        print(f"\n  Issues found:")
        if not profit_ok:
            print(f"  - Profit too low: ${stats['net_pnl']:,.0f} vs ${baseline['profit']*0.8:,.0f} minimum")
        if not pf_ok:
            print(f"  - PF too low: {stats['profit_factor']:.2f} vs 1.5 minimum")
        if not losing_months_ok:
            print(f"  - Too many losing months: {stats['losing_months']} vs 1 maximum")

    print(f"\n{'='*70}\n")


async def main():
    print("\n" + "="*70)
    print("ROUND 4 COMBINED OPTIMIZATIONS TEST")
    print("="*70)
    print(f"\nOptimizations enabled:")
    print(f"  1. Kelly 25% Position Sizing: {'YES' if USE_KELLY_SIZING else 'NO'}")
    print(f"  2. Fixed 4:1 R:R Exit: {'YES' if USE_4_TO_1_RR else 'NO'}")
    print(f"  3. Fixed Day Multipliers: {'YES' if USE_FIXED_DAY_MULTS else 'NO'}")
    print(f"  4. PIN_BAR Entry Signal: {'YES' if USE_PIN_BAR_ENTRY else 'NO'}")

    start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 30, tzinfo=timezone.utc)

    print(f"\nPeriod: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("\nFetching data...")

    df = await fetch_data(SYMBOL, "H1", start_date, end_date)
    if df.empty:
        print("ERROR: No data fetched!")
        return

    print(f"Data loaded: {len(df)} bars")

    # Detect column names
    if 'Open' in df.columns:
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
    else:
        col_map = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}

    print("\nRunning backtest with Round 4 combined optimizations...")
    trades, max_dd, condition_stats, skip_stats, entry_stats = run_backtest(df, col_map)

    if not trades:
        print("ERROR: No trades generated!")
        return

    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, condition_stats, skip_stats, entry_stats)


if __name__ == "__main__":
    asyncio.run(main())
