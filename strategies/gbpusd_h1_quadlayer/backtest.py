"""
SURGE-WSI H1 v6.4 GBPUSD - DUAL-LAYER QUALITY FILTER
=====================================================

Enhancement dari v6.3:
- Layer 1: MONTHLY PROFILE (dari market analysis data)
  - Bulan dengan tradeable_pct < 60% → +15 quality requirement
  - Bulan dengan tradeable_pct < 70% → +10 quality requirement
  - Bulan dengan tradeable_pct >= 70% → no adjustment

- Layer 2: REAL-TIME TECHNICAL (sama seperti v6.3)
  - ATR Stability, Efficiency, Trend Strength

Result: Dual protection against poor market conditions

Market Analysis Data (GBPUSD Monthly):
- January 2024: 67% tradeable
- February 2024: 55% tradeable (POOR) - ini yg bikin loss
- March 2024: 70% tradeable
- April 2024: 80% tradeable (EXCELLENT)
- dst...

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
from src.utils.telegram import TelegramNotifier, TelegramFormatter

# Import shared trading filters from strategy package
from gbpusd_h1_quadlayer.trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    ChoppinessFilter,
    calculate_choppiness_index,
    DirectionalMomentumFilter,
    calculate_lot_size,
    calculate_sl_tp,
    get_monthly_quality_adjustment,
    MONTHLY_TRADEABLE_PCT,
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
    CHOP_CHOPPY_THRESHOLD,
    CHOP_TRENDING_THRESHOLD,
    DIR_CONSEC_LOSS_CAUTION,
    DIR_CONSEC_LOSS_WARNING,
    DIR_CONSEC_LOSS_EXTREME,
)
from gbpusd_h1_quadlayer.strategy_config import (
    SYMBOL, PIP_SIZE, PIP_VALUE,
    RISK, TECHNICAL, INTRA_MONTH, PATTERN,
    MONTHLY_RISK_MULT,
)

# Optional vector database integration
try:
    from src.data.vector_provider import CachedFeatureProvider
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    CachedFeatureProvider = None

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# VECTOR DATABASE CONFIG
# ============================================================
USE_VECTOR_FEATURES = True  # Set False to disable vector feature caching
VECTOR_SYNC_ON_START = False  # Sync to Qdrant at backtest start


# ============================================================
# TELEGRAM NOTIFICATION CONFIG
# ============================================================
SEND_TO_TELEGRAM = True  # Set False to disable Telegram notifications

# ============================================================
# LAYER 4 PATTERN FILTER CONFIG
# ============================================================
USE_PATTERN_FILTER = True  # Enable Layer 4 Pattern Filter

# ============================================================
# LAYER 5 CHOPPINESS INDEX FILTER CONFIG (DEPRECATED)
# Based on E.W. Dreiss (1993) - Chaos Theory Applied to Markets
# ============================================================
USE_CHOPPINESS_FILTER = False  # DISABLED - use DirectionalMomentumFilter instead

# ============================================================
# LAYER 6 DIRECTIONAL MOMENTUM FILTER CONFIG
# Detects when one direction is consistently failing
# ============================================================
USE_DIRECTIONAL_FILTER = False  # Disabled - ADX regime detection is primary


# ============================================================
# CONFIGURATION v6.4 - DUAL-LAYER QUALITY FILTER
# Using values from strategy_config.py
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

# Layer 3: Intra-Month Risk Thresholds (for print_results display)
MONTHLY_LOSS_THRESHOLD_1 = INTRA_MONTH.loss_threshold_1
MONTHLY_LOSS_THRESHOLD_2 = INTRA_MONTH.loss_threshold_2
MONTHLY_LOSS_THRESHOLD_3 = INTRA_MONTH.loss_threshold_3
MONTHLY_LOSS_STOP = INTRA_MONTH.loss_stop
CONSECUTIVE_LOSS_THRESHOLD = INTRA_MONTH.consec_loss_quality
CONSECUTIVE_LOSS_MAX = INTRA_MONTH.consec_loss_day_stop

# Layer 4: Pattern filter enabled
PATTERN_FILTER_ENABLED = True

# NOTE: IntraMonthRiskManager, PatternBasedFilter, and get_monthly_quality_adjustment
# are now imported from trading_filters.py - no duplicate definitions needed


# ==========================================================
# LAYER 2: REAL-TIME TECHNICAL THRESHOLDS
# ==========================================================
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

# Risk Multipliers
MONTHLY_RISK = {
    1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,  # Feb lowered to 0.6
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
    """Market condition assessment"""
    atr_stability: float
    efficiency: float
    trend_strength: float
    technical_quality: float   # From Layer 2 (technical)
    monthly_adjustment: int    # From Layer 1 (profile)
    final_quality: float       # Combined threshold
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


def assess_market_condition(df: pd.DataFrame, col_map: dict, idx: int,
                           atr_series: pd.Series, current_time: datetime) -> MarketCondition:
    """
    DUAL-LAYER market condition assessment

    Layer 1: Monthly profile adjustment (from market analysis data)
    Layer 2: Real-time technical indicators
    """
    lookback = 20
    start_idx = max(0, idx - lookback)

    h, l, c = col_map['high'], col_map['low'], col_map['close']

    # ==========================================
    # LAYER 2: TECHNICAL INDICATORS
    # ==========================================

    # 1. ATR Stability
    recent_atr = atr_series.iloc[start_idx:idx+1]
    if len(recent_atr) > 5 and recent_atr.mean() > 0:
        atr_cv = recent_atr.std() / recent_atr.mean()
    else:
        atr_cv = 0.5

    # 2. Price Efficiency
    if idx >= lookback:
        net_move = abs(df[c].iloc[idx] - df[c].iloc[start_idx])
        total_move = sum(abs(df[c].iloc[i] - df[c].iloc[i-1]) for i in range(start_idx+1, idx+1))
        efficiency = net_move / total_move if total_move > 0 else 0
    else:
        efficiency = 0.1

    # 3. Trend Strength (ADX)
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
        technical_quality = MIN_QUALITY_GOOD  # 60
        tech_label = "TECH_GOOD"
    elif score >= 40:
        technical_quality = BASE_QUALITY  # 65
        tech_label = "TECH_NORMAL"
    else:
        technical_quality = MAX_QUALITY_BAD  # 80
        tech_label = "TECH_BAD"

    # ==========================================
    # LAYER 1: MONTHLY PROFILE ADJUSTMENT
    # ==========================================
    monthly_adj = get_monthly_quality_adjustment(current_time)

    # ==========================================
    # COMBINE LAYERS
    # ==========================================
    final_quality = technical_quality + monthly_adj

    # Determine overall label
    if monthly_adj >= 15:
        label = "POOR_MONTH"  # February 2024, etc.
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
    """
    Standard regime detection using EMA crossover.

    Same as original v6.4 dual filter backtest.
    The Pattern-Based Filter handles choppy market detection.
    """
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
                        pois.append({'price': current[l], 'direction': 'BUY', 'quality': quality, 'idx': i})

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
    """Run backtest with 5-LAYER quality filter (including Choppiness Index)"""
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

    # Layer 5: Choppiness Index filter (DEPRECATED)
    chop_filter = ChoppinessFilter() if USE_CHOPPINESS_FILTER else None

    # Layer 6: Directional Momentum filter (NEW!)
    directional_filter = DirectionalMomentumFilter() if USE_DIRECTIONAL_FILTER else None

    # Vector feature provider for fast cached feature access
    feature_provider = None
    if USE_VECTOR_FEATURES and VECTOR_AVAILABLE and CachedFeatureProvider is not None:
        try:
            feature_provider = CachedFeatureProvider(df)
            if feature_provider.initialize():
                print(f"[VECTOR] Pre-computed {feature_provider.total_features} feature vectors")
            else:
                feature_provider = None
                print("[VECTOR] Feature pre-computation failed, using standard mode")
        except Exception as e:
            print(f"[VECTOR] Initialization error: {e}, using standard mode")
            feature_provider = None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0, 'CHOP_ADJUSTED': 0, 'DIR_ADJUSTED': 0}

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

                # Layer 3: Record trade for intra-month tracking
                risk_manager.record_trade(pnl, current_time)

                # Layer 4: Record trade for pattern filter (if enabled)
                if USE_PATTERN_FILTER:
                    pattern_filter.record_trade(position.direction, pnl, current_time)

                # Layer 6: Record trade for directional filter (if enabled)
                if USE_DIRECTIONAL_FILTER and directional_filter:
                    directional_filter.record_trade(position.direction, pnl, current_time)

                position = None
            continue

        if current_time.weekday() >= 5:
            continue
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        hour = current_time.hour
        if not (7 <= hour <= 11 or 13 <= hour <= 17):
            continue
        session = "london" if 7 <= hour <= 11 else "newyork"

        # LAYER 3: Check intra-month risk manager
        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            if 'MONTH' in skip_reason:
                skip_stats['MONTH_STOPPED'] += 1
            elif 'DAY' in skip_reason:
                skip_stats['DAY_STOPPED'] += 1
            continue

        # LAYER 4 & 6: Reset filters if month changed
        month_key = (current_time.year, current_time.month)
        if month_key != current_month_key:
            current_month_key = month_key
            if USE_PATTERN_FILTER:
                pattern_filter.reset_for_month(current_time.month)
            if USE_DIRECTIONAL_FILTER and directional_filter:
                directional_filter.reset_for_month(current_time.month)

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # TRIPLE-LAYER QUALITY: Assess market condition
        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)

        # Layer 1 + Layer 2 + Layer 3 (intra-month dynamic)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        if intra_month_adj > 0:
            skip_stats['DYNAMIC_ADJ'] += 1

        condition_stats[market_cond.label] = condition_stats.get(market_cond.label, 0) + 1

        # Detect POIs with COMBINED quality threshold
        pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
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

            # LAYER 4: Pattern-based filter check (optional)
            pattern_size_mult = 1.0
            pattern_extra_q = 0
            if USE_PATTERN_FILTER:
                pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(poi['direction'])
                if not pattern_can_trade:
                    skip_stats['PATTERN_STOPPED'] += 1
                    continue

            # LAYER 5: Choppiness Index filter (DEPRECATED - disabled)
            # This filter NEVER blocks trades, only adjusts size and quality
            chop_size_mult = 1.0
            chop_extra_q = 0
            if USE_CHOPPINESS_FILTER and chop_filter:
                # Update choppiness with current price data
                h, l, c = col_map['high'], col_map['low'], col_map['close']
                highs = df[h].iloc[max(0, i-20):i+1].tolist()
                lows = df[l].iloc[max(0, i-20):i+1].tolist()
                closes = df[c].iloc[max(0, i-20):i+1].tolist()
                chop_filter.update(highs, lows, closes, current_time)

                # Get adjustments based on choppiness
                _, chop_extra_q, chop_size_mult, chop_reason = chop_filter.check_trade(poi['direction'])
                if chop_extra_q > 0 or chop_size_mult < 1.0:
                    skip_stats['CHOP_ADJUSTED'] += 1

            # LAYER 6: Directional Momentum filter (NEW!)
            # Detects when one direction is consistently failing
            # This filter NEVER blocks trades, only adjusts size and quality
            dir_size_mult = 1.0
            dir_extra_q = 0
            if USE_DIRECTIONAL_FILTER and directional_filter:
                # Get adjustments based on directional performance
                _, dir_extra_q, dir_size_mult, dir_reason = directional_filter.check_trade(poi['direction'])
                if dir_extra_q > 0 or dir_size_mult < 1.0:
                    skip_stats['DIR_ADJUSTED'] += 1

            # Combine all extra quality requirements
            total_extra_q = pattern_extra_q + chop_extra_q + dir_extra_q
            if total_extra_q > 0:
                effective_quality = dynamic_quality + total_extra_q
                # Re-check if POI meets stricter quality
                if poi['quality'] < effective_quality:
                    continue

            # Combine all size multipliers
            combined_size_mult = pattern_size_mult * chop_size_mult * dir_size_mult

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
                session=session,
                dynamic_quality=dynamic_quality,
                market_condition=market_cond.label,
                monthly_adj=market_cond.monthly_adjustment + intra_month_adj
            )
            break

    return trades, max_dd, condition_stats, skip_stats


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


async def send_telegram_report(stats: dict, trades: List[Trade], condition_stats: dict,
                               start_date: datetime, end_date: datetime):
    """Send backtest results to Telegram with visual chart"""
    if not SEND_TO_TELEGRAM:
        return

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        if not await telegram.initialize():
            print("Failed to initialize Telegram")
            return

        # ============================================================
        # GENERATE VISUAL REPORT IMAGE
        # ============================================================
        try:
            from src.utils.report_generator import BacktestReportGenerator

            # Prepare data for report generator (full trade details)
            trades_data = [
                {
                    'entry_time': str(t.entry_time),
                    'exit_time': str(t.exit_time),
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'lot_size': t.lot_size,
                    'sl': t.entry_price - (t.atr_pips * 0.0001 * 1.5) if t.direction == 'BUY' else t.entry_price + (t.atr_pips * 0.0001 * 1.5),
                    'tp': t.exit_price if 'TP' in t.exit_reason else None,
                    'pnl': t.pnl,
                    'exit_reason': t.exit_reason,
                    'session': t.session,
                    'market_condition': t.market_condition,
                    'quality_score': t.quality_score,
                }
                for t in trades
            ]

            monthly_stats = [
                {'month': f"{m.year}-{m.month:02d}", 'pnl': pnl}
                for m, pnl in stats['monthly'].items()
            ]

            summary = {
                'initial_balance': INITIAL_BALANCE,
                'net_pnl': stats['net_pnl'],
                'return_pct': (stats['net_pnl'] / INITIAL_BALANCE) * 100,
                'profit_factor': stats['profit_factor'],
                'win_rate': stats['win_rate'],
                'total_trades': stats['total_trades'],
                'winners': stats['winners'],
                'losers': stats['losers'],
                'avg_win': stats['avg_win'],
                'avg_loss': stats['avg_loss'],
                'losing_months': stats['losing_months'],
                'max_drawdown': stats.get('max_dd', 0),
            }

            # Generate image (save to strategy's reports folder)
            reports_dir = str(STRATEGY_DIR / "reports")
            generator = BacktestReportGenerator(output_dir=reports_dir)
            image_path = generator.generate_telegram_summary_image(
                summary=summary,
                monthly_stats=monthly_stats,
                filename=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

            if image_path:
                # Send image with caption
                caption = (
                    f"<b>SURGE-WSI v6.4 GBPUSD Backtest</b>\n"
                    f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
                    f"{'✅' if stats['net_pnl'] >= 5000 else '❌'} Profit: ${stats['net_pnl']:+,.0f}\n"
                    f"{'✅' if stats['profit_factor'] >= 2.0 else '❌'} PF: {stats['profit_factor']:.2f}\n"
                    f"{'✅' if stats['losing_months'] == 0 else '❌'} Losing Months: {stats['losing_months']}/13"
                )
                await telegram.send_photo(image_path, caption=caption)
                print(f"\n[TELEGRAM] Report image sent: {image_path}")

            # Generate and send PDF
            pdf_path = generator.generate_pdf_report(
                trades=trades_data,
                monthly_stats=monthly_stats,
                summary=summary,
                filename=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            if pdf_path:
                await telegram.send_document(pdf_path, caption="📄 <b>Full Backtest Report (PDF)</b>")
                print(f"[TELEGRAM] PDF report sent: {pdf_path}")

        except Exception as img_error:
            print(f"[TELEGRAM] Image generation failed: {img_error}, sending text instead")

            # Fallback: Send text report
            msg = TelegramFormatter.tree_header("BACKTEST RESULTS", "📊")
            msg += f"<b>H1 v6.4 GBPUSD - Triple-Layer Quality</b>\n"
            msg += f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"

            msg += TelegramFormatter.tree_section("Performance", TelegramFormatter.CHART)
            msg += TelegramFormatter.tree_item("Total Trades", str(stats['total_trades']))
            msg += TelegramFormatter.tree_item("Win Rate", f"{stats['win_rate']:.1f}%")
            msg += TelegramFormatter.tree_item("Profit Factor", f"{stats['profit_factor']:.2f}", last=True)

            msg += TelegramFormatter.tree_section("Profit/Loss", TelegramFormatter.MONEY)
            pnl_emoji = TelegramFormatter.CHECK if stats['net_pnl'] >= 0 else TelegramFormatter.CROSS
            msg += TelegramFormatter.tree_item("Net P/L", f"{pnl_emoji} ${stats['net_pnl']:+,.0f}")
            msg += TelegramFormatter.tree_item("Return", f"{(stats['net_pnl']/INITIAL_BALANCE)*100:+.1f}%", last=True)

            msg += "\n<b>Targets:</b>\n"
            msg += f"{'✅' if stats['net_pnl'] >= 5000 else '❌'} Profit >= $5K\n"
            msg += f"{'✅' if stats['profit_factor'] >= 2.0 else '❌'} PF >= 2.0\n"
            msg += f"{'✅' if stats['losing_months'] == 0 else '❌'} ZERO losing months"

            await telegram.send(msg)

        # ============================================================
        # MESSAGE 2: Monthly breakdown (compact text)
        # ============================================================
        msg2 = "<pre>"
        msg2 += f"📅 MONTHLY BREAKDOWN\n"
        msg2 += f"{'Month':<9} {'P/L':>9} {'T%':>4} {'Adj':>4}\n"
        msg2 += f"{'-'*9} {'-'*9} {'-'*4} {'-'*4}\n"

        for month, pnl in stats['monthly'].items():
            year = month.year
            mon = month.month
            tradeable = MONTHLY_TRADEABLE_PCT.get((year, mon), 70)
            adj = get_monthly_quality_adjustment(datetime(year, mon, 1))
            status = "✓" if pnl >= 0 else "✗"
            month_str = f"{year}-{mon:02d}"
            msg2 += f"{month_str:<9} ${pnl:>+7,.0f} {tradeable:>3}% +{adj:<2} {status}\n"

        msg2 += "</pre>"

        await telegram.send(msg2)

        print("\n[TELEGRAM] Results sent successfully!")

    except Exception as e:
        print(f"\n[TELEGRAM] Failed to send: {e}")


def print_results(stats: dict, trades: List[Trade], condition_stats: dict, skip_stats: dict = None):
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS - H1 v6.4 GBPUSD TRIPLE-LAYER QUALITY")
    print(f"{'='*70}")

    print(f"\n[TRIPLE-LAYER QUALITY CONFIGURATION]")
    print(f"{'-'*50}")
    print(f"  Layer 1 - Monthly Profile (from market analysis):")
    print(f"    tradeable < 30%: +50 quality (NO TRADE)")
    print(f"    tradeable < 40%: +35 quality (HALT)")
    print(f"    tradeable < 50%: +25 quality (extreme)")
    print(f"    tradeable < 60%: +15 quality (very poor)")
    print(f"    tradeable < 70%: +10 quality (below avg)")
    print(f"    tradeable < 75%: +5 quality (slight)")
    print(f"  Layer 2 - Technical (ATR stability, efficiency, trend):")
    print(f"    GOOD market: base quality = {MIN_QUALITY_GOOD}")
    print(f"    NORMAL market: base quality = {BASE_QUALITY}")
    print(f"    BAD market: base quality = {MAX_QUALITY_BAD}")
    print(f"  Layer 3 - Intra-Month Dynamic (OPTIMIZED):")
    print(f"    Monthly loss < ${MONTHLY_LOSS_THRESHOLD_1}: +5 quality")
    print(f"    Monthly loss < ${MONTHLY_LOSS_THRESHOLD_2}: +10 quality")
    print(f"    Monthly loss < ${MONTHLY_LOSS_THRESHOLD_3}: +15 quality")
    print(f"    Monthly loss < ${MONTHLY_LOSS_STOP}: STOP trading")
    print(f"    {CONSECUTIVE_LOSS_THRESHOLD}+ consecutive losses: +5 quality")
    print(f"    {CONSECUTIVE_LOSS_MAX}+ consecutive losses: STOP for day")
    if PATTERN_FILTER_ENABLED:
        print(f"  Layer 4 - Pattern-Based Filter:")
        print(f"    Warmup: first {WARMUP_TRADES} trades (observe only)")
        print(f"    Rolling WR halt: <{ROLLING_WR_HALT*100:.0f}% (window={ROLLING_WINDOW})")
        print(f"    Rolling WR caution: <{ROLLING_WR_CAUTION*100:.0f}% ({CAUTION_SIZE_MULT*100:.0f}% size)")
        print(f"    Both directions fail: {BOTH_DIRECTIONS_FAIL_THRESHOLD}+ each = HALT")
        print(f"    Recovery: need {RECOVERY_WIN_REQUIRED} win, {RECOVERY_SIZE_MULT*100:.0f}% size")
    if USE_CHOPPINESS_FILTER:
        print(f"  Layer 5 - Choppiness Index Filter (DEPRECATED):")
        print(f"    CHOP > {CHOP_CHOPPY_THRESHOLD}: Choppy (50% size, +10 quality)")
        print(f"    CHOP > 70: Extreme Choppy (30% size, +20 quality)")
        print(f"    CHOP < {CHOP_TRENDING_THRESHOLD}: Trending (full size)")
        print(f"    (Never blocks trades, only adjusts)")
    if USE_DIRECTIONAL_FILTER:
        print(f"  Layer 6 - Directional Momentum Filter (NEW!):")
        print(f"    {DIR_CONSEC_LOSS_CAUTION} consec losses: CAUTION (60% size, +5 quality)")
        print(f"    {DIR_CONSEC_LOSS_WARNING} consec losses: WARNING (35% size, +15 quality)")
        print(f"    {DIR_CONSEC_LOSS_EXTREME}+ consec losses: EXTREME (20% size, +25 quality)")
        print(f"    (Per-direction tracking, never blocks)")
    print(f"  Combined = Layer1 + Layer2 + Layer3 + Layer4 + Layer6")

    if skip_stats:
        print(f"\n[PROTECTION STATS]")
        print(f"{'-'*50}")
        print(f"  Month circuit breaker: {skip_stats.get('MONTH_STOPPED', 0)}")
        print(f"  Day stop: {skip_stats.get('DAY_STOPPED', 0)}")
        print(f"  Dynamic quality adj: {skip_stats.get('DYNAMIC_ADJ', 0)}")
        print(f"  Pattern filter stops: {skip_stats.get('PATTERN_STOPPED', 0)}")
        print(f"  Choppiness adjusted: {skip_stats.get('CHOP_ADJUSTED', 0)}")
        print(f"  Directional adjusted: {skip_stats.get('DIR_ADJUSTED', 0)}")

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
        # Show monthly quality adjustment
        year = month.year
        mon = month.month
        tradeable = MONTHLY_TRADEABLE_PCT.get((year, mon), 70)
        adj = get_monthly_quality_adjustment(datetime(year, mon, 1))
        print(f"  [{status}] {month}: ${pnl:+,.2f} (tradeable={tradeable}%, adj=+{adj})")

    # By market condition
    print(f"\n[TRADES BY MARKET CONDITION]")
    print(f"{'-'*50}")
    for cond in ['GOOD', 'NORMAL', 'BAD', 'CAUTION', 'POOR_MONTH']:
        cond_trades = [t for t in trades if t.market_condition == cond]
        if cond_trades:
            wins = len([t for t in cond_trades if t.pnl > 0])
            total = len(cond_trades)
            wr = wins / total * 100
            net = sum(t.pnl for t in cond_trades)
            avg_q = sum(t.dynamic_quality for t in cond_trades) / total
            print(f"  {cond:12} {total:>3} trades, {wr:>5.1f}% WR, Q={avg_q:.0f}, ${net:>+10,.0f}")

    # Show February 2024 specifically
    print(f"\n[FEBRUARY 2024 DETAIL]")
    print(f"{'-'*50}")
    feb_trades = [t for t in trades if t.entry_time.year == 2024 and t.entry_time.month == 2]
    if feb_trades:
        feb_pnl = sum(t.pnl for t in feb_trades)
        feb_wins = len([t for t in feb_trades if t.pnl > 0])
        print(f"  Trades: {len(feb_trades)}")
        print(f"  Wins: {feb_wins}")
        print(f"  P/L: ${feb_pnl:+,.2f}")
        for t in feb_trades:
            print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction} Q={t.quality_score:.0f} Qreq={t.dynamic_quality:.0f} ${t.pnl:+,.0f}")
    else:
        print(f"  No trades (filtered out by high quality requirement)")

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
    timeframe = "H1"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    print(f"SURGE-WSI H1 v6.4 GBPUSD - TRIPLE-LAYER QUALITY FILTER")
    print(f"{'='*70}")
    print(f"Triple-Layer Quality Filter:")
    print(f"  Layer 1: Monthly profile (from market analysis)")
    print(f"  Layer 2: Real-time technical indicators")
    print(f"  Layer 3: Intra-month dynamic risk (adaptive)")
    print(f"  Combined = Layer1 + Layer2 + Layer3")
    print(f"{'='*70}")
    print(f"Vector Features: {'ENABLED' if USE_VECTOR_FEATURES and VECTOR_AVAILABLE else 'DISABLED'}")

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

    print(f"\nRunning backtest with TRIPLE-LAYER quality filter...")
    trades, max_dd, condition_stats, skip_stats = run_backtest(df, col_map)

    if not trades:
        print("No trades executed")
        return

    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, condition_stats, skip_stats)

    # Send to Telegram
    await send_telegram_report(stats, trades, condition_stats, start, end)

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
        'session': t.session,
        'dynamic_quality': t.dynamic_quality,
        'market_condition': t.market_condition,
        'monthly_adj': t.monthly_adj
    } for t in trades])

    # Save to strategy's reports folder
    output_path = STRATEGY_DIR / "reports" / "quadlayer_trades.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
