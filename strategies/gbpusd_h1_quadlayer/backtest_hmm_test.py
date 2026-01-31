"""
SURGE-WSI H1 v6.8 GBPUSD - HMM REGIME DETECTION TEST
=====================================================

This is a TEST backtest to evaluate HMM (Hidden Markov Model) regime
detection as an additional filter for the v6.8 strategy.

HMM Configuration:
- GaussianHMM with 2-3 components (states)
- Features: returns, rolling volatility, ATR ratio
- Train on rolling 500-bar lookback window
- Only trade in favorable regimes (Trending state)

Purpose: Test if HMM can improve results by filtering out choppy/volatile regimes

Baseline (v6.8.0):
- 115 trades, 49.6% WR, PF 5.09, $22,346 profit, 0 losing months

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
from dataclasses import dataclass, field
from enum import Enum

from config import config
from src.data.db_handler import DBHandler

# Import HMM
from hmmlearn.hmm import GaussianHMM

# Import shared trading filters from strategy package
from gbpusd_h1_quadlayer.trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    ChoppinessFilter,
    calculate_choppiness_index,
    DirectionalMomentumFilter,
    H4BiasFilter,
    H4Bias,
    get_h4_bias,
    MarketStructureFilter,
    StructureType,
    StructureSignal,
    detect_swing_points,
    detect_market_structure,
    STRUCTURE_SWING_LOOKBACK,
    STRUCTURE_BOS_CONFIDENCE,
    STRUCTURE_CHOCH_CONFIDENCE,
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

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# HMM REGIME DETECTION CONFIGURATION
# ============================================================
USE_HMM_FILTER = True           # Enable HMM regime filter
HMM_N_COMPONENTS = 2            # Number of hidden states (2=trend/range - simpler, more stable)
HMM_LOOKBACK = 300              # Training lookback window (reduced for faster adaptation)
HMM_RETRAIN_INTERVAL = 100      # Retrain every N bars
HMM_STRICT_MODE = True          # True = only TRENDING, False = allow all states
HMM_WARMUP_BARS = 100           # Trade normally during warmup (before HMM is trained)

# Feature engineering
HMM_RETURN_WINDOW = 1           # Returns lookback
HMM_VOL_WINDOW = 20             # Volatility (std) lookback
HMM_ATR_WINDOW = 14             # ATR lookback


# ============================================================
# CONFIGURATION (same as v6.8)
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

# Layer 2 thresholds
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

# Session filter config (from v6.8)
USE_SESSION_POI_FILTER = True
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]

# Entry signals
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True
USE_PATTERN_FILTER = True


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


# ============================================================
# HMM REGIME DETECTOR CLASS
# ============================================================
class HMMRegimeDetector:
    """
    Hidden Markov Model regime detector.
    Classifies market into states based on returns and volatility patterns.
    """

    def __init__(self, n_components: int = 3, lookback: int = 500):
        self.n_components = n_components
        self.lookback = lookback
        self.model: Optional[GaussianHMM] = None
        self.is_trained = False
        self.state_labels: Dict[int, str] = {}  # State -> label mapping
        self.favorable_states: List[int] = []   # States where trading is allowed
        self.last_train_idx = 0
        self.current_state = -1
        self.training_attempts = 0
        self.training_successes = 0

        # Statistics
        self.state_counts: Dict[int, int] = {}
        self.state_returns: Dict[int, List[float]] = {}

    def prepare_features(self, df: pd.DataFrame, col_map: dict) -> np.ndarray:
        """
        Prepare features for HMM training/prediction.
        Features:
        1. Log returns (normalized)
        2. Rolling volatility (normalized)
        3. ATR ratio
        """
        c = col_map['close']
        h = col_map['high']
        l = col_map['low']

        # Log returns
        close = df[c].copy()
        returns = np.log(close / close.shift(HMM_RETURN_WINDOW))

        # Rolling volatility
        volatility = returns.rolling(HMM_VOL_WINDOW).std()

        # ATR ratio
        tr1 = df[h] - df[l]
        tr2 = abs(df[h] - close.shift(1))
        tr3 = abs(df[l] - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(HMM_ATR_WINDOW).mean()
        atr_mean = atr.rolling(50).mean()
        atr_ratio = atr / (atr_mean + 1e-10)

        # Combine features
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'atr_ratio': atr_ratio
        }).dropna()

        # Normalize features to prevent numerical issues
        features_arr = features.values
        if len(features_arr) > 0:
            # Standardize each feature
            means = np.mean(features_arr, axis=0)
            stds = np.std(features_arr, axis=0)
            stds[stds < 1e-10] = 1.0  # Prevent division by zero
            features_arr = (features_arr - means) / stds

        return features_arr

    def train(self, df: pd.DataFrame, col_map: dict, idx: int) -> bool:
        """
        Train HMM on historical data up to idx.
        Returns True if training successful.
        """
        self.training_attempts += 1

        try:
            # Get training data
            start_idx = max(0, idx - self.lookback)
            train_df = df.iloc[start_idx:idx+1].copy()

            if len(train_df) < 100:  # Need minimum data
                return False

            features = self.prepare_features(train_df, col_map)

            if len(features) < 100:
                return False

            # Add small noise to prevent singular covariance matrices
            features = features + np.random.randn(*features.shape) * 1e-6

            # Train HMM with diagonal covariance (more stable)
            self.model = GaussianHMM(
                n_components=self.n_components,
                covariance_type='diag',  # Use diagonal for stability
                n_iter=200,
                random_state=42,
                tol=1e-4
            )
            self.model.fit(features)

            # Predict states for training data
            states = self.model.predict(features)

            # Analyze states to determine labels
            self._analyze_states(features, states, train_df, col_map)

            self.is_trained = True
            self.last_train_idx = idx
            self.training_successes += 1

            return True

        except Exception as e:
            # Silently fail - HMM training can be unstable
            return False

    def _analyze_states(self, features: np.ndarray, states: np.ndarray,
                       df: pd.DataFrame, col_map: dict):
        """
        Analyze state characteristics to assign labels.
        - Low volatility + consistent returns = TRENDING
        - High volatility + erratic returns = VOLATILE
        - Medium volatility + low returns = RANGING
        """
        self.state_labels = {}
        state_stats = {}

        for state in range(self.n_components):
            mask = states == state
            if not mask.any():
                continue

            state_features = features[mask]

            # Calculate state characteristics
            avg_volatility = np.mean(state_features[:, 1])
            avg_abs_return = np.mean(np.abs(state_features[:, 0]))
            avg_atr_ratio = np.mean(state_features[:, 2])

            # Efficiency: directional movement vs total movement
            state_returns = state_features[:, 0]
            efficiency = abs(np.sum(state_returns)) / (np.sum(np.abs(state_returns)) + 1e-10)

            state_stats[state] = {
                'volatility': avg_volatility,
                'abs_return': avg_abs_return,
                'atr_ratio': avg_atr_ratio,
                'efficiency': efficiency,
                'count': mask.sum()
            }

        if not state_stats:
            # Fallback: allow all states
            for s in range(self.n_components):
                self.state_labels[s] = "UNKNOWN"
                self.favorable_states.append(s)
            return

        # Sort states by efficiency (higher = more trending)
        sorted_states = sorted(state_stats.items(),
                              key=lambda x: x[1]['efficiency'],
                              reverse=True)

        # Assign labels based on characteristics
        self.favorable_states = []
        for i, (state, stats) in enumerate(sorted_states):
            if self.n_components == 2:
                if i == 0:
                    self.state_labels[state] = "TRENDING"
                    self.favorable_states.append(state)
                else:
                    self.state_labels[state] = "RANGING"
            else:  # 3 components
                if i == 0:
                    self.state_labels[state] = "TRENDING"
                    self.favorable_states.append(state)
                elif i == 1:
                    # Check if this is ranging or volatile
                    if len(sorted_states) > 2 and stats['volatility'] > sorted_states[2][1]['volatility']:
                        self.state_labels[state] = "VOLATILE"
                    else:
                        self.state_labels[state] = "RANGING"
                        # Also allow ranging if efficiency is decent
                        if stats['efficiency'] > 0.15:
                            self.favorable_states.append(state)
                else:
                    self.state_labels[state] = "CHOPPY"

        # In strict mode, only allow the best trending state
        # In normal mode, allow BOTH states (don't filter - just analyze)
        if HMM_STRICT_MODE:
            self.favorable_states = [sorted_states[0][0]]  # Only TRENDING state
        else:
            # Allow ALL states - HMM will just provide regime information, not filter
            self.favorable_states = [s[0] for s in sorted_states]

    def predict(self, df: pd.DataFrame, col_map: dict, idx: int) -> Tuple[int, str]:
        """
        Predict current regime state.
        Returns (state_id, state_label)
        """
        if not self.is_trained or self.model is None:
            return -1, "UNKNOWN"

        try:
            # Get recent data for prediction
            start_idx = max(0, idx - 50)
            pred_df = df.iloc[start_idx:idx+1]
            features = self.prepare_features(pred_df, col_map)

            if len(features) == 0:
                return -1, "UNKNOWN"

            # Predict state for last observation
            states = self.model.predict(features)
            current_state = states[-1]

            self.current_state = current_state

            # Update statistics
            self.state_counts[current_state] = self.state_counts.get(current_state, 0) + 1

            label = self.state_labels.get(current_state, "UNKNOWN")
            return current_state, label

        except Exception as e:
            return -1, "UNKNOWN"

    def should_trade(self, state: int, bar_idx: int) -> Tuple[bool, str]:
        """
        Check if trading is allowed in current state.
        Returns (can_trade, reason)
        """
        # Allow trading during warmup period (before HMM is ready)
        if bar_idx < HMM_WARMUP_BARS:
            return True, "HMM_WARMUP"

        # If HMM not trained after warmup, allow trading (fallback)
        if not self.is_trained:
            return True, "HMM_FALLBACK"

        # If prediction failed, allow trading
        if state < 0:
            return True, "HMM_PRED_FAIL"

        if state in self.favorable_states:
            label = self.state_labels.get(state, "UNKNOWN")
            return True, f"HMM_{label}"
        else:
            label = self.state_labels.get(state, "UNKNOWN")
            return False, f"HMM_SKIP_{label}"

    def get_stats(self) -> Dict:
        """Get HMM statistics"""
        return {
            'is_trained': self.is_trained,
            'n_components': self.n_components,
            'favorable_states': self.favorable_states,
            'state_labels': self.state_labels,
            'state_counts': self.state_counts,
            'current_state': self.current_state
        }


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
    # HMM fields
    hmm_state: int = -1
    hmm_label: str = ""


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
# HELPER FUNCTIONS (same as v6.8)
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
    """DUAL-LAYER market condition assessment"""
    lookback = 20
    start_idx = max(0, idx - lookback)

    h, l, c = col_map['high'], col_map['low'], col_map['close']

    # LAYER 2: TECHNICAL INDICATORS
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

    # LAYER 1: MONTHLY PROFILE ADJUSTMENT
    monthly_adj = get_monthly_quality_adjustment(current_time)

    # COMBINE LAYERS
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


# ============================================================
# MAIN BACKTEST WITH HMM FILTER
# ============================================================
def run_backtest_with_hmm(df: pd.DataFrame, col_map: dict) -> Tuple[List[Trade], float, dict, dict, dict]:
    """Run backtest with HMM regime filter"""
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

    # HMM Regime Detector
    hmm_detector = HMMRegimeDetector(n_components=HMM_N_COMPONENTS, lookback=HMM_LOOKBACK) if USE_HMM_FILTER else None

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0, 'HMM_FILTERED': 0}
    hmm_stats = {
        'trades_by_state': {},
        'wins_by_state': {},
        'pnl_by_state': {},
        'filtered_by_state': {}
    }
    entry_stats = {'ORDER_BLOCK': 0, 'EMA_PULLBACK': 0}

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

        # HMM: Retrain periodically
        if USE_HMM_FILTER and hmm_detector:
            if not hmm_detector.is_trained or (i - hmm_detector.last_train_idx) >= HMM_RETRAIN_INTERVAL:
                hmm_detector.train(df, col_map, i)

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

                # Track HMM state performance
                if USE_HMM_FILTER and position.hmm_state >= 0:
                    state = position.hmm_state
                    hmm_stats['trades_by_state'][state] = hmm_stats['trades_by_state'].get(state, 0) + 1
                    hmm_stats['pnl_by_state'][state] = hmm_stats['pnl_by_state'].get(state, 0) + pnl
                    if pnl > 0:
                        hmm_stats['wins_by_state'][state] = hmm_stats['wins_by_state'].get(state, 0) + 1

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

        # LAYER 3: Check intra-month risk manager
        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            if 'MONTH' in skip_reason:
                skip_stats['MONTH_STOPPED'] += 1
            elif 'DAY' in skip_reason:
                skip_stats['DAY_STOPPED'] += 1
            continue

        # Reset pattern filter if month changed
        month_key = (current_time.year, current_time.month)
        if month_key != current_month_key:
            current_month_key = month_key
            if USE_PATTERN_FILTER:
                pattern_filter.reset_for_month(current_time.month)

        # HMM REGIME CHECK
        current_hmm_state = -1
        current_hmm_label = ""
        if USE_HMM_FILTER and hmm_detector:
            # Predict current regime (or return -1 if not trained)
            current_hmm_state, current_hmm_label = hmm_detector.predict(df, col_map, i)
            can_trade_hmm, hmm_reason = hmm_detector.should_trade(current_hmm_state, i)
            if not can_trade_hmm:
                skip_stats['HMM_FILTERED'] += 1
                hmm_stats['filtered_by_state'][current_hmm_state] = hmm_stats['filtered_by_state'].get(current_hmm_state, 0) + 1
                continue

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # TRIPLE-LAYER QUALITY
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

            # SESSION FILTER
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

            # LAYER 4: Pattern filter
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
                hmm_state=current_hmm_state,
                hmm_label=current_hmm_label
            )
            break

    return trades, max_dd, condition_stats, skip_stats, entry_stats, hmm_stats, hmm_detector


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
                  skip_stats: dict, entry_stats: dict, hmm_stats: dict,
                  hmm_detector: Optional[HMMRegimeDetector]):
    print(f"\n{'='*70}")
    print(f"HMM REGIME DETECTION TEST - GBPUSD H1 v6.8")
    print(f"{'='*70}")

    print(f"\n[HMM CONFIGURATION]")
    print(f"{'-'*50}")
    print(f"  HMM Filter: {'ENABLED' if USE_HMM_FILTER else 'DISABLED'}")
    print(f"  Number of States: {HMM_N_COMPONENTS}")
    print(f"  Training Lookback: {HMM_LOOKBACK} bars")
    print(f"  Retrain Interval: {HMM_RETRAIN_INTERVAL} bars")

    if hmm_detector and hmm_detector.is_trained:
        print(f"\n[HMM STATE ANALYSIS]")
        print(f"{'-'*50}")
        print(f"  State Labels: {hmm_detector.state_labels}")
        print(f"  Favorable States: {hmm_detector.favorable_states}")

        print(f"\n  State Distribution:")
        for state, count in sorted(hmm_detector.state_counts.items()):
            label = hmm_detector.state_labels.get(state, "UNKNOWN")
            is_favorable = "(*)" if state in hmm_detector.favorable_states else ""
            print(f"    State {state} ({label}){is_favorable}: {count} bars")

    print(f"\n[HMM TRADE STATISTICS]")
    print(f"{'-'*50}")
    print(f"  Trades filtered by HMM: {skip_stats.get('HMM_FILTERED', 0)}")

    if hmm_stats['trades_by_state']:
        print(f"\n  Performance by HMM State:")
        for state in sorted(hmm_stats['trades_by_state'].keys()):
            label = hmm_detector.state_labels.get(state, "UNKNOWN") if hmm_detector else "UNKNOWN"
            n_trades = hmm_stats['trades_by_state'].get(state, 0)
            n_wins = hmm_stats['wins_by_state'].get(state, 0)
            pnl = hmm_stats['pnl_by_state'].get(state, 0)
            wr = (n_wins / n_trades * 100) if n_trades > 0 else 0
            filtered = hmm_stats['filtered_by_state'].get(state, 0)
            print(f"    State {state} ({label}): {n_trades} trades, {wr:.1f}% WR, ${pnl:+,.0f} | Filtered: {filtered}")

    print(f"\n[PROTECTION STATS]")
    print(f"{'-'*50}")
    for key, val in skip_stats.items():
        print(f"  {key}: {val}")

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
        year = month.year
        mon = month.month
        tradeable = MONTHLY_TRADEABLE_PCT.get((year, mon), SEASONAL_TEMPLATE.get(mon, 65))
        adj = get_monthly_quality_adjustment(datetime(year, mon, 1))
        print(f"  [{status}] {month}: ${pnl:+,.2f} (tradeable={tradeable}%, adj=+{adj})")

    # Comparison with baseline
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH BASELINE (v6.8.0)")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Baseline v6.8':<15} {'HMM Test':<15} {'Diff':<15}")
    print(f"{'-'*60}")

    baseline = {
        'trades': 115,
        'win_rate': 49.6,
        'pf': 5.09,
        'net_pnl': 22346,
        'losing_months': 0
    }

    trades_diff = stats['total_trades'] - baseline['trades']
    wr_diff = stats['win_rate'] - baseline['win_rate']
    pf_diff = stats['profit_factor'] - baseline['pf']
    pnl_diff = stats['net_pnl'] - baseline['net_pnl']
    lm_diff = stats['losing_months'] - baseline['losing_months']

    print(f"{'Trades':<20} {baseline['trades']:<15} {stats['total_trades']:<15} {trades_diff:+d}")
    print(f"{'Win Rate (%)':<20} {baseline['win_rate']:<15.1f} {stats['win_rate']:<15.1f} {wr_diff:+.1f}")
    print(f"{'Profit Factor':<20} {baseline['pf']:<15.2f} {stats['profit_factor']:<15.2f} {pf_diff:+.2f}")
    print(f"{'Net P/L ($)':<20} {baseline['net_pnl']:<15,.0f} {stats['net_pnl']:<15,.0f} {pnl_diff:+,.0f}")
    print(f"{'Losing Months':<20} {baseline['losing_months']:<15} {stats['losing_months']:<15} {lm_diff:+d}")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")

    # Decision logic
    improved = (
        stats['net_pnl'] >= baseline['net_pnl'] * 0.95 and  # Allow 5% margin
        stats['losing_months'] <= baseline['losing_months'] and
        stats['profit_factor'] >= baseline['pf'] * 0.9
    )

    if improved:
        print(f"[KEEP] HMM filter maintains or improves performance")
        print(f"  - Net P/L within acceptable range")
        print(f"  - No new losing months")
        print(f"  - Profit factor acceptable")
    else:
        print(f"[REJECT] HMM filter does not improve results")
        reasons = []
        if stats['net_pnl'] < baseline['net_pnl'] * 0.95:
            reasons.append(f"- Net P/L dropped by ${baseline['net_pnl'] - stats['net_pnl']:,.0f}")
        if stats['losing_months'] > baseline['losing_months']:
            reasons.append(f"- Added {stats['losing_months'] - baseline['losing_months']} losing month(s)")
        if stats['profit_factor'] < baseline['pf'] * 0.9:
            reasons.append(f"- Profit factor dropped from {baseline['pf']:.2f} to {stats['profit_factor']:.2f}")
        for r in reasons:
            print(r)

    print(f"{'='*70}")


async def main():
    timeframe = "H1"
    # Match baseline v6.8.0 period: 2025-01-01 to 2026-01-31 (13 months)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    print(f"{'='*70}")
    print(f"SURGE-WSI H1 v6.8 - HMM REGIME DETECTION TEST")
    print(f"{'='*70}")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"HMM States: {HMM_N_COMPONENTS}")
    print(f"Training Lookback: {HMM_LOOKBACK} bars")
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

    print(f"\nRunning backtest with HMM regime filter...")
    trades, max_dd, condition_stats, skip_stats, entry_stats, hmm_stats, hmm_detector = run_backtest_with_hmm(df, col_map)

    if not trades:
        print("No trades executed")
        return

    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, condition_stats, skip_stats, entry_stats, hmm_stats, hmm_detector)

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
        'hmm_state': t.hmm_state,
        'hmm_label': t.hmm_label
    } for t in trades])

    output_path = STRATEGY_DIR / "reports" / f"hmm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path.parent.mkdir(exist_ok=True)
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
