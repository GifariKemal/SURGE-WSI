"""
SURGE-WSI H1 v6.9 GBPUSD - QUAD-LAYER QUALITY FILTER
=====================================================

Updated to v6.9 to match Live Executor and MQL5 EA.

Layers:
- Layer 1: MONTHLY PROFILE (dari market analysis data)
- Layer 2: REAL-TIME TECHNICAL (ATR Stability, Efficiency, Trend)
- Layer 3: INTRA-MONTH RISK (dynamic quality adjustment)
- Layer 4: PATTERN-BASED FILTER (choppy market detection)

v6.9 Changes (synced with Live/MQL5):
- MAX_ATR: 30 → 25 pips
- Thursday mult: 0.4 → 0.8
- Friday mult: 0.5 → 0.3
- Hour 7, 11: skip (0.0)
- London session: 8-10 UTC (not 7-11)
- Session POI filter v6.8: skip OB@8,16 and EMA@13,14

Author: SURIOTA Team
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
# CONFIGURATION v6.9 - QUAD-LAYER QUALITY FILTER (synced)
# ============================================================
SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = 1.0

SL_ATR_MULT = 1.5
TP_RATIO = 1.5
MAX_LOSS_PER_TRADE_PCT = 0.15

PIP_VALUE = 10.0
PIP_SIZE = 0.0001
MAX_LOT = 5.0
MIN_ATR = 8.0
MAX_ATR = 25.0  # v6.9: Changed from 30.0 to match Live/MQL5

# Base Quality Thresholds
BASE_QUALITY = 65
MIN_QUALITY_GOOD = 60
MAX_QUALITY_BAD = 80

# ============================================================
# LAYER 3: DYNAMIC INTRA-MONTH RISK ADJUSTMENT
# Adaptive system - OPTIMIZED MODE
# Balance between protection and allowing recovery trades
# ============================================================
MONTHLY_LOSS_THRESHOLD_1 = -150    # First warning: +5 quality
MONTHLY_LOSS_THRESHOLD_2 = -250    # Second warning: +10 quality
MONTHLY_LOSS_THRESHOLD_3 = -350    # Third warning: +15 quality
MONTHLY_LOSS_STOP = -400           # Circuit breaker: stop for month

CONSECUTIVE_LOSS_THRESHOLD = 3     # After 3 consecutive losses: +5 quality
CONSECUTIVE_LOSS_MAX = 6           # After 6 consecutive losses: stop for day

# Profit protection disabled
PROFIT_PROTECTION_THRESHOLD = 0

# No base protection
MIN_BASE_QUALITY_ADJUSTMENT = 0

# ============================================================
# LAYER 4: PATTERN-BASED CHOPPY MARKET DETECTOR
# Detects "whipsaw" markets where BOTH directions lose
# This pattern can appear in ANY month, not just April
# ============================================================
PATTERN_FILTER_ENABLED = True

# Warmup settings - filter observes but doesn't halt during warmup
WARMUP_TRADES = 15                 # First 15 trades are warmup (observe only)

# Probe trade settings - small "test" trade after halt
PROBE_TRADE_SIZE = 0.4             # Probe trade is 40% size
PROBE_QUALITY_EXTRA = 5            # Extra +5 quality for probe trades

# Choppy market detection thresholds (relaxed)
DIRECTION_TEST_WINDOW = 8          # Check last 8 trades for direction bias
MIN_DIRECTION_WIN_RATE = 0.10      # If a direction has <10% WR, avoid it
BOTH_DIRECTIONS_FAIL_THRESHOLD = 4 # If 4 trades in BOTH directions lose, HALT

# Rolling performance tracking (relaxed)
ROLLING_WINDOW = 10                # Track last 10 trades
ROLLING_WR_HALT = 0.10             # If rolling WR drops below 10%, halt trading
ROLLING_WR_CAUTION = 0.25          # If rolling WR below 25%, reduce size to 60%
CAUTION_SIZE_MULT = 0.6            # Size during caution mode

# Recovery settings
RECOVERY_WIN_REQUIRED = 1          # Need 1 win to exit halt state
RECOVERY_SIZE_MULT = 0.5           # Trade at 50% size during recovery

# ==========================================================
# LAYER 1: MONTHLY PROFILE (dari market analysis)
# Tradeable percentage berdasarkan historical analysis
# ==========================================================
MONTHLY_TRADEABLE_PCT = {
    # 2024
    (2024, 1): 67,   # January - OK
    (2024, 2): 55,   # February - POOR!
    (2024, 3): 70,   # March - Good
    (2024, 4): 80,   # April - Excellent
    (2024, 5): 62,   # May - Below avg
    (2024, 6): 68,   # June - OK
    (2024, 7): 78,   # July - Good
    (2024, 8): 65,   # August - Average
    (2024, 9): 72,   # September - Good
    (2024, 10): 58,  # October - Below avg
    (2024, 11): 66,  # November - OK
    (2024, 12): 60,  # December - Low (holidays)
    # 2025 - Pattern filter handles all months dynamically
    (2025, 1): 65, (2025, 2): 55, (2025, 3): 70, (2025, 4): 70,  # April - pattern filter handles
    (2025, 5): 62, (2025, 6): 68, (2025, 7): 78, (2025, 8): 65,
    (2025, 9): 72, (2025, 10): 58, (2025, 11): 66, (2025, 12): 60,
    # 2026
    (2026, 1): 65, (2026, 2): 55, (2026, 3): 70, (2026, 4): 80,
    (2026, 5): 62, (2026, 6): 68, (2026, 7): 78, (2026, 8): 65,
    (2026, 9): 72, (2026, 10): 58, (2026, 11): 66, (2026, 12): 60,
}

def get_monthly_quality_adjustment(dt: datetime) -> int:
    """
    Layer 1: Get quality adjustment based on monthly tradeable percentage

    Returns:
    - +50 if tradeable < 30% (NO TRADE - effectively halt)
    - +35 if tradeable < 40% (HALT - near impossible to trade)
    - +25 if tradeable < 50% (extremely poor month - near trading halt)
    - +15 if tradeable < 60% (very poor month)
    - +10 if tradeable < 70% (below average month)
    - +5  if tradeable < 75% (slightly below average)
    - 0   if tradeable >= 75% (good month)
    """
    key = (dt.year, dt.month)
    tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)  # Default 70% if unknown

    if tradeable_pct < 30:
        return 50  # NO TRADE - effectively halt all trading
    elif tradeable_pct < 40:
        return 35  # HALT - near impossible to trade
    elif tradeable_pct < 50:
        return 25  # Extremely poor - near halt
    elif tradeable_pct < 60:
        return 15  # Very poor month - high quality required
    elif tradeable_pct < 70:
        return 10  # Below average - moderate increase
    elif tradeable_pct < 75:
        return 5   # Slightly below average
    else:
        return 0   # Good month - no adjustment


class IntraMonthRiskManager:
    """
    Layer 3: Dynamic Intra-Month Risk Adjustment

    Tracks real-time monthly P&L and adjusts quality requirements
    to prevent losing months through adaptive risk management.
    Includes profit protection to avoid giving back gains.
    """

    def __init__(self):
        self.current_month = None
        self.monthly_pnl = 0.0
        self.monthly_peak = 0.0  # Track peak profit for protection
        self.consecutive_losses = 0
        self.daily_losses = 0
        self.current_day = None
        self.month_stopped = False
        self.day_stopped = False
        self.profit_protection_active = False

    def new_trade_check(self, dt: datetime) -> tuple:
        """
        Check if trading is allowed and get dynamic adjustment

        Returns:
            (can_trade: bool, dynamic_adj: int, reason: str)
        """
        month_key = (dt.year, dt.month)
        day_key = dt.date()

        # Reset for new month
        if self.current_month != month_key:
            self.current_month = month_key
            self.monthly_pnl = 0.0
            self.monthly_peak = 0.0
            self.consecutive_losses = 0
            self.month_stopped = False
            self.day_stopped = False
            self.profit_protection_active = False

        # Reset for new day
        if self.current_day != day_key:
            self.current_day = day_key
            self.daily_losses = 0
            self.day_stopped = False

        # Check circuit breakers
        if self.month_stopped:
            return False, 0, "MONTH_STOPPED"

        if self.day_stopped:
            return False, 0, "DAY_STOPPED"

        # Calculate dynamic adjustment based on monthly P&L
        dynamic_adj = MIN_BASE_QUALITY_ADJUSTMENT  # Start with base (0)

        # Monthly loss circuit breaker - tight at -$200
        if self.monthly_pnl <= MONTHLY_LOSS_STOP:
            self.month_stopped = True
            return False, 0, f"MONTH_CIRCUIT_BREAKER (${self.monthly_pnl:.0f})"

        # Profit protection (disabled if threshold is 0)
        if PROFIT_PROTECTION_THRESHOLD > 0:
            if self.monthly_peak >= PROFIT_PROTECTION_THRESHOLD and self.monthly_pnl < 0:
                self.profit_protection_active = True
                dynamic_adj += 10

        # Monthly loss thresholds - optimized
        if self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_3:
            dynamic_adj += 15  # At -$350
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_2:
            dynamic_adj += 10  # At -$250
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_1:
            dynamic_adj += 5   # At -$150

        # Adjustment for consecutive losses
        if self.consecutive_losses >= CONSECUTIVE_LOSS_MAX:
            self.day_stopped = True
            return False, 0, f"CONSECUTIVE_LOSS_STOP ({self.consecutive_losses})"

        if self.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            dynamic_adj += 5   # After 3 consecutive losses

        return True, dynamic_adj, "OK"

    def record_trade(self, pnl: float, dt: datetime):
        """Record trade result and update tracking"""
        self.monthly_pnl += pnl

        # Update peak profit
        if self.monthly_pnl > self.monthly_peak:
            self.monthly_peak = self.monthly_pnl

        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on win

    def get_status(self) -> dict:
        """Get current risk manager status"""
        return {
            'month': self.current_month,
            'monthly_pnl': self.monthly_pnl,
            'monthly_peak': self.monthly_peak,
            'consecutive_losses': self.consecutive_losses,
            'month_stopped': self.month_stopped,
            'day_stopped': self.day_stopped,
            'profit_protection': self.profit_protection_active
        }


class PatternBasedFilter:
    """
    Layer 4: Pattern-Based Choppy Market Detector

    Detects "whipsaw" markets where BOTH directions lose consistently.
    This pattern can appear in ANY month, learned from April 2025 analysis.

    Key features:
    1. Probe trades - small test trades to verify market tradability
    2. Direction-specific win rate tracking
    3. Rolling win rate monitoring
    4. Automatic halt when choppy market detected
    5. Recovery mode with gradual size increase
    """

    def __init__(self):
        self.trade_history = []  # List of (direction, pnl, timestamp)
        self.is_halted = False
        self.halt_reason = ""
        self.in_recovery = False
        self.recovery_wins = 0
        self.current_month = None
        self.probe_taken = False  # Has probe trade been taken this month?

    def reset_for_month(self, month: int):
        """Soft reset - keep history but reset monthly flags"""
        self.current_month = month
        self.probe_taken = False
        # Don't reset trade_history - we want to learn across months!
        # Only reset halt if we had recovery
        if self.in_recovery and self.recovery_wins >= RECOVERY_WIN_REQUIRED:
            self.is_halted = False
            self.in_recovery = False
            self.recovery_wins = 0

    def _get_rolling_stats(self) -> dict:
        """Calculate rolling statistics from recent trades"""
        if len(self.trade_history) < 3:
            return {'rolling_wr': 1.0, 'buy_wr': 1.0, 'sell_wr': 1.0, 'both_fail': False}

        recent = self.trade_history[-ROLLING_WINDOW:]
        wins = sum(1 for _, pnl, _ in recent if pnl > 0)
        rolling_wr = wins / len(recent) if recent else 1.0

        # Direction-specific stats
        buy_trades = [(d, p) for d, p, _ in recent if d == 'BUY']
        sell_trades = [(d, p) for d, p, _ in recent if d == 'SELL']

        buy_wr = sum(1 for _, p in buy_trades if p > 0) / len(buy_trades) if buy_trades else 1.0
        sell_wr = sum(1 for _, p in sell_trades if p > 0) / len(sell_trades) if sell_trades else 1.0

        # Check if BOTH directions are failing
        both_fail = False
        recent_window = self.trade_history[-BOTH_DIRECTIONS_FAIL_THRESHOLD*2:]
        if len(recent_window) >= BOTH_DIRECTIONS_FAIL_THRESHOLD * 2:
            buy_losses = sum(1 for d, p, _ in recent_window if d == 'BUY' and p < 0)
            sell_losses = sum(1 for d, p, _ in recent_window if d == 'SELL' and p < 0)
            if buy_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD and sell_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD:
                both_fail = True

        return {
            'rolling_wr': rolling_wr,
            'buy_wr': buy_wr,
            'sell_wr': sell_wr,
            'both_fail': both_fail
        }

    def check_trade(self, direction: str) -> tuple:
        """
        Check if trade is allowed based on pattern analysis

        Returns:
            (can_trade: bool, extra_quality: int, size_multiplier: float, reason: str)
        """
        if not PATTERN_FILTER_ENABLED:
            return True, 0, 1.0, "FILTER_DISABLED"

        # During warmup, always allow trades (just observe)
        if len(self.trade_history) < WARMUP_TRADES:
            return True, 0, 1.0, "WARMUP"

        # Check if halted
        if self.is_halted and not self.in_recovery:
            return False, 0, 1.0, f"HALTED: {self.halt_reason}"

        stats = self._get_rolling_stats()

        # Check for choppy market (both directions failing)
        if stats['both_fail']:
            self.is_halted = True
            self.halt_reason = "BOTH_DIRECTIONS_FAIL"
            self.in_recovery = True
            self.recovery_wins = 0
            return False, 0, 1.0, "CHOPPY_MARKET_DETECTED"

        # Check rolling win rate
        if stats['rolling_wr'] < ROLLING_WR_HALT:
            self.is_halted = True
            self.halt_reason = f"LOW_ROLLING_WR ({stats['rolling_wr']:.0%})"
            self.in_recovery = True
            self.recovery_wins = 0
            return False, 0, 1.0, f"LOW_WIN_RATE ({stats['rolling_wr']:.0%})"

        # Check direction-specific win rate (only if enough data)
        if len(self.trade_history) >= 10:
            if direction == 'BUY' and stats['buy_wr'] < MIN_DIRECTION_WIN_RATE:
                return False, 0, 1.0, f"BUY_DIRECTION_WEAK ({stats['buy_wr']:.0%})"
            if direction == 'SELL' and stats['sell_wr'] < MIN_DIRECTION_WIN_RATE:
                return False, 0, 1.0, f"SELL_DIRECTION_WEAK ({stats['sell_wr']:.0%})"

        # Determine size and quality adjustments
        size_mult = 1.0
        extra_q = 0

        # Recovery mode - trade smaller with probe
        if self.in_recovery:
            size_mult = RECOVERY_SIZE_MULT
            extra_q = PROBE_QUALITY_EXTRA

        # Caution mode - rolling WR below threshold
        elif stats['rolling_wr'] < ROLLING_WR_CAUTION:
            size_mult = CAUTION_SIZE_MULT
            extra_q = 3

        return True, extra_q, size_mult, "OK"

    def record_trade(self, direction: str, pnl: float, timestamp):
        """Record trade result and update pattern state"""
        if not PATTERN_FILTER_ENABLED:
            return

        self.trade_history.append((direction, pnl, timestamp))

        # Keep history manageable (last 50 trades)
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]

        # Mark probe as taken
        if not self.probe_taken:
            self.probe_taken = True

        # Track recovery
        if self.in_recovery:
            if pnl > 0:
                self.recovery_wins += 1
                if self.recovery_wins >= RECOVERY_WIN_REQUIRED:
                    self.is_halted = False
                    self.in_recovery = False
                    self.recovery_wins = 0
            else:
                self.recovery_wins = 0  # Reset on loss

    def get_stats(self) -> dict:
        """Get filter statistics"""
        stats = self._get_rolling_stats()
        return {
            'total_history': len(self.trade_history),
            'is_halted': self.is_halted,
            'in_recovery': self.in_recovery,
            'recovery_wins': self.recovery_wins,
            'rolling_wr': stats['rolling_wr'],
            'buy_wr': stats['buy_wr'],
            'sell_wr': stats['sell_wr'],
            'both_fail': stats['both_fail']
        }


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

# v6.9: Updated Thu 0.8, Fri 0.3 to match Live/MQL5
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}

# v6.9: Hour 7, 11 = 0.0 (skip) to match Live/MQL5
HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}

# ============================================================
# v6.8 SESSION POI FILTER
# Skip specific signal types at certain hours based on analysis
# ============================================================
SESSION_POI_FILTER_ENABLED = True
SKIP_ORDER_BLOCK_HOURS = [8, 16]    # Skip Order Block entries at these hours
SKIP_EMA_PULLBACK_HOURS = [13, 14]  # Skip EMA Pullback entries at these hours


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
    signal_type: str = ""  # ORDER_BLOCK or EMA_PULLBACK
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


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))


def calculate_adx(df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
    """Calculate ADX indicator"""
    h, l, c = col_map['high'], col_map['low'], col_map['close']

    plus_dm = (df[h] - df[h].shift(1)).clip(lower=0)
    minus_dm = (df[l].shift(1) - df[l]).clip(lower=0)

    tr = pd.concat([
        df[h] - df[l],
        abs(df[h] - df[c].shift(1)),
        abs(df[l] - df[c].shift(1))
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 0.0001))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 0.0001))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(period).mean()
    return adx


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


def detect_ema_pullback(df: pd.DataFrame, col_map: dict, idx: int, atr_pips: float,
                        ema20: pd.Series, rsi: pd.Series, adx: pd.Series,
                        min_quality: float) -> Optional[dict]:
    """
    Detect EMA Pullback signal - v6.9 matching MT5 logic

    Criteria:
    - Body ratio > 0.4
    - ADX > 20 (trending market)
    - RSI between 30-70 (not overbought/oversold)
    - Price within 1.5x ATR distance from EMA20
    """
    if idx < 2:
        return None

    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    bar = df.iloc[idx]

    body = abs(bar[c] - bar[o])
    total_range = bar[h] - bar[l]
    if total_range < 0.0003:
        return None

    body_ratio = body / total_range

    # Filter criteria
    if body_ratio < 0.4:
        return None
    if pd.isna(adx.iloc[idx]) or adx.iloc[idx] < 20:
        return None
    if pd.isna(rsi.iloc[idx]) or rsi.iloc[idx] < 30 or rsi.iloc[idx] > 70:
        return None

    is_bullish = bar[c] > bar[o]
    is_bearish = bar[c] < bar[o]

    current_ema20 = ema20.iloc[idx]
    current_ema50 = calculate_ema(df[c], 50).iloc[idx]

    # Trend check
    curr_bullish = bar[c] > current_ema20 > current_ema50
    curr_bearish = bar[c] < current_ema20 < current_ema50

    atr_distance = atr_pips * PIP_SIZE * 1.5

    # BUY: Uptrend pullback to EMA
    if curr_bullish and is_bullish:
        dist = bar[l] - current_ema20
        if dist <= atr_distance:
            touch_quality = max(0, 30 - (dist / PIP_SIZE))
            adx_quality = min(25, (adx.iloc[idx] - 15) * 1.5)
            rsi_quality = 25 if abs(50 - rsi.iloc[idx]) < 20 else 15
            body_quality = min(20, body_ratio * 30)
            quality = min(100, max(55, touch_quality + adx_quality + rsi_quality + body_quality))
            if quality >= min_quality:
                return {'direction': 'BUY', 'quality': quality, 'signal_type': 'EMA_PULLBACK'}

    # SELL: Downtrend pullback to EMA
    if curr_bearish and is_bearish:
        dist = current_ema20 - bar[h]
        if dist <= atr_distance:
            touch_quality = max(0, 30 - (dist / PIP_SIZE))
            adx_quality = min(25, (adx.iloc[idx] - 15) * 1.5)
            rsi_quality = 25 if abs(50 - rsi.iloc[idx]) < 20 else 15
            body_quality = min(20, body_ratio * 30)
            quality = min(100, max(55, touch_quality + adx_quality + rsi_quality + body_quality))
            if quality >= min_quality:
                return {'direction': 'SELL', 'quality': quality, 'signal_type': 'EMA_PULLBACK'}

    return None


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
    """Run backtest with QUAD-LAYER quality filter (including April special)"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    # Pre-calculate indicators for EMA Pullback detection
    c = col_map['close']
    ema20_series = calculate_ema(df[c], 20)
    rsi_series = calculate_rsi(df[c], 14)
    adx_series = calculate_adx(df, col_map, 14)

    # Layer 3: Intra-month risk manager
    risk_manager = IntraMonthRiskManager()

    # Layer 4: Pattern-based choppy market filter
    pattern_filter = PatternBasedFilter()
    current_month_key = None

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
    skip_stats = {'MONTH_STOPPED': 0, 'DAY_STOPPED': 0, 'DYNAMIC_ADJ': 0, 'PATTERN_STOPPED': 0, 'SESSION_POI': 0}

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

                # Layer 4: Record trade for pattern filter
                pattern_filter.record_trade(position.direction, pnl, current_time)

                position = None
            continue

        if current_time.weekday() >= 5:
            continue
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        hour = current_time.hour
        # v6.9: London 8-10, NY 13-17 (matching Live/MQL5)
        if not (8 <= hour <= 10 or 13 <= hour <= 17):
            continue
        session = "london" if 8 <= hour <= 10 else "newyork"

        # LAYER 3: Check intra-month risk manager
        can_trade, intra_month_adj, skip_reason = risk_manager.new_trade_check(current_time)
        if not can_trade:
            if 'MONTH' in skip_reason:
                skip_stats['MONTH_STOPPED'] += 1
            elif 'DAY' in skip_reason:
                skip_stats['DAY_STOPPED'] += 1
            continue

        # LAYER 4: Reset pattern filter if month changed
        month_key = (current_time.year, current_time.month)
        if month_key != current_month_key:
            current_month_key = month_key
            pattern_filter.reset_for_month(current_time.month)

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # QUAD-LAYER QUALITY: Assess market condition
        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)

        # Layer 1 + Layer 2 + Layer 3 (intra-month dynamic)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        if intra_month_adj > 0:
            skip_stats['DYNAMIC_ADJ'] += 1

        condition_stats[market_cond.label] = condition_stats.get(market_cond.label, 0) + 1

        # ============================================================
        # SIGNAL DETECTION: EMA_PULLBACK first, then ORDER_BLOCK
        # (EMA_PULLBACK generates ~74% of trades in MT5 backtest)
        # ============================================================
        signal = None
        signal_type = None

        # 1. Try EMA Pullback first (higher priority in v6.9)
        if hour not in SKIP_EMA_PULLBACK_HOURS:  # Skip EMA at hour 13, 14
            ema_signal = detect_ema_pullback(df, col_map, i, current_atr,
                                             ema20_series, rsi_series, adx_series,
                                             dynamic_quality)
            if ema_signal:
                # Check regime alignment
                if ema_signal['direction'] == 'BUY' and regime == Regime.BULLISH:
                    signal = ema_signal
                    signal_type = 'EMA_PULLBACK'
                elif ema_signal['direction'] == 'SELL' and regime == Regime.BEARISH:
                    signal = ema_signal
                    signal_type = 'EMA_PULLBACK'

        # 2. If no EMA signal, try Order Block
        if not signal and hour not in SKIP_ORDER_BLOCK_HOURS:  # Skip OB at hour 8, 16
            pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
            for poi in pois:
                if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                    continue
                if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                    continue

                zone_size = abs(current_bar[col_map['high']] - current_bar[col_map['low']]) * 2
                if abs(current_price - poi['price']) > zone_size:
                    continue

                signal = poi
                signal_type = 'ORDER_BLOCK'
                break

        if not signal:
            continue

        # Check entry trigger
        prev_bar = df.iloc[i-1]
        has_trigger, entry_type = check_entry_trigger(current_bar, prev_bar, signal['direction'], col_map)
        if not has_trigger:
            continue

        risk_mult, should_skip = calculate_risk_multiplier(current_time, entry_type, signal['quality'])
        if should_skip:
            continue

        # LAYER 4: Pattern-based filter check
        pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(signal['direction'])
        if not pattern_can_trade:
            skip_stats['PATTERN_STOPPED'] += 1
            continue

        # Apply pattern extra quality requirement
        if pattern_extra_q > 0:
            effective_quality = dynamic_quality + pattern_extra_q
            # Re-check if signal meets stricter quality
            if signal['quality'] < effective_quality:
                continue

        sl_pips = current_atr * SL_ATR_MULT
        tp_pips = sl_pips * TP_RATIO
        risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult * pattern_size_mult
        lot_size = risk_amount / (sl_pips * PIP_VALUE)
        lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

        if signal['direction'] == 'BUY':
            sl_price = current_price - (sl_pips * PIP_SIZE)
            tp_price = current_price + (tp_pips * PIP_SIZE)
        else:
            sl_price = current_price + (sl_pips * PIP_SIZE)
            tp_price = current_price - (tp_pips * PIP_SIZE)

        position = Trade(
            entry_time=current_time,
            direction=signal['direction'],
            entry_price=current_price,
            sl_price=sl_price,
            tp_price=tp_price,
            lot_size=lot_size,
            risk_amount=risk_amount,
            atr_pips=current_atr,
            quality_score=signal['quality'],
            entry_type=entry_type,
            signal_type=signal_type,
            session=session,
            dynamic_quality=dynamic_quality,
            market_condition=market_cond.label,
            monthly_adj=market_cond.monthly_adjustment + intra_month_adj
        )

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

            # Generate image
            generator = BacktestReportGenerator(output_dir="reports")
            image_path = generator.generate_telegram_summary_image(
                summary=summary,
                monthly_stats=monthly_stats,
                filename=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

            if image_path:
                # Send image with caption
                caption = (
                    f"<b>SURGE-WSI v6.9 GBPUSD Backtest</b>\n"
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
            msg += f"<b>H1 v6.9 GBPUSD - Quad-Layer Quality</b>\n"
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
    print(f"BACKTEST RESULTS - H1 v6.9 GBPUSD QUAD-LAYER QUALITY")
    print(f"{'='*70}")

    print(f"\n[QUAD-LAYER QUALITY CONFIGURATION]")
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
    print(f"  Combined = Layer1 + Layer2 + Layer3 + Layer4")

    if skip_stats:
        print(f"\n[PROTECTION STATS]")
        print(f"{'-'*50}")
        print(f"  Month circuit breaker: {skip_stats.get('MONTH_STOPPED', 0)}")
        print(f"  Day stop: {skip_stats.get('DAY_STOPPED', 0)}")
        print(f"  Dynamic quality adj: {skip_stats.get('DYNAMIC_ADJ', 0)}")
        print(f"  Pattern filter stops: {skip_stats.get('PATTERN_STOPPED', 0)}")
        print(f"  Session POI filter: {skip_stats.get('SESSION_POI', 0)}")

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

    print(f"SURGE-WSI H1 v6.9 GBPUSD - QUAD-LAYER QUALITY FILTER")
    print(f"{'='*70}")
    print(f"Quad-Layer Quality Filter:")
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

    print(f"\nRunning backtest with QUAD-LAYER quality filter...")
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
        'signal_type': t.signal_type,
        'session': t.session,
        'dynamic_quality': t.dynamic_quality,
        'market_condition': t.market_condition,
        'monthly_adj': t.monthly_adj
    } for t in trades])

    output_path = Path(__file__).parent.parent / "results" / "h1_v6_9_quad_filter_trades.csv"
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
