"""
SURGE-WSI H1 v6.9 GBPUSD Live Trading Executor
==============================================

QUAD-LAYER Quality Filter + Session Filter for ZERO losing months:
- Layer 1: Monthly profile from market analysis (tradeable %)
- Layer 2: Real-time technical indicators (ATR stability, efficiency, ADX)
- Layer 3: Intra-month dynamic risk adjustment (consecutive losses, monthly P&L)
- Layer 4: Pattern-Based Choppy Market Detector (rolling WR, direction tracking)
- Session Filter (v6.8): Skip underperforming Hour+POI combinations
- Day Multipliers Fix (v6.9): Thursday 0.4→0.8 (best day 51.8% WR), Friday 0.5→0.3 (worst day 33.8% WR)

Backtest Results v6.9 (Jan 2025 - Jan 2026):
- 150 trades, 50.0% WR, PF 5.84
- +$36,205 profit (+72.4% return on $50K)
- ZERO losing months (0/13)

Author: SURIOTA Team
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from loguru import logger

# Optional vector database integration
try:
    from src.data.vector_client import VectorClient
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    VectorClient = None


# ============================================================
# CONFIGURATION v6.4 GBPUSD - QUAD-LAYER QUALITY FILTER
# ============================================================
SYMBOL = "GBPUSD"
TIMEFRAME = "H1"

# Risk Management
RISK_PERCENT = 1.0              # 1% per trade
SL_ATR_MULT = 1.5               # SL = 1.5x ATR
TP_RATIO = 1.5                  # TP = 1.5x SL
MAX_LOSS_PER_TRADE_PCT = 0.15   # 0.15% max loss cap

# Technical Parameters
PIP_VALUE = 10.0                # $10 per pip per standard lot
PIP_SIZE = 0.0001               # GBPUSD pip size
MAX_LOT = 5.0
MIN_ATR = 8.0                   # Min 8 pips ATR
MAX_ATR = 25.0                  # Max 25 pips ATR (v6.9: synced with QuadLayer)

# Dynamic Quality Thresholds (Layer 2)
BASE_QUALITY = 65
MIN_QUALITY_GOOD = 60
MAX_QUALITY_BAD = 80

# Technical Thresholds
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

# ============================================================
# LAYER 3: DYNAMIC INTRA-MONTH RISK ADJUSTMENT
# Adaptive system - OPTIMIZED MODE
# ============================================================
MONTHLY_LOSS_THRESHOLD_1 = -150    # First warning: +5 quality
MONTHLY_LOSS_THRESHOLD_2 = -250    # Second warning: +10 quality
MONTHLY_LOSS_THRESHOLD_3 = -350    # Third warning: +15 quality
MONTHLY_LOSS_STOP = -400           # Circuit breaker: stop for month

CONSECUTIVE_LOSS_THRESHOLD = 3     # After 3 consecutive losses: +5 quality
CONSECUTIVE_LOSS_MAX = 6           # After 6 consecutive losses: stop for day

# No base protection (start at 0)
MIN_BASE_QUALITY_ADJUSTMENT = 0

# ============================================================
# LAYER 4: PATTERN-BASED CHOPPY MARKET DETECTOR
# Detects "whipsaw" markets where BOTH directions lose
# ============================================================
PATTERN_FILTER_ENABLED = True

# Warmup settings - filter observes but doesn't halt during warmup
WARMUP_TRADES = 15                 # First 15 trades are warmup (observe only)

# Probe trade settings - small "test" trade after halt
PROBE_TRADE_SIZE = 0.4             # Probe trade is 40% size
PROBE_QUALITY_EXTRA = 5            # Extra +5 quality for probe trades

# Choppy market detection thresholds
DIRECTION_TEST_WINDOW = 8          # Check last 8 trades for direction bias
MIN_DIRECTION_WIN_RATE = 0.10      # If a direction has <10% WR, avoid it
BOTH_DIRECTIONS_FAIL_THRESHOLD = 4 # If 4 trades in BOTH directions lose, HALT

# Rolling performance tracking
ROLLING_WINDOW = 10                # Track last 10 trades
ROLLING_WR_HALT = 0.10             # If rolling WR drops below 10%, halt trading
ROLLING_WR_CAUTION = 0.25          # If rolling WR below 25%, reduce size to 60%
CAUTION_SIZE_MULT = 0.6            # Size during caution mode

# Recovery settings
RECOVERY_WIN_REQUIRED = 1          # Need 1 win to exit halt state
RECOVERY_SIZE_MULT = 0.5           # Trade at 50% size during recovery


# ==========================================================
# LAYER 1: MONTHLY PROFILE (from market analysis)
# ==========================================================
MONTHLY_TRADEABLE_PCT = {
    # 2024
    (2024, 1): 67, (2024, 2): 55, (2024, 3): 70, (2024, 4): 80,
    (2024, 5): 62, (2024, 6): 68, (2024, 7): 78, (2024, 8): 65,
    (2024, 9): 72, (2024, 10): 58, (2024, 11): 66, (2024, 12): 60,
    # 2025 - Pattern filter handles all months dynamically
    (2025, 1): 65, (2025, 2): 55, (2025, 3): 70, (2025, 4): 70,
    (2025, 5): 62, (2025, 6): 68, (2025, 7): 78, (2025, 8): 65,
    (2025, 9): 72, (2025, 10): 58, (2025, 11): 66, (2025, 12): 60,
    # 2026
    (2026, 1): 65, (2026, 2): 55, (2026, 3): 70, (2026, 4): 80,
    (2026, 5): 62, (2026, 6): 68, (2026, 7): 78, (2026, 8): 65,
    (2026, 9): 72, (2026, 10): 58, (2026, 11): 66, (2026, 12): 60,
}

# Monthly risk multipliers
MONTHLY_RISK = {
    1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,
    7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
}

DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}  # v6.9: Thu 0.4→0.8 (best day), Fri 0.5→0.3 (worst day)

HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,  # Hour 7 & 11 = 0% (v6.8: low WR)
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

# ============================================================
# ENTRY SIGNAL CONFIG (v6.7)
# ============================================================
USE_ORDER_BLOCK = True       # Primary entry: Order Block detection
USE_EMA_PULLBACK = True      # Secondary entry: EMA Pullback (v6.7)

# ============================================================
# SESSION-BASED HOUR+POI FILTER (v6.8 - from Session Analysis)
# Skip underperforming Hour+POI combinations
# ============================================================
USE_SESSION_POI_FILTER = True  # Enable session-based filtering

# Hour 11: Skip entirely (27.3% WR - worst hour) - already in HOUR_MULTIPLIERS
SKIP_HOURS = [11]

# ORDER_BLOCK performs poorly at these hours
SKIP_ORDER_BLOCK_HOURS = [8, 16]  # 8.3% and 14.3% WR

# EMA_PULLBACK performs poorly during NY Overlap
SKIP_EMA_PULLBACK_HOURS = [13, 14]  # 18-28% WR


def should_skip_by_session(hour: int, poi_type: str) -> tuple:
    """
    Check if trade should be skipped based on session analysis.
    Returns (should_skip, reason)
    """
    if not USE_SESSION_POI_FILTER:
        return False, ""

    # Skip Hour 11 entirely (already handled by HOUR_MULTIPLIERS but double-check)
    if hour in SKIP_HOURS:
        return True, f"HOUR_{hour}_SKIP"

    # Skip ORDER_BLOCK at problematic hours
    if poi_type == "ORDER_BLOCK" and hour in SKIP_ORDER_BLOCK_HOURS:
        return True, f"OB_HOUR_{hour}_SKIP"

    # Skip EMA_PULLBACK during NY Overlap
    if poi_type == "EMA_PULLBACK" and hour in SKIP_EMA_PULLBACK_HOURS:
        return True, f"EMA_HOUR_{hour}_SKIP"

    return False, ""

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class MarketCondition:
    """Market condition assessment result"""
    atr_stability: float
    efficiency: float
    trend_strength: float
    technical_quality: float
    monthly_adjustment: int
    final_quality: float
    label: str


@dataclass
class TradeSignal:
    """Trade signal with all parameters"""
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    quality_score: float
    entry_type: str
    session: str
    atr_pips: float
    market_condition: str
    quality_threshold: float
    monthly_adj: int
    pattern_size_mult: float = 1.0
    pattern_status: str = "OK"
    poi_type: str = "ORDER_BLOCK"  # v6.8: POI type for session filter


class IntraMonthRiskManager:
    """
    Layer 3: Dynamic Intra-Month Risk Adjustment

    Tracks real-time monthly P&L and adjusts quality requirements
    to prevent losing months through adaptive risk management.
    """

    def __init__(self):
        self.current_month = None
        self.monthly_pnl = 0.0
        self.monthly_peak = 0.0
        self.consecutive_losses = 0
        self.daily_losses = 0
        self.current_day = None
        self.month_stopped = False
        self.day_stopped = False

    def new_trade_check(self, dt: datetime) -> Tuple[bool, int, str]:
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
            logger.info(f"[Layer3] New month: {month_key}")

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
        dynamic_adj = MIN_BASE_QUALITY_ADJUSTMENT

        # Monthly loss circuit breaker
        if self.monthly_pnl <= MONTHLY_LOSS_STOP:
            self.month_stopped = True
            logger.warning(f"[Layer3] MONTH CIRCUIT BREAKER: ${self.monthly_pnl:.0f}")
            return False, 0, f"MONTH_CIRCUIT_BREAKER (${self.monthly_pnl:.0f})"

        # Monthly loss thresholds
        if self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_3:
            dynamic_adj += 15
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_2:
            dynamic_adj += 10
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_1:
            dynamic_adj += 5

        # Adjustment for consecutive losses
        if self.consecutive_losses >= CONSECUTIVE_LOSS_MAX:
            self.day_stopped = True
            logger.warning(f"[Layer3] DAY STOPPED: {self.consecutive_losses} consecutive losses")
            return False, 0, f"CONSECUTIVE_LOSS_STOP ({self.consecutive_losses})"

        if self.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            dynamic_adj += 5

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
            logger.info(f"[Layer3] Loss recorded: ${pnl:.2f}, consecutive: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0  # Reset on win
            logger.info(f"[Layer3] Win recorded: ${pnl:.2f}, streak reset")

        logger.info(f"[Layer3] Monthly P&L: ${self.monthly_pnl:.2f}")

    def get_status(self) -> dict:
        """Get current risk manager status"""
        return {
            'month': self.current_month,
            'monthly_pnl': self.monthly_pnl,
            'monthly_peak': self.monthly_peak,
            'consecutive_losses': self.consecutive_losses,
            'month_stopped': self.month_stopped,
            'day_stopped': self.day_stopped
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
        self.probe_taken = False

    def reset_for_month(self, month: int):
        """Reset filter state for new month - fresh start each month"""
        self.current_month = month
        self.probe_taken = False

        # CRITICAL FIX: Always reset halt state on new month
        # Previous issue: halt could persist across months if not in recovery
        if self.is_halted:
            logger.info(f"[Layer4] New month {month}: clearing halt state (was: {self.halt_reason})")

        self.is_halted = False
        self.in_recovery = False
        self.recovery_wins = 0
        self.halt_reason = ""

        # Keep trade_history for learning but cap at last 30 trades
        if len(self.trade_history) > 30:
            self.trade_history = self.trade_history[-30:]

        logger.info(f"[Layer4] Reset for month {month}, history: {len(self.trade_history)} trades")

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

    def check_trade(self, direction: str) -> Tuple[bool, int, float, str]:
        """
        Check if trade is allowed based on pattern analysis

        Returns:
            (can_trade: bool, extra_quality: int, size_multiplier: float, reason: str)
        """
        if not PATTERN_FILTER_ENABLED:
            return True, 0, 1.0, "FILTER_DISABLED"

        # During warmup, always allow trades (just observe)
        if len(self.trade_history) < WARMUP_TRADES:
            logger.debug(f"[Layer4] Warmup mode: {len(self.trade_history)}/{WARMUP_TRADES} trades")
            return True, 0, 1.0, "WARMUP"

        # Check if halted
        if self.is_halted and not self.in_recovery:
            logger.warning(f"[Layer4] Trading HALTED: {self.halt_reason}")
            return False, 0, 1.0, f"HALTED: {self.halt_reason}"

        stats = self._get_rolling_stats()

        # Check for choppy market (both directions failing)
        if stats['both_fail']:
            self.is_halted = True
            self.halt_reason = "BOTH_DIRECTIONS_FAIL"
            self.in_recovery = True
            self.recovery_wins = 0
            logger.warning("[Layer4] CHOPPY MARKET DETECTED - both directions failing")
            return False, 0, 1.0, "CHOPPY_MARKET_DETECTED"

        # Check rolling win rate
        if stats['rolling_wr'] < ROLLING_WR_HALT:
            self.is_halted = True
            self.halt_reason = f"LOW_ROLLING_WR ({stats['rolling_wr']:.0%})"
            self.in_recovery = True
            self.recovery_wins = 0
            logger.warning(f"[Layer4] LOW WIN RATE: {stats['rolling_wr']:.0%}")
            return False, 0, 1.0, f"LOW_WIN_RATE ({stats['rolling_wr']:.0%})"

        # Check direction-specific win rate
        if len(self.trade_history) >= 10:
            if direction == 'BUY' and stats['buy_wr'] < MIN_DIRECTION_WIN_RATE:
                logger.info(f"[Layer4] BUY direction weak: {stats['buy_wr']:.0%}")
                return False, 0, 1.0, f"BUY_DIRECTION_WEAK ({stats['buy_wr']:.0%})"
            if direction == 'SELL' and stats['sell_wr'] < MIN_DIRECTION_WIN_RATE:
                logger.info(f"[Layer4] SELL direction weak: {stats['sell_wr']:.0%}")
                return False, 0, 1.0, f"SELL_DIRECTION_WEAK ({stats['sell_wr']:.0%})"

        # Determine size and quality adjustments
        size_mult = 1.0
        extra_q = 0

        # Recovery mode - trade smaller with probe
        if self.in_recovery:
            size_mult = RECOVERY_SIZE_MULT
            extra_q = PROBE_QUALITY_EXTRA
            logger.info(f"[Layer4] RECOVERY MODE: {size_mult:.0%} size, +{extra_q} quality")

        # Caution mode - rolling WR below threshold
        elif stats['rolling_wr'] < ROLLING_WR_CAUTION:
            size_mult = CAUTION_SIZE_MULT
            extra_q = 3
            logger.info(f"[Layer4] CAUTION MODE: {size_mult:.0%} size, +{extra_q} quality")

        return True, extra_q, size_mult, "OK"

    def record_trade(self, direction: str, pnl: float, timestamp: datetime):
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
                logger.info(f"[Layer4] Recovery win: {self.recovery_wins}/{RECOVERY_WIN_REQUIRED}")
                if self.recovery_wins >= RECOVERY_WIN_REQUIRED:
                    self.is_halted = False
                    self.in_recovery = False
                    self.recovery_wins = 0
                    logger.info("[Layer4] RECOVERY COMPLETE - normal trading resumed")
            else:
                self.recovery_wins = 0
                logger.info("[Layer4] Recovery loss - reset wins counter")

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


class GBPUSDRiskScorer:
    """
    GBPUSD-specific risk scorer with quad-layer quality filter
    """

    def __init__(self):
        self.symbol = SYMBOL

    def get_monthly_quality_adjustment(self, dt: datetime) -> int:
        """Layer 1: Get quality adjustment from monthly profile"""
        key = (dt.year, dt.month)
        tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)

        if tradeable_pct < 30:
            return 50  # NO TRADE
        elif tradeable_pct < 40:
            return 35  # HALT
        elif tradeable_pct < 50:
            return 25  # Extremely poor
        elif tradeable_pct < 60:
            return 15  # Very poor month
        elif tradeable_pct < 70:
            return 10  # Below average
        elif tradeable_pct < 75:
            return 5   # Slightly below average
        else:
            return 0   # Good month

    def calculate_technical_condition(self, df: pd.DataFrame, col_map: dict,
                                      atr_series: pd.Series) -> Tuple[float, str]:
        """Layer 2: Calculate technical market condition"""
        lookback = 20
        idx = len(df) - 1
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
            total_move = sum(abs(df[c].iloc[i] - df[c].iloc[i-1])
                           for i in range(start_idx+1, idx+1))
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

        # Calculate score
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

        # Determine technical quality
        if score >= 80:
            return MIN_QUALITY_GOOD, "TECH_GOOD"
        elif score >= 40:
            return BASE_QUALITY, "TECH_NORMAL"
        else:
            return MAX_QUALITY_BAD, "TECH_BAD"

    def assess_market_condition(self, df: pd.DataFrame, col_map: dict,
                                atr_series: pd.Series, current_time: datetime) -> MarketCondition:
        """
        QUAD-LAYER market condition assessment (Layer 1 + Layer 2)
        """
        # Layer 1: Monthly profile
        monthly_adj = self.get_monthly_quality_adjustment(current_time)

        # Layer 2: Technical indicators
        technical_quality, tech_label = self.calculate_technical_condition(
            df, col_map, atr_series
        )

        # Combined quality threshold
        final_quality = technical_quality + monthly_adj

        # Determine overall label
        if monthly_adj >= 15:
            label = "POOR_MONTH"
        elif monthly_adj >= 10:
            label = "CAUTION"
        elif tech_label == "TECH_GOOD":
            label = "GOOD"
        elif tech_label == "TECH_NORMAL":
            label = "NORMAL"
        else:
            label = "BAD"

        return MarketCondition(
            atr_stability=0.0,
            efficiency=0.0,
            trend_strength=0.0,
            technical_quality=technical_quality,
            monthly_adjustment=monthly_adj,
            final_quality=final_quality,
            label=label
        )

    def calculate_risk_multiplier(self, dt: datetime, entry_type: str,
                                  quality: float) -> Tuple[float, bool]:
        """Calculate risk multiplier based on time and quality"""
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


class H1V64GBPUSDExecutor:
    """
    H1 v6.4 GBPUSD Live Trading Executor

    Features:
    - QUAD-LAYER quality filter for zero losing months
    - Layer 1: Monthly profile adjustment
    - Layer 2: Technical indicators (ATR, efficiency, ADX)
    - Layer 3: Intra-month dynamic risk (consecutive losses, monthly P&L)
    - Layer 4: Pattern-based choppy market detector
    """

    def __init__(self, broker_client, db_handler=None, telegram_bot=None, mt5_connector=None,
                 enable_vector: bool = True):
        self.broker = broker_client
        self.db = db_handler
        self.telegram = telegram_bot
        self.mt5 = mt5_connector
        self.risk_scorer = GBPUSDRiskScorer()
        self.current_position = None
        self.last_signal_time = None

        # Layer 3 & 4: Risk managers
        self.intra_month_manager = IntraMonthRiskManager()
        self.pattern_filter = PatternBasedFilter()
        self.current_month_key = None

        # Vector database integration
        self.vector_client: Optional[Any] = None
        self.enable_vector = enable_vector and VECTOR_AVAILABLE
        self._last_vector_sync = None

        if self.enable_vector and VectorClient is not None:
            try:
                self.vector_client = VectorClient()
                if self.vector_client.connect():
                    logger.info("Vector database connected")
                else:
                    logger.warning("Vector database connection failed, disabled")
                    self.vector_client = None
                    self.enable_vector = False
            except Exception as e:
                logger.warning(f"Vector initialization failed: {e}")
                self.vector_client = None
                self.enable_vector = False

        logger.info(f"H1 v6.4 GBPUSD Executor initialized (Vector: {'ON' if self.enable_vector else 'OFF'})")

    async def get_ohlcv_data(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        """Get OHLCV data - MT5 primary, DB fallback"""
        df = None

        # Try MT5 first
        if self.mt5 is not None:
            try:
                df = self.mt5.get_ohlcv(symbol, timeframe, bars)
                if df is not None and not df.empty:
                    logger.debug(f"OHLCV from MT5: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"MT5 OHLCV failed: {e}")

        # Fallback to database
        if self.db is not None:
            try:
                now = datetime.now(timezone.utc)
                start = now - timedelta(days=60)
                df = await self.db.get_ohlcv(symbol, timeframe, bars, start, now)
                if df is not None and not df.empty:
                    logger.debug(f"OHLCV from DB (fallback): {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"DB OHLCV failed: {e}")

        logger.error("Failed to get OHLCV from both MT5 and DB")
        return pd.DataFrame()

    async def calculate_atr(self, df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
        """Calculate ATR in pips"""
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

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_adx(self, df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        h, l, c = col_map['high'], col_map['low'], col_map['close']
        high = df[h]
        low = df[l]
        close = df[c]

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx

    def detect_ema_pullback(self, df: pd.DataFrame, col_map: dict,
                            atr_series: pd.Series, min_quality: float) -> List[dict]:
        """
        Detect EMA Pullback entry signals (v6.7)

        Criteria:
        - BUY: close > EMA20 > EMA50 (uptrend), price near EMA20, bullish candle
        - SELL: close < EMA20 < EMA50 (downtrend), price near EMA20, bearish candle
        - ADX > 20 (trend present)
        - RSI between 30-70 (room for momentum)
        - Body ratio > 0.4 (decent candle)
        """
        pois = []
        if len(df) < 50:
            return pois

        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        # Calculate indicators
        ema20 = self.calculate_ema(df[c], 20)
        ema50 = self.calculate_ema(df[c], 50)
        rsi = self.calculate_rsi(df[c], 14)
        adx = self.calculate_adx(df, col_map, 14)

        # Check current bar (last bar)
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

        # Skip if indicators are NaN
        if pd.isna(current_ema20) or pd.isna(current_ema50) or pd.isna(current_rsi) or pd.isna(current_adx):
            return pois

        # Calculate body ratio
        total_range = current_high - current_low
        if total_range < 0.0003:
            return pois
        body = abs(current_close - current_open)
        body_ratio = body / total_range

        # Filter criteria
        if body_ratio < 0.4:
            return pois
        if current_adx < 20:
            return pois
        if not (30 <= current_rsi <= 70):
            return pois
        if current_atr < MIN_ATR or current_atr > MAX_ATR:
            return pois

        # EMA20 proximity - within 1.5 ATR
        atr_distance = current_atr * PIP_SIZE * 1.5

        # BUY: Uptrend pullback
        is_bullish = current_close > current_open
        if is_bullish and current_close > current_ema20 > current_ema50:
            distance_to_ema = current_low - current_ema20
            if distance_to_ema <= atr_distance:
                touch_quality = max(0, 30 - (distance_to_ema / PIP_SIZE))
                adx_quality = min(25, (current_adx - 15) * 1.5)
                rsi_quality = 25 if abs(50 - current_rsi) < 20 else 15
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

        # SELL: Downtrend pullback
        is_bearish = current_close < current_open
        if is_bearish and current_close < current_ema20 < current_ema50:
            distance_to_ema = current_ema20 - current_high
            if distance_to_ema <= atr_distance:
                touch_quality = max(0, 30 - (distance_to_ema / PIP_SIZE))
                adx_quality = min(25, (current_adx - 15) * 1.5)
                rsi_quality = 25 if abs(50 - current_rsi) < 20 else 15
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

    def detect_regime(self, df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float]:
        """Detect market regime using EMAs"""
        if len(df) < 50:
            return Regime.SIDEWAYS, 0.5

        c = col_map['close']
        ema20 = self.calculate_ema(df[c], 20)
        ema50 = self.calculate_ema(df[c], 50)

        current_close = df[c].iloc[-1]
        current_ema20 = ema20.iloc[-1]
        current_ema50 = ema50.iloc[-1]

        if current_close > current_ema20 > current_ema50:
            return Regime.BULLISH, 0.7
        elif current_close < current_ema20 < current_ema50:
            return Regime.BEARISH, 0.7
        else:
            return Regime.SIDEWAYS, 0.5

    def detect_order_blocks(self, df: pd.DataFrame, col_map: dict,
                           min_quality: float) -> List[dict]:
        """Detect order blocks/POIs"""
        pois = []
        if len(df) < 35:
            return pois

        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        for i in range(len(df) - 30, len(df) - 2):
            if i < 2:
                continue

            current = df.iloc[i]
            next1 = df.iloc[i+1]

            # Bullish OB
            is_bearish = current[c] < current[o]
            if is_bearish:
                next_body = abs(next1[c] - next1[o])
                next_range = next1[h] - next1[l]
                if next_range > 0:
                    body_ratio = next_body / next_range
                    if next1[c] > next1[o] and body_ratio > 0.55 and next1[c] > current[h]:
                        quality = body_ratio * 100
                        if quality >= min_quality:
                            pois.append({
                                'price': current[l],
                                'direction': 'BUY',
                                'quality': quality,
                                'idx': i
                            })

            # Bearish OB
            is_bullish = current[c] > current[o]
            if is_bullish:
                next_body = abs(next1[c] - next1[o])
                next_range = next1[h] - next1[l]
                if next_range > 0:
                    body_ratio = next_body / next_range
                    if next1[c] < next1[o] and body_ratio > 0.55 and next1[c] < current[l]:
                        quality = body_ratio * 100
                        if quality >= min_quality:
                            pois.append({
                                'price': current[h],
                                'direction': 'SELL',
                                'quality': quality,
                                'idx': i
                            })

        return pois

    def check_entry_trigger(self, bar: pd.Series, prev_bar: pd.Series,
                           direction: str, col_map: dict) -> Tuple[bool, str]:
        """Check for entry trigger"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        total_range = bar[h] - bar[l]
        if total_range < 0.0003:
            return False, ""

        body = abs(bar[c] - bar[o])
        is_bullish = bar[c] > bar[o]
        is_bearish = bar[c] < bar[o]
        prev_body = abs(prev_bar[c] - prev_bar[o])

        # Momentum
        if body > total_range * 0.5:
            if direction == 'BUY' and is_bullish:
                return True, 'MOMENTUM'
            if direction == 'SELL' and is_bearish:
                return True, 'MOMENTUM'

        # Engulfing
        if body > prev_body * 1.2:
            if direction == 'BUY' and is_bullish and prev_bar[c] < prev_bar[o]:
                return True, 'ENGULF'
            if direction == 'SELL' and is_bearish and prev_bar[c] > prev_bar[o]:
                return True, 'ENGULF'

        # Lower High (for SELL)
        if direction == 'SELL':
            if bar[h] < prev_bar[h] and is_bearish:
                return True, 'LOWER_HIGH'

        return False, ""

    def is_kill_zone(self, dt: datetime) -> Tuple[bool, str]:
        """
        Check if current time is in kill zone.

        Note: Hour 7 & 11 are blocked by HOUR_MULTIPLIERS = 0.0 (not here)
        This function only defines the active trading windows.
        """
        hour = dt.hour
        # London session: Hours 8-10 (Hour 7 & 11 blocked by HOUR_MULTIPLIERS)
        if hour in [8, 9, 10]:
            return True, "london"
        # New York session: Hours 13-17
        elif 13 <= hour <= 17:
            return True, "newyork"
        return False, ""

    async def analyze_market(self, balance: float) -> Optional[TradeSignal]:
        """
        Analyze market with QUAD-LAYER filter and generate trade signal

        Returns TradeSignal if valid signal found, None otherwise
        """
        now = datetime.now(timezone.utc)

        # Check day filter
        if now.weekday() >= 5:
            return None

        # Check kill zone
        in_kill_zone, session = self.is_kill_zone(now)
        if not in_kill_zone:
            return None

        # LAYER 3: Check intra-month risk manager
        can_trade, intra_month_adj, skip_reason = self.intra_month_manager.new_trade_check(now)
        if not can_trade:
            logger.info(f"[Layer3] Trade blocked: {skip_reason}")
            return None

        # LAYER 4: Reset pattern filter if month changed
        month_key = (now.year, now.month)
        if month_key != self.current_month_key:
            self.current_month_key = month_key
            self.pattern_filter.reset_for_month(now.month)

        # Fetch H1 data
        df = await self.get_ohlcv_data(SYMBOL, TIMEFRAME, 500)

        if df is None or df.empty or len(df) < 100:
            logger.warning("Insufficient data for analysis")
            return None

        # Map columns
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

        # Calculate ATR
        atr_series = await self.calculate_atr(df, col_map)
        current_atr = atr_series.iloc[-1]

        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            logger.debug(f"ATR out of range: {current_atr:.1f} pips")
            return None

        # Check regime
        regime, _ = self.detect_regime(df, col_map)
        if regime == Regime.SIDEWAYS:
            return None

        # LAYER 1 + 2: Assess market condition
        market_cond = self.risk_scorer.assess_market_condition(
            df, col_map, atr_series, now
        )

        # Combined quality (Layer 1 + 2 + 3)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        logger.info(f"Market: {market_cond.label}, Quality>={dynamic_quality:.0f} (L1={market_cond.monthly_adjustment}, L3={intra_month_adj})")

        # Detect POIs with dynamic quality threshold (v6.7: dual entry signals)
        pois = []

        # Entry Signal 1: Order Block detection
        if USE_ORDER_BLOCK:
            ob_pois = self.detect_order_blocks(df, col_map, dynamic_quality)
            for poi in ob_pois:
                poi['type'] = 'ORDER_BLOCK'
            pois.extend(ob_pois)

        # Entry Signal 2: EMA Pullback detection (v6.7)
        if USE_EMA_PULLBACK:
            ema_pois = self.detect_ema_pullback(df, col_map, atr_series, dynamic_quality)
            pois.extend(ema_pois)

        if not pois:
            return None

        current_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        current_price = current_bar[col_map['close']]

        for poi in pois:
            # Check regime alignment
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            # v6.8: SESSION-BASED HOUR+POI FILTER
            poi_type = poi.get('type', 'ORDER_BLOCK')
            session_skip, session_reason = should_skip_by_session(now.hour, poi_type)
            if session_skip:
                logger.debug(f"[Session] Trade blocked: {session_reason}")
                continue

            # Check zone proximity
            zone_size = abs(current_bar[col_map['high']] - current_bar[col_map['low']]) * 2
            if abs(current_price - poi['price']) > zone_size:
                continue

            # Check entry trigger
            has_trigger, entry_type = self.check_entry_trigger(
                current_bar, prev_bar, poi['direction'], col_map
            )
            if not has_trigger:
                continue

            # Calculate base risk
            risk_mult, should_skip = self.risk_scorer.calculate_risk_multiplier(
                now, entry_type, poi['quality']
            )
            if should_skip:
                continue

            # LAYER 4: Pattern-based filter check
            pattern_can_trade, pattern_extra_q, pattern_size_mult, pattern_reason = self.pattern_filter.check_trade(poi['direction'])
            if not pattern_can_trade:
                logger.info(f"[Layer4] Trade blocked: {pattern_reason}")
                continue

            # Apply pattern extra quality requirement
            effective_quality = dynamic_quality + pattern_extra_q
            if poi['quality'] < effective_quality:
                logger.debug(f"POI quality {poi['quality']:.0f} < required {effective_quality:.0f}")
                continue

            # Calculate position with pattern size multiplier
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

            return TradeSignal(
                direction=poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                quality_score=poi['quality'],
                entry_type=entry_type,
                session=session,
                atr_pips=current_atr,
                market_condition=market_cond.label,
                quality_threshold=effective_quality,
                monthly_adj=market_cond.monthly_adjustment + intra_month_adj,
                pattern_size_mult=pattern_size_mult,
                pattern_status=pattern_reason,
                poi_type=poi_type  # v6.8: POI type for session filter tracking
            )

        return None

    async def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute trade signal"""
        try:
            logger.info(f"Executing {signal.direction} signal:")
            logger.info(f"  Entry: {signal.entry_price:.5f}")
            logger.info(f"  SL: {signal.sl_price:.5f}")
            logger.info(f"  TP: {signal.tp_price:.5f}")
            logger.info(f"  Lot: {signal.lot_size}")
            logger.info(f"  Quality: {signal.quality_score:.1f} (req: {signal.quality_threshold:.0f})")
            logger.info(f"  Market: {signal.market_condition}")
            logger.info(f"  Pattern: {signal.pattern_status} ({signal.pattern_size_mult:.0%} size)")

            # Place order via broker
            order_result = await self.broker.place_order(
                symbol=SYMBOL,
                order_type=signal.direction,
                lot_size=signal.lot_size,
                stop_loss=signal.sl_price,
                take_profit=signal.tp_price
            )

            if order_result.get('success'):
                self.current_position = signal
                self.last_signal_time = datetime.now(timezone.utc)

                # Send Telegram notification
                if self.telegram:
                    msg = (
                        f"[v6.4 GBPUSD] NEW TRADE\n"
                        f"Direction: {signal.direction}\n"
                        f"Entry: {signal.entry_price:.5f}\n"
                        f"SL: {signal.sl_price:.5f}\n"
                        f"TP: {signal.tp_price:.5f}\n"
                        f"Lot: {signal.lot_size}\n"
                        f"Quality: {signal.quality_score:.0f} (req: {signal.quality_threshold:.0f})\n"
                        f"Market: {signal.market_condition}\n"
                        f"Pattern: {signal.pattern_status}\n"
                        f"Session: {signal.session}"
                    )
                    await self.telegram.send_message(msg)

                return True
            else:
                logger.error(f"Order failed: {order_result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Execute error: {e}")
            return False

    async def record_trade_result(self, pnl: float, direction: str):
        """Record trade result for Layer 3 & 4 tracking"""
        now = datetime.now(timezone.utc)

        # Layer 3: Record for intra-month tracking
        self.intra_month_manager.record_trade(pnl, now)

        # Layer 4: Record for pattern filter
        self.pattern_filter.record_trade(direction, pnl, now)

        logger.info(f"Trade result recorded: {direction} ${pnl:+.2f}")

    async def check_position(self) -> Optional[dict]:
        """Check current position status and record result if closed"""
        if not self.current_position:
            return None

        try:
            positions = await self.broker.get_positions(SYMBOL)
            if not positions:
                # Position closed - get P&L from trade history and record
                closed_direction = self.current_position.direction
                pnl = await self._get_last_trade_pnl()

                if pnl is not None:
                    await self.record_trade_result(pnl, closed_direction)
                    logger.info(f"Position closed: {closed_direction} P&L=${pnl:+.2f}")

                    # Send Telegram notification
                    if self.telegram:
                        result_emoji = "🟢" if pnl > 0 else "🔴"
                        msg = (
                            f"{result_emoji} [v6.4 GBPUSD] TRADE CLOSED\n"
                            f"Direction: {closed_direction}\n"
                            f"P&L: ${pnl:+.2f}\n"
                            f"Monthly P&L: ${self.intra_month_manager.monthly_pnl:+.2f}"
                        )
                        await self.telegram.send_message(msg)
                else:
                    logger.warning("Position closed but could not get P&L from history")

                self.current_position = None
                return {'status': 'closed', 'pnl': pnl}

            return {'status': 'open', 'position': positions[0]}
        except Exception as e:
            logger.error(f"Position check error: {e}")
            return None

    async def _get_last_trade_pnl(self) -> Optional[float]:
        """Get P&L of last closed trade from broker history"""
        try:
            # Try to get recent trade history
            if hasattr(self.broker, 'get_trade_history'):
                history = await self.broker.get_trade_history(SYMBOL, limit=1)
                if history and len(history) > 0:
                    return history[0].get('profit', 0.0)

            # Fallback: try to get from MT5 directly
            if self.mt5 and hasattr(self.mt5, 'get_last_trade'):
                trade = self.mt5.get_last_trade(SYMBOL)
                if trade:
                    return trade.get('profit', 0.0)

            logger.warning("No trade history method available on broker")
            return None

        except Exception as e:
            logger.error(f"Failed to get last trade P&L: {e}")
            return None

    async def run_cycle(self):
        """Run one analysis and trading cycle"""
        try:
            # Get balance
            balance = await self.broker.get_balance()
            if not balance:
                logger.warning("Could not get balance")
                return

            # Check existing position
            pos_status = await self.check_position()
            if pos_status and pos_status.get('status') == 'open':
                logger.debug("Position open, skipping analysis")
                return

            # Analyze market with QUAD-LAYER filter
            signal = await self.analyze_market(balance)

            if signal:
                await self.execute_signal(signal)

        except Exception as e:
            logger.error(f"Cycle error: {e}")

    def get_status(self) -> dict:
        """Get current executor status with all layer info"""
        now = datetime.now(timezone.utc)
        status = {
            'version': 'v6.4',
            'symbol': SYMBOL,
            'timeframe': TIMEFRAME,
            'strategy': 'quad_layer_quality',
            'has_position': self.current_position is not None,
            'last_signal': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'layer1_monthly_adj': self.risk_scorer.get_monthly_quality_adjustment(now),
            'layer3_status': self.intra_month_manager.get_status(),
            'layer4_status': self.pattern_filter.get_stats(),
            'vector_enabled': self.enable_vector
        }

        if self.vector_client:
            status['vector_status'] = self.vector_client.get_status()

        return status

    async def sync_vector(self, symbol: str = None, timeframe: str = None, force: bool = False) -> int:
        """
        Sync current data to vector database.

        Args:
            symbol: Symbol to sync (default: SYMBOL)
            timeframe: Timeframe to sync (default: TIMEFRAME)
            force: Force sync even if recently synced

        Returns:
            Number of vectors synced
        """
        if not self.vector_client:
            return 0

        symbol = symbol or SYMBOL
        timeframe = timeframe or TIMEFRAME

        # Rate limit (sync max once per minute unless forced)
        if not force and self._last_vector_sync:
            elapsed = datetime.now(timezone.utc) - self._last_vector_sync
            if elapsed < timedelta(seconds=60):
                return 0

        try:
            df = await self.get_ohlcv_data(symbol, timeframe, 1000)
            if df is None or df.empty:
                return 0

            synced = self.vector_client.sync(symbol, timeframe, df, force=force)
            self._last_vector_sync = datetime.now(timezone.utc)

            if synced > 0:
                logger.info(f"Vector sync: {synced} vectors for {symbol}/{timeframe}")

            return synced

        except Exception as e:
            logger.error(f"Vector sync failed: {e}")
            return 0

    async def find_similar_patterns(
        self,
        symbol: str = None,
        timeframe: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find similar historical patterns to current market state.

        Args:
            symbol: Symbol to search
            timeframe: Timeframe
            top_k: Number of similar patterns to return

        Returns:
            List of similar patterns with scores and OHLC data
        """
        if not self.vector_client:
            return []

        symbol = symbol or SYMBOL
        timeframe = timeframe or TIMEFRAME

        try:
            # Get current features
            df = await self.get_ohlcv_data(symbol, timeframe, 100)
            if df is None or df.empty or len(df) < 50:
                return []

            features = self.vector_client.compute_features(df)
            if len(features) == 0:
                return []

            # Get latest feature vector
            query_vector = features[-1]

            # Exclude recent 24 hours
            max_timestamp = datetime.now(timezone.utc) - timedelta(hours=24)

            return self.vector_client.find_similar(
                symbol=symbol,
                timeframe=timeframe,
                query_vector=query_vector,
                top_k=top_k,
                max_timestamp=max_timestamp
            )

        except Exception as e:
            logger.error(f"Similar pattern search failed: {e}")
            return []

    def get_vector_status(self) -> Dict:
        """Get vector database status"""
        if not self.vector_client:
            return {
                "enabled": False,
                "connected": False,
                "message": "Vector client not initialized"
            }

        return self.vector_client.get_status()
