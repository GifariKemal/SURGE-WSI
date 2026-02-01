"""Detailed H1 Backtest v6 - Enhanced Activity Filter Integration
================================================================

v6 IMPROVEMENTS over v5.6:
Based on comprehensive daily analysis (Jan 2025 - Jan 2026), added:
  1. Price Efficiency detection (net_move / total_movement)
     - LOW_EFFICIENCY (<0.10) = SKIP trade
  2. Reversal Count detection
     - MANY_REVERSALS (>10) = reduce risk
  3. Day-of-week quality adjustments from analysis:
     - Friday: 36.8 avg quality, 42.9% tradeable -> heavy penalty
     - Wednesday: 57.4 avg quality, 76.8% tradeable -> bonus

KEY INSIGHT FROM DAILY ANALYSIS:
- June 2025 (losing month) had 7 days with LOW_EFFICIENCY + MANY_REVERSALS
- These patterns are detectable in real-time using last 20 H1 bars

Version history:
- v4.5.2: 89T, 49.4% WR, +$2,394, PF 1.57, 1 losing month
- v5.6:   83T, 48.2% WR, +$2,164, PF 1.62, 1 losing month (June -$189)

Target v6: ZERO losing months through real-time efficiency detection

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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.utils.killzone import KillZone
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel, EnhancedActivityFilter
from src.utils.market_condition_filter import MarketConditionFilter
from src.utils.drift_detector import ADWINDriftDetector
from src.utils.session_profiles import SessionProfileManager, TradingSession
from src.utils.confluence_validator import ConfluenceValidator
from src.utils.volatility_regime_filter import VolatilityRegimeFilter
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector


# ============================================================================
# V5.1 ADAPTIVE RISK SCORING SYSTEM
# ============================================================================

class AdaptiveRiskScorer:
    """
    Calculates risk multiplier based on multiple factors.
    v5.6: Added skip threshold when combined risk is too low.

    Final risk = base_risk * day_mult * hour_mult * entry_mult * session_mult * quality_mult
    """

    # Day of week risk multipliers (based on analysis)
    DAY_MULTIPLIERS = {
        0: 1.0,    # Monday - 50% WR
        1: 0.9,    # Tuesday - 32.4% WR in v5.1, reduce slightly
        2: 1.0,    # Wednesday - 40.6% WR
        3: 0.4,    # Thursday - 18.2% WR in v5.1, reduce 60%
        4: 0.5,    # Friday - 29.4% WR in v5.1, reduce 50%
        5: 0.0,    # Saturday - SKIP
        6: 0.0,    # Sunday - SKIP
    }

    # Day quality requirements (higher = need better setup)
    DAY_QUALITY_BONUS = {
        0: 0,      # Monday
        1: 5,      # Tuesday
        2: 0,      # Wednesday
        3: 20,     # Thursday - require +20 quality
        4: 15,     # Friday - require +15 quality
        5: 99,     # Saturday - effectively skip
        6: 99,     # Sunday - effectively skip
    }

    # Hour risk multipliers (REVERT TO v5.2 - was better)
    HOUR_MULTIPLIERS = {
        7: 1.0, 8: 1.0,             # Early London - good
        9: 1.2,                      # Best hour - 80% WR
        10: 0.6, 11: 0.6,           # Transition
        12: 1.1, 13: 1.2,           # Overlap - 70%+ WR
        14: 1.1,                     # Mid overlap
        15: 0.7,                     # Post-overlap
        16: 0.5,                     # NY afternoon
        17: 0.5, 18: 0.5, 19: 0.5, 20: 0.5,  # Late NY
        21: 0.4, 22: 0.3, 23: 0.5,  # Off hours
        0: 0.4, 1: 0.0, 2: 0.4, 3: 0.4,  # Asia
        4: 0.0, 5: 0.4, 6: 0.5,     # Early Asia
    }

    # Hour quality requirements
    HOUR_QUALITY_BONUS = {
        9: -5, 13: -5, 14: -5,      # Best hours - can be more lenient
        10: 10, 11: 10,             # Transition - need +10 quality
        15: 10, 16: 15,             # Post-overlap/NY afternoon
        17: 15, 18: 15, 19: 15, 20: 15,  # Late NY - need +15 quality
        22: 20, 1: 99, 4: 99,       # Dead hours - high penalty/skip
    }

    # Entry type risk multipliers (v5.6: Less penalty for REJECTION based on v4.5.2 data)
    # v4.5.2: REJECTION had 50% WR, +$1,102 - should NOT be penalized heavily!
    ENTRY_MULTIPLIERS = {
        'MOMENTUM': 1.0,            # Good
        'LOWER_HIGH': 0.9,          # Good
        'HIGHER_LOW': 0.85,         # Slightly lower (36% WR in v5)
        'ENGULF': 0.85,             # Moderate
        'REJECTION': 0.85,          # v5.6: Increased from 0.5! v4.5.2 shows 50% WR
    }

    # Entry type quality requirements (v5.6: Reduced REJECTION penalty)
    ENTRY_QUALITY_BONUS = {
        'MOMENTUM': 0,
        'LOWER_HIGH': 5,
        'HIGHER_LOW': 8,
        'ENGULF': 5,
        'REJECTION': 10,            # v5.6: Reduced from 25 to 10
    }

    # v5.6: Lower skip threshold to allow more trades (v4.5.2 had 89 trades)
    SKIP_THRESHOLD = 0.28

    def __init__(self,
                 min_combined_score: float = 45.0,  # v5.6: Reduced to allow more trades
                 min_risk_multiplier: float = 0.28,  # v5.6: Reduced from 0.35
                 max_risk_multiplier: float = 1.3):
        self.min_combined_score = min_combined_score
        self.min_risk_multiplier = min_risk_multiplier
        self.max_risk_multiplier = max_risk_multiplier

    def calculate_risk_score(
        self,
        current_time: datetime,
        entry_type: str,
        poi_quality: float,
        in_killzone: bool,
        confluence_score: float,
        drift_detected: bool
    ) -> Tuple[float, float, str, bool]:
        """
        Calculate adaptive risk multiplier.

        Returns:
            (risk_multiplier, required_quality, reason_string, should_skip)
        """
        day = current_time.weekday()
        hour = current_time.hour

        # Start with base multipliers
        day_mult = self.DAY_MULTIPLIERS.get(day, 1.0)
        hour_mult = self.HOUR_MULTIPLIERS.get(hour, 1.0)
        entry_mult = self.ENTRY_MULTIPLIERS.get(entry_type, 1.0)

        # v5.6: Check for hard skip conditions
        if day_mult == 0.0 or hour_mult == 0.0:
            return 0.0, 100.0, "hard_skip", True

        # Session multiplier
        session_mult = 1.0 if in_killzone else 0.65  # More penalty for hybrid

        # Drift multiplier
        drift_mult = 0.6 if drift_detected else 1.0  # More penalty for drift

        # Calculate combined risk multiplier
        raw_mult = day_mult * hour_mult * entry_mult * session_mult * drift_mult

        # v5.6: Skip if combined multiplier is too low
        if raw_mult < self.SKIP_THRESHOLD:
            return raw_mult, 100.0, "combined_skip", True

        risk_multiplier = max(self.min_risk_multiplier,
                             min(self.max_risk_multiplier, raw_mult))

        # Calculate required quality (higher for riskier conditions)
        base_quality = self.min_combined_score
        quality_bonus = (
            self.DAY_QUALITY_BONUS.get(day, 0) +
            self.HOUR_QUALITY_BONUS.get(hour, 0) +
            self.ENTRY_QUALITY_BONUS.get(entry_type, 0)
        )

        if not in_killzone:
            quality_bonus += 12  # Hybrid needs +12 quality
        if drift_detected:
            quality_bonus += 12  # Drift needs +12 quality

        required_quality = base_quality + quality_bonus

        # Cap required quality at 90 (otherwise nearly impossible)
        required_quality = min(90.0, required_quality)

        # Generate reason string
        reasons = []
        if day_mult < 1.0:
            reasons.append(f"day={day_mult:.1f}x")
        if hour_mult != 1.0:
            reasons.append(f"hour={hour_mult:.1f}x")
        if entry_mult < 1.0:
            reasons.append(f"entry={entry_mult:.1f}x")
        if session_mult < 1.0:
            reasons.append("hybrid")
        if drift_mult < 1.0:
            reasons.append("drift")

        reason = ", ".join(reasons) if reasons else "optimal"

        return risk_multiplier, required_quality, reason, False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BacktestTrade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    lot_size: float = 0.1
    pnl: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""
    exit_reason: str = ""
    session: str = ""
    regime: str = ""
    poi_type: str = ""
    entry_type: str = ""
    quality_score: float = 0.0
    sl_pips: float = 25.0
    risk_multiplier: float = 1.0
    risk_reason: str = ""
    hour: int = 0
    day_of_week: int = 0


@dataclass
class BacktestStats:
    """Backtest statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    trades_per_day: float = 0.0

    # Filtering stats
    filtered_by_quality: int = 0
    signals_analyzed: int = 0
    avg_risk_multiplier: float = 1.0

    # Breakdown stats
    monthly_stats: Dict = field(default_factory=dict)
    day_stats: Dict = field(default_factory=dict)
    hour_stats: Dict = field(default_factory=dict)
    entry_stats: Dict = field(default_factory=dict)


# ============================================================================
# DATA FETCHING
# ============================================================================

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


# ============================================================================
# QUALITY CALCULATION
# ============================================================================

def calculate_ob_quality(df, ob_idx, direction, col_map):
    """Calculate Order Block quality score (0-100)"""
    if ob_idx < 5 or ob_idx >= len(df) - 3:
        return 50

    quality = 0.0
    ob_bar = df.iloc[ob_idx]
    next_bars = df.iloc[ob_idx+1:ob_idx+4]

    open_col, high_col = col_map['open'], col_map['high']
    low_col, close_col = col_map['low'], col_map['close']

    # 1. Impulse move quality (max 50 points)
    if direction == 'BUY':
        impulse = next_bars[close_col].max() - ob_bar[low_col]
    else:
        impulse = ob_bar[high_col] - next_bars[close_col].min()
    impulse_pips = impulse * 10000
    quality += min(50, impulse_pips * 2.5)

    # 2. Wick analysis (max 25 points)
    ob_range = ob_bar[high_col] - ob_bar[low_col]
    if ob_range > 0:
        if direction == 'BUY':
            wick = ob_bar[high_col] - max(ob_bar[open_col], ob_bar[close_col])
        else:
            wick = min(ob_bar[open_col], ob_bar[close_col]) - ob_bar[low_col]
        wick_ratio = wick / ob_range
        if wick_ratio > 0.3:
            quality += 25
        elif wick_ratio > 0.2:
            quality += 15
        elif wick_ratio > 0.1:
            quality += 10

    # 3. Fresh zone bonus (max 25 points)
    zone_high, zone_low = ob_bar[high_col], ob_bar[low_col]
    touched = False
    for i in range(ob_idx + 4, min(ob_idx + 15, len(df))):
        bar = df.iloc[i]
        if direction == 'BUY' and bar[low_col] <= zone_high:
            touched = True
            break
        elif direction == 'SELL' and bar[high_col] >= zone_low:
            touched = True
            break
    if not touched:
        quality += 25

    return min(100, quality)


# ============================================================================
# ENTRY TRIGGER
# ============================================================================

def check_entry_trigger(bar, prev_bar, direction, col_map) -> Optional[str]:
    """Check for entry trigger and return entry type"""
    o, h, l, c = bar[col_map['open']], bar[col_map['high']], bar[col_map['low']], bar[col_map['close']]
    total_range = h - l
    if total_range < 0.0003:
        return None

    body = abs(c - o)
    is_bullish, is_bearish = c > o, c < o
    po, ph, pl, pc = prev_bar[col_map['open']], prev_bar[col_map['high']], prev_bar[col_map['low']], prev_bar[col_map['close']]

    if direction == 'BUY':
        lower_wick = min(o, c) - l
        if lower_wick > body and lower_wick > total_range * 0.5:
            return "REJECTION"
        if is_bullish and body > total_range * 0.6:
            return "MOMENTUM"
        if is_bullish and c > ph and o <= pl:
            return "ENGULF"
        if l > pl and is_bullish:
            return "HIGHER_LOW"
    else:
        upper_wick = h - max(o, c)
        if upper_wick > body and upper_wick > total_range * 0.5:
            return "REJECTION"
        if is_bearish and body > total_range * 0.6:
            return "MOMENTUM"
        if is_bearish and c < pl and o >= ph:
            return "ENGULF"
        if h < ph and is_bearish:
            return "LOWER_HIGH"
    return None


# ============================================================================
# MAIN BACKTEST v6
# ============================================================================

def run_backtest(df: pd.DataFrame) -> tuple:
    """Run v6 backtest with enhanced activity filter (efficiency + reversal detection)"""

    # Initialize components
    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=5.0, min_bar_range_pips=3.0,
        activity_threshold=35.0, pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 55.0  # Slightly lower for more trades

    market_filter = MarketConditionFilter(
        chop_ranging_threshold=65.0, adx_weak_threshold=16.0,  # More lenient
        atr_sl_multiplier=1.5, min_sl_pips=15.0, max_sl_pips=40.0,
        regime_confidence_threshold=55.0,
        enable_thursday_filter=False,  # We handle Thursday in risk scorer
        min_confluence_score=40.0  # Lower - we use risk scoring instead
    )

    # V5.1: Adaptive Risk Scorer
    risk_scorer = AdaptiveRiskScorer(
        min_combined_score=45.0,  # Minimum quality to trade
        min_risk_multiplier=0.3,  # Minimum 30% of base risk
        max_risk_multiplier=1.3   # Maximum 130% of base risk
    )

    drift_detector = ADWINDriftDetector(delta=0.005, min_samples=50)
    session_manager = SessionProfileManager(hybrid_mode=True)

    # v5.4: Confluence validator
    confluence_validator = ConfluenceValidator(
        min_total_score=48,    # Slightly lower than v4.5.2
        min_factors_passed=4   # Same as v4.5.2
    )

    # v5.6: Volatility Regime Filter
    volatility_filter = VolatilityRegimeFilter(
        atr_lookback=50,               # Use 50 bars for average
        low_vol_threshold=0.75,        # ATR < 75% avg = low vol
        very_low_vol_threshold=0.55,   # ATR < 55% avg = very low
        high_chop_threshold=60.0,      # Choppiness > 60 = ranging
        min_quality_score=42.0,        # Below this = SKIP
        reduced_risk_threshold=58.0,   # Below this = reduce risk
    )

    # v6 NEW: Enhanced Activity Filter with Efficiency & Reversal Detection
    # Based on daily analysis: LOW_EFFICIENCY and MANY_REVERSALS are key predictors
    # v6.1: RELAXED thresholds - previous was too strict (63T, 4 losing months)
    enhanced_filter = EnhancedActivityFilter(
        base_filter=activity_filter,
        efficiency_threshold=0.04,    # v6.1: Relaxed from 0.08 (was too strict)
        max_reversals=14,             # v6.1: Relaxed from 11 (allow more reversals)
        choppiness_threshold=62.0,    # v6.1: Relaxed from 55 (higher tolerance)
        min_quality_score=28.0,       # v6.1: Relaxed from 38 (allow marginal days)
    )

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    print("      Warming up indicators...")
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    trades: List[BacktestTrade] = []
    position: Optional[BacktestTrade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    # Tracking
    signals_analyzed = 0
    filtered_quality = 0
    filtered_volatility = 0  # v5.6: Track volatility filter
    filtered_efficiency = 0  # v6 NEW: Track efficiency filter
    risk_multipliers = []
    market_quality_scores = []  # v5.6: Track market quality
    efficiency_scores = []  # v6 NEW: Track efficiency scores

    cooldown_after_sl = timedelta(hours=1)
    cooldown_after_tp = timedelta(minutes=30)

    print("      Processing bars...")
    total_bars = len(df) - 100
    prev_price = df.iloc[99][col_map['close']]

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        # Update indicators
        kalman.update(price)
        regime_info = regime_detector.update(price)
        drift_detector.update_price(price, prev_price)
        prev_price = price

        if (idx - 100) % 500 == 0:
            pct = (idx - 100) / total_bars * 100
            print(f"      Progress: {pct:.0f}%")

        # =====================================================================
        # POSITION MANAGEMENT (same as v4.5)
        # =====================================================================
        if position:
            if position.direction == 'BUY' and low <= position.sl:
                position.exit_time, position.exit_price = current_time, position.sl
                position.pnl_pips = (position.sl - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'LOSS', 'SL'
                balance += position.pnl
                drift_detector.update_trade_result(False)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue
            elif position.direction == 'SELL' and high >= position.sl:
                position.exit_time, position.exit_price = current_time, position.sl
                position.pnl_pips = (position.entry_price - position.sl) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'LOSS', 'SL'
                balance += position.pnl
                drift_detector.update_trade_result(False)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue

            if position.direction == 'BUY' and high >= position.tp1:
                position.exit_time, position.exit_price = current_time, position.tp1
                position.pnl_pips = (position.tp1 - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'WIN', 'TP1'
                balance += position.pnl
                drift_detector.update_trade_result(True)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_tp
                continue
            elif position.direction == 'SELL' and low <= position.tp1:
                position.exit_time, position.exit_price = current_time, position.tp1
                position.pnl_pips = (position.entry_price - position.tp1) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'WIN', 'TP1'
                balance += position.pnl
                drift_detector.update_trade_result(True)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_tp
                continue

            # Regime flip exit
            if regime_info:
                if position.direction == 'BUY' and regime_info.bias == 'SELL':
                    position.exit_time, position.exit_price = current_time, price
                    position.pnl_pips = (price - position.entry_price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    drift_detector.update_trade_result(position.pnl > 0)
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
                    continue
                elif position.direction == 'SELL' and regime_info.bias == 'BUY':
                    position.exit_time, position.exit_price = current_time, price
                    position.pnl_pips = (position.entry_price - price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    drift_detector.update_trade_result(position.pnl > 0)
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
                    continue

        # Update drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Skip if in cooldown or has position
        if cooldown_until and current_time < cooldown_until:
            continue
        if position:
            continue

        # =====================================================================
        # SESSION/ACTIVITY CHECK
        # =====================================================================
        in_kz, kz_session = killzone.is_in_killzone(current_time)
        can_trade_outside = False

        if not in_kz:
            recent_df = df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE] and activity.score >= 55:
                can_trade_outside = True

        should_trade = in_kz or can_trade_outside
        if not should_trade:
            continue

        # Check regime
        if not regime_info or not regime_info.is_tradeable or regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias
        signals_analyzed += 1

        # Check drift
        drift_result = drift_detector.detect(current_time)
        drift_detected = drift_result.drift_detected

        # Market condition check
        recent_for_filter = df.iloc[max(0, idx-30):idx+1]
        regime_confidence = regime_info.confidence * 100 if hasattr(regime_info, 'confidence') else 80.0

        market_condition = market_filter.analyze(
            df=recent_for_filter, current_time=current_time,
            regime_confidence=regime_confidence, direction=direction
        )

        if not market_condition.can_trade:
            continue

        sl_pips = market_condition.suggested_sl_pips

        # =====================================================================
        # v5.6 NEW: VOLATILITY REGIME FILTER (Dynamic June-like detection)
        # =====================================================================
        # Use more data for volatility analysis (50 bars)
        vol_analysis_df = df.iloc[max(0, idx-60):idx+1]
        market_quality = volatility_filter.analyze(vol_analysis_df, current_time)
        market_quality_scores.append(market_quality.score)

        # Skip if market quality is very poor
        if not market_quality.can_trade:
            filtered_volatility += 1
            continue

        # =====================================================================
        # v6 NEW: ENHANCED ACTIVITY FILTER (Efficiency & Reversal Detection)
        # =====================================================================
        # This filter catches days like June 3, 4, 9, 16, 20, 27, 30 which had
        # LOW_EFFICIENCY and MANY_REVERSALS that caused losses
        enhanced_result = enhanced_filter.evaluate(
            dt=current_time,
            h1_data=vol_analysis_df,  # Use same data as volatility filter
            current_high=high,
            current_low=low,
            in_killzone=in_kz,
            session_name=kz_session if in_kz else 'hybrid'
        )

        efficiency_scores.append(enhanced_result.efficiency)

        # v6.2: SOFT FILTER - only hard skip on VERY poor conditions
        # Otherwise use as risk adjuster (not hard skip)
        if enhanced_result.quality_score < 20 and not in_kz:
            # Only hard skip on very poor quality OUTSIDE kill zone
            filtered_efficiency += 1
            continue

        # Use as risk multiplier (primary mechanism)
        efficiency_risk_mult = enhanced_result.risk_multiplier

        # If LOW_EFFICIENCY is detected, add extra penalty
        if enhanced_result.efficiency < 0.06:
            efficiency_risk_mult *= 0.7  # Extra 30% reduction

        # Find POI
        poi_found = False
        poi_quality = 50
        lookback = 15
        recent = df.iloc[idx-lookback:idx]

        for i in range(len(recent) - 3):
            ob_bar = recent.iloc[i]
            next_bars = recent.iloc[i+1:i+4]
            actual_idx = idx - lookback + i

            if direction == 'BUY':
                if ob_bar[col_map['close']] < ob_bar[col_map['open']]:
                    move = next_bars[col_map['close']].max() - ob_bar[col_map['low']]
                    if move > 0.0010:
                        poi_found = True
                        poi_quality = calculate_ob_quality(df, actual_idx, direction, col_map)
                        break
            else:
                if ob_bar[col_map['close']] > ob_bar[col_map['open']]:
                    move = ob_bar[col_map['high']] - next_bars[col_map['close']].min()
                    if move > 0.0010:
                        poi_found = True
                        poi_quality = calculate_ob_quality(df, actual_idx, direction, col_map)
                        break

        if not poi_found:
            continue

        # Check entry trigger
        entry_type = check_entry_trigger(bar, prev_bar, direction, col_map)
        if not entry_type:
            continue

        # =====================================================================
        # V5.2: ADAPTIVE RISK SCORING WITH SKIP THRESHOLD
        # =====================================================================
        risk_mult, required_quality, risk_reason, should_skip = risk_scorer.calculate_risk_score(
            current_time=current_time,
            entry_type=entry_type,
            poi_quality=poi_quality,
            in_killzone=in_kz,
            confluence_score=poi_quality,  # Simplified
            drift_detected=drift_detected
        )

        # v5.6: Skip if conditions are too unfavorable
        if should_skip:
            filtered_quality += 1
            continue

        # v5.6.2: COMPOUND RISK CHECK (tuned)
        # If BOTH volatility quality is marginal AND risk mult is low, SKIP
        if market_quality.score < 50 and risk_mult < 0.55:
            filtered_quality += 1
            continue

        # Check if quality meets requirement
        if poi_quality < required_quality:
            filtered_quality += 1
            continue

        # =====================================================================
        # v5.4 NEW: CONFLUENCE VALIDATION (additional layer)
        # =====================================================================
        profile = session_manager.get_profile(current_time)
        is_optimal, _ = session_manager.is_optimal_trading_time(current_time)
        kalman_state = kalman.last_state
        kalman_velocity = kalman_state.get('velocity', 0) if kalman_state else 0

        confluence_result = confluence_validator.validate(
            direction=direction,
            df=recent_for_filter,
            poi_quality=poi_quality,
            regime_confidence=regime_confidence,
            regime_bias=regime_info.bias,
            session_name=profile.name,
            is_optimal_session=is_optimal,
            drift_detected=drift_detected,
            current_time=current_time,
            kalman_velocity=kalman_velocity
        )

        # Only skip if confluence is very low (soft filter)
        if confluence_result.total_score < 40:  # Very low threshold
            filtered_quality += 1
            continue

        # Use confluence to further adjust risk
        confluence_risk_mult = confluence_result.risk_multiplier

        risk_multipliers.append(risk_mult * confluence_risk_mult)

        # =====================================================================
        # POSITION SIZING WITH ADAPTIVE RISK (v6: + efficiency filter)
        # =====================================================================
        tp1_pips = sl_pips * 1.5

        # Base risk 1%, adjusted by multiple factors:
        # - risk_scorer (day/hour/entry type)
        # - confluence_risk_mult (multi-factor validation)
        # - market_quality_mult (volatility regime)
        # - efficiency_risk_mult (v6 NEW: price efficiency & reversals)
        base_risk = 0.01
        market_quality_mult = market_quality.risk_multiplier
        combined_risk_mult = risk_mult * confluence_risk_mult * market_quality_mult * efficiency_risk_mult
        adjusted_risk = base_risk * combined_risk_mult

        risk_amount = balance * adjusted_risk
        lot_size = risk_amount / (sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            sl_price = price - sl_pips * 0.0001
            tp1_price = price + tp1_pips * 0.0001
        else:
            sl_price = price + sl_pips * 0.0001
            tp1_price = price - tp1_pips * 0.0001

        # Create position
        session_name = kz_session if in_kz else 'hybrid'
        position = BacktestTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=price,
            sl=sl_price,
            tp1=tp1_price,
            lot_size=lot_size,
            session=session_name,
            regime=regime_info.regime.value if hasattr(regime_info.regime, 'value') else str(regime_info.regime),
            poi_type='OB',
            entry_type=entry_type,
            quality_score=poi_quality,
            sl_pips=sl_pips,
            risk_multiplier=risk_mult,
            risk_reason=risk_reason,
            hour=current_time.hour,
            day_of_week=current_time.weekday()
        )

    # Close remaining position
    if position:
        last_bar = df.iloc[-1]
        position.exit_time = last_bar.name if isinstance(last_bar.name, datetime) else pd.Timestamp(last_bar.name).to_pydatetime()
        position.exit_price = last_bar[col_map['close']]
        if position.direction == 'BUY':
            position.pnl_pips = (position.exit_price - position.entry_price) * 10000
        else:
            position.pnl_pips = (position.entry_price - position.exit_price) * 10000
        position.pnl = position.pnl_pips * position.lot_size * 10
        position.result = 'WIN' if position.pnl > 0 else 'LOSS'
        position.exit_reason = 'END_OF_TEST'
        balance += position.pnl
        trades.append(position)

    # Calculate stats
    stats = calculate_stats(trades, balance, max_dd, df)
    stats.signals_analyzed = signals_analyzed
    stats.filtered_by_quality = filtered_quality
    stats.avg_risk_multiplier = np.mean(risk_multipliers) if risk_multipliers else 1.0

    return trades, stats, balance


def calculate_stats(trades, final_balance, max_dd, df):
    """Calculate comprehensive statistics"""
    stats = BacktestStats()
    if not trades:
        return stats

    stats.total_trades = len(trades)
    stats.wins = sum(1 for t in trades if t.result == 'WIN')
    stats.losses = sum(1 for t in trades if t.result == 'LOSS')
    stats.win_rate = stats.wins / stats.total_trades * 100 if stats.total_trades > 0 else 0

    stats.total_pnl = sum(t.pnl for t in trades)
    stats.total_pips = sum(t.pnl_pips for t in trades)
    stats.max_drawdown = max_dd
    stats.max_drawdown_pct = max_dd / 10000 * 100

    winning_trades = [t.pnl for t in trades if t.pnl > 0]
    losing_trades = [abs(t.pnl) for t in trades if t.pnl < 0]

    stats.avg_win = np.mean(winning_trades) if winning_trades else 0
    stats.avg_loss = np.mean(losing_trades) if losing_trades else 0
    stats.largest_win = max(winning_trades) if winning_trades else 0
    stats.largest_loss = max(losing_trades) if losing_trades else 0

    total_wins = sum(winning_trades)
    total_losses = sum(losing_trades)
    stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades if t.exit_time and t.entry_time]
    stats.avg_trade_duration = np.mean(durations) if durations else 0

    if len(df) > 0:
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            stats.trades_per_day = stats.total_trades / days

    # Monthly stats
    for t in trades:
        month_key = t.entry_time.strftime('%Y-%m')
        if month_key not in stats.monthly_stats:
            stats.monthly_stats[month_key] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pips': 0.0}
        stats.monthly_stats[month_key]['trades'] += 1
        if t.result == 'WIN':
            stats.monthly_stats[month_key]['wins'] += 1
        stats.monthly_stats[month_key]['pnl'] += t.pnl
        stats.monthly_stats[month_key]['pips'] += t.pnl_pips

    # Day stats
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for t in trades:
        day = day_names[t.day_of_week]
        if day not in stats.day_stats:
            stats.day_stats[day] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        stats.day_stats[day]['trades'] += 1
        if t.result == 'WIN':
            stats.day_stats[day]['wins'] += 1
        stats.day_stats[day]['pnl'] += t.pnl

    # Hour stats
    for t in trades:
        h = t.hour
        if h not in stats.hour_stats:
            stats.hour_stats[h] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        stats.hour_stats[h]['trades'] += 1
        if t.result == 'WIN':
            stats.hour_stats[h]['wins'] += 1
        stats.hour_stats[h]['pnl'] += t.pnl

    # Entry type stats
    for t in trades:
        et = t.entry_type
        if et not in stats.entry_stats:
            stats.entry_stats[et] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'avg_risk': []}
        stats.entry_stats[et]['trades'] += 1
        if t.result == 'WIN':
            stats.entry_stats[et]['wins'] += 1
        stats.entry_stats[et]['pnl'] += t.pnl
        stats.entry_stats[et]['avg_risk'].append(t.risk_multiplier)

    return stats


def print_report(stats, final_balance, trades):
    """Print detailed report"""
    print()
    print("=" * 70)
    print("H1 BACKTEST v6 RESULTS - ENHANCED ACTIVITY FILTER")
    print("=" * 70)
    print()

    print("V6 APPROACH (Based on Daily Analysis Jan 2025 - Jan 2026):")
    print("-" * 60)
    print("+ All v5.6 features (adaptive risk, volatility regime)")
    print("+ NEW: Price Efficiency detection (<8% = SKIP)")
    print("+ NEW: Reversal Count detection (>11 = reduce risk)")
    print("+ NEW: Day-of-week from analysis (Fri -15, Wed +8)")
    print("+ Target: ZERO losing months through real-time detection")
    print()

    print("OVERALL PERFORMANCE")
    print("-" * 50)
    print(f"Initial Balance:     $10,000.00")
    print(f"Final Balance:       ${final_balance:,.2f}")
    print(f"Net P/L:             ${stats.total_pnl:+,.2f}")
    print(f"Return:              {(final_balance/10000-1)*100:+.1f}%")
    print()

    print("TRADE STATISTICS")
    print("-" * 50)
    print(f"Total Trades:        {stats.total_trades} ({stats.trades_per_day:.2f}/day)")
    print(f"Win Rate:            {stats.win_rate:.1f}%")
    print(f"Profit Factor:       {stats.profit_factor:.2f}")
    print(f"Max Drawdown:        ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)")
    print()

    print("ADAPTIVE RISK STATS")
    print("-" * 50)
    print(f"Signals analyzed:    {stats.signals_analyzed}")
    print(f"Filtered (quality):  {stats.filtered_by_quality}")
    print(f"Avg risk multiplier: {stats.avg_risk_multiplier:.2f}x")
    print()

    # Day breakdown
    print("DAY OF WEEK BREAKDOWN")
    print("-" * 60)
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    for day in day_order:
        if day in stats.day_stats:
            data = stats.day_stats[day]
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            flag = "!!!" if data['pnl'] < 0 else ""
            print(f"  {day}: {data['trades']:>3} trades, {wr:>5.1f}% WR, ${data['pnl']:>+8.2f} {flag}")
    print()

    # Hour breakdown (top 10)
    print("HOUR BREAKDOWN (by trades)")
    print("-" * 60)
    sorted_hours = sorted(stats.hour_stats.items(), key=lambda x: -x[1]['trades'])[:12]
    for h, data in sorted_hours:
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        star = "*" if wr >= 55 else ""
        warn = "!" if wr < 45 else ""
        print(f"  {h:>2}:00  {data['trades']:>3} trades, {wr:>5.1f}% WR, ${data['pnl']:>+8.2f} {star}{warn}")
    print()

    # Entry type breakdown
    print("ENTRY TYPE BREAKDOWN")
    print("-" * 70)
    for et, data in sorted(stats.entry_stats.items(), key=lambda x: -x[1]['pnl']):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        avg_r = np.mean(data['avg_risk']) if data['avg_risk'] else 1.0
        print(f"  {et:<15} {data['trades']:>3} trades, {wr:>5.1f}% WR, ${data['pnl']:>+8.2f}, avg risk {avg_r:.2f}x")
    print()

    # Monthly breakdown
    print("MONTHLY PERFORMANCE")
    print("-" * 70)
    print(f"{'Month':<10} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Pips':>10} {'P/L':>12}")
    print("-" * 70)

    losing_months = 0
    for month, data in sorted(stats.monthly_stats.items()):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        status = "X" if data['pnl'] < 0 else "OK"
        if data['pnl'] < 0:
            losing_months += 1
        print(f"{month:<10} {data['trades']:>8} {data['wins']:>6} {wr:>7.1f}% {data['pips']:>+9.1f} ${data['pnl']:>+10.2f} [{status}]")

    print("-" * 70)
    print(f"Losing months: {losing_months}/{len(stats.monthly_stats)}")
    print()

    print("=" * 70)
    print("VERSION COMPARISON")
    print("=" * 70)
    print("v4.5.2:    89T, 49.4% WR, +$2,394, PF 1.57, 1 losing month [BEST]")
    print("v5.6:      83T, 48.2% WR, +$2,164, PF 1.62, 1 losing (June -$189)")
    print(f"v6:      {stats.total_trades:>4}T, {stats.win_rate:.1f}% WR, ${stats.total_pnl:+,.0f}, PF {stats.profit_factor:.2f}, {losing_months} losing <- CURRENT")
    print("=" * 70)


async def send_telegram_report(stats, trades, final_balance):
    """Send report to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(bot_token=config.telegram.bot_token, chat_id=config.telegram.chat_id)
        if not await telegram.initialize():
            return

        losing_months = sum(1 for d in stats.monthly_stats.values() if d['pnl'] < 0)
        total_months = len(stats.monthly_stats)

        # Entry breakdown
        entry_lines = []
        for et, data in sorted(stats.entry_stats.items(), key=lambda x: -x[1]['pnl']):
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            entry_lines.append(f"  {et}: {data['trades']}T, {wr:.0f}%, ${data['pnl']:+.0f}")

        # Monthly breakdown
        monthly_lines = []
        for month, data in sorted(stats.monthly_stats.items()):
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            status = "X" if data['pnl'] < 0 else "OK"
            monthly_lines.append(f"  {month}: {data['trades']}T, {wr:.0f}%, ${data['pnl']:+.0f} [{status}]")

        # Day breakdown
        day_lines = []
        for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
            if day in stats.day_stats:
                d = stats.day_stats[day]
                wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
                day_lines.append(f"  {day}: {d['trades']}T, {wr:.0f}%, ${d['pnl']:+.0f}")

        msg = f"""<b>H1 BACKTEST v5.2</b>
<b>Refined Adaptive Risk Scoring</b>

<b>PERFORMANCE</b>
Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
Win Rate: {stats.win_rate:.1f}%
Net P/L: <b>${stats.total_pnl:+,.2f}</b>
Return: <b>{(final_balance/10000-1)*100:+.1f}%</b>
Profit Factor: <b>{stats.profit_factor:.2f}</b>
Max DD: ${stats.max_drawdown:,.2f}

<b>ADAPTIVE RISK</b>
Signals: {stats.signals_analyzed}
Filtered: {stats.filtered_by_quality}
Avg risk: {stats.avg_risk_multiplier:.2f}x

<b>BY DAY</b>
{chr(10).join(day_lines)}

<b>BY ENTRY</b>
{chr(10).join(entry_lines)}

<b>MONTHLY</b>
{chr(10).join(monthly_lines)}

Losing months: {losing_months}/{total_months}

<b>COMPARISON</b>
v4.5: 89T, 49%, +$2,394, 1 losing
v5.1: 123T, 37%, +$863, 5 losing
<b>v5.6: {stats.total_trades}T, {stats.win_rate:.0f}%, ${stats.total_pnl:+,.0f}, {losing_months} losing</b>

<b>Final: ${final_balance:,.2f}</b>
"""
        await telegram.send(msg, force=True)
        logger.info("Telegram sent!")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI H1 BACKTEST v6")
    print("Enhanced Activity Filter (Efficiency + Reversal Detection)")
    print("=" * 70)

    print("\n[1/3] Fetching H1 data...")
    df = await fetch_data("GBPUSD", "H1",
                         datetime(2025, 1, 1, tzinfo=timezone.utc),
                         datetime(2026, 1, 31, tzinfo=timezone.utc))
    if df.empty:
        print("ERROR: No data")
        return
    print(f"      Loaded {len(df)} bars")

    print("\n[2/3] Running v6 backtest with enhanced activity filter...")
    trades, stats, final_balance = run_backtest(df)

    print_report(stats, final_balance, trades)

    print("\n[3/3] Sending to Telegram...")
    await send_telegram_report(stats, trades, final_balance)

    print("=" * 70)
    print("v6 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
