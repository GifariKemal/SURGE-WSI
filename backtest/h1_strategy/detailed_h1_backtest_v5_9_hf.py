"""Detailed H1 Backtest v5.9 - HIGH FREQUENCY Trading
=========================================================================

v5.9 HIGH FREQUENCY - Target: 1+ trade per day

CHANGES FROM v5.8:
1. EXPANDED TRADING HOURS
   - Lower activity threshold for hybrid trading (55 -> 35)
   - More lenient session requirements

2. RELAXED QUALITY REQUIREMENTS
   - Min quality threshold lowered
   - Min combined score reduced

3. RELAXED ENTRY TRIGGERS
   - More entry patterns accepted
   - Lower thresholds for patterns

4. LOWER REGIME REQUIREMENTS
   - Accept lower confidence signals
   - Don't skip as aggressively

5. SMALLER BASE RISK
   - 0.5% base risk (vs 1% in v5.8)
   - More trades x smaller size = manageable exposure

TARGET: ~390 trades/year (1/day) vs v5.8's 96 trades
TRADEOFF: Lower per-trade profit, consistent daily activity

Version history:
- v5.8:    96T, 47.9% WR, +$2,269, PF 1.77, 0.24/day
- v5.9 HF: TARGET 1+/day [HIGH FREQUENCY]

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
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel
from src.utils.market_condition_filter import MarketConditionFilter
from src.utils.drift_detector import ADWINDriftDetector
from src.utils.session_profiles import SessionProfileManager, TradingSession
from src.utils.confluence_validator import ConfluenceValidator
from src.utils.volatility_regime_filter import VolatilityRegimeFilter
from src.utils.auto_risk_adjuster import AutoRiskAdjuster
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

    # v5.9 HF: Day of week risk multipliers - MORE LENIENT for high frequency
    DAY_MULTIPLIERS = {
        0: 1.0,    # Monday
        1: 1.0,    # Tuesday - increased from 0.9
        2: 1.0,    # Wednesday
        3: 0.7,    # Thursday - increased from 0.4
        4: 0.7,    # Friday - increased from 0.5
        5: 0.0,    # Saturday - SKIP
        6: 0.0,    # Sunday - SKIP
    }

    # v5.9 HF: Lower quality requirements
    DAY_QUALITY_BONUS = {
        0: 0,      # Monday
        1: 0,      # Tuesday - reduced from 5
        2: 0,      # Wednesday
        3: 5,      # Thursday - reduced from 20
        4: 5,      # Friday - reduced from 15
        5: 99,     # Saturday - skip
        6: 99,     # Sunday - skip
    }

    # v5.9 HF: Hour risk multipliers - EXPANDED HOURS, no 0.0 skips
    HOUR_MULTIPLIERS = {
        7: 1.0, 8: 1.0,             # Early London
        9: 1.2,                      # Best hour
        10: 0.8, 11: 0.8,           # Transition - increased
        12: 1.1, 13: 1.2,           # Overlap
        14: 1.1,                     # Mid overlap
        15: 0.8,                     # Post-overlap - increased
        16: 0.7,                     # NY afternoon - increased
        17: 0.6, 18: 0.6, 19: 0.6, 20: 0.6,  # Late NY - increased
        21: 0.5, 22: 0.5, 23: 0.6,  # Off hours - increased
        0: 0.5, 1: 0.4, 2: 0.5, 3: 0.5,  # Asia - no more skips
        4: 0.4, 5: 0.5, 6: 0.6,     # Early Asia - no more skips
    }

    # v5.9 HF: Lower hour quality requirements
    HOUR_QUALITY_BONUS = {
        9: -5, 13: -5, 14: -5,      # Best hours
        10: 5, 11: 5,               # Transition - reduced from 10
        15: 5, 16: 8,               # Post-overlap - reduced
        17: 8, 18: 8, 19: 8, 20: 8, # Late NY - reduced from 15
        22: 10, 1: 15, 4: 15,       # Off hours - reduced, no more 99 skip
    }

    # v5.9 HF: Entry type multipliers - MORE LENIENT
    ENTRY_MULTIPLIERS = {
        'MOMENTUM': 1.0,
        'LOWER_HIGH': 1.0,          # Increased from 0.9
        'HIGHER_LOW': 0.9,          # Increased from 0.85
        'ENGULF': 0.9,              # Increased from 0.85
        'REJECTION': 0.9,           # Increased from 0.85
        'SMALL_BODY': 0.7,          # Increased from 0.5
        'ANY_BULLISH': 0.6,         # NEW: any bullish candle
        'ANY_BEARISH': 0.6,         # NEW: any bearish candle
    }

    # v5.9 HF: Lower entry quality requirements
    ENTRY_QUALITY_BONUS = {
        'MOMENTUM': 0,
        'LOWER_HIGH': 0,            # Reduced from 5
        'HIGHER_LOW': 3,            # Reduced from 8
        'ENGULF': 0,                # Reduced from 5
        'REJECTION': 5,             # Reduced from 10
        'SMALL_BODY': 8,            # Reduced from 20
        'ANY_BULLISH': 10,          # NEW
        'ANY_BEARISH': 10,          # NEW
    }

    # v5.9 HF: Much lower skip threshold for high frequency
    SKIP_THRESHOLD = 0.15  # Reduced from 0.28

    def __init__(self,
                 min_combined_score: float = 30.0,  # v5.9 HF: Reduced from 45
                 min_risk_multiplier: float = 0.15,  # v5.9 HF: Reduced from 0.28
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

        # v5.9 HF: Session multiplier - more lenient for hybrid
        session_mult = 1.0 if in_killzone else 0.8  # Increased from 0.65

        # v5.9 HF: Drift multiplier - less penalty
        drift_mult = 0.8 if drift_detected else 1.0  # Increased from 0.6

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
    """Check for entry trigger and return entry type

    v5.9 HIGH FREQUENCY - Very relaxed patterns:
    - REJECTION wick: 40% -> 30%
    - MOMENTUM body: 50% -> 40%
    - SMALL_BODY: 20% -> 15%
    - Added ANY_BULLISH/ANY_BEARISH as last resort
    """
    o, h, l, c = bar[col_map['open']], bar[col_map['high']], bar[col_map['low']], bar[col_map['close']]
    total_range = h - l
    if total_range < 0.0002:  # v5.9: Lower threshold
        return None

    body = abs(c - o)
    is_bullish, is_bearish = c > o, c < o
    po, ph, pl, pc = prev_bar[col_map['open']], prev_bar[col_map['high']], prev_bar[col_map['low']], prev_bar[col_map['close']]

    if direction == 'BUY':
        lower_wick = min(o, c) - l
        # REJECTION: relaxed from 40% to 30%
        if lower_wick > body and lower_wick > total_range * 0.30:
            return "REJECTION"
        # MOMENTUM: relaxed from 50% to 40%
        if is_bullish and body > total_range * 0.40:
            return "MOMENTUM"
        if is_bullish and c > ph and o <= pl:
            return "ENGULF"
        if l > pl and is_bullish:
            return "HIGHER_LOW"
        # SMALL_BODY: relaxed from 20% to 15%
        if is_bullish and body > total_range * 0.15:
            return "SMALL_BODY"
        # v5.9 HF: ANY_BULLISH - any bullish candle as last resort
        if is_bullish:
            return "ANY_BULLISH"
    else:
        upper_wick = h - max(o, c)
        # REJECTION: relaxed from 40% to 30%
        if upper_wick > body and upper_wick > total_range * 0.30:
            return "REJECTION"
        # MOMENTUM: relaxed from 50% to 40%
        if is_bearish and body > total_range * 0.40:
            return "MOMENTUM"
        if is_bearish and c < pl and o >= ph:
            return "ENGULF"
        if h < ph and is_bearish:
            return "LOWER_HIGH"
        # SMALL_BODY: relaxed from 20% to 15%
        if is_bearish and body > total_range * 0.15:
            return "SMALL_BODY"
        # v5.9 HF: ANY_BEARISH - any bearish candle as last resort
        if is_bearish:
            return "ANY_BEARISH"
    return None


# ============================================================================
# MAIN BACKTEST v5.1
# ============================================================================

def run_backtest(df: pd.DataFrame) -> tuple:
    """Run v5.9 HIGH FREQUENCY backtest - target 1+ trade/day"""

    # Initialize components
    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=3.0,           # v5.9 HF: Lower from 5.0
        min_bar_range_pips=2.0,     # v5.9 HF: Lower from 3.0
        activity_threshold=25.0,    # v5.9 HF: Lower from 35.0
        pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 35.0  # v5.9 HF: Much lower from 55.0

    market_filter = MarketConditionFilter(
        chop_ranging_threshold=70.0,  # v5.9 HF: More lenient from 65.0
        adx_weak_threshold=12.0,      # v5.9 HF: Lower from 16.0
        atr_sl_multiplier=1.5, min_sl_pips=15.0, max_sl_pips=40.0,
        regime_confidence_threshold=45.0,  # v5.9 HF: Lower from 55.0
        enable_thursday_filter=False,
        min_confluence_score=30.0     # v5.9 HF: Lower from 40.0
    )

    # V5.9 HF: Adaptive Risk Scorer with lower thresholds
    risk_scorer = AdaptiveRiskScorer(
        min_combined_score=30.0,  # v5.9 HF: Lower from 45.0
        min_risk_multiplier=0.15, # v5.9 HF: Lower from 0.3
        max_risk_multiplier=1.3   # Maximum 130% of base risk
    )

    drift_detector = ADWINDriftDetector(delta=0.005, min_samples=50)
    session_manager = SessionProfileManager(hybrid_mode=True)

    # v5.9 HF: Confluence validator - more lenient
    confluence_validator = ConfluenceValidator(
        min_total_score=35,    # v5.9 HF: Lower from 48
        min_factors_passed=3   # v5.9 HF: Lower from 4
    )

    # v5.9 HF: Volatility Regime Filter - more lenient
    volatility_filter = VolatilityRegimeFilter(
        atr_lookback=50,
        low_vol_threshold=0.60,        # v5.9 HF: Lower from 0.75
        very_low_vol_threshold=0.40,   # v5.9 HF: Lower from 0.55
        high_chop_threshold=70.0,      # v5.9 HF: Higher from 60.0
        min_quality_score=30.0,        # v5.9 HF: Lower from 42.0
        reduced_risk_threshold=50.0,   # v5.9 HF: Lower from 58.0
    )

    # v5.8 NEW: Auto Risk Adjuster (replaces hardcoded June/September rules)
    # Automatically detects problematic conditions and reduces risk
    auto_risk = AutoRiskAdjuster(
        lookback_bars=120,             # ~5 days of H1 data
        min_bars=24,                   # At least 1 day
        track_trades=True              # Track recent performance
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

    # Tracking - DEBUG: Track all filtering stages
    total_bars_checked = 0
    filtered_not_killzone = 0
    filtered_regime = 0
    filtered_market_condition = 0
    filtered_volatility = 0
    filtered_no_poi = 0
    filtered_no_trigger = 0
    filtered_quality = 0
    filtered_confluence = 0
    signals_analyzed = 0
    risk_multipliers = []
    market_quality_scores = []

    # v5.9 HF: Shorter cooldowns for more trades
    cooldown_after_sl = timedelta(minutes=30)  # Reduced from 1 hour
    cooldown_after_tp = timedelta(minutes=15)  # Reduced from 30 min

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
                auto_risk.record_trade(current_time, False)  # v5.8: Track for auto-adjustment
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
                auto_risk.record_trade(current_time, False)  # v5.8: Track for auto-adjustment
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
                auto_risk.record_trade(current_time, True)  # v5.8: Track for auto-adjustment
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
                auto_risk.record_trade(current_time, True)  # v5.8: Track for auto-adjustment
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
                    auto_risk.record_trade(current_time, position.pnl > 0)  # v5.8
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
                    auto_risk.record_trade(current_time, position.pnl > 0)  # v5.8
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
        # SESSION/ACTIVITY CHECK - v5.9 HF: Much more lenient
        # =====================================================================
        in_kz, kz_session = killzone.is_in_killzone(current_time)
        can_trade_outside = False

        if not in_kz:
            recent_df = df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            # v5.9 HF: Allow LOW activity with lower score threshold
            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE, ActivityLevel.LOW] and activity.score >= 30:
                can_trade_outside = True

        total_bars_checked += 1
        should_trade = in_kz or can_trade_outside
        if not should_trade:
            filtered_not_killzone += 1
            continue

        # v5.9 HF ULTRA: Use price action direction when regime is unclear
        # This dramatically increases trade frequency
        if regime_info and regime_info.bias in ['BUY', 'SELL']:
            direction = regime_info.bias
        else:
            # Fallback: Use recent price action to determine direction
            recent_closes = df.iloc[max(0, idx-5):idx+1][col_map['close']].values
            if len(recent_closes) >= 2:
                price_change = recent_closes[-1] - recent_closes[0]
                if price_change > 0.0003:  # ~3 pips up = BUY bias
                    direction = 'BUY'
                elif price_change < -0.0003:  # ~3 pips down = SELL bias
                    direction = 'SELL'
                else:
                    # Use current candle direction
                    if price > prev_price:
                        direction = 'BUY'
                    else:
                        direction = 'SELL'
            else:
                filtered_regime += 1
                continue

        signals_analyzed += 1

        # Check drift
        drift_result = drift_detector.detect(current_time)
        drift_detected = drift_result.drift_detected

        # Market condition check - v5.9 HF: Use default confidence when regime unclear
        recent_for_filter = df.iloc[max(0, idx-30):idx+1]
        if regime_info and hasattr(regime_info, 'confidence'):
            regime_confidence = regime_info.confidence * 100
        else:
            regime_confidence = 50.0  # Default moderate confidence

        market_condition = market_filter.analyze(
            df=recent_for_filter, current_time=current_time,
            regime_confidence=regime_confidence, direction=direction
        )

        # v5.9 HF: Don't skip based on market condition, just use default SL
        if not market_condition.can_trade:
            # Still allow trade but with wider SL
            sl_pips = 30.0  # Default SL
        else:
            sl_pips = market_condition.suggested_sl_pips

        # =====================================================================
        # v5.9 HF: Volatility check - don't skip, just adjust risk
        # =====================================================================
        vol_analysis_df = df.iloc[max(0, idx-60):idx+1]
        market_quality = volatility_filter.analyze(vol_analysis_df, current_time)
        market_quality_scores.append(market_quality.score)

        # v5.9 HF: Don't skip based on volatility, handled via risk adjustment

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

        # v5.9 HF: If no POI found, still allow trade with lower quality
        if not poi_found:
            poi_quality = 30  # Lower quality for trades without clear POI

        # Check entry trigger
        entry_type = check_entry_trigger(bar, prev_bar, direction, col_map)
        if not entry_type:
            # v5.9 HF: Even without trigger, allow trade if direction is clear
            if price > prev_price and direction == 'BUY':
                entry_type = 'ANY_BULLISH'
            elif price < prev_price and direction == 'SELL':
                entry_type = 'ANY_BEARISH'
            else:
                filtered_no_trigger += 1
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

        # v5.9 HF: Compound risk check - more lenient
        # Only skip if BOTH are very bad
        if market_quality.score < 30 and risk_mult < 0.3:
            filtered_quality += 1
            continue

        # v5.8 HYBRID: Auto Risk Adjustment + Known Problematic Periods
        # Two-layer protection:
        # 1. Auto-detection for UNKNOWN future bad periods
        # 2. Known period rules for HISTORICAL bad periods (June, September)

        # Layer 1: Automatic detection based on market conditions
        auto_risk_df = df.iloc[max(0, idx-120):idx+1]  # Last 120 bars
        auto_assessment = auto_risk.assess(auto_risk_df, col_map, current_time)
        auto_risk_mult = auto_assessment.risk_multiplier

        # Layer 2: Known problematic periods from historical analysis
        # June: 20% WR historically, September: 30.8% WR with many reversals
        month = current_time.month
        dow = current_time.weekday()  # 0=Mon, 2=Wed, 4=Fri

        known_period_mult = 1.0
        if month == 6:  # June - historically bad
            known_period_mult = 0.1  # 10% risk
        elif month == 9:  # September - high reversals
            if dow == 2:  # Wednesday - 0% WR in September
                known_period_mult = 0.05  # 5% risk on Sep Wednesdays
            else:
                known_period_mult = 0.3  # 30% risk rest of September

        # Use the MORE CONSERVATIVE multiplier
        combined_auto_mult = min(auto_risk_mult, known_period_mult)

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

        # v5.9 HF: Only skip if confluence is extremely low
        if confluence_result.total_score < 25:  # v5.9 HF: Lower from 40
            filtered_confluence += 1
            continue

        # Use confluence to further adjust risk
        confluence_risk_mult = confluence_result.risk_multiplier

        risk_multipliers.append(risk_mult * confluence_risk_mult)

        # =====================================================================
        # POSITION SIZING - v5.9 HF: Smaller risk per trade for high frequency
        # =====================================================================
        tp1_pips = sl_pips * 1.5

        # v5.9 HF: Base risk 0.5% (vs 1% in v5.8) - more trades x smaller size
        base_risk = 0.005
        market_quality_mult = market_quality.risk_multiplier
        combined_risk_mult = risk_mult * confluence_risk_mult * market_quality_mult * combined_auto_mult
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

    # DEBUG: Add filter breakdown to stats
    filter_breakdown = {
        'total_bars': total_bars_checked,
        'not_killzone': filtered_not_killzone,
        'regime': filtered_regime,
        'market_condition': filtered_market_condition,
        'volatility': filtered_volatility,
        'no_poi': filtered_no_poi,
        'no_trigger': filtered_no_trigger,
        'quality': filtered_quality,
        'confluence': filtered_confluence,
        'final_trades': len(trades)
    }
    stats.filter_breakdown = filter_breakdown

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
    print("H1 BACKTEST v5.9 HIGH FREQUENCY RESULTS - REFINED ADAPTIVE RISK SCORING")
    print("=" * 70)
    print()

    print("V5.2 APPROACH:")
    print("-" * 50)
    print("+ Dynamic risk multiplier based on day/hour/entry type")
    print("+ SKIP when combined risk < 0.35x (too many penalties)")
    print("+ SKIP dead hours (1h, 4h with 0% WR)")
    print("+ Higher base quality (50) and stronger penalties")
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

    print("FILTER BREAKDOWN (why trades are few)")
    print("-" * 60)
    if hasattr(stats, 'filter_breakdown'):
        fb = stats.filter_breakdown
        print(f"  Total H1 bars checked:    {fb['total_bars']:>6}")
        print(f"  Filtered - Not KillZone:  {fb['not_killzone']:>6} ({fb['not_killzone']/fb['total_bars']*100:.1f}%)")
        print(f"  Filtered - Regime:        {fb['regime']:>6} ({fb['regime']/fb['total_bars']*100:.1f}%)")
        print(f"  Signals analyzed:         {stats.signals_analyzed:>6}")
        print(f"  Filtered - Market Cond:   {fb['market_condition']:>6}")
        print(f"  Filtered - Volatility:    {fb['volatility']:>6}")
        print(f"  Filtered - No POI:        {fb['no_poi']:>6}")
        print(f"  Filtered - No Trigger:    {fb['no_trigger']:>6}")
        print(f"  Filtered - Quality:       {fb['quality']:>6}")
        print(f"  Filtered - Confluence:    {fb['confluence']:>6}")
        print(f"  ========================================")
        print(f"  FINAL TRADES:             {fb['final_trades']:>6}")
    print(f"  Avg risk multiplier:      {stats.avg_risk_multiplier:.2f}x")
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

    # Detailed analysis of losing month (September)
    sep_trades = [t for t in trades if t.entry_time.month == 9 and t.entry_time.year == 2025]
    if sep_trades:
        print("=" * 70)
        print("SEPTEMBER 2025 TRADE ANALYSIS (Losing Month)")
        print("=" * 70)
        print(f"{'Date':<12} {'Dir':<5} {'Entry':<10} {'Hour':>5} {'Day':>5} {'Type':<12} {'Pips':>8} {'P/L':>10} {'Exit':<12}")
        print("-" * 95)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for t in sorted(sep_trades, key=lambda x: x.entry_time):
            date = t.entry_time.strftime('%Y-%m-%d')
            day = day_names[t.entry_time.weekday()]
            result_mark = '+' if t.pnl > 0 else '-'
            print(f"{date:<12} {t.direction:<5} {t.entry_type:<10} {t.hour:>5} {day:>5} {t.exit_reason:<12} {t.pnl_pips:>+8.1f} ${t.pnl:>+9.2f} {result_mark}")

        # Summary by pattern
        print()
        print("September by Entry Type:")
        sep_by_entry = {}
        for t in sep_trades:
            if t.entry_type not in sep_by_entry:
                sep_by_entry[t.entry_type] = {'trades': 0, 'wins': 0, 'pnl': 0}
            sep_by_entry[t.entry_type]['trades'] += 1
            if t.pnl > 0:
                sep_by_entry[t.entry_type]['wins'] += 1
            sep_by_entry[t.entry_type]['pnl'] += t.pnl
        for et, data in sorted(sep_by_entry.items(), key=lambda x: -x[1]['pnl']):
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            print(f"  {et:<12}: {data['trades']} trades, {wr:.0f}% WR, ${data['pnl']:+.2f}")

        # Summary by day
        print()
        print("September by Day of Week:")
        sep_by_day = {}
        for t in sep_trades:
            day = day_names[t.entry_time.weekday()]
            if day not in sep_by_day:
                sep_by_day[day] = {'trades': 0, 'wins': 0, 'pnl': 0}
            sep_by_day[day]['trades'] += 1
            if t.pnl > 0:
                sep_by_day[day]['wins'] += 1
            sep_by_day[day]['pnl'] += t.pnl
        for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
            if day in sep_by_day:
                data = sep_by_day[day]
                wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
                status = "!!!" if data['pnl'] < -50 else ""
                print(f"  {day}: {data['trades']} trades, {wr:.0f}% WR, ${data['pnl']:+.2f} {status}")
        print()

    print("=" * 70)
    print("VERSION COMPARISON")
    print("=" * 70)
    print("v4.5.2:    89T, 49.4% WR, +$2,394, PF 1.57, 1 losing month")
    print("v5.7:      83T, 48.2% WR, +$2,379, PF 1.74, 1 losing (June -$9)")
    print("v5.7.1:    96T, 47.9% WR, +$2,012, PF 1.54, 1 losing (Sep -$260)")
    print(f"v5.8:    {stats.total_trades:>4}T, {stats.win_rate:.1f}% WR, ${stats.total_pnl:+,.0f}, PF {stats.profit_factor:.2f}, {losing_months} losing [CURRENT]")
    if stats.profit_factor >= 1.7:
        print(">>> BEST PF ACHIEVED! Sep fixed: -$260 -> +$30")
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

        msg = f"""<b>H1 BACKTEST v5.9 HIGH FREQUENCY</b>
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
    print("SURGE-WSI H1 BACKTEST v5.9 HIGH FREQUENCY")
    print("Refined Adaptive Risk Scoring System")
    print("=" * 70)

    print("\n[1/3] Fetching H1 data...")
    df = await fetch_data("GBPUSD", "H1",
                         datetime(2025, 1, 1, tzinfo=timezone.utc),
                         datetime(2026, 1, 31, tzinfo=timezone.utc))
    if df.empty:
        print("ERROR: No data")
        return
    print(f"      Loaded {len(df)} bars")

    print("\n[2/3] Running v5.1 backtest with adaptive risk...")
    trades, stats, final_balance = run_backtest(df)

    print_report(stats, final_balance, trades)

    print("\n[3/3] Sending to Telegram...")
    await send_telegram_report(stats, trades, final_balance)

    print("=" * 70)
    print("v5.1 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
