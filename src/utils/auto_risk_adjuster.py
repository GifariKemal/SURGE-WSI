"""
Automatic Risk Adjuster - Dynamic risk reduction based on market conditions
============================================================================

Instead of hardcoding "if month == 6" or "if month == 9", this module
automatically detects problematic market conditions and reduces risk.

Key indicators for problematic periods:
1. HIGH REVERSAL RATE - price keeps changing direction (whipsaw)
2. LOW PRICE EFFICIENCY - net_move / total_movement is low
3. HIGH CHOPPINESS - market is ranging/consolidating
4. POOR RECENT PERFORMANCE - if we're losing, reduce exposure

Philosophy: The system should READ market conditions dynamically, not rely
on calendar-based rules that may not apply next year.

Author: SURIOTA Team
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class MarketConditionAssessment:
    """Result of automatic market condition assessment"""
    risk_multiplier: float          # 0.05 to 1.0
    condition_score: float          # 0-100, higher = better
    reversal_rate: float            # 0-1, higher = more reversals
    price_efficiency: float         # 0-1, higher = cleaner trends
    choppiness: float               # 0-100, higher = more choppy
    recent_win_rate: Optional[float]  # Recent trade win rate if available
    reasons: List[str]              # Why risk was adjusted
    severity: str                   # 'NORMAL', 'CAUTION', 'DANGER'


class AutoRiskAdjuster:
    """
    Automatically adjusts risk based on detected market conditions.

    This replaces manual rules like "reduce risk in June/September" with
    dynamic detection of the CONDITIONS that made those months bad.
    """

    # Thresholds for problematic conditions (TUNED - less aggressive)
    # Only trigger when conditions are REALLY bad, not just slightly elevated
    REVERSAL_RATE_HIGH = 0.62       # >62% of bars change direction = bad
    REVERSAL_RATE_DANGER = 0.70     # >70% = very bad

    EFFICIENCY_LOW = 0.08           # <8% price efficiency = choppy
    EFFICIENCY_DANGER = 0.05        # <5% = very choppy

    CHOPPINESS_HIGH = 60.0          # >60 choppiness index = ranging
    CHOPPINESS_DANGER = 68.0        # >68 = very choppy

    WIN_RATE_LOW = 0.28             # <28% recent win rate = reduce risk
    WIN_RATE_DANGER = 0.20          # <20% = significantly reduce

    # Risk multipliers for different severity levels
    RISK_MULT_CAUTION = 0.6         # 60% risk when caution
    RISK_MULT_DANGER = 0.2          # 20% risk when danger
    RISK_MULT_EXTREME = 0.1         # 10% risk when extreme danger

    def __init__(self,
                 lookback_bars: int = 120,      # ~5 days of H1 data
                 min_bars: int = 24,            # At least 1 day
                 track_trades: bool = True):
        """
        Initialize AutoRiskAdjuster.

        Args:
            lookback_bars: Number of bars to analyze for conditions
            min_bars: Minimum bars required for analysis
            track_trades: Whether to track recent trade performance
        """
        self.lookback_bars = lookback_bars
        self.min_bars = min_bars
        self.track_trades = track_trades

        # Recent trade tracking
        self.recent_trades: List[Tuple[datetime, bool]] = []  # (time, won)
        self.max_recent_trades = 20

        logger.info(f"AutoRiskAdjuster initialized: lookback={lookback_bars} bars, "
                   f"track_trades={track_trades}")

    def record_trade(self, time: datetime, won: bool):
        """Record a trade result for win rate tracking"""
        if not self.track_trades:
            return

        self.recent_trades.append((time, won))

        # Keep only recent trades
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades = self.recent_trades[-self.max_recent_trades:]

    def get_recent_win_rate(self, current_time: datetime,
                           lookback_days: int = 30) -> Optional[float]:
        """Get win rate of recent trades within lookback period"""
        if not self.recent_trades:
            return None

        cutoff = current_time - timedelta(days=lookback_days)
        recent = [won for time, won in self.recent_trades if time >= cutoff]

        if len(recent) < 3:  # Need at least 3 trades
            return None

        return sum(recent) / len(recent)

    def calculate_reversal_rate(self, df: pd.DataFrame, col_map: dict) -> float:
        """
        Calculate how often price direction changes.
        High reversal rate = whipsaw market = bad for trend following.
        """
        if len(df) < 3:
            return 0.0

        closes = df[col_map['close']].values
        opens = df[col_map['open']].values

        # Direction: 1 = bullish, -1 = bearish
        directions = np.where(closes > opens, 1, -1)

        # Count direction changes
        changes = np.abs(np.diff(directions)) / 2  # 0 or 1
        reversal_rate = np.mean(changes)

        return reversal_rate

    def calculate_price_efficiency(self, df: pd.DataFrame, col_map: dict) -> float:
        """
        Calculate price efficiency = net_move / total_movement.
        Low efficiency = price moves a lot but goes nowhere = choppy.
        """
        if len(df) < 2:
            return 1.0

        # Net move (where we ended up)
        net_move = abs(df[col_map['close']].iloc[-1] - df[col_map['open']].iloc[0])

        # Total movement (sum of all bar ranges)
        highs = df[col_map['high']].values
        lows = df[col_map['low']].values
        total_movement = np.sum(highs - lows)

        if total_movement == 0:
            return 1.0

        efficiency = net_move / total_movement
        return min(1.0, efficiency)

    def calculate_choppiness(self, df: pd.DataFrame, col_map: dict,
                            period: int = 14) -> float:
        """
        Calculate Choppiness Index.
        High value (>61.8) = ranging/choppy
        Low value (<38.2) = trending
        """
        if len(df) < period + 1:
            return 50.0  # Neutral

        highs = df[col_map['high']].values
        lows = df[col_map['low']].values
        closes = df[col_map['close']].values

        # ATR calculation
        tr_values = []
        for i in range(1, len(df)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return 50.0

        # Sum of ATR over period
        atr_sum = sum(tr_values[-period:])

        # Highest high - lowest low over period
        hh = max(highs[-period:])
        ll = min(lows[-period:])
        price_range = hh - ll

        if price_range == 0:
            return 50.0

        # Choppiness Index formula
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)

        return max(0, min(100, chop))

    def assess(self, df: pd.DataFrame, col_map: dict,
               current_time: datetime) -> MarketConditionAssessment:
        """
        Assess current market conditions and return risk adjustment.

        This is the main method - call this before each trade to get
        the appropriate risk multiplier.
        """
        reasons = []
        danger_count = 0
        caution_count = 0

        # Use recent data for analysis
        recent_df = df.tail(self.lookback_bars)

        if len(recent_df) < self.min_bars:
            return MarketConditionAssessment(
                risk_multiplier=1.0,
                condition_score=50.0,
                reversal_rate=0.0,
                price_efficiency=1.0,
                choppiness=50.0,
                recent_win_rate=None,
                reasons=["Insufficient data"],
                severity='NORMAL'
            )

        # Calculate indicators
        reversal_rate = self.calculate_reversal_rate(recent_df, col_map)
        price_efficiency = self.calculate_price_efficiency(recent_df, col_map)
        choppiness = self.calculate_choppiness(recent_df, col_map)
        recent_wr = self.get_recent_win_rate(current_time)

        # Check reversal rate
        if reversal_rate >= self.REVERSAL_RATE_DANGER:
            danger_count += 1
            reasons.append(f"DANGER: High reversals ({reversal_rate:.0%})")
        elif reversal_rate >= self.REVERSAL_RATE_HIGH:
            caution_count += 1
            reasons.append(f"CAUTION: Elevated reversals ({reversal_rate:.0%})")

        # Check price efficiency
        if price_efficiency <= self.EFFICIENCY_DANGER:
            danger_count += 1
            reasons.append(f"DANGER: Very low efficiency ({price_efficiency:.1%})")
        elif price_efficiency <= self.EFFICIENCY_LOW:
            caution_count += 1
            reasons.append(f"CAUTION: Low efficiency ({price_efficiency:.1%})")

        # Check choppiness
        if choppiness >= self.CHOPPINESS_DANGER:
            danger_count += 1
            reasons.append(f"DANGER: Very choppy ({choppiness:.1f})")
        elif choppiness >= self.CHOPPINESS_HIGH:
            caution_count += 1
            reasons.append(f"CAUTION: Choppy market ({choppiness:.1f})")

        # Check recent win rate
        if recent_wr is not None:
            if recent_wr <= self.WIN_RATE_DANGER:
                danger_count += 1
                reasons.append(f"DANGER: Poor recent WR ({recent_wr:.0%})")
            elif recent_wr <= self.WIN_RATE_LOW:
                caution_count += 1
                reasons.append(f"CAUTION: Low recent WR ({recent_wr:.0%})")

        # Determine severity and risk multiplier (TUNED - less aggressive)
        # Need multiple indicators before reducing risk significantly
        if danger_count >= 3:
            severity = 'DANGER'
            risk_mult = self.RISK_MULT_EXTREME
        elif danger_count >= 2:
            severity = 'DANGER'
            risk_mult = self.RISK_MULT_DANGER
        elif danger_count >= 1 and caution_count >= 2:
            severity = 'CAUTION'
            risk_mult = self.RISK_MULT_CAUTION
        elif caution_count >= 3:
            severity = 'CAUTION'
            risk_mult = self.RISK_MULT_CAUTION
        elif caution_count >= 2:
            severity = 'CAUTION'
            risk_mult = 0.8  # Mild reduction
        else:
            severity = 'NORMAL'
            risk_mult = 1.0
            if not reasons:
                reasons.append("Market conditions normal")

        # Calculate overall condition score (0-100, higher = better)
        condition_score = (
            (1 - reversal_rate) * 30 +           # 30% weight
            price_efficiency * 100 * 0.3 +        # 30% weight
            (100 - choppiness) * 0.4              # 40% weight
        )

        return MarketConditionAssessment(
            risk_multiplier=risk_mult,
            condition_score=condition_score,
            reversal_rate=reversal_rate,
            price_efficiency=price_efficiency,
            choppiness=choppiness,
            recent_win_rate=recent_wr,
            reasons=reasons,
            severity=severity
        )

    def get_status_emoji(self, severity: str) -> str:
        """Get emoji for severity level"""
        return {
            'NORMAL': '🟢',
            'CAUTION': '🟡',
            'DANGER': '🔴'
        }.get(severity, '⚪')


# Convenience function for quick assessment
def assess_market_risk(df: pd.DataFrame, col_map: dict,
                       current_time: datetime) -> float:
    """
    Quick function to get risk multiplier.
    Returns 0.05 to 1.0
    """
    adjuster = AutoRiskAdjuster()
    result = adjuster.assess(df, col_map, current_time)
    return result.risk_multiplier
