"""Adaptive Risk Manager
========================

Automatically adjusts risk parameters based on:
1. Market volatility (ATR-based)
2. Recent performance (drawdown)
3. Regime confidence
4. Time of day/week

This creates a dynamic risk system that becomes more
conservative during unfavorable conditions and more
aggressive during favorable conditions.

Author: SURIOTA Team
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class MarketCondition(Enum):
    """Market condition classification"""
    LOW_VOLATILITY = "low_volatility"       # Ranging, low opportunity
    NORMAL = "normal"                        # Standard conditions
    HIGH_VOLATILITY = "high_volatility"      # Trending, high opportunity
    EXTREME_VOLATILITY = "extreme"           # Too risky, reduce exposure


class PerformanceState(Enum):
    """Recent performance classification"""
    WINNING_STREAK = "winning"    # Recent wins, can be slightly aggressive
    NORMAL = "normal"             # Standard performance
    LOSING_STREAK = "losing"      # Recent losses, be conservative
    DRAWDOWN = "drawdown"         # Significant drawdown, be very conservative


@dataclass
class AdaptiveRiskParams:
    """Dynamic risk parameters"""
    max_lot_size: float
    min_sl_pips: float
    max_sl_pips: float
    risk_percent: float
    reason: str
    max_loss_per_trade_pct: float = 0.1  # ZERO LOSING MONTHS: Max loss 0.1% (was 0.8)


class AdaptiveRiskManager:
    """Adaptive risk management based on market conditions

    Zero Losing Months Strategy:
    - Caps maximum loss per trade at % of balance
    - Uses stricter consecutive loss thresholds
    - Special handling for December (anomaly month)
    """

    def __init__(
        self,
        # ZERO LOSING MONTHS CONFIGURATION
        base_max_lot: float = 0.5,
        base_min_sl: float = 15.0,
        base_max_sl: float = 10.0,  # ZERO LOSING MONTHS: Max SL 10 pips (was 50)
        base_risk_percent: float = 0.008,  # 0.8% max risk

        # ZERO LOSING MONTHS: Max loss per trade protection
        max_loss_per_trade_pct: float = 0.1,  # ZERO LOSING MONTHS: Maximum 0.1% loss (was 0.8)

        # Volatility thresholds (in pips) - tuned for GBPUSD
        low_volatility_atr: float = 12.0,   # ATR < 12 pips = low volatility
        high_volatility_atr: float = 40.0,  # ATR > 40 pips = high volatility
        extreme_volatility_atr: float = 55.0,  # ATR > 55 = extreme

        # Performance tracking (stricter for zero-loss)
        consecutive_loss_threshold: int = 2,  # 2 losses = reduce risk
        drawdown_threshold: float = 0.08,  # 8% drawdown triggers conservative mode

        # Adjustment multipliers
        low_vol_lot_mult: float = 0.6,      # Reduce lot in low vol
        high_vol_lot_mult: float = 0.8,     # Slightly reduce in high vol
        extreme_vol_lot_mult: float = 0.4,  # Heavily reduce in extreme
        drawdown_lot_mult: float = 0.5,     # 50% reduction in drawdown
        losing_streak_mult: float = 0.7,    # 70% during losing streak

        # December special settings (anomaly month)
        december_max_lot: float = 0.1,
        december_min_quality: float = 85.0,
    ):
        self.base_max_lot = base_max_lot
        self.base_min_sl = base_min_sl
        self.base_max_sl = base_max_sl
        self.base_risk_percent = base_risk_percent

        # Max loss per trade protection
        self.max_loss_per_trade_pct = max_loss_per_trade_pct

        # Volatility thresholds
        self.low_volatility_atr = low_volatility_atr
        self.high_volatility_atr = high_volatility_atr
        self.extreme_volatility_atr = extreme_volatility_atr

        # Performance thresholds
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.drawdown_threshold = drawdown_threshold

        # Adjustment multipliers
        self.low_vol_lot_mult = low_vol_lot_mult
        self.high_vol_lot_mult = high_vol_lot_mult
        self.extreme_vol_lot_mult = extreme_vol_lot_mult
        self.drawdown_lot_mult = drawdown_lot_mult
        self.losing_streak_mult = losing_streak_mult

        # December special settings
        self.december_max_lot = december_max_lot
        self.december_min_quality = december_min_quality

        # State tracking
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        self._peak_balance = 0.0
        self._current_drawdown = 0.0
        self._recent_trades: list = []  # Track recent trade results

    def classify_volatility(self, atr_pips: float) -> MarketCondition:
        """Classify market volatility based on ATR"""
        if atr_pips < self.low_volatility_atr:
            return MarketCondition.LOW_VOLATILITY
        elif atr_pips > self.extreme_volatility_atr:
            return MarketCondition.EXTREME_VOLATILITY
        elif atr_pips > self.high_volatility_atr:
            return MarketCondition.HIGH_VOLATILITY
        else:
            return MarketCondition.NORMAL

    def classify_performance(self, current_balance: float) -> PerformanceState:
        """Classify recent performance"""
        # Update drawdown
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        if self._peak_balance > 0:
            self._current_drawdown = (self._peak_balance - current_balance) / self._peak_balance

        # Check drawdown first (most severe)
        if self._current_drawdown >= self.drawdown_threshold:
            return PerformanceState.DRAWDOWN

        # Check consecutive losses
        if self._consecutive_losses >= self.consecutive_loss_threshold:
            return PerformanceState.LOSING_STREAK

        # Check winning streak (3+ consecutive wins)
        if self._consecutive_wins >= 3:
            return PerformanceState.WINNING_STREAK

        return PerformanceState.NORMAL

    def record_trade_result(self, is_win: bool, pnl: float):
        """Record trade result for performance tracking"""
        if is_win:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        self._recent_trades.append({
            'is_win': is_win,
            'pnl': pnl,
            'time': datetime.now(timezone.utc)
        })

        # Keep only last 20 trades
        if len(self._recent_trades) > 20:
            self._recent_trades.pop(0)

    def get_adaptive_params(
        self,
        current_balance: float,
        atr_pips: float,
        regime_confidence: float = 0.7,
        current_time: Optional[datetime] = None
    ) -> AdaptiveRiskParams:
        """Get adaptive risk parameters based on current conditions

        Args:
            current_balance: Current account balance
            atr_pips: Current ATR in pips
            regime_confidence: HMM regime probability (0-1)
            current_time: Current time for time-based adjustments

        Returns:
            AdaptiveRiskParams with adjusted settings
        """
        reasons = []

        # Start with base settings
        max_lot = self.base_max_lot
        min_sl = self.base_min_sl
        max_sl = self.base_max_sl
        risk_pct = self.base_risk_percent

        # 1. Volatility-based adjustment
        vol_condition = self.classify_volatility(atr_pips)

        if vol_condition == MarketCondition.LOW_VOLATILITY:
            max_lot *= self.low_vol_lot_mult
            min_sl = max(10.0, min_sl - 5)  # Tighter SL in low vol
            reasons.append(f"LowVol(ATR={atr_pips:.1f})")

        elif vol_condition == MarketCondition.HIGH_VOLATILITY:
            max_lot *= self.high_vol_lot_mult
            # ZERO LOSING MONTHS: Keep max_sl at 10, don't increase
            reasons.append(f"HighVol(ATR={atr_pips:.1f})")

        elif vol_condition == MarketCondition.EXTREME_VOLATILITY:
            max_lot *= self.extreme_vol_lot_mult
            # ZERO LOSING MONTHS: Keep max_sl at 10, don't increase
            risk_pct *= 0.5  # Half risk in extreme conditions
            reasons.append(f"EXTREME(ATR={atr_pips:.1f})")

        # 2. Performance-based adjustment
        perf_state = self.classify_performance(current_balance)

        if perf_state == PerformanceState.DRAWDOWN:
            max_lot *= self.drawdown_lot_mult
            risk_pct *= 0.5
            reasons.append(f"DD({self._current_drawdown:.1%})")

        elif perf_state == PerformanceState.LOSING_STREAK:
            max_lot *= self.losing_streak_mult
            reasons.append(f"LoseStreak({self._consecutive_losses})")

        elif perf_state == PerformanceState.WINNING_STREAK:
            # Slight increase during winning streak (max 1.2x)
            max_lot = min(max_lot * 1.1, self.base_max_lot * 1.2)
            reasons.append(f"WinStreak({self._consecutive_wins})")

        # 3. Regime confidence adjustment
        if regime_confidence < 0.6:
            max_lot *= 0.8
            reasons.append(f"LowConf({regime_confidence:.0%})")
        elif regime_confidence > 0.85:
            max_lot = min(max_lot * 1.1, self.base_max_lot * 1.2)
            reasons.append(f"HighConf({regime_confidence:.0%})")

        # 4. Time-based adjustment
        max_loss_pct = self.max_loss_per_trade_pct

        if current_time:
            # Friday after 18:00 UTC - reduce exposure
            if current_time.weekday() == 4 and current_time.hour >= 18:
                max_lot *= 0.5
                max_loss_pct *= 0.7  # Reduce max loss too
                reasons.append("FridayClose")

            # December - special handling (anomaly month)
            if current_time.month == 12:
                if current_time.day >= 15:
                    # Late December - almost no trading
                    max_lot = min(max_lot, self.december_max_lot)
                    max_loss_pct *= 0.3
                    reasons.append("DecHoliday")
                else:
                    # Early December - conservative
                    max_lot *= 0.5
                    max_loss_pct *= 0.5
                    reasons.append("DecEarly")

        # ZERO LOSING MONTHS: Enforce absolute limits
        max_lot = max(0.01, min(max_lot, 1.0))  # Between 0.01 and 1.0
        min_sl = max(10.0, min_sl)  # At least 10 pips
        max_sl = min(10.0, max_sl)  # ZERO LOSING MONTHS: Max 10 pips (was 70)
        risk_pct = max(0.005, min(risk_pct, 0.02))  # 0.5% - 2%
        max_loss_pct = max(0.1, min(max_loss_pct, 0.5))  # ZERO LOSING MONTHS: 0.1% - 0.5%

        reason_str = " | ".join(reasons) if reasons else "Normal"

        return AdaptiveRiskParams(
            max_lot_size=round(max_lot, 2),
            min_sl_pips=round(min_sl, 1),
            max_sl_pips=round(max_sl, 1),
            risk_percent=round(risk_pct, 4),
            reason=reason_str,
            max_loss_per_trade_pct=round(max_loss_pct, 2)
        )

    def get_status(self) -> Dict:
        """Get current adaptive risk status"""
        return {
            'consecutive_wins': self._consecutive_wins,
            'consecutive_losses': self._consecutive_losses,
            'current_drawdown': f"{self._current_drawdown:.1%}",
            'peak_balance': self._peak_balance,
            'recent_trades': len(self._recent_trades),
            'recent_win_rate': sum(1 for t in self._recent_trades if t['is_win']) / len(self._recent_trades) * 100 if self._recent_trades else 0
        }

    def reset(self):
        """Reset performance tracking"""
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        self._peak_balance = 0.0
        self._current_drawdown = 0.0
        self._recent_trades = []


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR in pips from OHLC data"""
    if len(df) < period:
        return 15.0  # Default if not enough data

    high = df['high'].values if 'high' in df.columns else df['High'].values
    low = df['low'].values if 'low' in df.columns else df['Low'].values
    close = df['close'].values if 'close' in df.columns else df['Close'].values

    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    atr = np.mean(tr[-period:])

    # Convert to pips (assuming 4-digit forex pair)
    atr_pips = atr / 0.0001

    return atr_pips
