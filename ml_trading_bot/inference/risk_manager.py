"""
Risk Manager
============

Dynamic position sizing and risk controls based on:
- Kelly Criterion (Half Kelly)
- Market regime
- Signal confidence
- Account state
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, date


@dataclass
class TradeParameters:
    """Parameters for a trade"""
    lot_size: float
    stop_loss_pips: float
    take_profit_pips: float
    risk_amount: float
    risk_pct: float
    regime: int
    confidence: float
    approved: bool
    reason: str


class RiskManager:
    """
    Dynamic risk management for ML Trading Bot

    Features:
    - Kelly Criterion position sizing (Half Kelly by default)
    - Regime-based position scaling
    - Confidence-based adjustments
    - Daily loss limits
    - Maximum position limits
    """

    # Regime-based risk multipliers
    REGIME_MULTIPLIERS = {
        0: 1.0,   # trending_low_vol: full size
        1: 0.5,   # crisis_high_vol: half size
        2: 0.7    # ranging_choppy: reduced
    }

    # Regime-based SL/TP multipliers (ATR-based)
    SL_MULTIPLIERS = {
        0: 1.2,   # Tighter in trending
        1: 2.0,   # Wider in crisis
        2: 1.5    # Medium in ranging
    }

    TP_MULTIPLIERS = {
        0: 2.0,   # Larger target in trending
        1: 2.5,   # Larger target in crisis (if entering)
        2: 1.5    # Smaller target in ranging
    }

    def __init__(
        self,
        account_risk_pct: float = 0.01,       # Reduced from 2% to 1%
        max_daily_loss_pct: float = 0.02,     # Reduced from 5% to 2%
        kelly_fraction: float = 0.25,         # Quarter Kelly (was 0.5)
        min_lot_size: float = 0.01,
        max_lot_size: float = 1.0,
        max_positions: int = 1,               # One position at a time
        min_confidence: float = 0.55,
        pip_value: float = 10.0  # USD per pip for 1 lot GBPUSD
    ):
        """
        Initialize Risk Manager

        Args:
            account_risk_pct: Max risk per trade (default 2%)
            max_daily_loss_pct: Max daily loss limit (default 5%)
            kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly)
            min_lot_size: Minimum position size
            max_lot_size: Maximum position size
            max_positions: Maximum concurrent positions
            min_confidence: Minimum signal confidence to trade
            pip_value: USD value per pip for 1 standard lot
        """
        self.account_risk_pct = account_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.kelly_fraction = kelly_fraction
        self.min_lot_size = min_lot_size
        self.max_lot_size = max_lot_size
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.pip_value = pip_value

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = date.today()
        self.current_positions = 0

    def _reset_daily_if_needed(self):
        """Reset daily counters if new day"""
        today = date.today()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction

        Kelly % = W - [(1-W) / R]
        where W = win rate, R = win/loss ratio

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade (pips or currency)
            avg_loss: Average losing trade (pips or currency, positive)

        Returns:
            Kelly fraction (capped at kelly_fraction)
        """
        if avg_loss <= 0 or win_rate <= 0:
            return self.kelly_fraction * 0.5  # Conservative default

        R = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / R)

        # Apply fraction (Half Kelly, Quarter Kelly, etc.)
        adjusted_kelly = kelly * self.kelly_fraction

        # Cap between 0 and kelly_fraction
        return max(0, min(adjusted_kelly, self.kelly_fraction))

    def calculate_position_size(
        self,
        account_balance: float,
        signal_confidence: float,
        regime: int,
        atr_pips: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> TradeParameters:
        """
        Calculate position size with all risk adjustments

        Args:
            account_balance: Current account balance
            signal_confidence: Model confidence (0-1)
            regime: Current market regime (0, 1, 2)
            atr_pips: Current ATR in pips
            win_rate: Historical win rate (optional)
            avg_win: Average win in pips (optional)
            avg_loss: Average loss in pips (optional)

        Returns:
            TradeParameters with position sizing details
        """
        self._reset_daily_if_needed()

        # Check daily loss limit
        if abs(self.daily_pnl) >= account_balance * self.max_daily_loss_pct:
            return TradeParameters(
                lot_size=0,
                stop_loss_pips=0,
                take_profit_pips=0,
                risk_amount=0,
                risk_pct=0,
                regime=regime,
                confidence=signal_confidence,
                approved=False,
                reason="Daily loss limit reached"
            )

        # Check position limit
        if self.current_positions >= self.max_positions:
            return TradeParameters(
                lot_size=0,
                stop_loss_pips=0,
                take_profit_pips=0,
                risk_amount=0,
                risk_pct=0,
                regime=regime,
                confidence=signal_confidence,
                approved=False,
                reason="Maximum positions reached"
            )

        # Check confidence threshold
        if signal_confidence < self.min_confidence:
            return TradeParameters(
                lot_size=0,
                stop_loss_pips=0,
                take_profit_pips=0,
                risk_amount=0,
                risk_pct=0,
                regime=regime,
                confidence=signal_confidence,
                approved=False,
                reason=f"Confidence {signal_confidence:.1%} below threshold {self.min_confidence:.1%}"
            )

        # Calculate SL and TP
        sl_multiplier = self.SL_MULTIPLIERS.get(regime, 1.5)
        tp_multiplier = self.TP_MULTIPLIERS.get(regime, 2.0)

        stop_loss_pips = atr_pips * sl_multiplier
        take_profit_pips = atr_pips * tp_multiplier

        # Base position from account risk
        base_risk_amount = account_balance * self.account_risk_pct
        base_lots = base_risk_amount / (stop_loss_pips * self.pip_value)

        # Kelly adjustment (if stats provided)
        if win_rate and avg_win and avg_loss:
            kelly_factor = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        else:
            # Use confidence as proxy for Kelly
            kelly_factor = signal_confidence * self.kelly_fraction

        # Regime adjustment
        regime_multiplier = self.REGIME_MULTIPLIERS.get(regime, 1.0)

        # Confidence scaling (linear from min_confidence to 1.0)
        confidence_factor = (signal_confidence - self.min_confidence) / (1 - self.min_confidence)
        confidence_factor = 0.5 + (confidence_factor * 0.5)  # Scale 0.5 to 1.0

        # Calculate final lot size
        final_lots = base_lots * kelly_factor * regime_multiplier * confidence_factor

        # Apply limits
        final_lots = max(self.min_lot_size, min(final_lots, self.max_lot_size))
        final_lots = round(final_lots, 2)

        # Calculate actual risk
        actual_risk = final_lots * stop_loss_pips * self.pip_value
        actual_risk_pct = actual_risk / account_balance

        return TradeParameters(
            lot_size=final_lots,
            stop_loss_pips=round(stop_loss_pips, 1),
            take_profit_pips=round(take_profit_pips, 1),
            risk_amount=round(actual_risk, 2),
            risk_pct=round(actual_risk_pct, 4),
            regime=regime,
            confidence=signal_confidence,
            approved=True,
            reason="Trade approved"
        )

    def record_trade_result(self, pnl: float):
        """
        Record trade result for daily tracking

        Args:
            pnl: Profit/loss in account currency
        """
        self._reset_daily_if_needed()
        self.daily_pnl += pnl
        self.daily_trades += 1

    def open_position(self):
        """Record position opened"""
        self.current_positions += 1

    def close_position(self):
        """Record position closed"""
        self.current_positions = max(0, self.current_positions - 1)

    def get_status(self, account_balance: float) -> Dict[str, Any]:
        """
        Get current risk status

        Args:
            account_balance: Current account balance

        Returns:
            Dict with risk status
        """
        self._reset_daily_if_needed()

        remaining_risk = (account_balance * self.max_daily_loss_pct) - abs(self.daily_pnl)

        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'daily_risk_used_pct': abs(self.daily_pnl) / account_balance if account_balance > 0 else 0,
            'daily_risk_remaining': max(0, remaining_risk),
            'current_positions': self.current_positions,
            'can_trade': (
                remaining_risk > 0 and
                self.current_positions < self.max_positions
            )
        }

    def assess_regime_risk(self, regime: int) -> Dict[str, Any]:
        """
        Get risk assessment for a regime

        Args:
            regime: Market regime (0, 1, 2)

        Returns:
            Dict with regime risk info
        """
        regime_names = {
            0: 'trending_low_vol',
            1: 'crisis_high_vol',
            2: 'ranging_choppy'
        }

        return {
            'regime': regime,
            'regime_name': regime_names.get(regime, 'unknown'),
            'position_multiplier': self.REGIME_MULTIPLIERS.get(regime, 1.0),
            'sl_multiplier': self.SL_MULTIPLIERS.get(regime, 1.5),
            'tp_multiplier': self.TP_MULTIPLIERS.get(regime, 2.0),
            'recommendation': {
                0: 'Normal trading, trend following strategies',
                1: 'Reduce size or skip, high volatility',
                2: 'Reduced size, consider mean reversion'
            }.get(regime, 'Unknown')
        }


def calculate_trade_params(
    account_balance: float,
    signal_confidence: float,
    regime: int,
    atr_pips: float,
    **kwargs
) -> TradeParameters:
    """
    Convenience function to calculate trade parameters

    Usage:
        from ml_trading_bot.inference.risk_manager import calculate_trade_params
        params = calculate_trade_params(10000, 0.7, 0, 15.5)
    """
    rm = RiskManager(**kwargs)
    return rm.calculate_position_size(
        account_balance, signal_confidence, regime, atr_pips
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Risk Manager")
    print("=" * 60)

    rm = RiskManager(
        account_risk_pct=0.02,
        max_daily_loss_pct=0.05,
        kelly_fraction=0.5
    )

    # Test scenarios
    scenarios = [
        {"name": "Normal trending", "confidence": 0.75, "regime": 0, "atr": 15.0},
        {"name": "High vol crisis", "confidence": 0.70, "regime": 1, "atr": 45.0},
        {"name": "Ranging choppy", "confidence": 0.65, "regime": 2, "atr": 12.0},
        {"name": "Low confidence", "confidence": 0.52, "regime": 0, "atr": 15.0},
        {"name": "Very high confidence", "confidence": 0.90, "regime": 0, "atr": 15.0},
    ]

    account = 10000

    print(f"\nAccount Balance: ${account:,.2f}")
    print(f"Base Risk per Trade: {rm.account_risk_pct:.1%}")
    print(f"Kelly Fraction: {rm.kelly_fraction}")
    print()

    for s in scenarios:
        params = rm.calculate_position_size(
            account_balance=account,
            signal_confidence=s['confidence'],
            regime=s['regime'],
            atr_pips=s['atr']
        )

        print(f"{s['name']}:")
        print(f"  Confidence: {s['confidence']:.0%}, Regime: {s['regime']}, ATR: {s['atr']} pips")
        print(f"  Approved: {params.approved}")
        if params.approved:
            print(f"  Lot Size: {params.lot_size}")
            print(f"  SL: {params.stop_loss_pips} pips, TP: {params.take_profit_pips} pips")
            print(f"  Risk: ${params.risk_amount:.2f} ({params.risk_pct:.2%})")
        else:
            print(f"  Reason: {params.reason}")
        print()

    # Test regime assessments
    print("\nRegime Risk Assessments:")
    for regime in [0, 1, 2]:
        assessment = rm.assess_regime_risk(regime)
        print(f"  Regime {regime} ({assessment['regime_name']}):")
        print(f"    Position mult: {assessment['position_multiplier']}")
        print(f"    SL mult: {assessment['sl_multiplier']}, TP mult: {assessment['tp_multiplier']}")
        print(f"    Recommendation: {assessment['recommendation']}")

    print("\n" + "=" * 60)
    print("Risk manager test complete!")
    print("=" * 60)
