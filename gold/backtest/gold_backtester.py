"""Gold (XAUUSD) Backtester
==========================

Specialized backtester for XAUUSD with Gold-optimized parameters.

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from gold.config.gold_settings import GoldConfig, get_gold_config
from src.analysis.kalman_filter import KalmanNoiseReducer
from src.analysis.regime_detector import HMMRegimeDetector
from src.analysis.poi_detector import POIDetector


@dataclass
class GoldTrade:
    """Trade record"""
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: float = 0
    exit_time: datetime = None
    sl: float = 0
    tp: float = 0
    lot_size: float = 0.01
    pnl: float = 0
    pnl_pips: float = 0
    exit_reason: str = ""
    regime: str = ""
    poi_type: str = ""


@dataclass
class GoldBacktestResult:
    """Backtest result"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0
    total_pnl: float = 0
    total_pnl_pips: float = 0
    pnl_pct: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    avg_win_pips: float = 0
    avg_loss_pips: float = 0
    profit_factor: float = 0
    max_drawdown: float = 0
    max_drawdown_pct: float = 0
    final_balance: float = 0
    sharpe_ratio: float = 0
    trades: List[GoldTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    monthly_pnl: Dict[str, float] = field(default_factory=dict)


class GoldBacktester:
    """Backtester optimized for XAUUSD"""

    def __init__(
        self,
        config: GoldConfig = None,
        initial_balance: float = 10000.0
    ):
        self.config = config or get_gold_config()
        self.initial_balance = initial_balance
        self.balance = initial_balance

        # Initialize components
        self.kalman = KalmanNoiseReducer()
        self.regime_detector = HMMRegimeDetector()
        self.poi_detector = POIDetector(
            swing_length=10,
            ob_min_strength=self.config.poi.min_ob_strength,
            fvg_min_pips=self.config.poi.fvg_min_pips,
            max_poi_age_bars=self.config.poi.max_ob_age_bars,
            pip_size=self.config.symbol.pip_size  # Gold uses 0.1
        )

        # State
        self.trades: List[GoldTrade] = []
        self.equity_curve: List[float] = [initial_balance]
        self.position: Optional[GoldTrade] = None

        # Settings from config
        self.pip_size = self.config.symbol.pip_size
        self.spread_pips = self.config.symbol.spread_pips
        self.pip_value = self.config.risk.pip_value_per_lot

    def reset(self):
        """Reset backtester state"""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.position = None
        self.kalman.reset()
        self.regime_detector.reset()
        self.poi_detector.reset()

    def _is_trading_session(self, dt: datetime) -> bool:
        """Check if current time is in trading session"""
        hour = dt.hour
        weekday = dt.weekday()

        # Skip weekends
        if weekday >= 5:
            return False

        # Skip Friday late
        if weekday == 4 and hour >= 18 and self.config.session.skip_friday_late:
            return False

        # Check primary session (New York for Gold)
        start = self.config.session.primary_session_start
        end = self.config.session.primary_session_end

        return start <= hour < end

    def _calculate_lot_size(self, sl_pips: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.balance * self.config.risk.risk_per_trade
        lot_size = risk_amount / (sl_pips * self.pip_value)

        # Clamp to limits
        lot_size = max(self.config.risk.min_lot_size, lot_size)
        lot_size = min(self.config.risk.max_lot_size, lot_size)

        return round(lot_size, 2)

    def _check_activity(self, velocity: float, atr_pips: float) -> bool:
        """Check if market activity is sufficient"""
        if not self.config.intel_filter.enabled:
            return True

        velocity_pips = abs(velocity) / self.pip_size

        # Check velocity
        if velocity_pips < self.config.intel_filter.min_velocity_pips * 0.5:
            return False

        # Check ATR
        if atr_pips < self.config.intel_filter.min_atr_pips * 0.5:
            return False

        # Calculate activity score
        vel_score = min(30, velocity_pips / self.config.intel_filter.high_velocity_pips * 30)
        atr_score = min(30, atr_pips / self.config.intel_filter.high_atr_pips * 30)
        total_score = vel_score + atr_score

        return total_score >= self.config.intel_filter.activity_threshold

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """Close current position"""
        if not self.position:
            return

        pos = self.position

        # Calculate P&L
        if pos.direction == "BUY":
            pnl_pips = (exit_price - pos.entry_price - self.spread_pips * self.pip_size) / self.pip_size
        else:
            pnl_pips = (pos.entry_price - exit_price - self.spread_pips * self.pip_size) / self.pip_size

        pnl = pnl_pips * pos.lot_size * self.pip_value

        # Update position
        pos.exit_price = exit_price
        pos.exit_time = exit_time
        pos.pnl = pnl
        pos.pnl_pips = pnl_pips
        pos.exit_reason = reason

        # Update balance
        self.balance += pnl
        self.equity_curve.append(self.balance)
        self.trades.append(pos)

        # Clear position
        self.position = None

    def run(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> GoldBacktestResult:
        """Run backtest

        Args:
            htf_df: Higher timeframe data (H4)
            ltf_df: Lower timeframe data (M15)
            start_date: Start of backtest period
            end_date: End of backtest period

        Returns:
            GoldBacktestResult with detailed results
        """
        self.reset()

        if htf_df is None or ltf_df is None or htf_df.empty or ltf_df.empty:
            return self._empty_result()

        # Normalize column names to lowercase
        htf_df = htf_df.copy()
        ltf_df = ltf_df.copy()
        htf_df.columns = [c.lower() for c in htf_df.columns]
        ltf_df.columns = [c.lower() for c in ltf_df.columns]

        # Filter by date range if specified
        if start_date:
            htf_df = htf_df[htf_df.index >= start_date]
            ltf_df = ltf_df[ltf_df.index >= start_date]
        if end_date:
            htf_df = htf_df[htf_df.index <= end_date]
            ltf_df = ltf_df[ltf_df.index <= end_date]

        # Warmup
        warmup = min(self.config.warmup_bars_htf, len(htf_df) // 4)
        for i in range(warmup):
            price = htf_df.iloc[i]['close']
            state = self.kalman.update(price)
            if state:
                self.regime_detector.update(state.smoothed_price)

        # Calculate ATR for activity check
        htf_df = htf_df.copy()
        htf_df['tr'] = np.maximum(
            htf_df['high'] - htf_df['low'],
            np.maximum(
                abs(htf_df['high'] - htf_df['close'].shift(1)),
                abs(htf_df['low'] - htf_df['close'].shift(1))
            )
        )
        htf_df['atr'] = htf_df['tr'].rolling(14).mean()

        # Main backtest loop
        for i in range(warmup, len(htf_df)):
            bar = htf_df.iloc[i]
            bar_time = htf_df.index[i]
            price = bar['close']
            high = bar['high']
            low = bar['low']
            atr = bar['atr'] if not pd.isna(bar['atr']) else self.config.symbol.avg_atr_pips * self.pip_size

            # Update Kalman filter
            state = self.kalman.update(price)
            if not state:
                continue

            smoothed = state.smoothed_price
            velocity = state.velocity

            # Update regime
            regime_info = self.regime_detector.update(smoothed)

            # Update POI detection
            if i > 30:
                poi_data = htf_df.iloc[i-30:i+1].copy()
                self.poi_detector.detect(poi_data)

            # Check existing position
            if self.position:
                # Check SL/TP
                if self.position.direction == "BUY":
                    if low <= self.position.sl:
                        self._close_position(self.position.sl, bar_time, "SL")
                    elif high >= self.position.tp:
                        self._close_position(self.position.tp, bar_time, "TP")
                else:  # SELL
                    if high >= self.position.sl:
                        self._close_position(self.position.sl, bar_time, "SL")
                    elif low <= self.position.tp:
                        self._close_position(self.position.tp, bar_time, "TP")
                continue

            # Check for new entry
            # 1. Check trading session
            if not self._is_trading_session(bar_time):
                continue

            # 2. Check regime
            if not regime_info or not regime_info.is_tradeable:
                continue

            direction = regime_info.bias
            if direction == "NONE":
                continue

            # 3. Check activity
            atr_pips = atr / self.pip_size
            if not self._check_activity(velocity, atr_pips):
                continue

            # 4. Check POI
            poi_result = self.poi_detector.last_result
            if not poi_result:
                continue

            tolerance = self.config.poi.tolerance_pips
            at_poi, poi_info = poi_result.price_at_poi(price, direction, tolerance_pips=tolerance, pip_size=self.pip_size)
            if not at_poi:
                continue

            # Calculate entry parameters
            sl_pips = min(self.config.risk.max_sl_pips, max(self.config.risk.min_sl_pips, atr_pips))
            tp_pips = sl_pips * self.config.risk.min_rr_ratio
            lot_size = self._calculate_lot_size(sl_pips)

            # Create position
            if direction == "BUY":
                entry = price + self.spread_pips * self.pip_size
                sl = entry - sl_pips * self.pip_size
                tp = entry + tp_pips * self.pip_size
            else:
                entry = price
                sl = entry + sl_pips * self.pip_size
                tp = entry - tp_pips * self.pip_size

            self.position = GoldTrade(
                direction=direction,
                entry_price=entry,
                entry_time=bar_time,
                sl=sl,
                tp=tp,
                lot_size=lot_size,
                regime=str(regime_info.regime),
                poi_type=poi_info.get('type', 'unknown') if poi_info else 'unknown'
            )

        # Close any remaining position at end
        if self.position:
            final_price = htf_df.iloc[-1]['close']
            self._close_position(final_price, htf_df.index[-1], "EOD")

        return self._calculate_results()

    def _calculate_results(self) -> GoldBacktestResult:
        """Calculate backtest results"""
        if not self.trades:
            return self._empty_result()

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pips = sum(t.pnl_pips for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        avg_win_pips = np.mean([t.pnl_pips for t in wins]) if wins else 0
        avg_loss_pips = np.mean([t.pnl_pips for t in losses]) if losses else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        peak = self.initial_balance
        max_dd = 0
        max_dd_pct = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak * 100
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        # Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

        # Monthly P&L
        monthly_pnl = {}
        for trade in self.trades:
            month_key = trade.exit_time.strftime('%Y-%m') if trade.exit_time else trade.entry_time.strftime('%Y-%m')
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade.pnl

        return GoldBacktestResult(
            total_trades=len(self.trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pips=total_pnl_pips,
            pnl_pct=total_pnl / self.initial_balance * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pips=avg_win_pips,
            avg_loss_pips=avg_loss_pips,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            final_balance=self.balance,
            sharpe_ratio=sharpe,
            trades=self.trades,
            equity_curve=self.equity_curve,
            monthly_pnl=monthly_pnl
        )

    def _empty_result(self) -> GoldBacktestResult:
        """Return empty result"""
        return GoldBacktestResult(
            final_balance=self.initial_balance,
            equity_curve=[self.initial_balance]
        )


def print_result(result: GoldBacktestResult, title: str = "BACKTEST RESULT"):
    """Print backtest result"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()
    print(f"Total Trades: {result.total_trades}")
    print(f"Wins: {result.wins} | Losses: {result.losses}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print()
    print(f"Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pips:,.1f} pips)")
    print(f"P&L %: {result.pnl_pct:.2f}%")
    print()
    print(f"Avg Win: ${result.avg_win:.2f} ({result.avg_win_pips:.1f} pips)")
    print(f"Avg Loss: ${result.avg_loss:.2f} ({result.avg_loss_pips:.1f} pips)")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print()
    print(f"Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print()
    print(f"Final Balance: ${result.final_balance:,.2f}")
    print()

    if result.monthly_pnl:
        print("Monthly P&L:")
        for month, pnl in sorted(result.monthly_pnl.items()):
            emoji = "+" if pnl > 0 else ""
            print(f"  {month}: {emoji}${pnl:.2f}")
    print()
