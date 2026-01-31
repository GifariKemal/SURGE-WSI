"""
ML Model Backtester
===================

Backtest ML trading models with realistic simulation including:
- Transaction costs (spread + commission)
- Slippage
- Position sizing based on regime
- Walk-forward validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    lot_size: float
    pnl_pips: float
    pnl_usd: float
    regime: int
    signal_confidence: float


@dataclass
class BacktestResult:
    """Backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_pips: float
    total_pnl_usd: float
    avg_win_pips: float
    avg_loss_pips: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    initial_balance: float
    final_balance: float
    total_return_pct: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None
    monthly_returns: pd.Series = None


class MLBacktester:
    """
    Backtest ML trading models

    Features:
    - Realistic transaction costs
    - Regime-based position sizing
    - Confidence threshold filtering
    - Detailed trade logging
    """

    def __init__(
        self,
        initial_balance: float = 10000,
        risk_per_trade: float = 0.02,
        spread_pips: float = 1.0,
        commission_pips: float = 0.5,
        slippage_pips: float = 0.5,
        confidence_threshold: float = 0.55,
        pip_value: float = 10.0,  # USD per pip for 1 lot
        use_regime_sizing: bool = True
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.spread_pips = spread_pips
        self.commission_pips = commission_pips
        self.slippage_pips = slippage_pips
        self.confidence_threshold = confidence_threshold
        self.pip_value = pip_value
        self.use_regime_sizing = use_regime_sizing

        # Regime multipliers
        self.regime_multipliers = {
            0: 1.0,   # trending_low_vol
            1: 0.5,   # crisis_high_vol
            2: 0.7    # ranging_choppy
        }

    def calculate_position_size(
        self,
        balance: float,
        atr_pips: float,
        regime: int,
        confidence: float
    ) -> Tuple[float, float, float]:
        """
        Calculate position size, SL, and TP

        Returns:
            (lot_size, sl_pips, tp_pips)
        """
        # Base risk amount
        risk_amount = balance * self.risk_per_trade

        # SL/TP based on ATR
        sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}.get(regime, 1.5)
        tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}.get(regime, 2.0)

        sl_pips = atr_pips * sl_mult
        tp_pips = atr_pips * tp_mult

        # Base lot size
        base_lots = risk_amount / (sl_pips * self.pip_value)

        # Apply regime multiplier
        if self.use_regime_sizing:
            regime_mult = self.regime_multipliers.get(regime, 1.0)
        else:
            regime_mult = 1.0

        # Confidence scaling
        conf_factor = (confidence - 0.5) * 2  # 0.5->0, 1.0->1.0
        conf_factor = max(0.3, min(conf_factor, 1.0))

        final_lots = base_lots * regime_mult * conf_factor
        final_lots = max(0.01, min(final_lots, 1.0))

        return round(final_lots, 2), sl_pips, tp_pips

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        regimes: np.ndarray,
        confidences: np.ndarray,
        atr_col: str = 'atr_14'
    ) -> BacktestResult:
        """
        Run backtest simulation

        Args:
            df: DataFrame with OHLCV data
            signals: Array of signals (1=buy, -1=sell, 0=hold)
            regimes: Array of regime labels
            confidences: Array of signal confidences
            atr_col: Column name for ATR

        Returns:
            BacktestResult with detailed metrics
        """
        balance = self.initial_balance
        trades: List[Trade] = []
        equity_curve = [balance]

        position = None  # Current position
        position_entry_idx = None

        # Get price columns
        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'

        close = df[close_col].values
        high = df[high_col].values
        low = df[low_col].values

        # ATR for position sizing
        if atr_col in df.columns:
            atr = df[atr_col].values * 10000  # Convert to pips
        else:
            atr = np.full(len(df), 15.0)  # Default 15 pips

        for i in range(1, len(df)):
            current_time = df.index[i]
            current_price = close[i]
            current_high = high[i]
            current_low = low[i]

            # Check if we have an open position
            if position is not None:
                entry_price = position['entry_price']
                direction = position['direction']
                sl_price = position['sl_price']
                tp_price = position['tp_price']

                # Check stop loss
                if direction == 1:  # Long
                    hit_sl = current_low <= sl_price
                    hit_tp = current_high >= tp_price
                    exit_price = sl_price if hit_sl else (tp_price if hit_tp else None)
                else:  # Short
                    hit_sl = current_high >= sl_price
                    hit_tp = current_low <= tp_price
                    exit_price = sl_price if hit_sl else (tp_price if hit_tp else None)

                # Check for exit signal (opposite signal)
                exit_signal = (direction == 1 and signals[i] == -1) or \
                              (direction == -1 and signals[i] == 1)

                if hit_sl or hit_tp or exit_signal:
                    # Close position
                    if exit_signal and not (hit_sl or hit_tp):
                        exit_price = current_price

                    # Calculate P&L
                    if direction == 1:
                        pnl_pips = (exit_price - entry_price) * 10000
                    else:
                        pnl_pips = (entry_price - exit_price) * 10000

                    # Subtract costs
                    total_cost = self.spread_pips + self.commission_pips + self.slippage_pips
                    pnl_pips -= total_cost

                    pnl_usd = pnl_pips * self.pip_value * position['lot_size']
                    balance += pnl_usd

                    # Record trade
                    trades.append(Trade(
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        lot_size=position['lot_size'],
                        pnl_pips=pnl_pips,
                        pnl_usd=pnl_usd,
                        regime=position['regime'],
                        signal_confidence=position['confidence']
                    ))

                    position = None

            # Check for new entry signal
            if position is None and signals[i] != 0:
                confidence = confidences[i]

                # Apply confidence threshold
                if confidence >= self.confidence_threshold:
                    direction = signals[i]
                    regime = regimes[i]
                    current_atr = atr[i] if not np.isnan(atr[i]) else 15.0

                    # Calculate position size
                    lot_size, sl_pips, tp_pips = self.calculate_position_size(
                        balance, current_atr, regime, confidence
                    )

                    # Entry price with slippage
                    entry_price = current_price
                    if direction == 1:
                        entry_price += (self.slippage_pips / 10000)
                        sl_price = entry_price - (sl_pips / 10000)
                        tp_price = entry_price + (tp_pips / 10000)
                    else:
                        entry_price -= (self.slippage_pips / 10000)
                        sl_price = entry_price + (sl_pips / 10000)
                        tp_price = entry_price - (tp_pips / 10000)

                    position = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'direction': direction,
                        'lot_size': lot_size,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'regime': regime,
                        'confidence': confidence
                    }

            equity_curve.append(balance)

        # Close any remaining position at market
        if position is not None:
            exit_price = close[-1]
            direction = position['direction']

            if direction == 1:
                pnl_pips = (exit_price - position['entry_price']) * 10000
            else:
                pnl_pips = (position['entry_price'] - exit_price) * 10000

            pnl_pips -= (self.spread_pips + self.commission_pips)
            pnl_usd = pnl_pips * self.pip_value * position['lot_size']
            balance += pnl_usd

            trades.append(Trade(
                entry_time=position['entry_time'],
                exit_time=df.index[-1],
                direction=direction,
                entry_price=position['entry_price'],
                exit_price=exit_price,
                lot_size=position['lot_size'],
                pnl_pips=pnl_pips,
                pnl_usd=pnl_usd,
                regime=position['regime'],
                signal_confidence=position['confidence']
            ))
            equity_curve[-1] = balance

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, df.index)

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        index: pd.DatetimeIndex
    ) -> BacktestResult:
        """Calculate backtest metrics"""

        if len(trades) == 0:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl_pips=0,
                total_pnl_usd=0,
                avg_win_pips=0,
                avg_loss_pips=0,
                profit_factor=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                total_return_pct=0,
                trades=[],
                equity_curve=pd.Series(equity_curve, index=index[:len(equity_curve)])
            )

        # Basic stats
        winning = [t for t in trades if t.pnl_pips > 0]
        losing = [t for t in trades if t.pnl_pips <= 0]

        total_pnl_pips = sum(t.pnl_pips for t in trades)
        total_pnl_usd = sum(t.pnl_usd for t in trades)

        win_rate = len(winning) / len(trades) if trades else 0

        avg_win = np.mean([t.pnl_pips for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t.pnl_pips for t in losing])) if losing else 0

        gross_profit = sum(t.pnl_pips for t in winning)
        gross_loss = abs(sum(t.pnl_pips for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Equity curve analysis
        equity_series = pd.Series(equity_curve, index=index[:len(equity_curve)])

        # Max drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Sharpe ratio (annualized)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Hourly data
        else:
            sharpe = 0

        # Monthly returns
        monthly = equity_series.resample('M').last().pct_change().dropna()

        final_balance = equity_curve[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance

        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl_pips=total_pnl_pips,
            total_pnl_usd=total_pnl_usd,
            avg_win_pips=avg_win,
            avg_loss_pips=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return_pct=total_return,
            trades=trades,
            equity_curve=equity_series,
            monthly_returns=monthly
        )


def run_full_backtest(
    test_period: str = "2024-01-01",
    confidence_threshold: float = 0.55
) -> BacktestResult:
    """
    Run full backtest with saved models

    Args:
        test_period: Start of test period
        confidence_threshold: Min confidence to trade

    Returns:
        BacktestResult
    """
    from ml_trading_bot.models.regime_detector import RegimeDetector
    from ml_trading_bot.models.signal_classifier import SignalClassifier
    from ml_trading_bot.training.data_loader import DataLoader
    from ml_trading_bot.features.technical import TechnicalFeatures
    from ml_trading_bot.features.session import SessionFeatures
    from ml_trading_bot.features.regime import RegimeFeatures

    print("=" * 60)
    print("ML MODEL BACKTEST")
    print("=" * 60)

    # Load models
    print("\n1. Loading models...")
    models_dir = Path(__file__).parent.parent / "saved_models"

    detector = RegimeDetector()
    detector.load(str(models_dir / "regime_hmm.pkl"))

    classifier = SignalClassifier()
    classifier.load(str(models_dir / "signal_classifier.pkl"))

    # Load test data
    print("\n2. Loading test data...")
    loader = DataLoader()
    df = loader.load_ohlcv_sync("2023-06-01", "2026-01-30")  # Include warmup

    daily_profiles = loader.load_daily_profiles()
    df = loader.merge_ohlcv_with_daily_profiles(df, daily_profiles)

    # Compute features
    print("\n3. Computing features...")
    tech = TechnicalFeatures()
    df = tech.add_all_features(df)

    session = SessionFeatures()
    df = session.add_all_features(df)

    regime = RegimeFeatures()
    df = regime.add_all_features(df)

    print(f"   Total data: {len(df)} bars")

    # Filter to test period
    test_start = pd.Timestamp(test_period, tz='UTC') if df.index.tz else pd.Timestamp(test_period)
    df_test = df[df.index >= test_start].copy()
    print(f"   Test period: {len(df_test)} bars ({df_test.index[0]} to {df_test.index[-1]})")

    # Generate predictions
    print("\n4. Generating predictions...")
    regimes = detector.predict(df_test)
    signals = classifier.predict(df_test)
    proba = classifier.predict_proba(df_test)

    # Get confidence for each signal
    confidences = np.zeros(len(signals))
    for i in range(len(signals)):
        if signals[i] == 1:
            confidences[i] = proba['buy'][i]
        elif signals[i] == -1:
            confidences[i] = proba['sell'][i]
        else:
            confidences[i] = proba['hold'][i]

    signal_counts = {
        'buy': sum(signals == 1),
        'sell': sum(signals == -1),
        'hold': sum(signals == 0)
    }
    print(f"   Signals: BUY={signal_counts['buy']}, SELL={signal_counts['sell']}, HOLD={signal_counts['hold']}")

    # Run backtest
    print("\n5. Running backtest simulation...")
    backtester = MLBacktester(
        initial_balance=10000,
        risk_per_trade=0.02,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.5,
        confidence_threshold=confidence_threshold,
        use_regime_sizing=True
    )

    result = backtester.run_backtest(
        df_test,
        signals,
        regimes,
        confidences,
        atr_col='atr_14'
    )

    return result


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report"""

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Initial Balance:    ${result.initial_balance:,.2f}")
    print(f"Final Balance:      ${result.final_balance:,.2f}")
    print(f"Total Return:       {result.total_return_pct:+.2%}")
    print(f"Total P&L (USD):    ${result.total_pnl_usd:+,.2f}")
    print(f"Total P&L (pips):   {result.total_pnl_pips:+,.1f}")

    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades:       {result.total_trades}")
    print(f"Winning Trades:     {result.winning_trades}")
    print(f"Losing Trades:      {result.losing_trades}")
    print(f"Win Rate:           {result.win_rate:.1%}")
    print(f"Avg Win (pips):     {result.avg_win_pips:+.1f}")
    print(f"Avg Loss (pips):    {result.avg_loss_pips:.1f}")
    print(f"Profit Factor:      {result.profit_factor:.2f}")

    print("\n--- RISK METRICS ---")
    print(f"Max Drawdown:       {result.max_drawdown_pct:.2%}")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")

    if result.monthly_returns is not None and len(result.monthly_returns) > 0:
        print("\n--- MONTHLY RETURNS ---")
        for date, ret in result.monthly_returns.items():
            month_str = date.strftime("%Y-%m")
            bar = "#" * int(abs(ret) * 100) if not np.isnan(ret) else ""
            sign = "+" if ret > 0 else ""
            print(f"  {month_str}: {sign}{ret:.2%} |{bar}")

    # Trade breakdown by regime
    if result.trades:
        print("\n--- TRADES BY REGIME ---")
        regime_names = {0: "TREND", 1: "CRISIS", 2: "RANGE"}
        for regime_id in [0, 1, 2]:
            regime_trades = [t for t in result.trades if t.regime == regime_id]
            if regime_trades:
                wins = sum(1 for t in regime_trades if t.pnl_pips > 0)
                total_pnl = sum(t.pnl_pips for t in regime_trades)
                wr = wins / len(regime_trades) if regime_trades else 0
                print(f"  {regime_names[regime_id]:6s}: {len(regime_trades):3d} trades, "
                      f"WR={wr:.0%}, P&L={total_pnl:+.1f} pips")

    # Direction breakdown
    if result.trades:
        print("\n--- TRADES BY DIRECTION ---")
        long_trades = [t for t in result.trades if t.direction == 1]
        short_trades = [t for t in result.trades if t.direction == -1]

        if long_trades:
            long_wins = sum(1 for t in long_trades if t.pnl_pips > 0)
            long_pnl = sum(t.pnl_pips for t in long_trades)
            print(f"  LONG:  {len(long_trades):3d} trades, "
                  f"WR={long_wins/len(long_trades):.0%}, P&L={long_pnl:+.1f} pips")

        if short_trades:
            short_wins = sum(1 for t in short_trades if t.pnl_pips > 0)
            short_pnl = sum(t.pnl_pips for t in short_trades)
            print(f"  SHORT: {len(short_trades):3d} trades, "
                  f"WR={short_wins/len(short_trades):.0%}, P&L={short_pnl:+.1f} pips")


if __name__ == "__main__":
    # Run backtest
    result = run_full_backtest(
        test_period="2024-01-01",
        confidence_threshold=0.55
    )

    # Print report
    print_backtest_report(result)

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE!")
    print("=" * 60)
