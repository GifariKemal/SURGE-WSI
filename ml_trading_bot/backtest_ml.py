"""
ML Trading Bot Backtest
=======================

Backtest ML models (Regime + Signal) with quick fixes:
- Quarter Kelly position sizing
- 2% daily loss limit
- 10% max drawdown
- ADX > 40 override

Period: January 2025 - January 2026 (13 months)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger

# ML Models
from ml_trading_bot.models import RegimeDetector, SignalClassifier
from ml_trading_bot.features import TechnicalFeatures, SessionFeatures, RegimeFeatures
from ml_trading_bot.features.profile_proxy import ProfileProxyFeatures


@dataclass
class BacktestTrade:
    """Single backtest trade"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    lot_size: float = 0.01
    pnl: float = 0.0
    pips: float = 0.0
    regime: str = ""
    signal_confidence: float = 0.0
    adx_override: bool = False
    close_reason: str = ""


@dataclass
class BacktestStats:
    """Backtest statistics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Monthly breakdown
    monthly_pnl: Dict[str, float] = field(default_factory=dict)
    monthly_trades: Dict[str, int] = field(default_factory=dict)


class MLBacktester:
    """
    ML Trading Bot Backtester

    Uses trained ML models to simulate trading with quick fixes.
    """

    def __init__(
        self,
        symbol: str = "GBPUSD",
        initial_balance: float = 10000.0,
        confidence_threshold: float = 0.55,
        kelly_fraction: float = 0.25,      # Quarter Kelly
        base_risk_pct: float = 0.01,       # 1% risk per trade
        max_daily_loss_pct: float = 0.02,  # 2% daily loss
        max_drawdown_pct: float = 0.10,    # 10% max drawdown
        adx_override_threshold: float = 40.0,
        pip_value: float = 10.0            # USD per pip per lot
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance

        # Risk parameters (with quick fixes)
        self.confidence_threshold = confidence_threshold
        self.kelly_fraction = kelly_fraction
        self.base_risk_pct = base_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.adx_override_threshold = adx_override_threshold
        self.pip_value = pip_value

        # ML Models
        self.regime_detector = None
        self.signal_classifier = None

        # Feature engineers
        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

        # Tracking
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.daily_pnl: Dict[str, float] = {}

        # State
        self.current_position: Optional[BacktestTrade] = None
        self.circuit_breaker_dates: set = set()
        self.max_dd_triggered = False

    def load_models(self) -> bool:
        """Load trained ML models"""
        try:
            models_dir = Path(__file__).parent / "saved_models"

            self.regime_detector = RegimeDetector()
            self.regime_detector.load(models_dir / "regime_hmm.pkl")

            self.signal_classifier = SignalClassifier()
            self.signal_classifier.load(models_dir / "signal_classifier.pkl")

            logger.info("ML models loaded for backtest")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to dataframe"""
        df = df.copy()
        df = self.tech_features.add_all_features(df)
        df = self.session_features.add_all_features(df)
        df = self.regime_features.add_all_features(df)
        df = self.profile_proxy.add_all_features(df)
        return df

    def calculate_position_size(
        self,
        signal_confidence: float,
        regime: int,
        atr_pips: float
    ) -> tuple:
        """
        Calculate position size with Quarter Kelly

        Returns: (lot_size, sl_pips, tp_pips, risk_amount)
        """
        # Regime multipliers
        regime_mult = {0: 1.0, 1: 0.5, 2: 0.7}.get(regime, 0.7)

        # SL/TP multipliers based on regime
        sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}.get(regime, 1.5)
        tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}.get(regime, 1.5)

        sl_pips = atr_pips * sl_mult
        tp_pips = atr_pips * tp_mult

        # Base risk amount
        base_risk = self.balance * self.base_risk_pct

        # Kelly adjustment (using confidence as proxy)
        kelly_factor = signal_confidence * self.kelly_fraction

        # Confidence scaling
        conf_factor = (signal_confidence - self.confidence_threshold) / (1 - self.confidence_threshold)
        conf_factor = 0.5 + (conf_factor * 0.5)

        # Calculate lot size
        risk_amount = base_risk * kelly_factor * regime_mult * conf_factor
        lot_size = risk_amount / (sl_pips * self.pip_value)

        # Limits
        lot_size = max(0.01, min(lot_size, 1.0))
        lot_size = round(lot_size, 2)

        actual_risk = lot_size * sl_pips * self.pip_value

        return lot_size, sl_pips, tp_pips, actual_risk

    def check_daily_loss(self, current_date: str) -> bool:
        """Check if daily loss limit hit"""
        if current_date in self.circuit_breaker_dates:
            return True

        daily_loss = self.daily_pnl.get(current_date, 0.0)
        if daily_loss < 0 and abs(daily_loss) >= self.balance * self.max_daily_loss_pct:
            self.circuit_breaker_dates.add(current_date)
            return True

        return False

    def check_max_drawdown(self) -> bool:
        """Check if max drawdown hit"""
        if self.max_dd_triggered:
            return True

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown >= self.max_drawdown_pct:
            self.max_dd_triggered = True
            logger.warning(f"MAX DRAWDOWN: {drawdown:.2%} - Stopping backtest")
            return True

        return False

    def run(self, df: pd.DataFrame) -> BacktestStats:
        """
        Run backtest on historical data (OPTIMIZED)

        Args:
            df: H1 OHLCV data with datetime index

        Returns:
            BacktestStats with results
        """
        logger.info(f"Starting backtest: {len(df)} bars")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")
        logger.info(f"Risk params: Kelly={self.kelly_fraction}, DailyLoss={self.max_daily_loss_pct:.1%}, MaxDD={self.max_drawdown_pct:.1%}")

        # Prepare features
        logger.info("Preparing features...")
        df = self.prepare_features(df)
        df = df.dropna()
        logger.info(f"After feature prep: {len(df)} bars, {len(df.columns)} features")

        # Need at least 50 bars for prediction
        if len(df) < 50:
            logger.error("Not enough data for backtest")
            return BacktestStats()

        # PRE-COMPUTE ALL PREDICTIONS (OPTIMIZATION)
        logger.info("Pre-computing ML predictions (this may take a moment)...")

        # Get all regime predictions at once
        regimes = self.regime_detector.predict(df)
        regime_probs = self.regime_detector.predict_proba(df)

        # Get all signal predictions at once
        signals = self.signal_classifier.predict(df)
        signal_probs_dict = self.signal_classifier.predict_proba(df)
        # Convert to array: max confidence for each sample
        signal_probs = np.column_stack([
            signal_probs_dict['sell'],
            signal_probs_dict['hold'],
            signal_probs_dict['buy']
        ])

        logger.info(f"Predictions computed. Running simulation...")

        # Initialize equity curve
        self.equity_curve = [self.initial_balance]

        # Progress tracking
        total_bars = len(df) - 50
        progress_interval = total_bars // 10

        # Iterate through bars (skip first 50 for warmup)
        for idx, i in enumerate(range(50, len(df))):
            # Progress logging
            if progress_interval > 0 and idx % progress_interval == 0:
                pct = idx / total_bars * 100
                logger.info(f"Progress: {pct:.0f}% ({idx}/{total_bars} bars)")

            bar = df.iloc[i]
            bar_time = df.index[i]
            current_date = bar_time.strftime('%Y-%m-%d')

            # Check max drawdown
            if self.check_max_drawdown():
                break

            # Check daily circuit breaker
            if self.check_daily_loss(current_date):
                if self.current_position:
                    self._check_position_exit(bar, bar_time)
                continue

            # Manage existing position
            if self.current_position:
                self._check_position_exit(bar, bar_time)
                continue

            # Get pre-computed predictions
            regime = regimes[i]
            regime_conf = regime_probs[i].max() if regime_probs is not None else 0.7
            regime_names = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}
            regime_name = regime_names.get(regime, 'unknown')

            # ADX Override check
            adx_override = False
            current_adx = bar.get('adx_14', 0)
            if current_adx > self.adx_override_threshold and regime == 2:
                adx_override = True
                regime = 0
                regime_name = 'trending_override'

            # Get signal
            signal = signals[i]
            signal_conf = signal_probs[i].max() if signal_probs is not None else 0.5

            # Skip if HOLD or low confidence
            if signal == 0:
                continue
            if signal_conf < self.confidence_threshold:
                continue

            # Get direction
            direction = 'BUY' if signal == 1 else 'SELL'

            # Get ATR for position sizing
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            # Calculate position size
            lot_size, sl_pips, tp_pips, risk_amount = self.calculate_position_size(
                signal_conf, regime, atr_pips
            )

            # Entry price
            entry_price = bar['close']

            # Calculate SL/TP prices
            pip_size = 0.0001
            if direction == 'BUY':
                sl_price = entry_price - sl_pips * pip_size
                tp_price = entry_price + tp_pips * pip_size
            else:
                sl_price = entry_price + sl_pips * pip_size
                tp_price = entry_price - tp_pips * pip_size

            # Open position
            self.current_position = BacktestTrade(
                entry_time=bar_time,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                regime=regime_name,
                signal_confidence=signal_conf,
                adx_override=adx_override
            )

        # Close any remaining position
        if self.current_position:
            last_bar = df.iloc[-1]
            self._close_position(last_bar['close'], df.index[-1], "END_OF_DATA")

        logger.info("Backtest complete!")
        stats = self._calculate_stats()
        return stats

    def _check_position_exit(self, bar: pd.Series, bar_time: datetime):
        """Check if position should be closed"""
        if not self.current_position:
            return

        pos = self.current_position
        high = bar['high']
        low = bar['low']
        close = bar['close']

        # Check SL
        if pos.direction == 'BUY':
            if low <= pos.sl_price:
                self._close_position(pos.sl_price, bar_time, "SL")
                return
            if high >= pos.tp_price:
                self._close_position(pos.tp_price, bar_time, "TP")
                return
        else:  # SELL
            if high >= pos.sl_price:
                self._close_position(pos.sl_price, bar_time, "SL")
                return
            if low <= pos.tp_price:
                self._close_position(pos.tp_price, bar_time, "TP")
                return

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """Close current position"""
        if not self.current_position:
            return

        pos = self.current_position
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.close_reason = reason

        # Calculate P/L
        if pos.direction == 'BUY':
            pips = (exit_price - pos.entry_price) / 0.0001
        else:
            pips = (pos.entry_price - exit_price) / 0.0001

        pnl = pips * self.pip_value * pos.lot_size
        pos.pips = pips
        pos.pnl = pnl

        # Update balance
        self.balance += pnl
        self.equity_curve.append(self.balance)

        # Track daily P/L
        current_date = exit_time.strftime('%Y-%m-%d')
        self.daily_pnl[current_date] = self.daily_pnl.get(current_date, 0.0) + pnl

        # Save trade
        self.trades.append(pos)
        self.current_position = None

    def _calculate_stats(self) -> BacktestStats:
        """Calculate backtest statistics"""
        stats = BacktestStats()

        if not self.trades:
            return stats

        stats.total_trades = len(self.trades)

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        stats.total_pnl = sum(t.pnl for t in self.trades)

        if wins:
            stats.avg_win = sum(t.pnl for t in wins) / len(wins)
            stats.largest_win = max(t.pnl for t in wins)

        if losses:
            stats.avg_loss = abs(sum(t.pnl for t in losses) / len(losses))
            stats.largest_loss = abs(min(t.pnl for t in losses))

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        stats.max_drawdown = max_dd
        stats.max_drawdown_pct = max_dd / self.initial_balance * 100

        # Monthly breakdown
        for trade in self.trades:
            month = trade.entry_time.strftime('%Y-%m')
            stats.monthly_pnl[month] = stats.monthly_pnl.get(month, 0) + trade.pnl
            stats.monthly_trades[month] = stats.monthly_trades.get(month, 0) + 1

        return stats

    def print_results(self, stats: BacktestStats):
        """Print backtest results"""
        print("\n" + "=" * 70)
        print("ML TRADING BOT BACKTEST RESULTS")
        print("=" * 70)

        print(f"\nPeriod: Jan 2025 - Jan 2026 (13 months)")
        print(f"Symbol: {self.symbol}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"Total Return: {(self.balance/self.initial_balance - 1)*100:+.2f}%")

        print("\n" + "-" * 70)
        print("RISK PARAMETERS (Quick Fixes Applied)")
        print("-" * 70)
        print(f"Kelly Fraction: {self.kelly_fraction} (Quarter Kelly)")
        print(f"Base Risk/Trade: {self.base_risk_pct:.1%}")
        print(f"Daily Loss Limit: {self.max_daily_loss_pct:.1%}")
        print(f"Max Drawdown Limit: {self.max_drawdown_pct:.1%}")
        print(f"ADX Override Threshold: {self.adx_override_threshold}")

        print("\n" + "-" * 70)
        print("TRADE STATISTICS")
        print("-" * 70)
        print(f"Total Trades: {stats.total_trades}")
        print(f"Winning Trades: {stats.winning_trades}")
        print(f"Losing Trades: {stats.losing_trades}")
        print(f"Win Rate: {stats.win_rate:.1f}%")
        print(f"Profit Factor: {stats.profit_factor:.2f}")

        print("\n" + "-" * 70)
        print("P/L ANALYSIS")
        print("-" * 70)
        print(f"Total P/L: ${stats.total_pnl:+,.2f}")
        print(f"Avg Win: ${stats.avg_win:,.2f}")
        print(f"Avg Loss: ${stats.avg_loss:,.2f}")
        print(f"Largest Win: ${stats.largest_win:,.2f}")
        print(f"Largest Loss: ${stats.largest_loss:,.2f}")
        print(f"Max Drawdown: ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)")

        print("\n" + "-" * 70)
        print("MONTHLY BREAKDOWN")
        print("-" * 70)
        print(f"{'Month':<10} {'Trades':>8} {'P/L':>12} {'Cumulative':>12}")
        print("-" * 42)

        cumulative = 0
        for month in sorted(stats.monthly_pnl.keys()):
            trades = stats.monthly_trades.get(month, 0)
            pnl = stats.monthly_pnl.get(month, 0)
            cumulative += pnl
            print(f"{month:<10} {trades:>8} ${pnl:>+10,.2f} ${cumulative:>+10,.2f}")

        # ADX Override stats
        adx_trades = [t for t in self.trades if t.adx_override]
        if adx_trades:
            print("\n" + "-" * 70)
            print("ADX OVERRIDE TRADES")
            print("-" * 70)
            print(f"Total ADX Override Trades: {len(adx_trades)}")
            adx_wins = len([t for t in adx_trades if t.pnl > 0])
            print(f"ADX Override Win Rate: {adx_wins/len(adx_trades)*100:.1f}%")
            print(f"ADX Override P/L: ${sum(t.pnl for t in adx_trades):+,.2f}")

        # Circuit breaker days
        if self.circuit_breaker_dates:
            print("\n" + "-" * 70)
            print("CIRCUIT BREAKER EVENTS")
            print("-" * 70)
            print(f"Days with circuit breaker: {len(self.circuit_breaker_dates)}")
            for date in sorted(self.circuit_breaker_dates)[:5]:
                print(f"  - {date}")
            if len(self.circuit_breaker_dates) > 5:
                print(f"  ... and {len(self.circuit_breaker_dates)-5} more")

        print("\n" + "=" * 70)


def main():
    """Run ML backtest"""
    import MetaTrader5 as mt5

    # Connect to MT5
    if not mt5.initialize(
        path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        login=10009310110,
        password="P-WyAnG8",
        server="MetaQuotes-Demo"
    ):
        logger.error("Failed to connect to MT5")
        return

    logger.info("Connected to MT5")

    # Get 13 months of H1 data (Jan 2025 - Jan 2026)
    # ~9000 H1 bars for 13 months
    from datetime import datetime

    end_date = datetime(2026, 1, 31)
    start_date = datetime(2025, 1, 1)

    logger.info(f"Fetching data: {start_date} to {end_date}")

    rates = mt5.copy_rates_range(
        "GBPUSD",
        mt5.TIMEFRAME_H1,
        start_date,
        end_date
    )

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        logger.error("Failed to get historical data")
        return

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    logger.info(f"Got {len(df)} H1 bars")

    # Create backtester with quick fixes
    backtester = MLBacktester(
        symbol="GBPUSD",
        initial_balance=10000.0,
        confidence_threshold=0.55,
        kelly_fraction=0.25,         # Quarter Kelly
        base_risk_pct=0.01,          # 1% risk
        max_daily_loss_pct=0.02,     # 2% daily loss
        max_drawdown_pct=0.10,       # 10% max drawdown
        adx_override_threshold=40.0
    )

    # Load models
    if not backtester.load_models():
        logger.error("Failed to load ML models")
        return

    # Run backtest
    stats = backtester.run(df)

    # Print results
    backtester.print_results(stats)


if __name__ == "__main__":
    main()
