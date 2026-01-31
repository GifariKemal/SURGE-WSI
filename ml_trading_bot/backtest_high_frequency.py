"""
ML Trading Bot - High Frequency Backtest
=========================================

Target: 120+ trades dalam 13 bulan
Strategi: Relax filters untuk lebih banyak signal

Changes dari Zero-Loss:
- Multiple trading hours (London + NY session)
- All regimes allowed (dengan position sizing adjustment)
- Lower confidence threshold
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
    close_reason: str = ""


class HighFrequencyBacktester:
    """
    High Frequency ML Backtester

    Target: 120+ trades in 13 months (~10 trades/month)
    """

    # Trading hours (UTC) - Extended: Asian late + London + NY
    TRADING_HOURS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Regime settings
    REGIME_NAMES = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}

    # Position size multipliers per regime
    REGIME_MULTIPLIERS = {
        0: 1.0,   # trending - full size
        1: 0.7,   # crisis - reduced (was 0.5)
        2: 0.5    # ranging - half size (was skip)
    }

    # SL/TP multipliers
    SL_MULTIPLIERS = {0: 1.5, 1: 2.0, 2: 1.2}
    TP_MULTIPLIERS = {0: 2.0, 1: 2.5, 2: 1.5}

    def __init__(
        self,
        symbol: str = "GBPUSD",
        initial_balance: float = 10000.0,
        confidence_threshold: float = 0.50,      # Lower threshold
        base_risk_pct: float = 0.01,             # 1% per trade
        max_daily_loss_pct: float = 0.03,        # 3% daily limit
        max_drawdown_pct: float = 0.15,          # 15% max DD
        pip_value: float = 10.0,
        min_atr_pips: float = 5.0,               # Lower ATR filter
        cooldown_hours: int = 0                  # No cooldown
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance

        self.confidence_threshold = confidence_threshold
        self.base_risk_pct = base_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.pip_value = pip_value
        self.min_atr_pips = min_atr_pips
        self.cooldown_hours = cooldown_hours

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
        self.last_trade_time: Optional[datetime] = None
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

            logger.info("ML models loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features"""
        df = df.copy()
        df = self.tech_features.add_all_features(df)
        df = self.session_features.add_all_features(df)
        df = self.regime_features.add_all_features(df)
        df = self.profile_proxy.add_all_features(df)
        return df

    def calculate_position_size(self, signal_conf: float, regime: int, atr_pips: float) -> tuple:
        """Calculate position size"""
        regime_mult = self.REGIME_MULTIPLIERS.get(regime, 0.5)
        sl_mult = self.SL_MULTIPLIERS.get(regime, 1.5)
        tp_mult = self.TP_MULTIPLIERS.get(regime, 1.5)

        sl_pips = atr_pips * sl_mult
        tp_pips = atr_pips * tp_mult

        # Confidence factor
        conf_factor = (signal_conf - 0.5) / 0.5  # 0 to 1
        conf_factor = max(0.3, min(conf_factor, 1.0))

        # Risk amount
        risk_amount = self.balance * self.base_risk_pct * regime_mult * conf_factor
        lot_size = risk_amount / (sl_pips * self.pip_value)

        # Limits
        lot_size = max(0.01, min(lot_size, 0.5))
        lot_size = round(lot_size, 2)

        return lot_size, sl_pips, tp_pips

    def check_daily_loss(self, current_date: str) -> bool:
        """Check daily loss limit"""
        if current_date in self.circuit_breaker_dates:
            return True
        daily_loss = self.daily_pnl.get(current_date, 0.0)
        if daily_loss < 0 and abs(daily_loss) >= self.balance * self.max_daily_loss_pct:
            self.circuit_breaker_dates.add(current_date)
            return True
        return False

    def check_max_drawdown(self) -> bool:
        """Check max drawdown"""
        if self.max_dd_triggered:
            return True
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown >= self.max_drawdown_pct:
            self.max_dd_triggered = True
            logger.warning(f"MAX DRAWDOWN: {drawdown:.2%}")
            return True
        return False

    def run(self, df: pd.DataFrame) -> dict:
        """Run backtest"""
        logger.info(f"Starting HIGH FREQUENCY backtest: {len(df)} bars")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Trading hours: {self.TRADING_HOURS}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")

        # Prepare features
        logger.info("Preparing features...")
        df = self.prepare_features(df)
        df = df.dropna()
        logger.info(f"After prep: {len(df)} bars, {len(df.columns)} features")

        if len(df) < 50:
            logger.error("Not enough data")
            return {}

        # Pre-compute predictions
        logger.info("Pre-computing ML predictions...")
        regimes = self.regime_detector.predict(df)
        regime_probs = self.regime_detector.predict_proba(df)
        signal_probs = self.signal_classifier.predict_proba(df)

        # Use probability-based signals (more sensitive than classified signals)
        buy_probs = signal_probs['buy']
        sell_probs = signal_probs['sell']
        hold_probs = signal_probs['hold']

        # Generate signals from probabilities: BUY if buy > sell and buy > threshold
        prob_threshold = 0.35  # Lower threshold for more signals
        signals = np.where(
            (buy_probs > sell_probs) & (buy_probs > prob_threshold), 1,
            np.where((sell_probs > buy_probs) & (sell_probs > prob_threshold), -1, 0)
        )

        # Confidence is max of buy/sell probability
        signal_conf_arr = np.maximum(buy_probs, sell_probs)

        logger.info("Running simulation...")
        self.equity_curve = [self.initial_balance]

        # Stats tracking
        stats = {
            'skipped_hour': 0,
            'skipped_cooldown': 0,
            'skipped_signal': 0,
            'skipped_confidence': 0,
            'skipped_atr': 0,
            'skipped_daily_limit': 0
        }

        for i in range(50, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]
            current_date = bar_time.strftime('%Y-%m-%d')
            current_hour = bar_time.hour

            # Check max drawdown
            if self.check_max_drawdown():
                break

            # Check daily limit
            if self.check_daily_loss(current_date):
                stats['skipped_daily_limit'] += 1
                if self.current_position:
                    self._check_exit(bar, bar_time)
                continue

            # Manage existing position
            if self.current_position:
                self._check_exit(bar, bar_time)
                continue

            # Hour filter
            if current_hour not in self.TRADING_HOURS:
                stats['skipped_hour'] += 1
                continue

            # Cooldown check
            if self.last_trade_time:
                hours_since = (bar_time - self.last_trade_time).total_seconds() / 3600
                if hours_since < self.cooldown_hours:
                    stats['skipped_cooldown'] += 1
                    continue

            # Get predictions
            regime = regimes[i]
            signal = signals[i]
            signal_conf = signal_conf_arr[i]

            # Signal filter (skip HOLD)
            if signal == 0:
                stats['skipped_signal'] += 1
                continue

            # Confidence filter
            if signal_conf < self.confidence_threshold:
                stats['skipped_confidence'] += 1
                continue

            # ATR filter
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < self.min_atr_pips:
                stats['skipped_atr'] += 1
                continue

            # Calculate position
            direction = 'BUY' if signal == 1 else 'SELL'
            lot_size, sl_pips, tp_pips = self.calculate_position_size(signal_conf, regime, atr_pips)

            entry_price = bar['close']
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
                regime=self.REGIME_NAMES.get(regime, 'unknown'),
                signal_confidence=signal_conf
            )
            self.last_trade_time = bar_time

        # Close remaining position
        if self.current_position:
            last_bar = df.iloc[-1]
            self._close_position(last_bar['close'], df.index[-1], "END")

        logger.info("Backtest complete!")

        # Calculate final stats
        return self._calculate_stats(stats)

    def _check_exit(self, bar: pd.Series, bar_time: datetime):
        """Check position exit"""
        if not self.current_position:
            return

        pos = self.current_position
        high, low = bar['high'], bar['low']

        if pos.direction == 'BUY':
            if low <= pos.sl_price:
                self._close_position(pos.sl_price, bar_time, "SL")
            elif high >= pos.tp_price:
                self._close_position(pos.tp_price, bar_time, "TP")
        else:
            if high >= pos.sl_price:
                self._close_position(pos.sl_price, bar_time, "SL")
            elif low <= pos.tp_price:
                self._close_position(pos.tp_price, bar_time, "TP")

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """Close position"""
        if not self.current_position:
            return

        pos = self.current_position
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.close_reason = reason

        if pos.direction == 'BUY':
            pips = (exit_price - pos.entry_price) / 0.0001
        else:
            pips = (pos.entry_price - exit_price) / 0.0001

        pnl = pips * self.pip_value * pos.lot_size
        pos.pips = pips
        pos.pnl = pnl

        self.balance += pnl
        self.equity_curve.append(self.balance)

        current_date = exit_time.strftime('%Y-%m-%d')
        self.daily_pnl[current_date] = self.daily_pnl.get(current_date, 0.0) + pnl

        self.trades.append(pos)
        self.current_position = None

    def _calculate_stats(self, filter_stats: dict) -> dict:
        """Calculate statistics"""
        if not self.trades:
            return {'total_trades': 0}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        # Monthly breakdown
        monthly_pnl = {}
        monthly_trades = {}
        for t in self.trades:
            month = t.entry_time.strftime('%Y-%m')
            monthly_pnl[month] = monthly_pnl.get(month, 0) + t.pnl
            monthly_trades[month] = monthly_trades.get(month, 0) + 1

        # Regime breakdown
        regime_stats = {}
        for t in self.trades:
            if t.regime not in regime_stats:
                regime_stats[t.regime] = {'trades': 0, 'wins': 0, 'pnl': 0}
            regime_stats[t.regime]['trades'] += 1
            regime_stats[t.regime]['pnl'] += t.pnl
            if t.pnl > 0:
                regime_stats[t.regime]['wins'] += 1

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_pnl': sum(t.pnl for t in self.trades),
            'total_pips': sum(t.pips for t in self.trades),
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': gross_profit / len(wins) if wins else 0,
            'avg_loss': gross_loss / len(losses) if losses else 0,
            'largest_win': max(t.pnl for t in wins) if wins else 0,
            'largest_loss': abs(min(t.pnl for t in losses)) if losses else 0,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd / self.initial_balance * 100,
            'monthly_pnl': monthly_pnl,
            'monthly_trades': monthly_trades,
            'regime_stats': regime_stats,
            'filter_stats': filter_stats,
            'final_balance': self.balance
        }

    def print_results(self, stats: dict):
        """Print results"""
        print("\n" + "=" * 70)
        print("HIGH FREQUENCY ML BACKTEST RESULTS")
        print("=" * 70)

        print(f"\nPeriod: Jan 2025 - Jan 2026")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${stats['final_balance']:,.2f}")
        print(f"Total Return: {(stats['final_balance']/self.initial_balance - 1)*100:+.2f}%")

        print("\n" + "-" * 70)
        print("PARAMETERS")
        print("-" * 70)
        print(f"Trading Hours: {self.TRADING_HOURS[0]:02d}:00 - {self.TRADING_HOURS[-1]:02d}:00 UTC")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Base Risk/Trade: {self.base_risk_pct:.1%}")
        print(f"Daily Loss Limit: {self.max_daily_loss_pct:.1%}")
        print(f"Max Drawdown Limit: {self.max_drawdown_pct:.1%}")
        print(f"Cooldown: {self.cooldown_hours} hours")

        print("\n" + "-" * 70)
        print("TRADE STATISTICS")
        print("-" * 70)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Winning Trades: {stats['winning_trades']}")
        print(f"Losing Trades: {stats['losing_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")

        print("\n" + "-" * 70)
        print("P/L ANALYSIS")
        print("-" * 70)
        print(f"Total P/L: ${stats['total_pnl']:+,.2f}")
        print(f"Total Pips: {stats['total_pips']:+,.1f}")
        print(f"Avg Win: ${stats['avg_win']:.2f}")
        print(f"Avg Loss: ${stats['avg_loss']:.2f}")
        print(f"Largest Win: ${stats['largest_win']:.2f}")
        print(f"Largest Loss: ${stats['largest_loss']:.2f}")
        print(f"Max Drawdown: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_pct']:.1f}%)")

        print("\n" + "-" * 70)
        print("MONTHLY BREAKDOWN")
        print("-" * 70)
        print(f"{'Month':<10} {'Trades':>8} {'P/L':>12} {'Cumulative':>12}")
        print("-" * 42)

        cumulative = 0
        for month in sorted(stats['monthly_pnl'].keys()):
            trades = stats['monthly_trades'].get(month, 0)
            pnl = stats['monthly_pnl'].get(month, 0)
            cumulative += pnl
            print(f"{month:<10} {trades:>8} ${pnl:>+10,.2f} ${cumulative:>+10,.2f}")

        print("\n" + "-" * 70)
        print("REGIME BREAKDOWN")
        print("-" * 70)
        for regime, rs in stats['regime_stats'].items():
            wr = rs['wins'] / rs['trades'] * 100 if rs['trades'] > 0 else 0
            print(f"{regime:<20} Trades: {rs['trades']:>4} | WR: {wr:>5.1f}% | P/L: ${rs['pnl']:>+8.2f}")

        print("\n" + "-" * 70)
        print("FILTER STATISTICS")
        print("-" * 70)
        fs = stats['filter_stats']
        print(f"Skipped by hour: {fs['skipped_hour']}")
        print(f"Skipped by cooldown: {fs['skipped_cooldown']}")
        print(f"Skipped by signal (HOLD): {fs['skipped_signal']}")
        print(f"Skipped by confidence: {fs['skipped_confidence']}")
        print(f"Skipped by ATR: {fs['skipped_atr']}")
        print(f"Skipped by daily limit: {fs['skipped_daily_limit']}")

        print("\n" + "=" * 70)

        # Target check
        if stats['total_trades'] >= 120:
            print(f"[OK] TARGET MET: {stats['total_trades']} trades (target: 120+)")
        else:
            print(f"[X] TARGET NOT MET: {stats['total_trades']} trades (target: 120+)")
        print("=" * 70)


def main():
    """Run high frequency backtest"""
    import MetaTrader5 as mt5

    if not mt5.initialize(
        path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        login=10009310110,
        password="P-WyAnG8",
        server="MetaQuotes-Demo"
    ):
        logger.error("Failed to connect to MT5")
        return

    logger.info("Connected to MT5")

    # Get data
    end_date = datetime(2026, 1, 31)
    start_date = datetime(2025, 1, 1)

    rates = mt5.copy_rates_range("GBPUSD", mt5.TIMEFRAME_H1, start_date, end_date)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        logger.error("Failed to get data")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    logger.info(f"Got {len(df)} H1 bars")

    # Run backtest
    backtester = HighFrequencyBacktester(
        initial_balance=10000.0,
        confidence_threshold=0.50,
        base_risk_pct=0.01,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
        cooldown_hours=0,
        min_atr_pips=5.0
    )

    if not backtester.load_models():
        return

    stats = backtester.run(df)
    backtester.print_results(stats)


if __name__ == "__main__":
    main()
