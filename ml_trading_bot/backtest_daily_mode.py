"""
DAILY MODE BACKTEST
====================

Goal: 1-2 trades per day with consistent action
Trade-off: Accept lower win rate for more frequency

Testing multiple configurations to find optimal daily trading setup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from loguru import logger
import MetaTrader5 as mt5

from ml_trading_bot.models import RegimeDetector, SignalClassifier
from ml_trading_bot.features import TechnicalFeatures, SessionFeatures, RegimeFeatures
from ml_trading_bot.features.profile_proxy import ProfileProxyFeatures


@dataclass
class Trade:
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
    signal_conf: float = 0.0
    close_reason: str = ""


class DailyModeBacktester:
    """
    Daily Mode Backtester - Optimized for frequent trading
    """

    def __init__(
        self,
        initial_balance: float = 50000.0,
        risk_pct: float = 0.005,  # 0.5% risk (smaller for more trades)
        min_confidence: float = 0.40,
        allowed_hours: List[int] = None,  # None = all hours
        allowed_days: List[int] = None,  # None = all days
        direction_filter: str = None,  # None = both directions
        skip_regimes: List[str] = None,
        max_trades_per_day: int = 3,
        name: str = "Daily Mode"
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        self.min_confidence = min_confidence
        self.allowed_hours = allowed_hours
        self.allowed_days = allowed_days
        self.direction_filter = direction_filter
        self.skip_regimes = skip_regimes or []
        self.max_trades_per_day = max_trades_per_day
        self.name = name

        self.regime_detector = None
        self.signal_classifier = None

        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_position: Optional[Trade] = None
        self.daily_trade_count: Dict[str, int] = {}

        self.pip_value = 10.0

    def load_models(self) -> bool:
        try:
            models_dir = Path(__file__).parent / "saved_models"
            self.regime_detector = RegimeDetector()
            self.regime_detector.load(models_dir / "regime_hmm.pkl")
            self.signal_classifier = SignalClassifier()
            self.signal_classifier.load(models_dir / "signal_classifier.pkl")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.tech_features.add_all_features(df)
        df = self.session_features.add_all_features(df)
        df = self.regime_features.add_all_features(df)
        df = self.profile_proxy.add_all_features(df)
        return df

    def calculate_lot_size(self, sl_pips: float) -> float:
        risk_amount = self.balance * self.risk_pct
        lot_size = risk_amount / (sl_pips * self.pip_value)
        lot_size = max(0.01, min(lot_size, 10.0))
        return round(lot_size, 2)

    def run(self, df: pd.DataFrame) -> dict:
        """Run backtest and return results"""
        df = self.prepare_features(df)
        df = df.dropna()

        regimes = self.regime_detector.predict(df)
        signals = self.signal_classifier.predict(df)
        signal_probs_dict = self.signal_classifier.predict_proba(df)
        signal_probs = np.column_stack([
            signal_probs_dict['sell'],
            signal_probs_dict['hold'],
            signal_probs_dict['buy']
        ])

        regime_names = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}

        self.equity_curve = [self.initial_balance]
        self.trades = []
        self.balance = self.initial_balance
        self.daily_trade_count = {}

        for i in range(50, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            if self.current_position:
                self._check_exit(bar, bar_time)
                continue

            hour = bar_time.hour
            day = bar_time.weekday()
            date_str = bar_time.strftime('%Y-%m-%d')

            # Check max trades per day
            if self.daily_trade_count.get(date_str, 0) >= self.max_trades_per_day:
                continue

            # FILTER: Hours
            if self.allowed_hours and hour not in self.allowed_hours:
                continue

            # FILTER: Days
            if self.allowed_days and day not in self.allowed_days:
                continue

            regime = regimes[i]
            regime_name = regime_names.get(regime, 'unknown')

            # FILTER: Regime
            if regime_name in self.skip_regimes:
                continue

            signal = signals[i]
            signal_conf = signal_probs[i].max()

            # FILTER: Signal must not be HOLD
            if signal == 0:
                continue

            # FILTER: Confidence
            if signal_conf < self.min_confidence:
                continue

            direction = 'BUY' if signal == 1 else 'SELL'

            # FILTER: Direction
            if self.direction_filter and direction != self.direction_filter:
                continue

            # Calculate SL/TP
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            sl_mult = 1.5
            tp_mult = 1.5  # 1:1 RR for daily mode

            sl_pips = atr_pips * sl_mult
            tp_pips = atr_pips * tp_mult

            lot_size = self.calculate_lot_size(sl_pips)

            entry_price = bar['close']
            pip_size = 0.0001

            if direction == 'BUY':
                sl_price = entry_price - sl_pips * pip_size
                tp_price = entry_price + tp_pips * pip_size
            else:
                sl_price = entry_price + sl_pips * pip_size
                tp_price = entry_price - tp_pips * pip_size

            self.current_position = Trade(
                entry_time=bar_time,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                regime=regime_name,
                signal_conf=signal_conf
            )

            # Update daily trade count
            self.daily_trade_count[date_str] = self.daily_trade_count.get(date_str, 0) + 1

        if self.current_position:
            last_bar = df.iloc[-1]
            self._close_position(last_bar['close'], df.index[-1], "END")

        return self._calculate_results()

    def _check_exit(self, bar, bar_time):
        if not self.current_position:
            return

        pos = self.current_position
        high = bar['high']
        low = bar['low']

        if pos.direction == 'BUY':
            if low <= pos.sl_price:
                self._close_position(pos.sl_price, bar_time, "SL")
                return
            if high >= pos.tp_price:
                self._close_position(pos.tp_price, bar_time, "TP")
                return
        else:
            if high >= pos.sl_price:
                self._close_position(pos.sl_price, bar_time, "SL")
                return
            if low <= pos.tp_price:
                self._close_position(pos.tp_price, bar_time, "TP")
                return

    def _close_position(self, exit_price, exit_time, reason):
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

        self.trades.append(pos)
        self.current_position = None

    def _calculate_results(self) -> dict:
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_pips = sum(t.pips for t in self.trades)

        # Calculate trading days
        unique_days = set()
        for t in self.trades:
            unique_days.add(t.entry_time.strftime('%Y-%m-%d'))
        trading_days = len(unique_days)

        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Trades per day
        total_calendar_days = 260  # Approx trading days in 13 months
        trades_per_day = len(self.trades) / total_calendar_days if total_calendar_days > 0 else 0

        return {
            'name': self.name,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': total_pnl,
            'total_pips': total_pips,
            'final_balance': self.balance,
            'return_pct': (self.balance / self.initial_balance - 1) * 100,
            'max_dd': max_dd,
            'max_dd_pct': max_dd / self.initial_balance * 100,
            'profit_factor': pf,
            'trades_per_day': trades_per_day,
            'trading_days': trading_days,
            'avg_win': sum(t.pnl for t in wins) / len(wins) if wins else 0,
            'avg_loss': abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0,
            'risk_pct': self.risk_pct * 100,
            'min_conf': self.min_confidence
        }


def main():
    print("=" * 70)
    print("DAILY MODE BACKTEST - Finding Optimal Configuration")
    print("=" * 70)

    # Connect to MT5
    if not mt5.initialize(
        path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        login=10009310110,
        password="P-WyAnG8",
        server="MetaQuotes-Demo"
    ):
        logger.error("Failed to connect to MT5")
        return

    # Get data
    end_date = datetime(2026, 1, 31)
    start_date = datetime(2025, 1, 1)

    rates = mt5.copy_rates_range(
        "GBPUSD",
        mt5.TIMEFRAME_H1,
        start_date,
        end_date
    )

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        logger.error("Failed to get data")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    print(f"\nData loaded: {len(df)} H1 bars")
    print(f"Period: {df.index[0]} to {df.index[-1]}")

    # Test configurations
    configs = [
        {
            'name': 'Daily Relaxed (All)',
            'risk_pct': 0.005,  # 0.5%
            'min_confidence': 0.40,
            'allowed_hours': None,  # All hours
            'allowed_days': None,  # All days
            'direction_filter': None,  # Both directions
            'skip_regimes': [],
            'max_trades_per_day': 3
        },
        {
            'name': 'Daily London+NY',
            'risk_pct': 0.005,
            'min_confidence': 0.42,
            'allowed_hours': list(range(7, 18)),  # 07:00-17:00 UTC
            'allowed_days': [0, 1, 2, 3, 4],  # Mon-Fri
            'direction_filter': None,
            'skip_regimes': [],
            'max_trades_per_day': 2
        },
        {
            'name': 'Daily Skip Choppy',
            'risk_pct': 0.005,
            'min_confidence': 0.40,
            'allowed_hours': None,
            'allowed_days': [0, 1, 2, 3, 4],
            'direction_filter': None,
            'skip_regimes': ['ranging_choppy'],
            'max_trades_per_day': 3
        },
        {
            'name': 'Daily BUY Only',
            'risk_pct': 0.005,
            'min_confidence': 0.40,
            'allowed_hours': None,
            'allowed_days': [0, 1, 2, 3, 4],
            'direction_filter': 'BUY',
            'skip_regimes': [],
            'max_trades_per_day': 2
        },
        {
            'name': 'Daily SELL Only',
            'risk_pct': 0.005,
            'min_confidence': 0.40,
            'allowed_hours': None,
            'allowed_days': [0, 1, 2, 3, 4],
            'direction_filter': 'SELL',
            'skip_regimes': [],
            'max_trades_per_day': 2
        },
        {
            'name': 'Daily Conf 0.45',
            'risk_pct': 0.005,
            'min_confidence': 0.45,
            'allowed_hours': None,
            'allowed_days': [0, 1, 2, 3, 4],
            'direction_filter': None,
            'skip_regimes': [],
            'max_trades_per_day': 3
        },
        {
            'name': 'Daily High Risk',
            'risk_pct': 0.01,  # 1%
            'min_confidence': 0.42,
            'allowed_hours': list(range(6, 16)),
            'allowed_days': [0, 1, 2, 3, 4],
            'direction_filter': None,
            'skip_regimes': [],
            'max_trades_per_day': 2
        },
        {
            'name': 'Daily Low Risk',
            'risk_pct': 0.0025,  # 0.25%
            'min_confidence': 0.38,
            'allowed_hours': None,
            'allowed_days': None,
            'direction_filter': None,
            'skip_regimes': [],
            'max_trades_per_day': 5
        },
    ]

    results = []

    # Load models once
    backtester = DailyModeBacktester()
    if not backtester.load_models():
        return

    for config in configs:
        print(f"\nTesting: {config['name']}...")

        bt = DailyModeBacktester(
            initial_balance=50000.0,
            risk_pct=config['risk_pct'],
            min_confidence=config['min_confidence'],
            allowed_hours=config['allowed_hours'],
            allowed_days=config['allowed_days'],
            direction_filter=config['direction_filter'],
            skip_regimes=config['skip_regimes'],
            max_trades_per_day=config['max_trades_per_day'],
            name=config['name']
        )
        bt.regime_detector = backtester.regime_detector
        bt.signal_classifier = backtester.signal_classifier

        result = bt.run(df)
        results.append(result)

    # Print comparison table
    print("\n" + "=" * 100)
    print("DAILY MODE BACKTEST RESULTS COMPARISON")
    print("=" * 100)
    print(f"\n{'Config':<22} {'Trades':>7} {'T/Day':>6} {'WR%':>6} {'P/L':>10} {'Ret%':>7} {'MaxDD%':>7} {'PF':>5}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
        print(f"{r['name']:<22} {r['total_trades']:>7} {r['trades_per_day']:>6.2f} "
              f"{r['win_rate']:>6.1f} ${r['total_pnl']:>+9,.0f} {r['return_pct']:>+6.1f}% "
              f"{r['max_dd_pct']:>6.1f}% {r['profit_factor']:>5.2f}")

    # Find best for daily trading
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Best by trades per day (min 0.5)
    daily_viable = [r for r in results if r['trades_per_day'] >= 0.3]
    if daily_viable:
        best_daily = max(daily_viable, key=lambda x: x['total_pnl'])
        print(f"\nBest for Daily Trading (profitable + frequent):")
        print(f"  Config: {best_daily['name']}")
        print(f"  Trades: {best_daily['total_trades']} ({best_daily['trades_per_day']:.2f}/day)")
        print(f"  Win Rate: {best_daily['win_rate']:.1f}%")
        print(f"  Total P/L: ${best_daily['total_pnl']:+,.2f}")
        print(f"  Return: {best_daily['return_pct']:+.1f}%")
        print(f"  Max Drawdown: {best_daily['max_dd_pct']:.1f}%")
        print(f"  Profit Factor: {best_daily['profit_factor']:.2f}")

    # Best overall profit
    best_profit = max(results, key=lambda x: x['total_pnl'])
    print(f"\nBest by Total Profit:")
    print(f"  Config: {best_profit['name']}")
    print(f"  Trades: {best_profit['total_trades']} ({best_profit['trades_per_day']:.2f}/day)")
    print(f"  Win Rate: {best_profit['win_rate']:.1f}%")
    print(f"  Total P/L: ${best_profit['total_pnl']:+,.2f}")

    # Best win rate (with min trades)
    wr_viable = [r for r in results if r['total_trades'] >= 50]
    if wr_viable:
        best_wr = max(wr_viable, key=lambda x: x['win_rate'])
        print(f"\nBest Win Rate (min 50 trades):")
        print(f"  Config: {best_wr['name']}")
        print(f"  Trades: {best_wr['total_trades']}")
        print(f"  Win Rate: {best_wr['win_rate']:.1f}%")

    # Detailed breakdown for best daily config
    if daily_viable:
        print("\n" + "=" * 100)
        print(f"DETAILED RESULTS: {best_daily['name']}")
        print("=" * 100)

        # Re-run to get trades
        bt = DailyModeBacktester(
            initial_balance=50000.0,
            risk_pct=0.005,
            min_confidence=0.40,
            allowed_hours=None,
            allowed_days=[0, 1, 2, 3, 4],
            direction_filter=None,
            skip_regimes=['ranging_choppy'] if 'Choppy' in best_daily['name'] else [],
            max_trades_per_day=3,
            name=best_daily['name']
        )
        bt.regime_detector = backtester.regime_detector
        bt.signal_classifier = backtester.signal_classifier
        bt.run(df)

        # Monthly breakdown
        print(f"\n{'Month':<10} {'Trades':>8} {'W/L':>10} {'P/L':>12}")
        print("-" * 45)

        monthly = {}
        for t in bt.trades:
            month = t.entry_time.strftime('%Y-%m')
            if month not in monthly:
                monthly[month] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
            monthly[month]['trades'] += 1
            if t.pnl > 0:
                monthly[month]['wins'] += 1
            else:
                monthly[month]['losses'] += 1
            monthly[month]['pnl'] += t.pnl

        for month in sorted(monthly.keys()):
            m = monthly[month]
            print(f"{month:<10} {m['trades']:>8} {m['wins']:>4}/{m['losses']:<4} ${m['pnl']:>+10,.2f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
