"""
DAILY SWEET SPOT BACKTEST
==========================

Combining best filters to find optimal daily trading configuration.

Key insights from previous test:
- BUY only is better than SELL (+2.8% vs -10.2%)
- London+NY hours have best win rate (57.1%)
- Lower confidence = more trades but lower WR
- Skip choppy might help

Now testing combinations to find sweet spot.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
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


class SweetSpotBacktester:
    def __init__(
        self,
        initial_balance: float = 50000.0,
        risk_pct: float = 0.005,
        min_confidence: float = 0.40,
        allowed_hours: List[int] = None,
        allowed_days: List[int] = None,
        direction_filter: str = None,
        skip_regimes: List[str] = None,
        max_trades_per_day: int = 3,
        tp_multiplier: float = 1.5,
        sl_multiplier: float = 1.5,
        name: str = "Config"
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
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
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

    def run(self, df: pd.DataFrame, precomputed=None) -> dict:
        if precomputed is None:
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
        else:
            df, regimes, signals, signal_probs = precomputed

        regime_names = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}

        self.equity_curve = [self.initial_balance]
        self.trades = []
        self.balance = self.initial_balance
        self.daily_trade_count = {}
        self.current_position = None

        for i in range(50, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            if self.current_position:
                self._check_exit(bar, bar_time)
                continue

            hour = bar_time.hour
            day = bar_time.weekday()
            date_str = bar_time.strftime('%Y-%m-%d')

            if self.daily_trade_count.get(date_str, 0) >= self.max_trades_per_day:
                continue

            if self.allowed_hours and hour not in self.allowed_hours:
                continue

            if self.allowed_days and day not in self.allowed_days:
                continue

            regime = regimes[i]
            regime_name = regime_names.get(regime, 'unknown')

            if regime_name in self.skip_regimes:
                continue

            signal = signals[i]
            signal_conf = signal_probs[i].max()

            if signal == 0:
                continue

            if signal_conf < self.min_confidence:
                continue

            direction = 'BUY' if signal == 1 else 'SELL'

            if self.direction_filter and direction != self.direction_filter:
                continue

            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            sl_pips = atr_pips * self.sl_multiplier
            tp_pips = atr_pips * self.tp_multiplier

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

        unique_days = set()
        for t in self.trades:
            unique_days.add(t.entry_time.strftime('%Y-%m-%d'))
        trading_days = len(unique_days)

        peak = self.initial_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        total_calendar_days = 260
        trades_per_day = len(self.trades) / total_calendar_days

        # Count profitable months
        monthly_pnl = {}
        for t in self.trades:
            month = t.entry_time.strftime('%Y-%m')
            monthly_pnl[month] = monthly_pnl.get(month, 0) + t.pnl

        profitable_months = sum(1 for pnl in monthly_pnl.values() if pnl > 0)
        total_months = len(monthly_pnl)

        return {
            'name': self.name,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': total_pnl,
            'final_balance': self.balance,
            'return_pct': (self.balance / self.initial_balance - 1) * 100,
            'max_dd': max_dd,
            'max_dd_pct': max_dd / self.initial_balance * 100,
            'profit_factor': pf,
            'trades_per_day': trades_per_day,
            'trading_days': trading_days,
            'profitable_months': profitable_months,
            'total_months': total_months,
            'risk_pct': self.risk_pct * 100,
            'min_conf': self.min_confidence
        }


def main():
    print("=" * 80)
    print("DAILY SWEET SPOT - Finding Optimal Combination")
    print("=" * 80)

    if not mt5.initialize(
        path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        login=10009310110,
        password="P-WyAnG8",
        server="MetaQuotes-Demo"
    ):
        logger.error("Failed to connect to MT5")
        return

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

    print(f"\nData: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Pre-compute features once
    print("\nPre-computing features...")
    bt_base = SweetSpotBacktester()
    if not bt_base.load_models():
        return

    df_feat = bt_base.prepare_features(df)
    df_feat = df_feat.dropna()
    regimes = bt_base.regime_detector.predict(df_feat)
    signals = bt_base.signal_classifier.predict(df_feat)
    signal_probs_dict = bt_base.signal_classifier.predict_proba(df_feat)
    signal_probs = np.column_stack([
        signal_probs_dict['sell'],
        signal_probs_dict['hold'],
        signal_probs_dict['buy']
    ])
    precomputed = (df_feat, regimes, signals, signal_probs)

    print("Features ready. Testing combinations...\n")

    # Combination matrix
    configs = []

    # Base hours options
    hour_sets = {
        'all': None,
        'london_ny': list(range(7, 18)),  # 07:00-17:00
        'core': list(range(8, 16)),  # 08:00-15:00
        'extended': list(range(1, 20)),  # 01:00-19:00
    }

    # Confidence levels
    conf_levels = [0.38, 0.40, 0.42, 0.45]

    # Risk levels
    risk_levels = [0.0025, 0.005, 0.0075, 0.01]

    # TP/SL multipliers (RR ratio)
    rr_options = [
        (1.2, 1.5),  # 1:1.25 RR
        (1.5, 1.5),  # 1:1 RR
        (1.5, 2.0),  # 1:1.33 RR
        (1.5, 2.5),  # 1:1.67 RR
    ]

    # Direction
    directions = ['BUY', None]  # BUY only or Both

    # Skip regimes
    regime_skips = [[], ['ranging_choppy']]

    # Max trades per day
    max_trades = [2, 3, 5]

    # Generate focused combinations based on insights
    print("Testing focused combinations based on previous insights...")

    # Insight-based configs
    focused_configs = [
        # BUY only + London/NY + various settings
        {'name': 'BUY+LonNY+Conf40+R0.5%', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+LonNY+Conf42+R0.5%', 'hours': 'london_ny', 'conf': 0.42, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+LonNY+Conf40+R0.75%', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.0075, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+LonNY+Conf40+R1%', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.01, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},

        # BUY only + Core hours
        {'name': 'BUY+Core+Conf40+R0.5%', 'hours': 'core', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+Core+Conf42+R0.75%', 'hours': 'core', 'conf': 0.42, 'risk': 0.0075, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},

        # BUY only + Extended hours (more trades)
        {'name': 'BUY+Ext+Conf40+R0.5%', 'hours': 'extended', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+Ext+Conf38+R0.25%', 'hours': 'extended', 'conf': 0.38, 'risk': 0.0025, 'dir': 'BUY', 'skip': [], 'max_t': 5, 'sl': 1.5, 'tp': 1.5},

        # BUY only + All hours
        {'name': 'BUY+All+Conf40+R0.5%', 'hours': 'all', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+All+Conf38+R0.25%', 'hours': 'all', 'conf': 0.38, 'risk': 0.0025, 'dir': 'BUY', 'skip': [], 'max_t': 5, 'sl': 1.5, 'tp': 1.5},

        # Skip choppy regime
        {'name': 'BUY+LonNY+NoChop+Conf40', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': ['ranging_choppy'], 'max_t': 3, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+All+NoChop+Conf38', 'hours': 'all', 'conf': 0.38, 'risk': 0.005, 'dir': 'BUY', 'skip': ['ranging_choppy'], 'max_t': 5, 'sl': 1.5, 'tp': 1.5},

        # Different RR ratios
        {'name': 'BUY+LonNY+RR1.33+Conf40', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 2.0},
        {'name': 'BUY+LonNY+RR1.67+Conf40', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 2.5},
        {'name': 'BUY+All+RR1.33+Conf38', 'hours': 'all', 'conf': 0.38, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 5, 'sl': 1.5, 'tp': 2.0},

        # Both directions but with filters
        {'name': 'Both+LonNY+NoChop+Conf45', 'hours': 'london_ny', 'conf': 0.45, 'risk': 0.005, 'dir': None, 'skip': ['ranging_choppy'], 'max_t': 2, 'sl': 1.5, 'tp': 1.5},
        {'name': 'Both+Core+NoChop+Conf45', 'hours': 'core', 'conf': 0.45, 'risk': 0.005, 'dir': None, 'skip': ['ranging_choppy'], 'max_t': 2, 'sl': 1.5, 'tp': 1.5},

        # High frequency low risk
        {'name': 'BUY+All+Conf35+R0.2%+Max5', 'hours': 'all', 'conf': 0.35, 'risk': 0.002, 'dir': 'BUY', 'skip': [], 'max_t': 5, 'sl': 1.5, 'tp': 1.5},
        {'name': 'BUY+Ext+Conf35+R0.25%+Max5', 'hours': 'extended', 'conf': 0.35, 'risk': 0.0025, 'dir': 'BUY', 'skip': [], 'max_t': 5, 'sl': 1.5, 'tp': 1.5},

        # Conservative daily
        {'name': 'BUY+LonNY+Conf45+R0.5%', 'hours': 'london_ny', 'conf': 0.45, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 2, 'sl': 1.5, 'tp': 2.0},
        {'name': 'BUY+Core+Conf45+R0.75%', 'hours': 'core', 'conf': 0.45, 'risk': 0.0075, 'dir': 'BUY', 'skip': [], 'max_t': 2, 'sl': 1.5, 'tp': 2.0},

        # Aggressive but filtered
        {'name': 'BUY+LonNY+Conf40+R1%+RR1.5', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.01, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 2.25},
        {'name': 'BUY+Core+Conf42+R1%+RR1.5', 'hours': 'core', 'conf': 0.42, 'risk': 0.01, 'dir': 'BUY', 'skip': [], 'max_t': 2, 'sl': 1.5, 'tp': 2.25},

        # Sweet spot candidates
        {'name': 'SWEET1:BUY+LonNY+C40+R0.75%', 'hours': 'london_ny', 'conf': 0.40, 'risk': 0.0075, 'dir': 'BUY', 'skip': [], 'max_t': 3, 'sl': 1.5, 'tp': 2.0},
        {'name': 'SWEET2:BUY+Ext+C38+R0.5%', 'hours': 'extended', 'conf': 0.38, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 4, 'sl': 1.5, 'tp': 1.75},
        {'name': 'SWEET3:BUY+All+C40+R0.5%+RR1.5', 'hours': 'all', 'conf': 0.40, 'risk': 0.005, 'dir': 'BUY', 'skip': [], 'max_t': 4, 'sl': 1.5, 'tp': 2.25},
    ]

    results = []

    for cfg in focused_configs:
        bt = SweetSpotBacktester(
            initial_balance=50000.0,
            risk_pct=cfg['risk'],
            min_confidence=cfg['conf'],
            allowed_hours=hour_sets[cfg['hours']],
            allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri always
            direction_filter=cfg['dir'],
            skip_regimes=cfg['skip'],
            max_trades_per_day=cfg['max_t'],
            sl_multiplier=cfg['sl'],
            tp_multiplier=cfg['tp'],
            name=cfg['name']
        )
        bt.regime_detector = bt_base.regime_detector
        bt.signal_classifier = bt_base.signal_classifier

        result = bt.run(df_feat, precomputed)
        results.append(result)
        print(f"  {cfg['name']}: {result['total_trades']} trades, {result['win_rate']:.1f}% WR, ${result['total_pnl']:+,.0f}")

    # Sort and display results
    print("\n" + "=" * 120)
    print("SWEET SPOT RESULTS - Sorted by Score (Profit * Frequency * WinRate)")
    print("=" * 120)

    # Calculate score: balance profit, frequency, and win rate
    for r in results:
        # Score = PnL factor * frequency factor * WR factor
        pnl_factor = max(0, r['total_pnl']) / 5000  # Normalize
        freq_factor = min(r['trades_per_day'], 1.5)  # Cap at 1.5
        wr_factor = r['win_rate'] / 50  # 50% = 1.0
        dd_penalty = max(0, 1 - r['max_dd_pct'] / 10)  # Penalize high DD

        r['score'] = pnl_factor * freq_factor * wr_factor * dd_penalty
        r['score'] = round(r['score'], 3)

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"\n{'Config':<32} {'Trades':>6} {'T/Day':>6} {'WR%':>6} {'P/L':>10} {'Ret%':>7} {'DD%':>6} {'PF':>5} {'PrfMo':>6} {'Score':>6}")
    print("-" * 110)

    for r in results_sorted:
        prf_mo = f"{r['profitable_months']}/{r['total_months']}"
        print(f"{r['name']:<32} {r['total_trades']:>6} {r['trades_per_day']:>6.2f} "
              f"{r['win_rate']:>6.1f} ${r['total_pnl']:>+9,.0f} {r['return_pct']:>+6.1f}% "
              f"{r['max_dd_pct']:>5.1f}% {r['profit_factor']:>5.2f} {prf_mo:>6} {r['score']:>6.3f}")

    # Top 5 analysis
    print("\n" + "=" * 120)
    print("TOP 5 SWEET SPOT CONFIGURATIONS")
    print("=" * 120)

    for i, r in enumerate(results_sorted[:5]):
        print(f"\n#{i+1} {r['name']}")
        print(f"    Trades: {r['total_trades']} ({r['trades_per_day']:.2f}/day)")
        print(f"    Win Rate: {r['win_rate']:.1f}%")
        print(f"    Profit: ${r['total_pnl']:+,.2f} ({r['return_pct']:+.1f}%)")
        print(f"    Max Drawdown: {r['max_dd_pct']:.1f}%")
        print(f"    Profit Factor: {r['profit_factor']:.2f}")
        print(f"    Profitable Months: {r['profitable_months']}/{r['total_months']}")

    # Find best for different priorities
    print("\n" + "=" * 120)
    print("BEST FOR DIFFERENT PRIORITIES")
    print("=" * 120)

    # Most trades per day (profitable)
    daily_profitable = [r for r in results if r['total_pnl'] > 0]
    if daily_profitable:
        best_freq = max(daily_profitable, key=lambda x: x['trades_per_day'])
        print(f"\nMost Frequent (Profitable): {best_freq['name']}")
        print(f"    {best_freq['trades_per_day']:.2f} trades/day, ${best_freq['total_pnl']:+,.0f}, {best_freq['win_rate']:.1f}% WR")

    # Best win rate (min 100 trades)
    wr_viable = [r for r in results if r['total_trades'] >= 100]
    if wr_viable:
        best_wr = max(wr_viable, key=lambda x: x['win_rate'])
        print(f"\nHighest Win Rate (100+ trades): {best_wr['name']}")
        print(f"    {best_wr['win_rate']:.1f}% WR, {best_wr['total_trades']} trades, ${best_wr['total_pnl']:+,.0f}")

    # Lowest drawdown (profitable)
    if daily_profitable:
        best_dd = min(daily_profitable, key=lambda x: x['max_dd_pct'])
        print(f"\nLowest Drawdown (Profitable): {best_dd['name']}")
        print(f"    {best_dd['max_dd_pct']:.1f}% DD, ${best_dd['total_pnl']:+,.0f}, {best_dd['win_rate']:.1f}% WR")

    # Best profit
    best_profit = max(results, key=lambda x: x['total_pnl'])
    print(f"\nHighest Profit: {best_profit['name']}")
    print(f"    ${best_profit['total_pnl']:+,.0f}, {best_profit['trades_per_day']:.2f} t/day, {best_profit['max_dd_pct']:.1f}% DD")

    # Most consistent (profitable months)
    best_consistent = max(results, key=lambda x: x['profitable_months'])
    print(f"\nMost Consistent: {best_consistent['name']}")
    print(f"    {best_consistent['profitable_months']}/{best_consistent['total_months']} profitable months")

    # RECOMMENDATION
    print("\n" + "=" * 120)
    print("RECOMMENDATION FOR DAILY TRADING")
    print("=" * 120)

    # Find sweet spot: good frequency + good profit + acceptable DD
    sweet_candidates = [r for r in results if
                        r['trades_per_day'] >= 0.4 and  # At least 0.4/day
                        r['total_pnl'] > 0 and  # Profitable
                        r['max_dd_pct'] <= 10 and  # Max 10% DD
                        r['win_rate'] >= 50]  # At least 50% WR

    if sweet_candidates:
        sweet_spot = max(sweet_candidates, key=lambda x: x['score'])
        print(f"\nRECOMMENDED CONFIG: {sweet_spot['name']}")
        print("-" * 60)
        print(f"  Trades/Day: {sweet_spot['trades_per_day']:.2f}")
        print(f"  Win Rate: {sweet_spot['win_rate']:.1f}%")
        print(f"  Total Profit: ${sweet_spot['total_pnl']:+,.2f}")
        print(f"  Return: {sweet_spot['return_pct']:+.1f}%")
        print(f"  Max Drawdown: {sweet_spot['max_dd_pct']:.1f}%")
        print(f"  Profit Factor: {sweet_spot['profit_factor']:.2f}")
        print(f"  Profitable Months: {sweet_spot['profitable_months']}/{sweet_spot['total_months']}")
        print(f"  Risk per Trade: {sweet_spot['risk_pct']:.2f}%")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
