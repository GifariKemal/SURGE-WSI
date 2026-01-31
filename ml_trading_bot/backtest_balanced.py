"""
BALANCED STRATEGY BACKTEST
==========================

Configuration for $50K balance, 1% risk, more trades.

3 Modes Available:
1. SAFE: 85.7% WR, 14 trades (Skip ranging + BUY only)
2. BALANCED: 74.1% WR, 27 trades
3. MORE_TRADES: 71.9% WR, 32 trades

All modes use:
- $50,000 starting balance
- 1% risk per trade
- Automatic lot sizing
- Skip Wednesday
- BUY only direction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
import MetaTrader5 as mt5

from ml_trading_bot.models import RegimeDetector, SignalClassifier
from ml_trading_bot.features import TechnicalFeatures, SessionFeatures, RegimeFeatures
from ml_trading_bot.features.profile_proxy import ProfileProxyFeatures


# ========== STRATEGY MODES ==========
MODES = {
    'SAFE': {
        'description': '85.7% WR, 14 trades, Max DD 1.1%',
        'skip_regime': ['ranging_choppy'],
        'min_conf': 0.49,
        'allowed_hours': None,  # All hours
        'allowed_days': [0, 1, 3, 4],  # Skip Wednesday
        'direction': 'BUY'
    },
    'BALANCED': {
        'description': '74.1% WR, 27 trades, Max DD 3.4%',
        'skip_regime': None,
        'min_conf': 0.49,
        'allowed_hours': list(range(1, 15)),  # 01:00-14:00
        'allowed_days': [0, 1, 3, 4],  # Skip Wednesday
        'direction': 'BUY'
    },
    'MORE_TRADES': {
        'description': '71.9% WR, 32 trades, Max DD 3.4%',
        'skip_regime': None,
        'min_conf': 0.49,
        'allowed_hours': None,  # All hours
        'allowed_days': [0, 1, 3, 4],  # Skip Wednesday
        'direction': 'BUY'
    }
}

# SELECT MODE HERE
SELECTED_MODE = 'BALANCED'  # Change to 'SAFE' or 'MORE_TRADES' as needed


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


class BalancedBacktester:
    """
    Balanced Strategy Backtester

    Features:
    - $50K balance
    - 1% risk per trade
    - Automatic lot sizing
    """

    def __init__(
        self,
        initial_balance: float = 50000.0,
        risk_pct: float = 0.01,  # 1% risk
        mode: str = 'BALANCED'
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        self.mode = mode
        self.config = MODES[mode]

        self.regime_detector = None
        self.signal_classifier = None

        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_position: Optional[Trade] = None

        self.pip_value = 10.0  # $10 per pip per lot

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
        """
        AUTOMATIC LOT SIZING based on 1% risk

        Formula: lot_size = risk_amount / (sl_pips * pip_value)

        Example with $50K balance, 1% risk, 30 pip SL:
        - risk_amount = $50,000 * 0.01 = $500
        - lot_size = $500 / (30 * $10) = 1.67 lots
        """
        risk_amount = self.balance * self.risk_pct
        lot_size = risk_amount / (sl_pips * self.pip_value)
        lot_size = max(0.01, min(lot_size, 10.0))  # Min 0.01, Max 10 lots
        return round(lot_size, 2)

    def run(self, df: pd.DataFrame):
        """Run backtest"""
        print("\n" + "=" * 70)
        print(f"BALANCED STRATEGY BACKTEST - {self.mode} MODE")
        print("=" * 70)
        print(f"\nMode: {self.mode}")
        print(f"Description: {self.config['description']}")
        print(f"\nSettings:")
        print(f"  Initial Balance: ${self.initial_balance:,.2f}")
        print(f"  Risk per Trade: {self.risk_pct*100:.1f}%")
        print(f"  Allowed Hours: {self.config['allowed_hours']}")
        print(f"  Allowed Days: {self.config['allowed_days']} (Skip Wednesday)")
        print(f"  Skip Regime: {self.config['skip_regime']}")
        print(f"  Min Confidence: {self.config['min_conf']}")
        print(f"  Direction: {self.config['direction']} only")

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

        for i in range(50, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            if self.current_position:
                self._check_exit(bar, bar_time)
                continue

            hour = bar_time.hour
            day = bar_time.weekday()

            # FILTER: Hours
            if self.config['allowed_hours'] and hour not in self.config['allowed_hours']:
                continue

            # FILTER: Days
            if self.config['allowed_days'] and day not in self.config['allowed_days']:
                continue

            regime = regimes[i]
            regime_name = regime_names.get(regime, 'unknown')

            current_adx = bar.get('adx_14', 0)
            if current_adx > 40 and regime == 2:
                regime = 0
                regime_name = 'trending_override'

            # FILTER: Regime
            if self.config['skip_regime'] and regime_name in self.config['skip_regime']:
                continue

            signal = signals[i]
            signal_conf = signal_probs[i].max()

            if signal == 0:
                continue
            if signal_conf < self.config['min_conf']:
                continue

            direction = 'BUY' if signal == 1 else 'SELL'

            # FILTER: Direction
            if self.config['direction'] and direction != self.config['direction']:
                continue

            # Calculate SL/TP
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}.get(regime, 1.5)
            tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}.get(regime, 1.5)

            sl_pips = atr_pips * sl_mult
            tp_pips = atr_pips * tp_mult

            # AUTOMATIC LOT SIZING
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

        if self.current_position:
            last_bar = df.iloc[-1]
            self._close_position(last_bar['close'], df.index[-1], "END")

        self._print_results()

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

    def _print_results(self):
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nInitial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        total_return = (self.balance / self.initial_balance - 1) * 100
        print(f"Total Return: {total_return:+.2f}%")

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        print(f"\nTotal Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(wins)}")
        print(f"Losing Trades: {len(losses)}")

        if self.trades:
            wr = len(wins) / len(self.trades) * 100
            print(f"Win Rate: {wr:.1f}%")

        total_pnl = sum(t.pnl for t in self.trades)
        total_pips = sum(t.pips for t in self.trades)
        print(f"\nTotal P/L: ${total_pnl:+,.2f}")
        print(f"Total Pips: {total_pips:+.1f}")

        if wins:
            avg_win = sum(t.pnl for t in wins) / len(wins)
            avg_win_pips = sum(t.pips for t in wins) / len(wins)
            print(f"Average Win: ${avg_win:,.2f} ({avg_win_pips:+.1f} pips)")

        if losses:
            avg_loss = abs(sum(t.pnl for t in losses) / len(losses))
            avg_loss_pips = abs(sum(t.pips for t in losses) / len(losses))
            print(f"Average Loss: ${avg_loss:,.2f} ({avg_loss_pips:.1f} pips)")

        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        max_dd_pct = max_dd / self.initial_balance * 100
        print(f"\nMax Drawdown: ${max_dd:,.2f} ({max_dd_pct:.1f}%)")

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        print(f"Profit Factor: {pf:.2f}")

        # Monthly breakdown
        print("\n" + "-" * 70)
        print("MONTHLY BREAKDOWN")
        print("-" * 70)
        print(f"{'Month':<10} {'Trades':>8} {'W/L':>10} {'P/L':>12} {'Cumulative':>12}")
        print("-" * 52)

        monthly = {}
        for t in self.trades:
            month = t.entry_time.strftime('%Y-%m')
            if month not in monthly:
                monthly[month] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
            monthly[month]['trades'] += 1
            if t.pnl > 0:
                monthly[month]['wins'] += 1
            else:
                monthly[month]['losses'] += 1
            monthly[month]['pnl'] += t.pnl

        cumulative = 0
        for month in sorted(monthly.keys()):
            m = monthly[month]
            cumulative += m['pnl']
            print(f"{month:<10} {m['trades']:>8} {m['wins']:>4}/{m['losses']:<4} ${m['pnl']:>+10,.2f} ${cumulative:>+10,.2f}")

        # Trade log
        print("\n" + "-" * 70)
        print("TRADE LOG")
        print("-" * 70)
        print(f"{'#':<3} {'Entry Time':<20} {'Dir':<5} {'Lot':>6} {'Entry':>10} {'Exit':>10} {'Pips':>8} {'P/L':>10} {'Rsn'}")
        print("-" * 95)

        for i, t in enumerate(self.trades):
            wl = "W" if t.pnl > 0 else "L"
            print(f"{i+1:<3} {str(t.entry_time):<20} {t.direction:<5} {t.lot_size:>6.2f} "
                  f"{t.entry_price:>10.5f} {t.exit_price:>10.5f} {t.pips:>+8.1f} ${t.pnl:>+9,.2f} {t.close_reason} {wl}")

        print("\n" + "=" * 70)

        # Summary stats
        avg_lot = sum(t.lot_size for t in self.trades) / len(self.trades) if self.trades else 0
        print(f"\nAVERAGE LOT SIZE: {avg_lot:.2f} lots")
        print(f"RISK PER TRADE: ${self.initial_balance * self.risk_pct:,.2f} ({self.risk_pct*100:.1f}%)")


def main():
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

    # Run backtest with selected mode
    backtester = BalancedBacktester(
        initial_balance=50000.0,
        risk_pct=0.01,  # 1% risk
        mode=SELECTED_MODE
    )

    if not backtester.load_models():
        return

    backtester.run(df)


if __name__ == "__main__":
    main()
