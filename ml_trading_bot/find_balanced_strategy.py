"""
Balanced Strategy Finder
========================

Find strategy with:
- More trades (target: 20+ per year)
- High win rate (target: 70%+)
- 1% risk per trade
- $50K starting balance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger
import MetaTrader5 as mt5

from ml_trading_bot.models import RegimeDetector, SignalClassifier
from ml_trading_bot.features import TechnicalFeatures, SessionFeatures, RegimeFeatures
from ml_trading_bot.features.profile_proxy import ProfileProxyFeatures


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pips: float
    is_win: bool
    hour: int
    day_of_week: int
    regime_name: str
    signal_confidence: float
    adx: float
    lot_size: float


class BalancedStrategyFinder:
    def __init__(self):
        self.regime_detector = None
        self.signal_classifier = None
        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

        self.df = None
        self.regimes = None
        self.signals = None
        self.signal_probs = None

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

    def precompute(self, df: pd.DataFrame):
        """Pre-compute all features and predictions"""
        logger.info("Pre-computing features...")
        df = df.copy()
        df = self.tech_features.add_all_features(df)
        df = self.session_features.add_all_features(df)
        df = self.regime_features.add_all_features(df)
        df = self.profile_proxy.add_all_features(df)
        df = df.dropna()
        self.df = df

        logger.info("Pre-computing predictions...")
        self.regimes = self.regime_detector.predict(df)
        self.signals = self.signal_classifier.predict(df)
        signal_probs_dict = self.signal_classifier.predict_proba(df)
        self.signal_probs = np.column_stack([
            signal_probs_dict['sell'],
            signal_probs_dict['hold'],
            signal_probs_dict['buy']
        ])
        logger.info("Pre-computation done")

    def simulate(
        self,
        initial_balance: float = 50000.0,
        risk_pct: float = 0.01,  # 1% risk
        allowed_hours: List[int] = None,
        allowed_days: List[int] = None,
        allowed_regimes: List[str] = None,
        skip_regimes: List[str] = None,
        min_conf: float = 0.49,
        min_adx: float = 0,
        allowed_directions: List[str] = None
    ) -> Tuple[List[Trade], float, float]:
        """
        Simulate with automatic lot sizing based on risk %
        """
        regime_names = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}
        pip_value = 10.0  # $10 per pip per lot

        trades = []
        balance = initial_balance
        in_position = False
        current_entry = None
        last_exit_idx = 0

        for i in range(50, len(self.df)):
            bar = self.df.iloc[i]
            bar_time = self.df.index[i]

            # Skip if still in position from before
            if i <= last_exit_idx:
                continue

            hour = bar_time.hour
            day = bar_time.weekday()

            # Filters
            if allowed_hours and hour not in allowed_hours:
                continue
            if allowed_days and day not in allowed_days:
                continue

            regime = self.regimes[i]
            regime_name = regime_names.get(regime, 'unknown')

            current_adx = bar.get('adx_14', 0)
            if current_adx > 40 and regime == 2:
                regime = 0
                regime_name = 'trending_override'

            if allowed_regimes and regime_name not in allowed_regimes:
                continue
            if skip_regimes and regime_name in skip_regimes:
                continue
            if current_adx < min_adx:
                continue

            signal = self.signals[i]
            signal_conf = self.signal_probs[i].max()

            if signal == 0:
                continue
            if signal_conf < min_conf:
                continue

            direction = 'BUY' if signal == 1 else 'SELL'
            if allowed_directions and direction not in allowed_directions:
                continue

            # Calculate position size with AUTOMATIC LOT SIZING
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            # SL/TP based on regime
            sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}.get(regime, 1.5)
            tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}.get(regime, 1.5)
            sl_pips = atr_pips * sl_mult
            tp_pips = atr_pips * tp_mult

            # AUTOMATIC LOT SIZING: risk_amount = lot_size * sl_pips * pip_value
            risk_amount = balance * risk_pct
            lot_size = risk_amount / (sl_pips * pip_value)
            lot_size = max(0.01, min(lot_size, 10.0))  # Max 10 lots
            lot_size = round(lot_size, 2)

            entry_price = bar['close']
            pip_size = 0.0001

            if direction == 'BUY':
                sl_price = entry_price - sl_pips * pip_size
                tp_price = entry_price + tp_pips * pip_size
            else:
                sl_price = entry_price + sl_pips * pip_size
                tp_price = entry_price - tp_pips * pip_size

            # Find exit
            exit_idx = i
            exit_price = entry_price
            for j in range(i + 1, len(self.df)):
                bar_j = self.df.iloc[j]
                high = bar_j['high']
                low = bar_j['low']

                if direction == 'BUY':
                    if low <= sl_price:
                        exit_idx = j
                        exit_price = sl_price
                        break
                    if high >= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        break
                else:
                    if high >= sl_price:
                        exit_idx = j
                        exit_price = sl_price
                        break
                    if low <= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        break

            last_exit_idx = exit_idx

            # Calculate P/L
            if direction == 'BUY':
                pips = (exit_price - entry_price) / 0.0001
            else:
                pips = (entry_price - exit_price) / 0.0001

            pnl = pips * pip_value * lot_size
            balance += pnl

            trade = Trade(
                entry_time=bar_time,
                exit_time=self.df.index[exit_idx],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pips=pips,
                is_win=pnl > 0,
                hour=hour,
                day_of_week=day,
                regime_name=regime_name,
                signal_confidence=signal_conf,
                adx=current_adx,
                lot_size=lot_size
            )
            trades.append(trade)

        # Calculate max drawdown
        peak = initial_balance
        max_dd = 0
        running_balance = initial_balance
        for t in trades:
            running_balance += t.pnl
            if running_balance > peak:
                peak = running_balance
            dd = peak - running_balance
            if dd > max_dd:
                max_dd = dd

        return trades, balance, max_dd

    def find_best_configs(self):
        """Find best balanced configurations"""
        print("\n" + "=" * 70)
        print("BALANCED STRATEGY SEARCH")
        print("=" * 70)
        print("Target: 20+ trades, 70%+ WR, 1% risk, $50K balance")

        # Test configurations
        configs = []

        # Hour filters to test
        hour_options = [
            None,  # All hours
            list(range(0, 6)) + list(range(8, 15)) + list(range(20, 24)),  # Skip 6-7, 15-19
            list(range(1, 6)) + list(range(8, 13)) + [14, 15, 20, 21, 22],  # Best hours
            list(range(0, 12)),  # Morning only
            list(range(1, 15)),  # 01:00-14:00
        ]

        # Day filters
        day_options = [
            None,  # All days
            [0, 1, 3, 4],  # Skip Wednesday
            [1, 2, 3],  # Tue-Thu
        ]

        # Regime filters
        regime_options = [
            None,  # All regimes
            ['ranging_choppy'],  # Skip ranging (use skip_regimes)
        ]

        # Confidence thresholds
        conf_options = [0.49, 0.50, 0.52]

        # Direction options
        dir_options = [
            None,  # Both
            ['BUY'],  # BUY only (historically better)
        ]

        results = []

        for hours in hour_options:
            for days in day_options:
                for skip_regime in [None, ['ranging_choppy']]:
                    for min_conf in conf_options:
                        for dirs in dir_options:
                            trades, final_balance, max_dd = self.simulate(
                                initial_balance=50000.0,
                                risk_pct=0.01,
                                allowed_hours=hours,
                                allowed_days=days,
                                skip_regimes=skip_regime,
                                min_conf=min_conf,
                                allowed_directions=dirs
                            )

                            if len(trades) < 10:  # Minimum 10 trades
                                continue

                            wins = len([t for t in trades if t.is_win])
                            losses = len([t for t in trades if not t.is_win])
                            wr = wins / len(trades) * 100
                            total_pnl = final_balance - 50000
                            total_pips = sum(t.pips for t in trades)
                            dd_pct = max_dd / 50000 * 100

                            result = {
                                'hours': hours,
                                'days': days,
                                'skip_regime': skip_regime,
                                'min_conf': min_conf,
                                'directions': dirs,
                                'trades': len(trades),
                                'wins': wins,
                                'losses': losses,
                                'wr': wr,
                                'pnl': total_pnl,
                                'pips': total_pips,
                                'max_dd': max_dd,
                                'dd_pct': dd_pct,
                                'final_balance': final_balance
                            }
                            results.append(result)

        # Sort by: high WR, then by trade count, then by PnL
        results.sort(key=lambda x: (-x['wr'], -x['trades'], -x['pnl']))

        print(f"\nTested {len(results)} configurations")
        print(f"\n--- TOP 20 BALANCED CONFIGS (sorted by WR, trades, PnL) ---")

        for i, r in enumerate(results[:20]):
            print(f"\n{i+1}. {r['trades']} trades | {r['wins']}W/{r['losses']}L | {r['wr']:.1f}% WR")
            print(f"   P/L: ${r['pnl']:+,.2f} | Pips: {r['pips']:+.1f}")
            print(f"   Max DD: ${r['max_dd']:,.2f} ({r['dd_pct']:.1f}%)")
            print(f"   Hours: {r['hours']}")
            print(f"   Days: {r['days']}")
            print(f"   Skip Regime: {r['skip_regime']}")
            print(f"   Min Conf: {r['min_conf']}, Directions: {r['directions']}")

        # Find best balanced config (WR >= 65%, trades >= 20)
        balanced = [r for r in results if r['wr'] >= 65 and r['trades'] >= 20]
        balanced.sort(key=lambda x: (-x['wr'], -x['pnl']))

        print(f"\n" + "=" * 70)
        print("BEST BALANCED CONFIGS (WR >= 65%, Trades >= 20)")
        print("=" * 70)

        for i, r in enumerate(balanced[:10]):
            print(f"\n{i+1}. {r['trades']} trades | {r['wins']}W/{r['losses']}L | {r['wr']:.1f}% WR")
            print(f"   P/L: ${r['pnl']:+,.2f} | Final Balance: ${r['final_balance']:,.2f}")
            print(f"   Max DD: ${r['max_dd']:,.2f} ({r['dd_pct']:.1f}%)")
            print(f"   Hours: {r['hours']}")
            print(f"   Days: {r['days']}")
            print(f"   Skip Regime: {r['skip_regime']}")
            print(f"   Min Conf: {r['min_conf']}, Directions: {r['directions']}")

        if balanced:
            return balanced[0]
        return results[0] if results else None


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

    finder = BalancedStrategyFinder()
    if not finder.load_models():
        return

    finder.precompute(df)
    best = finder.find_best_configs()

    if best:
        print("\n" + "=" * 70)
        print("RECOMMENDED CONFIGURATION")
        print("=" * 70)
        print(f"Trades: {best['trades']}")
        print(f"Win Rate: {best['wr']:.1f}%")
        print(f"P/L: ${best['pnl']:+,.2f}")
        print(f"Max DD: {best['dd_pct']:.1f}%")
        print(f"\nFilters:")
        print(f"  Hours: {best['hours']}")
        print(f"  Days: {best['days']}")
        print(f"  Skip Regime: {best['skip_regime']}")
        print(f"  Min Conf: {best['min_conf']}")
        print(f"  Directions: {best['directions']}")


if __name__ == "__main__":
    main()
