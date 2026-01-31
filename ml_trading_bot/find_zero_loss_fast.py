"""
Fast Zero Loss Finder
=====================

Pre-computes everything ONCE, then tests filter combinations quickly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from loguru import logger
import MetaTrader5 as mt5

from ml_trading_bot.models import RegimeDetector, SignalClassifier
from ml_trading_bot.features import TechnicalFeatures, SessionFeatures, RegimeFeatures
from ml_trading_bot.features.profile_proxy import ProfileProxyFeatures


@dataclass
class PotentialTrade:
    """Pre-computed potential trade"""
    bar_idx: int
    entry_time: datetime
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    hour: int
    day_of_week: int
    regime: int
    regime_name: str
    signal_confidence: float
    adx: float
    rsi: float
    atr_pips: float


@dataclass
class TradeResult:
    """Result of a trade"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    pnl: float
    pips: float
    is_win: bool
    hour: int
    day_of_week: int
    regime_name: str
    signal_confidence: float
    adx: float
    close_reason: str


class FastZeroLossFinder:
    """
    Optimized zero-loss finder with pre-computation
    """

    def __init__(self):
        self.regime_detector = None
        self.signal_classifier = None
        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

        # Pre-computed data
        self.df = None
        self.regimes = None
        self.signals = None
        self.signal_probs = None
        self.potential_trades: List[PotentialTrade] = []

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

    def precompute_all(self, df: pd.DataFrame):
        """
        Pre-compute ALL features and predictions ONCE
        """
        logger.info("Pre-computing features (this happens only once)...")

        # Add features
        df = df.copy()
        df = self.tech_features.add_all_features(df)
        df = self.session_features.add_all_features(df)
        df = self.regime_features.add_all_features(df)
        df = self.profile_proxy.add_all_features(df)
        df = df.dropna()
        self.df = df

        logger.info("Pre-computing ML predictions...")

        # Predictions
        self.regimes = self.regime_detector.predict(df)
        regime_probs = self.regime_detector.predict_proba(df)
        self.signals = self.signal_classifier.predict(df)
        signal_probs_dict = self.signal_classifier.predict_proba(df)
        self.signal_probs = np.column_stack([
            signal_probs_dict['sell'],
            signal_probs_dict['hold'],
            signal_probs_dict['buy']
        ])

        logger.info("Pre-computing potential trades...")

        # Identify all potential trade entries
        regime_names = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}

        for i in range(50, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            signal = self.signals[i]
            if signal == 0:  # HOLD
                continue

            signal_conf = self.signal_probs[i].max()
            if signal_conf < 0.49:  # Min threshold
                continue

            regime = self.regimes[i]
            regime_name = regime_names.get(regime, 'unknown')

            current_adx = bar.get('adx_14', 0)
            if current_adx > 40 and regime == 2:
                regime = 0
                regime_name = 'trending_override'

            direction = 'BUY' if signal == 1 else 'SELL'
            rsi = bar.get('rsi_14', 50)
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            # Position sizing
            regime_mult = {0: 1.0, 1: 0.5, 2: 0.7}.get(regime, 0.7)
            sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}.get(regime, 1.5)
            tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}.get(regime, 1.5)

            sl_pips = atr_pips * sl_mult
            tp_pips = atr_pips * tp_mult

            entry_price = bar['close']
            pip_size = 0.0001

            if direction == 'BUY':
                sl_price = entry_price - sl_pips * pip_size
                tp_price = entry_price + tp_pips * pip_size
            else:
                sl_price = entry_price + sl_pips * pip_size
                tp_price = entry_price - tp_pips * pip_size

            # Lot size (approximate)
            lot_size = 0.1 * regime_mult * signal_conf

            pt = PotentialTrade(
                bar_idx=i,
                entry_time=bar_time,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                hour=bar_time.hour,
                day_of_week=bar_time.weekday(),
                regime=regime,
                regime_name=regime_name,
                signal_confidence=signal_conf,
                adx=current_adx,
                rsi=rsi,
                atr_pips=atr_pips
            )
            self.potential_trades.append(pt)

        logger.info(f"Pre-computed {len(self.potential_trades)} potential trades")

    def simulate_filtered(
        self,
        allowed_hours: List[int] = None,
        allowed_days: List[int] = None,
        allowed_regimes: List[str] = None,
        allowed_directions: List[str] = None,
        min_adx: float = 0,
        max_adx: float = 100,
        min_conf: float = 0.49,
        min_rsi: float = 0,
        max_rsi: float = 100
    ) -> List[TradeResult]:
        """
        Fast simulation using pre-computed data
        Properly handles sequential trades (one at a time)
        """
        # Filter potential trades
        filtered = []
        for pt in self.potential_trades:
            if allowed_hours and pt.hour not in allowed_hours:
                continue
            if allowed_days and pt.day_of_week not in allowed_days:
                continue
            if allowed_regimes and pt.regime_name not in allowed_regimes:
                continue
            if allowed_directions and pt.direction not in allowed_directions:
                continue
            if pt.adx < min_adx or pt.adx > max_adx:
                continue
            if pt.signal_confidence < min_conf:
                continue
            if pt.rsi < min_rsi or pt.rsi > max_rsi:
                continue
            filtered.append(pt)

        if not filtered:
            return []

        # Simulate trades sequentially (one at a time)
        results = []
        last_exit_idx = 0

        for pt in filtered:
            # Skip if still in previous trade
            if pt.bar_idx <= last_exit_idx:
                continue

            # Find exit for this trade
            exit_idx, exit_time, exit_price, close_reason = self._find_exit(pt)
            last_exit_idx = exit_idx

            if pt.direction == 'BUY':
                pips = (exit_price - pt.entry_price) / 0.0001
            else:
                pips = (pt.entry_price - exit_price) / 0.0001

            pnl = pips * 10.0 * pt.lot_size

            result = TradeResult(
                entry_time=pt.entry_time,
                exit_time=exit_time,
                direction=pt.direction,
                pnl=pnl,
                pips=pips,
                is_win=pnl > 0,
                hour=pt.hour,
                day_of_week=pt.day_of_week,
                regime_name=pt.regime_name,
                signal_confidence=pt.signal_confidence,
                adx=pt.adx,
                close_reason=close_reason
            )
            results.append(result)

        return results

    def _find_exit(self, pt: PotentialTrade) -> Tuple[int, datetime, float, str]:
        """Find exit point for a potential trade"""
        for i in range(pt.bar_idx + 1, len(self.df)):
            bar = self.df.iloc[i]
            bar_time = self.df.index[i]
            high = bar['high']
            low = bar['low']

            if pt.direction == 'BUY':
                if low <= pt.sl_price:
                    return i, bar_time, pt.sl_price, "SL"
                if high >= pt.tp_price:
                    return i, bar_time, pt.tp_price, "TP"
            else:
                if high >= pt.sl_price:
                    return i, bar_time, pt.sl_price, "SL"
                if low <= pt.tp_price:
                    return i, bar_time, pt.tp_price, "TP"

        # End of data
        return len(self.df) - 1, self.df.index[-1], self.df.iloc[-1]['close'], "END"

    def find_zero_loss_configs(self):
        """
        Test many filter combinations quickly
        """
        print("\n" + "=" * 70)
        print("FAST ZERO LOSS CONFIGURATION SEARCH")
        print("=" * 70)

        # Baseline
        all_trades = self.simulate_filtered()
        wins = len([t for t in all_trades if t.is_win])
        losses = len([t for t in all_trades if not t.is_win])
        print(f"\nBaseline: {len(all_trades)} trades, {wins}W/{losses}L")

        # Analyze losses
        losing = [t for t in all_trades if not t.is_win]
        print(f"\n--- LOSING TRADES ({len(losing)}) ---")
        for t in losing:
            print(f"  {t.entry_time} | {t.direction} | H{t.hour:02d} | Day{t.day_of_week} | {t.regime_name} | ADX={t.adx:.1f} | Conf={t.signal_confidence:.2f}")

        # Find common patterns
        print(f"\n--- PATTERNS IN LOSING TRADES ---")
        losing_hours = set(t.hour for t in losing)
        losing_days = set(t.day_of_week for t in losing)
        losing_regimes = set(t.regime_name for t in losing)
        losing_dirs = set(t.direction for t in losing)
        print(f"Hours: {sorted(losing_hours)}")
        print(f"Days: {sorted(losing_days)}")
        print(f"Regimes: {sorted(losing_regimes)}")
        print(f"Directions: {sorted(losing_dirs)}")

        # Test combinations
        print(f"\n--- TESTING FILTER COMBINATIONS ---")

        hour_filters = [
            None,
            [4, 9, 12, 20],  # Hours with 100% WR
            [4, 9, 12],
            [4, 12, 20],
            [4, 9],
            [4],
            [9],
            [12],
            [20],
            list(range(4, 10)),  # 04-09
            list(range(8, 13)),  # 08-12
            list(range(14, 21)),  # 14-20
        ]

        day_filters = [
            None,
            [0, 1, 3, 4],  # Skip Wed
            [1, 2, 3],  # Tue-Thu
            [0, 3, 4],  # Mon, Thu, Fri
            [3, 4],  # Thu, Fri
            [1, 3],  # Tue, Thu
        ]

        regime_filters = [
            None,
            ['trending_low_vol'],
            ['trending_low_vol', 'trending_override'],
            ['crisis_high_vol'],
            ['trending_low_vol', 'crisis_high_vol'],
        ]

        direction_filters = [
            None,
            ['BUY'],
            ['SELL'],
        ]

        adx_mins = [0, 20, 25, 30, 35, 40, 45, 50]
        conf_mins = [0.49, 0.50, 0.52, 0.55, 0.58, 0.60]

        zero_loss_configs = []
        tested = 0

        for hours in hour_filters:
            for days in day_filters:
                for regimes in regime_filters:
                    for dirs in direction_filters:
                        for min_adx in adx_mins:
                            for min_conf in conf_mins:
                                tested += 1

                                trades = self.simulate_filtered(
                                    allowed_hours=hours,
                                    allowed_days=days,
                                    allowed_regimes=regimes,
                                    allowed_directions=dirs,
                                    min_adx=min_adx,
                                    min_conf=min_conf
                                )

                                if not trades:
                                    continue

                                wins = len([t for t in trades if t.is_win])
                                losses = len([t for t in trades if not t.is_win])

                                if losses == 0 and wins > 0:
                                    total_pnl = sum(t.pnl for t in trades)
                                    config = {
                                        'hours': hours,
                                        'days': days,
                                        'regimes': regimes,
                                        'directions': dirs,
                                        'min_adx': min_adx,
                                        'min_conf': min_conf,
                                        'trades': len(trades),
                                        'wins': wins,
                                        'pnl': total_pnl
                                    }
                                    zero_loss_configs.append(config)

        print(f"Tested {tested} combinations")
        print(f"Found {len(zero_loss_configs)} zero-loss configs")

        # Sort by trade count
        zero_loss_configs.sort(key=lambda x: (-x['trades'], -x['pnl']))

        print(f"\n--- TOP 30 ZERO-LOSS CONFIGS ---")
        for i, cfg in enumerate(zero_loss_configs[:30]):
            print(f"\n{i+1}. {cfg['trades']} trades, {cfg['wins']}W/0L, ${cfg['pnl']:+,.2f}")
            print(f"   Hours: {cfg['hours']}")
            print(f"   Days: {cfg['days']}")
            print(f"   Regimes: {cfg['regimes']}")
            print(f"   Directions: {cfg['directions']}")
            print(f"   Min ADX: {cfg['min_adx']}, Min Conf: {cfg['min_conf']}")

        # Show the best config with trades detail
        if zero_loss_configs:
            best = zero_loss_configs[0]
            print(f"\n" + "=" * 70)
            print("BEST ZERO-LOSS CONFIG - TRADE DETAILS")
            print("=" * 70)

            trades = self.simulate_filtered(
                allowed_hours=best['hours'],
                allowed_days=best['days'],
                allowed_regimes=best['regimes'],
                allowed_directions=best['directions'],
                min_adx=best['min_adx'],
                min_conf=best['min_conf']
            )

            print(f"\n{'#':<3} {'Entry Time':<20} {'Dir':<5} {'Pips':>8} {'P/L':>10} {'Regime':<20} {'ADX':<6}")
            for i, t in enumerate(trades):
                print(f"{i+1:<3} {str(t.entry_time):<20} {t.direction:<5} {t.pips:>+8.1f} ${t.pnl:>+9,.2f} {t.regime_name:<20} {t.adx:<6.1f}")

            total_pnl = sum(t.pnl for t in trades)
            total_pips = sum(t.pips for t in trades)
            print(f"\nTOTAL: {total_pips:+.1f} pips, ${total_pnl:+,.2f}")

        return zero_loss_configs


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

    logger.info(f"Got {len(df)} H1 bars")

    finder = FastZeroLossFinder()
    if not finder.load_models():
        return

    finder.precompute_all(df)
    finder.find_zero_loss_configs()


if __name__ == "__main__":
    main()
