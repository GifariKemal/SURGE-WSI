"""
Zero Loss Finder V2
===================

More granular analysis to find TRUE zero-loss configurations.
Analyzes individual trades from actual backtest execution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger
import MetaTrader5 as mt5

from ml_trading_bot.models import RegimeDetector, SignalClassifier
from ml_trading_bot.features import TechnicalFeatures, SessionFeatures, RegimeFeatures
from ml_trading_bot.features.profile_proxy import ProfileProxyFeatures


@dataclass
class Trade:
    """Trade record"""
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
    regime: str
    signal_confidence: float
    adx: float
    rsi: float
    atr_pips: float
    close_reason: str


class ZeroLossFinderV2:
    """Find zero-loss configurations with precise backtest logic"""

    def __init__(self):
        self.regime_detector = None
        self.signal_classifier = None
        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

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

    def simulate_trades(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.49,
        allowed_hours: List[int] = None,
        allowed_days: List[int] = None,
        allowed_regimes: List[str] = None,
        allowed_directions: List[str] = None,
        min_adx: float = 0,
        max_adx: float = 100,
        min_rsi: float = 0,
        max_rsi: float = 100
    ) -> List[Trade]:
        """
        Simulate trades with filters and return trade list
        """
        df = self.prepare_features(df)
        df = df.dropna()

        # Pre-compute predictions
        regimes = self.regime_detector.predict(df)
        regime_probs = self.regime_detector.predict_proba(df)
        signals = self.signal_classifier.predict(df)
        signal_probs_dict = self.signal_classifier.predict_proba(df)
        signal_probs = np.column_stack([
            signal_probs_dict['sell'],
            signal_probs_dict['hold'],
            signal_probs_dict['buy']
        ])

        regime_names = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}

        trades = []
        current_position = None
        balance = 10000.0
        pip_value = 10.0

        for i in range(50, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            # Check existing position
            if current_position is not None:
                pos = current_position
                high = bar['high']
                low = bar['low']

                closed = False
                exit_price = 0
                close_reason = ""

                if pos['direction'] == 'BUY':
                    if low <= pos['sl_price']:
                        exit_price = pos['sl_price']
                        close_reason = "SL"
                        closed = True
                    elif high >= pos['tp_price']:
                        exit_price = pos['tp_price']
                        close_reason = "TP"
                        closed = True
                else:
                    if high >= pos['sl_price']:
                        exit_price = pos['sl_price']
                        close_reason = "SL"
                        closed = True
                    elif low <= pos['tp_price']:
                        exit_price = pos['tp_price']
                        close_reason = "TP"
                        closed = True

                if closed:
                    if pos['direction'] == 'BUY':
                        pips = (exit_price - pos['entry_price']) / 0.0001
                    else:
                        pips = (pos['entry_price'] - exit_price) / 0.0001

                    pnl = pips * pip_value * pos['lot_size']
                    balance += pnl

                    trade = Trade(
                        entry_time=pos['entry_time'],
                        exit_time=bar_time,
                        direction=pos['direction'],
                        entry_price=pos['entry_price'],
                        exit_price=exit_price,
                        pnl=pnl,
                        pips=pips,
                        is_win=pnl > 0,
                        hour=pos['hour'],
                        day_of_week=pos['day_of_week'],
                        regime=pos['regime_name'],
                        signal_confidence=pos['signal_conf'],
                        adx=pos['adx'],
                        rsi=pos['rsi'],
                        atr_pips=pos['atr_pips'],
                        close_reason=close_reason
                    )
                    trades.append(trade)
                    current_position = None
                continue

            # New entry logic
            hour = bar_time.hour
            day_of_week = bar_time.weekday()

            # Hour filter
            if allowed_hours and hour not in allowed_hours:
                continue

            # Day filter
            if allowed_days and day_of_week not in allowed_days:
                continue

            regime = regimes[i]
            regime_name = regime_names.get(regime, 'unknown')

            # ADX override
            current_adx = bar.get('adx_14', 0)
            if current_adx > 40 and regime == 2:
                regime = 0
                regime_name = 'trending_override'

            # Regime filter
            if allowed_regimes and regime_name not in allowed_regimes:
                continue

            # ADX filter
            if current_adx < min_adx or current_adx > max_adx:
                continue

            signal = signals[i]
            signal_conf = signal_probs[i].max()

            if signal == 0:
                continue
            if signal_conf < confidence_threshold:
                continue

            direction = 'BUY' if signal == 1 else 'SELL'

            # Direction filter
            if allowed_directions and direction not in allowed_directions:
                continue

            # RSI filter
            rsi = bar.get('rsi_14', 50)
            if rsi < min_rsi or rsi > max_rsi:
                continue

            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            # Position sizing
            regime_mult = {0: 1.0, 1: 0.5, 2: 0.7}.get(regime, 0.7)
            sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}.get(regime, 1.5)
            tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}.get(regime, 1.5)

            sl_pips = atr_pips * sl_mult
            tp_pips = atr_pips * tp_mult

            base_risk = balance * 0.04
            kelly_factor = signal_conf * 1.2
            conf_factor = (signal_conf - confidence_threshold) / (1 - confidence_threshold)
            conf_factor = 0.5 + (conf_factor * 0.5)

            risk_amount = base_risk * kelly_factor * regime_mult * conf_factor
            lot_size = risk_amount / (sl_pips * pip_value)
            lot_size = max(0.01, min(lot_size, 1.0))
            lot_size = round(lot_size, 2)

            entry_price = bar['close']
            pip_size = 0.0001

            if direction == 'BUY':
                sl_price = entry_price - sl_pips * pip_size
                tp_price = entry_price + tp_pips * pip_size
            else:
                sl_price = entry_price + sl_pips * pip_size
                tp_price = entry_price - tp_pips * pip_size

            current_position = {
                'entry_time': bar_time,
                'direction': direction,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'lot_size': lot_size,
                'hour': hour,
                'day_of_week': day_of_week,
                'regime': regime,
                'regime_name': regime_name,
                'signal_conf': signal_conf,
                'adx': current_adx,
                'rsi': rsi,
                'atr_pips': atr_pips
            }

        return trades

    def find_zero_loss_configs(self, df: pd.DataFrame):
        """
        Systematically find zero-loss configurations
        """
        print("\n" + "=" * 70)
        print("ZERO LOSS CONFIGURATION SEARCH V2")
        print("=" * 70)

        # First, get all trades with minimal filters
        all_trades = self.simulate_trades(df)
        print(f"\nBaseline: {len(all_trades)} trades")
        wins = len([t for t in all_trades if t.is_win])
        losses = len([t for t in all_trades if not t.is_win])
        print(f"W/L: {wins}/{losses} ({wins/len(all_trades)*100:.1f}% WR)")

        # Analyze losing trades
        losing_trades = [t for t in all_trades if not t.is_win]
        print(f"\n--- LOSING TRADES ANALYSIS ---")
        print(f"{'Time':<20} {'Dir':<5} {'Hour':<5} {'Day':<5} {'Regime':<20} {'ADX':<6} {'Conf':<6} {'RSI':<6}")
        for t in losing_trades:
            print(f"{str(t.entry_time):<20} {t.direction:<5} {t.hour:<5} {t.day_of_week:<5} {t.regime:<20} {t.adx:<6.1f} {t.signal_confidence:<6.2f} {t.rsi:<6.1f}")

        # Find common characteristics of losing trades
        losing_hours = set(t.hour for t in losing_trades)
        losing_days = set(t.day_of_week for t in losing_trades)
        losing_regimes = set(t.regime for t in losing_trades)
        losing_directions = set(t.direction for t in losing_trades)

        print(f"\nLosing trade characteristics:")
        print(f"  Hours: {sorted(losing_hours)}")
        print(f"  Days: {sorted(losing_days)}")
        print(f"  Regimes: {sorted(losing_regimes)}")
        print(f"  Directions: {sorted(losing_directions)}")

        # Test filter combinations
        results = []

        # Hour combinations
        all_hours = list(range(24))
        safe_hours = [h for h in all_hours if h not in losing_hours]

        # Day combinations
        all_days = list(range(5))

        # Regime options
        regime_options = [
            None,
            ['trending_low_vol'],
            ['trending_low_vol', 'trending_override'],
            ['crisis_high_vol'],
            ['trending_low_vol', 'crisis_high_vol'],
        ]

        # Direction options
        direction_options = [
            None,
            ['BUY'],
            ['SELL']
        ]

        # ADX thresholds
        adx_thresholds = [0, 20, 25, 30, 35, 40, 45, 50]

        # RSI ranges
        rsi_ranges = [
            (0, 100),
            (30, 70),
            (35, 65),
            (40, 60)
        ]

        print(f"\n--- TESTING FILTER COMBINATIONS ---")

        zero_loss_configs = []

        # Test each regime first
        for regimes in regime_options:
            for directions in direction_options:
                for min_adx in adx_thresholds:
                    for rsi_min, rsi_max in rsi_ranges:
                        trades = self.simulate_trades(
                            df,
                            allowed_regimes=regimes,
                            allowed_directions=directions,
                            min_adx=min_adx,
                            min_rsi=rsi_min,
                            max_rsi=rsi_max
                        )

                        if len(trades) == 0:
                            continue

                        wins = len([t for t in trades if t.is_win])
                        losses = len([t for t in trades if not t.is_win])
                        total_pnl = sum(t.pnl for t in trades)

                        if losses == 0 and wins > 0:
                            config = {
                                'regimes': regimes,
                                'directions': directions,
                                'min_adx': min_adx,
                                'rsi_range': (rsi_min, rsi_max),
                                'trades': len(trades),
                                'wins': wins,
                                'pnl': total_pnl
                            }
                            zero_loss_configs.append(config)

        # Sort by trade count
        zero_loss_configs.sort(key=lambda x: x['trades'], reverse=True)

        print(f"\nFound {len(zero_loss_configs)} zero-loss configurations")

        if zero_loss_configs:
            print(f"\n--- TOP 20 ZERO-LOSS CONFIGS ---")
            for i, cfg in enumerate(zero_loss_configs[:20]):
                print(f"\n{i+1}. {cfg['trades']} trades, {cfg['wins']}W/0L, ${cfg['pnl']:+,.2f}")
                print(f"   Regimes: {cfg['regimes']}")
                print(f"   Directions: {cfg['directions']}")
                print(f"   Min ADX: {cfg['min_adx']}")
                print(f"   RSI range: {cfg['rsi_range']}")

        # Now test adding hour/day filters to best base configs
        print(f"\n--- TESTING WITH HOUR/DAY FILTERS ---")

        hour_combos = [
            None,  # No filter
            [1, 2, 3, 4, 5],
            [8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 8, 9, 10],
            list(range(1, 12)),  # 01:00-11:00
        ]

        day_combos = [
            None,  # No filter
            [0, 1, 3, 4],  # Skip Wednesday
            [1, 2, 3],  # Tue-Thu
            [0, 1, 2, 3, 4],  # All weekdays
        ]

        best_configs_with_time = []

        for hours in hour_combos:
            for days in day_combos:
                for regimes in [None, ['trending_low_vol'], ['trending_low_vol', 'crisis_high_vol']]:
                    for directions in [None, ['BUY']]:
                        for min_adx in [0, 20, 30, 40]:
                            trades = self.simulate_trades(
                                df,
                                allowed_hours=hours,
                                allowed_days=days,
                                allowed_regimes=regimes,
                                allowed_directions=directions,
                                min_adx=min_adx
                            )

                            if len(trades) == 0:
                                continue

                            wins = len([t for t in trades if t.is_win])
                            losses = len([t for t in trades if not t.is_win])
                            total_pnl = sum(t.pnl for t in trades)

                            if losses == 0 and wins >= 1:
                                config = {
                                    'hours': hours,
                                    'days': days,
                                    'regimes': regimes,
                                    'directions': directions,
                                    'min_adx': min_adx,
                                    'trades': len(trades),
                                    'wins': wins,
                                    'pnl': total_pnl
                                }
                                best_configs_with_time.append(config)

        best_configs_with_time.sort(key=lambda x: x['trades'], reverse=True)

        print(f"\nFound {len(best_configs_with_time)} zero-loss configs with time filters")

        if best_configs_with_time:
            print(f"\n--- TOP 20 ZERO-LOSS WITH TIME FILTERS ---")
            for i, cfg in enumerate(best_configs_with_time[:20]):
                print(f"\n{i+1}. {cfg['trades']} trades, {cfg['wins']}W/0L, ${cfg['pnl']:+,.2f}")
                print(f"   Hours: {cfg['hours']}")
                print(f"   Days: {cfg['days']}")
                print(f"   Regimes: {cfg['regimes']}")
                print(f"   Directions: {cfg['directions']}")
                print(f"   Min ADX: {cfg['min_adx']}")

        # Return best config
        if best_configs_with_time:
            return best_configs_with_time[0]
        return None


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

    finder = ZeroLossFinderV2()
    if not finder.load_models():
        return

    best_config = finder.find_zero_loss_configs(df)

    if best_config:
        print("\n" + "=" * 70)
        print("BEST ZERO-LOSS CONFIGURATION")
        print("=" * 70)
        for key, value in best_config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
