"""
Zero Loss Trade Finder
======================

Systematically analyze trades to find filter combinations
that result in ZERO losses while maximizing trade count.

Strategy: Find common characteristics of winning trades
that are NOT present in losing trades.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from loguru import logger
import MetaTrader5 as mt5

# ML Models
from ml_trading_bot.models import RegimeDetector, SignalClassifier
from ml_trading_bot.features import TechnicalFeatures, SessionFeatures, RegimeFeatures
from ml_trading_bot.features.profile_proxy import ProfileProxyFeatures


@dataclass
class TradeRecord:
    """Detailed trade record for analysis"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pips: float
    is_win: bool

    # Context at entry
    hour: int
    day_of_week: int  # 0=Monday
    regime: int
    regime_name: str
    signal_confidence: float
    adx: float
    atr_pips: float
    rsi: float
    close_reason: str

    # Additional metrics
    trend_strength: float = 0.0
    volatility_percentile: float = 0.0
    session: str = ""


class ZeroLossFinder:
    """
    Find trading configurations with zero losses
    """

    def __init__(self):
        # ML Models
        self.regime_detector = None
        self.signal_classifier = None

        # Feature engineers
        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

        # Trade records
        self.all_trades: List[TradeRecord] = []

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
        """Add all features to dataframe"""
        df = df.copy()
        df = self.tech_features.add_all_features(df)
        df = self.session_features.add_all_features(df)
        df = self.regime_features.add_all_features(df)
        df = self.profile_proxy.add_all_features(df)
        return df

    def collect_all_trades(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.49,
        kelly_fraction: float = 1.2,
        base_risk_pct: float = 0.04
    ) -> List[TradeRecord]:
        """
        Collect ALL possible trades with detailed context
        """
        logger.info(f"Collecting trades with conf={confidence_threshold}, kelly={kelly_fraction}, risk={base_risk_pct}")

        # Prepare features
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

        # Trading params
        initial_balance = 10000.0
        balance = initial_balance
        pip_value = 10.0

        trades = []
        current_position = None

        regime_names = {0: 'trending_low_vol', 1: 'crisis_high_vol', 2: 'ranging_choppy'}

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
                    # Calculate P/L
                    if pos['direction'] == 'BUY':
                        pips = (exit_price - pos['entry_price']) / 0.0001
                    else:
                        pips = (pos['entry_price'] - exit_price) / 0.0001

                    pnl = pips * pip_value * pos['lot_size']
                    balance += pnl

                    # Determine session
                    hour = pos['entry_time'].hour
                    if 0 <= hour < 8:
                        session = "Asian"
                    elif 8 <= hour < 13:
                        session = "London"
                    elif 13 <= hour < 17:
                        session = "Overlap"
                    else:
                        session = "NY"

                    trade = TradeRecord(
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
                        regime=pos['regime'],
                        regime_name=pos['regime_name'],
                        signal_confidence=pos['signal_conf'],
                        adx=pos['adx'],
                        atr_pips=pos['atr_pips'],
                        rsi=pos['rsi'],
                        close_reason=close_reason,
                        trend_strength=pos['trend_strength'],
                        volatility_percentile=pos['vol_pct'],
                        session=session
                    )
                    trades.append(trade)
                    current_position = None

                continue

            # New entry logic
            regime = regimes[i]
            regime_conf = regime_probs[i].max() if regime_probs is not None else 0.7
            regime_name = regime_names.get(regime, 'unknown')

            # ADX check
            current_adx = bar.get('adx_14', 0)
            if current_adx > 40 and regime == 2:
                regime = 0
                regime_name = 'trending_override'

            signal = signals[i]
            signal_conf = signal_probs[i].max()

            # Skip HOLD or low confidence
            if signal == 0:
                continue
            if signal_conf < confidence_threshold:
                continue

            direction = 'BUY' if signal == 1 else 'SELL'

            # Get metrics
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 5:
                atr_pips = 15

            rsi = bar.get('rsi_14', 50)
            trend_strength = bar.get('adx_14', 20)
            vol_pct = bar.get('atr_percentile', 50) if 'atr_percentile' in df.columns else 50

            # Position sizing
            regime_mult = {0: 1.0, 1: 0.5, 2: 0.7}.get(regime, 0.7)
            sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}.get(regime, 1.5)
            tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}.get(regime, 1.5)

            sl_pips = atr_pips * sl_mult
            tp_pips = atr_pips * tp_mult

            base_risk = balance * base_risk_pct
            kelly_factor = signal_conf * kelly_fraction
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
                'hour': bar_time.hour,
                'day_of_week': bar_time.weekday(),
                'regime': regime,
                'regime_name': regime_name,
                'signal_conf': signal_conf,
                'adx': current_adx,
                'atr_pips': atr_pips,
                'rsi': rsi,
                'trend_strength': trend_strength,
                'vol_pct': vol_pct
            }

        self.all_trades = trades
        return trades

    def analyze_trade_patterns(self):
        """Analyze winning vs losing trade patterns"""
        if not self.all_trades:
            logger.error("No trades to analyze")
            return

        wins = [t for t in self.all_trades if t.is_win]
        losses = [t for t in self.all_trades if not t.is_win]

        print("\n" + "=" * 70)
        print("TRADE PATTERN ANALYSIS")
        print("=" * 70)
        print(f"\nTotal Trades: {len(self.all_trades)}")
        print(f"Wins: {len(wins)} ({len(wins)/len(self.all_trades)*100:.1f}%)")
        print(f"Losses: {len(losses)} ({len(losses)/len(self.all_trades)*100:.1f}%)")

        # Hour analysis
        print("\n" + "-" * 70)
        print("HOUR ANALYSIS")
        print("-" * 70)
        print(f"{'Hour':<6} {'Wins':<6} {'Losses':<8} {'WR':<8} {'Status'}")
        print("-" * 40)

        hour_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        for t in self.all_trades:
            if t.is_win:
                hour_stats[t.hour]['wins'] += 1
            else:
                hour_stats[t.hour]['losses'] += 1

        safe_hours = []
        danger_hours = []

        for hour in sorted(hour_stats.keys()):
            stats = hour_stats[hour]
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total * 100 if total > 0 else 0

            if stats['losses'] == 0 and stats['wins'] > 0:
                status = "SAFE (0 losses)"
                safe_hours.append(hour)
            elif wr < 50:
                status = "DANGER"
                danger_hours.append(hour)
            else:
                status = ""

            print(f"{hour:02d}:00  {stats['wins']:<6} {stats['losses']:<8} {wr:.0f}%    {status}")

        # Day of week analysis
        print("\n" + "-" * 70)
        print("DAY OF WEEK ANALYSIS")
        print("-" * 70)

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for t in self.all_trades:
            if t.is_win:
                day_stats[t.day_of_week]['wins'] += 1
            else:
                day_stats[t.day_of_week]['losses'] += 1

        safe_days = []
        danger_days = []

        print(f"{'Day':<12} {'Wins':<6} {'Losses':<8} {'WR':<8} {'Status'}")
        print("-" * 45)

        for day in range(5):
            stats = day_stats[day]
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total * 100 if total > 0 else 0

            if stats['losses'] == 0 and stats['wins'] > 0:
                status = "SAFE (0 losses)"
                safe_days.append(day)
            elif wr < 50:
                status = "DANGER"
                danger_days.append(day)
            else:
                status = ""

            print(f"{day_names[day]:<12} {stats['wins']:<6} {stats['losses']:<8} {wr:.0f}%    {status}")

        # Regime analysis
        print("\n" + "-" * 70)
        print("REGIME ANALYSIS")
        print("-" * 70)

        regime_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        for t in self.all_trades:
            if t.is_win:
                regime_stats[t.regime_name]['wins'] += 1
            else:
                regime_stats[t.regime_name]['losses'] += 1

        safe_regimes = []

        print(f"{'Regime':<20} {'Wins':<6} {'Losses':<8} {'WR':<8} {'Status'}")
        print("-" * 50)

        for regime in sorted(regime_stats.keys()):
            stats = regime_stats[regime]
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total * 100 if total > 0 else 0

            if stats['losses'] == 0 and stats['wins'] > 0:
                status = "SAFE (0 losses)"
                safe_regimes.append(regime)
            elif wr < 50:
                status = "DANGER"
            else:
                status = ""

            print(f"{regime:<20} {stats['wins']:<6} {stats['losses']:<8} {wr:.0f}%    {status}")

        # Session analysis
        print("\n" + "-" * 70)
        print("SESSION ANALYSIS")
        print("-" * 70)

        session_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        for t in self.all_trades:
            if t.is_win:
                session_stats[t.session]['wins'] += 1
            else:
                session_stats[t.session]['losses'] += 1

        safe_sessions = []

        print(f"{'Session':<12} {'Wins':<6} {'Losses':<8} {'WR':<8} {'Status'}")
        print("-" * 45)

        for session in ['Asian', 'London', 'Overlap', 'NY']:
            stats = session_stats[session]
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total * 100 if total > 0 else 0

            if stats['losses'] == 0 and stats['wins'] > 0:
                status = "SAFE (0 losses)"
                safe_sessions.append(session)
            elif wr < 50:
                status = "DANGER"
            else:
                status = ""

            print(f"{session:<12} {stats['wins']:<6} {stats['losses']:<8} {wr:.0f}%    {status}")

        # ADX analysis
        print("\n" + "-" * 70)
        print("ADX THRESHOLD ANALYSIS")
        print("-" * 70)

        for threshold in [20, 25, 30, 35, 40, 45, 50]:
            above_wins = len([t for t in wins if t.adx >= threshold])
            above_losses = len([t for t in losses if t.adx >= threshold])
            total = above_wins + above_losses
            wr = above_wins / total * 100 if total > 0 else 0
            status = "SAFE" if above_losses == 0 and above_wins > 0 else ""
            print(f"ADX >= {threshold}: {above_wins}W / {above_losses}L ({wr:.0f}% WR) {status}")

        # Confidence analysis
        print("\n" + "-" * 70)
        print("CONFIDENCE THRESHOLD ANALYSIS")
        print("-" * 70)

        for threshold in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
            above_wins = len([t for t in wins if t.signal_confidence >= threshold])
            above_losses = len([t for t in losses if t.signal_confidence >= threshold])
            total = above_wins + above_losses
            wr = above_wins / total * 100 if total > 0 else 0
            status = "SAFE" if above_losses == 0 and above_wins > 0 else ""
            print(f"Conf >= {threshold:.2f}: {above_wins}W / {above_losses}L ({wr:.0f}% WR) {status}")

        # RSI analysis
        print("\n" + "-" * 70)
        print("RSI ANALYSIS")
        print("-" * 70)

        for rsi_low, rsi_high in [(30, 70), (35, 65), (40, 60), (45, 55)]:
            in_range_wins = len([t for t in wins if rsi_low <= t.rsi <= rsi_high])
            in_range_losses = len([t for t in losses if rsi_low <= t.rsi <= rsi_high])
            total = in_range_wins + in_range_losses
            wr = in_range_wins / total * 100 if total > 0 else 0
            status = "SAFE" if in_range_losses == 0 and in_range_wins > 0 else ""
            print(f"RSI {rsi_low}-{rsi_high}: {in_range_wins}W / {in_range_losses}L ({wr:.0f}% WR) {status}")

        # Direction analysis
        print("\n" + "-" * 70)
        print("DIRECTION ANALYSIS")
        print("-" * 70)

        for direction in ['BUY', 'SELL']:
            dir_wins = len([t for t in wins if t.direction == direction])
            dir_losses = len([t for t in losses if t.direction == direction])
            total = dir_wins + dir_losses
            wr = dir_wins / total * 100 if total > 0 else 0
            status = "SAFE" if dir_losses == 0 and dir_wins > 0 else ""
            print(f"{direction}: {dir_wins}W / {dir_losses}L ({wr:.0f}% WR) {status}")

        return {
            'safe_hours': safe_hours,
            'danger_hours': danger_hours,
            'safe_days': safe_days,
            'danger_days': danger_days,
            'safe_regimes': safe_regimes,
            'safe_sessions': safe_sessions
        }

    def test_filter_combinations(self) -> List[Dict]:
        """
        Test many filter combinations to find zero-loss configs
        """
        if not self.all_trades:
            logger.error("No trades to filter")
            return []

        print("\n" + "=" * 70)
        print("TESTING FILTER COMBINATIONS")
        print("=" * 70)

        results = []

        # Generate filter combinations
        hour_filters = [
            None,  # No filter
            list(range(1, 6)),  # 01-05
            list(range(8, 12)),  # 08-11
            list(range(13, 17)),  # 13-16
            [2, 3, 4, 5, 9, 10, 14, 15],  # Best hours from analysis
            [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 20, 21, 22],  # Skip worst hours
        ]

        day_filters = [
            None,  # No filter
            [0, 1, 3, 4],  # Skip Wednesday
            [1, 2, 3],  # Tue-Thu
            [0, 1, 2, 3],  # Mon-Thu
        ]

        regime_filters = [
            None,  # No filter
            ['trending_low_vol'],
            ['trending_low_vol', 'trending_override'],
            ['trending_low_vol', 'crisis_high_vol'],
        ]

        adx_thresholds = [0, 20, 25, 30, 35, 40, 45, 50]
        conf_thresholds = [0.49, 0.52, 0.55, 0.58, 0.60, 0.65]

        # Test combinations
        combo_count = 0
        zero_loss_configs = []

        for hours in hour_filters:
            for days in day_filters:
                for regimes in regime_filters:
                    for min_adx in adx_thresholds:
                        for min_conf in conf_thresholds:
                            combo_count += 1

                            # Filter trades
                            filtered = self.all_trades.copy()

                            if hours is not None:
                                filtered = [t for t in filtered if t.hour in hours]

                            if days is not None:
                                filtered = [t for t in filtered if t.day_of_week in days]

                            if regimes is not None:
                                filtered = [t for t in filtered if t.regime_name in regimes]

                            if min_adx > 0:
                                filtered = [t for t in filtered if t.adx >= min_adx]

                            if min_conf > 0.49:
                                filtered = [t for t in filtered if t.signal_confidence >= min_conf]

                            if len(filtered) == 0:
                                continue

                            wins = len([t for t in filtered if t.is_win])
                            losses = len([t for t in filtered if not t.is_win])
                            total_pnl = sum(t.pnl for t in filtered)

                            result = {
                                'hours': hours,
                                'days': days,
                                'regimes': regimes,
                                'min_adx': min_adx,
                                'min_conf': min_conf,
                                'trades': len(filtered),
                                'wins': wins,
                                'losses': losses,
                                'wr': wins / len(filtered) * 100,
                                'pnl': total_pnl
                            }
                            results.append(result)

                            # Track zero-loss configs with trades
                            if losses == 0 and wins > 0:
                                zero_loss_configs.append(result)

        print(f"\nTested {combo_count} combinations")
        print(f"Found {len(zero_loss_configs)} zero-loss configurations")

        # Sort zero-loss by trade count
        zero_loss_configs.sort(key=lambda x: x['trades'], reverse=True)

        print("\n" + "-" * 70)
        print("TOP ZERO-LOSS CONFIGURATIONS (by trade count)")
        print("-" * 70)

        for i, cfg in enumerate(zero_loss_configs[:20]):
            print(f"\n{i+1}. {cfg['trades']} trades, {cfg['wins']}W/0L, ${cfg['pnl']:+,.2f}")
            print(f"   Hours: {cfg['hours']}")
            print(f"   Days: {cfg['days']}")
            print(f"   Regimes: {cfg['regimes']}")
            print(f"   Min ADX: {cfg['min_adx']}, Min Conf: {cfg['min_conf']}")

        # Also show high trade count configs (regardless of losses)
        results.sort(key=lambda x: (-x['trades'], -x['wr']))
        high_wr = [r for r in results if r['wr'] >= 70 and r['trades'] >= 5]

        print("\n" + "-" * 70)
        print("HIGH WIN RATE CONFIGS (>= 70% WR, >= 5 trades)")
        print("-" * 70)

        for i, cfg in enumerate(high_wr[:15]):
            print(f"\n{i+1}. {cfg['trades']} trades, {cfg['wins']}W/{cfg['losses']}L ({cfg['wr']:.0f}%), ${cfg['pnl']:+,.2f}")
            print(f"   Hours: {cfg['hours']}")
            print(f"   Days: {cfg['days']}")
            print(f"   Regimes: {cfg['regimes']}")
            print(f"   Min ADX: {cfg['min_adx']}, Min Conf: {cfg['min_conf']}")

        return zero_loss_configs

    def find_multi_confirmation_trades(self):
        """
        Find trades where multiple confirmations align
        """
        print("\n" + "=" * 70)
        print("MULTI-CONFIRMATION ANALYSIS")
        print("=" * 70)

        # Define confirmation criteria
        def count_confirmations(trade: TradeRecord) -> int:
            count = 0

            # 1. Trending regime
            if trade.regime_name in ['trending_low_vol', 'trending_override']:
                count += 1

            # 2. High ADX
            if trade.adx >= 35:
                count += 1

            # 3. High confidence
            if trade.signal_confidence >= 0.55:
                count += 1

            # 4. Good session (London or Overlap)
            if trade.session in ['London', 'Overlap']:
                count += 1

            # 5. Good day (not Wednesday)
            if trade.day_of_week != 2:
                count += 1

            # 6. RSI not extreme
            if 35 <= trade.rsi <= 65:
                count += 1

            return count

        # Analyze by confirmation count
        for min_confirms in range(1, 7):
            filtered = [t for t in self.all_trades if count_confirmations(t) >= min_confirms]
            if not filtered:
                continue

            wins = len([t for t in filtered if t.is_win])
            losses = len([t for t in filtered if not t.is_win])
            total_pnl = sum(t.pnl for t in filtered)
            wr = wins / len(filtered) * 100

            status = "*** ZERO LOSS ***" if losses == 0 else ""
            print(f"\n>= {min_confirms} confirmations: {len(filtered)} trades, {wins}W/{losses}L ({wr:.0f}%), ${total_pnl:+,.2f} {status}")

            if losses == 0 and wins > 0:
                print("  WINNING TRADES:")
                for t in filtered:
                    print(f"    - {t.entry_time}: {t.direction} @ {t.entry_price:.5f}, +${t.pnl:.2f}")
                    print(f"      Hour={t.hour}, Day={t.day_of_week}, Regime={t.regime_name}, ADX={t.adx:.1f}, Conf={t.signal_confidence:.2f}")


def main():
    """Find zero-loss trading configurations"""

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

    logger.info(f"Got {len(df)} H1 bars")

    # Create finder
    finder = ZeroLossFinder()

    if not finder.load_models():
        return

    # Collect all trades with aggressive params (to get more data)
    trades = finder.collect_all_trades(
        df,
        confidence_threshold=0.49,
        kelly_fraction=1.2,
        base_risk_pct=0.04
    )

    print(f"\nCollected {len(trades)} total trades")

    # Analyze patterns
    patterns = finder.analyze_trade_patterns()

    # Test filter combinations
    zero_loss_configs = finder.test_filter_combinations()

    # Multi-confirmation analysis
    finder.find_multi_confirmation_trades()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal zero-loss configurations found: {len(zero_loss_configs)}")
    if zero_loss_configs:
        best = zero_loss_configs[0]
        print(f"\nBEST CONFIG (most trades with zero loss):")
        print(f"  Trades: {best['trades']}")
        print(f"  P/L: ${best['pnl']:+,.2f}")
        print(f"  Hours: {best['hours']}")
        print(f"  Days: {best['days']}")
        print(f"  Regimes: {best['regimes']}")
        print(f"  Min ADX: {best['min_adx']}")
        print(f"  Min Conf: {best['min_conf']}")


if __name__ == "__main__":
    main()
