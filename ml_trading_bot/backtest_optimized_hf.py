"""
Optimized High Frequency Backtest
=================================

Based on hybrid backtest insights:
- ONLY use EMA crossover signals (best performer: 56.8% WR)
- Focus on trending regime (best for EMA cross)
- Add ADX confirmation for trend strength
- Better SL/TP ratios

Target: 120+ trades, profitable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger

from ml_trading_bot.models import RegimeDetector
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
    adx: float = 0.0
    close_reason: str = ""


class OptimizedHFBacktester:
    """
    Optimized strategy - Multi-signal with low ADX threshold
    - EMA crossover + MACD confirmation
    - Lower ADX threshold (15) for more signals
    - All regimes allowed
    """

    TRADING_HOURS = list(range(4, 21))  # 04:00 - 20:00 UTC (longer hours)

    REGIME_NAMES = {0: 'trending', 1: 'crisis', 2: 'ranging'}

    # Trade all regimes but with different sizing
    REGIME_MULT = {0: 1.0, 1: 0.5, 2: 0.7}

    # Tighter SL, wider TP for better R:R
    SL_MULT = {0: 1.0, 1: 1.5, 2: 1.0}
    TP_MULT = {0: 1.6, 1: 2.0, 2: 1.4}

    def __init__(
        self,
        initial_balance: float = 10000.0,
        base_risk_pct: float = 0.01,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.12,
        pip_value: float = 10.0,
        min_adx: float = 15.0,           # Lower ADX filter for more trades
        ema_fast: int = 9,
        ema_slow: int = 21
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance

        self.base_risk_pct = base_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.pip_value = pip_value
        self.min_adx = min_adx
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

        self.regime_detector = None
        self.tech_features = TechnicalFeatures()
        self.session_features = SessionFeatures()
        self.regime_features = RegimeFeatures()
        self.profile_proxy = ProfileProxyFeatures()

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_pnl: Dict[str, float] = {}

        self.position: Optional[Trade] = None
        self.circuit_breaker_dates: set = set()
        self.max_dd_triggered = False

    def load_models(self) -> bool:
        try:
            models_dir = Path(__file__).parent / "saved_models"
            self.regime_detector = RegimeDetector()
            self.regime_detector.load(models_dir / "regime_hmm.pkl")
            logger.info("Regime detector loaded")
            return True
        except Exception as e:
            logger.error(f"Failed: {e}")
            return False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.tech_features.add_all_features(df)
        df = self.session_features.add_all_features(df)
        df = self.regime_features.add_all_features(df)
        df = self.profile_proxy.add_all_features(df)
        return df

    def generate_signal(self, bar: pd.Series, prev_bar: pd.Series) -> str:
        """
        Multi-signal: EMA crossover + MACD + RSI extreme
        """
        ema_9 = bar.get('ema_9', 0)
        ema_21 = bar.get('ema_21', 0)
        prev_ema_9 = prev_bar.get('ema_9', 0)
        prev_ema_21 = prev_bar.get('ema_21', 0)
        adx = bar.get('adx_14', 0)
        macd = bar.get('macd', 0)
        macd_signal = bar.get('macd_signal', 0)
        prev_macd = prev_bar.get('macd', 0)
        prev_macd_signal = prev_bar.get('macd_signal', 0)
        rsi = bar.get('rsi_14', 50)

        # ADX filter - lowered threshold
        if adx < self.min_adx:
            return None

        # Signal 1: EMA Crossover (best performer)
        if prev_ema_9 <= prev_ema_21 and ema_9 > ema_21:
            return 'BUY'
        if prev_ema_9 >= prev_ema_21 and ema_9 < ema_21:
            return 'SELL'

        # Signal 2: MACD crossover with trend confirmation
        if adx > 20:  # Only in trending
            if prev_macd <= prev_macd_signal and macd > macd_signal and macd < 0:
                return 'BUY'  # Bullish crossover below zero
            if prev_macd >= prev_macd_signal and macd < macd_signal and macd > 0:
                return 'SELL'  # Bearish crossover above zero

        # Signal 3: RSI extreme with trend
        if rsi < 30 and ema_9 > ema_21:  # Oversold in uptrend
            return 'BUY'
        if rsi > 70 and ema_9 < ema_21:  # Overbought in downtrend
            return 'SELL'

        return None

    def calculate_position(self, regime: int, atr_pips: float) -> tuple:
        regime_mult = self.REGIME_MULT.get(regime, 0.5)
        sl_mult = self.SL_MULT.get(regime, 1.0)
        tp_mult = self.TP_MULT.get(regime, 1.5)

        sl_pips = max(atr_pips * sl_mult, 12)
        tp_pips = max(atr_pips * tp_mult, 18)

        risk_amount = self.balance * self.base_risk_pct * regime_mult
        lot_size = risk_amount / (sl_pips * self.pip_value)
        lot_size = max(0.01, min(lot_size, 0.3))
        lot_size = round(lot_size, 2)

        return lot_size, sl_pips, tp_pips

    def check_daily_loss(self, date: str) -> bool:
        if date in self.circuit_breaker_dates:
            return True
        loss = self.daily_pnl.get(date, 0.0)
        if loss < 0 and abs(loss) >= self.balance * self.max_daily_loss_pct:
            self.circuit_breaker_dates.add(date)
            return True
        return False

    def check_max_drawdown(self) -> bool:
        if self.max_dd_triggered:
            return True
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd = (self.peak_balance - self.balance) / self.peak_balance
        if dd >= self.max_drawdown_pct:
            self.max_dd_triggered = True
            logger.warning(f"MAX DD: {dd:.2%}")
            return True
        return False

    def run(self, df: pd.DataFrame) -> dict:
        logger.info(f"Starting OPTIMIZED backtest: {len(df)} bars")

        logger.info("Preparing features...")
        df = self.prepare_features(df)
        df = df.dropna()
        logger.info(f"After prep: {len(df)} bars")

        logger.info("Computing regimes...")
        regimes = self.regime_detector.predict(df)

        logger.info("Running simulation...")
        self.equity_curve = [self.initial_balance]

        stats = {'skipped_hour': 0, 'skipped_adx': 0, 'skipped_no_signal': 0, 'skipped_daily': 0}

        for i in range(51, len(df)):
            bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            bar_time = df.index[i]
            current_date = bar_time.strftime('%Y-%m-%d')
            hour = bar_time.hour

            if self.check_max_drawdown():
                break

            if self.check_daily_loss(current_date):
                stats['skipped_daily'] += 1
                if self.position:
                    self._check_exit(bar, bar_time)
                continue

            if self.position:
                self._check_exit(bar, bar_time)
                continue

            if hour not in self.TRADING_HOURS:
                stats['skipped_hour'] += 1
                continue

            direction = self.generate_signal(bar, prev_bar)

            if direction is None:
                stats['skipped_no_signal'] += 1
                continue

            regime = regimes[i]
            atr_pips = bar.get('atr_14', 0.0015) * 10000
            if atr_pips < 8:
                atr_pips = 15

            lot_size, sl_pips, tp_pips = self.calculate_position(regime, atr_pips)

            entry_price = bar['close']
            pip_size = 0.0001

            if direction == 'BUY':
                sl_price = entry_price - sl_pips * pip_size
                tp_price = entry_price + tp_pips * pip_size
            else:
                sl_price = entry_price + sl_pips * pip_size
                tp_price = entry_price - tp_pips * pip_size

            adx = bar.get('adx_14', 0)

            self.position = Trade(
                entry_time=bar_time,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                regime=self.REGIME_NAMES.get(regime, 'unknown'),
                adx=adx
            )

        if self.position:
            last_bar = df.iloc[-1]
            self._close(last_bar['close'], df.index[-1], "END")

        logger.info("Backtest complete!")
        return self._calc_stats(stats)

    def _check_exit(self, bar: pd.Series, bar_time: datetime):
        if not self.position:
            return

        pos = self.position
        high, low = bar['high'], bar['low']

        if pos.direction == 'BUY':
            if low <= pos.sl_price:
                self._close(pos.sl_price, bar_time, "SL")
            elif high >= pos.tp_price:
                self._close(pos.tp_price, bar_time, "TP")
        else:
            if high >= pos.sl_price:
                self._close(pos.sl_price, bar_time, "SL")
            elif low <= pos.tp_price:
                self._close(pos.tp_price, bar_time, "TP")

    def _close(self, exit_price: float, exit_time: datetime, reason: str):
        if not self.position:
            return

        pos = self.position
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

        date = exit_time.strftime('%Y-%m-%d')
        self.daily_pnl[date] = self.daily_pnl.get(date, 0.0) + pnl

        self.trades.append(pos)
        self.position = None

    def _calc_stats(self, filter_stats: dict) -> dict:
        if not self.trades:
            return {'total_trades': 0}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        peak = self.initial_balance
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        monthly_pnl = {}
        monthly_trades = {}
        for t in self.trades:
            m = t.entry_time.strftime('%Y-%m')
            monthly_pnl[m] = monthly_pnl.get(m, 0) + t.pnl
            monthly_trades[m] = monthly_trades.get(m, 0) + 1

        regime_stats = {}
        for t in self.trades:
            if t.regime not in regime_stats:
                regime_stats[t.regime] = {'trades': 0, 'wins': 0, 'pnl': 0}
            regime_stats[t.regime]['trades'] += 1
            regime_stats[t.regime]['pnl'] += t.pnl
            if t.pnl > 0:
                regime_stats[t.regime]['wins'] += 1

        # Win/Loss by close reason
        close_stats = {'TP': {'count': 0, 'pnl': 0}, 'SL': {'count': 0, 'pnl': 0}}
        for t in self.trades:
            if t.close_reason in close_stats:
                close_stats[t.close_reason]['count'] += 1
                close_stats[t.close_reason]['pnl'] += t.pnl

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
            'close_stats': close_stats,
            'filter_stats': filter_stats,
            'final_balance': self.balance
        }

    def print_results(self, stats: dict):
        print("\n" + "=" * 70)
        print("OPTIMIZED EMA CROSSOVER BACKTEST")
        print("=" * 70)

        print(f"\nPeriod: Jan 2025 - Jan 2026")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${stats['final_balance']:,.2f}")
        ret = (stats['final_balance']/self.initial_balance - 1)*100
        print(f"Total Return: {ret:+.2f}%")

        print("\n" + "-" * 70)
        print("STRATEGY")
        print("-" * 70)
        print(f"Signal: EMA {self.ema_fast}/{self.ema_slow} Crossover")
        print(f"ADX Filter: > {self.min_adx}")
        print(f"Hours: {self.TRADING_HOURS[0]:02d}:00 - {self.TRADING_HOURS[-1]:02d}:00 UTC")
        print(f"Risk/Trade: {self.base_risk_pct:.1%}")

        print("\n" + "-" * 70)
        print("STATISTICS")
        print("-" * 70)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win/Loss: {stats['winning_trades']}/{stats['losing_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Total P/L: ${stats['total_pnl']:+,.2f} ({stats['total_pips']:+.0f} pips)")
        print(f"Avg Win: ${stats['avg_win']:.2f} | Avg Loss: ${stats['avg_loss']:.2f}")
        print(f"Max DD: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_pct']:.1f}%)")

        print("\n" + "-" * 70)
        print("BY CLOSE REASON")
        print("-" * 70)
        for reason, s in stats['close_stats'].items():
            print(f"{reason}: {s['count']} trades, ${s['pnl']:+.2f}")

        print("\n" + "-" * 70)
        print("BY REGIME")
        print("-" * 70)
        for reg, s in stats['regime_stats'].items():
            wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
            print(f"{reg:<12} Trades: {s['trades']:>4} | WR: {wr:>5.1f}% | P/L: ${s['pnl']:>+8.2f}")

        print("\n" + "-" * 70)
        print("MONTHLY")
        print("-" * 70)
        print(f"{'Month':<10} {'Trades':>6} {'P/L':>10} {'Cum':>10}")
        cum = 0
        for m in sorted(stats['monthly_pnl'].keys()):
            t = stats['monthly_trades'].get(m, 0)
            p = stats['monthly_pnl'].get(m, 0)
            cum += p
            print(f"{m:<10} {t:>6} ${p:>+8.2f} ${cum:>+8.2f}")

        print("\n" + "=" * 70)
        if stats['total_trades'] >= 120:
            print(f"[OK] TARGET: {stats['total_trades']} trades")
        else:
            print(f"[X] Need more trades: {stats['total_trades']}/120")

        if ret > 0:
            print(f"[OK] PROFITABLE: {ret:+.2f}%")
        else:
            print(f"[X] LOSING: {ret:+.2f}%")
        print("=" * 70)


def main():
    import MetaTrader5 as mt5

    if not mt5.initialize(
        path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        login=10009310110,
        password="P-WyAnG8",
        server="MetaQuotes-Demo"
    ):
        logger.error("MT5 failed")
        return

    logger.info("MT5 connected")

    rates = mt5.copy_rates_range(
        "GBPUSD", mt5.TIMEFRAME_H1,
        datetime(2025, 1, 1), datetime(2026, 1, 31)
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        logger.error("No data")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    logger.info(f"Got {len(df)} bars")

    bt = OptimizedHFBacktester(
        initial_balance=10000.0,
        base_risk_pct=0.01,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.12,
        min_adx=15.0
    )

    if not bt.load_models():
        return

    stats = bt.run(df)
    bt.print_results(stats)


if __name__ == "__main__":
    main()
