"""Test Intelligent Activity Filter
====================================

Compare:
1. Kill Zone ON (traditional)
2. Kill Zone OFF (no time filter)
3. Intelligent Filter (velocity-based)

This test uses a custom backtest loop to properly integrate
the Intelligent Activity Filter with Kalman velocity.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector, MarketRegime
from src.analysis.poi_detector import POIDetector
from src.analysis.market_filter import MarketFilter
from src.trading.entry_trigger import EntryTrigger
from src.trading.risk_manager import RiskManager
from src.trading.exit_manager import ExitManager
from src.utils.killzone import KillZone
from src.utils.intelligent_activity_filter import IntelligentActivityFilter, MarketActivity

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


@dataclass
class TestConfig:
    name: str
    use_killzone: bool
    use_intelligent: bool
    activity_threshold: float = 40.0
    max_sl_pips: float = 30.0
    min_quality: float = 65.0
    max_loss_pct: float = 0.4


@dataclass
class SimpleTrade:
    """Simple trade for backtest"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    sl_price: float
    pnl: float
    quality: float


class SimpleBacktester:
    """Simplified backtester with Intelligent Activity Filter support"""

    def __init__(
        self,
        cfg: TestConfig,
        initial_balance: float = 10000.0,
        pip_value: float = 10.0,
        spread_pips: float = 1.5
    ):
        self.cfg = cfg
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.pip_value = pip_value
        self.spread_pips = spread_pips

        # Components
        self.kalman = MultiScaleKalman()
        self.regime_detector = HMMRegimeDetector()
        self.poi_detector = POIDetector()
        self.entry_trigger = EntryTrigger()
        self.entry_trigger.min_quality_score = cfg.min_quality
        self.risk_manager = RiskManager(
            pip_value=pip_value,
            max_sl_pips=cfg.max_sl_pips,
            max_loss_per_trade_pct=cfg.max_loss_pct
        )
        self.exit_manager = ExitManager()
        self.killzone = KillZone()
        self.market_filter = MarketFilter()

        # Intelligent filter
        if cfg.use_intelligent:
            self.intelligent_filter = IntelligentActivityFilter(
                activity_threshold=cfg.activity_threshold,
                min_velocity_pips=2.0,
                min_atr_pips=5.0,
                pip_size=0.0001
            )
        else:
            self.intelligent_filter = None

        # State
        self.trades: List[SimpleTrade] = []
        self.open_trade: Optional[SimpleTrade] = None

    def run(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, start: datetime, end: datetime) -> List[SimpleTrade]:
        """Run backtest"""
        self.trades = []
        self.open_trade = None
        self.balance = self.initial_balance

        # Warmup
        warmup_htf = htf_df[htf_df.index < start].tail(100)
        warmup_ltf = ltf_df[ltf_df.index < start].tail(500)

        for _, row in warmup_ltf.iterrows():
            kalman_state = self.kalman.update(row['close'])
            if self.intelligent_filter:
                self.intelligent_filter.update(row['high'], row['low'], row['close'])
                # kalman_state is a dict with 'velocity' key
                self.intelligent_filter.update_kalman_velocity(kalman_state['velocity'])

        for _, row in warmup_htf.iterrows():
            self.regime_detector.update(row['close'])

        if len(warmup_htf) >= 50:
            self.poi_detector.detect(warmup_htf)

        # Filter data
        htf = htf_df[(htf_df.index >= start) & (htf_df.index <= end)].copy()
        ltf = ltf_df[(ltf_df.index >= start) & (ltf_df.index <= end)].copy()

        if htf.empty or ltf.empty:
            return []

        # Main loop
        htf_idx = 0
        for i, (time, ltf_row) in enumerate(ltf.iterrows()):
            current_price = ltf_row['close']
            current_high = ltf_row['high']
            current_low = ltf_row['low']

            # Update Kalman
            kalman_state = self.kalman.update(current_price)

            # Update intelligent filter
            if self.intelligent_filter:
                self.intelligent_filter.update(current_high, current_low, current_price)
                # kalman_state is a dict with 'velocity' key
                self.intelligent_filter.update_kalman_velocity(kalman_state['velocity'])

            # Update HTF
            while htf_idx < len(htf) and htf.index[htf_idx] <= time:
                htf_row = htf.iloc[htf_idx]
                self.regime_detector.update(htf_row['close'])
                htf_idx += 1
                if htf_idx >= 50:
                    self.poi_detector.detect(htf.iloc[:htf_idx])

            # Should we look for entries?
            should_check = self._should_check_entry(time, current_high, current_low, current_price)

            if not should_check:
                # Still manage open position
                if self.open_trade:
                    self._manage_position(time, current_high, current_low, current_price)
                continue

            # Manage open position
            if self.open_trade:
                self._manage_position(time, current_high, current_low, current_price)
                continue  # One trade at a time

            # Check for entry
            ltf_window = ltf.iloc[max(0, i-50):i+1]
            htf_window = htf.iloc[:htf_idx] if htf_idx > 10 else None
            self._check_entry(time, current_price, ltf_window, htf_window)

        # Close any open trade
        if self.open_trade and len(ltf) > 0:
            last_row = ltf.iloc[-1]
            self._close_trade(ltf.index[-1], last_row['close'])

        return self.trades

    def _should_check_entry(self, dt: datetime, high: float, low: float, close: float) -> bool:
        """Determine if we should check for entries"""
        # Weekend check always applies
        weekday = dt.weekday()
        hour = dt.hour
        if weekday == 5 or (weekday == 6 and hour < 22):
            return False
        if weekday == 4 and hour >= 20:
            return False

        if self.cfg.use_killzone:
            # Traditional Kill Zone
            in_kz, _ = self.killzone.is_in_killzone(dt)
            return in_kz

        elif self.cfg.use_intelligent:
            # Intelligent Activity Filter
            result = self.intelligent_filter.check(dt, high, low, close)
            return result.should_trade

        else:
            # No filter (KZ_OFF)
            return True

    def _check_entry(self, current_time: datetime, current_price: float,
                     ltf_window: pd.DataFrame, htf_window: Optional[pd.DataFrame]):
        """Check for entry signal"""
        # Skip December late
        if current_time.month == 12 and current_time.day >= 15:
            return

        # Get regime
        regime_info = self.regime_detector.last_info
        if regime_info is None or not regime_info.is_tradeable:
            return

        # Get POI
        poi_result = self.poi_detector.last_result
        if poi_result is None:
            return

        # Direction
        if regime_info.regime == MarketRegime.BULLISH:
            direction = "BUY"
            pois = poi_result.bullish_pois
        elif regime_info.regime == MarketRegime.BEARISH:
            direction = "SELL"
            pois = poi_result.bearish_pois
        else:
            return

        if not pois:
            return

        # Check if in POI
        in_poi, poi = poi_result.price_at_poi(current_price, direction)
        if not in_poi:
            return

        # Trend filter
        if htf_window is not None and len(htf_window) >= 10:
            aligned, _ = self.market_filter.check_trend_alignment(htf_window, direction)
            if not aligned:
                return

        # Get adaptive quality threshold from intelligent filter
        if self.intelligent_filter:
            activity_result = self.intelligent_filter.check(
                current_time,
                ltf_window.iloc[-1]['high'],
                ltf_window.iloc[-1]['low'],
                current_price
            )
            min_quality = activity_result.quality_threshold
        else:
            min_quality = self.cfg.min_quality

        # Entry trigger
        poi_info = {
            'high': poi.get('top', poi.get('high', poi.get('mid', 0) + 0.001)),
            'low': poi.get('bottom', poi.get('low', poi.get('mid', 0) - 0.001)),
            'mid': poi.get('mid', current_price),
            'strength': poi.get('strength', 70.0)
        }

        # Temporarily set quality threshold
        original_quality = self.entry_trigger.min_quality_score
        self.entry_trigger.min_quality_score = min_quality

        has_entry, signal = self.entry_trigger.check_for_entry(
            ltf_df=ltf_window,
            direction=direction,
            poi_info=poi_info,
            current_price=current_price,
            require_full_confirmation=True
        )

        self.entry_trigger.min_quality_score = original_quality

        if not has_entry or signal is None:
            return

        # Open trade
        quality_score = poi.get('strength', 70.0)
        sl_pips = signal.sl_pips

        risk_params = self.risk_manager.calculate_lot_size(
            account_balance=self.balance,
            quality_score=quality_score,
            sl_pips=sl_pips
        )

        # Create trade
        entry_price = current_price + (self.spread_pips * 0.0001 if direction == "BUY" else 0)

        self.open_trade = SimpleTrade(
            entry_time=current_time,
            exit_time=None,
            direction=direction,
            entry_price=entry_price,
            exit_price=0,
            sl_price=signal.stop_loss,
            pnl=0,
            quality=quality_score
        )

    def _manage_position(self, current_time: datetime, high: float, low: float, close: float):
        """Manage open position"""
        trade = self.open_trade

        if trade.direction == "BUY":
            if low <= trade.sl_price:
                self._close_trade(current_time, trade.sl_price)
                return
            # Simple TP: 2:1 RR
            sl_dist = trade.entry_price - trade.sl_price
            tp_price = trade.entry_price + sl_dist * 2
            if high >= tp_price:
                self._close_trade(current_time, tp_price)
                return
        else:
            if high >= trade.sl_price:
                self._close_trade(current_time, trade.sl_price)
                return
            sl_dist = trade.sl_price - trade.entry_price
            tp_price = trade.entry_price - sl_dist * 2
            if low <= tp_price:
                self._close_trade(current_time, tp_price)
                return

    def _close_trade(self, time: datetime, price: float):
        """Close trade"""
        trade = self.open_trade
        trade.exit_time = time
        trade.exit_price = price

        if trade.direction == "BUY":
            pips = (price - trade.entry_price) / 0.0001
        else:
            pips = (trade.entry_price - price) / 0.0001

        # Calculate PnL with 0.1 lot
        trade.pnl = pips * self.pip_value * 0.1

        self.balance += trade.pnl
        self.trades.append(trade)
        self.open_trade = None


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime):
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=200000,
                            start_time=start, end_time=end)
    await db.disconnect()

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df


def run_year_test(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, year: int, cfg: TestConfig) -> Dict:
    """Run test for a year"""
    monthly_results = []
    total_trades = 0
    total_wins = 0
    total_pnl = 0

    for month in range(1, 12):  # Skip December
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 11:
            end = datetime(year, 12, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)

        bt = SimpleBacktester(cfg)
        trades = bt.run(htf_df, ltf_df, start, end)

        month_pnl = sum(t.pnl for t in trades)
        month_trades = len(trades)
        month_wins = sum(1 for t in trades if t.pnl > 0)

        monthly_results.append({
            'month': month,
            'pnl': month_pnl,
            'trades': month_trades,
            'wins': month_wins
        })

        total_trades += month_trades
        total_wins += month_wins
        total_pnl += month_pnl

    losing = [m for m in monthly_results if m['pnl'] < 0]

    return {
        'year': year,
        'config': cfg.name,
        'total_pnl': total_pnl,
        'total_return': total_pnl / 100,  # Percentage
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'monthly': monthly_results,
        'losing_months': len(losing),
    }


async def main():
    print("\n" + "="*70)
    print("TEST INTELLIGENT ACTIVITY FILTER")
    print("Compare: Kill Zone vs Intelligent Filter")
    print("="*70)

    # Test configurations
    CONFIGS = [
        TestConfig("KZ_ON", use_killzone=True, use_intelligent=False),
        TestConfig("KZ_OFF", use_killzone=False, use_intelligent=False),
        TestConfig("INTEL_35", use_killzone=False, use_intelligent=True, activity_threshold=35.0),
        TestConfig("INTEL_40", use_killzone=False, use_intelligent=True, activity_threshold=40.0),
        TestConfig("INTEL_50", use_killzone=False, use_intelligent=True, activity_threshold=50.0),
    ]

    print("\nFetching data...")
    start = datetime(2021, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data('GBPUSD', 'H4', start, end)
    ltf_df = await fetch_data('GBPUSD', 'M15', start, end)

    print(f"H4={len(htf_df)}, M15={len(ltf_df)}")

    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing: {cfg.name}")
        if cfg.use_intelligent:
            print(f"  Activity Threshold: {cfg.activity_threshold}")
        print("="*70)

        years_results = []
        for year in [2024, 2025]:
            r = run_year_test(htf_df, ltf_df, year, cfg)
            years_results.append(r)

            print(f"\n{year}:")
            for m in r['monthly']:
                status = "+" if m['pnl'] >= 0 else "-"
                wr = m['wins'] / m['trades'] * 100 if m['trades'] > 0 else 0
                print(f"  M{m['month']:02d}: {status}${abs(m['pnl']):.0f} ({m['trades']}t, {wr:.0f}%WR)")

            print(f"  >> {r['losing_months']}L, {r['total_trades']}T, +{r['total_return']:.1f}%")

        total_losing = sum(r['losing_months'] for r in years_results)
        total_trades = sum(r['total_trades'] for r in years_results)
        total_return = sum(r['total_return'] for r in years_results)
        total_wins = sum(r['total_wins'] for r in years_results)

        all_results.append({
            'config': cfg,
            'years': years_results,
            'total_losing': total_losing,
            'total_trades': total_trades,
            'total_return': total_return,
            'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
            'trades_per_month': total_trades / 22,
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - 2024 + 2025 Combined")
    print("="*70)
    print(f"\n{'Config':<12} | {'Trades':>6} | {'/Mon':>5} | {'WR':>5} | {'Lose':>4} | {'Return':>8}")
    print("-" * 60)

    for r in all_results:
        cfg = r['config']
        print(f"{cfg.name:<12} | {r['total_trades']:>6} | {r['trades_per_month']:>5.1f} | "
              f"{r['win_rate']:>4.0f}% | {r['total_losing']:>4} | {r['total_return']:>+7.1f}%")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    kz_on = next((r for r in all_results if r['config'].name == "KZ_ON"), None)
    kz_off = next((r for r in all_results if r['config'].name == "KZ_OFF"), None)

    if kz_on and kz_off:
        print(f"\nKZ_OFF vs KZ_ON:")
        trade_diff = kz_off['total_trades'] - kz_on['total_trades']
        trade_pct = trade_diff / kz_on['total_trades'] * 100 if kz_on['total_trades'] > 0 else 0
        lose_diff = kz_off['total_losing'] - kz_on['total_losing']
        print(f"  Trades: {trade_diff:+d} ({trade_pct:+.0f}%)")
        print(f"  Losing months: {lose_diff:+d}")

    # Best intelligent config
    intel_results = [r for r in all_results if r['config'].use_intelligent]
    if intel_results:
        # Best = most trades with <= 3 losing months
        good_intel = [r for r in intel_results if r['total_losing'] <= 3]
        if good_intel:
            best_intel = max(good_intel, key=lambda x: x['total_trades'])
        else:
            best_intel = min(intel_results, key=lambda x: x['total_losing'])

        print(f"\nBest Intelligent: {best_intel['config'].name}")
        print(f"  Threshold: {best_intel['config'].activity_threshold}")
        print(f"  Trades: {best_intel['total_trades']} ({best_intel['trades_per_month']:.1f}/month)")
        print(f"  Losing: {best_intel['total_losing']} months")
        print(f"  Return: +{best_intel['total_return']:.1f}%")

        if kz_on:
            diff = best_intel['total_trades'] - kz_on['total_trades']
            pct = diff / kz_on['total_trades'] * 100 if kz_on['total_trades'] > 0 else 0
            print(f"\n  vs KZ_ON: {diff:+d} trades ({pct:+.0f}%)")

    # Send to Telegram
    if TELEGRAM_AVAILABLE:
        print("\n" + "="*70)
        print("Sending to Telegram...")

        msg = "🧠 <b>INTELLIGENT FILTER TEST</b>\n"
        msg += "<i>2024 + 2025 Combined</i>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        msg += "📊 <b>Results:</b>\n"
        msg += "<pre>\n"
        msg += f"{'Config':<12}|Trds|/Mo |WR% |Los|Return\n"
        msg += "-" * 42 + "\n"

        for r in all_results:
            cfg = r['config']
            emoji = "✓" if r['total_losing'] <= 2 else "△" if r['total_losing'] <= 4 else "✗"
            msg += f"{cfg.name:<12}|{r['total_trades']:>4}|{r['trades_per_month']:>4.1f}|{r['win_rate']:>3.0f}%| {r['total_losing']}{emoji}|{r['total_return']:>+5.0f}%\n"

        msg += "</pre>\n"

        if intel_results:
            good_intel = [r for r in intel_results if r['total_losing'] <= 3]
            if good_intel:
                best = max(good_intel, key=lambda x: x['total_trades'])
                msg += f"\n🎯 <b>Recommended: {best['config'].name}</b>\n"
                msg += f"├ Threshold: {best['config'].activity_threshold}\n"
                msg += f"├ Trades: {best['total_trades']} ({best['trades_per_month']:.1f}/mo)\n"
                msg += f"├ Losing: {best['total_losing']} months\n"
                msg += f"└ Return: +{best['total_return']:.1f}%\n"

        msg += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</i>"

        try:
            bot = Bot(token=config.telegram.bot_token)
            await bot.send_message(chat_id=config.telegram.chat_id, text=msg, parse_mode='HTML')
            print("Sent!")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
