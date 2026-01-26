"""Backtest Problem Months with Enhanced Filters
=================================================

Test enhanced filters on problem months:
- March 2025: Choppy market
- May 2025: High volatility
- November 2025: No trades
- December 2025: Counter-trend losses

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector, MarketRegime
from src.analysis.poi_detector import POIDetector
from src.analysis.market_filter import MarketFilter, MarketCondition, RelaxedEntryFilter
from src.trading.entry_trigger import EntryTrigger
from src.trading.risk_manager import RiskManager
from src.trading.exit_manager import ExitManager
from src.utils.killzone import KillZone


@dataclass
class PartialTPConfig:
    """Partial TP configuration"""
    tp1_rr: float = 1.0
    tp2_rr: float = 2.0
    tp3_rr: float = 3.0
    tp1_size: float = 0.5
    tp2_size: float = 0.3
    tp3_size: float = 0.2


@dataclass
class EnhancedTrade:
    """Trade with enhanced tracking"""
    entry_time: datetime
    direction: str
    entry_price: float
    stop_loss: float
    sl_pips: float
    volume: float
    quality: float
    market_condition: str
    trend_aligned: bool

    # Results
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    partial_pnl: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    result: str = "open"


@dataclass
class MonthResult:
    """Monthly backtest result"""
    month: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    net_pnl: float
    max_dd: float
    filtered_count: int  # Trades filtered out by new filters
    original_pnl: float  # For comparison


class EnhancedBacktester:
    """Backtester with enhanced market filters"""

    def __init__(
        self,
        symbol: str,
        initial_balance: float = 10000.0,
        pip_value: float = 10.0,
        spread_pips: float = 1.5,
        use_trend_filter: bool = True,
        use_volatility_sl: bool = True,
        use_trend_alignment: bool = True,
        relax_on_low_activity: bool = True
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.pip_value = pip_value
        self.spread_pips = spread_pips

        # Enhanced filter flags
        self.use_trend_filter = use_trend_filter
        self.use_volatility_sl = use_volatility_sl
        self.use_trend_alignment = use_trend_alignment
        self.relax_on_low_activity = relax_on_low_activity

        # Components
        self.kalman = MultiScaleKalman()
        self.regime_detector = HMMRegimeDetector()
        self.poi_detector = POIDetector()
        self.entry_trigger = EntryTrigger(min_quality_score=50.0)
        self.killzone = KillZone()
        self.risk_manager = RiskManager(pip_value=pip_value)
        self.exit_manager = ExitManager()

        # NEW: Enhanced filters
        self.market_filter = MarketFilter()
        self.relaxed_filter = RelaxedEntryFilter()

        # Tracking
        self.trades: List[EnhancedTrade] = []
        self.filtered_trades = 0
        self.equity_curve = []
        self.max_equity = initial_balance
        self.max_drawdown = 0.0

    def run(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> MonthResult:
        """Run backtest with enhanced filters"""

        # Reset state
        self.balance = self.initial_balance
        self.trades = []
        self.filtered_trades = 0
        self.equity_curve = [self.initial_balance]
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0

        # Prepare data
        htf = htf_df.copy()
        ltf = ltf_df.copy()

        close_col = 'close' if 'close' in htf.columns else 'Close'
        high_col = 'high' if 'high' in htf.columns else 'High'
        low_col = 'low' if 'low' in htf.columns else 'Low'

        # Warmup Kalman
        warmup_size = min(50, len(htf) // 3)
        for i in range(warmup_size):
            self.kalman.update(htf[close_col].iloc[i])

        # Get regime for period
        try:
            smoothed = []
            for i in range(len(htf)):
                result = self.kalman.update(htf[close_col].iloc[i])
                smoothed.append(result['prices']['fused'])
            regime_info = self.regime_detector.detect_regime(np.array(smoothed))
            base_direction = 'BUY' if regime_info.regime == MarketRegime.BULLISH else 'SELL'
        except:
            base_direction = 'BUY'  # Default

        # Track active trade
        active_trade: Optional[EnhancedTrade] = None
        recent_trades_7d = 0

        # Main loop - iterate through HTF bars
        for i in range(warmup_size, len(htf)):
            bar_time = htf.index[i]

            # Skip if before start date
            if bar_time < start_date:
                continue
            if bar_time > end_date:
                break

            current_price = htf[close_col].iloc[i]
            bar_high = htf[high_col].iloc[i]
            bar_low = htf[low_col].iloc[i]

            # Update equity for active trade
            if active_trade:
                # Check SL hit
                if active_trade.direction == 'BUY':
                    if bar_low <= active_trade.stop_loss:
                        # SL hit
                        sl_pnl = -active_trade.sl_pips * self.pip_value * active_trade.volume
                        active_trade.pnl = active_trade.partial_pnl + sl_pnl
                        active_trade.exit_price = active_trade.stop_loss
                        active_trade.exit_time = bar_time
                        active_trade.result = "loss"
                        self.balance += active_trade.pnl
                        self.trades.append(active_trade)
                        active_trade = None
                        continue
                else:  # SELL
                    if bar_high >= active_trade.stop_loss:
                        sl_pnl = -active_trade.sl_pips * self.pip_value * active_trade.volume
                        active_trade.pnl = active_trade.partial_pnl + sl_pnl
                        active_trade.exit_price = active_trade.stop_loss
                        active_trade.exit_time = bar_time
                        active_trade.result = "loss"
                        self.balance += active_trade.pnl
                        self.trades.append(active_trade)
                        active_trade = None
                        continue

                # Check TP levels
                tp_config = PartialTPConfig()
                tp1 = active_trade.entry_price + (active_trade.sl_pips * 0.0001 * tp_config.tp1_rr) * (1 if active_trade.direction == 'BUY' else -1)
                tp2 = active_trade.entry_price + (active_trade.sl_pips * 0.0001 * tp_config.tp2_rr) * (1 if active_trade.direction == 'BUY' else -1)
                tp3 = active_trade.entry_price + (active_trade.sl_pips * 0.0001 * tp_config.tp3_rr) * (1 if active_trade.direction == 'BUY' else -1)

                remaining_vol = active_trade.volume

                if active_trade.direction == 'BUY':
                    if not active_trade.tp1_hit and bar_high >= tp1:
                        active_trade.tp1_hit = True
                        close_vol = remaining_vol * tp_config.tp1_size
                        pips = (tp1 - active_trade.entry_price) / 0.0001
                        active_trade.partial_pnl += pips * self.pip_value * close_vol
                        remaining_vol -= close_vol
                        # Move SL to breakeven
                        active_trade.stop_loss = active_trade.entry_price + 0.0001

                    if not active_trade.tp2_hit and bar_high >= tp2:
                        active_trade.tp2_hit = True
                        close_vol = active_trade.volume * tp_config.tp2_size
                        pips = (tp2 - active_trade.entry_price) / 0.0001
                        active_trade.partial_pnl += pips * self.pip_value * close_vol
                        remaining_vol -= close_vol

                    if not active_trade.tp3_hit and bar_high >= tp3:
                        active_trade.tp3_hit = True
                        pips = (tp3 - active_trade.entry_price) / 0.0001
                        active_trade.partial_pnl += pips * self.pip_value * remaining_vol
                        active_trade.pnl = active_trade.partial_pnl
                        active_trade.exit_price = tp3
                        active_trade.exit_time = bar_time
                        active_trade.result = "win"
                        self.balance += active_trade.pnl
                        self.trades.append(active_trade)
                        active_trade = None
                        continue

                else:  # SELL
                    if not active_trade.tp1_hit and bar_low <= tp1:
                        active_trade.tp1_hit = True
                        close_vol = remaining_vol * tp_config.tp1_size
                        pips = (active_trade.entry_price - tp1) / 0.0001
                        active_trade.partial_pnl += pips * self.pip_value * close_vol
                        remaining_vol -= close_vol
                        active_trade.stop_loss = active_trade.entry_price - 0.0001

                    if not active_trade.tp2_hit and bar_low <= tp2:
                        active_trade.tp2_hit = True
                        close_vol = active_trade.volume * tp_config.tp2_size
                        pips = (active_trade.entry_price - tp2) / 0.0001
                        active_trade.partial_pnl += pips * self.pip_value * close_vol
                        remaining_vol -= close_vol

                    if not active_trade.tp3_hit and bar_low <= tp3:
                        active_trade.tp3_hit = True
                        pips = (active_trade.entry_price - tp3) / 0.0001
                        active_trade.partial_pnl += pips * self.pip_value * remaining_vol
                        active_trade.pnl = active_trade.partial_pnl
                        active_trade.exit_price = tp3
                        active_trade.exit_time = bar_time
                        active_trade.result = "win"
                        self.balance += active_trade.pnl
                        self.trades.append(active_trade)
                        active_trade = None
                        continue

            # Track equity
            self.equity_curve.append(self.balance)
            if self.balance > self.max_equity:
                self.max_equity = self.balance
            dd = (self.max_equity - self.balance) / self.max_equity * 100
            if dd > self.max_drawdown:
                self.max_drawdown = dd

            # Skip if we have active trade
            if active_trade:
                continue

            # Check kill zone
            if hasattr(bar_time, 'hour'):
                in_kz, kz_name = self.killzone.is_in_killzone(bar_time)
                if not in_kz:
                    continue

            # === ENHANCED FILTER 1: Market Condition ===
            if self.use_trend_filter:
                htf_recent = htf.iloc[max(0, i-30):i+1]
                market_analysis = self.market_filter.analyze_market(htf_recent)

                if not market_analysis.can_trade:
                    self.filtered_trades += 1
                    continue

            # === ENHANCED FILTER 2: Trend Alignment ===
            direction = base_direction
            if self.use_trend_alignment:
                htf_recent = htf.iloc[max(0, i-20):i+1]
                aligned, reason = self.market_filter.check_trend_alignment(htf_recent, direction)

                if not aligned:
                    self.filtered_trades += 1
                    continue

            # Get LTF data for entry
            ltf_mask = ltf.index <= bar_time
            ltf_recent = ltf[ltf_mask].tail(100)

            if len(ltf_recent) < 30:
                continue

            # === ENHANCED FILTER 3: Relaxed Entry on Low Activity ===
            if self.relax_on_low_activity:
                min_quality, require_full = self.relaxed_filter.get_entry_params(recent_trades_7d)
                self.entry_trigger.min_quality_score = min_quality
            else:
                require_full = False

            # Check for entry signal
            poi_info = {'high': bar_high, 'low': bar_low}
            has_entry, signal = self.entry_trigger.check_for_entry(
                ltf_recent, direction, poi_info, current_price,
                require_full_confirmation=require_full
            )

            if not has_entry or signal is None:
                continue

            # === ENHANCED FILTER 4: Volatility-Adjusted SL ===
            sl_pips = signal.sl_pips
            if self.use_volatility_sl:
                htf_recent = htf.iloc[max(0, i-20):i+1]
                sl_pips = self.market_filter.get_dynamic_sl(htf_recent, signal.sl_pips)

            # Calculate position size
            risk_params = self.risk_manager.calculate_lot_size(
                self.balance, signal.quality_score, sl_pips
            )

            # Create trade
            market_cond = market_analysis.condition.value if self.use_trend_filter else "unknown"

            active_trade = EnhancedTrade(
                entry_time=bar_time,
                direction=direction,
                entry_price=current_price,
                stop_loss=signal.stop_loss,
                sl_pips=sl_pips,
                volume=risk_params.lot_size,
                quality=signal.quality_score,
                market_condition=market_cond,
                trend_aligned=True
            )

            recent_trades_7d += 1

        # Close any remaining trade at last price
        if active_trade:
            final_price = htf[close_col].iloc[-1]
            if active_trade.direction == 'BUY':
                pips = (final_price - active_trade.entry_price) / 0.0001
            else:
                pips = (active_trade.entry_price - final_price) / 0.0001

            remaining_vol = active_trade.volume
            if active_trade.tp1_hit:
                remaining_vol -= active_trade.volume * 0.5
            if active_trade.tp2_hit:
                remaining_vol -= active_trade.volume * 0.3

            active_trade.pnl = active_trade.partial_pnl + (pips * self.pip_value * remaining_vol)
            active_trade.exit_price = final_price
            active_trade.exit_time = htf.index[-1]
            active_trade.result = "win" if active_trade.pnl > 0 else "loss"
            self.balance += active_trade.pnl
            self.trades.append(active_trade)

        # Calculate results
        wins = len([t for t in self.trades if t.result == "win"])
        losses = len([t for t in self.trades if t.result == "loss"])
        total = len(self.trades)

        win_rate = (wins / total * 100) if total > 0 else 0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        net_pnl = self.balance - self.initial_balance

        return MonthResult(
            month="",
            trades=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            profit_factor=pf,
            net_pnl=net_pnl,
            max_dd=self.max_drawdown,
            filtered_count=self.filtered_trades,
            original_pnl=0.0
        )


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )

    if not await db.connect():
        return pd.DataFrame()

    df = await db.get_ohlcv(symbol, timeframe, 50000, start, end)
    await db.disconnect()
    return df


async def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("\n" + "="*70)
    print("BACKTEST PROBLEM MONTHS WITH ENHANCED FILTERS")
    print("Testing: March, May, November, December 2025")
    print("="*70)

    symbol = "GBPUSD"

    # Problem months with original results
    problem_months = [
        (datetime(2025, 3, 1, tzinfo=timezone.utc),
         datetime(2025, 3, 31, tzinfo=timezone.utc),
         "March 2025", -447),

        (datetime(2025, 5, 1, tzinfo=timezone.utc),
         datetime(2025, 5, 31, tzinfo=timezone.utc),
         "May 2025", -258),

        (datetime(2025, 11, 1, tzinfo=timezone.utc),
         datetime(2025, 11, 30, tzinfo=timezone.utc),
         "November 2025", 0),

        (datetime(2025, 12, 1, tzinfo=timezone.utc),
         datetime(2025, 12, 31, tzinfo=timezone.utc),
         "December 2025", -1544),
    ]

    # Test configurations
    configs = [
        ("ORIGINAL (No Filters)", False, False, False, False),
        ("+ Trend Filter", True, False, False, False),
        ("+ Volatility SL", True, True, False, False),
        ("+ Trend Alignment", True, True, True, False),
        ("+ Relaxed Entry", True, True, True, True),
    ]

    results_table = []

    for start, end, month_name, original_pnl in problem_months:
        print(f"\n{'='*60}")
        print(f"TESTING: {month_name}")
        print(f"Original P/L: ${original_pnl:+.0f}")
        print(f"{'='*60}")

        # Fetch data
        warmup_start = start - timedelta(days=30)
        htf_df = await fetch_data(symbol, "H4", warmup_start, end)
        ltf_df = await fetch_data(symbol, "M15", warmup_start, end)

        if htf_df.empty or ltf_df.empty:
            print(f"  [SKIP] No data")
            continue

        month_results = []

        for config_name, tf, vs, ta, re in configs:
            bt = EnhancedBacktester(
                symbol=symbol,
                initial_balance=10000.0,
                use_trend_filter=tf,
                use_volatility_sl=vs,
                use_trend_alignment=ta,
                relax_on_low_activity=re
            )

            result = bt.run(htf_df, ltf_df, start, end)
            result.month = month_name
            result.original_pnl = original_pnl

            improvement = result.net_pnl - original_pnl

            print(f"\n  {config_name}:")
            print(f"    Trades: {result.trades} (filtered: {result.filtered_count})")
            print(f"    Win Rate: {result.win_rate:.1f}%")
            print(f"    P/L: ${result.net_pnl:+.0f} (vs ${original_pnl:+.0f})")
            print(f"    Improvement: ${improvement:+.0f}")

            month_results.append({
                'month': month_name,
                'config': config_name,
                'trades': result.trades,
                'filtered': result.filtered_count,
                'win_rate': result.win_rate,
                'pf': result.profit_factor,
                'pnl': result.net_pnl,
                'original': original_pnl,
                'improvement': improvement
            })

        results_table.extend(month_results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BEST CONFIGURATION PER MONTH")
    print("="*70)

    df = pd.DataFrame(results_table)

    for month in df['month'].unique():
        month_df = df[df['month'] == month]
        best = month_df.loc[month_df['improvement'].idxmax()]

        print(f"\n{month}:")
        print(f"  Original: ${best['original']:+.0f}")
        print(f"  Best Config: {best['config']}")
        print(f"  New P/L: ${best['pnl']:+.0f}")
        print(f"  Improvement: ${best['improvement']:+.0f}")

    # Total improvement
    print("\n" + "="*70)
    print("TOTAL IMPROVEMENT (Best Config)")
    print("="*70)

    total_original = sum(m[3] for m in problem_months)

    # Get best config for each month
    best_totals = {}
    for config_name, _, _, _, _ in configs:
        config_df = df[df['config'] == config_name]
        total_new = config_df['pnl'].sum()
        improvement = total_new - total_original
        best_totals[config_name] = (total_new, improvement)

    print(f"\nOriginal Total: ${total_original:+.0f}")
    print("\nBy Configuration:")
    for config_name, (total_new, improvement) in best_totals.items():
        print(f"  {config_name}: ${total_new:+.0f} ({improvement:+.0f})")

    best_config = max(best_totals.items(), key=lambda x: x[1][1])
    print(f"\nBest Overall: {best_config[0]}")
    print(f"  New Total: ${best_config[1][0]:+.0f}")
    print(f"  Improvement: ${best_config[1][1]:+.0f}")

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
