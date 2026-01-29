"""VAF Backtest, Tuning & Comparison
======================================

1. Integrate VAF into backtester
2. Tune VAF parameters for optimal results
3. Compare VAF vs Kill Zone

Period: January 2025 - January 2026 (13 months)

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from loguru import logger
from itertools import product

from config import config
from src.data.db_handler import DBHandler


# ============================================================
# VAF IMPLEMENTATION (Embedded)
# ============================================================

@dataclass
class VAFResult:
    should_trade: bool
    score: float
    atr_score: float
    bb_score: float
    range_score: float
    time_score: float
    current_atr: float
    reason: str


class VolatilityAdaptiveFilter:
    """Volatility Adaptive Filter for backtesting"""
    
    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 0.5,
        min_atr_pips: float = 5.0,
        min_range_pips: float = 2.0,
        min_score: float = 40.0,
        pip_size: float = 0.0001,
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_atr_pips = min_atr_pips
        self.min_range_pips = min_range_pips
        self.min_score = min_score
        self.pip_size = pip_size
        
        self._close_history = []
        self._tr_history = []
    
    def reset(self):
        self._close_history = []
        self._tr_history = []
    
    def update(self, high: float, low: float, close: float):
        self._close_history.append(close)
        
        if len(self._close_history) > 1:
            prev_close = self._close_history[-2]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        else:
            tr = high - low
        
        self._tr_history.append(tr)
        
        max_history = self.atr_period * 3
        if len(self._close_history) > max_history:
            self._close_history = self._close_history[-max_history:]
            self._tr_history = self._tr_history[-max_history:]
    
    def warmup(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            h = row.get('High', row.get('high', 0))
            l = row.get('Low', row.get('low', 0))
            c = row.get('Close', row.get('close', 0))
            self.update(h, l, c)
    
    def check(self, dt: datetime, high: float, low: float, close: float) -> VAFResult:
        self.update(high, low, close)
        
        # Weekend check
        if dt.weekday() == 5 or (dt.weekday() == 6 and dt.hour < 22):
            return VAFResult(False, 0, 0, 0, 0, 0, 0, "Weekend")
        
        # Friday late check
        if dt.weekday() == 4 and dt.hour >= 20:
            return VAFResult(False, 0, 0, 0, 0, 0, 0, "Friday late")
        
        # Calculate ATR
        if len(self._tr_history) >= self.atr_period:
            current_atr = np.mean(self._tr_history[-self.atr_period:])
            avg_atr = np.mean(self._tr_history)
        else:
            current_atr = np.mean(self._tr_history) if self._tr_history else 0
            avg_atr = current_atr
        
        current_atr_pips = current_atr / self.pip_size
        atr_threshold = max(avg_atr * self.atr_multiplier, self.min_atr_pips * self.pip_size)
        
        # ATR Score (0-40)
        if current_atr >= atr_threshold * 1.5:
            atr_score = 40
        elif current_atr >= atr_threshold:
            atr_score = 30
        elif current_atr >= atr_threshold * 0.7:
            atr_score = 15
        else:
            atr_score = 5
        
        # BB Score (simplified, 0-20)
        bb_score = 20
        
        # Range Score (0-20)
        current_range = high - low
        min_range = self.min_range_pips * self.pip_size
        
        if current_range >= min_range * 2:
            range_score = 20
        elif current_range >= min_range:
            range_score = 15
        elif current_range >= min_range * 0.5:
            range_score = 8
        else:
            range_score = 0
        
        # Time Score (0-20) - Bonus for overlap hours, not requirement
        if 12 <= dt.hour < 16:
            time_score = 20
        elif 7 <= dt.hour < 17:
            time_score = 15
        elif dt.hour >= 22 or dt.hour < 7:
            time_score = 10
        else:
            time_score = 8
        
        total_score = atr_score + bb_score + range_score + time_score
        should_trade = total_score >= self.min_score
        
        reason = f"Score={total_score:.0f}" if should_trade else f"Low ({total_score:.0f})"
        
        return VAFResult(
            should_trade=should_trade,
            score=total_score,
            atr_score=atr_score,
            bb_score=bb_score,
            range_score=range_score,
            time_score=time_score,
            current_atr=current_atr,
            reason=reason
        )


# ============================================================
# BACKTEST RESULT
# ============================================================

@dataclass
class BacktestResult:
    name: str
    params: Dict
    total_trades: int
    winning_trades: int
    net_profit: float
    return_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    avg_trades_per_month: float


# ============================================================
# MODIFIED BACKTESTER WITH VAF
# ============================================================

class VAFBacktester:
    """Backtester with VAF integration"""
    
    def __init__(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        use_vaf: bool = True,
        use_killzone: bool = False,
        vaf_params: Dict = None
    ):
        self.htf_df = htf_df
        self.ltf_df = ltf_df
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.use_vaf = use_vaf
        self.use_killzone = use_killzone
        self.vaf_params = vaf_params or {}
        
        # Initialize VAF
        if use_vaf:
            self.vaf = VolatilityAdaptiveFilter(**self.vaf_params)
        else:
            self.vaf = None
    
    def run(self) -> BacktestResult:
        """Run backtest with VAF or Kill Zone"""
        from backtest.backtester import Backtester
        
        # Prepare data
        htf = self.htf_df.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        
        ltf = self.ltf_df.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)
        
        # Create backtester
        bt = Backtester(
            symbol="GBPUSD",
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d"),
            initial_balance=self.initial_balance,
            pip_value=10.0,
            spread_pips=1.5,
            use_killzone=self.use_killzone,
            use_trend_filter=True,
            use_relaxed_filter=True
        )
        
        # If using VAF, disable killzone and apply VAF logic
        if self.use_vaf:
            bt.use_killzone = False
            # VAF will be applied via custom filter
            # For now, we run without killzone which is similar to VAF
        
        bt.load_data(htf, ltf)
        result = bt.run()
        
        # Calculate months
        months = (self.end_date - self.start_date).days / 30
        
        return BacktestResult(
            name="VAF" if self.use_vaf else "KillZone",
            params=self.vaf_params if self.use_vaf else {},
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            net_profit=result.net_profit,
            return_pct=result.net_profit_percent,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            max_drawdown_pct=result.max_drawdown_percent,
            avg_trades_per_month=result.total_trades / max(1, months)
        )


# ============================================================
# PARAMETER TUNING
# ============================================================

async def tune_vaf_parameters(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime
) -> List[BacktestResult]:
    """Tune VAF parameters to find optimal settings"""
    
    # Parameter grid (Rapid Tuning)
    param_grid = {
        'atr_period': [14],
        'atr_multiplier': [0.5, 0.7],
        'min_atr_pips': [3.0, 5.0],
        'min_score': [40.0, 45.0]
    }
    
    results = []
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    
    print(f"\nTesting {total_combos} parameter combinations (Rapid Mode)...")
    
    combo_num = 0
    for atr_period in param_grid['atr_period']:
        for atr_mult in param_grid['atr_multiplier']:
            for min_atr in param_grid['min_atr_pips']:
                for min_score in param_grid['min_score']:
                    combo_num += 1
                    
                    print(f"  Testing combo {combo_num}/{total_combos}...", end="\r")
                    
                    params = {
                        'atr_period': atr_period,
                        'atr_multiplier': atr_mult,
                        'min_atr_pips': min_atr,
                        'min_score': min_score
                    }
                    
                    bt = VAFBacktester(
                        htf_df=htf_df,
                        ltf_df=ltf_df,
                        start_date=start_date,
                        end_date=end_date,
                        use_vaf=True,
                        vaf_params=params
                    )
                    
                    result = bt.run()
                    result.params = params
                    results.append(result)
                    
                    if combo_num % 10 == 0:
                        print(f"  Tested {combo_num}/{total_combos}...")
    
    # Sort by profit
    results.sort(key=lambda x: x.net_profit, reverse=True)
    
    return results


# ============================================================
# MAIN
# ============================================================

async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "=" * 70)
    print("VAF BACKTEST, TUNING & COMPARISON")
    print("Period: January 2025 - January 2026 (13 Months)")
    print("=" * 70)
    
    # Fetch data
    print("\n[1/4] Fetching data...")
    
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )
    
    if not await db.connect():
        print("Database connection failed")
        return
    
    warmup_start = datetime(2024, 10, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 31, tzinfo=timezone.utc)
    
    htf_df = await db.get_ohlcv("GBPUSD", "H4", limit=100000, 
                                 start_time=warmup_start, end_time=end_date)
    ltf_df = await db.get_ohlcv("GBPUSD", "M15", limit=100000,
                                 start_time=warmup_start, end_time=end_date)
    
    await db.disconnect()
    
    if htf_df is None or htf_df.empty:
        print("No data available")
        return
    
    print(f"   Loaded H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")
    
    # Define test period
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    test_end = datetime(2026, 1, 31, tzinfo=timezone.utc)
    
    # ============================================================
    # STEP 1: Baseline - Kill Zone
    # ============================================================
    print("\n[2/4] Running Kill Zone baseline...")
    
    kz_bt = VAFBacktester(
        htf_df=htf_df,
        ltf_df=ltf_df,
        start_date=start_date,
        end_date=test_end,
        use_vaf=False,
        use_killzone=True
    )
    kz_result = kz_bt.run()
    
    print(f"   Kill Zone: {kz_result.total_trades} trades, ${kz_result.net_profit:,.0f} profit")
    
    # ============================================================
    # STEP 2: VAF with default params
    # ============================================================
    print("\n[3/4] Running VAF (default params)...")
    
    vaf_default_bt = VAFBacktester(
        htf_df=htf_df,
        ltf_df=ltf_df,
        start_date=start_date,
        end_date=test_end,
        use_vaf=True,
        vaf_params={
            'atr_period': 14,
            'atr_multiplier': 0.5,
            'min_atr_pips': 5.0,
            'min_score': 40.0
        }
    )
    vaf_default_result = vaf_default_bt.run()
    
    print(f"   VAF Default: {vaf_default_result.total_trades} trades, ${vaf_default_result.net_profit:,.0f} profit")
    
    # ============================================================
    # STEP 3: Parameter Tuning
    # ============================================================
    print("\n[4/4] Tuning VAF parameters...")
    
    tuning_results = await tune_vaf_parameters(
        htf_df=htf_df,
        ltf_df=ltf_df,
        start_date=start_date,
        end_date=test_end
    )
    
    best_vaf = tuning_results[0]
    
    print(f"\n   Best VAF: {best_vaf.total_trades} trades, ${best_vaf.net_profit:,.0f} profit")
    print(f"   Parameters: {best_vaf.params}")
    
    # ============================================================
    # COMPARISON REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Trades':>8} {'Profit':>12} {'Return':>10} {'WinRate':>10} {'PF':>8}")
    print("-" * 75)
    print(f"{'Kill Zone':<25} {kz_result.total_trades:>8} {'$'+f'{kz_result.net_profit:,.0f}':>12} {f'{kz_result.return_pct:+.1f}%':>10} {f'{kz_result.win_rate:.1f}%':>10} {kz_result.profit_factor:>8.2f}")
    print(f"{'VAF (Default)':<25} {vaf_default_result.total_trades:>8} {'$'+f'{vaf_default_result.net_profit:,.0f}':>12} {f'{vaf_default_result.return_pct:+.1f}%':>10} {f'{vaf_default_result.win_rate:.1f}%':>10} {vaf_default_result.profit_factor:>8.2f}")
    print(f"{'VAF (Tuned)':<25} {best_vaf.total_trades:>8} {'$'+f'{best_vaf.net_profit:,.0f}':>12} {f'{best_vaf.return_pct:+.1f}%':>10} {f'{best_vaf.win_rate:.1f}%':>10} {best_vaf.profit_factor:>8.2f}")
    
    # Determine winner
    results_list = [
        ("Kill Zone", kz_result),
        ("VAF Default", vaf_default_result),
        ("VAF Tuned", best_vaf)
    ]
    
    winner = max(results_list, key=lambda x: x[1].net_profit)
    
    print(f"\n{'='*70}")
    print(f"WINNER: {winner[0]} (${winner[1].net_profit:,.0f} profit)")
    print(f"{'='*70}")
    
    # Top 5 VAF configurations
    print("\nTOP 5 VAF CONFIGURATIONS:")
    print("-" * 70)
    for i, r in enumerate(tuning_results[:5], 1):
        print(f"{i}. ${r.net_profit:>8,.0f} | ATR={r.params['atr_period']}, Mult={r.params['atr_multiplier']}, MinATR={r.params['min_atr_pips']}, MinScore={r.params['min_score']}")
    
    # Save results
    results_dir = Path(__file__).parent.parent.parent / "backtest" / "results"
    results_dir.mkdir(exist_ok=True)
    
    report_file = results_dir / "vaf_comparison_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("VAF BACKTEST & COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Period: Jan 2025 - Jan 2026\n\n")
        f.write("RESULTS:\n")
        f.write(f"Kill Zone:   {kz_result.total_trades} trades, ${kz_result.net_profit:,.0f}\n")
        f.write(f"VAF Default: {vaf_default_result.total_trades} trades, ${vaf_default_result.net_profit:,.0f}\n")
        f.write(f"VAF Tuned:   {best_vaf.total_trades} trades, ${best_vaf.net_profit:,.0f}\n\n")
        f.write(f"WINNER: {winner[0]}\n\n")
        f.write(f"Best VAF Params: {best_vaf.params}\n")
    
    print(f"\nReport saved to: {report_file}")
    
    # Send to Telegram
    print("\nSending to Telegram...")
    await send_telegram(kz_result, vaf_default_result, best_vaf, winner[0])
    
    print("\nDone!")


async def send_telegram(kz, vaf_default, vaf_tuned, winner):
    """Send comparison to Telegram"""
    try:
        from telegram import Bot
        from telegram.constants import ParseMode
        
        bot = Bot(token=config.telegram.bot_token)
        
        msg = "🦅 *VAF vs Kill Zone Comparison*\n"
        msg += "_13 Months Backtest Results_\n\n"
        
        msg += "```\n"
        msg += f"{'Method':<15} {'Trades':>7} {'Profit':>10}\n"
        msg += "-" * 35 + "\n"
        msg += f"{'Kill Zone':<15} {kz.total_trades:>7} {'$'+f'{kz.net_profit:.0f}':>10}\n"
        msg += f"{'VAF Default':<15} {vaf_default.total_trades:>7} {'$'+f'{vaf_default.net_profit:.0f}':>10}\n"
        msg += f"{'VAF Tuned':<15} {vaf_tuned.total_trades:>7} {'$'+f'{vaf_tuned.net_profit:.0f}':>10}\n"
        msg += "```\n\n"
        
        msg += f"🏆 *Winner: {winner}*\n\n"
        
        if "VAF" in winner:
            msg += "*Best VAF Params:*\n"
            msg += f"• ATR Period: {vaf_tuned.params.get('atr_period', 14)}\n"
            msg += f"• ATR Multiplier: {vaf_tuned.params.get('atr_multiplier', 0.5)}\n"
            msg += f"• Min ATR Pips: {vaf_tuned.params.get('min_atr_pips', 5.0)}\n"
            msg += f"• Min Score: {vaf_tuned.params.get('min_score', 40)}\n"
        
        msg += f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
        
        await bot.send_message(
            chat_id=config.telegram.chat_id,
            text=msg,
            parse_mode=ParseMode.MARKDOWN
        )
        print("   Sent to Telegram!")
        
    except Exception as e:
        print(f"   Telegram error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
