"""
Detailed VAF vs KillZone Backtest Report
=========================================
Comprehensive comparison with monthly breakdown, win rates, and key metrics.
Uses real data from TimescaleDB.
"""
import asyncio
import sys
import os
import io
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import asyncpg
from src.utils.volatility_adaptive_filter import VolatilityAdaptiveFilter
from src.utils.killzone import KillZone
from src.utils.telegram import TelegramNotifier
from config import config


class KillZoneWrapper:
    def __init__(self):
        self.kz = KillZone()
        
    def check(self, dt):
        in_kz, _ = self.kz.is_in_killzone(dt)
        return in_kz


class BacktestStats:
    """Track detailed backtest statistics"""
    def __init__(self, name: str):
        self.name = name
        self.trades = []
        self.monthly_stats = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0,
            'gross_profit': 0.0, 'gross_loss': 0.0
        })
    
    def add_trade(self, dt: datetime, pnl: float):
        self.trades.append({'time': dt, 'pnl': pnl})
        month_key = dt.strftime('%Y-%m')
        
        self.monthly_stats[month_key]['trades'] += 1
        if pnl >= 0:
            self.monthly_stats[month_key]['wins'] += 1
            self.monthly_stats[month_key]['gross_profit'] += pnl
        else:
            self.monthly_stats[month_key]['losses'] += 1
            self.monthly_stats[month_key]['gross_loss'] += abs(pnl)
    
    @property
    def total_trades(self):
        return len(self.trades)
    
    @property
    def total_pnl(self):
        return sum(t['pnl'] for t in self.trades)
    
    @property
    def wins(self):
        return sum(1 for t in self.trades if t['pnl'] >= 0)
    
    @property
    def losses(self):
        return sum(1 for t in self.trades if t['pnl'] < 0)
    
    @property
    def win_rate(self):
        if self.total_trades == 0:
            return 0
        return (self.wins / self.total_trades) * 100
    
    @property
    def gross_profit(self):
        return sum(t['pnl'] for t in self.trades if t['pnl'] >= 0)
    
    @property
    def gross_loss(self):
        return abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
    
    @property
    def profit_factor(self):
        if self.gross_loss == 0:
            return 999.0
        return self.gross_profit / self.gross_loss
    
    @property
    def max_drawdown(self):
        if not self.trades:
            return 0
        cumsum = np.cumsum([t['pnl'] for t in self.trades])
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    @property
    def avg_trade(self):
        if self.total_trades == 0:
            return 0
        return self.total_pnl / self.total_trades
    
    def get_monthly_summary(self):
        """Get sorted monthly summary"""
        return dict(sorted(self.monthly_stats.items()))


async def fetch_ohlcv_from_db(timeframe: str = 'M15', symbol: str = 'GBPUSD'):
    """Fetch OHLCV data from TimescaleDB"""
    conn_str = config.database.connection_string
    print("[DB] Connecting to database...")
    
    try:
        conn = await asyncpg.connect(conn_str)
        print("[DB] Connected!")
        
        # Fetch data with correct column names
        query = """
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY time ASC
        """
        
        rows = await conn.fetch(query, symbol, timeframe)
        await conn.close()
        
        if not rows:
            print(f"[DB] No data found for {symbol} {timeframe}")
            return None
        
        df = pd.DataFrame([dict(r) for r in rows])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        print(f"[DB] Loaded {len(df):,} candles from {df.index[0].date()} to {df.index[-1].date()}")
        return df
        
    except Exception as e:
        print(f"[DB] Error: {e}")
        return None


async def run_detailed_backtest():
    print("=" * 70)
    print("       DETAILED VAF vs KILL ZONE BACKTEST")
    print("=" * 70)
    
    # Fetch data
    df = await fetch_ohlcv_from_db(timeframe='M15', symbol='GBPUSD')
    
    if df is None or df.empty:
        print("[ERROR] No data available. Please sync data to database first.")
        return
    
    # Initialize
    kz = KillZoneWrapper()
    vaf = VolatilityAdaptiveFilter(atr_period=14, atr_multiplier=0.5, min_score=40.0)
    
    kz_stats = BacktestStats("Kill Zone")
    vaf_stats = BacktestStats("VAF")
    
    # Warmup
    print("[INFO] Warming up filters...")
    vaf.warmup(df.head(200))
    
    # Run backtest
    print("[INFO] Running backtest on", len(df), "candles...")
    
    for i in range(200, len(df) - 1):
        idx = df.index[i]
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        
        # Simple signal based on candle direction
        signal = 'BUY' if row['close'] > row['open'] else 'SELL'
        
        # Calculate PnL (using 0.1 lot for more realistic numbers)
        move = next_row['close'] - row['close']
        pnl = (move if signal == 'BUY' else -move) * 10000  # 0.1 lot = $1/pip
        
        # Kill Zone check
        if kz.check(idx):
            kz_stats.add_trade(idx, pnl)
        
        # VAF check
        vaf_res = vaf.check(idx, row['high'], row['low'], row['close'])
        if vaf_res.should_trade:
            vaf_stats.add_trade(idx, pnl)
    
    # Print Results
    print("\n" + "=" * 70)
    print("                      OVERALL SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Kill Zone':>20} {'VAF':>20}")
    print("-" * 70)
    print(f"{'Total Trades':<25} {kz_stats.total_trades:>20,} {vaf_stats.total_trades:>20,}")
    print(f"{'Winners':<25} {kz_stats.wins:>20,} {vaf_stats.wins:>20,}")
    print(f"{'Losers':<25} {kz_stats.losses:>20,} {vaf_stats.losses:>20,}")
    print(f"{'Win Rate':<25} {kz_stats.win_rate:>19.1f}% {vaf_stats.win_rate:>19.1f}%")
    print(f"{'Gross Profit':<25} ${kz_stats.gross_profit:>18,.2f} ${vaf_stats.gross_profit:>18,.2f}")
    print(f"{'Gross Loss':<25} ${kz_stats.gross_loss:>18,.2f} ${vaf_stats.gross_loss:>18,.2f}")
    print(f"{'Net P/L':<25} ${kz_stats.total_pnl:>18,.2f} ${vaf_stats.total_pnl:>18,.2f}")
    print(f"{'Profit Factor':<25} {kz_stats.profit_factor:>20.2f} {vaf_stats.profit_factor:>20.2f}")
    print(f"{'Max Drawdown':<25} ${kz_stats.max_drawdown:>18,.2f} ${vaf_stats.max_drawdown:>18,.2f}")
    print(f"{'Avg Trade':<25} ${kz_stats.avg_trade:>18,.2f} ${vaf_stats.avg_trade:>18,.2f}")
    
    # Monthly Breakdown
    print("\n" + "=" * 70)
    print("                      MONTHLY BREAKDOWN")
    print("=" * 70)
    
    kz_monthly = kz_stats.get_monthly_summary()
    vaf_monthly = vaf_stats.get_monthly_summary()
    all_months = sorted(set(kz_monthly.keys()) | set(vaf_monthly.keys()))
    
    # Only show last 12 months for brevity
    recent_months = all_months[-12:] if len(all_months) > 12 else all_months
    
    print(f"\n{'Month':<12} {'KZ Trades':>10} {'KZ P/L':>15} {'VAF Trades':>12} {'VAF P/L':>15} {'Winner':>10}")
    print("-" * 80)
    
    kz_wins = 0
    vaf_wins = 0
    
    for month in recent_months:
        kz_m = kz_monthly.get(month, {'trades': 0, 'gross_profit': 0, 'gross_loss': 0})
        vaf_m = vaf_monthly.get(month, {'trades': 0, 'gross_profit': 0, 'gross_loss': 0})
        
        kz_net = kz_m['gross_profit'] - kz_m['gross_loss']
        vaf_net = vaf_m['gross_profit'] - vaf_m['gross_loss']
        
        if vaf_net > kz_net:
            winner = "VAF"
            vaf_wins += 1
        elif kz_net > vaf_net:
            winner = "KZ"
            kz_wins += 1
        else:
            winner = "TIE"
        
        print(f"{month:<12} {kz_m['trades']:>10,} ${kz_net:>14,.2f} {vaf_m['trades']:>12,} ${vaf_net:>14,.2f} {winner:>10}")
    
    print("-" * 80)
    print(f"{'Monthly Wins':<12} {'':>10} {'':>15} {'':>12} {'':>15} KZ:{kz_wins} VAF:{vaf_wins}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("                        CONCLUSION")
    print("=" * 70)
    
    if vaf_stats.total_pnl > kz_stats.total_pnl:
        diff = vaf_stats.total_pnl - kz_stats.total_pnl
        print(f"\n[WINNER] VAF WINS by ${diff:,.2f}")
        print(f"  - VAF took {vaf_stats.total_trades:,} trades vs KZ {kz_stats.total_trades:,}")
        print(f"  - VAF profit factor: {vaf_stats.profit_factor:.2f} vs KZ: {kz_stats.profit_factor:.2f}")
        print(f"  - VAF won {vaf_wins}/{len(recent_months)} months")
    else:
        diff = kz_stats.total_pnl - vaf_stats.total_pnl
        print(f"\n[WINNER] KILL ZONE WINS by ${diff:,.2f}")
        print(f"  - KZ took {kz_stats.total_trades:,} trades vs VAF {vaf_stats.total_trades:,}")
        print(f"  - KZ profit factor: {kz_stats.profit_factor:.2f} vs VAF: {vaf_stats.profit_factor:.2f}")
        print(f"  - KZ won {kz_wins}/{len(recent_months)} months")
    
    # Send to Telegram
    if config.telegram.enabled and config.telegram.bot_token:
        print("\n[TELEGRAM] Sending detailed report...")
        bot = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )
        
        if await bot.initialize():
            # Message 1: Summary
            msg1 = "📊 <b>DETAILED BACKTEST REPORT</b>\n"
            msg1 += f"<i>Period: {df.index[0].date()} to {df.index[-1].date()}</i>\n"
            msg1 += f"<i>Candles: {len(df):,} (M15)</i>\n\n"
            
            msg1 += "<b>🛑 Kill Zone (Fixed Time)</b>\n"
            msg1 += f"├ Trades: {kz_stats.total_trades:,}\n"
            msg1 += f"├ Win Rate: {kz_stats.win_rate:.1f}%\n"
            msg1 += f"├ Profit Factor: {kz_stats.profit_factor:.2f}\n"
            msg1 += f"├ Max DD: ${kz_stats.max_drawdown:,.0f}\n"
            msg1 += f"└ Net P/L: <b>${kz_stats.total_pnl:,.2f}</b>\n\n"
            
            msg1 += "<b>⚡ VAF (Volatility Adaptive)</b>\n"
            msg1 += f"├ Trades: {vaf_stats.total_trades:,}\n"
            msg1 += f"├ Win Rate: {vaf_stats.win_rate:.1f}%\n"
            msg1 += f"├ Profit Factor: {vaf_stats.profit_factor:.2f}\n"
            msg1 += f"├ Max DD: ${vaf_stats.max_drawdown:,.0f}\n"
            msg1 += f"└ Net P/L: <b>${vaf_stats.total_pnl:,.2f}</b>\n\n"
            
            if vaf_stats.total_pnl > kz_stats.total_pnl:
                diff = vaf_stats.total_pnl - kz_stats.total_pnl
                msg1 += f"🏆 <b>VAF WINS by ${diff:,.2f}</b> 🚀\n"
                msg1 += f"Monthly: VAF {vaf_wins} vs KZ {kz_wins}"
            else:
                diff = kz_stats.total_pnl - vaf_stats.total_pnl
                msg1 += f"🏆 <b>Kill Zone WINS by ${diff:,.2f}</b>\n"
                msg1 += f"Monthly: KZ {kz_wins} vs VAF {vaf_wins}"
            
            await bot.send(msg1)
            
            # Message 2: Monthly breakdown (last 6 months)
            msg2 = "📅 <b>MONTHLY BREAKDOWN</b>\n"
            msg2 += "<i>(Last 6 months)</i>\n\n"
            msg2 += "<pre>"
            msg2 += f"{'Month':<8} {'KZ':>10} {'VAF':>10} {'Win':>5}\n"
            msg2 += "-" * 35 + "\n"
            
            for month in recent_months[-6:]:
                kz_m = kz_monthly.get(month, {'gross_profit': 0, 'gross_loss': 0})
                vaf_m = vaf_monthly.get(month, {'gross_profit': 0, 'gross_loss': 0})
                kz_net = kz_m['gross_profit'] - kz_m['gross_loss']
                vaf_net = vaf_m['gross_profit'] - vaf_m['gross_loss']
                winner = "VAF" if vaf_net > kz_net else ("KZ" if kz_net > vaf_net else "-")
                msg2 += f"{month:<8} ${kz_net:>8,.0f} ${vaf_net:>8,.0f} {winner:>5}\n"
            
            msg2 += "</pre>"
            
            await bot.send(msg2)
            print("[TELEGRAM] Reports sent successfully!")
        else:
            print("[TELEGRAM] Failed to initialize bot.")
    else:
        print("[INFO] Telegram not configured.")
    
    print("\n[DONE] Backtest complete!")


if __name__ == "__main__":
    asyncio.run(run_detailed_backtest())
