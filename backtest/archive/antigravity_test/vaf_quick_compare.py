"""
VAF Quick Compare & Telegram Report
===================================
Runs a quick comparison between Kill Zone and VAF for the last 3 months
and sends a formatted report to Telegram.
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.volatility_adaptive_filter import VolatilityAdaptiveFilter
from src.utils.killzone import KillZone
from src.utils.telegram import TelegramNotifier
from config import config

# Wrapper for KillZone to match VAF interface style for this test
class KillZoneWrapper:
    def __init__(self):
        self.kz = KillZone()
        
    def check(self, dt):
        in_kz, _ = self.kz.is_in_killzone(dt)
        return in_kz

async def run_quick_compare():
    print("🚀 Starting Quick VAF vs KillZone Comparison...")
    
    # 1. Load Data (Mocking data loading for speed, using random walk if file not found)
    # In real scenario, we load from CSV/DB. Here we try to load the file used in previous steps
    data_path = r"C:\Users\Administrator\Music\SURGE-WSI\data\mt5_data\GBPUSD_M15_20250101_20260201.csv"
    
    if os.path.exists(data_path):
        print(f"📂 Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        print("⚠️ Data file not found. Generating synthetic data for demonstration...")
        # Generate 3 months of M15 data
        dates = pd.date_range(start='2025-11-01', end='2026-02-01', freq='15min')
        df = pd.DataFrame(index=dates)
        
        # Random walk with daily seasonality to mimic volatility
        np.random.seed(42)
        n = len(dates)
        
        # Volatility pattern (higher during London/NY hours)
        hours = dates.hour
        vol_multiplier = np.where((hours >= 8) & (hours <= 17), 1.5, 0.5)
        
        returns = np.random.normal(0, 0.0005 * vol_multiplier, n)
        price = 1.2500 * np.exp(np.cumsum(returns))
        
        df['close'] = price
        df['open'] = df['close'].shift(1).fillna(1.2500)
        
        # Add High/Low
        noise = np.random.rand(n) * 0.0010 * vol_multiplier
        df['high'] = df[['open', 'close']].max(axis=1) + noise
        df['low'] = df[['open', 'close']].min(axis=1) - noise
        
        df.dropna(inplace=True)

    print(f"📊 Analyzing {len(df)} candles...")

    # 2. Initialize Filters
    kz = KillZoneWrapper()
    vaf = VolatilityAdaptiveFilter(
        atr_period=14,
        atr_multiplier=0.5,
        min_score=40.0
    )
    
    # Warmup VAF
    print("🔥 Warming up VAF...")
    vaf.warmup(df)

    # 3. Run Backtest Loop
    kz_trades = 0
    kz_pnl = 0.0
    
    vaf_trades = 0
    vaf_pnl = 0.0
    
    print("🔄 Running simulation...")
    
    # Simple simulation loop
    for i in range(100, len(df)):
        idx = df.index[i]
        row = df.iloc[i]
        
        # Generate raw signal (random/simple logic)
        # We assume a raw signal exists, and we check if filters allow it
        raw_signal = 'BUY' if row['close'] > row['open'] else 'SELL'
        
        # Calculate theoretical PnL (simple: next bar move)
        if i + 1 < len(df):
            next_row = df.iloc[i+1]
            move = next_row['close'] - row['close']
            trade_pnl = move if raw_signal == 'BUY' else -move
            trade_pnl_usd = trade_pnl * 100000 # Standard lot
        else:
            trade_pnl_usd = 0
            
        # --- Check Kill Zone ---
        if kz.check(idx):
            kz_trades += 1
            kz_pnl += trade_pnl_usd
            
        # --- Check VAF ---
        vaf_res = vaf.check(idx, row['high'], row['low'], row['close'])
        if vaf_res.should_trade:
            vaf_trades += 1
            vaf_pnl += trade_pnl_usd

    # 4. Prepare Results
    print("\n📈 RESULTS:")
    print(f"Kill Zone: {kz_trades} trades, PnL: ${kz_pnl:,.2f}")
    print(f"VAF      : {vaf_trades} trades, PnL: ${vaf_pnl:,.2f}")
    
    # 5. Send to Telegram
    if config.telegram.enabled and config.telegram.bot_token:
        print("\n📤 Sending report to Telegram...")
        bot = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )
        
        # Initialize bot
        if await bot.initialize():
            msg = "📊 <b>VAF vs KillZone Comparison Report</b>\n\n"
            msg += f"📅 Period: {df.index[0].date()} to {df.index[-1].date()}\n"
            msg += f"🕯️ Candles: {len(df)}\n\n"
            
            msg += "<b>🛑 Kill Zone (Fixed Time)</b>\n"
            msg += f"├ Trades: {kz_trades}\n"
            msg += f"└ PnL: <b>${kz_pnl:,.2f}</b>\n\n"
            
            msg += "<b>⚡ VAF (Dynamic Volatility)</b>\n"
            msg += f"├ Trades: {vaf_trades}\n"
            msg += f"└ PnL: <b>${vaf_pnl:,.2f}</b>\n\n"
            
            if vaf_pnl > kz_pnl:
                diff = vaf_pnl - kz_pnl
                msg += f"✅ <b>VAF Outperformed by ${diff:,.2f}</b> 🚀"
            else:
                diff = kz_pnl - vaf_pnl
                msg += f"⚠️ Kill Zone Outperformed by ${diff:,.2f}"
                
            await bot.send(msg)
            print("✅ Message sent!")
        else:
            print("❌ Failed to initialize Telegram bot.")
    else:
        print("❌ Telegram not configured or disabled.")

if __name__ == "__main__":
    asyncio.run(run_quick_compare())
