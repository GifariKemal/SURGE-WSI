"""
VAF Quick Compare with Database Data
=====================================
Fetches real OHLCV data from TimescaleDB and compares VAF vs KillZone performance.
Sends results to Telegram.
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

import asyncpg
from src.utils.volatility_adaptive_filter import VolatilityAdaptiveFilter
from src.utils.killzone import KillZone
from src.utils.telegram import TelegramNotifier
from config import config

# Wrapper for KillZone
class KillZoneWrapper:
    def __init__(self):
        self.kz = KillZone()
        
    def check(self, dt):
        in_kz, _ = self.kz.is_in_killzone(dt)
        return in_kz


async def fetch_ohlcv_from_db(timeframe: str = 'M15', symbol: str = 'GBPUSD', limit: int = 10000):
    """Fetch OHLCV data from TimescaleDB"""
    
    # Build connection string
    conn_str = config.database.connection_string
    print(f"🔌 Connecting to database: {config.database.host}:{config.database.port}/{config.database.database}")
    
    try:
        conn = await asyncpg.connect(conn_str)
        print("✅ Database connected!")
        
        # First, check what tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        print(f"📋 Available tables: {[t['table_name'] for t in tables]}")
        
        # Try different table name patterns
        table_candidates = [
            f"ohlcv_{timeframe.lower()}",  # ohlcv_m15
            f"ohlcv_{timeframe}",            # ohlcv_M15
            "ohlcv",                         # generic ohlcv
            "candles",                       # alternative name
            f"{symbol.lower()}_{timeframe.lower()}"  # gbpusd_m15
        ]
        
        table_name = None
        for candidate in table_candidates:
            check = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = $1
                )
            """, candidate)
            if check:
                table_name = candidate
                print(f"✅ Found table: {table_name}")
                break
        
        if not table_name:
            # Try to find any table with 'ohlcv' or 'candle' in name
            for t in tables:
                if 'ohlcv' in t['table_name'].lower() or 'candle' in t['table_name'].lower():
                    table_name = t['table_name']
                    print(f"✅ Using table: {table_name}")
                    break
        
        if not table_name:
            print("❌ No OHLCV table found in database!")
            await conn.close()
            return None
        
        # Get column names
        columns = await conn.fetch(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = $1
        """, table_name)
        col_names = [c['column_name'] for c in columns]
        print(f"📊 Columns: {col_names}")
        
        # Determine column names (handle different naming conventions)
        time_col = next((c for c in col_names if c.lower() in ['time', 'timestamp', 'datetime', 'date']), 'time')
        open_col = next((c for c in col_names if c.lower() in ['open', 'open_price', 'o']), 'open')
        high_col = next((c for c in col_names if c.lower() in ['high', 'high_price', 'h']), 'high')
        low_col = next((c for c in col_names if c.lower() in ['low', 'low_price', 'l']), 'low')
        close_col = next((c for c in col_names if c.lower() in ['close', 'close_price', 'c']), 'close')
        
        # Check if symbol column exists
        symbol_filter = ""
        if 'symbol' in col_names:
            symbol_filter = f"WHERE symbol = '{symbol}'"
        
        # Fetch data
        query = f"""
            SELECT {time_col} as time, {open_col} as open, {high_col} as high, 
                   {low_col} as low, {close_col} as close
            FROM {table_name}
            {symbol_filter}
            ORDER BY {time_col} DESC
            LIMIT {limit}
        """
        
        rows = await conn.fetch(query)
        await conn.close()
        
        if not rows:
            print("❌ No data found in table!")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(r) for r in rows])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)  # Sort ascending
        
        print(f"📈 Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None


async def run_backtest_with_db():
    print("🚀 Starting VAF vs KillZone Backtest (Database Data)")
    print("=" * 60)
    
    # 1. Fetch data from database
    df = await fetch_ohlcv_from_db(timeframe='M15', symbol='GBPUSD', limit=10000)
    
    if df is None or df.empty:
        print("\n⚠️ Could not load data from database.")
        print("Please ensure:")
        print("  1. TimescaleDB is running (docker-compose up -d)")
        print("  2. OHLCV data has been synced to the database")
        print("  3. Database credentials in .env are correct")
        return
    
    print(f"\n📊 Analyzing {len(df)} candles from database...")
    
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
    
    for i in range(100, len(df)):
        idx = df.index[i]
        row = df.iloc[i]
        
        # Simple signal logic
        raw_signal = 'BUY' if row['close'] > row['open'] else 'SELL'
        
        # Calculate theoretical PnL
        if i + 1 < len(df):
            next_row = df.iloc[i+1]
            move = next_row['close'] - row['close']
            trade_pnl = move if raw_signal == 'BUY' else -move
            trade_pnl_usd = trade_pnl * 100000  # Standard lot
        else:
            trade_pnl_usd = 0
            
        # Check Kill Zone
        if kz.check(idx):
            kz_trades += 1
            kz_pnl += trade_pnl_usd
            
        # Check VAF
        vaf_res = vaf.check(idx, row['high'], row['low'], row['close'])
        if vaf_res.should_trade:
            vaf_trades += 1
            vaf_pnl += trade_pnl_usd

    # 4. Results
    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS (Database Data)")
    print("=" * 60)
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Candles: {len(df)}")
    print()
    print(f"🛑 Kill Zone (Fixed Time):")
    print(f"   Trades: {kz_trades}")
    print(f"   PnL: ${kz_pnl:,.2f}")
    print()
    print(f"⚡ VAF (Dynamic Volatility):")
    print(f"   Trades: {vaf_trades}")
    print(f"   PnL: ${vaf_pnl:,.2f}")
    print()
    
    if vaf_pnl > kz_pnl:
        diff = vaf_pnl - kz_pnl
        print(f"✅ VAF Outperformed by ${diff:,.2f} 🚀")
    else:
        diff = kz_pnl - vaf_pnl
        print(f"⚠️ Kill Zone Outperformed by ${diff:,.2f}")
    
    # 5. Send to Telegram
    if config.telegram.enabled and config.telegram.bot_token:
        print("\n📤 Sending report to Telegram...")
        bot = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )
        
        if await bot.initialize():
            msg = "📊 <b>VAF vs KillZone Backtest</b>\n"
            msg += "<i>(Real Database Data)</i>\n\n"
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
            print("✅ Report sent to Telegram!")
        else:
            print("❌ Failed to initialize Telegram bot.")
    else:
        print("⚠️ Telegram not configured.")


if __name__ == "__main__":
    asyncio.run(run_backtest_with_db())
