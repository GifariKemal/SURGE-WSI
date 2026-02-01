"""
SURGE-WSI CORRELATED PORTFOLIO BACKTEST
========================================
Both pairs trade during London+NY session (optimal for GBP pairs).
Uses correlation filter to avoid doubling up on same direction.

Strategy:
- GBPUSD: Primary pair (60% allocation)
- GBPJPY: Secondary pair (40% allocation)
- Skip GBPJPY trade if GBPUSD has same direction open position
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.gbpusd_h1_quadlayer.trading_filters import IntraMonthRiskManager, PatternBasedFilter

# ============================================================
# PORTFOLIO CONFIGURATION
# ============================================================
INITIAL_BALANCE = 50_000.0
GBPUSD_ALLOCATION = 0.60  # 60% for GBPUSD
GBPJPY_ALLOCATION = 0.40  # 40% for GBPJPY

# Risk per trade
RISK_PERCENT = 0.5
SL_ATR_MULT = 1.2
TP_RATIO = 3.5
MAX_LOT = 5.0

# Both pairs trade during London+NY
TRADING_HOURS = list(range(8, 18))  # 08:00-17:59 UTC

# Entry signals
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True
USE_TRIPLE_LAYER_FILTER = True

# Backtest period
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 2, 1)

@dataclass
class PairConfig:
    symbol: str
    pip_size: float
    pip_value: float
    allocation: float

PAIRS = {
    'GBPUSD': PairConfig('GBPUSD', 0.0001, 10.0, GBPUSD_ALLOCATION),
    'GBPJPY': PairConfig('GBPJPY', 0.01, 6.5, GBPJPY_ALLOCATION)
}

@dataclass
class Trade:
    pair: str
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""
    quality_score: float = 0.0
    entry_type: str = ""
    poi_type: str = ""

async def fetch_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from MT5"""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        logger.error("MT5 init failed")
        return pd.DataFrame()

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

    logger.info(f"[MT5] {symbol}: {len(df)} bars")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators"""
    df = df.copy()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # EMAs
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Trend
    df['trend'] = 0
    df.loc[df['ema_9'] > df['ema_21'], 'trend'] = 1
    df.loc[df['ema_9'] < df['ema_21'], 'trend'] = -1

    return df

def detect_signals(df: pd.DataFrame, pair_config: PairConfig) -> list:
    """Detect trading signals"""
    signals = []

    for i in range(50, len(df)):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        hour = bar.name.hour

        # Only trade during optimal hours
        if hour not in TRADING_HOURS:
            continue

        atr = bar['atr']
        if pd.isna(atr) or atr <= 0:
            continue

        atr_pips = atr / pair_config.pip_size
        if atr_pips < 10 or atr_pips > 50:
            continue

        bullish_ema = bar['ema_9'] > bar['ema_21'] > bar['ema_50']
        bearish_ema = bar['ema_9'] < bar['ema_21'] < bar['ema_50']

        rsi = bar['rsi']
        rsi_bullish = 40 < rsi < 70
        rsi_bearish = 30 < rsi < 60

        signal = None

        # ORDER BLOCK
        if USE_ORDER_BLOCK:
            if (bullish_ema and rsi_bullish and
                prev_bar['close'] < prev_bar['open'] and
                bar['close'] > bar['open'] and
                bar['close'] > prev_bar['high']):
                signal = {
                    'time': bar.name, 'direction': 'BUY',
                    'entry_price': bar['close'], 'atr': atr,
                    'atr_pips': atr_pips, 'entry_type': 'ENGULF',
                    'poi_type': 'ORDER_BLOCK'
                }
            elif (bearish_ema and rsi_bearish and
                  prev_bar['close'] > prev_bar['open'] and
                  bar['close'] < bar['open'] and
                  bar['close'] < prev_bar['low']):
                signal = {
                    'time': bar.name, 'direction': 'SELL',
                    'entry_price': bar['close'], 'atr': atr,
                    'atr_pips': atr_pips, 'entry_type': 'ENGULF',
                    'poi_type': 'ORDER_BLOCK'
                }

        # EMA PULLBACK
        if USE_EMA_PULLBACK and signal is None:
            if (bullish_ema and rsi_bullish and
                bar['low'] <= bar['ema_21'] * 1.002 and
                bar['close'] > bar['ema_9'] and
                bar['close'] > bar['open']):
                signal = {
                    'time': bar.name, 'direction': 'BUY',
                    'entry_price': bar['close'], 'atr': atr,
                    'atr_pips': atr_pips, 'entry_type': 'MOMENTUM',
                    'poi_type': 'EMA_PULLBACK'
                }
            elif (bearish_ema and rsi_bearish and
                  bar['high'] >= bar['ema_21'] * 0.998 and
                  bar['close'] < bar['ema_9'] and
                  bar['close'] < bar['open']):
                signal = {
                    'time': bar.name, 'direction': 'SELL',
                    'entry_price': bar['close'], 'atr': atr,
                    'atr_pips': atr_pips, 'entry_type': 'MOMENTUM',
                    'poi_type': 'EMA_PULLBACK'
                }

        if signal:
            signals.append(signal)

    return signals

async def run_portfolio_backtest():
    """Run correlated portfolio backtest"""

    print("=" * 70)
    print("SURGE-WSI CORRELATED PORTFOLIO BACKTEST")
    print("=" * 70)
    print(f"GBPUSD: {GBPUSD_ALLOCATION*100:.0f}% allocation")
    print(f"GBPJPY: {GBPJPY_ALLOCATION*100:.0f}% allocation")
    print(f"Trading Hours: {TRADING_HOURS[0]:02d}:00 - {TRADING_HOURS[-1]+1:02d}:00 UTC")
    print("Correlation Filter: Skip same-direction trades")
    print("=" * 70)

    # Fetch data
    print("\nFetching data...")
    df_usd = await fetch_data("GBPUSD", START_DATE, END_DATE)
    df_jpy = await fetch_data("GBPJPY", START_DATE, END_DATE)

    if df_usd.empty or df_jpy.empty:
        print("ERROR: Could not fetch data")
        return

    # Add indicators
    df_usd = compute_indicators(df_usd)
    df_jpy = compute_indicators(df_jpy)

    # Detect signals
    print("\nDetecting signals...")
    signals_usd = detect_signals(df_usd, PAIRS['GBPUSD'])
    signals_jpy = detect_signals(df_jpy, PAIRS['GBPJPY'])
    print(f"GBPUSD: {len(signals_usd)} signals")
    print(f"GBPJPY: {len(signals_jpy)} signals")

    # Initialize filters
    risk_usd = IntraMonthRiskManager()
    pattern_usd = PatternBasedFilter()
    risk_jpy = IntraMonthRiskManager()
    pattern_jpy = PatternBasedFilter()

    # Track open positions for correlation filter
    open_usd_direction = None
    open_jpy_direction = None

    # Combine and sort all signals
    all_signals = []
    for s in signals_usd:
        s['pair'] = 'GBPUSD'
        all_signals.append(s)
    for s in signals_jpy:
        s['pair'] = 'GBPJPY'
        all_signals.append(s)

    all_signals.sort(key=lambda x: x['time'])

    # Track balance and trades
    balance = INITIAL_BALANCE
    trades = []
    last_trade_time = {'GBPUSD': None, 'GBPJPY': None}

    correlation_skipped = 0

    for signal in all_signals:
        pair = signal['pair']
        config = PAIRS[pair]
        direction = signal['direction']

        # Minimum time between trades (4 hours)
        if last_trade_time[pair]:
            if (signal['time'] - last_trade_time[pair]).total_seconds() < 4 * 3600:
                continue

        # Get appropriate filters
        risk_mgr = risk_usd if pair == 'GBPUSD' else risk_jpy
        pattern_flt = pattern_usd if pair == 'GBPUSD' else pattern_jpy
        df = df_usd if pair == 'GBPUSD' else df_jpy

        # Quality filter
        if USE_TRIPLE_LAYER_FILTER:
            risk_ok, risk_adj, _ = risk_mgr.new_trade_check(signal['time'])
            if not risk_ok:
                continue

            pattern_ok, pattern_size, _ = pattern_flt.check_trade_allowed()
            if not pattern_ok:
                continue

            quality = 70 + risk_adj
            if quality >= 100:
                continue
        else:
            quality = 70.0
            pattern_size = 1.0

        # CORRELATION FILTER
        # Skip GBPJPY if GBPUSD has same direction position open
        if pair == 'GBPJPY' and open_usd_direction == direction:
            correlation_skipped += 1
            continue

        # Calculate position size based on allocation
        pair_balance = balance * config.allocation
        atr_pips = signal['atr_pips']
        sl_pips = atr_pips * SL_ATR_MULT
        tp_pips = sl_pips * TP_RATIO

        risk_amount = pair_balance * (RISK_PERCENT / 100)
        lot_size = risk_amount / (sl_pips * config.pip_value)
        lot_size = min(lot_size, MAX_LOT)
        lot_size = round(lot_size * pattern_size, 2)

        if lot_size < 0.01:
            continue

        # Find exit
        entry_idx = df.index.get_loc(signal['time'])
        entry_price = signal['entry_price']

        if direction == 'BUY':
            sl_price = entry_price - (sl_pips * config.pip_size)
            tp_price = entry_price + (tp_pips * config.pip_size)
        else:
            sl_price = entry_price + (sl_pips * config.pip_size)
            tp_price = entry_price - (tp_pips * config.pip_size)

        # Mark position open
        if pair == 'GBPUSD':
            open_usd_direction = direction
        else:
            open_jpy_direction = direction

        # Simulate trade
        exit_price = None
        exit_time = None
        exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + 48, len(df))):
            future_bar = df.iloc[j]

            if direction == 'BUY':
                if future_bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_time = future_bar.name
                    exit_reason = 'SL'
                    break
                elif future_bar['high'] >= tp_price:
                    exit_price = tp_price
                    exit_time = future_bar.name
                    exit_reason = 'TP'
                    break
            else:
                if future_bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_time = future_bar.name
                    exit_reason = 'SL'
                    break
                elif future_bar['low'] <= tp_price:
                    exit_price = tp_price
                    exit_time = future_bar.name
                    exit_reason = 'TP'
                    break

        # Timeout
        if exit_price is None:
            if entry_idx + 48 < len(df):
                exit_bar = df.iloc[entry_idx + 48]
                exit_price = exit_bar['close']
                exit_time = exit_bar.name
                exit_reason = 'TIMEOUT'
            else:
                continue

        # Mark position closed
        if pair == 'GBPUSD':
            open_usd_direction = None
        else:
            open_jpy_direction = None

        # Calculate P&L
        if direction == 'BUY':
            pips_gained = (exit_price - entry_price) / config.pip_size
        else:
            pips_gained = (entry_price - exit_price) / config.pip_size

        pnl = pips_gained * lot_size * config.pip_value
        balance += pnl

        # Record trade
        trade = Trade(
            pair=pair,
            entry_time=signal['time'],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            lot_size=lot_size,
            pnl=pnl,
            exit_reason=exit_reason,
            quality_score=quality,
            entry_type=signal['entry_type'],
            poi_type=signal['poi_type']
        )
        trades.append(trade)

        # Update filters
        risk_mgr.record_trade(pnl, exit_time, direction)
        pattern_flt.record_trade(direction, pnl, exit_time)

        last_trade_time[pair] = signal['time']

    # Calculate metrics
    total_pnl = balance - INITIAL_BALANCE
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Per pair breakdown
    usd_trades = [t for t in trades if t.pair == 'GBPUSD']
    jpy_trades = [t for t in trades if t.pair == 'GBPJPY']
    usd_pnl = sum(t.pnl for t in usd_trades)
    jpy_pnl = sum(t.pnl for t in jpy_trades)
    usd_wins = [t for t in usd_trades if t.pnl > 0]
    jpy_wins = [t for t in jpy_trades if t.pnl > 0]
    usd_wr = len(usd_wins) / len(usd_trades) * 100 if usd_trades else 0
    jpy_wr = len(jpy_wins) / len(jpy_trades) * 100 if jpy_trades else 0

    # Monthly breakdown
    monthly = {}
    for t in trades:
        m = t.entry_time.strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'total': 0, 'usd': 0, 'jpy': 0}
        monthly[m]['total'] += t.pnl
        if t.pair == 'GBPUSD':
            monthly[m]['usd'] += t.pnl
        else:
            monthly[m]['jpy'] += t.pnl

    losing_months = sum(1 for m in monthly.values() if m['total'] < 0)

    # Print results
    print("\n" + "=" * 70)
    print("CORRELATED PORTFOLIO RESULTS")
    print("=" * 70)

    print("\n[PORTFOLIO SUMMARY]")
    print("-" * 50)
    print(f"  Initial Balance:     ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Balance:       ${balance:,.2f}")
    print(f"  Net P/L:             ${total_pnl:+,.2f}")
    print(f"  Total Return:        {(total_pnl/INITIAL_BALANCE)*100:+.1f}%")

    print("\n[TRADE STATISTICS]")
    print("-" * 50)
    print(f"  Total Trades:        {len(trades)}")
    print(f"  Win Rate:            {win_rate:.1f}%")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Correlation Skipped: {correlation_skipped}")

    print("\n[PER-PAIR BREAKDOWN]")
    print("-" * 50)
    print(f"  GBPUSD ({GBPUSD_ALLOCATION*100:.0f}%):")
    print(f"    Trades: {len(usd_trades)}, WR: {usd_wr:.1f}%, P/L: ${usd_pnl:+,.2f}")
    print(f"  GBPJPY ({GBPJPY_ALLOCATION*100:.0f}%):")
    print(f"    Trades: {len(jpy_trades)}, WR: {jpy_wr:.1f}%, P/L: ${jpy_pnl:+,.2f}")

    print("\n[MONTHLY BREAKDOWN]")
    print("-" * 50)
    for m, data in sorted(monthly.items()):
        status = "WIN " if data['total'] >= 0 else "LOSS"
        print(f"  [{status}] {m}: ${data['total']:+,.2f} (USD: ${data['usd']:+,.2f}, JPY: ${data['jpy']:+,.2f})")

    print("\n" + "=" * 70)
    print(f"[{'OK' if total_pnl > 0 else 'X'}] Profit: ${total_pnl:+,.2f}")
    print(f"[{'OK' if profit_factor >= 2 else 'X'}] PF: {profit_factor:.2f}")
    print(f"[{'OK' if losing_months <= 2 else 'X'}] Losing Months: {losing_months}/{len(monthly)}")
    print("=" * 70)

    # Send to Telegram
    try:
        from src.utils.telegram import TelegramNotifier
        notifier = TelegramNotifier()
        await notifier.initialize()

        msg = f"""📊 *CORRELATED PORTFOLIO BACKTEST*

*Period:* {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}

💰 Net P/L: ${total_pnl:+,.2f}
📈 Return: {(total_pnl/INITIAL_BALANCE)*100:+.1f}%
🎯 Win Rate: {win_rate:.1f}%
📊 Profit Factor: {profit_factor:.2f}
📅 Losing Months: {losing_months}/{len(monthly)}

*Per-Pair:*
🇺🇸 GBPUSD: {len(usd_trades)} trades, {usd_wr:.1f}% WR, ${usd_pnl:+,.2f}
🇯🇵 GBPJPY: {len(jpy_trades)} trades, {jpy_wr:.1f}% WR, ${jpy_pnl:+,.2f}

🔗 Correlation skipped: {correlation_skipped} trades
"""
        await notifier.send_message(msg)
        print("\n[TELEGRAM] Sent!")
    except Exception as e:
        print(f"\n[TELEGRAM] Error: {e}")

    return {'pnl': total_pnl, 'pf': profit_factor, 'wr': win_rate}

if __name__ == "__main__":
    asyncio.run(run_portfolio_backtest())
