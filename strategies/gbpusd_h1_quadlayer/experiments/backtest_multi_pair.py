"""
SURGE-WSI MULTI-PAIR BACKTEST
=============================
2 Pairs trading simultaneously (like live demo)
- GBPUSD: 60% allocation
- GBPJPY: 40% allocation
- No session filter - both trade during optimal hours
- Independent risk management per pair
"""

import asyncio
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.gbpusd_h1_quadlayer.trading_filters import IntraMonthRiskManager, PatternBasedFilter

# ============================================================
# MULTI-PAIR CONFIGURATION
# ============================================================
INITIAL_BALANCE = 50_000.0

# Pair config - NO SPLIT, each uses full balance
PAIRS_CONFIG = {
    'GBPUSD': {
        'allocation': 1.0,       # 100% - full balance
        'pip_size': 0.0001,
        'pip_value': 10.0,       # $10 per pip per lot
    },
    'GBPJPY': {
        'allocation': 1.0,       # 100% - full balance
        'pip_size': 0.01,
        'pip_value': 6.5,        # ~$6.5 per pip per lot
    }
}
# Risk: 0.5% per pair, max 1% if both trade simultaneously

# Risk parameters (same for all pairs)
RISK_PERCENT = 0.5
SL_ATR_MULT = 1.2
TP_RATIO = 3.5
MAX_LOT = 5.0

# Trading hours (London + NY) - same for all pairs
TRADING_HOURS = list(range(8, 21))  # 08:00-20:59 UTC (full London+NY)

# Backtest period
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 2, 1)

# ============================================================
# DATA STRUCTURES
# ============================================================
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

# ============================================================
# DATA FETCHING
# ============================================================
async def fetch_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch H1 data from MT5"""
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

    print(f"  {symbol}: {len(df)} bars loaded")
    return df

# ============================================================
# SIGNAL DETECTION
# ============================================================
def detect_signals(df: pd.DataFrame, symbol: str, pip_size: float) -> List[dict]:
    """Detect entry signals for a pair"""
    signals = []

    for i in range(50, len(df)):
        bar = df.iloc[i]
        prev = df.iloc[i-1]
        hour = bar.name.hour

        if hour not in TRADING_HOURS:
            continue

        atr = bar['atr']
        if pd.isna(atr) or atr <= 0:
            continue

        atr_pips = atr / pip_size
        if atr_pips < 10 or atr_pips > 50:
            continue

        # EMA alignment
        bullish = bar['ema_9'] > bar['ema_21'] > bar['ema_50']
        bearish = bar['ema_9'] < bar['ema_21'] < bar['ema_50']

        # RSI
        rsi = bar['rsi']
        rsi_bull = 40 < rsi < 70
        rsi_bear = 30 < rsi < 60

        signal = None

        # ORDER BLOCK - Engulfing
        if bullish and rsi_bull:
            if prev['close'] < prev['open'] and bar['close'] > bar['open'] and bar['close'] > prev['high']:
                signal = {'dir': 'BUY', 'type': 'ORDER_BLOCK'}

        if bearish and rsi_bear and signal is None:
            if prev['close'] > prev['open'] and bar['close'] < bar['open'] and bar['close'] < prev['low']:
                signal = {'dir': 'SELL', 'type': 'ORDER_BLOCK'}

        # EMA PULLBACK
        if bullish and rsi_bull and signal is None:
            if bar['low'] <= bar['ema_21'] * 1.002 and bar['close'] > bar['ema_9']:
                signal = {'dir': 'BUY', 'type': 'EMA_PULLBACK'}

        if bearish and rsi_bear and signal is None:
            if bar['high'] >= bar['ema_21'] * 0.998 and bar['close'] < bar['ema_9']:
                signal = {'dir': 'SELL', 'type': 'EMA_PULLBACK'}

        if signal:
            signals.append({
                'time': bar.name,
                'price': bar['close'],
                'atr_pips': atr_pips,
                **signal
            })

    return signals

# ============================================================
# SIMULATE TRADE
# ============================================================
def simulate_trade(df: pd.DataFrame, signal: dict, config: dict, lot_size: float) -> Tuple[float, datetime, str]:
    """Simulate a single trade, return (pnl, exit_time, exit_reason)"""

    entry_idx = df.index.get_loc(signal['time'])
    entry_price = signal['price']
    direction = signal['dir']
    atr_pips = signal['atr_pips']

    pip_size = config['pip_size']
    pip_value = config['pip_value']

    sl_pips = atr_pips * SL_ATR_MULT
    tp_pips = sl_pips * TP_RATIO

    if direction == 'BUY':
        sl = entry_price - (sl_pips * pip_size)
        tp = entry_price + (tp_pips * pip_size)
    else:
        sl = entry_price + (sl_pips * pip_size)
        tp = entry_price - (tp_pips * pip_size)

    # Find exit
    for j in range(entry_idx + 1, min(entry_idx + 48, len(df))):
        bar = df.iloc[j]

        if direction == 'BUY':
            if bar['low'] <= sl:
                pips = -sl_pips
                return pips * lot_size * pip_value, bar.name, 'SL'
            if bar['high'] >= tp:
                pips = tp_pips
                return pips * lot_size * pip_value, bar.name, 'TP'
        else:
            if bar['high'] >= sl:
                pips = -sl_pips
                return pips * lot_size * pip_value, bar.name, 'SL'
            if bar['low'] <= tp:
                pips = tp_pips
                return pips * lot_size * pip_value, bar.name, 'TP'

    # Timeout
    if entry_idx + 48 < len(df):
        exit_bar = df.iloc[entry_idx + 48]
        if direction == 'BUY':
            pips = (exit_bar['close'] - entry_price) / pip_size
        else:
            pips = (entry_price - exit_bar['close']) / pip_size
        return pips * lot_size * pip_value, exit_bar.name, 'TIMEOUT'

    return 0, signal['time'], 'NO_EXIT'

# ============================================================
# MAIN BACKTEST
# ============================================================
async def run_multi_pair_backtest():
    """Run multi-pair backtest"""

    print("=" * 70)
    print("SURGE-WSI MULTI-PAIR BACKTEST")
    print("=" * 70)
    print(f"Initial Balance: ${INITIAL_BALANCE:,.0f}")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print()
    for pair, cfg in PAIRS_CONFIG.items():
        print(f"  {pair}: {cfg['allocation']*100:.0f}% = ${INITIAL_BALANCE * cfg['allocation']:,.0f}")
    print("=" * 70)

    # Fetch data for all pairs
    print("\nLoading data...")
    data = {}
    for pair in PAIRS_CONFIG:
        data[pair] = await fetch_data(pair, START_DATE, END_DATE)
        if data[pair].empty:
            print(f"ERROR: No data for {pair}")
            return

    # Detect signals for all pairs
    print("\nDetecting signals...")
    signals = {}
    for pair, cfg in PAIRS_CONFIG.items():
        signals[pair] = detect_signals(data[pair], pair, cfg['pip_size'])
        print(f"  {pair}: {len(signals[pair])} signals")

    # SHARED balance for all pairs
    shared_balance = INITIAL_BALANCE

    # Initialize per-pair state (filters are independent, balance is shared)
    pair_state = {}
    for pair, cfg in PAIRS_CONFIG.items():
        pair_state[pair] = {
            'risk_mgr': IntraMonthRiskManager(),
            'pattern_flt': PatternBasedFilter(),
            'last_trade': None,
            'trades': []
        }

    # Process signals chronologically
    all_signals = []
    for pair, sigs in signals.items():
        for s in sigs:
            s['pair'] = pair
            all_signals.append(s)
    all_signals.sort(key=lambda x: x['time'])

    print(f"\nProcessing {len(all_signals)} total signals...")

    for signal in all_signals:
        pair = signal['pair']
        cfg = PAIRS_CONFIG[pair]
        state = pair_state[pair]

        # Cooldown 2 hours between trades per pair
        if state['last_trade']:
            if (signal['time'] - state['last_trade']).total_seconds() < 2 * 3600:
                continue

        # Risk manager check
        can_trade, quality_adj, reason = state['risk_mgr'].new_trade_check(signal['time'])
        if not can_trade:
            continue

        # Pattern filter check
        pattern_ok, size_mult, _ = state['pattern_flt'].check_trade_allowed()
        if not pattern_ok:
            continue

        # Quality threshold
        quality = 70 + quality_adj
        if quality >= 100:
            continue

        # Calculate lot size using SHARED balance
        sl_pips = signal['atr_pips'] * SL_ATR_MULT
        risk_amount = shared_balance * (RISK_PERCENT / 100)
        lot_size = risk_amount / (sl_pips * cfg['pip_value'])
        lot_size = min(lot_size, MAX_LOT)
        lot_size = round(lot_size * size_mult, 2)

        if lot_size < 0.01:
            continue

        # Simulate trade
        pnl, exit_time, exit_reason = simulate_trade(data[pair], signal, cfg, lot_size)

        if exit_reason == 'NO_EXIT':
            continue

        # Update shared balance and state
        shared_balance += pnl
        state['last_trade'] = signal['time']
        state['risk_mgr'].record_trade(pnl, exit_time, signal['dir'])
        state['pattern_flt'].record_trade(signal['dir'], pnl, exit_time)

        # Record trade
        state['trades'].append(Trade(
            pair=pair,
            entry_time=signal['time'],
            exit_time=exit_time,
            direction=signal['dir'],
            entry_price=signal['price'],
            lot_size=lot_size,
            pnl=pnl,
            exit_reason=exit_reason
        ))

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("MULTI-PAIR BACKTEST RESULTS")
    print("=" * 70)

    total_pnl = 0
    total_trades = 0
    total_wins = 0

    print("\n[PER-PAIR RESULTS]")
    print("-" * 60)

    total_pnl = shared_balance - INITIAL_BALANCE

    for pair, state in pair_state.items():
        trades = state['trades']
        if not trades:
            continue

        pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 999

        print(f"\n  {pair}:")
        print(f"    Trades: {len(trades)}")
        print(f"    Win Rate: {wr:.1f}%")
        print(f"    Profit Factor: {pf:.2f}")
        print(f"    Net P/L: ${pnl:+,.2f}")

        total_trades += len(trades)
        total_wins += wins

    # Portfolio summary
    all_trades = []
    for state in pair_state.values():
        all_trades.extend(state['trades'])

    if all_trades:
        total_wr = total_wins / total_trades * 100
        gross_profit = sum(t.pnl for t in all_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in all_trades if t.pnl <= 0))
        total_pf = gross_profit / gross_loss if gross_loss > 0 else 999

        # Monthly breakdown
        monthly = {}
        for t in all_trades:
            m = t.entry_time.strftime('%Y-%m')
            if m not in monthly:
                monthly[m] = 0
            monthly[m] += t.pnl

        losing_months = sum(1 for v in monthly.values() if v < 0)

        print("\n" + "=" * 60)
        print("[PORTFOLIO SUMMARY]")
        print("=" * 60)
        print(f"  Initial Balance:  ${INITIAL_BALANCE:,.2f}")
        print(f"  Final Balance:    ${shared_balance:,.2f}")
        print(f"  Net P/L:          ${total_pnl:+,.2f}")
        print(f"  Total Return:     {(total_pnl/INITIAL_BALANCE)*100:+.1f}%")
        print()
        print(f"  Total Trades:     {total_trades}")
        print(f"  Win Rate:         {total_wr:.1f}%")
        print(f"  Profit Factor:    {total_pf:.2f}")
        print(f"  Losing Months:    {losing_months}/{len(monthly)}")

        print("\n[MONTHLY BREAKDOWN]")
        print("-" * 60)
        for m, pnl in sorted(monthly.items()):
            status = "WIN " if pnl >= 0 else "LOSS"
            print(f"  [{status}] {m}: ${pnl:+,.2f}")

        print("\n" + "=" * 60)
        if total_pnl > 5000:
            print("✅ PROFITABLE - Ready for live demo")
        else:
            print("⚠️ MARGINAL - Consider tuning parameters")
        print("=" * 60)

        # Send to Telegram
        try:
            from src.utils.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            await notifier.initialize()

            # Per pair summary
            pair_summary = ""
            for pair, state in pair_state.items():
                trades = state['trades']
                if trades:
                    pnl = sum(t.pnl for t in trades)
                    wr = sum(1 for t in trades if t.pnl > 0) / len(trades) * 100
                    pair_summary += f"  • {pair}: {len(trades)} trades, {wr:.0f}% WR, ${pnl:+,.0f}\n"

            msg = f"""📊 *MULTI-PAIR BACKTEST*

*Period:* {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}

*Portfolio:*
💰 Net P/L: ${total_pnl:+,.2f}
📈 Return: {(total_pnl/INITIAL_BALANCE)*100:+.1f}%
🎯 Win Rate: {total_wr:.1f}%
📊 PF: {total_pf:.2f}
📅 Losing Months: {losing_months}/{len(monthly)}

*Per Pair:*
{pair_summary}
*Allocation:*
  GBPUSD: 60% (${INITIAL_BALANCE*0.6:,.0f})
  GBPJPY: 40% (${INITIAL_BALANCE*0.4:,.0f})
"""
            await notifier.send_message(msg)
            print("\n[TELEGRAM] Results sent!")
        except Exception as e:
            print(f"\n[TELEGRAM] Error: {e}")

    return total_pnl

if __name__ == "__main__":
    asyncio.run(run_multi_pair_backtest())
