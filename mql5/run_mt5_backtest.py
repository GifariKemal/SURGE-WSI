"""
MT5 Backtest Runner for GBPUSD H1 QuadLayer v6.9
Runs backtest using MT5 Python API and exports results
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import json

# Initialize MT5
print("Initializing MT5...")
if not mt5.initialize():
    print(f"MT5 initialization failed: {mt5.last_error()}")
    quit()

print(f"MT5 Version: {mt5.version()}")
print(f"Terminal Info: {mt5.terminal_info()}")

# Get account info
account_info = mt5.account_info()
if account_info:
    print(f"\nAccount: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Balance: ${account_info.balance:,.2f}")
    print(f"Leverage: 1:{account_info.leverage}")

# Strategy parameters (matching EA)
SYMBOL = "GBPUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
START_DATE = datetime(2024, 2, 1)
END_DATE = datetime(2026, 1, 30)
INITIAL_BALANCE = 50000
RISK_PERCENT = 1.0
SL_ATR_MULT = 1.5
TP_RATIO = 1.5

# Day Multipliers (v6.9)
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}

# Hour Multipliers
HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

print(f"\n{'='*60}")
print("GBPUSD H1 QuadLayer v6.9 Backtest")
print(f"{'='*60}")
print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"Initial Balance: ${INITIAL_BALANCE:,}")
print(f"Risk: {RISK_PERCENT}% | SL: {SL_ATR_MULT}x ATR | TP: {TP_RATIO}:1")

# Get historical data
print(f"\nFetching {SYMBOL} H1 data...")
rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, START_DATE, END_DATE)

if rates is None or len(rates) == 0:
    print(f"Failed to get rates: {mt5.last_error()}")
    mt5.shutdown()
    quit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
print(f"Loaded {len(df)} bars from {df['time'].min()} to {df['time'].max()}")

# Calculate indicators
def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()

print("Calculating indicators...")
df['atr'] = calculate_atr(df)
df['ema20'] = calculate_ema(df['close'], 20)
df['ema50'] = calculate_ema(df['close'], 50)
df['rsi'] = calculate_rsi(df['close'])
df['adx'] = calculate_adx(df)
df['atr_pips'] = df['atr'] / 0.0001

# Get symbol info
symbol_info = mt5.symbol_info(SYMBOL)
pip_value = symbol_info.trade_tick_value * 10 if symbol_info else 10.0

# Backtest simulation
print("\nRunning backtest simulation...")
trades = []
balance = INITIAL_BALANCE
position = None
monthly_pnl = {}

for i in range(50, len(df)):
    bar = df.iloc[i]
    prev_bar = df.iloc[i-1]
    current_time = bar['time']

    # Check existing position
    if position:
        # Check SL/TP
        if position['direction'] == 'BUY':
            if bar['low'] <= position['sl']:
                pnl = -position['risk_amount']
                position['exit_price'] = position['sl']
                position['exit_reason'] = 'SL'
            elif bar['high'] >= position['tp']:
                pnl = position['risk_amount'] * TP_RATIO
                position['exit_price'] = position['tp']
                position['exit_reason'] = 'TP'
            else:
                continue
        else:  # SELL
            if bar['high'] >= position['sl']:
                pnl = -position['risk_amount']
                position['exit_price'] = position['sl']
                position['exit_reason'] = 'SL'
            elif bar['low'] <= position['tp']:
                pnl = position['risk_amount'] * TP_RATIO
                position['exit_price'] = position['tp']
                position['exit_reason'] = 'TP'
            else:
                continue

        # Close position
        balance += pnl
        position['pnl'] = pnl
        position['exit_time'] = current_time
        trades.append(position.copy())

        # Track monthly P&L
        month_key = current_time.strftime('%Y-%m')
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl

        position = None
        continue

    # Check filters
    day = current_time.weekday()
    hour = current_time.hour

    day_mult = DAY_MULTIPLIERS.get(day, 0)
    hour_mult = HOUR_MULTIPLIERS.get(hour, 0)

    if day_mult <= 0 or hour_mult <= 0:
        continue

    # Check kill zone
    in_london = 8 <= hour <= 10
    in_ny = 13 <= hour <= 17
    if not (in_london or in_ny):
        continue

    # Check ATR
    atr_pips = bar['atr_pips']
    if pd.isna(atr_pips) or atr_pips < 8 or atr_pips > 25:
        continue

    # Check regime
    close = bar['close']
    ema20 = bar['ema20']
    ema50 = bar['ema50']

    is_bullish = close > ema20 > ema50
    is_bearish = close < ema20 < ema50

    if not is_bullish and not is_bearish:
        continue

    # Check entry signals
    signal = None
    signal_type = None
    quality = 0

    # Order Block detection
    body_ratio = abs(bar['close'] - bar['open']) / (bar['high'] - bar['low'] + 1e-10)
    prev_bearish = prev_bar['close'] < prev_bar['open']
    prev_bullish = prev_bar['close'] > prev_bar['open']
    curr_bullish = bar['close'] > bar['open']
    curr_bearish = bar['close'] < bar['open']

    if prev_bearish and curr_bullish and body_ratio > 0.55 and bar['close'] > prev_bar['high']:
        if is_bullish:
            signal = 'BUY'
            signal_type = 'ORDER_BLOCK'
            quality = body_ratio * 100
    elif prev_bullish and curr_bearish and body_ratio > 0.55 and bar['close'] < prev_bar['low']:
        if is_bearish:
            signal = 'SELL'
            signal_type = 'ORDER_BLOCK'
            quality = body_ratio * 100

    # Session filter for Order Block
    if signal_type == 'ORDER_BLOCK':
        if hour == 8 or hour == 16:
            signal = None

    # EMA Pullback detection (if no Order Block signal)
    if not signal and body_ratio > 0.4 and bar['adx'] > 20 and 30 <= bar['rsi'] <= 70:
        atr_distance = atr_pips * 0.0001 * 1.5

        if curr_bullish and is_bullish:
            dist = bar['low'] - ema20
            if dist <= atr_distance:
                signal = 'BUY'
                signal_type = 'EMA_PULLBACK'
                quality = min(100, max(55, 60 + (bar['adx'] - 20) * 0.5))
        elif curr_bearish and is_bearish:
            dist = ema20 - bar['high']
            if dist <= atr_distance:
                signal = 'SELL'
                signal_type = 'EMA_PULLBACK'
                quality = min(100, max(55, 60 + (bar['adx'] - 20) * 0.5))

    # Session filter for EMA Pullback
    if signal_type == 'EMA_PULLBACK':
        if hour == 13 or hour == 14:
            signal = None

    # Quality filter
    if signal and quality < 60:
        signal = None

    # Execute trade
    if signal:
        risk_mult = day_mult * hour_mult * (quality / 100)
        risk_amount = balance * (RISK_PERCENT / 100) * risk_mult
        sl_pips = atr_pips * SL_ATR_MULT
        tp_pips = sl_pips * TP_RATIO

        entry_price = bar['close']

        if signal == 'BUY':
            sl = entry_price - sl_pips * 0.0001
            tp = entry_price + tp_pips * 0.0001
        else:
            sl = entry_price + sl_pips * 0.0001
            tp = entry_price - tp_pips * 0.0001

        position = {
            'direction': signal,
            'entry_time': current_time,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'risk_amount': risk_amount,
            'signal_type': signal_type,
            'quality': quality,
            'session': 'london' if in_london else 'newyork'
        }

# Results
print(f"\n{'='*60}")
print("BACKTEST RESULTS")
print(f"{'='*60}")

total_trades = len(trades)
if total_trades > 0:
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = total_trades - wins
    win_rate = wins / total_trades * 100

    total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    net_pnl = total_profit - total_loss
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    avg_win = total_profit / wins if wins > 0 else 0
    avg_loss = total_loss / losses if losses > 0 else 0

    # Count losing months
    losing_months = sum(1 for m, pnl in monthly_pnl.items() if pnl < 0)
    total_months = len(monthly_pnl)

    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"Net P/L:         ${net_pnl:+,.2f}")
    print(f"Total Return:    {(net_pnl/INITIAL_BALANCE)*100:+.1f}%")
    print(f"Avg Win:         ${avg_win:,.2f}")
    print(f"Avg Loss:        ${avg_loss:,.2f}")
    print(f"Final Balance:   ${balance:,.2f}")
    print(f"Losing Months:   {losing_months}/{total_months}")

    print(f"\n{'='*60}")
    print("MONTHLY BREAKDOWN")
    print(f"{'='*60}")
    for month, pnl in sorted(monthly_pnl.items()):
        status = "WIN" if pnl >= 0 else "LOSS"
        print(f"  [{status:4}] {month}: ${pnl:+,.2f}")

    # Export to CSV
    trades_df = pd.DataFrame(trades)
    output_path = r'C:\Users\Administrator\Music\SURGE-WSI\mql5\quadlayer_v69_mt5_trades.csv'
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades exported to: {output_path}")

    # Export summary
    summary = {
        'strategy': 'GBPUSD H1 QuadLayer v6.9',
        'period': f"{START_DATE.date()} to {END_DATE.date()}",
        'initial_balance': INITIAL_BALANCE,
        'final_balance': round(balance, 2),
        'total_trades': total_trades,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'net_pnl': round(net_pnl, 2),
        'return_pct': round((net_pnl/INITIAL_BALANCE)*100, 1),
        'losing_months': losing_months,
        'total_months': total_months,
        'monthly_pnl': {k: round(v, 2) for k, v in monthly_pnl.items()}
    }

    summary_path = r'C:\Users\Administrator\Music\SURGE-WSI\mql5\quadlayer_v69_mt5_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary exported to: {summary_path}")

else:
    print("No trades executed!")

mt5.shutdown()
print("\nMT5 shutdown complete.")
