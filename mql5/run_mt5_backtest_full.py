"""
MT5 Backtest Runner for GBPUSD H1 QuadLayer v6.9
FULL VERSION with Layer 3 & Layer 4 filters
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Initialize MT5
print("Initializing MT5...")
if not mt5.initialize():
    print(f"MT5 initialization failed: {mt5.last_error()}")
    quit()

print(f"MT5 Version: {mt5.version()}")

# ============================================================
# CONFIGURATION - MATCHING PYTHON BACKTEST EXACTLY
# ============================================================
SYMBOL = "GBPUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
START_DATE = datetime(2025, 1, 1)  # Match Python backtest period
END_DATE = datetime(2026, 1, 30)
INITIAL_BALANCE = 50000
RISK_PERCENT = 1.0
SL_ATR_MULT = 1.5
TP_RATIO = 1.5
MIN_ATR = 8.0
MAX_ATR = 25.0
PIP_SIZE = 0.0001

# Day Multipliers (v6.9)
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}

# Hour Multipliers
HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

# Layer 3: Intra-Month Risk Manager Config
MONTHLY_LOSS_THRESHOLD_1 = -150
MONTHLY_LOSS_THRESHOLD_2 = -250
MONTHLY_LOSS_THRESHOLD_3 = -350
MONTHLY_LOSS_STOP = -400
CONSECUTIVE_LOSS_THRESHOLD = 3
CONSECUTIVE_LOSS_MAX = 6

# Layer 4: Pattern Filter Config
WARMUP_TRADES = 15
ROLLING_WINDOW = 10
ROLLING_WR_HALT = 0.10
ROLLING_WR_CAUTION = 0.25
CAUTION_SIZE_MULT = 0.6
BOTH_DIRECTIONS_FAIL_THRESHOLD = 4
RECOVERY_SIZE_MULT = 0.5

# Session Filter
SKIP_HOURS = [11]
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]

# Monthly Tradeable % (from backtest)
MONTHLY_TRADEABLE_PCT = {
    (2025, 1): 65, (2025, 2): 55, (2025, 3): 70, (2025, 4): 70,
    (2025, 5): 62, (2025, 6): 68, (2025, 7): 78, (2025, 8): 65,
    (2025, 9): 72, (2025, 10): 58, (2025, 11): 66, (2025, 12): 60,
    (2026, 1): 65,
}

def get_monthly_quality_adjustment(dt):
    """Layer 1: Get quality adjustment from monthly profile"""
    key = (dt.year, dt.month)
    tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)

    if tradeable_pct < 30:
        return 50
    elif tradeable_pct < 40:
        return 35
    elif tradeable_pct < 50:
        return 25
    elif tradeable_pct < 60:
        return 15
    elif tradeable_pct < 70:
        return 10
    elif tradeable_pct < 75:
        return 5
    else:
        return 0

# ============================================================
# LAYER 3: INTRA-MONTH RISK MANAGER
# ============================================================
class IntraMonthRiskManager:
    def __init__(self):
        self.current_month = None
        self.monthly_pnl = 0.0
        self.consecutive_losses = 0
        self.month_stopped = False
        self.day_stopped = False
        self.current_day = None
        self.daily_losses = 0

    def new_trade_check(self, dt):
        month_key = (dt.year, dt.month)
        day_key = dt.date()

        # Reset for new month
        if self.current_month != month_key:
            self.current_month = month_key
            self.monthly_pnl = 0.0
            self.consecutive_losses = 0
            self.month_stopped = False
            self.day_stopped = False

        # Reset for new day
        if self.current_day != day_key:
            self.current_day = day_key
            self.daily_losses = 0
            self.day_stopped = False

        if self.month_stopped:
            return False, 0, "MONTH_STOPPED"
        if self.day_stopped:
            return False, 0, "DAY_STOPPED"

        # Monthly loss circuit breaker
        if self.monthly_pnl <= MONTHLY_LOSS_STOP:
            self.month_stopped = True
            return False, 0, "MONTH_CIRCUIT_BREAKER"

        # Dynamic adjustment
        dynamic_adj = 0
        if self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_3:
            dynamic_adj = 15
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_2:
            dynamic_adj = 10
        elif self.monthly_pnl <= MONTHLY_LOSS_THRESHOLD_1:
            dynamic_adj = 5

        # Consecutive losses
        if self.consecutive_losses >= CONSECUTIVE_LOSS_MAX:
            self.day_stopped = True
            return False, 0, "CONSECUTIVE_LOSS_STOP"
        if self.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            dynamic_adj += 5

        return True, dynamic_adj, "OK"

    def record_trade(self, pnl):
        self.monthly_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_losses += 1
        else:
            self.consecutive_losses = 0

# ============================================================
# LAYER 4: PATTERN-BASED FILTER
# ============================================================
class PatternBasedFilter:
    def __init__(self):
        self.trade_history = []
        self.is_halted = False
        self.in_recovery = False
        self.recovery_wins = 0
        self.current_month = None

    def reset_for_month(self, month):
        self.current_month = month
        self.is_halted = False
        self.in_recovery = False
        self.recovery_wins = 0
        if len(self.trade_history) > 30:
            self.trade_history = self.trade_history[-30:]

    def _get_rolling_stats(self):
        if len(self.trade_history) < 3:
            return {'rolling_wr': 1.0, 'buy_wr': 1.0, 'sell_wr': 1.0, 'both_fail': False}

        recent = self.trade_history[-ROLLING_WINDOW:]
        wins = sum(1 for d, pnl in recent if pnl > 0)
        rolling_wr = wins / len(recent) if recent else 1.0

        buy_trades = [(d, p) for d, p in recent if d == 'BUY']
        sell_trades = [(d, p) for d, p in recent if d == 'SELL']

        buy_wr = sum(1 for _, p in buy_trades if p > 0) / len(buy_trades) if buy_trades else 1.0
        sell_wr = sum(1 for _, p in sell_trades if p > 0) / len(sell_trades) if sell_trades else 1.0

        # Check if BOTH directions are failing
        both_fail = False
        recent_window = self.trade_history[-BOTH_DIRECTIONS_FAIL_THRESHOLD*2:]
        if len(recent_window) >= BOTH_DIRECTIONS_FAIL_THRESHOLD * 2:
            buy_losses = sum(1 for d, p in recent_window if d == 'BUY' and p < 0)
            sell_losses = sum(1 for d, p in recent_window if d == 'SELL' and p < 0)
            if buy_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD and sell_losses >= BOTH_DIRECTIONS_FAIL_THRESHOLD:
                both_fail = True

        return {'rolling_wr': rolling_wr, 'buy_wr': buy_wr, 'sell_wr': sell_wr, 'both_fail': both_fail}

    def check_trade(self, direction):
        # During warmup, always allow
        if len(self.trade_history) < WARMUP_TRADES:
            return True, 0, 1.0, "WARMUP"

        if self.is_halted and not self.in_recovery:
            return False, 0, 1.0, "HALTED"

        stats = self._get_rolling_stats()

        if stats['both_fail']:
            self.is_halted = True
            self.in_recovery = True
            self.recovery_wins = 0
            return False, 0, 1.0, "CHOPPY_MARKET"

        if stats['rolling_wr'] < ROLLING_WR_HALT:
            self.is_halted = True
            self.in_recovery = True
            self.recovery_wins = 0
            return False, 0, 1.0, "LOW_WIN_RATE"

        size_mult = 1.0
        extra_q = 0

        if self.in_recovery:
            size_mult = RECOVERY_SIZE_MULT
            extra_q = 5
        elif stats['rolling_wr'] < ROLLING_WR_CAUTION:
            size_mult = CAUTION_SIZE_MULT
            extra_q = 3

        return True, extra_q, size_mult, "OK"

    def record_trade(self, direction, pnl):
        self.trade_history.append((direction, pnl))
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]

        if self.in_recovery:
            if pnl > 0:
                self.recovery_wins += 1
                if self.recovery_wins >= 1:
                    self.is_halted = False
                    self.in_recovery = False
                    self.recovery_wins = 0
            else:
                self.recovery_wins = 0

# ============================================================
# INDICATOR CALCULATIONS
# ============================================================
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
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()

# ============================================================
# MAIN BACKTEST
# ============================================================
print(f"\n{'='*60}")
print("GBPUSD H1 QuadLayer v6.9 FULL Backtest")
print(f"{'='*60}")
print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"Initial Balance: ${INITIAL_BALANCE:,}")

# Get historical data
rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
if rates is None or len(rates) == 0:
    print(f"Failed to get rates: {mt5.last_error()}")
    mt5.shutdown()
    quit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
print(f"Loaded {len(df)} bars")

# Calculate indicators
df['atr'] = calculate_atr(df)
df['ema20'] = calculate_ema(df['close'], 20)
df['ema50'] = calculate_ema(df['close'], 50)
df['rsi'] = calculate_rsi(df['close'])
df['adx'] = calculate_adx(df)
df['atr_pips'] = df['atr'] / PIP_SIZE

# Initialize managers
risk_manager = IntraMonthRiskManager()
pattern_filter = PatternBasedFilter()
current_month_key = None

# Backtest
trades = []
balance = INITIAL_BALANCE
position = None
monthly_pnl = {}
skip_stats = {'LAYER3': 0, 'LAYER4': 0, 'SESSION': 0, 'ATR': 0, 'REGIME': 0}

print("Running backtest with FULL Quad-Layer filter...")

for i in range(50, len(df)):
    bar = df.iloc[i]
    prev_bar = df.iloc[i-1]
    current_time = bar['time'].to_pydatetime()

    # Check existing position
    if position:
        if position['direction'] == 'BUY':
            if bar['low'] <= position['sl']:
                pnl = -position['risk_amount']
                exit_reason = 'SL'
            elif bar['high'] >= position['tp']:
                pnl = position['risk_amount'] * TP_RATIO
                exit_reason = 'TP'
            else:
                continue
        else:
            if bar['high'] >= position['sl']:
                pnl = -position['risk_amount']
                exit_reason = 'SL'
            elif bar['low'] <= position['tp']:
                pnl = position['risk_amount'] * TP_RATIO
                exit_reason = 'TP'
            else:
                continue

        balance += pnl
        position['pnl'] = pnl
        position['exit_reason'] = exit_reason

        # Record to managers
        risk_manager.record_trade(pnl)
        pattern_filter.record_trade(position['direction'], pnl)

        trades.append(position.copy())
        month_key = current_time.strftime('%Y-%m')
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl
        position = None
        continue

    # Reset pattern filter on new month
    month_key = (current_time.year, current_time.month)
    if month_key != current_month_key:
        current_month_key = month_key
        pattern_filter.reset_for_month(current_time.month)

    # Time filters
    day = current_time.weekday()
    hour = current_time.hour

    day_mult = DAY_MULTIPLIERS.get(day, 0)
    hour_mult = HOUR_MULTIPLIERS.get(hour, 0)
    if day_mult <= 0 or hour_mult <= 0:
        continue

    in_london = 8 <= hour <= 10
    in_ny = 13 <= hour <= 17
    if not (in_london or in_ny):
        continue

    # LAYER 3: Intra-month risk manager
    can_trade, intra_adj, reason = risk_manager.new_trade_check(current_time)
    if not can_trade:
        skip_stats['LAYER3'] += 1
        continue

    # ATR filter
    atr_pips = bar['atr_pips']
    if pd.isna(atr_pips) or atr_pips < MIN_ATR or atr_pips > MAX_ATR:
        skip_stats['ATR'] += 1
        continue

    # Regime
    close = bar['close']
    ema20 = bar['ema20']
    ema50 = bar['ema50']
    is_bullish = close > ema20 > ema50
    is_bearish = close < ema20 < ema50
    if not is_bullish and not is_bearish:
        skip_stats['REGIME'] += 1
        continue

    # Layer 1: Monthly quality adjustment
    monthly_adj = get_monthly_quality_adjustment(current_time)
    base_quality = 65
    dynamic_quality = base_quality + monthly_adj + intra_adj

    # Signal detection
    signal = None
    signal_type = None
    quality = 0

    body_ratio = abs(bar['close'] - bar['open']) / (bar['high'] - bar['low'] + 1e-10)
    prev_bearish = prev_bar['close'] < prev_bar['open']
    prev_bullish = prev_bar['close'] > prev_bar['open']
    curr_bullish = bar['close'] > bar['open']
    curr_bearish = bar['close'] < bar['open']

    # Order Block
    if prev_bearish and curr_bullish and body_ratio > 0.55 and bar['close'] > prev_bar['high']:
        if is_bullish and hour not in SKIP_ORDER_BLOCK_HOURS:
            signal = 'BUY'
            signal_type = 'ORDER_BLOCK'
            quality = body_ratio * 100
    elif prev_bullish and curr_bearish and body_ratio > 0.55 and bar['close'] < prev_bar['low']:
        if is_bearish and hour not in SKIP_ORDER_BLOCK_HOURS:
            signal = 'SELL'
            signal_type = 'ORDER_BLOCK'
            quality = body_ratio * 100

    # EMA Pullback
    if not signal and body_ratio > 0.4 and bar['adx'] > 20 and 30 <= bar['rsi'] <= 70:
        atr_distance = atr_pips * PIP_SIZE * 1.5
        if curr_bullish and is_bullish and hour not in SKIP_EMA_PULLBACK_HOURS:
            dist = bar['low'] - ema20
            if dist <= atr_distance:
                signal = 'BUY'
                signal_type = 'EMA_PULLBACK'
                quality = min(100, max(55, 60 + (bar['adx'] - 20) * 0.5))
        elif curr_bearish and is_bearish and hour not in SKIP_EMA_PULLBACK_HOURS:
            dist = ema20 - bar['high']
            if dist <= atr_distance:
                signal = 'SELL'
                signal_type = 'EMA_PULLBACK'
                quality = min(100, max(55, 60 + (bar['adx'] - 20) * 0.5))

    if not signal or quality < dynamic_quality:
        continue

    # LAYER 4: Pattern filter
    pattern_ok, pattern_extra_q, pattern_size_mult, pattern_reason = pattern_filter.check_trade(signal)
    if not pattern_ok:
        skip_stats['LAYER4'] += 1
        continue

    effective_quality = dynamic_quality + pattern_extra_q
    if quality < effective_quality:
        continue

    # Execute
    risk_mult = day_mult * hour_mult * (quality / 100) * pattern_size_mult
    risk_amount = balance * (RISK_PERCENT / 100) * risk_mult
    sl_pips = atr_pips * SL_ATR_MULT
    tp_pips = sl_pips * TP_RATIO
    entry_price = bar['close']

    if signal == 'BUY':
        sl = entry_price - sl_pips * PIP_SIZE
        tp = entry_price + tp_pips * PIP_SIZE
    else:
        sl = entry_price + sl_pips * PIP_SIZE
        tp = entry_price - tp_pips * PIP_SIZE

    position = {
        'direction': signal,
        'entry_time': current_time,
        'entry_price': entry_price,
        'sl': sl,
        'tp': tp,
        'risk_amount': risk_amount,
        'signal_type': signal_type,
        'quality': quality
    }

# Results
print(f"\n{'='*60}")
print("BACKTEST RESULTS (FULL QUAD-LAYER)")
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

    losing_months = sum(1 for m, pnl in monthly_pnl.items() if pnl < 0)
    total_months = len(monthly_pnl)

    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"Net P/L:         ${net_pnl:+,.2f}")
    print(f"Total Return:    {(net_pnl/INITIAL_BALANCE)*100:+.1f}%")
    print(f"Final Balance:   ${balance:,.2f}")
    print(f"Losing Months:   {losing_months}/{total_months}")

    print(f"\n{'='*60}")
    print("SKIP STATISTICS")
    print(f"{'='*60}")
    for k, v in skip_stats.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print("MONTHLY BREAKDOWN")
    print(f"{'='*60}")
    for month, pnl in sorted(monthly_pnl.items()):
        status = "WIN " if pnl >= 0 else "LOSS"
        print(f"  [{status}] {month}: ${pnl:+,.2f}")

    # Export
    trades_df = pd.DataFrame(trades)
    output_path = r'C:\Users\Administrator\Music\SURGE-WSI\mql5\quadlayer_v69_mt5_full_trades.csv'
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades exported to: {output_path}")

    summary = {
        'strategy': 'GBPUSD H1 QuadLayer v6.9 (Full)',
        'period': f"{START_DATE.date()} to {END_DATE.date()}",
        'total_trades': total_trades,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'net_pnl': round(net_pnl, 2),
        'losing_months': losing_months,
        'total_months': total_months,
        'monthly_pnl': {k: round(v, 2) for k, v in monthly_pnl.items()}
    }
    summary_path = r'C:\Users\Administrator\Music\SURGE-WSI\mql5\quadlayer_v69_mt5_full_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary exported to: {summary_path}")

mt5.shutdown()
