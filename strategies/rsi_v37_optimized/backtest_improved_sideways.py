"""
RSI v3.7 IMPROVED v3 - Better SIDEWAYS Detection
=================================================
Testing on Losing Months: Nov 2024 & Apr 2025

Problem: SIDEWAYS filter allows too many trades in losing months
- Nov 2024: 99.2% SIDEWAYS but still -$1,222
- Apr 2025: 85.8% SIDEWAYS but still -$1,717

Solution: Make SIDEWAYS detection STRICTER
1. Tighter SMA slope threshold
2. Add Bollinger Band squeeze detection
3. Add price range compression check
4. Require multiple conditions for "true" SIDEWAYS
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# Configuration
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '61045904'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'iy#K5L7sF')
MT5_SERVER = os.getenv('MT5_SERVER', 'FinexBisnisSolusi-Demo')
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe")

SYMBOL = "GBPUSD"
INITIAL_BALANCE = 10000.0

# Strategy Parameters
RSI_PERIOD = 10
RSI_OVERSOLD = 42
RSI_OVERBOUGHT = 58
ATR_PERIOD = 14
SL_MULT = 1.5
TP_LOW = 2.4
TP_MED = 3.0
TP_HIGH = 3.6
MAX_HOLDING_HOURS = 46
MIN_ATR_PCT = 20
MAX_ATR_PCT = 80
RISK_PER_TRADE = 0.01

USE_CONSEC_LOSS_FILTER = True
CONSEC_LOSS_LIMIT = 3

# ============================================
# SIDEWAYS DETECTION PARAMETERS
# ============================================

# Original criteria
ORIG_SMA_SLOPE_THRESHOLD = 0.5  # abs(slope) < 0.5 = SIDEWAYS

# Improved criteria
IMPROVED_SMA_SLOPE_THRESHOLD = 0.3  # Stricter slope

# Bollinger Band Squeeze
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_PERCENTILE = 30  # Width below 30th percentile = squeeze

# Price Range Compression
RANGE_LOOKBACK = 20  # Hours
RANGE_COMPRESSION_THRESHOLD = 50  # Below median = compressed


def connect_mt5():
    if not MT5_PASSWORD:
        print("ERROR: MT5_PASSWORD not set")
        return False
    if not mt5.initialize(path=MT5_PATH):
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return False
    return True


def get_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df


def prepare_data(df):
    """Prepare data with all indicators"""

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(ATR_PERIOD).mean()

    # ATR Percentile
    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # SMAs
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # Original Regime (baseline)
    conditions_orig = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > ORIG_SMA_SLOPE_THRESHOLD),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -ORIG_SMA_SLOPE_THRESHOLD),
    ]
    df['regime_original'] = np.select(conditions_orig, ['BULL', 'BEAR'], default='SIDEWAYS')

    # Improved Regime - Stricter slope
    conditions_strict = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > IMPROVED_SMA_SLOPE_THRESHOLD),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -IMPROVED_SMA_SLOPE_THRESHOLD),
    ]
    df['regime_strict'] = np.select(conditions_strict, ['BULL', 'BEAR'], default='SIDEWAYS')

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_STD * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - BB_STD * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

    # BB Width Percentile (squeeze detection)
    def bb_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['bb_width_pct'] = df['bb_width'].rolling(100).apply(bb_pct_func, raw=True)
    df['bb_squeeze'] = df['bb_width_pct'] < BB_SQUEEZE_PERCENTILE

    # Price Range (High-Low over lookback)
    df['range_high'] = df['high'].rolling(RANGE_LOOKBACK).max()
    df['range_low'] = df['low'].rolling(RANGE_LOOKBACK).min()
    df['range_size'] = (df['range_high'] - df['range_low']) / df['close'] * 100

    # Range percentile
    def range_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['range_pct'] = df['range_size'].rolling(100).apply(range_pct_func, raw=True)
    df['range_compressed'] = df['range_pct'] < RANGE_COMPRESSION_THRESHOLD

    # Combined strict SIDEWAYS (multiple conditions)
    df['sideways_strict'] = (
        (df['regime_strict'] == 'SIDEWAYS') &
        (abs(df['sma_slope']) < IMPROVED_SMA_SLOPE_THRESHOLD)
    )

    # Ultra-strict SIDEWAYS (all conditions)
    df['sideways_ultra'] = (
        (df['regime_strict'] == 'SIDEWAYS') &
        (abs(df['sma_slope']) < IMPROVED_SMA_SLOPE_THRESHOLD) &
        (df['bb_squeeze'] | df['range_compressed'])  # At least one compression signal
    )

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)


def run_backtest(df, sideways_mode='original', target_months=None):
    """
    Run backtest with different SIDEWAYS detection modes

    sideways_mode:
        - 'original': Original SIDEWAYS filter (slope < 0.5)
        - 'strict': Stricter slope threshold (< 0.3)
        - 'bb_squeeze': Original + BB squeeze required
        - 'range_compress': Original + Range compression required
        - 'ultra': Multiple conditions required
        - 'none': No SIDEWAYS filter (allow all)
    """
    balance = INITIAL_BALANCE
    position = None
    consecutive_losses = 0
    trades = []
    filtered_sideways = 0

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        month_str = current_time.strftime('%Y-%m')

        if target_months and month_str not in target_months:
            continue

        if row['weekday'] >= 5:
            continue

        # Position management
        if position:
            exit_reason = None
            exit_price = None
            pnl = 0
            bars_held = i - position['entry_idx']

            if bars_held >= MAX_HOLDING_HOURS:
                exit_price = row['close']
                pnl = (exit_price - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - exit_price) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL HIT'
                    elif row['high'] >= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP HIT'
                else:
                    if row['high'] >= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL HIT'
                    elif row['low'] <= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP HIT'

            if exit_reason:
                balance += pnl
                if pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                pips = (exit_price - position['entry']) / 0.0001 if position['dir'] == 1 else (position['entry'] - exit_price) / 0.0001

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'BUY' if position['dir'] == 1 else 'SELL',
                    'pips': pips,
                    'pnl': pnl,
                    'result': 'WIN' if pnl > 0 else 'LOSS',
                    'exit_reason': exit_reason,
                    'sma_slope': position.get('sma_slope', 0),
                    'bb_width_pct': position.get('bb_width_pct', 0),
                    'range_pct': position.get('range_pct', 0),
                })

                position = None

        # Entry logic
        if not position:
            hour = row['hour']
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            # SIDEWAYS filter based on mode
            is_sideways = True
            if sideways_mode == 'original':
                is_sideways = row['regime_original'] == 'SIDEWAYS'
            elif sideways_mode == 'strict':
                is_sideways = row['regime_strict'] == 'SIDEWAYS'
            elif sideways_mode == 'bb_squeeze':
                is_sideways = (row['regime_original'] == 'SIDEWAYS') and row['bb_squeeze']
            elif sideways_mode == 'range_compress':
                is_sideways = (row['regime_original'] == 'SIDEWAYS') and row['range_compressed']
            elif sideways_mode == 'ultra':
                is_sideways = row['sideways_ultra']
            elif sideways_mode == 'none':
                is_sideways = True

            if not is_sideways:
                filtered_sideways += 1
                continue

            # Consecutive loss filter
            if USE_CONSEC_LOSS_FILTER and consecutive_losses >= CONSEC_LOSS_LIMIT:
                consecutive_losses = 0
                continue

            # RSI signal
            rsi = row['rsi']
            signal = 0
            if rsi < RSI_OVERSOLD:
                signal = 1
            elif rsi > RSI_OVERBOUGHT:
                signal = -1

            if not signal:
                continue

            # Execute trade
            entry = row['close']
            atr = row['atr'] if row['atr'] > 0 else entry * 0.002

            if atr_pct < 40:
                tp_mult = TP_LOW
            elif atr_pct > 60:
                tp_mult = TP_HIGH
            else:
                tp_mult = TP_MED

            if 12 <= hour < 16:
                tp_mult += 0.35

            if signal == 1:
                sl = entry - atr * SL_MULT
                tp = entry + atr * tp_mult
            else:
                sl = entry + atr * SL_MULT
                tp = entry - atr * tp_mult

            risk = balance * RISK_PER_TRADE
            size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

            position = {
                'dir': signal,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'size': size,
                'entry_idx': i,
                'entry_time': current_time,
                'sma_slope': row['sma_slope'],
                'bb_width_pct': row['bb_width_pct'],
                'range_pct': row['range_pct'],
            }

    return trades, balance, filtered_sideways


def print_results(name, trades, balance, filtered=0):
    """Print results"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    if len(trades) == 0:
        print(f"No trades (filtered: {filtered})")
        return 0, 0, 0

    df = pd.DataFrame(trades)
    wins = len(df[df['result'] == 'WIN'])
    total_pnl = df['pnl'].sum()
    win_rate = wins / len(df) * 100
    sl_rate = len(df[df['exit_reason'] == 'SL HIT']) / len(df) * 100

    print(f"Trades: {len(df)} | Filtered: {filtered}")
    print(f"Wins: {wins} ({win_rate:.1f}%)")
    print(f"SL Hit Rate: {sl_rate:.1f}%")
    print(f"P/L: ${total_pnl:+,.2f}")
    print(f"Return: {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:+.2f}%")

    return len(df), win_rate, total_pnl


def main():
    print("=" * 70)
    print("RSI v3.7 IMPROVED v3 - Better SIDEWAYS Detection")
    print("=" * 70)
    print("\nTesting different SIDEWAYS detection criteria:")
    print("  1. Original: SMA slope < 0.5")
    print("  2. Strict: SMA slope < 0.3")
    print("  3. BB Squeeze: Original + Bollinger Band squeeze")
    print("  4. Range Compress: Original + Price range compressed")
    print("  5. Ultra: Strict + (BB squeeze OR Range compress)")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

        df = get_data(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
        print(f"Loaded {len(df)} H1 bars")

        print("Preparing indicators...")
        df = prepare_data(df)

        losing_months = ['2024-11', '2025-04']

        # ============================================
        # Test all modes on LOSING months
        # ============================================
        print(f"\n{'#'*70}")
        print(f"TESTING ON LOSING MONTHS: {', '.join(losing_months)}")
        print(f"{'#'*70}")

        modes = ['none', 'original', 'strict', 'bb_squeeze', 'range_compress', 'ultra']
        results_losing = {}

        for mode in modes:
            trades, balance, filtered = run_backtest(df, mode, losing_months)
            n, wr, pnl = print_results(f"Mode: {mode.upper()}", trades, balance, filtered)
            results_losing[mode] = {'trades': n, 'win_rate': wr, 'pnl': pnl, 'balance': balance}

        # ============================================
        # Comparison table
        # ============================================
        print("\n" + "#" * 70)
        print("COMPARISON - LOSING MONTHS")
        print("#" * 70)

        print(f"\n{'Mode':<18} {'Trades':>8} {'Win%':>8} {'SL%':>8} {'P/L':>14}")
        print("-" * 60)

        for mode in modes:
            r = results_losing[mode]
            if r['trades'] > 0:
                print(f"{mode.upper():<18} {r['trades']:>8} {r['win_rate']:>7.1f}% {'-':>8} ${r['pnl']:>+12,.2f}")
            else:
                print(f"{mode.upper():<18} {'0':>8} {'-':>8} {'-':>8} {'-':>14}")

        # Find best
        best_mode = max(results_losing, key=lambda x: results_losing[x]['pnl'])
        orig_pnl = results_losing['original']['pnl']
        best_pnl = results_losing[best_mode]['pnl']

        print(f"\n{'='*60}")
        print(f"BEST MODE: {best_mode.upper()}")
        print(f"Improvement: ${best_pnl - orig_pnl:+,.2f} vs original")
        print(f"{'='*60}")

        # ============================================
        # Test on PROFITABLE months for comparison
        # ============================================
        profit_months = ['2024-12', '2025-01']

        print(f"\n{'#'*70}")
        print(f"COMPARISON: PROFITABLE MONTHS ({', '.join(profit_months)})")
        print(f"{'#'*70}")

        results_profit = {}
        for mode in ['original', best_mode]:
            trades, balance, filtered = run_backtest(df, mode, profit_months)
            n, wr, pnl = print_results(f"Mode: {mode.upper()}", trades, balance, filtered)
            results_profit[mode] = {'trades': n, 'win_rate': wr, 'pnl': pnl}

        # ============================================
        # Full backtest with best mode
        # ============================================
        print(f"\n{'#'*70}")
        print(f"FULL BACKTEST (2024-2025)")
        print(f"{'#'*70}")

        for mode in ['original', best_mode]:
            trades, balance, filtered = run_backtest(df, mode, None)
            print_results(f"Mode: {mode.upper()} (Full)", trades, balance, filtered)

        # ============================================
        # Analyze what the filter is catching
        # ============================================
        print(f"\n{'='*70}")
        print("ANALYSIS: What does the best filter catch?")
        print(f"{'='*70}")

        # Run without filter on losing months
        trades_none, _, _ = run_backtest(df, 'none', losing_months)
        trades_best, _, _ = run_backtest(df, best_mode, losing_months)

        if trades_none and trades_best:
            none_df = pd.DataFrame(trades_none)
            best_df = pd.DataFrame(trades_best)

            none_times = set(none_df['entry_time'].astype(str))
            best_times = set(best_df['entry_time'].astype(str))

            # Trades filtered out
            filtered_out = none_df[~none_df['entry_time'].astype(str).isin(best_times)]

            if len(filtered_out) > 0:
                f_wins = len(filtered_out[filtered_out['result'] == 'WIN'])
                f_losses = len(filtered_out[filtered_out['result'] == 'LOSS'])
                f_pnl = filtered_out['pnl'].sum()

                print(f"\nTrades filtered out by {best_mode.upper()}:")
                print(f"  Total: {len(filtered_out)}")
                print(f"  Wins filtered: {f_wins}")
                print(f"  Losses filtered: {f_losses}")
                print(f"  P/L prevented: ${f_pnl:+,.2f}")

                if f_pnl < 0:
                    print(f"\n  >> SUCCESS: Filter prevented ${abs(f_pnl):,.2f} in losses!")

                # Analyze characteristics of filtered trades
                print(f"\nCharacteristics of filtered trades:")
                print(f"  Avg SMA slope: {filtered_out['sma_slope'].mean():.3f}")
                print(f"  Avg BB width %: {filtered_out['bb_width_pct'].mean():.1f}")
                print(f"  Avg Range %: {filtered_out['range_pct'].mean():.1f}")

                # Compare with kept trades
                kept = none_df[none_df['entry_time'].astype(str).isin(best_times)]
                if len(kept) > 0:
                    print(f"\nCharacteristics of kept trades:")
                    print(f"  Avg SMA slope: {kept['sma_slope'].mean():.3f}")
                    print(f"  Avg BB width %: {kept['bb_width_pct'].mean():.1f}")
                    print(f"  Avg Range %: {kept['range_pct'].mean():.1f}")

    finally:
        mt5.shutdown()
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
