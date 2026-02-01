"""
RSI v3.7 HYBRID FILTER - Best of Both Worlds
=============================================
Combines Original + BB_SQUEEZE adaptively

Strategy:
- Normal conditions: Use ORIGINAL filter (more trades, more profit)
- Trending signs detected: Add BB_SQUEEZE requirement (avoid losses)

Trending Signs:
- ADX > threshold
- OR SMA slope > threshold
- OR price momentum strong
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
# HYBRID FILTER PARAMETERS
# ============================================

# Original SIDEWAYS threshold
SMA_SLOPE_THRESHOLD = 0.5

# BB Squeeze settings
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_PERCENTILE = 30

# ADX settings
ADX_PERIOD = 14

# Hybrid trigger thresholds (when to require BB_SQUEEZE)
HYBRID_ADX_TRIGGER = 20          # ADX > 20 = potential trend
HYBRID_SLOPE_TRIGGER = 0.35      # abs(slope) > 0.35 = potential trend
HYBRID_MOMENTUM_TRIGGER = 0.3    # price change % in 10 bars


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


def calculate_adx(df, period=14):
    """Calculate ADX"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(period).mean()

    return adx


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

    # SMAs and slope
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # ADX
    df['adx'] = calculate_adx(df, ADX_PERIOD)

    # Momentum (price change over 10 bars)
    df['momentum'] = (df['close'] / df['close'].shift(10) - 1) * 100

    # Original Regime
    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > SMA_SLOPE_THRESHOLD),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -SMA_SLOPE_THRESHOLD),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_STD * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - BB_STD * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

    # BB Width Percentile
    def bb_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['bb_width_pct'] = df['bb_width'].rolling(100).apply(bb_pct_func, raw=True)
    df['bb_squeeze'] = df['bb_width_pct'] < BB_SQUEEZE_PERCENTILE

    # Trending signs detection
    df['trending_adx'] = df['adx'] > HYBRID_ADX_TRIGGER
    df['trending_slope'] = abs(df['sma_slope']) > HYBRID_SLOPE_TRIGGER
    df['trending_momentum'] = abs(df['momentum']) > HYBRID_MOMENTUM_TRIGGER

    # Any trending sign
    df['has_trending_sign'] = df['trending_adx'] | df['trending_slope'] | df['trending_momentum']

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)


def run_backtest(df, filter_mode='hybrid', target_months=None):
    """
    Run backtest with different filter modes

    filter_mode:
        - 'original': Original SIDEWAYS filter only
        - 'bb_squeeze': Always require BB squeeze
        - 'hybrid': Original + BB squeeze when trending signs detected
        - 'hybrid_v2': Stricter hybrid (any 2 trending signs)
    """
    balance = INITIAL_BALANCE
    position = None
    consecutive_losses = 0
    trades = []

    stats = {
        'signals': 0,
        'filtered_regime': 0,
        'filtered_bb': 0,
        'filtered_hybrid': 0,
        'hybrid_triggered': 0,
    }

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
                    'hybrid_active': position.get('hybrid_active', False),
                    'adx': position.get('adx', 0),
                    'slope': position.get('slope', 0),
                    'bb_squeeze': position.get('bb_squeeze', False),
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

            # RSI signal check first
            rsi = row['rsi']
            signal = 0
            if rsi < RSI_OVERSOLD:
                signal = 1
            elif rsi > RSI_OVERBOUGHT:
                signal = -1

            if not signal:
                continue

            stats['signals'] += 1

            # Apply filters based on mode
            skip_trade = False
            hybrid_active = False

            # Step 1: Always check original SIDEWAYS regime
            if row['regime'] != 'SIDEWAYS':
                skip_trade = True
                stats['filtered_regime'] += 1

            # Step 2: Apply additional filter based on mode
            if not skip_trade:
                if filter_mode == 'original':
                    # No additional filter
                    pass

                elif filter_mode == 'bb_squeeze':
                    # Always require BB squeeze
                    if not row['bb_squeeze']:
                        skip_trade = True
                        stats['filtered_bb'] += 1

                elif filter_mode == 'hybrid':
                    # Require BB squeeze only when trending signs detected
                    if row['has_trending_sign']:
                        stats['hybrid_triggered'] += 1
                        hybrid_active = True
                        if not row['bb_squeeze']:
                            skip_trade = True
                            stats['filtered_hybrid'] += 1

                elif filter_mode == 'hybrid_v2':
                    # Require BB squeeze if ANY 2 trending signs
                    trending_count = sum([
                        row['trending_adx'],
                        row['trending_slope'],
                        row['trending_momentum']
                    ])
                    if trending_count >= 2:
                        stats['hybrid_triggered'] += 1
                        hybrid_active = True
                        if not row['bb_squeeze']:
                            skip_trade = True
                            stats['filtered_hybrid'] += 1

                elif filter_mode == 'hybrid_adx':
                    # Require BB squeeze only when ADX > threshold
                    if row['trending_adx']:
                        stats['hybrid_triggered'] += 1
                        hybrid_active = True
                        if not row['bb_squeeze']:
                            skip_trade = True
                            stats['filtered_hybrid'] += 1

                elif filter_mode == 'hybrid_slope':
                    # Require BB squeeze only when slope > threshold
                    if row['trending_slope']:
                        stats['hybrid_triggered'] += 1
                        hybrid_active = True
                        if not row['bb_squeeze']:
                            skip_trade = True
                            stats['filtered_hybrid'] += 1

            # Step 3: Consecutive loss filter
            if not skip_trade and USE_CONSEC_LOSS_FILTER:
                if consecutive_losses >= CONSEC_LOSS_LIMIT:
                    consecutive_losses = 0
                    skip_trade = True

            if skip_trade:
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
                'hybrid_active': hybrid_active,
                'adx': row['adx'],
                'slope': row['sma_slope'],
                'bb_squeeze': row['bb_squeeze'],
            }

    return trades, balance, stats


def print_results(name, trades, balance, stats=None):
    """Print results"""
    if len(trades) == 0:
        print(f"\n{name}: No trades")
        return 0, 0, 0

    df = pd.DataFrame(trades)
    wins = len(df[df['result'] == 'WIN'])
    total_pnl = df['pnl'].sum()
    win_rate = wins / len(df) * 100
    sl_rate = len(df[df['exit_reason'] == 'SL HIT']) / len(df) * 100

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Trades: {len(df)} | Wins: {wins} ({win_rate:.1f}%)")
    print(f"SL Hit Rate: {sl_rate:.1f}%")
    print(f"P/L: ${total_pnl:+,.2f}")
    print(f"Return: {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:+.2f}%")

    if stats:
        print(f"\nFilter Stats:")
        print(f"  Signals: {stats.get('signals', 0)}")
        print(f"  Filtered (regime): {stats.get('filtered_regime', 0)}")
        print(f"  Filtered (BB): {stats.get('filtered_bb', 0)}")
        print(f"  Hybrid triggered: {stats.get('hybrid_triggered', 0)}")
        print(f"  Filtered (hybrid): {stats.get('filtered_hybrid', 0)}")

    return len(df), win_rate, total_pnl


def main():
    print("=" * 70)
    print("RSI v3.7 HYBRID FILTER")
    print("=" * 70)
    print("\nHybrid Strategy:")
    print("  - Normal: Use ORIGINAL filter (more opportunities)")
    print("  - Trending detected: Add BB_SQUEEZE (avoid losses)")
    print("\nTrending Detection Triggers:")
    print(f"  - ADX > {HYBRID_ADX_TRIGGER}")
    print(f"  - SMA Slope > {HYBRID_SLOPE_TRIGGER}")
    print(f"  - Momentum > {HYBRID_MOMENTUM_TRIGGER}%")
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
        profit_months = ['2024-12', '2025-01']

        modes = ['original', 'bb_squeeze', 'hybrid', 'hybrid_v2', 'hybrid_adx', 'hybrid_slope']

        # ============================================
        # Test on LOSING months
        # ============================================
        print(f"\n{'#'*70}")
        print(f"LOSING MONTHS: {', '.join(losing_months)}")
        print(f"{'#'*70}")

        results_losing = {}
        for mode in modes:
            trades, balance, stats = run_backtest(df, mode, losing_months)
            n, wr, pnl = print_results(f"Mode: {mode.upper()}", trades, balance, stats)
            results_losing[mode] = {'trades': n, 'win_rate': wr, 'pnl': pnl}

        # ============================================
        # Test on PROFITABLE months
        # ============================================
        print(f"\n{'#'*70}")
        print(f"PROFITABLE MONTHS: {', '.join(profit_months)}")
        print(f"{'#'*70}")

        results_profit = {}
        for mode in modes:
            trades, balance, stats = run_backtest(df, mode, profit_months)
            n, wr, pnl = print_results(f"Mode: {mode.upper()}", trades, balance, stats)
            results_profit[mode] = {'trades': n, 'win_rate': wr, 'pnl': pnl}

        # ============================================
        # FULL BACKTEST
        # ============================================
        print(f"\n{'#'*70}")
        print(f"FULL BACKTEST (2024-2025)")
        print(f"{'#'*70}")

        results_full = {}
        for mode in modes:
            trades, balance, stats = run_backtest(df, mode, None)
            n, wr, pnl = print_results(f"Mode: {mode.upper()}", trades, balance, stats)
            results_full[mode] = {'trades': n, 'win_rate': wr, 'pnl': pnl, 'balance': balance}

        # ============================================
        # COMPARISON SUMMARY
        # ============================================
        print("\n" + "#" * 70)
        print("COMPARISON SUMMARY")
        print("#" * 70)

        print(f"\n{'Mode':<15} {'Losing':>12} {'Profit':>12} {'Full':>14} {'Net Chg':>12}")
        print("-" * 70)

        for mode in modes:
            l_pnl = results_losing[mode]['pnl']
            p_pnl = results_profit[mode]['pnl']
            f_pnl = results_full[mode]['pnl']
            orig_full = results_full['original']['pnl']
            net_change = f_pnl - orig_full

            marker = ""
            if mode != 'original':
                if l_pnl > results_losing['original']['pnl'] and f_pnl > orig_full * 0.8:
                    marker = " << GOOD"
                elif l_pnl > results_losing['original']['pnl']:
                    marker = " (better losing)"

            print(f"{mode.upper():<15} ${l_pnl:>+10,.0f} ${p_pnl:>+10,.0f} ${f_pnl:>+12,.0f} ${net_change:>+10,.0f}{marker}")

        # ============================================
        # Find best hybrid
        # ============================================
        print("\n" + "=" * 70)
        print("FINDING OPTIMAL HYBRID")
        print("=" * 70)

        # Score: Maximize (losing month improvement) while minimizing (full backtest loss)
        scores = {}
        orig_losing = results_losing['original']['pnl']
        orig_full = results_full['original']['pnl']

        for mode in modes:
            if mode == 'original':
                continue

            losing_improvement = results_losing[mode]['pnl'] - orig_losing
            full_retention = results_full[mode]['pnl'] / orig_full * 100

            # Score = losing improvement * full retention percentage
            # We want high losing improvement AND high retention
            score = losing_improvement * (full_retention / 100)

            scores[mode] = {
                'losing_improvement': losing_improvement,
                'full_retention': full_retention,
                'score': score,
            }

            print(f"\n{mode.upper()}:")
            print(f"  Losing month improvement: ${losing_improvement:+,.2f}")
            print(f"  Full backtest retention: {full_retention:.1f}%")
            print(f"  Score: {score:,.0f}")

        best_mode = max(scores, key=lambda x: scores[x]['score'])

        print(f"\n{'='*70}")
        print(f"BEST HYBRID MODE: {best_mode.upper()}")
        print(f"{'='*70}")

        print(f"\nLosing months: ${results_losing['original']['pnl']:+,.2f} -> ${results_losing[best_mode]['pnl']:+,.2f}")
        print(f"Full backtest: ${results_full['original']['pnl']:+,.2f} -> ${results_full[best_mode]['pnl']:+,.2f}")
        print(f"Retention: {scores[best_mode]['full_retention']:.1f}%")

        # ============================================
        # Monthly breakdown with best mode
        # ============================================
        print("\n" + "=" * 70)
        print(f"MONTHLY BREAKDOWN: ORIGINAL vs {best_mode.upper()}")
        print("=" * 70)

        all_months = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06',
                      '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12',
                      '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
                      '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12']

        print(f"\n{'Month':<10} {'Orig Trades':>12} {'Orig P/L':>12} {'Hybrid Trades':>14} {'Hybrid P/L':>12} {'Diff':>10}")
        print("-" * 75)

        total_orig = 0
        total_hybrid = 0

        for month in all_months:
            trades_orig, bal_orig, _ = run_backtest(df, 'original', [month])
            trades_hybrid, bal_hybrid, _ = run_backtest(df, best_mode, [month])

            if len(trades_orig) > 0 or len(trades_hybrid) > 0:
                pnl_orig = sum(t['pnl'] for t in trades_orig) if trades_orig else 0
                pnl_hybrid = sum(t['pnl'] for t in trades_hybrid) if trades_hybrid else 0
                diff = pnl_hybrid - pnl_orig

                total_orig += pnl_orig
                total_hybrid += pnl_hybrid

                marker = ""
                if month in losing_months:
                    marker = " [LOSS MONTH]"
                elif month in profit_months:
                    marker = " [PROFIT MONTH]"

                print(f"{month:<10} {len(trades_orig):>12} ${pnl_orig:>+10,.0f} {len(trades_hybrid):>14} ${pnl_hybrid:>+10,.0f} ${diff:>+8,.0f}{marker}")

        print("-" * 75)
        print(f"{'TOTAL':<10} {'-':>12} ${total_orig:>+10,.0f} {'-':>14} ${total_hybrid:>+10,.0f} ${total_hybrid-total_orig:>+8,.0f}")

    finally:
        mt5.shutdown()
        print("\n" + "=" * 70)
        print("Backtest complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
