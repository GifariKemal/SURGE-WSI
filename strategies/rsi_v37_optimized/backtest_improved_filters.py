"""
RSI v3.7 IMPROVED - Backtest with New Filters
==============================================
Testing on Losing Months: Nov 2024 & Apr 2025

Improvements:
1. ADX Trend Strength Filter - Avoid trading when trend is strong
2. Multi-Timeframe Regime Confirmation - H1 + H4 alignment
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

# Strategy Parameters (Original)
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

# Original Filters
USE_REGIME_FILTER = True
ALLOWED_REGIMES = ['SIDEWAYS']
USE_CONSEC_LOSS_FILTER = True
CONSEC_LOSS_LIMIT = 3

# ============================================
# NEW FILTERS
# ============================================

# 1. ADX Trend Strength Filter
USE_ADX_FILTER = True
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # Below this = weak trend (good for mean reversion)

# 2. Multi-Timeframe Regime Confirmation
USE_MTF_CONFIRMATION = True
MTF_ALIGNMENT_REQUIRED = True  # H1 and H4 must both be SIDEWAYS


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
    """Calculate ADX (Average Directional Index)"""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed TR and DM
    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


def calculate_regime(df):
    """Calculate market regime based on SMA crossover and slope"""
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')
    return df


def prepare_h1_data(df_h1, df_h4=None):
    """Prepare H1 data with all indicators"""

    # RSI
    delta = df_h1['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df_h1['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df_h1['high'] - df_h1['low'],
                    np.maximum(abs(df_h1['high'] - df_h1['close'].shift(1)),
                              abs(df_h1['low'] - df_h1['close'].shift(1))))
    df_h1['atr'] = pd.Series(tr, index=df_h1.index).rolling(ATR_PERIOD).mean()

    # ATR Percentile
    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df_h1['atr_pct'] = df_h1['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # H1 Regime
    df_h1 = calculate_regime(df_h1)

    # ADX (New Filter #1)
    df_h1['adx'], df_h1['plus_di'], df_h1['minus_di'] = calculate_adx(df_h1, ADX_PERIOD)

    # Multi-Timeframe Regime (New Filter #2)
    if df_h4 is not None and USE_MTF_CONFIRMATION:
        df_h4 = calculate_regime(df_h4)
        df_h4['h4_regime'] = df_h4['regime']

        # Map H4 regime to H1 timeframe
        df_h1['h4_regime'] = 'UNKNOWN'
        for idx in df_h1.index:
            # Find the most recent H4 bar before or at this H1 time
            h4_before = df_h4[df_h4.index <= idx]
            if len(h4_before) > 0:
                df_h1.loc[idx, 'h4_regime'] = h4_before['h4_regime'].iloc[-1]
    else:
        df_h1['h4_regime'] = df_h1['regime']  # Fallback to H1 regime

    df_h1['hour'] = df_h1.index.hour
    df_h1['weekday'] = df_h1.index.weekday
    df_h1['day_name'] = df_h1.index.day_name()

    return df_h1.ffill().fillna(0)


def run_backtest(df, use_new_filters=True, target_months=None):
    """
    Run backtest with optional new filters

    Args:
        df: DataFrame with H1 data and indicators
        use_new_filters: If True, apply ADX and MTF filters
        target_months: List of months to include (e.g., ['2024-11', '2025-04'])
    """
    balance = INITIAL_BALANCE
    position = None
    consecutive_losses = 0
    trades = []

    filtered_out = {
        'adx': 0,
        'mtf': 0,
        'regime': 0,
        'consec': 0,
        'atr_pct': 0,
        'hour': 0,
    }

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        month_str = current_time.strftime('%Y-%m')

        # Filter by target months if specified
        if target_months and month_str not in target_months:
            continue

        weekday = row['weekday']
        hour = row['hour']

        if weekday >= 5:
            continue

        # Position management (same as original)
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
                result = 'WIN' if pnl > 0 else 'LOSS'

                if pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                if position['dir'] == 1:
                    pips = (exit_price - position['entry']) / 0.0001
                else:
                    pips = (position['entry'] - exit_price) / 0.0001

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'BUY' if position['dir'] == 1 else 'SELL',
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pips': pips,
                    'pnl': pnl,
                    'result': result,
                    'exit_reason': exit_reason,
                    'adx': position.get('adx', 0),
                    'h1_regime': position.get('h1_regime', ''),
                    'h4_regime': position.get('h4_regime', ''),
                })

                position = None

        # Entry logic
        if not position:
            # Hour filter
            if hour < 7 or hour >= 22 or hour == 12:
                filtered_out['hour'] += 1
                continue

            # ATR% filter
            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                filtered_out['atr_pct'] += 1
                continue

            skip_trade = False
            skip_reason = ''

            # Original regime filter
            regime = row['regime']
            if USE_REGIME_FILTER and regime not in ALLOWED_REGIMES:
                skip_trade = True
                skip_reason = 'regime'
                filtered_out['regime'] += 1

            # NEW FILTER #1: ADX Trend Strength
            if not skip_trade and use_new_filters and USE_ADX_FILTER:
                adx_value = row['adx']
                if adx_value > ADX_THRESHOLD:
                    skip_trade = True
                    skip_reason = 'adx'
                    filtered_out['adx'] += 1

            # NEW FILTER #2: Multi-Timeframe Confirmation
            if not skip_trade and use_new_filters and USE_MTF_CONFIRMATION:
                h4_regime = row['h4_regime']
                if MTF_ALIGNMENT_REQUIRED and h4_regime not in ALLOWED_REGIMES:
                    skip_trade = True
                    skip_reason = 'mtf'
                    filtered_out['mtf'] += 1

            # Consecutive loss filter
            if not skip_trade and USE_CONSEC_LOSS_FILTER:
                if consecutive_losses >= CONSEC_LOSS_LIMIT:
                    skip_trade = True
                    skip_reason = 'consec'
                    consecutive_losses = 0
                    filtered_out['consec'] += 1

            if skip_trade:
                continue

            # RSI signal
            rsi = row['rsi']
            signal = 0

            if rsi < RSI_OVERSOLD:
                signal = 1
            elif rsi > RSI_OVERBOUGHT:
                signal = -1

            if signal:
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
                    'adx': row['adx'],
                    'h1_regime': regime,
                    'h4_regime': row['h4_regime'],
                }

    return trades, balance, filtered_out


def print_results(name, trades, final_balance, filtered_out=None):
    """Print backtest results"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    if len(trades) == 0:
        print("No trades executed")
        return

    df = pd.DataFrame(trades)
    wins = len(df[df['result'] == 'WIN'])
    losses = len(df[df['result'] == 'LOSS'])
    total_pnl = df['pnl'].sum()
    win_rate = wins / len(df) * 100

    print(f"Trades: {len(df)}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P/L: ${total_pnl:+,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Return: {(final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:+.2f}%")

    # By direction
    buys = df[df['direction'] == 'BUY']
    sells = df[df['direction'] == 'SELL']
    print(f"\nBY DIRECTION:")
    print(f"  BUY:  {len(buys)} trades, ${buys['pnl'].sum():+,.2f}")
    print(f"  SELL: {len(sells)} trades, ${sells['pnl'].sum():+,.2f}")

    # By exit reason
    print(f"\nBY EXIT REASON:")
    for reason in df['exit_reason'].unique():
        subset = df[df['exit_reason'] == reason]
        print(f"  {reason}: {len(subset)} trades, ${subset['pnl'].sum():+,.2f}")

    # SL Hit rate
    sl_hits = len(df[df['exit_reason'] == 'SL HIT'])
    tp_hits = len(df[df['exit_reason'] == 'TP HIT'])
    print(f"\nSL Hit Rate: {sl_hits/len(df)*100:.1f}%")
    print(f"TP Hit Rate: {tp_hits/len(df)*100:.1f}%")

    if filtered_out:
        print(f"\nFILTERED OUT SIGNALS:")
        for reason, count in filtered_out.items():
            if count > 0:
                print(f"  {reason}: {count}")


def main():
    print("=" * 70)
    print("RSI v3.7 IMPROVED - Testing New Filters")
    print("=" * 70)
    print("\nNew Filters:")
    print(f"  1. ADX Filter: Trade only when ADX < {ADX_THRESHOLD} (weak trend)")
    print(f"  2. MTF Confirmation: H1 + H4 both must be SIDEWAYS")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        # Fetch data
        print("\nFetching H1 data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

        df_h1 = get_data(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
        print(f"H1: {len(df_h1)} bars")

        print("Fetching H4 data...")
        df_h4 = get_data(SYMBOL, mt5.TIMEFRAME_H4, start_date, end_date)
        print(f"H4: {len(df_h4)} bars")

        print("\nPreparing indicators...")
        df_h1 = prepare_h1_data(df_h1, df_h4)

        # Target months (losing months)
        losing_months = ['2024-11', '2025-04']

        print(f"\n{'#'*70}")
        print(f"TESTING ON LOSING MONTHS: {', '.join(losing_months)}")
        print(f"{'#'*70}")

        # ============================================
        # TEST 1: Original filters (baseline)
        # ============================================
        print("\n" + "=" * 70)
        print("BASELINE: Original Filters Only")
        print("=" * 70)

        trades_orig, balance_orig, filtered_orig = run_backtest(
            df_h1,
            use_new_filters=False,
            target_months=losing_months
        )
        print_results("ORIGINAL RSI v3.7", trades_orig, balance_orig, filtered_orig)

        # ============================================
        # TEST 2: With ADX filter only
        # ============================================
        print("\n" + "=" * 70)
        print("TEST: Original + ADX Filter (ADX < 25)")
        print("=" * 70)

        # Temporarily disable MTF
        global USE_MTF_CONFIRMATION
        USE_MTF_CONFIRMATION = False

        trades_adx, balance_adx, filtered_adx = run_backtest(
            df_h1,
            use_new_filters=True,
            target_months=losing_months
        )
        print_results("WITH ADX FILTER", trades_adx, balance_adx, filtered_adx)

        USE_MTF_CONFIRMATION = True

        # ============================================
        # TEST 3: With MTF filter only
        # ============================================
        print("\n" + "=" * 70)
        print("TEST: Original + MTF Confirmation (H1+H4 SIDEWAYS)")
        print("=" * 70)

        # Temporarily disable ADX
        global USE_ADX_FILTER
        USE_ADX_FILTER = False

        trades_mtf, balance_mtf, filtered_mtf = run_backtest(
            df_h1,
            use_new_filters=True,
            target_months=losing_months
        )
        print_results("WITH MTF FILTER", trades_mtf, balance_mtf, filtered_mtf)

        USE_ADX_FILTER = True

        # ============================================
        # TEST 4: Both new filters combined
        # ============================================
        print("\n" + "=" * 70)
        print("TEST: Original + ADX + MTF (FULL IMPROVEMENT)")
        print("=" * 70)

        trades_both, balance_both, filtered_both = run_backtest(
            df_h1,
            use_new_filters=True,
            target_months=losing_months
        )
        print_results("WITH BOTH FILTERS", trades_both, balance_both, filtered_both)

        # ============================================
        # COMPARISON SUMMARY
        # ============================================
        print("\n" + "#" * 70)
        print("COMPARISON SUMMARY - LOSING MONTHS")
        print("#" * 70)

        results = [
            ("Original", trades_orig, balance_orig),
            ("+ ADX Filter", trades_adx, balance_adx),
            ("+ MTF Filter", trades_mtf, balance_mtf),
            ("+ ADX + MTF", trades_both, balance_both),
        ]

        print(f"\n{'Configuration':<20} {'Trades':>8} {'Wins':>6} {'Win%':>8} {'P/L':>12} {'Return':>10}")
        print("-" * 70)

        for name, trades, balance in results:
            if len(trades) > 0:
                df = pd.DataFrame(trades)
                wins = len(df[df['result'] == 'WIN'])
                win_rate = wins / len(df) * 100
                pnl = df['pnl'].sum()
                ret = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
                print(f"{name:<20} {len(df):>8} {wins:>6} {win_rate:>7.1f}% ${pnl:>+10,.2f} {ret:>+9.2f}%")
            else:
                print(f"{name:<20} {'0':>8} {'-':>6} {'-':>8} {'-':>12} {'-':>10}")

        # Calculate improvement
        if len(trades_orig) > 0 and len(trades_both) > 0:
            orig_pnl = pd.DataFrame(trades_orig)['pnl'].sum()
            both_pnl = pd.DataFrame(trades_both)['pnl'].sum()
            improvement = both_pnl - orig_pnl

            print(f"\n{'='*70}")
            print(f"IMPROVEMENT: ${improvement:+,.2f}")
            if orig_pnl < 0 and both_pnl > orig_pnl:
                reduction = (1 - both_pnl / orig_pnl) * 100 if orig_pnl != 0 else 0
                print(f"Loss reduction: {reduction:.1f}%")
            print(f"{'='*70}")

        # ============================================
        # DETAILED ANALYSIS OF FILTERED TRADES
        # ============================================
        print("\n" + "=" * 70)
        print("ANALYSIS: What did the new filters prevent?")
        print("=" * 70)

        # Get trades that were in original but not in improved
        if len(trades_orig) > 0:
            orig_df = pd.DataFrame(trades_orig)
            orig_times = set(orig_df['entry_time'].astype(str))

            if len(trades_both) > 0:
                both_df = pd.DataFrame(trades_both)
                both_times = set(both_df['entry_time'].astype(str))

                # Trades that were filtered out
                filtered_trades = orig_df[~orig_df['entry_time'].astype(str).isin(both_times)]

                if len(filtered_trades) > 0:
                    filtered_wins = len(filtered_trades[filtered_trades['result'] == 'WIN'])
                    filtered_losses = len(filtered_trades[filtered_trades['result'] == 'LOSS'])
                    filtered_pnl = filtered_trades['pnl'].sum()

                    print(f"\nTrades filtered out by new filters:")
                    print(f"  Total: {len(filtered_trades)}")
                    print(f"  Wins filtered: {filtered_wins}")
                    print(f"  Losses filtered: {filtered_losses}")
                    print(f"  P/L prevented: ${filtered_pnl:+,.2f}")

                    if filtered_pnl < 0:
                        print(f"\n  >> GOOD: Filters prevented ${abs(filtered_pnl):,.2f} in losses!")
                    else:
                        print(f"\n  >> NOTE: Filters also prevented ${filtered_pnl:,.2f} in profits")
                        print(f"     But net effect is positive if overall P/L improved")

        # ============================================
        # ADX Distribution Analysis
        # ============================================
        print("\n" + "=" * 70)
        print("ADX ANALYSIS")
        print("=" * 70)

        if len(trades_orig) > 0:
            orig_df = pd.DataFrame(trades_orig)
            low_adx = orig_df[orig_df['adx'] < ADX_THRESHOLD]
            high_adx = orig_df[orig_df['adx'] >= ADX_THRESHOLD]

            print(f"\nOriginal trades by ADX:")
            print(f"  ADX < {ADX_THRESHOLD} (weak trend): {len(low_adx)} trades, ${low_adx['pnl'].sum():+,.2f}")
            print(f"  ADX >= {ADX_THRESHOLD} (strong trend): {len(high_adx)} trades, ${high_adx['pnl'].sum():+,.2f}")

            if len(high_adx) > 0 and high_adx['pnl'].sum() < 0:
                print(f"\n  >> High ADX trades lost money - ADX filter is effective!")

        # ============================================
        # H4 Regime Analysis
        # ============================================
        print("\n" + "=" * 70)
        print("H4 REGIME ANALYSIS")
        print("=" * 70)

        if len(trades_orig) > 0:
            orig_df = pd.DataFrame(trades_orig)

            for regime in orig_df['h4_regime'].unique():
                subset = orig_df[orig_df['h4_regime'] == regime]
                wins = len(subset[subset['result'] == 'WIN'])
                print(f"  H4={regime}: {len(subset)} trades, {wins}W, ${subset['pnl'].sum():+,.2f}")

    finally:
        mt5.shutdown()
        print("\n" + "=" * 70)
        print("Backtest complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
