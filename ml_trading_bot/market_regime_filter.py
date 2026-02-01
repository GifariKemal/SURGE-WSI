"""
Market Regime Filter for RSI v3.7
=================================
Detect Bull/Bear/Sideways and filter trades accordingly.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

MT5_LOGIN = 61045904
MT5_PASSWORD = "iy#K5L7sF"
MT5_SERVER = "FinexBisnisSolusi-Demo"
MT5_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

def connect_mt5():
    if not mt5.initialize(path=MT5_PATH):
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return False
    return True

def get_h1_data(symbol, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def detect_regime(df, method='sma'):
    """
    Detect market regime: BULL, BEAR, SIDEWAYS

    Methods:
    - sma: Based on SMA crossover and slope
    - adx: Based on ADX and DI
    - bb: Based on Bollinger Band width
    - combined: Combination of all
    """

    # Calculate indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # SMA slope (20-period)
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # ADX calculation
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean()

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm_arr = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm_arr = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    smoothed_tr = pd.Series(tr, index=df.index).rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm_arr, index=df.index).rolling(14).mean() / smoothed_tr
    minus_di = 100 * pd.Series(minus_dm_arr, index=df.index).rolling(14).mean() / smoothed_tr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_std'] * 2) / df['bb_mid'] * 100  # Width as percentage

    # Choppiness Index
    atr_sum = df['atr'].rolling(14).sum()
    high_max = df['high'].rolling(14).max()
    low_min = df['low'].rolling(14).min()
    range_14 = high_max - low_min
    df['chop'] = 100 * np.log10(atr_sum / range_14) / np.log10(14)

    # Regime detection based on method
    if method == 'sma':
        # Bull: SMA20 > SMA50, positive slope
        # Bear: SMA20 < SMA50, negative slope
        # Sideways: Small slope, SMA close together
        conditions = [
            (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
            (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
        ]
        choices = ['BULL', 'BEAR']
        df['regime'] = np.select(conditions, choices, default='SIDEWAYS')

    elif method == 'adx':
        # Bull: ADX > 25, +DI > -DI
        # Bear: ADX > 25, -DI > +DI
        # Sideways: ADX < 25
        conditions = [
            (df['adx'] > 25) & (df['plus_di'] > df['minus_di']),
            (df['adx'] > 25) & (df['minus_di'] > df['plus_di']),
        ]
        choices = ['BULL', 'BEAR']
        df['regime'] = np.select(conditions, choices, default='SIDEWAYS')

    elif method == 'bb':
        # Trending: Wide BB (> 2%)
        # Sideways: Narrow BB (< 1.5%)
        df['trending'] = df['bb_width'] > 2.0
        conditions = [
            (df['trending']) & (df['close'] > df['bb_mid']),
            (df['trending']) & (df['close'] < df['bb_mid']),
        ]
        choices = ['BULL', 'BEAR']
        df['regime'] = np.select(conditions, choices, default='SIDEWAYS')

    elif method == 'combined':
        # Score-based: combine multiple signals
        bull_score = (
            (df['sma_20'] > df['sma_50']).astype(int) +
            (df['sma_slope'] > 0.3).astype(int) +
            (df['plus_di'] > df['minus_di']).astype(int) +
            (df['close'] > df['sma_200']).astype(int)
        )
        bear_score = (
            (df['sma_20'] < df['sma_50']).astype(int) +
            (df['sma_slope'] < -0.3).astype(int) +
            (df['minus_di'] > df['plus_di']).astype(int) +
            (df['close'] < df['sma_200']).astype(int)
        )
        trend_strength = df['adx'] > 20

        conditions = [
            (bull_score >= 3) & trend_strength,
            (bear_score >= 3) & trend_strength,
        ]
        choices = ['BULL', 'BEAR']
        df['regime'] = np.select(conditions, choices, default='SIDEWAYS')

    return df

def run_regime_backtest(df, regime_filter=None, test_start='2024-10-01', test_end='2026-02-01'):
    """
    Run RSI v3.7 with regime filter.
    regime_filter: None (no filter), 'BULL', 'BEAR', 'SIDEWAYS', or list like ['BULL', 'SIDEWAYS']
    """
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MAX_HOLDING = 46
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR percentile
    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0

    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    regime_trades = {'BULL': 0, 'BEAR': 0, 'SIDEWAYS': 0}
    regime_pnl = {'BULL': 0, 'BEAR': 0, 'SIDEWAYS': 0}
    trades_filtered = 0

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        in_test = current_time >= pd.Timestamp(test_start) and current_time < pd.Timestamp(test_end)

        month_str = current_time.strftime('%Y-%m')
        weekday = row['weekday']
        hour = row['hour']
        current_regime = row['regime'] if 'regime' in df.columns else 'UNKNOWN'

        if weekday >= 5:
            continue

        if position:
            exit_reason = None
            pnl = 0

            if (i - position['entry_idx']) >= MAX_HOLDING:
                pnl = (row['close'] - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - row['close']) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL'
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP'
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL'
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP'

            if exit_reason:
                balance += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                if month_str not in monthly_pnl:
                    monthly_pnl[month_str] = 0
                monthly_pnl[month_str] += pnl

                # Track by regime
                entry_regime = position.get('regime', 'UNKNOWN')
                if entry_regime in regime_pnl:
                    regime_pnl[entry_regime] += pnl

                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        if not position and in_test:
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            # Regime filter
            if regime_filter is not None:
                allowed_regimes = regime_filter if isinstance(regime_filter, list) else [regime_filter]
                if current_regime not in allowed_regimes:
                    trades_filtered += 1
                    continue

            rsi = row['rsi']
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                tp_mult = base_tp + 0.35 if 12 <= hour < 16 else base_tp
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i, 'regime': current_regime}

                if current_regime in regime_trades:
                    regime_trades[current_regime] += 1

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - 10000) / 10000 * 100
    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    losing_months = sum(1 for m in monthly_pnl.values() if m < 0)

    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profitable_months': profitable_months,
        'losing_months': losing_months,
        'total_months': len(monthly_pnl),
        'monthly_pnl': monthly_pnl,
        'trades_filtered': trades_filtered,
        'regime_trades': regime_trades,
        'regime_pnl': regime_pnl
    }

def main():
    print("=" * 70)
    print("MARKET REGIME FILTER TEST")
    print("Bull / Bear / Sideways Detection")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data("GBPUSD", start_date, end_date)

        if df is None:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")

        # Test different regime detection methods
        methods = ['sma', 'adx', 'bb', 'combined']

        for method in methods:
            print(f"\n{'='*70}")
            print(f"REGIME METHOD: {method.upper()}")
            print("=" * 70)

            df_test = df.copy()
            df_test = detect_regime(df_test, method=method)

            # Show regime distribution in test period
            test_data = df_test[(df_test.index >= '2024-10-01') & (df_test.index < '2026-02-01')]
            regime_counts = test_data['regime'].value_counts()
            print(f"\nRegime Distribution (Oct 2024 - Jan 2026):")
            for regime, count in regime_counts.items():
                pct = count / len(test_data) * 100
                print(f"  {regime}: {count} bars ({pct:.1f}%)")

            # Baseline (no filter)
            print(f"\n--- Baseline (No Filter) ---")
            baseline = run_regime_backtest(df_test.copy(), regime_filter=None)
            print(f"Return: {baseline['total_return']:+.1f}% | Trades: {baseline['total_trades']} | Loss Months: {baseline['losing_months']}")

            # Test different regime filters
            filter_configs = [
                ('BULL only', ['BULL']),
                ('BEAR only', ['BEAR']),
                ('SIDEWAYS only', ['SIDEWAYS']),
                ('BULL+BEAR', ['BULL', 'BEAR']),
                ('BULL+SIDEWAYS', ['BULL', 'SIDEWAYS']),
                ('BEAR+SIDEWAYS', ['BEAR', 'SIDEWAYS']),
            ]

            print(f"\n--- With Regime Filter ---")
            results = []
            for name, filter_val in filter_configs:
                result = run_regime_backtest(df_test.copy(), regime_filter=filter_val)
                status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
                print(f"{name:<18}: {status:<8} | Ret={result['total_return']:+6.1f}% | "
                      f"Trades={result['total_trades']:>3} | Filtered={result['trades_filtered']}")
                results.append({'name': f'{method}_{name}', 'filter': filter_val, **result})

            # Show which regimes are profitable
            print(f"\n--- Regime Performance (Baseline) ---")
            for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
                trades = baseline['regime_trades'].get(regime, 0)
                pnl = baseline['regime_pnl'].get(regime, 0)
                if trades > 0:
                    print(f"  {regime}: {trades} trades, ${pnl:+,.2f}")

        # Final analysis - find best config
        print("\n" + "=" * 70)
        print("ANALYZING LOSING MONTHS")
        print("=" * 70)

        # Use combined method for analysis
        df_analysis = df.copy()
        df_analysis = detect_regime(df_analysis, method='combined')

        baseline = run_regime_backtest(df_analysis.copy(), regime_filter=None)

        print("\nMonthly P/L with Regime Info:")
        for month, pnl in sorted(baseline['monthly_pnl'].items()):
            # Get dominant regime for month
            month_start = pd.Timestamp(month + '-01')
            month_end = month_start + pd.DateOffset(months=1)
            month_data = df_analysis[(df_analysis.index >= month_start) & (df_analysis.index < month_end)]
            if len(month_data) > 0:
                dominant_regime = month_data['regime'].mode().iloc[0] if len(month_data['regime'].mode()) > 0 else 'UNKNOWN'
                regime_pcts = month_data['regime'].value_counts(normalize=True) * 100
                regime_str = ', '.join([f"{r}:{p:.0f}%" for r, p in regime_pcts.items()])
            else:
                dominant_regime = 'UNKNOWN'
                regime_str = ''

            marker = " <-- LOSS" if pnl < 0 else ""
            print(f"  {month}: ${pnl:+,.2f} [{regime_str}]{marker}")

    finally:
        mt5.shutdown()
        print("\nDone!")

if __name__ == "__main__":
    main()
