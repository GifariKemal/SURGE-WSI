"""
Regime-Based Filter for RSI v3.7
================================
Detect unfavorable market regimes and pause trading.
Uses market conditions from losing months to learn patterns.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

def create_regime_features(df):
    """Create features for regime detection (monthly aggregated)."""
    df['returns'] = df['close'].pct_change()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean()

    # Choppiness
    atr_sum = df['atr'].rolling(14).sum()
    high_max = df['high'].rolling(14).max()
    low_min = df['low'].rolling(14).min()
    range_14 = high_max - low_min
    df['chop'] = 100 * np.log10(atr_sum / range_14) / np.log10(14)

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm_arr = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm_arr = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    plus_dm_s = pd.Series(plus_dm_arr, index=df.index)
    minus_dm_s = pd.Series(minus_dm_arr, index=df.index)
    tr_s = pd.Series(tr, index=df.index)
    smoothed_tr = tr_s.rolling(14).mean()
    plus_di = 100 * plus_dm_s.rolling(14).mean() / smoothed_tr
    minus_di = 100 * minus_dm_s.rolling(14).mean() / smoothed_tr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(14).mean()

    # Trend features
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50'] * 100

    df['volatility'] = df['returns'].rolling(24).std()

    df['month'] = df.index.to_period('M')

    return df

def aggregate_monthly_features(df):
    """Aggregate hourly data into monthly regime features."""
    monthly = df.groupby('month').agg({
        'returns': ['mean', 'std', 'min', 'max'],
        'rsi': ['mean', 'std', 'min', 'max'],
        'atr': ['mean', 'std'],
        'chop': ['mean', 'std'],
        'adx': ['mean', 'std'],
        'trend_strength': ['mean', 'std'],
        'volatility': ['mean', 'std', 'max']
    })

    monthly.columns = ['_'.join(col) for col in monthly.columns]
    monthly = monthly.reset_index()

    # Add RSI signal counts
    df['rsi_oversold'] = (df['rsi'] < 42).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 58).astype(int)

    signal_counts = df.groupby('month').agg({
        'rsi_oversold': 'sum',
        'rsi_overbought': 'sum'
    }).reset_index()

    monthly = monthly.merge(signal_counts, on='month')
    monthly['signal_imbalance'] = abs(monthly['rsi_oversold'] - monthly['rsi_overbought'])

    return monthly

def run_regime_backtest(df, regime_detector, scaler, monthly_features, regime_cols,
                        bad_regime_threshold=0.5, use_regime_filter=True,
                        test_start='2024-10-01', test_end='2026-02-01'):
    """Run RSI v3.7 with regime-based filter."""
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MAX_HOLDING = 46
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0

    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    trades_taken = 0
    trades_filtered = 0
    paused_months = set()

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        in_test_period = current_time >= pd.Timestamp(test_start) and current_time < pd.Timestamp(test_end)

        month_str = current_time.strftime('%Y-%m')
        month_period = current_time.to_period('M')
        weekday = row['weekday'] if 'weekday' in df.columns else current_time.weekday()
        hour = row['hour'] if 'hour' in df.columns else current_time.hour

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
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        if not position and in_test_period:
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            # Regime Filter - check if current month is in bad regime
            if use_regime_filter and regime_detector is not None:
                if month_str in paused_months:
                    trades_filtered += 1
                    continue

                # Get current month's features
                month_data = monthly_features[monthly_features['month'] == month_period]
                if len(month_data) > 0:
                    try:
                        features = month_data[regime_cols].values
                        if not np.isnan(features).any():
                            features_scaled = scaler.transform(features)

                            # Check if bad regime
                            if hasattr(regime_detector, 'predict_proba'):
                                proba = regime_detector.predict_proba(features_scaled)[0]
                                bad_prob = proba[1] if len(proba) > 1 else proba[0]
                                if bad_prob > bad_regime_threshold:
                                    paused_months.add(month_str)
                                    trades_filtered += 1
                                    continue
                            else:
                                pred = regime_detector.predict(features_scaled)[0]
                                if pred == -1:  # Anomaly = bad regime
                                    paused_months.add(month_str)
                                    trades_filtered += 1
                                    continue
                    except Exception:
                        pass

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
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}
                trades_taken += 1

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
        'paused_months': len(paused_months)
    }

def main():
    print("=" * 70)
    print("REGIME-BASED FILTER + RSI v3.7")
    print("Detect bad market regimes and pause trading")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2015, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data("GBPUSD", start_date, end_date)

        if df is None:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")

        print("Creating regime features...")
        df = create_regime_features(df)
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday

        # Create monthly aggregated features
        monthly_features = aggregate_monthly_features(df)
        print(f"Monthly data: {len(monthly_features)} months")

        regime_cols = [
            'returns_mean', 'returns_std', 'returns_min', 'returns_max',
            'rsi_mean', 'rsi_std', 'rsi_min', 'rsi_max',
            'atr_mean', 'atr_std',
            'chop_mean', 'chop_std',
            'adx_mean', 'adx_std',
            'trend_strength_mean', 'trend_strength_std',
            'volatility_mean', 'volatility_std', 'volatility_max',
            'rsi_oversold', 'rsi_overbought', 'signal_imbalance'
        ]

        # Now run RSI v3.7 backtest on historical data to label months
        print("\nLabeling historical months (good/bad)...")

        # Run baseline backtest on full history to get monthly P/L
        baseline_pnl = {}
        # Simple simulation to get monthly returns

        RSI_OS = 42
        RSI_OB = 58
        SL_MULT = 1.5
        MAX_HOLDING = 46

        df_sim = df.copy()
        df_sim['atr_pct'] = df_sim['atr'].rolling(100).apply(
            lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50.0, raw=True
        )
        df_sim = df_sim.ffill().fillna(0)

        balance = 10000.0
        position = None

        for i in range(200, len(df_sim) - 20):
            row = df_sim.iloc[i]
            current_time = df_sim.index[i]
            month_str = current_time.strftime('%Y-%m')
            weekday = current_time.weekday()
            hour = current_time.hour

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
                    if month_str not in baseline_pnl:
                        baseline_pnl[month_str] = 0
                    baseline_pnl[month_str] += pnl
                    position = None

            if not position:
                if hour < 7 or hour >= 22 or hour == 12:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < 20 or atr_pct > 80:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)
                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    tp_mult = 3.0 + (0.35 if 12 <= hour < 16 else 0)
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        # Label months
        monthly_features['is_bad'] = 0
        for month_str, pnl in baseline_pnl.items():
            month_period = pd.Period(month_str, freq='M')
            mask = monthly_features['month'] == month_period
            if mask.any():
                monthly_features.loc[mask, 'is_bad'] = 1 if pnl < 0 else 0

        # Prepare training data (use data before 2024-10)
        train_mask = monthly_features['month'] < pd.Period('2024-10', freq='M')
        monthly_clean = monthly_features.dropna(subset=regime_cols)

        X_train = monthly_clean[train_mask][regime_cols].values
        y_train = monthly_clean[train_mask]['is_bad'].values

        print(f"\nTraining data: {len(X_train)} months")
        print(f"Bad months in training: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print("\n" + "-" * 70)
        print("Testing Regime Detection Methods...")
        print("-" * 70)

        # Baseline
        print("\nBaseline (No Filter):")
        baseline = run_regime_backtest(df.copy(), None, None, monthly_features, regime_cols, use_regime_filter=False)
        print(f"  Return: {baseline['total_return']:+.1f}% | DD: {baseline['max_drawdown']:.1f}%")
        print(f"  Trades: {baseline['total_trades']} | LOSING MONTHS: {baseline['losing_months']}")

        results = [{'name': 'Baseline', **baseline}]

        # RandomForest for regime classification
        print("\nRandomForest Regime Detector:")
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3,
                                     class_weight='balanced', random_state=42)
        rf.fit(X_train_scaled, y_train)

        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            result = run_regime_backtest(
                df.copy(), rf, scaler, monthly_features, regime_cols,
                bad_regime_threshold=thresh, use_regime_filter=True
            )
            status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
            print(f"  Thresh {thresh:.1f}: {status:<8} | Ret={result['total_return']:+6.1f}% | "
                  f"Trades={result['total_trades']:>3} | Paused={result['paused_months']}")
            results.append({'name': f'RF_{thresh}', **result})

        # Isolation Forest for anomaly detection
        print("\nIsolation Forest (Anomaly = Bad Regime):")
        for contamination in [0.1, 0.15, 0.2, 0.25, 0.3]:
            iso = IsolationForest(contamination=contamination, random_state=42)
            iso.fit(X_train_scaled)

            result = run_regime_backtest(
                df.copy(), iso, scaler, monthly_features, regime_cols,
                use_regime_filter=True
            )
            status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
            print(f"  Contam {contamination:.2f}: {status:<8} | Ret={result['total_return']:+6.1f}% | "
                  f"Trades={result['total_trades']:>3} | Paused={result['paused_months']}")
            results.append({'name': f'IsoForest_{contamination}', **result})

        # Sort and show best
        results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        print("\n" + "=" * 70)
        print("TOP CONFIGURATIONS")
        print("=" * 70)

        for i, r in enumerate(results[:8], 1):
            print(f"\n{i}. {r['name']}")
            print(f"   Losing: {r['losing_months']} | Return: {r['total_return']:+.1f}% | DD: {r['max_drawdown']:.1f}%")
            print(f"   Trades: {r['total_trades']} | Paused Months: {r.get('paused_months', 0)}")

        # Best monthly P/L
        best = results[0]
        print("\n" + "=" * 70)
        print(f"BEST: {best['name']}")
        print("=" * 70)
        print("Monthly P/L:")
        for m, p in sorted(best['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            print(f"  {m}: ${p:+,.2f}{marker}")

        zero_loss = [r for r in results if r['losing_months'] == 0]
        if zero_loss:
            print("\n" + "=" * 70)
            print("ZERO-LOSS CONFIGURATIONS!")
            print("=" * 70)
            for r in zero_loss:
                print(f"  {r['name']}: Return={r['total_return']:+.1f}%, Trades={r['total_trades']}")

    finally:
        mt5.shutdown()
        print("\nDone!")

if __name__ == "__main__":
    main()
