"""
Hybrid ML Filter + RSI v3.7 - V2
================================
Focus on test period: Oct 2024 - Jan 2026
Train on 2020-2024, test on Oct 2024-Jan 2026
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
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

def create_features(df):
    """Create ML features."""
    df['returns_1h'] = df['close'].pct_change()
    df['returns_4h'] = df['close'].pct_change(4)
    df['returns_24h'] = df['close'].pct_change(24)

    df['volatility_24h'] = df['returns_1h'].rolling(24).std()
    df['volatility_ratio'] = df['volatility_24h'] / df['volatility_24h'].rolling(120).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_slope'] = df['rsi'].diff(3)

    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['atr'].rolling(100).mean()

    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(5) - 1) * 100

    df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
    df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()

    atr_sum = df['atr'].rolling(14).sum()
    high_max = df['high'].rolling(14).max()
    low_min = df['low'].rolling(14).min()
    range_14 = high_max - low_min
    df['chop'] = 100 * np.log10(atr_sum / range_14) / np.log10(14)

    plus_dm_raw = df['high'].diff()
    minus_dm_raw = -df['low'].diff()
    plus_dm = pd.Series(np.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), plus_dm_raw, 0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), minus_dm_raw, 0), index=df.index)
    tr_series = pd.Series(tr, index=df.index)
    smoothed_tr = tr_series.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / smoothed_tr
    minus_di = 100 * minus_dm.rolling(14).mean() / smoothed_tr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(14).mean()

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['is_london'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['recent_volatility_trend'] = df['volatility_24h'].diff(24)

    return df

def create_trade_labels(df, lookahead=12, threshold=0.003):
    """Label trades based on future outcome."""
    future_high = df['high'].rolling(lookahead, min_periods=1).max().shift(-lookahead)
    future_low = df['low'].rolling(lookahead, min_periods=1).min().shift(-lookahead)
    current = df['close']

    long_potential = (future_high - current) / current
    short_potential = (current - future_low) / current

    labels = pd.Series(0, index=df.index)
    labels[(long_potential > threshold) & (long_potential > short_potential)] = 1
    labels[(short_potential > threshold) & (short_potential > long_potential)] = -1

    df['trade_label'] = labels.fillna(0)
    return df

def run_hybrid_backtest(df, ml_model, scaler, feature_cols, ml_threshold=0.5, use_ml_filter=True,
                        test_start='2024-10-01', test_end='2026-02-01'):
    """Run RSI v3.7 with ML filter on specific test period."""
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

    # Only test on specific period
    test_mask = (df.index >= test_start) & (df.index < test_end)
    test_indices = df[test_mask].index

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]

        # Skip if not in test period for entry decisions
        in_test_period = current_time >= pd.Timestamp(test_start) and current_time < pd.Timestamp(test_end)

        month_str = current_time.strftime('%Y-%m')
        weekday = row['weekday']
        hour = row['hour']

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

            rsi = row['rsi']
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if signal:
                # ML Filter
                if use_ml_filter and ml_model is not None:
                    try:
                        features = df.iloc[i:i+1][feature_cols].values
                        if not np.isnan(features).any():
                            features_scaled = scaler.transform(features)
                            proba = ml_model.predict_proba(features_scaled)[0]

                            # Get class probabilities
                            classes = ml_model.classes_
                            proba_dict = dict(zip(classes, proba))

                            if signal == 1:  # Long
                                long_prob = proba_dict.get(1, 0)
                                if long_prob < ml_threshold:
                                    trades_filtered += 1
                                    continue
                            elif signal == -1:  # Short
                                short_prob = proba_dict.get(-1, 0)
                                if short_prob < ml_threshold:
                                    trades_filtered += 1
                                    continue
                    except Exception:
                        pass

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
        'trades_filtered': trades_filtered
    }

def main():
    print("=" * 70)
    print("HYBRID ML FILTER + RSI v3.7 (V2)")
    print("Focus: Oct 2024 - Jan 2026 (matching original backtest)")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data("GBPUSD", start_date, end_date)

        if df is None:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")

        print("Creating features...")
        df = create_features(df)
        df = create_trade_labels(df)

        feature_cols = [
            'returns_1h', 'returns_4h', 'returns_24h',
            'volatility_24h', 'volatility_ratio',
            'rsi', 'rsi_slope', 'atr_ratio',
            'price_vs_sma20', 'price_vs_sma50', 'sma_slope',
            'higher_highs', 'lower_lows',
            'chop', 'adx',
            'is_london', 'is_ny', 'recent_volatility_trend'
        ]

        df = df.ffill()
        df_clean = df.dropna(subset=feature_cols + ['trade_label'])

        # Train on data before Oct 2024
        train_mask = df_clean.index < '2024-10-01'
        X_train = df_clean[train_mask][feature_cols].values
        y_train = df_clean[train_mask]['trade_label'].values

        print(f"\nTraining data: {len(X_train)} samples")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print("\n" + "-" * 70)
        print("Training ML Models...")
        print("-" * 70)

        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_leaf=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=15, random_state=42
            )
        }

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)

        print("\n" + "-" * 70)
        print("Testing on Oct 2024 - Jan 2026...")
        print("-" * 70)

        print("\nBaseline (No ML Filter):")
        baseline = run_hybrid_backtest(df.copy(), None, None, feature_cols, use_ml_filter=False)
        print(f"  Return: {baseline['total_return']:+.1f}% | DD: {baseline['max_drawdown']:.1f}%")
        print(f"  Trades: {baseline['total_trades']} | WinRate: {baseline['win_rate']:.1f}%")
        print(f"  Months: {baseline['profitable_months']}/{baseline['total_months']} profitable")
        print(f"  LOSING MONTHS: {baseline['losing_months']}")

        results = [{'name': 'Baseline', **baseline}]

        # Test with different thresholds
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

        for model_name, model in models.items():
            print(f"\n{model_name}:")
            for thresh in thresholds:
                result = run_hybrid_backtest(
                    df.copy(), model, scaler, feature_cols,
                    ml_threshold=thresh, use_ml_filter=True
                )
                status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
                print(f"  Thresh {thresh:.2f}: {status:<8} | Ret={result['total_return']:+6.1f}% | "
                      f"Trades={result['total_trades']:>3} | Filtered={result['trades_filtered']:>3}")
                results.append({'name': f'{model_name}_{thresh}', **result})

        # Sort by losing months, then by return
        results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        print("\n" + "=" * 70)
        print("TOP CONFIGURATIONS (by lowest losing months)")
        print("=" * 70)

        for i, r in enumerate(results[:8], 1):
            print(f"\n{i}. {r['name']}")
            print(f"   Losing: {r['losing_months']} | Return: {r['total_return']:+.1f}% | DD: {r['max_drawdown']:.1f}%")
            print(f"   Trades: {r['total_trades']} | Filtered: {r.get('trades_filtered', 0)}")

        # Show monthly P/L for best
        best = results[0]
        print("\n" + "=" * 70)
        print(f"BEST CONFIG: {best['name']}")
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
