"""
Hurst Exponent Test
===================
Test Hurst exponent as a regime filter for mean reversion.

Theory:
- H < 0.5: Mean-reverting (anti-persistent) - GOOD for our strategy
- H = 0.5: Random walk - avoid trading
- H > 0.5: Trending (persistent) - BAD for mean reversion

Only trade when Hurst indicates mean-reverting conditions.

Sources:
- https://robotwealth.com/demystifying-the-hurst-exponent-part-1/
- https://blog.quantinsti.com/hurst-exponent/
"""
import pandas as pd
import numpy as np
import psycopg2
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}


def calculate_hurst(series, max_lag=100):
    """
    Calculate Hurst exponent using the Rescaled Range (R/S) method.

    Returns:
    - H < 0.5: Mean-reverting
    - H = 0.5: Random walk
    - H > 0.5: Trending
    """
    if len(series) < max_lag:
        return 0.5  # Default to random walk if not enough data

    series = np.array(series)
    lags = range(2, min(max_lag, len(series) // 2))

    if len(lags) < 3:
        return 0.5

    # Calculate variance of lagged differences
    tau = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        tau.append(np.std(diff))

    # Log-log regression to get Hurst
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)

    # Handle any inf/nan values
    valid = np.isfinite(log_lags) & np.isfinite(log_tau)
    if valid.sum() < 3:
        return 0.5

    # Linear regression
    reg = np.polyfit(log_lags[valid], log_tau[valid], 1)
    hurst = reg[0]

    return max(0, min(1, hurst))  # Clamp to [0, 1]


def calculate_hurst_rs(series, max_lag=20):
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis.
    More accurate but slower.
    """
    if len(series) < max_lag * 2:
        return 0.5

    series = np.array(series)
    N = len(series)
    max_k = min(max_lag, N // 4)

    if max_k < 4:
        return 0.5

    RS_list = []
    n_list = []

    for n in range(10, max_k):
        RS = []
        for start in range(0, N, n):
            if start + n > N:
                break
            subseries = series[start:start + n]

            # Mean
            mean = np.mean(subseries)

            # Cumulative deviations from mean
            Y = np.cumsum(subseries - mean)

            # Range
            R = np.max(Y) - np.min(Y)

            # Standard deviation
            S = np.std(subseries)

            if S > 0:
                RS.append(R / S)

        if RS:
            RS_list.append(np.mean(RS))
            n_list.append(n)

    if len(RS_list) < 3:
        return 0.5

    # Log-log regression
    log_n = np.log(n_list)
    log_RS = np.log(RS_list)

    valid = np.isfinite(log_n) & np.isfinite(log_RS)
    if valid.sum() < 3:
        return 0.5

    reg = np.polyfit(log_n[valid], log_RS[valid], 1)
    hurst = reg[0]

    return max(0, min(1, hurst))


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT time, open, high, low, close
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)
    conn.close()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    print('=' * 100)
    print('HURST EXPONENT REGIME FILTER TEST')
    print('=' * 100)

    # 1. Calculate Hurst exponent over different windows
    print('\n1. HURST EXPONENT ANALYSIS')
    print('-' * 80)

    # Calculate rolling Hurst
    window = 100
    hurst_values = []

    print('   Calculating rolling Hurst exponent (this may take a while)...')

    for i in range(window, len(df), 10):  # Sample every 10 bars for speed
        series = df['close'].iloc[i-window:i].values
        h = calculate_hurst(series, max_lag=50)
        hurst_values.append(h)

    hurst_values = np.array(hurst_values)
    print(f'\n   Hurst Statistics (window={window}):')
    print(f'   Mean: {np.mean(hurst_values):.3f}')
    print(f'   Median: {np.median(hurst_values):.3f}')
    print(f'   Std: {np.std(hurst_values):.3f}')
    print(f'   Min: {np.min(hurst_values):.3f}')
    print(f'   Max: {np.max(hurst_values):.3f}')

    # Distribution
    mean_reverting = (hurst_values < 0.45).sum()
    random_walk = ((hurst_values >= 0.45) & (hurst_values <= 0.55)).sum()
    trending = (hurst_values > 0.55).sum()
    total = len(hurst_values)

    print(f'\n   Regime Distribution:')
    print(f'   Mean-reverting (H < 0.45): {mean_reverting} ({mean_reverting/total*100:.1f}%)')
    print(f'   Random walk (0.45 <= H <= 0.55): {random_walk} ({random_walk/total*100:.1f}%)')
    print(f'   Trending (H > 0.55): {trending} ({trending/total*100:.1f}%)')

    # Calculate Hurst for full dataframe (precompute for backtest)
    print('\n   Calculating Hurst for all bars (for backtest)...')

    df['hurst'] = 0.5  # Default
    for i in range(window, len(df)):
        if i % 1000 == 0:
            print(f'      Progress: {i}/{len(df)}')
        series = df['close'].iloc[i-window:i].values
        df.iloc[i, df.columns.get_loc('hurst')] = calculate_hurst(series, max_lag=50)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.7 parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(hurst_threshold=None, require_mean_reverting=False):
        """Backtest with Hurst filter"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        filtered_by_hurst = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                if (i - position['entry_idx']) >= MAX_HOLDING:
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    balance += pnl
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
                else:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if hour < 7 or hour >= 22:
                    continue
                if hour in SKIP_HOURS:
                    continue

                # Hurst filter
                if hurst_threshold is not None:
                    hurst = row['hurst']
                    if require_mean_reverting and hurst > hurst_threshold:
                        filtered_by_hurst += 1
                        continue

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, yearly_pnl, filtered_by_hurst

    # 2. Backtest with Hurst filter
    print('\n2. BACKTEST WITH HURST FILTER')
    print('-' * 80)
    print(f'   {"Hurst Threshold":^20} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 90)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest()
    print(f'   {"v3.7 Baseline":<20} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]

    for threshold in thresholds:
        ret, trades, wr, max_dd, _, filtered = backtest(hurst_threshold=threshold, require_mean_reverting=True)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   H < {threshold:<15} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # 3. Analyze Hurst by year
    print('\n3. HURST BY YEAR')
    print('-' * 80)

    for year in sorted(df.index.year.unique()):
        year_data = df[df.index.year == year]
        year_hurst = year_data['hurst'].dropna()
        if len(year_hurst) > 0:
            mean_h = year_hurst.mean()
            regime = 'Mean-reverting' if mean_h < 0.45 else ('Trending' if mean_h > 0.55 else 'Random walk')
            print(f'   {year}: Mean Hurst = {mean_h:.3f} ({regime})')

    # 4. Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    avg_hurst = np.mean(hurst_values)
    dominant_regime = 'Mean-reverting' if avg_hurst < 0.45 else ('Trending' if avg_hurst > 0.55 else 'Random walk')

    print(f"""
1. HURST EXPONENT ANALYSIS:
   - Average Hurst: {avg_hurst:.3f} ({dominant_regime})
   - GBPUSD H1 shows {"MEAN-REVERTING" if avg_hurst < 0.45 else "RANDOM WALK/TRENDING"} behavior

2. REGIME DISTRIBUTION:
   - Mean-reverting (H < 0.45): {mean_reverting/total*100:.1f}%
   - Random walk (0.45-0.55): {random_walk/total*100:.1f}%
   - Trending (H > 0.55): {trending/total*100:.1f}%

3. FILTER IMPACT:
   - Filtering by Hurst {"HELPS" if baseline_ret < backtest(0.50, True)[0] else "HURTS"} performance
   - Hurst filter may reduce trades without improving win rate

4. RECOMMENDATION:
   - If Hurst is consistently {"< 0.5 -> Market is mean-reverting -> Keep trading" if avg_hurst < 0.5 else "> 0.5 -> Market may be trending -> Be cautious"}
   - Hurst filter adds complexity but may not provide benefit
   - Our strategy already works well because GBPUSD H1 {"IS" if avg_hurst < 0.5 else "is mostly"} mean-reverting
""")


if __name__ == "__main__":
    main()
