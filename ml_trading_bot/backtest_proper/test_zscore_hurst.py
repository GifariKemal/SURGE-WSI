"""
Test Z-Score and Hurst Exponent Techniques
============================================
Based on research from:
- QuantStart, StatOasis, PyQuant News
- arXiv papers on Hurst exponent optimization

Techniques tested:
1. Z-Score Mean Reversion (instead of RSI)
2. Hurst Exponent Filter (only trade when H < 0.5)
3. Half-life for holding period optimization
4. Dynamic Z-score thresholds based on volatility
5. Combining RSI with Z-score confirmation
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


def calculate_hurst(series, lags_range=(2, 100)):
    """
    Calculate Hurst Exponent using R/S analysis
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    lags = range(lags_range[0], min(lags_range[1], len(series) // 2))
    tau = []
    rs = []

    for lag in lags:
        returns = series.diff().dropna()
        if len(returns) < lag:
            continue

        # Split into chunks
        chunks = [returns[i:i+lag] for i in range(0, len(returns) - lag, lag)]
        if not chunks:
            continue

        rs_values = []
        for chunk in chunks:
            if len(chunk) < lag:
                continue
            mean_adj = chunk - chunk.mean()
            cumsum = mean_adj.cumsum()
            r = cumsum.max() - cumsum.min()
            s = chunk.std()
            if s > 0:
                rs_values.append(r / s)

        if rs_values:
            tau.append(lag)
            rs.append(np.mean(rs_values))

    if len(tau) < 5:
        return 0.5  # Default to random walk if not enough data

    # Linear regression in log-log space
    log_tau = np.log(tau)
    log_rs = np.log(rs)
    hurst = np.polyfit(log_tau, log_rs, 1)[0]

    return max(0, min(1, hurst))  # Clamp to [0, 1]


def calculate_half_life(series):
    """
    Calculate half-life of mean reversion
    Based on Ornstein-Uhlenbeck process
    """
    lag = series.shift(1)
    delta = series.diff()

    df = pd.DataFrame({'lag': lag, 'delta': delta}).dropna()
    if len(df) < 30:
        return 50  # Default

    # OLS regression: delta = phi * (lag - mean) + noise
    # half_life = -log(2) / log(1 + phi)
    from scipy import stats
    slope, _, _, _, _ = stats.linregress(df['lag'], df['delta'])

    if slope >= 0:
        return 100  # Not mean reverting

    half_life = -np.log(2) / slope
    return max(1, min(100, half_life))


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

    # Calculate indicators
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

    # Z-Score of price
    df['price_mean'] = df['close'].rolling(20).mean()
    df['price_std'] = df['close'].rolling(20).std()
    df['zscore'] = (df['close'] - df['price_mean']) / (df['price_std'] + 1e-10)

    # Rolling Hurst Exponent (calculated every 100 bars for performance)
    df['hurst'] = 0.5
    for i in range(200, len(df), 10):  # Update every 10 bars
        hurst = calculate_hurst(df['close'].iloc[max(0, i-200):i])
        for j in range(i, min(i+10, len(df))):
            df.iloc[j, df.columns.get_loc('hurst')] = hurst

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.5 BASELINE parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35

    def backtest_baseline():
        """v3.5 BASELINE"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        balance += (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        balance += (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        balance += (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        position = None
                    elif row['low'] <= position['tp']:
                        balance += (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                hour = row['hour']
                if hour < 7 or hour >= 22:
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
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    def backtest_zscore_only(z_entry=2.0, z_exit=0.5):
        """
        Pure Z-Score Strategy
        Entry: Z-score > +entry (sell) or < -entry (buy)
        Exit: Z-score returns to +-exit
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                z = row['zscore']
                pnl = 0
                closed = False

                if position['dir'] == 1:  # Long
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif z >= z_exit:  # Exit on mean reversion
                        pnl = (row['close'] - position['entry']) * position['size']
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        closed = True
                else:  # Short
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif z <= -z_exit:  # Exit on mean reversion
                        pnl = (position['entry'] - row['close']) * position['size']
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        closed = True

                if closed:
                    balance += pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                hour = row['hour']
                if hour < 7 or hour >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                z = row['zscore']
                signal = 1 if z < -z_entry else (-1 if z > z_entry else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    def backtest_hurst_filter():
        """
        RSI v3.5 + Hurst Filter
        Only trade when Hurst < 0.5 (mean reverting regime)
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        filtered = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        balance += (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        balance += (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        balance += (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        position = None
                    elif row['low'] <= position['tp']:
                        balance += (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                hour = row['hour']
                if hour < 7 or hour >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                # Hurst filter - only trade in mean reverting regime
                if row['hurst'] >= 0.5:
                    filtered += 1
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
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        print(f"    Filtered by Hurst >= 0.5: {filtered}")
        return ret, trades, wr, max_dd

    def backtest_rsi_zscore_combo():
        """
        RSI + Z-Score Confirmation
        Entry: RSI signal + Z-score in same direction
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        filtered = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        balance += (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        balance += (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        balance += (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        position = None
                    elif row['low'] <= position['tp']:
                        balance += (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                hour = row['hour']
                if hour < 7 or hour >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                z = row['zscore']

                # Both must agree
                rsi_signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)
                z_signal = 1 if z < -1.5 else (-1 if z > 1.5 else 0)

                if rsi_signal != z_signal or rsi_signal == 0:
                    if rsi_signal != 0:
                        filtered += 1
                    continue

                signal = rsi_signal
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
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        print(f"    Filtered by Z-score mismatch: {filtered}")
        return ret, trades, wr, max_dd

    def backtest_low_hurst_bonus():
        """
        RSI v3.5 + TP bonus when Hurst is very low (strong mean reversion)
        Add +0.5x TP when H < 0.4
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        bonus_trades = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        balance += (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        balance += (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        balance += (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        position = None
                    elif row['low'] <= position['tp']:
                        balance += (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                hour = row['hour']
                if hour < 7 or hour >= 22:
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

                    # Time bonus
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp

                    # Hurst bonus - larger TP when strong mean reversion
                    if row['hurst'] < 0.4:
                        tp_mult += 0.5
                        bonus_trades += 1

                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        print(f"    Trades with Hurst bonus: {bonus_trades}")
        return ret, trades, wr, max_dd

    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    print('=' * 100)
    print('Z-SCORE AND HURST EXPONENT TESTS')
    print('=' * 100)

    print('\n' + '-' * 100)
    print('1. v3.5 BASELINE (RSI 42/58 + Dynamic TP + Time TP)')
    print('-' * 100)
    ret, trades, wr, max_dd = backtest_baseline()
    baseline_ret = ret
    print(f"   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%")

    print('\n' + '-' * 100)
    print('2. PURE Z-SCORE STRATEGY (different thresholds)')
    print('-' * 100)

    for z_entry in [1.5, 2.0, 2.5, 3.0]:
        for z_exit in [0.0, 0.5, 1.0]:
            ret, trades, wr, max_dd = backtest_zscore_only(z_entry, z_exit)
            diff = ret - baseline_ret
            marker = ' <-- BETTER!' if diff > 0 else ''
            if trades > 100:  # Only show if enough trades
                print(f"   Z-entry={z_entry}, Z-exit={z_exit}: +{ret:.1f}% ({diff:+.1f}) | {trades} trades | WR {wr:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('3. RSI v3.5 + HURST FILTER (H < 0.5)')
    print('-' * 100)
    ret, trades, wr, max_dd = backtest_hurst_filter()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('4. RSI + Z-SCORE CONFIRMATION')
    print('-' * 100)
    ret, trades, wr, max_dd = backtest_rsi_zscore_combo()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('5. RSI v3.5 + HURST TP BONUS (H < 0.4 = +0.5x TP)')
    print('-' * 100)
    ret, trades, wr, max_dd = backtest_low_hurst_bonus()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)


if __name__ == "__main__":
    main()
