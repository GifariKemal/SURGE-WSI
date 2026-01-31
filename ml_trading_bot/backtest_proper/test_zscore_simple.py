"""
Test Z-Score Techniques (Simplified - No Hurst)
================================================
Focus on Z-score based entry/exit without expensive calculations
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

    # Z-Score of price (different lookbacks)
    for lb in [10, 20, 50]:
        df[f'zscore_{lb}'] = (df['close'] - df['close'].rolling(lb).mean()) / (df['close'].rolling(lb).std() + 1e-10)

    # Z-Score of RSI
    df['rsi_zscore'] = (df['rsi'] - df['rsi'].rolling(50).mean()) / (df['rsi'].rolling(50).std() + 1e-10)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # Baseline parameters
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

    def backtest_zscore_confirm(z_col='zscore_20', z_threshold=1.0):
        """
        RSI + Z-Score Confirmation
        Entry: RSI signal + Z-score in same extreme direction
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
                z = row[z_col]

                # RSI signal
                rsi_signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if rsi_signal == 0:
                    continue

                # Z-score must confirm
                if rsi_signal == 1 and z > -z_threshold:  # Buy needs negative Z
                    filtered += 1
                    continue
                if rsi_signal == -1 and z < z_threshold:  # Sell needs positive Z
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
        return ret, trades, wr, max_dd, filtered

    def backtest_extreme_zscore_bonus(z_threshold=2.0, tp_bonus=0.3):
        """
        RSI v3.5 + TP bonus when Z-score is very extreme
        Larger TP when price is far from mean (likely to revert more)
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

                    # Z-score bonus - larger TP when price is far from mean
                    z = abs(row['zscore_20'])
                    if z > z_threshold:
                        tp_mult += tp_bonus
                        bonus_trades += 1

                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, bonus_trades

    def backtest_rsi_zscore_replace():
        """
        Replace RSI with Z-Score of RSI
        Entry: RSI Z-score < -1.5 (buy) or > 1.5 (sell)
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

                rsi_z = row['rsi_zscore']
                signal = 1 if rsi_z < -1.5 else (-1 if rsi_z > 1.5 else 0)

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

    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    print('=' * 100)
    print('Z-SCORE ENHANCEMENT TESTS')
    print('=' * 100)

    print('\n' + '-' * 100)
    print('1. v3.5 BASELINE (RSI 42/58 + Dynamic TP + Time TP)')
    print('-' * 100)
    ret, trades, wr, max_dd = backtest_baseline()
    baseline_ret = ret
    print(f"   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%")

    print('\n' + '-' * 100)
    print('2. RSI + Z-SCORE CONFIRMATION (different lookbacks)')
    print('-' * 100)

    for lb in [10, 20, 50]:
        for z_th in [0.5, 1.0, 1.5]:
            ret, trades, wr, max_dd, filtered = backtest_zscore_confirm(f'zscore_{lb}', z_th)
            diff = ret - baseline_ret
            marker = ' <-- BETTER!' if diff > 0 else ''
            print(f"   LB={lb}, Z>{z_th}: +{ret:.1f}% ({diff:+.1f}) | {trades} trades | Filtered: {filtered}{marker}")

    print('\n' + '-' * 100)
    print('3. EXTREME Z-SCORE TP BONUS')
    print('-' * 100)

    for z_th in [1.5, 2.0, 2.5]:
        for bonus in [0.2, 0.3, 0.5]:
            ret, trades, wr, max_dd, bonus_count = backtest_extreme_zscore_bonus(z_th, bonus)
            diff = ret - baseline_ret
            marker = ' <-- BETTER!' if diff > 0 else ''
            print(f"   Z>{z_th} +{bonus}x: +{ret:.1f}% ({diff:+.1f}) | Bonus trades: {bonus_count}{marker}")

    print('\n' + '-' * 100)
    print('4. RSI Z-SCORE (Replace fixed thresholds)')
    print('-' * 100)
    ret, trades, wr, max_dd = backtest_rsi_zscore_replace()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}%{marker}")

    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)


if __name__ == "__main__":
    main()
