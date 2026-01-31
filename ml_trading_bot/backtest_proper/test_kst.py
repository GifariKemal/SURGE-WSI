"""
KST (Know Sure Thing) Test
==========================
Test KST indicator for mean reversion.

KST is a smoothed, weighted sum of 4 ROC indicators.
Created by Martin Pring.

Formula:
    ROC1 = ROC(10), smoothed with SMA(10)
    ROC2 = ROC(15), smoothed with SMA(10)
    ROC3 = ROC(20), smoothed with SMA(10)
    ROC4 = ROC(30), smoothed with SMA(15)

    KST = (ROC1 * 1) + (ROC2 * 2) + (ROC3 * 3) + (ROC4 * 4)
    Signal = SMA(KST, 9)

Sources:
- Martin Pring "Technical Analysis Explained"
- https://www.investopedia.com/terms/k/know-sure-thing-kst.asp
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


def calculate_kst(close, r1=10, r2=15, r3=20, r4=30, s1=10, s2=10, s3=10, s4=15, sig=9):
    """Calculate KST indicator."""
    roc1 = ((close - close.shift(r1)) / close.shift(r1)) * 100
    roc2 = ((close - close.shift(r2)) / close.shift(r2)) * 100
    roc3 = ((close - close.shift(r3)) / close.shift(r3)) * 100
    roc4 = ((close - close.shift(r4)) / close.shift(r4)) * 100

    sroc1 = roc1.rolling(s1).mean()
    sroc2 = roc2.rolling(s2).mean()
    sroc3 = roc3.rolling(s3).mean()
    sroc4 = roc4.rolling(s4).mean()

    kst = (sroc1 * 1) + (sroc2 * 2) + (sroc3 * 3) + (sroc4 * 4)
    signal = kst.rolling(sig).mean()

    return kst, signal


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

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    print('=' * 100)
    print('KST (KNOW SURE THING) TEST')
    print('=' * 100)
    print('Formula: KST = weighted sum of 4 smoothed ROC')

    # Analyze distribution
    kst, signal = calculate_kst(df['close'])
    sample = kst.dropna()
    print(f'\nKST Distribution:')
    print(f'   Mean: {sample.mean():.3f} | Std: {sample.std():.3f}')
    print(f'   5th pct: {sample.quantile(0.05):.3f} | 95th pct: {sample.quantile(0.95):.3f}')

    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(use_kst=False, fast=False, os_level=-1, ob_level=1):
        if use_kst:
            if fast:
                kst, signal = calculate_kst(df['close'], r1=5, r2=7, r3=10, r4=15, s1=5, s2=5, s3=5, s4=7, sig=5)
            else:
                kst, signal = calculate_kst(df['close'])
            indicator = kst
        else:
            indicator = df['rsi']

        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                if (i - position['entry_idx']) >= MAX_HOLDING:
                    pnl = (row['close'] - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - row['close']) * position['size']
                    balance += pnl
                    wins += 1 if pnl > 0 else 0
                    losses += 1 if pnl <= 0 else 0
                    position = None
                else:
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
                if hour < 7 or hour >= 22 or hour in SKIP_HOURS:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                ind_val = indicator.iloc[i]
                if pd.isna(ind_val):
                    continue

                if use_kst:
                    signal = 1 if ind_val < os_level else (-1 if ind_val > ob_level else 0)
                else:
                    signal = 1 if ind_val < 42 else (-1 if ind_val > 58 else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    tp_mult = base_tp + TIME_TP_BONUS if 12 <= hour < 16 else base_tp
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_kst=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    print('\n2. KST CONFIGURATIONS (Standard)')
    print('-' * 80)
    print(f'   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        (False, -1.0, 1.0, 'KST std -1.0/+1.0'),
        (False, -0.5, 0.5, 'KST std -0.5/+0.5'),
        (False, -0.3, 0.3, 'KST std -0.3/+0.3'),
        (False, -0.2, 0.2, 'KST std -0.2/+0.2'),
        (False, -0.1, 0.1, 'KST std -0.1/+0.1'),
        (True, -0.5, 0.5, 'KST fast -0.5/+0.5'),
        (True, -0.3, 0.3, 'KST fast -0.3/+0.3'),
        (True, -0.2, 0.2, 'KST fast -0.2/+0.2'),
        (True, -0.1, 0.1, 'KST fast -0.1/+0.1'),
    ]

    for fast, os, ob, name in configs:
        ret, trades, wr, max_dd = backtest(use_kst=True, fast=fast, os_level=os, ob_level=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')


if __name__ == "__main__":
    main()
