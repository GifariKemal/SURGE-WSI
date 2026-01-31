"""
Pivot Points Test
=================
Test Pivot Points for mean reversion signals.

Classic Pivot Points calculated from previous day's H/L/C.

Formula (Standard):
    Pivot = (High + Low + Close) / 3
    R1 = 2 * Pivot - Low
    S1 = 2 * Pivot - High
    R2 = Pivot + (High - Low)
    S2 = Pivot - (High - Low)
    R3 = High + 2 * (Pivot - Low)
    S3 = Low - 2 * (High - Pivot)

Entry signals:
- Price at or below S1 = oversold (BUY)
- Price at or above R1 = overbought (SELL)

Sources:
- https://www.investopedia.com/terms/p/pivotpoint.asp
- https://school.stockcharts.com/doku.php?id=technical_indicators:pivot_points
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


def calculate_pivot_points(df):
    """
    Calculate daily pivot points from H1 data.
    Returns pivot, S1, S2, S3, R1, R2, R3 for each bar.
    """
    # Extract date from index
    df = df.copy()
    df['date'] = df.index.date

    # Calculate daily H/L/C
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)  # Use PREVIOUS day's values

    # Calculate pivot points
    daily['pivot'] = (daily['high'] + daily['low'] + daily['close']) / 3
    daily['r1'] = 2 * daily['pivot'] - daily['low']
    daily['s1'] = 2 * daily['pivot'] - daily['high']
    daily['r2'] = daily['pivot'] + (daily['high'] - daily['low'])
    daily['s2'] = daily['pivot'] - (daily['high'] - daily['low'])
    daily['r3'] = daily['high'] + 2 * (daily['pivot'] - daily['low'])
    daily['s3'] = daily['low'] - 2 * (daily['high'] - daily['pivot'])

    # Map back to H1 data
    df['pivot'] = df['date'].map(daily['pivot'])
    df['r1'] = df['date'].map(daily['r1'])
    df['s1'] = df['date'].map(daily['s1'])
    df['r2'] = df['date'].map(daily['r2'])
    df['s2'] = df['date'].map(daily['s2'])
    df['r3'] = df['date'].map(daily['r3'])
    df['s3'] = df['date'].map(daily['s3'])

    return df


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

    # Calculate pivot points
    df = calculate_pivot_points(df)

    # Standard RSI(10) for comparison
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR for SL/TP
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
    print('PIVOT POINTS TEST')
    print('=' * 100)
    print('Formula: Pivot = (H + L + C) / 3, S1/R1, S2/R2, S3/R3')
    print('Entry: Price at support/resistance levels')

    # v3.7 parameters
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(use_pivot=False, level='s1r1', proximity_pct=0.0):
        """
        Backtest with Pivot Points.
        level: 's1r1', 's2r2', 's3r3', 'pivot'
        proximity_pct: how close to level to trigger signal (as % of price)
        """
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
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    balance += pnl
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
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
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

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                if use_pivot:
                    price = row['close']
                    pivot = row['pivot']

                    if pd.isna(pivot):
                        continue

                    proximity = price * proximity_pct / 100

                    if level == 's1r1':
                        support = row['s1']
                        resistance = row['r1']
                    elif level == 's2r2':
                        support = row['s2']
                        resistance = row['r2']
                    elif level == 's3r3':
                        support = row['s3']
                        resistance = row['r3']
                    else:  # pivot
                        support = pivot
                        resistance = pivot

                    if pd.isna(support) or pd.isna(resistance):
                        continue

                    # Price at or below support = BUY
                    # Price at or above resistance = SELL
                    if price <= support + proximity:
                        sig = 1
                    elif price >= resistance - proximity:
                        sig = -1
                    else:
                        sig = 0
                else:
                    rsi = row['rsi']
                    sig = 1 if rsi < 42 else (-1 if rsi > 58 else 0)

                if sig:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp
                    sl = entry - atr * SL_MULT if sig == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if sig == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': sig, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_pivot=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Pivot Point Levels
    print('\n2. PIVOT POINT LEVELS')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    configs = [
        ('s1r1', 0.0, 'S1/R1 exact'),
        ('s1r1', 0.05, 'S1/R1 0.05% proximity'),
        ('s1r1', 0.1, 'S1/R1 0.1% proximity'),
        ('s1r1', 0.2, 'S1/R1 0.2% proximity'),
        ('s2r2', 0.0, 'S2/R2 exact'),
        ('s2r2', 0.1, 'S2/R2 0.1% proximity'),
        ('s2r2', 0.2, 'S2/R2 0.2% proximity'),
        ('s3r3', 0.0, 'S3/R3 exact'),
        ('s3r3', 0.1, 'S3/R3 0.1% proximity'),
        ('pivot', 0.1, 'Pivot 0.1% proximity'),
        ('pivot', 0.2, 'Pivot 0.2% proximity'),
    ]

    for level, prox, name in configs:
        ret, trades, wr, max_dd = backtest(use_pivot=True, level=level, proximity_pct=prox)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    print('\n' + '=' * 100)
    print('FINDINGS: Pivot Points are daily S/R levels - may not suit H1 mean reversion')
    print('=' * 100)


if __name__ == "__main__":
    main()
