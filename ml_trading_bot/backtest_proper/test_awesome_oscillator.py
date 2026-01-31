"""
Awesome Oscillator (AO) Test
============================
Test Bill Williams' Awesome Oscillator for mean reversion.

AO measures market momentum using midpoint price difference of MAs.

Formula:
    Median Price = (High + Low) / 2
    AO = SMA(Median, 5) - SMA(Median, 34)

Interpretation:
- AO > 0: Bullish momentum
- AO < 0: Bearish momentum
- Zero line crossover: Momentum shift
- Saucer pattern: Momentum continuation

For mean reversion:
- Extreme negative AO = oversold
- Extreme positive AO = overbought

Sources:
- Bill Williams "New Trading Dimensions"
- https://www.investopedia.com/terms/a/awesome-oscillator.asp
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


def calculate_ao(high, low, fast=5, slow=34):
    """Calculate Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = median_price.rolling(fast).mean() - median_price.rolling(slow).mean()
    return ao


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
    print('AWESOME OSCILLATOR (AO) TEST')
    print('=' * 100)
    print('Formula: AO = SMA(Median, 5) - SMA(Median, 34)')

    # Analyze AO distribution (in pips)
    ao_sample = calculate_ao(df['high'], df['low']).dropna() * 10000
    print(f'\nAO(5,34) Distribution (in pips):')
    print(f'   Mean: {ao_sample.mean():.1f}')
    print(f'   Std: {ao_sample.std():.1f}')
    print(f'   5th percentile: {ao_sample.quantile(0.05):.1f}')
    print(f'   95th percentile: {ao_sample.quantile(0.95):.1f}')

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

    def backtest(use_ao=False, fast=5, slow=34, os_pips=-20, ob_pips=20):
        if use_ao:
            ao = calculate_ao(df['high'], df['low'], fast, slow)
        else:
            ao = None

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
                if hour < 7 or hour >= 22:
                    continue
                if hour in SKIP_HOURS:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                if use_ao:
                    ao_val = ao.iloc[i]
                    if pd.isna(ao_val):
                        continue
                    ao_pips = ao_val * 10000
                    signal = 1 if ao_pips < os_pips else (-1 if ao_pips > ob_pips else 0)
                else:
                    rsi = row['rsi']
                    signal = 1 if rsi < 42 else (-1 if rsi > 58 else 0)

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

    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_ao=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    print('\n2. AWESOME OSCILLATOR CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        (5, 34, -20, 20, 'AO(5,34) -20/+20 pips'),
        (5, 34, -15, 15, 'AO(5,34) -15/+15 pips'),
        (5, 34, -10, 10, 'AO(5,34) -10/+10 pips'),
        (5, 34, -5, 5, 'AO(5,34) -5/+5 pips'),
        (3, 20, -10, 10, 'AO(3,20) -10/+10 fast'),
        (3, 20, -5, 5, 'AO(3,20) -5/+5'),
        (5, 20, -10, 10, 'AO(5,20) -10/+10'),
        (7, 40, -15, 15, 'AO(7,40) -15/+15 slow'),
    ]

    for fast, slow, os, ob, name in configs:
        ret, trades, wr, max_dd = backtest(use_ao=True, fast=fast, slow=slow, os_pips=os, ob_pips=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')


if __name__ == "__main__":
    main()
