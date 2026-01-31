"""
TRIX Test
=========
Test TRIX indicator for mean reversion.

TRIX = Triple smoothed EMA rate of change.
Filters out noise through triple exponential smoothing.

Formula:
    EMA1 = EMA(Close, n)
    EMA2 = EMA(EMA1, n)
    EMA3 = EMA(EMA2, n)
    TRIX = (EMA3 - EMA3[1]) / EMA3[1] * 100

Interpretation:
- TRIX > 0: Bullish
- TRIX < 0: Bearish
- Zero line crossover: Signal

Sources:
- Jack Hutson (Stocks & Commodities)
- https://www.investopedia.com/terms/t/trix.asp
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


def calculate_trix(close, period=15):
    """Calculate TRIX indicator."""
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()

    trix = (ema3 - ema3.shift(1)) / (ema3.shift(1) + 1e-10) * 10000  # In basis points

    return trix


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
    print('TRIX TEST')
    print('=' * 100)
    print('Formula: TRIX = Rate of change of triple EMA')

    # Analyze distribution
    trix = calculate_trix(df['close'], 15)
    sample = trix.dropna()
    print(f'\nTRIX(15) Distribution (basis points):')
    print(f'   Mean: {sample.mean():.2f} | Std: {sample.std():.2f}')
    print(f'   5th pct: {sample.quantile(0.05):.2f} | 95th pct: {sample.quantile(0.95):.2f}')

    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(use_trix=False, period=15, os_level=-5, ob_level=5):
        if use_trix:
            indicator = calculate_trix(df['close'], period)
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

                if use_trix:
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

    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_trix=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    print('\n2. TRIX CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        (15, -5, 5, 'TRIX(15) -5/+5 bps'),
        (15, -3, 3, 'TRIX(15) -3/+3 bps'),
        (15, -2, 2, 'TRIX(15) -2/+2 bps'),
        (15, -1, 1, 'TRIX(15) -1/+1 bps'),
        (10, -5, 5, 'TRIX(10) -5/+5 bps'),
        (10, -3, 3, 'TRIX(10) -3/+3 bps'),
        (10, -2, 2, 'TRIX(10) -2/+2 bps'),
        (7, -5, 5, 'TRIX(7) -5/+5 fast'),
        (7, -3, 3, 'TRIX(7) -3/+3'),
        (5, -5, 5, 'TRIX(5) -5/+5 very fast'),
    ]

    for period, os, ob, name in configs:
        ret, trades, wr, max_dd = backtest(use_trix=True, period=period, os_level=os, ob_level=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')


if __name__ == "__main__":
    main()
