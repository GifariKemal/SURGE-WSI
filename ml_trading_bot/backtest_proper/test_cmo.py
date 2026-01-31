"""
Chande Momentum Oscillator (CMO) Test
=====================================
Test CMO indicator for mean reversion.

CMO is similar to RSI but uses raw momentum (not smoothed).

Formula:
    CMO = 100 * (Sum(Gains) - Sum(Losses)) / (Sum(Gains) + Sum(Losses))

Range: -100 to +100
- > +50: Overbought
- < -50: Oversold

Advantages over RSI:
- Centered around zero
- More volatile, potentially earlier signals

Sources:
- Tushar Chande "The New Technical Trader"
- https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
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


def calculate_cmo(close, period=14):
    """Calculate Chande Momentum Oscillator."""
    delta = close.diff()

    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)

    sum_gains = gains.rolling(period).sum()
    sum_losses = losses.rolling(period).sum()

    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses + 1e-10)

    return cmo


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
    print('CHANDE MOMENTUM OSCILLATOR (CMO) TEST')
    print('=' * 100)
    print('Formula: CMO = 100 * (SumGains - SumLosses) / (SumGains + SumLosses)')
    print('Range: -100 to +100 | OB: > +50 | OS: < -50')

    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(use_cmo=False, period=14, os_level=-50, ob_level=50):
        if use_cmo:
            indicator = calculate_cmo(df['close'], period)
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

                if use_cmo:
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

    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_cmo=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    print('\n2. CMO CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        (14, -50, 50, 'CMO(14) -50/+50 std'),
        (14, -40, 40, 'CMO(14) -40/+40'),
        (14, -30, 30, 'CMO(14) -30/+30'),
        (14, -20, 20, 'CMO(14) -20/+20'),
        (10, -50, 50, 'CMO(10) -50/+50'),
        (10, -30, 30, 'CMO(10) -30/+30'),
        (10, -16, 16, 'CMO(10) -16/+16 RSI-like'),
        (7, -40, 40, 'CMO(7) -40/+40 fast'),
        (7, -20, 20, 'CMO(7) -20/+20'),
        (5, -30, 30, 'CMO(5) -30/+30 very fast'),
    ]

    for period, os, ob, name in configs:
        ret, trades, wr, max_dd = backtest(use_cmo=True, period=period, os_level=os, ob_level=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')


if __name__ == "__main__":
    main()
