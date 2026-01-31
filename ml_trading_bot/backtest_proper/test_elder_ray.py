"""
Elder Ray Test
==============
Test Elder Ray (Bull Power / Bear Power) for mean reversion.

Created by Dr. Alexander Elder.

Formula:
    Bull Power = High - EMA(Close, period)
    Bear Power = Low - EMA(Close, period)

Interpretation:
- Bull Power > 0: Bulls stronger than average
- Bear Power < 0: Bears stronger than average
- Extreme values may indicate reversal

For mean reversion:
- Very negative Bull Power = oversold
- Very positive Bear Power = overbought

Sources:
- Alexander Elder "Trading for a Living"
- https://www.investopedia.com/terms/e/elderray.asp
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


def calculate_elder_ray(high, low, close, period=13):
    """Calculate Elder Ray Bull and Bear Power."""
    ema = close.ewm(span=period, adjust=False).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power


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
    print('ELDER RAY (BULL/BEAR POWER) TEST')
    print('=' * 100)
    print('Formula: Bull Power = High - EMA, Bear Power = Low - EMA')

    # Analyze distribution (in pips)
    bull, bear = calculate_elder_ray(df['high'], df['low'], df['close'], 13)
    print(f'\nElder Ray(13) Distribution (pips):')
    print(f'   Bull Power - 5th: {(bull.quantile(0.05)*10000):.1f}, 95th: {(bull.quantile(0.95)*10000):.1f}')
    print(f'   Bear Power - 5th: {(bear.quantile(0.05)*10000):.1f}, 95th: {(bear.quantile(0.95)*10000):.1f}')

    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(use_elder=False, period=13, bull_os=-20, bear_ob=20, use_bull=True):
        """
        use_bull=True: Use Bull Power (negative = oversold)
        use_bull=False: Use Bear Power (positive = overbought)
        """
        if use_elder:
            bull_power, bear_power = calculate_elder_ray(df['high'], df['low'], df['close'], period)

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

                if use_elder:
                    bull_pips = bull_power.iloc[i] * 10000
                    bear_pips = bear_power.iloc[i] * 10000
                    if pd.isna(bull_pips) or pd.isna(bear_pips):
                        continue

                    # Bull Power very negative = oversold (BUY)
                    # Bear Power very positive = overbought (SELL)
                    buy_signal = bull_pips < bull_os
                    sell_signal = bear_pips > bear_ob
                    signal = 1 if buy_signal else (-1 if sell_signal else 0)
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

    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_elder=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    print('\n2. ELDER RAY CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        (13, -20, 20, 'Elder(13) -20/+20 pips'),
        (13, -15, 15, 'Elder(13) -15/+15 pips'),
        (13, -10, 10, 'Elder(13) -10/+10 pips'),
        (13, -5, 5, 'Elder(13) -5/+5 pips'),
        (10, -15, 15, 'Elder(10) -15/+15 pips'),
        (10, -10, 10, 'Elder(10) -10/+10 pips'),
        (7, -15, 15, 'Elder(7) -15/+15 fast'),
        (7, -10, 10, 'Elder(7) -10/+10'),
        (5, -10, 10, 'Elder(5) -10/+10 very fast'),
    ]

    for period, bull_os, bear_ob, name in configs:
        ret, trades, wr, max_dd = backtest(use_elder=True, period=period, bull_os=bull_os, bear_ob=bear_ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')


if __name__ == "__main__":
    main()
