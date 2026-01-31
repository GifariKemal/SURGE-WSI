"""
Ehlers Fisher Transform Test
============================
Test Fisher Transform indicator for mean reversion.

Created by John Ehlers (digital signal processing expert).

Fisher Transform converts prices into a Gaussian normal distribution,
making turning points easier to identify.

Formula:
    Value = 0.5 * ln((1 + X) / (1 - X))
    Where X = normalized price (-1 to 1 range)

Typically applied to a normalized price indicator like:
    X = (Price - Lowest) / (Highest - Lowest) * 2 - 1

Signal line crossovers indicate buy/sell.

Sources:
- John Ehlers "Cybernetic Analysis for Stocks and Futures"
- https://www.investopedia.com/terms/f/fisher-transform.asp
- https://www.motivewave.com/studies/fisher_transform.htm
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


def calculate_fisher_transform(high, low, period=10):
    """
    Calculate Ehlers Fisher Transform.
    """
    # Median price
    hl2 = (high + low) / 2

    # Normalized price (-1 to 1)
    highest = hl2.rolling(period).max()
    lowest = hl2.rolling(period).min()

    # Avoid division by zero and clip to (-0.999, 0.999)
    raw_value = (hl2 - lowest) / (highest - lowest + 1e-10) * 2 - 1
    raw_value = raw_value.clip(-0.999, 0.999)

    # Smooth the value
    value = raw_value.ewm(span=5, adjust=False).mean()
    value = value.clip(-0.999, 0.999)

    # Fisher Transform
    fisher = 0.5 * np.log((1 + value) / (1 - value))

    # Signal line (lagged fisher)
    signal = fisher.shift(1)

    return fisher, signal


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
    print('EHLERS FISHER TRANSFORM TEST')
    print('=' * 100)
    print('Formula: Fisher = 0.5 * ln((1 + X) / (1 - X))')
    print('Entry: Fisher crossover signal line or threshold levels')

    # Analyze Fisher distribution
    fisher, signal = calculate_fisher_transform(df['high'], df['low'], 10)
    fisher_sample = fisher.dropna()
    print(f'\nFisher(10) Distribution:')
    print(f'   Mean: {fisher_sample.mean():.2f}')
    print(f'   Std: {fisher_sample.std():.2f}')
    print(f'   Min: {fisher_sample.min():.2f}')
    print(f'   Max: {fisher_sample.max():.2f}')

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

    def backtest(use_fisher=False, period=10, os_level=-1.5, ob_level=1.5, use_crossover=False):
        """
        Backtest with Fisher Transform.
        """
        if use_fisher:
            fisher, signal = calculate_fisher_transform(df['high'], df['low'], period)

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

                if use_fisher:
                    fisher_val = fisher.iloc[i]
                    signal_val = signal.iloc[i]
                    if pd.isna(fisher_val) or pd.isna(signal_val):
                        continue

                    if use_crossover:
                        # Fisher crosses above signal = BUY
                        # Fisher crosses below signal = SELL
                        prev_fisher = fisher.iloc[i-1]
                        prev_signal = signal.iloc[i-1]
                        if pd.isna(prev_fisher) or pd.isna(prev_signal):
                            continue
                        if fisher_val > signal_val and prev_fisher <= prev_signal:
                            sig = 1
                        elif fisher_val < signal_val and prev_fisher >= prev_signal:
                            sig = -1
                        else:
                            sig = 0
                    else:
                        # Threshold-based
                        if fisher_val < os_level:
                            sig = 1
                        elif fisher_val > ob_level:
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
    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_fisher=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Fisher Threshold-based
    print('\n2. FISHER THRESHOLD-BASED')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    configs = [
        (10, -1.5, 1.5, 'Fisher(10) -1.5/+1.5'),
        (10, -2.0, 2.0, 'Fisher(10) -2.0/+2.0'),
        (10, -1.0, 1.0, 'Fisher(10) -1.0/+1.0'),
        (10, -0.5, 0.5, 'Fisher(10) -0.5/+0.5'),
        (5, -1.5, 1.5, 'Fisher(5) -1.5/+1.5 fast'),
        (5, -1.0, 1.0, 'Fisher(5) -1.0/+1.0'),
        (14, -1.5, 1.5, 'Fisher(14) -1.5/+1.5'),
        (20, -1.5, 1.5, 'Fisher(20) -1.5/+1.5 slow'),
    ]

    for period, os, ob, name in configs:
        ret, trades, wr, max_dd = backtest(use_fisher=True, period=period, os_level=os, ob_level=ob, use_crossover=False)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Fisher Signal Crossover
    print('\n3. FISHER SIGNAL CROSSOVER')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    for period in [5, 10, 14, 20]:
        ret, trades, wr, max_dd = backtest(use_fisher=True, period=period, use_crossover=True)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        name = f'Fisher({period}) crossover'
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    print('\n' + '=' * 100)
    print('FINDINGS: Fisher Transform normalizes price for clearer turning points')
    print('=' * 100)


if __name__ == "__main__":
    main()
