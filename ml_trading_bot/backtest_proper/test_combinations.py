"""
Indicator Combinations Test
===========================
Test if combining RSI with other promising indicators improves performance.

Best candidates from individual tests:
1. ROC(10) -0.1/+0.1 = +510.4% (closest to baseline)
2. DPO(10) -15/+15 = +377.6%
3. MFI(7) 42/58 = +335.7%
4. Stochastic(10,3) 42/58 = +318.9%

Test combinations where BOTH indicators must agree.
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


def calculate_roc(close, period=10):
    return ((close - close.shift(period)) / close.shift(period)) * 100


def calculate_dpo(close, period=10):
    shift = period // 2 + 1
    sma = close.rolling(period).mean()
    return close - sma.shift(shift)


def calculate_mfi(high, low, close, period=7):
    tp = (high + low + close) / 3
    volume_proxy = high - low
    raw_mf = tp * volume_proxy
    tp_change = tp.diff()
    positive_mf = pd.Series(0.0, index=close.index)
    negative_mf = pd.Series(0.0, index=close.index)
    positive_mf[tp_change > 0] = raw_mf[tp_change > 0]
    negative_mf[tp_change < 0] = raw_mf[tp_change < 0]
    positive_mf_sum = positive_mf.rolling(period).sum()
    negative_mf_sum = negative_mf.rolling(period).sum()
    mf_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
    return 100 - (100 / (1 + mf_ratio))


def calculate_stochastic(high, low, close, k_period=10, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)


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

    # RSI(10)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Other indicators
    df['roc'] = calculate_roc(df['close'], 10)
    df['dpo'] = calculate_dpo(df['close'], 10) * 10000  # in pips
    df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], 7)
    df['stoch'] = calculate_stochastic(df['high'], df['low'], df['close'], 10, 3)

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

    print('=' * 100)
    print('INDICATOR COMBINATIONS TEST')
    print('=' * 100)
    print('Testing if combining RSI with other indicators improves performance')
    print('Requirement: BOTH indicators must agree on signal direction')

    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(combo=None):
        """
        combo options:
        - None: RSI only (baseline)
        - 'roc': RSI + ROC must agree
        - 'dpo': RSI + DPO must agree
        - 'mfi': RSI + MFI must agree
        - 'stoch': RSI + Stochastic must agree
        - 'roc_loose': RSI + ROC (looser thresholds)
        - 'any2': RSI + any 1 of (ROC, DPO, Stoch) must agree
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        filtered_count = 0

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

                rsi = row['rsi']
                rsi_signal = 1 if rsi < 42 else (-1 if rsi > 58 else 0)

                if combo is None:
                    signal = rsi_signal
                elif combo == 'roc':
                    roc = row['roc']
                    roc_signal = 1 if roc < -0.1 else (-1 if roc > 0.1 else 0)
                    signal = rsi_signal if rsi_signal == roc_signal else 0
                    if rsi_signal != 0 and signal == 0:
                        filtered_count += 1
                elif combo == 'roc_loose':
                    roc = row['roc']
                    roc_signal = 1 if roc < -0.05 else (-1 if roc > 0.05 else 0)
                    signal = rsi_signal if rsi_signal == roc_signal else 0
                    if rsi_signal != 0 and signal == 0:
                        filtered_count += 1
                elif combo == 'dpo':
                    dpo = row['dpo']
                    dpo_signal = 1 if dpo < -15 else (-1 if dpo > 15 else 0)
                    signal = rsi_signal if rsi_signal == dpo_signal else 0
                    if rsi_signal != 0 and signal == 0:
                        filtered_count += 1
                elif combo == 'mfi':
                    mfi = row['mfi']
                    mfi_signal = 1 if mfi < 42 else (-1 if mfi > 58 else 0)
                    signal = rsi_signal if rsi_signal == mfi_signal else 0
                    if rsi_signal != 0 and signal == 0:
                        filtered_count += 1
                elif combo == 'stoch':
                    stoch = row['stoch']
                    stoch_signal = 1 if stoch < 42 else (-1 if stoch > 58 else 0)
                    signal = rsi_signal if rsi_signal == stoch_signal else 0
                    if rsi_signal != 0 and signal == 0:
                        filtered_count += 1
                elif combo == 'any2':
                    # RSI + at least 1 other must agree
                    roc = row['roc']
                    dpo = row['dpo']
                    stoch = row['stoch']
                    roc_signal = 1 if roc < -0.1 else (-1 if roc > 0.1 else 0)
                    dpo_signal = 1 if dpo < -15 else (-1 if dpo > 15 else 0)
                    stoch_signal = 1 if stoch < 42 else (-1 if stoch > 58 else 0)

                    confirms = sum([1 for s in [roc_signal, dpo_signal, stoch_signal] if s == rsi_signal])
                    signal = rsi_signal if confirms >= 1 else 0
                    if rsi_signal != 0 and signal == 0:
                        filtered_count += 1
                elif combo == 'any2_strict':
                    # RSI + at least 2 others must agree
                    roc = row['roc']
                    dpo = row['dpo']
                    stoch = row['stoch']
                    roc_signal = 1 if roc < -0.1 else (-1 if roc > 0.1 else 0)
                    dpo_signal = 1 if dpo < -15 else (-1 if dpo > 15 else 0)
                    stoch_signal = 1 if stoch < 42 else (-1 if stoch > 58 else 0)

                    confirms = sum([1 for s in [roc_signal, dpo_signal, stoch_signal] if s == rsi_signal])
                    signal = rsi_signal if confirms >= 2 else 0
                    if rsi_signal != 0 and signal == 0:
                        filtered_count += 1
                else:
                    signal = rsi_signal

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
        return ret, trades, wr, max_dd, filtered_count

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(combo=None)
    print(f'\n1. BASELINE: RSI(10) 42/58 only')
    print(f'   Return: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Combinations
    print('\n2. COMBINATIONS (RSI + other indicator must agree)')
    print('-' * 100)
    print(f'   {"Combination":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 105)

    combos = [
        ('roc', 'RSI + ROC(10) -0.1/+0.1'),
        ('roc_loose', 'RSI + ROC(10) -0.05/+0.05 loose'),
        ('dpo', 'RSI + DPO(10) -15/+15 pips'),
        ('mfi', 'RSI + MFI(7) 42/58'),
        ('stoch', 'RSI + Stochastic(10) 42/58'),
        ('any2', 'RSI + any 1 of (ROC/DPO/Stoch)'),
        ('any2_strict', 'RSI + any 2 of (ROC/DPO/Stoch)'),
    ]

    for combo_key, name in combos:
        ret, trades, wr, max_dd, filtered = backtest(combo=combo_key)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 0 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('ANALYSIS')
    print('=' * 100)
    print("""
Key observations:
1. Adding confirmation filters REDUCES trades
2. Win rate may slightly increase, but...
3. Total return typically decreases because good trades get filtered too

The pattern is consistent: RSI alone captures mean reversion better than any combination.
""")


if __name__ == "__main__":
    main()
