"""
Z-Score Entry/Exit Test
=======================
Test Z-score based entry and exit thresholds from quantitative finance.

Z-Score Formula:
    Z = (Price - MA) / StdDev

Academic thresholds from pairs trading literature:
- Entry: Z = +/- 1.5 or 2.0 (1.5-2 std dev from mean)
- Exit: Z = 0 (mean reversion complete) or +/- 1.0
- Stop: Z = +/- 3.0 (extreme)

Sources:
- QuantStart: Pairs Trading Mean Reversion
- Hudson & Thames: Optimal Trading Thresholds
- Academic pairs trading literature
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


def calculate_zscore(close, lookback=20):
    """
    Calculate Z-score of price.
    Z = (Price - MA) / StdDev
    """
    ma = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    zscore = (close - ma) / (std + 1e-10)
    return zscore


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
    print('Z-SCORE ENTRY/EXIT TEST')
    print('=' * 100)
    print('Formula: Z = (Price - MA) / StdDev')
    print('Academic: Entry at Z = +/-1.5 to 2.0, Exit at Z = 0')

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

    def backtest(use_zscore=False, lookback=20, entry_z=2.0, exit_z=0.0, stop_z=3.0):
        """
        Backtest with Z-score entry/exit.

        Entry: |Z| > entry_z
        Exit: |Z| < exit_z (mean reversion complete)
        Stop: |Z| > stop_z (extreme, cut loss)
        """
        if use_zscore:
            zscore = calculate_zscore(df['close'], lookback)
        else:
            zscore = None

        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        zscore_exits = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                exit_trade = False
                pnl = 0

                # Z-score based exit (if using zscore)
                if use_zscore:
                    current_z = zscore.iloc[i]
                    # Exit if Z reverts to exit threshold
                    if position['dir'] == 1 and current_z >= -exit_z:
                        pnl = (row['close'] - position['entry']) * position['size']
                        exit_trade = True
                        zscore_exits += 1
                    elif position['dir'] == -1 and current_z <= exit_z:
                        pnl = (position['entry'] - row['close']) * position['size']
                        exit_trade = True
                        zscore_exits += 1
                    # Stop if Z goes even more extreme
                    elif position['dir'] == 1 and current_z < -stop_z:
                        pnl = (row['close'] - position['entry']) * position['size']
                        exit_trade = True
                    elif position['dir'] == -1 and current_z > stop_z:
                        pnl = (position['entry'] - row['close']) * position['size']
                        exit_trade = True

                # Max holding timeout
                if not exit_trade and (i - position['entry_idx']) >= MAX_HOLDING:
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    exit_trade = True

                # Regular SL/TP (only if not using zscore exit)
                if not exit_trade and not use_zscore:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            exit_trade = True
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            exit_trade = True
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            exit_trade = True
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            exit_trade = True

                if exit_trade:
                    balance += pnl
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
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

                if use_zscore:
                    z = zscore.iloc[i]
                    # Entry based on Z-score
                    signal = 1 if z < -entry_z else (-1 if z > entry_z else 0)
                else:
                    # RSI baseline
                    rsi = row['rsi']
                    signal = 1 if rsi < 42 else (-1 if rsi > 58 else 0)

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
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, yearly_pnl, zscore_exits

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest(use_zscore=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Z-score configurations
    print('\n2. Z-SCORE ENTRY/EXIT CONFIGURATIONS')
    print('-' * 80)
    print('   Entry at |Z| > entry_z, Exit at |Z| < exit_z')
    print(f'\n   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Z-Exits":^8}')
    print('   ' + '-' * 105)

    configs = [
        # (lookback, entry_z, exit_z, stop_z, name)
        (20, 1.5, 0.0, 3.0, 'Z(20) 1.5/0/3.0'),
        (20, 2.0, 0.0, 3.0, 'Z(20) 2.0/0/3.0 academic'),
        (20, 2.0, 0.5, 3.0, 'Z(20) 2.0/0.5/3.0'),
        (20, 2.0, 1.0, 3.0, 'Z(20) 2.0/1.0/3.0'),
        (20, 1.5, 0.5, 2.5, 'Z(20) 1.5/0.5/2.5'),
        (50, 2.0, 0.0, 3.0, 'Z(50) 2.0/0/3.0'),
        (100, 2.0, 0.0, 3.0, 'Z(100) 2.0/0/3.0'),
        (20, 1.0, 0.0, 2.0, 'Z(20) 1.0/0/2.0 tight'),
        (20, 2.5, 0.0, 4.0, 'Z(20) 2.5/0/4.0 wide'),
    ]

    for lookback, entry_z, exit_z, stop_z, name in configs:
        ret, trades, wr, max_dd, _, z_exits = backtest(
            use_zscore=True, lookback=lookback, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {z_exits:>6}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. Z-SCORE CONCEPT (Academic):
   - Measures how many std devs price is from mean
   - Entry when price is "stretched" from mean
   - Exit when price reverts to mean
   - Common in pairs trading literature

2. THRESHOLDS:
   - Entry: Z = +/- 1.5 to 2.0 (1.5-2 sigma)
   - Exit: Z = 0 (mean) or 0.5-1.0 (partial reversion)
   - Stop: Z = +/- 3.0 (extreme, 3 sigma event)

3. RECOMMENDATION:
   - Compare Z-score vs RSI performance
   - Z-score is a "pure" mean reversion measure
   - If improves returns -> Consider adopting
""")


if __name__ == "__main__":
    main()
