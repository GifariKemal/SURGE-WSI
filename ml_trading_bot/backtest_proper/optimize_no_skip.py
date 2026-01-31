"""
RSI Optimization - No Skip, Improve Probability
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
    df = pd.read_sql('''
        SELECT time, open, high, low, close
        FROM ohlcv WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    ''', conn)
    conn.close()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Indicators
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    def backtest(
        min_atr_pct=20, max_atr_pct=80,
        dynamic_tp=True,
        tp_low=2.4, tp_med=3.0, tp_high=3.6,
        buy_tp_bonus=0, sell_tp_bonus=0,
        rsi_oversold=35, rsi_overbought=65,
        sl_mult=1.5
    ):
        balance = 10000.0
        wins = losses = 0
        buy_wins = buy_losses = sell_wins = sell_losses = 0
        position = None

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        balance += (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        buy_losses += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        balance += (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        buy_wins += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        balance += (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        sell_losses += 1
                        position = None
                    elif row['low'] <= position['tp']:
                        balance += (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        sell_wins += 1
                        position = None

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < rsi_oversold else (-1 if rsi > rsi_overbought else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002

                    if dynamic_tp:
                        base_tp = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_med)
                    else:
                        base_tp = tp_med

                    if signal == 1:
                        tp_mult = base_tp + buy_tp_bonus
                    else:
                        tp_mult = base_tp + sell_tp_bonus

                    sl = entry - atr * sl_mult if signal == 1 else entry + atr * sl_mult
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        buy_trades = buy_wins + buy_losses
        sell_trades = sell_wins + sell_losses
        buy_wr = buy_wins / buy_trades * 100 if buy_trades else 0
        sell_wr = sell_wins / sell_trades * 100 if sell_trades else 0
        return ret, trades, wr, buy_trades, buy_wr, sell_trades, sell_wr

    print('=' * 90)
    print('RSI OPTIMIZATION - NO SKIP, IMPROVE PROBABILITY')
    print('=' * 90)

    # Baseline v3.3
    ret, trades, wr, bt, bwr, st, swr = backtest()
    print(f'\nv3.3 BASELINE: +{ret:.1f}% | {trades} trades | WR {wr:.1f}%')
    print(f'    BUY: {bt} trades, WR {bwr:.1f}% | SELL: {st} trades, WR {swr:.1f}%')
    baseline_ret = ret

    print('\n' + '=' * 90)
    print('TEST 1: VOLATILITY FILTER RANGE')
    print('-' * 90)

    vol_tests = [
        (20, 80, 'Current 20-80'),
        (0, 100, 'No filter'),
        (10, 90, 'Wide 10-90'),
        (30, 80, 'Shift up 30-80'),
        (20, 90, 'Expand 20-90'),
        (25, 85, 'Narrow 25-85'),
    ]

    for min_v, max_v, name in vol_tests:
        ret, trades, wr, bt, bwr, st, swr = backtest(min_atr_pct=min_v, max_atr_pct=max_v)
        diff = ret - baseline_ret
        marker = ' <-- BETTER' if diff > 5 else ''
        print(f'{name:<20}: +{ret:>6.1f}% ({diff:+.1f}) | {trades:>4} trades | WR {wr:.1f}%{marker}')

    print('\n' + '=' * 90)
    print('TEST 2: DYNAMIC TP MULTIPLIERS')
    print('-' * 90)

    tp_tests = [
        (2.4, 3.0, 3.6, 'Current 2.4/3.0/3.6'),
        (2.0, 2.5, 3.0, 'Smaller 2.0/2.5/3.0'),
        (2.5, 3.0, 4.0, 'Bigger high 2.5/3.0/4.0'),
        (2.0, 3.0, 4.0, 'Wide 2.0/3.0/4.0'),
        (3.0, 3.5, 4.0, 'All bigger 3.0/3.5/4.0'),
        (2.4, 3.0, 4.5, 'High vol 4.5x'),
        (2.0, 2.5, 4.0, 'Low smaller, high bigger'),
    ]

    for low, med, high, name in tp_tests:
        ret, trades, wr, bt, bwr, st, swr = backtest(tp_low=low, tp_med=med, tp_high=high)
        diff = ret - baseline_ret
        marker = ' <-- BETTER' if diff > 5 else ''
        print(f'{name:<25}: +{ret:>6.1f}% ({diff:+.1f}) | WR {wr:.1f}%{marker}')

    print('\n' + '=' * 90)
    print('TEST 3: BUY vs SELL TP ADJUSTMENT')
    print('-' * 90)

    bonus_tests = [
        (0, 0, 'No adjustment'),
        (0.5, 0, 'BUY +0.5x TP'),
        (0, -0.3, 'SELL -0.3x TP'),
        (0.5, -0.3, 'BUY +0.5, SELL -0.3'),
        (0.3, -0.5, 'BUY +0.3, SELL -0.5'),
        (1.0, 0, 'BUY +1.0x TP'),
        (0, -0.5, 'SELL -0.5x TP'),
    ]

    for buy_b, sell_b, name in bonus_tests:
        ret, trades, wr, bt, bwr, st, swr = backtest(buy_tp_bonus=buy_b, sell_tp_bonus=sell_b)
        diff = ret - baseline_ret
        marker = ' <-- BETTER' if diff > 5 else ''
        print(f'{name:<25}: +{ret:>6.1f}% ({diff:+.1f}) | BUY WR {bwr:.1f}% | SELL WR {swr:.1f}%{marker}')

    print('\n' + '=' * 90)
    print('TEST 4: RSI THRESHOLDS')
    print('-' * 90)

    rsi_tests = [
        (35, 65, 'Current 35/65'),
        (30, 70, 'Stricter 30/70'),
        (40, 60, 'Looser 40/60'),
        (33, 67, 'Slight strict 33/67'),
        (37, 63, 'Slight loose 37/63'),
        (30, 65, 'Asymmetric 30/65'),
        (35, 70, 'Asymmetric 35/70'),
    ]

    for os, ob, name in rsi_tests:
        ret, trades, wr, bt, bwr, st, swr = backtest(rsi_oversold=os, rsi_overbought=ob)
        diff = ret - baseline_ret
        marker = ' <-- BETTER' if diff > 5 else ''
        print(f'{name:<20}: +{ret:>6.1f}% ({diff:+.1f}) | {trades:>4} trades | WR {wr:.1f}%{marker}')

    print('\n' + '=' * 90)
    print('TEST 5: SL MULTIPLIER')
    print('-' * 90)

    sl_tests = [
        (1.5, 'Current 1.5x'),
        (1.2, 'Tighter 1.2x'),
        (1.8, 'Wider 1.8x'),
        (2.0, 'Wide 2.0x'),
        (1.3, 'Tight 1.3x'),
        (1.0, 'Very tight 1.0x'),
    ]

    for sl, name in sl_tests:
        ret, trades, wr, bt, bwr, st, swr = backtest(sl_mult=sl)
        diff = ret - baseline_ret
        marker = ' <-- BETTER' if diff > 5 else ''
        print(f'{name:<15}: +{ret:>6.1f}% ({diff:+.1f}) | {trades:>4} trades | WR {wr:.1f}%{marker}')

    # Find best combinations
    print('\n' + '=' * 90)
    print('TEST 6: BEST COMBINATIONS')
    print('-' * 90)

    combos = [
        {'name': 'v3.3 Baseline', 'params': {}},
        {'name': 'Wider vol 10-90', 'params': {'min_atr_pct': 10, 'max_atr_pct': 90}},
        {'name': 'High vol TP 4.5x', 'params': {'tp_high': 4.5}},
        {'name': 'SELL faster exit', 'params': {'sell_tp_bonus': -0.5}},
        {'name': 'Wider SL 1.8x', 'params': {'sl_mult': 1.8}},
        {'name': 'Vol 10-90 + TP 4.5x', 'params': {'min_atr_pct': 10, 'max_atr_pct': 90, 'tp_high': 4.5}},
        {'name': 'Vol 10-90 + SELL -0.5', 'params': {'min_atr_pct': 10, 'max_atr_pct': 90, 'sell_tp_bonus': -0.5}},
        {'name': 'TP 4.5x + SELL -0.5', 'params': {'tp_high': 4.5, 'sell_tp_bonus': -0.5}},
        {'name': 'All combined', 'params': {'min_atr_pct': 10, 'max_atr_pct': 90, 'tp_high': 4.5, 'sell_tp_bonus': -0.5}},
    ]

    best_ret = baseline_ret
    best_name = 'v3.3 Baseline'

    for combo in combos:
        ret, trades, wr, bt, bwr, st, swr = backtest(**combo['params'])
        diff = ret - baseline_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_name = combo['name']
            marker = ' <-- NEW BEST'
        print(f"{combo['name']:<25}: +{ret:>6.1f}% ({diff:+.1f}) | {trades:>4} trades | WR {wr:.1f}%{marker}")

    print('\n' + '=' * 90)
    print(f'BEST CONFIG: {best_name} with +{best_ret:.1f}%')
    print('=' * 90)


if __name__ == "__main__":
    main()
