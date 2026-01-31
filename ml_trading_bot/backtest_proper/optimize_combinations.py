"""
Test Best Combinations with RSI 40/60
"""
import pandas as pd
import numpy as np
import psycopg2
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {'host': 'localhost', 'port': 5434, 'database': 'surge_wsi', 'user': 'surge_wsi', 'password': 'surge_wsi_secret'}

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
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    def backtest(min_atr_pct=20, max_atr_pct=80, tp_low=2.4, tp_med=3.0, tp_high=3.6, rsi_os=35, rsi_ob=65, sl_mult=1.5):
        balance = 10000.0
        wins = losses = 0
        position = None
        max_dd = 0
        peak = balance
        yearly_pnl = {}

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < rsi_os else (-1 if rsi > rsi_ob else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    tp_mult = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_med)
                    sl = entry - atr * sl_mult if signal == 1 else entry + atr * sl_mult
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100

        # Check consistency - how many years profitable
        profitable_years = sum(1 for v in yearly_pnl.values() if v > 0)

        return ret, trades, wr, max_dd, profitable_years, yearly_pnl

    print('=' * 100)
    print('TESTING BEST COMBINATIONS')
    print('=' * 100)

    configs = [
        ('v3.3 Baseline (RSI 35/65)', {'rsi_os': 35, 'rsi_ob': 65}),
        ('RSI 40/60 only', {'rsi_os': 40, 'rsi_ob': 60}),
        ('RSI 40/60 + Vol 10-90', {'rsi_os': 40, 'rsi_ob': 60, 'min_atr_pct': 10, 'max_atr_pct': 90}),
        ('RSI 40/60 + SL 1.0x', {'rsi_os': 40, 'rsi_ob': 60, 'sl_mult': 1.0}),
        ('RSI 40/60 + Vol 10-90 + SL 1.0x', {'rsi_os': 40, 'rsi_ob': 60, 'min_atr_pct': 10, 'max_atr_pct': 90, 'sl_mult': 1.0}),
        ('RSI 40/60 + SL 1.2x', {'rsi_os': 40, 'rsi_ob': 60, 'sl_mult': 1.2}),
        ('RSI 40/60 + Vol 10-90 + SL 1.2x', {'rsi_os': 40, 'rsi_ob': 60, 'min_atr_pct': 10, 'max_atr_pct': 90, 'sl_mult': 1.2}),
        ('RSI 38/62', {'rsi_os': 38, 'rsi_ob': 62}),
        ('RSI 38/62 + Vol 10-90', {'rsi_os': 38, 'rsi_ob': 62, 'min_atr_pct': 10, 'max_atr_pct': 90}),
        ('RSI 37/63', {'rsi_os': 37, 'rsi_ob': 63}),
        ('RSI 37/63 + Vol 10-90', {'rsi_os': 37, 'rsi_ob': 62, 'min_atr_pct': 10, 'max_atr_pct': 90}),
    ]

    print(f"\n{'Config':<35} {'Return':>10} {'Trades':>8} {'WR':>7} {'MaxDD':>8} {'ProfYrs':>8}")
    print('-' * 85)

    best_ret = 0
    best_name = ''
    results = []

    for name, params in configs:
        ret, trades, wr, max_dd, prof_yrs, yearly = backtest(**params)
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_name = name
            marker = ' *'
        results.append((name, ret, trades, wr, max_dd, prof_yrs, yearly, params))
        print(f"{name:<35} {ret:>+9.1f}% {trades:>8} {wr:>6.1f}% {max_dd:>7.1f}% {prof_yrs:>6}/6{marker}")

    # Show yearly breakdown of top 3
    print('\n' + '=' * 100)
    print('YEARLY BREAKDOWN - TOP 3 CONFIGS')
    print('-' * 100)

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:3]

    for name, ret, trades, wr, max_dd, prof_yrs, yearly, params in sorted_results:
        print(f"\n{name}: +{ret:.1f}% total")
        print(f"  Parameters: {params}")
        for year in sorted(yearly.keys()):
            pnl = yearly[year]
            marker = ' (loss)' if pnl < 0 else ''
            print(f"    {year}: ${pnl:>+10,.0f}{marker}")

    print('\n' + '=' * 100)
    print(f'BEST CONFIG: {best_name}')
    print(f'Return: +{best_ret:.1f}%')
    print('=' * 100)


if __name__ == "__main__":
    main()
