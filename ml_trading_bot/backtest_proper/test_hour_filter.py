"""
Fine-tune Hour Skip Filter
==========================
Found skipping hour 12:00 gives +45.3% improvement!
Let's test various combinations to find optimal hours to skip.
"""
import pandas as pd
import numpy as np
import psycopg2
from itertools import combinations
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

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

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

    # v3.6 parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46

    def backtest(skip_hours=None):
        """v3.6 with optional hour skip filter"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                # Time-based exit (v3.6)
                if (i - position['entry_idx']) >= MAX_HOLDING:
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    balance += pnl
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
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
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
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

                # Skip hours filter
                if skip_hours and hour in skip_hours:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

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
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, yearly_pnl

    print('=' * 100)
    print('HOUR SKIP FILTER OPTIMIZATION')
    print('=' * 100)

    # Baseline
    ret, trades, wr, max_dd, prof_yrs, yearly = backtest()
    baseline_ret = ret
    print(f'\n1. BASELINE (v3.6, no hour skip)')
    print(f'   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%')
    print(f'   Yearly: {prof_yrs}/6 profitable')

    # Test single hours
    print(f'\n2. SKIP SINGLE HOURS')
    print('-' * 80)

    single_results = []
    for hour in range(7, 22):
        ret, trades, wr, max_dd, prof_yrs, yearly = backtest(skip_hours=[hour])
        diff = ret - baseline_ret
        single_results.append((hour, ret, diff, trades, wr, max_dd, prof_yrs))
        marker = ' <<<' if diff > 20 else ''
        print(f'   Skip {hour:02d}:00: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades:4d} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Sort by improvement
    best_singles = sorted(single_results, key=lambda x: x[2], reverse=True)[:5]
    print(f'\n   Top 5 hours to skip:')
    for h, ret, diff, trades, wr, max_dd, prof_yrs in best_singles:
        print(f'   - {h:02d}:00 -> +{diff:.1f}%')

    # Test combinations of 2 hours
    print(f'\n3. SKIP 2 HOURS COMBINATIONS (top performers)')
    print('-' * 80)

    top_hours = [h[0] for h in best_singles[:6]]  # Top 6 single hours
    combo_results = []

    for combo in combinations(top_hours, 2):
        ret, trades, wr, max_dd, prof_yrs, yearly = backtest(skip_hours=list(combo))
        diff = ret - baseline_ret
        combo_results.append((combo, ret, diff, trades, wr, max_dd, prof_yrs))

    combo_results.sort(key=lambda x: x[2], reverse=True)
    for combo, ret, diff, trades, wr, max_dd, prof_yrs in combo_results[:10]:
        hours_str = '+'.join([f'{h:02d}' for h in combo])
        marker = ' <<<' if diff > 50 else ''
        print(f'   Skip {hours_str}:00: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades:4d} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Test combinations of 3 hours
    print(f'\n4. SKIP 3 HOURS COMBINATIONS (top performers)')
    print('-' * 80)

    combo3_results = []
    for combo in combinations(top_hours, 3):
        ret, trades, wr, max_dd, prof_yrs, yearly = backtest(skip_hours=list(combo))
        diff = ret - baseline_ret
        combo3_results.append((combo, ret, diff, trades, wr, max_dd, prof_yrs))

    combo3_results.sort(key=lambda x: x[2], reverse=True)
    for combo, ret, diff, trades, wr, max_dd, prof_yrs in combo3_results[:10]:
        hours_str = '+'.join([f'{h:02d}' for h in combo])
        marker = ' <<<' if diff > 50 else ''
        print(f'   Skip {hours_str}:00: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades:4d} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Best overall
    all_results = single_results + [(c[0],) + c[1:] for c in combo_results] + [(c[0],) + c[1:] for c in combo3_results]
    best = max(all_results, key=lambda x: x[2] if isinstance(x[2], float) else x[1])

    print(f'\n' + '=' * 100)
    print(f'BEST CONFIGURATION')
    print('=' * 100)

    if isinstance(best[0], tuple):
        hours_str = ', '.join([f'{h:02d}:00' for h in best[0]])
        skip_hours = list(best[0])
    else:
        hours_str = f'{best[0]:02d}:00'
        skip_hours = [best[0]]

    ret, trades, wr, max_dd, prof_yrs, yearly = backtest(skip_hours=skip_hours)
    diff = ret - baseline_ret

    print(f'\nSkip hours: {hours_str}')
    print(f'Return: +{ret:.1f}% (vs baseline +{baseline_ret:.1f}%, improvement: {diff:+.1f}%)')
    print(f'Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%')
    print(f'Profitable years: {prof_yrs}/6')
    print(f'\nYearly breakdown:')
    for year in sorted(yearly.keys()):
        pnl = yearly[year]
        status = '' if pnl > 0 else ' (LOSS)'
        print(f'  {year}: ${pnl:>+12,.2f}{status}')


if __name__ == "__main__":
    main()
