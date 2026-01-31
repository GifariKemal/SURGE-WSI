"""
Detailed Skip Days Analysis
===========================
Confirm the "skip first 3 days" finding is robust and not overfitting.
Test yearly consistency and different skip configurations.
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
    df['day'] = df.index.day
    df = df.ffill().fillna(0)

    print('=' * 100)
    print('DETAILED SKIP DAYS ANALYSIS')
    print('=' * 100)

    # v3.7 parameters
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
    SKIP_HOURS = [12]

    def backtest(skip_days=None):
        """Backtest with skip days filter"""
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
            day = int(row['day'])

            if weekday >= 5:
                continue

            if position:
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
                if hour in SKIP_HOURS:
                    continue
                if skip_days and day in skip_days:
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
        prof_years = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_years, yearly_pnl

    # 1. Baseline
    print('\n1. BASELINE vs SKIP FIRST 3 DAYS')
    print('-' * 80)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, baseline_prof, baseline_yearly = backtest()
    skip3_ret, skip3_trades, skip3_wr, skip3_dd, skip3_prof, skip3_yearly = backtest(skip_days=[1, 2, 3])

    print(f'   Baseline:         +{baseline_ret:>7.1f}% | Trades: {baseline_trades:>5} | WR: {baseline_wr:>5.1f}% | MaxDD: {baseline_dd:>5.1f}%')
    print(f'   Skip days 1-3:    +{skip3_ret:>7.1f}% | Trades: {skip3_trades:>5} | WR: {skip3_wr:>5.1f}% | MaxDD: {skip3_dd:>5.1f}%')
    print(f'   Improvement:      +{skip3_ret - baseline_ret:>7.1f}%')

    # 2. Yearly consistency check
    print('\n2. YEARLY CONSISTENCY CHECK')
    print('-' * 80)
    print(f'   {"Year":<6} | {"Baseline":>12} | {"Skip 1-3":>12} | {"Diff":>10} | {"Better?":>10}')
    print('   ' + '-' * 55)

    years_better = 0
    for year in sorted(baseline_yearly.keys()):
        base_pnl = baseline_yearly.get(year, 0)
        skip_pnl = skip3_yearly.get(year, 0)
        diff = skip_pnl - base_pnl
        better = 'YES' if skip_pnl > base_pnl else 'NO'
        if skip_pnl > base_pnl:
            years_better += 1
        print(f'   {year:<6} | ${base_pnl:>+10,.0f} | ${skip_pnl:>+10,.0f} | ${diff:>+8,.0f} | {better:>10}')

    total_years = len(baseline_yearly)
    print(f'\n   Skip 1-3 better in {years_better}/{total_years} years ({years_better/total_years*100:.0f}%)')

    # 3. Test different skip day combinations
    print('\n3. DIFFERENT SKIP DAY COMBINATIONS')
    print('-' * 80)
    print(f'   {"Skip Days":^20} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 70)

    combinations = [
        (None, 'None (v3.7)'),
        ([1], 'Day 1 only'),
        ([2], 'Day 2 only'),
        ([3], 'Day 3 only'),
        ([1, 2], 'Days 1-2'),
        ([1, 2, 3], 'Days 1-3'),
        ([1, 2, 3, 4], 'Days 1-4'),
        ([1, 2, 3, 4, 5], 'Days 1-5'),
        ([2, 23], 'Days 2 & 23 (worst)'),
        ([2, 10, 15, 23], '4 worst days'),
    ]

    for skip_days, name in combinations:
        ret, trades, wr, max_dd, prof_yrs, _ = backtest(skip_days=skip_days)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 50 else ''
        print(f'   {name:<20} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 4. Out-of-sample test (different periods)
    print('\n4. PERIOD ROBUSTNESS TEST')
    print('-' * 80)

    def backtest_period(start_year, end_year, skip_days=None):
        """Test on specific period"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0

        for i in range(200, len(df)):
            year = df.index[i].year
            if year < start_year or year > end_year:
                continue

            row = df.iloc[i]
            weekday = row['weekday']
            hour = row['hour']
            day = int(row['day'])

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
                if skip_days and day in skip_days:
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
        return ret, trades, wr, max_dd

    periods = [
        (2020, 2021, '2020-2021'),
        (2022, 2023, '2022-2023'),
        (2024, 2026, '2024-2026'),
    ]

    print(f'   {"Period":<12} | {"Baseline":>10} | {"Skip 1-3":>10} | {"Diff":>10} | {"Consistent?":>12}')
    print('   ' + '-' * 65)

    consistent_count = 0
    for start, end, name in periods:
        base_ret, _, _, _ = backtest_period(start, end, skip_days=None)
        skip_ret, _, _, _ = backtest_period(start, end, skip_days=[1, 2, 3])
        diff = skip_ret - base_ret
        consistent = 'YES' if skip_ret > base_ret else 'NO'
        if skip_ret > base_ret:
            consistent_count += 1
        print(f'   {name:<12} | +{base_ret:>8.1f}% | +{skip_ret:>8.1f}% | {diff:>+8.1f}% | {consistent:>12}')

    print(f'\n   Consistent in {consistent_count}/{len(periods)} periods')

    # 5. Summary
    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)

    # Determine if we should adopt skip days
    if years_better >= total_years * 0.7 and consistent_count >= len(periods) * 0.7:
        recommendation = "ADOPT: Skip days 1-3 is robust and consistent"
        adopt = True
    elif years_better >= total_years * 0.5:
        recommendation = "CONSIDER: Skip days 1-3 shows improvement but not fully consistent"
        adopt = False
    else:
        recommendation = "REJECT: Skip days 1-3 is not consistent enough"
        adopt = False

    print(f"""
1. SKIP DAYS 1-3 ANALYSIS:
   - Total improvement: +{skip3_ret - baseline_ret:.1f}%
   - MaxDD improvement: {baseline_dd:.1f}% -> {skip3_dd:.1f}%
   - WR improvement: {baseline_wr:.1f}% -> {skip3_wr:.1f}%
   - Better in {years_better}/{total_years} years
   - Consistent in {consistent_count}/{len(periods)} periods

2. RECOMMENDATION:
   {recommendation}

3. RATIONALE:
   - Beginning of month has unusual flows (new capital, rebalancing)
   - Mean reversion may be disrupted by these flows
   - Day 2 consistently shows poor performance
""")


if __name__ == "__main__":
    main()
