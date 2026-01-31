"""
Calendar Effects Test for Forex
================================
Test month-end rebalancing flows and other calendar effects.

Research shows:
- Last 2-3 trading days of month: Portfolio rebalancing
- First trading days of month: New capital inflows
- Turn of month effect is well documented in forex

This test checks if filtering by calendar days improves performance.
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


def get_month_position(date):
    """
    Get position within the month.
    Returns:
    - 'first_3': First 3 trading days
    - 'last_3': Last 3 trading days
    - 'mid': Middle of month
    """
    day = date.day
    days_in_month = pd.Timestamp(date).days_in_month

    if day <= 3:
        return 'first_3'
    elif day >= days_in_month - 2:
        return 'last_3'
    else:
        return 'mid'


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
    df['month_pos'] = df.index.to_series().apply(get_month_position)
    df = df.ffill().fillna(0)

    print('=' * 100)
    print('CALENDAR EFFECTS ANALYSIS (Month-End Rebalancing)')
    print('=' * 100)

    # 1. Analyze returns by month position
    print('\n1. PRICE MOVEMENT BY MONTH POSITION')
    print('-' * 80)

    df['hourly_return'] = df['close'].pct_change() * 100

    for pos in ['first_3', 'mid', 'last_3']:
        subset = df[df['month_pos'] == pos]
        mean_ret = subset['hourly_return'].mean()
        std_ret = subset['hourly_return'].std()
        count = len(subset)
        print(f'   {pos:12s}: Mean Return = {mean_ret:+.4f}%, Std = {std_ret:.4f}%, Count = {count}')

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

    def backtest(skip_month_end=False, skip_month_start=False, only_mid_month=False,
                 skip_days=None, only_days=None):
        """Backtest with calendar filters"""
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
            day = row['day']
            month_pos = row['month_pos']
            days_in_month = pd.Timestamp(df.index[i]).days_in_month

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

                # Calendar filters
                if skip_month_end and day >= days_in_month - 2:
                    continue
                if skip_month_start and day <= 3:
                    continue
                if only_mid_month and month_pos != 'mid':
                    continue
                if skip_days and day in skip_days:
                    continue
                if only_days and day not in only_days:
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
        return ret, trades, wr, max_dd, yearly_pnl

    # 2. Test calendar filters
    print('\n2. BACKTEST WITH CALENDAR FILTERS')
    print('-' * 80)
    print(f'   {"Filter":^30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 80)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest()
    print(f'   {"v3.7 Baseline":<30} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}%')

    filters = [
        (dict(skip_month_end=True), 'Skip last 3 days'),
        (dict(skip_month_start=True), 'Skip first 3 days'),
        (dict(only_mid_month=True), 'Mid-month only'),
        (dict(skip_month_end=True, skip_month_start=True), 'Skip first+last 3 days'),
        (dict(skip_days=[1, 2]), 'Skip 1st & 2nd day'),
        (dict(skip_days=list(range(28, 32))), 'Skip days 28-31'),
    ]

    for kwargs, name in filters:
        ret, trades, wr, max_dd, _ = backtest(**kwargs)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Analyze trades by day of month
    print('\n3. TRADE PERFORMANCE BY DAY OF MONTH')
    print('-' * 80)

    # Collect trade results by day
    day_results = {d: {'wins': 0, 'losses': 0, 'pnl': 0} for d in range(1, 32)}

    balance = 10000.0
    position = None

    for i in range(200, len(df)):
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
                day_results[position['entry_day']]['pnl'] += pnl
                if pnl > 0:
                    day_results[position['entry_day']]['wins'] += 1
                else:
                    day_results[position['entry_day']]['losses'] += 1
                position = None
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        day_results[position['entry_day']]['pnl'] += pnl
                        day_results[position['entry_day']]['losses'] += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        day_results[position['entry_day']]['pnl'] += pnl
                        day_results[position['entry_day']]['wins'] += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        day_results[position['entry_day']]['pnl'] += pnl
                        day_results[position['entry_day']]['losses'] += 1
                        position = None
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        day_results[position['entry_day']]['pnl'] += pnl
                        day_results[position['entry_day']]['wins'] += 1
                        position = None

        if not position:
            if hour < 7 or hour >= 22:
                continue
            if hour in SKIP_HOURS:
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
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i, 'entry_day': day}

    # Find worst days
    worst_days = []
    best_days = []
    for day, data in day_results.items():
        total = data['wins'] + data['losses']
        if total > 10:  # Only consider days with enough trades
            wr = data['wins'] / total * 100
            if data['pnl'] < 0:
                worst_days.append((day, data['pnl'], wr, total))
            else:
                best_days.append((day, data['pnl'], wr, total))

    worst_days.sort(key=lambda x: x[1])
    best_days.sort(key=lambda x: x[1], reverse=True)

    print('   Worst performing days:')
    for day, pnl, wr, total in worst_days[:5]:
        print(f'      Day {day:2d}: PnL = ${pnl:>+10,.2f} | WR = {wr:5.1f}% | Trades = {total}')

    print('\n   Best performing days:')
    for day, pnl, wr, total in best_days[:5]:
        print(f'      Day {day:2d}: PnL = ${pnl:>+10,.2f} | WR = {wr:5.1f}% | Trades = {total}')

    # 4. Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. MONTH-END REBALANCING EFFECT:
   - Last 3 days of month show different patterns
   - Portfolio rebalancing causes unusual flows
   - Effect may not be strong enough to filter

2. TURN OF MONTH EFFECT:
   - First few days of month = new capital inflows
   - May cause momentum rather than mean reversion
   - Consider direction-specific filtering

3. RECOMMENDATION:
   - Calendar filters may reduce trade count too much
   - Only apply if specific days show consistent losses
   - Test "skip worst days" based on historical analysis
""")


if __name__ == "__main__":
    main()
