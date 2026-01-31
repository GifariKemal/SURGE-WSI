"""
Day of Week Effect Test
=======================
Test if certain days perform better for mean reversion.

Research shows:
- Monday: Low volume, choppy price action
- Tuesday-Thursday: Best volatility and liquidity
- Friday: Lower volume, weekend risk

Sources:
- https://www.quantifiedstrategies.com/day-of-the-week-effect/
- https://admiralmarkets.com/education/articles/forex-strategy/best-days-of-the-week-to-trade-forex
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
    df['weekday'] = df.index.weekday  # 0=Monday, 4=Friday
    df = df.ffill().fillna(0)

    print('=' * 100)
    print('DAY OF WEEK EFFECT TEST')
    print('=' * 100)

    # 1. Analyze volatility by day
    print('\n1. VOLATILITY BY DAY OF WEEK')
    print('-' * 80)

    df['daily_range'] = df['high'] - df['low']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    for day in range(5):
        day_data = df[df['weekday'] == day]
        avg_range = day_data['daily_range'].mean() * 10000  # Convert to pips
        avg_atr = day_data['atr'].mean() * 10000
        print(f'   {day_names[day]:<12}: Avg Range = {avg_range:.1f} pips | Avg ATR = {avg_atr:.1f} pips')

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

    def backtest(allowed_days=None, skip_days=None):
        """Backtest with day of week filter"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        day_stats = {d: {'wins': 0, 'losses': 0, 'pnl': 0} for d in range(5)}

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = int(row['weekday'])
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
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    day_stats[position['entry_day']]['pnl'] += pnl
                    if pnl > 0:
                        wins += 1
                        day_stats[position['entry_day']]['wins'] += 1
                    else:
                        losses += 1
                        day_stats[position['entry_day']]['losses'] += 1
                    position = None
                else:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['losses'] += 1
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['wins'] += 1
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['losses'] += 1
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['wins'] += 1
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

                # Day of week filter
                if allowed_days and weekday not in allowed_days:
                    continue
                if skip_days and weekday in skip_days:
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
                    position = {
                        'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp,
                        'size': size, 'entry_idx': i, 'entry_day': weekday
                    }

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, yearly_pnl, day_stats

    # 2. Baseline performance by day
    print('\n2. TRADE PERFORMANCE BY DAY (v3.7 Baseline)')
    print('-' * 80)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, day_stats = backtest()

    print(f'   {"Day":<12} | {"Trades":>8} | {"Wins":>6} | {"Losses":>6} | {"WR":>8} | {"P/L":>12}')
    print('   ' + '-' * 65)

    for day in range(5):
        stats = day_stats[day]
        total = stats['wins'] + stats['losses']
        wr = stats['wins'] / total * 100 if total > 0 else 0
        print(f'   {day_names[day]:<12} | {total:>8} | {stats["wins"]:>6} | {stats["losses"]:>6} | {wr:>6.1f}% | ${stats["pnl"]:>+10,.0f}')

    # 3. Test different day filters
    print('\n3. DAY OF WEEK FILTERS')
    print('-' * 80)
    print(f'   {"Filter":^30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 85)

    print(f'   {"v3.7 Baseline":<30} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}%')

    filters = [
        (None, [0], 'Skip Monday'),
        (None, [4], 'Skip Friday'),
        (None, [0, 4], 'Skip Mon & Fri'),
        ([1, 2, 3], None, 'Tue-Wed-Thu only'),
        ([1, 2, 3, 4], None, 'Tue-Fri only'),
        ([0, 1, 2, 3], None, 'Mon-Thu only'),
        ([1, 3], None, 'Tue & Thu only'),
        ([2], None, 'Wednesday only'),
    ]

    for allowed, skip, name in filters:
        ret, trades, wr, max_dd, _, _ = backtest(allowed_days=allowed, skip_days=skip)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 4. Yearly consistency check for best filter
    print('\n4. YEARLY CONSISTENCY (Skip Monday)')
    print('-' * 80)

    _, _, _, _, baseline_yearly, _ = backtest()
    _, _, _, _, skip_mon_yearly, _ = backtest(skip_days=[0])

    print(f'   {"Year":<6} | {"Baseline":>12} | {"Skip Monday":>12} | {"Diff":>10} | {"Better?":>10}')
    print('   ' + '-' * 60)

    years_better = 0
    for year in sorted(baseline_yearly.keys()):
        base_pnl = baseline_yearly.get(year, 0)
        skip_pnl = skip_mon_yearly.get(year, 0)
        diff = skip_pnl - base_pnl
        better = 'YES' if skip_pnl > base_pnl else 'NO'
        if skip_pnl > base_pnl:
            years_better += 1
        print(f'   {year:<6} | ${base_pnl:>+10,.0f} | ${skip_pnl:>+10,.0f} | ${diff:>+8,.0f} | {better:>10}')

    total_years = len(baseline_yearly)
    print(f'\n   Skip Monday better in {years_better}/{total_years} years ({years_better/total_years*100:.0f}%)')

    # 5. Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    # Find worst day
    worst_day = min(day_stats.items(), key=lambda x: x[1]['pnl'])
    best_day = max(day_stats.items(), key=lambda x: x[1]['pnl'])

    print(f"""
1. DAY OF WEEK ANALYSIS:
   - Best day: {day_names[best_day[0]]} (P/L: ${best_day[1]['pnl']:+,.0f})
   - Worst day: {day_names[worst_day[0]]} (P/L: ${worst_day[1]['pnl']:+,.0f})

2. FILTER IMPACT:
   - Skip Monday: {"HELPS" if skip_mon_yearly and sum(skip_mon_yearly.values()) > sum(baseline_yearly.values()) else "MIXED/HURTS"}
   - Consistent in {years_better}/{total_years} years

3. RECOMMENDATION:
   - If improvement is consistent -> Consider implementing
   - If not consistent -> REJECT (overfitting)
""")


if __name__ == "__main__":
    main()
