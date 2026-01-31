"""
Trade Frequency Analysis - RSI v3.3
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

    # Collect trades with full details
    trades = []
    position = None
    balance = 10000.0

    for i in range(200, len(df)):
        row = df.iloc[i]
        idx = df.index[i]

        if row['weekday'] >= 5:
            continue

        if position:
            pnl = 0
            result = None
            bars_held = i - position['entry_idx']

            if position['dir'] == 1:
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    result = 'SL'
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    result = 'TP'
            else:
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    result = 'SL'
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    result = 'TP'

            if result:
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': idx,
                    'year': position['entry_time'].year,
                    'month': position['entry_time'].month,
                    'hour': position['entry_time'].hour,
                    'weekday': position['entry_time'].weekday(),
                    'direction': 'BUY' if position['dir'] == 1 else 'SELL',
                    'result': result,
                    'pnl': pnl,
                    'pnl_pips': pnl / position['size'] / 0.0001 if position['size'] > 0 else 0,
                    'bars_held': bars_held,
                    'atr_pct': position['atr_pct']
                })
                position = None

        if not position:
            if row['hour'] < 7 or row['hour'] >= 22:
                continue
            atr_pct = row['atr_pct']
            if atr_pct < 20 or atr_pct > 80:
                continue

            rsi = row['rsi']
            signal = 1 if rsi < 35 else (-1 if rsi > 65 else 0)

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                tp_mult = 2.4 if atr_pct < 40 else (3.6 if atr_pct > 60 else 3.0)
                sl = entry - atr * 1.5 if signal == 1 else entry + atr * 1.5
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

                position = {
                    'dir': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'entry_time': idx,
                    'entry_idx': i,
                    'atr_pct': atr_pct
                }

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['result'] == 'TP']
    losses = trades_df[trades_df['result'] == 'SL']

    print('=' * 80)
    print('RSI v3.3 - DETAILED TRADE ANALYSIS')
    print('=' * 80)

    # ============================================================================
    # HOURLY DISTRIBUTION
    # ============================================================================
    print()
    print('HOURLY DISTRIBUTION (Entry Time):')
    print('-' * 80)
    hourly = trades_df.groupby('hour').agg({
        'pnl': ['count', 'sum'],
        'result': lambda x: (x == 'TP').sum()
    }).reset_index()
    hourly.columns = ['hour', 'trades', 'total_pnl', 'wins']
    hourly['wr'] = hourly['wins'] / hourly['trades'] * 100
    hourly['avg_pnl'] = hourly['total_pnl'] / hourly['trades']

    print(f"{'Hour':<6} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Total PnL':>12} {'Avg PnL':>10}")
    print('-' * 60)
    for _, row in hourly.sort_values('hour').iterrows():
        print(f"{int(row['hour']):02d}:00  {int(row['trades']):>8} {int(row['wins']):>6} {row['wr']:>7.1f}% {row['total_pnl']:>+11.0f} {row['avg_pnl']:>+10.1f}")

    best_hour = hourly.loc[hourly['total_pnl'].idxmax()]
    worst_hour = hourly.loc[hourly['total_pnl'].idxmin()]
    print()
    print(f"Best hour:  {int(best_hour['hour']):02d}:00 (${best_hour['total_pnl']:+,.0f})")
    print(f"Worst hour: {int(worst_hour['hour']):02d}:00 (${worst_hour['total_pnl']:+,.0f})")

    # ============================================================================
    # MONTHLY PERFORMANCE
    # ============================================================================
    print()
    print('=' * 80)
    print('MONTHLY PERFORMANCE (All Years Combined):')
    print('-' * 80)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = trades_df.groupby('month').agg({
        'pnl': ['count', 'sum'],
        'result': lambda x: (x == 'TP').sum()
    }).reset_index()
    monthly.columns = ['month', 'trades', 'total_pnl', 'wins']
    monthly['wr'] = monthly['wins'] / monthly['trades'] * 100

    print(f"{'Month':<6} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Total PnL':>12}")
    print('-' * 50)
    for _, row in monthly.sort_values('month').iterrows():
        m = months[int(row['month'])-1]
        marker = ' <-- BEST' if row['total_pnl'] == monthly['total_pnl'].max() else (' <-- WORST' if row['total_pnl'] == monthly['total_pnl'].min() else '')
        print(f"{m:<6} {int(row['trades']):>8} {int(row['wins']):>6} {row['wr']:>7.1f}% {row['total_pnl']:>+11.0f}{marker}")

    # ============================================================================
    # YEARLY PERFORMANCE
    # ============================================================================
    print()
    print('=' * 80)
    print('YEARLY PERFORMANCE:')
    print('-' * 80)
    yearly = trades_df.groupby('year').agg({
        'pnl': ['count', 'sum'],
        'result': lambda x: (x == 'TP').sum()
    }).reset_index()
    yearly.columns = ['year', 'trades', 'total_pnl', 'wins']
    yearly['wr'] = yearly['wins'] / yearly['trades'] * 100

    print(f"{'Year':<6} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Total PnL':>12}")
    print('-' * 50)
    for _, row in yearly.iterrows():
        print(f"{int(row['year']):<6} {int(row['trades']):>8} {int(row['wins']):>6} {row['wr']:>7.1f}% {row['total_pnl']:>+11.0f}")

    # ============================================================================
    # TRADE DURATION
    # ============================================================================
    print()
    print('=' * 80)
    print('TRADE DURATION (bars/hours held):')
    print('-' * 80)
    print(f"Average: {trades_df['bars_held'].mean():.1f} hours")
    print(f"Median:  {trades_df['bars_held'].median():.0f} hours")
    print(f"Min:     {trades_df['bars_held'].min():.0f} hours")
    print(f"Max:     {trades_df['bars_held'].max():.0f} hours")
    print()
    print('Duration distribution:')
    duration_bins = [0, 6, 12, 24, 48, 72, 1000]
    duration_labels = ['<6h', '6-12h', '12-24h', '1-2 days', '2-3 days', '>3 days']
    trades_df['duration_cat'] = pd.cut(trades_df['bars_held'], bins=duration_bins, labels=duration_labels)
    dur_dist = trades_df.groupby('duration_cat').size()
    for cat in duration_labels:
        count = dur_dist.get(cat, 0)
        pct = count / len(trades_df) * 100
        print(f"  {cat:<10}: {count:>5} trades ({pct:>5.1f}%)")

    # ============================================================================
    # WIN/LOSS STREAKS
    # ============================================================================
    print()
    print('=' * 80)
    print('WIN/LOSS STREAKS:')
    print('-' * 80)

    results = trades_df['result'].tolist()
    streaks = []
    current_streak = 1
    for i in range(1, len(results)):
        if results[i] == results[i-1]:
            current_streak += 1
        else:
            streaks.append((results[i-1], current_streak))
            current_streak = 1
    streaks.append((results[-1], current_streak))

    win_streaks = [s[1] for s in streaks if s[0] == 'TP']
    loss_streaks = [s[1] for s in streaks if s[0] == 'SL']

    print(f"Longest winning streak: {max(win_streaks)} trades")
    print(f"Longest losing streak:  {max(loss_streaks)} trades")
    print(f"Average win streak:     {np.mean(win_streaks):.1f} trades")
    print(f"Average loss streak:    {np.mean(loss_streaks):.1f} trades")

    # ============================================================================
    # PNL DISTRIBUTION
    # ============================================================================
    print()
    print('=' * 80)
    print('PNL DISTRIBUTION:')
    print('-' * 80)
    print(f"Average win:  +${wins['pnl'].mean():,.2f} (+{wins['pnl_pips'].mean():.1f} pips)")
    print(f"Average loss: -${abs(losses['pnl'].mean()):,.2f} (-{abs(losses['pnl_pips'].mean()):.1f} pips)")
    print(f"Largest win:  +${wins['pnl'].max():,.2f}")
    print(f"Largest loss: -${abs(losses['pnl'].min()):,.2f}")
    print()
    print(f"Risk/Reward ratio: 1:{abs(wins['pnl'].mean() / losses['pnl'].mean()):.2f}")
    print(f"Profit Factor: {wins['pnl'].sum() / abs(losses['pnl'].sum()):.2f}")

    # ============================================================================
    # VOLATILITY REGIME PERFORMANCE
    # ============================================================================
    print()
    print('=' * 80)
    print('PERFORMANCE BY VOLATILITY REGIME:')
    print('-' * 80)
    trades_df['vol_regime'] = pd.cut(trades_df['atr_pct'],
                                      bins=[0, 40, 60, 100],
                                      labels=['Low (20-40)', 'Medium (40-60)', 'High (60-80)'])
    vol_perf = trades_df.groupby('vol_regime').agg({
        'pnl': ['count', 'sum', 'mean'],
        'result': lambda x: (x == 'TP').sum()
    }).reset_index()
    vol_perf.columns = ['regime', 'trades', 'total_pnl', 'avg_pnl', 'wins']
    vol_perf['wr'] = vol_perf['wins'] / vol_perf['trades'] * 100

    print(f"{'Regime':<15} {'Trades':>8} {'WR%':>8} {'Total PnL':>12} {'Avg PnL':>10}")
    print('-' * 60)
    for _, row in vol_perf.iterrows():
        print(f"{row['regime']:<15} {int(row['trades']):>8} {row['wr']:>7.1f}% {row['total_pnl']:>+11.0f} {row['avg_pnl']:>+10.1f}")

    # ============================================================================
    # BUY vs SELL PERFORMANCE
    # ============================================================================
    print()
    print('=' * 80)
    print('BUY vs SELL PERFORMANCE:')
    print('-' * 80)
    dir_perf = trades_df.groupby('direction').agg({
        'pnl': ['count', 'sum', 'mean'],
        'result': lambda x: (x == 'TP').sum()
    }).reset_index()
    dir_perf.columns = ['direction', 'trades', 'total_pnl', 'avg_pnl', 'wins']
    dir_perf['wr'] = dir_perf['wins'] / dir_perf['trades'] * 100

    for _, row in dir_perf.iterrows():
        print(f"{row['direction']}: {int(row['trades'])} trades, WR {row['wr']:.1f}%, Total ${row['total_pnl']:+,.0f}, Avg ${row['avg_pnl']:+.2f}")

    # ============================================================================
    # WEEKDAY PERFORMANCE
    # ============================================================================
    print()
    print('=' * 80)
    print('WEEKDAY PERFORMANCE:')
    print('-' * 80)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekday_perf = trades_df.groupby('weekday').agg({
        'pnl': ['count', 'sum', 'mean'],
        'result': lambda x: (x == 'TP').sum()
    }).reset_index()
    weekday_perf.columns = ['weekday', 'trades', 'total_pnl', 'avg_pnl', 'wins']
    weekday_perf['wr'] = weekday_perf['wins'] / weekday_perf['trades'] * 100

    print(f"{'Day':<12} {'Trades':>8} {'WR%':>8} {'Total PnL':>12} {'Avg PnL':>10}")
    print('-' * 55)
    for _, row in weekday_perf.sort_values('weekday').iterrows():
        day = days[int(row['weekday'])]
        marker = ' <-- BEST' if row['total_pnl'] == weekday_perf['total_pnl'].max() else ''
        print(f"{day:<12} {int(row['trades']):>8} {row['wr']:>7.1f}% {row['total_pnl']:>+11.0f} {row['avg_pnl']:>+10.1f}{marker}")

    print()
    print('=' * 80)


if __name__ == "__main__":
    main()
