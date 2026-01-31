"""
Test New Optimization Techniques for RSI v3.6
==============================================
Based on research findings:
1. ADX Regime Filter - Only trade when ADX < 25 (ranging market)
2. Day of Week Analysis - Check which days perform best
3. Adaptive RSI Thresholds - Adjust based on volatility
4. Hour of Day Analysis - Check which hours perform best
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


def calculate_adx(df, period=14):
    """Calculate ADX indicator"""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed averages
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()

    return adx


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

    # ADX
    df['adx'] = calculate_adx(df, 14)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df = df.ffill().fillna(0)

    # v3.6 baseline parameters
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

    def backtest(adx_filter=None, skip_days=None, skip_hours=None,
                 adaptive_rsi=False, skip_months=None):
        """v3.6 with optional filters"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        day_stats = {i: {'wins': 0, 'losses': 0, 'pnl': 0} for i in range(5)}
        hour_stats = {i: {'wins': 0, 'losses': 0, 'pnl': 0} for i in range(24)}
        filtered_adx = 0
        filtered_day = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']
            month = row['month']

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
                    day_stats[position['entry_day']]['pnl'] += pnl
                    hour_stats[position['entry_hour']]['pnl'] += pnl
                    if pnl > 0:
                        wins += 1
                        day_stats[position['entry_day']]['wins'] += 1
                        hour_stats[position['entry_hour']]['wins'] += 1
                    else:
                        losses += 1
                        day_stats[position['entry_day']]['losses'] += 1
                        hour_stats[position['entry_hour']]['losses'] += 1
                    position = None
                else:
                    # SL/TP check
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['losses'] += 1
                            hour_stats[position['entry_hour']]['pnl'] += pnl
                            hour_stats[position['entry_hour']]['losses'] += 1
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['wins'] += 1
                            hour_stats[position['entry_hour']]['pnl'] += pnl
                            hour_stats[position['entry_hour']]['wins'] += 1
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['losses'] += 1
                            hour_stats[position['entry_hour']]['pnl'] += pnl
                            hour_stats[position['entry_hour']]['losses'] += 1
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            day_stats[position['entry_day']]['pnl'] += pnl
                            day_stats[position['entry_day']]['wins'] += 1
                            hour_stats[position['entry_hour']]['pnl'] += pnl
                            hour_stats[position['entry_hour']]['wins'] += 1
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

                # Skip days filter
                if skip_days and weekday in skip_days:
                    filtered_day += 1
                    continue

                # Skip months filter
                if skip_months and month in skip_months:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                # ADX filter - only trade in ranging market
                if adx_filter and row['adx'] > adx_filter:
                    filtered_adx += 1
                    continue

                rsi = row['rsi']

                # Adaptive RSI thresholds based on ADX
                if adaptive_rsi:
                    adx = row['adx']
                    if adx < 15:  # Very ranging - use wider thresholds
                        rsi_os, rsi_ob = 38, 62
                    elif adx < 20:  # Ranging
                        rsi_os, rsi_ob = 40, 60
                    else:  # Slightly trending
                        rsi_os, rsi_ob = 44, 56
                else:
                    rsi_os, rsi_ob = RSI_OS, RSI_OB

                signal = 1 if rsi < rsi_os else (-1 if rsi > rsi_ob else 0)

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
                        'size': size, 'entry_idx': i, 'entry_day': weekday,
                        'entry_hour': hour
                    }

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return {
            'ret': ret, 'trades': trades, 'wr': wr, 'max_dd': max_dd,
            'yearly': yearly_pnl, 'day_stats': day_stats, 'hour_stats': hour_stats,
            'filtered_adx': filtered_adx, 'filtered_day': filtered_day
        }

    print('=' * 100)
    print('TESTING NEW OPTIMIZATION TECHNIQUES')
    print('=' * 100)

    # 1. BASELINE (v3.6)
    res = backtest()
    baseline_ret = res['ret']
    print(f"\n1. BASELINE (v3.6)")
    print(f"   Return: +{res['ret']:.1f}% | Trades: {res['trades']} | WR: {res['wr']:.1f}% | MaxDD: {res['max_dd']:.1f}%")

    # 2. DAY OF WEEK ANALYSIS
    print(f"\n2. DAY OF WEEK ANALYSIS")
    print("-" * 80)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for day, stats in res['day_stats'].items():
        total = stats['wins'] + stats['losses']
        wr = stats['wins'] / total * 100 if total > 0 else 0
        print(f"   {day_names[day]:10s}: Trades={total:4d} | WR={wr:5.1f}% | P/L=${stats['pnl']:>+10,.2f}")

    # Find worst performing days
    worst_days = sorted(res['day_stats'].items(), key=lambda x: x[1]['pnl'])[:2]
    print(f"\n   Worst days: {', '.join([day_names[d[0]] for d in worst_days])}")

    # 3. HOUR OF DAY ANALYSIS
    print(f"\n3. HOUR OF DAY ANALYSIS (Session 7-22 UTC)")
    print("-" * 80)
    for hour in range(7, 22):
        stats = res['hour_stats'][hour]
        total = stats['wins'] + stats['losses']
        wr = stats['wins'] / total * 100 if total > 0 else 0
        bar = '#' * int(stats['pnl'] / 1000) if stats['pnl'] > 0 else ''
        print(f"   {hour:02d}:00: Trades={total:4d} | WR={wr:5.1f}% | P/L=${stats['pnl']:>+10,.2f} {bar}")

    # Find worst hours
    session_hours = {h: res['hour_stats'][h] for h in range(7, 22)}
    worst_hours = sorted(session_hours.items(), key=lambda x: x[1]['pnl'])[:3]
    print(f"\n   Worst hours: {', '.join([f'{h[0]}:00' for h in worst_hours])}")

    # 4. ADX FILTER TEST
    print(f"\n4. ADX REGIME FILTER TEST")
    print("-" * 80)
    print(f"   (Only trade when ADX < threshold, indicating ranging market)")

    best_adx_ret = baseline_ret
    best_adx = None

    for adx_thresh in [15, 20, 25, 30, 35, 40, None]:
        res = backtest(adx_filter=adx_thresh)
        diff = res['ret'] - baseline_ret
        label = f"ADX<{adx_thresh}" if adx_thresh else "No filter"
        marker = ''
        if res['ret'] > best_adx_ret:
            best_adx_ret = res['ret']
            best_adx = adx_thresh
            marker = ' <-- BEST'
        filtered = f"Filtered: {res['filtered_adx']}" if adx_thresh else ""
        print(f"   {label:12s}: +{res['ret']:>6.1f}% ({diff:+6.1f}) | Trades: {res['trades']:4d} | WR: {res['wr']:.1f}% | MaxDD: {res['max_dd']:.1f}% {filtered}{marker}")

    # 5. SKIP WORST DAYS TEST
    print(f"\n5. SKIP WORST PERFORMING DAYS")
    print("-" * 80)

    # Test skipping individual days
    for day in range(5):
        res = backtest(skip_days=[day])
        diff = res['ret'] - baseline_ret
        print(f"   Skip {day_names[day]:10s}: +{res['ret']:.1f}% ({diff:+.1f}) | Trades: {res['trades']} | WR: {res['wr']:.1f}%")

    # Test skipping worst 2 days
    worst_day_indices = [d[0] for d in worst_days]
    res = backtest(skip_days=worst_day_indices)
    diff = res['ret'] - baseline_ret
    print(f"   Skip {'+'.join([day_names[d] for d in worst_day_indices]):20s}: +{res['ret']:.1f}% ({diff:+.1f}) | Trades: {res['trades']} | WR: {res['wr']:.1f}%")

    # 6. SKIP WORST HOURS TEST
    print(f"\n6. SKIP WORST PERFORMING HOURS")
    print("-" * 80)

    worst_hour_indices = [h[0] for h in worst_hours]
    for hour in worst_hour_indices:
        res = backtest(skip_hours=[hour])
        diff = res['ret'] - baseline_ret
        print(f"   Skip {hour:02d}:00: +{res['ret']:.1f}% ({diff:+.1f}) | Trades: {res['trades']} | WR: {res['wr']:.1f}%")

    # Test skipping worst 3 hours
    res = backtest(skip_hours=worst_hour_indices)
    diff = res['ret'] - baseline_ret
    print(f"   Skip all 3 worst hours: +{res['ret']:.1f}% ({diff:+.1f}) | Trades: {res['trades']} | WR: {res['wr']:.1f}%")

    # 7. ADAPTIVE RSI THRESHOLDS
    print(f"\n7. ADAPTIVE RSI THRESHOLDS (based on ADX)")
    print("-" * 80)
    print(f"   ADX < 15: RSI 38/62 (wider, more ranging)")
    print(f"   ADX 15-20: RSI 40/60")
    print(f"   ADX > 20: RSI 44/56 (tighter, less ranging)")

    res = backtest(adaptive_rsi=True)
    diff = res['ret'] - baseline_ret
    print(f"\n   Result: +{res['ret']:.1f}% ({diff:+.1f}) | Trades: {res['trades']} | WR: {res['wr']:.1f}% | MaxDD: {res['max_dd']:.1f}%")

    # 8. COMBINATION TEST
    print(f"\n8. COMBINATION TESTS")
    print("-" * 80)

    # Try best ADX + skip worst day
    if best_adx:
        for day in worst_day_indices:
            res = backtest(adx_filter=best_adx, skip_days=[day])
            diff = res['ret'] - baseline_ret
            print(f"   ADX<{best_adx} + Skip {day_names[day]}: +{res['ret']:.1f}% ({diff:+.1f}) | Trades: {res['trades']} | WR: {res['wr']:.1f}%")

    # ADX + Adaptive RSI
    if best_adx:
        res = backtest(adx_filter=best_adx, adaptive_rsi=True)
        diff = res['ret'] - baseline_ret
        print(f"   ADX<{best_adx} + Adaptive RSI: +{res['ret']:.1f}% ({diff:+.1f}) | Trades: {res['trades']} | WR: {res['wr']:.1f}%")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Baseline v3.6: +{baseline_ret:.1f}%")
    if best_adx and best_adx_ret > baseline_ret:
        print(f"Best ADX filter: ADX<{best_adx} -> +{best_adx_ret:.1f}% ({best_adx_ret - baseline_ret:+.1f}%)")
    else:
        print(f"ADX filter: No improvement")


if __name__ == "__main__":
    main()
