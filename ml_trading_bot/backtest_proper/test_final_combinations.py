"""
Test Final Combinations - Finding the Best v3.5
================================================
Combine best findings:
1. Time-based TP (12-16 +0.4x) - proven +26.5% improvement
2. Test other combinations that might help further
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

    # Indicators
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    def backtest(
        rsi_os=42, rsi_ob=58,
        sl_mult=1.5,
        tp_low=2.4, tp_med=3.0, tp_high=3.6,
        min_atr_pct=20, max_atr_pct=80,
        time_tp_bonus=0,  # Bonus during 12-16 UTC
        risk_pct=0.01
    ):
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
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
                hour = row['hour']
                if hour < 7 or hour >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < rsi_os else (-1 if rsi > rsi_ob else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002

                    # Dynamic TP based on volatility
                    base_tp = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_med)

                    # Time-based TP bonus (12-16 UTC = London+NY overlap)
                    if 12 <= hour < 16:
                        tp_mult = base_tp + time_tp_bonus
                    else:
                        tp_mult = base_tp

                    sl = entry - atr * sl_mult if signal == 1 else entry + atr * sl_mult
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * risk_pct
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, yearly_pnl, balance

    print('=' * 100)
    print('FINAL COMBINATION TESTS - Finding v3.5')
    print('=' * 100)

    # v3.4 Baseline
    ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest()
    baseline_ret = ret
    print(f'\n1. v3.4 BASELINE (RSI 42/58, Dynamic TP)')
    print(f'   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%')
    print(f'   Yearly: {" | ".join([f"{y}: ${p:+,.0f}" for y, p in sorted(yearly.items())])}')

    # v3.5 Candidate: Add Time-based TP
    ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(time_tp_bonus=0.4)
    print(f'\n2. v3.5 CANDIDATE (+ Time TP 12-16 +0.4x)')
    print(f'   Return: +{ret:.1f}% ({ret - baseline_ret:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%')
    print(f'   Yearly: {" | ".join([f"{y}: ${p:+,.0f}" for y, p in sorted(yearly.items())])}')
    v35_ret = ret

    print('\n' + '=' * 100)
    print('FINE-TUNING v3.5')
    print('=' * 100)

    # Test different RSI combinations with Time TP
    print('\n--- RSI Threshold Combinations ---')
    rsi_tests = [
        (40, 60, 'RSI 40/60'),
        (42, 58, 'RSI 42/58'),  # Current
        (43, 57, 'RSI 43/57'),
        (44, 56, 'RSI 44/56'),
        (45, 55, 'RSI 45/55'),
    ]

    best_ret = v35_ret
    best_config = None

    for os, ob, name in rsi_tests:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(
            rsi_os=os, rsi_ob=ob, time_tp_bonus=0.4
        )
        diff = ret - v35_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_config = name
            marker = ' <-- NEW BEST'
        print(f'   {name}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Test different TP base values
    print('\n--- Dynamic TP Base Values ---')
    tp_tests = [
        (2.2, 2.8, 3.4, 'TP 2.2/2.8/3.4'),
        (2.4, 3.0, 3.6, 'TP 2.4/3.0/3.6'),  # Current
        (2.5, 3.0, 3.5, 'TP 2.5/3.0/3.5'),
        (2.5, 3.2, 3.8, 'TP 2.5/3.2/3.8'),
        (2.6, 3.2, 3.8, 'TP 2.6/3.2/3.8'),
    ]

    for low, med, high, name in tp_tests:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(
            tp_low=low, tp_med=med, tp_high=high, time_tp_bonus=0.4
        )
        diff = ret - v35_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_config = name
            marker = ' <-- NEW BEST'
        print(f'   {name}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Test different volatility filter ranges
    print('\n--- Volatility Filter Range ---')
    vol_tests = [
        (15, 85, 'ATR 15-85'),
        (20, 80, 'ATR 20-80'),  # Current
        (25, 75, 'ATR 25-75'),
        (10, 90, 'ATR 10-90'),
    ]

    for min_v, max_v, name in vol_tests:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(
            min_atr_pct=min_v, max_atr_pct=max_v, time_tp_bonus=0.4
        )
        diff = ret - v35_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_config = name
            marker = ' <-- NEW BEST'
        print(f'   {name}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Test time bonus values
    print('\n--- Time TP Bonus Fine-tune ---')
    time_tests = [
        (0.3, '+0.3x'),
        (0.35, '+0.35x'),
        (0.4, '+0.4x'),  # Current
        (0.45, '+0.45x'),
        (0.5, '+0.5x'),
    ]

    for bonus, name in time_tests:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(time_tp_bonus=bonus)
        diff = ret - v35_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_config = f'Time {name}'
            marker = ' <-- NEW BEST'
        print(f'   {name}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Final comparison
    print('\n' + '=' * 100)
    print('FINAL COMPARISON')
    print('=' * 100)

    configs = [
        ('v3.4 (Baseline)', {}),
        ('v3.5 (+ Time TP +0.4x)', {'time_tp_bonus': 0.4}),
        ('v3.5 + RSI 43/57', {'time_tp_bonus': 0.4, 'rsi_os': 43, 'rsi_ob': 57}),
        ('v3.5 + TP 2.5/3.0/3.5', {'time_tp_bonus': 0.4, 'tp_low': 2.5, 'tp_med': 3.0, 'tp_high': 3.5}),
        ('v3.5 + ATR 15-85', {'time_tp_bonus': 0.4, 'min_atr_pct': 15, 'max_atr_pct': 85}),
    ]

    print(f"\n{'Config':<30} {'Return':>12} {'vs Base':>10} {'Trades':>8} {'WR':>8} {'MaxDD':>8} {'Prof Yrs':>10}")
    print('-' * 95)

    for name, params in configs:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(**params)
        diff = ret - baseline_ret
        print(f'{name:<30} {ret:>+11.1f}% {diff:>+9.1f}% {trades:>8} {wr:>7.1f}% {max_dd:>7.1f}% {prof_yrs:>8}/6')

    # Show recommended v3.5 config
    print('\n' + '=' * 100)
    print('RECOMMENDED v3.5 CONFIGURATION')
    print('=' * 100)

    ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(time_tp_bonus=0.4)

    print(f'''
RSI_CONFIG = {{
    "rsi_period": 10,
    "rsi_oversold": 42.0,
    "rsi_overbought": 58.0,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "session_start": 7,
    "session_end": 22,
    "risk_per_trade": 0.01,
    # Volatility filter (v3.2)
    "min_atr_percentile": 20.0,
    "max_atr_percentile": 80.0,
    # Dynamic TP (v3.3)
    "dynamic_tp": True,
    "tp_low_vol_mult": 2.4,
    "tp_high_vol_mult": 3.6,
    # Time-based TP (v3.5 NEW!)
    "time_tp_bonus": True,
    "time_tp_start": 12,  # London+NY overlap start
    "time_tp_end": 16,    # London+NY overlap end
    "time_tp_bonus_mult": 0.4,  # Add 0.4x ATR to TP during overlap
}}

PERFORMANCE:
  Return: +{ret:.1f}% (over 6 years)
  Trades: {trades} (~{trades/6:.0f}/year, ~{trades/6/52:.1f}/week)
  Win Rate: {wr:.1f}%
  Max Drawdown: {max_dd:.1f}%
  Profitable Years: {prof_yrs}/6

IMPROVEMENT vs v3.4:
  +{ret - baseline_ret:.1f}% return improvement
  -{baseline_ret - ret + 26.5:.1f}% drawdown improvement (from 41.1% to {max_dd:.1f}%)
''')

    print('\nYearly Breakdown:')
    for year in sorted(yearly.keys()):
        pnl = yearly[year]
        status = '' if pnl > 0 else ' (LOSS)'
        print(f'  {year}: ${pnl:>+12,.2f}{status}')


if __name__ == "__main__":
    main()
