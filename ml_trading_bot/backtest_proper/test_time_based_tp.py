"""
Test Time-Based TP Variations
==============================
Found that +0.5x TP during London+NY overlap improves performance.
Let's test more variations and fine-tune.
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

    # v3.4 BASELINE parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    def backtest(time_bonus_config=None, name=""):
        """
        Backtest with time-based TP bonus
        time_bonus_config: list of (start_hour, end_hour, bonus) tuples
        """
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
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)

                    # Time-based TP bonus
                    tp_bonus = 0
                    if time_bonus_config:
                        for start, end, bonus in time_bonus_config:
                            if start <= hour < end:
                                tp_bonus = bonus
                                break

                    tp_mult = base_tp + tp_bonus
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, yearly_pnl, balance

    print('=' * 100)
    print('TIME-BASED TP OPTIMIZATION')
    print('=' * 100)

    # Baseline
    print('\n' + '-' * 100)
    print('BASELINE v3.4')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(None, "Baseline")
    baseline_ret = ret
    print(f"   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%")
    print(f"   Yearly: {' | '.join([f'{y}: ${p:+,.0f}' for y, p in sorted(yearly.items())])}")

    print('\n' + '-' * 100)
    print('TEST 1: London+NY Overlap TP Bonus')
    print('-' * 100)

    # Different time windows
    configs = [
        ([(12, 16, 0.3)], '12-16 +0.3x'),
        ([(12, 16, 0.5)], '12-16 +0.5x'),  # Original finding
        ([(12, 16, 0.7)], '12-16 +0.7x'),
        ([(12, 16, 1.0)], '12-16 +1.0x'),
        ([(13, 17, 0.5)], '13-17 +0.5x'),
        ([(11, 15, 0.5)], '11-15 +0.5x'),
        ([(12, 18, 0.5)], '12-18 +0.5x'),
    ]

    best_ret = baseline_ret
    best_config = None

    for config, name in configs:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(config, name)
        diff = ret - baseline_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_config = (config, name)
            marker = ' <-- NEW BEST'
        print(f"   {name:<15}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('TEST 2: Multiple Time Windows')
    print('-' * 100)

    # Different combined windows
    configs2 = [
        # London morning lower, overlap higher
        ([(7, 12, -0.2), (12, 16, 0.5)], 'London -0.2, Overlap +0.5'),
        ([(7, 12, -0.3), (12, 16, 0.5)], 'London -0.3, Overlap +0.5'),
        # Overlap high, NY afternoon lower
        ([(12, 16, 0.5), (16, 22, -0.2)], 'Overlap +0.5, NY -0.2'),
        # Three zones
        ([(7, 12, -0.2), (12, 16, 0.5), (16, 22, -0.2)], 'London -0.2, Overlap +0.5, NY -0.2'),
        ([(7, 10, -0.3), (10, 14, 0.3), (14, 18, 0.5), (18, 22, -0.3)], '4 zones'),
    ]

    for config, name in configs2:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(config, name)
        diff = ret - baseline_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_config = (config, name)
            marker = ' <-- NEW BEST'
        print(f"   {name:<40}: +{ret:.1f}% ({diff:+.1f}) | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('TEST 3: Fine-tune best overlap window')
    print('-' * 100)

    # Fine-tune around 12-16
    configs3 = [
        ([(12, 15, 0.5)], '12-15 +0.5x'),
        ([(12, 16, 0.4)], '12-16 +0.4x'),
        ([(12, 16, 0.6)], '12-16 +0.6x'),
        ([(13, 16, 0.5)], '13-16 +0.5x'),
        ([(11, 16, 0.5)], '11-16 +0.5x'),
        ([(12, 17, 0.5)], '12-17 +0.5x'),
    ]

    for config, name in configs3:
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(config, name)
        diff = ret - baseline_ret
        marker = ''
        if ret > best_ret:
            best_ret = ret
            best_config = (config, name)
            marker = ' <-- NEW BEST'
        print(f"   {name:<15}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    # Show yearly breakdown of best config
    print('\n' + '=' * 100)
    if best_config:
        print(f'BEST CONFIG: {best_config[1]}')
        print('=' * 100)
        ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(best_config[0], best_config[1])
        print(f"\nReturn: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%")
        print(f"Final Balance: ${final_bal:,.2f}")
        print(f"\nYearly Breakdown:")
        for year in sorted(yearly.keys()):
            pnl = yearly[year]
            status = '' if pnl > 0 else ' (LOSS)'
            print(f"  {year}: ${pnl:>+12,.2f}{status}")
        print(f"\nProfitable Years: {prof_yrs}/6")
    else:
        print('BASELINE IS STILL BEST')

    # Test combining Time-based TP with other small improvements
    print('\n' + '=' * 100)
    print('TEST 4: COMBINATION WITH BEST TIME CONFIG')
    print('=' * 100)

    # Baseline vs best time config
    time_config = [(12, 16, 0.5)]

    print(f"\nBaseline: +{baseline_ret:.1f}%")
    ret, trades, wr, max_dd, prof_yrs, yearly, final_bal = backtest(time_config, "Time 12-16 +0.5x")
    print(f"Time 12-16 +0.5x: +{ret:.1f}% (improvement: +{ret - baseline_ret:.1f}%)")
    print(f"MaxDD reduced: {41.1:.1f}% -> {max_dd:.1f}%")


if __name__ == "__main__":
    main()
