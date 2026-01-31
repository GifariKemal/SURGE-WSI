"""
Test RSI Threshold vs MaxDD Trade-off
=====================================
RSI 45/55 showed lowest MaxDD (28.8%) in previous test.
Let's explore this trade-off more thoroughly.
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
    df = df.ffill().fillna(0)

    # Fixed parameters
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46

    def backtest(rsi_os=42, rsi_ob=58, skip_hours=None):
        """v3.7 base with configurable RSI thresholds"""
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
                if skip_hours and hour in skip_hours:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
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
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, yearly_pnl

    print('=' * 100)
    print('RSI THRESHOLD vs MAX DRAWDOWN TRADE-OFF')
    print('=' * 100)

    # Test RSI thresholds with skip_hours=[12]
    print(f"\nWith skip_hours=[12] (v3.7 default):")
    print("-" * 80)
    print(f"{'RSI':^10} | {'Return':^10} | {'Trades':^8} | {'WR':^8} | {'MaxDD':^8} | {'Sharpe*':^10}")
    print("-" * 80)

    results = []
    for rsi_os, rsi_ob in [(40, 60), (41, 59), (42, 58), (43, 57), (44, 56), (45, 55), (46, 54)]:
        ret, trades, wr, max_dd, prof_yrs, yearly = backtest(rsi_os=rsi_os, rsi_ob=rsi_ob, skip_hours=[12])
        # Simple risk-adjusted return (return / maxdd)
        sharpe_approx = ret / max_dd if max_dd > 0 else 0
        results.append((rsi_os, rsi_ob, ret, trades, wr, max_dd, sharpe_approx, prof_yrs, yearly))
        print(f"{rsi_os}/{rsi_ob:>2}     | +{ret:>7.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {sharpe_approx:>8.2f}")

    # Find best by Sharpe-like metric (return/maxdd)
    best_sharpe = max(results, key=lambda x: x[6])
    # Find best by return
    best_return = max(results, key=lambda x: x[2])
    # Find best by MaxDD
    best_dd = min(results, key=lambda x: x[5])

    print(f"\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    print(f"\nBest by Return: RSI {best_return[0]}/{best_return[1]}")
    print(f"  Return: +{best_return[2]:.1f}% | MaxDD: {best_return[5]:.1f}% | Ret/DD: {best_return[6]:.2f}")

    print(f"\nBest by MaxDD: RSI {best_dd[0]}/{best_dd[1]}")
    print(f"  Return: +{best_dd[2]:.1f}% | MaxDD: {best_dd[5]:.1f}% | Ret/DD: {best_dd[6]:.2f}")

    print(f"\nBest by Risk-Adjusted (Ret/DD): RSI {best_sharpe[0]}/{best_sharpe[1]}")
    print(f"  Return: +{best_sharpe[2]:.1f}% | MaxDD: {best_sharpe[5]:.1f}% | Ret/DD: {best_sharpe[6]:.2f}")

    # Show yearly breakdown for best configs
    print(f"\n" + "-" * 80)
    print(f"Yearly breakdown for Best Risk-Adjusted (RSI {best_sharpe[0]}/{best_sharpe[1]}):")
    for year in sorted(best_sharpe[8].keys()):
        pnl = best_sharpe[8][year]
        status = '  ' if pnl > 0 else ' (LOSS)'
        print(f"  {year}: ${pnl:>+12,.2f}{status}")


if __name__ == "__main__":
    main()
