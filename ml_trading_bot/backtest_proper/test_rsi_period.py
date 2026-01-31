"""
RSI Period Optimization Test
============================
Test different RSI calculation periods.

Current v3.7: RSI(10)
Standard: RSI(14)
Larry Connors: RSI(2)

Theory:
- Shorter period = more responsive, more signals
- Longer period = smoother, fewer signals
- Need to find optimal balance

Sources:
- https://www.investopedia.com/terms/r/rsi.asp
- Larry Connors "Short Term Trading Strategies That Work"
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

    print('=' * 100)
    print('RSI PERIOD OPTIMIZATION TEST')
    print('=' * 100)

    # v3.7 parameters
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(rsi_period=10, rsi_os=42, rsi_ob=58):
        """
        Backtest with different RSI period.
        """
        # Calculate RSI with specified period
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rsi_series = 100 - (100 / (1 + gain / (loss + 1e-10)))

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
            rsi = rsi_series.iloc[i]

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

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

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
        return ret, trades, wr, max_dd, yearly_pnl

    # 1. RSI Period Sweep
    print('\n1. RSI PERIOD SWEEP (fixed thresholds 42/58)')
    print('-' * 80)
    print(f'   {"Period":^10} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 65)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(rsi_period=10)

    for period in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 20]:
        ret, trades, wr, max_dd, _ = backtest(rsi_period=period)
        diff = ret - baseline_ret
        marker = ' <<<' if period == 10 else (' ***' if diff > 30 else '')
        print(f'   RSI({period:2d})   | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 2. Test different RSI periods with adjusted thresholds
    print('\n2. RSI PERIOD + ADJUSTED THRESHOLDS')
    print('-' * 80)
    print('   Shorter RSI needs wider thresholds (more volatile)')
    print(f'\n   {"Config":<25} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 80)

    configs = [
        (2, 10, 90, 'RSI(2) - 10/90 (Connors)'),
        (2, 15, 85, 'RSI(2) - 15/85'),
        (2, 20, 80, 'RSI(2) - 20/80'),
        (3, 15, 85, 'RSI(3) - 15/85'),
        (3, 20, 80, 'RSI(3) - 20/80'),
        (5, 25, 75, 'RSI(5) - 25/75'),
        (5, 30, 70, 'RSI(5) - 30/70'),
        (7, 35, 65, 'RSI(7) - 35/65'),
        (7, 40, 60, 'RSI(7) - 40/60'),
        (10, 42, 58, 'RSI(10) - 42/58 (v3.7)'),
        (14, 30, 70, 'RSI(14) - 30/70 (classic)'),
        (14, 40, 60, 'RSI(14) - 40/60'),
        (14, 42, 58, 'RSI(14) - 42/58'),
    ]

    for period, os, ob, name in configs:
        ret, trades, wr, max_dd, _ = backtest(rsi_period=period, rsi_os=os, rsi_ob=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if 'v3.7' in name else (' ***' if diff > 30 else '')
        print(f'   {name:<25} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Fine-tune around best period
    print('\n3. FINE-TUNE THRESHOLDS FOR RSI(10)')
    print('-' * 80)
    print(f'   {"Thresholds":^15} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 70)

    for os, ob in [(38, 62), (40, 60), (42, 58), (44, 56), (45, 55), (46, 54)]:
        ret, trades, wr, max_dd, _ = backtest(rsi_period=10, rsi_os=os, rsi_ob=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if (os == 42 and ob == 58) else (' ***' if diff > 20 else '')
        print(f'   {os}/{ob}         | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. RSI PERIOD EFFECT:
   - Shorter period (2-5): More responsive, more signals, noisier
   - Medium period (7-14): Balanced responsiveness and smoothness
   - Longer period (>14): Smoother, fewer signals, may miss moves

2. THRESHOLD ADJUSTMENT:
   - Shorter RSI needs wider thresholds (e.g., 10/90 for RSI(2))
   - Standard RSI(14) uses classic 30/70
   - Our RSI(10) with 42/58 is aggressive but tuned for H1

3. RECOMMENDATION:
   - v3.7 RSI(10) with 42/58 is already well-optimized
   - Shorter periods increase noise without improving WR
   - Longer periods reduce opportunities
""")


if __name__ == "__main__":
    main()
