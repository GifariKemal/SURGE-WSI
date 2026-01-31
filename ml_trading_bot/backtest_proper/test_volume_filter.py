"""
Tick Volume Filter Test
=======================
Test if filtering by tick volume improves signal quality.

Theory:
- Higher volume = more market participation = better signal
- Low volume = choppy price action = more false signals
- Use tick volume (available in forex) as proxy for real volume

Sources:
- https://www.fxpro.com/help-section/education/beginners/articles/trading-with-metatraders-forex-volume-indicator
- https://fxssi.com/forex-volumes-indicator
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

    print('=' * 100)
    print('TICK VOLUME FILTER TEST (using range as proxy)')
    print('=' * 100)

    # Use range as proxy for volume (higher range = more activity)
    # This is a reasonable proxy since wider range = more market activity
    df['tick_volume'] = (df['high'] - df['low']) / df['close'] * 1000000
    print('\n   Note: Using price range as volume proxy (no tick volume in database)')

    # Volume statistics
    print('\n1. TICK VOLUME STATISTICS')
    print('-' * 80)
    print(f'   Mean volume: {df["tick_volume"].mean():,.0f}')
    print(f'   Median volume: {df["tick_volume"].median():,.0f}')
    print(f'   Std volume: {df["tick_volume"].std():,.0f}')

    # Calculate volume percentile
    df['vol_pct'] = df['tick_volume'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

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

    # Volume by hour
    print('\n2. VOLUME BY HOUR')
    print('-' * 80)

    for hour in range(7, 22):
        hour_data = df[df['hour'] == hour]
        avg_vol = hour_data['tick_volume'].mean()
        print(f'   {hour:02d}:00 UTC: {avg_vol:>12,.0f}')

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

    def backtest(min_vol_pct=None, max_vol_pct=None):
        """Backtest with volume filter"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        filtered_by_vol = 0

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
                if hour in SKIP_HOURS:
                    continue

                # Volume filter
                vol_pct = row['vol_pct']
                if min_vol_pct and vol_pct < min_vol_pct:
                    filtered_by_vol += 1
                    continue
                if max_vol_pct and vol_pct > max_vol_pct:
                    filtered_by_vol += 1
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
        return ret, trades, wr, max_dd, yearly_pnl, filtered_by_vol

    # 3. Test volume filters
    print('\n3. BACKTEST WITH VOLUME FILTER')
    print('-' * 80)
    print(f'   {"Filter":^25} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 95)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest()
    print(f'   {"v3.7 Baseline":<25} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    filters = [
        (20, None, 'Vol > 20th pct'),
        (30, None, 'Vol > 30th pct'),
        (40, None, 'Vol > 40th pct'),
        (50, None, 'Vol > 50th pct (median)'),
        (None, 80, 'Vol < 80th pct'),
        (20, 80, 'Vol 20-80th pct'),
        (30, 70, 'Vol 30-70th pct'),
    ]

    for min_v, max_v, name in filters:
        ret, trades, wr, max_dd, _, filtered = backtest(min_vol_pct=min_v, max_vol_pct=max_v)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<25} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. VOLUME FILTER ANALYSIS:
   - Tick volume can indicate market activity
   - Higher volume = more participation = potentially better signals

2. FILTER IMPACT:
   - Volume filters tend to reduce trade count
   - Win rate may or may not improve
   - Overall return often decreases due to fewer trades

3. RECOMMENDATION:
   - Volume filter may help in some markets
   - For GBPUSD H1, our ATR filter already captures volatility
   - Test if volume adds value beyond ATR filter
""")


if __name__ == "__main__":
    main()
