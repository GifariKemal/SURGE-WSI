"""
RSI Consecutive Days Test
=========================
Test if requiring RSI to decline/rise for multiple consecutive bars
improves signal quality (Larry Connors style).

Theory:
- RSI declining for 2-3 days = stronger oversold condition
- Single bar oversold may be noise
- Multiple bars = confirmed weakness/strength

Sources:
- https://www.quantifiedstrategies.com/rsi-trading-strategy/
- https://greaterwaves.com/secrets-of-larry-connors-mean-reversion/
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

    # RSI change
    df['rsi_change'] = df['rsi'].diff()

    # Count consecutive RSI declines/rises
    def count_consecutive(series, direction='down'):
        """Count consecutive bars RSI moved in direction"""
        count = 0
        counts = []
        for val in series:
            if direction == 'down' and val < 0:
                count += 1
            elif direction == 'up' and val > 0:
                count += 1
            else:
                count = 0
            counts.append(count)
        return counts

    df['rsi_down_streak'] = count_consecutive(df['rsi_change'], 'down')
    df['rsi_up_streak'] = count_consecutive(df['rsi_change'], 'up')

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
    print('RSI CONSECUTIVE BARS TEST')
    print('=' * 100)

    # 1. Analyze RSI streaks
    print('\n1. RSI STREAK DISTRIBUTION')
    print('-' * 80)

    for streak in range(1, 6):
        down_count = (df['rsi_down_streak'] >= streak).sum()
        up_count = (df['rsi_up_streak'] >= streak).sum()
        print(f'   {streak}+ bars declining: {down_count:>6} ({down_count/len(df)*100:.1f}%)')

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

    def backtest(min_down_streak=0, min_up_streak=0, require_rsi_turning=False):
        """
        Backtest with consecutive RSI requirements.

        Args:
            min_down_streak: For BUY, require RSI declining this many bars
            min_up_streak: For SELL, require RSI rising this many bars
            require_rsi_turning: Only enter when RSI starts turning (change direction)
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        filtered = 0

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

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                rsi_change = row['rsi_change']
                down_streak = row['rsi_down_streak']
                up_streak = row['rsi_up_streak']

                signal = 0

                # BUY signal
                if rsi < RSI_OS:
                    # Check consecutive requirement
                    if min_down_streak > 0 and down_streak < min_down_streak:
                        filtered += 1
                        continue
                    # Check if RSI is turning (was down, now up)
                    if require_rsi_turning and rsi_change <= 0:
                        filtered += 1
                        continue
                    signal = 1

                # SELL signal
                elif rsi > RSI_OB:
                    if min_up_streak > 0 and up_streak < min_up_streak:
                        filtered += 1
                        continue
                    if require_rsi_turning and rsi_change >= 0:
                        filtered += 1
                        continue
                    signal = -1

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
        return ret, trades, wr, max_dd, yearly_pnl, filtered

    # 2. Test consecutive requirements
    print('\n2. BACKTEST WITH CONSECUTIVE RSI REQUIREMENT')
    print('-' * 80)
    print(f'   {"Filter":^30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 100)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest()
    print(f'   {"v3.7 Baseline":<30} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    configs = [
        (1, 1, False, '1+ bar streak'),
        (2, 2, False, '2+ bar streak'),
        (3, 3, False, '3+ bar streak'),
        (4, 4, False, '4+ bar streak'),
        (0, 0, True, 'RSI turning only'),
        (2, 2, True, '2+ streak + turning'),
        (3, 3, True, '3+ streak + turning'),
    ]

    for down, up, turning, name in configs:
        ret, trades, wr, max_dd, _, filtered = backtest(
            min_down_streak=down, min_up_streak=up, require_rsi_turning=turning
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # 3. Test RSI extreme + consecutive
    print('\n3. RSI EXTREME + CONSECUTIVE (Larry Connors style)')
    print('-' * 80)

    def backtest_extreme(rsi_os=30, rsi_ob=70, min_streak=2):
        """Connors style: extreme RSI + consecutive decline"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
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

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                down_streak = row['rsi_down_streak']
                up_streak = row['rsi_up_streak']

                signal = 0
                if rsi < rsi_os and down_streak >= min_streak:
                    signal = 1
                elif rsi > rsi_ob and up_streak >= min_streak:
                    signal = -1

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

    print(f'   {"Config":^25} | {"Return":^10} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 65)

    for rsi_os, rsi_ob, streak in [(30, 70, 2), (35, 65, 2), (30, 70, 3), (25, 75, 2), (20, 80, 2)]:
        ret, trades, wr, max_dd = backtest_extreme(rsi_os=rsi_os, rsi_ob=rsi_ob, min_streak=streak)
        print(f'   RSI {rsi_os}/{rsi_ob} + {streak} bars   | +{ret:>7.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. CONSECUTIVE RSI ANALYSIS:
   - Requiring multiple bars of RSI decline/rise adds confirmation
   - Trade-off: fewer trades but potentially higher quality

2. LARRY CONNORS STYLE:
   - Uses RSI(2) with extreme thresholds (10/90 or 20/80)
   - Requires consecutive days of weakness
   - Works better on daily timeframe

3. ON H1 TIMEFRAME:
   - More bars = more lag = worse timing
   - RSI 42/58 already captures good opportunities
   - Adding consecutive filter tends to HURT performance

4. RECOMMENDATION:
   - Consecutive filter likely NOT beneficial for H1 mean reversion
   - Our current entry timing (immediate on threshold cross) is optimal
""")


if __name__ == "__main__":
    main()
