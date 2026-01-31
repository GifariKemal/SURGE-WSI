"""
Test Time-Based Exit Strategies
================================
Based on research from arXiv papers on Optimal Mean Reversion Trading

Techniques tested:
1. Time-based exit - Close after X hours if no SL/TP
2. Maximum holding period - Force close after N bars
3. Minimum holding period - Don't exit early
4. Adaptive time stop - Based on ATR percentile
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

    # Baseline parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35

    def backtest_baseline():
        """v3.5 BASELINE"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
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
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    def backtest_time_exit(max_hours):
        """
        Time-based exit - Close after max_hours if no SL/TP
        Based on research showing mean reversion has optimal holding period
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        time_exits = 0
        time_exit_profit = 0
        time_exit_loss = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                # Check time-based exit first
                bars_held = i - position['entry_idx']
                if bars_held >= max_hours:
                    # Force close at current price
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']

                    balance += pnl
                    if pnl > 0:
                        wins += 1
                        time_exit_profit += 1
                    else:
                        losses += 1
                        time_exit_loss += 1
                    time_exits += 1
                    position = None
                else:
                    # Normal SL/TP check
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
        return ret, trades, wr, max_dd, time_exits, time_exit_profit, time_exit_loss

    def backtest_breakeven_move(be_trigger_atr=1.0):
        """
        Move SL to breakeven after price moves X ATR in our favor
        Don't use trailing - just move to breakeven once
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        be_triggered = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                # Check for breakeven trigger
                if not position.get('be_done'):
                    if position['dir'] == 1:
                        if row['high'] - position['entry'] >= position['atr'] * be_trigger_atr:
                            position['sl'] = position['entry'] + position['atr'] * 0.1  # Small profit
                            position['be_done'] = True
                            be_triggered += 1
                    else:
                        if position['entry'] - row['low'] >= position['atr'] * be_trigger_atr:
                            position['sl'] = position['entry'] - position['atr'] * 0.1
                            position['be_done'] = True
                            be_triggered += 1

                # Normal SL/TP check
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        balance += pnl
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        balance += (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        balance += pnl
                        if pnl > 0:
                            wins += 1
                        else:
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
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'atr': atr, 'be_done': False}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, be_triggered

    def backtest_tighter_sl(sl_mult):
        """
        Test tighter SL to cut losses faster
        Research suggests: "cut losses quickly"
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
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
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp

                    # Tighter SL
                    sl = entry - atr * sl_mult if signal == 1 else entry + atr * sl_mult
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    print('=' * 100)
    print('TIME-BASED EXIT AND SL OPTIMIZATION TESTS')
    print('=' * 100)

    print('\n' + '-' * 100)
    print('1. v3.5 BASELINE')
    print('-' * 100)
    ret, trades, wr, max_dd = backtest_baseline()
    baseline_ret = ret
    print(f"   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%")

    print('\n' + '-' * 100)
    print('2. TIME-BASED EXIT (Close after X hours)')
    print('-' * 100)

    for hours in [6, 12, 24, 36, 48, 72]:
        ret, trades, wr, max_dd, time_exits, te_win, te_loss = backtest_time_exit(hours)
        diff = ret - baseline_ret
        marker = ' <-- BETTER!' if diff > 0 else ''
        print(f"   Max {hours}h: +{ret:.1f}% ({diff:+.1f}) | WR: {wr:.1f}% | Time exits: {time_exits} (W:{te_win}/L:{te_loss}){marker}")

    print('\n' + '-' * 100)
    print('3. BREAKEVEN MOVE (Move SL to entry after X ATR profit)')
    print('-' * 100)

    for trigger in [0.5, 1.0, 1.5, 2.0]:
        ret, trades, wr, max_dd, be_count = backtest_breakeven_move(trigger)
        diff = ret - baseline_ret
        marker = ' <-- BETTER!' if diff > 0 else ''
        print(f"   BE at {trigger}x ATR: +{ret:.1f}% ({diff:+.1f}) | WR: {wr:.1f}% | BE triggered: {be_count}{marker}")

    print('\n' + '-' * 100)
    print('4. TIGHTER SL (Cut losses faster)')
    print('-' * 100)

    for sl in [1.0, 1.2, 1.3, 1.5, 1.8, 2.0]:
        ret, trades, wr, max_dd = backtest_tighter_sl(sl)
        diff = ret - baseline_ret
        marker = ' <-- BETTER!' if diff > 0 else ''
        print(f"   SL {sl}x ATR: +{ret:.1f}% ({diff:+.1f}) | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)


if __name__ == "__main__":
    main()
