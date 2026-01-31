"""
RSI Mean Exit Test
==================
Test exiting when RSI returns to neutral (50) instead of fixed TP.

Theory:
- Mean reversion = price returns to mean
- RSI 50 = neutral, no more oversold/overbought
- Exit at RSI 50 may capture the "reversion" more accurately

This changes the EXIT strategy, not entry.

Sources:
- https://www.quantifiedstrategies.com/trading-exit-strategies/
- https://www.algomatictrading.com/post/13-essential-exits-for-mean-reversion-strategies
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

    print('=' * 100)
    print('RSI MEAN EXIT TEST')
    print('=' * 100)

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

    def backtest(exit_rsi_mean=False, rsi_exit_level=50, use_tp_as_backup=True):
        """
        Backtest with RSI mean exit.

        Args:
            exit_rsi_mean: If True, exit when RSI crosses the exit level
            rsi_exit_level: RSI level to exit (default 50 = neutral)
            use_tp_as_backup: Still use TP if RSI doesn't revert
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        rsi_exits = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']
            rsi = row['rsi']

            if weekday >= 5:
                continue

            if position:
                exit_trade = False
                pnl = 0

                # Check RSI mean exit first
                if exit_rsi_mean:
                    if position['dir'] == 1 and rsi >= rsi_exit_level:
                        # Long position: RSI crossed above exit level
                        pnl = (row['close'] - position['entry']) * position['size']
                        exit_trade = True
                        rsi_exits += 1
                    elif position['dir'] == -1 and rsi <= rsi_exit_level:
                        # Short position: RSI crossed below exit level
                        pnl = (position['entry'] - row['close']) * position['size']
                        exit_trade = True
                        rsi_exits += 1

                # Check max holding
                if not exit_trade and (i - position['entry_idx']) >= MAX_HOLDING:
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    exit_trade = True

                # Check SL/TP
                if not exit_trade:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            exit_trade = True
                        elif use_tp_as_backup and row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            exit_trade = True
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            exit_trade = True
                        elif use_tp_as_backup and row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            exit_trade = True

                if exit_trade:
                    balance += pnl
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
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
        return ret, trades, wr, max_dd, yearly_pnl, rsi_exits

    # Test different exit strategies
    print('\n1. RSI MEAN EXIT STRATEGIES')
    print('-' * 80)
    print(f'   {"Strategy":^35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"RSI Exits":^10}')
    print('   ' + '-' * 105)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest(exit_rsi_mean=False)
    print(f'   {"v3.7 Baseline (TP only)":<35} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    strategies = [
        (50, True, 'RSI 50 + TP backup'),
        (50, False, 'RSI 50 only (no TP)'),
        (48, True, 'RSI 48 + TP backup'),
        (52, True, 'RSI 52 + TP backup'),
        (45, True, 'RSI 45 + TP backup'),
        (55, True, 'RSI 55 + TP backup'),
    ]

    for rsi_level, use_tp, name in strategies:
        ret, trades, wr, max_dd, _, rsi_exits = backtest(
            exit_rsi_mean=True, rsi_exit_level=rsi_level, use_tp_as_backup=use_tp
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {rsi_exits:>8}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. RSI MEAN EXIT CONCEPT:
   - Exit when RSI returns to neutral (50)
   - Captures the "mean reversion" directly
   - May exit earlier than fixed TP

2. TRADE-OFFS:
   - RSI 50 exit may be too early (misses further profit)
   - RSI 50 exit may be too late (price already reversed)
   - TP backup ensures we don't hold losers forever

3. RECOMMENDATION:
   - Compare RSI exit vs fixed TP
   - If RSI exit improves risk-adjusted returns -> Consider
   - If not -> Keep fixed TP (simpler)
""")


if __name__ == "__main__":
    main()
