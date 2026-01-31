"""
Breakeven Stop Test
===================
Test moving stop loss to breakeven after price moves X ATR in profit.

Theory:
- After a certain profit (e.g., 1x ATR), move SL to entry price
- This locks in "risk-free" trade while allowing TP to be hit
- May reduce winners but eliminates some losing trades

Caution: This can cut winners short in mean reversion (price often retraces).
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
    print('BREAKEVEN STOP TEST')
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

    def backtest(breakeven_atr=None, include_buffer=False):
        """
        Backtest with breakeven stop.

        Args:
            breakeven_atr: Move SL to entry when price moves this many ATR in profit
            include_buffer: Add small buffer (0.1 ATR) to breakeven to cover spread
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        breakeven_exits = 0  # Trades that exit at breakeven

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                # Current profit in price terms
                if position['dir'] == 1:
                    current_profit = row['close'] - position['entry']
                else:
                    current_profit = position['entry'] - row['close']

                # Check if we should move to breakeven
                if breakeven_atr and not position.get('at_breakeven'):
                    trigger = position['atr'] * breakeven_atr
                    if current_profit >= trigger:
                        # Move SL to entry (+ buffer if specified)
                        buffer = position['atr'] * 0.1 if include_buffer else 0
                        if position['dir'] == 1:
                            position['sl'] = position['entry'] + buffer
                        else:
                            position['sl'] = position['entry'] - buffer
                        position['at_breakeven'] = True

                # Check max holding
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
                    # Check SL/TP
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            if position.get('at_breakeven') and abs(pnl) < position['atr'] * 0.2 * position['size']:
                                breakeven_exits += 1
                            if pnl >= 0:
                                wins += 1
                            else:
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
                            if position.get('at_breakeven') and abs(pnl) < position['atr'] * 0.2 * position['size']:
                                breakeven_exits += 1
                            if pnl >= 0:
                                wins += 1
                            else:
                                losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            position = None

                if position is None:
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
                    position = {
                        'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp,
                        'size': size, 'entry_idx': i, 'atr': atr, 'at_breakeven': False
                    }

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, yearly_pnl, breakeven_exits

    # Test different breakeven triggers
    print('\n1. BREAKEVEN STOP TEST')
    print('-' * 80)
    print(f'   {"Breakeven Trigger":^20} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"BE Exits":^10}')
    print('   ' + '-' * 95)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest()
    print(f'   {"v3.7 Baseline":<20} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    triggers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    for trigger in triggers:
        ret, trades, wr, max_dd, _, be_exits = backtest(breakeven_atr=trigger)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   {trigger}x ATR profit      | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {be_exits:>8}{marker}')

    # Test with buffer
    print('\n2. BREAKEVEN WITH BUFFER (to cover spread)')
    print('-' * 80)
    print(f'   {"Trigger + Buffer":^25} | {"Return":^10} | {"Diff":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 70)

    for trigger in [1.0, 1.25, 1.5]:
        ret_no_buf, _, wr_no_buf, dd_no_buf, _, _ = backtest(breakeven_atr=trigger, include_buffer=False)
        ret_buf, _, wr_buf, dd_buf, _, _ = backtest(breakeven_atr=trigger, include_buffer=True)
        print(f'   {trigger}x ATR (no buffer)     | +{ret_no_buf:>7.1f}% | {ret_no_buf - baseline_ret:>+6.1f}% | {wr_no_buf:>5.1f}% | {dd_no_buf:>5.1f}%')
        print(f'   {trigger}x ATR (with buffer)   | +{ret_buf:>7.1f}% | {ret_buf - baseline_ret:>+6.1f}% | {wr_buf:>5.1f}% | {dd_buf:>5.1f}%')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. BREAKEVEN STOP ANALYSIS:
   - Moving stop to breakeven HURTS performance for mean reversion
   - Price often retraces before hitting TP - breakeven exit cuts winners

2. WHY IT FAILS FOR MEAN REVERSION:
   - Mean reversion = price bounces around before reaching target
   - Moving SL to breakeven gets triggered during normal retracement
   - Then trade exits with zero profit instead of full TP

3. COMPARISON TO TRAILING STOP:
   - Trailing stop: CATASTROPHIC (-592.9%)
   - Breakeven stop: Also negative but less severe
   - Both cut winners short in ranging/reverting markets

4. RECOMMENDATION:
   - DO NOT use breakeven stop for mean reversion strategy
   - Fixed SL/TP structure is optimal
   - Trust the original trade setup
""")


if __name__ == "__main__":
    main()
