"""
Partial Profit Taking Test
==========================
Test taking partial profit at first target, letting rest run to full TP.

Theory:
- Take 50% profit at 1.5x ATR (lock in gains)
- Let remaining 50% run to full TP (2.4-3.6x ATR)
- Reduces psychological pressure, secures some profit

This changes POSITION MANAGEMENT, not entry.

Sources:
- https://www.babypips.com/learn/forex/scaling-in-and-out
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
    print('PARTIAL PROFIT TAKING TEST')
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

    def backtest(partial_tp=False, partial_pct=0.5, partial_mult=1.5):
        """
        Backtest with partial profit taking.

        Args:
            partial_tp: If True, take partial profit at first target
            partial_pct: Percentage of position to close at first target (0.5 = 50%)
            partial_mult: ATR multiplier for first profit target
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        partial_wins = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                exit_trade = False
                pnl = 0

                # Check partial profit first
                if partial_tp and not position.get('partial_taken', False):
                    partial_target = position['entry'] + position['atr'] * partial_mult * position['dir']

                    if position['dir'] == 1:
                        if row['high'] >= partial_target:
                            # Take partial profit
                            partial_pnl = (partial_target - position['entry']) * position['size'] * partial_pct
                            balance += partial_pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + partial_pnl
                            position['partial_taken'] = True
                            position['size'] *= (1 - partial_pct)  # Reduce remaining size
                            partial_wins += 1
                    else:
                        if row['low'] <= partial_target:
                            partial_pnl = (position['entry'] - partial_target) * position['size'] * partial_pct
                            balance += partial_pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + partial_pnl
                            position['partial_taken'] = True
                            position['size'] *= (1 - partial_pct)
                            partial_wins += 1

                # Check max holding
                if (i - position['entry_idx']) >= MAX_HOLDING:
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
                            if not position.get('partial_taken', False):
                                losses += 1
                            else:
                                # Partial was taken, so overall might be positive
                                wins += 1 if balance > position.get('start_balance', balance) else 0
                                losses += 1 if balance <= position.get('start_balance', balance) else 0
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            exit_trade = True
                            wins += 1
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            exit_trade = True
                            if not position.get('partial_taken', False):
                                losses += 1
                            else:
                                wins += 1 if balance > position.get('start_balance', balance) else 0
                                losses += 1 if balance <= position.get('start_balance', balance) else 0
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            exit_trade = True
                            wins += 1

                if exit_trade:
                    balance += pnl
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    if not partial_tp:
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
                        'size': size, 'entry_idx': i, 'atr': atr,
                        'start_balance': balance, 'partial_taken': False
                    }

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, yearly_pnl, partial_wins

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest(partial_tp=False)
    print(f'\n   v3.7 Baseline (no partial): +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Test partial profit configurations
    print('\n2. PARTIAL PROFIT CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<40} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Partials":^10}')
    print('   ' + '-' * 105)

    configs = [
        (0.25, 1.0, '25% at 1.0x ATR'),
        (0.25, 1.5, '25% at 1.5x ATR'),
        (0.50, 1.0, '50% at 1.0x ATR'),
        (0.50, 1.5, '50% at 1.5x ATR'),
        (0.50, 2.0, '50% at 2.0x ATR'),
        (0.75, 1.0, '75% at 1.0x ATR'),
        (0.75, 1.5, '75% at 1.5x ATR'),
    ]

    for partial_pct, partial_mult, name in configs:
        ret, trades, wr, max_dd, _, partials = backtest(
            partial_tp=True, partial_pct=partial_pct, partial_mult=partial_mult
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<40} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {partials:>8}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. PARTIAL PROFIT CONCEPT:
   - Take some profit early, let rest run
   - Reduces overall risk exposure
   - But also reduces profit potential

2. TRADE-OFFS:
   - Higher partial % = more locked profit but less upside
   - Earlier partial target = more secure but less profit
   - Need to balance security vs opportunity

3. RECOMMENDATION:
   - If partial profit improves risk-adjusted returns -> Consider
   - If not -> Keep full position until TP/SL
""")


if __name__ == "__main__":
    main()
