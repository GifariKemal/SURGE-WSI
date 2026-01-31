"""
Session-Specific Parameters Test
================================
Test different SL/TP parameters for different trading sessions.

Theory:
- Asian session: Lower volatility, need tighter targets
- London session: Higher volatility, can afford wider targets
- US session: Mixed, depends on news

This changes PARAMETERS based on time, not filtering.

Sources:
- https://www.babypips.com/learn/forex/forex-trading-sessions
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
    print('SESSION-SPECIFIC PARAMETERS TEST')
    print('=' * 100)
    print('   Sessions (UTC):')
    print('   - Asian: 00:00-07:00')
    print('   - London: 07:00-15:00')
    print('   - US: 13:00-22:00')
    print('   - Overlap (London/US): 13:00-15:00')

    # v3.7 parameters
    RSI_OS = 42
    RSI_OB = 58
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(session_params=None):
        """
        Backtest with session-specific SL/TP parameters.

        session_params: dict with keys 'asian', 'london', 'us'
                       Each value is (sl_mult, tp_mult)
        If None, use v3.7 defaults for all sessions.
        """
        # v3.7 defaults
        default_sl = 1.5
        default_tp_low = 2.4
        default_tp_med = 3.0
        default_tp_high = 3.6
        time_tp_bonus = 0.35

        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        session_stats = {'asian': [0, 0], 'london': [0, 0], 'us': [0, 0]}

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            # Determine session
            if hour < 7:
                session = 'asian'
            elif hour < 15:
                session = 'london'
            else:
                session = 'us'

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
                        session_stats[position['session']][0] += 1
                    else:
                        losses += 1
                        session_stats[position['session']][1] += 1
                    position = None
                else:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            session_stats[position['session']][1] += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            session_stats[position['session']][0] += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            session_stats[position['session']][1] += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            session_stats[position['session']][0] += 1
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

                    # Get session-specific parameters
                    if session_params and session in session_params:
                        sl_mult, tp_mult = session_params[session]
                    else:
                        sl_mult = default_sl
                        base_tp = default_tp_low if atr_pct < 40 else (default_tp_high if atr_pct > 60 else default_tp_med)
                        if 12 <= hour < 16:
                            tp_mult = base_tp + time_tp_bonus
                        else:
                            tp_mult = base_tp

                    sl = entry - atr * sl_mult if signal == 1 else entry + atr * sl_mult
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {
                        'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp,
                        'size': size, 'entry_idx': i, 'session': session
                    }

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd, yearly_pnl, session_stats

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, baseline_stats = backtest()
    print(f'\n   v3.7 Baseline: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # Show session breakdown
    print('\n   Session breakdown (v3.7):')
    for sess, (w, l) in baseline_stats.items():
        if w + l > 0:
            wr = w / (w + l) * 100
            print(f'   - {sess.capitalize()}: {w+l} trades, WR: {wr:.1f}%')

    # 2. Test session-specific parameters
    print('\n2. SESSION-SPECIFIC PARAMETERS')
    print('-' * 80)
    print(f'   {"Config":<50} | {"Return":^10} | {"Diff":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        # (asian_sl_tp, london_sl_tp, us_sl_tp, name)
        # Lower SL/TP for Asian (lower vol), higher for London/US
        ({'asian': (1.2, 2.0), 'london': (1.5, 3.0), 'us': (1.5, 3.0)}, 'Tighter Asian (1.2/2.0)'),
        ({'asian': (1.3, 2.2), 'london': (1.5, 3.0), 'us': (1.5, 3.0)}, 'Slightly tighter Asian (1.3/2.2)'),
        ({'asian': (1.5, 2.4), 'london': (1.6, 3.2), 'us': (1.5, 3.0)}, 'Wider London (1.6/3.2)'),
        ({'asian': (1.5, 2.4), 'london': (1.8, 3.6), 'us': (1.5, 3.0)}, 'Much wider London (1.8/3.6)'),
        ({'asian': (1.5, 2.4), 'london': (1.5, 3.0), 'us': (1.6, 3.2)}, 'Wider US (1.6/3.2)'),
        ({'asian': (1.2, 2.0), 'london': (1.6, 3.2), 'us': (1.6, 3.2)}, 'Tight Asian, Wide London/US'),
        ({'asian': (1.8, 3.0), 'london': (1.2, 2.4), 'us': (1.2, 2.4)}, 'Inverted (wide Asian, tight London/US)'),
        ({'asian': (1.5, 2.8), 'london': (1.5, 3.2), 'us': (1.5, 2.8)}, 'Wider London TP only'),
        ({'asian': (1.3, 2.4), 'london': (1.5, 3.0), 'us': (1.7, 3.2)}, 'Progressive SL increase'),
    ]

    for params, name in configs:
        ret, trades, wr, max_dd, _, stats = backtest(session_params=params)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<50} | +{ret:>7.1f}% | {diff:>+6.1f}% | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Analyze which session parameters might help
    print('\n3. INDIVIDUAL SESSION PARAMETER CHANGES')
    print('-' * 80)

    # Test just changing Asian
    print('\n   Changing Asian session parameters only:')
    print(f'   {"Asian SL/TP":<20} | {"Return":^10} | {"Diff":^8} | {"WR":^8}')
    print('   ' + '-' * 55)

    for sl, tp in [(1.0, 2.0), (1.2, 2.2), (1.3, 2.4), (1.5, 2.4), (1.5, 2.8), (1.8, 3.0)]:
        ret, trades, wr, max_dd, _, _ = backtest(session_params={'asian': (sl, tp)})
        diff = ret - baseline_ret
        print(f'   SL={sl}, TP={tp}      | +{ret:>7.1f}% | {diff:>+6.1f}% | {wr:>5.1f}%')

    # Test just changing London
    print('\n   Changing London session parameters only:')
    print(f'   {"London SL/TP":<20} | {"Return":^10} | {"Diff":^8} | {"WR":^8}')
    print('   ' + '-' * 55)

    for sl, tp in [(1.2, 2.4), (1.5, 2.8), (1.5, 3.2), (1.6, 3.2), (1.8, 3.6), (2.0, 4.0)]:
        ret, trades, wr, max_dd, _, _ = backtest(session_params={'london': (sl, tp)})
        diff = ret - baseline_ret
        print(f'   SL={sl}, TP={tp}      | +{ret:>7.1f}% | {diff:>+6.1f}% | {wr:>5.1f}%')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. SESSION CHARACTERISTICS:
   - Asian (00-07 UTC): Lower volatility, GBPUSD less active
   - London (07-15 UTC): Highest volatility for GBP pairs
   - US (13-22 UTC): Second highest volatility, overlaps with London

2. PARAMETER ADJUSTMENTS:
   - Tighter SL/TP for low-volatility sessions may lock profits earlier
   - Wider SL/TP for high-volatility sessions may capture bigger moves
   - But: ATR already adapts dynamically!

3. RECOMMENDATION:
   - v3.7's ATR-based dynamic SL/TP already adapts to volatility
   - Session-specific fixed parameters may not improve further
   - Keep the ATR-adaptive approach unless significant improvement found
""")


if __name__ == "__main__":
    main()
