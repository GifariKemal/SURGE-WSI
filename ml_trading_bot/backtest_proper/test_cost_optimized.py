"""
Optimize Strategy for Transaction Costs
=======================================
Test techniques to improve strategy under realistic costs:
1. Wider TP to improve reward:cost ratio
2. Tighter RSI thresholds (fewer but higher quality trades)
3. Only trade during low-spread hours (overlap)
4. Larger ATR multiplier for TP
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


def get_session_spread(hour):
    """Get typical spread based on trading session"""
    if hour >= 22 or hour < 7:
        return 2.0, 1.5
    if 7 <= hour < 12:
        return 0.6, 0.3
    if 12 <= hour < 16:
        return 0.4, 0.2
    if 16 <= hour < 22:
        return 0.8, 0.4
    return 1.0, 0.5


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

    # RSI (SMA-based)
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

    def backtest(rsi_os=42, rsi_ob=58, tp_mult_base=3.0, tp_low=2.4, tp_high=3.6,
                 time_tp_bonus=0.35, session_start=7, session_end=22, skip_hours=None,
                 with_costs=True, sl_mult=1.5, max_holding=46, min_atr_pct=20, max_atr_pct=80):
        """Configurable backtest with session-based costs"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        if skip_hours is None:
            skip_hours = [12]

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            spread_pips, slip_pips = get_session_spread(hour) if with_costs else (0, 0)

            if position:
                if (i - position['entry_idx']) >= max_holding:
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    if with_costs:
                        pnl -= slip_pips * 0.0001 * position['size']
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
                            if with_costs:
                                pnl -= slip_pips * 0.0001 * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            if with_costs:
                                pnl -= slip_pips * 0.0001 * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            if with_costs:
                                pnl -= slip_pips * 0.0001 * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            if with_costs:
                                pnl -= slip_pips * 0.0001 * position['size']
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
                if hour < session_start or hour >= session_end:
                    continue
                if hour in skip_hours:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < rsi_os else (-1 if rsi > rsi_ob else 0)

                if signal:
                    entry = row['close']
                    if with_costs:
                        if signal == 1:
                            entry += (spread_pips + slip_pips) * 0.0001
                        else:
                            entry -= (spread_pips + slip_pips) * 0.0001

                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_mult_base)
                    if session_start <= hour < 16:
                        tp_mult = base_tp + time_tp_bonus
                    else:
                        tp_mult = base_tp
                    sl = entry - atr * sl_mult if signal == 1 else entry + atr * sl_mult
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
    print('OPTIMIZING STRATEGY FOR TRANSACTION COSTS')
    print('=' * 100)

    # Baseline with costs
    ret, trades, wr, max_dd, prof_yrs, _ = backtest(with_costs=True)
    baseline_ret = ret
    print(f'\nBASELINE (v3.7 with session costs):')
    print(f'Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%')

    # 1. WIDER TP MULTIPLIERS
    print('\n1. WIDER TP MULTIPLIERS (improve reward:cost ratio)')
    print('-' * 80)

    for tp_base, tp_low, tp_high in [(3.0, 2.4, 3.6), (3.5, 2.8, 4.2), (4.0, 3.2, 4.8), (4.5, 3.6, 5.4), (5.0, 4.0, 6.0)]:
        ret, trades, wr, max_dd, prof_yrs, _ = backtest(with_costs=True, tp_mult_base=tp_base, tp_low=tp_low, tp_high=tp_high)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   TP {tp_low}/{tp_base}/{tp_high}: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # 2. TIGHTER RSI THRESHOLDS
    print('\n2. TIGHTER RSI THRESHOLDS (higher quality signals)')
    print('-' * 80)

    for rsi_os, rsi_ob in [(42, 58), (40, 60), (38, 62), (35, 65), (32, 68), (30, 70)]:
        ret, trades, wr, max_dd, prof_yrs, _ = backtest(with_costs=True, rsi_os=rsi_os, rsi_ob=rsi_ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   RSI {rsi_os}/{rsi_ob}: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # 3. OVERLAP-ONLY TRADING
    print('\n3. TRADING SESSION RESTRICTIONS')
    print('-' * 80)

    sessions = [
        (7, 22, [12], 'v3.7 (07-22, skip 12)'),
        (7, 16, [], '07-16 (London only)'),
        (13, 16, [], '13-16 (Overlap only)'),
        (9, 17, [12], '09-17 (Core hours)'),
        (8, 20, [12], '08-20 (Extended)'),
    ]

    for start, end, skip, name in sessions:
        ret, trades, wr, max_dd, prof_yrs, _ = backtest(with_costs=True, session_start=start, session_end=end, skip_hours=skip)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   {name:25s}: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # 4. WIDER SL (reduce SL hits from spread)
    print('\n4. WIDER SL MULTIPLIER (reduce SL hits from spread)')
    print('-' * 80)

    for sl_mult in [1.5, 1.7, 2.0, 2.2, 2.5]:
        ret, trades, wr, max_dd, prof_yrs, _ = backtest(with_costs=True, sl_mult=sl_mult)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   SL {sl_mult}x ATR: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # 5. COMBINATION TEST
    print('\n5. BEST COMBINATIONS')
    print('-' * 80)

    combos = [
        # (rsi_os, rsi_ob, tp_base, tp_low, tp_high, sl_mult, session_start, session_end, skip, name)
        (42, 58, 3.0, 2.4, 3.6, 1.5, 7, 22, [12], 'Baseline'),
        (40, 60, 3.5, 2.8, 4.2, 1.7, 7, 22, [12], 'Wider TP + SL'),
        (38, 62, 4.0, 3.2, 4.8, 2.0, 7, 22, [12], 'Much Wider TP + SL'),
        (35, 65, 4.5, 3.6, 5.4, 2.0, 7, 22, [12], 'Strict RSI + Wide TP'),
        (35, 65, 4.0, 3.2, 4.8, 2.0, 9, 17, [12], 'Strict RSI + Core Hours'),
    ]

    best_ret = -1000
    best_config = None

    for rsi_os, rsi_ob, tp_base, tp_low, tp_high, sl_mult, start, end, skip, name in combos:
        ret, trades, wr, max_dd, prof_yrs, yearly = backtest(
            with_costs=True, rsi_os=rsi_os, rsi_ob=rsi_ob,
            tp_mult_base=tp_base, tp_low=tp_low, tp_high=tp_high,
            sl_mult=sl_mult, session_start=start, session_end=end, skip_hours=skip
        )
        diff = ret - baseline_ret
        if ret > best_ret:
            best_ret = ret
            best_config = (name, rsi_os, rsi_ob, tp_base, tp_low, tp_high, sl_mult, start, end, skip, yearly, max_dd, wr, trades)
        marker = ' <<<' if ret == best_ret else ''
        print(f'   {name:25s}: +{ret:>6.1f}% ({diff:+6.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}')

    # Best config details
    if best_config:
        name, rsi_os, rsi_ob, tp_base, tp_low, tp_high, sl_mult, start, end, skip, yearly, max_dd, wr, trades = best_config
        print(f'\n' + '=' * 100)
        print(f'BEST CONFIGURATION: {name}')
        print('=' * 100)
        print(f'RSI: {rsi_os}/{rsi_ob} | TP: {tp_low}/{tp_base}/{tp_high} | SL: {sl_mult}x')
        print(f'Session: {start}:00-{end}:00 | Skip: {skip}')
        print(f'Return: +{best_ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%')
        print(f'\nYearly breakdown:')
        for year in sorted(yearly.keys()):
            pnl = yearly[year]
            status = '  ' if pnl > 0 else ' (LOSS)'
            print(f'   {year}: ${pnl:>+12,.2f}{status}')


if __name__ == "__main__":
    main()
