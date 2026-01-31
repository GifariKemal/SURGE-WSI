"""
Final Optimization Analysis
===========================
Deep dive into remaining optimization opportunities for v3.7 strategy.

Areas to explore:
1. SL multiplier fine-tuning (currently 1.5x)
2. TP multiplier fine-tuning (currently 2.4/3.0/3.6)
3. Signal strength position sizing
4. R:R ratio analysis
5. Trade distribution analysis
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

    # RSI(10)
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
    print('FINAL OPTIMIZATION ANALYSIS')
    print('=' * 100)

    # Current v3.7 parameters
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(sl_mult=1.5, tp_low=2.4, tp_med=3.0, tp_high=3.6,
                 signal_strength_sizing=False, rsi_os=42, rsi_ob=58):
        """Full backtest with configurable parameters."""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0

        # Track trade details
        trade_results = []

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                exit_reason = None
                pnl = 0

                if (i - position['entry_idx']) >= MAX_HOLDING:
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    exit_reason = 'timeout'
                else:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            exit_reason = 'sl'
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            exit_reason = 'tp'
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            exit_reason = 'sl'
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            exit_reason = 'tp'

                if exit_reason:
                    balance += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    trade_results.append({
                        'pnl': pnl,
                        'exit': exit_reason,
                        'rsi_entry': position['rsi_entry'],
                        'holding': i - position['entry_idx']
                    })
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if hour < 7 or hour >= 22 or hour in SKIP_HOURS:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < rsi_os else (-1 if rsi > rsi_ob else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002

                    # Dynamic TP
                    base_tp = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_med)
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp

                    sl = entry - atr * sl_mult if signal == 1 else entry + atr * sl_mult
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult

                    # Position sizing
                    base_risk = balance * 0.01

                    if signal_strength_sizing:
                        # More extreme RSI = larger position
                        if signal == 1:
                            extremity = (rsi_os - rsi) / rsi_os  # 0 to ~1
                        else:
                            extremity = (rsi - rsi_ob) / (100 - rsi_ob)
                        size_mult = 1 + extremity * 0.5  # 1.0 to 1.5x
                        risk = base_risk * min(size_mult, 1.5)
                    else:
                        risk = base_risk

                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {
                        'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp,
                        'size': size, 'entry_idx': i, 'rsi_entry': rsi
                    }

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100

        # Calculate additional metrics
        if trade_results:
            avg_win = np.mean([t['pnl'] for t in trade_results if t['pnl'] > 0]) if wins > 0 else 0
            avg_loss = abs(np.mean([t['pnl'] for t in trade_results if t['pnl'] <= 0])) if losses > 0 else 1
            profit_factor = (avg_win * wins) / (avg_loss * losses) if losses > 0 else 999

            tp_exits = sum(1 for t in trade_results if t['exit'] == 'tp')
            sl_exits = sum(1 for t in trade_results if t['exit'] == 'sl')
            timeout_exits = sum(1 for t in trade_results if t['exit'] == 'timeout')
        else:
            profit_factor = 0
            tp_exits = sl_exits = timeout_exits = 0
            avg_win = avg_loss = 0

        return ret, trades, wr, max_dd, profit_factor, tp_exits, sl_exits, timeout_exits, avg_win, avg_loss

    # 1. BASELINE
    print('\n' + '=' * 100)
    print('1. BASELINE v3.7')
    print('=' * 100)
    ret, trades, wr, max_dd, pf, tp_ex, sl_ex, to_ex, avg_w, avg_l = backtest()
    print(f'   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%')
    print(f'   Profit Factor: {pf:.2f} | Avg Win: ${avg_w:.0f} | Avg Loss: ${avg_l:.0f}')
    print(f'   Exit Types: TP={tp_ex} ({tp_ex/trades*100:.1f}%) | SL={sl_ex} ({sl_ex/trades*100:.1f}%) | Timeout={to_ex} ({to_ex/trades*100:.1f}%)')
    print(f'   R:R Ratio (realized): {avg_w/avg_l:.2f}:1' if avg_l > 0 else '')
    baseline_ret = ret

    # 2. SL MULTIPLIER FINE-TUNING
    print('\n' + '=' * 100)
    print('2. SL MULTIPLIER FINE-TUNING')
    print('=' * 100)
    print(f'   {"SL Mult":<10} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"PF":^6}')
    print('   ' + '-' * 75)

    for sl in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5]:
        ret, trades, wr, max_dd, pf, _, _, _, _, _ = backtest(sl_mult=sl)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 0 else ''
        print(f'   {sl:<10} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {pf:>5.2f}{marker}')

    # 3. TP MULTIPLIER VARIATIONS
    print('\n' + '=' * 100)
    print('3. TP MULTIPLIER VARIATIONS (keeping SL=1.5)')
    print('=' * 100)
    print(f'   {"TP Low/Med/High":<20} | {"Return":^10} | {"Diff":^8} | {"WR":^8} | {"MaxDD":^8} | {"PF":^6}')
    print('   ' + '-' * 80)

    tp_configs = [
        (2.0, 2.5, 3.0, '2.0/2.5/3.0 tighter'),
        (2.2, 2.8, 3.4, '2.2/2.8/3.4'),
        (2.4, 3.0, 3.6, '2.4/3.0/3.6 current'),
        (2.6, 3.2, 3.8, '2.6/3.2/3.8'),
        (2.8, 3.5, 4.2, '2.8/3.5/4.2 wider'),
        (3.0, 3.5, 4.0, '3.0/3.5/4.0'),
        (2.0, 3.0, 4.0, '2.0/3.0/4.0 spread'),
    ]

    for tp_l, tp_m, tp_h, name in tp_configs:
        ret, trades, wr, max_dd, pf, _, _, _, _, _ = backtest(tp_low=tp_l, tp_med=tp_m, tp_high=tp_h)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 0 else ''
        print(f'   {name:<20} | +{ret:>7.1f}% | {diff:>+6.1f}% | {wr:>5.1f}% | {max_dd:>5.1f}% | {pf:>5.2f}{marker}')

    # 4. SIGNAL STRENGTH SIZING
    print('\n' + '=' * 100)
    print('4. SIGNAL STRENGTH SIZING')
    print('=' * 100)
    print('   More extreme RSI = larger position (up to 1.5x)')

    ret_normal, trades, wr, max_dd, pf, _, _, _, _, _ = backtest(signal_strength_sizing=False)
    print(f'   Normal sizing:   +{ret_normal:.1f}% | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | PF: {pf:.2f}')

    ret_signal, trades, wr, max_dd, pf, _, _, _, _, _ = backtest(signal_strength_sizing=True)
    diff = ret_signal - ret_normal
    print(f'   Strength sizing: +{ret_signal:.1f}% | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | PF: {pf:.2f} | Diff: {diff:+.1f}%')

    # 5. RSI THRESHOLD MICRO-TUNING
    print('\n' + '=' * 100)
    print('5. RSI THRESHOLD MICRO-TUNING (around 42/58)')
    print('=' * 100)
    print(f'   {"RSI OS/OB":<15} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 75)

    for os, ob in [(40, 60), (41, 59), (42, 58), (43, 57), (44, 56), (41, 58), (42, 59), (43, 58)]:
        ret, trades, wr, max_dd, _, _, _, _, _, _ = backtest(rsi_os=os, rsi_ob=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 0 else ''
        print(f'   {os}/{ob:<10} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 6. COMBINED BEST PARAMETERS
    print('\n' + '=' * 100)
    print('6. TESTING POTENTIALLY IMPROVED COMBINATIONS')
    print('=' * 100)

    # Test if any SL showed improvement
    best_configs = [
        (1.5, 2.4, 3.0, 3.6, 42, 58, False, 'v3.7 Current'),
        (1.4, 2.4, 3.0, 3.6, 42, 58, False, 'SL=1.4'),
        (1.6, 2.4, 3.0, 3.6, 42, 58, False, 'SL=1.6'),
        (1.5, 2.6, 3.2, 3.8, 42, 58, False, 'TP wider'),
        (1.4, 2.6, 3.2, 3.8, 42, 58, False, 'SL=1.4 + TP wider'),
        (1.5, 2.4, 3.0, 3.6, 42, 58, True, 'Signal strength sizing'),
        (1.5, 2.4, 3.0, 3.6, 41, 59, False, 'RSI 41/59'),
    ]

    print(f'   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"WR":^8} | {"MaxDD":^8} | {"PF":^6}')
    print('   ' + '-' * 90)

    for sl, tp_l, tp_m, tp_h, rsi_os, rsi_ob, sig_size, name in best_configs:
        ret, trades, wr, max_dd, pf, _, _, _, _, _ = backtest(
            sl_mult=sl, tp_low=tp_l, tp_med=tp_m, tp_high=tp_h,
            rsi_os=rsi_os, rsi_ob=rsi_ob, signal_strength_sizing=sig_size
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 0 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {wr:>5.1f}% | {max_dd:>5.1f}% | {pf:>5.2f}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)


if __name__ == "__main__":
    main()
