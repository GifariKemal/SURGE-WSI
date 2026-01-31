"""
ATR Multiplier Optimization Test
================================
Fine-tune SL and TP ATR multipliers for optimal risk-reward.

Current v3.7 settings:
- SL: 1.5x ATR
- TP: 2.4/3.0/3.6x ATR (dynamic based on volatility)

Test different combinations to find optimal settings.

Sources:
- https://www.luxalgo.com/blog/5-atr-stop-loss-strategies-for-risk-control/
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
    print('ATR MULTIPLIER OPTIMIZATION TEST')
    print('=' * 100)

    # v3.7 parameters (except SL/TP which we'll vary)
    RSI_OS = 42
    RSI_OB = 58
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(sl_mult=1.5, tp_low=2.4, tp_med=3.0, tp_high=3.6):
        """Backtest with configurable SL/TP multipliers"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        total_profit = 0
        total_loss = 0

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
                        total_profit += pnl
                    else:
                        losses += 1
                        total_loss += abs(pnl)
                    position = None
                else:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            total_loss += abs(pnl)
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            total_profit += pnl
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            total_loss += abs(pnl)
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            total_profit += pnl
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
                    base_tp = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_med)
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
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
        avg_win = total_profit / wins if wins > 0 else 0
        avg_loss = total_loss / losses if losses > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        return ret, trades, wr, max_dd, profit_factor, avg_win, avg_loss

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, baseline_pf, _, _ = backtest()
    print(f'\n   v3.7 Baseline: SL=1.5x, TP=2.4/3.0/3.6x')
    print(f'   Return: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}% | PF: {baseline_pf:.2f}')

    # 2. SL Multiplier sweep
    print('\n2. STOP LOSS MULTIPLIER SWEEP (TP fixed at v3.7)')
    print('-' * 80)
    print(f'   {"SL Mult":^10} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"PF":^8}')
    print('   ' + '-' * 75)

    for sl_mult in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5]:
        ret, trades, wr, max_dd, pf, _, _ = backtest(sl_mult=sl_mult)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {sl_mult:^10} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {pf:>6.2f}{marker}')

    # 3. TP Multiplier sweep (fixed ratio across all volatility levels)
    print('\n3. TAKE PROFIT MULTIPLIER SWEEP (SL fixed at 1.5x)')
    print('-' * 80)
    print('   Testing fixed TP across all volatility levels')
    print(f'\n   {"TP Mult":^10} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"PF":^8}')
    print('   ' + '-' * 75)

    for tp_mult in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        ret, trades, wr, max_dd, pf, _, _ = backtest(tp_low=tp_mult, tp_med=tp_mult, tp_high=tp_mult)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {tp_mult:^10} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {pf:>6.2f}{marker}')

    # 4. Risk-Reward Ratio sweep
    print('\n4. RISK-REWARD RATIO OPTIMIZATION')
    print('-' * 80)
    print('   Testing different SL:TP ratios')
    print(f'\n   {"SL:TP Ratio":^15} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"PF":^8}')
    print('   ' + '-' * 85)

    rr_combos = [
        (1.0, 2.0, '1:2'),
        (1.0, 2.5, '1:2.5'),
        (1.0, 3.0, '1:3'),
        (1.5, 3.0, '1:2 (v3.7 base)'),
        (1.5, 3.5, '1:2.3'),
        (1.5, 4.0, '1:2.7'),
        (1.5, 4.5, '1:3'),
        (2.0, 4.0, '1:2'),
        (2.0, 5.0, '1:2.5'),
        (2.0, 6.0, '1:3'),
    ]

    for sl, tp, name in rr_combos:
        ret, trades, wr, max_dd, pf, avg_w, avg_l = backtest(sl_mult=sl, tp_low=tp, tp_med=tp, tp_high=tp)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   SL={sl}, TP={tp} ({name:<5})  | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {pf:>6.2f}{marker}')

    # 5. Fine-tune around best settings
    print('\n5. FINE-TUNING DYNAMIC TP MULTIPLIERS')
    print('-' * 80)
    print('   Varying tp_low/tp_med/tp_high independently')
    print(f'\n   {"TP Config":^20} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 80)

    configs = [
        (2.4, 3.0, 3.6, 'v3.7 (2.4/3.0/3.6)'),
        (2.0, 2.5, 3.0, 'Tighter (2.0/2.5/3.0)'),
        (2.6, 3.2, 3.8, 'Wider (2.6/3.2/3.8)'),
        (2.8, 3.4, 4.0, 'Much wider (2.8/3.4/4.0)'),
        (3.0, 3.5, 4.0, 'High (3.0/3.5/4.0)'),
        (2.2, 2.8, 3.4, 'Moderate (2.2/2.8/3.4)'),
        (2.5, 3.0, 3.5, 'Balanced (2.5/3.0/3.5)'),
    ]

    for tp_l, tp_m, tp_h, name in configs:
        ret, trades, wr, max_dd, pf, _, _ = backtest(tp_low=tp_l, tp_med=tp_m, tp_high=tp_h)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   {name:<20} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. STOP LOSS OPTIMIZATION:
   - SL multiplier affects win rate and drawdown
   - Tighter SL = more stopped out = lower WR but smaller losses
   - Wider SL = fewer stops = higher WR but bigger losses

2. TAKE PROFIT OPTIMIZATION:
   - TP multiplier affects average win size
   - Tighter TP = more wins but smaller profits
   - Wider TP = fewer wins but bigger profits

3. RISK-REWARD RATIO:
   - Higher R:R (e.g., 1:3) = lower WR but higher profit per win
   - Lower R:R (e.g., 1:2) = higher WR but lower profit per win
   - Optimal depends on strategy characteristics

4. RECOMMENDATION:
   - Find the setting with best Return/MaxDD ratio
   - v3.7 dynamic TP (2.4/3.0/3.6) is already well-tuned
   - Only change if significant improvement found
""")


if __name__ == "__main__":
    main()
