"""
Dynamic RSI Threshold Test
==========================
Test adaptive RSI thresholds based on volatility.

Theory:
- Low volatility: Use tighter thresholds (e.g., 45/55) - market is quieter
- High volatility: Use wider thresholds (e.g., 35/65) - need more extreme signals

This is NOT a filter - it changes the entry parameters dynamically.

Sources:
- https://medium.com/@FMZQuant/volatility-optimized-rsi-mean-reversion-trading-strategy-a83eda318fab
- https://www.tradingview.com/script/K9FLcueo-Dynamic-RSI-Mean-Reversion-Strategy/
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
    print('DYNAMIC RSI THRESHOLD TEST')
    print('=' * 100)

    # v3.7 parameters
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(dynamic_rsi=False, low_vol_os=45, low_vol_ob=55,
                 med_vol_os=42, med_vol_ob=58, high_vol_os=38, high_vol_ob=62):
        """
        Backtest with dynamic RSI thresholds.

        If dynamic_rsi=True:
        - Low vol (ATR < 40th pct): Use low_vol_os/low_vol_ob (tighter)
        - Med vol (40-60th pct): Use med_vol_os/med_vol_ob
        - High vol (> 60th pct): Use high_vol_os/high_vol_ob (wider)
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

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

                # Determine RSI thresholds
                if dynamic_rsi:
                    if atr_pct < 40:
                        rsi_os, rsi_ob = low_vol_os, low_vol_ob
                    elif atr_pct > 60:
                        rsi_os, rsi_ob = high_vol_os, high_vol_ob
                    else:
                        rsi_os, rsi_ob = med_vol_os, med_vol_ob
                else:
                    rsi_os, rsi_ob = 42, 58  # v3.7 fixed

                signal = 1 if rsi < rsi_os else (-1 if rsi > rsi_ob else 0)

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
        return ret, trades, wr, max_dd, yearly_pnl

    # 1. Baseline
    print('\n1. BASELINE vs DYNAMIC RSI THRESHOLDS')
    print('-' * 80)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(dynamic_rsi=False)
    print(f'   v3.7 Baseline (fixed 42/58): +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Test different dynamic threshold schemes
    print('\n2. DYNAMIC THRESHOLD CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":^40} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        # (low_os, low_ob, med_os, med_ob, high_os, high_ob, name)
        (45, 55, 42, 58, 38, 62, 'Tighter low, wider high'),
        (44, 56, 42, 58, 40, 60, 'Slight adaptive'),
        (46, 54, 42, 58, 36, 64, 'More adaptive'),
        (48, 52, 42, 58, 35, 65, 'Extreme adaptive'),
        (42, 58, 42, 58, 42, 58, 'Same as baseline (control)'),
        (40, 60, 42, 58, 44, 56, 'Inverse (wider low, tighter high)'),
        (43, 57, 42, 58, 41, 59, 'Minimal change'),
    ]

    for low_os, low_ob, med_os, med_ob, high_os, high_ob, name in configs:
        ret, trades, wr, max_dd, _ = backtest(
            dynamic_rsi=True,
            low_vol_os=low_os, low_vol_ob=low_ob,
            med_vol_os=med_os, med_vol_ob=med_ob,
            high_vol_os=high_os, high_vol_ob=high_ob
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<40} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Test volatility-percentile based thresholds
    print('\n3. PERCENTILE-BASED DYNAMIC THRESHOLDS')
    print('-' * 80)
    print('   RSI threshold = 50 +/- (8 * (1 - atr_pct/100)) or similar formula')

    def backtest_percentile_based(base_offset=8, vol_factor=0.5):
        """
        RSI thresholds based on formula:
        - Low vol: offset increases (tighter around 50)
        - High vol: offset decreases (wider from 50)

        Formula: offset = base_offset * (1 + vol_factor * (50 - atr_pct) / 50)
        """
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

                # Dynamic threshold calculation
                # Higher vol = smaller offset = wider thresholds
                offset = base_offset * (1 + vol_factor * (50 - atr_pct) / 50)
                offset = max(5, min(15, offset))  # Clamp between 5 and 15

                rsi_os = 50 - offset
                rsi_ob = 50 + offset

                rsi = row['rsi']
                signal = 1 if rsi < rsi_os else (-1 if rsi > rsi_ob else 0)

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

    print(f'\n   {"Base Offset / Vol Factor":^30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 85)

    for base_offset in [6, 8, 10]:
        for vol_factor in [0.3, 0.5, 0.7]:
            ret, trades, wr, max_dd = backtest_percentile_based(base_offset, vol_factor)
            diff = ret - baseline_ret
            marker = ' <<<' if diff > 30 else ''
            print(f'   offset={base_offset}, factor={vol_factor}          | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. DYNAMIC RSI THRESHOLD CONCEPT:
   - Adjust RSI thresholds based on current volatility
   - Low vol: Tighter thresholds (market is quiet, smaller moves matter)
   - High vol: Wider thresholds (need more extreme readings)

2. RESULTS:
   - Dynamic thresholds may or may not improve performance
   - Too aggressive adaptation can reduce opportunities
   - Fixed 42/58 is already well-tuned

3. RECOMMENDATION:
   - If dynamic thresholds improve returns -> Consider implementing
   - If not -> Keep fixed 42/58 (simpler is better)
""")


if __name__ == "__main__":
    main()
