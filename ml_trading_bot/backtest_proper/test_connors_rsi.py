"""
ConnorsRSI Test
===============
Test Larry Connors' composite RSI indicator.

ConnorsRSI Formula:
    CRSI = (RSI(Close,3) + RSI(Streak,2) + PercentRank(ROC,100)) / 3

Where:
- RSI(Close,3): 3-period RSI of closing prices
- RSI(Streak,2): 2-period RSI of consecutive up/down streak
- PercentRank(ROC,100): Percentile rank of 1-day ROC over 100 periods

Uses 90/10 thresholds instead of traditional 70/30.

Sources:
- https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/connorsrsi
- https://www.backtrader.com/recipes/indicators/crsi/crsi/
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


def calculate_rsi(series, period):
    """Calculate RSI for a series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_streak(close):
    """
    Calculate up/down streak.
    Positive: number of consecutive up days
    Negative: number of consecutive down days
    """
    streak = pd.Series(0, index=close.index)
    diff = close.diff()

    for i in range(1, len(close)):
        if diff.iloc[i] > 0:
            streak.iloc[i] = max(1, streak.iloc[i-1] + 1) if streak.iloc[i-1] >= 0 else 1
        elif diff.iloc[i] < 0:
            streak.iloc[i] = min(-1, streak.iloc[i-1] - 1) if streak.iloc[i-1] <= 0 else -1
        else:
            streak.iloc[i] = 0

    return streak


def calculate_percent_rank(series, period):
    """
    Calculate percent rank of current value over lookback period.
    Returns 0-100 indicating what % of values were less than current.
    """
    def pct_rank(x):
        if len(x) < 2:
            return 50
        current = x.iloc[-1]
        historical = x.iloc[:-1]
        return (historical < current).sum() / len(historical) * 100

    return series.rolling(period + 1).apply(pct_rank, raw=False)


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

    # Standard RSI(10) for comparison
    df['rsi10'] = calculate_rsi(df['close'], 10)

    # ConnorsRSI components
    df['rsi3'] = calculate_rsi(df['close'], 3)
    df['streak'] = calculate_streak(df['close'])
    df['rsi_streak'] = calculate_rsi(df['streak'], 2)
    df['roc'] = df['close'].pct_change() * 100  # 1-period ROC
    df['pct_rank'] = calculate_percent_rank(df['roc'], 100)

    # ConnorsRSI = average of 3 components
    df['crsi'] = (df['rsi3'] + df['rsi_streak'] + df['pct_rank']) / 3

    # ATR for SL/TP
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(50)  # Fill NaN with neutral values

    print('=' * 100)
    print('CONNORSRSI TEST')
    print('=' * 100)
    print('Formula: CRSI = (RSI(3) + RSI(Streak,2) + PercentRank(ROC,100)) / 3')

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

    def backtest(use_crsi=False, os_level=42, ob_level=58):
        """
        Backtest with either standard RSI or ConnorsRSI.
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

                # Use ConnorsRSI or standard RSI
                if use_crsi:
                    indicator = row['crsi']
                else:
                    indicator = row['rsi10']

                signal = 1 if indicator < os_level else (-1 if indicator > ob_level else 0)

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

    # 1. Baseline with standard RSI
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(use_crsi=False, os_level=42, ob_level=58)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. ConnorsRSI with various thresholds
    print('\n2. CONNORSRSI WITH DIFFERENT THRESHOLDS')
    print('-' * 80)
    print('   Connors recommends 90/10 for CRSI (more extreme than traditional 70/30)')
    print(f'\n   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 85)

    configs = [
        (10, 90, 'CRSI 10/90 (Connors std)'),
        (15, 85, 'CRSI 15/85'),
        (20, 80, 'CRSI 20/80'),
        (25, 75, 'CRSI 25/75'),
        (30, 70, 'CRSI 30/70'),
        (35, 65, 'CRSI 35/65'),
        (40, 60, 'CRSI 40/60'),
        (42, 58, 'CRSI 42/58 (same as v3.7)'),
    ]

    for os, ob, name in configs:
        ret, trades, wr, max_dd, _ = backtest(use_crsi=True, os_level=os, ob_level=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Show CRSI component values
    print('\n3. CONNORSRSI COMPONENT ANALYSIS')
    print('-' * 80)
    sample = df.dropna().iloc[-500:]
    print(f'   RSI(3) mean: {sample["rsi3"].mean():.1f}, std: {sample["rsi3"].std():.1f}')
    print(f'   RSI(Streak,2) mean: {sample["rsi_streak"].mean():.1f}, std: {sample["rsi_streak"].std():.1f}')
    print(f'   PercentRank mean: {sample["pct_rank"].mean():.1f}, std: {sample["pct_rank"].std():.1f}')
    print(f'   CRSI mean: {sample["crsi"].mean():.1f}, std: {sample["crsi"].std():.1f}')
    print(f'   Standard RSI(10) mean: {sample["rsi10"].mean():.1f}, std: {sample["rsi10"].std():.1f}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. CONNORSRSI CONCEPT:
   - Composite of 3 RSI-based measures
   - RSI(3) for short-term momentum
   - RSI(Streak) for consecutive move strength
   - PercentRank for relative move size
   - Designed for stocks on daily timeframe (Larry Connors)

2. H1 FOREX APPLICATION:
   - CRSI may behave differently on H1 forex
   - Thresholds may need adjustment from 90/10

3. RECOMMENDATION:
   - Compare CRSI vs standard RSI performance
   - If CRSI improves returns -> Consider adopting
   - If not -> Keep simple RSI(10)
""")


if __name__ == "__main__":
    main()
