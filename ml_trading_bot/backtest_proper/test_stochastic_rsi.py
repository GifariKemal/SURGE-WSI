"""
Stochastic RSI Test
===================
Test the Stochastic RSI indicator - RSI of RSI.

Formula:
    StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)

Then smooth with %K and %D:
    %K = SMA(StochRSI, smoothK)
    %D = SMA(%K, smoothD)

More responsive than regular RSI, oscillates between 0-100.

Sources:
- https://www.quantifiedstrategies.com/stochastic-rsi/
- https://www.investopedia.com/terms/s/stochrsi.asp
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


def calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    """
    Calculate Stochastic RSI.

    Returns: stoch_rsi, %K, %D (all 0-100 scale)
    """
    # First calculate RSI
    rsi = calculate_rsi(close, rsi_period)

    # Then apply Stochastic formula to RSI
    lowest_rsi = rsi.rolling(stoch_period).min()
    highest_rsi = rsi.rolling(stoch_period).max()

    stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10) * 100

    # Smooth with %K and %D
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()

    return stoch_rsi, k, d


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

    # ATR for SL/TP
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(50)

    print('=' * 100)
    print('STOCHASTIC RSI TEST')
    print('=' * 100)
    print('Formula: StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)')

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

    def backtest(use_stoch_rsi=False, rsi_period=14, stoch_period=14,
                 smooth_k=3, smooth_d=3, os_level=20, ob_level=80, use_k=True):
        """
        Backtest with either standard RSI or Stochastic RSI.
        """
        if use_stoch_rsi:
            stoch_rsi, k, d = calculate_stoch_rsi(df['close'], rsi_period, stoch_period, smooth_k, smooth_d)
            indicator = k if use_k else stoch_rsi
        else:
            indicator = df['rsi10']

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

                ind_val = indicator.iloc[i]
                signal = 1 if ind_val < os_level else (-1 if ind_val > ob_level else 0)

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
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(use_stoch_rsi=False, os_level=42, ob_level=58)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Stochastic RSI with various settings
    print('\n2. STOCHASTIC RSI CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<45} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 100)

    configs = [
        # (rsi_period, stoch_period, smooth_k, smooth_d, os, ob, name)
        (14, 14, 3, 3, 20, 80, 'StochRSI(14,14,3,3) 20/80 std'),
        (14, 14, 3, 3, 10, 90, 'StochRSI(14,14,3,3) 10/90'),
        (14, 14, 3, 3, 30, 70, 'StochRSI(14,14,3,3) 30/70'),
        (10, 10, 3, 3, 20, 80, 'StochRSI(10,10,3,3) 20/80'),
        (10, 10, 3, 3, 30, 70, 'StochRSI(10,10,3,3) 30/70'),
        (10, 10, 3, 3, 40, 60, 'StochRSI(10,10,3,3) 40/60'),
        (10, 14, 3, 3, 20, 80, 'StochRSI(10,14,3,3) 20/80'),
        (5, 5, 3, 3, 20, 80, 'StochRSI(5,5,3,3) 20/80 fast'),
        (7, 7, 3, 3, 25, 75, 'StochRSI(7,7,3,3) 25/75'),
    ]

    for rsi_p, stoch_p, k, d, os, ob, name in configs:
        ret, trades, wr, max_dd, _ = backtest(
            use_stoch_rsi=True, rsi_period=rsi_p, stoch_period=stoch_p,
            smooth_k=k, smooth_d=d, os_level=os, ob_level=ob
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<45} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Compare raw StochRSI vs %K smoothed
    print('\n3. RAW STOCHRSI vs SMOOTHED %K')
    print('-' * 80)

    for use_k, name in [(True, '%K Smoothed'), (False, 'Raw StochRSI')]:
        ret, trades, wr, max_dd, _ = backtest(
            use_stoch_rsi=True, rsi_period=10, stoch_period=10,
            smooth_k=3, smooth_d=3, os_level=20, ob_level=80, use_k=use_k
        )
        diff = ret - baseline_ret
        print(f'   {name:<20}: +{ret:.1f}% (diff: {diff:+.1f}%) | Trades: {trades} | WR: {wr:.1f}%')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. STOCHASTIC RSI CONCEPT:
   - Applies Stochastic formula to RSI values
   - More responsive/sensitive than regular RSI
   - Oscillates 0-100 but reaches extremes more often
   - Standard thresholds: 20/80 (vs RSI's 30/70)

2. H1 FOREX APPLICATION:
   - May generate more signals (more responsive)
   - Could be more prone to false signals
   - Need to balance sensitivity vs reliability

3. RECOMMENDATION:
   - Compare StochRSI vs standard RSI performance
   - If more responsive signals improve returns -> Consider
   - If too noisy -> Keep standard RSI(10)
""")


if __name__ == "__main__":
    main()
