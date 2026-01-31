"""
Hull Moving Average (HMA) Test
==============================
Test HMA for mean reversion signals.

HMA reduces lag while maintaining smoothness.
Created by Alan Hull.

Formula:
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

Where WMA = Weighted Moving Average

Entry signals:
- Price crosses below HMA = oversold (BUY)
- Price crosses above HMA = overbought (SELL)

Or use HMA slope/direction change.

Sources:
- https://alanhull.com/hull-moving-average
- https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average
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


def wma(series, period):
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def calculate_hma(close, period=9):
    """
    Calculate Hull Moving Average.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))

    wma_half = wma(close, half_period)
    wma_full = wma(close, period)

    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)

    return hma


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
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR for SL/TP
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
    print('HULL MOVING AVERAGE (HMA) TEST')
    print('=' * 100)
    print('Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))')
    print('Entry: Price vs HMA crossover or HMA direction change')

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

    def backtest(use_hma=False, period=9, threshold_pct=0.0, use_slope=False):
        """
        Backtest with HMA.
        """
        if use_hma:
            hma = calculate_hma(df['close'], period)
            hma_slope = hma.diff()

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
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
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

                if use_hma:
                    hma_val = hma.iloc[i]
                    if pd.isna(hma_val):
                        continue

                    if use_slope:
                        # Use HMA slope direction change
                        slope = hma_slope.iloc[i]
                        prev_slope = hma_slope.iloc[i-1]
                        if pd.isna(slope) or pd.isna(prev_slope):
                            continue
                        # Slope turns up = BUY, slope turns down = SELL
                        if slope > 0 and prev_slope <= 0:
                            signal = 1
                        elif slope < 0 and prev_slope >= 0:
                            signal = -1
                        else:
                            signal = 0
                    else:
                        # Use price vs HMA crossover
                        price = row['close']
                        threshold = hma_val * threshold_pct / 100
                        if price < hma_val - threshold:
                            signal = 1
                        elif price > hma_val + threshold:
                            signal = -1
                        else:
                            signal = 0
                else:
                    rsi = row['rsi']
                    signal = 1 if rsi < 42 else (-1 if rsi > 58 else 0)

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

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_hma=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. HMA Price Crossover
    print('\n2. HMA PRICE CROSSOVER')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    configs = [
        (9, 0.0, 'HMA(9) 0% threshold'),
        (9, 0.1, 'HMA(9) 0.1% threshold'),
        (9, 0.2, 'HMA(9) 0.2% threshold'),
        (16, 0.0, 'HMA(16) 0% threshold'),
        (16, 0.1, 'HMA(16) 0.1% threshold'),
        (16, 0.2, 'HMA(16) 0.2% threshold'),
        (25, 0.1, 'HMA(25) 0.1% threshold'),
        (36, 0.1, 'HMA(36) 0.1% threshold'),
        (49, 0.1, 'HMA(49) 0.1% threshold'),
    ]

    for period, thresh, name in configs:
        ret, trades, wr, max_dd = backtest(use_hma=True, period=period, threshold_pct=thresh, use_slope=False)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. HMA Slope Direction Change
    print('\n3. HMA SLOPE DIRECTION CHANGE')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    for period in [9, 16, 25, 36, 49]:
        ret, trades, wr, max_dd = backtest(use_hma=True, period=period, use_slope=True)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        name = f'HMA({period}) slope change'
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    print('\n' + '=' * 100)
    print('FINDINGS: HMA reduces lag but is still a trend-following indicator')
    print('=' * 100)


if __name__ == "__main__":
    main()
