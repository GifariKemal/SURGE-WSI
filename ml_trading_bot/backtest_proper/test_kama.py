"""
KAMA (Kaufman Adaptive Moving Average) Test
============================================
Test KAMA for mean reversion signals.

KAMA adapts to market noise using Efficiency Ratio (ER).
- High ER (trending) = faster MA
- Low ER (choppy) = slower MA

Formula:
    ER = Change / Volatility = |Close - Close[n]| / Sum(|Close - Close[1]|, n)
    SC = (ER * (fast - slow) + slow)^2
    KAMA = KAMA[1] + SC * (Close - KAMA[1])

Standard: n=10, fast=2, slow=30

Entry signals:
- Price crosses below KAMA = oversold (BUY)
- Price crosses above KAMA = overbought (SELL)

Sources:
- Perry Kaufman "Trading Systems and Methods"
- https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
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


def calculate_kama(close, n=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    """
    # Change
    change = abs(close - close.shift(n))

    # Volatility (sum of absolute changes)
    volatility = close.diff().abs().rolling(n).sum()

    # Efficiency Ratio
    er = change / (volatility + 1e-10)

    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

    # Scaled smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # KAMA calculation
    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[n-1] = close.iloc[:n].mean()  # Initialize with SMA

    for i in range(n, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

    return kama, er


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
    print('KAMA (KAUFMAN ADAPTIVE MOVING AVERAGE) TEST')
    print('=' * 100)
    print('Formula: KAMA adapts speed based on Efficiency Ratio')
    print('Entry: Price vs KAMA crossover')

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

    def backtest(use_kama=False, n=10, fast=2, slow=30, threshold_pct=0.0):
        """
        Backtest with KAMA.
        threshold_pct: % distance from KAMA to trigger signal
        """
        if use_kama:
            kama, er = calculate_kama(df['close'], n, fast, slow)

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

                if use_kama:
                    kama_val = kama.iloc[i]
                    if pd.isna(kama_val):
                        continue
                    price = row['close']
                    threshold = kama_val * threshold_pct / 100
                    # Price below KAMA = oversold (BUY)
                    # Price above KAMA = overbought (SELL)
                    if price < kama_val - threshold:
                        signal = 1
                    elif price > kama_val + threshold:
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
    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_kama=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. KAMA configurations
    print('\n2. KAMA CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    configs = [
        (10, 2, 30, 0.0, 'KAMA(10,2,30) 0% threshold'),
        (10, 2, 30, 0.1, 'KAMA(10,2,30) 0.1% threshold'),
        (10, 2, 30, 0.2, 'KAMA(10,2,30) 0.2% threshold'),
        (10, 2, 30, 0.3, 'KAMA(10,2,30) 0.3% threshold'),
        (5, 2, 30, 0.1, 'KAMA(5,2,30) fast'),
        (5, 2, 30, 0.2, 'KAMA(5,2,30) 0.2%'),
        (20, 2, 30, 0.1, 'KAMA(20,2,30) slow'),
        (10, 2, 20, 0.1, 'KAMA(10,2,20) faster adapt'),
        (10, 3, 40, 0.1, 'KAMA(10,3,40) wider range'),
    ]

    for n, fast, slow, thresh, name in configs:
        ret, trades, wr, max_dd = backtest(use_kama=True, n=n, fast=fast, slow=slow, threshold_pct=thresh)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    print('\n' + '=' * 100)
    print('FINDINGS: KAMA uses price vs adaptive MA crossover - different from RSI momentum')
    print('=' * 100)


if __name__ == "__main__":
    main()
