"""
MFI (Money Flow Index) Test
===========================
Test MFI indicator for mean reversion.

MFI is like RSI but incorporates volume (we use range as proxy).

Formula:
    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = TP * Volume
    Money Flow Ratio = Positive MF / Negative MF
    MFI = 100 - (100 / (1 + Money Flow Ratio))

Range: 0-100 (like RSI)
- > 80: Overbought
- < 20: Oversold

Sources:
- https://www.investopedia.com/terms/m/mfi.asp
- https://school.stockcharts.com/doku.php?id=technical_indicators:money_flow_index_mfi
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


def calculate_mfi(high, low, close, period=14):
    """
    Calculate Money Flow Index.
    Using range (high-low) as volume proxy for forex.
    """
    tp = (high + low + close) / 3
    # Use range as volume proxy
    volume_proxy = high - low

    raw_mf = tp * volume_proxy
    tp_change = tp.diff()

    positive_mf = pd.Series(0.0, index=close.index)
    negative_mf = pd.Series(0.0, index=close.index)

    positive_mf[tp_change > 0] = raw_mf[tp_change > 0]
    negative_mf[tp_change < 0] = raw_mf[tp_change < 0]

    positive_mf_sum = positive_mf.rolling(period).sum()
    negative_mf_sum = negative_mf.rolling(period).sum()

    mf_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
    mfi = 100 - (100 / (1 + mf_ratio))

    return mfi


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
    df = df.ffill().fillna(50)

    print('=' * 100)
    print('MFI (MONEY FLOW INDEX) TEST')
    print('=' * 100)
    print('Formula: MFI = RSI with volume weighting (range as proxy)')
    print('Range: 0-100 | OB: > 80 | OS: < 20')

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

    def backtest(use_mfi=False, period=14, os_level=20, ob_level=80):
        if use_mfi:
            indicator = calculate_mfi(df['high'], df['low'], df['close'], period)
        else:
            indicator = df['rsi']

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

                ind_val = indicator.iloc[i]
                if pd.isna(ind_val):
                    continue

                if use_mfi:
                    signal = 1 if ind_val < os_level else (-1 if ind_val > ob_level else 0)
                else:
                    signal = 1 if ind_val < 42 else (-1 if ind_val > 58 else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    tp_mult = base_tp + TIME_TP_BONUS if 12 <= hour < 16 else base_tp
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    baseline_ret, baseline_trades, baseline_wr, baseline_dd = backtest(use_mfi=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    print('\n2. MFI CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 90)

    configs = [
        (14, 20, 80, 'MFI(14) 20/80 std'),
        (14, 30, 70, 'MFI(14) 30/70'),
        (14, 40, 60, 'MFI(14) 40/60'),
        (14, 42, 58, 'MFI(14) 42/58 RSI-like'),
        (10, 20, 80, 'MFI(10) 20/80'),
        (10, 30, 70, 'MFI(10) 30/70'),
        (10, 42, 58, 'MFI(10) 42/58'),
        (7, 30, 70, 'MFI(7) 30/70 fast'),
        (7, 42, 58, 'MFI(7) 42/58'),
    ]

    for period, os, ob, name in configs:
        ret, trades, wr, max_dd = backtest(use_mfi=True, period=period, os_level=os, ob_level=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<30} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')


if __name__ == "__main__":
    main()
