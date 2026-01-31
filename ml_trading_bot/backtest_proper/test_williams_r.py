"""
Williams %R Test
================
Test Williams %R indicator for mean reversion.

Williams %R Formula:
    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

Range: 0 to -100
- Near 0: Overbought (price near high of range)
- Near -100: Oversold (price near low of range)
- Standard thresholds: -20 (OB), -80 (OS)

Very similar to Stochastic but inverted scale.
Reported 81% win rate in some backtests.

Sources:
- https://www.quantifiedstrategies.com/williams-r-trading-strategy/
- https://www.investopedia.com/terms/w/williamsr.asp
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


def calculate_williams_r(high, low, close, period=14):
    """
    Calculate Williams %R.
    Returns values from 0 to -100.
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    return williams_r


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
    print('WILLIAMS %R TEST')
    print('=' * 100)
    print('Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100')
    print('Range: 0 to -100 | OB: near 0 (-20) | OS: near -100 (-80)')

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

    def backtest(use_williams=False, period=14, os_level=-80, ob_level=-20):
        """
        Backtest with Williams %R.
        Note: Williams %R uses negative values!
        OS = -80 (oversold, BUY)
        OB = -20 (overbought, SELL)
        """
        if use_williams:
            indicator = calculate_williams_r(df['high'], df['low'], df['close'], period)
        else:
            indicator = df['rsi']

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

                if use_williams:
                    # Williams: < -80 = oversold (BUY), > -20 = overbought (SELL)
                    signal = 1 if ind_val < os_level else (-1 if ind_val > ob_level else 0)
                else:
                    # RSI: < 42 = oversold (BUY), > 58 = overbought (SELL)
                    signal = 1 if ind_val < 42 else (-1 if ind_val > 58 else 0)

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
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(use_williams=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Williams %R with various settings
    print('\n2. WILLIAMS %R CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    configs = [
        # (period, os, ob, name)
        (14, -80, -20, 'WillR(14) -80/-20 std'),
        (14, -90, -10, 'WillR(14) -90/-10 extreme'),
        (14, -70, -30, 'WillR(14) -70/-30 loose'),
        (10, -80, -20, 'WillR(10) -80/-20'),
        (10, -85, -15, 'WillR(10) -85/-15'),
        (10, -75, -25, 'WillR(10) -75/-25'),
        (7, -80, -20, 'WillR(7) -80/-20 fast'),
        (7, -85, -15, 'WillR(7) -85/-15'),
        (5, -80, -20, 'WillR(5) -80/-20 very fast'),
        (21, -80, -20, 'WillR(21) -80/-20 slow'),
    ]

    for period, os, ob, name in configs:
        ret, trades, wr, max_dd, _ = backtest(use_williams=True, period=period, os_level=os, ob_level=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Williams %R with tighter thresholds (like RSI 42/58)
    print('\n3. WILLIAMS %R WITH TIGHTER THRESHOLDS')
    print('-' * 80)
    print('   Converting RSI 42/58 concept to Williams scale')
    print('   RSI 42 ≈ WillR -58, RSI 58 ≈ WillR -42')
    print(f'\n   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    for period in [7, 10, 14]:
        for os, ob in [(-58, -42), (-60, -40), (-65, -35)]:
            ret, trades, wr, max_dd, _ = backtest(use_williams=True, period=period, os_level=os, ob_level=ob)
            diff = ret - baseline_ret
            marker = ' <<<' if diff > 30 else ''
            name = f'WillR({period}) {os}/{ob}'
            print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. WILLIAMS %R CONCEPT:
   - Inverted Stochastic oscillator (0 to -100)
   - -20 = overbought (price at high of range)
   - -80 = oversold (price at low of range)
   - Measures where close is in recent range

2. vs RSI:
   - RSI measures momentum (price change)
   - Williams %R measures position in range
   - Different calculations, similar signals

3. RECOMMENDATION:
   - Compare Williams %R vs RSI performance
   - If Williams %R improves returns -> Consider switching
   - If not -> Keep RSI(10) which is optimized for this strategy
""")


if __name__ == "__main__":
    main()
