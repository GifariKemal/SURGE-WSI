"""
True Strength Index (TSI) Test
==============================
Test TSI indicator for mean reversion.

TSI Formula:
    PC = Price Change = Close - Close[1]
    Double Smoothed PC = EMA(EMA(PC, r), s)
    Double Smoothed Absolute PC = EMA(EMA(|PC|, r), s)
    TSI = 100 * (Double Smoothed PC / Double Smoothed Absolute PC)

Standard parameters:
- r (long period) = 25
- s (short period) = 13

Range: -100 to +100
- > +25: Overbought
- < -25: Oversold

TSI is a double-smoothed momentum indicator that reduces noise.
Created by William Blau, published in "Stocks & Commodities" magazine.

Sources:
- https://www.investopedia.com/terms/t/tsi.asp
- https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
- SSRN papers on momentum indicators
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


def calculate_tsi(close, r=25, s=13):
    """
    Calculate True Strength Index (TSI).

    TSI = 100 * (Double Smoothed PC / Double Smoothed Absolute PC)

    Parameters:
    - r: long smoothing period (default 25)
    - s: short smoothing period (default 13)
    """
    # Price Change
    pc = close.diff()

    # Double smoothed price change
    pc_smoothed1 = pc.ewm(span=r, adjust=False).mean()
    pc_smoothed2 = pc_smoothed1.ewm(span=s, adjust=False).mean()

    # Double smoothed absolute price change
    apc_smoothed1 = pc.abs().ewm(span=r, adjust=False).mean()
    apc_smoothed2 = apc_smoothed1.ewm(span=s, adjust=False).mean()

    # TSI
    tsi = 100 * pc_smoothed2 / (apc_smoothed2 + 1e-10)

    return tsi


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
    print('TRUE STRENGTH INDEX (TSI) TEST')
    print('=' * 100)
    print('Formula: TSI = 100 * (Double Smoothed PC / Double Smoothed |PC|)')
    print('Range: -100 to +100 | OB: > +25 | OS: < -25')

    # Analyze TSI distribution
    tsi_sample = calculate_tsi(df['close'], 25, 13).dropna()
    print(f'\nTSI(25,13) Distribution:')
    print(f'   Mean: {tsi_sample.mean():.2f}')
    print(f'   Std: {tsi_sample.std():.2f}')
    print(f'   < -25 (oversold): {(tsi_sample < -25).sum() / len(tsi_sample) * 100:.1f}%')
    print(f'   -25 to +25 (neutral): {((tsi_sample >= -25) & (tsi_sample <= 25)).sum() / len(tsi_sample) * 100:.1f}%')
    print(f'   > +25 (overbought): {(tsi_sample > 25).sum() / len(tsi_sample) * 100:.1f}%')

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

    def backtest(use_tsi=False, r=25, s=13, os_level=-25, ob_level=25):
        """
        Backtest with TSI indicator.
        """
        if use_tsi:
            indicator = calculate_tsi(df['close'], r, s)
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

                if use_tsi:
                    # TSI: < -25 = oversold (BUY), > +25 = overbought (SELL)
                    signal = 1 if ind_val < os_level else (-1 if ind_val > ob_level else 0)
                else:
                    # RSI baseline
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
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(use_tsi=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. TSI with various settings
    print('\n2. TSI CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    configs = [
        # (r, s, os, ob, name)
        (25, 13, -25, 25, 'TSI(25,13) -25/+25 std'),
        (25, 13, -20, 20, 'TSI(25,13) -20/+20 tighter'),
        (25, 13, -15, 15, 'TSI(25,13) -15/+15 very tight'),
        (25, 13, -30, 30, 'TSI(25,13) -30/+30 wider'),
        (25, 13, -10, 10, 'TSI(25,13) -10/+10 extreme tight'),
        (13, 7, -25, 25, 'TSI(13,7) -25/+25 fast'),
        (13, 7, -20, 20, 'TSI(13,7) -20/+20 fast tight'),
        (13, 7, -15, 15, 'TSI(13,7) -15/+15'),
        (8, 5, -25, 25, 'TSI(8,5) -25/+25 very fast'),
        (8, 5, -20, 20, 'TSI(8,5) -20/+20'),
        (40, 20, -25, 25, 'TSI(40,20) -25/+25 slow'),
    ]

    for r, s, os, ob, name in configs:
        ret, trades, wr, max_dd, _ = backtest(use_tsi=True, r=r, s=s, os_level=os, ob_level=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. TSI with RSI-equivalent thresholds
    print('\n3. TSI WITH RSI-EQUIVALENT THRESHOLDS')
    print('-' * 80)
    print('   Converting RSI 42/58 concept to TSI scale')
    print('   RSI 42/58 -> TSI approximately -8/+8 to -16/+16')
    print(f'\n   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    for r, s in [(25, 13), (13, 7), (8, 5)]:
        for os, ob in [(-8, 8), (-10, 10), (-12, 12), (-16, 16)]:
            ret, trades, wr, max_dd, _ = backtest(use_tsi=True, r=r, s=s, os_level=os, ob_level=ob)
            diff = ret - baseline_ret
            marker = ' <<<' if diff > 30 else ''
            name = f'TSI({r},{s}) {os}/+{ob}'
            print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. TSI CONCEPT:
   - Double-smoothed momentum indicator
   - Reduces noise compared to single-smoothed RSI
   - Range -100 to +100
   - Standard: OB > +25, OS < -25

2. vs RSI:
   - TSI double smoothing may reduce false signals
   - But also reduces responsiveness
   - TSI may lag behind RSI for quick mean reversion

3. RECOMMENDATION:
   - Compare TSI vs RSI performance
   - If TSI improves returns -> Consider switching
   - Note: TSI typically used for trending, not mean reversion
""")


if __name__ == "__main__":
    main()
