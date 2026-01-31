"""
Internal Bar Strength (IBS) Test
================================
Test IBS indicator for mean reversion - from academic research.

IBS Formula:
    IBS = (Close - Low) / (High - Low)

Range: 0 to 1
- IBS < 0.2: Close near low = Oversold (BUY)
- IBS > 0.8: Close near high = Overbought (SELL)

Academic source: arXiv:2306.12434 "Using Internal Bar Strength as a Key
Indicator for Trading Country ETFs"

Reported 78% win rate on SPY with 0.8% avg gain per trade.

Sources:
- https://arxiv.org/abs/2306.12434
- https://www.quantifiedstrategies.com/internal-bar-strength-ibs-indicator-strategy/
- https://alvarezquanttrading.com/blog/internal-bar-strength-for-mean-reversion/
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


def calculate_ibs(high, low, close):
    """
    Calculate Internal Bar Strength.
    IBS = (Close - Low) / (High - Low)
    Returns values from 0 to 1.
    """
    return (close - low) / (high - low + 1e-10)


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

    # IBS
    df['ibs'] = calculate_ibs(df['high'], df['low'], df['close'])

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
    df = df.ffill().fillna(0.5)

    print('=' * 100)
    print('INTERNAL BAR STRENGTH (IBS) TEST')
    print('=' * 100)
    print('Formula: IBS = (Close - Low) / (High - Low)')
    print('Academic source: arXiv:2306.12434')

    # Analyze IBS distribution
    ibs_sample = df['ibs'].dropna()
    print(f'\nIBS Distribution:')
    print(f'   Mean: {ibs_sample.mean():.3f}')
    print(f'   < 0.2 (oversold): {(ibs_sample < 0.2).sum() / len(ibs_sample) * 100:.1f}%')
    print(f'   0.2-0.8 (neutral): {((ibs_sample >= 0.2) & (ibs_sample <= 0.8)).sum() / len(ibs_sample) * 100:.1f}%')
    print(f'   > 0.8 (overbought): {(ibs_sample > 0.8).sum() / len(ibs_sample) * 100:.1f}%')

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

    def backtest(use_ibs=False, ibs_os=0.2, ibs_ob=0.8, combine_rsi=False):
        """
        Backtest with IBS indicator.
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

                ibs = row['ibs']
                rsi = row['rsi']

                if use_ibs:
                    if combine_rsi:
                        # Both IBS and RSI must agree
                        ibs_signal = 1 if ibs < ibs_os else (-1 if ibs > ibs_ob else 0)
                        rsi_signal = 1 if rsi < 42 else (-1 if rsi > 58 else 0)
                        signal = ibs_signal if ibs_signal == rsi_signal else 0
                    else:
                        # IBS only
                        signal = 1 if ibs < ibs_os else (-1 if ibs > ibs_ob else 0)
                else:
                    # RSI baseline
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
        return ret, trades, wr, max_dd, yearly_pnl

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest(use_ibs=False)
    print(f'\n1. v3.7 Baseline RSI(10) 42/58: +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. IBS only configurations
    print('\n2. IBS ONLY CONFIGURATIONS')
    print('-' * 80)
    print(f'   {"Config":<25} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 85)

    configs = [
        (0.2, 0.8, 'IBS 0.2/0.8 std'),
        (0.1, 0.9, 'IBS 0.1/0.9 extreme'),
        (0.15, 0.85, 'IBS 0.15/0.85'),
        (0.25, 0.75, 'IBS 0.25/0.75'),
        (0.3, 0.7, 'IBS 0.3/0.7 loose'),
        (0.35, 0.65, 'IBS 0.35/0.65'),
        (0.4, 0.6, 'IBS 0.4/0.6 very loose'),
    ]

    for os, ob, name in configs:
        ret, trades, wr, max_dd, _ = backtest(use_ibs=True, ibs_os=os, ibs_ob=ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<25} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. IBS + RSI combined
    print('\n3. IBS + RSI COMBINED (both must agree)')
    print('-' * 80)
    print(f'   {"Config":<25} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 85)

    for os, ob, name in [(0.2, 0.8, 'IBS 0.2/0.8'), (0.3, 0.7, 'IBS 0.3/0.7'), (0.4, 0.6, 'IBS 0.4/0.6')]:
        ret, trades, wr, max_dd, _ = backtest(use_ibs=True, ibs_os=os, ibs_ob=ob, combine_rsi=True)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        combined_name = f'{name} + RSI'
        print(f'   {combined_name:<25} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. IBS CONCEPT (Academic):
   - Measures where close is within H-L range
   - IBS < 0.2 = closed near low = oversold
   - IBS > 0.8 = closed near high = overbought
   - Works best on stock indices (SPY, QQQ)

2. FOREX APPLICATION:
   - IBS may behave differently on forex H1
   - Forex has 24h market, different dynamics
   - May not show same mean reversion as stocks

3. RECOMMENDATION:
   - Compare IBS vs RSI performance
   - If IBS improves returns -> Consider adopting
   - Note: IBS primarily works on stocks/indices
""")


if __name__ == "__main__":
    main()
