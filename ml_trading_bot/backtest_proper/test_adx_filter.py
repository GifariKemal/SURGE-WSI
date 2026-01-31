"""
ADX Filter Test
===============
Test using ADX (Average Directional Index) to filter trades.

Theory:
- ADX measures trend STRENGTH (not direction)
- ADX < 20: Weak trend / ranging market -> GOOD for mean reversion
- ADX 20-25: Trend developing
- ADX > 25: Strong trend -> BAD for mean reversion
- ADX > 40: Very strong trend

For mean reversion, we want LOW ADX (ranging conditions).

Sources:
- https://eodhd.com/financial-academy/backtesting-strategies-examples/does-combining-adx-and-rsi-create-a-better-profitable-trading-strategy
- https://www.quantifiedstrategies.com/rsi-adx-trading-strategy/
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


def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    Returns: ADX, +DI, -DI
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    atr = tr.ewm(span=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(span=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(span=period, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * smooth_plus_dm / (atr + 1e-10)
    minus_di = 100 * smooth_minus_dm / (atr + 1e-10)

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx, plus_di, minus_di


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

    # ADX
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'], 14)

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
    print('ADX FILTER TEST')
    print('=' * 100)
    print('Theory: Low ADX = ranging market = GOOD for mean reversion')

    # Analyze ADX distribution
    adx_sample = df['adx'].dropna()
    print(f'\nADX Distribution:')
    print(f'   Mean: {adx_sample.mean():.1f}')
    print(f'   Median: {adx_sample.median():.1f}')
    print(f'   < 20 (ranging): {(adx_sample < 20).sum() / len(adx_sample) * 100:.1f}%')
    print(f'   20-25 (weak trend): {((adx_sample >= 20) & (adx_sample < 25)).sum() / len(adx_sample) * 100:.1f}%')
    print(f'   25-40 (strong trend): {((adx_sample >= 25) & (adx_sample < 40)).sum() / len(adx_sample) * 100:.1f}%')
    print(f'   > 40 (very strong): {(adx_sample >= 40).sum() / len(adx_sample) * 100:.1f}%')

    # v3.7 parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(adx_max=None, adx_min=None):
        """
        Backtest with ADX filter.
        adx_max: Only trade if ADX < max (for mean reversion, want low ADX)
        adx_min: Only trade if ADX > min (optional)
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        filtered = 0

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
                adx = row['adx']

                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                # Apply ADX filter
                if signal:
                    if adx_max is not None and adx > adx_max:
                        filtered += 1
                        signal = 0
                    if adx_min is not None and adx < adx_min:
                        filtered += 1
                        signal = 0

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
        return ret, trades, wr, max_dd, yearly_pnl, filtered

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest()
    print(f'\n1. v3.7 Baseline (no ADX filter): +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. ADX Max filter (only trade in ranging conditions)
    print('\n2. ADX MAX FILTER (only trade when ADX < threshold)')
    print('-' * 80)
    print('   For mean reversion, we want LOW ADX (ranging/choppy market)')
    print(f'\n   {"ADX Max":^10} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 90)

    for adx_max in [15, 20, 25, 30, 35, 40, 50]:
        ret, trades, wr, max_dd, _, filtered = backtest(adx_max=adx_max)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   ADX < {adx_max:<3} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # 3. ADX Min filter (only trade when trending - opposite of mean reversion theory)
    print('\n3. ADX MIN FILTER (only trade when ADX > threshold)')
    print('-' * 80)
    print('   This is opposite to mean reversion theory - just for comparison')
    print(f'\n   {"ADX Min":^10} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 90)

    for adx_min in [15, 20, 25, 30]:
        ret, trades, wr, max_dd, _, filtered = backtest(adx_min=adx_min)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   ADX > {adx_min:<3} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # 4. ADX Range filter
    print('\n4. ADX RANGE FILTER')
    print('-' * 80)
    print(f'   {"Range":^15} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8}')
    print('   ' + '-' * 70)

    ranges = [
        (10, 25, 'ADX 10-25'),
        (15, 30, 'ADX 15-30'),
        (20, 40, 'ADX 20-40'),
    ]

    for adx_min, adx_max, name in ranges:
        ret, trades, wr, max_dd, _, filtered = backtest(adx_min=adx_min, adx_max=adx_max)
        diff = ret - baseline_ret
        print(f'   {name:<15} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}%')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. ADX FILTER CONCEPT:
   - ADX measures trend strength (not direction)
   - Low ADX = ranging market = traditionally good for mean reversion
   - High ADX = strong trend = traditionally bad for mean reversion

2. EXPECTED RESULT:
   - If theory holds: ADX < 25 should improve WR
   - Mean reversion should work better in ranging conditions

3. ACTUAL RESULTS:
   - Compare filtered vs unfiltered performance
   - If ADX filter improves returns -> Consider implementing
   - If not -> Keep strategy without ADX filter

4. NOTE:
   - ADX adds complexity without guaranteeing improvement
   - ATR filter may already capture similar information
   - Simpler strategies often perform better
""")


if __name__ == "__main__":
    main()
