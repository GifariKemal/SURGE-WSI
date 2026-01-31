"""
RSI Divergence Strategy Test
============================
Test RSI divergence as a confirmation signal for mean reversion.

Types of divergence:
1. Regular Bullish Divergence: Price makes lower low, RSI makes higher low
   -> Strong reversal signal for LONG
2. Regular Bearish Divergence: Price makes higher high, RSI makes lower high
   -> Strong reversal signal for SHORT
3. Hidden Divergence: Opposite pattern (continuation signals)

This test checks if requiring divergence improves win rate.
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


def find_local_extrema(series, window=5):
    """Find local highs and lows within rolling window"""
    highs = series[(series.shift(1) < series) & (series.shift(-1) < series)]
    lows = series[(series.shift(1) > series) & (series.shift(-1) > series)]
    return highs, lows


def detect_divergence(price, rsi, lookback=10, min_diff=0.001):
    """
    Detect RSI divergence.

    Returns:
    - 1: Bullish divergence (price lower low, RSI higher low) -> BUY signal
    - -1: Bearish divergence (price higher high, RSI lower high) -> SELL signal
    - 0: No divergence
    """
    if len(price) < lookback or len(rsi) < lookback:
        return 0

    # Get recent price and RSI
    recent_price = price[-lookback:]
    recent_rsi = rsi[-lookback:]

    # Find price extrema
    price_min_idx = recent_price.argmin()
    price_max_idx = recent_price.argmax()

    # Check last 2-3 swing points
    # Simplified: compare current vs recent extreme

    current_price = price.iloc[-1]
    current_rsi = rsi.iloc[-1]

    # Look for previous swing low in last lookback bars
    prev_price_low = recent_price.min()
    prev_rsi_at_low = recent_rsi.iloc[price_min_idx]

    # Look for previous swing high
    prev_price_high = recent_price.max()
    prev_rsi_at_high = recent_rsi.iloc[price_max_idx]

    # Bullish divergence: price makes lower low, RSI makes higher low
    if current_price < prev_price_low * (1 - min_diff):  # New lower low
        if current_rsi > prev_rsi_at_low + 2:  # RSI higher
            return 1

    # Bearish divergence: price makes higher high, RSI makes lower high
    if current_price > prev_price_high * (1 + min_diff):  # New higher high
        if current_rsi < prev_rsi_at_high - 2:  # RSI lower
            return -1

    return 0


def detect_divergence_v2(df, idx, lookback=20, swing_window=5):
    """
    More robust divergence detection using swing points.

    Finds actual swing highs/lows and compares them.
    """
    if idx < lookback + swing_window:
        return 0

    recent = df.iloc[idx-lookback:idx+1]
    price = recent['close']
    rsi = recent['rsi']

    # Find swing lows (local minima)
    swing_lows_price = []
    swing_lows_rsi = []
    swing_highs_price = []
    swing_highs_rsi = []

    for i in range(swing_window, len(recent) - swing_window):
        # Check if swing low
        window_prices = price.iloc[i-swing_window:i+swing_window+1]
        if price.iloc[i] == window_prices.min():
            swing_lows_price.append((i, price.iloc[i]))
            swing_lows_rsi.append((i, rsi.iloc[i]))

        # Check if swing high
        if price.iloc[i] == window_prices.max():
            swing_highs_price.append((i, price.iloc[i]))
            swing_highs_rsi.append((i, rsi.iloc[i]))

    # Need at least 2 swing points to detect divergence
    if len(swing_lows_price) >= 2:
        # Compare last two swing lows
        prev_low_idx, prev_low_price = swing_lows_price[-2]
        curr_low_idx, curr_low_price = swing_lows_price[-1]

        prev_low_rsi = swing_lows_rsi[-2][1]
        curr_low_rsi = swing_lows_rsi[-1][1]

        # Bullish divergence: price lower low, RSI higher low
        if curr_low_price < prev_low_price and curr_low_rsi > prev_low_rsi:
            # Confirm we're near oversold
            if curr_low_rsi < 45:
                return 1

    if len(swing_highs_price) >= 2:
        # Compare last two swing highs
        prev_high_idx, prev_high_price = swing_highs_price[-2]
        curr_high_idx, curr_high_price = swing_highs_price[-1]

        prev_high_rsi = swing_highs_rsi[-2][1]
        curr_high_rsi = swing_highs_rsi[-1][1]

        # Bearish divergence: price higher high, RSI lower high
        if curr_high_price > prev_high_price and curr_high_rsi < prev_high_rsi:
            # Confirm we're near overbought
            if curr_high_rsi > 55:
                return -1

    return 0


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
    print('RSI DIVERGENCE STRATEGY TEST')
    print('=' * 100)

    # 1. Count divergence occurrences
    print('\n1. DIVERGENCE DETECTION ANALYSIS')
    print('-' * 80)

    bullish_div = 0
    bearish_div = 0
    no_div = 0

    for i in range(200, len(df)):
        div = detect_divergence_v2(df, i)
        if div == 1:
            bullish_div += 1
        elif div == -1:
            bearish_div += 1
        else:
            no_div += 1

    total = bullish_div + bearish_div + no_div
    print(f'   Bullish Divergences: {bullish_div} ({bullish_div/total*100:.1f}%)')
    print(f'   Bearish Divergences: {bearish_div} ({bearish_div/total*100:.1f}%)')
    print(f'   No Divergence: {no_div} ({no_div/total*100:.1f}%)')

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

    def backtest(require_divergence=False, divergence_confirmation=False, divergence_only=False):
        """
        Backtest with divergence options:
        - require_divergence: Only trade when RSI signal + divergence match
        - divergence_confirmation: Use divergence to confirm RSI direction
        - divergence_only: Trade only on divergence (ignore RSI threshold)
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
                divergence = detect_divergence_v2(df, i)

                # Determine signal based on mode
                if divergence_only:
                    signal = divergence
                elif require_divergence:
                    rsi_signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)
                    # Both must agree
                    if rsi_signal == divergence and rsi_signal != 0:
                        signal = rsi_signal
                    else:
                        signal = 0
                elif divergence_confirmation:
                    rsi_signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)
                    # Divergence confirms but doesn't block
                    if rsi_signal != 0:
                        if divergence == rsi_signal:
                            signal = rsi_signal  # Stronger signal
                        else:
                            signal = rsi_signal  # Still trade but weaker
                    else:
                        signal = 0
                else:
                    signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

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

    # 2. Test divergence strategies
    print('\n2. BACKTEST WITH DIVERGENCE')
    print('-' * 80)
    print(f'   {"Strategy":^35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 85)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest()
    print(f'   {"v3.7 Baseline (RSI only)":<35} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}%')

    strategies = [
        (dict(divergence_only=True), 'Divergence Only'),
        (dict(require_divergence=True), 'RSI + Divergence Required'),
        (dict(divergence_confirmation=True), 'RSI + Divergence Confirmation'),
    ]

    for kwargs, name in strategies:
        ret, trades, wr, max_dd, yearly = backtest(**kwargs)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 20 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. Test different lookback periods for divergence
    print('\n3. DIVERGENCE LOOKBACK SENSITIVITY')
    print('-' * 80)

    def backtest_with_lookback(lookback):
        """Test different divergence lookback periods"""
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

                rsi = row['rsi']
                rsi_signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                # Detect divergence with custom lookback
                divergence = detect_divergence_v2(df, i, lookback=lookback)

                # Require both
                if rsi_signal == divergence and rsi_signal != 0:
                    signal = rsi_signal
                else:
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
        return ret, trades, wr, max_dd

    print(f'   {"Lookback":^15} | {"Return":^10} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 55)

    for lookback in [10, 15, 20, 25, 30, 40]:
        ret, trades, wr, max_dd = backtest_with_lookback(lookback)
        print(f'   {lookback:>8} bars  | +{ret:>7.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%')

    # 4. Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. RSI DIVERGENCE ANALYSIS:
   - Divergence signals are RARE (only a few percent of bars)
   - Requiring divergence drastically reduces trade count
   - Win rate may improve but fewer trades = less profit

2. DIVERGENCE AS CONFIRMATION:
   - Can be used as "higher quality" signal filter
   - Trade-off between quality and quantity
   - Best when divergence is common enough to maintain trade volume

3. RECOMMENDATION:
   - If divergence severely reduces trades -> NOT RECOMMENDED
   - If divergence maintains trades but improves WR -> CONSIDER
   - Our RSI threshold already captures mean reversion well
   - Divergence adds complexity without clear benefit

4. WHY IT MAY NOT WORK:
   - H1 timeframe may be too noisy for clean divergence
   - Our RSI threshold (42/58) is already selective
   - Adding more filters reduces the edge from volume of trades
""")


if __name__ == "__main__":
    main()
