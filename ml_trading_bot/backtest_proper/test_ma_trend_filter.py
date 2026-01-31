"""
Moving Average Trend Filter Test
================================
Test if trading only with the trend (defined by MA) improves results.

Theory:
- Price above MA = uptrend, only take BUY signals
- Price below MA = downtrend, only take SELL signals
- This is COUNTER to mean reversion (which trades against trend)

Sources:
- https://www.mindmathmoney.com/articles/the-ultimate-guide-to-the-rsi-indicator-mastering-rsi-trading-strategies-and-settings-2025
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

    # Calculate MAs
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma100'] = df['close'].rolling(100).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()

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
    print('MOVING AVERAGE TREND FILTER TEST')
    print('=' * 100)

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

    def backtest(ma_filter=None, with_trend=True, against_trend=False):
        """
        Backtest with MA trend filter.

        Args:
            ma_filter: Column name of MA to use (e.g., 'sma50')
            with_trend: Only take signals aligned with trend
            against_trend: Only take signals against trend (counter-trend)
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
                price = row['close']

                signal = 0

                if rsi < RSI_OS:
                    signal = 1
                elif rsi > RSI_OB:
                    signal = -1

                # Apply MA filter
                if signal and ma_filter:
                    ma_value = row[ma_filter]
                    if ma_value > 0:
                        uptrend = price > ma_value
                        downtrend = price < ma_value

                        if with_trend:
                            # Only BUY in uptrend, only SELL in downtrend
                            if signal == 1 and not uptrend:
                                filtered += 1
                                signal = 0
                            elif signal == -1 and not downtrend:
                                filtered += 1
                                signal = 0
                        elif against_trend:
                            # Only BUY in downtrend (counter-trend), SELL in uptrend
                            if signal == 1 and uptrend:
                                filtered += 1
                                signal = 0
                            elif signal == -1 and downtrend:
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

    # Test MA filters
    print('\n1. WITH-TREND FILTER (trade with MA trend)')
    print('-' * 80)
    print('   Note: Mean reversion SHOULD work against trend, so with-trend may hurt')
    print(f'\n   {"Filter":^30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 100)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest()
    print(f'   {"v3.7 Baseline":<30} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    for ma in ['sma20', 'sma50', 'sma100', 'sma200', 'ema50']:
        ret, trades, wr, max_dd, _, filtered = backtest(ma_filter=ma, with_trend=True)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   With {ma.upper():<24} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    print('\n2. AGAINST-TREND FILTER (counter-trend only)')
    print('-' * 80)
    print('   Note: This is classic mean reversion - buy when oversold in downtrend')
    print(f'\n   {"Filter":^30} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 100)

    print(f'   {"v3.7 Baseline":<30} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    for ma in ['sma20', 'sma50', 'sma100', 'sma200', 'ema50']:
        ret, trades, wr, max_dd, _, filtered = backtest(ma_filter=ma, against_trend=True)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   Against {ma.upper():<21} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. WITH-TREND FILTER:
   - Trading WITH trend filters out mean reversion signals
   - Mean reversion by definition trades AGAINST short-term moves
   - Expected to HURT performance

2. AGAINST-TREND FILTER:
   - Only taking signals against MA trend
   - May improve quality by confirming oversold in downtrend
   - Or may reduce opportunity too much

3. MEAN REVERSION PRINCIPLE:
   - Our strategy buys oversold (price dropped) and sells overbought (price rose)
   - This is inherently counter-trend on short timeframe
   - Adding trend filter conflicts with the strategy's core logic

4. RECOMMENDATION:
   - MA trend filter likely NOT beneficial for pure mean reversion
   - Keep the strategy trend-agnostic for maximum opportunity
""")


if __name__ == "__main__":
    main()
