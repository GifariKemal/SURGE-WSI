"""
Keltner Channel Test
====================
Test Keltner Channel for mean reversion signals.

Keltner Channel:
- Middle: EMA(20)
- Upper: EMA + ATR * multiplier
- Lower: EMA - ATR * multiplier

Mean Reversion Strategy:
- Buy when price touches/crosses below lower band
- Sell when price touches/crosses above upper band

This can REPLACE or COMBINE with RSI signals.

Sources:
- https://www.quantifiedstrategies.com/keltner-bands-trading-strategies/
- https://eodhd.com/financial-academy/backtesting-strategies-examples/algorithmic-trading-with-the-keltner-channel-in-python
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


def calculate_keltner(close, high, low, ema_period=20, atr_mult=2.0, atr_period=14):
    """
    Calculate Keltner Channel.
    Returns: middle (EMA), upper band, lower band
    """
    # Middle line (EMA)
    middle = close.ewm(span=ema_period, adjust=False).mean()

    # ATR
    tr = np.maximum(high - low,
                    np.maximum(abs(high - close.shift(1)),
                              abs(low - close.shift(1))))
    atr = tr.rolling(atr_period).mean()

    # Bands
    upper = middle + atr * atr_mult
    lower = middle - atr * atr_mult

    return middle, upper, lower, atr


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

    # RSI for comparison and combination
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
    print('KELTNER CHANNEL TEST')
    print('=' * 100)
    print('Keltner = EMA +/- (ATR * multiplier)')

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

    def backtest(use_keltner=False, keltner_only=False, ema_period=20, atr_mult=2.0,
                 require_both=False):
        """
        Backtest with Keltner Channel.

        use_keltner: Use Keltner as additional confirmation
        keltner_only: Use ONLY Keltner (ignore RSI)
        require_both: Require BOTH RSI and Keltner to agree
        """
        if use_keltner or keltner_only:
            middle, upper, lower, _ = calculate_keltner(
                df['close'], df['high'], df['low'],
                ema_period=ema_period, atr_mult=atr_mult
            )

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

                price = row['close']
                rsi = row['rsi']

                # Determine signal
                if keltner_only:
                    # Only use Keltner
                    kelt_signal = 0
                    if price <= lower.iloc[i]:
                        kelt_signal = 1  # Buy at lower band
                    elif price >= upper.iloc[i]:
                        kelt_signal = -1  # Sell at upper band
                    signal = kelt_signal
                elif require_both:
                    # Require both RSI and Keltner
                    rsi_signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)
                    kelt_signal = 0
                    if price <= lower.iloc[i]:
                        kelt_signal = 1
                    elif price >= upper.iloc[i]:
                        kelt_signal = -1
                    # Only trade if both agree
                    signal = rsi_signal if rsi_signal == kelt_signal else 0
                elif use_keltner:
                    # RSI signal with Keltner confirmation
                    rsi_signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)
                    if rsi_signal == 1 and price > lower.iloc[i]:
                        signal = 0  # RSI oversold but not at lower band
                    elif rsi_signal == -1 and price < upper.iloc[i]:
                        signal = 0  # RSI overbought but not at upper band
                    else:
                        signal = rsi_signal
                else:
                    # Standard RSI only (v3.7 baseline)
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

    # 1. Baseline
    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _ = backtest()
    print(f'\n1. v3.7 Baseline (RSI only): +{baseline_ret:.1f}% | Trades: {baseline_trades} | WR: {baseline_wr:.1f}% | MaxDD: {baseline_dd:.1f}%')

    # 2. Keltner Channel ONLY (replace RSI)
    print('\n2. KELTNER CHANNEL ONLY (replace RSI)')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    for ema_p, atr_m in [(10, 1.5), (15, 1.5), (20, 1.5), (20, 2.0), (20, 2.5), (20, 3.0), (30, 2.0)]:
        ret, trades, wr, max_dd, _ = backtest(keltner_only=True, ema_period=ema_p, atr_mult=atr_m)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        name = f'Keltner({ema_p}, {atr_m}x ATR)'
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 3. RSI + Keltner confirmation
    print('\n3. RSI + KELTNER CONFIRMATION')
    print('-' * 80)
    print('   RSI signal only valid if price at Keltner band')
    print(f'\n   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    for ema_p, atr_m in [(20, 1.5), (20, 2.0), (20, 2.5), (15, 2.0)]:
        ret, trades, wr, max_dd, _ = backtest(use_keltner=True, ema_period=ema_p, atr_mult=atr_m)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        name = f'RSI + Keltner({ema_p}, {atr_m}x)'
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 4. RSI AND Keltner must agree
    print('\n4. RSI AND KELTNER MUST AGREE')
    print('-' * 80)
    print(f'   {"Config":<35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 95)

    for ema_p, atr_m in [(20, 1.5), (20, 2.0), (20, 2.5)]:
        ret, trades, wr, max_dd, _ = backtest(require_both=True, ema_period=ema_p, atr_mult=atr_m)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        name = f'Both agree ({ema_p}, {atr_m}x)'
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print("""
1. KELTNER CHANNEL CONCEPT:
   - EMA-based channel with ATR bands
   - Price at lower band = potential buy (oversold)
   - Price at upper band = potential sell (overbought)
   - Similar concept to Bollinger Bands but uses ATR

2. COMPARISON:
   - Keltner ONLY: Uses price/band relationship
   - RSI ONLY: Uses momentum oscillator
   - Combined: May filter signals or add confluence

3. RECOMMENDATION:
   - If Keltner improves over RSI -> Consider switching
   - If combined improves both -> Consider confluence
   - If neither improves -> Keep simple RSI(10)
""")


if __name__ == "__main__":
    main()
