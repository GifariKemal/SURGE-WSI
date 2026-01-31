"""
Test More Optimization Techniques for RSI v3.7
===============================================
Ideas from research:
1. RSI Slope/Momentum - Only enter when RSI is turning
2. Volatility Expansion Filter - Trade when volatility is expanding
3. Previous Candle Filter - Confirmation from previous candle
4. RSI Threshold Tightening - Tighter thresholds (40/60 vs 42/58)
5. Multi-bar RSI Confirmation - RSI must be oversold for 2+ bars
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

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # RSI change (slope)
    df['rsi_change'] = df['rsi'].diff()

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    # ATR expansion (current vs previous)
    df['atr_expanding'] = df['atr'] > df['atr'].shift(1)

    # Previous candle info
    df['prev_bullish'] = df['close'].shift(1) > df['open'].shift(1)
    df['prev_bearish'] = df['close'].shift(1) < df['open'].shift(1)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.7 baseline parameters
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

    def backtest(rsi_slope_filter=False, atr_expanding_filter=False,
                 prev_candle_filter=False, rsi_os=RSI_OS, rsi_ob=RSI_OB,
                 multi_bar_rsi=False):
        """v3.7 with optional extra filters"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        filtered_count = 0

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
                rsi_chg = row['rsi_change']

                # Check RSI signal with custom thresholds
                if rsi < rsi_os:
                    signal = 1
                elif rsi > rsi_ob:
                    signal = -1
                else:
                    continue

                # RSI Slope Filter - RSI should be turning
                if rsi_slope_filter:
                    if signal == 1 and rsi_chg < 0:  # BUY but RSI still falling
                        filtered_count += 1
                        continue
                    if signal == -1 and rsi_chg > 0:  # SELL but RSI still rising
                        filtered_count += 1
                        continue

                # ATR Expanding Filter - Only trade when volatility expanding
                if atr_expanding_filter and not row['atr_expanding']:
                    filtered_count += 1
                    continue

                # Previous Candle Confirmation
                if prev_candle_filter:
                    if signal == 1 and not row['prev_bearish']:  # BUY needs prev bearish
                        filtered_count += 1
                        continue
                    if signal == -1 and not row['prev_bullish']:  # SELL needs prev bullish
                        filtered_count += 1
                        continue

                # Multi-bar RSI confirmation
                if multi_bar_rsi:
                    prev_rsi = df.iloc[i-1]['rsi']
                    if signal == 1 and prev_rsi >= rsi_os:  # BUY needs 2 bars oversold
                        filtered_count += 1
                        continue
                    if signal == -1 and prev_rsi <= rsi_ob:  # SELL needs 2 bars overbought
                        filtered_count += 1
                        continue

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
        return ret, trades, wr, max_dd, filtered_count

    print('=' * 100)
    print('TESTING MORE OPTIMIZATION TECHNIQUES')
    print('=' * 100)

    # 1. BASELINE (v3.7)
    ret, trades, wr, max_dd, _ = backtest()
    baseline_ret = ret
    print(f"\n1. BASELINE (v3.7)")
    print(f"   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%")

    # 2. RSI SLOPE FILTER
    print(f"\n2. RSI SLOPE FILTER (RSI must be turning)")
    print("-" * 80)
    ret, trades, wr, max_dd, filtered = backtest(rsi_slope_filter=True)
    diff = ret - baseline_ret
    print(f"   Result: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | Filtered: {filtered}")

    # 3. ATR EXPANDING FILTER
    print(f"\n3. ATR EXPANDING FILTER (volatility must be expanding)")
    print("-" * 80)
    ret, trades, wr, max_dd, filtered = backtest(atr_expanding_filter=True)
    diff = ret - baseline_ret
    print(f"   Result: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | Filtered: {filtered}")

    # 4. PREVIOUS CANDLE FILTER
    print(f"\n4. PREVIOUS CANDLE FILTER (confirmation from prev candle)")
    print("-" * 80)
    ret, trades, wr, max_dd, filtered = backtest(prev_candle_filter=True)
    diff = ret - baseline_ret
    print(f"   Result: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | Filtered: {filtered}")

    # 5. MULTI-BAR RSI
    print(f"\n5. MULTI-BAR RSI (RSI oversold/overbought for 2+ bars)")
    print("-" * 80)
    ret, trades, wr, max_dd, filtered = backtest(multi_bar_rsi=True)
    diff = ret - baseline_ret
    print(f"   Result: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | Filtered: {filtered}")

    # 6. RSI THRESHOLD VARIATIONS
    print(f"\n6. RSI THRESHOLD VARIATIONS")
    print("-" * 80)

    for rsi_os, rsi_ob in [(40, 60), (41, 59), (43, 57), (44, 56), (45, 55)]:
        ret, trades, wr, max_dd, _ = backtest(rsi_os=rsi_os, rsi_ob=rsi_ob)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 10 else ''
        print(f"   RSI {rsi_os}/{rsi_ob}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    # 7. COMBINATIONS
    print(f"\n7. COMBINATION TESTS")
    print("-" * 80)

    combos = [
        (True, False, False, False, "RSI Slope only"),
        (False, True, False, False, "ATR Expanding only"),
        (False, False, True, False, "Prev Candle only"),
        (True, True, False, False, "RSI Slope + ATR Expanding"),
        (True, False, True, False, "RSI Slope + Prev Candle"),
        (False, True, True, False, "ATR Expanding + Prev Candle"),
    ]

    for slope, atr_exp, prev, multi, name in combos:
        ret, trades, wr, max_dd, filtered = backtest(
            rsi_slope_filter=slope, atr_expanding_filter=atr_exp,
            prev_candle_filter=prev, multi_bar_rsi=multi
        )
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 10 else ''
        print(f"   {name:30s}: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}%{marker}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Baseline v3.7: +{baseline_ret:.1f}%")
    print("\nNote: Most filters reduce trades without improving win rate.")
    print("The RSI strategy works best with FEWER filters, not more.")


if __name__ == "__main__":
    main()
