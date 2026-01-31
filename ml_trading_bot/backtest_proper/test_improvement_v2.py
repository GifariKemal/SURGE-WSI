"""
Test Improvement Techniques v2 - Focus on Probability Enhancement
=================================================================
Keep 1% risk - find techniques that improve WIN RATE or R:R

Ideas tested:
1. Asymmetric TP - Different TP for BUY vs SELL
2. Time-based TP adjustment - Larger TP during high volume hours
3. RSI momentum confirmation - Only trade when RSI moving in signal direction
4. Multiple timeframe RSI - Confirm with higher timeframe
5. Price action confirmation - Only trade at support/resistance
6. Improved exit timing
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

    # Load H1 data
    df = pd.read_sql("""
        SELECT time, open, high, low, close
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)

    # Load H4 data for multi-timeframe
    df_h4 = pd.read_sql("""
        SELECT time, open, high, low, close
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H4'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)
    conn.close()

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df_h4['time'] = pd.to_datetime(df_h4['time'])
    df_h4.set_index('time', inplace=True)

    # H1 Indicators
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_prev'] = df['rsi'].shift(1)  # Previous RSI for momentum
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    # Support/Resistance levels (rolling high/low)
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()
    df['range'] = df['resistance'] - df['support']

    # H4 RSI
    delta_h4 = df_h4['close'].diff()
    gain_h4 = delta_h4.where(delta_h4 > 0, 0).rolling(10).mean()
    loss_h4 = (-delta_h4.where(delta_h4 < 0, 0)).rolling(10).mean()
    df_h4['rsi'] = 100 - (100 / (1 + gain_h4 / (loss_h4 + 1e-10)))

    df = df.ffill().fillna(0)
    df_h4 = df_h4.ffill().fillna(0)

    # v3.4 BASELINE parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    def get_h4_rsi(timestamp):
        """Get H4 RSI for given H1 timestamp"""
        h4_time = timestamp.floor('4H')
        if h4_time in df_h4.index:
            return df_h4.loc[h4_time, 'rsi']
        else:
            # Find nearest H4 bar
            idx = df_h4.index.get_indexer([h4_time], method='ffill')[0]
            if idx >= 0 and idx < len(df_h4):
                return df_h4.iloc[idx]['rsi']
        return 50  # Neutral default

    def backtest_baseline():
        """v3.4 BASELINE"""
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    tp_mult = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_asymmetric_tp(buy_bonus=0, sell_bonus=0):
        """
        Asymmetric TP for BUY vs SELL
        Research shows BUY has higher WR, so we can use larger TP for BUY
        """
        balance = 10000.0
        wins = losses = 0
        buy_wins = buy_losses = sell_wins = sell_losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        buy_losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        buy_wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        sell_losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        sell_wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)

                    # Asymmetric TP
                    if signal == 1:
                        tp_mult = base_tp + buy_bonus
                    else:
                        tp_mult = base_tp + sell_bonus

                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)

        buy_t = buy_wins + buy_losses
        sell_t = sell_wins + sell_losses
        buy_wr = buy_wins / buy_t * 100 if buy_t else 0
        sell_wr = sell_wins / sell_t * 100 if sell_t else 0

        return ret, trades, wr, max_dd, prof_yrs, balance, buy_wr, sell_wr

    def backtest_time_tp():
        """
        Time-based TP adjustment
        - Larger TP during London+NY overlap (12-16 UTC) - more momentum
        - Normal TP during London morning (7-12) and NY afternoon (16-22)
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                hour = row['hour']
                if hour < 7 or hour >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)

                    # Time-based TP bonus
                    if 12 <= hour < 16:  # London+NY overlap
                        tp_mult = base_tp + 0.5  # Larger TP during high volume
                    else:
                        tp_mult = base_tp

                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_rsi_momentum():
        """
        RSI Momentum Confirmation
        - Only BUY when RSI is rising (current > previous)
        - Only SELL when RSI is falling (current < previous)
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
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                rsi_prev = row['rsi_prev']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                # Momentum filter - RSI should be moving in reversal direction
                if signal == 1 and rsi <= rsi_prev:  # Want RSI rising for BUY
                    filtered += 1
                    continue
                if signal == -1 and rsi >= rsi_prev:  # Want RSI falling for SELL
                    filtered += 1
                    continue

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    tp_mult = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        print(f"    Filtered by momentum: {filtered}")
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_h4_confirmation():
        """
        H4 RSI Confirmation
        - Only BUY when H4 RSI also < 50 (confirms oversold)
        - Only SELL when H4 RSI also > 50 (confirms overbought)
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
            timestamp = df.index[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                h4_rsi = get_h4_rsi(timestamp)
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                # H4 confirmation
                if signal == 1 and h4_rsi > 55:  # Want H4 RSI also oversold-ish
                    filtered += 1
                    continue
                if signal == -1 and h4_rsi < 45:  # Want H4 RSI also overbought-ish
                    filtered += 1
                    continue

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    tp_mult = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        print(f"    Filtered by H4 RSI: {filtered}")
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_sr_proximity():
        """
        Support/Resistance Proximity
        - BUY only when price is near support (bottom 30% of range)
        - SELL only when price is near resistance (top 30% of range)
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
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                # S/R proximity check
                if row['range'] > 0:
                    price_position = (row['close'] - row['support']) / row['range']
                else:
                    price_position = 0.5

                if signal == 1 and price_position > 0.4:  # Want price near support for BUY
                    filtered += 1
                    continue
                if signal == -1 and price_position < 0.6:  # Want price near resistance for SELL
                    filtered += 1
                    continue

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    tp_mult = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        print(f"    Filtered by S/R: {filtered}")
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_wider_tp():
        """
        Test wider TP multipliers to capture more profit
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        tp_low = 3.0
        tp_med = 4.0
        tp_high = 5.0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

                if closed:
                    balance += pnl
                    year = df.index[i].year
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if row['hour'] < 7 or row['hour'] >= 22:
                    continue
                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    tp_mult = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_med)
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, balance

    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    print('=' * 100)
    print('TESTING IMPROVEMENT TECHNIQUES v2 - Probability Enhancement')
    print('Focus: Improve WR or R:R while keeping 1% risk')
    print('=' * 100)

    print('\n' + '-' * 100)
    print('1. BASELINE v3.4 (RSI 42/58, 1% Risk)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_baseline()
    baseline_ret = ret
    print(f"   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | Profitable: {prof_yrs}/6 years")
    print(f"   Final Balance: ${final_bal:,.2f}")

    print('\n' + '-' * 100)
    print('2. ASYMMETRIC TP (Different TP for BUY vs SELL)')
    print('-' * 100)

    asymmetric_tests = [
        (0.3, 0, 'BUY +0.3x'),
        (0.5, 0, 'BUY +0.5x'),
        (0, -0.3, 'SELL -0.3x (faster exit)'),
        (0, -0.5, 'SELL -0.5x'),
        (0.3, -0.3, 'BUY +0.3, SELL -0.3'),
        (0.5, -0.5, 'BUY +0.5, SELL -0.5'),
    ]

    for buy_b, sell_b, name in asymmetric_tests:
        ret, trades, wr, max_dd, prof_yrs, final_bal, buy_wr, sell_wr = backtest_asymmetric_tp(buy_b, sell_b)
        diff = ret - baseline_ret
        marker = ' <-- BETTER!' if diff > 0 else ''
        print(f"   {name:<25}: +{ret:.1f}% ({diff:+.1f}) | BUY WR {buy_wr:.1f}% | SELL WR {sell_wr:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('3. TIME-BASED TP (+0.5x during London+NY overlap)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_time_tp()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('4. RSI MOMENTUM CONFIRMATION (trade only when RSI reversing)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_rsi_momentum()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('5. H4 RSI CONFIRMATION (higher timeframe alignment)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_h4_confirmation()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('6. SUPPORT/RESISTANCE PROXIMITY (trade only near S/R)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_sr_proximity()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '-' * 100)
    print('7. WIDER TP (3.0/4.0/5.0 instead of 2.4/3.0/3.6)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_wider_tp()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")

    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)


if __name__ == "__main__":
    main()
