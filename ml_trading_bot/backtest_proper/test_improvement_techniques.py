"""
Test Improvement Techniques - NOT Filters
==========================================
Teknik yang MENINGKATKAN performance, bukan mengurangi trades

Techniques tested:
1. Kelly Criterion Position Sizing - Optimal risk based on win rate
2. Partial Exit (50% at midpoint) - Lock in profits earlier
3. Scaled Entry - Add to position when RSI more extreme
4. Dynamic Risk - Adjust risk based on consecutive wins/losses
5. Pyramiding - Add to winning positions

Based on research from:
- Kelly Criterion formula: f* = (W × R − L) / R
- Half-Kelly for smoother equity curve
- Partial profit taking strategies
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

    # Calculate indicators
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.4 BASELINE parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    def backtest_baseline(risk_pct=0.01):
        """v3.4 BASELINE - Current best"""
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
                    risk = balance * risk_pct
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_kelly():
        """
        Kelly Criterion Position Sizing
        Formula: f* = (W × R − L) / R
        where W = win rate, L = loss rate, R = reward/risk ratio

        With WR ~37% and R:R ~2:1, Kelly suggests ~6.1%
        We use Half-Kelly (~3%) for safety
        """
        # First pass to calculate win rate and R:R
        balance = 10000.0
        wins = losses = 0
        total_win_pips = 0
        total_loss_pips = 0
        position = None

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        total_loss_pips += abs(position['entry'] - position['sl'])
                        losses += 1
                        position = None
                    elif row['high'] >= position['tp']:
                        total_win_pips += abs(position['tp'] - position['entry'])
                        wins += 1
                        position = None
                else:
                    if row['high'] >= position['sl']:
                        total_loss_pips += abs(position['sl'] - position['entry'])
                        losses += 1
                        position = None
                    elif row['low'] <= position['tp']:
                        total_win_pips += abs(position['entry'] - position['tp'])
                        wins += 1
                        position = None

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
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp}

        trades = wins + losses
        if trades == 0:
            return 0, 0, 0, 0, 0, 10000

        W = wins / trades
        L = losses / trades
        avg_win = total_win_pips / wins if wins > 0 else 0
        avg_loss = total_loss_pips / losses if losses > 0 else 0.0001
        R = avg_win / avg_loss if avg_loss > 0 else 2.0

        # Kelly formula: f* = (W × R − L) / R
        kelly = (W * R - L) / R if R > 0 else 0
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%
        half_kelly = kelly / 2  # Half Kelly for safety

        print(f"    Kelly Analysis: WR={W*100:.1f}%, R:R={R:.2f}, Kelly={kelly*100:.1f}%, Half-Kelly={half_kelly*100:.1f}%")

        # Second pass with Half-Kelly sizing
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
                    risk = balance * half_kelly  # Half-Kelly risk
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_partial_exit():
        """
        Partial Exit Strategy
        - Close 50% at midpoint (entry + half of distance to TP)
        - Close remaining 50% at full TP
        - SL remains same for both portions
        """
        balance = 10000.0
        wins = losses = partial_wins = 0
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
                    # Check SL first
                    if row['low'] <= position['sl']:
                        # Hit SL - close all remaining
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    else:
                        # Check partial TP (midpoint)
                        if not position.get('partial_closed') and row['high'] >= position['partial_tp']:
                            # Close 50% at midpoint
                            partial_pnl = (position['partial_tp'] - position['entry']) * (position['size'] / 2)
                            balance += partial_pnl
                            year = df.index[i].year
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + partial_pnl
                            position['partial_closed'] = True
                            position['size'] = position['size'] / 2  # Remaining 50%
                            partial_wins += 1

                        # Check full TP
                        if row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            wins += 1
                            closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    else:
                        if not position.get('partial_closed') and row['low'] <= position['partial_tp']:
                            partial_pnl = (position['entry'] - position['partial_tp']) * (position['size'] / 2)
                            balance += partial_pnl
                            year = df.index[i].year
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + partial_pnl
                            position['partial_closed'] = True
                            position['size'] = position['size'] / 2
                            partial_wins += 1

                        if row['low'] <= position['tp']:
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

                    # Midpoint for partial exit
                    if signal == 1:
                        partial_tp = entry + (tp - entry) / 2
                    else:
                        partial_tp = entry - (entry - tp) / 2

                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {
                        'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp,
                        'partial_tp': partial_tp, 'size': size, 'partial_closed': False
                    }

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        print(f"    Partial exits triggered: {partial_wins}")
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_scaled_entry():
        """
        Scaled Entry Strategy
        - Initial entry at RSI 42/58 with 50% size
        - Add 50% more if RSI reaches 35/65 (more extreme)
        - Same SL/TP for both portions
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        scale_ins = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                # Check for scale-in opportunity
                if not position.get('scaled') and row['hour'] >= 7 and row['hour'] < 22:
                    rsi = row['rsi']
                    if position['dir'] == 1 and rsi < 35:
                        # Add to long position
                        add_risk = balance * 0.005  # 0.5% more
                        add_size = min(add_risk / abs(position['entry'] - position['sl']), 50000)
                        position['size'] += add_size
                        position['scaled'] = True
                        scale_ins += 1
                    elif position['dir'] == -1 and rsi > 65:
                        # Add to short position
                        add_risk = balance * 0.005
                        add_size = min(add_risk / abs(position['sl'] - position['entry']), 50000)
                        position['size'] += add_size
                        position['scaled'] = True
                        scale_ins += 1

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
                    risk = balance * 0.005  # Start with 0.5% (will add 0.5% more on scale-in)
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'scaled': False}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        print(f"    Scale-ins triggered: {scale_ins}")
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_dynamic_risk():
        """
        Dynamic Risk based on streak
        - After 2 consecutive wins: increase risk to 1.5%
        - After 2 consecutive losses: reduce risk to 0.5%
        - Otherwise: 1% baseline
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        streak = 0  # Positive = wins, Negative = losses

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
                        streak = streak - 1 if streak < 0 else -1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        streak = streak + 1 if streak > 0 else 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        streak = streak - 1 if streak < 0 else -1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        streak = streak + 1 if streak > 0 else 1
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

                    # Dynamic risk based on streak
                    if streak >= 2:
                        risk_pct = 0.015  # 1.5% after 2+ wins
                    elif streak <= -2:
                        risk_pct = 0.005  # 0.5% after 2+ losses
                    else:
                        risk_pct = 0.01  # 1% baseline

                    risk = balance * risk_pct
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        return ret, trades, wr, max_dd, prof_yrs, balance

    def backtest_trailing_stop():
        """
        Trailing Stop Strategy
        - Move SL to breakeven when price reaches 1.5x ATR profit
        - Trail SL at 1.5x ATR behind price
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        be_triggered = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['weekday'] >= 5:
                continue

            if position:
                pnl = 0
                closed = False

                # Update trailing stop
                if position['dir'] == 1:
                    profit_pips = row['high'] - position['entry']
                    if profit_pips >= position['atr'] * 1.5:
                        # Move SL to breakeven + small profit
                        new_sl = max(position['sl'], position['entry'] + position['atr'] * 0.5)
                        if new_sl > position['sl']:
                            if position['sl'] < position['entry']:
                                be_triggered += 1
                            position['sl'] = new_sl

                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    profit_pips = position['entry'] - row['low']
                    if profit_pips >= position['atr'] * 1.5:
                        new_sl = min(position['sl'], position['entry'] - position['atr'] * 0.5)
                        if new_sl < position['sl']:
                            if position['sl'] > position['entry']:
                                be_triggered += 1
                            position['sl'] = new_sl

                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        if pnl > 0:
                            wins += 1
                        else:
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
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'atr': atr}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        prof_yrs = sum(1 for v in yearly_pnl.values() if v > 0)
        print(f"    Breakeven triggered: {be_triggered}")
        return ret, trades, wr, max_dd, prof_yrs, balance

    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    print('=' * 100)
    print('TESTING IMPROVEMENT TECHNIQUES - NOT FILTERS')
    print('These techniques aim to IMPROVE returns, not reduce trades')
    print('=' * 100)

    print('\n' + '-' * 100)
    print('1. BASELINE v3.4 (RSI 42/58, 1% Risk)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_baseline()
    baseline_ret = ret
    print(f"   Return: +{ret:.1f}% | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}% | Profitable: {prof_yrs}/6 years")
    print(f"   Final Balance: ${final_bal:,.2f}")

    print('\n' + '-' * 100)
    print('2. KELLY CRITERION POSITION SIZING (Half-Kelly)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_kelly()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")
    print(f"   Final Balance: ${final_bal:,.2f}")

    print('\n' + '-' * 100)
    print('3. PARTIAL EXIT (50% at midpoint, 50% at full TP)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_partial_exit()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")
    print(f"   Final Balance: ${final_bal:,.2f}")

    print('\n' + '-' * 100)
    print('4. SCALED ENTRY (Add 50% when RSI more extreme)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_scaled_entry()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")
    print(f"   Final Balance: ${final_bal:,.2f}")

    print('\n' + '-' * 100)
    print('5. DYNAMIC RISK (Adjust based on win/loss streak)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_dynamic_risk()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")
    print(f"   Final Balance: ${final_bal:,.2f}")

    print('\n' + '-' * 100)
    print('6. TRAILING STOP (Move SL to breakeven + trail)')
    print('-' * 100)
    ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_trailing_stop()
    diff = ret - baseline_ret
    marker = ' <-- BETTER!' if diff > 0 else ''
    print(f"   Return: +{ret:.1f}% ({diff:+.1f}) | Trades: {trades} | WR: {wr:.1f}% | MaxDD: {max_dd:.1f}%{marker}")
    print(f"   Final Balance: ${final_bal:,.2f}")

    # Test different risk levels
    print('\n' + '=' * 100)
    print('RISK LEVEL COMPARISON')
    print('=' * 100)

    for risk in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        ret, trades, wr, max_dd, prof_yrs, final_bal = backtest_baseline(risk_pct=risk)
        print(f"   Risk {risk*100:.1f}%: Return +{ret:.1f}% | MaxDD {max_dd:.1f}% | Final ${final_bal:,.2f}")

    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)


if __name__ == "__main__":
    main()
