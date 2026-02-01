"""
Analyze Losing Months Deep Dive
================================
What's different about Nov 2024 and Apr 2025?
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

MT5_LOGIN = 61045904
MT5_PASSWORD = "iy#K5L7sF"
MT5_SERVER = "FinexBisnisSolusi-Demo"
MT5_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

def connect_mt5():
    if not mt5.initialize(path=MT5_PATH):
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return False
    return True

def get_h1_data(symbol, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def prepare_indicators(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean()

    def atr_pct_func(x):
        if len(x) == 0: return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(24).std() * 100
    df['vol_ma'] = df['volatility'].rolling(120).mean()
    df['vol_ratio'] = df['volatility'] / df['vol_ma']

    # Range size
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)

def get_trade_details(df, test_start='2024-10-01', test_end='2026-02-01'):
    """Run backtest and return individual trade details."""
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    MAX_HOLDING = 46

    trades = []
    position = None

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        in_test = current_time >= pd.Timestamp(test_start) and current_time < pd.Timestamp(test_end)

        month_str = current_time.strftime('%Y-%m')
        weekday = row['weekday']
        hour = row['hour']

        if weekday >= 5:
            continue

        if position:
            exit_reason = None
            pnl = 0

            if (i - position['entry_idx']) >= MAX_HOLDING:
                pnl = (row['close'] - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - row['close']) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL'
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP'
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL'
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP'

            if exit_reason:
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'month': position['entry_time'].strftime('%Y-%m'),
                    'direction': 'LONG' if position['dir'] == 1 else 'SHORT',
                    'entry': position['entry'],
                    'sl': position['sl'],
                    'tp': position['tp'],
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'regime': position['regime'],
                    'rsi': position['rsi'],
                    'atr_pct': position['atr_pct'],
                    'volatility': position['volatility'],
                    'vol_ratio': position['vol_ratio'],
                    'hour': position['hour'],
                    'weekday': position['entry_time'].weekday()
                }
                trades.append(trade)
                position = None

        if not position and in_test:
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < 20 or atr_pct > 80:
                continue

            # Only SIDEWAYS
            if row['regime'] != 'SIDEWAYS':
                continue

            rsi = row['rsi']
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                tp_mult = 3.0 + (0.35 if 12 <= hour < 16 else 0)
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = 10000 * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {
                    'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i,
                    'entry_time': current_time, 'regime': row['regime'], 'rsi': rsi,
                    'atr_pct': atr_pct, 'volatility': row['volatility'], 'vol_ratio': row['vol_ratio'],
                    'hour': hour
                }

    return pd.DataFrame(trades)

def main():
    print("=" * 70)
    print("DEEP ANALYSIS: LOSING MONTHS")
    print("What's different about Nov 2024 and Apr 2025?")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data("GBPUSD", start_date, end_date)

        if df is None:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")
        df = prepare_indicators(df)

        # Get all trades
        trades_df = get_trade_details(df)
        print(f"\nTotal trades: {len(trades_df)}")

        # Separate by month type
        losing_months = ['2024-11', '2025-04']
        winning_months = [m for m in trades_df['month'].unique() if m not in losing_months]

        lose_trades = trades_df[trades_df['month'].isin(losing_months)]
        win_trades = trades_df[trades_df['month'].isin(winning_months)]

        print(f"\nLosing month trades: {len(lose_trades)}")
        print(f"Winning month trades: {len(win_trades)}")

        # Compare statistics
        print("\n" + "=" * 70)
        print("COMPARISON: LOSING vs WINNING MONTHS")
        print("=" * 70)

        metrics = ['rsi', 'atr_pct', 'volatility', 'vol_ratio', 'hour']

        print(f"\n{'Metric':<15} {'Losing Mo':<15} {'Winning Mo':<15} {'Difference':<15}")
        print("-" * 60)

        for metric in metrics:
            lose_mean = lose_trades[metric].mean()
            win_mean = win_trades[metric].mean()
            diff = lose_mean - win_mean
            print(f"{metric:<15} {lose_mean:<15.2f} {win_mean:<15.2f} {diff:+.2f}")

        # Win rate comparison
        lose_wr = (lose_trades['pnl'] > 0).mean() * 100
        win_wr = (win_trades['pnl'] > 0).mean() * 100
        print(f"\n{'Win Rate':<15} {lose_wr:<15.1f}% {win_wr:<15.1f}% {lose_wr - win_wr:+.1f}%")

        # Direction breakdown
        print("\n" + "=" * 70)
        print("DIRECTION ANALYSIS")
        print("=" * 70)

        for month_type, trades in [('LOSING', lose_trades), ('WINNING', win_trades)]:
            print(f"\n{month_type} MONTHS:")
            for direction in ['LONG', 'SHORT']:
                dir_trades = trades[trades['direction'] == direction]
                if len(dir_trades) > 0:
                    dir_wr = (dir_trades['pnl'] > 0).mean() * 100
                    dir_pnl = dir_trades['pnl'].sum()
                    print(f"  {direction}: {len(dir_trades)} trades, WR={dir_wr:.1f}%, P/L=${dir_pnl:+,.2f}")

        # Exit reason breakdown
        print("\n" + "=" * 70)
        print("EXIT REASON ANALYSIS")
        print("=" * 70)

        for month_type, trades in [('LOSING', lose_trades), ('WINNING', win_trades)]:
            print(f"\n{month_type} MONTHS:")
            for reason in ['TP', 'SL', 'TIMEOUT']:
                reason_trades = trades[trades['exit_reason'] == reason]
                if len(reason_trades) > 0:
                    pct = len(reason_trades) / len(trades) * 100
                    reason_pnl = reason_trades['pnl'].sum()
                    print(f"  {reason}: {len(reason_trades)} ({pct:.1f}%), P/L=${reason_pnl:+,.2f}")

        # Hour analysis
        print("\n" + "=" * 70)
        print("HOUR ANALYSIS (Entry Hour)")
        print("=" * 70)

        print("\nLOSING MONTHS - Trades by hour:")
        lose_hour = lose_trades.groupby('hour').agg({'pnl': ['count', 'sum', 'mean']}).round(2)
        lose_hour.columns = ['Count', 'Total P/L', 'Avg P/L']
        print(lose_hour.to_string())

        print("\nWINNING MONTHS - Avg P/L by hour:")
        win_hour = win_trades.groupby('hour')['pnl'].mean().round(2)
        print(win_hour.to_string())

        # Specific losing months breakdown
        print("\n" + "=" * 70)
        print("INDIVIDUAL LOSING MONTH BREAKDOWN")
        print("=" * 70)

        for month in losing_months:
            month_trades = trades_df[trades_df['month'] == month]
            print(f"\n{month}:")
            print(f"  Total trades: {len(month_trades)}")
            print(f"  Win rate: {(month_trades['pnl'] > 0).mean()*100:.1f}%")
            print(f"  Total P/L: ${month_trades['pnl'].sum():+,.2f}")
            print(f"  Avg RSI at entry: {month_trades['rsi'].mean():.1f}")
            print(f"  Avg Vol Ratio: {month_trades['vol_ratio'].mean():.2f}")

            # Show worst trades
            worst = month_trades.nsmallest(3, 'pnl')
            print(f"  Worst trades:")
            for _, t in worst.iterrows():
                print(f"    {t['entry_time']}: {t['direction']} ${t['pnl']:+.2f} ({t['exit_reason']})")

        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)

        # Check if specific hours are problematic
        lose_hour_pnl = lose_trades.groupby('hour')['pnl'].sum()
        worst_hours = lose_hour_pnl.nsmallest(3).index.tolist()
        print(f"\n1. Worst hours in losing months: {worst_hours}")

        # Check vol_ratio
        lose_high_vol = lose_trades[lose_trades['vol_ratio'] > 1.5]['pnl'].sum()
        lose_low_vol = lose_trades[lose_trades['vol_ratio'] <= 1.5]['pnl'].sum()
        print(f"\n2. Vol Ratio impact in losing months:")
        print(f"   High vol (>1.5): ${lose_high_vol:+,.2f}")
        print(f"   Low vol (<=1.5): ${lose_low_vol:+,.2f}")

        # Check weekday
        lose_weekday = lose_trades.groupby('weekday')['pnl'].sum()
        print(f"\n3. Weekday P/L in losing months:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        for d, pnl in lose_weekday.items():
            print(f"   {days[d]}: ${pnl:+,.2f}")

    finally:
        mt5.shutdown()
        print("\nDone!")

if __name__ == "__main__":
    main()
