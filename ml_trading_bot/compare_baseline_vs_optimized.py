"""
Compare RSI v3.7 Baseline vs Optimized (SIDEWAYS + ConsecLoss3)
================================================================
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

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)

def run_backtest(df, use_regime_filter=False, use_consec_filter=False, consec_limit=3,
                 test_start='2024-10-01', test_end='2026-02-01'):
    """Run RSI v3.7 backtest with optional filters."""
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MAX_HOLDING = 46

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    trades = []
    consecutive_losses = 0
    trades_filtered = 0

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
                balance += pnl
                if pnl > 0:
                    wins += 1
                    consecutive_losses = 0
                else:
                    losses += 1
                    consecutive_losses += 1

                if month_str not in monthly_pnl:
                    monthly_pnl[month_str] = 0
                monthly_pnl[month_str] += pnl

                trades.append({
                    'time': current_time,
                    'month': month_str,
                    'pnl': pnl,
                    'exit': exit_reason
                })
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        if not position and in_test:
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < 20 or atr_pct > 80:
                continue

            skip = False

            # Regime filter
            if use_regime_filter and row['regime'] != 'SIDEWAYS':
                skip = True
                trades_filtered += 1

            # Consecutive loss filter
            if not skip and use_consec_filter and consecutive_losses >= consec_limit:
                skip = True
                trades_filtered += 1
                consecutive_losses = 0

            if skip:
                continue

            rsi = row['rsi']
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                tp_mult = base_tp + 0.35 if 12 <= hour < 16 else base_tp
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - 10000) / 10000 * 100
    profit_factor = (sum(t['pnl'] for t in trades if t['pnl'] > 0) /
                     abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))) if any(t['pnl'] < 0 for t in trades) else 0

    return {
        'final_balance': balance,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'monthly_pnl': monthly_pnl,
        'trades_filtered': trades_filtered
    }

def main():
    print("=" * 70)
    print("RSI v3.7 COMPARISON")
    print("Baseline vs Optimized (SIDEWAYS + ConsecLoss3)")
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

        print(f"Data: {len(df)} bars")
        print(f"Test Period: Oct 2024 - Jan 2026")
        df = prepare_indicators(df)

        # Run both versions
        print("\nRunning backtests...")

        baseline = run_backtest(df.copy(), use_regime_filter=False, use_consec_filter=False)
        optimized = run_backtest(df.copy(), use_regime_filter=True, use_consec_filter=True, consec_limit=3)

        # Comparison table
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)

        metrics = [
            ('Final Balance', f"${baseline['final_balance']:,.2f}", f"${optimized['final_balance']:,.2f}"),
            ('Total Return', f"{baseline['total_return']:+.1f}%", f"{optimized['total_return']:+.1f}%"),
            ('Max Drawdown', f"{baseline['max_drawdown']:.1f}%", f"{optimized['max_drawdown']:.1f}%"),
            ('Total Trades', f"{baseline['total_trades']}", f"{optimized['total_trades']}"),
            ('Wins', f"{baseline['wins']}", f"{optimized['wins']}"),
            ('Losses', f"{baseline['losses']}", f"{optimized['losses']}"),
            ('Win Rate', f"{baseline['win_rate']:.1f}%", f"{optimized['win_rate']:.1f}%"),
            ('Profit Factor', f"{baseline['profit_factor']:.2f}", f"{optimized['profit_factor']:.2f}"),
            ('Trades Filtered', f"{baseline['trades_filtered']}", f"{optimized['trades_filtered']}"),
        ]

        print(f"\n{'Metric':<20} {'Baseline':<20} {'Optimized':<20} {'Better':<10}")
        print("-" * 70)

        for metric, base_val, opt_val in metrics:
            # Determine which is better
            better = ""
            if metric in ['Total Return', 'Win Rate', 'Profit Factor', 'Wins']:
                # Higher is better
                base_num = float(base_val.replace('$', '').replace(',', '').replace('%', '').replace('+', ''))
                opt_num = float(opt_val.replace('$', '').replace(',', '').replace('%', '').replace('+', ''))
                better = "OPT" if opt_num > base_num else ("BASE" if base_num > opt_num else "=")
            elif metric in ['Max Drawdown', 'Losses']:
                # Lower is better
                base_num = float(base_val.replace('$', '').replace(',', '').replace('%', ''))
                opt_num = float(opt_val.replace('$', '').replace(',', '').replace('%', ''))
                better = "OPT" if opt_num < base_num else ("BASE" if base_num < opt_num else "=")

            print(f"{metric:<20} {base_val:<20} {opt_val:<20} {better:<10}")

        # Monthly comparison
        print("\n" + "=" * 70)
        print("MONTHLY P/L COMPARISON")
        print("=" * 70)

        all_months = sorted(set(list(baseline['monthly_pnl'].keys()) + list(optimized['monthly_pnl'].keys())))

        base_losing = 0
        opt_losing = 0

        print(f"\n{'Month':<12} {'Baseline':<15} {'Optimized':<15} {'Diff':<12} {'Notes'}")
        print("-" * 70)

        for month in all_months:
            base_pnl = baseline['monthly_pnl'].get(month, 0)
            opt_pnl = optimized['monthly_pnl'].get(month, 0)
            diff = opt_pnl - base_pnl

            base_status = "LOSS" if base_pnl < 0 else ""
            opt_status = "LOSS" if opt_pnl < 0 else ""

            if base_pnl < 0:
                base_losing += 1
            if opt_pnl < 0:
                opt_losing += 1

            notes = ""
            if base_pnl < 0 and opt_pnl >= 0:
                notes = "FIXED!"
            elif base_pnl >= 0 and opt_pnl < 0:
                notes = "WORSE"
            elif base_pnl < 0 and opt_pnl < 0:
                if opt_pnl > base_pnl:
                    notes = "Improved"
                else:
                    notes = "Worse loss"

            print(f"{month:<12} ${base_pnl:>+10,.2f}  ${opt_pnl:>+10,.2f}  ${diff:>+8,.2f}  {notes}")

        print("-" * 70)
        print(f"{'LOSING MONTHS':<12} {base_losing:<15} {opt_losing:<15}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(f"""
BASELINE (RSI v3.7 Original):
  - Return: {baseline['total_return']:+.1f}%
  - Drawdown: {baseline['max_drawdown']:.1f}%
  - Trades: {baseline['total_trades']}
  - Win Rate: {baseline['win_rate']:.1f}%
  - Losing Months: {base_losing}

OPTIMIZED (SIDEWAYS + ConsecLoss3):
  - Return: {optimized['total_return']:+.1f}%
  - Drawdown: {optimized['max_drawdown']:.1f}%
  - Trades: {optimized['total_trades']}
  - Win Rate: {optimized['win_rate']:.1f}%
  - Losing Months: {opt_losing}

IMPROVEMENT:
  - Losing months reduced: {base_losing} -> {opt_losing} ({base_losing - opt_losing} fewer)
  - Drawdown reduced: {baseline['max_drawdown']:.1f}% -> {optimized['max_drawdown']:.1f}%
  - Return change: {baseline['total_return']:+.1f}% -> {optimized['total_return']:+.1f}%
""")

        if opt_losing < base_losing:
            print(">>> RECOMMENDATION: Use OPTIMIZED version <<<")
        else:
            print(">>> RECOMMENDATION: Use BASELINE version <<<")

    finally:
        mt5.shutdown()
        print("\nDone!")

if __name__ == "__main__":
    main()
