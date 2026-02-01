"""
RSI v3.7 Deep Parameter Sweep
=============================
Exhaustive search for 0 losing months configuration.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# MT5 Connection
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
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def run_backtest(df, params):
    RSI_OS = params['rsi_os']
    RSI_OB = params['rsi_ob']
    SL_MULT = params['sl_mult']
    TP_LOW = params['tp_low']
    TP_MED = params['tp_med']
    TP_HIGH = params['tp_high']
    MAX_HOLDING = params['max_holding']
    MIN_ATR_PCT = params['atr_min']
    MAX_ATR_PCT = params['atr_max']
    TIME_TP_BONUS = params.get('tp_bonus', 0.35)
    SKIP_HOURS = params.get('skip_hours', [12])
    START_HOUR = params.get('start_hour', 7)
    END_HOUR = params.get('end_hour', 22)

    # RSI(10) using SMA
    rsi_period = 10
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = np.where(loss == 0, 100, gain / loss)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    # ATR(14)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()

    def atr_percentile(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0

    df['atr_pct'] = df['atr'].rolling(100).apply(atr_percentile, raw=True)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}

    for i in range(200, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        month = current_time.strftime('%Y-%m')
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
                else:
                    losses += 1

                if month not in monthly_pnl:
                    monthly_pnl[month] = 0
                monthly_pnl[month] += pnl
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        if not position:
            if hour < START_HOUR or hour >= END_HOUR or hour in SKIP_HOURS:
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
                tp_mult = base_tp + TIME_TP_BONUS if 12 <= hour < 16 else base_tp
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - 10000) / 10000 * 100

    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    losing_months = sum(1 for m in monthly_pnl.values() if m < 0)
    total_months = len(monthly_pnl)

    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profitable_months': profitable_months,
        'losing_months': losing_months,
        'total_months': total_months,
        'final_balance': balance,
        'monthly_pnl': monthly_pnl
    }

def main():
    print("=" * 70)
    print("RSI v3.7 DEEP PARAMETER SWEEP")
    print("Target: 0 Losing Months")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching GBPUSD H1 data...")
        start_date = datetime(2024, 10, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df_original = get_h1_data("GBPUSD", start_date, end_date)

        if df_original is None:
            print("Failed to get data!")
            return

        print(f"Loaded {len(df_original)} bars")

        # Deep sweep parameters
        rsi_os_range = [30, 32, 34, 36, 38, 40, 42, 44]
        rsi_ob_range = [56, 58, 60, 62, 64, 66, 68, 70]
        sl_range = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2]
        tp_base_range = [2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
        atr_min_range = [15, 20, 25, 30]
        atr_max_range = [70, 75, 80, 85]

        best_results = []
        tested = 0
        total_combinations = len(rsi_os_range) * len(rsi_ob_range) * len(sl_range) * len(tp_base_range)

        print(f"\nTesting {total_combinations} base combinations...")
        print("Looking for configurations with 0-2 losing months...\n")

        for rsi_os in rsi_os_range:
            for rsi_ob in rsi_ob_range:
                # RSI spread must be reasonable
                if rsi_ob - rsi_os < 12:
                    continue

                for sl_mult in sl_range:
                    for tp_base in tp_base_range:
                        # Minimum R:R check
                        if tp_base / sl_mult < 1.3:
                            continue

                        params = {
                            'rsi_os': rsi_os, 'rsi_ob': rsi_ob,
                            'sl_mult': sl_mult,
                            'tp_low': tp_base, 'tp_med': tp_base + 0.6, 'tp_high': tp_base + 1.2,
                            'max_holding': 46,
                            'atr_min': 20, 'atr_max': 80
                        }

                        df = df_original.copy()
                        result = run_backtest(df, params)
                        tested += 1

                        # Keep top performers
                        if result['losing_months'] <= 2 and result['total_return'] > 20:
                            result['params'] = params.copy()
                            best_results.append(result)

                            if result['losing_months'] == 0:
                                print(f"*** 0-LOSS FOUND! Return: {result['total_return']:.1f}%, "
                                      f"RSI: {rsi_os}/{rsi_ob}, SL: {sl_mult}x, TP: {tp_base}x")
                            elif result['losing_months'] == 1:
                                print(f"  1-loss: Return: {result['total_return']:.1f}%, "
                                      f"RSI: {rsi_os}/{rsi_ob}, SL: {sl_mult}x, TP: {tp_base}x")

                if tested % 500 == 0:
                    print(f"  Progress: {tested} tested, {len(best_results)} good configs found")

        print(f"\nTotal tested: {tested}")
        print(f"Good configs found: {len(best_results)}")

        if best_results:
            # Sort by losing months, then by return
            best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

            print("\n" + "=" * 70)
            print("TOP 10 CONFIGURATIONS")
            print("=" * 70)

            for i, r in enumerate(best_results[:10], 1):
                p = r['params']
                print(f"\n{i}. Loss Months: {r['losing_months']} | Return: {r['total_return']:.1f}% | "
                      f"DD: {r['max_drawdown']:.1f}% | Trades: {r['total_trades']} | WR: {r['win_rate']:.1f}%")
                print(f"   RSI: {p['rsi_os']}/{p['rsi_ob']} | SL: {p['sl_mult']}x | TP: {p['tp_low']}/{p['tp_med']}/{p['tp_high']}")

            # Best one
            best = best_results[0]
            print("\n" + "=" * 70)
            print(f"BEST CONFIGURATION: {best['losing_months']} Losing Months")
            print("=" * 70)
            print(f"Total Return: {best['total_return']:.2f}%")
            print(f"Max Drawdown: {best['max_drawdown']:.2f}%")
            print(f"Trades: {best['total_trades']}")
            print(f"Win Rate: {best['win_rate']:.1f}%")

            print(f"\nParameters:")
            for k, v in best['params'].items():
                print(f"  {k}: {v}")

            print(f"\nMonthly P/L:")
            for month, pnl in sorted(best['monthly_pnl'].items()):
                marker = " <<<< LOSS" if pnl < 0 else ""
                print(f"  {month}: ${pnl:+,.2f}{marker}")

            # If 0 loss found, do ATR filter optimization
            if best['losing_months'] <= 1:
                print("\n" + "=" * 70)
                print("FINE-TUNING ATR FILTER...")
                print("=" * 70)

                best_zero = None
                base_params = best['params'].copy()

                for atr_min in atr_min_range:
                    for atr_max in atr_max_range:
                        if atr_max - atr_min < 40:
                            continue

                        params = base_params.copy()
                        params['atr_min'] = atr_min
                        params['atr_max'] = atr_max

                        df = df_original.copy()
                        result = run_backtest(df, params)

                        if result['losing_months'] == 0 and result['total_return'] > 10:
                            if best_zero is None or result['total_return'] > best_zero['total_return']:
                                best_zero = result
                                best_zero['params'] = params.copy()
                                print(f"  0-loss with ATR {atr_min}-{atr_max}: Return {result['total_return']:.1f}%")

                if best_zero:
                    print("\n" + "=" * 70)
                    print("OPTIMAL 0-LOSS CONFIGURATION")
                    print("=" * 70)
                    print(f"Total Return: {best_zero['total_return']:.2f}%")
                    print(f"Max Drawdown: {best_zero['max_drawdown']:.2f}%")
                    print(f"Trades: {best_zero['total_trades']}")
                    print(f"Win Rate: {best_zero['win_rate']:.1f}%")
                    print(f"\nOptimal Parameters:")
                    for k, v in best_zero['params'].items():
                        print(f"  {k}: {v}")
                    print(f"\nMonthly P/L (All Positive!):")
                    for month, pnl in sorted(best_zero['monthly_pnl'].items()):
                        print(f"  {month}: ${pnl:+,.2f}")

        else:
            print("\nNo good configurations found. Try wider parameter ranges.")

    finally:
        mt5.shutdown()
        print("\nMT5 disconnected")

if __name__ == "__main__":
    main()
