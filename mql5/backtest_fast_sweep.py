"""
RSI v3.7 Fast Parameter Sweep
=============================
Quick search for 0 losing months configuration.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys

# Force flush output
def log(msg):
    print(msg)
    sys.stdout.flush()

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

    # RSI(10) using SMA
    rsi_period = 10
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = np.where(loss == 0, 100, gain / loss)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

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
            if hour < 7 or hour >= 22 or hour == 12:
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
                tp_mult = base_tp + 0.35 if 12 <= hour < 16 else base_tp
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

    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profitable_months': profitable_months,
        'losing_months': losing_months,
        'total_months': len(monthly_pnl),
        'monthly_pnl': monthly_pnl
    }

def main():
    log("=" * 60)
    log("RSI v3.7 FAST PARAMETER SWEEP")
    log("=" * 60)

    if not connect_mt5():
        log("MT5 connection failed")
        return

    try:
        log("Fetching data...")
        start_date = datetime(2024, 10, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df_original = get_h1_data("GBPUSD", start_date, end_date)

        if df_original is None:
            log("Failed to get data!")
            return

        log(f"Loaded {len(df_original)} bars")

        # Key parameter ranges
        rsi_os_values = [30, 34, 38, 42, 44, 46]
        rsi_ob_values = [54, 56, 58, 62, 66, 70]
        sl_values = [1.2, 1.5, 1.8, 2.0, 2.2]
        tp_values = [2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
        atr_min_values = [15, 20, 25, 30]
        atr_max_values = [70, 75, 80, 85]

        best_results = []
        tested = 0

        log("\nSearching for 0-loss configuration...")

        for rsi_os in rsi_os_values:
            for rsi_ob in rsi_ob_values:
                if rsi_ob - rsi_os < 10:
                    continue

                for sl in sl_values:
                    for tp in tp_values:
                        if tp / sl < 1.2:
                            continue

                        for atr_min in atr_min_values:
                            for atr_max in atr_max_values:
                                if atr_max - atr_min < 40:
                                    continue

                                params = {
                                    'rsi_os': rsi_os, 'rsi_ob': rsi_ob,
                                    'sl_mult': sl,
                                    'tp_low': tp, 'tp_med': tp + 0.6, 'tp_high': tp + 1.2,
                                    'max_holding': 46,
                                    'atr_min': atr_min, 'atr_max': atr_max
                                }

                                df = df_original.copy()
                                result = run_backtest(df, params)
                                tested += 1

                                if result['losing_months'] == 0 and result['total_return'] > 10:
                                    result['params'] = params.copy()
                                    best_results.append(result)
                                    log(f"*** 0-LOSS! Ret:{result['total_return']:.1f}% RSI:{rsi_os}/{rsi_ob} SL:{sl} TP:{tp} ATR:{atr_min}-{atr_max}")

                                elif result['losing_months'] == 1 and result['total_return'] > 30:
                                    result['params'] = params.copy()
                                    best_results.append(result)

                        if tested % 1000 == 0:
                            log(f"  Tested: {tested}, Found: {len(best_results)}")

        log(f"\nTotal tested: {tested}")
        log(f"Good configs: {len(best_results)}")

        if best_results:
            best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

            log("\n" + "=" * 60)
            log("TOP CONFIGURATIONS")
            log("=" * 60)

            for i, r in enumerate(best_results[:5], 1):
                p = r['params']
                log(f"\n{i}. Loss:{r['losing_months']} | Ret:{r['total_return']:.1f}% | DD:{r['max_drawdown']:.1f}%")
                log(f"   RSI:{p['rsi_os']}/{p['rsi_ob']} SL:{p['sl_mult']} TP:{p['tp_low']} ATR:{p['atr_min']}-{p['atr_max']}")
                log(f"   Monthly: {dict(sorted(r['monthly_pnl'].items()))}")

            best = best_results[0]
            if best['losing_months'] == 0:
                log("\n" + "=" * 60)
                log("ZERO LOSS CONFIGURATION FOUND!")
                log("=" * 60)
                log(f"Return: {best['total_return']:.2f}%")
                log(f"Max DD: {best['max_drawdown']:.2f}%")
                log(f"Trades: {best['total_trades']}")
                log(f"WinRate: {best['win_rate']:.1f}%")
                log("\nParameters:")
                for k, v in best['params'].items():
                    log(f"  {k}: {v}")
                log("\nMonthly P/L:")
                for m, p in sorted(best['monthly_pnl'].items()):
                    log(f"  {m}: ${p:+,.2f}")

    finally:
        mt5.shutdown()
        log("\nDone!")

if __name__ == "__main__":
    main()
