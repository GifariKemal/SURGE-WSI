"""
RSI v3.7 Parameter Tuning Sweep
===============================
Test multiple parameter combinations to find optimal settings.
Target: Minimize losing months while maintaining profitability.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# MT5 Connection
MT5_LOGIN = 61045904
MT5_PASSWORD = "iy#K5L7sF"
MT5_SERVER = "FinexBisnisSolusi-Demo"
MT5_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

def connect_mt5():
    """Connect to MT5"""
    if not mt5.initialize(path=MT5_PATH):
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        return False
    print(f"Connected to MT5: {mt5.account_info().server}")
    return True

def get_h1_data(symbol, start_date, end_date):
    """Get H1 OHLCV data from MT5"""
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def run_backtest(df, params):
    """Run backtest with given parameters"""
    # Unpack parameters
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

    # RSI(10) using SMA method
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

    # ATR Percentile
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

    # Backtest
    balance = 10000.0
    initial_balance = 10000.0
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
            if hour < 7 or hour >= 22 or hour in SKIP_HOURS:
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

    # Calculate results
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - initial_balance) / initial_balance * 100

    # Profit factor
    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    losing_months = sum(1 for m in monthly_pnl.values() if m < 0)
    total_months = len(monthly_pnl)

    # Calculate gross profit/loss for profit factor
    gross_profit = sum(v for v in monthly_pnl.values() if v > 0)
    gross_loss = abs(sum(v for v in monthly_pnl.values() if v < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'profitable_months': profitable_months,
        'losing_months': losing_months,
        'total_months': total_months,
        'final_balance': balance,
        'monthly_pnl': monthly_pnl
    }

def main():
    print("=" * 70)
    print("RSI v3.7 PARAMETER TUNING SWEEP")
    print("Target: 0 Losing Months")
    print("=" * 70)

    if not connect_mt5():
        return

    try:
        # Get data
        print("\nFetching GBPUSD H1 data...")
        start_date = datetime(2024, 10, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data("GBPUSD", start_date, end_date)

        if df is None:
            print("Failed to get data!")
            return

        print(f"Loaded {len(df)} H1 bars")

        # Define parameter sets to test
        param_sets = {
            'v3.7 Current': {
                'rsi_os': 42, 'rsi_ob': 58,
                'sl_mult': 1.5,
                'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
                'max_holding': 46,
                'atr_min': 20, 'atr_max': 80
            },
            'Opsi A (Konservatif)': {
                'rsi_os': 38, 'rsi_ob': 62,
                'sl_mult': 1.5,
                'tp_low': 2.8, 'tp_med': 3.4, 'tp_high': 4.0,
                'max_holding': 36,
                'atr_min': 25, 'atr_max': 75
            },
            'Opsi B (Wider SL)': {
                'rsi_os': 42, 'rsi_ob': 58,
                'sl_mult': 2.0,
                'tp_low': 3.0, 'tp_med': 3.6, 'tp_high': 4.2,
                'max_holding': 48,
                'atr_min': 20, 'atr_max': 80
            },
            'Opsi C (Balanced)': {
                'rsi_os': 40, 'rsi_ob': 60,
                'sl_mult': 1.7,
                'tp_low': 2.6, 'tp_med': 3.2, 'tp_high': 3.8,
                'max_holding': 42,
                'atr_min': 22, 'atr_max': 78
            },
            'Opsi D (Ultra Konservatif)': {
                'rsi_os': 35, 'rsi_ob': 65,
                'sl_mult': 1.8,
                'tp_low': 3.0, 'tp_med': 3.6, 'tp_high': 4.2,
                'max_holding': 40,
                'atr_min': 25, 'atr_max': 70
            },
            'Opsi E (Aggressive TP)': {
                'rsi_os': 40, 'rsi_ob': 60,
                'sl_mult': 1.6,
                'tp_low': 3.2, 'tp_med': 4.0, 'tp_high': 4.8,
                'max_holding': 50,
                'atr_min': 20, 'atr_max': 80
            },
            'Opsi F (Tight Filter)': {
                'rsi_os': 38, 'rsi_ob': 62,
                'sl_mult': 1.7,
                'tp_low': 2.8, 'tp_med': 3.4, 'tp_high': 4.0,
                'max_holding': 44,
                'atr_min': 30, 'atr_max': 70
            },
        }

        results = []

        print("\n" + "=" * 70)
        print("RUNNING BACKTESTS...")
        print("=" * 70)

        for name, params in param_sets.items():
            print(f"\nTesting: {name}...")
            df_copy = df.copy()
            result = run_backtest(df_copy, params)
            result['name'] = name
            result['params'] = params
            results.append(result)

            print(f"  Return: {result['total_return']:.1f}% | DD: {result['max_drawdown']:.1f}% | "
                  f"Trades: {result['total_trades']} | WR: {result['win_rate']:.1f}% | "
                  f"Months: {result['profitable_months']}/{result['total_months']} (+) {result['losing_months']} (-)")

        # Sort by losing months (ascending), then by return (descending)
        results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        print("\n" + "=" * 70)
        print("RESULTS COMPARISON (Sorted by Fewest Losing Months)")
        print("=" * 70)

        print(f"\n{'Rank':<5}{'Name':<25}{'Return':<10}{'MaxDD':<8}{'Trades':<8}{'WinRate':<8}{'PF':<6}{'Win Mo':<8}{'Loss Mo':<8}")
        print("-" * 86)

        for i, r in enumerate(results, 1):
            print(f"{i:<5}{r['name']:<25}{r['total_return']:>7.1f}%  {r['max_drawdown']:>5.1f}%  "
                  f"{r['total_trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>4.2f}  "
                  f"{r['profitable_months']:>6}  {r['losing_months']:>6}")

        # Best result
        best = results[0]
        print("\n" + "=" * 70)
        print(f"BEST CONFIGURATION: {best['name']}")
        print("=" * 70)
        print(f"Losing Months: {best['losing_months']}")
        print(f"Total Return: {best['total_return']:.2f}%")
        print(f"Max Drawdown: {best['max_drawdown']:.2f}%")
        print(f"Profit Factor: {best['profit_factor']:.2f}")
        print(f"\nParameters:")
        for k, v in best['params'].items():
            print(f"  {k}: {v}")

        # Show monthly breakdown for best
        print(f"\nMonthly P/L for {best['name']}:")
        print("-" * 40)
        for month, pnl in sorted(best['monthly_pnl'].items()):
            status = "+" if pnl > 0 else ""
            marker = " <<<< LOSS" if pnl < 0 else ""
            print(f"  {month}: ${status}{pnl:,.2f}{marker}")

        # If still has losing months, run extended sweep
        if best['losing_months'] > 0:
            print("\n" + "=" * 70)
            print("EXTENDED SWEEP - Searching for 0 losing months...")
            print("=" * 70)

            # Extended parameter grid
            rsi_os_range = [35, 36, 37, 38]
            rsi_ob_range = [62, 63, 64, 65]
            sl_range = [1.7, 1.8, 1.9, 2.0]
            tp_base_range = [3.0, 3.2, 3.4]

            best_zero_loss = None
            tested = 0

            for rsi_os in rsi_os_range:
                for rsi_ob in rsi_ob_range:
                    for sl_mult in sl_range:
                        for tp_base in tp_base_range:
                            params = {
                                'rsi_os': rsi_os, 'rsi_ob': rsi_ob,
                                'sl_mult': sl_mult,
                                'tp_low': tp_base, 'tp_med': tp_base + 0.6, 'tp_high': tp_base + 1.2,
                                'max_holding': 42,
                                'atr_min': 25, 'atr_max': 70
                            }

                            df_copy = df.copy()
                            result = run_backtest(df_copy, params)
                            tested += 1

                            if result['losing_months'] == 0:
                                if best_zero_loss is None or result['total_return'] > best_zero_loss['total_return']:
                                    best_zero_loss = result
                                    best_zero_loss['params'] = params
                                    print(f"  Found 0-loss config! Return: {result['total_return']:.1f}%, "
                                          f"RSI: {rsi_os}/{rsi_ob}, SL: {sl_mult}, TP: {tp_base}")

            print(f"\nTested {tested} combinations")

            if best_zero_loss:
                print("\n" + "=" * 70)
                print("ZERO LOSS CONFIGURATION FOUND!")
                print("=" * 70)
                print(f"Total Return: {best_zero_loss['total_return']:.2f}%")
                print(f"Max Drawdown: {best_zero_loss['max_drawdown']:.2f}%")
                print(f"Trades: {best_zero_loss['total_trades']}")
                print(f"Win Rate: {best_zero_loss['win_rate']:.1f}%")
                print(f"Profitable Months: {best_zero_loss['profitable_months']}/{best_zero_loss['total_months']}")
                print(f"\nOptimal Parameters:")
                for k, v in best_zero_loss['params'].items():
                    print(f"  {k}: {v}")

                print(f"\nMonthly P/L:")
                for month, pnl in sorted(best_zero_loss['monthly_pnl'].items()):
                    print(f"  {month}: ${pnl:+,.2f}")
            else:
                print("\nNo 0-loss configuration found in this range.")
                print("Minimum losing months achieved:", best['losing_months'])

    finally:
        mt5.shutdown()
        print("\nMT5 disconnected")

if __name__ == "__main__":
    main()
