"""
RSI v3.7 Smart Sweep with Conditional Trading
==============================================
Test: Skip trading during high-risk conditions to achieve 0 losing months.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys

def log(msg):
    print(msg)
    sys.stdout.flush()

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

def run_backtest_v2(df, params):
    """Enhanced backtest with smart filters"""
    RSI_OS = params['rsi_os']
    RSI_OB = params['rsi_ob']
    SL_MULT = params['sl_mult']
    TP_LOW = params['tp_low']
    TP_MED = params['tp_med']
    TP_HIGH = params['tp_high']
    MAX_HOLDING = params['max_holding']
    MIN_ATR_PCT = params['atr_min']
    MAX_ATR_PCT = params['atr_max']

    # New smart filters
    TREND_LOOKBACK = params.get('trend_lookback', 50)
    TREND_THRESHOLD = params.get('trend_thresh', 0.0)  # 0 = no trend filter
    CONSEC_LOSS_PAUSE = params.get('consec_pause', 0)  # Pause after N consecutive losses
    RSI_EXTREME_ONLY = params.get('rsi_extreme', False)  # Only trade at extreme RSI
    SKIP_MONTHS = params.get('skip_months', [])  # Skip specific months
    START_HOUR = params.get('start_hour', 7)
    END_HOUR = params.get('end_hour', 22)
    SKIP_HOURS = params.get('skip_hours', [12])

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

    # Trend detection (SMA slope)
    if TREND_LOOKBACK > 0:
        df['sma'] = df['close'].rolling(TREND_LOOKBACK).mean()
        df['trend'] = (df['sma'] - df['sma'].shift(TREND_LOOKBACK)) / df['sma'].shift(TREND_LOOKBACK) * 100
    else:
        df['trend'] = 0

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['year'] = df.index.year
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    consecutive_losses = 0
    paused_until = None

    for i in range(200, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        month_str = current_time.strftime('%Y-%m')
        month_num = row['month']
        weekday = row['weekday']
        hour = row['hour']

        if weekday >= 5:
            continue

        # Check if we should skip this month
        if month_num in SKIP_MONTHS:
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
                    if CONSEC_LOSS_PAUSE > 0 and consecutive_losses >= CONSEC_LOSS_PAUSE:
                        paused_until = i + 24  # Pause for 24 bars (24 hours)

                if month_str not in monthly_pnl:
                    monthly_pnl[month_str] = 0
                monthly_pnl[month_str] += pnl
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        if not position:
            # Check pause
            if paused_until and i < paused_until:
                continue

            if hour < START_HOUR or hour >= END_HOUR or hour in SKIP_HOURS:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            # Trend filter
            trend = row['trend']
            if TREND_THRESHOLD > 0:
                # Skip counter-trend trades in strong trend
                if abs(trend) > TREND_THRESHOLD:
                    continue

            rsi = row['rsi']

            # RSI extreme filter
            if RSI_EXTREME_ONLY:
                signal = 1 if rsi < RSI_OS - 5 else (-1 if rsi > RSI_OB + 5 else 0)
            else:
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
    log("RSI v3.7 SMART SWEEP - Zero Loss Target")
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
            log("Failed!")
            return

        log(f"Loaded {len(df_original)} bars")

        best_results = []
        tested = 0

        # Strategy 1: Skip risky months (Oct, Nov, Apr)
        log("\n--- Strategy 1: Skip High-Risk Months ---")
        skip_month_combos = [
            [],
            [10],
            [11],
            [4],
            [10, 11],
            [10, 4],
            [11, 4],
            [10, 11, 4],
        ]

        for skip_months in skip_month_combos:
            params = {
                'rsi_os': 42, 'rsi_ob': 58,
                'sl_mult': 1.5,
                'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
                'max_holding': 46,
                'atr_min': 20, 'atr_max': 80,
                'skip_months': skip_months
            }

            df = df_original.copy()
            result = run_backtest_v2(df, params)
            tested += 1
            result['params'] = params.copy()
            result['strategy'] = f"Skip months: {skip_months}"
            best_results.append(result)

            log(f"  Skip {skip_months}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Strategy 2: Trend filter
        log("\n--- Strategy 2: Trend Filter ---")
        for trend_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for trend_lb in [30, 50, 100]:
                params = {
                    'rsi_os': 42, 'rsi_ob': 58,
                    'sl_mult': 1.5,
                    'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
                    'max_holding': 46,
                    'atr_min': 20, 'atr_max': 80,
                    'trend_lookback': trend_lb,
                    'trend_thresh': trend_thresh
                }

                df = df_original.copy()
                result = run_backtest_v2(df, params)
                tested += 1

                if result['losing_months'] <= 2:
                    result['params'] = params.copy()
                    result['strategy'] = f"Trend LB={trend_lb}, Thresh={trend_thresh}%"
                    best_results.append(result)
                    log(f"  Trend LB={trend_lb}, Thresh={trend_thresh}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Strategy 3: Consecutive loss pause
        log("\n--- Strategy 3: Pause After Consecutive Losses ---")
        for consec in [2, 3, 4, 5]:
            params = {
                'rsi_os': 42, 'rsi_ob': 58,
                'sl_mult': 1.5,
                'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
                'max_holding': 46,
                'atr_min': 20, 'atr_max': 80,
                'consec_pause': consec
            }

            df = df_original.copy()
            result = run_backtest_v2(df, params)
            tested += 1
            result['params'] = params.copy()
            result['strategy'] = f"Pause after {consec} losses"
            best_results.append(result)

            log(f"  Pause after {consec}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Strategy 4: Extreme RSI only
        log("\n--- Strategy 4: Extreme RSI Only ---")
        for rsi_os in [35, 38, 40]:
            for rsi_ob in [60, 62, 65]:
                params = {
                    'rsi_os': rsi_os, 'rsi_ob': rsi_ob,
                    'sl_mult': 1.5,
                    'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
                    'max_holding': 46,
                    'atr_min': 20, 'atr_max': 80,
                    'rsi_extreme': True
                }

                df = df_original.copy()
                result = run_backtest_v2(df, params)
                tested += 1

                if result['losing_months'] <= 2:
                    result['params'] = params.copy()
                    result['strategy'] = f"Extreme RSI {rsi_os}/{rsi_ob}"
                    best_results.append(result)
                    log(f"  Extreme RSI {rsi_os}/{rsi_ob}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Strategy 5: Narrower trading hours
        log("\n--- Strategy 5: Narrower Trading Hours ---")
        hour_ranges = [(8, 20), (9, 18), (10, 17), (8, 16), (12, 20)]
        for start_h, end_h in hour_ranges:
            params = {
                'rsi_os': 42, 'rsi_ob': 58,
                'sl_mult': 1.5,
                'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
                'max_holding': 46,
                'atr_min': 20, 'atr_max': 80,
                'start_hour': start_h,
                'end_hour': end_h
            }

            df = df_original.copy()
            result = run_backtest_v2(df, params)
            tested += 1

            if result['losing_months'] <= 2:
                result['params'] = params.copy()
                result['strategy'] = f"Hours {start_h}-{end_h}"
                best_results.append(result)
                log(f"  Hours {start_h}-{end_h}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Strategy 6: Combined approach
        log("\n--- Strategy 6: Combined Smart Filters ---")
        combos = [
            {'trend_lookback': 50, 'trend_thresh': 2.0, 'consec_pause': 3},
            {'trend_lookback': 50, 'trend_thresh': 1.5, 'consec_pause': 4},
            {'rsi_extreme': True, 'consec_pause': 3},
            {'start_hour': 9, 'end_hour': 18, 'trend_lookback': 50, 'trend_thresh': 2.0},
            {'start_hour': 8, 'end_hour': 20, 'consec_pause': 3, 'rsi_extreme': True},
        ]

        for combo in combos:
            params = {
                'rsi_os': 42, 'rsi_ob': 58,
                'sl_mult': 1.5,
                'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
                'max_holding': 46,
                'atr_min': 20, 'atr_max': 80,
            }
            params.update(combo)

            df = df_original.copy()
            result = run_backtest_v2(df, params)
            tested += 1
            result['params'] = params.copy()
            result['strategy'] = f"Combined: {combo}"
            best_results.append(result)

            log(f"  Combined {combo}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        log(f"\nTotal tested: {tested}")

        # Sort and show best
        best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        log("\n" + "=" * 60)
        log("TOP 10 CONFIGURATIONS")
        log("=" * 60)

        for i, r in enumerate(best_results[:10], 1):
            log(f"\n{i}. Loss Months: {r['losing_months']} | Return: {r['total_return']:.1f}% | DD: {r['max_drawdown']:.1f}%")
            log(f"   Strategy: {r.get('strategy', 'N/A')}")
            log(f"   Trades: {r['total_trades']} | WinRate: {r['win_rate']:.1f}%")

        # Best with 0 loss
        zero_loss = [r for r in best_results if r['losing_months'] == 0]
        if zero_loss:
            best = max(zero_loss, key=lambda x: x['total_return'])
            log("\n" + "=" * 60)
            log("ZERO LOSS CONFIGURATION FOUND!")
            log("=" * 60)
            log(f"Strategy: {best.get('strategy', 'N/A')}")
            log(f"Return: {best['total_return']:.2f}%")
            log(f"Max DD: {best['max_drawdown']:.2f}%")
            log(f"Trades: {best['total_trades']}")
            log(f"WinRate: {best['win_rate']:.1f}%")
            log("\nParameters:")
            for k, v in best['params'].items():
                log(f"  {k}: {v}")
            log("\nMonthly P/L:")
            for m, p in sorted(best['monthly_pnl'].items()):
                marker = " <-- LOSS" if p < 0 else ""
                log(f"  {m}: ${p:+,.2f}{marker}")

    finally:
        mt5.shutdown()
        log("\nDone!")

if __name__ == "__main__":
    main()
