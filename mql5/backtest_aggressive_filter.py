"""
RSI v3.7 Aggressive Auto-Detection Filter
==========================================
More aggressive filtering with adjusted base parameters.
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

    # Filters
    TREND_SMA = params.get('trend_sma', 50)
    TREND_STRENGTH = params.get('trend_strength', 0)
    ADX_THRESH = params.get('adx_thresh', 0)
    MOM_PERIOD = params.get('mom_period', 20)
    MOM_THRESH = params.get('mom_thresh', 0)
    ONLY_WITH_TREND = params.get('only_with_trend', False)
    MIN_RSI_EXTREME = params.get('min_rsi_extreme', 0)  # Extra RSI margin required

    # RSI
    rsi_period = 10
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()

    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0

    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # Trend indicators
    df['sma'] = df['close'].rolling(TREND_SMA).mean()
    df['sma_slope'] = (df['sma'] - df['sma'].shift(TREND_SMA)) / df['sma'].shift(TREND_SMA) * 100
    df['momentum'] = (df['close'] - df['close'].shift(MOM_PERIOD)) / df['close'].shift(MOM_PERIOD) * 100

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr_adx = np.maximum(df['high'] - df['low'],
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                  abs(df['low'] - df['close'].shift(1))))
    atr_adx = tr_adx.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_adx)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(14).mean()

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
                else:
                    losses += 1

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
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            rsi = row['rsi']

            # Apply RSI extreme margin
            effective_os = RSI_OS - MIN_RSI_EXTREME
            effective_ob = RSI_OB + MIN_RSI_EXTREME

            signal = 1 if rsi < effective_os else (-1 if rsi > effective_ob else 0)

            if not signal:
                continue

            skip = False

            # Trend Strength Filter
            if TREND_STRENGTH > 0:
                slope = row['sma_slope']
                if signal == 1 and slope < -TREND_STRENGTH:
                    skip = True
                elif signal == -1 and slope > TREND_STRENGTH:
                    skip = True

            # ADX Filter
            if ADX_THRESH > 0 and not skip:
                if row['adx'] > ADX_THRESH:
                    skip = True

            # Momentum Filter
            if MOM_THRESH > 0 and not skip:
                mom = row['momentum']
                if signal == 1 and mom < -MOM_THRESH:
                    skip = True
                elif signal == -1 and mom > MOM_THRESH:
                    skip = True

            # Only With Trend
            if ONLY_WITH_TREND and not skip:
                if signal == 1 and row['close'] < row['sma']:
                    skip = True
                elif signal == -1 and row['close'] > row['sma']:
                    skip = True

            if skip:
                continue

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
    log("=" * 70)
    log("RSI v3.7 AGGRESSIVE FILTER SWEEP")
    log("=" * 70)

    if not connect_mt5():
        log("MT5 failed")
        return

    try:
        log("Fetching data...")
        start_date = datetime(2024, 10, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df_original = get_h1_data("GBPUSD", start_date, end_date)

        if df_original is None:
            return

        log(f"Loaded {len(df_original)} bars\n")

        best_results = []
        tested = 0

        # Extensive grid search
        rsi_os_vals = [38, 40, 42]
        rsi_ob_vals = [58, 60, 62]
        sl_vals = [1.5, 1.7, 2.0]
        tp_vals = [2.4, 2.8, 3.2]
        trend_str_vals = [0, 1.5, 2.0, 2.5, 3.0]
        adx_vals = [0, 25, 30, 35]
        mom_vals = [0, 1.5, 2.0, 2.5]
        rsi_extreme_vals = [0, 3, 5]

        log("Testing combinations...")

        for rsi_os in rsi_os_vals:
            for rsi_ob in rsi_ob_vals:
                for sl in sl_vals:
                    for tp in tp_vals:
                        for trend_str in trend_str_vals:
                            for adx in adx_vals:
                                for mom in mom_vals:
                                    for rsi_ext in rsi_extreme_vals:
                                        # Skip if no filter at all
                                        if trend_str == 0 and adx == 0 and mom == 0 and rsi_ext == 0:
                                            continue

                                        params = {
                                            'rsi_os': rsi_os, 'rsi_ob': rsi_ob,
                                            'sl_mult': sl,
                                            'tp_low': tp, 'tp_med': tp + 0.6, 'tp_high': tp + 1.2,
                                            'max_holding': 46,
                                            'atr_min': 20, 'atr_max': 80,
                                            'trend_strength': trend_str,
                                            'adx_thresh': adx,
                                            'mom_thresh': mom,
                                            'mom_period': 20,
                                            'trend_sma': 50,
                                            'min_rsi_extreme': rsi_ext
                                        }

                                        df = df_original.copy()
                                        result = run_backtest(df, params)
                                        tested += 1

                                        if result['losing_months'] == 0 and result['total_return'] > 20:
                                            result['params'] = params.copy()
                                            best_results.append(result)
                                            log(f"*** 0-LOSS! Ret:{result['total_return']:.1f}% RSI:{rsi_os}/{rsi_ob} SL:{sl} TP:{tp} "
                                                f"Trend:{trend_str} ADX:{adx} Mom:{mom} RSIext:{rsi_ext}")

                                        elif result['losing_months'] == 1 and result['total_return'] > 40:
                                            result['params'] = params.copy()
                                            best_results.append(result)

                                        if tested % 5000 == 0:
                                            log(f"  Progress: {tested} tested, {len([r for r in best_results if r['losing_months'] == 0])} zero-loss found")

        log(f"\nTotal tested: {tested}")
        log(f"Zero-loss configs: {len([r for r in best_results if r['losing_months'] == 0])}")
        log(f"One-loss configs: {len([r for r in best_results if r['losing_months'] == 1])}")

        if best_results:
            best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

            log("\n" + "=" * 70)
            log("TOP 10 CONFIGURATIONS")
            log("=" * 70)

            for i, r in enumerate(best_results[:10], 1):
                p = r['params']
                log(f"\n{i}. Loss:{r['losing_months']} | Ret:{r['total_return']:.1f}% | DD:{r['max_drawdown']:.1f}% | Trades:{r['total_trades']}")
                log(f"   RSI:{p['rsi_os']}/{p['rsi_ob']} SL:{p['sl_mult']} TP:{p['tp_low']}")
                log(f"   Trend:{p['trend_strength']} ADX:{p['adx_thresh']} Mom:{p['mom_thresh']} RSIext:{p['min_rsi_extreme']}")

            # Best zero loss
            zero_loss = [r for r in best_results if r['losing_months'] == 0]
            if zero_loss:
                best = max(zero_loss, key=lambda x: x['total_return'])
                log("\n" + "=" * 70)
                log("BEST ZERO-LOSS CONFIGURATION")
                log("=" * 70)
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
