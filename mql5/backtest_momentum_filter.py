"""
RSI v3.7 + Momentum-Based Auto Filter
=====================================
Skip counter-trend trades when momentum strongly against signal.
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

    # Momentum Filter params
    MOM_PERIOD = params.get('mom_period', 10)
    MOM_THRESHOLD = params.get('mom_threshold', 0)  # Skip if momentum against signal > this %
    SMA_PERIOD = params.get('sma_period', 20)
    SMA_FILTER = params.get('sma_filter', False)  # Only trade with SMA direction
    CONSEC_BARS = params.get('consec_bars', 0)  # Min consecutive bars in same direction
    THURSDAY_MULT = params.get('thursday_mult', 1.0)

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

    # Momentum (% change over N bars)
    df['momentum'] = (df['close'] - df['close'].shift(MOM_PERIOD)) / df['close'].shift(MOM_PERIOD) * 100

    # SMA for trend
    df['sma'] = df['close'].rolling(SMA_PERIOD).mean()

    # Count consecutive up/down bars
    df['bar_dir'] = np.sign(df['close'] - df['close'].shift(1))
    df['consec'] = df['bar_dir'].groupby((df['bar_dir'] != df['bar_dir'].shift()).cumsum()).cumcount() + 1

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    skipped = 0

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
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if not signal:
                continue

            # ========== AUTO FILTERS ==========
            skip_trade = False
            mom = row['momentum']

            # Momentum filter: Skip if momentum strongly against signal
            if MOM_THRESHOLD > 0:
                if signal == 1 and mom < -MOM_THRESHOLD:  # Buy when strong down momentum
                    skip_trade = True
                elif signal == -1 and mom > MOM_THRESHOLD:  # Sell when strong up momentum
                    skip_trade = True

            # SMA filter: Only trade with trend
            if SMA_FILTER and not skip_trade:
                if signal == 1 and row['close'] < row['sma']:  # Buy below SMA
                    skip_trade = True
                elif signal == -1 and row['close'] > row['sma']:  # Sell above SMA
                    skip_trade = True

            # Consecutive bars filter
            if CONSEC_BARS > 0 and not skip_trade:
                bar_dir = row['bar_dir']
                consec = row['consec']
                # Skip if recent bars strongly against signal
                if signal == 1 and bar_dir == -1 and consec >= CONSEC_BARS:
                    skip_trade = True
                elif signal == -1 and bar_dir == 1 and consec >= CONSEC_BARS:
                    skip_trade = True

            if skip_trade:
                skipped += 1
                continue

            # ================================

            entry = row['close']
            atr = row['atr'] if row['atr'] > 0 else entry * 0.002
            base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
            tp_mult = base_tp + 0.35 if 12 <= hour < 16 else base_tp
            sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
            tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
            risk = balance * 0.01

            # Thursday multiplier
            if weekday == 3:
                risk *= THURSDAY_MULT

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
        'monthly_pnl': monthly_pnl,
        'skipped': skipped
    }

def main():
    log("=" * 70)
    log("RSI v3.7 + Momentum-Based Auto Filter")
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

        log(f"Loaded {len(df_original)} bars")

        # Check momentum stats
        df_debug = df_original.copy()
        for period in [5, 10, 15, 20]:
            df_debug[f'mom_{period}'] = (df_debug['close'] - df_debug['close'].shift(period)) / df_debug['close'].shift(period) * 100
            log(f"  Momentum({period}): min={df_debug[f'mom_{period}'].min():.2f}%, max={df_debug[f'mom_{period}'].max():.2f}%")

        base_params = {
            'rsi_os': 42, 'rsi_ob': 58,
            'sl_mult': 1.5,
            'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
            'max_holding': 46,
            'atr_min': 20, 'atr_max': 80
        }

        best_results = []

        configs = [
            {'name': 'Original', 'mom_threshold': 0, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 1.0},
            # Momentum filters
            {'name': 'Mom>0.3%', 'mom_period': 10, 'mom_threshold': 0.3, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 1.0},
            {'name': 'Mom>0.5%', 'mom_period': 10, 'mom_threshold': 0.5, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 1.0},
            {'name': 'Mom>0.7%', 'mom_period': 10, 'mom_threshold': 0.7, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 1.0},
            {'name': 'Mom>1.0%', 'mom_period': 10, 'mom_threshold': 1.0, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 1.0},
            {'name': 'Mom5>0.5%', 'mom_period': 5, 'mom_threshold': 0.5, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 1.0},
            {'name': 'Mom20>0.5%', 'mom_period': 20, 'mom_threshold': 0.5, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 1.0},
            # SMA filter
            {'name': 'SMA20 Filter', 'mom_threshold': 0, 'sma_period': 20, 'sma_filter': True, 'consec_bars': 0, 'thursday_mult': 1.0},
            {'name': 'SMA50 Filter', 'mom_threshold': 0, 'sma_period': 50, 'sma_filter': True, 'consec_bars': 0, 'thursday_mult': 1.0},
            # Consecutive bars
            {'name': 'Consec>=3', 'mom_threshold': 0, 'sma_filter': False, 'consec_bars': 3, 'thursday_mult': 1.0},
            {'name': 'Consec>=4', 'mom_threshold': 0, 'sma_filter': False, 'consec_bars': 4, 'thursday_mult': 1.0},
            {'name': 'Consec>=5', 'mom_threshold': 0, 'sma_filter': False, 'consec_bars': 5, 'thursday_mult': 1.0},
            # Combined
            {'name': 'Mom0.5+SMA20', 'mom_period': 10, 'mom_threshold': 0.5, 'sma_period': 20, 'sma_filter': True, 'consec_bars': 0, 'thursday_mult': 1.0},
            {'name': 'Mom0.5+Consec3', 'mom_period': 10, 'mom_threshold': 0.5, 'sma_filter': False, 'consec_bars': 3, 'thursday_mult': 1.0},
            {'name': 'SMA20+Consec3', 'mom_threshold': 0, 'sma_period': 20, 'sma_filter': True, 'consec_bars': 3, 'thursday_mult': 1.0},
            {'name': 'All: Mom0.5+SMA20+C3', 'mom_period': 10, 'mom_threshold': 0.5, 'sma_period': 20, 'sma_filter': True, 'consec_bars': 3, 'thursday_mult': 1.0},
            # Thursday reduction
            {'name': 'Thu50%', 'mom_threshold': 0, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 0.5},
            {'name': 'Mom0.5+Thu50%', 'mom_period': 10, 'mom_threshold': 0.5, 'sma_filter': False, 'consec_bars': 0, 'thursday_mult': 0.5},
        ]

        log("\n" + "-" * 70)
        log("Testing Momentum Filter Configurations...")
        log("-" * 70)

        for cfg in configs:
            params = base_params.copy()
            params.update(cfg)

            df = df_original.copy()
            result = run_backtest(df, params)
            result['name'] = cfg['name']
            result['config'] = cfg
            best_results.append(result)

            status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
            log(f"{cfg['name']:<25} | {status:<7} | Ret:{result['total_return']:>6.1f}% | "
                f"WR:{result['win_rate']:>5.1f}% | Trades:{result['total_trades']:>3} | Skip:{result['skipped']:>3}")

        # Sort
        best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        log("\n" + "=" * 70)
        log("TOP 5 CONFIGURATIONS")
        log("=" * 70)

        for i, r in enumerate(best_results[:5], 1):
            log(f"\n{i}. {r['name']}")
            log(f"   Loss: {r['losing_months']} | Return: {r['total_return']:.1f}% | DD: {r['max_drawdown']:.1f}% | WR: {r['win_rate']:.1f}%")

        # Best config monthly
        best = best_results[0]
        log("\n" + "=" * 70)
        log(f"BEST: {best['name']}")
        log("=" * 70)
        log("Monthly P/L:")
        for m, p in sorted(best['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            log(f"  {m}: ${p:+,.2f}{marker}")

        # Extended search if still has losses
        if best['losing_months'] > 0:
            log("\n" + "=" * 70)
            log("EXTENDED GRID SEARCH")
            log("=" * 70)

            for mom_p in [5, 10, 15]:
                for mom_t in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    for sma_p in [20, 30]:
                        for consec in [0, 3, 4]:
                            params = base_params.copy()
                            params['mom_period'] = mom_p
                            params['mom_threshold'] = mom_t
                            params['sma_period'] = sma_p
                            params['sma_filter'] = True
                            params['consec_bars'] = consec
                            params['thursday_mult'] = 0.5

                            df = df_original.copy()
                            result = run_backtest(df, params)

                            if result['losing_months'] == 0 and result['total_return'] > 20:
                                log(f"0-LOSS! MomP{mom_p}T{mom_t}+SMA{sma_p}+C{consec}+Thu0.5 | Ret:{result['total_return']:.1f}% Trades:{result['total_trades']}")
                                result['config'] = params
                                best_results.append(result)

    finally:
        mt5.shutdown()
        log("\nDone!")

if __name__ == "__main__":
    main()
