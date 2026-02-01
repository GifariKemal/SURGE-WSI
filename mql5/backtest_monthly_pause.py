"""
RSI v3.7 + Monthly Performance Auto-Pause
==========================================
Auto-pause trading when current month is losing beyond threshold.
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

    # Monthly auto-pause params
    MONTH_LOSS_PAUSE = params.get('month_loss_pause', 0)  # Pause if month loss > this $
    MONTH_LOSS_PCT_PAUSE = params.get('month_loss_pct_pause', 0)  # Pause if month loss > this % of balance
    CONSEC_LOSS_PAUSE = params.get('consec_loss_pause', 0)  # Pause after N consecutive losses
    DAILY_LOSS_PAUSE = params.get('daily_loss_pause', 0)  # Pause if daily loss > this $

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

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    daily_pnl = {}
    skipped = 0
    consecutive_losses = 0
    paused_months = set()

    for i in range(200, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        month_str = current_time.strftime('%Y-%m')
        day_str = current_time.strftime('%Y-%m-%d')
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

                if day_str not in daily_pnl:
                    daily_pnl[day_str] = 0
                daily_pnl[day_str] += pnl

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

            # ========== AUTO-PAUSE CHECKS ==========
            skip_trade = False

            # Check if month is paused
            if month_str in paused_months:
                skipped += 1
                continue

            # Check month loss threshold (absolute)
            if MONTH_LOSS_PAUSE > 0:
                current_month_pnl = monthly_pnl.get(month_str, 0)
                if current_month_pnl < -MONTH_LOSS_PAUSE:
                    paused_months.add(month_str)
                    skipped += 1
                    continue

            # Check month loss threshold (percentage)
            if MONTH_LOSS_PCT_PAUSE > 0:
                current_month_pnl = monthly_pnl.get(month_str, 0)
                if current_month_pnl < -(balance * MONTH_LOSS_PCT_PAUSE / 100):
                    paused_months.add(month_str)
                    skipped += 1
                    continue

            # Check consecutive losses
            if CONSEC_LOSS_PAUSE > 0 and consecutive_losses >= CONSEC_LOSS_PAUSE:
                skipped += 1
                consecutive_losses = 0  # Reset after skip
                continue

            # Check daily loss
            if DAILY_LOSS_PAUSE > 0:
                current_day_pnl = daily_pnl.get(day_str, 0)
                if current_day_pnl < -DAILY_LOSS_PAUSE:
                    skipped += 1
                    continue

            # ========================================

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
        'monthly_pnl': monthly_pnl,
        'skipped': skipped,
        'paused_months': len(paused_months)
    }

def main():
    log("=" * 70)
    log("RSI v3.7 + Monthly Performance Auto-Pause")
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

        base_params = {
            'rsi_os': 42, 'rsi_ob': 58,
            'sl_mult': 1.5,
            'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
            'max_holding': 46,
            'atr_min': 20, 'atr_max': 80
        }

        best_results = []

        configs = [
            {'name': 'Original', 'month_loss_pause': 0, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            # Monthly absolute loss pause
            {'name': 'MonthLoss>$100', 'month_loss_pause': 100, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            {'name': 'MonthLoss>$150', 'month_loss_pause': 150, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            {'name': 'MonthLoss>$200', 'month_loss_pause': 200, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            {'name': 'MonthLoss>$250', 'month_loss_pause': 250, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            {'name': 'MonthLoss>$300', 'month_loss_pause': 300, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            # Monthly % loss pause
            {'name': 'MonthLoss>1%', 'month_loss_pause': 0, 'month_loss_pct_pause': 1.0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            {'name': 'MonthLoss>1.5%', 'month_loss_pause': 0, 'month_loss_pct_pause': 1.5, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            {'name': 'MonthLoss>2%', 'month_loss_pause': 0, 'month_loss_pct_pause': 2.0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            {'name': 'MonthLoss>3%', 'month_loss_pause': 0, 'month_loss_pct_pause': 3.0, 'consec_loss_pause': 0, 'daily_loss_pause': 0},
            # Consecutive losses pause
            {'name': 'Consec>=2', 'month_loss_pause': 0, 'month_loss_pct_pause': 0, 'consec_loss_pause': 2, 'daily_loss_pause': 0},
            {'name': 'Consec>=3', 'month_loss_pause': 0, 'month_loss_pct_pause': 0, 'consec_loss_pause': 3, 'daily_loss_pause': 0},
            {'name': 'Consec>=4', 'month_loss_pause': 0, 'month_loss_pct_pause': 0, 'consec_loss_pause': 4, 'daily_loss_pause': 0},
            # Daily loss pause
            {'name': 'DailyLoss>$50', 'month_loss_pause': 0, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 50},
            {'name': 'DailyLoss>$100', 'month_loss_pause': 0, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 100},
            # Combined
            {'name': 'MonthLoss$150+Consec3', 'month_loss_pause': 150, 'month_loss_pct_pause': 0, 'consec_loss_pause': 3, 'daily_loss_pause': 0},
            {'name': 'MonthLoss1.5%+Consec3', 'month_loss_pause': 0, 'month_loss_pct_pause': 1.5, 'consec_loss_pause': 3, 'daily_loss_pause': 0},
            {'name': 'Month$100+Daily$50', 'month_loss_pause': 100, 'month_loss_pct_pause': 0, 'consec_loss_pause': 0, 'daily_loss_pause': 50},
        ]

        log("-" * 70)
        log("Testing Auto-Pause Configurations...")
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
                f"Trades:{result['total_trades']:>3} | Skip:{result['skipped']:>3} | Paused:{result['paused_months']}")

        # Sort
        best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        log("\n" + "=" * 70)
        log("TOP 5 CONFIGURATIONS")
        log("=" * 70)

        for i, r in enumerate(best_results[:5], 1):
            log(f"\n{i}. {r['name']}")
            log(f"   Loss: {r['losing_months']} | Return: {r['total_return']:.1f}% | DD: {r['max_drawdown']:.1f}%")
            log(f"   Trades: {r['total_trades']} | Skipped: {r['skipped']} | Months Paused: {r['paused_months']}")

        # Best monthly
        best = best_results[0]
        log("\n" + "=" * 70)
        log(f"BEST: {best['name']}")
        log("=" * 70)
        log("Monthly P/L:")
        for m, p in sorted(best['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            log(f"  {m}: ${p:+,.2f}{marker}")

        # Check if we found 0-loss
        zero_loss = [r for r in best_results if r['losing_months'] == 0]
        if zero_loss:
            log("\n" + "=" * 70)
            log("ZERO-LOSS CONFIGURATIONS FOUND!")
            log("=" * 70)
            for r in zero_loss:
                log(f"\n{r['name']}: Return={r['total_return']:.1f}%, Trades={r['total_trades']}")

    finally:
        mt5.shutdown()
        log("\nDone!")

if __name__ == "__main__":
    main()
