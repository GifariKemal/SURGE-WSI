"""
RSI v3.7 + MarketConditionFilter Integration
=============================================
Using existing MarketConditionFilter (Choppiness + ADX + Thursday)
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

def calculate_choppiness(df, period=14):
    """Choppiness Index: High = Ranging, Low = Trending"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(period).sum()
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    price_range = highest - lowest
    price_range = price_range.replace(0, np.nan)

    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    return chop

def calculate_adx(df, period=14):
    """ADX: Trend Strength"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

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

    # MarketConditionFilter params
    CHOP_SKIP = params.get('chop_skip', 100)  # Skip if chop > this
    ADX_SKIP = params.get('adx_skip', 0)      # Skip if adx < this
    THURSDAY_SKIP = params.get('thursday_skip', False)

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

    # Market Condition Indicators
    df['chop'] = calculate_choppiness(df)
    df['adx'] = calculate_adx(df)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    skipped_by_filter = 0

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

            # ========== MARKET CONDITION FILTER ==========
            chop = row['chop']
            adx = row['adx']

            # Choppiness filter (skip ranging markets)
            if chop > CHOP_SKIP:
                skipped_by_filter += 1
                continue

            # ADX filter (skip weak trends)
            if adx < ADX_SKIP:
                skipped_by_filter += 1
                continue

            # Thursday filter
            if THURSDAY_SKIP and weekday == 3:  # Thursday
                skipped_by_filter += 1
                continue

            # =============================================

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
        'skipped': skipped_by_filter
    }

def main():
    log("=" * 60)
    log("RSI v3.7 + MarketConditionFilter")
    log("=" * 60)

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

        # Test different filter combinations
        configs = [
            {'name': 'v3.7 Original', 'chop_skip': 100, 'adx_skip': 0, 'thursday_skip': False},
            {'name': 'Chop < 65', 'chop_skip': 65, 'adx_skip': 0, 'thursday_skip': False},
            {'name': 'Chop < 62', 'chop_skip': 62, 'adx_skip': 0, 'thursday_skip': False},
            {'name': 'Chop < 60', 'chop_skip': 60, 'adx_skip': 0, 'thursday_skip': False},
            {'name': 'ADX > 18', 'chop_skip': 100, 'adx_skip': 18, 'thursday_skip': False},
            {'name': 'ADX > 20', 'chop_skip': 100, 'adx_skip': 20, 'thursday_skip': False},
            {'name': 'ADX > 22', 'chop_skip': 100, 'adx_skip': 22, 'thursday_skip': False},
            {'name': 'Skip Thursday', 'chop_skip': 100, 'adx_skip': 0, 'thursday_skip': True},
            {'name': 'Chop<65 + ADX>18', 'chop_skip': 65, 'adx_skip': 18, 'thursday_skip': False},
            {'name': 'Chop<62 + ADX>20', 'chop_skip': 62, 'adx_skip': 20, 'thursday_skip': False},
            {'name': 'Chop<65 + Thu Skip', 'chop_skip': 65, 'adx_skip': 0, 'thursday_skip': True},
            {'name': 'ADX>20 + Thu Skip', 'chop_skip': 100, 'adx_skip': 20, 'thursday_skip': True},
            {'name': 'All Filters Light', 'chop_skip': 65, 'adx_skip': 18, 'thursday_skip': True},
            {'name': 'All Filters Medium', 'chop_skip': 62, 'adx_skip': 20, 'thursday_skip': True},
            {'name': 'All Filters Strict', 'chop_skip': 60, 'adx_skip': 22, 'thursday_skip': True},
        ]

        log("Testing MarketConditionFilter combinations...")
        log("-" * 60)

        for cfg in configs:
            params = base_params.copy()
            params['chop_skip'] = cfg['chop_skip']
            params['adx_skip'] = cfg['adx_skip']
            params['thursday_skip'] = cfg['thursday_skip']

            df = df_original.copy()
            result = run_backtest(df, params)
            result['name'] = cfg['name']
            result['config'] = cfg
            best_results.append(result)

            status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
            log(f"{cfg['name']:<25} | {status:<7} | Ret:{result['total_return']:>6.1f}% | "
                f"DD:{result['max_drawdown']:>5.1f}% | Trades:{result['total_trades']:>3} | Skip:{result['skipped']:>4}")

        # Sort
        best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        log("\n" + "=" * 60)
        log("TOP CONFIGURATIONS")
        log("=" * 60)

        for i, r in enumerate(best_results[:5], 1):
            log(f"\n{i}. {r['name']}")
            log(f"   Loss Months: {r['losing_months']} | Return: {r['total_return']:.1f}% | DD: {r['max_drawdown']:.1f}%")
            log(f"   Trades: {r['total_trades']} | WinRate: {r['win_rate']:.1f}% | Skipped: {r['skipped']}")

        # Best config
        best = best_results[0]
        log("\n" + "=" * 60)
        log(f"BEST: {best['name']}")
        log("=" * 60)
        log(f"Losing Months: {best['losing_months']}")
        log(f"Return: {best['total_return']:.2f}%")
        log(f"Max DD: {best['max_drawdown']:.2f}%")
        log(f"Trades: {best['total_trades']}")
        log(f"WinRate: {best['win_rate']:.1f}%")
        log(f"\nConfig: {best['config']}")
        log(f"\nMonthly P/L:")
        for m, p in sorted(best['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            log(f"  {m}: ${p:+,.2f}{marker}")

        # If still has losses, try more aggressive
        if best['losing_months'] > 0:
            log("\n" + "=" * 60)
            log("EXTENDED SEARCH - More Aggressive Filters")
            log("=" * 60)

            for chop in [55, 58, 60, 62]:
                for adx in [22, 24, 26, 28]:
                    params = base_params.copy()
                    params['chop_skip'] = chop
                    params['adx_skip'] = adx
                    params['thursday_skip'] = True

                    df = df_original.copy()
                    result = run_backtest(df, params)

                    if result['losing_months'] == 0 and result['total_return'] > 20:
                        log(f"0-LOSS! Chop<{chop} ADX>{adx} Thu=Skip | Ret:{result['total_return']:.1f}% Trades:{result['total_trades']}")
                        result['config'] = {'chop_skip': chop, 'adx_skip': adx, 'thursday_skip': True}
                        best_results.append(result)

            zero_loss = [r for r in best_results if r['losing_months'] == 0]
            if zero_loss:
                best_zero = max(zero_loss, key=lambda x: x['total_return'])
                log(f"\nBest 0-Loss Config: {best_zero['config']}")
                log(f"Return: {best_zero['total_return']:.2f}%")
                log(f"\nMonthly P/L:")
                for m, p in sorted(best_zero['monthly_pnl'].items()):
                    log(f"  {m}: ${p:+,.2f}")

    finally:
        mt5.shutdown()
        log("\nDone!")

if __name__ == "__main__":
    main()
