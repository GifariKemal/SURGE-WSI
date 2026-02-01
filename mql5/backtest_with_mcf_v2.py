"""
RSI v3.7 + MarketConditionFilter v2
===================================
Fixed ADX calculation and added debug info
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - Vectorized"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    atr_sum = tr.rolling(period).sum()
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    price_range = highest - lowest
    price_range = price_range.replace(0, np.nan)

    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    return chop.fillna(50)

def calculate_adx(high, low, close, period=14):
    """ADX - Fixed calculation"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm.astype(float)) & (minus_dm > 0), minus_dm, 0)

    # Smoothed averages
    tr_smooth = pd.Series(tr).rolling(period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(period).mean()

    # DI lines
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)

    # DX
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * di_diff / (di_sum + 1e-10)

    # ADX
    adx = dx.rolling(period).mean()
    return adx.fillna(25)

def calculate_sma_slope(close, period=50):
    """SMA Slope as % change"""
    sma = close.rolling(period).mean()
    slope = (sma - sma.shift(period)) / sma.shift(period) * 100
    return slope.fillna(0)

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
    CHOP_SKIP = params.get('chop_skip', 100)
    ADX_MIN = params.get('adx_min', 0)
    SLOPE_FILTER = params.get('slope_filter', 0)  # Skip counter-trend if slope > this
    THURSDAY_MULT = params.get('thursday_mult', 1.0)  # Position multiplier on Thursday

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
    df['chop'] = calculate_choppiness(df['high'], df['low'], df['close'])
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'])
    df['slope'] = calculate_sma_slope(df['close'], 50)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    skipped_chop = 0
    skipped_adx = 0
    skipped_slope = 0

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

            chop = row['chop']
            adx = row['adx']
            slope = row['slope']

            # Choppiness filter
            if chop > CHOP_SKIP:
                skipped_chop += 1
                continue

            # ADX filter (skip if ADX too low = no trend)
            if adx < ADX_MIN:
                skipped_adx += 1
                continue

            rsi = row['rsi']
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if not signal:
                continue

            # Slope filter (skip counter-trend)
            if SLOPE_FILTER > 0:
                if signal == 1 and slope < -SLOPE_FILTER:  # Buy in downtrend
                    skipped_slope += 1
                    continue
                elif signal == -1 and slope > SLOPE_FILTER:  # Sell in uptrend
                    skipped_slope += 1
                    continue

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
        'skipped': {'chop': skipped_chop, 'adx': skipped_adx, 'slope': skipped_slope}
    }

def main():
    log("=" * 70)
    log("RSI v3.7 + MarketConditionFilter v2")
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

        # Debug: Check indicator values
        df_debug = df_original.copy()
        df_debug['chop'] = calculate_choppiness(df_debug['high'], df_debug['low'], df_debug['close'])
        df_debug['adx'] = calculate_adx(df_debug['high'], df_debug['low'], df_debug['close'])
        df_debug['slope'] = calculate_sma_slope(df_debug['close'], 50)

        log(f"\nIndicator Stats:")
        log(f"  Choppiness: min={df_debug['chop'].min():.1f}, max={df_debug['chop'].max():.1f}, mean={df_debug['chop'].mean():.1f}")
        log(f"  ADX: min={df_debug['adx'].min():.1f}, max={df_debug['adx'].max():.1f}, mean={df_debug['adx'].mean():.1f}")
        log(f"  Slope: min={df_debug['slope'].min():.2f}%, max={df_debug['slope'].max():.2f}%, mean={df_debug['slope'].mean():.2f}%")

        base_params = {
            'rsi_os': 42, 'rsi_ob': 58,
            'sl_mult': 1.5,
            'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
            'max_holding': 46,
            'atr_min': 20, 'atr_max': 80
        }

        best_results = []

        # Test configurations based on indicator stats
        configs = [
            {'name': 'Original', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 0, 'thursday_mult': 1.0},
            {'name': 'Slope>1%', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 1.0, 'thursday_mult': 1.0},
            {'name': 'Slope>1.5%', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 1.5, 'thursday_mult': 1.0},
            {'name': 'Slope>2%', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 2.0, 'thursday_mult': 1.0},
            {'name': 'Slope>2.5%', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 2.5, 'thursday_mult': 1.0},
            {'name': 'Slope>3%', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 3.0, 'thursday_mult': 1.0},
            {'name': 'Chop<55', 'chop_skip': 55, 'adx_min': 0, 'slope_filter': 0, 'thursday_mult': 1.0},
            {'name': 'Chop<50', 'chop_skip': 50, 'adx_min': 0, 'slope_filter': 0, 'thursday_mult': 1.0},
            {'name': 'Chop<55+Slope>2%', 'chop_skip': 55, 'adx_min': 0, 'slope_filter': 2.0, 'thursday_mult': 1.0},
            {'name': 'Chop<50+Slope>2%', 'chop_skip': 50, 'adx_min': 0, 'slope_filter': 2.0, 'thursday_mult': 1.0},
            {'name': 'Slope>2%+Thu0.5', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 2.0, 'thursday_mult': 0.5},
            {'name': 'Slope>2.5%+Thu0.5', 'chop_skip': 100, 'adx_min': 0, 'slope_filter': 2.5, 'thursday_mult': 0.5},
            {'name': 'Full:Chop55+Slope2+Thu0.5', 'chop_skip': 55, 'adx_min': 0, 'slope_filter': 2.0, 'thursday_mult': 0.5},
        ]

        log("\n" + "-" * 70)
        log("Testing Filter Combinations...")
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
            skipped = result['skipped']
            log(f"{cfg['name']:<30} | {status:<7} | Ret:{result['total_return']:>6.1f}% | "
                f"Trades:{result['total_trades']:>3} | Skip(C/A/S):{skipped['chop']}/{skipped['adx']}/{skipped['slope']}")

        # Sort
        best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        log("\n" + "=" * 70)
        log("TOP 5 CONFIGURATIONS")
        log("=" * 70)

        for i, r in enumerate(best_results[:5], 1):
            log(f"\n{i}. {r['name']}")
            log(f"   Loss Months: {r['losing_months']} | Return: {r['total_return']:.1f}% | DD: {r['max_drawdown']:.1f}%")
            log(f"   Trades: {r['total_trades']} | WinRate: {r['win_rate']:.1f}%")
            log(f"   Config: chop<{r['config']['chop_skip']}, slope>{r['config']['slope_filter']}%, thu_mult={r['config']['thursday_mult']}")

        # Show monthly for best
        best = best_results[0]
        log("\n" + "=" * 70)
        log(f"BEST CONFIG: {best['name']}")
        log("=" * 70)
        log(f"Monthly P/L:")
        for m, p in sorted(best['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            log(f"  {m}: ${p:+,.2f}{marker}")

    finally:
        mt5.shutdown()
        log("\nDone!")

if __name__ == "__main__":
    main()
