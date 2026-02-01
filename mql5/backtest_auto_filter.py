"""
RSI v3.7 Auto-Detection Filter Development
===========================================
Develop smart filters that automatically detect risky conditions.
Target: 0 losing months WITHOUT skipping specific months.
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

def run_backtest_auto(df, params):
    """Backtest with auto-detection filters"""
    RSI_OS = params['rsi_os']
    RSI_OB = params['rsi_ob']
    SL_MULT = params['sl_mult']
    TP_LOW = params['tp_low']
    TP_MED = params['tp_med']
    TP_HIGH = params['tp_high']
    MAX_HOLDING = params['max_holding']
    MIN_ATR_PCT = params['atr_min']
    MAX_ATR_PCT = params['atr_max']

    # Auto-detection parameters
    # Trend Detection
    TREND_SMA_PERIOD = params.get('trend_sma', 50)
    TREND_STRENGTH_THRESH = params.get('trend_strength', 0)  # % change threshold
    TREND_ALIGN_ONLY = params.get('trend_align', False)  # Only trade with trend

    # Momentum Detection
    MOM_PERIOD = params.get('mom_period', 20)
    MOM_THRESH = params.get('mom_thresh', 0)  # Skip if momentum too strong against

    # Volatility Regime
    VOL_SPIKE_MULT = params.get('vol_spike', 0)  # Skip if ATR > X * avg ATR

    # Price Distance from MA
    MA_DIST_THRESH = params.get('ma_dist', 0)  # Skip if price too far from MA

    # RSI Divergence
    RSI_DIV_CHECK = params.get('rsi_div', False)

    # Rolling Performance
    ROLL_WIN_THRESH = params.get('roll_win', 0)  # Min win rate in last N trades
    ROLL_LOOKBACK = params.get('roll_lookback', 10)

    # ADX Trend Strength
    ADX_PERIOD = params.get('adx_period', 14)
    ADX_THRESH = params.get('adx_thresh', 0)  # Skip if ADX > threshold (strong trend)

    # RSI(10) using SMA
    rsi_period = 10
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss_series = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = np.where(loss_series == 0, 100, gain / loss_series)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_avg'] = df['atr'].rolling(100).mean()

    def atr_percentile(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0

    df['atr_pct'] = df['atr'].rolling(100).apply(atr_percentile, raw=True)

    # Trend indicators
    df['sma'] = df['close'].rolling(TREND_SMA_PERIOD).mean()
    df['sma_slope'] = (df['sma'] - df['sma'].shift(TREND_SMA_PERIOD)) / df['sma'].shift(TREND_SMA_PERIOD) * 100
    df['price_vs_sma'] = (df['close'] - df['sma']) / df['sma'] * 100

    # Momentum
    df['momentum'] = (df['close'] - df['close'].shift(MOM_PERIOD)) / df['close'].shift(MOM_PERIOD) * 100

    # ADX calculation
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr_adx = np.maximum(df['high'] - df['low'],
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                  abs(df['low'] - df['close'].shift(1))))

    atr_adx = tr_adx.rolling(ADX_PERIOD).mean()
    plus_di = 100 * (plus_dm.rolling(ADX_PERIOD).mean() / atr_adx)
    minus_di = 100 * (minus_dm.rolling(ADX_PERIOD).mean() / atr_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(ADX_PERIOD).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    recent_trades = []  # Track recent trade results

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
                    recent_trades.append(1)
                else:
                    losses += 1
                    recent_trades.append(0)

                # Keep only last N trades
                if len(recent_trades) > ROLL_LOOKBACK:
                    recent_trades.pop(0)

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

            # ============ AUTO-DETECTION FILTERS ============

            skip_trade = False

            # Filter 1: Trend Strength - Skip if strong trend against signal
            if TREND_STRENGTH_THRESH > 0:
                sma_slope = row['sma_slope']
                if signal == 1 and sma_slope < -TREND_STRENGTH_THRESH:  # Buying in downtrend
                    skip_trade = True
                elif signal == -1 and sma_slope > TREND_STRENGTH_THRESH:  # Selling in uptrend
                    skip_trade = True

            # Filter 2: Trend Alignment - Only trade with trend
            if TREND_ALIGN_ONLY and not skip_trade:
                if signal == 1 and row['close'] < row['sma']:  # Buy below MA
                    skip_trade = True
                elif signal == -1 and row['close'] > row['sma']:  # Sell above MA
                    skip_trade = True

            # Filter 3: Momentum - Skip if momentum strongly against
            if MOM_THRESH > 0 and not skip_trade:
                mom = row['momentum']
                if signal == 1 and mom < -MOM_THRESH:  # Buying when strong down momentum
                    skip_trade = True
                elif signal == -1 and mom > MOM_THRESH:  # Selling when strong up momentum
                    skip_trade = True

            # Filter 4: Volatility Spike - Skip if volatility too high
            if VOL_SPIKE_MULT > 0 and not skip_trade:
                if row['atr'] > row['atr_avg'] * VOL_SPIKE_MULT:
                    skip_trade = True

            # Filter 5: Price Distance from MA - Skip if price too extended
            if MA_DIST_THRESH > 0 and not skip_trade:
                price_dist = abs(row['price_vs_sma'])
                if price_dist > MA_DIST_THRESH:
                    skip_trade = True

            # Filter 6: ADX Trend Strength - Skip if trend too strong
            if ADX_THRESH > 0 and not skip_trade:
                if row['adx'] > ADX_THRESH:
                    skip_trade = True

            # Filter 7: Rolling Performance - Skip if recent performance poor
            if ROLL_WIN_THRESH > 0 and len(recent_trades) >= ROLL_LOOKBACK and not skip_trade:
                recent_win_rate = sum(recent_trades) / len(recent_trades) * 100
                if recent_win_rate < ROLL_WIN_THRESH:
                    skip_trade = True

            if skip_trade:
                continue

            # ============ EXECUTE TRADE ============
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
    log("RSI v3.7 AUTO-DETECTION FILTER DEVELOPMENT")
    log("Target: 0 Losing Months with Smart Filters")
    log("=" * 70)

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

        base_params = {
            'rsi_os': 42, 'rsi_ob': 58,
            'sl_mult': 1.5,
            'tp_low': 2.4, 'tp_med': 3.0, 'tp_high': 3.6,
            'max_holding': 46,
            'atr_min': 20, 'atr_max': 80
        }

        best_results = []
        tested = 0

        # Test 1: Trend Strength Filter
        log("\n--- Filter 1: Trend Strength (SMA Slope) ---")
        for sma_period in [20, 30, 50, 100]:
            for strength in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
                params = base_params.copy()
                params['trend_sma'] = sma_period
                params['trend_strength'] = strength

                df = df_original.copy()
                result = run_backtest_auto(df, params)
                tested += 1

                if result['losing_months'] <= 1 and result['total_return'] > 30:
                    result['params'] = params.copy()
                    result['filter'] = f"Trend SMA={sma_period}, Strength={strength}%"
                    best_results.append(result)
                    log(f"  SMA={sma_period}, Str={strength}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Test 2: Momentum Filter
        log("\n--- Filter 2: Momentum ---")
        for mom_period in [10, 20, 30]:
            for mom_thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
                params = base_params.copy()
                params['mom_period'] = mom_period
                params['mom_thresh'] = mom_thresh

                df = df_original.copy()
                result = run_backtest_auto(df, params)
                tested += 1

                if result['losing_months'] <= 1 and result['total_return'] > 30:
                    result['params'] = params.copy()
                    result['filter'] = f"Momentum P={mom_period}, T={mom_thresh}%"
                    best_results.append(result)
                    log(f"  Mom P={mom_period}, T={mom_thresh}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Test 3: ADX Filter
        log("\n--- Filter 3: ADX Trend Strength ---")
        for adx_thresh in [20, 25, 30, 35, 40, 45, 50]:
            params = base_params.copy()
            params['adx_thresh'] = adx_thresh

            df = df_original.copy()
            result = run_backtest_auto(df, params)
            tested += 1

            if result['losing_months'] <= 2 and result['total_return'] > 20:
                result['params'] = params.copy()
                result['filter'] = f"ADX < {adx_thresh}"
                best_results.append(result)
                log(f"  ADX<{adx_thresh}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%, Trades={result['total_trades']}")

        # Test 4: Price Distance from MA
        log("\n--- Filter 4: Price Distance from MA ---")
        for ma_dist in [1.0, 1.5, 2.0, 2.5, 3.0]:
            params = base_params.copy()
            params['ma_dist'] = ma_dist

            df = df_original.copy()
            result = run_backtest_auto(df, params)
            tested += 1

            if result['losing_months'] <= 2:
                result['params'] = params.copy()
                result['filter'] = f"MA Dist < {ma_dist}%"
                best_results.append(result)
                log(f"  MA Dist<{ma_dist}%: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Test 5: Volatility Spike
        log("\n--- Filter 5: Volatility Spike ---")
        for vol_spike in [1.5, 2.0, 2.5, 3.0]:
            params = base_params.copy()
            params['vol_spike'] = vol_spike

            df = df_original.copy()
            result = run_backtest_auto(df, params)
            tested += 1

            if result['losing_months'] <= 2:
                result['params'] = params.copy()
                result['filter'] = f"Vol Spike < {vol_spike}x"
                best_results.append(result)
                log(f"  Vol<{vol_spike}x: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        # Test 6: Combined Filters
        log("\n--- Filter 6: Combined Approaches ---")
        combinations = [
            {'trend_sma': 50, 'trend_strength': 2.0, 'adx_thresh': 35},
            {'trend_sma': 50, 'trend_strength': 1.5, 'adx_thresh': 30},
            {'trend_sma': 30, 'trend_strength': 2.5, 'mom_thresh': 2.0, 'mom_period': 20},
            {'trend_sma': 50, 'trend_strength': 3.0, 'vol_spike': 2.0},
            {'adx_thresh': 30, 'mom_thresh': 2.0, 'mom_period': 20},
            {'adx_thresh': 25, 'trend_strength': 2.0, 'trend_sma': 50},
            {'trend_sma': 50, 'trend_strength': 2.0, 'ma_dist': 2.0},
            {'adx_thresh': 35, 'ma_dist': 2.0, 'vol_spike': 2.0},
            # More aggressive combinations
            {'trend_sma': 30, 'trend_strength': 3.0, 'adx_thresh': 30},
            {'trend_sma': 50, 'trend_strength': 4.0, 'adx_thresh': 25},
            {'mom_thresh': 3.0, 'mom_period': 20, 'adx_thresh': 30},
            {'trend_sma': 50, 'trend_strength': 3.5, 'mom_thresh': 2.5, 'mom_period': 20},
        ]

        for combo in combinations:
            params = base_params.copy()
            params.update(combo)

            df = df_original.copy()
            result = run_backtest_auto(df, params)
            tested += 1
            result['params'] = params.copy()
            result['filter'] = f"Combined: {combo}"
            best_results.append(result)

            if result['losing_months'] <= 1:
                log(f"  {combo}: Loss={result['losing_months']}, Ret={result['total_return']:.1f}%")

        log(f"\nTotal tested: {tested}")

        # Sort results
        best_results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        log("\n" + "=" * 70)
        log("TOP 15 CONFIGURATIONS")
        log("=" * 70)

        for i, r in enumerate(best_results[:15], 1):
            log(f"\n{i}. Loss: {r['losing_months']} | Ret: {r['total_return']:.1f}% | DD: {r['max_drawdown']:.1f}% | Trades: {r['total_trades']}")
            log(f"   Filter: {r.get('filter', 'N/A')}")

        # Find zero loss configurations
        zero_loss = [r for r in best_results if r['losing_months'] == 0]
        if zero_loss:
            best = max(zero_loss, key=lambda x: x['total_return'])
            log("\n" + "=" * 70)
            log("ZERO LOSS AUTO-DETECTION FILTER FOUND!")
            log("=" * 70)
            log(f"Filter: {best.get('filter', 'N/A')}")
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
        else:
            # Show best with 1 losing month
            one_loss = [r for r in best_results if r['losing_months'] == 1]
            if one_loss:
                best = max(one_loss, key=lambda x: x['total_return'])
                log("\n" + "=" * 70)
                log("BEST 1-LOSS CONFIGURATION (Close to Zero!)")
                log("=" * 70)
                log(f"Filter: {best.get('filter', 'N/A')}")
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
