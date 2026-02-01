"""
RSI v3.7 - Deep Parameter Analysis & Optimization
==================================================
Analyze optimal parameters for the strategy
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '61045904'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'iy#K5L7sF')
MT5_SERVER = os.getenv('MT5_SERVER', 'FinexBisnisSolusi-Demo')
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe")

SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50000.0  # Match MQL5 backtest

def connect_mt5():
    if not MT5_PASSWORD:
        return False
    if not mt5.initialize(path=MT5_PATH):
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return False
    return True

def get_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_indicators(df, rsi_period, atr_period, sma_fast, sma_slow, sma_slope_lookback):
    """Calculate all indicators with given parameters"""

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(atr_period).mean()

    # ATR Percentile
    def atr_pct_func(x):
        if len(x) < 10:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100
    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # SMAs
    df['sma_fast'] = df['close'].rolling(sma_fast).mean()
    df['sma_slow'] = df['close'].rolling(sma_slow).mean()
    df['sma_slope'] = (df['sma_fast'] / df['sma_fast'].shift(sma_slope_lookback) - 1) * 100

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)

def run_backtest(df, params):
    """Run backtest with given parameters"""

    # Unpack parameters
    rsi_oversold = params['rsi_oversold']
    rsi_overbought = params['rsi_overbought']
    sl_mult = params['sl_mult']
    tp_low = params['tp_low']
    tp_med = params['tp_med']
    tp_high = params['tp_high']
    tp_bonus = params['tp_bonus']
    max_holding = params['max_holding']
    min_atr_pct = params['min_atr_pct']
    max_atr_pct = params['max_atr_pct']
    sma_slope_threshold = params['sma_slope_threshold']
    consec_limit = params['consec_limit']
    consec_cooldown = params['consec_cooldown']
    risk_per_trade = params['risk_per_trade']

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown = 0
    max_drawdown_pct = 0

    position = None
    consecutive_losses = 0
    cooldown_bars = 0
    trades = []

    for i in range(200, len(df) - 20):
        row = df.iloc[i]

        # Decrease cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1
            if cooldown_bars == 0:
                consecutive_losses = 0

        if row['weekday'] >= 5:
            continue

        # Position management
        if position:
            exit_reason = None
            exit_price = None
            pnl = 0
            bars_held = i - position['entry_idx']

            if bars_held >= max_holding:
                exit_price = row['close']
                pnl = (exit_price - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - exit_price) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL'
                    elif row['high'] >= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP'
                else:
                    if row['high'] >= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL'
                    elif row['low'] <= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP'

            if exit_reason:
                balance += pnl

                if balance > peak_balance:
                    peak_balance = balance
                dd = peak_balance - balance
                dd_pct = dd / peak_balance * 100
                if dd > max_drawdown:
                    max_drawdown = dd
                    max_drawdown_pct = dd_pct

                if pnl < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= consec_limit:
                        cooldown_bars = consec_cooldown
                else:
                    consecutive_losses = 0
                    cooldown_bars = 0

                trades.append({
                    'pnl': pnl,
                    'result': 'WIN' if pnl > 0 else 'LOSS',
                    'exit_reason': exit_reason,
                })
                position = None

        # Entry logic
        if not position:
            hour = row['hour']
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                continue

            # Regime filter
            sma_fast_val = row['sma_fast']
            sma_slow_val = row['sma_slow']
            slope = row['sma_slope']

            is_bull = sma_fast_val > sma_slow_val and slope > sma_slope_threshold
            is_bear = sma_fast_val < sma_slow_val and slope < -sma_slope_threshold
            is_sideways = not is_bull and not is_bear

            if not is_sideways:
                continue

            # Cooldown check
            if cooldown_bars > 0:
                continue

            # RSI signal
            rsi = row['rsi']
            signal = 0
            if rsi < rsi_oversold:
                signal = 1
            elif rsi > rsi_overbought:
                signal = -1

            if not signal:
                continue

            # Calculate SL/TP
            entry = row['close']
            atr = row['atr'] if row['atr'] > 0 else entry * 0.002

            if atr_pct < 40:
                tp_mult = tp_low
            elif atr_pct > 60:
                tp_mult = tp_high
            else:
                tp_mult = tp_med

            if 12 <= hour < 16:
                tp_mult += tp_bonus

            if signal == 1:
                sl = entry - atr * sl_mult
                tp = entry + atr * tp_mult
            else:
                sl = entry + atr * sl_mult
                tp = entry - atr * tp_mult

            risk = balance * risk_per_trade / 100
            size = risk / abs(entry - sl) if abs(entry - sl) > 0 else 0

            position = {
                'dir': signal,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'size': size,
                'entry_idx': i,
            }

    # Calculate metrics
    if len(trades) == 0:
        return None

    total_pnl = sum(t['pnl'] for t in trades)
    wins = len([t for t in trades if t['result'] == 'WIN'])
    losses = len([t for t in trades if t['result'] == 'LOSS'])
    win_rate = wins / len(trades) * 100 if trades else 0

    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    avg_win = gross_profit / wins if wins > 0 else 0
    avg_loss = gross_loss / losses if losses > 0 else 0

    # Calculate max consecutive losses
    max_consec_loss = 0
    current_consec = 0
    for t in trades:
        if t['result'] == 'LOSS':
            current_consec += 1
            max_consec_loss = max(max_consec_loss, current_consec)
        else:
            current_consec = 0

    return {
        'total_pnl': total_pnl,
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_consec_loss': max_consec_loss,
        'final_balance': balance,
        'return_pct': (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
    }

def main():
    print("=" * 80)
    print("RSI v3.7 - DEEP PARAMETER ANALYSIS & OPTIMIZATION")
    print("=" * 80)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        # Use same period as MQL5 backtest
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc)

        df = get_data(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
        if df is None:
            print("Failed to get data")
            return
        print(f"Loaded {len(df)} H1 bars")

        # Calculate indicators with default params
        print("Calculating indicators...")
        df = calculate_indicators(df,
                                  rsi_period=10,
                                  atr_period=14,
                                  sma_fast=20,
                                  sma_slow=50,
                                  sma_slope_lookback=10)

        # ============================================
        # BASELINE TEST
        # ============================================
        print("\n" + "=" * 80)
        print("BASELINE (Current Parameters)")
        print("=" * 80)

        baseline_params = {
            'rsi_oversold': 42,
            'rsi_overbought': 58,
            'sl_mult': 1.5,
            'tp_low': 2.4,
            'tp_med': 3.0,
            'tp_high': 3.6,
            'tp_bonus': 0.35,
            'max_holding': 46,
            'min_atr_pct': 20,
            'max_atr_pct': 80,
            'sma_slope_threshold': 0.5,
            'consec_limit': 3,
            'consec_cooldown': 2,
            'risk_per_trade': 1.0,
        }

        baseline = run_backtest(df, baseline_params)
        if baseline:
            print(f"\nBaseline Results:")
            print(f"  Net Profit: ${baseline['total_pnl']:+,.2f}")
            print(f"  Return: {baseline['return_pct']:+.2f}%")
            print(f"  Trades: {baseline['trades']}")
            print(f"  Win Rate: {baseline['win_rate']:.1f}%")
            print(f"  Profit Factor: {baseline['profit_factor']:.2f}")
            print(f"  Max Drawdown: {baseline['max_drawdown_pct']:.1f}%")
            print(f"  Max Consec Loss: {baseline['max_consec_loss']}")

        # ============================================
        # PARAMETER SENSITIVITY ANALYSIS
        # ============================================
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 80)

        # Test RSI levels
        print("\n--- RSI Oversold/Overbought Levels ---")
        rsi_results = []
        for oversold in [35, 38, 40, 42, 45]:
            for overbought in [55, 58, 60, 62, 65]:
                if overbought - oversold < 10:
                    continue
                params = baseline_params.copy()
                params['rsi_oversold'] = oversold
                params['rsi_overbought'] = overbought
                result = run_backtest(df, params)
                if result:
                    rsi_results.append({
                        'oversold': oversold,
                        'overbought': overbought,
                        'pnl': result['total_pnl'],
                        'trades': result['trades'],
                        'win_rate': result['win_rate'],
                        'pf': result['profit_factor'],
                        'dd': result['max_drawdown_pct'],
                    })

        rsi_df = pd.DataFrame(rsi_results).sort_values('pnl', ascending=False)
        print("\nTop 5 RSI Combinations:")
        print(rsi_df.head(10).to_string(index=False))

        # Test SL/TP multipliers
        print("\n--- SL/TP Multipliers ---")
        sltp_results = []
        for sl in [1.0, 1.2, 1.5, 1.8, 2.0]:
            for tp in [2.0, 2.5, 3.0, 3.5, 4.0]:
                if tp <= sl:
                    continue
                params = baseline_params.copy()
                params['sl_mult'] = sl
                params['tp_low'] = tp * 0.8
                params['tp_med'] = tp
                params['tp_high'] = tp * 1.2
                result = run_backtest(df, params)
                if result:
                    sltp_results.append({
                        'sl_mult': sl,
                        'tp_mult': tp,
                        'pnl': result['total_pnl'],
                        'trades': result['trades'],
                        'win_rate': result['win_rate'],
                        'pf': result['profit_factor'],
                        'dd': result['max_drawdown_pct'],
                    })

        sltp_df = pd.DataFrame(sltp_results).sort_values('pnl', ascending=False)
        print("\nTop 5 SL/TP Combinations:")
        print(sltp_df.head(10).to_string(index=False))

        # Test ATR Percentile Range
        print("\n--- ATR Percentile Range ---")
        atr_results = []
        for min_atr in [10, 15, 20, 25, 30]:
            for max_atr in [70, 75, 80, 85, 90]:
                params = baseline_params.copy()
                params['min_atr_pct'] = min_atr
                params['max_atr_pct'] = max_atr
                result = run_backtest(df, params)
                if result:
                    atr_results.append({
                        'min_atr': min_atr,
                        'max_atr': max_atr,
                        'pnl': result['total_pnl'],
                        'trades': result['trades'],
                        'win_rate': result['win_rate'],
                        'pf': result['profit_factor'],
                        'dd': result['max_drawdown_pct'],
                    })

        atr_df = pd.DataFrame(atr_results).sort_values('pnl', ascending=False)
        print("\nTop 5 ATR Percentile Ranges:")
        print(atr_df.head(10).to_string(index=False))

        # Test SMA Slope Threshold
        print("\n--- SMA Slope Threshold ---")
        slope_results = []
        for slope in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            params = baseline_params.copy()
            params['sma_slope_threshold'] = slope
            result = run_backtest(df, params)
            if result:
                slope_results.append({
                    'slope_threshold': slope,
                    'pnl': result['total_pnl'],
                    'trades': result['trades'],
                    'win_rate': result['win_rate'],
                    'pf': result['profit_factor'],
                    'dd': result['max_drawdown_pct'],
                })

        slope_df = pd.DataFrame(slope_results).sort_values('pnl', ascending=False)
        print("\nSMA Slope Threshold Results:")
        print(slope_df.to_string(index=False))

        # Test Max Holding Hours
        print("\n--- Max Holding Hours ---")
        holding_results = []
        for hours in [24, 36, 46, 56, 72]:
            params = baseline_params.copy()
            params['max_holding'] = hours
            result = run_backtest(df, params)
            if result:
                holding_results.append({
                    'max_holding': hours,
                    'pnl': result['total_pnl'],
                    'trades': result['trades'],
                    'win_rate': result['win_rate'],
                    'pf': result['profit_factor'],
                    'dd': result['max_drawdown_pct'],
                })

        holding_df = pd.DataFrame(holding_results).sort_values('pnl', ascending=False)
        print("\nMax Holding Hours Results:")
        print(holding_df.to_string(index=False))

        # Test Consecutive Loss Settings
        print("\n--- Consecutive Loss Filter ---")
        consec_results = []
        for limit in [2, 3, 4, 5]:
            for cooldown in [1, 2, 3, 4, 5]:
                params = baseline_params.copy()
                params['consec_limit'] = limit
                params['consec_cooldown'] = cooldown
                result = run_backtest(df, params)
                if result:
                    consec_results.append({
                        'limit': limit,
                        'cooldown': cooldown,
                        'pnl': result['total_pnl'],
                        'trades': result['trades'],
                        'win_rate': result['win_rate'],
                        'pf': result['profit_factor'],
                        'dd': result['max_drawdown_pct'],
                        'max_consec': result['max_consec_loss'],
                    })

        consec_df = pd.DataFrame(consec_results).sort_values('pnl', ascending=False)
        print("\nTop Consecutive Loss Settings:")
        print(consec_df.head(10).to_string(index=False))

        # ============================================
        # FIND OPTIMAL COMBINATION
        # ============================================
        print("\n" + "=" * 80)
        print("FINDING OPTIMAL PARAMETER COMBINATION")
        print("=" * 80)

        # Get best from each category
        best_rsi = rsi_df.iloc[0] if len(rsi_df) > 0 else None
        best_sltp = sltp_df.iloc[0] if len(sltp_df) > 0 else None
        best_atr = atr_df.iloc[0] if len(atr_df) > 0 else None
        best_slope = slope_df.iloc[0] if len(slope_df) > 0 else None
        best_holding = holding_df.iloc[0] if len(holding_df) > 0 else None
        best_consec = consec_df.iloc[0] if len(consec_df) > 0 else None

        # Build optimized params
        optimized_params = baseline_params.copy()

        if best_rsi is not None:
            optimized_params['rsi_oversold'] = int(best_rsi['oversold'])
            optimized_params['rsi_overbought'] = int(best_rsi['overbought'])

        if best_sltp is not None:
            optimized_params['sl_mult'] = best_sltp['sl_mult']
            tp = best_sltp['tp_mult']
            optimized_params['tp_low'] = tp * 0.8
            optimized_params['tp_med'] = tp
            optimized_params['tp_high'] = tp * 1.2

        if best_atr is not None:
            optimized_params['min_atr_pct'] = int(best_atr['min_atr'])
            optimized_params['max_atr_pct'] = int(best_atr['max_atr'])

        if best_slope is not None:
            optimized_params['sma_slope_threshold'] = best_slope['slope_threshold']

        if best_holding is not None:
            optimized_params['max_holding'] = int(best_holding['max_holding'])

        if best_consec is not None:
            optimized_params['consec_limit'] = int(best_consec['limit'])
            optimized_params['consec_cooldown'] = int(best_consec['cooldown'])

        print("\nOptimized Parameters:")
        for key, value in optimized_params.items():
            baseline_val = baseline_params[key]
            changed = "  <-- CHANGED" if value != baseline_val else ""
            print(f"  {key}: {value}{changed}")

        # Test optimized params
        optimized = run_backtest(df, optimized_params)

        print("\n" + "=" * 80)
        print("COMPARISON: BASELINE vs OPTIMIZED")
        print("=" * 80)

        if optimized:
            print(f"\n{'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Change':>15}")
            print("-" * 70)
            print(f"{'Net Profit':<25} ${baseline['total_pnl']:>13,.0f} ${optimized['total_pnl']:>13,.0f} ${optimized['total_pnl']-baseline['total_pnl']:>+13,.0f}")
            print(f"{'Return %':<25} {baseline['return_pct']:>14.1f}% {optimized['return_pct']:>14.1f}% {optimized['return_pct']-baseline['return_pct']:>+14.1f}%")
            print(f"{'Trades':<25} {baseline['trades']:>15} {optimized['trades']:>15} {optimized['trades']-baseline['trades']:>+15}")
            print(f"{'Win Rate':<25} {baseline['win_rate']:>14.1f}% {optimized['win_rate']:>14.1f}% {optimized['win_rate']-baseline['win_rate']:>+14.1f}%")
            print(f"{'Profit Factor':<25} {baseline['profit_factor']:>15.2f} {optimized['profit_factor']:>15.2f} {optimized['profit_factor']-baseline['profit_factor']:>+15.2f}")
            print(f"{'Max Drawdown':<25} {baseline['max_drawdown_pct']:>14.1f}% {optimized['max_drawdown_pct']:>14.1f}% {optimized['max_drawdown_pct']-baseline['max_drawdown_pct']:>+14.1f}%")
            print(f"{'Max Consec Loss':<25} {baseline['max_consec_loss']:>15} {optimized['max_consec_loss']:>15} {optimized['max_consec_loss']-baseline['max_consec_loss']:>+15}")
            print(f"{'Avg Win':<25} ${baseline['avg_win']:>13,.0f} ${optimized['avg_win']:>13,.0f} ${optimized['avg_win']-baseline['avg_win']:>+13,.0f}")
            print(f"{'Avg Loss':<25} ${baseline['avg_loss']:>13,.0f} ${optimized['avg_loss']:>13,.0f} ${optimized['avg_loss']-baseline['avg_loss']:>+13,.0f}")

        # ============================================
        # GENERATE MQL5 OPTIMIZED PARAMETERS
        # ============================================
        print("\n" + "=" * 80)
        print("RECOMMENDED MQL5 EA PARAMETERS")
        print("=" * 80)

        print(f"""
// Strategy Parameters
input int      RSI_Period = 10;
input int      RSI_Oversold = {optimized_params['rsi_oversold']};
input int      RSI_Overbought = {optimized_params['rsi_overbought']};

// SL/TP Settings
input double   SL_Multiplier = {optimized_params['sl_mult']};
input double   TP_Low = {optimized_params['tp_low']:.1f};
input double   TP_Med = {optimized_params['tp_med']:.1f};
input double   TP_High = {optimized_params['tp_high']:.1f};
input double   TP_Session_Bonus = {optimized_params['tp_bonus']};
input int      Max_Holding_Hours = {optimized_params['max_holding']};

// ATR Percentile Filter
input int      Min_ATR_Percentile = {optimized_params['min_atr_pct']};
input int      Max_ATR_Percentile = {optimized_params['max_atr_pct']};

// Regime Filter
input double   SMA_Slope_Threshold = {optimized_params['sma_slope_threshold']};

// Consecutive Loss Filter
input int      ConsecLoss_Limit = {optimized_params['consec_limit']};
input int      ConsecLoss_Cooldown_Bars = {optimized_params['consec_cooldown']};

// Risk Management
input double   Risk_Per_Trade = {optimized_params['risk_per_trade']};
""")

    finally:
        mt5.shutdown()
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)


if __name__ == "__main__":
    main()
