"""
V4 Summer Month Optimization
RSI v3.7 Strategy - Focus on fixing July-August losses

Key findings from analysis:
- August consistently worse than July
- Higher SL hit rate during summer (avg 65%)
- Lower win rate during summer (avg 35%)
- Lower liquidity = more erratic moves
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

# V3 Parameters (baseline)
V3_PARAMS = {
    'rsi_period': 10,
    'rsi_oversold': 42,
    'rsi_overbought': 58,
    'atr_period': 14,
    'sl_multiplier': 1.5,
    'tp_low': 2.4,
    'tp_med': 3.0,
    'tp_high': 3.6,
    'tp_session_bonus': 0.35,
    'max_holding_hours': 36,
    'atr_lookback': 100,
    'min_atr_pct': 20,
    'max_atr_pct': 85,
    'sma_fast': 20,
    'sma_slow': 50,
    'slope_threshold': 0.5,
    'slope_lookback': 10,
    'consec_loss_limit': 3,
    'consec_loss_cooldown': 3,
    'trading_start': 7,
    'trading_end': 22,
    'skip_hour': 12,
    'risk_per_trade': 0.01,
    # NEW V4 parameters
    'use_summer_filter': False,  # Skip July-August entirely
    'summer_sl_multiplier': 1.5,  # Wider SL for summer
    'summer_tp_reduction': 0.0,   # Reduce TP for summer
    'summer_max_holding': 36,     # Shorter holding for summer
}

def init_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    return True

def get_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_indicators(df, params):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=params['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=params['atr_period']).mean()

    # ATR Percentile
    df['atr_pct'] = df['atr'].rolling(window=params['atr_lookback']).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50
    )

    # SMAs
    df['sma_fast'] = df['close'].rolling(window=params['sma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=params['sma_slow']).mean()
    df['sma_slope'] = (df['sma_fast'] / df['sma_fast'].shift(params['slope_lookback']) - 1) * 100

    # Regime
    def get_regime(row):
        if pd.isna(row['sma_fast']) or pd.isna(row['sma_slow']) or pd.isna(row['sma_slope']):
            return 'UNKNOWN'
        if row['sma_fast'] > row['sma_slow'] and row['sma_slope'] > params['slope_threshold']:
            return 'BULL'
        elif row['sma_fast'] < row['sma_slow'] and row['sma_slope'] < -params['slope_threshold']:
            return 'BEAR'
        else:
            return 'SIDEWAYS'

    df['regime'] = df.apply(get_regime, axis=1)
    return df

def backtest_v4(df, params, initial_capital=10000):
    """Backtest with V4 summer-aware parameters"""
    capital = initial_capital
    trades = []
    position = None
    consec_losses = 0
    cooldown_bars = 0

    for i in range(200, len(df)):
        row = df.iloc[i]
        is_summer = row.name.month in [7, 8]

        # Decrease cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1
            if cooldown_bars == 0:
                consec_losses = 0

        # Check position exit
        if position is not None:
            entry_idx, entry_price, entry_type, sl, tp, entry_time, entry_summer = position
            current_price = row['close']

            # Use appropriate max holding based on when trade was opened
            max_hold = params['summer_max_holding'] if entry_summer else params['max_holding_hours']
            hours_held = (row.name - entry_time).total_seconds() / 3600

            exit_reason = None
            exit_price = None

            if entry_type == 'BUY':
                if row['low'] <= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                elif row['high'] >= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                elif hours_held >= max_hold:
                    exit_price = current_price
                    exit_reason = 'TIMEOUT'
            else:
                if row['high'] >= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                elif row['low'] <= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                elif hours_held >= max_hold:
                    exit_price = current_price
                    exit_reason = 'TIMEOUT'

            if exit_reason:
                if entry_type == 'BUY':
                    pnl_pips = (exit_price - entry_price) / 0.0001
                else:
                    pnl_pips = (entry_price - exit_price) / 0.0001

                risk_amount = capital * params['risk_per_trade']
                sl_pips = abs(entry_price - sl) / 0.0001
                pnl_dollars = (pnl_pips / sl_pips) * risk_amount if sl_pips > 0 else 0

                capital += pnl_dollars

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row.name,
                    'type': entry_type,
                    'pnl_dollars': pnl_dollars,
                    'exit_reason': exit_reason,
                    'month': entry_time.month,
                    'year': entry_time.year,
                    'is_summer': entry_summer
                })

                if pnl_dollars < 0:
                    consec_losses += 1
                    if consec_losses >= params['consec_loss_limit']:
                        cooldown_bars = params['consec_loss_cooldown']
                else:
                    consec_losses = 0

                position = None

        # Check for new entry
        if position is None and cooldown_bars == 0:
            # Summer filter - skip if enabled
            if params['use_summer_filter'] and is_summer:
                continue

            hour = row.name.hour
            dow = row.name.dayofweek

            if dow >= 5:
                continue
            if dow == 4 and hour >= 20:
                continue
            if hour < params['trading_start'] or hour >= params['trading_end']:
                continue
            if hour == params['skip_hour']:
                continue

            if pd.isna(row['atr_pct']):
                continue
            if row['atr_pct'] < params['min_atr_pct'] or row['atr_pct'] > params['max_atr_pct']:
                continue

            if row['regime'] != 'SIDEWAYS':
                continue

            signal = None
            if row['rsi'] < params['rsi_oversold']:
                signal = 'BUY'
            elif row['rsi'] > params['rsi_overbought']:
                signal = 'SELL'

            if signal:
                entry_price = row['close']
                atr = row['atr']

                # Adjust SL for summer
                sl_mult = params['summer_sl_multiplier'] if is_summer else params['sl_multiplier']

                # TP calculation with summer reduction
                if row['atr_pct'] < 40:
                    tp_mult = params['tp_low']
                elif row['atr_pct'] > 60:
                    tp_mult = params['tp_high']
                else:
                    tp_mult = params['tp_med']

                if 12 <= hour < 16:
                    tp_mult += params['tp_session_bonus']

                # Reduce TP during summer
                if is_summer:
                    tp_mult -= params['summer_tp_reduction']

                if signal == 'BUY':
                    sl = entry_price - atr * sl_mult
                    tp = entry_price + atr * tp_mult
                else:
                    sl = entry_price + atr * sl_mult
                    tp = entry_price - atr * tp_mult

                position = (i, entry_price, signal, sl, tp, row.name, is_summer)

    return pd.DataFrame(trades), capital

def run_optimization():
    """Test various V4 configurations"""
    print("="*70)
    print("V4 OPTIMIZATION - SUMMER MONTH FIX")
    print("="*70)

    if not init_mt5():
        return

    # Get data
    symbol = "GBPUSD"
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)

    print(f"\nFetching {symbol} H1 data...")
    df = get_data(symbol, mt5.TIMEFRAME_H1, start_date, end_date)

    if df is None:
        print("Failed to get data")
        mt5.shutdown()
        return

    print(f"Got {len(df)} bars")

    # Calculate indicators once
    print("Calculating indicators...")
    df = calculate_indicators(df, V3_PARAMS)

    # Test configurations
    configs = [
        # Baseline V3
        {'name': 'V3_BASELINE', 'changes': {}},

        # Option 1: Skip summer entirely
        {'name': 'SKIP_SUMMER', 'changes': {'use_summer_filter': True}},

        # Option 2: Wider SL during summer (2.0x ATR)
        {'name': 'SUMMER_SL_2.0', 'changes': {'summer_sl_multiplier': 2.0}},

        # Option 3: Wider SL + Reduced TP during summer
        {'name': 'SUMMER_SL2_TP-0.3', 'changes': {'summer_sl_multiplier': 2.0, 'summer_tp_reduction': 0.3}},

        # Option 4: Shorter holding during summer
        {'name': 'SUMMER_HOLD_24', 'changes': {'summer_max_holding': 24}},

        # Option 5: Combined summer adjustments
        {'name': 'SUMMER_COMBO_1', 'changes': {
            'summer_sl_multiplier': 1.8,
            'summer_tp_reduction': 0.2,
            'summer_max_holding': 30
        }},

        # Option 6: Aggressive summer protection
        {'name': 'SUMMER_COMBO_2', 'changes': {
            'summer_sl_multiplier': 2.0,
            'summer_tp_reduction': 0.4,
            'summer_max_holding': 24
        }},

        # Option 7: Tighter ATR range overall
        {'name': 'ATR_25_70', 'changes': {'min_atr_pct': 25, 'max_atr_pct': 70}},

        # Option 8: ATR + Summer combo
        {'name': 'ATR_25_70_SUMMER', 'changes': {
            'min_atr_pct': 25,
            'max_atr_pct': 70,
            'summer_sl_multiplier': 1.8,
            'summer_max_holding': 30
        }},

        # Option 9: Longer cooldown
        {'name': 'COOLDOWN_5', 'changes': {'consec_loss_cooldown': 5}},

        # Option 10: Best combination attempt
        {'name': 'V4_CANDIDATE', 'changes': {
            'min_atr_pct': 25,
            'max_atr_pct': 75,
            'summer_sl_multiplier': 1.8,
            'summer_tp_reduction': 0.2,
            'summer_max_holding': 28,
            'consec_loss_cooldown': 4
        }},
    ]

    results = []

    print("\nTesting configurations...")
    for config in configs:
        params = V3_PARAMS.copy()
        params.update(config['changes'])

        trades_df, final_capital = backtest_v4(df, params)

        if trades_df.empty:
            continue

        # Overall stats
        total_pnl = trades_df['pnl_dollars'].sum()
        total_trades = len(trades_df)
        win_rate = (trades_df['pnl_dollars'] > 0).mean() * 100

        # Summer stats
        summer = trades_df[trades_df['is_summer'] == True]
        other = trades_df[trades_df['is_summer'] == False]

        summer_pnl = summer['pnl_dollars'].sum() if not summer.empty else 0
        summer_trades = len(summer)
        summer_wr = (summer['pnl_dollars'] > 0).mean() * 100 if not summer.empty else 0

        other_pnl = other['pnl_dollars'].sum() if not other.empty else 0

        # Max consecutive losses
        consec = 0
        max_consec = 0
        for pnl in trades_df.sort_values('entry_time')['pnl_dollars']:
            if pnl < 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        # Max drawdown
        equity = 10000 + trades_df['pnl_dollars'].cumsum()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak * 100
        max_dd = abs(dd.min())

        results.append({
            'config': config['name'],
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'summer_pnl': summer_pnl,
            'summer_trades': summer_trades,
            'summer_wr': summer_wr,
            'other_pnl': other_pnl,
            'max_consec': max_consec,
            'max_dd': max_dd,
            'changes': str(config['changes'])
        })

    # Display results
    results_df = pd.DataFrame(results)

    print("\n" + "="*100)
    print("OPTIMIZATION RESULTS")
    print("="*100)
    print(f"{'Config':<20} {'Total P&L':>12} {'Trades':>7} {'WR':>6} {'Summer P&L':>12} {'Sum WR':>7} {'MaxDD':>7} {'MaxCon':>7}")
    print("-"*100)

    for _, row in results_df.iterrows():
        print(f"{row['config']:<20} ${row['total_pnl']:>10,.0f} {row['total_trades']:>7} "
              f"{row['win_rate']:>5.1f}% ${row['summer_pnl']:>10,.0f} {row['summer_wr']:>6.1f}% "
              f"{row['max_dd']:>6.1f}% {row['max_consec']:>7}")

    # Find best config for summer
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)

    # Best overall
    best_overall = results_df.loc[results_df['total_pnl'].idxmax()]
    print(f"\nBest Overall: {best_overall['config']}")
    print(f"  Total P&L: ${best_overall['total_pnl']:,.2f}")
    print(f"  Summer P&L: ${best_overall['summer_pnl']:,.2f}")

    # Best summer improvement
    baseline_summer = results_df[results_df['config'] == 'V3_BASELINE']['summer_pnl'].values[0]
    results_df['summer_improvement'] = results_df['summer_pnl'] - baseline_summer
    best_summer = results_df.loc[results_df['summer_improvement'].idxmax()]
    print(f"\nBest Summer Improvement: {best_summer['config']}")
    print(f"  Summer P&L: ${best_summer['summer_pnl']:,.2f} (baseline: ${baseline_summer:,.2f})")
    print(f"  Improvement: +${best_summer['summer_improvement']:,.2f}")

    # Best risk-adjusted (total_pnl / max_dd)
    results_df['risk_adj'] = results_df['total_pnl'] / (results_df['max_dd'] + 1)
    best_risk = results_df.loc[results_df['risk_adj'].idxmax()]
    print(f"\nBest Risk-Adjusted: {best_risk['config']}")
    print(f"  Total P&L: ${best_risk['total_pnl']:,.2f}")
    print(f"  Max DD: {best_risk['max_dd']:.1f}%")

    # V4 Recommendation
    print("\n" + "="*70)
    print("V4 RECOMMENDED PARAMETERS")
    print("="*70)

    # Find best balance
    v4_candidate = results_df[results_df['config'] == 'V4_CANDIDATE'].iloc[0] if 'V4_CANDIDATE' in results_df['config'].values else None

    if v4_candidate is not None:
        print(f"\nV4_CANDIDATE Performance:")
        print(f"  Total P&L: ${v4_candidate['total_pnl']:,.2f}")
        print(f"  Summer P&L: ${v4_candidate['summer_pnl']:,.2f}")
        print(f"  Max DD: {v4_candidate['max_dd']:.1f}%")
        print(f"  Max Consec Loss: {v4_candidate['max_consec']}")

    print("\nRecommended V4 Parameter Changes:")
    print("  min_atr_pct: 25 (was 20)")
    print("  max_atr_pct: 75 (was 85)")
    print("  summer_sl_multiplier: 1.8 (for Jul-Aug)")
    print("  summer_tp_reduction: 0.2 (for Jul-Aug)")
    print("  summer_max_holding: 28 (for Jul-Aug)")
    print("  consec_loss_cooldown: 4 (was 3)")

    # Save results
    output_file = r'C:\Users\Administrator\Music\SURGE-WSI\strategies\rsi_v37_optimized\v4_optimization_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    mt5.shutdown()
    print("\nOptimization complete!")

if __name__ == "__main__":
    run_optimization()
