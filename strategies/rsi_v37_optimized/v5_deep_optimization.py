"""
V5 Deep Optimization - Based on V4 Analysis

Key Issues Found in V4:
1. July STILL worst month (-$6,552, 24.3% WR) despite summer adjustments
2. Hour 10 and 15 are terrible exit times
3. Thursday is the only losing day
4. 6 losing months total (Mar, Apr, Jun, Jul, Sep, Oct)

V5 Optimization Ideas:
1. Enhanced July protection (skip July or stricter filters)
2. Avoid entering trades that would likely exit at bad hours
3. Thursday adjustment
4. Longer cooldown for consecutive losses
5. Skip "problem months" entirely (optional)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

# V4 Parameters (baseline)
V4_PARAMS = {
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
    'summer_max_holding': 24,
    # V5 new params
    'skip_july': False,
    'skip_thursday': False,
    'skip_problem_months': False,  # Mar, Apr, Jun, Jul, Sep, Oct
    'july_max_holding': 24,
    'thursday_max_holding': 24,
    'avoid_entry_hours': [],  # Hours to avoid entry
    'extended_cooldown': 3,
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

def backtest_v5(df, params, initial_capital=10000):
    """Backtest with V5 enhancements"""
    capital = initial_capital
    trades = []
    position = None
    consec_losses = 0
    cooldown_bars = 0

    problem_months = [3, 4, 6, 7, 9, 10]  # Mar, Apr, Jun, Jul, Sep, Oct

    for i in range(200, len(df)):
        row = df.iloc[i]
        month = row.name.month
        dow = row.name.dayofweek
        hour = row.name.hour
        is_summer = month in [7, 8]
        is_july = month == 7
        is_thursday = dow == 3
        is_problem_month = month in problem_months

        # Decrease cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1
            if cooldown_bars == 0:
                consec_losses = 0

        # Check position exit
        if position is not None:
            entry_idx, entry_price, entry_type, sl, tp, entry_time, pos_month, pos_dow = position
            current_price = row['close']

            # Determine max holding based on conditions
            max_hold = params['max_holding_hours']
            if pos_month == 7:  # July
                max_hold = params['july_max_holding']
            elif pos_month in [7, 8]:  # Summer
                max_hold = params['summer_max_holding']
            elif pos_dow == 3:  # Thursday
                max_hold = params.get('thursday_max_holding', params['max_holding_hours'])

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
                    'exit_hour': row.name.hour,
                    'month': entry_time.month,
                    'year': entry_time.year,
                    'dow': entry_time.dayofweek
                })

                if pnl_dollars < 0:
                    consec_losses += 1
                    if consec_losses >= params['consec_loss_limit']:
                        cooldown_bars = params.get('extended_cooldown', params['consec_loss_cooldown'])
                else:
                    consec_losses = 0

                position = None

        # Check for new entry
        if position is None and cooldown_bars == 0:
            # V5 Filters
            if params['skip_july'] and is_july:
                continue
            if params['skip_thursday'] and is_thursday:
                continue
            if params['skip_problem_months'] and is_problem_month:
                continue
            if hour in params.get('avoid_entry_hours', []):
                continue

            # Weekend filter
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

                sl_mult = params['sl_multiplier']

                if row['atr_pct'] < 40:
                    tp_mult = params['tp_low']
                elif row['atr_pct'] > 60:
                    tp_mult = params['tp_high']
                else:
                    tp_mult = params['tp_med']

                if 12 <= hour < 16:
                    tp_mult += params['tp_session_bonus']

                if signal == 'BUY':
                    sl = entry_price - atr * sl_mult
                    tp = entry_price + atr * tp_mult
                else:
                    sl = entry_price + atr * sl_mult
                    tp = entry_price - atr * tp_mult

                position = (i, entry_price, signal, sl, tp, row.name, month, dow)

    return pd.DataFrame(trades), capital

def run_v5_optimization():
    """Test various V5 configurations"""
    print("="*80)
    print("V5 DEEP OPTIMIZATION")
    print("="*80)

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
    df = calculate_indicators(df, V4_PARAMS)

    # Test configurations
    configs = [
        # Baseline V4
        {'name': 'V4_BASELINE', 'changes': {}},

        # July specific fixes
        {'name': 'SKIP_JULY', 'changes': {'skip_july': True}},
        {'name': 'JULY_HOLD_18', 'changes': {'july_max_holding': 18}},
        {'name': 'JULY_HOLD_12', 'changes': {'july_max_holding': 12}},

        # Thursday fixes
        {'name': 'SKIP_THURSDAY', 'changes': {'skip_thursday': True}},
        {'name': 'THU_HOLD_24', 'changes': {'thursday_max_holding': 24}},

        # Problem months
        {'name': 'SKIP_PROBLEMS', 'changes': {'skip_problem_months': True}},

        # Avoid bad entry hours (that lead to bad exit times)
        {'name': 'AVOID_ENTRY_9_14', 'changes': {'avoid_entry_hours': [9, 14]}},
        {'name': 'AVOID_ENTRY_9', 'changes': {'avoid_entry_hours': [9]}},

        # Extended cooldown
        {'name': 'COOLDOWN_4', 'changes': {'extended_cooldown': 4}},
        {'name': 'COOLDOWN_5', 'changes': {'extended_cooldown': 5}},
        {'name': 'COOLDOWN_6', 'changes': {'extended_cooldown': 6}},

        # Combined approaches
        {'name': 'V5_COMBO_1', 'changes': {
            'skip_july': True,
            'skip_thursday': True,
            'extended_cooldown': 4
        }},
        {'name': 'V5_COMBO_2', 'changes': {
            'july_max_holding': 12,
            'thursday_max_holding': 24,
            'extended_cooldown': 4
        }},
        {'name': 'V5_COMBO_3', 'changes': {
            'skip_july': True,
            'extended_cooldown': 5,
            'avoid_entry_hours': [9]
        }},
        {'name': 'V5_CANDIDATE', 'changes': {
            'july_max_holding': 12,
            'thursday_max_holding': 24,
            'extended_cooldown': 5,
        }},
    ]

    results = []

    print("\nTesting configurations...")
    for config in configs:
        params = V4_PARAMS.copy()
        params.update(config['changes'])

        trades_df, final_capital = backtest_v5(df, params)

        if trades_df.empty:
            continue

        # Overall stats
        total_pnl = trades_df['pnl_dollars'].sum()
        total_trades = len(trades_df)
        win_rate = (trades_df['pnl_dollars'] > 0).mean() * 100

        # July stats
        july = trades_df[trades_df['month'] == 7]
        july_pnl = july['pnl_dollars'].sum() if not july.empty else 0
        july_trades = len(july)

        # Thursday stats
        thu = trades_df[trades_df['dow'] == 3]
        thu_pnl = thu['pnl_dollars'].sum() if not thu.empty else 0

        # Losing months count
        monthly = trades_df.groupby('month')['pnl_dollars'].sum()
        losing_months = (monthly < 0).sum()

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
            'july_pnl': july_pnl,
            'july_trades': july_trades,
            'thu_pnl': thu_pnl,
            'losing_months': losing_months,
            'max_consec': max_consec,
            'max_dd': max_dd,
        })

    # Display results
    results_df = pd.DataFrame(results)

    print("\n" + "="*110)
    print("V5 OPTIMIZATION RESULTS")
    print("="*110)
    print(f"{'Config':<18} {'Total P&L':>12} {'Trades':>7} {'WR':>6} {'July P&L':>10} {'Thu P&L':>10} {'LossMo':>7} {'MaxDD':>7} {'MaxCon':>7}")
    print("-"*110)

    for _, row in results_df.iterrows():
        print(f"{row['config']:<18} ${row['total_pnl']:>10,.0f} {row['total_trades']:>7} "
              f"{row['win_rate']:>5.1f}% ${row['july_pnl']:>8,.0f} ${row['thu_pnl']:>8,.0f} "
              f"{row['losing_months']:>7} {row['max_dd']:>6.1f}% {row['max_consec']:>7}")

    # Find best configs
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)

    # Best overall profit
    best_profit = results_df.loc[results_df['total_pnl'].idxmax()]
    print(f"\nBest Total P&L: {best_profit['config']}")
    print(f"  P&L: ${best_profit['total_pnl']:,.2f}, Max DD: {best_profit['max_dd']:.1f}%")

    # Best July improvement
    baseline_july = results_df[results_df['config'] == 'V4_BASELINE']['july_pnl'].values[0]
    results_df['july_improvement'] = results_df['july_pnl'] - baseline_july
    best_july = results_df.loc[results_df['july_improvement'].idxmax()]
    print(f"\nBest July Fix: {best_july['config']}")
    print(f"  July P&L: ${best_july['july_pnl']:,.2f} (was ${baseline_july:,.2f})")

    # Best risk-adjusted
    results_df['risk_score'] = results_df['total_pnl'] / (results_df['max_dd'] + 1)
    best_risk = results_df.loc[results_df['risk_score'].idxmax()]
    print(f"\nBest Risk-Adjusted: {best_risk['config']}")
    print(f"  P&L: ${best_risk['total_pnl']:,.2f}, Max DD: {best_risk['max_dd']:.1f}%")

    # Fewest losing months
    best_months = results_df.loc[results_df['losing_months'].idxmin()]
    print(f"\nFewest Losing Months: {best_months['config']}")
    print(f"  Losing Months: {best_months['losing_months']}, P&L: ${best_months['total_pnl']:,.2f}")

    # V5 Recommendation
    print("\n" + "="*80)
    print("V5 RECOMMENDED PARAMETERS")
    print("="*80)

    v5_candidate = results_df[results_df['config'] == 'V5_CANDIDATE']
    if not v5_candidate.empty:
        v5 = v5_candidate.iloc[0]
        print(f"\nV5_CANDIDATE Performance:")
        print(f"  Total P&L: ${v5['total_pnl']:,.2f}")
        print(f"  July P&L: ${v5['july_pnl']:,.2f}")
        print(f"  Max DD: {v5['max_dd']:.1f}%")
        print(f"  Max Consec Loss: {v5['max_consec']}")
        print(f"  Losing Months: {v5['losing_months']}")

    print("\nRecommended V5 Changes from V4:")
    print("  july_max_holding: 12h (aggressive July protection)")
    print("  thursday_max_holding: 24h (reduce Thursday exposure)")
    print("  consec_loss_cooldown: 5 bars (longer recovery)")

    # Save results
    output_file = r'C:\Users\Administrator\Music\SURGE-WSI\strategies\rsi_v37_optimized\v5_optimization_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    mt5.shutdown()
    print("\nV5 Optimization complete!")

if __name__ == "__main__":
    run_v5_optimization()
