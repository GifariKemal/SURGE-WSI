"""
Deep Analysis: Why July-August Are Losing Months
RSI v3.7 Strategy - GBPUSD H1

Analyzes:
1. Monthly performance breakdown
2. July-August specific patterns
3. Market conditions during those months
4. Parameter sensitivity for summer months
5. Recommendations for v4
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STRATEGY PARAMETERS (V3 Optimized)
# ============================================================
PARAMS = {
    'rsi_period': 10,
    'rsi_oversold': 42,
    'rsi_overbought': 58,
    'atr_period': 14,
    'sl_multiplier': 1.5,
    'tp_low': 2.4,
    'tp_med': 3.0,
    'tp_high': 3.6,
    'tp_session_bonus': 0.35,
    'max_holding_hours': 36,  # V3 optimized
    'atr_lookback': 100,
    'min_atr_pct': 20,
    'max_atr_pct': 85,  # V3 optimized
    'sma_fast': 20,
    'sma_slow': 50,
    'slope_threshold': 0.5,
    'slope_lookback': 10,
    'consec_loss_limit': 3,
    'consec_loss_cooldown': 3,  # V3 optimized
    'trading_start': 7,
    'trading_end': 22,
    'skip_hour': 12,
    'risk_per_trade': 0.01,
}

def init_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    return True

def get_data(symbol, timeframe, start_date, end_date):
    """Get OHLC data from MT5"""
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_indicators(df, params):
    """Calculate all strategy indicators"""
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

    # SMA Slope
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

    # Bollinger Bands (for additional analysis)
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_width_pct'] = df['bb_width'].rolling(window=100).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50
    )

    return df

def backtest_strategy(df, params, initial_capital=10000):
    """Run backtest with detailed trade tracking"""
    capital = initial_capital
    trades = []
    position = None
    consec_losses = 0
    cooldown_bars = 0

    for i in range(200, len(df)):  # Skip warmup
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # Decrease cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1
            if cooldown_bars == 0:
                consec_losses = 0

        # Check position exit
        if position is not None:
            entry_idx, entry_price, entry_type, sl, tp, entry_time = position
            current_price = row['close']
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
                elif hours_held >= params['max_holding_hours']:
                    exit_price = current_price
                    exit_reason = 'TIMEOUT'
            else:  # SELL
                if row['high'] >= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                elif row['low'] <= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                elif hours_held >= params['max_holding_hours']:
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
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'sl': sl,
                    'tp': tp,
                    'pnl_pips': pnl_pips,
                    'pnl_dollars': pnl_dollars,
                    'exit_reason': exit_reason,
                    'hours_held': hours_held,
                    'regime': df.iloc[entry_idx]['regime'],
                    'atr_pct': df.iloc[entry_idx]['atr_pct'],
                    'rsi': df.iloc[entry_idx]['rsi'],
                    'bb_width_pct': df.iloc[entry_idx]['bb_width_pct'],
                    'month': entry_time.month,
                    'year': entry_time.year,
                    'capital_after': capital
                })

                # Update consecutive losses
                if pnl_dollars < 0:
                    consec_losses += 1
                    if consec_losses >= params['consec_loss_limit']:
                        cooldown_bars = params['consec_loss_cooldown']
                else:
                    consec_losses = 0

                position = None

        # Check for new entry
        if position is None and cooldown_bars == 0:
            hour = row.name.hour
            dow = row.name.dayofweek

            # Time filters
            if dow >= 5:  # Weekend
                continue
            if dow == 4 and hour >= 20:  # Friday evening
                continue
            if hour < params['trading_start'] or hour >= params['trading_end']:
                continue
            if hour == params['skip_hour']:
                continue

            # ATR filter
            if pd.isna(row['atr_pct']):
                continue
            if row['atr_pct'] < params['min_atr_pct'] or row['atr_pct'] > params['max_atr_pct']:
                continue

            # Regime filter
            if row['regime'] != 'SIDEWAYS':
                continue

            # RSI signal
            signal = None
            if row['rsi'] < params['rsi_oversold']:
                signal = 'BUY'
            elif row['rsi'] > params['rsi_overbought']:
                signal = 'SELL'

            if signal:
                entry_price = row['close']
                atr = row['atr']

                # TP multiplier
                if row['atr_pct'] < 40:
                    tp_mult = params['tp_low']
                elif row['atr_pct'] > 60:
                    tp_mult = params['tp_high']
                else:
                    tp_mult = params['tp_med']

                if 12 <= hour < 16:
                    tp_mult += params['tp_session_bonus']

                if signal == 'BUY':
                    sl = entry_price - atr * params['sl_multiplier']
                    tp = entry_price + atr * tp_mult
                else:
                    sl = entry_price + atr * params['sl_multiplier']
                    tp = entry_price - atr * tp_mult

                position = (i, entry_price, signal, sl, tp, row.name)

    return pd.DataFrame(trades), capital

def analyze_monthly_performance(trades_df):
    """Analyze performance by month"""
    if trades_df.empty:
        return None

    monthly = trades_df.groupby(['year', 'month']).agg({
        'pnl_dollars': ['sum', 'count', 'mean'],
        'exit_reason': lambda x: (x == 'SL').sum(),
        'hours_held': 'mean',
        'atr_pct': 'mean',
        'bb_width_pct': 'mean'
    }).round(2)

    monthly.columns = ['total_pnl', 'trades', 'avg_pnl', 'sl_hits', 'avg_hours', 'avg_atr_pct', 'avg_bb_width']
    monthly['win_rate'] = ((monthly['trades'] - monthly['sl_hits']) / monthly['trades'] * 100).round(1)
    monthly['sl_rate'] = (monthly['sl_hits'] / monthly['trades'] * 100).round(1)

    return monthly

def analyze_summer_months(trades_df, df_prices):
    """Deep analysis of July-August specifically"""
    summer_trades = trades_df[trades_df['month'].isin([7, 8])]
    other_trades = trades_df[~trades_df['month'].isin([7, 8])]

    print("\n" + "="*70)
    print("SUMMER MONTHS (JULY-AUGUST) DEEP ANALYSIS")
    print("="*70)

    if summer_trades.empty:
        print("No summer trades found")
        return

    # Basic comparison
    print("\n--- Performance Comparison ---")
    print(f"{'Metric':<30} {'Jul-Aug':<15} {'Other Months':<15}")
    print("-"*60)
    print(f"{'Total Trades':<30} {len(summer_trades):<15} {len(other_trades):<15}")
    print(f"{'Total P&L':<30} ${summer_trades['pnl_dollars'].sum():,.2f}     ${other_trades['pnl_dollars'].sum():,.2f}")
    print(f"{'Avg P&L/Trade':<30} ${summer_trades['pnl_dollars'].mean():,.2f}       ${other_trades['pnl_dollars'].mean():,.2f}")
    print(f"{'Win Rate':<30} {(summer_trades['pnl_dollars']>0).mean()*100:.1f}%          {(other_trades['pnl_dollars']>0).mean()*100:.1f}%")
    print(f"{'SL Hit Rate':<30} {(summer_trades['exit_reason']=='SL').mean()*100:.1f}%          {(other_trades['exit_reason']=='SL').mean()*100:.1f}%")
    print(f"{'Avg ATR Percentile':<30} {summer_trades['atr_pct'].mean():.1f}            {other_trades['atr_pct'].mean():.1f}")
    print(f"{'Avg BB Width Pct':<30} {summer_trades['bb_width_pct'].mean():.1f}            {other_trades['bb_width_pct'].mean():.1f}")
    print(f"{'Avg Hours Held':<30} {summer_trades['hours_held'].mean():.1f}            {other_trades['hours_held'].mean():.1f}")

    # Analyze by trade direction
    print("\n--- By Direction (Summer) ---")
    for direction in ['BUY', 'SELL']:
        dir_trades = summer_trades[summer_trades['type'] == direction]
        if not dir_trades.empty:
            print(f"{direction}: {len(dir_trades)} trades, P&L: ${dir_trades['pnl_dollars'].sum():,.2f}, "
                  f"WR: {(dir_trades['pnl_dollars']>0).mean()*100:.1f}%")

    # Analyze by exit reason
    print("\n--- By Exit Reason (Summer) ---")
    for reason in ['SL', 'TP', 'TIMEOUT']:
        reason_trades = summer_trades[summer_trades['exit_reason'] == reason]
        if not reason_trades.empty:
            print(f"{reason}: {len(reason_trades)} trades ({len(reason_trades)/len(summer_trades)*100:.1f}%), "
                  f"P&L: ${reason_trades['pnl_dollars'].sum():,.2f}")

    # Market condition analysis
    print("\n--- Market Conditions During Summer ---")
    summer_prices = df_prices[df_prices.index.month.isin([7, 8])]
    other_prices = df_prices[~df_prices.index.month.isin([7, 8])]

    print(f"Avg Daily Range (ATR): Summer={summer_prices['atr'].mean()*10000:.1f} pips, Other={other_prices['atr'].mean()*10000:.1f} pips")
    print(f"Avg BB Width: Summer={summer_prices['bb_width'].mean():.3f}%, Other={other_prices['bb_width'].mean():.3f}%")
    print(f"SIDEWAYS Regime %: Summer={((summer_prices['regime']=='SIDEWAYS').sum()/len(summer_prices)*100):.1f}%, "
          f"Other={((other_prices['regime']=='SIDEWAYS').sum()/len(other_prices)*100):.1f}%")

    # Consecutive losses analysis
    print("\n--- Consecutive Losses Analysis (Summer) ---")
    summer_trades_sorted = summer_trades.sort_values('entry_time')
    consec = 0
    max_consec = 0
    consec_streaks = []

    for _, trade in summer_trades_sorted.iterrows():
        if trade['pnl_dollars'] < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            if consec > 0:
                consec_streaks.append(consec)
            consec = 0

    if consec > 0:
        consec_streaks.append(consec)

    print(f"Max Consecutive Losses: {max_consec}")
    print(f"Avg Loss Streak: {np.mean(consec_streaks) if consec_streaks else 0:.1f}")

    return summer_trades, other_trades

def test_parameter_variations(df, base_params, initial_capital=10000):
    """Test various parameter combinations to fix summer months"""
    results = []

    # Parameters to test for summer fix
    test_configs = [
        # Baseline (V3)
        {'name': 'V3_BASELINE', 'changes': {}},

        # Tighter ATR range (less volatile periods)
        {'name': 'ATR_25_75', 'changes': {'min_atr_pct': 25, 'max_atr_pct': 75}},
        {'name': 'ATR_30_70', 'changes': {'min_atr_pct': 30, 'max_atr_pct': 70}},
        {'name': 'ATR_25_70', 'changes': {'min_atr_pct': 25, 'max_atr_pct': 70}},

        # Shorter holding time
        {'name': 'HOLD_24H', 'changes': {'max_holding_hours': 24}},
        {'name': 'HOLD_30H', 'changes': {'max_holding_hours': 30}},

        # Stricter RSI
        {'name': 'RSI_40_60', 'changes': {'rsi_oversold': 40, 'rsi_overbought': 60}},
        {'name': 'RSI_38_62', 'changes': {'rsi_oversold': 38, 'rsi_overbought': 62}},

        # Longer cooldown
        {'name': 'COOLDOWN_4', 'changes': {'consec_loss_cooldown': 4}},
        {'name': 'COOLDOWN_5', 'changes': {'consec_loss_cooldown': 5}},

        # Lower TP (take profits earlier)
        {'name': 'TP_LOWER', 'changes': {'tp_low': 2.0, 'tp_med': 2.5, 'tp_high': 3.0}},

        # Higher SL (wider stop)
        {'name': 'SL_1.8', 'changes': {'sl_multiplier': 1.8}},
        {'name': 'SL_2.0', 'changes': {'sl_multiplier': 2.0}},

        # Combinations
        {'name': 'COMBO_1', 'changes': {'min_atr_pct': 25, 'max_atr_pct': 75, 'max_holding_hours': 30}},
        {'name': 'COMBO_2', 'changes': {'min_atr_pct': 25, 'max_atr_pct': 70, 'consec_loss_cooldown': 4}},
        {'name': 'COMBO_3', 'changes': {'rsi_oversold': 40, 'rsi_overbought': 60, 'max_holding_hours': 30}},
        {'name': 'COMBO_4', 'changes': {'min_atr_pct': 25, 'max_atr_pct': 70, 'max_holding_hours': 30, 'consec_loss_cooldown': 4}},
        {'name': 'COMBO_5', 'changes': {'sl_multiplier': 1.8, 'tp_low': 2.2, 'tp_med': 2.8, 'tp_high': 3.4}},
    ]

    print("\n" + "="*70)
    print("PARAMETER OPTIMIZATION FOR SUMMER MONTHS")
    print("="*70)

    for config in test_configs:
        params = base_params.copy()
        params.update(config['changes'])

        trades_df, final_capital = backtest_strategy(df, params, initial_capital)

        if trades_df.empty:
            continue

        # Overall stats
        total_pnl = trades_df['pnl_dollars'].sum()
        win_rate = (trades_df['pnl_dollars'] > 0).mean() * 100

        # Summer stats
        summer = trades_df[trades_df['month'].isin([7, 8])]
        summer_pnl = summer['pnl_dollars'].sum() if not summer.empty else 0
        summer_trades = len(summer)
        summer_wr = (summer['pnl_dollars'] > 0).mean() * 100 if not summer.empty else 0

        # Max consecutive losses
        trades_sorted = trades_df.sort_values('entry_time')
        consec = 0
        max_consec = 0
        for _, trade in trades_sorted.iterrows():
            if trade['pnl_dollars'] < 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        # Max drawdown
        equity = initial_capital + trades_df['pnl_dollars'].cumsum()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak * 100
        max_dd = abs(dd.min())

        results.append({
            'config': config['name'],
            'total_pnl': total_pnl,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'summer_pnl': summer_pnl,
            'summer_trades': summer_trades,
            'summer_wr': summer_wr,
            'max_consec_loss': max_consec,
            'max_dd': max_dd,
            'changes': str(config['changes'])
        })

    results_df = pd.DataFrame(results)

    # Sort by summer P&L improvement
    results_df = results_df.sort_values('summer_pnl', ascending=False)

    print("\n--- Results Sorted by Summer P&L ---")
    print(f"{'Config':<15} {'Total P&L':>12} {'Summer P&L':>12} {'Summer WR':>10} {'Max DD':>8} {'MaxConsec':>10}")
    print("-"*75)

    for _, row in results_df.iterrows():
        print(f"{row['config']:<15} ${row['total_pnl']:>10,.0f} ${row['summer_pnl']:>10,.0f} "
              f"{row['summer_wr']:>9.1f}% {row['max_dd']:>7.1f}% {row['max_consec_loss']:>10}")

    return results_df

def find_optimal_v4_params(df, base_params, initial_capital=10000):
    """Grid search for optimal V4 parameters"""
    print("\n" + "="*70)
    print("V4 PARAMETER OPTIMIZATION - GRID SEARCH")
    print("="*70)

    best_score = -float('inf')
    best_params = None
    best_result = None
    all_results = []

    # Grid search ranges
    atr_min_range = [20, 25, 30]
    atr_max_range = [70, 75, 80]
    holding_range = [24, 30, 36]
    cooldown_range = [3, 4, 5]
    sl_range = [1.5, 1.8, 2.0]

    total_combos = len(atr_min_range) * len(atr_max_range) * len(holding_range) * len(cooldown_range) * len(sl_range)
    print(f"Testing {total_combos} combinations...")

    combo_count = 0
    for atr_min in atr_min_range:
        for atr_max in atr_max_range:
            if atr_min >= atr_max:
                continue
            for holding in holding_range:
                for cooldown in cooldown_range:
                    for sl in sl_range:
                        combo_count += 1

                        params = base_params.copy()
                        params['min_atr_pct'] = atr_min
                        params['max_atr_pct'] = atr_max
                        params['max_holding_hours'] = holding
                        params['consec_loss_cooldown'] = cooldown
                        params['sl_multiplier'] = sl

                        trades_df, final_capital = backtest_strategy(df, params, initial_capital)

                        if trades_df.empty or len(trades_df) < 50:
                            continue

                        # Calculate metrics
                        total_pnl = trades_df['pnl_dollars'].sum()
                        win_rate = (trades_df['pnl_dollars'] > 0).mean() * 100

                        # Summer metrics
                        summer = trades_df[trades_df['month'].isin([7, 8])]
                        summer_pnl = summer['pnl_dollars'].sum() if not summer.empty else 0

                        # Max DD
                        equity = initial_capital + trades_df['pnl_dollars'].cumsum()
                        peak = equity.expanding().max()
                        dd = (equity - peak) / peak * 100
                        max_dd = abs(dd.min())

                        # Max consecutive losses
                        consec = 0
                        max_consec = 0
                        for pnl in trades_df.sort_values('entry_time')['pnl_dollars']:
                            if pnl < 0:
                                consec += 1
                                max_consec = max(max_consec, consec)
                            else:
                                consec = 0

                        # Scoring function: Balance profit, summer performance, and risk
                        # Prioritize: total profit, summer not too negative, low max DD, low consec losses
                        score = (
                            total_pnl * 0.4 +
                            summer_pnl * 0.3 +  # Weight summer more
                            (100 - max_dd) * 100 * 0.2 +
                            (15 - max_consec) * 500 * 0.1
                        )

                        all_results.append({
                            'atr_min': atr_min,
                            'atr_max': atr_max,
                            'holding': holding,
                            'cooldown': cooldown,
                            'sl': sl,
                            'total_pnl': total_pnl,
                            'summer_pnl': summer_pnl,
                            'trades': len(trades_df),
                            'win_rate': win_rate,
                            'max_dd': max_dd,
                            'max_consec': max_consec,
                            'score': score
                        })

                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                            best_result = all_results[-1]

    print(f"\nTested {combo_count} valid combinations")

    # Show top 10 results
    results_df = pd.DataFrame(all_results).sort_values('score', ascending=False)

    print("\n--- TOP 10 PARAMETER COMBINATIONS ---")
    print(f"{'ATR':<10} {'Hold':>6} {'Cool':>6} {'SL':>5} {'Total P&L':>12} {'Summer':>10} {'MaxDD':>7} {'MaxCon':>7} {'Score':>10}")
    print("-"*85)

    for _, row in results_df.head(10).iterrows():
        print(f"{row['atr_min']}-{row['atr_max']:<6} {row['holding']:>6} {row['cooldown']:>6} {row['sl']:>5.1f} "
              f"${row['total_pnl']:>10,.0f} ${row['summer_pnl']:>8,.0f} {row['max_dd']:>6.1f}% {row['max_consec']:>7} {row['score']:>10,.0f}")

    return best_params, best_result, results_df

def main():
    print("="*70)
    print("RSI v3.7 DEEP ANALYSIS - JULY/AUGUST LOSING MONTHS")
    print("="*70)

    if not init_mt5():
        return

    # Get data for multiple years
    symbol = "GBPUSD"
    timeframe = mt5.TIMEFRAME_H1
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)

    print(f"\nFetching {symbol} H1 data from {start_date.date()} to {end_date.date()}...")
    df = get_data(symbol, timeframe, start_date, end_date)

    if df is None or df.empty:
        print("Failed to get data")
        mt5.shutdown()
        return

    print(f"Got {len(df)} bars")

    # Calculate indicators
    print("Calculating indicators...")
    df = calculate_indicators(df, PARAMS)

    # Run baseline backtest
    print("\nRunning V3 baseline backtest...")
    trades_df, final_capital = backtest_strategy(df, PARAMS)
    print(f"Total trades: {len(trades_df)}")
    print(f"Final capital: ${final_capital:,.2f}")
    print(f"Total P&L: ${trades_df['pnl_dollars'].sum():,.2f}")

    # Monthly analysis
    print("\n" + "="*70)
    print("MONTHLY PERFORMANCE BREAKDOWN")
    print("="*70)
    monthly = analyze_monthly_performance(trades_df)
    if monthly is not None:
        print(monthly.to_string())

    # Summer analysis
    analyze_summer_months(trades_df, df)

    # Test parameter variations
    test_parameter_variations(df, PARAMS)

    # Grid search for optimal V4
    best_params, best_result, all_results = find_optimal_v4_params(df, PARAMS)

    # Print V4 recommendations
    print("\n" + "="*70)
    print("V4 RECOMMENDED PARAMETERS")
    print("="*70)

    if best_params:
        print(f"\nBest configuration found:")
        print(f"  Min ATR Percentile: {best_params['min_atr_pct']}")
        print(f"  Max ATR Percentile: {best_params['max_atr_pct']}")
        print(f"  Max Holding Hours:  {best_params['max_holding_hours']}")
        print(f"  ConsecLoss Cooldown: {best_params['consec_loss_cooldown']}")
        print(f"  SL Multiplier:      {best_params['sl_multiplier']}")
        print(f"\nExpected Performance:")
        print(f"  Total P&L:          ${best_result['total_pnl']:,.2f}")
        print(f"  Summer P&L:         ${best_result['summer_pnl']:,.2f}")
        print(f"  Max Drawdown:       {best_result['max_dd']:.1f}%")
        print(f"  Max Consec Losses:  {best_result['max_consec']}")
        print(f"  Win Rate:           {best_result['win_rate']:.1f}%")

    # Save results
    output_file = r'C:\Users\Administrator\Music\SURGE-WSI\strategies\rsi_v37_optimized\v4_optimization_results.csv'
    all_results.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")

    mt5.shutdown()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
