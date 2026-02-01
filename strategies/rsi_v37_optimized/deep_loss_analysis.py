"""
Deep Analysis: Why Nov 2024 & Apr 2025 Lost Money
==================================================
Investigate anomalies and patterns in losing months
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '61045904'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'iy#K5L7sF')
MT5_SERVER = os.getenv('MT5_SERVER', 'FinexBisnisSolusi-Demo')
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe")

SYMBOL = "GBPUSD"
INITIAL_BALANCE = 10000.0

# Strategy Parameters
RSI_PERIOD = 10
RSI_OVERSOLD = 42
RSI_OVERBOUGHT = 58
ATR_PERIOD = 14
SL_MULT = 1.5
TP_LOW = 2.4
TP_MED = 3.0
TP_HIGH = 3.6
MAX_HOLDING_HOURS = 46
MIN_ATR_PCT = 20
MAX_ATR_PCT = 80
RISK_PER_TRADE = 0.01

# Filters
USE_REGIME_FILTER = True
ALLOWED_REGIMES = ['SIDEWAYS']
USE_CONSEC_LOSS_FILTER = True
CONSEC_LOSS_LIMIT = 3


def connect_mt5():
    if not MT5_PASSWORD:
        print("ERROR: MT5_PASSWORD not set")
        return False
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


def prepare_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(ATR_PERIOD).mean()

    # ATR Percentile
    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # SMAs for regime
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # Regime
    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['day_name'] = df.index.day_name()

    # Additional metrics for analysis
    df['daily_range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    return df.ffill().fillna(0)


def run_detailed_backtest(df, target_months):
    balance = INITIAL_BALANCE
    position = None
    consecutive_losses = 0
    trade_log = []

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        month_str = current_time.strftime('%Y-%m')
        weekday = row['weekday']
        hour = row['hour']

        if weekday >= 5:
            continue

        # Position management
        if position:
            exit_reason = None
            exit_price = None
            pnl = 0
            bars_held = i - position['entry_idx']

            if bars_held >= MAX_HOLDING_HOURS:
                exit_price = row['close']
                pnl = (exit_price - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - exit_price) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL HIT'
                    elif row['high'] >= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP HIT'
                else:
                    if row['high'] >= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL HIT'
                    elif row['low'] <= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP HIT'

            if exit_reason:
                balance += pnl
                result = 'WIN' if pnl > 0 else 'LOSS'

                if pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                if position['dir'] == 1:
                    pips = (exit_price - position['entry']) / 0.0001
                else:
                    pips = (position['entry'] - exit_price) / 0.0001

                entry_month = position['entry_time'].strftime('%Y-%m')
                if entry_month in target_months:
                    # Calculate what happened AFTER entry
                    max_adverse = 0
                    max_favorable = 0
                    for j in range(position['entry_idx'] + 1, i + 1):
                        if j < len(df):
                            if position['dir'] == 1:
                                adverse = (position['entry'] - df.iloc[j]['low']) / 0.0001
                                favorable = (df.iloc[j]['high'] - position['entry']) / 0.0001
                            else:
                                adverse = (df.iloc[j]['high'] - position['entry']) / 0.0001
                                favorable = (position['entry'] - df.iloc[j]['low']) / 0.0001
                            max_adverse = max(max_adverse, adverse)
                            max_favorable = max(max_favorable, favorable)

                    sl_pips = abs(position['entry'] - position['sl']) / 0.0001
                    tp_pips = abs(position['tp'] - position['entry']) / 0.0001

                    trade_log.append({
                        'Entry Time': position['entry_time'],
                        'Exit Time': current_time,
                        'Direction': 'BUY' if position['dir'] == 1 else 'SELL',
                        'Entry Price': position['entry'],
                        'SL': position['sl'],
                        'TP': position['tp'],
                        'Exit Price': exit_price,
                        'Pips': round(pips, 1),
                        'P/L ($)': round(pnl, 2),
                        'Result': result,
                        'Exit Reason': exit_reason,
                        'Hours Held': bars_held,
                        'RSI Entry': round(position['rsi'], 1),
                        'ATR%': round(position['atr_pct'], 1),
                        'ATR (pips)': round(position['atr'] / 0.0001, 1),
                        'Regime': position['regime'],
                        'Entry Hour': position['entry_hour'],
                        'Entry Day': position['entry_day'],
                        'SL Pips': round(sl_pips, 1),
                        'TP Pips': round(tp_pips, 1),
                        'R:R': round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0,
                        'Max Adverse (pips)': round(max_adverse, 1),
                        'Max Favorable (pips)': round(max_favorable, 1),
                        'Almost TP?': 'YES' if max_favorable >= tp_pips * 0.8 else 'NO',
                        'Deep Drawdown?': 'YES' if max_adverse > sl_pips * 0.8 else 'NO',
                    })

                position = None

        # Entry logic
        if not position:
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            skip_trade = False
            regime = row['regime']

            if USE_REGIME_FILTER and regime not in ALLOWED_REGIMES:
                skip_trade = True

            if not skip_trade and USE_CONSEC_LOSS_FILTER:
                if consecutive_losses >= CONSEC_LOSS_LIMIT:
                    skip_trade = True
                    consecutive_losses = 0

            if skip_trade:
                continue

            rsi = row['rsi']
            signal = 0

            if rsi < RSI_OVERSOLD:
                signal = 1
            elif rsi > RSI_OVERBOUGHT:
                signal = -1

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002

                if atr_pct < 40:
                    tp_mult = TP_LOW
                elif atr_pct > 60:
                    tp_mult = TP_HIGH
                else:
                    tp_mult = TP_MED

                if 12 <= hour < 16:
                    tp_mult += 0.35

                if signal == 1:
                    sl = entry - atr * SL_MULT
                    tp = entry + atr * tp_mult
                else:
                    sl = entry + atr * SL_MULT
                    tp = entry - atr * tp_mult

                risk = balance * RISK_PER_TRADE
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

                position = {
                    'dir': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'entry_idx': i,
                    'entry_time': current_time,
                    'entry_hour': hour,
                    'entry_day': row['day_name'],
                    'rsi': rsi,
                    'atr_pct': atr_pct,
                    'atr': atr,
                    'regime': regime,
                }

    return trade_log


def analyze_market_conditions(df, target_months):
    """Analyze market conditions during losing months"""
    analysis = {}

    for month in target_months:
        year, mon = map(int, month.split('-'))
        month_data = df[(df.index.year == year) & (df.index.month == mon)]

        if len(month_data) == 0:
            continue

        # Calculate monthly metrics
        price_change = (month_data['close'].iloc[-1] - month_data['close'].iloc[0]) / month_data['close'].iloc[0] * 100
        avg_atr = month_data['atr'].mean()
        avg_atr_pct = month_data['atr_pct'].mean()

        # Regime distribution
        regime_counts = month_data['regime'].value_counts()
        sideways_pct = regime_counts.get('SIDEWAYS', 0) / len(month_data) * 100
        bull_pct = regime_counts.get('BULL', 0) / len(month_data) * 100
        bear_pct = regime_counts.get('BEAR', 0) / len(month_data) * 100

        # Volatility analysis
        daily_ranges = month_data.groupby(month_data.index.date)['daily_range'].mean()

        # Trend strength
        sma_distance = abs(month_data['sma_20'] - month_data['sma_50']).mean() / month_data['close'].mean() * 100

        analysis[month] = {
            'price_change_pct': round(price_change, 2),
            'avg_atr_pips': round(avg_atr / 0.0001, 1),
            'avg_atr_pct': round(avg_atr_pct, 1),
            'sideways_pct': round(sideways_pct, 1),
            'bull_pct': round(bull_pct, 1),
            'bear_pct': round(bear_pct, 1),
            'sma_distance_pct': round(sma_distance, 3),
            'avg_daily_range_pips': round(daily_ranges.mean() / 0.0001, 1),
        }

    return analysis


def main():
    print("=" * 70)
    print("DEEP ANALYSIS: Why Nov 2024 & Apr 2025 Lost Money")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data(SYMBOL, start_date, end_date)

        if df is None:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")
        df = prepare_indicators(df)

        losing_months = ['2024-11', '2025-04']
        profitable_months = ['2024-12', '2025-01']

        print("\n" + "=" * 70)
        print("MARKET CONDITIONS COMPARISON")
        print("=" * 70)

        all_months = losing_months + profitable_months
        market_analysis = analyze_market_conditions(df, all_months)

        print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Month", "Price%", "ATR(pip)", "ATR%", "SIDEWAY%", "BULL%", "BEAR%"
        ))
        print("-" * 70)

        for month in all_months:
            if month in market_analysis:
                m = market_analysis[month]
                status = "LOSS" if month in losing_months else "PROFIT"
                print("{:<12} {:>+10.2f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}  [{}]".format(
                    month, m['price_change_pct'], m['avg_atr_pips'], m['avg_atr_pct'],
                    m['sideways_pct'], m['bull_pct'], m['bear_pct'], status
                ))

        print("\n" + "=" * 70)
        print("TRADE-BY-TRADE ANALYSIS - LOSING MONTHS")
        print("=" * 70)

        trades = run_detailed_backtest(df, losing_months)
        trades_df = pd.DataFrame(trades)

        for month in losing_months:
            month_trades = trades_df[trades_df['Entry Time'].dt.strftime('%Y-%m') == month].copy()

            if len(month_trades) == 0:
                print(f"\n{month}: No trades")
                continue

            print(f"\n{'='*70}")
            print(f"{month} DETAILED ANALYSIS")
            print(f"{'='*70}")

            wins = len(month_trades[month_trades['Result'] == 'WIN'])
            losses = len(month_trades[month_trades['Result'] == 'LOSS'])
            total_pnl = month_trades['P/L ($)'].sum()

            print(f"\nSUMMARY: {len(month_trades)} trades | {wins}W/{losses}L | ${total_pnl:+,.2f}")

            # Direction analysis
            buys = month_trades[month_trades['Direction'] == 'BUY']
            sells = month_trades[month_trades['Direction'] == 'SELL']

            print(f"\nBY DIRECTION:")
            print(f"  BUY:  {len(buys)} trades, {len(buys[buys['Result']=='WIN'])}W/{len(buys[buys['Result']=='LOSS'])}L, ${buys['P/L ($)'].sum():+,.2f}")
            print(f"  SELL: {len(sells)} trades, {len(sells[sells['Result']=='WIN'])}W/{len(sells[sells['Result']=='LOSS'])}L, ${sells['P/L ($)'].sum():+,.2f}")

            # Exit reason analysis
            print(f"\nBY EXIT REASON:")
            for reason in month_trades['Exit Reason'].unique():
                subset = month_trades[month_trades['Exit Reason'] == reason]
                print(f"  {reason}: {len(subset)} trades, ${subset['P/L ($)'].sum():+,.2f}")

            # Hour analysis
            print(f"\nBY ENTRY HOUR (losing hours):")
            hour_pnl = month_trades.groupby('Entry Hour')['P/L ($)'].sum().sort_values()
            for hour, pnl in hour_pnl.items():
                if pnl < 0:
                    count = len(month_trades[month_trades['Entry Hour'] == hour])
                    print(f"  Hour {hour:02d}: {count} trades, ${pnl:+,.2f}")

            # Day analysis
            print(f"\nBY DAY (losing days):")
            day_pnl = month_trades.groupby('Entry Day')['P/L ($)'].sum().sort_values()
            for day, pnl in day_pnl.items():
                if pnl < 0:
                    count = len(month_trades[month_trades['Entry Day'] == day])
                    print(f"  {day}: {count} trades, ${pnl:+,.2f}")

            # Anomaly detection
            print(f"\nANOMALIES DETECTED:")

            # Trades that almost hit TP but reversed
            almost_tp = month_trades[month_trades['Almost TP?'] == 'YES']
            almost_tp_losses = almost_tp[almost_tp['Result'] == 'LOSS']
            if len(almost_tp_losses) > 0:
                print(f"  - {len(almost_tp_losses)} trades almost hit TP (>80%) but reversed to SL")
                for _, t in almost_tp_losses.iterrows():
                    print(f"    * {t['Entry Time'].strftime('%Y-%m-%d %H:%M')} {t['Direction']}: Max Favorable={t['Max Favorable (pips)']:.0f}p vs TP={t['TP Pips']:.0f}p")

            # Deep drawdown trades
            deep_dd = month_trades[month_trades['Deep Drawdown?'] == 'YES']
            if len(deep_dd) > 0:
                print(f"  - {len(deep_dd)} trades had deep drawdown (>80% of SL)")

            # RSI extremes
            rsi_extreme_losses = month_trades[(month_trades['Result'] == 'LOSS') &
                                              ((month_trades['RSI Entry'] < 35) | (month_trades['RSI Entry'] > 65))]
            if len(rsi_extreme_losses) > 0:
                print(f"  - {len(rsi_extreme_losses)} losses with extreme RSI (<35 or >65)")

            # ATR% analysis
            low_atr_trades = month_trades[month_trades['ATR%'] < 30]
            high_atr_trades = month_trades[month_trades['ATR%'] > 70]
            if len(low_atr_trades) > 0:
                low_atr_pnl = low_atr_trades['P/L ($)'].sum()
                print(f"  - {len(low_atr_trades)} trades in low ATR% (<30): ${low_atr_pnl:+,.2f}")
            if len(high_atr_trades) > 0:
                high_atr_pnl = high_atr_trades['P/L ($)'].sum()
                print(f"  - {len(high_atr_trades)} trades in high ATR% (>70): ${high_atr_pnl:+,.2f}")

            # Consecutive loss streaks
            print(f"\nLOSS STREAKS:")
            streak = 0
            max_streak = 0
            streak_pnl = 0
            for _, t in month_trades.iterrows():
                if t['Result'] == 'LOSS':
                    streak += 1
                    streak_pnl += t['P/L ($)']
                    max_streak = max(max_streak, streak)
                else:
                    if streak >= 2:
                        print(f"  - {streak} consecutive losses: ${streak_pnl:+,.2f}")
                    streak = 0
                    streak_pnl = 0
            if streak >= 2:
                print(f"  - {streak} consecutive losses (end of month): ${streak_pnl:+,.2f}")
            print(f"  Maximum streak: {max_streak}")

            # List worst trades
            print(f"\nWORST 5 TRADES:")
            worst = month_trades.nsmallest(5, 'P/L ($)')
            for _, t in worst.iterrows():
                print(f"  {t['Entry Time'].strftime('%Y-%m-%d %H:%M')} {t['Direction']} RSI={t['RSI Entry']:.0f} ATR%={t['ATR%']:.0f}")
                print(f"    Entry={t['Entry Price']:.5f} SL={t['SL']:.5f} TP={t['TP']:.5f}")
                print(f"    Exit={t['Exit Price']:.5f} by {t['Exit Reason']} after {t['Hours Held']}h")
                print(f"    Max Adverse={t['Max Adverse (pips)']:.0f}p, Max Favorable={t['Max Favorable (pips)']:.0f}p")
                print(f"    P/L: ${t['P/L ($)']:+,.2f} ({t['Pips']:+.1f} pips)")
                print()

        # Compare with profitable months
        print("\n" + "=" * 70)
        print("COMPARISON: LOSING vs PROFITABLE MONTHS")
        print("=" * 70)

        profitable_trades = run_detailed_backtest(df, profitable_months)
        profitable_df = pd.DataFrame(profitable_trades)

        print("\n{:<20} {:>15} {:>15}".format("Metric", "LOSING", "PROFITABLE"))
        print("-" * 50)

        # Calculate averages
        if len(trades_df) > 0 and len(profitable_df) > 0:
            metrics = [
                ("Avg RSI Entry", trades_df['RSI Entry'].mean(), profitable_df['RSI Entry'].mean()),
                ("Avg ATR%", trades_df['ATR%'].mean(), profitable_df['ATR%'].mean()),
                ("Avg ATR (pips)", trades_df['ATR (pips)'].mean(), profitable_df['ATR (pips)'].mean()),
                ("Avg Hours Held", trades_df['Hours Held'].mean(), profitable_df['Hours Held'].mean()),
                ("SL Hit %", len(trades_df[trades_df['Exit Reason']=='SL HIT'])/len(trades_df)*100,
                            len(profitable_df[profitable_df['Exit Reason']=='SL HIT'])/len(profitable_df)*100),
                ("TP Hit %", len(trades_df[trades_df['Exit Reason']=='TP HIT'])/len(trades_df)*100,
                            len(profitable_df[profitable_df['Exit Reason']=='TP HIT'])/len(profitable_df)*100),
                ("BUY Win Rate", len(trades_df[(trades_df['Direction']=='BUY')&(trades_df['Result']=='WIN')])/len(trades_df[trades_df['Direction']=='BUY'])*100 if len(trades_df[trades_df['Direction']=='BUY'])>0 else 0,
                                len(profitable_df[(profitable_df['Direction']=='BUY')&(profitable_df['Result']=='WIN')])/len(profitable_df[profitable_df['Direction']=='BUY'])*100 if len(profitable_df[profitable_df['Direction']=='BUY'])>0 else 0),
                ("SELL Win Rate", len(trades_df[(trades_df['Direction']=='SELL')&(trades_df['Result']=='WIN')])/len(trades_df[trades_df['Direction']=='SELL'])*100 if len(trades_df[trades_df['Direction']=='SELL'])>0 else 0,
                                 len(profitable_df[(profitable_df['Direction']=='SELL')&(profitable_df['Result']=='WIN')])/len(profitable_df[profitable_df['Direction']=='SELL'])*100 if len(profitable_df[profitable_df['Direction']=='SELL'])>0 else 0),
            ]

            for name, losing_val, profit_val in metrics:
                diff = profit_val - losing_val
                indicator = "<<" if abs(diff) > 5 else ""
                print("{:<20} {:>15.1f} {:>15.1f} {}".format(name, losing_val, profit_val, indicator))

        print("\n" + "=" * 70)
        print("ROOT CAUSE ANALYSIS")
        print("=" * 70)

        # Market condition difference
        losing_market = [market_analysis[m] for m in losing_months if m in market_analysis]
        profit_market = [market_analysis[m] for m in profitable_months if m in market_analysis]

        if losing_market and profit_market:
            avg_losing_price_change = np.mean([m['price_change_pct'] for m in losing_market])
            avg_profit_price_change = np.mean([m['price_change_pct'] for m in profit_market])

            avg_losing_sideways = np.mean([m['sideways_pct'] for m in losing_market])
            avg_profit_sideways = np.mean([m['sideways_pct'] for m in profit_market])

            print(f"\n1. TREND STRENGTH:")
            print(f"   Losing months avg price change: {avg_losing_price_change:+.2f}%")
            print(f"   Profitable months avg price change: {avg_profit_price_change:+.2f}%")
            if abs(avg_losing_price_change) > abs(avg_profit_price_change):
                print(f"   >> ISSUE: Losing months had STRONGER trends - RSI mean reversion struggles!")

            print(f"\n2. SIDEWAYS REGIME:")
            print(f"   Losing months SIDEWAYS%: {avg_losing_sideways:.1f}%")
            print(f"   Profitable months SIDEWAYS%: {avg_profit_sideways:.1f}%")
            if avg_losing_sideways < avg_profit_sideways:
                print(f"   >> ISSUE: Less SIDEWAYS time = fewer quality setups!")

            print(f"\n3. STRATEGY LIMITATION:")
            print(f"   - RSI mean reversion works best in RANGING markets")
            print(f"   - Strong trends cause repeated SL hits against the trend")
            print(f"   - The SIDEWAYS filter helps but isn't perfect")

    finally:
        mt5.shutdown()
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
