"""
Analyze Dec 2024 & Jan 2025 - RSI v3.7 OPTIMIZED
================================================
Generate detailed Excel report for profitable months
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# Configuration
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '61045904'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
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
    print(f"Connected: {MT5_LOGIN}")
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
                exit_reason = 'TIMEOUT (46h max)'
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
                if entry_month in target_months or month_str in target_months:
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
                        'Regime': position['regime'],
                        'Entry Hour': position['entry_hour'],
                        'Entry Day': position['entry_day'],
                        'Balance After': round(balance, 2),
                        'ConsecLoss After': consecutive_losses,
                        'Signal Reason': position['signal_reason'],
                        'Trade Notes': position['trade_notes']
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
            signal_reason = ""

            if rsi < RSI_OVERSOLD:
                signal = 1
                signal_reason = f"RSI={rsi:.1f} < {RSI_OVERSOLD} (Oversold -> BUY)"
            elif rsi > RSI_OVERBOUGHT:
                signal = -1
                signal_reason = f"RSI={rsi:.1f} > {RSI_OVERBOUGHT} (Overbought -> SELL)"

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                trade_notes = []

                if atr_pct < 40:
                    tp_mult = TP_LOW
                    trade_notes.append(f"Low ATR%={atr_pct:.0f} -> TP={TP_LOW}x")
                elif atr_pct > 60:
                    tp_mult = TP_HIGH
                    trade_notes.append(f"High ATR%={atr_pct:.0f} -> TP={TP_HIGH}x")
                else:
                    tp_mult = TP_MED
                    trade_notes.append(f"Med ATR%={atr_pct:.0f} -> TP={TP_MED}x")

                if 12 <= hour < 16:
                    tp_mult += 0.35
                    trade_notes.append(f"Hour={hour} (12-16) -> TP bonus +0.35x")

                if signal == 1:
                    sl = entry - atr * SL_MULT
                    tp = entry + atr * tp_mult
                else:
                    sl = entry + atr * SL_MULT
                    tp = entry - atr * tp_mult

                risk = balance * RISK_PER_TRADE
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

                sl_pips = abs(entry - sl) / 0.0001
                tp_pips = abs(tp - entry) / 0.0001
                rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0

                trade_notes.append(f"SL={sl_pips:.0f}p, TP={tp_pips:.0f}p, R:R=1:{rr_ratio:.1f}")
                trade_notes.append(f"Regime={regime}, ConsecLoss={consecutive_losses}")

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
                    'regime': regime,
                    'signal_reason': signal_reason,
                    'trade_notes': ' | '.join(trade_notes)
                }

    return trade_log


def main():
    print("=" * 60)
    print("ANALYZE DEC 2024 & JAN 2025")
    print("Profitable Months Analysis")
    print("=" * 60)

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
        print("Calculating indicators...")
        df = prepare_indicators(df)

        print("Running detailed backtest...")
        trades = run_detailed_backtest(df, target_months=['2024-12', '2025-01'])

        print(f"\nTotal trades: {len(trades)}")

        trades_df = pd.DataFrame(trades)

        dec_2024 = trades_df[trades_df['Entry Time'].dt.strftime('%Y-%m') == '2024-12'].copy()
        jan_2025 = trades_df[trades_df['Entry Time'].dt.strftime('%Y-%m') == '2025-01'].copy()

        print(f"\nDecember 2024: {len(dec_2024)} trades")
        print(f"January 2025: {len(jan_2025)} trades")

        def calc_stats(df_month, name):
            if len(df_month) == 0:
                return
            wins = len(df_month[df_month['Result'] == 'WIN'])
            losses = len(df_month[df_month['Result'] == 'LOSS'])
            total_pnl = df_month['P/L ($)'].sum()
            win_rate = wins / len(df_month) * 100

            print(f"\n{name}:")
            print(f"  Trades: {len(df_month)}")
            print(f"  Wins: {wins} ({win_rate:.1f}%)")
            print(f"  Losses: {losses}")
            print(f"  Total P/L: ${total_pnl:+,.2f}")

            buys = df_month[df_month['Direction'] == 'BUY']
            sells = df_month[df_month['Direction'] == 'SELL']
            print(f"  BUY trades: {len(buys)} (P/L: ${buys['P/L ($)'].sum():+,.2f})")
            print(f"  SELL trades: {len(sells)} (P/L: ${sells['P/L ($)'].sum():+,.2f})")

            print(f"  Exit reasons:")
            for reason in df_month['Exit Reason'].unique():
                count = len(df_month[df_month['Exit Reason'] == reason])
                pnl = df_month[df_month['Exit Reason'] == reason]['P/L ($)'].sum()
                print(f"    {reason}: {count} (${pnl:+,.2f})")

        calc_stats(dec_2024, "DECEMBER 2024")
        calc_stats(jan_2025, "JANUARY 2025")

        # Summary
        summary_data = []
        for name, df_month in [('December 2024', dec_2024), ('January 2025', jan_2025)]:
            if len(df_month) == 0:
                continue
            wins = len(df_month[df_month['Result'] == 'WIN'])
            losses = len(df_month[df_month['Result'] == 'LOSS'])
            buys = df_month[df_month['Direction'] == 'BUY']
            sells = df_month[df_month['Direction'] == 'SELL']

            summary_data.append({
                'Month': name,
                'Total Trades': len(df_month),
                'Wins': wins,
                'Losses': losses,
                'Win Rate': f"{wins/len(df_month)*100:.1f}%",
                'Total P/L': df_month['P/L ($)'].sum(),
                'BUY Trades': len(buys),
                'BUY P/L': buys['P/L ($)'].sum(),
                'SELL Trades': len(sells),
                'SELL P/L': sells['P/L ($)'].sum(),
                'Avg RSI Entry': df_month['RSI Entry'].mean(),
                'Avg Pips': df_month['Pips'].mean(),
                'SL Hits': len(df_month[df_month['Exit Reason'] == 'SL HIT']),
                'TP Hits': len(df_month[df_month['Exit Reason'] == 'TP HIT']),
                'Timeouts': len(df_month[df_month['Exit Reason'].str.contains('TIMEOUT')])
            })

        summary_df = pd.DataFrame(summary_data)

        # Export
        output_file = os.path.join(os.path.dirname(__file__), 'dec2024_jan2025_analysis.xlsx')

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            if len(dec_2024) > 0:
                dec_2024.to_excel(writer, sheet_name='Dec 2024 Trades', index=False)
            if len(jan_2025) > 0:
                jan_2025.to_excel(writer, sheet_name='Jan 2025 Trades', index=False)
            trades_df.to_excel(writer, sheet_name='All Trades', index=False)

        print(f"\n{'='*60}")
        print(f"Excel exported: {output_file}")
        print(f"{'='*60}")

    finally:
        mt5.shutdown()
        print("\nDone!")


if __name__ == "__main__":
    main()
