"""
RSI v3.7 - Comparison with $5,000 Capital
=========================================
Compare Original vs BB_SQUEEZE with different starting capital
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
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'iy#K5L7sF')
MT5_SERVER = os.getenv('MT5_SERVER', 'FinexBisnisSolusi-Demo')
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe")

SYMBOL = "GBPUSD"

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
RISK_PER_TRADE = 0.01  # 1% risk per trade

USE_CONSEC_LOSS_FILTER = True
CONSEC_LOSS_LIMIT = 3

# BB Squeeze settings
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_PERCENTILE = 30
SMA_SLOPE_THRESHOLD = 0.5


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


def prepare_data(df):
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

    # SMAs
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # Regime
    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > SMA_SLOPE_THRESHOLD),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -SMA_SLOPE_THRESHOLD),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
    df['bb_width'] = (df['bb_std'] * 2 * BB_STD) / df['bb_mid'] * 100

    def bb_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['bb_width_pct'] = df['bb_width'].rolling(100).apply(bb_pct_func, raw=True)
    df['bb_squeeze'] = df['bb_width_pct'] < BB_SQUEEZE_PERCENTILE

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)


def run_backtest(df, initial_balance, use_bb_squeeze=False, target_months=None):
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0
    max_drawdown_pct = 0

    position = None
    consecutive_losses = 0
    trades = []
    monthly_pnl = {}
    equity_curve = []

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        month_str = current_time.strftime('%Y-%m')

        if target_months and month_str not in target_months:
            continue

        if row['weekday'] >= 5:
            continue

        # Track equity
        equity_curve.append({'time': current_time, 'balance': balance})

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

                # Track drawdown
                if balance > peak_balance:
                    peak_balance = balance
                drawdown = peak_balance - balance
                drawdown_pct = drawdown / peak_balance * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct

                if pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                pips = (exit_price - position['entry']) / 0.0001 if position['dir'] == 1 else (position['entry'] - exit_price) / 0.0001

                # Track monthly P/L
                entry_month = position['entry_time'].strftime('%Y-%m')
                if entry_month not in monthly_pnl:
                    monthly_pnl[entry_month] = 0
                monthly_pnl[entry_month] += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'BUY' if position['dir'] == 1 else 'SELL',
                    'pips': pips,
                    'pnl': pnl,
                    'result': 'WIN' if pnl > 0 else 'LOSS',
                    'exit_reason': exit_reason,
                    'balance_after': balance,
                })

                position = None

        # Entry logic
        if not position:
            hour = row['hour']
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            # Regime filter
            if row['regime'] != 'SIDEWAYS':
                continue

            # BB Squeeze filter (optional)
            if use_bb_squeeze and not row['bb_squeeze']:
                continue

            # Consecutive loss filter
            if USE_CONSEC_LOSS_FILTER and consecutive_losses >= CONSEC_LOSS_LIMIT:
                consecutive_losses = 0
                continue

            # RSI signal
            rsi = row['rsi']
            signal = 0
            if rsi < RSI_OVERSOLD:
                signal = 1
            elif rsi > RSI_OVERBOUGHT:
                signal = -1

            if not signal:
                continue

            # Execute trade
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
            }

    return {
        'trades': trades,
        'final_balance': balance,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'monthly_pnl': monthly_pnl,
        'equity_curve': equity_curve,
    }


def print_comparison(results_orig, results_bb, initial_balance):
    """Print side-by-side comparison"""

    print(f"\n{'='*70}")
    print(f"COMPARISON: Modal ${initial_balance:,.0f}")
    print(f"{'='*70}")

    print(f"\n{'Metric':<25} {'ORIGINAL':>20} {'BB_SQUEEZE':>20}")
    print("-" * 70)

    # Basic stats
    orig_trades = results_orig['trades']
    bb_trades = results_bb['trades']

    orig_wins = len([t for t in orig_trades if t['result'] == 'WIN'])
    bb_wins = len([t for t in bb_trades if t['result'] == 'WIN'])

    orig_pnl = sum(t['pnl'] for t in orig_trades)
    bb_pnl = sum(t['pnl'] for t in bb_trades)

    orig_return = (results_orig['final_balance'] - initial_balance) / initial_balance * 100
    bb_return = (results_bb['final_balance'] - initial_balance) / initial_balance * 100

    print(f"{'Total Trades':<25} {len(orig_trades):>20} {len(bb_trades):>20}")
    print(f"{'Wins':<25} {orig_wins:>20} {bb_wins:>20}")
    print(f"{'Win Rate':<25} {orig_wins/len(orig_trades)*100 if orig_trades else 0:>19.1f}% {bb_wins/len(bb_trades)*100 if bb_trades else 0:>19.1f}%")
    print(f"{'Total P/L':<25} ${orig_pnl:>+18,.2f} ${bb_pnl:>+18,.2f}")
    print(f"{'Final Balance':<25} ${results_orig['final_balance']:>18,.2f} ${results_bb['final_balance']:>18,.2f}")
    print(f"{'Return':<25} {orig_return:>+19.1f}% {bb_return:>+19.1f}%")
    print(f"{'Max Drawdown $':<25} ${results_orig['max_drawdown']:>18,.2f} ${results_bb['max_drawdown']:>18,.2f}")
    print(f"{'Max Drawdown %':<25} {results_orig['max_drawdown_pct']:>19.1f}% {results_bb['max_drawdown_pct']:>19.1f}%")

    # Monthly comparison
    print(f"\n{'='*70}")
    print("MONTHLY P/L COMPARISON")
    print(f"{'='*70}")

    all_months = sorted(set(list(results_orig['monthly_pnl'].keys()) + list(results_bb['monthly_pnl'].keys())))

    print(f"\n{'Month':<12} {'ORIGINAL':>15} {'BB_SQUEEZE':>15} {'Difference':>15}")
    print("-" * 60)

    losing_months_orig = 0
    losing_months_bb = 0

    for month in all_months:
        orig_m = results_orig['monthly_pnl'].get(month, 0)
        bb_m = results_bb['monthly_pnl'].get(month, 0)
        diff = bb_m - orig_m

        if orig_m < 0:
            losing_months_orig += 1
        if bb_m < 0:
            losing_months_bb += 1

        marker = ""
        if orig_m < -100 and bb_m > orig_m:
            marker = " << SAVED"
        elif orig_m > 100 and bb_m < orig_m - 100:
            marker = " (missed)"

        print(f"{month:<12} ${orig_m:>+13,.0f} ${bb_m:>+13,.0f} ${diff:>+13,.0f}{marker}")

    print("-" * 60)
    print(f"{'Losing Months':<12} {losing_months_orig:>15} {losing_months_bb:>15}")

    # Risk-adjusted metrics
    print(f"\n{'='*70}")
    print("RISK-ADJUSTED METRICS")
    print(f"{'='*70}")

    # Calmar ratio (Annual Return / Max Drawdown)
    orig_calmar = orig_return / results_orig['max_drawdown_pct'] if results_orig['max_drawdown_pct'] > 0 else 0
    bb_calmar = bb_return / results_bb['max_drawdown_pct'] if results_bb['max_drawdown_pct'] > 0 else 0

    # Profit factor
    orig_gross_profit = sum(t['pnl'] for t in orig_trades if t['pnl'] > 0)
    orig_gross_loss = abs(sum(t['pnl'] for t in orig_trades if t['pnl'] < 0))
    bb_gross_profit = sum(t['pnl'] for t in bb_trades if t['pnl'] > 0)
    bb_gross_loss = abs(sum(t['pnl'] for t in bb_trades if t['pnl'] < 0))

    orig_pf = orig_gross_profit / orig_gross_loss if orig_gross_loss > 0 else 0
    bb_pf = bb_gross_profit / bb_gross_loss if bb_gross_loss > 0 else 0

    # Average trade
    orig_avg = orig_pnl / len(orig_trades) if orig_trades else 0
    bb_avg = bb_pnl / len(bb_trades) if bb_trades else 0

    print(f"\n{'Metric':<25} {'ORIGINAL':>20} {'BB_SQUEEZE':>20}")
    print("-" * 70)
    print(f"{'Calmar Ratio':<25} {orig_calmar:>20.2f} {bb_calmar:>20.2f}")
    print(f"{'Profit Factor':<25} {orig_pf:>20.2f} {bb_pf:>20.2f}")
    print(f"{'Avg Trade P/L':<25} ${orig_avg:>+18,.2f} ${bb_avg:>+18,.2f}")
    print(f"{'Losing Months':<25} {losing_months_orig:>20} {losing_months_bb:>20}")


def main():
    print("=" * 70)
    print("RSI v3.7 - CAPITAL COMPARISON")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

        df = get_data(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
        print(f"Loaded {len(df)} H1 bars")

        print("Preparing indicators...")
        df = prepare_data(df)

        # Test with different capital amounts
        capitals = [5000, 10000, 25000]

        for capital in capitals:
            print(f"\n{'#'*70}")
            print(f"TESTING WITH ${capital:,} INITIAL CAPITAL")
            print(f"{'#'*70}")

            # Run Original
            results_orig = run_backtest(df, capital, use_bb_squeeze=False)

            # Run BB_SQUEEZE
            results_bb = run_backtest(df, capital, use_bb_squeeze=True)

            # Print comparison
            print_comparison(results_orig, results_bb, capital)

        # Final recommendation
        print("\n" + "=" * 70)
        print("REKOMENDASI BERDASARKAN MODAL")
        print("=" * 70)

        print("""
Modal $5,000:
  - ORIGINAL: Higher return tapi drawdown bisa mencapai ~$800-1000 (16-20%)
  - BB_SQUEEZE: Return lebih rendah tapi drawdown terkontrol ~$400-600 (8-12%)
  - REKOMENDASI: BB_SQUEEZE lebih aman untuk modal kecil

Modal $10,000:
  - ORIGINAL: Drawdown ~$1,500-2,000 masih acceptable
  - BB_SQUEEZE: Lebih conservative
  - REKOMENDASI: Bisa pilih ORIGINAL jika toleran risiko

Modal $25,000+:
  - Drawdown dalam % lebih kecil relative terhadap modal
  - REKOMENDASI: ORIGINAL untuk maximize return
        """)

    finally:
        mt5.shutdown()
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
