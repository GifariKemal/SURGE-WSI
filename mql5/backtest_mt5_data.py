"""
RSI v3.7 Backtest using MT5 Data
================================
Run backtest using real MT5 historical data from Finex broker.
Period: 2025-01 to 2026-01
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os

# MT5 Connection
MT5_LOGIN = 61045904
MT5_PASSWORD = "iy#K5L7sF"
MT5_SERVER = "FinexBisnisSolusi-Demo"
MT5_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

def connect_mt5():
    """Connect to MT5"""
    if not mt5.initialize(path=MT5_PATH):
        print(f"MT5 init failed: {mt5.last_error()}")
        return False

    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        return False

    print(f"Connected to MT5: {mt5.account_info().server}")
    return True

def get_h1_data(symbol, start_date, end_date):
    """Get H1 OHLCV data from MT5"""
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Failed to get data: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def run_backtest(df):
    """Run RSI v3.7 backtest"""
    # RSI(10) using Wilder's smoothing (EMA with alpha=1/period)
    rsi_period = 10
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Wilder's smoothing: EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()

    # Safe division: when avg_loss is 0, RSI = 100
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Fill initial NaN with neutral value

    # ATR(14)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()

    # ATR Percentile: What % of historical values are below current value
    def atr_percentile(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()  # Compare against historical only
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0

    df['atr_pct'] = df['atr'].rolling(100).apply(atr_percentile, raw=True)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.7 Parameters
    SL_MULT = 1.5
    TP_LOW, TP_MED, TP_HIGH = 2.4, 3.0, 3.6
    MIN_ATR_PCT, MAX_ATR_PCT = 20, 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]
    RSI_OS, RSI_OB = 42, 58

    # Backtest
    balance = 10000.0
    initial_balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0

    trades_list = []
    equity_curve = []
    monthly_pnl = {}

    for i in range(200, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        month = current_time.strftime('%Y-%m')
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

                if month not in monthly_pnl:
                    monthly_pnl[month] = 0
                monthly_pnl[month] += pnl

                trades_list.append({
                    'time': current_time,
                    'type': 'CLOSE',
                    'direction': 'BUY' if position['dir'] == 1 else 'SELL',
                    'price': row['close'],
                    'pnl': pnl,
                    'exit': exit_reason,
                    'balance': balance
                })
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        equity_curve.append({'time': current_time, 'equity': balance})

        if not position:
            if hour < 7 or hour >= 22 or hour in SKIP_HOURS:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            rsi = row['rsi']
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                tp_mult = base_tp + TIME_TP_BONUS if 12 <= hour < 16 else base_tp
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

                trades_list.append({
                    'time': current_time,
                    'type': 'OPEN',
                    'direction': 'BUY' if signal == 1 else 'SELL',
                    'price': entry,
                    'sl': sl,
                    'tp': tp,
                    'rsi': rsi,
                    'atr_pct': atr_pct,
                    'size': size
                })

    # Calculate results
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - initial_balance) / initial_balance * 100

    wins_list = [t['pnl'] for t in trades_list if t.get('pnl', 0) > 0]
    losses_list = [t['pnl'] for t in trades_list if t.get('pnl', 0) < 0]
    gross_profit = sum(wins_list) if wins_list else 0
    gross_loss = abs(sum(losses_list)) if losses_list else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    avg_win = np.mean(wins_list) if wins_list else 0
    avg_loss = abs(np.mean(losses_list)) if losses_list else 0

    tp_exits = sum(1 for t in trades_list if t.get('exit') == 'TP')
    sl_exits = sum(1 for t in trades_list if t.get('exit') == 'SL')
    timeout_exits = sum(1 for t in trades_list if t.get('exit') == 'TIMEOUT')

    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    total_months = len(monthly_pnl)

    results = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'timeout_exits': timeout_exits,
        'profitable_months': profitable_months,
        'total_months': total_months,
        'monthly_pnl': monthly_pnl,
        'trades': trades_list,
        'equity_curve': equity_curve
    }

    return results

def export_results(results, trades_df, equity_df):
    """Export results to files"""
    output_dir = "C:/Users/Administrator/Music/SURGE-WSI/mql5/backtest_results"
    os.makedirs(output_dir, exist_ok=True)

    # Export trades to CSV
    trades_df.to_csv(f"{output_dir}/RSI_v37_trades_2025_2026.csv", index=False)

    # Export equity curve
    equity_df.to_csv(f"{output_dir}/RSI_v37_equity_2025_2026.csv", index=False)

    # Export summary
    with open(f"{output_dir}/RSI_v37_summary_2025_2026.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RSI MEAN REVERSION v3.7 - BACKTEST REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Period: 2025-01-01 to 2026-01-31\n")
        f.write(f"Symbol: GBPUSD H1\n")
        f.write(f"Data Source: Finex MT5 Demo\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        f.write(f"Initial Balance: ${results['initial_balance']:,.2f}\n")
        f.write(f"Final Balance:   ${results['final_balance']:,.2f}\n")
        f.write(f"Total Return:    +{results['total_return']:.2f}%\n")
        f.write(f"Max Drawdown:    {results['max_drawdown']:.2f}%\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("TRADES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Trades:    {results['total_trades']}\n")
        f.write(f"Winners:         {results['wins']} ({results['win_rate']:.1f}%)\n")
        f.write(f"Losers:          {results['losses']}\n")
        f.write(f"Profit Factor:   {results['profit_factor']:.2f}\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("RISK/REWARD\n")
        f.write("-" * 60 + "\n")
        f.write(f"Avg Win:         ${results['avg_win']:.2f}\n")
        f.write(f"Avg Loss:        ${results['avg_loss']:.2f}\n")
        f.write(f"R:R Ratio:       1:{results['rr_ratio']:.2f}\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("EXIT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"TP Exits:        {results['tp_exits']} ({results['tp_exits']/results['total_trades']*100:.0f}%)\n")
        f.write(f"SL Exits:        {results['sl_exits']} ({results['sl_exits']/results['total_trades']*100:.0f}%)\n")
        f.write(f"Timeout Exits:   {results['timeout_exits']} ({results['timeout_exits']/results['total_trades']*100:.0f}%)\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("MONTHLY P/L\n")
        f.write("-" * 60 + "\n")
        for month, pnl in sorted(results['monthly_pnl'].items()):
            status = "+" if pnl > 0 else ""
            f.write(f"{month}: ${status}{pnl:,.2f}\n")
        f.write("\n")
        f.write(f"Profitable Months: {results['profitable_months']}/{results['total_months']} ({results['profitable_months']/results['total_months']*100:.0f}%)\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults exported to: {output_dir}")
    return output_dir

def main():
    print("=" * 60)
    print("RSI v3.7 BACKTEST - MT5 DATA (Finex)")
    print("Period: 2025-01-01 to 2026-01-31")
    print("=" * 60)

    # Connect to MT5
    if not connect_mt5():
        print("Failed to connect to MT5!")
        return

    try:
        # Get H1 data
        print("\nFetching GBPUSD H1 data from MT5...")
        start_date = datetime(2024, 10, 1, tzinfo=timezone.utc)  # Extra for warmup
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

        df = get_h1_data("GBPUSD", start_date, end_date)
        if df is None:
            print("Failed to get data!")
            return

        print(f"Loaded {len(df)} H1 bars")
        print(f"Data range: {df.index[0]} to {df.index[-1]}")

        # Run backtest
        print("\nRunning backtest...")
        results = run_backtest(df)

        # Print summary
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Return: +{results['total_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"R:R Ratio: 1:{results['rr_ratio']:.2f}")
        print(f"Profitable Months: {results['profitable_months']}/{results['total_months']}")

        # Export results
        trades_df = pd.DataFrame([t for t in results['trades'] if t['type'] == 'CLOSE'])
        equity_df = pd.DataFrame(results['equity_curve'])
        output_dir = export_results(results, trades_df, equity_df)

        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE!")
        print("=" * 60)

    finally:
        mt5.shutdown()
        print("\nMT5 disconnected")

if __name__ == "__main__":
    main()
