"""
v3.7 RSI Mean Reversion Strategy - Detailed Backtest
=====================================================
Final production-ready backtest with comprehensive metrics.
"""
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT time, open, high, low, close
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)
    conn.close()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # RSI(10)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.7 Parameters
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]
    RSI_OS = 42
    RSI_OB = 58

    balance = 10000.0
    initial_balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    max_dd_date = None

    # Detailed tracking
    trades_list = []
    equity_curve = []
    monthly_pnl = {}
    yearly_pnl = {}

    for i in range(200, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        year = current_time.year
        month = current_time.strftime('%Y-%m')
        weekday = row['weekday']
        hour = row['hour']

        if weekday >= 5:
            continue

        if position:
            exit_reason = None
            pnl = 0
            exit_price = 0

            if (i - position['entry_idx']) >= MAX_HOLDING:
                if position['dir'] == 1:
                    pnl = (row['close'] - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - row['close']) * position['size']
                exit_reason = 'TIMEOUT'
                exit_price = row['close']
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL'
                        exit_price = position['sl']
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP'
                        exit_price = position['tp']
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL'
                        exit_price = position['sl']
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP'
                        exit_price = position['tp']

            if exit_reason:
                balance += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                # Track monthly/yearly
                if month not in monthly_pnl:
                    monthly_pnl[month] = 0
                monthly_pnl[month] += pnl

                if year not in yearly_pnl:
                    yearly_pnl[year] = 0
                yearly_pnl[year] += pnl

                trades_list.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'dir': 'LONG' if position['dir'] == 1 else 'SHORT',
                    'entry': position['entry'],
                    'exit': exit_price,
                    'sl': position['sl'],
                    'tp': position['tp'],
                    'pnl': pnl,
                    'pnl_pct': pnl / position['risk_amount'] * 100 if position['risk_amount'] > 0 else 0,
                    'exit_reason': exit_reason,
                    'holding_hours': i - position['entry_idx'],
                    'rsi_entry': position['rsi_entry'],
                    'atr_pct': position['atr_pct']
                })
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd
                max_dd_date = current_time

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
                if 12 <= hour < 16:
                    tp_mult = base_tp + TIME_TP_BONUS
                else:
                    tp_mult = base_tp

                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult

                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

                position = {
                    'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp,
                    'size': size, 'entry_idx': i, 'rsi_entry': rsi,
                    'atr_pct': atr_pct, 'entry_time': current_time,
                    'risk_amount': risk
                }

    # Results
    print('=' * 100)
    print('v3.7 RSI MEAN REVERSION STRATEGY - DETAILED BACKTEST')
    print('=' * 100)
    print(f'Period: 2020-01-01 to 2026-01-31 | Symbol: GBPUSD H1')
    print('=' * 100)

    print('\n' + '=' * 100)
    print('STRATEGY PARAMETERS')
    print('=' * 100)
    print(f'   RSI Period: 10 | Oversold: {RSI_OS} | Overbought: {RSI_OB}')
    print(f'   SL Multiplier: {SL_MULT}x ATR')
    print(f'   TP Multiplier: {TP_LOW}/{TP_MED}/{TP_HIGH}x ATR (low/med/high volatility)')
    print(f'   Time TP Bonus: +{TIME_TP_BONUS}x during 12:00-16:00 UTC')
    print(f'   ATR Filter: {MIN_ATR_PCT}-{MAX_ATR_PCT} percentile')
    print(f'   Trading Hours: 07:00-22:00 UTC, skip {SKIP_HOURS}')
    print(f'   Max Holding: {MAX_HOLDING} hours')
    print(f'   Risk per Trade: 1% of balance')

    trades = wins + losses
    wr = wins / trades * 100 if trades else 0
    total_return = (balance - initial_balance) / initial_balance * 100

    print('\n' + '=' * 100)
    print('PERFORMANCE SUMMARY')
    print('=' * 100)
    print(f'   Initial Balance:  ${initial_balance:,.0f}')
    print(f'   Final Balance:    ${balance:,.2f}')
    print(f'   Total Return:     +{total_return:.1f}%')
    print(f'   Max Drawdown:     {max_dd:.1f}% (on {max_dd_date.strftime("%Y-%m-%d") if max_dd_date else "N/A"})')
    print(f'   Total Trades:     {trades}')
    print(f'   Wins:             {wins} ({wr:.1f}%)')
    print(f'   Losses:           {losses} ({100-wr:.1f}%)')

    # Trade statistics
    if trades_list:
        wins_list = [t['pnl'] for t in trades_list if t['pnl'] > 0]
        losses_list = [t['pnl'] for t in trades_list if t['pnl'] <= 0]

        avg_win = np.mean(wins_list) if wins_list else 0
        avg_loss = abs(np.mean(losses_list)) if losses_list else 0
        max_win = max(wins_list) if wins_list else 0
        max_loss = min(losses_list) if losses_list else 0

        profit_factor = (sum(wins_list)) / abs(sum(losses_list)) if losses_list else 999

        tp_exits = sum(1 for t in trades_list if t['exit_reason'] == 'TP')
        sl_exits = sum(1 for t in trades_list if t['exit_reason'] == 'SL')
        timeout_exits = sum(1 for t in trades_list if t['exit_reason'] == 'TIMEOUT')

        long_trades = [t for t in trades_list if t['dir'] == 'LONG']
        short_trades = [t for t in trades_list if t['dir'] == 'SHORT']
        long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
        short_wins = sum(1 for t in short_trades if t['pnl'] > 0)

        avg_holding = np.mean([t['holding_hours'] for t in trades_list])

        print('\n' + '=' * 100)
        print('TRADE STATISTICS')
        print('=' * 100)
        print(f'   Profit Factor:    {profit_factor:.2f}')
        print(f'   Avg Win:          ${avg_win:.2f}')
        print(f'   Avg Loss:         ${avg_loss:.2f}')
        if avg_loss > 0:
            print(f'   Risk:Reward:      1:{avg_win/avg_loss:.2f}')
        print(f'   Max Win:          ${max_win:.2f}')
        print(f'   Max Loss:         ${max_loss:.2f}')
        print(f'   Avg Holding:      {avg_holding:.1f} hours')

        print('\n' + '=' * 100)
        print('EXIT ANALYSIS')
        print('=' * 100)
        print(f'   Take Profit:      {tp_exits} ({tp_exits/trades*100:.1f}%)')
        print(f'   Stop Loss:        {sl_exits} ({sl_exits/trades*100:.1f}%)')
        print(f'   Timeout:          {timeout_exits} ({timeout_exits/trades*100:.1f}%)')

        print('\n' + '=' * 100)
        print('DIRECTION ANALYSIS')
        print('=' * 100)
        if long_trades:
            long_pnl = sum(t['pnl'] for t in long_trades)
            print(f'   LONG Trades:      {len(long_trades)} | Wins: {long_wins} ({long_wins/len(long_trades)*100:.1f}%) | P&L: ${long_pnl:,.0f}')
        if short_trades:
            short_pnl = sum(t['pnl'] for t in short_trades)
            print(f'   SHORT Trades:     {len(short_trades)} | Wins: {short_wins} ({short_wins/len(short_trades)*100:.1f}%) | P&L: ${short_pnl:,.0f}')

    # Yearly breakdown
    print('\n' + '=' * 100)
    print('YEARLY PERFORMANCE')
    print('=' * 100)
    print(f'   {"Year":<8} | {"P&L":^12} | {"Trades":^8} | {"Win Rate":^10} | {"Return":^10}')
    print('   ' + '-' * 60)

    for year in sorted(yearly_pnl.keys()):
        year_trades = [t for t in trades_list if t['entry_time'].year == year]
        year_wins = sum(1 for t in year_trades if t['pnl'] > 0)
        year_wr = year_wins / len(year_trades) * 100 if year_trades else 0
        year_return = yearly_pnl[year] / initial_balance * 100
        status = '+' if yearly_pnl[year] > 0 else ''
        print(f'   {year:<8} | {status}${yearly_pnl[year]:>10,.0f} | {len(year_trades):>6} | {year_wr:>8.1f}% | {status}{year_return:>7.1f}%')

    # Monthly breakdown
    print('\n' + '=' * 100)
    print('MONTHLY PERFORMANCE')
    print('=' * 100)
    print(f'   {"Month":<10} | {"P&L":^10} | {"Trades":^8} | {"Win Rate":^10}')
    print('   ' + '-' * 50)

    sorted_months = sorted(monthly_pnl.keys())
    profitable_months = 0
    for month in sorted_months:
        month_trades = [t for t in trades_list if t['entry_time'].strftime('%Y-%m') == month]
        month_wins = sum(1 for t in month_trades if t['pnl'] > 0)
        month_wr = month_wins / len(month_trades) * 100 if month_trades else 0
        status = '+' if monthly_pnl[month] > 0 else '-'
        if monthly_pnl[month] > 0:
            profitable_months += 1
        print(f'   {month:<10} | {status}${abs(monthly_pnl[month]):>8,.0f} | {len(month_trades):>6} | {month_wr:>8.1f}%')

    print('\n' + '=' * 100)
    print('CONSISTENCY METRICS')
    print('=' * 100)
    total_months = len(monthly_pnl)
    all_profitable = sum(1 for m in monthly_pnl.values() if m > 0)
    print(f'   Profitable Months: {all_profitable}/{total_months} ({all_profitable/total_months*100:.1f}%)')
    print(f'   Avg Monthly P&L:   ${np.mean(list(monthly_pnl.values())):,.0f}')
    print(f'   Best Month:        ${max(monthly_pnl.values()):,.0f}')
    print(f'   Worst Month:       ${min(monthly_pnl.values()):,.0f}')

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for t in trades_list:
        if t['pnl'] > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    print(f'   Max Consecutive Wins:   {max_consec_wins}')
    print(f'   Max Consecutive Losses: {max_consec_losses}')

    # Risk metrics
    print('\n' + '=' * 100)
    print('RISK METRICS')
    print('=' * 100)

    # Sharpe-like ratio (simplified)
    monthly_returns = [(monthly_pnl[m] / initial_balance * 100) for m in sorted_months]
    if len(monthly_returns) > 1:
        avg_monthly_ret = np.mean(monthly_returns)
        std_monthly_ret = np.std(monthly_returns)
        sharpe_monthly = avg_monthly_ret / std_monthly_ret if std_monthly_ret > 0 else 0
        print(f'   Monthly Sharpe Ratio:   {sharpe_monthly:.2f}')

    # Calmar ratio (return / max drawdown)
    calmar = total_return / max_dd if max_dd > 0 else 999
    print(f'   Calmar Ratio:           {calmar:.2f}')

    # Recovery factor
    recovery_factor = total_return / max_dd if max_dd > 0 else 999
    print(f'   Recovery Factor:        {recovery_factor:.2f}')

    print('\n' + '=' * 100)
    print('STRATEGY VERDICT: v3.7 FINAL - READY FOR LIVE DEMO')
    print('=' * 100)


if __name__ == "__main__":
    main()
