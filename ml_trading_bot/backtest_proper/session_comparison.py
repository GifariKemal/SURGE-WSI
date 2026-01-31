"""
Session Comparison Backtest
===========================

Compare RSI performance across different trading sessions:
- Asian Session (Tokyo/Sydney)
- London Session
- NY Session
- London+NY Overlap
- 24 Hours (no filter)

Using OPTIMIZED RSI parameters:
- RSI(10) with 35/65 thresholds
- SL: 1.5x ATR, TP: 3.0x ATR
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV data"""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= %s AND time <= %s
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


def calculate_indicators(df: pd.DataFrame, rsi_period: int = 10) -> pd.DataFrame:
    """Calculate RSI and ATR"""
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = tr.rolling(14).mean()

    # BB Middle
    df['bb_mid'] = df['close'].rolling(20).mean()

    # Hour
    df['hour'] = df.index.hour

    return df.ffill().fillna(0)


def check_session(hour: int, session_start: int, session_end: int) -> bool:
    """Check if hour is in session (handles overnight sessions)"""
    if session_start < session_end:
        # Normal session (e.g., 7-19)
        return session_start <= hour < session_end
    else:
        # Overnight session (e.g., 22-6)
        return hour >= session_start or hour < session_end


def run_backtest(
    df: pd.DataFrame,
    session_start: int,
    session_end: int,
    rsi_oversold: float = 35,
    rsi_overbought: float = 65,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
) -> dict:
    """Run RSI backtest with given session"""

    balance = 10000.0
    trades = []
    position = None
    cooldown = 0

    for i in range(50, len(df)):
        if cooldown > 0:
            cooldown -= 1
            continue

        row = df.iloc[i]
        hour = row['hour']

        # Manage position
        if position is not None:
            if position['direction'] == 1:  # LONG
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({
                        'pnl': pnl,
                        'result': 'SL',
                        'entry_time': position['entry_time'],
                        'hour': position['entry_hour']
                    })
                    position = None
                    cooldown = 3
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({
                        'pnl': pnl,
                        'result': 'TP',
                        'entry_time': position['entry_time'],
                        'hour': position['entry_hour']
                    })
                    position = None
                    cooldown = 2
                elif row['high'] >= row['bb_mid']:
                    pnl = (row['bb_mid'] - position['entry']) * position['size']
                    balance += pnl
                    result = 'MEAN_WIN' if pnl > 0 else 'MEAN_LOSS'
                    trades.append({
                        'pnl': pnl,
                        'result': result,
                        'entry_time': position['entry_time'],
                        'hour': position['entry_hour']
                    })
                    position = None
                    cooldown = 2
            else:  # SHORT
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    trades.append({
                        'pnl': pnl,
                        'result': 'SL',
                        'entry_time': position['entry_time'],
                        'hour': position['entry_hour']
                    })
                    position = None
                    cooldown = 3
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    trades.append({
                        'pnl': pnl,
                        'result': 'TP',
                        'entry_time': position['entry_time'],
                        'hour': position['entry_hour']
                    })
                    position = None
                    cooldown = 2
                elif row['low'] <= row['bb_mid']:
                    pnl = (position['entry'] - row['bb_mid']) * position['size']
                    balance += pnl
                    result = 'MEAN_WIN' if pnl > 0 else 'MEAN_LOSS'
                    trades.append({
                        'pnl': pnl,
                        'result': result,
                        'entry_time': position['entry_time'],
                        'hour': position['entry_hour']
                    })
                    position = None
                    cooldown = 2

        # Check for new signal
        if position is None:
            # Session filter
            if not check_session(hour, session_start, session_end):
                continue

            rsi = row['rsi']
            signal = 0

            if rsi < rsi_oversold:
                signal = 1  # BUY
            elif rsi > rsi_overbought:
                signal = -1  # SELL

            if signal != 0:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002

                if signal == 1:
                    sl = entry - atr * sl_mult
                    tp = entry + atr * tp_mult
                else:
                    sl = entry + atr * sl_mult
                    tp = entry - atr * tp_mult

                risk = balance * 0.01
                size = risk / abs(entry - sl) if abs(entry - sl) > 0 else 0
                size = min(size, 100000)

                position = {
                    'entry_time': df.index[i],
                    'entry_hour': hour,
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size
                }

    # Close remaining
    if position:
        final = df.iloc[-1]['close']
        if position['direction'] == 1:
            pnl = (final - position['entry']) * position['size']
        else:
            pnl = (position['entry'] - final) * position['size']
        balance += pnl
        trades.append({
            'pnl': pnl,
            'result': 'CLOSE',
            'entry_time': position['entry_time'],
            'hour': position['entry_hour']
        })

    # Calculate stats
    if len(trades) == 0:
        return {'trades': 0, 'return_pct': 0, 'win_rate': 0, 'profit_factor': 0}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0

    # Max drawdown
    cumulative = [10000]
    for t in trades:
        cumulative.append(cumulative[-1] + t['pnl'])
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak * 100
    max_dd = max(drawdown)

    # Hour distribution
    hour_dist = trades_df.groupby('hour').agg({
        'pnl': ['count', 'sum']
    }).round(2)

    return {
        'trades': len(trades_df),
        'wins': len(wins),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'return_pct': (balance - 10000) / 10000 * 100,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'final_balance': balance,
        'trades_df': trades_df
    }


def main():
    print("=" * 100)
    print("SESSION COMPARISON - RSI OPTIMIZED STRATEGY")
    print("=" * 100)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_data('2020-01-01', '2026-01-31')
    print(f"      Loaded {len(df):,} bars")

    # Calculate indicators
    print("\n[2/3] Calculating indicators (RSI 10)...")
    df = calculate_indicators(df, rsi_period=10)

    # Define sessions to test
    sessions = [
        ("Asian (Tokyo)", 0, 8),
        ("Asian Extended", 0, 9),
        ("London Only", 7, 16),
        ("London Full", 7, 19),
        ("NY Overlap", 13, 17),
        ("NY Session", 13, 22),
        ("NY Extended", 12, 22),
        ("London+NY", 7, 22),
        ("Extended All", 6, 23),
        ("24 Hours", 0, 24),
        ("Night (Asian+Late NY)", 22, 8),  # Overnight
    ]

    print("\n[3/3] Running backtests...")
    results = []

    for name, start, end in sessions:
        result = run_backtest(df, session_start=start, session_end=end)
        result['session'] = name
        result['hours'] = f"{start:02d}:00-{end:02d}:00"
        results.append(result)
        print(f"      {name}: {result['trades']} trades, {result['return_pct']:.1f}%")

    # Sort by return
    results = sorted(results, key=lambda x: x['return_pct'], reverse=True)

    # Display results
    print("\n" + "=" * 100)
    print("SESSION COMPARISON RESULTS (sorted by Return)")
    print("=" * 100)
    print(f"{'Session':<25} {'Hours':<15} {'Trades':>8} {'Return':>12} {'WR':>8} {'PF':>8} {'MaxDD':>8}")
    print("-" * 100)

    for r in results:
        print(f"{r['session']:<25} {r['hours']:<15} {r['trades']:>8} {r['return_pct']:>+11.1f}% {r['win_rate']:>7.1f}% {r['profit_factor']:>7.2f} {r['max_drawdown']:>7.1f}%")

    # Best session
    best = results[0]

    print("\n" + "=" * 100)
    print("BEST SESSION")
    print("=" * 100)
    print(f"""
  Session:        {best['session']}
  Hours:          {best['hours']} UTC

  Performance (2020-2026):
  ------------------------
  Trades:         {best['trades']}
  Trades/Year:    {best['trades'] / 6:.1f}
  Total Return:   {best['return_pct']:.1f}%
  Annual Return:  {best['return_pct'] / 6:.1f}%
  Win Rate:       {best['win_rate']:.1f}%
  Profit Factor:  {best['profit_factor']:.2f}
  Max Drawdown:   {best['max_drawdown']:.1f}%
""")

    # Analyze by hour
    print("\n" + "=" * 100)
    print("PROFIT BY HOUR (24h analysis)")
    print("=" * 100)

    # Run 24h backtest and analyze
    result_24h = run_backtest(df, session_start=0, session_end=24)
    trades_df = result_24h['trades_df']

    hour_stats = trades_df.groupby('hour').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    hour_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
    hour_stats = hour_stats.sort_values('total_pnl', ascending=False)

    print(f"\n{'Hour':<8} {'Trades':>8} {'Total P/L':>15} {'Avg P/L':>12} {'Rating':<10}")
    print("-" * 60)

    for hour, row in hour_stats.iterrows():
        rating = "+++" if row['total_pnl'] > 1000 else "++" if row['total_pnl'] > 500 else "+" if row['total_pnl'] > 0 else "-" if row['total_pnl'] > -500 else "--"
        print(f"{hour:02d}:00   {int(row['trades']):>8} {row['total_pnl']:>+14.2f}$ {row['avg_pnl']:>+11.2f}$ {rating:<10}")

    # Profitable hours
    profitable_hours = hour_stats[hour_stats['total_pnl'] > 0].index.tolist()
    losing_hours = hour_stats[hour_stats['total_pnl'] <= 0].index.tolist()

    print(f"\nProfitable Hours: {sorted(profitable_hours)}")
    print(f"Losing Hours:     {sorted(losing_hours)}")

    # Recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    # Find optimal custom session based on profitable hours
    if profitable_hours:
        custom_start = min(profitable_hours)
        custom_end = max(profitable_hours) + 1

        print(f"""
  Based on hourly analysis:

  CURRENT:    07:00-19:00 UTC (London)
  OPTIMAL:    {custom_start:02d}:00-{custom_end:02d}:00 UTC (Custom)

  Best performing hours: {sorted(profitable_hours[:5])}
  Worst performing hours: {sorted(losing_hours[:3]) if losing_hours else 'None'}
""")


if __name__ == "__main__":
    main()
