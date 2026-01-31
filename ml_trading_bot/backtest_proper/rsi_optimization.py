"""
RSI Parameter Optimization
==========================

Optimize RSI strategy parameters:
1. RSI Threshold (oversold/overbought levels)
2. Session Hours (start/end)
3. ATR Multipliers (SL/TP)
4. RSI Period

Database: ml_trading_bot PostgreSQL
Symbol: GBPUSD H1
Period: 2020-2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from itertools import product
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


def calculate_indicators(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
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

    # BB Middle (for mean exit)
    df['bb_mid'] = df['close'].rolling(20).mean()

    # Hour
    df['hour'] = df.index.hour

    return df.fillna(method='ffill').fillna(0)


def run_backtest(
    df: pd.DataFrame,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    session_start: int = 7,
    session_end: int = 19,
    sl_mult: float = 1.75,
    tp_mult: float = 2.5,
    use_mean_exit: bool = True
) -> dict:
    """Run RSI backtest with given parameters"""

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
                    trades.append({'pnl': pnl, 'result': 'SL', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 3
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2
                elif use_mean_exit and row['high'] >= row['bb_mid']:
                    pnl = (row['bb_mid'] - position['entry']) * position['size']
                    balance += pnl
                    result = 'MEAN_WIN' if pnl > 0 else 'MEAN_LOSS'
                    trades.append({'pnl': pnl, 'result': result, 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2
            else:  # SHORT
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 3
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2
                elif use_mean_exit and row['low'] <= row['bb_mid']:
                    pnl = (position['entry'] - row['bb_mid']) * position['size']
                    balance += pnl
                    result = 'MEAN_WIN' if pnl > 0 else 'MEAN_LOSS'
                    trades.append({'pnl': pnl, 'result': result, 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2

        # Check for new signal
        if position is None:
            # Session filter
            if hour < session_start or hour >= session_end:
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
        trades.append({'pnl': pnl, 'result': 'CLOSE', 'entry_time': position['entry_time']})

    # Calculate stats
    if len(trades) == 0:
        return {'trades': 0, 'return_pct': 0, 'win_rate': 0, 'profit_factor': 0}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0

    # Calculate max drawdown
    cumulative = [10000]
    for t in trades:
        cumulative.append(cumulative[-1] + t['pnl'])
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak * 100
    max_dd = max(drawdown)

    return {
        'trades': len(trades_df),
        'wins': len(wins),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'return_pct': (balance - 10000) / 10000 * 100,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'final_balance': balance
    }


def main():
    print("=" * 90)
    print("RSI PARAMETER OPTIMIZATION")
    print("=" * 90)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_data('2020-01-01', '2026-01-31')
    print(f"      Loaded {len(df):,} bars")

    # Calculate base indicators
    print("\n[2/4] Calculating indicators...")
    df = calculate_indicators(df)

    # Define parameter ranges
    print("\n[3/4] Running optimization...")

    # Parameter grid
    rsi_levels = [
        (25, 75),  # More extreme
        (30, 70),  # Standard (baseline)
        (35, 65),  # Less extreme
        (20, 80),  # Very extreme
    ]

    session_ranges = [
        (7, 19),   # London full (baseline)
        (8, 17),   # Core hours
        (7, 17),   # London only
        (8, 12),   # Morning only
        (13, 19),  # Afternoon only
        (6, 20),   # Extended
    ]

    atr_mults = [
        (1.5, 2.0),   # Tighter
        (1.75, 2.5),  # Standard (baseline)
        (2.0, 3.0),   # Wider
        (1.5, 3.0),   # Tight SL, Wide TP
    ]

    mean_exit_options = [True, False]

    results = []
    total_combinations = len(rsi_levels) * len(session_ranges) * len(atr_mults) * len(mean_exit_options)
    current = 0

    for (rsi_os, rsi_ob) in rsi_levels:
        for (sess_start, sess_end) in session_ranges:
            for (sl_m, tp_m) in atr_mults:
                for mean_exit in mean_exit_options:
                    current += 1
                    if current % 20 == 0:
                        print(f"      Progress: {current}/{total_combinations}")

                    result = run_backtest(
                        df,
                        rsi_oversold=rsi_os,
                        rsi_overbought=rsi_ob,
                        session_start=sess_start,
                        session_end=sess_end,
                        sl_mult=sl_m,
                        tp_mult=tp_m,
                        use_mean_exit=mean_exit
                    )

                    result['rsi_os'] = rsi_os
                    result['rsi_ob'] = rsi_ob
                    result['session'] = f"{sess_start}-{sess_end}"
                    result['sl_mult'] = sl_m
                    result['tp_mult'] = tp_m
                    result['mean_exit'] = mean_exit

                    results.append(result)

    # Summary
    print("\n[4/4] Analyzing results...")

    # Filter meaningful results (min 100 trades)
    good_results = [r for r in results if r['trades'] >= 100]

    # Sort by return
    good_results = sorted(good_results, key=lambda x: x['return_pct'], reverse=True)

    print("\n" + "=" * 120)
    print("TOP 20 CONFIGURATIONS (sorted by Return)")
    print("=" * 120)
    print(f"{'RSI':>10} {'Session':>10} {'SL/TP':>10} {'MeanExit':>10} {'Trades':>8} {'Return':>10} {'WR':>8} {'PF':>6} {'MaxDD':>8}")
    print("-" * 120)

    for r in good_results[:20]:
        rsi_str = f"{r['rsi_os']}/{r['rsi_ob']}"
        sltp_str = f"{r['sl_mult']}/{r['tp_mult']}"
        mean_str = "Yes" if r['mean_exit'] else "No"
        print(f"{rsi_str:>10} {r['session']:>10} {sltp_str:>10} {mean_str:>10} {r['trades']:>8} {r['return_pct']:>9.1f}% {r['win_rate']:>7.1f}% {r['profit_factor']:>5.2f} {r['max_drawdown']:>7.1f}%")

    # Best configuration
    best = good_results[0]

    print("\n" + "=" * 90)
    print("OPTIMAL CONFIGURATION")
    print("=" * 90)
    print(f"""
  RSI Oversold:     {best['rsi_os']}
  RSI Overbought:   {best['rsi_ob']}
  Session:          {best['session']} UTC
  SL Multiplier:    {best['sl_mult']}x ATR
  TP Multiplier:    {best['tp_mult']}x ATR
  Mean Exit:        {'Yes' if best['mean_exit'] else 'No'}

  RESULTS (2020-2026):
  --------------------
  Trades:           {best['trades']}
  Trades/Year:      {best['trades'] / 6:.1f}
  Total Return:     {best['return_pct']:.2f}%
  Annual Return:    {best['return_pct'] / 6:.2f}%
  Win Rate:         {best['win_rate']:.1f}%
  Profit Factor:    {best['profit_factor']:.2f}
  Max Drawdown:     {best['max_drawdown']:.1f}%
  Final Balance:    ${best['final_balance']:,.2f}
""")

    # Compare with baseline
    baseline = next((r for r in results if r['rsi_os'] == 30 and r['rsi_ob'] == 70 and r['session'] == '7-19' and r['sl_mult'] == 1.75 and r['tp_mult'] == 2.5 and r['mean_exit'] == True), None)

    if baseline:
        print("\n" + "=" * 90)
        print("COMPARISON: OPTIMAL vs BASELINE")
        print("=" * 90)
        print(f"""
                            OPTIMAL         BASELINE
  -------------------------------------------------
  RSI Levels:               {best['rsi_os']}/{best['rsi_ob']}            30/70
  Session:                  {best['session']}           7-19
  SL/TP:                    {best['sl_mult']}/{best['tp_mult']}           1.75/2.5
  Mean Exit:                {'Yes' if best['mean_exit'] else 'No'}              Yes

  Trades:                   {best['trades']}             {baseline['trades']}
  Return:                   {best['return_pct']:.1f}%           {baseline['return_pct']:.1f}%
  Win Rate:                 {best['win_rate']:.1f}%           {baseline['win_rate']:.1f}%
  Profit Factor:            {best['profit_factor']:.2f}            {baseline['profit_factor']:.2f}
  Max Drawdown:             {best['max_drawdown']:.1f}%           {baseline['max_drawdown']:.1f}%

  Improvement:              {best['return_pct'] - baseline['return_pct']:+.1f}%
""")

    # RSI Period optimization
    print("\n" + "=" * 90)
    print("RSI PERIOD OPTIMIZATION")
    print("=" * 90)

    for period in [7, 10, 14, 21, 28]:
        df_temp = load_data('2020-01-01', '2026-01-31')
        df_temp = calculate_indicators(df_temp, rsi_period=period)

        result = run_backtest(
            df_temp,
            rsi_oversold=best['rsi_os'],
            rsi_overbought=best['rsi_ob'],
            session_start=int(best['session'].split('-')[0]),
            session_end=int(best['session'].split('-')[1]),
            sl_mult=best['sl_mult'],
            tp_mult=best['tp_mult'],
            use_mean_exit=best['mean_exit']
        )

        print(f"  RSI({period:2d}): {result['trades']:4d} trades, {result['return_pct']:6.1f}%, WR {result['win_rate']:.1f}%, PF {result['profit_factor']:.2f}")

    # Save optimal config
    print("\n" + "=" * 90)
    print("SAVING OPTIMAL CONFIGURATION")
    print("=" * 90)

    config = f"""
# RSI Strategy Optimal Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Backtest Period: 2020-2026 (6 years)

RSI_PERIOD = 14
RSI_OVERSOLD = {best['rsi_os']}
RSI_OVERBOUGHT = {best['rsi_ob']}

SESSION_START = {int(best['session'].split('-')[0])}  # UTC
SESSION_END = {int(best['session'].split('-')[1])}    # UTC

SL_ATR_MULT = {best['sl_mult']}
TP_ATR_MULT = {best['tp_mult']}

USE_MEAN_EXIT = {best['mean_exit']}

RISK_PER_TRADE = 0.01  # 1%

# Expected Performance:
# - Trades/Year: {best['trades'] / 6:.0f}
# - Annual Return: {best['return_pct'] / 6:.1f}%
# - Win Rate: {best['win_rate']:.1f}%
# - Max Drawdown: {best['max_drawdown']:.1f}%
"""
    print(config)


if __name__ == "__main__":
    main()
