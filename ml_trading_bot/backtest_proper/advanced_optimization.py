"""
Advanced RSI Optimization
=========================

Test additional filters to improve performance:
1. Day of Week filter
2. RSI Confirmation (consecutive bars)
3. Trend Filter (SMA 200)
4. Volatility Filter (ATR percentile)
5. Hour of Day analysis

Base: RSI(10) 35/65, Session 07-22, SL 1.5x, TP 3.0x = +651.9%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}


def load_data() -> pd.DataFrame:
    """Load OHLCV data"""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators"""
    df = df.copy()

    # RSI(10)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR(14)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = tr.rolling(14).mean()

    # Trend indicators
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['trend'] = np.where(df['close'] > df['sma_200'], 1, -1)  # 1=bullish, -1=bearish

    # Volatility percentile (ATR relative to last 100 bars)
    df['atr_percentile'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )

    # RSI previous bar (for confirmation)
    df['rsi_prev'] = df['rsi'].shift(1)
    df['rsi_prev2'] = df['rsi'].shift(2)

    # Time features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday  # 0=Monday, 4=Friday
    df['month'] = df.index.month

    return df.ffill().fillna(0)


def run_backtest(
    df: pd.DataFrame,
    rsi_oversold: float = 35,
    rsi_overbought: float = 65,
    session_start: int = 7,
    session_end: int = 22,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    # Additional filters
    allowed_days: list = None,  # None = all days, [0,1,2,3,4] = Mon-Fri
    rsi_confirmation: int = 1,  # Bars RSI must stay in zone
    use_trend_filter: bool = False,  # Only trade with trend
    min_atr_percentile: float = 0,  # Min ATR percentile (0-100)
    max_atr_percentile: float = 100,  # Max ATR percentile
    allowed_hours: list = None,  # Specific hours only
) -> dict:
    """Run backtest with filters"""

    balance = 10000.0
    trades = []
    position = None

    for i in range(200, len(df)):
        row = df.iloc[i]
        hour = row['hour']
        weekday = row['weekday']

        # Skip weekends
        if weekday >= 5:
            continue

        # Manage position
        if position is not None:
            current_price = row['close']
            pip_value = 0.0001

            if position['direction'] == 1:  # LONG
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL', 'weekday': position['weekday'], 'hour': position['hour']})
                    position = None
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'weekday': position['weekday'], 'hour': position['hour']})
                    position = None
            else:  # SHORT
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL', 'weekday': position['weekday'], 'hour': position['hour']})
                    position = None
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'weekday': position['weekday'], 'hour': position['hour']})
                    position = None

        # Check for new signal
        if position is None:
            # Session filter
            if hour < session_start or hour >= session_end:
                continue

            # Day filter
            if allowed_days is not None and weekday not in allowed_days:
                continue

            # Hour filter
            if allowed_hours is not None and hour not in allowed_hours:
                continue

            # ATR percentile filter
            atr_pct = row['atr_percentile']
            if atr_pct < min_atr_percentile or atr_pct > max_atr_percentile:
                continue

            rsi = row['rsi']
            signal = 0

            # RSI signal with confirmation
            if rsi < rsi_oversold:
                # Check confirmation (RSI was also oversold in previous bars)
                if rsi_confirmation > 1:
                    confirmed = True
                    for j in range(1, rsi_confirmation):
                        if df.iloc[i-j]['rsi'] >= rsi_oversold:
                            confirmed = False
                            break
                    if not confirmed:
                        continue
                signal = 1  # BUY

            elif rsi > rsi_overbought:
                if rsi_confirmation > 1:
                    confirmed = True
                    for j in range(1, rsi_confirmation):
                        if df.iloc[i-j]['rsi'] <= rsi_overbought:
                            confirmed = False
                            break
                    if not confirmed:
                        continue
                signal = -1  # SELL

            # Trend filter
            if use_trend_filter and signal != 0:
                trend = row['trend']
                if signal == 1 and trend == -1:  # BUY but downtrend
                    continue
                if signal == -1 and trend == 1:  # SELL but uptrend
                    continue

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
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'weekday': weekday,
                    'hour': hour
                }

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

    return {
        'trades': len(trades_df),
        'wins': len(wins),
        'win_rate': len(wins) / len(trades_df) * 100,
        'return_pct': (balance - 10000) / 10000 * 100,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'final_balance': balance,
        'trades_df': trades_df
    }


def main():
    print("=" * 100)
    print("ADVANCED RSI OPTIMIZATION")
    print("=" * 100)

    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"      Loaded {len(df):,} bars")

    print("\n[2/5] Calculating indicators...")
    df = calculate_indicators(df)

    # Baseline
    print("\n[3/5] Running baseline...")
    baseline = run_backtest(df)
    print(f"      BASELINE: {baseline['trades']} trades, {baseline['return_pct']:.1f}%")

    # =========================================================================
    # TEST 1: Day of Week
    # =========================================================================
    print("\n[4/5] Testing filters...")
    print("\n" + "-" * 80)
    print("TEST 1: DAY OF WEEK")
    print("-" * 80)

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_results = []

    for d in range(5):
        result = run_backtest(df, allowed_days=[d])
        day_results.append({
            'day': days[d],
            'trades': result['trades'],
            'return': result['return_pct'],
            'wr': result['win_rate'],
            'pf': result['profit_factor']
        })
        print(f"      {days[d]}: {result['trades']} trades, {result['return_pct']:+.1f}%, WR {result['win_rate']:.1f}%")

    # Find best days
    day_df = pd.DataFrame(day_results)
    best_days = day_df[day_df['return'] > 0].sort_values('return', ascending=False)

    if len(best_days) > 0:
        top_days = best_days['day'].tolist()[:3]
        top_day_indices = [days.index(d) for d in top_days]
        best_days_result = run_backtest(df, allowed_days=top_day_indices)
        print(f"\n      Best days ({', '.join(top_days)}): {best_days_result['return_pct']:+.1f}%")

    # =========================================================================
    # TEST 2: RSI Confirmation
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: RSI CONFIRMATION (consecutive bars)")
    print("-" * 80)

    for confirm in [1, 2, 3]:
        result = run_backtest(df, rsi_confirmation=confirm)
        diff = result['return_pct'] - baseline['return_pct']
        print(f"      {confirm} bar(s): {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}% vs baseline)")

    # =========================================================================
    # TEST 3: Trend Filter
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 3: TREND FILTER (trade with SMA 200 direction)")
    print("-" * 80)

    result_trend = run_backtest(df, use_trend_filter=True)
    diff = result_trend['return_pct'] - baseline['return_pct']
    print(f"      With Trend: {result_trend['trades']} trades, {result_trend['return_pct']:+.1f}% ({diff:+.1f}% vs baseline)")
    print(f"      No Filter:  {baseline['trades']} trades, {baseline['return_pct']:+.1f}%")

    # =========================================================================
    # TEST 4: Volatility Filter
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 4: VOLATILITY FILTER (ATR percentile)")
    print("-" * 80)

    vol_configs = [
        (0, 100, "All volatility"),
        (20, 100, "Skip low vol (>20th pct)"),
        (0, 80, "Skip high vol (<80th pct)"),
        (20, 80, "Medium volatility only"),
        (30, 70, "Narrow range (30-70)"),
        (50, 100, "High volatility only (>50th)"),
    ]

    for min_pct, max_pct, desc in vol_configs:
        result = run_backtest(df, min_atr_percentile=min_pct, max_atr_percentile=max_pct)
        diff = result['return_pct'] - baseline['return_pct']
        print(f"      {desc}: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")

    # =========================================================================
    # TEST 5: Hour Analysis
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 5: BEST HOURS (within 07-22 session)")
    print("-" * 80)

    hour_results = []
    for h in range(7, 22):
        result = run_backtest(df, allowed_hours=[h])
        hour_results.append({
            'hour': h,
            'trades': result['trades'],
            'return': result['return_pct'],
            'wr': result['win_rate'],
            'pf': result['profit_factor']
        })

    hour_df = pd.DataFrame(hour_results).sort_values('return', ascending=False)
    print("\n      Top 5 hours:")
    for _, row in hour_df.head(5).iterrows():
        print(f"        {int(row['hour']):02d}:00 - {row['trades']} trades, {row['return']:+.1f}%, WR {row['wr']:.1f}%")

    print("\n      Bottom 3 hours:")
    for _, row in hour_df.tail(3).iterrows():
        print(f"        {int(row['hour']):02d}:00 - {row['trades']} trades, {row['return']:+.1f}%, WR {row['wr']:.1f}%")

    # Test best hours combined
    best_hours = hour_df[hour_df['return'] > baseline['return_pct'] / 15]['hour'].tolist()  # Hours with good return
    if len(best_hours) >= 3:
        best_hours_int = [int(h) for h in best_hours[:8]]
        result_best_hours = run_backtest(df, allowed_hours=best_hours_int)
        print(f"\n      Best hours combined ({best_hours_int}): {result_best_hours['return_pct']:+.1f}%")

    # =========================================================================
    # TEST 6: Combined Best Filters
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 6: COMBINED OPTIMIZATIONS")
    print("-" * 80)

    # Try combining the best performing filters
    combinations = [
        {'name': 'Baseline', 'params': {}},
        {'name': '+ Best Days (Mon-Thu)', 'params': {'allowed_days': [0, 1, 2, 3]}},
        {'name': '+ Skip Friday', 'params': {'allowed_days': [0, 1, 2, 3]}},
        {'name': '+ Trend Filter', 'params': {'use_trend_filter': True}},
        {'name': '+ Medium Vol (20-80)', 'params': {'min_atr_percentile': 20, 'max_atr_percentile': 80}},
        {'name': 'Best Days + Trend', 'params': {'allowed_days': [0, 1, 2, 3], 'use_trend_filter': True}},
        {'name': 'Best Days + Med Vol', 'params': {'allowed_days': [0, 1, 2, 3], 'min_atr_percentile': 20, 'max_atr_percentile': 80}},
        {'name': 'All Combined', 'params': {'allowed_days': [0, 1, 2, 3], 'use_trend_filter': True, 'min_atr_percentile': 20, 'max_atr_percentile': 80}},
    ]

    print(f"\n{'Configuration':<30} {'Trades':>8} {'Return':>12} {'WR':>8} {'PF':>8} {'MaxDD':>8}")
    print("-" * 80)

    best_combo = None
    best_return = baseline['return_pct']

    for combo in combinations:
        result = run_backtest(df, **combo['params'])
        diff = result['return_pct'] - baseline['return_pct']

        marker = ""
        if result['return_pct'] > best_return:
            best_return = result['return_pct']
            best_combo = combo
            marker = " <-- BEST"

        print(f"{combo['name']:<30} {result['trades']:>8} {result['return_pct']:>+10.1f}% {result['win_rate']:>7.1f}% {result['profit_factor']:>7.2f} {result['max_drawdown']:>7.1f}%{marker}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("OPTIMIZATION SUMMARY")
    print("=" * 100)

    print(f"\nBASELINE: {baseline['return_pct']:+.1f}% ({baseline['trades']} trades)")

    if best_combo and best_return > baseline['return_pct']:
        improvement = best_return - baseline['return_pct']
        print(f"BEST:     {best_return:+.1f}% ({best_combo['name']}) -> +{improvement:.1f}% improvement")
        print(f"\nOptimal filters: {best_combo['params']}")
    else:
        print("\nNo filter combination improved baseline significantly.")
        print("Current configuration is already near-optimal.")

    # Individual filter summary
    print("\n" + "-" * 60)
    print("INDIVIDUAL FILTER IMPACT:")
    print("-" * 60)

    filters_impact = [
        ("Day of Week (skip Friday)", run_backtest(df, allowed_days=[0,1,2,3])['return_pct'] - baseline['return_pct']),
        ("RSI 2-bar Confirmation", run_backtest(df, rsi_confirmation=2)['return_pct'] - baseline['return_pct']),
        ("Trend Filter (SMA 200)", result_trend['return_pct'] - baseline['return_pct']),
        ("Medium Volatility (20-80)", run_backtest(df, min_atr_percentile=20, max_atr_percentile=80)['return_pct'] - baseline['return_pct']),
    ]

    for name, impact in sorted(filters_impact, key=lambda x: x[1], reverse=True):
        status = "IMPROVES" if impact > 0 else "HURTS" if impact < -10 else "NEUTRAL"
        print(f"  {name:<35} {impact:+.1f}% [{status}]")


if __name__ == "__main__":
    main()
