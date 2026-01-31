"""
Feature Comparison Backtest
============================

Compare RSI performance with different features:
1. Baseline (no features)
2. + Trailing Stop
3. + Break Even
4. + Spread Filter
5. + News Filter
6. All features combined

Using OPTIMIZED RSI parameters:
- RSI(10) with 35/65 thresholds
- SL: 1.5x ATR, TP: 3.0x ATR
- Session: 07:00-22:00 UTC
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

# NFP dates (1st Friday of each month, 13:30 UTC) - simplified for backtest
def get_nfp_dates(start_year, end_year):
    """Generate NFP dates for given years"""
    nfp_dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Find first Friday of month
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            # NFP at 13:30 UTC
            nfp_time = first_friday.replace(hour=13, minute=30)
            nfp_dates.append(nfp_time)
    return nfp_dates

# FOMC dates (approximate - 8 per year)
FOMC_DATES = []  # Would be populated with actual dates


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

    # Hour and day info
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    # Simulate spread (based on hour - wider during Asian, tighter during London)
    df['spread'] = df['hour'].apply(lambda h:
        0.5 if 7 <= h < 16 else  # London: tight
        0.8 if 13 <= h < 20 else  # NY overlap: medium
        1.5 if 0 <= h < 7 else  # Asian: wide
        2.0  # Late night: very wide
    )

    return df.ffill().fillna(0)


def is_nfp_window(dt, nfp_dates, minutes_before=30, minutes_after=30):
    """Check if datetime is within NFP window"""
    # Convert to naive datetime if timezone-aware (handles both pandas Timestamp and datetime)
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    for nfp in nfp_dates:
        window_start = nfp - timedelta(minutes=minutes_before)
        window_end = nfp + timedelta(minutes=minutes_after)
        if window_start <= dt <= window_end:
            return True
    return False


def run_backtest(
    df: pd.DataFrame,
    # Base parameters
    rsi_oversold: float = 35,
    rsi_overbought: float = 65,
    session_start: int = 7,
    session_end: int = 22,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    # Feature flags
    use_trailing_stop: bool = False,
    trailing_start_pips: float = 20,
    trailing_step_pips: float = 10,
    use_break_even: bool = False,
    break_even_pips: float = 15,
    break_even_offset: float = 2,
    use_spread_filter: bool = False,
    max_spread_pips: float = 3.0,
    use_news_filter: bool = False,
    news_minutes_before: int = 30,
    news_minutes_after: int = 30,
    nfp_dates: list = None,
) -> dict:
    """Run RSI backtest with optional features"""

    balance = 10000.0
    trades = []
    position = None
    cooldown = 0

    # Track for trailing stop
    highest_price = 0
    break_even_applied = False

    for i in range(50, len(df)):
        if cooldown > 0:
            cooldown -= 1
            continue

        row = df.iloc[i]
        current_time = df.index[i]
        hour = row['hour']
        weekday = row['weekday']

        # Skip weekends
        if weekday >= 5:
            continue

        # Manage position
        if position is not None:
            current_price = row['close']
            pip_value = 0.0001

            # Calculate current profit in pips
            if position['direction'] == 1:  # LONG
                profit_pips = (current_price - position['entry']) / pip_value

                # Track highest for trailing
                if current_price > highest_price:
                    highest_price = current_price

                # Check Break Even
                if use_break_even and not break_even_applied:
                    if profit_pips >= break_even_pips:
                        new_sl = position['entry'] + (break_even_offset * pip_value)
                        if new_sl > position['sl']:
                            position['sl'] = new_sl
                            break_even_applied = True

                # Check Trailing Stop
                if use_trailing_stop and profit_pips >= trailing_start_pips:
                    new_sl = highest_price - (trailing_step_pips * pip_value)
                    if new_sl > position['sl']:
                        position['sl'] = new_sl

                # Check exit conditions
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    exit_reason = 'BE' if break_even_applied else ('TRAIL' if position['sl'] > position['original_sl'] else 'SL')
                    trades.append({
                        'pnl': pnl,
                        'result': exit_reason,
                        'entry_time': position['entry_time'],
                        'pips': (position['sl'] - position['entry']) / pip_value
                    })
                    position = None
                    cooldown = 3
                    highest_price = 0
                    break_even_applied = False
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({
                        'pnl': pnl,
                        'result': 'TP',
                        'entry_time': position['entry_time'],
                        'pips': (position['tp'] - position['entry']) / pip_value
                    })
                    position = None
                    cooldown = 2
                    highest_price = 0
                    break_even_applied = False

            else:  # SHORT
                profit_pips = (position['entry'] - current_price) / pip_value

                # Track lowest for trailing (stored as highest)
                if current_price < highest_price or highest_price == position['entry']:
                    highest_price = current_price

                # Check Break Even
                if use_break_even and not break_even_applied:
                    if profit_pips >= break_even_pips:
                        new_sl = position['entry'] - (break_even_offset * pip_value)
                        if new_sl < position['sl']:
                            position['sl'] = new_sl
                            break_even_applied = True

                # Check Trailing Stop
                if use_trailing_stop and profit_pips >= trailing_start_pips:
                    new_sl = highest_price + (trailing_step_pips * pip_value)
                    if new_sl < position['sl']:
                        position['sl'] = new_sl

                # Check exit conditions
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    exit_reason = 'BE' if break_even_applied else ('TRAIL' if position['sl'] < position['original_sl'] else 'SL')
                    trades.append({
                        'pnl': pnl,
                        'result': exit_reason,
                        'entry_time': position['entry_time'],
                        'pips': (position['entry'] - position['sl']) / pip_value
                    })
                    position = None
                    cooldown = 3
                    highest_price = 0
                    break_even_applied = False
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    trades.append({
                        'pnl': pnl,
                        'result': 'TP',
                        'entry_time': position['entry_time'],
                        'pips': (position['entry'] - position['tp']) / pip_value
                    })
                    position = None
                    cooldown = 2
                    highest_price = 0
                    break_even_applied = False

        # Check for new signal
        if position is None:
            # Session filter
            if hour < session_start or hour >= session_end:
                continue

            # News filter
            if use_news_filter and nfp_dates:
                if is_nfp_window(current_time, nfp_dates, news_minutes_before, news_minutes_after):
                    continue

            # Spread filter
            if use_spread_filter:
                if row['spread'] > max_spread_pips:
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
                pip_value = 0.0001

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
                    'entry_time': current_time,
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'original_sl': sl,
                    'tp': tp,
                    'size': size
                }
                highest_price = entry
                break_even_applied = False

    # Close remaining position
    if position:
        final = df.iloc[-1]['close']
        pip_value = 0.0001
        if position['direction'] == 1:
            pnl = (final - position['entry']) * position['size']
            pips = (final - position['entry']) / pip_value
        else:
            pnl = (position['entry'] - final) * position['size']
            pips = (position['entry'] - final) / pip_value
        balance += pnl
        trades.append({'pnl': pnl, 'result': 'CLOSE', 'entry_time': position['entry_time'], 'pips': pips})

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

    # Count exit reasons
    exit_counts = trades_df['result'].value_counts().to_dict()

    return {
        'trades': len(trades_df),
        'wins': len(wins),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'return_pct': (balance - 10000) / 10000 * 100,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'final_balance': balance,
        'avg_pips': trades_df['pips'].mean() if 'pips' in trades_df else 0,
        'exit_counts': exit_counts
    }


def main():
    print("=" * 100)
    print("FEATURE COMPARISON BACKTEST")
    print("=" * 100)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_data('2020-01-01', '2026-01-31')
    print(f"      Loaded {len(df):,} bars")

    # Calculate indicators
    print("\n[2/3] Calculating indicators...")
    df = calculate_indicators(df, rsi_period=10)

    # Generate NFP dates
    nfp_dates = get_nfp_dates(2020, 2026)
    print(f"      Generated {len(nfp_dates)} NFP dates for news filter")

    # Define test configurations
    print("\n[3/3] Running backtests...")

    configs = [
        {
            'name': '1. BASELINE (no features)',
            'params': {}
        },
        {
            'name': '2. + Trailing Stop Only',
            'params': {
                'use_trailing_stop': True,
                'trailing_start_pips': 20,
                'trailing_step_pips': 10,
            }
        },
        {
            'name': '3. + Break Even Only',
            'params': {
                'use_break_even': True,
                'break_even_pips': 15,
                'break_even_offset': 2,
            }
        },
        {
            'name': '4. + Trailing + Break Even',
            'params': {
                'use_trailing_stop': True,
                'trailing_start_pips': 20,
                'trailing_step_pips': 10,
                'use_break_even': True,
                'break_even_pips': 15,
                'break_even_offset': 2,
            }
        },
        {
            'name': '5. + Spread Filter Only',
            'params': {
                'use_spread_filter': True,
                'max_spread_pips': 3.0,
            }
        },
        {
            'name': '6. + News Filter Only',
            'params': {
                'use_news_filter': True,
                'news_minutes_before': 30,
                'news_minutes_after': 30,
            }
        },
        {
            'name': '7. ALL FEATURES',
            'params': {
                'use_trailing_stop': True,
                'trailing_start_pips': 20,
                'trailing_step_pips': 10,
                'use_break_even': True,
                'break_even_pips': 15,
                'break_even_offset': 2,
                'use_spread_filter': True,
                'max_spread_pips': 3.0,
                'use_news_filter': True,
                'news_minutes_before': 30,
                'news_minutes_after': 30,
            }
        },
    ]

    results = []

    for config in configs:
        result = run_backtest(df, nfp_dates=nfp_dates, **config['params'])
        result['name'] = config['name']
        results.append(result)
        print(f"      {config['name']}: {result['trades']} trades, {result['return_pct']:.1f}%")

    # Display results
    print("\n" + "=" * 120)
    print("FEATURE COMPARISON RESULTS")
    print("=" * 120)
    print(f"{'Configuration':<35} {'Trades':>8} {'Return':>12} {'WR':>8} {'PF':>8} {'MaxDD':>8} {'Avg Pips':>10}")
    print("-" * 120)

    baseline = results[0]

    for r in results:
        diff = r['return_pct'] - baseline['return_pct']
        diff_str = f"({diff:+.0f}%)" if r['name'] != baseline['name'] else ""
        print(f"{r['name']:<35} {r['trades']:>8} {r['return_pct']:>+10.1f}% {r['win_rate']:>7.1f}% {r['profit_factor']:>7.2f} {r['max_drawdown']:>7.1f}% {r['avg_pips']:>9.1f} {diff_str}")

    # Analysis
    print("\n" + "=" * 120)
    print("ANALYSIS")
    print("=" * 120)

    # Find best and worst
    sorted_results = sorted(results, key=lambda x: x['return_pct'], reverse=True)
    best = sorted_results[0]
    worst = sorted_results[-1]

    print(f"\nBEST:  {best['name']} -> {best['return_pct']:+.1f}% return")
    print(f"WORST: {worst['name']} -> {worst['return_pct']:+.1f}% return")

    # Compare each feature vs baseline
    print("\n" + "-" * 80)
    print("FEATURE IMPACT vs BASELINE")
    print("-" * 80)

    for r in results[1:]:  # Skip baseline
        diff_return = r['return_pct'] - baseline['return_pct']
        diff_wr = r['win_rate'] - baseline['win_rate']
        diff_dd = r['max_drawdown'] - baseline['max_drawdown']

        verdict = "KEEP" if diff_return > 0 else "REMOVE"
        emoji = "+" if diff_return > 0 else "-"

        print(f"\n{r['name']}")
        print(f"  Return: {diff_return:+.1f}% | WR: {diff_wr:+.1f}% | MaxDD: {diff_dd:+.1f}%")
        print(f"  Verdict: [{verdict}] {emoji}")

    # Recommendation
    print("\n" + "=" * 120)
    print("RECOMMENDATION")
    print("=" * 120)

    # Find features that improve return
    good_features = []
    bad_features = []

    # Check trailing stop (index 1)
    if results[1]['return_pct'] > baseline['return_pct']:
        good_features.append('Trailing Stop')
    else:
        bad_features.append('Trailing Stop')

    # Check break even (index 2)
    if results[2]['return_pct'] > baseline['return_pct']:
        good_features.append('Break Even')
    else:
        bad_features.append('Break Even')

    # Check spread filter (index 4)
    if results[4]['return_pct'] > baseline['return_pct']:
        good_features.append('Spread Filter')
    else:
        bad_features.append('Spread Filter')

    # Check news filter (index 5)
    if results[5]['return_pct'] > baseline['return_pct']:
        good_features.append('News Filter')
    else:
        bad_features.append('News Filter')

    print(f"\nFEATURES TO KEEP:   {', '.join(good_features) if good_features else 'None'}")
    print(f"FEATURES TO REMOVE: {', '.join(bad_features) if bad_features else 'None'}")

    # Exit reason breakdown for all features config
    print("\n" + "-" * 80)
    print("EXIT REASON BREAKDOWN (All Features)")
    print("-" * 80)
    all_features = results[-1]
    for reason, count in all_features.get('exit_counts', {}).items():
        pct = count / all_features['trades'] * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
