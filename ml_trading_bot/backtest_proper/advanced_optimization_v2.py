"""
Advanced RSI Optimization v2
============================

New ideas to test:
1. Month of Year filter (seasonality)
2. RSI Momentum (slope/acceleration)
3. RSI Divergence detection
4. Higher Timeframe confirmation (H4/D1)
5. Candle Pattern filter (confirmation candle)
6. ADX Trend Strength filter
7. Bollinger Band position
8. Multiple RSI levels (extreme zones)
9. Time decay exit (max holding period)
10. Dynamic TP/SL based on volatility regime

Base: RSI v3.2 = +80.3% (with ATR 20-80 filter)
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


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators for testing"""
    df = df.copy()

    # === Basic Indicators ===
    # RSI(10)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # RSI Momentum (slope over 3 bars)
    df['rsi_slope'] = df['rsi'].diff(3)

    # RSI previous values
    df['rsi_prev'] = df['rsi'].shift(1)
    df['rsi_prev2'] = df['rsi'].shift(2)

    # ATR(14)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = tr.rolling(14).mean()

    # ATR Percentile
    df['atr_percentile'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )

    # === Trend Indicators ===
    # SMA
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # EMA
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

    # ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)

    atr_14 = df['atr']
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # === Volatility Indicators ===
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # === Candle Patterns ===
    df['candle_body'] = df['close'] - df['open']
    df['candle_range'] = df['high'] - df['low']
    df['candle_upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['candle_lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Bullish/Bearish candle
    df['is_bullish'] = df['close'] > df['open']
    df['is_bearish'] = df['close'] < df['open']

    # Previous candle
    df['prev_bullish'] = df['is_bullish'].shift(1)
    df['prev_bearish'] = df['is_bearish'].shift(1)

    # Engulfing pattern
    df['bullish_engulfing'] = (
        df['is_bullish'] &
        df['prev_bearish'] &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )
    df['bearish_engulfing'] = (
        df['is_bearish'] &
        df['prev_bullish'] &
        (df['close'] < df['open'].shift(1)) &
        (df['open'] > df['close'].shift(1))
    )

    # Pin bar (hammer/shooting star)
    body_pct = abs(df['candle_body']) / (df['candle_range'] + 1e-10)
    df['is_pin_bar'] = body_pct < 0.3  # Small body
    df['is_hammer'] = df['is_pin_bar'] & (df['candle_lower_wick'] > 2 * abs(df['candle_body']))
    df['is_shooting_star'] = df['is_pin_bar'] & (df['candle_upper_wick'] > 2 * abs(df['candle_body']))

    # === Higher Timeframe (H4 simulation) ===
    # Resample to H4 and merge back
    df['h4_close'] = df['close'].rolling(4).mean()
    df['h4_rsi'] = df['rsi'].rolling(4).mean()

    # === Time Features ===
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['week_of_month'] = (df.index.day - 1) // 7 + 1

    # === Price Levels ===
    # Distance from round numbers
    df['round_dist'] = df['close'].apply(lambda x: min(x % 0.01, 0.01 - x % 0.01) / 0.0001)

    # Recent swing high/low (20 bars)
    df['swing_high_20'] = df['high'].rolling(20).max()
    df['swing_low_20'] = df['low'].rolling(20).min()
    df['near_swing_high'] = (df['swing_high_20'] - df['close']) / df['atr'] < 1
    df['near_swing_low'] = (df['close'] - df['swing_low_20']) / df['atr'] < 1

    return df.ffill().fillna(0)


def run_backtest(
    df: pd.DataFrame,
    # Base parameters (v3.2)
    rsi_oversold: float = 35,
    rsi_overbought: float = 65,
    session_start: int = 7,
    session_end: int = 22,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    min_atr_pct: float = 20,
    max_atr_pct: float = 80,
    # New filters to test
    allowed_months: list = None,
    min_rsi_slope: float = None,  # RSI momentum
    max_rsi_slope: float = None,
    require_confirmation_candle: bool = False,
    require_engulfing: bool = False,
    require_pin_bar: bool = False,
    min_adx: float = None,
    max_adx: float = None,
    min_bb_width: float = None,
    max_bb_width: float = None,
    h4_alignment: bool = False,  # H4 RSI must also be oversold/overbought
    avoid_week1: bool = False,  # Avoid first week of month
    avoid_week4: bool = False,  # Avoid last week of month
    extreme_rsi_only: bool = False,  # RSI < 25 or > 75
    max_hold_bars: int = None,  # Time-based exit
    dynamic_tp: bool = False,  # Adjust TP based on volatility
    near_swing_filter: bool = False,  # Trade near swing levels
) -> dict:
    """Run backtest with new filters"""

    balance = 10000.0
    trades = []
    position = None
    bars_held = 0

    for i in range(200, len(df)):
        row = df.iloc[i]
        hour = row['hour']
        weekday = row['weekday']

        # Skip weekends
        if weekday >= 5:
            continue

        # Manage existing position
        if position is not None:
            bars_held += 1
            current_price = row['close']

            # Time-based exit
            if max_hold_bars and bars_held >= max_hold_bars:
                if position['direction'] == 1:
                    pnl = (current_price - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - current_price) * position['size']
                balance += pnl
                trades.append({
                    'pnl': pnl,
                    'result': 'TIME',
                    'bars': bars_held,
                    'month': position['month']
                })
                position = None
                bars_held = 0
                continue

            if position['direction'] == 1:  # LONG
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL', 'bars': bars_held, 'month': position['month']})
                    position = None
                    bars_held = 0
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'bars': bars_held, 'month': position['month']})
                    position = None
                    bars_held = 0
            else:  # SHORT
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL', 'bars': bars_held, 'month': position['month']})
                    position = None
                    bars_held = 0
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'bars': bars_held, 'month': position['month']})
                    position = None
                    bars_held = 0

        # Check for new signal
        if position is None:
            # Session filter
            if hour < session_start or hour >= session_end:
                continue

            # ATR percentile filter (v3.2 base)
            atr_pct = row['atr_percentile']
            if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                continue

            # Month filter
            if allowed_months is not None and row['month'] not in allowed_months:
                continue

            # Week of month filter
            if avoid_week1 and row['week_of_month'] == 1:
                continue
            if avoid_week4 and row['week_of_month'] >= 4:
                continue

            # ADX filter
            if min_adx is not None and row['adx'] < min_adx:
                continue
            if max_adx is not None and row['adx'] > max_adx:
                continue

            # Bollinger Band width filter
            if min_bb_width is not None and row['bb_width'] < min_bb_width:
                continue
            if max_bb_width is not None and row['bb_width'] > max_bb_width:
                continue

            rsi = row['rsi']
            signal = 0

            # RSI threshold (with optional extreme mode)
            oversold_level = 25 if extreme_rsi_only else rsi_oversold
            overbought_level = 75 if extreme_rsi_only else rsi_overbought

            if rsi < oversold_level:
                signal = 1  # BUY
            elif rsi > overbought_level:
                signal = -1  # SELL

            if signal == 0:
                continue

            # RSI Momentum filter
            if min_rsi_slope is not None:
                if signal == 1 and row['rsi_slope'] < min_rsi_slope:  # BUY needs upward momentum
                    continue
                if signal == -1 and row['rsi_slope'] > -min_rsi_slope:  # SELL needs downward momentum
                    continue

            if max_rsi_slope is not None:
                if abs(row['rsi_slope']) > max_rsi_slope:  # Too much momentum = choppy
                    continue

            # H4 alignment
            if h4_alignment:
                h4_rsi = row['h4_rsi']
                if signal == 1 and h4_rsi >= 40:  # H4 should also be oversold-ish
                    continue
                if signal == -1 and h4_rsi <= 60:  # H4 should also be overbought-ish
                    continue

            # Confirmation candle
            if require_confirmation_candle:
                if signal == 1 and not row['is_bullish']:
                    continue
                if signal == -1 and not row['is_bearish']:
                    continue

            # Engulfing pattern
            if require_engulfing:
                if signal == 1 and not row['bullish_engulfing']:
                    continue
                if signal == -1 and not row['bearish_engulfing']:
                    continue

            # Pin bar pattern
            if require_pin_bar:
                if signal == 1 and not row['is_hammer']:
                    continue
                if signal == -1 and not row['is_shooting_star']:
                    continue

            # Near swing level filter
            if near_swing_filter:
                if signal == 1 and not row['near_swing_low']:
                    continue
                if signal == -1 and not row['near_swing_high']:
                    continue

            # Calculate entry, SL, TP
            entry = row['close']
            atr = row['atr'] if row['atr'] > 0 else entry * 0.002

            # Dynamic TP based on volatility
            actual_tp_mult = tp_mult
            if dynamic_tp:
                if atr_pct < 40:  # Low vol = smaller TP
                    actual_tp_mult = tp_mult * 0.8
                elif atr_pct > 60:  # High vol = larger TP
                    actual_tp_mult = tp_mult * 1.2

            if signal == 1:
                sl = entry - atr * sl_mult
                tp = entry + atr * actual_tp_mult
            else:
                sl = entry + atr * sl_mult
                tp = entry - atr * actual_tp_mult

            risk = balance * 0.01
            size = risk / abs(entry - sl) if abs(entry - sl) > 0 else 0
            size = min(size, 100000)

            position = {
                'direction': signal,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'size': size,
                'month': row['month']
            }
            bars_held = 0

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
    print("ADVANCED RSI OPTIMIZATION v2 - New Ideas")
    print("=" * 100)

    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"      Loaded {len(df):,} bars")

    print("\n[2/6] Calculating indicators...")
    df = calculate_all_indicators(df)

    # Baseline (v3.2 with ATR filter)
    print("\n[3/6] Running v3.2 baseline...")
    baseline = run_backtest(df)
    print(f"      v3.2 BASELINE: {baseline['trades']} trades, +{baseline['return_pct']:.1f}%")

    results = []

    # =========================================================================
    # TEST 1: Month Seasonality
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: MONTH SEASONALITY")
    print("-" * 80)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    month_results = []
    for m in range(1, 13):
        result = run_backtest(df, allowed_months=[m])
        month_results.append({
            'month': month_names[m-1],
            'trades': result['trades'],
            'return': result['return_pct'],
            'wr': result['win_rate']
        })

    month_df = pd.DataFrame(month_results).sort_values('return', ascending=False)
    print("\n      Best months:")
    for _, row in month_df.head(4).iterrows():
        print(f"        {row['month']}: {row['trades']} trades, {row['return']:+.1f}%")
    print("\n      Worst months:")
    for _, row in month_df.tail(3).iterrows():
        print(f"        {row['month']}: {row['trades']} trades, {row['return']:+.1f}%")

    # Test best months combined
    best_months = month_df[month_df['return'] > 0]['month'].tolist()[:8]
    best_month_nums = [month_names.index(m) + 1 for m in best_months]
    if best_month_nums:
        result = run_backtest(df, allowed_months=best_month_nums)
        diff = result['return_pct'] - baseline['return_pct']
        print(f"\n      Best months combined: {result['return_pct']:+.1f}% ({diff:+.1f}% vs baseline)")
        results.append(('Best Months Only', diff, result))

    # =========================================================================
    # TEST 2: RSI Momentum (Slope)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: RSI MOMENTUM (Slope)")
    print("-" * 80)

    momentum_tests = [
        (2, None, "RSI rising (slope > 2)"),
        (5, None, "RSI rising fast (slope > 5)"),
        (None, 10, "Skip choppy (slope < 10)"),
        (2, 15, "Moderate momentum (2-15)"),
    ]

    for min_s, max_s, desc in momentum_tests:
        result = run_backtest(df, min_rsi_slope=min_s, max_rsi_slope=max_s)
        diff = result['return_pct'] - baseline['return_pct']
        print(f"      {desc}: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
        if diff > 0:
            results.append((desc, diff, result))

    # =========================================================================
    # TEST 3: ADX Filter (Trend Strength)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 3: ADX FILTER (Trend Strength)")
    print("-" * 80)

    adx_tests = [
        (None, 25, "Low ADX < 25 (ranging market)"),
        (25, None, "High ADX > 25 (trending)"),
        (15, 30, "Medium ADX 15-30"),
        (None, 20, "Very low ADX < 20"),
        (20, 40, "ADX 20-40"),
    ]

    for min_a, max_a, desc in adx_tests:
        result = run_backtest(df, min_adx=min_a, max_adx=max_a)
        diff = result['return_pct'] - baseline['return_pct']
        print(f"      {desc}: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
        if diff > 0:
            results.append((desc, diff, result))

    # =========================================================================
    # TEST 4: Bollinger Band Width
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 4: BOLLINGER BAND WIDTH")
    print("-" * 80)

    bb_tests = [
        (None, 2.0, "Narrow BB < 2%"),
        (1.0, 3.0, "Medium BB 1-3%"),
        (1.5, None, "Wide BB > 1.5%"),
        (0.8, 2.5, "Optimal BB 0.8-2.5%"),
    ]

    for min_b, max_b, desc in bb_tests:
        result = run_backtest(df, min_bb_width=min_b, max_bb_width=max_b)
        diff = result['return_pct'] - baseline['return_pct']
        print(f"      {desc}: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
        if diff > 0:
            results.append((desc, diff, result))

    # =========================================================================
    # TEST 5: Candle Patterns
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 5: CANDLE PATTERN CONFIRMATION")
    print("-" * 80)

    # Confirmation candle
    result = run_backtest(df, require_confirmation_candle=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      Confirmation candle: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
    if diff > 0:
        results.append(('Confirmation Candle', diff, result))

    # Engulfing only (likely too restrictive)
    result = run_backtest(df, require_engulfing=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      Engulfing pattern: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")

    # =========================================================================
    # TEST 6: H4 Timeframe Alignment
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 6: HIGHER TIMEFRAME (H4) ALIGNMENT")
    print("-" * 80)

    result = run_backtest(df, h4_alignment=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      H4 RSI alignment: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
    if diff > 0:
        results.append(('H4 Alignment', diff, result))

    # =========================================================================
    # TEST 7: Week of Month
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 7: WEEK OF MONTH")
    print("-" * 80)

    result = run_backtest(df, avoid_week1=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      Skip week 1: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
    if diff > 0:
        results.append(('Skip Week 1', diff, result))

    result = run_backtest(df, avoid_week4=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      Skip week 4+: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
    if diff > 0:
        results.append(('Skip Week 4', diff, result))

    # =========================================================================
    # TEST 8: Extreme RSI Levels
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 8: EXTREME RSI LEVELS (25/75)")
    print("-" * 80)

    result = run_backtest(df, extreme_rsi_only=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      RSI < 25 / > 75: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
    if diff > 0:
        results.append(('Extreme RSI 25/75', diff, result))

    # =========================================================================
    # TEST 9: Time-Based Exit
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 9: TIME-BASED EXIT (Max Hold)")
    print("-" * 80)

    for max_bars in [12, 24, 48, 72]:
        result = run_backtest(df, max_hold_bars=max_bars)
        diff = result['return_pct'] - baseline['return_pct']
        print(f"      Max {max_bars}h hold: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
        if diff > 0:
            results.append((f'Max {max_bars}h Hold', diff, result))

    # =========================================================================
    # TEST 10: Dynamic TP
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 10: DYNAMIC TP (Volatility-Adjusted)")
    print("-" * 80)

    result = run_backtest(df, dynamic_tp=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      Dynamic TP: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
    if diff > 0:
        results.append(('Dynamic TP', diff, result))

    # =========================================================================
    # TEST 11: Near Swing Levels
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 11: TRADE NEAR SWING LEVELS")
    print("-" * 80)

    result = run_backtest(df, near_swing_filter=True)
    diff = result['return_pct'] - baseline['return_pct']
    print(f"      Near swing levels: {result['trades']} trades, {result['return_pct']:+.1f}% ({diff:+.1f}%)")
    if diff > 0:
        results.append(('Near Swing Levels', diff, result))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("OPTIMIZATION SUMMARY")
    print("=" * 100)

    print(f"\nv3.2 BASELINE: +{baseline['return_pct']:.1f}% ({baseline['trades']} trades, WR {baseline['win_rate']:.1f}%)")

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print("\n" + "-" * 60)
        print("FILTERS THAT IMPROVE PERFORMANCE:")
        print("-" * 60)
        for name, diff, res in results[:10]:
            print(f"  {name:<35} {res['return_pct']:+.1f}% ({diff:+.1f}% improvement)")

        # Try combining top 2-3 improvements
        print("\n" + "-" * 60)
        print("TESTING COMBINATIONS:")
        print("-" * 60)

        # Get the parameter names from top results
        top_filters = results[:3]
        print(f"\n  Top improvements to combine:")
        for name, diff, _ in top_filters:
            print(f"    - {name}: +{diff:.1f}%")
    else:
        print("\nNo filters improved baseline significantly.")
        print("v3.2 configuration is already near-optimal!")


if __name__ == "__main__":
    main()
