"""Fetch and Analyze XAUUSD Data
================================

Fetch historical XAUUSD data from MT5 and analyze its characteristics
to determine optimal trading parameters.

This analysis will help us understand:
1. Price range and volatility patterns
2. Best trading sessions for Gold
3. Optimal pip/point size settings
4. ATR and velocity characteristics
5. Comparison with GBPUSD

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import json

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNING: MetaTrader5 not installed")

from config import config
from src.data.db_handler import DBHandler

# ============================================================
# CONFIGURATION
# ============================================================
SYMBOL = "XAUUSD"
COMPARISON_SYMBOL = "GBPUSD"

# Fetch 2 years of data for comprehensive analysis
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 31, tzinfo=timezone.utc)

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "data"
ANALYSIS_OUTPUT = Path(__file__).parent / "gold_characteristics.json"


def fetch_mt5_data(symbol: str, timeframe_str: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch historical data from MT5"""
    if not MT5_AVAILABLE:
        print(f"ERROR: MT5 not available")
        return None

    if not mt5.initialize():
        print(f"ERROR: MT5 init failed: {mt5.last_error()}")
        return None

    tf_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
    }
    tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_H4)

    print(f"  Fetching {symbol} {timeframe_str}...")
    rates = mt5.copy_rates_range(symbol, tf, start, end)

    if rates is None or len(rates) == 0:
        print(f"  ERROR: No data for {symbol} {timeframe_str}")
        mt5.shutdown()
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

    print(f"  Got {len(df)} bars for {symbol} {timeframe_str}")
    mt5.shutdown()

    return df[['open', 'high', 'low', 'close', 'volume']]


def analyze_price_characteristics(df: pd.DataFrame, symbol: str) -> Dict:
    """Analyze price movement characteristics"""

    # Basic stats
    avg_price = df['close'].mean()
    price_range = df['close'].max() - df['close'].min()

    # Daily range analysis
    df['daily_range'] = df['high'] - df['low']
    avg_daily_range = df['daily_range'].mean()

    # Volatility (standard deviation)
    df['returns'] = df['close'].pct_change()
    daily_volatility = df['returns'].std() * 100  # percentage

    # ATR calculation (14-period)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    avg_atr = df['atr_14'].mean()

    # Movement per bar
    df['bar_move'] = abs(df['close'] - df['open'])
    avg_bar_move = df['bar_move'].mean()

    # Determine pip/point size based on price level
    if avg_price > 100:  # Gold, indices
        point_size = 0.01  # 1 point = $0.01
        pip_size = 0.1     # 1 pip = $0.10
    else:  # Forex
        point_size = 0.00001
        pip_size = 0.0001

    # Convert to pips
    avg_daily_range_pips = avg_daily_range / pip_size
    avg_atr_pips = avg_atr / pip_size
    avg_bar_move_pips = avg_bar_move / pip_size

    return {
        'symbol': symbol,
        'total_bars': len(df),
        'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
        'avg_price': round(avg_price, 2),
        'price_range': round(price_range, 2),
        'point_size': point_size,
        'pip_size': pip_size,
        'avg_daily_range': round(avg_daily_range, 5),
        'avg_daily_range_pips': round(avg_daily_range_pips, 1),
        'daily_volatility_pct': round(daily_volatility, 4),
        'avg_atr': round(avg_atr, 5),
        'avg_atr_pips': round(avg_atr_pips, 1),
        'avg_bar_move': round(avg_bar_move, 5),
        'avg_bar_move_pips': round(avg_bar_move_pips, 2),
    }


def analyze_session_performance(df: pd.DataFrame, symbol: str) -> Dict:
    """Analyze performance by trading session"""

    # Add hour column
    df = df.copy()
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['range'] = df['high'] - df['low']

    # Define sessions (UTC)
    sessions = {
        'Asian': (0, 8),      # 00:00 - 08:00 UTC
        'London': (7, 16),    # 07:00 - 16:00 UTC
        'NewYork': (12, 21),  # 12:00 - 21:00 UTC
        'Overlap': (12, 16),  # London-NY overlap
    }

    session_stats = {}
    for session_name, (start_hour, end_hour) in sessions.items():
        if start_hour < end_hour:
            mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
        else:
            mask = (df['hour'] >= start_hour) | (df['hour'] < end_hour)

        session_df = df[mask]
        if len(session_df) > 0:
            avg_range = session_df['range'].mean()
            pip_size = 0.1 if session_df['close'].mean() > 100 else 0.0001
            session_stats[session_name] = {
                'bars': len(session_df),
                'avg_range': round(avg_range, 5),
                'avg_range_pips': round(avg_range / pip_size, 1),
                'pct_of_total': round(len(session_df) / len(df) * 100, 1)
            }

    # Find best session (highest avg range)
    best_session = max(session_stats.items(), key=lambda x: x[1]['avg_range_pips'])

    return {
        'sessions': session_stats,
        'best_session': best_session[0],
        'best_session_avg_pips': best_session[1]['avg_range_pips']
    }


def analyze_trend_characteristics(df: pd.DataFrame) -> Dict:
    """Analyze trend behavior"""

    df = df.copy()

    # Calculate EMAs
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    # Trend detection
    df['uptrend'] = (df['ema_20'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
    df['downtrend'] = (df['ema_20'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])
    df['sideways'] = ~df['uptrend'] & ~df['downtrend']

    uptrend_pct = df['uptrend'].sum() / len(df) * 100
    downtrend_pct = df['downtrend'].sum() / len(df) * 100
    sideways_pct = df['sideways'].sum() / len(df) * 100

    # Consecutive trend bars
    df['trend_change'] = (df['uptrend'] != df['uptrend'].shift(1)).cumsum()
    trend_durations = df.groupby('trend_change').size()
    avg_trend_duration = trend_durations.mean()
    max_trend_duration = trend_durations.max()

    return {
        'uptrend_pct': round(uptrend_pct, 1),
        'downtrend_pct': round(downtrend_pct, 1),
        'sideways_pct': round(sideways_pct, 1),
        'avg_trend_duration_bars': round(avg_trend_duration, 1),
        'max_trend_duration_bars': int(max_trend_duration)
    }


def analyze_velocity(df: pd.DataFrame) -> Dict:
    """Analyze price velocity characteristics"""

    df = df.copy()
    pip_size = 0.1 if df['close'].mean() > 100 else 0.0001

    # Velocity = change per bar
    df['velocity'] = (df['close'] - df['close'].shift(1)) / pip_size
    df['abs_velocity'] = abs(df['velocity'])

    # Acceleration
    df['acceleration'] = df['velocity'] - df['velocity'].shift(1)

    avg_velocity = df['abs_velocity'].mean()
    max_velocity = df['abs_velocity'].max()
    velocity_std = df['abs_velocity'].std()

    # Percentiles
    vel_25 = df['abs_velocity'].quantile(0.25)
    vel_50 = df['abs_velocity'].quantile(0.50)
    vel_75 = df['abs_velocity'].quantile(0.75)
    vel_90 = df['abs_velocity'].quantile(0.90)

    return {
        'avg_velocity_pips': round(avg_velocity, 2),
        'max_velocity_pips': round(max_velocity, 2),
        'velocity_std_pips': round(velocity_std, 2),
        'velocity_25th_pips': round(vel_25, 2),
        'velocity_50th_pips': round(vel_50, 2),
        'velocity_75th_pips': round(vel_75, 2),
        'velocity_90th_pips': round(vel_90, 2),
        'recommended_min_velocity': round(vel_50, 2),  # Use median as baseline
        'recommended_high_velocity': round(vel_75, 2),  # 75th percentile for "active"
    }


def suggest_parameters(gold_analysis: Dict, gbp_analysis: Dict) -> Dict:
    """Suggest optimal parameters for Gold based on analysis"""

    gold_price = gold_analysis['price_chars']
    gold_velocity = gold_analysis['velocity']
    gold_session = gold_analysis['sessions']

    # Pip size for Gold
    pip_size = gold_price['pip_size']  # 0.1

    # ATR-based parameters
    avg_atr = gold_price['avg_atr_pips']

    # Suggested parameters
    suggestions = {
        'symbol': 'XAUUSD',
        'pip_size': pip_size,
        'point_size': gold_price['point_size'],

        # Spread (typical Gold spread is 20-30 points = 2-3 pips)
        'spread_pips': 25,  # Conservative

        # Intelligent Filter parameters
        'intel_filter': {
            'activity_threshold': 50,  # Lower than GBPUSD due to less frequent moves
            'min_velocity_pips': round(gold_velocity['velocity_50th_pips'], 1),
            'high_velocity_pips': round(gold_velocity['velocity_75th_pips'], 1),
            'min_atr_pips': round(avg_atr * 0.5, 1),  # 50% of avg ATR
            'high_atr_pips': round(avg_atr * 1.0, 1),  # 100% of avg ATR
        },

        # Risk Management
        'risk_management': {
            'max_sl_pips': round(avg_atr * 1.5, 0),  # 1.5x ATR
            'default_sl_pips': round(avg_atr * 1.0, 0),  # 1x ATR
            'min_rr_ratio': 2.0,  # Minimum risk:reward
            'pip_value_per_lot': 1.0,  # $1 per pip per 0.01 lot (standard)
        },

        # POI Detection
        'poi_detection': {
            'tolerance_pips': round(avg_atr * 0.3, 0),  # 30% of ATR for entry zone
            'max_poi_age_bars': 30,  # Fresher POIs for Gold's faster moves
            'min_ob_strength': 0.3,  # Lower threshold
        },

        # Best trading times
        'best_sessions': {
            'primary': gold_session['best_session'],
            'recommended_hours_utc': '12:00-16:00' if gold_session['best_session'] == 'Overlap' else '07:00-16:00'
        },

        # Comparison with GBPUSD
        'vs_gbpusd': {
            'atr_ratio': round(gold_price['avg_atr_pips'] / gbp_analysis['price_chars']['avg_atr_pips'], 2),
            'velocity_ratio': round(gold_velocity['avg_velocity_pips'] / gbp_analysis['velocity']['avg_velocity_pips'], 2),
            'note': 'Gold typically has higher ATR but similar velocity patterns'
        }
    }

    return suggestions


async def save_to_database(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save data to TimescaleDB"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    count = await db.save_ohlcv(symbol, timeframe, df)
    await db.disconnect()
    print(f"  Saved {count} bars to database: {symbol} {timeframe}")
    return count


async def main():
    print("=" * 70)
    print("XAUUSD DATA FETCH & ANALYSIS")
    print("=" * 70)
    print()

    # ============================================================
    # STEP 1: Fetch Data from MT5
    # ============================================================
    print("[1/5] Fetching data from MT5...")
    print()

    # Fetch XAUUSD
    gold_h4 = fetch_mt5_data(SYMBOL, 'H4', START_DATE, END_DATE)
    gold_h1 = fetch_mt5_data(SYMBOL, 'H1', START_DATE, END_DATE)
    gold_m15 = fetch_mt5_data(SYMBOL, 'M15', START_DATE, END_DATE)
    gold_m5 = fetch_mt5_data(SYMBOL, 'M5', START_DATE - timedelta(days=90), END_DATE)  # Less M5 data

    # Fetch GBPUSD for comparison
    gbp_h4 = fetch_mt5_data(COMPARISON_SYMBOL, 'H4', START_DATE, END_DATE)

    if gold_h4 is None or gbp_h4 is None:
        print("ERROR: Failed to fetch data. Make sure MT5 is running.")
        return

    print()

    # ============================================================
    # STEP 2: Save to Database
    # ============================================================
    print("[2/5] Saving to database...")
    await save_to_database(gold_h4, SYMBOL, 'H4')
    await save_to_database(gold_h1, SYMBOL, 'H1')
    await save_to_database(gold_m15, SYMBOL, 'M15')
    if gold_m5 is not None:
        await save_to_database(gold_m5, SYMBOL, 'M5')
    print()

    # ============================================================
    # STEP 3: Analyze XAUUSD Characteristics
    # ============================================================
    print("[3/5] Analyzing XAUUSD characteristics...")
    print()

    gold_price_chars = analyze_price_characteristics(gold_h4, SYMBOL)
    gold_sessions = analyze_session_performance(gold_h4, SYMBOL)
    gold_trends = analyze_trend_characteristics(gold_h4)
    gold_velocity = analyze_velocity(gold_h4)

    gold_analysis = {
        'price_chars': gold_price_chars,
        'sessions': gold_sessions,
        'trends': gold_trends,
        'velocity': gold_velocity
    }

    # ============================================================
    # STEP 4: Analyze GBPUSD for Comparison
    # ============================================================
    print("[4/5] Analyzing GBPUSD for comparison...")
    print()

    gbp_price_chars = analyze_price_characteristics(gbp_h4, COMPARISON_SYMBOL)
    gbp_sessions = analyze_session_performance(gbp_h4, COMPARISON_SYMBOL)
    gbp_trends = analyze_trend_characteristics(gbp_h4)
    gbp_velocity = analyze_velocity(gbp_h4)

    gbp_analysis = {
        'price_chars': gbp_price_chars,
        'sessions': gbp_sessions,
        'trends': gbp_trends,
        'velocity': gbp_velocity
    }

    # ============================================================
    # STEP 5: Generate Recommendations
    # ============================================================
    print("[5/5] Generating parameter recommendations...")
    print()

    suggestions = suggest_parameters(gold_analysis, gbp_analysis)

    # ============================================================
    # OUTPUT RESULTS
    # ============================================================
    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print()

    print("XAUUSD PRICE CHARACTERISTICS:")
    print("-" * 40)
    for k, v in gold_price_chars.items():
        print(f"  {k}: {v}")
    print()

    print("XAUUSD SESSION ANALYSIS:")
    print("-" * 40)
    for session, stats in gold_sessions['sessions'].items():
        print(f"  {session}: {stats['avg_range_pips']:.1f} pips avg range ({stats['pct_of_total']:.1f}% of bars)")
    print(f"  Best Session: {gold_sessions['best_session']} ({gold_sessions['best_session_avg_pips']:.1f} pips)")
    print()

    print("XAUUSD TREND ANALYSIS:")
    print("-" * 40)
    for k, v in gold_trends.items():
        print(f"  {k}: {v}")
    print()

    print("XAUUSD VELOCITY ANALYSIS:")
    print("-" * 40)
    for k, v in gold_velocity.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 70)
    print("COMPARISON: XAUUSD vs GBPUSD")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'XAUUSD':<15} {'GBPUSD':<15} {'Ratio':<10}")
    print("-" * 70)
    print(f"{'Avg Price':<30} ${gold_price_chars['avg_price']:<14} {gbp_price_chars['avg_price']:<15} -")
    print(f"{'Pip Size':<30} {gold_price_chars['pip_size']:<15} {gbp_price_chars['pip_size']:<15} -")
    print(f"{'Avg ATR (pips)':<30} {gold_price_chars['avg_atr_pips']:<15} {gbp_price_chars['avg_atr_pips']:<15} {gold_price_chars['avg_atr_pips']/gbp_price_chars['avg_atr_pips']:.2f}x")
    print(f"{'Avg Velocity (pips)':<30} {gold_velocity['avg_velocity_pips']:<15} {gbp_velocity['avg_velocity_pips']:<15} {gold_velocity['avg_velocity_pips']/gbp_velocity['avg_velocity_pips']:.2f}x")
    print(f"{'Daily Range (pips)':<30} {gold_price_chars['avg_daily_range_pips']:<15} {gbp_price_chars['avg_daily_range_pips']:<15} {gold_price_chars['avg_daily_range_pips']/gbp_price_chars['avg_daily_range_pips']:.2f}x")
    print()

    print("=" * 70)
    print("RECOMMENDED PARAMETERS FOR XAUUSD")
    print("=" * 70)
    print()
    print(json.dumps(suggestions, indent=2))
    print()

    # Save analysis to file
    full_analysis = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'xauusd': gold_analysis,
        'gbpusd': gbp_analysis,
        'recommendations': suggestions
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANALYSIS_OUTPUT, 'w') as f:
        json.dump(full_analysis, f, indent=2, default=str)
    print(f"Analysis saved to: {ANALYSIS_OUTPUT}")

    # Save CSV for reference
    gold_h4.to_csv(OUTPUT_DIR / 'xauusd_h4.csv')
    print(f"H4 data saved to: {OUTPUT_DIR / 'xauusd_h4.csv'}")

    print()
    print("DONE!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
