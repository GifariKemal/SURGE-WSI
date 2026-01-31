"""Deep Market Analysis - Per Month/Day/Hour Analysis
======================================================

Analyze every trade in detail to understand:
1. What conditions led to WINS vs LOSSES
2. Patterns per month, day-of-week, hour
3. Market characteristics that predict success
4. Create truly adaptive rules

Goal: Find patterns that can achieve ZERO LOSS

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Dict, Tuple
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        return pd.DataFrame()
    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators for analysis"""
    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    close = df[col_map['close']]
    high = df[col_map['high']]
    low = df[col_map['low']]

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pips'] = df['atr'] / 0.0001

    # Choppiness Index
    atr_sum = tr.rolling(14).sum()
    highest_high = high.rolling(14).max()
    lowest_low = low.rolling(14).min()
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.nan)
    df['choppiness'] = 100 * np.log10(atr_sum / price_range) / np.log10(14)

    # ADX
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (atr_smooth + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (atr_smooth + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMAs
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()

    # Volatility (rolling std)
    df['volatility'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Trend strength (price vs EMAs)
    df['trend_strength'] = ((close - df['ema_50']) / df['ema_50'] * 100).abs()

    # Bar range
    df['bar_range_pips'] = (high - low) / 0.0001

    return df


def simulate_trades_with_details(df: pd.DataFrame) -> List[dict]:
    """
    Simulate trades and capture ALL details for analysis.
    Returns list of trade dictionaries with full context.
    """
    from src.utils.killzone import KillZone
    from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel

    killzone = KillZone()
    activity_filter = DynamicActivityFilter(min_atr_pips=5.0, min_bar_range_pips=3.0,
                                            activity_threshold=35.0, pip_size=0.0001)
    activity_filter.outside_kz_min_score = 60.0

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Warmup
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    trades = []
    position = None

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        kalman.update(price)
        regime_info = regime_detector.update(price)

        # Manage position
        if position:
            if position['direction'] == 'BUY':
                if low <= position['sl']:
                    position['exit_time'] = current_time
                    position['exit_price'] = position['sl']
                    position['pnl_pips'] = (position['sl'] - position['entry_price']) * 10000
                    position['result'] = 'LOSS'
                    position['exit_reason'] = 'SL'
                    trades.append(position)
                    position = None
                    continue
                elif high >= position['tp']:
                    position['exit_time'] = current_time
                    position['exit_price'] = position['tp']
                    position['pnl_pips'] = (position['tp'] - position['entry_price']) * 10000
                    position['result'] = 'WIN'
                    position['exit_reason'] = 'TP'
                    trades.append(position)
                    position = None
                    continue
            else:  # SELL
                if high >= position['sl']:
                    position['exit_time'] = current_time
                    position['exit_price'] = position['sl']
                    position['pnl_pips'] = (position['entry_price'] - position['sl']) * 10000
                    position['result'] = 'LOSS'
                    position['exit_reason'] = 'SL'
                    trades.append(position)
                    position = None
                    continue
                elif low <= position['tp']:
                    position['exit_time'] = current_time
                    position['exit_price'] = position['tp']
                    position['pnl_pips'] = (position['entry_price'] - position['tp']) * 10000
                    position['result'] = 'WIN'
                    position['exit_reason'] = 'TP'
                    trades.append(position)
                    position = None
                    continue

            # Regime flip
            if regime_info:
                if (position['direction'] == 'BUY' and regime_info.bias == 'SELL') or \
                   (position['direction'] == 'SELL' and regime_info.bias == 'BUY'):
                    position['exit_time'] = current_time
                    position['exit_price'] = price
                    if position['direction'] == 'BUY':
                        position['pnl_pips'] = (price - position['entry_price']) * 10000
                    else:
                        position['pnl_pips'] = (position['entry_price'] - price) * 10000
                    position['result'] = 'WIN' if position['pnl_pips'] > 0 else 'LOSS'
                    position['exit_reason'] = 'REGIME_FLIP'
                    trades.append(position)
                    position = None
                    continue

        if position:
            continue

        # Check trading conditions
        in_kz, session = killzone.is_in_killzone(current_time)
        can_trade_outside = False
        activity_score = 0.0

        if not in_kz:
            recent_df = df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            activity_score = activity.score
            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE] and activity.score >= 60:
                can_trade_outside = True

        if not (in_kz or can_trade_outside):
            continue

        if not regime_info or not regime_info.is_tradeable or regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias

        # Detect POI (simplified)
        poi_found = False
        poi_quality = 50
        recent = df.iloc[idx-15:idx]

        for i in range(len(recent) - 3):
            ob_bar = recent.iloc[i]
            next_bars = recent.iloc[i+1:i+4]

            if direction == 'BUY':
                if ob_bar[col_map['close']] < ob_bar[col_map['open']]:
                    move = next_bars[col_map['close']].max() - ob_bar[col_map['low']]
                    if move > 0.0010:
                        poi_found = True
                        poi_quality = min(100, move * 10000 * 5)
                        break
            else:
                if ob_bar[col_map['close']] > ob_bar[col_map['open']]:
                    move = ob_bar[col_map['high']] - next_bars[col_map['close']].min()
                    if move > 0.0010:
                        poi_found = True
                        poi_quality = min(100, move * 10000 * 5)
                        break

        if not poi_found:
            continue

        # Check entry trigger
        o, h_bar, l_bar, c = bar[col_map['open']], bar[col_map['high']], bar[col_map['low']], bar[col_map['close']]
        total_range = h_bar - l_bar
        if total_range < 0.0003:
            continue

        body = abs(c - o)
        entry_type = None

        if direction == 'BUY':
            lower_wick = min(o, c) - l_bar
            if lower_wick > body and lower_wick > total_range * 0.5:
                entry_type = "REJECTION"
            elif c > o and body > total_range * 0.6:
                entry_type = "MOMENTUM"
            elif l_bar > prev_bar[col_map['low']] and c > o:
                entry_type = "HIGHER_LOW"
        else:
            upper_wick = h_bar - max(o, c)
            if upper_wick > body and upper_wick > total_range * 0.5:
                entry_type = "REJECTION"
            elif c < o and body > total_range * 0.6:
                entry_type = "MOMENTUM"

        if not entry_type:
            continue

        # Get all indicators at entry
        sl_pips = 25.0
        tp_pips = sl_pips * 1.5

        if direction == 'BUY':
            sl_price = price - sl_pips * 0.0001
            tp_price = price + tp_pips * 0.0001
        else:
            sl_price = price + sl_pips * 0.0001
            tp_price = price - tp_pips * 0.0001

        # Capture EVERYTHING about this trade
        position = {
            'entry_time': current_time,
            'direction': direction,
            'entry_price': price,
            'sl': sl_price,
            'tp': tp_price,
            'entry_type': entry_type,
            'poi_quality': poi_quality,

            # Time features
            'month': current_time.month,
            'day_of_week': current_time.weekday(),
            'day_name': current_time.strftime('%A'),
            'hour': current_time.hour,
            'week_of_year': current_time.isocalendar()[1],

            # Session
            'session': session if in_kz else 'Hybrid',
            'in_killzone': in_kz,
            'activity_score': activity_score,

            # Market indicators at entry
            'atr_pips': bar['atr_pips'] if 'atr_pips' in bar.index else 20,
            'choppiness': bar['choppiness'] if 'choppiness' in bar.index else 50,
            'adx': bar['adx'] if 'adx' in bar.index else 25,
            'rsi': bar['rsi'] if 'rsi' in bar.index else 50,
            'volatility': bar['volatility'] if 'volatility' in bar.index else 10,
            'trend_strength': bar['trend_strength'] if 'trend_strength' in bar.index else 0.5,
            'bar_range_pips': bar['bar_range_pips'] if 'bar_range_pips' in bar.index else 10,

            # EMA alignment
            'price_vs_ema20': (price - bar['ema_20']) / bar['ema_20'] * 100 if 'ema_20' in bar.index else 0,
            'price_vs_ema50': (price - bar['ema_50']) / bar['ema_50'] * 100 if 'ema_50' in bar.index else 0,
            'ema_aligned': (bar['ema_20'] > bar['ema_50']) if direction == 'BUY' else (bar['ema_20'] < bar['ema_50']) if 'ema_20' in bar.index else False,

            # Regime
            'regime_confidence': regime_info.confidence * 100 if hasattr(regime_info, 'confidence') else 80,
        }

    # Close remaining position
    if position:
        last_bar = df.iloc[-1]
        position['exit_time'] = last_bar.name
        position['exit_price'] = last_bar[col_map['close']]
        if position['direction'] == 'BUY':
            position['pnl_pips'] = (position['exit_price'] - position['entry_price']) * 10000
        else:
            position['pnl_pips'] = (position['entry_price'] - position['exit_price']) * 10000
        position['result'] = 'WIN' if position['pnl_pips'] > 0 else 'LOSS'
        position['exit_reason'] = 'END'
        trades.append(position)

    return trades


def analyze_winning_conditions(trades: List[dict]) -> dict:
    """Analyze what conditions lead to WINS vs LOSSES"""

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    analysis = {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
    }

    # Analyze numeric features
    numeric_features = [
        'atr_pips', 'choppiness', 'adx', 'rsi', 'volatility',
        'trend_strength', 'bar_range_pips', 'poi_quality',
        'activity_score', 'regime_confidence'
    ]

    feature_analysis = {}
    for feature in numeric_features:
        win_values = [t[feature] for t in wins if feature in t and t[feature] is not None]
        loss_values = [t[feature] for t in losses if feature in t and t[feature] is not None]

        if win_values and loss_values:
            feature_analysis[feature] = {
                'win_mean': np.mean(win_values),
                'win_std': np.std(win_values),
                'loss_mean': np.mean(loss_values),
                'loss_std': np.std(loss_values),
                'diff': np.mean(win_values) - np.mean(loss_values),
                'win_median': np.median(win_values),
                'loss_median': np.median(loss_values),
            }

    analysis['features'] = feature_analysis

    # Day of week analysis
    day_analysis = {}
    for day in range(7):
        day_trades = [t for t in trades if t['day_of_week'] == day]
        day_wins = [t for t in day_trades if t['result'] == 'WIN']
        if day_trades:
            day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day]
            day_analysis[day_name] = {
                'trades': len(day_trades),
                'wins': len(day_wins),
                'win_rate': len(day_wins) / len(day_trades) * 100,
                'avg_pnl_pips': np.mean([t['pnl_pips'] for t in day_trades])
            }
    analysis['by_day'] = day_analysis

    # Hour analysis
    hour_analysis = {}
    for hour in range(24):
        hour_trades = [t for t in trades if t['hour'] == hour]
        hour_wins = [t for t in hour_trades if t['result'] == 'WIN']
        if hour_trades:
            hour_analysis[hour] = {
                'trades': len(hour_trades),
                'wins': len(hour_wins),
                'win_rate': len(hour_wins) / len(hour_trades) * 100,
                'avg_pnl_pips': np.mean([t['pnl_pips'] for t in hour_trades])
            }
    analysis['by_hour'] = hour_analysis

    # Month analysis
    month_analysis = {}
    for month in range(1, 13):
        month_trades = [t for t in trades if t['month'] == month]
        month_wins = [t for t in month_trades if t['result'] == 'WIN']
        if month_trades:
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
            month_analysis[month_name] = {
                'trades': len(month_trades),
                'wins': len(month_wins),
                'win_rate': len(month_wins) / len(month_trades) * 100,
                'total_pips': sum(t['pnl_pips'] for t in month_trades),
                'avg_choppiness': np.mean([t['choppiness'] for t in month_trades if t.get('choppiness')]),
                'avg_adx': np.mean([t['adx'] for t in month_trades if t.get('adx')]),
                'avg_atr': np.mean([t['atr_pips'] for t in month_trades if t.get('atr_pips')]),
            }
    analysis['by_month'] = month_analysis

    # Session analysis
    session_analysis = {}
    for session in ['London', 'New York', 'Hybrid']:
        session_trades = [t for t in trades if t['session'] == session]
        session_wins = [t for t in session_trades if t['result'] == 'WIN']
        if session_trades:
            session_analysis[session] = {
                'trades': len(session_trades),
                'wins': len(session_wins),
                'win_rate': len(session_wins) / len(session_trades) * 100,
                'total_pips': sum(t['pnl_pips'] for t in session_trades)
            }
    analysis['by_session'] = session_analysis

    # Entry type analysis
    entry_analysis = {}
    for entry_type in ['REJECTION', 'MOMENTUM', 'HIGHER_LOW', 'ENGULF']:
        type_trades = [t for t in trades if t.get('entry_type') == entry_type]
        type_wins = [t for t in type_trades if t['result'] == 'WIN']
        if type_trades:
            entry_analysis[entry_type] = {
                'trades': len(type_trades),
                'wins': len(type_wins),
                'win_rate': len(type_wins) / len(type_trades) * 100,
                'total_pips': sum(t['pnl_pips'] for t in type_trades)
            }
    analysis['by_entry_type'] = entry_analysis

    # EMA alignment analysis
    aligned_trades = [t for t in trades if t.get('ema_aligned')]
    not_aligned_trades = [t for t in trades if not t.get('ema_aligned')]

    if aligned_trades:
        aligned_wins = [t for t in aligned_trades if t['result'] == 'WIN']
        analysis['ema_aligned'] = {
            'trades': len(aligned_trades),
            'wins': len(aligned_wins),
            'win_rate': len(aligned_wins) / len(aligned_trades) * 100,
            'total_pips': sum(t['pnl_pips'] for t in aligned_trades)
        }

    if not_aligned_trades:
        not_aligned_wins = [t for t in not_aligned_trades if t['result'] == 'WIN']
        analysis['ema_not_aligned'] = {
            'trades': len(not_aligned_trades),
            'wins': len(not_aligned_wins),
            'win_rate': len(not_aligned_wins) / len(not_aligned_trades) * 100,
            'total_pips': sum(t['pnl_pips'] for t in not_aligned_trades)
        }

    return analysis


def find_optimal_thresholds(trades: List[dict]) -> dict:
    """Find optimal thresholds for each indicator to maximize win rate"""

    thresholds = {}

    # Test different thresholds for each feature
    features_to_test = {
        'choppiness': {'min': 30, 'max': 80, 'step': 5, 'direction': 'below'},  # Lower is better
        'adx': {'min': 10, 'max': 40, 'step': 2, 'direction': 'above'},  # Higher is better
        'rsi': {'min': 30, 'max': 70, 'step': 5, 'direction': 'range'},  # In range is better
        'poi_quality': {'min': 30, 'max': 80, 'step': 5, 'direction': 'above'},  # Higher is better
        'regime_confidence': {'min': 50, 'max': 90, 'step': 5, 'direction': 'above'},  # Higher is better
        'trend_strength': {'min': 0.1, 'max': 2.0, 'step': 0.1, 'direction': 'above'},  # Higher is better
    }

    for feature, config in features_to_test.items():
        best_wr = 0
        best_threshold = None
        best_trades = 0

        test_range = np.arange(config['min'], config['max'], config['step'])

        for threshold in test_range:
            if config['direction'] == 'below':
                filtered = [t for t in trades if t.get(feature, 100) < threshold]
            elif config['direction'] == 'above':
                filtered = [t for t in trades if t.get(feature, 0) > threshold]
            else:  # range - for RSI
                filtered = [t for t in trades if 30 < t.get(feature, 50) < 70]

            if len(filtered) >= 20:  # Minimum trades for significance
                wins = len([t for t in filtered if t['result'] == 'WIN'])
                wr = wins / len(filtered) * 100

                if wr > best_wr:
                    best_wr = wr
                    best_threshold = threshold
                    best_trades = len(filtered)

        if best_threshold is not None:
            thresholds[feature] = {
                'threshold': best_threshold,
                'direction': config['direction'],
                'win_rate': best_wr,
                'trades': best_trades
            }

    return thresholds


def find_perfect_conditions(trades: List[dict]) -> dict:
    """Find combination of conditions that achieve highest win rate"""

    results = []

    # Test combinations
    chop_thresholds = [55, 60, 65, 70]
    adx_thresholds = [18, 20, 22, 25]
    confidence_thresholds = [60, 65, 70, 75]
    quality_thresholds = [40, 50, 60]

    for chop in chop_thresholds:
        for adx in adx_thresholds:
            for conf in confidence_thresholds:
                for quality in quality_thresholds:
                    filtered = [
                        t for t in trades
                        if t.get('choppiness', 100) < chop
                        and t.get('adx', 0) > adx
                        and t.get('regime_confidence', 0) > conf
                        and t.get('poi_quality', 0) > quality
                    ]

                    if len(filtered) >= 15:
                        wins = len([t for t in filtered if t['result'] == 'WIN'])
                        wr = wins / len(filtered) * 100
                        total_pips = sum(t['pnl_pips'] for t in filtered)

                        results.append({
                            'chop_max': chop,
                            'adx_min': adx,
                            'confidence_min': conf,
                            'quality_min': quality,
                            'trades': len(filtered),
                            'wins': wins,
                            'win_rate': wr,
                            'total_pips': total_pips,
                            'pips_per_trade': total_pips / len(filtered)
                        })

    # Sort by win rate then by number of trades
    results.sort(key=lambda x: (-x['win_rate'], -x['trades']))

    return {
        'best_combinations': results[:10],
        'highest_wr': results[0] if results else None
    }


def print_analysis_report(analysis: dict, thresholds: dict, perfect: dict):
    """Print comprehensive analysis report"""

    print("\n" + "=" * 80)
    print("DEEP MARKET ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nTotal Trades: {analysis['total_trades']}")
    print(f"Win Rate: {analysis['win_rate']:.1f}%")

    # Feature Analysis
    print("\n" + "-" * 80)
    print("FEATURE ANALYSIS: WIN vs LOSS Comparison")
    print("-" * 80)
    print(f"{'Feature':<20} {'WIN Mean':>12} {'LOSS Mean':>12} {'Difference':>12} {'Better When':>15}")
    print("-" * 80)

    for feature, data in analysis['features'].items():
        diff = data['diff']
        if feature in ['choppiness']:
            better = f"< {data['win_mean']:.1f}" if diff < 0 else f"> {data['loss_mean']:.1f}"
        else:
            better = f"> {data['win_mean']:.1f}" if diff > 0 else f"< {data['loss_mean']:.1f}"

        print(f"{feature:<20} {data['win_mean']:>12.2f} {data['loss_mean']:>12.2f} {diff:>+12.2f} {better:>15}")

    # Day of Week
    print("\n" + "-" * 80)
    print("DAY OF WEEK ANALYSIS")
    print("-" * 80)
    print(f"{'Day':<10} {'Trades':>8} {'Wins':>8} {'Win Rate':>10} {'Avg Pips':>10}")

    for day, data in analysis['by_day'].items():
        status = "✅" if data['win_rate'] >= 50 else "❌"
        print(f"{day:<10} {data['trades']:>8} {data['wins']:>8} {data['win_rate']:>9.1f}% {data['avg_pnl_pips']:>+9.1f} {status}")

    # Hour Analysis
    print("\n" + "-" * 80)
    print("HOUR ANALYSIS (Best Hours)")
    print("-" * 80)

    sorted_hours = sorted(analysis['by_hour'].items(), key=lambda x: -x[1]['win_rate'])
    print(f"{'Hour':>6} {'Trades':>8} {'Win Rate':>10} {'Avg Pips':>10}")
    for hour, data in sorted_hours[:8]:
        if data['trades'] >= 3:
            status = "⭐" if data['win_rate'] >= 55 else ""
            print(f"{hour:>6}h {data['trades']:>8} {data['win_rate']:>9.1f}% {data['avg_pnl_pips']:>+9.1f} {status}")

    # Month Analysis
    print("\n" + "-" * 80)
    print("MONTHLY ANALYSIS")
    print("-" * 80)
    print(f"{'Month':<6} {'Trades':>7} {'WR%':>7} {'Pips':>10} {'Chop':>7} {'ADX':>7} {'ATR':>7}")

    for month, data in analysis['by_month'].items():
        status = "✅" if data['total_pips'] > 0 else "❌"
        print(f"{month:<6} {data['trades']:>7} {data['win_rate']:>6.0f}% {data['total_pips']:>+9.1f} "
              f"{data['avg_choppiness']:>6.1f} {data['avg_adx']:>6.1f} {data['avg_atr']:>6.1f} {status}")

    # Session Analysis
    print("\n" + "-" * 80)
    print("SESSION ANALYSIS")
    print("-" * 80)

    for session, data in analysis['by_session'].items():
        print(f"{session:<12}: {data['trades']} trades, {data['win_rate']:.1f}% WR, {data['total_pips']:+.1f} pips")

    # Entry Type Analysis
    print("\n" + "-" * 80)
    print("ENTRY TYPE ANALYSIS")
    print("-" * 80)

    for entry, data in analysis['by_entry_type'].items():
        print(f"{entry:<12}: {data['trades']} trades, {data['win_rate']:.1f}% WR, {data['total_pips']:+.1f} pips")

    # EMA Alignment
    print("\n" + "-" * 80)
    print("EMA ALIGNMENT IMPACT")
    print("-" * 80)

    if 'ema_aligned' in analysis:
        print(f"EMA Aligned:     {analysis['ema_aligned']['trades']} trades, "
              f"{analysis['ema_aligned']['win_rate']:.1f}% WR, {analysis['ema_aligned']['total_pips']:+.1f} pips")
    if 'ema_not_aligned' in analysis:
        print(f"EMA Not Aligned: {analysis['ema_not_aligned']['trades']} trades, "
              f"{analysis['ema_not_aligned']['win_rate']:.1f}% WR, {analysis['ema_not_aligned']['total_pips']:+.1f} pips")

    # Optimal Thresholds
    print("\n" + "-" * 80)
    print("OPTIMAL THRESHOLDS (Single Factor)")
    print("-" * 80)

    for feature, data in thresholds.items():
        direction = "<" if data['direction'] == 'below' else ">"
        print(f"{feature:<20}: {direction} {data['threshold']:.1f} -> {data['win_rate']:.1f}% WR ({data['trades']} trades)")

    # Perfect Conditions
    print("\n" + "-" * 80)
    print("BEST FILTER COMBINATIONS")
    print("-" * 80)
    print(f"{'Chop<':>7} {'ADX>':>7} {'Conf>':>7} {'Qual>':>7} {'Trades':>8} {'WR%':>7} {'Pips':>10}")

    for combo in perfect['best_combinations'][:5]:
        print(f"{combo['chop_max']:>7} {combo['adx_min']:>7} {combo['confidence_min']:>7} "
              f"{combo['quality_min']:>7} {combo['trades']:>8} {combo['win_rate']:>6.1f}% {combo['total_pips']:>+9.1f}")

    # Best combination
    if perfect['highest_wr']:
        best = perfect['highest_wr']
        print("\n" + "=" * 80)
        print("RECOMMENDED FILTER SETTINGS (Highest Win Rate)")
        print("=" * 80)
        print(f"  Choppiness: < {best['chop_max']}")
        print(f"  ADX: > {best['adx_min']}")
        print(f"  Regime Confidence: > {best['confidence_min']}%")
        print(f"  POI Quality: > {best['quality_min']}")
        print(f"\n  Expected: {best['trades']} trades, {best['win_rate']:.1f}% WR, {best['total_pips']:+.1f} pips")

    print("\n" + "=" * 80)


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 80)
    print("SURGE-WSI DEEP MARKET ANALYSIS")
    print("Analyzing 13 months of trades to find optimal conditions")
    print("=" * 80)

    print("\n[1/5] Fetching H1 data...")
    symbol = "GBPUSD"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    df = await fetch_data(symbol, "H1", start, end)
    if df.empty:
        print("ERROR: No data")
        return
    print(f"      Loaded {len(df)} bars")

    print("\n[2/5] Calculating indicators...")
    df = calculate_indicators(df)
    print("      Done")

    print("\n[3/5] Simulating trades with full detail capture...")
    trades = simulate_trades_with_details(df)
    print(f"      Captured {len(trades)} trades")

    print("\n[4/5] Analyzing winning conditions...")
    analysis = analyze_winning_conditions(trades)
    thresholds = find_optimal_thresholds(trades)
    perfect = find_perfect_conditions(trades)

    print("\n[5/5] Generating report...")
    print_analysis_report(analysis, thresholds, perfect)

    # Save detailed trade data for further analysis
    trade_df = pd.DataFrame(trades)
    output_path = Path(__file__).parent / "trade_analysis_data.csv"
    trade_df.to_csv(output_path, index=False)
    print(f"\nDetailed trade data saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
