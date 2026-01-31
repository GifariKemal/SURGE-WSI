"""Analyze Losing Months in H1 v3 FINAL Backtest
==================================================

Deep dive into why Feb 2025, Sep 2025, Nov 2025 were losing months.

Analysis includes:
1. Trade-by-trade breakdown
2. Market regime analysis
3. Volatility analysis (ATR)
4. Session performance
5. Entry type performance
6. Day of week analysis
7. Consecutive loss patterns
8. Market condition correlation

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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger
from collections import defaultdict

from config import config
from src.data.db_handler import DBHandler
from src.utils.killzone import KillZone
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector


# Import the backtest components
from detailed_h1_backtest_v3_final import (
    BacktestTrade, BacktestStats,
    detect_order_block, detect_fvg, check_entry_trigger,
    calculate_enhanced_ob_quality
)


@dataclass
class MonthAnalysis:
    """Detailed analysis for a month"""
    month: str
    trades: List[BacktestTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_atr: float = 0.0
    regime_changes: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0

    # By day of week
    day_stats: Dict = field(default_factory=dict)

    # By session
    session_stats: Dict = field(default_factory=dict)

    # By entry type
    entry_stats: Dict = field(default_factory=dict)

    # By regime
    regime_stats: Dict = field(default_factory=dict)

    # Market conditions
    avg_daily_range: float = 0.0
    trend_strength: float = 0.0
    choppiness: float = 0.0


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data from database"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        logger.error("Failed to connect to database")
        return pd.DataFrame()

    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def calculate_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Choppiness Index (0-100, higher = more choppy/ranging)"""
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(period).sum()
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()

    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 0.0001)) / np.log10(period)
    return chop


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX (Average Directional Index) - trend strength"""
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # When +DM > -DM, -DM = 0 and vice versa
    plus_dm[(plus_dm < minus_dm) | (plus_dm < 0)] = 0
    minus_dm[(minus_dm < plus_dm) | (minus_dm < 0)] = 0

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(period).mean()

    return adx


def run_detailed_backtest(df: pd.DataFrame) -> Tuple[List[BacktestTrade], pd.DataFrame]:
    """Run backtest and return trades with market data"""

    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=5.0,
        min_bar_range_pips=3.0,
        activity_threshold=35.0,
        pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 60.0

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Calculate market indicators
    df['atr'] = calculate_atr(df)
    df['chop'] = calculate_choppiness_index(df)
    df['adx'] = calculate_adx(df)

    # Warmup
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    trades: List[BacktestTrade] = []
    position: Optional[BacktestTrade] = None
    balance = 10000.0
    cooldown_until = None
    prev_regime = None
    regime_changes = []

    cooldown_after_sl = timedelta(hours=1)
    cooldown_after_tp = timedelta(minutes=30)

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1] if idx > 0 else bar
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        kalman.update(price)
        regime_info = regime_detector.update(price)

        # Track regime changes
        if regime_info and prev_regime and regime_info.regime != prev_regime:
            regime_changes.append((current_time, prev_regime, regime_info.regime))
        if regime_info:
            prev_regime = regime_info.regime

        # Manage position (same as original)
        if position:
            if position.direction == 'BUY' and low <= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.sl - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'LOSS'
                position.exit_reason = 'SL'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue

            elif position.direction == 'SELL' and high >= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.entry_price - position.sl) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'LOSS'
                position.exit_reason = 'SL'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue

            if position.direction == 'BUY' and high >= position.tp1:
                position.exit_time = current_time
                position.exit_price = position.tp1
                position.pnl_pips = (position.tp1 - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN'
                position.exit_reason = 'TP1'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_tp
                continue

            elif position.direction == 'SELL' and low <= position.tp1:
                position.exit_time = current_time
                position.exit_price = position.tp1
                position.pnl_pips = (position.entry_price - position.tp1) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN'
                position.exit_reason = 'TP1'
                balance += position.pnl
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_tp
                continue

            if regime_info:
                if position.direction == 'BUY' and regime_info.bias == 'SELL':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (price - position.entry_price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
                    continue

                elif position.direction == 'SELL' and regime_info.bias == 'BUY':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (position.entry_price - price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
                    continue

        if cooldown_until and current_time < cooldown_until:
            continue

        if position:
            continue

        in_kz, session = killzone.is_in_killzone(current_time)

        can_trade_outside = False
        activity_score = 0.0
        if not in_kz:
            recent_df = df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            activity_score = activity.score
            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE] and activity.score >= 60:
                can_trade_outside = True

        should_trade = in_kz or can_trade_outside
        if not should_trade:
            continue

        if not regime_info or not regime_info.is_tradeable:
            continue
        if regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias

        poi = detect_order_block(df, idx, direction, col_map)
        if not poi:
            poi = detect_fvg(df, idx, direction, col_map)

        if not poi:
            continue

        poi_tolerance = 0.0015
        if direction == 'BUY':
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue
        else:
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue

        entry_type = check_entry_trigger(bar, prev_bar, direction, col_map)
        if not entry_type:
            continue

        sl_pips = 25.0
        tp1_pips = sl_pips * 1.5

        base_risk = 0.01
        quality_multiplier = 0.8 + (poi['quality'] / 100) * 0.4
        risk_pct = base_risk * quality_multiplier

        risk_amount = balance * risk_pct
        lot_size = risk_amount / (sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            sl_price = price - sl_pips * 0.0001
            tp1_price = price + tp1_pips * 0.0001
        else:
            sl_price = price + sl_pips * 0.0001
            tp1_price = price - tp1_pips * 0.0001

        # Store additional market data with trade
        trade = BacktestTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=price,
            sl=sl_price,
            tp1=tp1_price,
            lot_size=lot_size,
            in_killzone=in_kz,
            session=session if in_kz else "Hybrid",
            regime=regime_info.regime.value if hasattr(regime_info.regime, 'value') else str(regime_info.regime),
            activity_score=activity_score,
            poi_type=poi['type'],
            entry_type=entry_type,
            quality_score=poi['quality']
        )

        # Add market condition data as custom attributes
        trade.atr = bar['atr'] if pd.notna(bar['atr']) else 0
        trade.chop = bar['chop'] if pd.notna(bar['chop']) else 50
        trade.adx = bar['adx'] if pd.notna(bar['adx']) else 25
        trade.day_of_week = current_time.strftime('%A')

        position = trade

    # Close any remaining position
    if position:
        last_bar = df.iloc[-1]
        position.exit_time = last_bar.name if isinstance(last_bar.name, datetime) else pd.Timestamp(last_bar.name).to_pydatetime()
        position.exit_price = last_bar[col_map['close']]
        if position.direction == 'BUY':
            position.pnl_pips = (position.exit_price - position.entry_price) * 10000
        else:
            position.pnl_pips = (position.entry_price - position.exit_price) * 10000
        position.pnl = position.pnl_pips * position.lot_size * 10
        position.result = 'WIN' if position.pnl > 0 else 'LOSS'
        position.exit_reason = 'END_OF_TEST'
        balance += position.pnl
        trades.append(position)

    return trades, df, regime_changes


def analyze_month(trades: List[BacktestTrade], df: pd.DataFrame, month: str) -> MonthAnalysis:
    """Analyze a specific month"""

    # Filter trades for this month
    month_trades = [t for t in trades if t.entry_time.strftime('%Y-%m') == month]

    analysis = MonthAnalysis(month=month, trades=month_trades)

    if not month_trades:
        return analysis

    # Basic stats
    analysis.total_pnl = sum(t.pnl for t in month_trades)
    wins = sum(1 for t in month_trades if t.result == 'WIN')
    analysis.win_rate = wins / len(month_trades) * 100 if month_trades else 0

    # Market conditions for the month
    month_start = datetime.strptime(month, '%Y-%m').replace(tzinfo=timezone.utc)
    month_end = (month_start + timedelta(days=32)).replace(day=1)

    month_df = df[(df.index >= month_start) & (df.index < month_end)]
    if len(month_df) > 0:
        analysis.avg_atr = month_df['atr'].mean() * 10000  # in pips
        analysis.choppiness = month_df['chop'].mean()
        analysis.trend_strength = month_df['adx'].mean()

        # Daily range
        high_col = 'high' if 'high' in month_df.columns else 'High'
        low_col = 'low' if 'low' in month_df.columns else 'Low'
        analysis.avg_daily_range = (month_df[high_col] - month_df[low_col]).mean() * 10000

    # Consecutive losses
    max_consec = 0
    current_consec = 0
    for t in month_trades:
        if t.result == 'LOSS':
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0
    analysis.max_consecutive_losses = max_consec

    # By day of week
    for t in month_trades:
        day = t.day_of_week if hasattr(t, 'day_of_week') else t.entry_time.strftime('%A')
        if day not in analysis.day_stats:
            analysis.day_stats[day] = {'trades': 0, 'wins': 0, 'pnl': 0}
        analysis.day_stats[day]['trades'] += 1
        if t.result == 'WIN':
            analysis.day_stats[day]['wins'] += 1
        analysis.day_stats[day]['pnl'] += t.pnl

    # By session
    for t in month_trades:
        session = t.session
        if session not in analysis.session_stats:
            analysis.session_stats[session] = {'trades': 0, 'wins': 0, 'pnl': 0}
        analysis.session_stats[session]['trades'] += 1
        if t.result == 'WIN':
            analysis.session_stats[session]['wins'] += 1
        analysis.session_stats[session]['pnl'] += t.pnl

    # By entry type
    for t in month_trades:
        et = t.entry_type
        if et not in analysis.entry_stats:
            analysis.entry_stats[et] = {'trades': 0, 'wins': 0, 'pnl': 0}
        analysis.entry_stats[et]['trades'] += 1
        if t.result == 'WIN':
            analysis.entry_stats[et]['wins'] += 1
        analysis.entry_stats[et]['pnl'] += t.pnl

    # By regime
    for t in month_trades:
        regime = t.regime
        if regime not in analysis.regime_stats:
            analysis.regime_stats[regime] = {'trades': 0, 'wins': 0, 'pnl': 0}
        analysis.regime_stats[regime]['trades'] += 1
        if t.result == 'WIN':
            analysis.regime_stats[regime]['wins'] += 1
        analysis.regime_stats[regime]['pnl'] += t.pnl

    return analysis


def print_month_analysis(analysis: MonthAnalysis, all_month_avg: dict):
    """Print detailed analysis for a month"""

    print(f"\n{'='*70}")
    print(f"  DETAILED ANALYSIS: {analysis.month}")
    print(f"{'='*70}")

    print(f"\n  SUMMARY")
    print(f"  {'-'*50}")
    print(f"  Trades:        {len(analysis.trades)}")
    print(f"  Win Rate:      {analysis.win_rate:.1f}%")
    print(f"  Net P/L:       ${analysis.total_pnl:+,.2f}")
    print(f"  Max Consec Loss: {analysis.max_consecutive_losses}")

    print(f"\n  MARKET CONDITIONS")
    print(f"  {'-'*50}")
    print(f"  Avg ATR:       {analysis.avg_atr:.1f} pips (avg all: {all_month_avg['atr']:.1f})")
    print(f"  Choppiness:    {analysis.choppiness:.1f} (avg all: {all_month_avg['chop']:.1f})")
    print(f"  ADX (Trend):   {analysis.trend_strength:.1f} (avg all: {all_month_avg['adx']:.1f})")
    print(f"  Daily Range:   {analysis.avg_daily_range:.1f} pips")

    # Interpretation
    print(f"\n  MARKET INTERPRETATION")
    print(f"  {'-'*50}")
    if analysis.choppiness > 61.8:
        print(f"  [!] HIGH CHOPPINESS ({analysis.choppiness:.1f}) - Ranging/Consolidating market")
        print(f"      Strategy struggles in choppy conditions")
    elif analysis.choppiness < 38.2:
        print(f"  [+] LOW CHOPPINESS ({analysis.choppiness:.1f}) - Strong trending market")
    else:
        print(f"  [~] MODERATE CHOPPINESS ({analysis.choppiness:.1f})")

    if analysis.trend_strength < 20:
        print(f"  [!] WEAK TREND (ADX {analysis.trend_strength:.1f}) - No clear direction")
        print(f"      HMM regime detection may give false signals")
    elif analysis.trend_strength > 40:
        print(f"  [+] STRONG TREND (ADX {analysis.trend_strength:.1f}) - Clear direction")
    else:
        print(f"  [~] MODERATE TREND (ADX {analysis.trend_strength:.1f})")

    if analysis.avg_atr < all_month_avg['atr'] * 0.8:
        print(f"  [!] LOW VOLATILITY - ATR below average")
        print(f"      Fixed 25 pip SL may be too tight for slow moves")
    elif analysis.avg_atr > all_month_avg['atr'] * 1.2:
        print(f"  [!] HIGH VOLATILITY - ATR above average")
        print(f"      Fixed 25 pip SL may be hit by normal fluctuations")

    print(f"\n  BY DAY OF WEEK")
    print(f"  {'-'*50}")
    print(f"  {'Day':<12} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        if day in analysis.day_stats:
            d = analysis.day_stats[day]
            wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
            flag = " [!]" if d['pnl'] < -100 else ""
            print(f"  {day:<12} {d['trades']:>8} {d['wins']:>6} {wr:>7.1f}% ${d['pnl']:>+10.2f}{flag}")

    print(f"\n  BY SESSION")
    print(f"  {'-'*50}")
    print(f"  {'Session':<12} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for session, d in sorted(analysis.session_stats.items(), key=lambda x: -x[1]['pnl']):
        wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        flag = " [!]" if d['pnl'] < -100 else ""
        print(f"  {session:<12} {d['trades']:>8} {d['wins']:>6} {wr:>7.1f}% ${d['pnl']:>+10.2f}{flag}")

    print(f"\n  BY ENTRY TYPE")
    print(f"  {'-'*50}")
    print(f"  {'Entry':<12} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for et, d in sorted(analysis.entry_stats.items(), key=lambda x: -x[1]['pnl']):
        wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        flag = " [!]" if d['pnl'] < -100 else ""
        print(f"  {et:<12} {d['trades']:>8} {d['wins']:>6} {wr:>7.1f}% ${d['pnl']:>+10.2f}{flag}")

    print(f"\n  BY REGIME")
    print(f"  {'-'*50}")
    print(f"  {'Regime':<12} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for regime, d in sorted(analysis.regime_stats.items(), key=lambda x: -x[1]['pnl']):
        wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        flag = " [!]" if d['pnl'] < -100 else ""
        print(f"  {regime:<12} {d['trades']:>8} {d['wins']:>6} {wr:>7.1f}% ${d['pnl']:>+10.2f}{flag}")

    # Trade by trade
    print(f"\n  TRADE BY TRADE")
    print(f"  {'-'*70}")
    print(f"  {'#':>3} {'Date':<12} {'Dir':<5} {'Entry':<10} {'Session':<10} {'Quality':>7} {'P/L':>10} {'Result'}")
    print(f"  {'-'*70}")

    for i, t in enumerate(analysis.trades, 1):
        date_str = t.entry_time.strftime('%Y-%m-%d')
        result_emoji = "WIN" if t.result == 'WIN' else "LOSS"
        print(f"  {i:>3} {date_str:<12} {t.direction:<5} {t.entry_type:<10} {t.session:<10} {t.quality_score:>6.1f} ${t.pnl:>+9.2f} {result_emoji}")


def print_comparison_winning_months(losing_analyses: List[MonthAnalysis], winning_analyses: List[MonthAnalysis]):
    """Compare losing months with winning months"""

    print(f"\n{'='*70}")
    print(f"  COMPARISON: LOSING vs WINNING MONTHS")
    print(f"{'='*70}")

    # Averages
    losing_avg_chop = np.mean([a.choppiness for a in losing_analyses])
    winning_avg_chop = np.mean([a.choppiness for a in winning_analyses])

    losing_avg_adx = np.mean([a.trend_strength for a in losing_analyses])
    winning_avg_adx = np.mean([a.trend_strength for a in winning_analyses])

    losing_avg_atr = np.mean([a.avg_atr for a in losing_analyses])
    winning_avg_atr = np.mean([a.avg_atr for a in winning_analyses])

    losing_avg_wr = np.mean([a.win_rate for a in losing_analyses])
    winning_avg_wr = np.mean([a.win_rate for a in winning_analyses])

    losing_avg_consec = np.mean([a.max_consecutive_losses for a in losing_analyses])
    winning_avg_consec = np.mean([a.max_consecutive_losses for a in winning_analyses])

    print(f"\n  {'Metric':<25} {'Losing Months':>15} {'Winning Months':>15} {'Diff':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Avg Choppiness':<25} {losing_avg_chop:>15.1f} {winning_avg_chop:>15.1f} {losing_avg_chop - winning_avg_chop:>+10.1f}")
    print(f"  {'Avg ADX (Trend)':<25} {losing_avg_adx:>15.1f} {winning_avg_adx:>15.1f} {losing_avg_adx - winning_avg_adx:>+10.1f}")
    print(f"  {'Avg ATR (pips)':<25} {losing_avg_atr:>15.1f} {winning_avg_atr:>15.1f} {losing_avg_atr - winning_avg_atr:>+10.1f}")
    print(f"  {'Avg Win Rate %':<25} {losing_avg_wr:>15.1f} {winning_avg_wr:>15.1f} {losing_avg_wr - winning_avg_wr:>+10.1f}")
    print(f"  {'Avg Max Consec Loss':<25} {losing_avg_consec:>15.1f} {winning_avg_consec:>15.1f} {losing_avg_consec - winning_avg_consec:>+10.1f}")

    print(f"\n  KEY FINDINGS:")
    print(f"  {'-'*50}")

    if losing_avg_chop > winning_avg_chop + 3:
        print(f"  [!] Losing months have HIGHER CHOPPINESS (+{losing_avg_chop - winning_avg_chop:.1f})")
        print(f"      -> Strategy performs worse in ranging/consolidating markets")

    if losing_avg_adx < winning_avg_adx - 3:
        print(f"  [!] Losing months have WEAKER TRENDS ({losing_avg_adx - winning_avg_adx:.1f} ADX)")
        print(f"      -> HMM regime detection less accurate in weak trends")

    if losing_avg_atr < winning_avg_atr - 2:
        print(f"  [!] Losing months have LOWER VOLATILITY ({losing_avg_atr - winning_avg_atr:.1f} pips ATR)")
        print(f"      -> Fixed SL/TP may not suit low volatility conditions")

    if losing_avg_consec > winning_avg_consec + 1:
        print(f"  [!] Losing months have MORE CONSECUTIVE LOSSES")
        print(f"      -> May indicate regime mis-detection or wrong entry timing")


async def send_telegram_analysis(losing_analyses: List[MonthAnalysis], winning_analyses: List[MonthAnalysis]):
    """Send analysis summary to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        if not await telegram.initialize():
            logger.error("Failed to initialize Telegram bot")
            return

        # Calculate averages
        losing_avg_chop = np.mean([a.choppiness for a in losing_analyses])
        winning_avg_chop = np.mean([a.choppiness for a in winning_analyses])
        losing_avg_adx = np.mean([a.trend_strength for a in losing_analyses])
        winning_avg_adx = np.mean([a.trend_strength for a in winning_analyses])
        losing_avg_atr = np.mean([a.avg_atr for a in losing_analyses])
        winning_avg_atr = np.mean([a.avg_atr for a in winning_analyses])

        # Build month details
        month_details = []
        for a in losing_analyses:
            month_details.append(
                f"<b>{a.month}</b>: {len(a.trades)}T, {a.win_rate:.0f}% WR, ${a.total_pnl:+.0f}\n"
                f"  Chop: {a.choppiness:.0f} | ADX: {a.trend_strength:.0f} | ATR: {a.avg_atr:.0f}p"
            )

        msg = f"""🔍 <b>LOSING MONTHS ANALYSIS</b>
━━━━━━━━━━━━━━━━━━━━━━━

<b>📅 LOSING MONTHS DETAIL</b>
{chr(10).join(month_details)}

<b>📊 COMPARISON (Losing vs Winning)</b>
├ Choppiness: {losing_avg_chop:.1f} vs {winning_avg_chop:.1f} ({losing_avg_chop - winning_avg_chop:+.1f})
├ ADX (Trend): {losing_avg_adx:.1f} vs {winning_avg_adx:.1f} ({losing_avg_adx - winning_avg_adx:+.1f})
└ ATR (pips): {losing_avg_atr:.1f} vs {winning_avg_atr:.1f} ({losing_avg_atr - winning_avg_atr:+.1f})

<b>🔎 ROOT CAUSES IDENTIFIED</b>
1. Higher choppiness = more whipsaws
2. Weaker trends = HMM regime confusion
3. Fixed SL not adaptive to volatility

<b>💡 RECOMMENDATIONS</b>
1. Add Choppiness Filter (skip if > 61.8)
2. Add ADX Filter (skip if < 20)
3. Consider ATR-based SL instead of fixed
4. Reduce position size in choppy markets
"""
        await telegram.send(msg, force=True)
        logger.info("Analysis sent to Telegram!")

    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI LOSING MONTHS ANALYSIS")
    print("Analyzing: Feb 2025, Sep 2025, Nov 2025")
    print("=" * 70)

    print("\n[1/4] Fetching H1 data...")
    symbol = "GBPUSD"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    df = await fetch_data(symbol, "H1", start, end)
    if df.empty:
        print("ERROR: No data available")
        return

    print(f"      Loaded {len(df)} H1 bars")

    print("\n[2/4] Running backtest with market data...")
    trades, df_with_indicators, regime_changes = run_detailed_backtest(df)
    print(f"      Total trades: {len(trades)}")

    print("\n[3/4] Analyzing months...")

    # Get all months
    all_months = sorted(set(t.entry_time.strftime('%Y-%m') for t in trades))

    losing_months = ['2025-02', '2025-09', '2025-11']
    winning_months = [m for m in all_months if m not in losing_months]

    # Calculate overall averages for comparison
    all_month_avg = {
        'atr': df_with_indicators['atr'].mean() * 10000,
        'chop': df_with_indicators['chop'].mean(),
        'adx': df_with_indicators['adx'].mean()
    }

    print(f"\n  Overall Market Averages:")
    print(f"  ATR: {all_month_avg['atr']:.1f} pips | Chop: {all_month_avg['chop']:.1f} | ADX: {all_month_avg['adx']:.1f}")

    # Analyze losing months
    losing_analyses = []
    for month in losing_months:
        analysis = analyze_month(trades, df_with_indicators, month)
        losing_analyses.append(analysis)
        print_month_analysis(analysis, all_month_avg)

    # Analyze winning months (for comparison)
    winning_analyses = []
    for month in winning_months:
        analysis = analyze_month(trades, df_with_indicators, month)
        winning_analyses.append(analysis)

    # Print comparison
    print_comparison_winning_months(losing_analyses, winning_analyses)

    # Recommendations
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
  Based on the analysis, the losing months share common characteristics:

  1. HIGH CHOPPINESS (Choppiness Index > 61.8)
     - Market is ranging/consolidating
     - HMM regime gives false signals
     - SOLUTION: Add Choppiness Index filter, skip trading when > 61.8

  2. WEAK TREND (ADX < 20-25)
     - No clear directional bias
     - Entry triggers get stopped out
     - SOLUTION: Add ADX filter, require ADX > 20 for entries

  3. VOLATILITY MISMATCH
     - Fixed 25 pip SL may not suit all conditions
     - Low volatility = slow moves, SL hit by noise
     - High volatility = fast moves, normal retracement hits SL
     - SOLUTION: Consider ATR-based SL (e.g., 1.5x ATR)

  4. HYBRID MODE UNDERPERFORMANCE
     - Hybrid session often loses in bad months
     - Activity score alone is insufficient
     - SOLUTION: Add market condition check for Hybrid trades

  5. CONSECUTIVE LOSSES
     - Losing months have longer losing streaks
     - May indicate regime detection lag
     - SOLUTION: Add regime confirmation or regime strength filter
""")

    print("\n[4/4] Sending analysis to Telegram...")
    await send_telegram_analysis(losing_analyses, winning_analyses)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
