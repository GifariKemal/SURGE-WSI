"""Detailed H1 Backtest v4.5 - Adaptive Market Intelligence
============================================================

v4.5 Enhancements based on academic research:
1. ADWIN Drift Detection - detect market regime changes early
2. Session-Specific Parameters - different settings per session
3. Multi-Factor Confluence Validation - 8-factor validation
4. Dynamic Position Sizing - Kelly-based adjustments

Research References:
- River ADWIN: https://riverml.xyz/dev/api/drift/ADWIN/
- NBER FX Microstructure: Market sessions have different characteristics
- Professional Traders: Multi-factor confluence validation

Previous versions:
- v2:       106 trades, 48.1% WR, +$2,018, PF 1.22 [BASELINE]
- v3 FINAL: 100 trades, 51.0% WR, +$2,669, PF 1.50
- v4:        89 trades, 49.4% WR, +$2,131, PF 1.49, 2 losing months

Target v4.5: Higher WR, 0-1 losing months

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
from typing import List, Dict, Optional
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.utils.killzone import KillZone
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel
from src.utils.market_condition_filter import MarketConditionFilter
from src.utils.drift_detector import ADWINDriftDetector
from src.utils.session_profiles import SessionProfileManager, TradingSession
from src.utils.confluence_validator import ConfluenceValidator
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector


# ============================================================================
# ENHANCED OB QUALITY (same as v4)
# ============================================================================

def calculate_enhanced_ob_quality(df, ob_idx, direction, col_map):
    """Enhanced OB quality scoring"""
    quality = 0.0
    if ob_idx < 5 or ob_idx >= len(df) - 3:
        return 50

    ob_bar = df.iloc[ob_idx]
    next_bars = df.iloc[ob_idx+1:ob_idx+4]
    open_col, high_col, low_col, close_col = col_map['open'], col_map['high'], col_map['low'], col_map['close']

    # 1. Base quality from impulse move
    if direction == 'BUY':
        impulse = next_bars[close_col].max() - ob_bar[low_col]
    else:
        impulse = ob_bar[high_col] - next_bars[close_col].min()
    impulse_pips = impulse * 10000
    quality += min(50, impulse_pips * 2.5)

    # 2. Wick analysis
    ob_range = ob_bar[high_col] - ob_bar[low_col]
    if ob_range > 0:
        if direction == 'BUY':
            upper_wick = ob_bar[high_col] - max(ob_bar[open_col], ob_bar[close_col])
            wick_ratio = upper_wick / ob_range
        else:
            lower_wick = min(ob_bar[open_col], ob_bar[close_col]) - ob_bar[low_col]
            wick_ratio = lower_wick / ob_range
        if wick_ratio > 0.3:
            quality += 25
        elif wick_ratio > 0.2:
            quality += 15
        elif wick_ratio > 0.1:
            quality += 10

    # 3. Fresh zone bonus
    zone_high, zone_low = ob_bar[high_col], ob_bar[low_col]
    touched = False
    for i in range(ob_idx + 4, min(ob_idx + 15, len(df))):
        bar = df.iloc[i]
        if direction == 'BUY':
            if bar[low_col] <= zone_high:
                touched = True
                break
        else:
            if bar[high_col] >= zone_low:
                touched = True
                break
    if not touched:
        quality += 25

    return min(100, quality)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BacktestTrade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    lot_size: float = 0.1
    pnl: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""
    exit_reason: str = ""
    session: str = ""
    regime: str = ""
    poi_type: str = ""
    entry_type: str = ""
    quality_score: float = 0.0
    sl_pips: float = 25.0
    confluence_score: float = 0.0
    drift_status: str = ""
    is_thursday: bool = False


@dataclass
class BacktestStats:
    """Backtest statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    trades_per_day: float = 0.0

    # Session breakdown
    session_stats: Dict = field(default_factory=dict)

    # V4.5 metrics
    signals_total: int = 0
    signals_filtered_drift: int = 0
    signals_filtered_confluence: int = 0
    signals_filtered_session: int = 0
    avg_confluence_score: float = 0.0
    drift_events: int = 0

    monthly_stats: Dict = field(default_factory=dict)
    thursday_trades: int = 0
    thursday_pnl: float = 0.0


# ============================================================================
# DATA FETCHING
# ============================================================================

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


# ============================================================================
# POI DETECTION
# ============================================================================

def detect_order_block(df, idx, direction, col_map, lookback=15):
    """Detect Order Block with enhanced quality"""
    if idx < lookback + 3:
        return None

    close_col, open_col = col_map['close'], col_map['open']
    high_col, low_col = col_map['high'], col_map['low']
    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 3):
        bar = recent.iloc[i]
        next_bars = recent.iloc[i+1:i+4]
        actual_idx = idx - lookback + i

        if direction == 'BUY':
            if bar[close_col] < bar[open_col]:
                move_up = next_bars[close_col].max() - bar[low_col]
                if move_up > 0.0010:
                    quality = calculate_enhanced_ob_quality(df, actual_idx, direction, col_map)
                    return {'type': 'OB', 'direction': 'BUY',
                            'zone_high': bar[high_col], 'zone_low': bar[low_col], 'quality': quality}
        else:
            if bar[close_col] > bar[open_col]:
                move_down = bar[high_col] - next_bars[close_col].min()
                if move_down > 0.0010:
                    quality = calculate_enhanced_ob_quality(df, actual_idx, direction, col_map)
                    return {'type': 'OB', 'direction': 'SELL',
                            'zone_high': bar[high_col], 'zone_low': bar[low_col], 'quality': quality}
    return None


def detect_fvg(df, idx, direction, col_map, lookback=8):
    """Detect Fair Value Gap"""
    if idx < lookback + 3:
        return None

    high_col, low_col = col_map['high'], col_map['low']
    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 2):
        bar1, bar3 = recent.iloc[i], recent.iloc[i+2]

        if direction == 'BUY':
            gap = bar3[low_col] - bar1[high_col]
            if gap > 0.0003:
                return {'type': 'FVG', 'direction': 'BUY',
                        'zone_high': bar3[low_col], 'zone_low': bar1[high_col],
                        'quality': min(100, gap * 10000 * 2)}
        else:
            gap = bar1[low_col] - bar3[high_col]
            if gap > 0.0003:
                return {'type': 'FVG', 'direction': 'SELL',
                        'zone_high': bar1[low_col], 'zone_low': bar3[high_col],
                        'quality': min(100, gap * 10000 * 2)}
    return None


# ============================================================================
# ENTRY TRIGGER
# ============================================================================

def check_entry_trigger(bar, prev_bar, direction, col_map):
    """Check for entry trigger"""
    o, h, l, c = bar[col_map['open']], bar[col_map['high']], bar[col_map['low']], bar[col_map['close']]
    total_range = h - l
    if total_range < 0.0003:
        return None

    body = abs(c - o)
    is_bullish, is_bearish = c > o, c < o
    po, ph, pl, pc = prev_bar[col_map['open']], prev_bar[col_map['high']], prev_bar[col_map['low']], prev_bar[col_map['close']]

    if direction == 'BUY':
        lower_wick = min(o, c) - l
        if lower_wick > body and lower_wick > total_range * 0.5:
            return "REJECTION"
        if is_bullish and body > total_range * 0.6:
            return "MOMENTUM"
        if is_bullish and c > ph and o <= pl:
            return "ENGULF"
        if l > pl and is_bullish:
            return "HIGHER_LOW"
    else:
        upper_wick = h - max(o, c)
        if upper_wick > body and upper_wick > total_range * 0.5:
            return "REJECTION"
        if is_bearish and body > total_range * 0.6:
            return "MOMENTUM"
        if is_bearish and c < pl and o >= ph:
            return "ENGULF"
    return None


# ============================================================================
# MAIN BACKTEST v4.5
# ============================================================================

def run_backtest(df: pd.DataFrame) -> tuple:
    """Run v4.5 backtest with adaptive intelligence"""

    # Initialize components
    killzone = KillZone()
    activity_filter = DynamicActivityFilter(min_atr_pips=5.0, min_bar_range_pips=3.0,
                                            activity_threshold=35.0, pip_size=0.0001)
    activity_filter.outside_kz_min_score = 60.0

    market_filter = MarketConditionFilter(
        chop_ranging_threshold=65.0, adx_weak_threshold=18.0,
        atr_sl_multiplier=1.5, min_sl_pips=15.0, max_sl_pips=40.0,
        regime_confidence_threshold=60.0, enable_thursday_filter=True,
        thursday_position_multiplier=0.6, skip_thursday_in_weak_market=True,
        min_confluence_score=45.0
    )

    # V4.5 NEW COMPONENTS (tuned v4.5.1)
    # Delta increased from 0.002 to 0.005 = less sensitive drift detection
    drift_detector = ADWINDriftDetector(delta=0.005, min_samples=50)
    session_manager = SessionProfileManager(hybrid_mode=True)
    # Reduced min_score from 55 to 50, min_factors from 5 to 4
    confluence_validator = ConfluenceValidator(min_total_score=50, min_factors_passed=4)

    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    print("      Warming up indicators...")
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    trades: List[BacktestTrade] = []
    position: Optional[BacktestTrade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    # Tracking
    signals_total = 0
    filtered_drift = 0
    filtered_confluence = 0
    filtered_session = 0
    drift_events = 0
    confluence_scores = []

    cooldown_after_sl = timedelta(hours=1)
    cooldown_after_tp = timedelta(minutes=30)

    print("      Processing bars...")
    total_bars = len(df) - 100
    prev_price = df.iloc[99][col_map['close']]

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        # Update indicators
        kalman.update(price)
        regime_info = regime_detector.update(price)
        drift_detector.update_price(price, prev_price)
        prev_price = price

        if (idx - 100) % 500 == 0:
            pct = (idx - 100) / total_bars * 100
            print(f"      Progress: {pct:.0f}% ({idx-100}/{total_bars} bars)")

        # Manage position
        if position:
            if position.direction == 'BUY' and low <= position.sl:
                position.exit_time, position.exit_price = current_time, position.sl
                position.pnl_pips = (position.sl - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'LOSS', 'SL'
                balance += position.pnl
                drift_detector.update_trade_result(False)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue
            elif position.direction == 'SELL' and high >= position.sl:
                position.exit_time, position.exit_price = current_time, position.sl
                position.pnl_pips = (position.entry_price - position.sl) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'LOSS', 'SL'
                balance += position.pnl
                drift_detector.update_trade_result(False)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue

            if position.direction == 'BUY' and high >= position.tp1:
                position.exit_time, position.exit_price = current_time, position.tp1
                position.pnl_pips = (position.tp1 - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'WIN', 'TP1'
                balance += position.pnl
                drift_detector.update_trade_result(True)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_tp
                continue
            elif position.direction == 'SELL' and low <= position.tp1:
                position.exit_time, position.exit_price = current_time, position.tp1
                position.pnl_pips = (position.entry_price - position.tp1) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result, position.exit_reason = 'WIN', 'TP1'
                balance += position.pnl
                drift_detector.update_trade_result(True)
                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_tp
                continue

            # Regime flip exit
            if regime_info:
                if position.direction == 'BUY' and regime_info.bias == 'SELL':
                    position.exit_time, position.exit_price = current_time, price
                    position.pnl_pips = (price - position.entry_price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    drift_detector.update_trade_result(position.pnl > 0)
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
                    continue
                elif position.direction == 'SELL' and regime_info.bias == 'BUY':
                    position.exit_time, position.exit_price = current_time, price
                    position.pnl_pips = (position.entry_price - price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    drift_detector.update_trade_result(position.pnl > 0)
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
                    continue

        # Update drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Skip if in cooldown or has position
        if cooldown_until and current_time < cooldown_until:
            continue
        if position:
            continue

        # =====================================================================
        # V4.5: SESSION CHECK
        # =====================================================================
        session = session_manager.get_current_session(current_time)
        profile = session_manager.get_profile(current_time)
        is_optimal, session_reason = session_manager.is_optimal_trading_time(current_time)

        in_kz, kz_session = killzone.is_in_killzone(current_time)
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

        # Check regime
        if not regime_info or not regime_info.is_tradeable or regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias
        signals_total += 1

        # =====================================================================
        # V4.5: DRIFT CHECK (v4.5.2 - information only, not hard filter)
        # Drift detection is used to REDUCE position size, not skip trades
        # =====================================================================
        drift_result = drift_detector.detect(current_time)
        drift_detected_for_sizing = False
        if drift_result.drift_detected:
            drift_events += 1
            drift_detected_for_sizing = True
            # Only log, don't filter (use for position sizing instead)

        # =====================================================================
        # V4.5: MARKET CONDITION CHECK
        # =====================================================================
        recent_for_filter = df.iloc[max(0, idx-30):idx+1]
        regime_confidence = regime_info.confidence * 100 if hasattr(regime_info, 'confidence') else 80.0

        market_condition = market_filter.analyze(
            df=recent_for_filter, current_time=current_time,
            regime_confidence=regime_confidence, direction=direction
        )

        if not market_condition.can_trade:
            filtered_session += 1
            continue

        sl_pips = market_condition.suggested_sl_pips
        is_thursday = current_time.weekday() == 3

        # Find POI
        poi = detect_order_block(df, idx, direction, col_map)
        if not poi:
            poi = detect_fvg(df, idx, direction, col_map)
        if not poi:
            continue

        # Check POI zone
        poi_tolerance = 0.0015
        if direction == 'BUY':
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue
        else:
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue

        # Check entry trigger
        entry_type = check_entry_trigger(bar, prev_bar, direction, col_map)
        if not entry_type:
            continue

        # =====================================================================
        # V4.5: CONFLUENCE VALIDATION
        # =====================================================================
        kalman_state = kalman.last_state
        kalman_velocity = kalman_state.get('velocity', 0) if kalman_state else 0

        confluence_result = confluence_validator.validate(
            direction=direction,
            df=recent_for_filter,
            poi_quality=poi['quality'],
            regime_confidence=regime_confidence,
            regime_bias=regime_info.bias,
            session_name=profile.name,
            is_optimal_session=is_optimal,
            drift_detected=drift_detected_for_sizing,  # Use soft drift flag
            current_time=current_time,
            kalman_velocity=kalman_velocity
        )

        confluence_scores.append(confluence_result.total_score)

        # v4.5.2: Don't filter by confluence, just use for position sizing
        # if not confluence_result.can_trade:
        #     filtered_confluence += 1
        #     continue

        # =====================================================================
        # POSITION SIZING (with all adjustments)
        # =====================================================================
        tp1_pips = sl_pips * 1.5

        # Get adjusted parameters (drift reduces position size)
        adjusted_params = session_manager.get_adjusted_parameters(
            dt=current_time,
            base_risk=0.01,
            drift_detected=drift_detected_for_sizing  # Use soft drift flag
        )

        risk_pct = adjusted_params['risk_pct'] * confluence_result.risk_multiplier
        risk_amount = balance * risk_pct
        lot_size = risk_amount / (sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            sl_price = price - sl_pips * 0.0001
            tp1_price = price + tp1_pips * 0.0001
        else:
            sl_price = price + sl_pips * 0.0001
            tp1_price = price - tp1_pips * 0.0001

        # Create position
        position = BacktestTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=price,
            sl=sl_price,
            tp1=tp1_price,
            lot_size=lot_size,
            session=profile.session.value,
            regime=regime_info.regime.value if hasattr(regime_info.regime, 'value') else str(regime_info.regime),
            poi_type=poi['type'],
            entry_type=entry_type,
            quality_score=poi['quality'],
            sl_pips=sl_pips,
            confluence_score=confluence_result.total_score,
            drift_status='stable' if not drift_detected_for_sizing else drift_result.drift_type,
            is_thursday=is_thursday
        )

    # Close remaining position
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

    # Calculate stats
    stats = calculate_stats(trades, balance, max_dd, df)
    stats.signals_total = signals_total
    stats.signals_filtered_drift = filtered_drift
    stats.signals_filtered_confluence = filtered_confluence
    stats.signals_filtered_session = filtered_session
    stats.drift_events = drift_events
    stats.avg_confluence_score = np.mean(confluence_scores) if confluence_scores else 0

    return trades, stats, balance


def calculate_stats(trades, final_balance, max_dd, df):
    """Calculate statistics"""
    stats = BacktestStats()
    if not trades:
        return stats

    stats.total_trades = len(trades)
    stats.wins = sum(1 for t in trades if t.result == 'WIN')
    stats.losses = sum(1 for t in trades if t.result == 'LOSS')
    stats.win_rate = stats.wins / stats.total_trades * 100 if stats.total_trades > 0 else 0

    stats.total_pnl = sum(t.pnl for t in trades)
    stats.total_pips = sum(t.pnl_pips for t in trades)
    stats.max_drawdown = max_dd
    stats.max_drawdown_pct = max_dd / 10000 * 100

    winning_trades = [t.pnl for t in trades if t.pnl > 0]
    losing_trades = [abs(t.pnl) for t in trades if t.pnl < 0]

    stats.avg_win = np.mean(winning_trades) if winning_trades else 0
    stats.avg_loss = np.mean(losing_trades) if losing_trades else 0
    stats.largest_win = max(winning_trades) if winning_trades else 0
    stats.largest_loss = max(losing_trades) if losing_trades else 0

    total_wins = sum(winning_trades)
    total_losses = sum(losing_trades)
    stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades if t.exit_time and t.entry_time]
    stats.avg_trade_duration = np.mean(durations) if durations else 0

    if len(df) > 0:
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            stats.trades_per_day = stats.total_trades / days

    # Session stats
    for session in ['london', 'new_york', 'overlap', 'asia', 'off_hours']:
        session_trades = [t for t in trades if t.session == session]
        stats.session_stats[session] = {
            'trades': len(session_trades),
            'wins': sum(1 for t in session_trades if t.result == 'WIN'),
            'pnl': sum(t.pnl for t in session_trades)
        }

    # Thursday stats
    thursday_trades = [t for t in trades if t.is_thursday]
    stats.thursday_trades = len(thursday_trades)
    stats.thursday_pnl = sum(t.pnl for t in thursday_trades)

    # Monthly stats
    for t in trades:
        month_key = t.entry_time.strftime('%Y-%m')
        if month_key not in stats.monthly_stats:
            stats.monthly_stats[month_key] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pips': 0.0}
        stats.monthly_stats[month_key]['trades'] += 1
        if t.result == 'WIN':
            stats.monthly_stats[month_key]['wins'] += 1
        stats.monthly_stats[month_key]['pnl'] += t.pnl
        stats.monthly_stats[month_key]['pips'] += t.pnl_pips

    return stats


def print_report(stats, final_balance, trades):
    """Print detailed report"""
    print()
    print("=" * 70)
    print("H1 DETAILED BACKTEST v4.5 RESULTS")
    print("(Adaptive Market Intelligence)")
    print("=" * 70)
    print()

    print("V4.5 ENHANCEMENTS:")
    print("-" * 50)
    print("+ ADWIN Drift Detection (market change detection)")
    print("+ Session-Specific Parameters (London/NY/Overlap)")
    print("+ 8-Factor Confluence Validation")
    print("+ Dynamic Position Sizing (Kelly-based)")
    print()

    print("OVERALL PERFORMANCE")
    print("-" * 50)
    print(f"Initial Balance:     $10,000.00")
    print(f"Final Balance:       ${final_balance:,.2f}")
    print(f"Net P/L:             ${stats.total_pnl:+,.2f}")
    print(f"Return:              {(final_balance/10000-1)*100:+.1f}%")
    print(f"Total Pips:          {stats.total_pips:+.1f}")
    print()

    print("TRADE STATISTICS")
    print("-" * 50)
    print(f"Total Trades:        {stats.total_trades} ({stats.trades_per_day:.2f}/day)")
    print(f"Wins:                {stats.wins}")
    print(f"Losses:              {stats.losses}")
    print(f"Win Rate:            {stats.win_rate:.1f}%")
    print(f"Profit Factor:       {stats.profit_factor:.2f}")
    print()

    print("P/L ANALYSIS")
    print("-" * 50)
    print(f"Average Win:         ${stats.avg_win:,.2f}")
    print(f"Average Loss:        ${stats.avg_loss:,.2f}")
    print(f"Largest Win:         ${stats.largest_win:,.2f}")
    print(f"Largest Loss:        ${stats.largest_loss:,.2f}")
    print(f"Max Drawdown:        ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)")
    print(f"Avg Duration:        {stats.avg_trade_duration:.1f} hours")
    print()

    print("V4.5 ADAPTIVE FILTERS")
    print("-" * 50)
    print(f"Signals analyzed:    {stats.signals_total}")
    print(f"Filtered by drift:   {stats.signals_filtered_drift}")
    print(f"Filtered by conf:    {stats.signals_filtered_confluence}")
    print(f"Filtered by session: {stats.signals_filtered_session}")
    print(f"Drift events:        {stats.drift_events}")
    print(f"Avg confluence:      {stats.avg_confluence_score:.1f}/100")
    thursday_wr = sum(1 for t in trades if t.is_thursday and t.result == 'WIN') / stats.thursday_trades * 100 if stats.thursday_trades > 0 else 0
    print(f"Thursday trades:     {stats.thursday_trades} ({thursday_wr:.0f}% WR, ${stats.thursday_pnl:+,.2f})")
    print()

    # Entry type breakdown
    entry_types = {}
    for t in trades:
        et = t.entry_type
        if et not in entry_types:
            entry_types[et] = {'count': 0, 'wins': 0, 'pnl': 0}
        entry_types[et]['count'] += 1
        if t.result == 'WIN':
            entry_types[et]['wins'] += 1
        entry_types[et]['pnl'] += t.pnl

    print("ENTRY TYPE BREAKDOWN")
    print("-" * 60)
    print(f"{'Type':<15} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for et, data in sorted(entry_types.items(), key=lambda x: -x[1]['pnl']):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"{et:<15} {data['count']:>8} {data['wins']:>6} {wr:>7.1f}% ${data['pnl']:>+10.2f}")
    print()

    print("SESSION BREAKDOWN")
    print("-" * 60)
    print(f"{'Session':<15} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for session, data in stats.session_stats.items():
        if data['trades'] > 0:
            wr = data['wins'] / data['trades'] * 100
            print(f"{session:<15} {data['trades']:>8} {data['wins']:>6} {wr:>7.1f}% ${data['pnl']:>+10.2f}")
    print()

    print("MONTHLY PERFORMANCE")
    print("-" * 70)
    print(f"{'Month':<10} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Pips':>10} {'P/L':>12}")
    print("-" * 70)

    losing_months = 0
    for month, data in sorted(stats.monthly_stats.items()):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        status = "[-]" if data['pnl'] < 0 else "[+]"
        if data['pnl'] < 0:
            losing_months += 1
        print(f"{month:<10} {data['trades']:>8} {data['wins']:>6} {wr:>7.1f}% {data['pips']:>+9.1f} ${data['pnl']:>+10.2f} {status}")

    print("-" * 70)
    print(f"Losing months: {losing_months}/{len(stats.monthly_stats)}")
    print()

    print("=" * 70)
    print("VERSION COMPARISON")
    print("=" * 70)
    print("v2:       106 trades, 48.1% WR, +$2,018, PF 1.22 [BASELINE]")
    print("v3 FINAL: 100 trades, 51.0% WR, +$2,669, PF 1.50")
    print("v4:        89 trades, 49.4% WR, +$2,131, PF 1.49, 2 losing months")
    print(f"v4.5:    {stats.total_trades:>4} trades, {stats.win_rate:.1f}% WR, ${stats.total_pnl:+,.0f}, PF {stats.profit_factor:.2f}, {losing_months} losing")
    print("=" * 70)
    print()


async def send_telegram_report(stats, trades, final_balance):
    """Send report to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(bot_token=config.telegram.bot_token, chat_id=config.telegram.chat_id)
        if not await telegram.initialize():
            logger.error("Failed to initialize Telegram bot")
            return

        entry_types = {}
        for t in trades:
            et = t.entry_type
            if et not in entry_types:
                entry_types[et] = {'count': 0, 'pnl': 0}
            entry_types[et]['count'] += 1
            entry_types[et]['pnl'] += t.pnl
        entry_str = "\n".join([f"  {k}: {v['count']}T, ${v['pnl']:+.0f}" for k, v in sorted(entry_types.items(), key=lambda x: -x[1]['pnl'])])

        losing_months = sum(1 for m, d in stats.monthly_stats.items() if d['pnl'] < 0)
        total_months = len(stats.monthly_stats)

        monthly_lines = []
        for month, data in sorted(stats.monthly_stats.items()):
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            status = "❌" if data['pnl'] < 0 else "✅"
            monthly_lines.append(f"  {month}: {data['trades']}T, {wr:.0f}%, ${data['pnl']:+.0f} {status}")
        monthly_str = "\n".join(monthly_lines)

        thursday_wr = sum(1 for t in trades if t.is_thursday and t.result == 'WIN') / stats.thursday_trades * 100 if stats.thursday_trades > 0 else 0

        msg = f"""🦅 <b>H1 BACKTEST v4.5</b>
━━━━━━━━━━━━━━━━━━━━━━━
<b>Adaptive Market Intelligence</b>
Period: Jan 2025 - Jan 2026

<b>📊 PERFORMANCE</b>
├ Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
├ Win Rate: {stats.win_rate:.1f}%
├ Net P/L: <b>${stats.total_pnl:+,.2f}</b>
├ Return: <b>{(final_balance/10000-1)*100:+.1f}%</b>
├ Profit Factor: <b>{stats.profit_factor:.2f}</b>
└ Max DD: ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)

<b>🔬 V4.5 ADAPTIVE FILTERS</b>
├ Signals: {stats.signals_total}
├ Drift filtered: {stats.signals_filtered_drift}
├ Confluence filtered: {stats.signals_filtered_confluence}
├ Drift events: {stats.drift_events}
├ Avg confluence: {stats.avg_confluence_score:.0f}/100
└ Thursday: {stats.thursday_trades}T, {thursday_wr:.0f}%, ${stats.thursday_pnl:+.0f}

<b>🎯 ENTRY TYPES</b>
{entry_str}

<b>📅 MONTHLY</b>
{monthly_str}
━━━━━━━━━━━━━━━━━━━━━━━
Losing months: {losing_months}/{total_months}

<b>📈 COMPARISON</b>
v3: 100T, 51%, +$2,669, PF 1.50
v4: 89T, 49%, +$2,131, PF 1.49
<b>v4.5: {stats.total_trades}T, {stats.win_rate:.0f}%, ${stats.total_pnl:+,.0f}, PF {stats.profit_factor:.2f}</b>

<b>Final: ${final_balance:,.2f}</b>
"""
        await telegram.send(msg, force=True)
        logger.info("Report sent to Telegram!")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI DETAILED H1 BACKTEST v4.5")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("Strategy: Adaptive Market Intelligence")
    print("=" * 70)

    print("\n[1/3] Fetching H1 data...")
    symbol = "GBPUSD"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    df = await fetch_data(symbol, "H1", start, end)
    if df.empty:
        print("ERROR: No data available")
        return

    print(f"      Loaded {len(df)} H1 bars")
    print(f"      Period: {df.index[0]} to {df.index[-1]}")

    print("\n[2/3] Running v4.5 backtest with adaptive intelligence...")
    trades, stats, final_balance = run_backtest(df)

    print_report(stats, final_balance, trades)

    print("[3/3] Sending report to Telegram...")
    await send_telegram_report(stats, trades, final_balance)

    print("=" * 70)
    print("BACKTEST v4.5 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
