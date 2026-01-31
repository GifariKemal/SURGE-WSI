"""
SURGE-WSI H1 v6.4 GBPUSD - DUAL-LAYER QUALITY FILTER
=====================================================

Enhancement dari v6.3:
- Layer 1: MONTHLY PROFILE (dari market analysis data)
  - Bulan dengan tradeable_pct < 60% → +15 quality requirement
  - Bulan dengan tradeable_pct < 70% → +10 quality requirement
  - Bulan dengan tradeable_pct >= 70% → no adjustment

- Layer 2: REAL-TIME TECHNICAL (sama seperti v6.3)
  - ATR Stability, Efficiency, Trend Strength

Result: Dual protection against poor market conditions

Market Analysis Data (GBPUSD Monthly):
- January 2024: 67% tradeable
- February 2024: 55% tradeable (POOR) - ini yg bikin loss
- March 2024: 70% tradeable
- April 2024: 80% tradeable (EXCELLENT)
- dst...

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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from config import config
from src.data.db_handler import DBHandler
from src.utils.telegram import TelegramNotifier, TelegramFormatter

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# TELEGRAM NOTIFICATION CONFIG
# ============================================================
SEND_TO_TELEGRAM = True  # Set False to disable Telegram notifications


# ============================================================
# CONFIGURATION v6.4 - DUAL-LAYER QUALITY FILTER
# ============================================================
SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = 1.0

SL_ATR_MULT = 1.5
TP_RATIO = 1.5
MAX_LOSS_PER_TRADE_PCT = 0.15

PIP_VALUE = 10.0
PIP_SIZE = 0.0001
MAX_LOT = 5.0
MIN_ATR = 8.0
MAX_ATR = 30.0

# Base Quality Thresholds
BASE_QUALITY = 65
MIN_QUALITY_GOOD = 60
MAX_QUALITY_BAD = 80

# ==========================================================
# LAYER 1: MONTHLY PROFILE (dari market analysis)
# Tradeable percentage berdasarkan historical analysis
# ==========================================================
MONTHLY_TRADEABLE_PCT = {
    # 2024
    (2024, 1): 67,   # January - OK
    (2024, 2): 55,   # February - POOR!
    (2024, 3): 70,   # March - Good
    (2024, 4): 80,   # April - Excellent
    (2024, 5): 62,   # May - Below avg
    (2024, 6): 68,   # June - OK
    (2024, 7): 78,   # July - Good
    (2024, 8): 65,   # August - Average
    (2024, 9): 72,   # September - Good
    (2024, 10): 58,  # October - Below avg
    (2024, 11): 66,  # November - OK
    (2024, 12): 60,  # December - Low (holidays)
    # 2025
    (2025, 1): 65, (2025, 2): 55, (2025, 3): 70, (2025, 4): 80,
    (2025, 5): 62, (2025, 6): 68, (2025, 7): 78, (2025, 8): 65,
    (2025, 9): 72, (2025, 10): 58, (2025, 11): 66, (2025, 12): 60,
    # 2026
    (2026, 1): 65, (2026, 2): 55, (2026, 3): 70, (2026, 4): 80,
    (2026, 5): 62, (2026, 6): 68, (2026, 7): 78, (2026, 8): 65,
    (2026, 9): 72, (2026, 10): 58, (2026, 11): 66, (2026, 12): 60,
}

def get_monthly_quality_adjustment(dt: datetime) -> int:
    """
    Get quality adjustment based on monthly tradeable percentage

    Returns:
    - +15 if tradeable < 60% (very poor month)
    - +10 if tradeable < 70% (below average month)
    - +5  if tradeable < 75% (slightly below average)
    - 0   if tradeable >= 75% (good month)
    """
    key = (dt.year, dt.month)
    tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)  # Default 70% if unknown

    if tradeable_pct < 60:
        return 15  # Very poor month - high quality required
    elif tradeable_pct < 70:
        return 10  # Below average - moderate increase
    elif tradeable_pct < 75:
        return 5   # Slightly below average
    else:
        return 0   # Good month - no adjustment

# ==========================================================
# LAYER 2: REAL-TIME TECHNICAL THRESHOLDS
# ==========================================================
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = 0.08
TREND_STRENGTH_THRESHOLD = 25

# Risk Multipliers
MONTHLY_RISK = {
    1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,  # Feb lowered to 0.6
    7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
}

DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}

HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.8, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.8,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}


@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    risk_amount: float
    atr_pips: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    exit_reason: str = ""
    quality_score: float = 0.0
    entry_type: str = ""
    session: str = ""
    dynamic_quality: float = 0.0
    market_condition: str = ""
    monthly_adj: int = 0


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class MarketCondition:
    """Market condition assessment"""
    atr_stability: float
    efficiency: float
    trend_strength: float
    technical_quality: float   # From Layer 2 (technical)
    monthly_adjustment: int    # From Layer 1 (profile)
    final_quality: float       # Combined threshold
    label: str


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
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


def calculate_atr(df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
    h, l, c = col_map['high'], col_map['low'], col_map['close']
    high = df[h]
    low = df[l]
    close = df[c].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr / PIP_SIZE


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def assess_market_condition(df: pd.DataFrame, col_map: dict, idx: int,
                           atr_series: pd.Series, current_time: datetime) -> MarketCondition:
    """
    DUAL-LAYER market condition assessment

    Layer 1: Monthly profile adjustment (from market analysis data)
    Layer 2: Real-time technical indicators
    """
    lookback = 20
    start_idx = max(0, idx - lookback)

    h, l, c = col_map['high'], col_map['low'], col_map['close']

    # ==========================================
    # LAYER 2: TECHNICAL INDICATORS
    # ==========================================

    # 1. ATR Stability
    recent_atr = atr_series.iloc[start_idx:idx+1]
    if len(recent_atr) > 5 and recent_atr.mean() > 0:
        atr_cv = recent_atr.std() / recent_atr.mean()
    else:
        atr_cv = 0.5

    # 2. Price Efficiency
    if idx >= lookback:
        net_move = abs(df[c].iloc[idx] - df[c].iloc[start_idx])
        total_move = sum(abs(df[c].iloc[i] - df[c].iloc[i-1]) for i in range(start_idx+1, idx+1))
        efficiency = net_move / total_move if total_move > 0 else 0
    else:
        efficiency = 0.1

    # 3. Trend Strength (ADX)
    if idx >= lookback:
        highs = df[h].iloc[start_idx:idx+1]
        lows = df[l].iloc[start_idx:idx+1]
        closes = df[c].iloc[start_idx:idx+1]

        plus_dm = (highs - highs.shift(1)).clip(lower=0)
        minus_dm = (lows.shift(1) - lows).clip(lower=0)

        tr = pd.concat([
            highs - lows,
            abs(highs - closes.shift(1)),
            abs(lows - closes.shift(1))
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(14).mean()

        trend_strength = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
    else:
        trend_strength = 25

    # Calculate technical score (0-100)
    score = 0

    if atr_cv < ATR_STABILITY_THRESHOLD:
        score += 33
    elif atr_cv < ATR_STABILITY_THRESHOLD * 1.5:
        score += 20

    if efficiency > EFFICIENCY_THRESHOLD:
        score += 33
    elif efficiency > EFFICIENCY_THRESHOLD * 0.5:
        score += 20

    if trend_strength > TREND_STRENGTH_THRESHOLD:
        score += 34
    elif trend_strength > TREND_STRENGTH_THRESHOLD * 0.7:
        score += 20

    # Technical quality threshold
    if score >= 80:
        technical_quality = MIN_QUALITY_GOOD  # 60
        tech_label = "TECH_GOOD"
    elif score >= 40:
        technical_quality = BASE_QUALITY  # 65
        tech_label = "TECH_NORMAL"
    else:
        technical_quality = MAX_QUALITY_BAD  # 80
        tech_label = "TECH_BAD"

    # ==========================================
    # LAYER 1: MONTHLY PROFILE ADJUSTMENT
    # ==========================================
    monthly_adj = get_monthly_quality_adjustment(current_time)

    # ==========================================
    # COMBINE LAYERS
    # ==========================================
    final_quality = technical_quality + monthly_adj

    # Determine overall label
    if monthly_adj >= 15:
        label = "POOR_MONTH"  # February 2024, etc.
    elif monthly_adj >= 10:
        label = "CAUTION"
    elif score >= 80:
        label = "GOOD"
    elif score >= 40:
        label = "NORMAL"
    else:
        label = "BAD"

    return MarketCondition(
        atr_stability=atr_cv,
        efficiency=efficiency,
        trend_strength=trend_strength,
        technical_quality=technical_quality,
        monthly_adjustment=monthly_adj,
        final_quality=final_quality,
        label=label
    )


def detect_regime(df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float]:
    if len(df) < 50:
        return Regime.SIDEWAYS, 0.5
    c = col_map['close']
    ema20 = calculate_ema(df[c], 20)
    ema50 = calculate_ema(df[c], 50)
    current_close = df[c].iloc[-1]
    current_ema20 = ema20.iloc[-1]
    current_ema50 = ema50.iloc[-1]
    if current_close > current_ema20 > current_ema50:
        return Regime.BULLISH, 0.7
    elif current_close < current_ema20 < current_ema50:
        return Regime.BEARISH, 0.7
    else:
        return Regime.SIDEWAYS, 0.5


def detect_order_blocks(df: pd.DataFrame, col_map: dict, min_quality: float) -> List[dict]:
    pois = []
    if len(df) < 35:
        return pois
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    for i in range(len(df) - 30, len(df) - 2):
        if i < 2:
            continue
        current = df.iloc[i]
        next1 = df.iloc[i+1]

        is_bearish = current[c] < current[o]
        if is_bearish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] > next1[o] and body_ratio > 0.55 and next1[c] > current[h]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[l], 'direction': 'BUY', 'quality': quality, 'idx': i})

        is_bullish = current[c] > current[o]
        if is_bullish:
            next_body = abs(next1[c] - next1[o])
            next_range = next1[h] - next1[l]
            if next_range > 0:
                body_ratio = next_body / next_range
                if next1[c] < next1[o] and body_ratio > 0.55 and next1[c] < current[l]:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        pois.append({'price': current[h], 'direction': 'SELL', 'quality': quality, 'idx': i})
    return pois


def check_entry_trigger(bar: pd.Series, prev_bar: pd.Series, direction: str, col_map: dict) -> Tuple[bool, str]:
    o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
    total_range = bar[h] - bar[l]
    if total_range < 0.0003:
        return False, ""
    body = abs(bar[c] - bar[o])
    is_bullish = bar[c] > bar[o]
    is_bearish = bar[c] < bar[o]
    prev_body = abs(prev_bar[c] - prev_bar[o])

    if body > total_range * 0.5:
        if direction == 'BUY' and is_bullish:
            return True, 'MOMENTUM'
        if direction == 'SELL' and is_bearish:
            return True, 'MOMENTUM'

    if body > prev_body * 1.2:
        if direction == 'BUY' and is_bullish and prev_bar[c] < prev_bar[o]:
            return True, 'ENGULF'
        if direction == 'SELL' and is_bearish and prev_bar[c] > prev_bar[o]:
            return True, 'ENGULF'

    if direction == 'SELL':
        if bar[h] < prev_bar[h] and is_bearish:
            return True, 'LOWER_HIGH'

    return False, ""


def calculate_risk_multiplier(dt: datetime, entry_type: str, quality: float) -> Tuple[float, bool]:
    day = dt.weekday()
    hour = dt.hour
    month = dt.month

    day_mult = DAY_MULTIPLIERS.get(day, 0.5)
    hour_mult = HOUR_MULTIPLIERS.get(hour, 0.0)
    entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 0.0)
    quality_mult = quality / 100.0
    month_mult = MONTHLY_RISK.get(month, 0.8)

    if day_mult == 0.0 or hour_mult == 0.0 or entry_mult == 0.0:
        return 0.0, True

    combined = day_mult * hour_mult * entry_mult * quality_mult * month_mult
    if combined < 0.30:
        return combined, True

    return max(0.30, min(1.2, combined)), False


def run_backtest(df: pd.DataFrame, col_map: dict) -> Tuple[List[Trade], float, dict]:
    """Run backtest with DUAL-LAYER quality filter"""
    trades = []
    balance = INITIAL_BALANCE
    peak_balance = balance
    max_dd = 0
    position: Optional[Trade] = None
    atr_series = calculate_atr(df, col_map)

    condition_stats = {'GOOD': 0, 'NORMAL': 0, 'BAD': 0, 'POOR_MONTH': 0, 'CAUTION': 0}

    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1]
        current_bar = df.iloc[i]
        current_time = df.index[i]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_price = current_bar[col_map['close']]
        current_atr = atr_series.iloc[i]

        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        if position:
            high = current_bar[col_map['high']]
            low = current_bar[col_map['low']]
            exit_price = None
            exit_reason = ""

            if position.direction == 'BUY':
                if low <= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = "SL"
                elif high >= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "TP"
            else:
                if high >= position.sl_price:
                    exit_price = position.sl_price
                    exit_reason = "SL"
                elif low <= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "TP"

            if exit_price:
                if position.direction == 'BUY':
                    pips = (exit_price - position.entry_price) / PIP_SIZE
                else:
                    pips = (position.entry_price - exit_price) / PIP_SIZE
                pnl = pips * position.lot_size * PIP_VALUE
                max_loss = balance * (MAX_LOSS_PER_TRADE_PCT / 100)
                if pnl < 0 and abs(pnl) > max_loss:
                    pnl = -max_loss
                    exit_reason = "SL_CAPPED"
                position.exit_time = current_time
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pips = pips
                position.exit_reason = exit_reason
                balance += pnl
                trades.append(position)
                position = None
            continue

        if current_time.weekday() >= 5:
            continue
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        hour = current_time.hour
        if not (7 <= hour <= 11 or 13 <= hour <= 17):
            continue
        session = "london" if 7 <= hour <= 11 else "newyork"

        regime, _ = detect_regime(current_slice, col_map)
        if regime == Regime.SIDEWAYS:
            continue

        # DUAL-LAYER QUALITY: Assess market condition
        market_cond = assess_market_condition(df, col_map, i, atr_series, current_time)
        dynamic_quality = market_cond.final_quality
        condition_stats[market_cond.label] = condition_stats.get(market_cond.label, 0) + 1

        # Detect POIs with COMBINED quality threshold
        pois = detect_order_blocks(current_slice, col_map, dynamic_quality)
        if not pois:
            continue

        for poi in pois:
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            zone_size = abs(current_bar[col_map['high']] - current_bar[col_map['low']]) * 2
            if abs(current_price - poi['price']) > zone_size:
                continue

            prev_bar = df.iloc[i-1]
            has_trigger, entry_type = check_entry_trigger(current_bar, prev_bar, poi['direction'], col_map)
            if not has_trigger:
                continue

            risk_mult, should_skip = calculate_risk_multiplier(current_time, entry_type, poi['quality'])
            if should_skip:
                continue

            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO
            risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult
            lot_size = risk_amount / (sl_pips * PIP_VALUE)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            position = Trade(
                entry_time=current_time,
                direction=poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                risk_amount=risk_amount,
                atr_pips=current_atr,
                quality_score=poi['quality'],
                entry_type=entry_type,
                session=session,
                dynamic_quality=dynamic_quality,
                market_condition=market_cond.label,
                monthly_adj=market_cond.monthly_adjustment
            )
            break

    return trades, max_dd, condition_stats


def calculate_stats(trades: List[Trade], max_dd: float) -> dict:
    if not trades:
        return {"error": "No trades"}

    total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total * 100) if total > 0 else 0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    net_pnl = gross_profit - gross_loss

    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0

    trade_df = pd.DataFrame([{'time': t.entry_time, 'pnl': t.pnl} for t in trades])
    trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
    monthly = trade_df.groupby('month')['pnl'].sum()
    losing_months = (monthly < 0).sum()

    first_trade = trades[0].entry_time
    last_trade = trades[-1].entry_time
    days = (last_trade - first_trade).days or 1
    trades_per_day = total / days

    return {
        'total_trades': total,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'profit_factor': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_dd_pct': (max_dd / INITIAL_BALANCE) * 100,
        'losing_months': losing_months,
        'total_months': len(monthly),
        'trades_per_day': trades_per_day,
        'final_balance': INITIAL_BALANCE + net_pnl,
        'monthly': monthly
    }


async def send_telegram_report(stats: dict, trades: List[Trade], condition_stats: dict,
                               start_date: datetime, end_date: datetime):
    """Send backtest results to Telegram"""
    if not SEND_TO_TELEGRAM:
        return

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        if not await telegram.initialize():
            print("Failed to initialize Telegram")
            return

        # ============================================================
        # MESSAGE 1: Main Performance Report (Tree Style)
        # ============================================================
        msg = TelegramFormatter.tree_header("BACKTEST RESULTS", "📊")
        msg += f"<b>H1 v6.4 GBPUSD - Dual-Layer Quality</b>\n"
        msg += f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"

        # Performance metrics
        msg += TelegramFormatter.tree_section("Performance", TelegramFormatter.CHART)
        msg += TelegramFormatter.tree_item("Total Trades", str(stats['total_trades']))
        msg += TelegramFormatter.tree_item("Trades/Day", f"{stats['trades_per_day']:.2f}")
        msg += TelegramFormatter.tree_item("Win Rate", f"{stats['win_rate']:.1f}%")
        msg += TelegramFormatter.tree_item("Profit Factor", f"{stats['profit_factor']:.2f}", last=True)

        # P/L section
        msg += TelegramFormatter.tree_section("Profit/Loss", TelegramFormatter.MONEY)
        msg += TelegramFormatter.tree_item("Initial", f"${INITIAL_BALANCE:,.0f}")
        msg += TelegramFormatter.tree_item("Final", f"${stats['final_balance']:,.0f}")
        pnl_emoji = TelegramFormatter.CHECK if stats['net_pnl'] >= 0 else TelegramFormatter.CROSS
        msg += TelegramFormatter.tree_item("Net P/L", f"{pnl_emoji} ${stats['net_pnl']:+,.0f}")
        msg += TelegramFormatter.tree_item("Return", f"{(stats['net_pnl']/INITIAL_BALANCE)*100:+.1f}%", last=True)

        # Risk metrics
        msg += TelegramFormatter.tree_section("Risk Metrics", TelegramFormatter.WARNING)
        msg += TelegramFormatter.tree_item("Max DD", f"{stats['max_dd_pct']:.1f}%")
        msg += TelegramFormatter.tree_item("Avg Win", f"${stats['avg_win']:,.0f}")
        msg += TelegramFormatter.tree_item("Avg Loss", f"${stats['avg_loss']:,.0f}")

        # Losing months
        if stats['losing_months'] == 0:
            msg += TelegramFormatter.tree_item("Losing Months",
                f"{TelegramFormatter.CHECK} 0/{stats['total_months']} (ZERO!)", last=True)
        else:
            msg += TelegramFormatter.tree_item("Losing Months",
                f"{TelegramFormatter.CROSS} {stats['losing_months']}/{stats['total_months']}", last=True)

        # Target check
        msg += "\n<b>Targets:</b>\n"
        msg += f"{TelegramFormatter.CHECK if stats['net_pnl'] >= 5000 else TelegramFormatter.CROSS} Profit >= $5K\n"
        msg += f"{TelegramFormatter.CHECK if stats['profit_factor'] >= 2.0 else TelegramFormatter.CROSS} PF >= 2.0\n"
        msg += f"{TelegramFormatter.CHECK if stats['losing_months'] == 0 else TelegramFormatter.CROSS} ZERO losing months"

        await telegram.send(msg)

        # ============================================================
        # MESSAGE 2: Monthly + Market Condition (Same pre block)
        # ============================================================
        msg2 = "<pre>"
        msg2 += f"📅 MONTHLY BREAKDOWN\n"
        msg2 += f"{'Month':<9} {'P/L':>9} {'T%':>4} {'Adj':>4}\n"
        msg2 += f"{'-'*9} {'-'*9} {'-'*4} {'-'*4}\n"

        for month, pnl in stats['monthly'].items():
            year = month.year
            mon = month.month
            tradeable = MONTHLY_TRADEABLE_PCT.get((year, mon), 70)
            adj = get_monthly_quality_adjustment(datetime(year, mon, 1))
            status = "✓" if pnl >= 0 else "✗"
            month_str = f"{year}-{mon:02d}"
            msg2 += f"{month_str:<9} ${pnl:>+7,.0f} {tradeable:>3}% +{adj:<2} {status}\n"

        msg2 += f"\n🎯 BY MARKET CONDITION\n"
        msg2 += f"{'Condition':<11} {'N':>4} {'WR':>4} {'P/L':>10}\n"
        msg2 += f"{'-'*11} {'-'*4} {'-'*4} {'-'*10}\n"

        for cond in ['GOOD', 'NORMAL', 'CAUTION', 'POOR_MONTH', 'BAD']:
            cond_trades = [t for t in trades if t.market_condition == cond]
            if cond_trades:
                wins = len([t for t in cond_trades if t.pnl > 0])
                total = len(cond_trades)
                wr = wins / total * 100
                net = sum(t.pnl for t in cond_trades)
                status = "✓" if net >= 0 else "✗"
                msg2 += f"{cond:<11} {total:>4} {wr:>3.0f}% ${net:>+8,.0f} {status}\n"

        msg2 += "</pre>"

        await telegram.send(msg2)

        print("\n[TELEGRAM] Results sent successfully!")

    except Exception as e:
        print(f"\n[TELEGRAM] Failed to send: {e}")


def print_results(stats: dict, trades: List[Trade], condition_stats: dict):
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS - H1 v6.4 GBPUSD DUAL-LAYER QUALITY")
    print(f"{'='*70}")

    print(f"\n[DUAL-LAYER QUALITY CONFIGURATION]")
    print(f"{'-'*50}")
    print(f"  Layer 1 - Monthly Profile (from market analysis):")
    print(f"    tradeable < 60%: +15 quality requirement")
    print(f"    tradeable < 70%: +10 quality requirement")
    print(f"    tradeable < 75%: +5 quality requirement")
    print(f"  Layer 2 - Technical (ATR stability, efficiency, trend):")
    print(f"    GOOD market: base quality = {MIN_QUALITY_GOOD}")
    print(f"    NORMAL market: base quality = {BASE_QUALITY}")
    print(f"    BAD market: base quality = {MAX_QUALITY_BAD}")
    print(f"  Combined = Layer1 + Layer2")

    print(f"\n[MARKET CONDITIONS OBSERVED]")
    print(f"{'-'*50}")
    for cond, count in sorted(condition_stats.items(), key=lambda x: -x[1]):
        print(f"  {cond}: {count} bars")

    print(f"\n[PERFORMANCE]")
    print(f"{'-'*50}")
    print(f"Total Trades:      {stats['total_trades']}")
    print(f"Trades/Day:        {stats['trades_per_day']:.2f}")
    print(f"Win Rate:          {stats['win_rate']:.1f}%")
    print(f"Profit Factor:     {stats['profit_factor']:.2f}")

    print(f"\n[PROFIT/LOSS]")
    print(f"{'-'*50}")
    print(f"Initial Balance:   ${INITIAL_BALANCE:,.2f}")
    print(f"Final Balance:     ${stats['final_balance']:,.2f}")
    print(f"Net P/L:           ${stats['net_pnl']:+,.2f}")
    print(f"Total Return:      {(stats['net_pnl']/INITIAL_BALANCE)*100:+.1f}%")
    print(f"Avg Win:           ${stats['avg_win']:,.2f}")
    print(f"Avg Loss:          ${stats['avg_loss']:,.2f}")

    print(f"\n[MONTHLY BREAKDOWN]")
    print(f"{'-'*50}")
    print(f"Losing Months:     {stats['losing_months']}/{stats['total_months']}")

    for month, pnl in stats['monthly'].items():
        status = "WIN " if pnl >= 0 else "LOSS"
        # Show monthly quality adjustment
        year = month.year
        mon = month.month
        tradeable = MONTHLY_TRADEABLE_PCT.get((year, mon), 70)
        adj = get_monthly_quality_adjustment(datetime(year, mon, 1))
        print(f"  [{status}] {month}: ${pnl:+,.2f} (tradeable={tradeable}%, adj=+{adj})")

    # By market condition
    print(f"\n[TRADES BY MARKET CONDITION]")
    print(f"{'-'*50}")
    for cond in ['GOOD', 'NORMAL', 'BAD', 'CAUTION', 'POOR_MONTH']:
        cond_trades = [t for t in trades if t.market_condition == cond]
        if cond_trades:
            wins = len([t for t in cond_trades if t.pnl > 0])
            total = len(cond_trades)
            wr = wins / total * 100
            net = sum(t.pnl for t in cond_trades)
            avg_q = sum(t.dynamic_quality for t in cond_trades) / total
            print(f"  {cond:12} {total:>3} trades, {wr:>5.1f}% WR, Q={avg_q:.0f}, ${net:>+10,.0f}")

    # Show February 2024 specifically
    print(f"\n[FEBRUARY 2024 DETAIL]")
    print(f"{'-'*50}")
    feb_trades = [t for t in trades if t.entry_time.year == 2024 and t.entry_time.month == 2]
    if feb_trades:
        feb_pnl = sum(t.pnl for t in feb_trades)
        feb_wins = len([t for t in feb_trades if t.pnl > 0])
        print(f"  Trades: {len(feb_trades)}")
        print(f"  Wins: {feb_wins}")
        print(f"  P/L: ${feb_pnl:+,.2f}")
        for t in feb_trades:
            print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction} Q={t.quality_score:.0f} Qreq={t.dynamic_quality:.0f} ${t.pnl:+,.0f}")
    else:
        print(f"  No trades (filtered out by high quality requirement)")

    print(f"\n{'='*70}")
    target_met = 0

    if stats['net_pnl'] >= 5000:
        print(f"[OK] PROFIT TARGET: ${stats['net_pnl']:+,.2f} >= $5,000")
        target_met += 1
    else:
        print(f"[X] PROFIT TARGET: ${stats['net_pnl']:+,.2f} < $5,000")

    if stats['profit_factor'] >= 2.0:
        print(f"[OK] PF TARGET: {stats['profit_factor']:.2f} >= 2.0")
        target_met += 1
    else:
        print(f"[X] PF TARGET: {stats['profit_factor']:.2f} < 2.0")

    if stats['losing_months'] == 0:
        print(f"[OK] ZERO LOSING MONTHS!")
        target_met += 1
    else:
        print(f"[X] LOSING MONTHS: {stats['losing_months']}")

    print(f"\nTargets Met: {target_met}/3")
    print(f"{'='*70}")


async def main():
    timeframe = "H1"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 15, tzinfo=timezone.utc)

    print(f"SURGE-WSI H1 v6.4 GBPUSD - DUAL-LAYER QUALITY FILTER")
    print(f"{'='*70}")
    print(f"Dual-Layer Quality Filter:")
    print(f"  Layer 1: Monthly profile (from market analysis)")
    print(f"  Layer 2: Real-time technical indicators")
    print(f"  Combined = Higher of both layers")
    print(f"{'='*70}")

    print(f"\nFetching {SYMBOL} {timeframe} data...")

    df = await fetch_data(SYMBOL, timeframe, start, end)

    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")

    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_map['open'] = col
        elif 'high' in col_lower:
            col_map['high'] = col
        elif 'low' in col_lower:
            col_map['low'] = col
        elif 'close' in col_lower:
            col_map['close'] = col

    print(f"\nRunning backtest with DUAL-LAYER quality filter...")
    trades, max_dd, condition_stats = run_backtest(df, col_map)

    if not trades:
        print("No trades executed")
        return

    stats = calculate_stats(trades, max_dd)
    print_results(stats, trades, condition_stats)

    # Send to Telegram
    await send_telegram_report(stats, trades, condition_stats, start, end)

    # Save trades
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'lot_size': t.lot_size,
        'atr_pips': t.atr_pips,
        'pnl': t.pnl,
        'exit_reason': t.exit_reason,
        'quality_score': t.quality_score,
        'entry_type': t.entry_type,
        'session': t.session,
        'dynamic_quality': t.dynamic_quality,
        'market_condition': t.market_condition,
        'monthly_adj': t.monthly_adj
    } for t in trades])

    output_path = Path(__file__).parent.parent / "results" / "h1_v6_4_dual_filter_trades.csv"
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
