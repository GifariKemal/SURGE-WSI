"""Detailed H1 Backtest v3 - Research-Based Improvements
=======================================================

Improvements from research:
1. Ensemble Regime Detection (HMM + ADX + CCI + Structure)
2. Enhanced OB Quality (volume + wicks + retests)
3. Kelly Criterion Position Sizing
4. Advanced Rejection Candle (volume + context)
5. Volatility-Adaptive Trailing Stops (ATR-based)

Expected improvements over v2:
- Win Rate: +5-10%
- Profit Factor: +15-25%

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger
from enum import Enum

from config import config
from src.data.db_handler import DBHandler
from src.utils.killzone import KillZone
from src.utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector


# ============================================================================
# IMPROVEMENT 1: Ensemble Regime Detection
# ============================================================================

class EnsembleRegime(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class EnsembleRegimeInfo:
    """Enhanced regime info with confidence from multiple sources"""
    regime: EnsembleRegime
    confidence: float  # 0-100
    hmm_vote: str
    adx_vote: str
    cci_vote: str
    structure_vote: str
    is_tradeable: bool
    bias: str


class EnsembleRegimeDetector:
    """
    Combines multiple indicators for regime detection:
    - HMM (existing)
    - ADX for trend strength
    - CCI for momentum direction
    - Price structure (higher highs/lows)
    """

    def __init__(self, adx_period: int = 14, cci_period: int = 20):
        self.hmm_detector = HMMRegimeDetector()
        self.adx_period = adx_period
        self.cci_period = cci_period

        # Price buffers
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []

        # ADX components
        self.tr_buffer: List[float] = []
        self.plus_dm_buffer: List[float] = []
        self.minus_dm_buffer: List[float] = []

        # CCI buffer
        self.typical_prices: List[float] = []

        self.prev_high = None
        self.prev_low = None
        self.prev_close = None

    def update(self, high: float, low: float, close: float) -> Optional[EnsembleRegimeInfo]:
        """Update with OHLC data and return ensemble regime"""

        # Update buffers
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        # Keep buffer size manageable
        max_buffer = 100
        if len(self.highs) > max_buffer:
            self.highs = self.highs[-max_buffer:]
            self.lows = self.lows[-max_buffer:]
            self.closes = self.closes[-max_buffer:]

        # Update HMM
        hmm_info = self.hmm_detector.update(close)

        # Need enough data
        if len(self.closes) < self.adx_period + 5:
            if self.prev_high is not None:
                self._update_adx_buffers(high, low, close)
            self.prev_high = high
            self.prev_low = low
            self.prev_close = close
            return None

        # Calculate indicators
        self._update_adx_buffers(high, low, close)
        adx_value, plus_di, minus_di = self._calculate_adx()
        cci_value = self._calculate_cci(high, low, close)
        structure = self._analyze_structure()

        self.prev_high = high
        self.prev_low = low
        self.prev_close = close

        # Get votes
        hmm_vote = hmm_info.bias if hmm_info else "NONE"
        adx_vote = self._adx_vote(adx_value, plus_di, minus_di)
        cci_vote = self._cci_vote(cci_value)
        structure_vote = structure

        # Ensemble voting with weights
        votes = {
            'BULLISH': 0,
            'BEARISH': 0,
            'SIDEWAYS': 0
        }

        weights = {
            'hmm': 0.35,      # HMM is primary
            'adx': 0.25,      # ADX for trend strength
            'cci': 0.20,      # CCI for momentum
            'structure': 0.20 # Price structure
        }

        # HMM vote
        if hmm_vote == 'BUY':
            votes['BULLISH'] += weights['hmm']
        elif hmm_vote == 'SELL':
            votes['BEARISH'] += weights['hmm']
        else:
            votes['SIDEWAYS'] += weights['hmm']

        # ADX vote
        if adx_vote == 'BULLISH':
            votes['BULLISH'] += weights['adx']
        elif adx_vote == 'BEARISH':
            votes['BEARISH'] += weights['adx']
        else:
            votes['SIDEWAYS'] += weights['adx']

        # CCI vote
        if cci_vote == 'BULLISH':
            votes['BULLISH'] += weights['cci']
        elif cci_vote == 'BEARISH':
            votes['BEARISH'] += weights['cci']
        else:
            votes['SIDEWAYS'] += weights['cci']

        # Structure vote
        if structure_vote == 'BULLISH':
            votes['BULLISH'] += weights['structure']
        elif structure_vote == 'BEARISH':
            votes['BEARISH'] += weights['structure']
        else:
            votes['SIDEWAYS'] += weights['structure']

        # Determine final regime
        max_vote = max(votes.values())
        if votes['BULLISH'] == max_vote and votes['BULLISH'] > 0.5:
            regime = EnsembleRegime.BULLISH
            bias = 'BUY'
        elif votes['BEARISH'] == max_vote and votes['BEARISH'] > 0.5:
            regime = EnsembleRegime.BEARISH
            bias = 'SELL'
        else:
            regime = EnsembleRegime.SIDEWAYS
            bias = 'NONE'

        confidence = max_vote * 100
        is_tradeable = confidence >= 55 and regime != EnsembleRegime.SIDEWAYS

        return EnsembleRegimeInfo(
            regime=regime,
            confidence=confidence,
            hmm_vote=hmm_vote,
            adx_vote=adx_vote,
            cci_vote=cci_vote,
            structure_vote=structure_vote,
            is_tradeable=is_tradeable,
            bias=bias
        )

    def _update_adx_buffers(self, high: float, low: float, close: float):
        """Update ADX component buffers"""
        if self.prev_high is None:
            return

        # True Range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close)
        )
        self.tr_buffer.append(tr)

        # Directional Movement
        up_move = high - self.prev_high
        down_move = self.prev_low - low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        self.plus_dm_buffer.append(plus_dm)
        self.minus_dm_buffer.append(minus_dm)

        # Keep buffers trimmed
        max_buf = self.adx_period * 3
        if len(self.tr_buffer) > max_buf:
            self.tr_buffer = self.tr_buffer[-max_buf:]
            self.plus_dm_buffer = self.plus_dm_buffer[-max_buf:]
            self.minus_dm_buffer = self.minus_dm_buffer[-max_buf:]

    def _calculate_adx(self) -> Tuple[float, float, float]:
        """Calculate ADX, +DI, -DI"""
        if len(self.tr_buffer) < self.adx_period:
            return 0, 0, 0

        period = self.adx_period

        # Smoothed averages (Wilder's smoothing)
        atr = sum(self.tr_buffer[-period:]) / period
        smoothed_plus_dm = sum(self.plus_dm_buffer[-period:]) / period
        smoothed_minus_dm = sum(self.minus_dm_buffer[-period:]) / period

        if atr == 0:
            return 0, 0, 0

        plus_di = (smoothed_plus_dm / atr) * 100
        minus_di = (smoothed_minus_dm / atr) * 100

        # DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0, plus_di, minus_di

        dx = abs(plus_di - minus_di) / di_sum * 100

        # ADX (simplified - just use DX for now)
        adx = dx

        return adx, plus_di, minus_di

    def _adx_vote(self, adx: float, plus_di: float, minus_di: float) -> str:
        """Get ADX vote"""
        if adx < 20:  # Weak trend
            return "SIDEWAYS"

        if plus_di > minus_di:
            return "BULLISH"
        elif minus_di > plus_di:
            return "BEARISH"
        return "SIDEWAYS"

    def _calculate_cci(self, high: float, low: float, close: float) -> float:
        """Calculate CCI"""
        typical_price = (high + low + close) / 3
        self.typical_prices.append(typical_price)

        if len(self.typical_prices) > self.cci_period * 2:
            self.typical_prices = self.typical_prices[-(self.cci_period * 2):]

        if len(self.typical_prices) < self.cci_period:
            return 0

        recent_tp = self.typical_prices[-self.cci_period:]
        sma = np.mean(recent_tp)
        mean_deviation = np.mean([abs(tp - sma) for tp in recent_tp])

        if mean_deviation == 0:
            return 0

        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci

    def _cci_vote(self, cci: float) -> str:
        """Get CCI vote"""
        if cci > 100:
            return "BULLISH"
        elif cci < -100:
            return "BEARISH"
        return "SIDEWAYS"

    def _analyze_structure(self) -> str:
        """Analyze price structure (higher highs/lows)"""
        if len(self.highs) < 10:
            return "SIDEWAYS"

        recent_highs = self.highs[-10:]
        recent_lows = self.lows[-10:]

        # Check for higher highs and higher lows (bullish)
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])

        # Check for lower highs and lower lows (bearish)
        lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
        ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])

        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        if bullish_score >= 12:  # Out of 18 possible
            return "BULLISH"
        elif bearish_score >= 12:
            return "BEARISH"
        return "SIDEWAYS"


# ============================================================================
# IMPROVEMENT 2: Enhanced OB Quality Calculator
# ============================================================================

def calculate_enhanced_ob_quality(
    df: pd.DataFrame,
    ob_idx: int,
    direction: str,
    col_map: dict
) -> float:
    """
    Enhanced Order Block quality scoring:
    - Base quality from impulse move
    - Volume confirmation (+20 pts)
    - Wick analysis (+15 pts)
    - Retest count bonus (+10 pts per retest, max 20)
    """

    quality = 0.0

    if ob_idx < 5 or ob_idx >= len(df) - 3:
        return quality

    ob_bar = df.iloc[ob_idx]
    next_bars = df.iloc[ob_idx+1:ob_idx+4]

    open_col = col_map['open']
    high_col = col_map['high']
    low_col = col_map['low']
    close_col = col_map['close']

    # 1. Base quality from impulse move (0-40 pts)
    if direction == 'BUY':
        impulse = next_bars[close_col].max() - ob_bar[low_col]
    else:
        impulse = ob_bar[high_col] - next_bars[close_col].min()

    impulse_pips = impulse * 10000
    quality += min(40, impulse_pips * 2)  # 2 pts per pip, max 40

    # 2. Volume confirmation (0-20 pts)
    if 'volume' in df.columns or 'Volume' in df.columns:
        vol_col = 'volume' if 'volume' in df.columns else 'Volume'
        avg_vol = df.iloc[max(0, ob_idx-20):ob_idx][vol_col].mean()
        ob_vol = ob_bar[vol_col]

        if avg_vol > 0:
            vol_ratio = ob_vol / avg_vol
            if vol_ratio > 1.5:
                quality += 20  # Strong volume
            elif vol_ratio > 1.2:
                quality += 15
            elif vol_ratio > 1.0:
                quality += 10
    else:
        quality += 10  # Default if no volume data

    # 3. Wick analysis (0-15 pts)
    ob_range = ob_bar[high_col] - ob_bar[low_col]
    if ob_range > 0:
        body = abs(ob_bar[close_col] - ob_bar[open_col])
        body_ratio = body / ob_range

        if direction == 'BUY':
            # For bullish OB (bearish candle), longer upper wick is better
            upper_wick = ob_bar[high_col] - max(ob_bar[open_col], ob_bar[close_col])
            wick_ratio = upper_wick / ob_range
        else:
            # For bearish OB (bullish candle), longer lower wick is better
            lower_wick = min(ob_bar[open_col], ob_bar[close_col]) - ob_bar[low_col]
            wick_ratio = lower_wick / ob_range

        if wick_ratio > 0.3:
            quality += 15
        elif wick_ratio > 0.2:
            quality += 10
        elif wick_ratio > 0.1:
            quality += 5

    # 4. Retest bonus (0-20 pts)
    zone_high = ob_bar[high_col]
    zone_low = ob_bar[low_col]
    retest_count = 0

    for i in range(ob_idx + 4, min(ob_idx + 30, len(df))):
        bar = df.iloc[i]
        bar_low = bar[low_col]
        bar_high = bar[high_col]
        bar_close = bar[close_col]

        # Check if price tested the zone but didn't close through it
        if direction == 'BUY':
            if bar_low <= zone_high and bar_close > zone_low:
                retest_count += 1
        else:
            if bar_high >= zone_low and bar_close < zone_high:
                retest_count += 1

        if retest_count >= 2:
            break

    quality += min(20, retest_count * 10)

    return min(100, quality)


# ============================================================================
# IMPROVEMENT 3: Kelly Criterion Position Sizing
# ============================================================================

class KellyPositionSizer:
    """
    Dynamic position sizing using Kelly Criterion:
    Kelly % = W - [(1-W) / R]

    Where:
    - W = Win probability
    - R = Win/Loss ratio

    Uses fractional Kelly (25-50%) for safety
    """

    def __init__(self,
                 base_risk: float = 0.01,
                 min_trades_for_kelly: int = 20,
                 kelly_fraction: float = 0.25,  # Use 25% of Kelly
                 max_risk: float = 0.02,
                 min_risk: float = 0.005):

        self.base_risk = base_risk
        self.min_trades = min_trades_for_kelly
        self.kelly_fraction = kelly_fraction
        self.max_risk = max_risk
        self.min_risk = min_risk

        self.wins = 0
        self.losses = 0
        self.total_win_amount = 0.0
        self.total_loss_amount = 0.0

    def add_trade_result(self, is_win: bool, pnl: float):
        """Record trade result"""
        if is_win:
            self.wins += 1
            self.total_win_amount += abs(pnl)
        else:
            self.losses += 1
            self.total_loss_amount += abs(pnl)

    def get_position_risk(self, quality_score: float = 50) -> float:
        """
        Get position risk percentage based on:
        - Kelly criterion (if enough trades)
        - Quality score adjustment
        """
        total_trades = self.wins + self.losses

        if total_trades < self.min_trades:
            # Not enough data, use base risk with quality adjustment
            quality_mult = 0.8 + (quality_score / 100) * 0.4  # 0.8 to 1.2
            return max(self.min_risk, min(self.max_risk, self.base_risk * quality_mult))

        # Calculate Kelly
        win_rate = self.wins / total_trades

        if self.losses == 0:
            avg_loss = 1  # Prevent division by zero
        else:
            avg_loss = self.total_loss_amount / self.losses

        if self.wins == 0:
            avg_win = 0
        else:
            avg_win = self.total_win_amount / self.wins

        if avg_loss == 0:
            win_loss_ratio = 1
        else:
            win_loss_ratio = avg_win / avg_loss

        # Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0

        # Apply fraction
        fractional_kelly = kelly_pct * self.kelly_fraction

        # Apply quality adjustment
        quality_mult = 0.8 + (quality_score / 100) * 0.4
        adjusted_risk = fractional_kelly * quality_mult

        # Clamp to limits
        return max(self.min_risk, min(self.max_risk, adjusted_risk))


# ============================================================================
# IMPROVEMENT 4: Advanced Rejection Candle Detection
# ============================================================================

def detect_advanced_rejection(
    bar: pd.Series,
    prev_bars: pd.DataFrame,
    direction: str,
    col_map: dict
) -> Tuple[bool, float]:
    """
    Advanced rejection candle detection:
    - Standard rejection criteria
    - Volume confirmation
    - Context analysis (previous bars)

    Returns: (is_rejection, confidence_score)
    """

    o = bar[col_map['open']]
    h = bar[col_map['high']]
    l = bar[col_map['low']]
    c = bar[col_map['close']]

    total_range = h - l
    if total_range < 0.0003:  # Min 3 pips
        return False, 0

    body = abs(c - o)
    confidence = 0.0

    if direction == 'BUY':
        lower_wick = min(o, c) - l
        wick_ratio = lower_wick / total_range

        # Basic rejection criteria
        if lower_wick < body or wick_ratio < 0.4:
            return False, 0

        # Score based on wick ratio
        if wick_ratio > 0.7:
            confidence += 40
        elif wick_ratio > 0.5:
            confidence += 30
        else:
            confidence += 20

        # Small body bonus (strong rejection)
        body_ratio = body / total_range
        if body_ratio < 0.2:
            confidence += 20
        elif body_ratio < 0.3:
            confidence += 10

        # Bullish close bonus
        if c > o:
            confidence += 15

    else:  # SELL
        upper_wick = h - max(o, c)
        wick_ratio = upper_wick / total_range

        if upper_wick < body or wick_ratio < 0.4:
            return False, 0

        if wick_ratio > 0.7:
            confidence += 40
        elif wick_ratio > 0.5:
            confidence += 30
        else:
            confidence += 20

        body_ratio = body / total_range
        if body_ratio < 0.2:
            confidence += 20
        elif body_ratio < 0.3:
            confidence += 10

        if c < o:
            confidence += 15

    # Volume confirmation
    if 'volume' in prev_bars.columns or 'Volume' in prev_bars.columns:
        vol_col = 'volume' if 'volume' in prev_bars.columns else 'Volume'
        if vol_col in bar.index:
            avg_vol = prev_bars[vol_col].mean() if len(prev_bars) > 0 else 0
            current_vol = bar[vol_col]

            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                if vol_ratio > 1.5:
                    confidence += 15
                elif vol_ratio > 1.2:
                    confidence += 10

    # Context: Preceded by momentum (good setup)
    if len(prev_bars) >= 3:
        prev_closes = prev_bars[col_map['close']].values
        if direction == 'BUY':
            # Previous bars were bearish (good context)
            bearish_count = sum(1 for i in range(1, len(prev_closes)) if prev_closes[i] < prev_closes[i-1])
            if bearish_count >= 2:
                confidence += 10
        else:
            bullish_count = sum(1 for i in range(1, len(prev_closes)) if prev_closes[i] > prev_closes[i-1])
            if bullish_count >= 2:
                confidence += 10

    is_rejection = confidence >= 50
    return is_rejection, min(100, confidence)


# ============================================================================
# IMPROVEMENT 5: Volatility-Adaptive Trailing Stop
# ============================================================================

class ATRTrailingStop:
    """
    ATR-based trailing stop that adapts to volatility:
    - Initial stop: 2x ATR
    - Trail activates at 1x ATR profit
    - Trail distance: 1.5x ATR
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self.tr_buffer: List[float] = []
        self.prev_close: Optional[float] = None

    def update(self, high: float, low: float, close: float):
        """Update ATR calculation"""
        if self.prev_close is not None:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
            self.tr_buffer.append(tr)

            if len(self.tr_buffer) > self.atr_period * 2:
                self.tr_buffer = self.tr_buffer[-(self.atr_period * 2):]

        self.prev_close = close

    def get_atr(self) -> float:
        """Get current ATR value"""
        if len(self.tr_buffer) < self.atr_period:
            return 0.0020  # Default 20 pips

        return np.mean(self.tr_buffer[-self.atr_period:])

    def calculate_levels(
        self,
        entry_price: float,
        direction: str,
        atr_mult_sl: float = 2.0,
        atr_mult_tp1: float = 1.5,
        atr_mult_tp2: float = 3.0
    ) -> Tuple[float, float, float]:
        """
        Calculate SL/TP levels based on ATR

        Returns: (sl_price, tp1_price, tp2_price)
        """
        atr = self.get_atr()

        # Clamp ATR to reasonable range (10-50 pips)
        min_atr = 0.0010
        max_atr = 0.0050
        atr = max(min_atr, min(max_atr, atr))

        if direction == 'BUY':
            sl = entry_price - (atr * atr_mult_sl)
            tp1 = entry_price + (atr * atr_mult_tp1)
            tp2 = entry_price + (atr * atr_mult_tp2)
        else:
            sl = entry_price + (atr * atr_mult_sl)
            tp1 = entry_price - (atr * atr_mult_tp1)
            tp2 = entry_price - (atr * atr_mult_tp2)

        return sl, tp1, tp2

    def get_trailing_stop(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        atr_mult_trail: float = 1.5
    ) -> float:
        """
        Calculate trailing stop level
        Only trails if in profit by at least 1 ATR
        """
        atr = self.get_atr()

        if direction == 'BUY':
            profit = current_price - entry_price
            if profit > atr:  # Trail only if 1 ATR in profit
                new_sl = current_price - (atr * atr_mult_trail)
                return max(current_sl, new_sl)  # Only move up
        else:
            profit = entry_price - current_price
            if profit > atr:
                new_sl = current_price + (atr * atr_mult_trail)
                return min(current_sl, new_sl)  # Only move down

        return current_sl


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
    initial_sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    lot_size: float = 0.1
    pnl: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""
    exit_reason: str = ""
    in_killzone: bool = True
    session: str = ""
    regime: str = ""
    regime_confidence: float = 0.0
    activity_score: float = 0.0
    poi_type: str = ""
    entry_type: str = ""
    quality_score: float = 0.0
    rejection_confidence: float = 0.0
    risk_pct: float = 0.01


@dataclass
class BacktestStats:
    """Backtest statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
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

    london_trades: int = 0
    london_pnl: float = 0.0
    newyork_trades: int = 0
    newyork_pnl: float = 0.0
    hybrid_trades: int = 0
    hybrid_pnl: float = 0.0

    bullish_trades: int = 0
    bullish_pnl: float = 0.0
    bearish_trades: int = 0
    bearish_pnl: float = 0.0

    tp1_exits: int = 0
    tp2_exits: int = 0
    sl_exits: int = 0
    trailing_exits: int = 0
    regime_flip_exits: int = 0

    monthly_stats: Dict = field(default_factory=dict)

    # v3 specific
    avg_regime_confidence: float = 0.0
    avg_quality_score: float = 0.0
    avg_risk_pct: float = 0.0


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
# POI DETECTION (Enhanced)
# ============================================================================

def detect_order_block_enhanced(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    col_map: dict,
    lookback: int = 15
) -> Optional[dict]:
    """Detect Order Block with enhanced quality scoring"""
    if idx < lookback + 3:
        return None

    close_col = col_map['close']
    open_col = col_map['open']
    high_col = col_map['high']
    low_col = col_map['low']

    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 3):
        bar = recent.iloc[i]
        next_bars = recent.iloc[i+1:i+4]
        actual_idx = idx - lookback + i

        if direction == 'BUY':
            if bar[close_col] < bar[open_col]:  # Bearish candle
                move_up = next_bars[close_col].max() - bar[low_col]
                if move_up > 0.0010:  # 10 pips
                    quality = calculate_enhanced_ob_quality(df, actual_idx, direction, col_map)
                    return {
                        'type': 'OB',
                        'direction': 'BUY',
                        'zone_high': bar[high_col],
                        'zone_low': bar[low_col],
                        'quality': quality
                    }
        else:
            if bar[close_col] > bar[open_col]:  # Bullish candle
                move_down = bar[high_col] - next_bars[close_col].min()
                if move_down > 0.0010:
                    quality = calculate_enhanced_ob_quality(df, actual_idx, direction, col_map)
                    return {
                        'type': 'OB',
                        'direction': 'SELL',
                        'zone_high': bar[high_col],
                        'zone_low': bar[low_col],
                        'quality': quality
                    }

    return None


def detect_fvg(df: pd.DataFrame, idx: int, direction: str, col_map: dict, lookback: int = 8) -> Optional[dict]:
    """Detect Fair Value Gap"""
    if idx < lookback + 3:
        return None

    high_col = col_map['high']
    low_col = col_map['low']

    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 2):
        bar1 = recent.iloc[i]
        bar3 = recent.iloc[i+2]

        if direction == 'BUY':
            gap = bar3[low_col] - bar1[high_col]
            if gap > 0.0003:  # 3 pips
                return {
                    'type': 'FVG',
                    'direction': 'BUY',
                    'zone_high': bar3[low_col],
                    'zone_low': bar1[high_col],
                    'quality': min(100, gap * 10000 * 2)
                }
        else:
            gap = bar1[low_col] - bar3[high_col]
            if gap > 0.0003:
                return {
                    'type': 'FVG',
                    'direction': 'SELL',
                    'zone_high': bar1[low_col],
                    'zone_low': bar3[high_col],
                    'quality': min(100, gap * 10000 * 2)
                }

    return None


# ============================================================================
# ENTRY TRIGGER (Multiple Types)
# ============================================================================

def check_entry_trigger_v3(
    bar: pd.Series,
    prev_bars: pd.DataFrame,
    direction: str,
    col_map: dict
) -> Tuple[Optional[str], float]:
    """
    Multiple entry types with confidence scoring

    Returns: (entry_type, confidence)
    """
    o = bar[col_map['open']]
    h = bar[col_map['high']]
    l = bar[col_map['low']]
    c = bar[col_map['close']]

    total_range = h - l
    if total_range < 0.0003:
        return None, 0

    body = abs(c - o)
    is_bullish = c > o
    is_bearish = c < o

    # Get previous bar
    if len(prev_bars) == 0:
        return None, 0

    prev_bar = prev_bars.iloc[-1]
    po = prev_bar[col_map['open']]
    ph = prev_bar[col_map['high']]
    pl = prev_bar[col_map['low']]
    pc = prev_bar[col_map['close']]

    if direction == 'BUY':
        # 1. Advanced rejection (highest confidence)
        is_rej, rej_conf = detect_advanced_rejection(bar, prev_bars, direction, col_map)
        if is_rej:
            return "REJECTION", rej_conf

        # 2. Bullish momentum candle
        if is_bullish and body > total_range * 0.6:
            return "MOMENTUM", 60

        # 3. Higher low + bullish close
        if l > pl and is_bullish:
            return "HIGHER_LOW", 55

    else:  # SELL
        # 1. Advanced rejection
        is_rej, rej_conf = detect_advanced_rejection(bar, prev_bars, direction, col_map)
        if is_rej:
            return "REJECTION", rej_conf

        # 2. Bearish momentum
        if is_bearish and body > total_range * 0.6:
            return "MOMENTUM", 60

        # 3. Lower high + bearish close
        if h < ph and is_bearish:
            return "LOWER_HIGH", 55

    return None, 0


# ============================================================================
# MAIN BACKTEST
# ============================================================================

def run_backtest_v3(df: pd.DataFrame, use_hybrid: bool = True) -> tuple:
    """Run backtest with all v3 improvements"""

    # Initialize components
    killzone = KillZone()
    activity_filter = DynamicActivityFilter(
        min_atr_pips=5.0,
        min_bar_range_pips=3.0,
        activity_threshold=35.0,
        pip_size=0.0001
    )
    activity_filter.outside_kz_min_score = 60.0

    kalman = MultiScaleKalman()

    # v3 improvements
    ensemble_regime = EnsembleRegimeDetector()
    kelly_sizer = KellyPositionSizer(
        base_risk=0.01,
        kelly_fraction=0.25,
        max_risk=0.02,
        min_risk=0.005
    )
    atr_trailing = ATRTrailingStop(atr_period=14)

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Warmup
    print("      Warming up indicators...")
    for _, row in df.head(100).iterrows():
        price = row[col_map['close']]
        high = row[col_map['high']]
        low = row[col_map['low']]

        kalman.update(price)
        ensemble_regime.update(high, low, price)
        atr_trailing.update(high, low, price)

    trades: List[BacktestTrade] = []
    position: Optional[BacktestTrade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    cooldown_after_sl = timedelta(hours=1)
    cooldown_after_tp = timedelta(minutes=30)

    print("      Processing bars...")
    total_bars = len(df) - 100

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bars = df.iloc[max(0, idx-5):idx]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        kalman.update(price)
        regime_info = ensemble_regime.update(high, low, price)
        atr_trailing.update(high, low, price)

        if (idx - 100) % 500 == 0:
            pct = (idx - 100) / total_bars * 100
            print(f"      Progress: {pct:.0f}% ({idx-100}/{total_bars} bars)")

        # Manage open position
        if position:
            # Update trailing stop (v3 improvement)
            new_sl = atr_trailing.get_trailing_stop(
                position.direction,
                position.entry_price,
                price,
                position.sl
            )
            if new_sl != position.sl:
                position.sl = new_sl

            # Check SL
            if position.direction == 'BUY' and low <= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.sl - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                position.exit_reason = 'TRAILING' if position.sl > position.initial_sl else 'SL'
                balance += position.pnl

                # Update Kelly
                kelly_sizer.add_trade_result(position.result == 'WIN', position.pnl)

                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue

            elif position.direction == 'SELL' and high >= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.entry_price - position.sl) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                position.exit_reason = 'TRAILING' if position.sl < position.initial_sl else 'SL'
                balance += position.pnl

                kelly_sizer.add_trade_result(position.result == 'WIN', position.pnl)

                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_sl
                continue

            # Check TP1 (1.5R)
            if position.direction == 'BUY' and high >= position.tp1:
                position.exit_time = current_time
                position.exit_price = position.tp1
                position.pnl_pips = (position.tp1 - position.entry_price) * 10000
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN'
                position.exit_reason = 'TP1'
                balance += position.pnl

                kelly_sizer.add_trade_result(True, position.pnl)

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

                kelly_sizer.add_trade_result(True, position.pnl)

                trades.append(position)
                position = None
                cooldown_until = current_time + cooldown_after_tp
                continue

            # Check regime flip exit
            if regime_info and regime_info.is_tradeable:
                if position.direction == 'BUY' and regime_info.bias == 'SELL':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (price - position.entry_price) * 10000
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl

                    kelly_sizer.add_trade_result(position.result == 'WIN', position.pnl)

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

                    kelly_sizer.add_trade_result(position.result == 'WIN', position.pnl)

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

        # Skip if cooldown active
        if cooldown_until and current_time < cooldown_until:
            continue

        # Skip if already in position
        if position:
            continue

        # Check Kill Zone
        in_kz, session = killzone.is_in_killzone(current_time)

        # Check hybrid mode
        can_trade_outside = False
        activity_score = 0.0
        if use_hybrid and not in_kz:
            recent_df = df.iloc[max(0, idx-20):idx+1]
            activity = activity_filter.check_activity(current_time, high, low, recent_df)
            activity_score = activity.score
            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE] and activity.score >= 60:
                can_trade_outside = True

        should_trade = in_kz or can_trade_outside
        if not should_trade:
            continue

        # Check ensemble regime (v3)
        if not regime_info or not regime_info.is_tradeable:
            continue
        if regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias

        # Detect POI with enhanced quality (v3)
        poi = detect_order_block_enhanced(df, idx, direction, col_map)
        if not poi:
            poi = detect_fvg(df, idx, direction, col_map)

        if not poi:
            continue

        # Only trade high quality POIs (v3 improvement)
        if poi['quality'] < 50:
            continue

        # Check price near POI zone
        poi_tolerance = 0.0015
        if direction == 'BUY':
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue
        else:
            if not (poi['zone_low'] - poi_tolerance <= price <= poi['zone_high'] + poi_tolerance):
                continue

        # Check entry trigger with confidence (v3)
        entry_type, entry_confidence = check_entry_trigger_v3(bar, prev_bars, direction, col_map)
        if not entry_type:
            continue

        # Combined quality score
        combined_quality = (poi['quality'] + entry_confidence) / 2

        # Calculate ATR-based SL/TP (v3)
        sl_price, tp1_price, tp2_price = atr_trailing.calculate_levels(
            price, direction,
            atr_mult_sl=2.0,
            atr_mult_tp1=1.5,
            atr_mult_tp2=3.0
        )

        # Kelly position sizing (v3)
        risk_pct = kelly_sizer.get_position_risk(combined_quality)
        risk_amount = balance * risk_pct

        sl_distance = abs(price - sl_price)
        if sl_distance > 0:
            lot_size = risk_amount / (sl_distance * 100000)  # 100k per lot
            lot_size = max(0.01, min(1.0, round(lot_size, 2)))
        else:
            lot_size = 0.01

        # Open position
        position = BacktestTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=price,
            sl=sl_price,
            initial_sl=sl_price,
            tp1=tp1_price,
            tp2=tp2_price,
            lot_size=lot_size,
            in_killzone=in_kz,
            session=session if in_kz else "Hybrid",
            regime=regime_info.regime.value,
            regime_confidence=regime_info.confidence,
            activity_score=activity_score,
            poi_type=poi['type'],
            entry_type=entry_type,
            quality_score=combined_quality,
            rejection_confidence=entry_confidence if entry_type == "REJECTION" else 0,
            risk_pct=risk_pct
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

    stats = calculate_stats_v3(trades, balance, max_dd, df)
    return trades, stats, balance


def calculate_stats_v3(trades: List[BacktestTrade], final_balance: float, max_dd: float, df: pd.DataFrame) -> BacktestStats:
    """Calculate statistics with v3 additions"""
    stats = BacktestStats()

    if not trades:
        return stats

    stats.total_trades = len(trades)
    stats.wins = sum(1 for t in trades if t.result == 'WIN')
    stats.losses = sum(1 for t in trades if t.result == 'LOSS')
    stats.breakeven = sum(1 for t in trades if t.result == 'BE')
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

    durations = []
    for t in trades:
        if t.exit_time and t.entry_time:
            duration = (t.exit_time - t.entry_time).total_seconds() / 3600
            durations.append(duration)
    stats.avg_trade_duration = np.mean(durations) if durations else 0

    # Trades per day
    if len(df) > 0:
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            stats.trades_per_day = stats.total_trades / days

    # By session
    london_trades = [t for t in trades if t.session == 'London']
    newyork_trades = [t for t in trades if t.session == 'New York']
    hybrid_trades = [t for t in trades if t.session == 'Hybrid']

    stats.london_trades = len(london_trades)
    stats.london_pnl = sum(t.pnl for t in london_trades)
    stats.newyork_trades = len(newyork_trades)
    stats.newyork_pnl = sum(t.pnl for t in newyork_trades)
    stats.hybrid_trades = len(hybrid_trades)
    stats.hybrid_pnl = sum(t.pnl for t in hybrid_trades)

    # By regime
    bullish_trades = [t for t in trades if t.regime == 'BULLISH']
    bearish_trades = [t for t in trades if t.regime == 'BEARISH']

    stats.bullish_trades = len(bullish_trades)
    stats.bullish_pnl = sum(t.pnl for t in bullish_trades)
    stats.bearish_trades = len(bearish_trades)
    stats.bearish_pnl = sum(t.pnl for t in bearish_trades)

    # By exit reason
    stats.tp1_exits = sum(1 for t in trades if t.exit_reason == 'TP1')
    stats.tp2_exits = sum(1 for t in trades if t.exit_reason == 'TP2')
    stats.sl_exits = sum(1 for t in trades if t.exit_reason == 'SL')
    stats.trailing_exits = sum(1 for t in trades if t.exit_reason == 'TRAILING')
    stats.regime_flip_exits = sum(1 for t in trades if t.exit_reason == 'REGIME_FLIP')

    # v3 specific stats
    stats.avg_regime_confidence = np.mean([t.regime_confidence for t in trades])
    stats.avg_quality_score = np.mean([t.quality_score for t in trades])
    stats.avg_risk_pct = np.mean([t.risk_pct for t in trades]) * 100

    # Monthly
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


def print_report_v3(stats: BacktestStats, final_balance: float, trades: List[BacktestTrade]):
    """Print detailed report with v3 additions"""
    print()
    print("=" * 70)
    print("H1 DETAILED BACKTEST v3 RESULTS")
    print("(Research-Based Improvements)")
    print("=" * 70)
    print()

    print("v3 IMPROVEMENTS APPLIED:")
    print("-" * 50)
    print("1. Ensemble Regime Detection (HMM + ADX + CCI + Structure)")
    print("2. Enhanced OB Quality (volume + wicks + retests)")
    print("3. Kelly Criterion Position Sizing")
    print("4. Advanced Rejection Candle (volume + context)")
    print("5. Volatility-Adaptive Trailing Stops (ATR-based)")
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

    print("v3 METRICS")
    print("-" * 50)
    print(f"Avg Regime Confidence: {stats.avg_regime_confidence:.1f}%")
    print(f"Avg Quality Score:     {stats.avg_quality_score:.1f}")
    print(f"Avg Position Risk:     {stats.avg_risk_pct:.2f}%")
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
    print("-" * 50)
    print(f"{'Session':<15} {'Trades':>8} {'P/L':>12}")
    print(f"{'London':<15} {stats.london_trades:>8} ${stats.london_pnl:>+10.2f}")
    print(f"{'New York':<15} {stats.newyork_trades:>8} ${stats.newyork_pnl:>+10.2f}")
    print(f"{'Hybrid':<15} {stats.hybrid_trades:>8} ${stats.hybrid_pnl:>+10.2f}")
    print()

    print("EXIT REASONS")
    print("-" * 50)
    print(f"TP1 (ATR-based):     {stats.tp1_exits}")
    print(f"TP2 (ATR-based):     {stats.tp2_exits}")
    print(f"Stop Loss:           {stats.sl_exits}")
    print(f"Trailing Stop:       {stats.trailing_exits}")
    print(f"Regime Flip:         {stats.regime_flip_exits}")
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

    # Comparison with v2
    print("COMPARISON WITH v2")
    print("-" * 50)
    print("v2 Results: 106 trades, 48.1% WR, +$2,018, PF 1.22")
    print(f"v3 Results: {stats.total_trades} trades, {stats.win_rate:.1f}% WR, ${stats.total_pnl:+,.0f}, PF {stats.profit_factor:.2f}")

    wr_diff = stats.win_rate - 48.1
    pnl_diff = stats.total_pnl - 2018
    pf_diff = stats.profit_factor - 1.22

    print()
    print(f"Win Rate Change:     {wr_diff:+.1f}%")
    print(f"P/L Change:          ${pnl_diff:+,.0f}")
    print(f"Profit Factor Change: {pf_diff:+.2f}")
    print()


async def send_telegram_report_v3(stats: BacktestStats, trades: List[BacktestTrade], final_balance: float):
    """Send report to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        # Entry breakdown
        entry_types = {}
        for t in trades:
            et = t.entry_type
            if et not in entry_types:
                entry_types[et] = {'count': 0, 'pnl': 0}
            entry_types[et]['count'] += 1
            entry_types[et]['pnl'] += t.pnl

        entry_str = "\n".join([f"  {k}: {v['count']} trades, ${v['pnl']:+.2f}" for k, v in sorted(entry_types.items(), key=lambda x: -x[1]['pnl'])])

        msg = f"""*H1 BACKTEST v3 (Research-Based)*
Period: Jan 2025 - Jan 2026 (13 months)

*v3 IMPROVEMENTS*
1. Ensemble Regime (HMM+ADX+CCI)
2. Enhanced OB Quality
3. Kelly Criterion Sizing
4. Advanced Rejection
5. ATR Trailing Stops

*PERFORMANCE*
  Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
  Win Rate: {stats.win_rate:.1f}%
  Net P/L: ${stats.total_pnl:+,.2f} ({(final_balance/10000-1)*100:+.1f}%)
  Profit Factor: {stats.profit_factor:.2f}
  Max DD: ${stats.max_drawdown:,.2f}

*v3 METRICS*
  Avg Regime Conf: {stats.avg_regime_confidence:.1f}%
  Avg Quality: {stats.avg_quality_score:.1f}
  Avg Risk: {stats.avg_risk_pct:.2f}%

*ENTRY TYPES*
{entry_str}

*vs v2*
v2: 106 trades, 48% WR, +$2,018
v3: {stats.total_trades} trades, {stats.win_rate:.0f}% WR, ${stats.total_pnl:+,.0f}

Final Balance: ${final_balance:,.2f}
"""
        await telegram.send(msg)
        logger.info("Report sent to Telegram!")

    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI DETAILED H1 BACKTEST v3")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("Strategy: Research-Based Improvements")
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

    print("\n[2/3] Running backtest v3...")
    trades, stats, final_balance = run_backtest_v3(df, use_hybrid=True)

    print_report_v3(stats, final_balance, trades)

    print("[3/3] Sending report to Telegram...")
    await send_telegram_report_v3(stats, trades, final_balance)

    print("=" * 70)
    print("BACKTEST v3 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
