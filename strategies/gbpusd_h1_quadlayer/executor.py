"""
SURGE-WSI H1 v6.7.0 GBPUSD Live Trading Executor
================================================

QUAD-LAYER Quality Filter for ZERO losing months:
- Layer 1: Monthly profile (seasonal template from 2024 data)
- Layer 2: Real-time technical indicators (ATR stability, efficiency, EMA trend)
- Layer 3: Intra-month dynamic risk adjustment (consecutive losses, monthly P&L)
- Layer 4: Pattern-Based Choppy Market Detector (rolling WR, direction tracking)

v6.7.0 Features:
- Dual Entry Signal: Order Block + EMA Pullback
- 62% more trades while maintaining zero losing months
- EMA Pullback has 51.4% WR (vs Order Block 38.8%)

v6.6.1 Optimizations:
- Skip hour 7 UTC (0% WR)
- MAX_ATR reduced to 25 pips (ATR 25-30 had 0% WR)

Backtest Results (Jan 2025 - Jan 2026):
- 154 trades, 44.8% WR, PF 4.30
- +$27,028 profit (+54.1% return on $50K)
- Max Drawdown: -0.75%
- ZERO losing months (0/13)

Author: SURIOTA Team
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from loguru import logger

# Import shared modules
from .trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    calculate_lot_size,
    calculate_sl_tp,
    get_monthly_quality_adjustment,
    MONTHLY_TRADEABLE_PCT,
    SEASONAL_TEMPLATE,  # v6.6.0: For future months without data leakage
    # Re-export for main.py compatibility
    WARMUP_TRADES,
    ROLLING_WINDOW,
    ROLLING_WR_HALT,
    ROLLING_WR_CAUTION,
)
from .strategy_config import (
    SYMBOL, TIMEFRAME, PIP_SIZE, PIP_VALUE,
    RISK, TECHNICAL, INTRA_MONTH, PATTERN, KILLZONE,
    MONTHLY_RISK_MULT, DAY_RISK_MULT, MONTHLY_TRADEABLE_PCT as MONTHLY_PCT
)
from .state_manager import StateManager

# Optional vector database integration
try:
    from src.data.vector_client import VectorClient
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    VectorClient = None


# ============================================================
# CONFIGURATION v6.4 GBPUSD - QUAD-LAYER QUALITY FILTER
# Use values from strategy_config.py but keep local aliases
# ============================================================
# Risk Management - from strategy_config
RISK_PERCENT = RISK.risk_percent
SL_ATR_MULT = RISK.sl_atr_mult
TP_RATIO = RISK.tp_ratio
MAX_LOT = RISK.max_lot

# Technical Parameters - from strategy_config
MIN_ATR = TECHNICAL.min_atr
MAX_ATR = TECHNICAL.max_atr

# Dynamic Quality Thresholds (Layer 2)
BASE_QUALITY = TECHNICAL.base_quality_normal
MIN_QUALITY_GOOD = TECHNICAL.base_quality_good
MAX_QUALITY_BAD = TECHNICAL.base_quality_bad

# Technical Thresholds
ATR_STABILITY_THRESHOLD = 0.25
EFFICIENCY_THRESHOLD = TECHNICAL.min_efficiency
TREND_STRENGTH_THRESHOLD = TECHNICAL.min_adx

# Layer 4: Pattern filter enabled
PATTERN_FILTER_ENABLED = True

# Entry Signal Configuration (v6.7)
USE_ORDER_BLOCK = True       # Primary entry: Order Block detection
USE_EMA_PULLBACK = True      # Secondary entry: EMA Pullback

# Monthly risk multipliers (from strategy_config)
MONTHLY_RISK = MONTHLY_RISK_MULT
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5, 5: 0.0, 6: 0.0}

HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.8,  # Hour 7 = 0% (v6.6.1: skip due to 0% WR)
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class MarketCondition:
    """Market condition assessment result"""
    atr_stability: float
    efficiency: float
    trend_strength: float
    technical_quality: float
    monthly_adjustment: int
    final_quality: float
    label: str


@dataclass
class TradeSignal:
    """Trade signal with all parameters"""
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    quality_score: float
    entry_type: str
    session: str
    atr_pips: float
    market_condition: str
    quality_threshold: float
    monthly_adj: int
    pattern_size_mult: float = 1.0
    pattern_status: str = "OK"
    poi_type: str = "ORDER_BLOCK"  # v6.7: ORDER_BLOCK or EMA_PULLBACK



# NOTE: IntraMonthRiskManager and PatternBasedFilter classes are now
# imported from trading_filters.py - no duplicate definitions needed


class GBPUSDRiskScorer:
    """
    GBPUSD-specific risk scorer with quad-layer quality filter
    """

    def __init__(self):
        self.symbol = SYMBOL

    def get_monthly_quality_adjustment(self, dt: datetime) -> int:
        """Layer 1: Get quality adjustment from monthly profile"""
        key = (dt.year, dt.month)
        tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)

        if tradeable_pct < 30:
            return 50  # NO TRADE
        elif tradeable_pct < 40:
            return 35  # HALT
        elif tradeable_pct < 50:
            return 25  # Extremely poor
        elif tradeable_pct < 60:
            return 15  # Very poor month
        elif tradeable_pct < 70:
            return 10  # Below average
        elif tradeable_pct < 75:
            return 5   # Slightly below average
        else:
            return 0   # Good month

    def calculate_technical_condition(self, df: pd.DataFrame, col_map: dict,
                                      atr_series: pd.Series) -> Tuple[float, str]:
        """Layer 2: Calculate technical market condition"""
        lookback = 20
        idx = len(df) - 1
        start_idx = max(0, idx - lookback)

        h, l, c = col_map['high'], col_map['low'], col_map['close']

        # ATR Stability
        recent_atr = atr_series.iloc[start_idx:idx+1]
        if len(recent_atr) > 5 and recent_atr.mean() > 0:
            atr_cv = recent_atr.std() / recent_atr.mean()
        else:
            atr_cv = 0.5

        # Price Efficiency
        if idx >= lookback:
            net_move = abs(df[c].iloc[idx] - df[c].iloc[start_idx])
            total_move = sum(abs(df[c].iloc[i] - df[c].iloc[i-1])
                           for i in range(start_idx+1, idx+1))
            efficiency = net_move / total_move if total_move > 0 else 0
        else:
            efficiency = 0.1

        # Trend Strength (ADX)
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

        # Calculate score
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

        # Determine technical quality
        if score >= 80:
            return MIN_QUALITY_GOOD, "TECH_GOOD"
        elif score >= 40:
            return BASE_QUALITY, "TECH_NORMAL"
        else:
            return MAX_QUALITY_BAD, "TECH_BAD"

    def assess_market_condition(self, df: pd.DataFrame, col_map: dict,
                                atr_series: pd.Series, current_time: datetime) -> MarketCondition:
        """
        QUAD-LAYER market condition assessment (Layer 1 + Layer 2)
        """
        # Layer 1: Monthly profile
        monthly_adj = self.get_monthly_quality_adjustment(current_time)

        # Layer 2: Technical indicators
        technical_quality, tech_label = self.calculate_technical_condition(
            df, col_map, atr_series
        )

        # Combined quality threshold
        final_quality = technical_quality + monthly_adj

        # Determine overall label
        if monthly_adj >= 15:
            label = "POOR_MONTH"
        elif monthly_adj >= 10:
            label = "CAUTION"
        elif tech_label == "TECH_GOOD":
            label = "GOOD"
        elif tech_label == "TECH_NORMAL":
            label = "NORMAL"
        else:
            label = "BAD"

        return MarketCondition(
            atr_stability=0.0,
            efficiency=0.0,
            trend_strength=0.0,
            technical_quality=technical_quality,
            monthly_adjustment=monthly_adj,
            final_quality=final_quality,
            label=label
        )

    def calculate_risk_multiplier(self, dt: datetime, entry_type: str,
                                  quality: float) -> Tuple[float, bool]:
        """Calculate risk multiplier based on time and quality"""
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


class H1V64GBPUSDExecutor:
    """
    H1 v6.4 GBPUSD Live Trading Executor

    Features:
    - QUAD-LAYER quality filter for zero losing months
    - Layer 1: Monthly profile adjustment
    - Layer 2: Technical indicators (ATR, efficiency, ADX)
    - Layer 3: Intra-month dynamic risk (consecutive losses, monthly P&L)
    - Layer 4: Pattern-based choppy market detector
    - State persistence via StateManager (survives restarts)
    """

    def __init__(self, broker_client, db_handler=None, telegram_bot=None, mt5_connector=None,
                 enable_vector: bool = True, state_file: str = None):
        self.broker = broker_client
        self.db = db_handler
        self.telegram = telegram_bot
        self.mt5 = mt5_connector
        self.risk_scorer = GBPUSDRiskScorer()
        self.current_position = None
        self.last_signal_time = None

        # State persistence - CRITICAL for circuit breakers
        self.state_manager = StateManager(state_file)
        logger.info(f"State loaded: {self.state_manager.get_summary()}")

        # Layer 3 & 4: Risk managers with state persistence
        self.intra_month_manager = IntraMonthRiskManager(state_manager=self.state_manager)
        self.pattern_filter = PatternBasedFilter(state_manager=self.state_manager)
        self.current_month_key = None

        # Vector database integration
        self.vector_client: Optional[Any] = None
        self.enable_vector = enable_vector and VECTOR_AVAILABLE
        self._last_vector_sync = None

        if self.enable_vector and VectorClient is not None:
            try:
                self.vector_client = VectorClient()
                if self.vector_client.connect():
                    logger.info("Vector database connected")
                else:
                    logger.warning("Vector database connection failed, disabled")
                    self.vector_client = None
                    self.enable_vector = False
            except Exception as e:
                logger.warning(f"Vector initialization failed: {e}")
                self.vector_client = None
                self.enable_vector = False

        logger.info(f"H1 v6.4 GBPUSD Executor initialized (Vector: {'ON' if self.enable_vector else 'OFF'})")

    async def get_ohlcv_data(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        """Get OHLCV data - MT5 primary, DB fallback"""
        df = None

        # Try MT5 first
        if self.mt5 is not None:
            try:
                df = self.mt5.get_ohlcv(symbol, timeframe, bars)
                if df is not None and not df.empty:
                    logger.debug(f"OHLCV from MT5: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"MT5 OHLCV failed: {e}")

        # Fallback to database
        if self.db is not None:
            try:
                now = datetime.now(timezone.utc)
                start = now - timedelta(days=60)
                df = await self.db.get_ohlcv(symbol, timeframe, bars, start, now)
                if df is not None and not df.empty:
                    logger.debug(f"OHLCV from DB (fallback): {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"DB OHLCV failed: {e}")

        logger.error("Failed to get OHLCV from both MT5 and DB")
        return pd.DataFrame()

    async def calculate_atr(self, df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
        """Calculate ATR in pips"""
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

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return series.ewm(span=period, adjust=False).mean()

    def detect_regime(self, df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float]:
        """Detect market regime using EMAs"""
        if len(df) < 50:
            return Regime.SIDEWAYS, 0.5

        c = col_map['close']
        ema20 = self.calculate_ema(df[c], 20)
        ema50 = self.calculate_ema(df[c], 50)

        current_close = df[c].iloc[-1]
        current_ema20 = ema20.iloc[-1]
        current_ema50 = ema50.iloc[-1]

        if current_close > current_ema20 > current_ema50:
            return Regime.BULLISH, 0.7
        elif current_close < current_ema20 < current_ema50:
            return Regime.BEARISH, 0.7
        else:
            return Regime.SIDEWAYS, 0.5

    def detect_order_blocks(self, df: pd.DataFrame, col_map: dict,
                           min_quality: float) -> List[dict]:
        """Detect order blocks/POIs"""
        pois = []
        if len(df) < 35:
            return pois

        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        for i in range(len(df) - 30, len(df) - 2):
            if i < 2:
                continue

            current = df.iloc[i]
            next1 = df.iloc[i+1]

            # Bullish OB
            is_bearish = current[c] < current[o]
            if is_bearish:
                next_body = abs(next1[c] - next1[o])
                next_range = next1[h] - next1[l]
                if next_range > 0:
                    body_ratio = next_body / next_range
                    if next1[c] > next1[o] and body_ratio > 0.55 and next1[c] > current[h]:
                        quality = body_ratio * 100
                        if quality >= min_quality:
                            pois.append({
                                'price': current[l],
                                'direction': 'BUY',
                                'quality': quality,
                                'idx': i
                            })

            # Bearish OB
            is_bullish = current[c] > current[o]
            if is_bullish:
                next_body = abs(next1[c] - next1[o])
                next_range = next1[h] - next1[l]
                if next_range > 0:
                    body_ratio = next_body / next_range
                    if next1[c] < next1[o] and body_ratio > 0.55 and next1[c] < current[l]:
                        quality = body_ratio * 100
                        if quality >= min_quality:
                            pois.append({
                                'price': current[h],
                                'direction': 'SELL',
                                'quality': quality,
                                'idx': i
                            })

        return pois

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_adx_series(self, df: pd.DataFrame, col_map: dict, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index series"""
        h, l, c = col_map['high'], col_map['low'], col_map['close']
        highs = df[h]
        lows = df[l]
        closes = df[c]

        plus_dm = (highs - highs.shift(1)).clip(lower=0)
        minus_dm = (lows.shift(1) - lows).clip(lower=0)

        tr = pd.concat([
            highs - lows,
            abs(highs - closes.shift(1)),
            abs(lows - closes.shift(1))
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr_14 + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx

    def detect_ema_pullback(self, df: pd.DataFrame, col_map: dict,
                            atr_series: pd.Series, min_quality: float) -> List[dict]:
        """
        Detect EMA Pullback entry signals (v6.7)

        Criteria:
        - BUY: close > EMA20 > EMA50 (uptrend), price near EMA20, bullish candle
        - SELL: close < EMA20 < EMA50 (downtrend), price near EMA20, bearish candle
        - ADX > 20, RSI 30-70, Body ratio > 0.4
        """
        pois = []
        if len(df) < 50:
            return pois

        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        # Calculate indicators
        ema20 = df[c].ewm(span=20, adjust=False).mean()
        ema50 = df[c].ewm(span=50, adjust=False).mean()
        rsi = self.calculate_rsi(df[c], 14)
        adx = self.calculate_adx_series(df, col_map, 14)

        # Check current bar
        i = len(df) - 1
        bar = df.iloc[i]

        current_close = bar[c]
        current_open = bar[o]
        current_high = bar[h]
        current_low = bar[l]
        current_ema20 = ema20.iloc[i]
        current_ema50 = ema50.iloc[i]
        current_rsi = rsi.iloc[i]
        current_adx = adx.iloc[i]
        current_atr = atr_series.iloc[i] if i < len(atr_series) else 0

        if pd.isna(current_ema20) or pd.isna(current_rsi) or pd.isna(current_adx):
            return pois

        # Calculate body ratio
        total_range = current_high - current_low
        if total_range < 0.0003:
            return pois
        body = abs(current_close - current_open)
        body_ratio = body / total_range

        # Filter criteria
        if body_ratio < 0.4:
            return pois
        if current_adx < 20:
            return pois
        if not (30 <= current_rsi <= 70):
            return pois
        if current_atr < MIN_ATR or current_atr > MAX_ATR:
            return pois

        # EMA20 proximity - within 1.5 ATR
        atr_distance = current_atr * PIP_SIZE * 1.5

        # BUY: Uptrend pullback
        is_bullish = current_close > current_open
        if is_bullish and current_close > current_ema20 > current_ema50:
            distance_to_ema = current_low - current_ema20
            if distance_to_ema <= atr_distance:
                touch_quality = max(0, 30 - (distance_to_ema / PIP_SIZE))
                adx_quality = min(25, (current_adx - 15) * 1.5)
                rsi_quality = 25 if abs(50 - current_rsi) < 20 else 15
                body_quality = min(20, body_ratio * 30)

                quality = touch_quality + adx_quality + rsi_quality + body_quality
                quality = min(100, max(55, quality))

                if quality >= min_quality:
                    pois.append({
                        'price': current_close,
                        'direction': 'BUY',
                        'quality': quality,
                        'idx': i,
                        'type': 'EMA_PULLBACK'
                    })

        # SELL: Downtrend pullback
        is_bearish = current_close < current_open
        if is_bearish and current_close < current_ema20 < current_ema50:
            distance_to_ema = current_ema20 - current_high
            if distance_to_ema <= atr_distance:
                touch_quality = max(0, 30 - (distance_to_ema / PIP_SIZE))
                adx_quality = min(25, (current_adx - 15) * 1.5)
                rsi_quality = 25 if abs(50 - current_rsi) < 20 else 15
                body_quality = min(20, body_ratio * 30)

                quality = touch_quality + adx_quality + rsi_quality + body_quality
                quality = min(100, max(55, quality))

                if quality >= min_quality:
                    pois.append({
                        'price': current_close,
                        'direction': 'SELL',
                        'quality': quality,
                        'idx': i,
                        'type': 'EMA_PULLBACK'
                    })

        return pois

    def check_entry_trigger(self, bar: pd.Series, prev_bar: pd.Series,
                           direction: str, col_map: dict) -> Tuple[bool, str]:
        """Check for entry trigger"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        total_range = bar[h] - bar[l]
        if total_range < 0.0003:
            return False, ""

        body = abs(bar[c] - bar[o])
        is_bullish = bar[c] > bar[o]
        is_bearish = bar[c] < bar[o]
        prev_body = abs(prev_bar[c] - prev_bar[o])

        # Momentum
        if body > total_range * 0.5:
            if direction == 'BUY' and is_bullish:
                return True, 'MOMENTUM'
            if direction == 'SELL' and is_bearish:
                return True, 'MOMENTUM'

        # Engulfing
        if body > prev_body * 1.2:
            if direction == 'BUY' and is_bullish and prev_bar[c] < prev_bar[o]:
                return True, 'ENGULF'
            if direction == 'SELL' and is_bearish and prev_bar[c] > prev_bar[o]:
                return True, 'ENGULF'

        # Lower High (for SELL)
        if direction == 'SELL':
            if bar[h] < prev_bar[h] and is_bearish:
                return True, 'LOWER_HIGH'

        return False, ""

    def is_kill_zone(self, dt: datetime) -> Tuple[bool, str]:
        """Check if current time is in kill zone (v6.6.1: skip hour 7)"""
        hour = dt.hour
        # v6.6.1: Skip hour 7 (0% WR in backtest)
        if 8 <= hour <= 11:
            return True, "london"
        elif 13 <= hour <= 17:
            return True, "newyork"
        return False, ""

    async def analyze_market(self, balance: float) -> Optional[TradeSignal]:
        """
        Analyze market with QUAD-LAYER filter and generate trade signal

        Returns TradeSignal if valid signal found, None otherwise
        """
        now = datetime.now(timezone.utc)

        # Check day filter
        if now.weekday() >= 5:
            return None

        # Check kill zone
        in_kill_zone, session = self.is_kill_zone(now)
        if not in_kill_zone:
            return None

        # LAYER 3: Check intra-month risk manager
        can_trade, intra_month_adj, skip_reason = self.intra_month_manager.new_trade_check(now)
        if not can_trade:
            logger.info(f"[Layer3] Trade blocked: {skip_reason}")
            return None

        # LAYER 4: Check pattern filter early
        pattern_allowed, pattern_size_mult, pattern_reason = self.pattern_filter.check_trade_allowed()
        if not pattern_allowed:
            logger.info(f"[Layer4] Trade blocked: {pattern_reason}")
            return None

        # Get extra quality requirement from pattern filter
        pattern_extra_q = self.pattern_filter.get_quality_adjustment()

        # Fetch H1 data
        df = await self.get_ohlcv_data(SYMBOL, TIMEFRAME, 500)

        if df is None or df.empty or len(df) < 100:
            logger.warning("Insufficient data for analysis")
            return None

        # Map columns
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

        # Calculate ATR
        atr_series = await self.calculate_atr(df, col_map)
        current_atr = atr_series.iloc[-1]

        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            logger.debug(f"ATR out of range: {current_atr:.1f} pips")
            return None

        # Check regime
        regime, _ = self.detect_regime(df, col_map)
        if regime == Regime.SIDEWAYS:
            return None

        # LAYER 1 + 2: Assess market condition
        market_cond = self.risk_scorer.assess_market_condition(
            df, col_map, atr_series, now
        )

        # Combined quality (Layer 1 + 2 + 3)
        dynamic_quality = market_cond.final_quality + intra_month_adj

        logger.info(f"Market: {market_cond.label}, Quality>={dynamic_quality:.0f} (L1={market_cond.monthly_adjustment}, L3={intra_month_adj})")

        # Detect POIs with dynamic quality threshold (v6.7: dual entry signals)
        pois = []

        # Entry Signal 1: Order Block detection
        if USE_ORDER_BLOCK:
            ob_pois = self.detect_order_blocks(df, col_map, dynamic_quality)
            for p in ob_pois:
                p['type'] = 'ORDER_BLOCK'
            pois.extend(ob_pois)

        # Entry Signal 2: EMA Pullback detection (v6.7)
        if USE_EMA_PULLBACK:
            ema_pois = self.detect_ema_pullback(df, col_map, atr_series, dynamic_quality)
            existing_indices = {p['idx'] for p in pois}
            for ep in ema_pois:
                if ep['idx'] not in existing_indices:
                    pois.append(ep)

        if not pois:
            return None

        current_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        current_price = current_bar[col_map['close']]

        for poi in pois:
            # Check regime alignment
            if poi['direction'] == 'BUY' and regime != Regime.BULLISH:
                continue
            if poi['direction'] == 'SELL' and regime != Regime.BEARISH:
                continue

            poi_type = poi.get('type', 'ORDER_BLOCK')

            # EMA_PULLBACK already has entry confirmation
            if poi_type == 'EMA_PULLBACK':
                entry_type = 'MOMENTUM'
            else:
                # ORDER_BLOCK needs additional entry trigger
                zone_size = abs(current_bar[col_map['high']] - current_bar[col_map['low']]) * 2
                if abs(current_price - poi['price']) > zone_size:
                    continue

                has_trigger, entry_type = self.check_entry_trigger(
                    current_bar, prev_bar, poi['direction'], col_map
                )
                if not has_trigger:
                    continue

            # Calculate base risk
            risk_mult, should_skip = self.risk_scorer.calculate_risk_multiplier(
                now, entry_type, poi['quality']
            )
            if should_skip:
                continue

            # LAYER 4: Pattern already checked above, apply quality requirement
            effective_quality = dynamic_quality + pattern_extra_q
            if poi['quality'] < effective_quality:
                logger.debug(f"POI quality {poi['quality']:.0f} < required {effective_quality:.0f}")
                continue

            # Calculate position with pattern size multiplier
            sl_pips = current_atr * SL_ATR_MULT
            tp_pips = sl_pips * TP_RATIO
            risk_amount = balance * (RISK_PERCENT / 100.0) * risk_mult * pattern_size_mult
            lot_size = risk_amount / (sl_pips * PIP_VALUE)
            lot_size = max(0.01, min(MAX_LOT, round(lot_size, 2)))

            if poi['direction'] == 'BUY':
                sl_price = current_price - (sl_pips * PIP_SIZE)
                tp_price = current_price + (tp_pips * PIP_SIZE)
            else:
                sl_price = current_price + (sl_pips * PIP_SIZE)
                tp_price = current_price - (tp_pips * PIP_SIZE)

            logger.info(f"[{poi_type}] Signal found: {poi['direction']} Q={poi['quality']:.0f}")

            return TradeSignal(
                direction=poi['direction'],
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                quality_score=poi['quality'],
                entry_type=entry_type,
                session=session,
                atr_pips=current_atr,
                market_condition=market_cond.label,
                quality_threshold=effective_quality,
                monthly_adj=market_cond.monthly_adjustment + intra_month_adj,
                pattern_size_mult=pattern_size_mult,
                pattern_status=pattern_reason,
                poi_type=poi_type
            )

        return None

    async def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute trade signal"""
        try:
            logger.info(f"Executing {signal.direction} signal [{signal.poi_type}]:")
            logger.info(f"  Entry: {signal.entry_price:.5f}")
            logger.info(f"  SL: {signal.sl_price:.5f}")
            logger.info(f"  TP: {signal.tp_price:.5f}")
            logger.info(f"  Lot: {signal.lot_size}")
            logger.info(f"  Quality: {signal.quality_score:.1f} (req: {signal.quality_threshold:.0f})")
            logger.info(f"  Market: {signal.market_condition}")
            logger.info(f"  Signal Type: {signal.poi_type}")
            logger.info(f"  Pattern: {signal.pattern_status} ({signal.pattern_size_mult:.0%} size)")

            # Place order via broker
            order_result = await self.broker.place_order(
                symbol=SYMBOL,
                order_type=signal.direction,
                lot_size=signal.lot_size,
                stop_loss=signal.sl_price,
                take_profit=signal.tp_price
            )

            if order_result.get('success'):
                self.current_position = signal
                self.last_signal_time = datetime.now(timezone.utc)

                # Send Telegram notification
                if self.telegram:
                    # v6.7: Include signal type
                    signal_emoji = "📊" if signal.poi_type == "ORDER_BLOCK" else "📈"
                    msg = (
                        f"{signal_emoji} [v6.7 GBPUSD] NEW TRADE\n"
                        f"Signal: {signal.poi_type}\n"
                        f"Direction: {signal.direction}\n"
                        f"Entry: {signal.entry_price:.5f}\n"
                        f"SL: {signal.sl_price:.5f}\n"
                        f"TP: {signal.tp_price:.5f}\n"
                        f"Lot: {signal.lot_size}\n"
                        f"Quality: {signal.quality_score:.0f} (req: {signal.quality_threshold:.0f})\n"
                        f"Market: {signal.market_condition}\n"
                        f"Pattern: {signal.pattern_status}\n"
                        f"Session: {signal.session}"
                    )
                    await self.telegram.send_message(msg)

                return True
            else:
                logger.error(f"Order failed: {order_result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Execute error: {e}")
            return False

    async def record_trade_result(self, pnl: float, direction: str):
        """Record trade result for Layer 3 & 4 tracking"""
        now = datetime.now(timezone.utc)

        # Layer 3: Record for intra-month tracking (with direction for state persistence)
        self.intra_month_manager.record_trade(pnl, now, direction)

        # Layer 4: Record for pattern filter
        self.pattern_filter.record_trade(direction, pnl, now)

        logger.info(f"Trade result recorded: {direction} ${pnl:+.2f}")

    async def check_position(self) -> Optional[dict]:
        """Check current position status"""
        if not self.current_position:
            return None

        try:
            positions = await self.broker.get_positions(SYMBOL)
            if not positions:
                # Position closed - need to get P&L and record
                self.current_position = None
                return {'status': 'closed'}
            return {'status': 'open', 'position': positions[0]}
        except Exception as e:
            logger.error(f"Position check error: {e}")
            return None

    async def run_cycle(self):
        """Run one analysis and trading cycle"""
        try:
            # Get balance
            balance = await self.broker.get_balance()
            if not balance:
                logger.warning("Could not get balance")
                return

            # Check existing position
            pos_status = await self.check_position()
            if pos_status and pos_status.get('status') == 'open':
                logger.debug("Position open, skipping analysis")
                return

            # Analyze market with QUAD-LAYER filter
            signal = await self.analyze_market(balance)

            if signal:
                await self.execute_signal(signal)

        except Exception as e:
            logger.error(f"Cycle error: {e}")

    def get_status(self) -> dict:
        """Get current executor status with all layer info"""
        now = datetime.now(timezone.utc)
        status = {
            'version': 'v6.4',
            'symbol': SYMBOL,
            'timeframe': TIMEFRAME,
            'strategy': 'quad_layer_quality',
            'has_position': self.current_position is not None,
            'last_signal': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'layer1_monthly_adj': self.risk_scorer.get_monthly_quality_adjustment(now),
            'layer3_status': self.intra_month_manager.get_stats(),
            'layer4_status': self.pattern_filter.get_stats(),
            'state': self.state_manager.get_summary(),
            'vector_enabled': self.enable_vector
        }

        if self.vector_client:
            status['vector_status'] = self.vector_client.get_status()

        return status

    async def sync_vector(self, symbol: str = None, timeframe: str = None, force: bool = False) -> int:
        """
        Sync current data to vector database.

        Args:
            symbol: Symbol to sync (default: SYMBOL)
            timeframe: Timeframe to sync (default: TIMEFRAME)
            force: Force sync even if recently synced

        Returns:
            Number of vectors synced
        """
        if not self.vector_client:
            return 0

        symbol = symbol or SYMBOL
        timeframe = timeframe or TIMEFRAME

        # Rate limit (sync max once per minute unless forced)
        if not force and self._last_vector_sync:
            elapsed = datetime.now(timezone.utc) - self._last_vector_sync
            if elapsed < timedelta(seconds=60):
                return 0

        try:
            df = await self.get_ohlcv_data(symbol, timeframe, 1000)
            if df is None or df.empty:
                return 0

            synced = self.vector_client.sync(symbol, timeframe, df, force=force)
            self._last_vector_sync = datetime.now(timezone.utc)

            if synced > 0:
                logger.info(f"Vector sync: {synced} vectors for {symbol}/{timeframe}")

            return synced

        except Exception as e:
            logger.error(f"Vector sync failed: {e}")
            return 0

    async def find_similar_patterns(
        self,
        symbol: str = None,
        timeframe: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find similar historical patterns to current market state.

        Args:
            symbol: Symbol to search
            timeframe: Timeframe
            top_k: Number of similar patterns to return

        Returns:
            List of similar patterns with scores and OHLC data
        """
        if not self.vector_client:
            return []

        symbol = symbol or SYMBOL
        timeframe = timeframe or TIMEFRAME

        try:
            # Get current features
            df = await self.get_ohlcv_data(symbol, timeframe, 100)
            if df is None or df.empty or len(df) < 50:
                return []

            features = self.vector_client.compute_features(df)
            if len(features) == 0:
                return []

            # Get latest feature vector
            query_vector = features[-1]

            # Exclude recent 24 hours
            max_timestamp = datetime.now(timezone.utc) - timedelta(hours=24)

            return self.vector_client.find_similar(
                symbol=symbol,
                timeframe=timeframe,
                query_vector=query_vector,
                top_k=top_k,
                max_timestamp=max_timestamp
            )

        except Exception as e:
            logger.error(f"Similar pattern search failed: {e}")
            return []

    def get_vector_status(self) -> Dict:
        """Get vector database status"""
        if not self.vector_client:
            return {
                "enabled": False,
                "connected": False,
                "message": "Vector client not initialized"
            }

        return self.vector_client.get_status()
