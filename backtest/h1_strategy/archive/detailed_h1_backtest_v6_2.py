"""
SURGE-WSI H1 v6.2 Backtest - Ultra High PF Version
===================================================
Target:
- Balance: $50,000
- Risk: 1% per trade (auto lot)
- PF > 2.0
- Fewer but higher quality trades

v6.1 Results: 225T, 35.6% WR, PF 1.21, 5 losing months
v6.2 Changes:
1. Higher R:R = 1:2.5 (62.5 pips TP vs 25 SL)
2. Stricter entry requirements (score >= 0.7)
3. Regime must align strongly
4. POI quality >= 0.6
5. Skip weak sessions entirely
6. Auto-skip after 2 consecutive losses

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
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from config import config
from src.data.db_handler import DBHandler

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION v6.2 - ULTRA HIGH PF
# ============================================================
class ConfigV62:
    # Account
    INITIAL_BALANCE = 50_000  # $50K
    RISK_PERCENT = 1.0  # 1% risk per trade

    # Strategy - Higher R:R for PF > 2
    SL_PIPS = 25  # Stop loss in pips
    TP_RATIO = 2.5  # Risk:Reward = 1:2.5
    TP_PIPS = SL_PIPS * TP_RATIO  # 62.5 pips TP

    # Quality Thresholds - STRICTER
    MIN_POI_QUALITY = 0.60  # Higher quality POIs only
    MIN_ENTRY_SCORE = 0.70  # Need strong confirmation
    MIN_REGIME_PROB = 0.65  # Strong regime only

    # Kill Zones - FOCUSED (only best sessions)
    KILL_ZONES = {
        'london_main': (8, 11),    # London Core
        'ny_open': (13, 16),       # NY Open
    }

    # Session Quality Multipliers
    SESSION_MULTIPLIERS = {
        'london_main': 1.0,     # Full size
        'ny_open': 1.0,         # Full size
    }

    # Day Multipliers - STRICTER
    DAY_MULTIPLIERS = {
        0: 0.7,   # Monday - cautious start
        1: 1.0,   # Tuesday - full
        2: 1.0,   # Wednesday - full
        3: 0.8,   # Thursday - slightly reduced
        4: 0.5,   # Friday - half size
        5: 0.0,   # Saturday - no trade
        6: 0.0,   # Sunday - no trade
    }

    # Known Bad Periods - MORE AGGRESSIVE REDUCTION
    BAD_PERIODS = {
        3: 0.5,   # March - choppy
        4: 0.5,   # April - choppy
        6: 0.2,   # June - very reduced
        9: 0.3,   # September - reduced
        10: 0.5,  # October - choppy
    }

    # Auto Risk Adjuster
    MAX_CONSECUTIVE_LOSSES = 2  # Skip after 2 losses
    LOSS_REDUCTION_FACTOR = 0.3  # Heavy reduction

    # Regime Settings
    REGIME_LOOKBACK = 50


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    risk_amount: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""
    quality_score: float = 0.0
    session: str = ""
    regime: str = ""
    trigger_type: str = ""


@dataclass
class POI:
    price: float
    direction: str
    strength: float
    timestamp: datetime
    impulse_ratio: float
    wick_ratio: float
    freshness: int
    quality: float


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


# ============================================================
# DATA FETCHING
# ============================================================
async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data from database"""
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


# ============================================================
# INDICATORS
# ============================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = calculate_ema(series, 12)
    ema26 = calculate_ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(df: pd.DataFrame, period: int = 14, col_map: dict = None) -> pd.Series:
    if col_map is None:
        col_map = {'high': 'high', 'low': 'low', 'close': 'close'}

    high = df[col_map['high']]
    low = df[col_map['low']]
    close = df[col_map['close']]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


# ============================================================
# REGIME DETECTOR (Enhanced v6.2)
# ============================================================
class RegimeDetectorV62:
    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float, str]:
        """Detect market regime with higher confidence requirement"""
        if len(df) < self.lookback:
            return Regime.SIDEWAYS, 0.3, "neutral"

        close_col = col_map['close']
        high_col = col_map['high']
        low_col = col_map['low']

        # Calculate indicators
        ema20 = calculate_ema(df[close_col], 20)
        ema50 = calculate_ema(df[close_col], 50)
        ema100 = calculate_ema(df[close_col], 100)
        rsi = calculate_rsi(df[close_col])
        macd_line, signal_line, histogram = calculate_macd(df[close_col])

        # Current values
        current_close = df[close_col].iloc[-1]
        current_ema20 = ema20.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_ema100 = ema100.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        # Scoring (more factors)
        bullish_score = 0
        bearish_score = 0

        # 1. Triple EMA alignment (strong signal)
        if current_close > current_ema20 > current_ema50 > current_ema100:
            bullish_score += 3
        elif current_close < current_ema20 < current_ema50 < current_ema100:
            bearish_score += 3

        # 2. RSI momentum
        if current_rsi > 55:
            bullish_score += 1
        elif current_rsi < 45:
            bearish_score += 1

        # 3. MACD
        if current_macd > current_signal and current_macd > 0:
            bullish_score += 1.5
        elif current_macd < current_signal and current_macd < 0:
            bearish_score += 1.5

        # 4. MACD histogram momentum
        if current_hist > prev_hist > 0:
            bullish_score += 1
        elif current_hist < prev_hist < 0:
            bearish_score += 1

        # 5. Price structure (HH/HL or LH/LL)
        recent = df.tail(20)
        swing_highs = recent[high_col].rolling(5).max()
        swing_lows = recent[low_col].rolling(5).min()

        if swing_highs.iloc[-1] > swing_highs.iloc[-10]:
            bullish_score += 0.5
        if swing_lows.iloc[-1] < swing_lows.iloc[-10]:
            bearish_score += 0.5

        # Determine regime
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return Regime.SIDEWAYS, 0.3, "neutral"

        if bullish_score > bearish_score * 1.5:  # Need 1.5x for strong regime
            prob = min(0.9, bullish_score / (total_score + 2))
            return Regime.BULLISH, prob, "bullish"
        elif bearish_score > bullish_score * 1.5:
            prob = min(0.9, bearish_score / (total_score + 2))
            return Regime.BEARISH, prob, "bearish"
        else:
            return Regime.SIDEWAYS, 0.4, "neutral"


# ============================================================
# POI DETECTOR (v6.2 - Higher Quality)
# ============================================================
class POIDetectorV62:
    def __init__(self, min_quality: float = 0.60):
        self.min_quality = min_quality

    def detect_order_blocks(self, df: pd.DataFrame, col_map: dict, lookback: int = 30) -> List[POI]:
        """Detect high-quality order blocks only"""
        pois = []

        if len(df) < lookback + 5:
            return pois

        for i in range(len(df) - lookback, len(df) - 2):
            if i < 2:
                continue

            # Bullish Order Block
            if self._is_bullish_ob(df, i, col_map):
                poi = self._create_poi(df, i, 'BUY', col_map)
                if poi and poi.quality >= self.min_quality:
                    pois.append(poi)

            # Bearish Order Block
            if self._is_bearish_ob(df, i, col_map):
                poi = self._create_poi(df, i, 'SELL', col_map)
                if poi and poi.quality >= self.min_quality:
                    pois.append(poi)

        return pois

    def _is_bullish_ob(self, df: pd.DataFrame, i: int, col_map: dict) -> bool:
        """Stricter bullish OB detection"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        current = df.iloc[i]
        next1 = df.iloc[i+1]

        # Current candle is bearish
        is_bearish = current[c] < current[o]
        if not is_bearish:
            return False

        # Next candle is STRONGLY bullish
        next_body = abs(next1[c] - next1[o])
        next_range = next1[h] - next1[l]

        if next_range == 0:
            return False

        body_ratio = next_body / next_range

        # Requirements:
        # 1. Next is bullish
        # 2. Body > 60% of range (strong candle)
        # 3. Close above current high
        is_strong = (next1[c] > next1[o] and
                    body_ratio > 0.6 and
                    next1[c] > current[h])

        return is_strong

    def _is_bearish_ob(self, df: pd.DataFrame, i: int, col_map: dict) -> bool:
        """Stricter bearish OB detection"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        current = df.iloc[i]
        next1 = df.iloc[i+1]

        # Current candle is bullish
        is_bullish = current[c] > current[o]
        if not is_bullish:
            return False

        # Next candle is STRONGLY bearish
        next_body = abs(next1[c] - next1[o])
        next_range = next1[h] - next1[l]

        if next_range == 0:
            return False

        body_ratio = next_body / next_range

        # Requirements:
        # 1. Next is bearish
        # 2. Body > 60% of range (strong candle)
        # 3. Close below current low
        is_strong = (next1[c] < next1[o] and
                    body_ratio > 0.6 and
                    next1[c] < current[l])

        return is_strong

    def _create_poi(self, df: pd.DataFrame, i: int, direction: str, col_map: dict) -> Optional[POI]:
        """Create POI with strict quality metrics"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        candle = df.iloc[i]
        next_candle = df.iloc[i+1]

        # Impulse ratio (stronger = better)
        impulse_body = abs(next_candle[c] - next_candle[o])
        impulse_range = next_candle[h] - next_candle[l]
        impulse_ratio = impulse_body / impulse_range if impulse_range > 0 else 0

        # Wick ratio
        total = candle[h] - candle[l]
        if total == 0:
            return None

        if direction == 'BUY':
            wick = candle[o] - candle[l]
        else:
            wick = candle[h] - candle[o]

        wick_ratio = 1 - (wick / total)

        # Freshness
        freshness = len(df) - i
        freshness_score = max(0, 1 - (freshness / 40))

        # Volume spike check (if available)
        volume_bonus = 0
        if 'Volume' in df.columns or 'volume' in df.columns:
            vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
            avg_vol = df[vol_col].rolling(20).mean().iloc[i]
            if df.iloc[i][vol_col] > avg_vol * 1.3:
                volume_bonus = 0.1

        # Calculate quality (weighted)
        quality = (impulse_ratio * 0.45 +
                  wick_ratio * 0.25 +
                  freshness_score * 0.20 +
                  volume_bonus)

        # Price level
        price = candle[l] if direction == 'BUY' else candle[h]

        timestamp = df.index[i] if hasattr(df.index, '__getitem__') else datetime.now()

        return POI(
            price=price,
            direction=direction,
            strength=impulse_ratio,
            timestamp=timestamp,
            impulse_ratio=impulse_ratio,
            wick_ratio=wick_ratio,
            freshness=freshness,
            quality=quality
        )


# ============================================================
# ENTRY TRIGGER (v6.2 - Ultra Strict)
# ============================================================
class EntryTriggerV62:
    """Ultra-strict entry system for PF > 2.0"""

    def __init__(self, min_score: float = 0.70):
        self.min_score = min_score

    def check_entry(self, df: pd.DataFrame, poi: POI, regime: Regime,
                   regime_prob: float, col_map: dict) -> Tuple[bool, float, str]:
        """Check entry with multiple confirmations"""
        if len(df) < 5:
            return False, 0, ""

        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        score = 0
        triggers = []

        # 1. Price at POI zone (MUST be in zone)
        price = current[c]
        zone_size = abs(current[h] - current[l]) * 1.5  # Tighter zone

        in_zone = abs(price - poi.price) <= zone_size
        if not in_zone:
            return False, 0, ""

        # 2. Regime alignment (MUST align for trades)
        if poi.direction == 'BUY' and regime == Regime.BULLISH:
            score += 0.25
            triggers.append("REGIME")
        elif poi.direction == 'SELL' and regime == Regime.BEARISH:
            score += 0.25
            triggers.append("REGIME")
        else:
            # No regime alignment = skip
            return False, 0, ""

        # 3. Regime probability bonus
        if regime_prob >= 0.7:
            score += 0.1
            triggers.append("STRONG_REGIME")

        # 4. Rejection candle (pin bar)
        if self._is_rejection(current, poi.direction, col_map):
            score += 0.25
            triggers.append("REJECTION")

        # 5. Engulfing pattern
        if self._is_engulfing(prev, current, poi.direction, col_map):
            score += 0.25
            triggers.append("ENGULF")

        # 6. Momentum confirmation
        if self._has_momentum(df, poi.direction, col_map):
            score += 0.15
            triggers.append("MOMENTUM")

        # 7. Structure break
        if self._has_structure_break(df, poi.direction, col_map):
            score += 0.15
            triggers.append("STRUCTURE")

        # 8. Multiple confirmations bonus
        if len(triggers) >= 4:
            score += 0.15
        elif len(triggers) >= 3:
            score += 0.05

        should_enter = score >= self.min_score and len(triggers) >= 3
        trigger_type = "+".join(triggers) if triggers else ""

        return should_enter, score, trigger_type

    def _is_rejection(self, candle: pd.Series, direction: str, col_map: dict) -> bool:
        """Strict rejection candle detection"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        body = abs(candle[c] - candle[o])
        total = candle[h] - candle[l]

        if total == 0:
            return False

        if direction == 'BUY':
            lower_wick = min(candle[o], candle[c]) - candle[l]
            # Stricter: wick > 2x body and > 50% of range
            return lower_wick > body * 2 and lower_wick > total * 0.5
        else:
            upper_wick = candle[h] - max(candle[o], candle[c])
            return upper_wick > body * 2 and upper_wick > total * 0.5

    def _has_momentum(self, df: pd.DataFrame, direction: str, col_map: dict) -> bool:
        """Check momentum with RSI"""
        rsi = calculate_rsi(df[col_map['close']])
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if direction == 'BUY':
            # Rising RSI from oversold area
            return current_rsi > prev_rsi and 35 < current_rsi < 65
        else:
            # Falling RSI from overbought area
            return current_rsi < prev_rsi and 35 < current_rsi < 65

    def _is_engulfing(self, prev: pd.Series, current: pd.Series, direction: str, col_map: dict) -> bool:
        """Strict engulfing pattern"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        prev_body = abs(prev[c] - prev[o])
        curr_body = abs(current[c] - current[o])

        if direction == 'BUY':
            prev_bearish = prev[c] < prev[o]
            curr_bullish = current[c] > current[o]
            engulfs = current[c] > prev[o] and current[o] < prev[c]
            # Body must be at least 1.5x bigger
            return prev_bearish and curr_bullish and engulfs and curr_body > prev_body * 1.5
        else:
            prev_bullish = prev[c] > prev[o]
            curr_bearish = current[c] < current[o]
            engulfs = current[c] < prev[o] and current[o] > prev[c]
            return prev_bullish and curr_bearish and engulfs and curr_body > prev_body * 1.5

    def _has_structure_break(self, df: pd.DataFrame, direction: str, col_map: dict) -> bool:
        """Check for structure break"""
        h, l, c = col_map['high'], col_map['low'], col_map['close']

        recent = df.tail(15)
        current = df.iloc[-1]

        if direction == 'BUY':
            recent_high = recent[h].iloc[:-1].max()
            return current[c] > recent_high
        else:
            recent_low = recent[l].iloc[:-1].min()
            return current[c] < recent_low


# ============================================================
# RISK MANAGER v6.2
# ============================================================
class RiskManagerV62:
    def __init__(self, config: ConfigV62):
        self.config = config
        self.consecutive_losses = 0
        self.recent_trades: List[Trade] = []

    def should_skip(self) -> bool:
        """Check if should skip due to consecutive losses"""
        return self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES

    def calculate_lot_size(self, balance: float, sl_pips: float,
                          quality_score: float, session_mult: float,
                          day_mult: float, month_mult: float) -> Tuple[float, float]:
        """Calculate lot size with all factors"""
        risk_percent = self.config.RISK_PERCENT / 100
        risk_amount = balance * risk_percent

        # Apply multipliers
        total_mult = quality_score * session_mult * day_mult * month_mult

        # Loss reduction
        if self.consecutive_losses >= 1:
            total_mult *= (1 - self.consecutive_losses * 0.2)  # 20% reduction per loss

        total_mult = max(0.3, total_mult)  # Minimum 30%

        adjusted_risk = risk_amount * total_mult

        # XAUUSD: 1 pip = $10 per 0.1 lot
        pip_value = 10.0
        lot_size = adjusted_risk / (sl_pips * pip_value)
        lot_size = max(0.01, round(lot_size, 2))

        return lot_size, adjusted_risk

    def update_after_trade(self, trade: Trade):
        """Update state after trade"""
        self.recent_trades.append(trade)
        if len(self.recent_trades) > 20:
            self.recent_trades = self.recent_trades[-20:]

        if trade.pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0


# ============================================================
# SESSION MANAGER v6.2
# ============================================================
class SessionManagerV62:
    def __init__(self, config: ConfigV62):
        self.config = config

    def get_session(self, dt: datetime) -> Tuple[str, float]:
        """Get session with multiplier"""
        hour = dt.hour

        for session_name, (start, end) in self.config.KILL_ZONES.items():
            if start <= hour < end:
                mult = self.config.SESSION_MULTIPLIERS.get(session_name, 1.0)
                return session_name, mult

        return "off_session", 0.0

    def get_day_multiplier(self, dt: datetime) -> float:
        return self.config.DAY_MULTIPLIERS.get(dt.weekday(), 0.0)

    def get_month_multiplier(self, dt: datetime) -> float:
        return self.config.BAD_PERIODS.get(dt.month, 1.0)


# ============================================================
# BACKTEST ENGINE v6.2
# ============================================================
class BacktestEngineV62:
    def __init__(self, config: ConfigV62 = None):
        self.config = config or ConfigV62()

        # Components
        self.regime_detector = RegimeDetectorV62(self.config.REGIME_LOOKBACK)
        self.poi_detector = POIDetectorV62(self.config.MIN_POI_QUALITY)
        self.entry_trigger = EntryTriggerV62(self.config.MIN_ENTRY_SCORE)
        self.risk_manager = RiskManagerV62(self.config)
        self.session_manager = SessionManagerV62(self.config)

        # State
        self.balance = self.config.INITIAL_BALANCE
        self.trades: List[Trade] = []
        self.active_trade: Optional[Trade] = None
        self.equity_curve = []
        self.skipped_trades = 0

    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest"""
        print(f"\n{'='*60}")
        print(f"SURGE-WSI H1 v6.2 Backtest - ULTRA HIGH PF")
        print(f"{'='*60}")
        print(f"Initial Balance: ${self.config.INITIAL_BALANCE:,.0f}")
        print(f"Risk per Trade: {self.config.RISK_PERCENT}%")
        print(f"SL: {self.config.SL_PIPS} pips | TP: {self.config.TP_PIPS} pips (1:{self.config.TP_RATIO})")
        print(f"Target PF: > 2.0")
        print(f"{'='*60}\n")

        # Detect columns
        col_map = self._detect_columns(df)
        if not col_map:
            return {"error": "Cannot detect columns"}

        print(f"Column mapping: {col_map}")

        # Set time index
        time_col = col_map.get('time', 'time')
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)

        print(f"Data: {df.index[0]} to {df.index[-1]} ({len(df)} bars)")

        warmup = 100

        for i in range(warmup, len(df)):
            current_slice = df.iloc[:i+1]
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar[col_map['close']]

            # Equity tracking
            equity = self.balance
            if self.active_trade:
                if self.active_trade.direction == 'BUY':
                    unrealized = (current_price - self.active_trade.entry_price) * self.active_trade.lot_size * 100
                else:
                    unrealized = (self.active_trade.entry_price - current_price) * self.active_trade.lot_size * 100
                equity += unrealized
            self.equity_curve.append({'time': current_time, 'equity': equity})

            # Check active trade
            if self.active_trade:
                self._check_exit(current_bar, current_time, col_map)
                continue

            # Skip if consecutive losses
            if self.risk_manager.should_skip():
                self.skipped_trades += 1
                # Reset after 5 bars
                if self.skipped_trades > 5:
                    self.risk_manager.consecutive_losses = 0
                    self.skipped_trades = 0
                continue

            # Check session
            session, session_mult = self.session_manager.get_session(current_time)
            if session_mult == 0:
                continue

            # Get multipliers
            day_mult = self.session_manager.get_day_multiplier(current_time)
            month_mult = self.session_manager.get_month_multiplier(current_time)

            if day_mult == 0:
                continue

            # Combined multiplier check
            combined_mult = session_mult * day_mult * month_mult
            if combined_mult < 0.4:  # Skip if total multiplier too low
                continue

            # Detect regime
            regime, regime_prob, bias = self.regime_detector.detect(current_slice, col_map)

            # Skip sideways or weak regime
            if regime == Regime.SIDEWAYS or regime_prob < self.config.MIN_REGIME_PROB:
                continue

            # Detect POIs
            pois = self.poi_detector.detect_order_blocks(current_slice, col_map)

            if not pois:
                continue

            # Check entry for each POI
            for poi in pois:
                # Regime alignment check
                if poi.direction == 'BUY' and regime != Regime.BULLISH:
                    continue
                if poi.direction == 'SELL' and regime != Regime.BEARISH:
                    continue

                # Check entry
                should_enter, entry_score, trigger_type = self.entry_trigger.check_entry(
                    current_slice, poi, regime, regime_prob, col_map
                )

                if not should_enter:
                    continue

                # Calculate lot
                lot_size, risk_amount = self.risk_manager.calculate_lot_size(
                    self.balance,
                    self.config.SL_PIPS,
                    entry_score,
                    session_mult,
                    day_mult,
                    month_mult
                )

                # SL/TP
                if poi.direction == 'BUY':
                    sl_price = current_price - (self.config.SL_PIPS * 0.1)
                    tp_price = current_price + (self.config.TP_PIPS * 0.1)
                else:
                    sl_price = current_price + (self.config.SL_PIPS * 0.1)
                    tp_price = current_price - (self.config.TP_PIPS * 0.1)

                # Create trade
                self.active_trade = Trade(
                    entry_time=current_time,
                    direction=poi.direction,
                    entry_price=current_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    lot_size=lot_size,
                    risk_amount=risk_amount,
                    quality_score=entry_score,
                    session=session,
                    regime=regime.value,
                    trigger_type=trigger_type
                )

                self.skipped_trades = 0
                break

        return self._calculate_results(df)

    def _detect_columns(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect OHLCV columns"""
        col_map = {}

        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower and 'open' not in col_map:
                col_map['open'] = col
            elif 'high' in col_lower and 'high' not in col_map:
                col_map['high'] = col
            elif 'low' in col_lower and 'low' not in col_map:
                col_map['low'] = col
            elif 'close' in col_lower and 'close' not in col_map:
                col_map['close'] = col
            elif 'time' in col_lower and 'time' not in col_map:
                col_map['time'] = col

        required = ['open', 'high', 'low', 'close']
        if all(k in col_map for k in required):
            return col_map
        return None

    def _check_exit(self, bar: pd.Series, current_time: datetime, col_map: dict):
        """Check trade exit"""
        if not self.active_trade:
            return

        trade = self.active_trade
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        exit_price = None
        exit_reason = ""

        if trade.direction == 'BUY':
            if low <= trade.sl_price:
                exit_price = trade.sl_price
                exit_reason = "SL"
            elif high >= trade.tp_price:
                exit_price = trade.tp_price
                exit_reason = "TP"
        else:
            if high >= trade.sl_price:
                exit_price = trade.sl_price
                exit_reason = "SL"
            elif low <= trade.tp_price:
                exit_price = trade.tp_price
                exit_reason = "TP"

        if exit_price:
            if trade.direction == 'BUY':
                pips = (exit_price - trade.entry_price) / 0.1
            else:
                pips = (trade.entry_price - exit_price) / 0.1

            pnl = pips * trade.lot_size * 10

            trade.exit_time = current_time
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.exit_reason = exit_reason

            self.balance += pnl
            self.trades.append(trade)
            self.risk_manager.update_after_trade(trade)
            self.active_trade = None

    def _calculate_results(self, df: pd.DataFrame) -> Dict:
        """Calculate results"""
        if not self.trades:
            return {"error": "No trades executed"}

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        net_pnl = gross_profit - gross_loss

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        avg_trade = net_pnl / total_trades if total_trades > 0 else 0

        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['peak'] - equity_df['equity']
        max_dd = equity_df['drawdown'].max()
        max_dd_pct = (max_dd / self.config.INITIAL_BALANCE) * 100

        # Monthly
        trade_df = pd.DataFrame([{
            'time': t.entry_time,
            'pnl': t.pnl,
        } for t in self.trades])

        trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
        monthly = trade_df.groupby('month')['pnl'].sum()

        losing_months = (monthly < 0).sum()
        total_months = len(monthly)

        total_return = (net_pnl / self.config.INITIAL_BALANCE) * 100

        first_trade = self.trades[0].entry_time
        last_trade = self.trades[-1].entry_time
        days = (last_trade - first_trade).days or 1
        trades_per_day = total_trades / days

        results = {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_pnl': net_pnl,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_drawdown': max_dd,
            'max_dd_pct': max_dd_pct,
            'total_return': total_return,
            'losing_months': losing_months,
            'total_months': total_months,
            'trades_per_day': trades_per_day,
            'final_balance': self.balance,
            'monthly': monthly
        }

        self._print_results(results)
        return results

    def _print_results(self, results: Dict):
        """Print results"""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS - H1 v6.2")
        print(f"{'='*60}")

        print(f"\n[PERFORMANCE]")
        print(f"{'-'*40}")
        print(f"Total Trades:      {results['total_trades']}")
        print(f"Trades/Day:        {results['trades_per_day']:.2f}")
        print(f"Win Rate:          {results['win_rate']:.1f}%")
        print(f"Profit Factor:     {results['profit_factor']:.2f}")

        print(f"\n[PROFIT/LOSS]")
        print(f"{'-'*40}")
        print(f"Gross Profit:      ${results['gross_profit']:,.2f}")
        print(f"Gross Loss:        ${results['gross_loss']:,.2f}")
        print(f"Net P/L:           ${results['net_pnl']:+,.2f}")
        print(f"Total Return:      {results['total_return']:+.1f}%")
        print(f"Final Balance:     ${results['final_balance']:,.2f}")

        print(f"\n[RISK]")
        print(f"{'-'*40}")
        print(f"Max Drawdown:      ${results['max_drawdown']:,.2f} ({results['max_dd_pct']:.1f}%)")
        print(f"Avg Win:           ${results['avg_win']:,.2f}")
        print(f"Avg Loss:          ${results['avg_loss']:,.2f}")

        print(f"\n[MONTHLY]")
        print(f"{'-'*40}")
        print(f"Losing Months:     {results['losing_months']}/{results['total_months']}")

        for month, pnl in results['monthly'].items():
            status = "WIN " if pnl >= 0 else "LOSS"
            print(f"  [{status}] {month}: ${pnl:+,.2f}")

        print(f"\n{'='*60}")
        if results['profit_factor'] >= 2.0:
            print(f"[OK] TARGET MET: PF {results['profit_factor']:.2f} >= 2.0")
        else:
            gap = 2.0 - results['profit_factor']
            print(f"[X] TARGET NOT MET: PF {results['profit_factor']:.2f} (need +{gap:.2f})")
        print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================
async def main():
    symbol = "XAUUSD"
    timeframe = "H1"
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 30, tzinfo=timezone.utc)

    print(f"Fetching {symbol} {timeframe} data...")

    df = await fetch_data(symbol, timeframe, start, end)

    if df.empty:
        print("Error: No data")
        return

    print(f"Fetched {len(df)} bars")

    config = ConfigV62()
    engine = BacktestEngineV62(config)
    results = engine.run(df)

    if results and 'error' not in results:
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'lot_size': t.lot_size,
            'pnl': t.pnl,
            'exit_reason': t.exit_reason,
            'quality_score': t.quality_score,
            'session': t.session,
            'trigger_type': t.trigger_type
        } for t in engine.trades])

        output_path = Path(__file__).parent.parent / "results" / "h1_v6_2_trades.csv"
        trades_df.to_csv(output_path, index=False)
        print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
