"""
SURGE-WSI H1 v6.1 Backtest
==========================
Target:
- Balance: $50,000
- Risk: 1% per trade (auto lot)
- PF > 2.0
- More trades than v5.8

Strategy Enhancements:
1. Extended Kill Zones with quality gates
2. Stricter POI quality (for higher PF)
3. Better R:R ratio (1:2)
4. Multi-confirmation entry
5. Auto lot sizing

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
# CONFIGURATION v6.1
# ============================================================
class ConfigV61:
    # Account
    INITIAL_BALANCE = 50_000  # $50K
    RISK_PERCENT = 1.0  # 1% risk per trade

    # Strategy
    SL_PIPS = 25  # Stop loss in pips
    TP_RATIO = 2.0  # Risk:Reward = 1:2 (for higher PF)
    TP_PIPS = SL_PIPS * TP_RATIO  # 50 pips TP

    # Quality Thresholds (stricter for PF > 2)
    MIN_POI_QUALITY = 0.55  # Higher quality POIs
    MIN_IMPULSE_RATIO = 0.65  # Stronger impulse
    MIN_ENTRY_SCORE = 0.6  # Better entries

    # Kill Zones (Extended for more trades)
    KILL_ZONES = {
        'asian_late': (5, 7),      # Late Asian (limited)
        'london_early': (7, 9),    # London Open
        'london_main': (9, 12),    # London Main
        'ny_open': (13, 16),       # NY Open
        'ny_main': (16, 19),       # NY Main
    }

    # Session Quality Multipliers
    SESSION_MULTIPLIERS = {
        'asian_late': 0.5,      # Half size in Asian
        'london_early': 0.8,    # Slightly reduced
        'london_main': 1.0,     # Full size
        'ny_open': 1.0,         # Full size
        'ny_main': 0.8,         # Slightly reduced
    }

    # Day Multipliers (based on historical performance)
    DAY_MULTIPLIERS = {
        0: 0.8,   # Monday - cautious
        1: 1.0,   # Tuesday - full
        2: 1.0,   # Wednesday - full
        3: 1.0,   # Thursday - full
        4: 0.7,   # Friday - reduced
        5: 0.0,   # Saturday - no trade
        6: 0.0,   # Sunday - no trade
    }

    # Known Bad Periods
    BAD_PERIODS = {
        6: 0.3,   # June - reduced
        9: 0.4,   # September - reduced
    }

    # Auto Risk Adjuster Thresholds
    MAX_CONSECUTIVE_LOSSES = 3
    LOSS_REDUCTION_FACTOR = 0.5

    # Regime Settings
    REGIME_LOOKBACK = 50
    MIN_TREND_STRENGTH = 0.6


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Trade:
    entry_time: datetime
    direction: str  # 'BUY' or 'SELL'
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


# ============================================================
# REGIME DETECTOR (Enhanced)
# ============================================================
class RegimeDetectorV61:
    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame, col_map: dict) -> Tuple[Regime, float, str]:
        """Detect market regime with probability and bias"""
        if len(df) < self.lookback:
            return Regime.SIDEWAYS, 0.5, "neutral"

        close_col = col_map['close']
        high_col = col_map['high']
        low_col = col_map['low']

        # Calculate trend indicators
        ema20 = calculate_ema(df[close_col], 20)
        ema50 = calculate_ema(df[close_col], 50)
        rsi = calculate_rsi(df[close_col])
        macd_line, signal_line, histogram = calculate_macd(df[close_col])

        # Current values
        current_close = df[close_col].iloc[-1]
        current_ema20 = ema20.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]

        # Scoring
        bullish_score = 0
        bearish_score = 0

        # EMA alignment
        if current_close > current_ema20 > current_ema50:
            bullish_score += 2
        elif current_close < current_ema20 < current_ema50:
            bearish_score += 2

        # RSI
        if current_rsi > 55:
            bullish_score += 1
        elif current_rsi < 45:
            bearish_score += 1

        # MACD
        if current_macd > current_signal and current_macd > 0:
            bullish_score += 1.5
        elif current_macd < current_signal and current_macd < 0:
            bearish_score += 1.5

        # Price action (higher highs/lows)
        recent = df.tail(self.lookback)
        recent_highs = recent[high_col].values[-10:]
        recent_lows = recent[low_col].values[-10:]

        if recent_highs[-1] > np.mean(recent_highs[:-1]):
            bullish_score += 0.5
        if recent_lows[-1] < np.mean(recent_lows[:-1]):
            bearish_score += 0.5

        # Determine regime
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return Regime.SIDEWAYS, 0.5, "neutral"

        if bullish_score > bearish_score * 1.3:
            prob = bullish_score / (total_score + 1)
            return Regime.BULLISH, min(prob, 0.9), "bullish"
        elif bearish_score > bullish_score * 1.3:
            prob = bearish_score / (total_score + 1)
            return Regime.BEARISH, min(prob, 0.9), "bearish"
        else:
            return Regime.SIDEWAYS, 0.5, "neutral"


# ============================================================
# POI DETECTOR (Enhanced for v6.1)
# ============================================================
class POIDetectorV61:
    def __init__(self, min_quality: float = 0.55):
        self.min_quality = min_quality

    def detect_order_blocks(self, df: pd.DataFrame, col_map: dict, lookback: int = 30) -> List[POI]:
        """Detect order blocks with quality scoring"""
        pois = []

        if len(df) < lookback + 5:
            return pois

        for i in range(len(df) - lookback, len(df) - 2):
            if i < 2:
                continue

            # Bullish Order Block (last down candle before up move)
            if self._is_bullish_ob(df, i, col_map):
                poi = self._create_poi(df, i, 'BUY', col_map)
                if poi and poi.quality >= self.min_quality:
                    pois.append(poi)

            # Bearish Order Block (last up candle before down move)
            if self._is_bearish_ob(df, i, col_map):
                poi = self._create_poi(df, i, 'SELL', col_map)
                if poi and poi.quality >= self.min_quality:
                    pois.append(poi)

        return pois

    def _is_bullish_ob(self, df: pd.DataFrame, i: int, col_map: dict) -> bool:
        """Check if candle at index i is a bullish order block"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        current = df.iloc[i]
        next1 = df.iloc[i+1]

        # Current candle is bearish
        is_bearish = current[c] < current[o]

        # Next candle is strongly bullish
        next_body = abs(next1[c] - next1[o])
        next_range = next1[h] - next1[l]
        is_strong_bullish = (next1[c] > next1[o] and
                           next_range > 0 and
                           next_body > next_range * 0.5 and
                           next1[c] > current[h])

        return is_bearish and is_strong_bullish

    def _is_bearish_ob(self, df: pd.DataFrame, i: int, col_map: dict) -> bool:
        """Check if candle at index i is a bearish order block"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        current = df.iloc[i]
        next1 = df.iloc[i+1]

        # Current candle is bullish
        is_bullish = current[c] > current[o]

        # Next candle is strongly bearish
        next_body = abs(next1[c] - next1[o])
        next_range = next1[h] - next1[l]
        is_strong_bearish = (next1[c] < next1[o] and
                           next_range > 0 and
                           next_body > next_range * 0.5 and
                           next1[c] < current[l])

        return is_bullish and is_strong_bearish

    def _create_poi(self, df: pd.DataFrame, i: int, direction: str, col_map: dict) -> Optional[POI]:
        """Create POI with quality metrics"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        candle = df.iloc[i]
        next_candle = df.iloc[i+1]

        # Calculate impulse ratio
        impulse_body = abs(next_candle[c] - next_candle[o])
        impulse_range = next_candle[h] - next_candle[l]
        impulse_ratio = impulse_body / impulse_range if impulse_range > 0 else 0

        # Calculate wick ratio (smaller wick = better)
        if direction == 'BUY':
            wick = candle[o] - candle[l]
            body = candle[o] - candle[c]
        else:
            wick = candle[h] - candle[o]
            body = candle[c] - candle[o]

        total = candle[h] - candle[l]
        wick_ratio = 1 - (wick / total) if total > 0 else 0

        # Freshness (how recent)
        freshness = len(df) - i
        freshness_score = max(0, 1 - (freshness / 50))

        # Calculate quality
        quality = (impulse_ratio * 0.4 + wick_ratio * 0.3 + freshness_score * 0.3)

        # Price level
        if direction == 'BUY':
            price = candle[l]
        else:
            price = candle[h]

        timestamp = df.index[i] if hasattr(df.index[i], 'timestamp') else df.iloc[i].get('time', datetime.now())

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
# ENTRY TRIGGER (Multi-confirmation)
# ============================================================
class EntryTriggerV61:
    """Multi-confirmation entry system for higher PF"""

    def __init__(self, min_score: float = 0.6):
        self.min_score = min_score

    def check_entry(self, df: pd.DataFrame, poi: POI, regime: Regime, col_map: dict) -> Tuple[bool, float, str]:
        """Check if entry conditions are met."""
        if len(df) < 5:
            return False, 0, ""

        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        current = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0
        triggers = []

        # 1. Price at POI zone
        price = current[c]
        zone_size = abs(current[h] - current[l]) * 2

        if poi.direction == 'BUY':
            in_zone = poi.price - zone_size <= price <= poi.price + zone_size
        else:
            in_zone = poi.price - zone_size <= price <= poi.price + zone_size

        if not in_zone:
            return False, 0, ""

        # 2. Regime alignment
        if poi.direction == 'BUY' and regime == Regime.BULLISH:
            score += 0.2
            triggers.append("REGIME")
        elif poi.direction == 'SELL' and regime == Regime.BEARISH:
            score += 0.2
            triggers.append("REGIME")
        elif regime == Regime.SIDEWAYS:
            score += 0.1  # Partial credit

        # 3. Rejection candle
        if self._is_rejection(current, poi.direction, col_map):
            score += 0.25
            triggers.append("REJECTION")

        # 4. Momentum confirmation
        if self._has_momentum(df, poi.direction, col_map):
            score += 0.2
            triggers.append("MOMENTUM")

        # 5. Engulfing pattern
        if self._is_engulfing(prev, current, poi.direction, col_map):
            score += 0.25
            triggers.append("ENGULF")

        # 6. Structure break (HH/LL)
        if self._has_structure_break(df, poi.direction, col_map):
            score += 0.15
            triggers.append("STRUCTURE")

        # Bonus for multiple confirmations
        if len(triggers) >= 3:
            score += 0.1

        should_enter = score >= self.min_score
        trigger_type = "+".join(triggers) if triggers else ""

        return should_enter, score, trigger_type

    def _is_rejection(self, candle: pd.Series, direction: str, col_map: dict) -> bool:
        """Check for rejection candle (pin bar)"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        body = abs(candle[c] - candle[o])
        total = candle[h] - candle[l]

        if total == 0:
            return False

        if direction == 'BUY':
            lower_wick = min(candle[o], candle[c]) - candle[l]
            return lower_wick > body * 1.5 and lower_wick > total * 0.5
        else:
            upper_wick = candle[h] - max(candle[o], candle[c])
            return upper_wick > body * 1.5 and upper_wick > total * 0.5

    def _has_momentum(self, df: pd.DataFrame, direction: str, col_map: dict) -> bool:
        """Check momentum in direction"""
        rsi = calculate_rsi(df[col_map['close']])
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if direction == 'BUY':
            return current_rsi > prev_rsi and 40 < current_rsi < 70
        else:
            return current_rsi < prev_rsi and 30 < current_rsi < 60

    def _is_engulfing(self, prev: pd.Series, current: pd.Series, direction: str, col_map: dict) -> bool:
        """Check for engulfing pattern"""
        o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']

        prev_body = abs(prev[c] - prev[o])
        curr_body = abs(current[c] - current[o])

        if direction == 'BUY':
            prev_bearish = prev[c] < prev[o]
            curr_bullish = current[c] > current[o]
            engulfs = current[c] > prev[o] and current[o] < prev[c]
            return prev_bearish and curr_bullish and engulfs and curr_body > prev_body
        else:
            prev_bullish = prev[c] > prev[o]
            curr_bearish = current[c] < current[o]
            engulfs = current[c] < prev[o] and current[o] > prev[c]
            return prev_bullish and curr_bearish and engulfs and curr_body > prev_body

    def _has_structure_break(self, df: pd.DataFrame, direction: str, col_map: dict) -> bool:
        """Check for structure break"""
        h, l, c = col_map['high'], col_map['low'], col_map['close']

        recent = df.tail(10)
        current = df.iloc[-1]

        if direction == 'BUY':
            recent_high = recent[h].iloc[:-1].max()
            return current[c] > recent_high
        else:
            recent_low = recent[l].iloc[:-1].min()
            return current[c] < recent_low


# ============================================================
# RISK MANAGER (Auto Lot Sizing)
# ============================================================
class RiskManagerV61:
    def __init__(self, config: ConfigV61):
        self.config = config
        self.consecutive_losses = 0
        self.recent_trades: List[Trade] = []

    def calculate_lot_size(self, balance: float, sl_pips: float,
                          quality_score: float, session_mult: float,
                          day_mult: float, month_mult: float) -> Tuple[float, float]:
        """Calculate lot size based on all factors"""
        # Base risk
        risk_percent = self.config.RISK_PERCENT / 100
        risk_amount = balance * risk_percent

        # Apply multipliers
        total_mult = quality_score * session_mult * day_mult * month_mult

        # Loss reduction
        if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            total_mult *= self.config.LOSS_REDUCTION_FACTOR

        adjusted_risk = risk_amount * total_mult

        # Calculate lot
        # For XAUUSD: 1 pip = $10 per 0.1 lot = $1 per 0.01 lot
        pip_value = 10.0  # $10 per pip per 0.1 lot
        lot_size = adjusted_risk / (sl_pips * pip_value)

        # Round to 2 decimals, min 0.01
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
# SESSION MANAGER
# ============================================================
class SessionManagerV61:
    def __init__(self, config: ConfigV61):
        self.config = config

    def get_session(self, dt: datetime) -> Tuple[str, float]:
        """Get current session and its multiplier"""
        hour = dt.hour

        for session_name, (start, end) in self.config.KILL_ZONES.items():
            if start <= hour < end:
                mult = self.config.SESSION_MULTIPLIERS.get(session_name, 1.0)
                return session_name, mult

        return "off_session", 0.0

    def get_day_multiplier(self, dt: datetime) -> float:
        """Get day of week multiplier"""
        return self.config.DAY_MULTIPLIERS.get(dt.weekday(), 0.0)

    def get_month_multiplier(self, dt: datetime) -> float:
        """Get month multiplier for bad periods"""
        return self.config.BAD_PERIODS.get(dt.month, 1.0)


# ============================================================
# BACKTEST ENGINE v6.1
# ============================================================
class BacktestEngineV61:
    def __init__(self, config: ConfigV61 = None):
        self.config = config or ConfigV61()

        # Components
        self.regime_detector = RegimeDetectorV61(self.config.REGIME_LOOKBACK)
        self.poi_detector = POIDetectorV61(self.config.MIN_POI_QUALITY)
        self.entry_trigger = EntryTriggerV61(self.config.MIN_ENTRY_SCORE)
        self.risk_manager = RiskManagerV61(self.config)
        self.session_manager = SessionManagerV61(self.config)

        # State
        self.balance = self.config.INITIAL_BALANCE
        self.trades: List[Trade] = []
        self.active_trade: Optional[Trade] = None
        self.equity_curve = []

    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest on H1 data"""
        print(f"\n{'='*60}")
        print(f"SURGE-WSI H1 v6.1 Backtest")
        print(f"{'='*60}")
        print(f"Initial Balance: ${self.config.INITIAL_BALANCE:,.0f}")
        print(f"Risk per Trade: {self.config.RISK_PERCENT}%")
        print(f"SL: {self.config.SL_PIPS} pips | TP: {self.config.TP_PIPS} pips (1:{self.config.TP_RATIO})")
        print(f"Target PF: > 2.0")
        print(f"{'='*60}\n")

        # Detect column names
        col_map = self._detect_columns(df)
        if not col_map:
            return {"error": "Cannot detect OHLCV columns"}

        print(f"Column mapping: {col_map}")

        # Ensure time column
        time_col = col_map.get('time', 'time')
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)

        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        print(f"Total bars: {len(df)}")

        # Warmup period
        warmup = 100

        # Main loop
        for i in range(warmup, len(df)):
            current_slice = df.iloc[:i+1]
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar[col_map['close']]

            # Record equity
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

            # Check session
            session, session_mult = self.session_manager.get_session(current_time)
            if session_mult == 0:
                continue

            # Get multipliers
            day_mult = self.session_manager.get_day_multiplier(current_time)
            month_mult = self.session_manager.get_month_multiplier(current_time)

            if day_mult == 0:
                continue

            # Detect regime
            regime, regime_prob, bias = self.regime_detector.detect(current_slice, col_map)

            # Skip sideways with low confidence
            if regime == Regime.SIDEWAYS and regime_prob < 0.6:
                continue

            # Detect POIs
            pois = self.poi_detector.detect_order_blocks(current_slice, col_map)

            if not pois:
                continue

            # Check entry for each POI
            for poi in pois:
                # Regime alignment check
                if poi.direction == 'BUY' and regime == Regime.BEARISH:
                    continue
                if poi.direction == 'SELL' and regime == Regime.BULLISH:
                    continue

                # Check entry conditions
                should_enter, entry_score, trigger_type = self.entry_trigger.check_entry(
                    current_slice, poi, regime, col_map
                )

                if not should_enter:
                    continue

                # Calculate lot size
                lot_size, risk_amount = self.risk_manager.calculate_lot_size(
                    self.balance,
                    self.config.SL_PIPS,
                    entry_score,
                    session_mult,
                    day_mult,
                    month_mult
                )

                # Calculate SL/TP (XAUUSD: 1 pip = 0.1)
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
                    regime=regime.value
                )

                break  # Only one trade at a time

        # Calculate results
        return self._calculate_results(df)

    def _detect_columns(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect OHLCV column names"""
        col_map = {}

        # Common patterns
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

        # Check required columns
        required = ['open', 'high', 'low', 'close']
        if all(k in col_map for k in required):
            return col_map

        return None

    def _check_exit(self, bar: pd.Series, current_time: datetime, col_map: dict):
        """Check if trade should exit"""
        if not self.active_trade:
            return

        trade = self.active_trade
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        exit_price = None
        exit_reason = ""

        if trade.direction == 'BUY':
            # Check SL
            if low <= trade.sl_price:
                exit_price = trade.sl_price
                exit_reason = "SL"
            # Check TP
            elif high >= trade.tp_price:
                exit_price = trade.tp_price
                exit_reason = "TP"
        else:
            # Check SL
            if high >= trade.sl_price:
                exit_price = trade.sl_price
                exit_reason = "SL"
            # Check TP
            elif low <= trade.tp_price:
                exit_price = trade.tp_price
                exit_reason = "TP"

        if exit_price:
            # Calculate PnL (XAUUSD: 1 pip = 0.1, pip value = $10 per 0.1 lot)
            if trade.direction == 'BUY':
                pips = (exit_price - trade.entry_price) / 0.1
            else:
                pips = (trade.entry_price - exit_price) / 0.1

            # PnL = pips * lot * pip_value_per_lot
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
        """Calculate backtest results"""
        if not self.trades:
            return {"error": "No trades executed"}

        # Basic stats
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

        # Average trade
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        avg_trade = net_pnl / total_trades if total_trades > 0 else 0

        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['peak'] - equity_df['equity']
        max_dd = equity_df['drawdown'].max()
        max_dd_pct = (max_dd / self.config.INITIAL_BALANCE) * 100

        # Monthly analysis
        trade_df = pd.DataFrame([{
            'time': t.entry_time,
            'pnl': t.pnl,
            'direction': t.direction
        } for t in self.trades])

        trade_df['month'] = pd.to_datetime(trade_df['time']).dt.to_period('M')
        monthly = trade_df.groupby('month')['pnl'].sum()

        losing_months = (monthly < 0).sum()
        total_months = len(monthly)

        # Return
        total_return = (net_pnl / self.config.INITIAL_BALANCE) * 100

        # Trade frequency
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
        """Print backtest results"""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS - H1 v6.1")
        print(f"{'='*60}")

        print(f"\n[PERFORMANCE SUMMARY]")
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

        print(f"\n[RISK METRICS]")
        print(f"{'-'*40}")
        print(f"Max Drawdown:      ${results['max_drawdown']:,.2f} ({results['max_dd_pct']:.1f}%)")
        print(f"Avg Win:           ${results['avg_win']:,.2f}")
        print(f"Avg Loss:          ${results['avg_loss']:,.2f}")
        print(f"Avg Trade:         ${results['avg_trade']:,.2f}")

        print(f"\n[MONTHLY BREAKDOWN]")
        print(f"{'-'*40}")
        print(f"Losing Months:     {results['losing_months']}/{results['total_months']}")

        monthly = results['monthly']
        for month, pnl in monthly.items():
            status = "WIN" if pnl >= 0 else "LOSS"
            print(f"  [{status}] {month}: ${pnl:+,.2f}")

        # PF Check
        print(f"\n{'='*60}")
        if results['profit_factor'] >= 2.0:
            print(f"[OK] TARGET MET: PF {results['profit_factor']:.2f} >= 2.0")
        else:
            print(f"[X] TARGET NOT MET: PF {results['profit_factor']:.2f} < 2.0")
            print(f"    Need to adjust parameters...")
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
    print(f"Period: {start.date()} to {end.date()}")

    df = await fetch_data(symbol, timeframe, start, end)

    if df.empty:
        print("Error: No data fetched")
        return

    print(f"Fetched {len(df)} bars")

    # Run backtest
    config = ConfigV61()
    engine = BacktestEngineV61(config)
    results = engine.run(df)

    # Save results
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
            'regime': t.regime
        } for t in engine.trades])

        output_path = Path(__file__).parent.parent / "results" / "h1_v6_1_trades.csv"
        trades_df.to_csv(output_path, index=False)
        print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
