"""Market Condition Filter v2 - Enhanced for H1 v4
==================================================

Based on analysis of losing months (Feb, Sep, Nov 2025):
- Feb 2025: 22% WR, all BEARISH, 5 consecutive losses
- Sep 2025: 35% WR, low ATR (15.4 vs 17.4), Thursday 0% WR
- Nov 2025: 33% WR, Thursday 0% WR

Key findings from research:
1. Choppiness Index > 61.8 = Ranging market (avoid)
2. ADX < 20 = Weak trend (avoid)
3. Thursday had 0% win rate in all losing months
4. Fixed SL doesn't adapt to volatility changes
5. Regime confidence needs minimum threshold

Research References:
- LuxAlgo: Choppiness Index (https://www.luxalgo.com/blog/choppiness-index-quantifying-consolidation/)
- FOREX.com: ADX Indicator (https://www.forex.com/en-us/trading-guides/adx-indicator/)
- ResearchGate: Day-of-week effect in FX markets
- QuantStart: HMM Regime Detection

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from loguru import logger


class MarketCondition(Enum):
    """Market condition states"""
    STRONG_TREND = "strong_trend"   # Best for trading
    MODERATE_TREND = "moderate_trend"  # OK for trading
    WEAK_TREND = "weak_trend"       # Caution
    RANGING = "ranging"             # Avoid trading
    UNKNOWN = "unknown"


@dataclass
class MarketConditionResult:
    """Result of market condition analysis"""
    # Market state
    condition: MarketCondition
    can_trade: bool

    # Indicators
    choppiness: float           # 0-100 (>61.8 = ranging)
    adx: float                  # 0-100 (>20 = trending)
    atr_pips: float             # Current ATR in pips

    # Stop Loss
    suggested_sl_pips: float    # ATR-based SL
    sl_multiplier: float        # ATR multiplier used

    # Thursday filter
    is_thursday: bool
    thursday_multiplier: float  # Position size multiplier

    # Regime
    regime_confidence_ok: bool

    # Details
    confluence_score: float     # Overall score (0-100)
    reasons: list               # List of reasons

    def __str__(self):
        status = "✅ TRADE" if self.can_trade else "❌ SKIP"
        return (
            f"{status} | {self.condition.value.upper()} | "
            f"Score: {self.confluence_score:.0f} | "
            f"Chop: {self.choppiness:.1f} | ADX: {self.adx:.1f} | "
            f"ATR: {self.atr_pips:.1f}p | SL: {self.suggested_sl_pips:.1f}p"
        )


class MarketConditionFilter:
    """
    Enhanced Market Condition Filter v2.

    Filters:
    1. Choppiness Index (skip > 61.8)
    2. ADX (skip < 20)
    3. ATR-based Stop Loss
    4. Thursday caution
    5. Regime confidence threshold
    """

    def __init__(
        self,
        # Choppiness settings
        chop_period: int = 14,
        chop_trending_threshold: float = 38.2,   # Below = strong trend
        chop_ranging_threshold: float = 65.0,    # Above = ranging (SKIP) - v4.2 balanced

        # ADX settings
        adx_period: int = 14,
        adx_weak_threshold: float = 18.0,        # Below = weak trend (SKIP) - v4.2 balanced
        adx_strong_threshold: float = 25.0,      # Above = strong trend

        # ATR-based SL settings
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,          # SL = ATR * multiplier
        min_sl_pips: float = 15.0,               # Minimum SL
        max_sl_pips: float = 40.0,               # Maximum SL

        # Regime settings
        regime_confidence_threshold: float = 60.0,  # Minimum confidence - v4.2 balanced

        # Thursday settings
        enable_thursday_filter: bool = True,
        thursday_position_multiplier: float = 0.6,  # Reduce position on Thursday - v4.2 balanced
        skip_thursday_in_weak_market: bool = True,  # Skip entirely if weak

        # General
        min_confluence_score: float = 45.0,      # v4.2 balanced
        pip_size: float = 0.0001
    ):
        """Initialize Enhanced Market Condition Filter"""
        # Choppiness
        self.chop_period = chop_period
        self.chop_trending_threshold = chop_trending_threshold
        self.chop_ranging_threshold = chop_ranging_threshold

        # ADX
        self.adx_period = adx_period
        self.adx_weak_threshold = adx_weak_threshold
        self.adx_strong_threshold = adx_strong_threshold

        # ATR
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

        # Regime
        self.regime_confidence_threshold = regime_confidence_threshold

        # Thursday
        self.enable_thursday_filter = enable_thursday_filter
        self.thursday_position_multiplier = thursday_position_multiplier
        self.skip_thursday_in_weak_market = skip_thursday_in_weak_market

        # General
        self.min_confluence_score = min_confluence_score
        self.pip_size = pip_size

        # Cache
        self._last_choppiness = 50.0
        self._last_adx = 25.0
        self._last_atr_pips = 20.0

        logger.info(
            f"MarketConditionFilter v2 initialized: "
            f"chop_skip>{chop_ranging_threshold}, "
            f"adx_skip<{adx_weak_threshold}, "
            f"atr_sl_mult={atr_sl_multiplier}, "
            f"thursday_mult={thursday_position_multiplier}"
        )

    def _get_col_map(self, df: pd.DataFrame) -> dict:
        """Get column name mapping"""
        return {
            'close': 'close' if 'close' in df.columns else 'Close',
            'open': 'open' if 'open' in df.columns else 'Open',
            'high': 'high' if 'high' in df.columns else 'High',
            'low': 'low' if 'low' in df.columns else 'Low',
        }

    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range"""
        if period is None:
            period = self.atr_period

        col_map = self._get_col_map(df)
        high = df[col_map['high']]
        low = df[col_map['low']]
        close = df[col_map['close']]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(period).mean()

    def calculate_choppiness(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Choppiness Index (0-100).

        Interpretation:
        - < 38.2: Market is strongly trending
        - > 61.8: Market is choppy/ranging
        - Between: Moderate/transition

        Formula: 100 * LOG10(SUM(TR, n) / (HH - LL)) / LOG10(n)
        """
        if period is None:
            period = self.chop_period

        col_map = self._get_col_map(df)
        high = df[col_map['high']]
        low = df[col_map['low']]
        close = df[col_map['close']]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Sum of TR
        atr_sum = tr.rolling(period).sum()

        # Highest High and Lowest Low
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        price_range = highest_high - lowest_low

        # Avoid division by zero
        price_range = price_range.replace(0, np.nan)

        # Choppiness Index
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)

        return chop

    def calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate ADX (Average Directional Index).

        Measures trend strength (not direction):
        - < 20: Weak or no trend
        - 20-25: Moderate trend developing
        - 25-40: Strong trend
        - > 40: Very strong trend
        """
        if period is None:
            period = self.adx_period

        col_map = self._get_col_map(df)
        high = df[col_map['high']]
        low = df[col_map['low']]
        close = df[col_map['close']]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed using Wilder's method
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return adx

    def analyze(
        self,
        df: pd.DataFrame,
        current_time: datetime = None,
        regime_confidence: float = 100.0,
        direction: str = None
    ) -> MarketConditionResult:
        """
        Analyze market conditions and determine if trading is advisable.

        Args:
            df: OHLCV DataFrame (minimum 30 bars recommended)
            current_time: Current datetime (for Thursday check)
            regime_confidence: HMM regime confidence (0-100)
            direction: Trade direction (for logging)

        Returns:
            MarketConditionResult with analysis
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        reasons = []
        can_trade = True

        # Check minimum data
        min_bars = max(self.chop_period, self.adx_period, self.atr_period) + 5
        if len(df) < min_bars:
            return MarketConditionResult(
                condition=MarketCondition.UNKNOWN,
                can_trade=True,  # Don't block if insufficient data
                choppiness=50.0,
                adx=25.0,
                atr_pips=20.0,
                suggested_sl_pips=25.0,
                sl_multiplier=self.atr_sl_multiplier,
                is_thursday=False,
                thursday_multiplier=1.0,
                regime_confidence_ok=True,
                confluence_score=50.0,
                reasons=["Insufficient data for market analysis"]
            )

        # =====================================================================
        # CALCULATE INDICATORS
        # =====================================================================

        # Choppiness Index
        chop_series = self.calculate_choppiness(df)
        choppiness = chop_series.iloc[-1] if not pd.isna(chop_series.iloc[-1]) else 50.0
        self._last_choppiness = choppiness

        # ADX
        adx_series = self.calculate_adx(df)
        adx = adx_series.iloc[-1] if not pd.isna(adx_series.iloc[-1]) else 25.0
        self._last_adx = adx

        # ATR
        atr_series = self.calculate_atr(df)
        atr = atr_series.iloc[-1] if not pd.isna(atr_series.iloc[-1]) else 0.0020
        atr_pips = atr / self.pip_size
        self._last_atr_pips = atr_pips

        # Calculate ATR-based Stop Loss
        suggested_sl_pips = atr_pips * self.atr_sl_multiplier
        suggested_sl_pips = max(self.min_sl_pips, min(self.max_sl_pips, suggested_sl_pips))

        # =====================================================================
        # FILTER 1: CHOPPINESS INDEX (Skip if > 61.8)
        # =====================================================================
        if choppiness > self.chop_ranging_threshold:
            can_trade = False
            reasons.append(f"❌ Ranging market (Chop={choppiness:.1f} > {self.chop_ranging_threshold})")
        elif choppiness < self.chop_trending_threshold:
            reasons.append(f"✅ Strong trend (Chop={choppiness:.1f})")
        else:
            reasons.append(f"⚠️ Moderate choppiness (Chop={choppiness:.1f})")

        # =====================================================================
        # FILTER 2: ADX (Skip if < 20)
        # =====================================================================
        if adx < self.adx_weak_threshold:
            can_trade = False
            reasons.append(f"❌ Weak trend (ADX={adx:.1f} < {self.adx_weak_threshold})")
        elif adx >= self.adx_strong_threshold:
            reasons.append(f"✅ Strong trend (ADX={adx:.1f})")
        else:
            reasons.append(f"⚠️ Developing trend (ADX={adx:.1f})")

        # =====================================================================
        # FILTER 3: REGIME CONFIDENCE
        # =====================================================================
        regime_confidence_ok = regime_confidence >= self.regime_confidence_threshold
        if not regime_confidence_ok:
            can_trade = False
            reasons.append(f"❌ Low regime confidence ({regime_confidence:.0f}% < {self.regime_confidence_threshold}%)")
        else:
            reasons.append(f"✅ Regime confidence OK ({regime_confidence:.0f}%)")

        # =====================================================================
        # FILTER 4: THURSDAY CAUTION
        # =====================================================================
        is_thursday = current_time.weekday() == 3  # Thursday = 3
        thursday_multiplier = 1.0

        if self.enable_thursday_filter and is_thursday:
            thursday_multiplier = self.thursday_position_multiplier
            reasons.append(f"⚠️ Thursday: Position reduced to {thursday_multiplier:.0%}")

            # If already in weak/ranging condition, skip Thursday entirely
            if self.skip_thursday_in_weak_market:
                is_weak = (
                    choppiness > 55 or  # Moderately choppy
                    adx < self.adx_strong_threshold  # Not strong trend
                )
                if is_weak:
                    can_trade = False
                    reasons.append("❌ Thursday + weak market = SKIP")

        # =====================================================================
        # DETERMINE MARKET CONDITION
        # =====================================================================
        if choppiness > self.chop_ranging_threshold:
            condition = MarketCondition.RANGING
        elif choppiness < self.chop_trending_threshold and adx >= self.adx_strong_threshold:
            condition = MarketCondition.STRONG_TREND
        elif adx >= self.adx_weak_threshold:
            condition = MarketCondition.MODERATE_TREND
        else:
            condition = MarketCondition.WEAK_TREND

        # =====================================================================
        # CALCULATE CONFLUENCE SCORE
        # =====================================================================
        # Choppiness score: 100 when trending (low chop), 0 when ranging (high chop)
        chop_range = self.chop_ranging_threshold - self.chop_trending_threshold
        if choppiness <= self.chop_trending_threshold:
            chop_score = 100.0
        elif choppiness >= self.chop_ranging_threshold:
            chop_score = 0.0
        else:
            chop_score = 100 * (self.chop_ranging_threshold - choppiness) / chop_range

        # ADX score: 0 when weak, 100 when strong
        adx_range = 40.0 - self.adx_weak_threshold
        if adx <= self.adx_weak_threshold:
            adx_score = 0.0
        elif adx >= 40.0:
            adx_score = 100.0
        else:
            adx_score = 100 * (adx - self.adx_weak_threshold) / adx_range

        # Confidence score
        conf_score = min(100, regime_confidence)

        # Weighted average: Chop 35%, ADX 35%, Confidence 30%
        confluence_score = (
            chop_score * 0.35 +
            adx_score * 0.35 +
            conf_score * 0.30
        )

        # Final check on confluence
        if confluence_score < self.min_confluence_score and can_trade:
            can_trade = False
            reasons.append(f"❌ Low confluence score ({confluence_score:.0f} < {self.min_confluence_score})")

        # Add ATR info
        reasons.append(f"📊 ATR: {atr_pips:.1f}p → SL: {suggested_sl_pips:.1f}p")

        return MarketConditionResult(
            condition=condition,
            can_trade=can_trade,
            choppiness=choppiness,
            adx=adx,
            atr_pips=atr_pips,
            suggested_sl_pips=suggested_sl_pips,
            sl_multiplier=self.atr_sl_multiplier,
            is_thursday=is_thursday,
            thursday_multiplier=thursday_multiplier,
            regime_confidence_ok=regime_confidence_ok,
            confluence_score=confluence_score,
            reasons=reasons
        )

    def get_last_values(self) -> dict:
        """Get last calculated indicator values"""
        return {
            'choppiness': self._last_choppiness,
            'adx': self._last_adx,
            'atr_pips': self._last_atr_pips
        }

    def should_skip_trading(
        self,
        df: pd.DataFrame,
        current_time: datetime = None,
        regime_confidence: float = 100.0
    ) -> Tuple[bool, str, float]:
        """
        Quick check if trading should be skipped.

        Returns:
            (should_skip, reason, suggested_sl_pips)
        """
        result = self.analyze(df, current_time, regime_confidence)

        if not result.can_trade:
            reason = "; ".join([r for r in result.reasons if r.startswith("❌")])
            return True, reason, result.suggested_sl_pips

        return False, "Market conditions OK", result.suggested_sl_pips


# ============================================================================
# STANDALONE FUNCTIONS (for backtest)
# ============================================================================

def calculate_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Choppiness Index (standalone function for backtest)"""
    mcf = MarketConditionFilter(chop_period=period)
    return mcf.calculate_choppiness(df, period)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX (standalone function for backtest)"""
    mcf = MarketConditionFilter(adx_period=period)
    return mcf.calculate_adx(df, period)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR (standalone function for backtest)"""
    mcf = MarketConditionFilter(atr_period=period)
    return mcf.calculate_atr(df, period)


def check_market_condition(
    df: pd.DataFrame,
    current_time: datetime = None,
    regime_confidence: float = 100.0
) -> Tuple[bool, float, str]:
    """
    Quick market condition check.

    Returns:
        (can_trade, confluence_score, condition_name)
    """
    mcf = MarketConditionFilter()
    result = mcf.analyze(df, current_time, regime_confidence)
    return result.can_trade, result.confluence_score, result.condition.value


if __name__ == "__main__":
    # Test
    import numpy as np

    np.random.seed(42)
    n = 100

    # Create sample trending data
    trend = np.linspace(0, 0.02, n)
    noise = np.random.randn(n) * 0.001
    close = 1.25 + trend + np.cumsum(noise)

    df = pd.DataFrame({
        'open': close - 0.0005,
        'high': close + np.random.rand(n) * 0.002,
        'low': close - np.random.rand(n) * 0.002,
        'close': close
    })

    filter = MarketConditionFilter()

    print("\n" + "=" * 60)
    print("MARKET CONDITION FILTER v2 TEST")
    print("=" * 60)

    # Test normal day
    from datetime import datetime, timezone
    monday = datetime(2025, 1, 27, 10, 0, tzinfo=timezone.utc)  # Monday
    result = filter.analyze(df, monday, regime_confidence=80.0)
    print(f"\nMonday Analysis:")
    print(f"  {result}")
    print(f"  Reasons: {result.reasons}")

    # Test Thursday
    thursday = datetime(2025, 1, 30, 10, 0, tzinfo=timezone.utc)  # Thursday
    result = filter.analyze(df, thursday, regime_confidence=80.0)
    print(f"\nThursday Analysis:")
    print(f"  {result}")
    print(f"  Reasons: {result.reasons}")

    # Test low confidence
    result = filter.analyze(df, monday, regime_confidence=50.0)
    print(f"\nLow Confidence (50%):")
    print(f"  {result}")
    print(f"  Reasons: {result.reasons}")

    print("\n" + "=" * 60)
