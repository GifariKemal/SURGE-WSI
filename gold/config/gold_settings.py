"""Gold (XAUUSD) Trading Configuration
=====================================

Optimized parameters for XAUUSD based on data analysis.
Gold characteristics differ significantly from Forex pairs:
- 5.8x higher ATR than GBPUSD
- 5.7x higher velocity
- Best session: New York

Author: SURIOTA Team
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class GoldIntelFilterConfig:
    """Intelligent Activity Filter settings for Gold"""
    enabled: bool = True
    activity_threshold: float = 50  # Lower than GBPUSD (60) - Gold moves are bigger

    # Velocity thresholds (in pips where 1 pip = $0.10)
    min_velocity_pips: float = 58.0  # 50th percentile from analysis
    high_velocity_pips: float = 120.0  # 75th percentile

    # ATR thresholds
    min_atr_pips: float = 90.0  # ~50% of avg ATR
    high_atr_pips: float = 180.0  # ~100% of avg ATR


@dataclass
class GoldRiskConfig:
    """Risk Management settings for Gold"""
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_daily_risk: float = 0.03  # 3% max daily risk

    # Stop Loss settings (in pips)
    default_sl_pips: float = 150.0  # ~80% of ATR
    max_sl_pips: float = 280.0  # 1.5x ATR
    min_sl_pips: float = 50.0  # Minimum SL

    # Take Profit
    min_rr_ratio: float = 2.0  # Minimum risk:reward

    # Position sizing
    pip_value_per_lot: float = 1.0  # $1 per pip per 0.01 lot
    max_lot_size: float = 1.0
    min_lot_size: float = 0.01

    # Exposure limits
    max_positions: int = 2  # Max open positions
    max_exposure_pct: float = 0.05  # 5% max exposure


@dataclass
class GoldPOIConfig:
    """POI Detection settings for Gold"""
    # POI tolerance (for entry zones)
    tolerance_pips: float = 50.0  # ~30% of ATR for entry zone buffer

    # Order Block settings
    max_ob_age_bars: int = 30  # Fresher OBs for Gold's faster moves
    min_ob_strength: float = 0.3  # Lower threshold
    ob_min_pips: float = 30.0  # Minimum OB size

    # FVG settings
    fvg_min_pips: float = 20.0  # Minimum FVG size
    fvg_max_age_bars: int = 25


@dataclass
class GoldSessionConfig:
    """Trading Session settings for Gold"""
    # Best sessions (UTC hours)
    london_start: int = 7
    london_end: int = 16
    newyork_start: int = 12
    newyork_end: int = 21

    # Primary session for Gold (New York)
    primary_session_start: int = 12
    primary_session_end: int = 20

    # Skip conditions
    skip_asian_session: bool = True  # Asian session has less range
    skip_friday_late: bool = True  # After 18:00 UTC on Friday


@dataclass
class GoldSymbolConfig:
    """Symbol-specific settings"""
    symbol: str = "XAUUSD"
    pip_size: float = 0.1  # 1 pip = $0.10 price movement
    point_size: float = 0.01  # Smallest price increment
    spread_pips: float = 25.0  # Typical spread (~$2.50)

    # Price characteristics from analysis
    avg_price: float = 2500.0  # Approximate average price
    avg_atr_pips: float = 187.0
    avg_velocity_pips: float = 93.0
    avg_daily_range_pips: float = 188.0


@dataclass
class GoldConfig:
    """Complete Gold Trading Configuration"""
    symbol: GoldSymbolConfig = field(default_factory=GoldSymbolConfig)
    intel_filter: GoldIntelFilterConfig = field(default_factory=GoldIntelFilterConfig)
    risk: GoldRiskConfig = field(default_factory=GoldRiskConfig)
    poi: GoldPOIConfig = field(default_factory=GoldPOIConfig)
    session: GoldSessionConfig = field(default_factory=GoldSessionConfig)

    # Timeframes
    htf_timeframe: str = "H4"  # Higher timeframe for regime/POI
    ltf_timeframe: str = "M15"  # Lower timeframe for entry
    entry_timeframe: str = "M5"  # Entry trigger timeframe

    # Analysis settings
    warmup_bars_htf: int = 300  # Warmup for HTF
    warmup_bars_ltf: int = 500  # Warmup for LTF

    # Regime settings
    regime_min_probability: float = 0.65  # Minimum regime confidence
    regime_lookback: int = 100  # Bars for regime detection


# Default configuration instance
GOLD_CONFIG = GoldConfig()


def get_gold_config() -> GoldConfig:
    """Get Gold configuration"""
    return GOLD_CONFIG


def print_config():
    """Print current Gold configuration"""
    cfg = GOLD_CONFIG

    print("=" * 60)
    print("GOLD (XAUUSD) TRADING CONFIGURATION")
    print("=" * 60)
    print()

    print("SYMBOL SETTINGS:")
    print(f"  Symbol: {cfg.symbol.symbol}")
    print(f"  Pip Size: {cfg.symbol.pip_size}")
    print(f"  Spread: {cfg.symbol.spread_pips} pips")
    print(f"  Avg ATR: {cfg.symbol.avg_atr_pips} pips")
    print()

    print("INTELLIGENT FILTER:")
    print(f"  Enabled: {cfg.intel_filter.enabled}")
    print(f"  Activity Threshold: {cfg.intel_filter.activity_threshold}")
    print(f"  Min Velocity: {cfg.intel_filter.min_velocity_pips} pips")
    print(f"  High Velocity: {cfg.intel_filter.high_velocity_pips} pips")
    print()

    print("RISK MANAGEMENT:")
    print(f"  Risk Per Trade: {cfg.risk.risk_per_trade * 100}%")
    print(f"  Default SL: {cfg.risk.default_sl_pips} pips")
    print(f"  Max SL: {cfg.risk.max_sl_pips} pips")
    print(f"  Min R:R Ratio: {cfg.risk.min_rr_ratio}")
    print()

    print("POI DETECTION:")
    print(f"  Tolerance: {cfg.poi.tolerance_pips} pips")
    print(f"  Max OB Age: {cfg.poi.max_ob_age_bars} bars")
    print(f"  Min OB Strength: {cfg.poi.min_ob_strength}")
    print()

    print("SESSIONS (UTC):")
    print(f"  Primary: {cfg.session.primary_session_start}:00 - {cfg.session.primary_session_end}:00")
    print(f"  Skip Asian: {cfg.session.skip_asian_session}")
    print()


if __name__ == "__main__":
    print_config()
