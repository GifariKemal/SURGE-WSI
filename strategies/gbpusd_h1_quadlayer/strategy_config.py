"""
Strategy Configuration - All configurable parameters in one place
=================================================================

This file contains all configurable parameters for the GBPUSD H1
Quad-Layer strategy. Edit values here instead of in the code.

Author: SURIOTA Team
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================
# PATHS
# ============================================================
STRATEGY_DIR = Path(__file__).parent
PROJECT_ROOT = STRATEGY_DIR.parent.parent

# MetaQuotes terminal path (adjust for your system)
METAQUOTES_TERMINAL_PATH = os.environ.get(
    'METAQUOTES_TERMINAL_PATH',
    r"C:\Program Files\MetaTrader 5\terminal64.exe"
)

# State file for persistence
STATE_FILE = STRATEGY_DIR / "state.json"


# ============================================================
# BROKER SETTINGS
# ============================================================
REQUIRED_BROKER = "MetaQuotes"   # Must contain this string
FORBIDDEN_BROKER = "Finex"       # Must NOT contain this string


# ============================================================
# SERVICE PORTS (for startup checks)
# ============================================================
POSTGRES_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.environ.get('POSTGRES_PORT', 5434))
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6381))


# ============================================================
# TRADING PARAMETERS
# ============================================================
SYMBOL = "GBPUSD"
TIMEFRAME = "H1"
PIP_SIZE = 0.0001
PIP_VALUE = 10.0  # USD per pip per lot


# ============================================================
# RISK MANAGEMENT
# ============================================================
@dataclass
class RiskConfig:
    """Risk management parameters"""
    # Position sizing
    risk_percent: float = 1.0       # 1% per trade
    max_lot: float = 5.0            # Maximum lot size
    min_lot: float = 0.01           # Minimum lot size

    # Stop loss / Take profit
    sl_atr_mult: float = 1.5        # SL = ATR * 1.5
    tp_ratio: float = 1.5           # TP = SL * 1.5
    min_sl_pips: float = 5.0        # Minimum SL in pips
    max_sl_pips: float = 50.0       # Maximum SL in pips

    # Loss limits
    max_loss_per_trade_pct: float = 0.15  # 0.15% max loss cap

    # Validation
    def validate(self):
        """Validate configuration values"""
        errors = []

        if not 0.1 <= self.risk_percent <= 5.0:
            errors.append(f"risk_percent {self.risk_percent} should be 0.1-5.0%")

        if self.max_lot <= 0:
            errors.append(f"max_lot {self.max_lot} must be positive")

        if self.min_lot <= 0:
            errors.append(f"min_lot {self.min_lot} must be positive")

        if self.min_lot >= self.max_lot:
            errors.append(f"min_lot {self.min_lot} must be < max_lot {self.max_lot}")

        if self.sl_atr_mult <= 0:
            errors.append(f"sl_atr_mult {self.sl_atr_mult} must be positive")

        if self.tp_ratio <= 0:
            errors.append(f"tp_ratio {self.tp_ratio} must be positive")

        if self.min_sl_pips <= 0:
            errors.append(f"min_sl_pips {self.min_sl_pips} must be positive")

        if errors:
            raise ValueError("Risk config validation failed:\n" + "\n".join(errors))


RISK = RiskConfig()


# ============================================================
# LAYER 1: MONTHLY PROFILE
# ============================================================
# Monthly quality adjustments based on historical tradeable percentage
# Higher adjustment = stricter quality requirement
MONTHLY_QUALITY_THRESHOLDS = {
    30: 50,   # tradeable < 30% → +50 quality (NO TRADE)
    40: 35,   # tradeable < 40% → +35 quality (HALT)
    50: 25,   # tradeable < 50% → +25 quality
    60: 15,   # tradeable < 60% → +15 quality
    70: 10,   # tradeable < 70% → +10 quality
    75: 5,    # tradeable < 75% → +5 quality
    100: 0,   # tradeable >= 75% → no adjustment
}

# Monthly tradeable percentage (historical data only - 2024)
# NOTE: For future months, use SEASONAL_TEMPLATE in trading_filters.py
MONTHLY_TRADEABLE_PCT: Dict[Tuple[int, int], int] = {
    # 2024 - Actual historical data (base for seasonal template)
    (2024, 1): 67, (2024, 2): 55, (2024, 3): 70, (2024, 4): 80,
    (2024, 5): 65, (2024, 6): 72, (2024, 7): 78, (2024, 8): 60,
    (2024, 9): 75, (2024, 10): 58, (2024, 11): 68, (2024, 12): 45,
}

# Seasonal template for future months (derived from 2024 with targeted adjustments)
# Key insight: April tends to be optimistic (80% → 70%) so we apply safety margin
# This avoids data leakage while capturing seasonal patterns
SEASONAL_TEMPLATE: Dict[int, int] = {
    1: 65, 2: 55, 3: 70, 4: 70, 5: 62, 6: 68,
    7: 78, 8: 60, 9: 72, 10: 58, 11: 66, 12: 45,
}


# ============================================================
# LAYER 2: TECHNICAL QUALITY
# ============================================================
@dataclass
class TechnicalConfig:
    """Layer 2 technical filter parameters"""
    # Base quality scores
    base_quality_good: int = 60     # Good market conditions
    base_quality_normal: int = 65   # Normal conditions
    base_quality_bad: int = 80      # Bad conditions

    # ATR thresholds (in pips)
    min_atr: float = 8.0            # Minimum ATR to trade
    max_atr: float = 25.0           # Maximum ATR to trade (v6.6.1: reduced from 30, ATR 25-30 had 0% WR)
    atr_low_mult: float = 0.8       # ATR < avg * this = low volatility
    atr_high_mult: float = 1.2      # ATR > avg * this = high volatility

    # Efficiency threshold
    min_efficiency: float = 0.3     # Minimum price efficiency

    # ADX threshold
    min_adx: float = 20.0           # Minimum ADX for trending


TECHNICAL = TechnicalConfig()


# ============================================================
# LAYER 3: INTRA-MONTH RISK
# ============================================================
@dataclass
class IntraMonthConfig:
    """Layer 3 intra-month risk parameters"""
    # Monthly P&L thresholds (USD)
    loss_threshold_1: float = -150.0  # +5 quality
    loss_threshold_2: float = -250.0  # +10 quality
    loss_threshold_3: float = -350.0  # +15 quality
    loss_stop: float = -400.0         # Stop trading for month

    # Consecutive loss thresholds
    consec_loss_quality: int = 3      # +5 quality after this many losses
    consec_loss_day_stop: int = 6     # Stop for day after this many losses

    def validate(self):
        """Validate configuration"""
        if self.loss_stop >= self.loss_threshold_3:
            raise ValueError("loss_stop must be < loss_threshold_3")
        if self.consec_loss_day_stop <= self.consec_loss_quality:
            raise ValueError("consec_loss_day_stop must be > consec_loss_quality")


INTRA_MONTH = IntraMonthConfig()


# ============================================================
# LAYER 4: PATTERN FILTER
# ============================================================
@dataclass
class PatternConfig:
    """Layer 4 pattern filter parameters"""
    # Warmup
    warmup_trades: int = 15           # Trades before filter activates

    # Rolling win rate
    rolling_window: int = 10          # Window size
    rolling_wr_halt: float = 0.10     # Halt if WR < this
    rolling_wr_caution: float = 0.25  # Reduce size if WR < this

    # Direction analysis
    direction_window: int = 8         # Window for direction check
    both_fail_threshold: int = 4      # Halt if both directions fail this many

    # Size multipliers
    recovery_size: float = 0.5        # Size in recovery mode
    caution_size: float = 0.6         # Size in caution mode
    probe_size: float = 0.4           # Size for probe trades

    # Quality
    probe_quality_extra: int = 5      # Extra quality during probe/recovery

    def validate(self):
        """Validate configuration"""
        if self.warmup_trades < 0:
            raise ValueError("warmup_trades must be >= 0")
        if not 0 < self.rolling_wr_halt < self.rolling_wr_caution < 1:
            raise ValueError("Must have: 0 < rolling_wr_halt < rolling_wr_caution < 1")


PATTERN = PatternConfig()


# ============================================================
# KILL ZONES (Trading Hours)
# ============================================================
@dataclass
class KillZoneConfig:
    """Trading hours configuration (UTC)"""
    # London session
    london_start: int = 8
    london_end: int = 12

    # New York session
    new_york_start: int = 13
    new_york_end: int = 17

    # Hour multipliers (quality adjustment)
    # Lower = better hour, higher = worse hour
    hour_multipliers: Dict[int, float] = None

    def __post_init__(self):
        if self.hour_multipliers is None:
            self.hour_multipliers = {
                8: 0.9, 9: 0.85, 10: 0.9, 11: 0.95,  # London
                12: 1.0,  # Lunch
                13: 0.9, 14: 0.85, 15: 0.8, 16: 0.85, 17: 0.95,  # New York
            }

    def is_kill_zone(self, hour: int) -> bool:
        """Check if hour is in kill zone"""
        return (self.london_start <= hour <= self.london_end or
                self.new_york_start <= hour <= self.new_york_end)


KILLZONE = KillZoneConfig()


# ============================================================
# MONTHLY RISK MULTIPLIERS
# ============================================================
# Adjust risk based on historical month performance
# Lower = reduce risk, Higher = normal/increased risk
MONTHLY_RISK_MULT: Dict[int, float] = {
    1: 0.9,   # January
    2: 0.6,   # February (historically worst)
    3: 0.8,   # March
    4: 1.0,   # April (good)
    5: 0.7,   # May
    6: 0.85,  # June
    7: 1.0,   # July (good)
    8: 0.75,  # August (summer)
    9: 0.9,   # September
    10: 0.6,  # October (volatile)
    11: 0.75, # November
    12: 0.8,  # December (holiday)
}


# ============================================================
# DAY MULTIPLIERS
# ============================================================
# Adjust risk based on day of week (0=Monday, 4=Friday)
DAY_RISK_MULT: Dict[int, float] = {
    0: 0.9,   # Monday (gap risk)
    1: 1.0,   # Tuesday
    2: 1.0,   # Wednesday
    3: 1.0,   # Thursday
    4: 0.7,   # Friday (weekend risk)
}


# ============================================================
# TELEGRAM
# ============================================================
@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    enabled: bool = True
    hourly_status: bool = True        # Send hourly status updates
    trade_notifications: bool = True  # Send trade notifications
    error_notifications: bool = True  # Send error notifications


TELEGRAM = TelegramConfig()


# ============================================================
# LOGGING
# ============================================================
@dataclass
class LogConfig:
    """Logging configuration"""
    level: str = "INFO"               # Console log level
    file_level: str = "DEBUG"         # File log level
    rotation: str = "1 day"           # Log rotation
    retention: str = "30 days"        # Log retention


LOG = LogConfig()


# ============================================================
# VALIDATION
# ============================================================
def validate_all():
    """Validate all configuration"""
    RISK.validate()
    INTRA_MONTH.validate()
    PATTERN.validate()


# Run validation on import
try:
    validate_all()
except ValueError as e:
    raise RuntimeError(f"Configuration validation failed: {e}")
