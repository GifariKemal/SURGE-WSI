"""SURGE-WSI Configuration Module"""
from pathlib import Path
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config directory
CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "settings.yaml"


class MT5Settings(BaseModel):
    """MetaTrader 5 connection settings"""
    login: Optional[int] = None
    password: Optional[str] = None
    server: str = "FinexBisnisSolusi-Server"
    terminal_path: Optional[str] = None


class DatabaseSettings(BaseModel):
    """TimescaleDB settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "surge_wsi"
    user: str = "surge"
    password: str = ""

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseModel):
    """Redis cache settings"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None


class TelegramSettings(BaseModel):
    """Telegram bot settings"""
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    enabled: bool = True


class KalmanSettings(BaseModel):
    """Layer 1: Kalman Filter settings"""
    fast_process_noise: float = 0.05
    fast_measurement_noise: float = 0.05
    medium_process_noise: float = 0.01
    medium_measurement_noise: float = 0.1
    slow_process_noise: float = 0.001
    slow_measurement_noise: float = 0.2


class RegimeSettings(BaseModel):
    """Layer 2: HMM Regime Detection settings"""
    n_states: int = 3
    lookback: int = 50
    min_samples: int = 100
    retrain_every: int = 200
    min_probability: float = 0.6


class KillZoneSettings(BaseModel):
    """Layer 3: ICT Kill Zone settings"""
    london_start: int = 8
    london_end: int = 12
    new_york_start: int = 13
    new_york_end: int = 17
    london_close_start: int = 15
    london_close_end: int = 17
    enabled: bool = True


class POISettings(BaseModel):
    """Layer 4: POI Detection settings (Order Blocks + FVG)"""
    swing_length: int = 10
    ob_min_strength: float = 0.6
    fvg_min_pips: float = 3.0
    max_poi_age_bars: int = 100
    use_order_blocks: bool = True
    use_fvg: bool = True
    use_bos: bool = True


class EntrySettings(BaseModel):
    """Layer 5: Entry Trigger settings"""
    swing_length: int = 3
    sweep_min_pips: float = 2.0
    mss_lookback: int = 10
    fvg_min_pips: float = 1.0
    min_quality_score: float = 65.0   # Higher quality for zero-loss strategy
    max_sl_pips: float = 50.0
    rejection_wick_ratio: float = 0.5


class ExitSettings(BaseModel):
    """Layer 6: Exit Management settings"""
    # Partial TP Strategy
    tp1_rr: float = 1.0
    tp1_percent: float = 0.5
    tp2_rr: float = 2.0
    tp2_percent: float = 0.3
    tp3_rr: float = 3.0
    tp3_percent: float = 0.2
    # Trailing Stop
    trailing_enabled: bool = True
    trailing_start_rr: float = 1.5
    trailing_step_pips: float = 10.0
    # Breakeven
    move_sl_to_be_at_tp1: bool = True


class RiskSettings(BaseModel):
    """Risk Management settings"""
    # Position sizing by zone quality
    high_quality_threshold: float = 80.0
    high_quality_risk: float = 0.015  # 1.5%
    medium_quality_threshold: float = 60.0
    medium_quality_risk: float = 0.01  # 1.0%
    low_quality_risk: float = 0.005   # 0.5%
    # Daily limits
    daily_profit_target: float = 100.0
    daily_loss_limit: float = 80.0    # 0.8% of $10K = $80 daily loss limit
    max_open_positions: int = 1
    # Per trade limits - ZERO LOSING MONTHS CONFIG
    max_lot_size: float = 0.5         # Conservative base for zero-loss strategy
    min_lot_size: float = 0.01
    min_sl_pips: float = 15.0         # Minimum SL distance
    max_sl_pips: float = 50.0         # Maximum SL distance
    # Loss protection
    max_loss_per_trade_pct: float = 0.8   # Maximum 0.8% loss per trade
    monthly_loss_stop_pct: float = 2.0    # Stop trading if monthly loss > 2%


class TradingSettings(BaseModel):
    """General trading settings"""
    symbol: str = "GBPUSD"
    magic_number: int = 20250125
    enabled: bool = False
    mode: str = "demo"  # demo or live
    timeframe_htf: str = "H4"
    timeframe_ltf: str = "M5"
    pip_value: float = 10.0


class Settings(BaseSettings):
    """Main configuration class"""
    mt5: MT5Settings = MT5Settings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    telegram: TelegramSettings = TelegramSettings()
    kalman: KalmanSettings = KalmanSettings()
    regime: RegimeSettings = RegimeSettings()
    killzone: KillZoneSettings = KillZoneSettings()
    poi: POISettings = POISettings()
    entry: EntrySettings = EntrySettings()
    exit: ExitSettings = ExitSettings()
    risk: RiskSettings = RiskSettings()
    trading: TradingSettings = TradingSettings()

    class Config:
        env_prefix = ""
        case_sensitive = False


def load_config() -> Settings:
    """Load configuration from YAML and environment variables"""
    settings = Settings()

    # Load from YAML if exists
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Update settings from YAML
                for section, values in yaml_config.items():
                    if hasattr(settings, section) and isinstance(values, dict):
                        section_obj = getattr(settings, section)
                        for key, value in values.items():
                            if hasattr(section_obj, key):
                                setattr(section_obj, key, value)

    # Override with environment variables
    settings.mt5.login = int(os.getenv("MT5_LOGIN", "0")) or None
    settings.mt5.password = os.getenv("MT5_PASSWORD")
    settings.mt5.server = os.getenv("MT5_SERVER", settings.mt5.server)
    settings.mt5.terminal_path = os.getenv("MT5_TERMINAL_PATH")

    settings.database.host = os.getenv("POSTGRES_HOST", settings.database.host)
    settings.database.port = int(os.getenv("POSTGRES_PORT", settings.database.port))
    settings.database.database = os.getenv("POSTGRES_DB", settings.database.database)
    settings.database.user = os.getenv("POSTGRES_USER", settings.database.user)
    settings.database.password = os.getenv("POSTGRES_PASSWORD", settings.database.password)

    settings.redis.host = os.getenv("REDIS_HOST", settings.redis.host)
    settings.redis.port = int(os.getenv("REDIS_PORT", settings.redis.port))

    settings.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    settings.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID")
    settings.telegram.enabled = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

    settings.trading.enabled = os.getenv("TRADING_ENABLED", "false").lower() == "true"
    settings.trading.mode = os.getenv("TRADING_MODE", settings.trading.mode)
    settings.trading.symbol = os.getenv("SYMBOL", settings.trading.symbol)

    return settings


# Global config instance
config = load_config()
