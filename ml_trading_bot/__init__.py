"""
SURGE ML Trading Bot
====================

Machine Learning-based trading system with:
- Market regime detection (HMM)
- Signal classification (XGBoost/Random Forest)
- Dynamic risk management (Half Kelly)

Modules:
--------
- features: Feature engineering pipeline
- models: ML model implementations
- training: Model training scripts
- inference: Real-time prediction
- backtest: Strategy backtesting
"""

__version__ = "0.1.0"
__author__ = "SURGE Trading Team"

from pathlib import Path

# Package paths
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "config"
MODELS_DIR = PACKAGE_DIR / "saved_models"
DATA_DIR = PROJECT_ROOT / "backtest" / "market_analysis"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
