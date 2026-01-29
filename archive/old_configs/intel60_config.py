"""INTEL_60 Live Trading Configuration
======================================

Best configuration from 13-month backtest (Jan 2025 - Jan 2026):
- 29 trades (2.2/month)
- 72% win rate (HIGHEST)
- 2 losing months (BEST)
- +13.8% return

Key Features:
- Intelligent Activity Filter replaces fixed Kill Zone
- Uses Kalman velocity for momentum detection
- Adaptive quality thresholds based on market activity
- Trades when market is MOVING, skips when QUIET

Author: SURIOTA Team
Date: 2026-01-29
"""

# =============================================================================
# INTEL_60 CONFIGURATION
# =============================================================================

INTEL_60_CONFIG = {
    'name': 'INTEL_60',

    # Intelligent Activity Filter Settings
    'use_intelligent_filter': True,
    'intelligent_threshold': 60.0,      # Activity score threshold (0-100)
    'min_velocity_pips': 2.0,           # Minimum Kalman velocity
    'min_atr_pips': 5.0,                # Minimum ATR for volatility

    # Adaptive Quality Thresholds (from Intelligent Filter)
    # SURGING market (score >= 80): quality_threshold = 60
    # ACTIVE market (score >= 60): quality_threshold = 65
    # MODERATE market (score >= 40): quality_threshold = 70
    # QUIET market (score < 40): SKIP TRADING

    # Risk Management
    'max_sl_pips': 30.0,               # Allow reasonable SL
    'max_loss_per_trade_pct': 0.4,     # Max 0.4% loss per trade
    'max_lot_size': 0.5,               # Maximum lot size

    # Filters
    'use_killzone': False,             # DISABLED - replaced by intelligent filter
    'use_trend_filter': True,          # Keep trend alignment
    'use_choppiness_filter': False,    # Disabled

    # Trading
    'skip_december': True,             # Skip December (anomaly month)
    'pip_value': 10.0,
    'spread_pips': 1.5,
}

# =============================================================================
# COMPARISON WITH OTHER CONFIGS (13-month backtest)
# =============================================================================
"""
| Config   | Trades | /Mon | WR  | Lose | Return |
|----------|--------|------|-----|------|--------|
| KZ_ON    | 24     | 1.8  | 62% | 3    | +14.2% |
| KZ_OFF   | 34     | 2.6  | 68% | 2    | +12.5% |
| INTEL_40 | 33     | 2.5  | 64% | 2    | +11.4% |
| INTEL_50 | 30     | 2.3  | 67% | 2    | +12.4% |
| INTEL_60 | 29     | 2.2  | 72% | 2    | +13.8% |  <-- SELECTED

INTEL_60 advantages over KZ_ON:
- +21% more trades (29 vs 24)
- +10% higher win rate (72% vs 62%)
- 1 fewer losing month (2 vs 3)
- Similar return (+13.8% vs +14.2%)
"""

# =============================================================================
# HOW IT WORKS
# =============================================================================
"""
The Intelligent Activity Filter replaces fixed Kill Zone hours with
velocity-based market activity detection:

1. KALMAN VELOCITY (30 points max)
   - Measures rate of price change
   - High velocity = strong momentum = trade
   - Low velocity = ranging = skip

2. ATR VOLATILITY (30 points max)
   - Measures market volatility
   - High ATR = opportunity = trade
   - Low ATR = dead market = skip

3. PRICE RANGE (20 points max)
   - Current bar range
   - Large range = active = trade
   - Small range = quiet = skip

4. MOMENTUM (20 points max)
   - Price change over lookback period
   - Strong momentum = trending = trade
   - Weak momentum = ranging = skip

TOTAL SCORE (0-100):
- >= 80: SURGING (best) - quality 60
- >= 60: ACTIVE (good) - quality 65
- >= 40: MODERATE (ok) - quality 70
- < 40: QUIET (skip) - no trading

This allows trading at ANY time when market is active,
not just during fixed Kill Zone hours.
"""
