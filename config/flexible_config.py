"""Flexible Trading Configuration
==================================

Optimized for MORE TRADES while maintaining reasonable safety.

Key Changes:
- Kill Zone: OFF or EXPANDED (allows trading outside major sessions)
- Quality: 65-68 (reasonable threshold)
- SL: 30 pips (allows most SMC setups)
- Loss Cap: 0.3-0.4% (balanced protection)

Target Results:
- 100+ trades/year
- Max 2-3 losing months
- 40-50% annual return

Author: SURIOTA Team
Date: 2026-01-29
"""

# =============================================================================
# FLEXIBLE CONFIGURATION - MORE TRADES
# =============================================================================

FLEXIBLE_CONFIG = {
    'name': 'FLEXIBLE',

    # Risk Management
    'max_sl_pips': 30.0,           # Allow most SMC setups (15-30 pips typical)
    'max_loss_per_trade_pct': 0.4, # Max 0.4% loss = $40 on $10k account
    'max_lot_size': 0.5,           # Maximum lot size

    # Entry Quality
    'min_quality_score': 65.0,     # Standard quality (MSS or decent sweep)

    # TIME FILTERS - KEY CHANGE
    'use_killzone': False,         # OFF - Allow trading anytime
    'use_extended_hours': True,    # Trade during extended hours

    # Other Filters
    'use_trend_filter': True,      # Keep trend alignment
    'use_relaxed_filter': False,
    'use_hybrid_mode': False,
    'use_choppiness_filter': False,

    # Trading Parameters
    'skip_december': True,
    'pip_value': 10.0,
    'spread_pips': 1.5,
}

# =============================================================================
# FLEXIBLE WITH EXPANDED KILL ZONE (Alternative)
# =============================================================================

FLEXIBLE_EXPANDED_KZ = {
    'name': 'FLEXIBLE_EXPANDED_KZ',

    # Risk Management
    'max_sl_pips': 30.0,
    'max_loss_per_trade_pct': 0.4,
    'max_lot_size': 0.5,

    # Entry Quality
    'min_quality_score': 65.0,

    # TIME FILTERS - Expanded Kill Zone
    'use_killzone': True,
    'killzone_config': {
        # Original: London 07-10, NY 12-15, Overlap 12-16
        # Expanded: Include Asian session and extended hours
        'asian_session': {'start': 0, 'end': 3},      # 00:00-03:00 UTC
        'london_session': {'start': 6, 'end': 11},    # 06:00-11:00 UTC (extended)
        'ny_session': {'start': 12, 'end': 17},       # 12:00-17:00 UTC (extended)
        'overlap': {'start': 12, 'end': 16},          # 12:00-16:00 UTC
    },

    # Other Filters
    'use_trend_filter': True,
    'use_relaxed_filter': False,
    'use_hybrid_mode': False,
    'use_choppiness_filter': False,

    'skip_december': True,
    'pip_value': 10.0,
    'spread_pips': 1.5,
}

# =============================================================================
# FLEXIBLE AGGRESSIVE (Maximum Trades)
# =============================================================================

FLEXIBLE_AGGRESSIVE = {
    'name': 'FLEXIBLE_AGGRESSIVE',

    # Risk Management - Looser
    'max_sl_pips': 40.0,           # Allow larger SL
    'max_loss_per_trade_pct': 0.5, # Max 0.5% loss
    'max_lot_size': 0.5,

    # Entry Quality - Lower threshold
    'min_quality_score': 60.0,     # Accept more signals

    # TIME FILTERS - All OFF
    'use_killzone': False,
    'use_trend_filter': False,     # Trade against trend too

    'use_relaxed_filter': False,
    'use_hybrid_mode': False,
    'use_choppiness_filter': False,

    'skip_december': True,
    'pip_value': 10.0,
    'spread_pips': 1.5,
}

# =============================================================================
# COMPARISON TABLE
# =============================================================================
"""
| Config              | KillZone | SL   | Quality | Loss% | Expected Trades |
|---------------------|----------|------|---------|-------|-----------------|
| ZERO_LOSS           | ON       | 10   | 75      | 0.1   | 35-50/year      |
| BALANCED            | ON       | 25   | 68      | 0.3   | 60-80/year      |
| FLEXIBLE            | OFF      | 30   | 65      | 0.4   | 100-120/year    |
| FLEXIBLE_AGGRESSIVE | OFF      | 40   | 60      | 0.5   | 120-150/year    |
"""
