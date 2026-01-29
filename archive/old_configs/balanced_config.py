"""Balanced Trading Configuration
=================================

Optimized configuration that balances:
- Trade frequency (more opportunities)
- Safety (minimal losing months)
- Flexibility (works across market conditions)

Comparison:
- ORIGINAL: 130+ trades/year, 4 losing months, +60% return
- ZERO_LOSS: 17-35 trades/year, 0-1 losing months, +25% return
- BALANCED: 80-100 trades/year, 1-2 losing months, +40-50% return (target)

Author: SURIOTA Team
Date: 2026-01-29
"""

# =============================================================================
# BALANCED CONFIGURATION OPTIONS
# =============================================================================

# Option A: Conservative Balanced (fewer losing months)
BALANCED_CONSERVATIVE = {
    'name': 'BALANCED_CONSERVATIVE',
    'max_sl_pips': 25.0,           # Allow trades with SL up to 25 pips (was 10)
    'max_loss_per_trade_pct': 0.3, # Max 0.3% loss per trade (was 0.1%)
    'min_quality_score': 68.0,     # Slightly above original 65 (was 75)

    # Expected: ~60-80 trades/year, 1-2 losing months, +35-45% return
}

# Option B: Moderate Balanced (good balance)
BALANCED_MODERATE = {
    'name': 'BALANCED_MODERATE',
    'max_sl_pips': 30.0,           # Allow trades with SL up to 30 pips
    'max_loss_per_trade_pct': 0.4, # Max 0.4% loss per trade
    'min_quality_score': 65.0,     # Original quality threshold

    # Expected: ~80-100 trades/year, 2-3 losing months, +40-50% return
}

# Option C: Active Trading (more trades, slightly more risk)
BALANCED_ACTIVE = {
    'name': 'BALANCED_ACTIVE',
    'max_sl_pips': 40.0,           # Allow trades with SL up to 40 pips
    'max_loss_per_trade_pct': 0.5, # Max 0.5% loss per trade
    'min_quality_score': 60.0,     # Lower quality threshold

    # Expected: ~100-130 trades/year, 3-4 losing months, +50-60% return
}

# =============================================================================
# RECOMMENDED CONFIG (Best Balance)
# =============================================================================

BALANCED_CONFIG = {
    'name': 'BALANCED_RECOMMENDED',

    # Risk Management - Balanced
    'max_sl_pips': 25.0,           # Allow most SMC setups (typical SL 15-25 pips)
    'max_loss_per_trade_pct': 0.3, # Max 0.3% loss = $30 on $10k account
    'max_lot_size': 0.5,           # Maximum lot size

    # Entry Quality - Balanced
    'min_quality_score': 68.0,     # Require at least MSS or good sweep

    # Trading Parameters
    'skip_december': True,         # Skip December trading
    'pip_value': 10.0,             # Pip value per standard lot
    'spread_pips': 1.5,            # Spread in pips

    # Filters
    'use_killzone': True,          # Use kill zone time filter
    'use_trend_filter': True,      # Use trend alignment filter
    'use_relaxed_filter': False,   # Don't use relaxed entry
    'use_hybrid_mode': False,      # Don't use hybrid mode
    'use_choppiness_filter': False, # Don't use choppiness filter
}

# =============================================================================
# WHY THESE VALUES?
# =============================================================================
"""
max_sl_pips = 25:
- Original SMC setups typically have SL 15-30 pips
- SL=10 filters 70-80% of valid setups
- SL=25 allows most quality setups while filtering extremes

max_loss_per_trade_pct = 0.3%:
- Original 0.8% can cause -$80 loss on $10k
- Zero Loss 0.1% limits to -$10, but also limits lot size
- 0.3% = max -$30 per trade, good balance

min_quality_score = 68:
- 65 allows MSS-only setups (quality=70)
- 68 ensures MSS or decent sweep
- 75 requires sweep+MSS (too restrictive)

Expected Results:
- Trade frequency: 6-10 trades/month (vs 1-3 with zero loss)
- Monthly: Most months positive, 1-2 small losing months/year
- Return: 35-50% annually (vs 25% with zero loss)
"""

# =============================================================================
# COMPARISON TABLE
# =============================================================================
"""
| Config           | SL   | Loss% | Quality | Trades/Yr | Losing | Return |
|------------------|------|-------|---------|-----------|--------|--------|
| ORIGINAL         | 50   | 0.8   | 65      | 130       | 4      | 60%    |
| ZERO_LOSS        | 10   | 0.1   | 75      | 35        | 0-1    | 25%    |
| BALANCED_CONS    | 25   | 0.3   | 68      | 70        | 1-2    | 40%    |
| BALANCED_MOD     | 30   | 0.4   | 65      | 90        | 2-3    | 50%    |
| BALANCED_ACTIVE  | 40   | 0.5   | 60      | 120       | 3-4    | 55%    |
"""
