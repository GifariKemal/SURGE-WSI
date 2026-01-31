"""
Fair Value Gap (FVG) Detection Module
=====================================

FVG = price imbalance from impulsive moves that creates gaps in price.
Price often returns to fill these gaps, making them good entry zones.

Bullish FVG: candle 3's low > candle 1's high (gap to the upside)
    - Price moved up so fast it left a gap
    - Expect price to return DOWN to fill this gap (confirmation for BUY)

Bearish FVG: candle 3's high < candle 1's low (gap to the downside)
    - Price moved down so fast it left a gap
    - Expect price to return UP to fill this gap (confirmation for SELL)

Author: SURIOTA Team
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd


# ============================================================
# FVG CONFIGURATION
# ============================================================
FVG_LOOKBACK = 50              # How many bars to look back for FVG zones (extended for more zones)
FVG_MIN_GAP_PIPS = 3           # Minimum gap size in pips to be valid (lowered to catch smaller gaps)
FVG_MAX_AGE = 100              # Maximum age (bars) for FVG zone to be valid (extended)
FVG_CONFIRMATION_PIPS = 50     # How close price must be to FVG zone for confirmation (widened)
PIP_SIZE = 0.0001              # Default pip size for GBPUSD


@dataclass
class FVGZone:
    """Fair Value Gap zone"""
    direction: str          # 'BULLISH' or 'BEARISH'
    upper_price: float      # Upper boundary of the gap
    lower_price: float      # Lower boundary of the gap
    gap_size_pips: float    # Size of gap in pips
    created_idx: int        # Bar index when FVG was created
    quality: float          # Quality score (based on gap size and confirmation)
    filled: bool = False    # Whether the gap has been filled


def detect_fvg(df: pd.DataFrame, col_map: dict, current_idx: int,
               lookback: int = FVG_LOOKBACK,
               min_gap_pips: float = FVG_MIN_GAP_PIPS,
               max_age: int = FVG_MAX_AGE,
               pip_size: float = PIP_SIZE) -> List[FVGZone]:
    """
    Detect Fair Value Gap (FVG) zones.

    FVG occurs when there's a gap between candle 1's range and candle 3's range,
    indicating an imbalance that price often returns to fill.

    Bullish FVG: candle 3's low > candle 1's high (gap to the upside)
        - Price moved up so fast it left a gap
        - Expect price to return DOWN to fill this gap (confirmation for BUY)

    Bearish FVG: candle 3's high < candle 1's low (gap to the downside)
        - Price moved down so fast it left a gap
        - Expect price to return UP to fill this gap (confirmation for SELL)

    Args:
        df: DataFrame with OHLC data
        col_map: Column mapping for OHLC
        current_idx: Current bar index
        lookback: How many bars to look back for FVG zones
        min_gap_pips: Minimum gap size in pips to be valid
        max_age: Maximum age (bars) for FVG zone to be valid
        pip_size: Pip size for the symbol

    Returns:
        List of FVGZone objects with direction, price range, and quality
    """
    fvg_zones = []

    if current_idx < 3:
        return fvg_zones

    h, l = col_map['high'], col_map['low']

    # Look back through recent bars for FVG patterns
    start_idx = max(3, current_idx - lookback)

    for i in range(start_idx, current_idx - 1):  # Need at least 3 bars: i-2, i-1, i
        if i < 2:
            continue

        # Candle 1 (oldest), Candle 2 (middle - impulsive), Candle 3 (newest)
        candle_1_high = df[h].iloc[i - 2]
        candle_1_low = df[l].iloc[i - 2]
        # candle_2 is the impulsive move (middle)
        candle_3_high = df[h].iloc[i]
        candle_3_low = df[l].iloc[i]

        # Check if FVG zone has been filled by subsequent price action
        zone_filled = False

        # BULLISH FVG: candle 3's low > candle 1's high (gap above)
        # This creates a gap that price may return to fill
        if candle_3_low > candle_1_high:
            gap_size_pips = (candle_3_low - candle_1_high) / pip_size

            if gap_size_pips >= min_gap_pips:
                # Check if filled by subsequent bars
                for fill_idx in range(i + 1, current_idx + 1):
                    if df[l].iloc[fill_idx] <= candle_1_high:
                        zone_filled = True
                        break

                # Check zone age
                zone_age = current_idx - i
                if zone_age <= max_age and not zone_filled:
                    # Quality based on gap size (bigger = more significant)
                    quality = min(100, 50 + (gap_size_pips * 2))

                    fvg_zones.append(FVGZone(
                        direction='BULLISH',
                        upper_price=candle_3_low,
                        lower_price=candle_1_high,
                        gap_size_pips=gap_size_pips,
                        created_idx=i,
                        quality=quality,
                        filled=zone_filled
                    ))

        # BEARISH FVG: candle 3's high < candle 1's low (gap below)
        # This creates a gap that price may return to fill
        if candle_3_high < candle_1_low:
            gap_size_pips = (candle_1_low - candle_3_high) / pip_size

            if gap_size_pips >= min_gap_pips:
                # Check if filled by subsequent bars
                for fill_idx in range(i + 1, current_idx + 1):
                    if df[h].iloc[fill_idx] >= candle_1_low:
                        zone_filled = True
                        break

                # Check zone age
                zone_age = current_idx - i
                if zone_age <= max_age and not zone_filled:
                    # Quality based on gap size (bigger = more significant)
                    quality = min(100, 50 + (gap_size_pips * 2))

                    fvg_zones.append(FVGZone(
                        direction='BEARISH',
                        upper_price=candle_1_low,
                        lower_price=candle_3_high,
                        gap_size_pips=gap_size_pips,
                        created_idx=i,
                        quality=quality,
                        filled=zone_filled
                    ))

    return fvg_zones


def check_fvg_confirmation(price: float, direction: str, fvg_zones: List[FVGZone],
                           confirmation_pips: float = FVG_CONFIRMATION_PIPS,
                           pip_size: float = PIP_SIZE) -> Tuple[bool, Optional[FVGZone]]:
    """
    Check if current price is near an FVG zone that confirms the trade direction.

    For BUY signals: Look for BULLISH FVG zones (price returning to fill gap from above)
    For SELL signals: Look for BEARISH FVG zones (price returning to fill gap from below)

    Args:
        price: Current price
        direction: Trade direction ('BUY' or 'SELL')
        fvg_zones: List of active FVG zones
        confirmation_pips: How close price must be to zone for confirmation
        pip_size: Pip size for the symbol

    Returns:
        Tuple of (is_confirmed, matching_fvg_zone)
    """
    if not fvg_zones:
        return False, None

    confirmation_distance = confirmation_pips * pip_size

    for zone in fvg_zones:
        # BUY confirmation: price is near a BULLISH FVG zone
        # (price dropped into the FVG zone, good spot to buy)
        if direction == 'BUY' and zone.direction == 'BULLISH':
            # Check if price is within or near the FVG zone
            distance_to_zone = min(
                abs(price - zone.upper_price),
                abs(price - zone.lower_price)
            )
            # Also confirm if price is inside the zone
            is_inside_zone = zone.lower_price <= price <= zone.upper_price

            if is_inside_zone or distance_to_zone <= confirmation_distance:
                return True, zone

        # SELL confirmation: price is near a BEARISH FVG zone
        # (price rallied into the FVG zone, good spot to sell)
        elif direction == 'SELL' and zone.direction == 'BEARISH':
            # Check if price is within or near the FVG zone
            distance_to_zone = min(
                abs(price - zone.upper_price),
                abs(price - zone.lower_price)
            )
            # Also confirm if price is inside the zone
            is_inside_zone = zone.lower_price <= price <= zone.upper_price

            if is_inside_zone or distance_to_zone <= confirmation_distance:
                return True, zone

    return False, None
