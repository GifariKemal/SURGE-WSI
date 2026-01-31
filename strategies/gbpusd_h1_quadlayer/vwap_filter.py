"""
VWAP (Volume-Weighted Average Price) Filter
=============================================

Implementation of VWAP as a directional filter for GBPUSD H1 strategy.

VWAP Formula:
    VWAP = sum(Typical Price * Volume) / sum(Volume)
    where Typical Price = (High + Low + Close) / 3

Filter Logic:
    - BUY only when price > VWAP (bullish bias)
    - SELL only when price < VWAP (bearish bias)

Daily Reset:
    - VWAP resets at 00:00 UTC each day
    - First few bars of each day use simple average until enough volume

Optional: VWAP Standard Deviation Bands
    - Upper Band = VWAP + N * StdDev
    - Lower Band = VWAP - N * StdDev
    - Can be used to filter trades that are too far from VWAP

Note: Forex uses tick volume as proxy for actual volume.
      Tick volume correlates well with activity/liquidity.

Author: SURIOTA Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


# ============================================================
# VWAP FILTER CONFIGURATION
# ============================================================

# Daily reset time (hour in UTC)
VWAP_RESET_HOUR = 0  # 00:00 UTC (can change to London open 08:00)

# Minimum bars needed before VWAP is valid
VWAP_MIN_BARS = 3

# Standard deviation bands (0 = disabled)
VWAP_STD_BANDS = 2.0  # Trade within 2 std devs of VWAP

# Buffer zone around VWAP (in pips) - too close = no clear direction
VWAP_BUFFER_PIPS = 5.0  # Don't trade if price within 5 pips of VWAP


@dataclass
class VWAPState:
    """State for daily VWAP calculation"""
    sum_pv: float = 0.0       # Sum of (price * volume)
    sum_volume: float = 0.0   # Sum of volume
    sum_pv2: float = 0.0      # Sum of (price^2 * volume) for std dev
    vwap: float = 0.0         # Current VWAP value
    std_dev: float = 0.0      # Current standard deviation
    upper_band: float = 0.0   # Upper band (VWAP + N*std)
    lower_band: float = 0.0   # Lower band (VWAP - N*std)
    bar_count: int = 0        # Bars since reset
    last_reset_date: Optional[datetime] = None


class VWAPFilter:
    """
    Daily VWAP filter for directional bias

    Uses tick volume as proxy for volume in Forex.
    Resets at start of each trading day.
    """

    def __init__(
        self,
        reset_hour: int = VWAP_RESET_HOUR,
        min_bars: int = VWAP_MIN_BARS,
        std_bands: float = VWAP_STD_BANDS,
        buffer_pips: float = VWAP_BUFFER_PIPS,
        pip_size: float = 0.0001
    ):
        """
        Initialize VWAP filter

        Args:
            reset_hour: Hour of day (UTC) to reset VWAP
            min_bars: Minimum bars before VWAP is valid
            std_bands: Standard deviation multiplier for bands (0 = disabled)
            buffer_pips: Buffer zone around VWAP in pips
            pip_size: Pip size for the symbol
        """
        self.reset_hour = reset_hour
        self.min_bars = min_bars
        self.std_bands = std_bands
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size

        self.state = VWAPState()
        self.history: List[Dict] = []  # Track VWAP history for analysis

    def _should_reset(self, current_time: datetime) -> bool:
        """Check if VWAP should reset based on time"""
        if self.state.last_reset_date is None:
            return True

        # Get date portion
        current_date = current_time.date()
        last_date = self.state.last_reset_date.date()

        # Reset if new day
        if current_date > last_date:
            return True

        # Reset if past reset hour and haven't reset today
        if (current_time.hour >= self.reset_hour and
            self.state.last_reset_date.hour < self.reset_hour and
            current_date == last_date):
            return True

        return False

    def _reset(self, current_time: datetime):
        """Reset VWAP for new day"""
        self.state = VWAPState(last_reset_date=current_time)

    def update(
        self,
        high: float,
        low: float,
        close: float,
        volume: float,
        current_time: datetime
    ) -> VWAPState:
        """
        Update VWAP with new bar data

        Args:
            high: Bar high price
            low: Bar low price
            close: Bar close price
            volume: Bar tick volume
            current_time: Bar timestamp

        Returns:
            Current VWAPState with updated values
        """
        # Check for daily reset
        if self._should_reset(current_time):
            self._reset(current_time)

        # Calculate typical price
        typical_price = (high + low + close) / 3.0

        # Handle zero volume (use price average instead)
        if volume <= 0:
            volume = 1.0  # Fallback: treat as equal weight

        # Update cumulative values
        self.state.sum_pv += typical_price * volume
        self.state.sum_volume += volume
        self.state.sum_pv2 += (typical_price ** 2) * volume
        self.state.bar_count += 1

        # Calculate VWAP
        if self.state.sum_volume > 0:
            self.state.vwap = self.state.sum_pv / self.state.sum_volume

            # Calculate standard deviation
            # Var = E[X^2] - E[X]^2
            # where E[X] = VWAP, E[X^2] = sum(P^2*V) / sum(V)
            mean_pv2 = self.state.sum_pv2 / self.state.sum_volume
            variance = mean_pv2 - (self.state.vwap ** 2)

            if variance > 0:
                self.state.std_dev = np.sqrt(variance)
            else:
                self.state.std_dev = 0.0

            # Calculate bands
            if self.std_bands > 0 and self.state.std_dev > 0:
                self.state.upper_band = self.state.vwap + (self.std_bands * self.state.std_dev)
                self.state.lower_band = self.state.vwap - (self.std_bands * self.state.std_dev)
            else:
                self.state.upper_band = self.state.vwap
                self.state.lower_band = self.state.vwap

        # Record history
        self.history.append({
            'time': current_time,
            'vwap': self.state.vwap,
            'std_dev': self.state.std_dev,
            'upper_band': self.state.upper_band,
            'lower_band': self.state.lower_band,
            'bar_count': self.state.bar_count,
            'close': close
        })

        return self.state

    def check_trade(
        self,
        direction: str,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Check if trade direction is aligned with VWAP

        Args:
            direction: 'BUY' or 'SELL'
            current_price: Current price

        Returns:
            (can_trade, reason) tuple
        """
        # Not enough bars yet
        if self.state.bar_count < self.min_bars:
            return True, "VWAP_WARMUP"  # Allow trading during warmup

        # Check buffer zone
        price_distance_pips = abs(current_price - self.state.vwap) / self.pip_size
        if price_distance_pips < self.buffer_pips:
            return False, "VWAP_BUFFER_ZONE"

        # Check directional alignment
        if direction == 'BUY':
            if current_price <= self.state.vwap:
                return False, "VWAP_BEARISH_BIAS"
            # Optional: Check if not too far above VWAP
            if self.std_bands > 0 and current_price > self.state.upper_band:
                return False, "VWAP_ABOVE_UPPER_BAND"
            return True, "VWAP_BULLISH_ALIGNED"

        elif direction == 'SELL':
            if current_price >= self.state.vwap:
                return False, "VWAP_BULLISH_BIAS"
            # Optional: Check if not too far below VWAP
            if self.std_bands > 0 and current_price < self.state.lower_band:
                return False, "VWAP_BELOW_LOWER_BAND"
            return True, "VWAP_BEARISH_ALIGNED"

        return True, "VWAP_UNKNOWN_DIRECTION"

    def get_stats(self) -> Dict:
        """Get VWAP filter statistics"""
        if not self.history:
            return {}

        # Count days
        dates = set(h['time'].date() for h in self.history)

        # Calculate average daily bars
        avg_bars = len(self.history) / len(dates) if dates else 0

        return {
            'total_updates': len(self.history),
            'unique_days': len(dates),
            'avg_bars_per_day': avg_bars,
            'current_vwap': self.state.vwap,
            'current_std_dev': self.state.std_dev,
            'current_bar_count': self.state.bar_count
        }


def calculate_vwap_series(
    df: pd.DataFrame,
    col_map: dict,
    reset_hour: int = 0
) -> pd.DataFrame:
    """
    Calculate VWAP series for entire DataFrame

    Args:
        df: DataFrame with OHLCV data
        col_map: Column name mapping
        reset_hour: Hour to reset VWAP daily

    Returns:
        DataFrame with VWAP columns added
    """
    h, l, c = col_map['high'], col_map['low'], col_map['close']
    v = col_map.get('volume', 'Volume')

    # Initialize columns
    df = df.copy()
    df['vwap'] = np.nan
    df['vwap_std'] = np.nan
    df['vwap_upper'] = np.nan
    df['vwap_lower'] = np.nan

    # Track daily cumulative values
    sum_pv = 0.0
    sum_volume = 0.0
    sum_pv2 = 0.0
    current_date = None

    for i, (idx, row) in enumerate(df.iterrows()):
        # Get timestamp
        if isinstance(idx, pd.Timestamp):
            ts = idx
        else:
            ts = pd.to_datetime(idx)

        # Check for daily reset
        row_date = ts.date()
        if current_date is None or row_date != current_date:
            # Reset for new day
            sum_pv = 0.0
            sum_volume = 0.0
            sum_pv2 = 0.0
            current_date = row_date

        # Calculate typical price
        typical_price = (row[h] + row[l] + row[c]) / 3.0

        # Get volume (fallback to 1 if not available or zero)
        volume = row.get(v, 1)
        if pd.isna(volume) or volume <= 0:
            volume = 1.0

        # Update cumulative
        sum_pv += typical_price * volume
        sum_volume += volume
        sum_pv2 += (typical_price ** 2) * volume

        # Calculate VWAP
        if sum_volume > 0:
            vwap = sum_pv / sum_volume
            mean_pv2 = sum_pv2 / sum_volume
            variance = mean_pv2 - (vwap ** 2)
            std_dev = np.sqrt(variance) if variance > 0 else 0

            df.loc[idx, 'vwap'] = vwap
            df.loc[idx, 'vwap_std'] = std_dev
            df.loc[idx, 'vwap_upper'] = vwap + (2 * std_dev)
            df.loc[idx, 'vwap_lower'] = vwap - (2 * std_dev)

    return df


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_vwap_signal(
    current_price: float,
    vwap: float,
    buffer_pips: float = 5.0,
    pip_size: float = 0.0001
) -> str:
    """
    Get simple VWAP signal based on price position

    Returns:
        'BULLISH' if price > VWAP + buffer
        'BEARISH' if price < VWAP - buffer
        'NEUTRAL' if price within buffer
    """
    buffer_price = buffer_pips * pip_size

    if current_price > vwap + buffer_price:
        return 'BULLISH'
    elif current_price < vwap - buffer_price:
        return 'BEARISH'
    else:
        return 'NEUTRAL'
