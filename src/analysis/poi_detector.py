"""POI Detector - Order Blocks + FVG using smartmoneyconcepts
=============================================================

Layer 4 of 6-Layer Architecture

Function: Detect Points of Interest (POI) for potential entries.
Uses smartmoneyconcepts library for:
- Order Blocks (OB)
- Fair Value Gaps (FVG)
- Break of Structure (BOS/CHOCH)
- Swing Highs/Lows

Output: List of active POIs with quality scores

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime, timezone
from loguru import logger

try:
    import sys
    import io
    # Suppress smartmoneyconcepts star emoji print (encoding issue on Windows)
    _original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    import smartmoneyconcepts as smc_module
    sys.stdout = _original_stdout
    # The library exports a class called 'smc' inside the module
    smc = smc_module.smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    smc = None
    logger.warning("smartmoneyconcepts not installed. Install with: pip install smartmoneyconcepts")
except Exception as e:
    sys.stdout = _original_stdout
    SMC_AVAILABLE = False
    smc = None
    logger.warning(f"smartmoneyconcepts import error: {e}")


class POIType(Enum):
    """Types of Points of Interest"""
    ORDER_BLOCK_BULLISH = "OB_BULL"
    ORDER_BLOCK_BEARISH = "OB_BEAR"
    FVG_BULLISH = "FVG_BULL"
    FVG_BEARISH = "FVG_BEAR"
    BREAKER_BULLISH = "BREAKER_BULL"
    BREAKER_BEARISH = "BREAKER_BEAR"


@dataclass
class OrderBlock:
    """Order Block structure"""
    id: str
    poi_type: POIType
    top: float
    bottom: float
    volume: float = 0
    strength: float = 0.0
    mitigated: bool = False
    created_at: datetime = None
    bar_index: int = 0

    @property
    def mid(self) -> float:
        """Middle of the order block"""
        return (self.top + self.bottom) / 2

    @property
    def size_pips(self) -> float:
        """Size in pips"""
        return abs(self.top - self.bottom) / 0.0001

    @property
    def direction(self) -> str:
        """Direction for trading"""
        if "BULL" in self.poi_type.value:
            return "BUY"
        return "SELL"

    def contains_price(self, price: float) -> bool:
        """Check if price is within the order block"""
        return self.bottom <= price <= self.top

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.poi_type.value,
            "top": self.top,
            "bottom": self.bottom,
            "mid": self.mid,
            "size_pips": self.size_pips,
            "strength": self.strength,
            "mitigated": self.mitigated,
            "direction": self.direction,
            "bar_index": self.bar_index,
        }


@dataclass
class FairValueGap:
    """Fair Value Gap structure"""
    id: str
    poi_type: POIType
    high: float
    low: float
    filled: bool = False
    fill_percentage: float = 0.0
    created_at: datetime = None
    bar_index: int = 0

    @property
    def mid(self) -> float:
        """Middle of the FVG"""
        return (self.high + self.low) / 2

    @property
    def size_pips(self) -> float:
        """Size in pips"""
        return abs(self.high - self.low) / 0.0001

    @property
    def direction(self) -> str:
        """Direction for trading"""
        if "BULL" in self.poi_type.value:
            return "BUY"
        return "SELL"

    def contains_price(self, price: float) -> bool:
        """Check if price is within the FVG"""
        return self.low <= price <= self.high

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.poi_type.value,
            "high": self.high,
            "low": self.low,
            "mid": self.mid,
            "size_pips": self.size_pips,
            "filled": self.filled,
            "fill_percentage": self.fill_percentage,
            "direction": self.direction,
            "bar_index": self.bar_index,
        }


@dataclass
class POIResult:
    """Combined POI detection result"""
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FairValueGap] = field(default_factory=list)
    swing_highs: List[Dict] = field(default_factory=list)
    swing_lows: List[Dict] = field(default_factory=list)
    bos_choch: List[Dict] = field(default_factory=list)

    @property
    def bullish_pois(self) -> List[Dict]:
        """Get all bullish POIs"""
        pois = []
        for ob in self.order_blocks:
            if ob.poi_type == POIType.ORDER_BLOCK_BULLISH and not ob.mitigated:
                pois.append(ob.to_dict())
        for fvg in self.fvgs:
            if fvg.poi_type == POIType.FVG_BULLISH and not fvg.filled:
                pois.append(fvg.to_dict())
        return pois

    @property
    def bearish_pois(self) -> List[Dict]:
        """Get all bearish POIs"""
        pois = []
        for ob in self.order_blocks:
            if ob.poi_type == POIType.ORDER_BLOCK_BEARISH and not ob.mitigated:
                pois.append(ob.to_dict())
        for fvg in self.fvgs:
            if fvg.poi_type == POIType.FVG_BEARISH and not fvg.filled:
                pois.append(fvg.to_dict())
        return pois

    def get_nearest_poi(self, price: float, direction: str) -> Optional[Dict]:
        """Get nearest POI to current price

        Args:
            price: Current price
            direction: 'BUY' or 'SELL'

        Returns:
            Nearest POI dict or None
        """
        if direction == "BUY":
            pois = self.bullish_pois
            # For BUY, look for POIs below current price
            valid_pois = [p for p in pois if p['mid'] < price]
        else:
            pois = self.bearish_pois
            # For SELL, look for POIs above current price
            valid_pois = [p for p in pois if p['mid'] > price]

        if not valid_pois:
            return None

        # Sort by distance to price
        valid_pois.sort(key=lambda p: abs(p['mid'] - price))
        return valid_pois[0]

    def price_at_poi(self, price: float, direction: str, tolerance_pips: float = 15.0) -> Tuple[bool, Optional[Dict]]:
        """Check if price is currently at a POI

        Args:
            price: Current price
            direction: 'BUY' or 'SELL'
            tolerance_pips: Buffer around POI in pips (default 15 pips)

        Returns:
            Tuple of (at_poi: bool, poi_info: Optional[Dict])
        """
        pois = self.bullish_pois if direction == "BUY" else self.bearish_pois
        tolerance = tolerance_pips * 0.0001  # Convert pips to price

        for poi in pois:
            low = poi.get('bottom', poi.get('low', 0))
            high = poi.get('top', poi.get('high', 0))

            # Extend zone by tolerance
            zone_low = low - tolerance
            zone_high = high + tolerance

            # For BUY, price should be at or below the POI zone
            # For SELL, price should be at or above the POI zone
            if direction == "BUY":
                # Bullish POI - we're looking for price at demand zone (below current)
                # Accept if price is within or slightly above the zone
                if zone_low <= price <= zone_high + tolerance:
                    return True, poi
            else:
                # Bearish POI - we're looking for price at supply zone (above current)
                # Accept if price is within or slightly below the zone
                if zone_low - tolerance <= price <= zone_high:
                    return True, poi

        return False, None


class POIDetector:
    """Point of Interest detector using smartmoneyconcepts

    Detects:
    - Order Blocks (supply/demand zones)
    - Fair Value Gaps (imbalance zones)
    - Break of Structure (trend confirmation)
    """

    def __init__(
        self,
        swing_length: int = 10,
        ob_min_strength: float = 0.4,  # Lowered from 0.6 for more OBs
        fvg_min_pips: float = 2.0,  # Lowered from 3.0 for more FVGs
        max_poi_age_bars: int = 50,  # Reduced from 100 for fresher POIs
        use_order_blocks: bool = True,
        use_fvg: bool = True,
        use_bos: bool = True
    ):
        """Initialize POI Detector

        Args:
            swing_length: Bars for swing detection
            ob_min_strength: Minimum order block strength
            fvg_min_pips: Minimum FVG size in pips
            max_poi_age_bars: Maximum age for valid POI
            use_order_blocks: Enable order block detection
            use_fvg: Enable FVG detection
            use_bos: Enable break of structure detection
        """
        self.swing_length = swing_length
        self.ob_min_strength = ob_min_strength
        self.fvg_min_pips = fvg_min_pips
        self.max_poi_age_bars = max_poi_age_bars
        self.use_order_blocks = use_order_blocks
        self.use_fvg = use_fvg
        self.use_bos = use_bos

        self._last_result: Optional[POIResult] = None
        self._poi_counter = 0

    def _generate_id(self) -> str:
        """Generate unique POI ID"""
        self._poi_counter += 1
        return f"POI_{self._poi_counter}_{datetime.now(timezone.utc).strftime('%H%M%S')}"

    def detect(self, df: pd.DataFrame) -> POIResult:
        """Detect all POIs on the given data

        Args:
            df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            POIResult with all detected POIs
        """
        result = POIResult()

        if len(df) < self.swing_length * 2:
            return result

        # Normalize column names
        df = self._normalize_columns(df)

        if not SMC_AVAILABLE:
            # Fallback to simple detection
            return self._fallback_detection(df)

        try:
            # Detect swing highs/lows first (required for order blocks)
            swing_hl = smc.swing_highs_lows(df, self.swing_length)

            # Detect Order Blocks
            if self.use_order_blocks:
                ob_result = smc.ob(df, swing_hl)
                result.order_blocks = self._parse_order_blocks(df, ob_result)

            # Detect Fair Value Gaps
            if self.use_fvg:
                fvg_result = smc.fvg(df)
                result.fvgs = self._parse_fvgs(df, fvg_result)

            # Detect Break of Structure
            if self.use_bos:
                bos_result = smc.bos_choch(df, swing_hl)
                result.bos_choch = self._parse_bos_choch(df, bos_result)

            # Parse swing highs/lows
            result.swing_highs, result.swing_lows = self._parse_swings(df, swing_hl)

            # If smartmoneyconcepts didn't find any valid POIs, use fallback
            if len(result.order_blocks) == 0 and len(result.fvgs) == 0:
                logger.debug("No POIs from smartmoneyconcepts, using fallback detection")
                fallback_result = self._fallback_detection(df)
                result.order_blocks = fallback_result.order_blocks
                result.fvgs = fallback_result.fvgs
                result.swing_highs = fallback_result.swing_highs
                result.swing_lows = fallback_result.swing_lows

        except Exception as e:
            logger.error(f"POI detection failed: {e}")
            return self._fallback_detection(df)

        self._last_result = result
        return result

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase"""
        df = df.copy()
        col_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }
        df.rename(columns=col_map, inplace=True)
        return df

    def _parse_order_blocks(self, df: pd.DataFrame, ob_result) -> List[OrderBlock]:
        """Parse order blocks from smartmoneyconcepts result"""
        order_blocks = []
        current_bar = len(df) - 1

        if ob_result is None:
            return order_blocks

        # smartmoneyconcepts returns DataFrame with OB info
        try:
            for i in range(len(ob_result)):
                row = ob_result.iloc[i]

                # Check if this is a valid OB
                ob_type = row.get('OB', 0)
                if ob_type == 0 or pd.isna(ob_type):
                    continue

                ob_top = row.get('Top', row.get('OBTop', 0))
                ob_bottom = row.get('Bottom', row.get('OBBottom', 0))
                ob_volume = row.get('OBVolume', row.get('Volume', 0))
                mitigated = row.get('MitigatedIndex', -1) > 0

                # Check age
                if current_bar - i > self.max_poi_age_bars:
                    continue

                # Determine type
                if ob_type == 1:  # Bullish OB
                    poi_type = POIType.ORDER_BLOCK_BULLISH
                else:  # Bearish OB
                    poi_type = POIType.ORDER_BLOCK_BEARISH

                # Calculate strength based on volume and size
                size_pips = abs(ob_top - ob_bottom) / 0.0001
                strength = min(1.0, size_pips / 20)  # Normalize

                # Validate values - must be valid price levels (not 0 or NaN)
                if ob_top <= 0 or ob_bottom <= 0 or pd.isna(ob_top) or pd.isna(ob_bottom):
                    logger.debug(f"Skipping OB with invalid values: top={ob_top}, bottom={ob_bottom}")
                    continue

                if strength >= self.ob_min_strength or not mitigated:
                    order_blocks.append(OrderBlock(
                        id=self._generate_id(),
                        poi_type=poi_type,
                        top=ob_top,
                        bottom=ob_bottom,
                        volume=ob_volume if not pd.isna(ob_volume) else 0,
                        strength=strength,
                        mitigated=mitigated,
                        bar_index=i,
                        created_at=df.index[i] if hasattr(df.index[i], 'strftime') else None
                    ))

        except Exception as e:
            logger.warning(f"Error parsing order blocks: {e}")

        return order_blocks

    def _parse_fvgs(self, df: pd.DataFrame, fvg_result) -> List[FairValueGap]:
        """Parse FVGs from smartmoneyconcepts result"""
        fvgs = []
        current_bar = len(df) - 1

        if fvg_result is None:
            return fvgs

        try:
            for i in range(len(fvg_result)):
                row = fvg_result.iloc[i]

                fvg_type = row.get('FVG', 0)
                if fvg_type == 0 or pd.isna(fvg_type):
                    continue

                fvg_top = row.get('FVGTop', row.get('Top', 0))
                fvg_bottom = row.get('FVGBottom', row.get('Bottom', 0))
                mitigated_idx = row.get('MitigatedIndex', -1)

                # Check age
                if current_bar - i > self.max_poi_age_bars:
                    continue

                # Validate values - must be valid price levels
                if fvg_top <= 0 or fvg_bottom <= 0 or pd.isna(fvg_top) or pd.isna(fvg_bottom):
                    logger.debug(f"Skipping FVG with invalid values: top={fvg_top}, bottom={fvg_bottom}")
                    continue

                # Check size
                size_pips = abs(fvg_top - fvg_bottom) / 0.0001
                if size_pips < self.fvg_min_pips:
                    continue

                # Determine type
                if fvg_type == 1:  # Bullish FVG
                    poi_type = POIType.FVG_BULLISH
                else:  # Bearish FVG
                    poi_type = POIType.FVG_BEARISH

                filled = mitigated_idx > 0

                fvgs.append(FairValueGap(
                    id=self._generate_id(),
                    poi_type=poi_type,
                    high=fvg_top,
                    low=fvg_bottom,
                    filled=filled,
                    bar_index=i,
                    created_at=df.index[i] if hasattr(df.index[i], 'strftime') else None
                ))

        except Exception as e:
            logger.warning(f"Error parsing FVGs: {e}")

        return fvgs

    def _parse_bos_choch(self, df: pd.DataFrame, bos_result) -> List[Dict]:
        """Parse BOS/CHOCH from smartmoneyconcepts result"""
        structures = []

        if bos_result is None:
            return structures

        try:
            for i in range(len(bos_result)):
                row = bos_result.iloc[i]

                bos_type = row.get('BOS', 0)
                choch_type = row.get('CHOCH', 0)

                if bos_type != 0 and not pd.isna(bos_type):
                    structures.append({
                        'type': 'BOS',
                        'direction': 'BULLISH' if bos_type == 1 else 'BEARISH',
                        'level': row.get('Level', df['close'].iloc[i]),
                        'bar_index': i,
                        'time': df.index[i] if hasattr(df.index[i], 'strftime') else None
                    })

                if choch_type != 0 and not pd.isna(choch_type):
                    structures.append({
                        'type': 'CHOCH',
                        'direction': 'BULLISH' if choch_type == 1 else 'BEARISH',
                        'level': row.get('Level', df['close'].iloc[i]),
                        'bar_index': i,
                        'time': df.index[i] if hasattr(df.index[i], 'strftime') else None
                    })

        except Exception as e:
            logger.warning(f"Error parsing BOS/CHOCH: {e}")

        return structures

    def _parse_swings(self, df: pd.DataFrame, swing_result) -> Tuple[List[Dict], List[Dict]]:
        """Parse swing highs and lows"""
        swing_highs = []
        swing_lows = []

        if swing_result is None:
            return swing_highs, swing_lows

        try:
            for i in range(len(swing_result)):
                row = swing_result.iloc[i]

                swing_type = row.get('HighLow', 0)

                if swing_type == 1:  # Swing High
                    swing_highs.append({
                        'price': df['high'].iloc[i],
                        'bar_index': i,
                        'time': df.index[i] if hasattr(df.index[i], 'strftime') else None
                    })
                elif swing_type == -1:  # Swing Low
                    swing_lows.append({
                        'price': df['low'].iloc[i],
                        'bar_index': i,
                        'time': df.index[i] if hasattr(df.index[i], 'strftime') else None
                    })

        except Exception as e:
            logger.warning(f"Error parsing swings: {e}")

        return swing_highs, swing_lows

    def _fallback_detection(self, df: pd.DataFrame) -> POIResult:
        """Fallback detection when smartmoneyconcepts not available"""
        result = POIResult()
        current_bar = len(df) - 1

        # Simple swing detection
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        close_col = 'close' if 'close' in df.columns else 'Close'
        open_col = 'open' if 'open' in df.columns else 'Open'

        highs = df[high_col].values
        lows = df[low_col].values
        closes = df[close_col].values
        opens = df[open_col].values

        window = self.swing_length

        for i in range(window, len(df) - window):
            # Check age
            if current_bar - i > self.max_poi_age_bars:
                continue

            # Swing High
            if highs[i] == max(highs[i-window:i+window+1]):
                result.swing_highs.append({
                    'price': highs[i],
                    'bar_index': i,
                })

            # Swing Low
            if lows[i] == min(lows[i-window:i+window+1]):
                result.swing_lows.append({
                    'price': lows[i],
                    'bar_index': i,
                })

        # Simple Order Block detection (last candle before impulsive move)
        for i in range(3, len(df)):
            # Check age
            if current_bar - i > self.max_poi_age_bars:
                continue

            # Bullish OB: Strong down candle followed by impulsive up move
            if closes[i-2] < opens[i-2]:  # Down candle (potential OB)
                # Check for impulsive up move after
                if closes[i-1] > opens[i-1] and closes[i] > closes[i-1]:
                    # Calculate impulsiveness
                    ob_body = abs(closes[i-2] - opens[i-2])
                    move_size = closes[i] - opens[i-2]

                    if move_size > ob_body * 1.5:  # Impulsive move
                        result.order_blocks.append(OrderBlock(
                            id=self._generate_id(),
                            poi_type=POIType.ORDER_BLOCK_BULLISH,
                            top=opens[i-2],
                            bottom=lows[i-2],
                            strength=min(1.0, move_size / ob_body / 2),
                            mitigated=False,
                            bar_index=i - 2
                        ))

            # Bearish OB: Strong up candle followed by impulsive down move
            if closes[i-2] > opens[i-2]:  # Up candle (potential OB)
                # Check for impulsive down move after
                if closes[i-1] < opens[i-1] and closes[i] < closes[i-1]:
                    # Calculate impulsiveness
                    ob_body = abs(closes[i-2] - opens[i-2])
                    move_size = opens[i-2] - closes[i]

                    if move_size > ob_body * 1.5:  # Impulsive move
                        result.order_blocks.append(OrderBlock(
                            id=self._generate_id(),
                            poi_type=POIType.ORDER_BLOCK_BEARISH,
                            top=highs[i-2],
                            bottom=opens[i-2],
                            strength=min(1.0, move_size / ob_body / 2),
                            mitigated=False,
                            bar_index=i - 2
                        ))

        # Simple FVG detection
        for i in range(2, len(df)):
            # Check age
            if current_bar - i > self.max_poi_age_bars:
                continue

            c0_high = highs[i-2]
            c0_low = lows[i-2]
            c2_high = highs[i]
            c2_low = lows[i]

            # Bullish FVG (gap up)
            if c2_low > c0_high:
                gap_pips = (c2_low - c0_high) / 0.0001
                if gap_pips >= self.fvg_min_pips:
                    result.fvgs.append(FairValueGap(
                        id=self._generate_id(),
                        poi_type=POIType.FVG_BULLISH,
                        high=c2_low,
                        low=c0_high,
                        bar_index=i - 1
                    ))

            # Bearish FVG (gap down)
            if c2_high < c0_low:
                gap_pips = (c0_low - c2_high) / 0.0001
                if gap_pips >= self.fvg_min_pips:
                    result.fvgs.append(FairValueGap(
                        id=self._generate_id(),
                        poi_type=POIType.FVG_BEARISH,
                        high=c0_low,
                        low=c2_high,
                        bar_index=i - 1
                    ))

        self._last_result = result
        return result

    def update_mitigation(self, df: pd.DataFrame):
        """Update POI mitigation status based on price action

        Args:
            df: Updated OHLCV data
        """
        if self._last_result is None:
            return

        current_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]

        # Check order blocks
        for ob in self._last_result.order_blocks:
            if not ob.mitigated:
                if ob.poi_type == POIType.ORDER_BLOCK_BULLISH:
                    # Bullish OB mitigated if price goes below it
                    if current_price < ob.bottom:
                        ob.mitigated = True
                else:
                    # Bearish OB mitigated if price goes above it
                    if current_price > ob.top:
                        ob.mitigated = True

        # Check FVGs
        for fvg in self._last_result.fvgs:
            if not fvg.filled:
                if fvg.contains_price(current_price):
                    # Calculate fill percentage
                    if fvg.poi_type == POIType.FVG_BULLISH:
                        fill_pct = (fvg.high - current_price) / (fvg.high - fvg.low)
                    else:
                        fill_pct = (current_price - fvg.low) / (fvg.high - fvg.low)
                    fvg.fill_percentage = min(1.0, fill_pct)
                    if fvg.fill_percentage >= 0.5:
                        fvg.filled = True

    @property
    def last_result(self) -> Optional[POIResult]:
        """Get last detection result"""
        return self._last_result

    def get_active_pois(self, direction: str = None) -> List[Dict]:
        """Get all active (non-mitigated) POIs

        Args:
            direction: Filter by direction ('BUY' or 'SELL')

        Returns:
            List of active POI dicts
        """
        if self._last_result is None:
            return []

        if direction == "BUY":
            return self._last_result.bullish_pois
        elif direction == "SELL":
            return self._last_result.bearish_pois
        else:
            return self._last_result.bullish_pois + self._last_result.bearish_pois

    def reset(self):
        """Reset detector state"""
        self._last_result = None
        self._poi_counter = 0
