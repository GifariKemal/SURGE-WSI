"""Entry Trigger - LTF Confirmation Module
==========================================

Layer 5 of 6-Layer Architecture

Function: Provide precise entry triggers using ICT concepts:
1. Liquidity Sweep - Price takes out previous highs/lows
2. Market Structure Shift (MSS) - Confirms reversal
3. FVG Entry - Enter on retest of Fair Value Gap
4. Rejection Candle - Wick > 50% of body

Author: SURIOTA Team
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
from loguru import logger


class ConfirmationType(Enum):
    """Types of LTF confirmation signals"""
    LIQUIDITY_SWEEP = "liquidity_sweep"
    MSS = "market_structure_shift"
    FVG_ENTRY = "fvg_entry"
    REJECTION_CANDLE = "rejection_candle"
    SWEEP_AND_MSS = "sweep_and_mss"


@dataclass
class SwingPoint:
    """Swing High or Low on LTF"""
    idx: int
    price: float
    time: any
    swing_type: str  # 'high' or 'low'


@dataclass
class LiquiditySweep:
    """Detected liquidity sweep (stop hunt)"""
    idx: int
    time: any
    sweep_type: str  # 'buy' or 'sell'
    swept_level: float
    sweep_low: float
    sweep_high: float
    close_price: float
    strength: float
    valid: bool = True


@dataclass
class MarketStructureShift:
    """Market Structure Shift (MSS)"""
    idx: int
    time: any
    mss_type: str  # 'bullish' or 'bearish'
    break_level: float
    close_price: float
    associated_sweep: Optional[LiquiditySweep] = None
    strength: float = 0.0


@dataclass
class LTFEntrySignal:
    """Complete LTF Entry Signal"""
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    sl_pips: float
    confirmation_type: ConfirmationType
    quality_score: float

    # Component details
    sweep: Optional[LiquiditySweep] = None
    mss: Optional[MarketStructureShift] = None
    fvg_entry: Optional[Dict] = None
    rejection_candle: Optional[Dict] = None

    # Timing
    idx: int = 0
    time: any = None

    @property
    def is_full_confirmation(self) -> bool:
        """Check if we have full sweep + MSS confirmation"""
        return self.sweep is not None and self.mss is not None

    def get_risk_reward_levels(self, rr_ratio: float = 2.0) -> Dict:
        """Calculate TP levels based on R:R ratio"""
        sl_distance = abs(self.entry_price - self.stop_loss)
        if self.direction == 'BUY':
            tp = self.entry_price + (sl_distance * rr_ratio)
        else:
            tp = self.entry_price - (sl_distance * rr_ratio)

        return {
            'entry': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': tp,
            'sl_pips': self.sl_pips,
            'tp_pips': self.sl_pips * rr_ratio,
            'rr_ratio': rr_ratio
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'sl_pips': self.sl_pips,
            'confirmation_type': self.confirmation_type.value,
            'quality_score': self.quality_score,
            'has_sweep': self.sweep is not None,
            'has_mss': self.mss is not None,
            'has_fvg': self.fvg_entry is not None,
            'has_rejection': self.rejection_candle is not None,
        }


class EntryTrigger:
    """LTF Entry Confirmation Engine

    Detects precise entry triggers when price is at an HTF POI.
    """

    def __init__(
        self,
        swing_length: int = 3,
        sweep_min_pips: float = 2.0,
        mss_lookback: int = 10,
        fvg_min_pips: float = 1.0,
        min_quality_score: float = 75.0,  # ZERO LOSING MONTHS: Higher quality (was 60)
        max_sl_pips: float = 10.0,  # ZERO LOSING MONTHS: Max SL (was 50)
        rejection_wick_ratio: float = 0.5
    ):
        """Initialize Entry Trigger

        Args:
            swing_length: Bars for swing detection
            sweep_min_pips: Minimum sweep distance in pips
            mss_lookback: Bars to look back for MSS
            fvg_min_pips: Minimum FVG size in pips
            min_quality_score: Minimum signal quality to accept
            max_sl_pips: Maximum stop loss in pips
            rejection_wick_ratio: Min wick/body ratio for rejection
        """
        self.swing_length = swing_length
        self.sweep_min_pips = sweep_min_pips
        self.mss_lookback = mss_lookback
        self.fvg_min_pips = fvg_min_pips
        self.min_quality_score = min_quality_score
        self.max_sl_pips = max_sl_pips
        self.rejection_wick_ratio = rejection_wick_ratio

        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.sweeps: List[LiquiditySweep] = []
        self.mss_events: List[MarketStructureShift] = []
        self.ltf_fvgs: List[Dict] = []

        self._last_signal: Optional[LTFEntrySignal] = None

    def _get_col(self, df: pd.DataFrame, col: str) -> str:
        """Get column name (handle case variations)"""
        return col.lower() if col.lower() in df.columns else col.title()

    def detect_swings(self, df: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Detect swing highs and lows"""
        high_col = self._get_col(df, 'high')
        low_col = self._get_col(df, 'low')

        highs = df[high_col].values
        lows = df[low_col].values

        swing_highs = []
        swing_lows = []
        window = self.swing_length

        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append(SwingPoint(
                    idx=i,
                    price=highs[i],
                    time=df.index[i] if hasattr(df.index[i], 'strftime') else i,
                    swing_type='high'
                ))

            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append(SwingPoint(
                    idx=i,
                    price=lows[i],
                    time=df.index[i] if hasattr(df.index[i], 'strftime') else i,
                    swing_type='low'
                ))

        self.swing_highs = swing_highs
        self.swing_lows = swing_lows
        return swing_highs, swing_lows

    def detect_liquidity_sweep(
        self,
        df: pd.DataFrame,
        direction: str,
        lookback: int = 20
    ) -> Optional[LiquiditySweep]:
        """Detect liquidity sweep"""
        if len(df) < lookback + self.swing_length:
            return None

        high_col = self._get_col(df, 'high')
        low_col = self._get_col(df, 'low')
        close_col = self._get_col(df, 'close')

        self.detect_swings(df.iloc[:-lookback])
        recent = df.iloc[-lookback:]

        if direction == 'BUY':
            if not self.swing_lows:
                return None

            recent_lows = sorted(self.swing_lows, key=lambda x: x.idx)[-3:]

            for swing_low in recent_lows:
                for i in range(1, len(recent)):
                    bar = recent.iloc[i]

                    if bar[low_col] < swing_low.price:
                        sweep_distance = (swing_low.price - bar[low_col]) / 0.0001

                        if bar[close_col] > swing_low.price and sweep_distance >= self.sweep_min_pips:
                            bar_range = bar[high_col] - bar[low_col]
                            recovery = (bar[close_col] - bar[low_col]) / bar_range if bar_range > 0 else 0

                            sweep = LiquiditySweep(
                                idx=len(df) - lookback + i,
                                time=bar.name if hasattr(bar.name, 'strftime') else i,
                                sweep_type='buy',
                                swept_level=swing_low.price,
                                sweep_low=bar[low_col],
                                sweep_high=bar[high_col],
                                close_price=bar[close_col],
                                strength=recovery
                            )
                            self.sweeps.append(sweep)
                            return sweep

        else:  # SELL
            if not self.swing_highs:
                return None

            recent_highs = sorted(self.swing_highs, key=lambda x: x.idx)[-3:]

            for swing_high in recent_highs:
                for i in range(1, len(recent)):
                    bar = recent.iloc[i]

                    if bar[high_col] > swing_high.price:
                        sweep_distance = (bar[high_col] - swing_high.price) / 0.0001

                        if bar[close_col] < swing_high.price and sweep_distance >= self.sweep_min_pips:
                            bar_range = bar[high_col] - bar[low_col]
                            recovery = (bar[high_col] - bar[close_col]) / bar_range if bar_range > 0 else 0

                            sweep = LiquiditySweep(
                                idx=len(df) - lookback + i,
                                time=bar.name if hasattr(bar.name, 'strftime') else i,
                                sweep_type='sell',
                                swept_level=swing_high.price,
                                sweep_low=bar[low_col],
                                sweep_high=bar[high_col],
                                close_price=bar[close_col],
                                strength=recovery
                            )
                            self.sweeps.append(sweep)
                            return sweep

        return None

    def detect_mss(
        self,
        df: pd.DataFrame,
        direction: str,
        after_sweep: Optional[LiquiditySweep] = None
    ) -> Optional[MarketStructureShift]:
        """Detect Market Structure Shift"""
        if len(df) < self.mss_lookback:
            return None

        close_col = self._get_col(df, 'close')
        self.detect_swings(df.iloc[:-self.mss_lookback])

        start_idx = after_sweep.idx if after_sweep else len(df) - self.mss_lookback

        if direction == 'BUY':
            if not self.swing_highs:
                return None

            relevant_highs = [sh for sh in self.swing_highs if sh.idx < start_idx]
            if not relevant_highs:
                return None

            target_high = max(relevant_highs[-3:], key=lambda x: x.price)

            for i in range(max(0, start_idx - len(df) + self.mss_lookback), self.mss_lookback):
                actual_idx = len(df) - self.mss_lookback + i
                if actual_idx < 0 or actual_idx >= len(df):
                    continue

                bar = df.iloc[actual_idx]
                if bar[close_col] > target_high.price:
                    mss = MarketStructureShift(
                        idx=actual_idx,
                        time=bar.name if hasattr(bar.name, 'strftime') else actual_idx,
                        mss_type='bullish',
                        break_level=target_high.price,
                        close_price=bar[close_col],
                        associated_sweep=after_sweep,
                        strength=(bar[close_col] - target_high.price) / 0.0001
                    )
                    self.mss_events.append(mss)
                    return mss

        else:  # SELL
            if not self.swing_lows:
                return None

            relevant_lows = [sl for sl in self.swing_lows if sl.idx < start_idx]
            if not relevant_lows:
                return None

            target_low = min(relevant_lows[-3:], key=lambda x: x.price)

            for i in range(max(0, start_idx - len(df) + self.mss_lookback), self.mss_lookback):
                actual_idx = len(df) - self.mss_lookback + i
                if actual_idx < 0 or actual_idx >= len(df):
                    continue

                bar = df.iloc[actual_idx]
                if bar[close_col] < target_low.price:
                    mss = MarketStructureShift(
                        idx=actual_idx,
                        time=bar.name if hasattr(bar.name, 'strftime') else actual_idx,
                        mss_type='bearish',
                        break_level=target_low.price,
                        close_price=bar[close_col],
                        associated_sweep=after_sweep,
                        strength=(target_low.price - bar[close_col]) / 0.0001
                    )
                    self.mss_events.append(mss)
                    return mss

        return None

    def detect_rejection_candle(
        self,
        df: pd.DataFrame,
        direction: str
    ) -> Optional[Dict]:
        """Detect rejection candle (wick > 50% of body)"""
        if len(df) < 2:
            return None

        open_col = self._get_col(df, 'open')
        high_col = self._get_col(df, 'high')
        low_col = self._get_col(df, 'low')
        close_col = self._get_col(df, 'close')

        bar = df.iloc[-1]
        body = abs(bar[close_col] - bar[open_col])
        upper_wick = bar[high_col] - max(bar[close_col], bar[open_col])
        lower_wick = min(bar[close_col], bar[open_col]) - bar[low_col]

        if body < 0.00001:  # Doji
            return None

        if direction == 'BUY':
            # Bullish rejection: long lower wick
            wick_ratio = lower_wick / body if body > 0 else 0
            if wick_ratio >= self.rejection_wick_ratio and bar[close_col] > bar[open_col]:
                return {
                    'type': 'bullish_rejection',
                    'wick_ratio': wick_ratio,
                    'bar_index': len(df) - 1,
                    'close': bar[close_col],
                    'low': bar[low_col]
                }

        else:  # SELL
            # Bearish rejection: long upper wick
            wick_ratio = upper_wick / body if body > 0 else 0
            if wick_ratio >= self.rejection_wick_ratio and bar[close_col] < bar[open_col]:
                return {
                    'type': 'bearish_rejection',
                    'wick_ratio': wick_ratio,
                    'bar_index': len(df) - 1,
                    'close': bar[close_col],
                    'high': bar[high_col]
                }

        return None

    def detect_ltf_fvg(
        self,
        df: pd.DataFrame,
        direction: str,
        after_mss: Optional[MarketStructureShift] = None
    ) -> Optional[Dict]:
        """Detect Fair Value Gap on LTF"""
        if len(df) < 5:
            return None

        high_col = self._get_col(df, 'high')
        low_col = self._get_col(df, 'low')

        start_idx = after_mss.idx if after_mss else len(df) - 10

        for i in range(max(2, start_idx - len(df) + 5), len(df)):
            c0_high = df[high_col].iloc[i-2]
            c0_low = df[low_col].iloc[i-2]
            c2_high = df[high_col].iloc[i]
            c2_low = df[low_col].iloc[i]

            if direction == 'BUY':
                if c2_low > c0_high:
                    gap_pips = (c2_low - c0_high) / 0.0001
                    if gap_pips >= self.fvg_min_pips:
                        fvg = {
                            'direction': 'bullish',
                            'high': c2_low,
                            'low': c0_high,
                            'mid': (c2_low + c0_high) / 2,
                            'size_pips': gap_pips,
                            'idx': i - 1
                        }
                        self.ltf_fvgs.append(fvg)
                        return fvg

            else:
                if c2_high < c0_low:
                    gap_pips = (c0_low - c2_high) / 0.0001
                    if gap_pips >= self.fvg_min_pips:
                        fvg = {
                            'direction': 'bearish',
                            'high': c0_low,
                            'low': c2_high,
                            'mid': (c0_low + c2_high) / 2,
                            'size_pips': gap_pips,
                            'idx': i - 1
                        }
                        self.ltf_fvgs.append(fvg)
                        return fvg

        return None

    def get_entry_signal(
        self,
        ltf_df: pd.DataFrame,
        direction: str,
        poi_info: Dict,
        require_full_confirmation: bool = True
    ) -> Optional[LTFEntrySignal]:
        """Get LTF entry signal when price is at HTF POI

        Args:
            ltf_df: Lower timeframe OHLCV data
            direction: Trading direction from regime
            poi_info: POI information from detector
            require_full_confirmation: Require both sweep and MSS

        Returns:
            LTFEntrySignal if confirmation found
        """
        if len(ltf_df) < 30:
            logger.debug(f"Entry rejected: LTF data too short ({len(ltf_df)} < 30)")
            return None

        # Step 1: Look for Liquidity Sweep
        sweep = self.detect_liquidity_sweep(ltf_df, direction)

        # Step 2: Look for MSS
        mss = self.detect_mss(ltf_df, direction, sweep)

        # Step 3: Look for rejection candle
        rejection = self.detect_rejection_candle(ltf_df, direction)

        logger.debug(f"Entry check - sweep:{sweep is not None}, mss:{mss is not None}, rejection:{rejection is not None}")

        # Check confirmation requirements
        if require_full_confirmation:
            if sweep is None or mss is None:
                # Fall back to rejection candle
                if rejection is None:
                    return None
        else:
            if sweep is None and mss is None and rejection is None:
                return None

        # Step 4: Look for FVG entry
        fvg = self.detect_ltf_fvg(ltf_df, direction, mss)

        # Get current price
        close_col = self._get_col(ltf_df, 'close')
        current_price = float(ltf_df[close_col].iloc[-1])

        # Determine entry and SL
        if direction == 'BUY':
            if fvg:
                entry_price = fvg['mid']
            elif mss:
                entry_price = mss.break_level
            else:
                entry_price = current_price

            if sweep:
                stop_loss = min(sweep.sweep_low, poi_info.get('low', sweep.sweep_low))
            elif rejection:
                stop_loss = rejection['low']
            else:
                stop_loss = poi_info.get('low', current_price - 0.0020)

            stop_loss -= 0.00005

        else:  # SELL
            if fvg:
                entry_price = fvg['mid']
            elif mss:
                entry_price = mss.break_level
            else:
                entry_price = current_price

            if sweep:
                stop_loss = max(sweep.sweep_high, poi_info.get('high', sweep.sweep_high))
            elif rejection:
                stop_loss = rejection['high']
            else:
                stop_loss = poi_info.get('high', current_price + 0.0020)

            stop_loss += 0.00005

        # Calculate SL pips
        sl_pips = abs(entry_price - stop_loss) / 0.0001

        # Check max SL
        if sl_pips > self.max_sl_pips:
            logger.debug(f"SL too large: {sl_pips:.1f} pips > {self.max_sl_pips}")
            return None

        # Calculate quality score
        quality = 50.0
        if sweep:
            quality += 20.0 * sweep.strength
        if mss:
            quality += 20.0
        if fvg:
            quality += 10.0
        if rejection:
            quality += 10.0 * min(rejection['wick_ratio'], 1.0)

        quality = min(100.0, quality)

        # Check minimum quality
        if quality < self.min_quality_score:
            logger.debug(f"Quality too low: {quality:.1f} < {self.min_quality_score}")
            return None

        # Determine confirmation type
        if sweep and mss:
            conf_type = ConfirmationType.SWEEP_AND_MSS
        elif mss:
            conf_type = ConfirmationType.MSS
        elif sweep:
            conf_type = ConfirmationType.LIQUIDITY_SWEEP
        elif rejection:
            conf_type = ConfirmationType.REJECTION_CANDLE
        else:
            conf_type = ConfirmationType.FVG_ENTRY

        signal = LTFEntrySignal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            sl_pips=sl_pips,
            confirmation_type=conf_type,
            quality_score=quality,
            sweep=sweep,
            mss=mss,
            fvg_entry=fvg,
            rejection_candle=rejection,
            idx=len(ltf_df) - 1,
            time=ltf_df.index[-1] if hasattr(ltf_df.index[-1], 'strftime') else None
        )

        self._last_signal = signal

        logger.info(
            f"LTF Entry Signal: {direction} at {entry_price:.5f}, "
            f"SL {stop_loss:.5f} ({sl_pips:.1f} pips), "
            f"Quality {quality:.1f}%, Type: {conf_type.value}"
        )

        return signal

    def check_for_entry(
        self,
        ltf_df: pd.DataFrame,
        direction: str,
        poi_info: Dict,
        current_price: float,
        require_full_confirmation: bool = False
    ) -> Tuple[bool, Optional[LTFEntrySignal]]:
        """Quick check if entry conditions are met

        Args:
            ltf_df: Lower timeframe data
            direction: 'BUY' or 'SELL'
            poi_info: POI information dict
            current_price: Current market price
            require_full_confirmation: If True, require both sweep AND MSS.
                                      If False, accept any single confirmation.
        """
        signal = self.get_entry_signal(
            ltf_df, direction, poi_info,
            require_full_confirmation=require_full_confirmation
        )

        if signal is None:
            return False, None

        return True, signal

    @property
    def last_signal(self) -> Optional[LTFEntrySignal]:
        """Get last generated signal"""
        return self._last_signal

    def reset(self):
        """Reset state"""
        self.swing_highs.clear()
        self.swing_lows.clear()
        self.sweeps.clear()
        self.mss_events.clear()
        self.ltf_fvgs.clear()
        self._last_signal = None
