"""Volatility Adaptive Filter (VAF)
====================================

A sophisticated market activity filter designed to replace fixed Kill Zones.
It adapts to changing market conditions using volatility and price action metrics.

Key Components:
1. ATR Filter: Ensures sufficient volatility for profitable trading
2. Bollinger Band Squeeze: Detects potential breakouts
3. Range Filter: Ensures bar-by-bar movement
4. Time Scoring: Flexible scoring based on market hours (no hard blocks except weekends)

Usage:
    vaf = VolatilityAdaptiveFilter(atr_period=14, min_score=40)
    result = vaf.check(datetime.now(), high, low, close)
    if result.should_trade:
        # Execute trade logic

Author: SURIOTA Team
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class VAFResult:
    """Volatility Adaptive Filter result"""
    should_trade: bool
    score: float  # 0-100
    
    # Component scores
    atr_score: float
    bb_score: float
    range_score: float
    time_score: float
    
    # Metrics
    current_atr: float
    atr_threshold: float
    bb_width: float
    bb_squeeze: bool
    current_range: float
    
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            'should_trade': self.should_trade,
            'score': round(self.score, 1),
            'atr_score': round(self.atr_score, 1),
            'bb_score': round(self.bb_score, 1),
            'range_score': round(self.range_score, 1),
            'time_score': round(self.time_score, 1),
            'metrics': {
                'atr': round(self.current_atr, 5),
                'atr_threshold': round(self.atr_threshold, 5),
                'bb_width': round(self.bb_width, 5),
                'range': round(self.current_range, 5)
            },
            'reason': self.reason
        }


class VolatilityAdaptiveFilter:
    """Volatility-based filter to replace fixed Kill Zones"""
    
    def __init__(
        self,
        # ATR settings
        atr_period: int = 14,
        atr_multiplier: float = 0.5,        # Trade if ATR > avg * multiplier
        min_atr_pips: float = 5.0,           # Absolute minimum ATR in pips
        
        # Bollinger Band settings
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_squeeze_threshold: float = 0.5,   # Width below this = squeeze
        
        # Range settings
        min_range_pips: float = 2.0,         # Minimum bar range
        
        # Time settings
        skip_weekends: bool = True,
        skip_friday_late: bool = True,       # Skip Friday after 20:00 UTC
        
        # Scoring
        min_score: float = 40.0,             # Minimum score to trade
        
        # Symbol settings
        pip_size: float = 0.0001,
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_atr_pips = min_atr_pips
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_squeeze_threshold = bb_squeeze_threshold
        
        self.min_range_pips = min_range_pips
        
        self.skip_weekends = skip_weekends
        self.skip_friday_late = skip_friday_late
        
        self.min_score = min_score
        self.pip_size = pip_size
        
        # History
        self._close_history: List[float] = []
        self._tr_history: List[float] = []
        self._bb_width_history: List[float] = []
    
    def update(self, high: float, low: float, close: float) -> None:
        """Update filter with new bar data"""
        self._close_history.append(close)
        
        # Calculate True Range
        if len(self._close_history) > 1:
            prev_close = self._close_history[-2]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
        else:
            tr = high - low
        
        self._tr_history.append(tr)
        
        # Calculate BB width
        if len(self._close_history) >= self.bb_period:
            closes = self._close_history[-self.bb_period:]
            sma = np.mean(closes)
            std = np.std(closes)
            upper = sma + self.bb_std * std
            lower = sma - self.bb_std * std
            width = (upper - lower) / sma if sma > 0 else 0
            self._bb_width_history.append(width)
        
        # Keep limited history
        max_history = max(self.atr_period, self.bb_period) * 3
        if len(self._close_history) > max_history:
            self._close_history = self._close_history[-max_history:]
            self._tr_history = self._tr_history[-max_history:]
            self._bb_width_history = self._bb_width_history[-max_history:]
    
    def calculate_atr(self) -> Tuple[float, float]:
        """Calculate current ATR and average ATR"""
        if len(self._tr_history) < self.atr_period:
            return 0.0, 0.0
        
        # Current ATR (SMA of TR for stability)
        current_atr = np.mean(self._tr_history[-self.atr_period:])
        
        # Average ATR over longer period
        avg_atr = np.mean(self._tr_history)
        
        return current_atr, avg_atr
    
    def calculate_bb_metrics(self) -> Tuple[float, bool, float]:
        """Calculate Bollinger Band metrics"""
        if not self._bb_width_history:
            return 0.0, False, 0.0
        
        current_width = self._bb_width_history[-1]
        avg_width = np.mean(self._bb_width_history)
        
        is_squeeze = current_width < avg_width * self.bb_squeeze_threshold
        
        return current_width, is_squeeze, avg_width
    
    def is_market_open(self, dt: datetime) -> Tuple[bool, str]:
        """Check if market is open"""
        weekday = dt.weekday()
        hour = dt.hour
        
        # Saturday
        if weekday == 5 and self.skip_weekends:
            return False, "Weekend (Saturday)"
        
        # Sunday before 22:00
        if weekday == 6 and hour < 22 and self.skip_weekends:
            return False, "Weekend (Sunday)"
        
        # Friday late
        if weekday == 4 and hour >= 20 and self.skip_friday_late:
            return False, "Friday late session"
        
        return True, "Market open"
    
    def check(
        self,
        dt: datetime,
        current_high: float,
        current_low: float,
        current_close: float
    ) -> VAFResult:
        """Check if conditions are favorable for trading"""
        
        # 1. Update history
        self.update(current_high, current_low, current_close)
        
        # 2. Check market hours
        is_open, time_reason = self.is_market_open(dt)
        if not is_open:
            return VAFResult(False, 0, 0, 0, 0, 0, 0, 0, 0, False, 0, time_reason)
        
        # 3. Calculate metrics
        current_atr, avg_atr = self.calculate_atr()
        current_atr_pips = current_atr / self.pip_size
        
        bb_width, bb_squeeze, avg_bb_width = self.calculate_bb_metrics()
        
        current_range = current_high - current_low
        current_range_pips = current_range / self.pip_size
        
        # 4. Scoring Logic
        
        # ATR Score (0-40)
        atr_threshold = max(avg_atr * self.atr_multiplier, self.min_atr_pips * self.pip_size)
        if current_atr >= atr_threshold * 1.5:
            atr_score = 40
        elif current_atr >= atr_threshold:
            atr_score = 30
        elif current_atr >= atr_threshold * 0.7:
            atr_score = 15
        else:
            atr_score = 5
            
        # BB Score (0-20)
        if bb_width > 0:
            if bb_squeeze:
                bb_score = 15  # Squeeze = potential breakout
            elif bb_width > avg_bb_width * 1.2:
                bb_score = 20  # Expanding = trend
            else:
                bb_score = 10
        else:
            bb_score = 15
            
        # Range Score (0-20)
        min_range = self.min_range_pips * self.pip_size
        if current_range >= min_range * 2:
            range_score = 20
        elif current_range >= min_range:
            range_score = 15
        else:
            range_score = 0
            
        # Time Score (0-20) - Bonus, not requirement
        hour = dt.hour
        if 12 <= hour < 16:  # Overlap
            time_score = 20
        elif 7 <= hour < 17:  # Primary
            time_score = 15
        elif hour >= 22 or hour < 7:  # Asian
            time_score = 10
        else:
            time_score = 8
            
        # Total Score
        total_score = atr_score + bb_score + range_score + time_score
        total_score = max(0, min(100, total_score))
        
        should_trade = total_score >= self.min_score
        
        if should_trade:
            reason = f"Active (Score={total_score:.0f}, ATR={current_atr_pips:.1f}p)"
        else:
            reason = f"Inactive (Score={total_score:.0f})"
            
        return VAFResult(
            should_trade=should_trade,
            score=total_score,
            atr_score=atr_score,
            bb_score=bb_score,
            range_score=range_score,
            time_score=time_score,
            current_atr=current_atr,
            atr_threshold=atr_threshold,
            bb_width=bb_width,
            bb_squeeze=bb_squeeze,
            current_range=current_range,
            reason=reason
        )
    
    def warmup(self, df: pd.DataFrame) -> None:
        """Warmup filter with historical data"""
        for _, row in df.iterrows():
            h = row.get('High', row.get('high', 0))
            l = row.get('Low', row.get('low', 0))
            c = row.get('Close', row.get('close', 0))
            self.update(h, l, c)
