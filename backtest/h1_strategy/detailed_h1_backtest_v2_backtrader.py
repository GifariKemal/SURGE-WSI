"""
SURGE-WSI H1 GBPUSD Backtest v2 - BACKTRADER EDITION
=====================================================

Realistic backtesting using Backtrader library with:
- Real spread costs (1.5 pips for GBPUSD)
- Commission simulation
- NO SL_CAPPED accounting trick (real losses)
- Proper position management
- Slippage simulation

This produces results closer to MQL5 Strategy Tester.

Author: SURIOTA Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import backtrader as bt

from config import config
from src.data.db_handler import DBHandler

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION v2 - MATCHING MQL5 EA v3
# ============================================================
SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50000.0
RISK_PERCENT = 1.0

SL_ATR_MULT = 1.5
TP_RATIO = 1.5

PIP_VALUE = 10.0  # $10 per pip per standard lot
PIP_SIZE = 0.0001
SPREAD_PIPS = 1.5  # Realistic spread for GBPUSD
COMMISSION_PER_LOT = 7.0  # $7 commission per round-trip lot

MAX_LOT = 5.0
MIN_LOT = 0.01
MIN_ATR = 8.0
MAX_ATR = 25.0

# Quality thresholds
BASE_QUALITY = 65
MIN_QUALITY_GOOD = 60
MAX_QUALITY_BAD = 80

# ============================================================
# MONTHLY TRADEABLE PROFILE (from market analysis)
# ============================================================
MONTHLY_TRADEABLE_PCT = {
    (2024, 1): 67, (2024, 2): 55, (2024, 3): 70, (2024, 4): 80,
    (2024, 5): 62, (2024, 6): 68, (2024, 7): 78, (2024, 8): 65,
    (2024, 9): 72, (2024, 10): 58, (2024, 11): 66, (2024, 12): 60,
    (2025, 1): 65, (2025, 2): 55, (2025, 3): 70, (2025, 4): 70,
    (2025, 5): 62, (2025, 6): 68, (2025, 7): 78, (2025, 8): 65,
    (2025, 9): 72, (2025, 10): 58, (2025, 11): 66, (2025, 12): 60,
    (2026, 1): 65, (2026, 2): 55, (2026, 3): 70, (2026, 4): 80,
}

def get_monthly_quality_adjustment(dt: datetime) -> int:
    """Layer 1: Monthly profile quality adjustment"""
    key = (dt.year, dt.month)
    tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)

    if tradeable_pct < 30:
        return 50
    elif tradeable_pct < 40:
        return 35
    elif tradeable_pct < 50:
        return 25
    elif tradeable_pct < 60:
        return 15
    elif tradeable_pct < 70:
        return 10
    elif tradeable_pct < 75:
        return 5
    else:
        return 0


# Risk Multipliers (matching MQL5 v3)
MONTHLY_RISK = {
    1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,
    7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
}
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}
HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}
ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}

# Session POI filter
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]


# ============================================================
# CUSTOM COMMISSION SCHEME (Spread + Commission)
# ============================================================
class ForexCommission(bt.CommInfoBase):
    """
    Forex commission with spread + per-lot commission
    With leverage support for margin trading
    Size is in units (100,000 = 1 lot)
    """
    params = (
        ('commission', COMMISSION_PER_LOT),
        ('spread_pips', SPREAD_PIPS),
        ('pip_value', PIP_VALUE),
        ('pip_size', PIP_SIZE),
        ('lot_units', 100000),  # 1 lot = 100,000 units
        ('leverage', 100.0),    # 100:1 leverage
        ('stocklike', False),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )

    def _getcommission(self, size, price, pseudoexec):
        """Calculate commission: spread cost + fixed commission per lot"""
        # Convert units to lots
        lots = abs(size) / self.p.lot_units
        # Spread cost (paid on entry and exit = round trip)
        spread_cost = self.p.spread_pips * self.p.pip_value * lots
        # Fixed commission per lot
        commission = self.p.commission * lots
        return spread_cost + commission

    def get_leverage(self):
        """Return leverage"""
        return self.p.leverage

    def getsize(self, price, cash):
        """Return the maximum size that can be bought with available cash"""
        # With leverage, we can control larger positions
        return int((cash * self.p.leverage) / price)


# ============================================================
# BACKTRADER INDICATORS
# ============================================================
class ATRPips(bt.Indicator):
    """ATR in pips"""
    lines = ('atr_pips',)
    params = (('period', 14), ('pip_size', PIP_SIZE))

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.period)

    def next(self):
        self.lines.atr_pips[0] = self.atr[0] / self.p.pip_size


class MarketRegime(bt.Indicator):
    """Detect market regime (bullish/bearish/sideways)"""
    lines = ('regime',)  # 1=bullish, -1=bearish, 0=sideways
    params = (('ema_fast', 20), ('ema_slow', 50))

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)

    def next(self):
        close = self.data.close[0]
        ema20 = self.ema_fast[0]
        ema50 = self.ema_slow[0]

        if close > ema20 > ema50:
            self.lines.regime[0] = 1  # Bullish
        elif close < ema20 < ema50:
            self.lines.regime[0] = -1  # Bearish
        else:
            self.lines.regime[0] = 0  # Sideways


class QualityFilter(bt.Indicator):
    """Calculate quality score based on technical and monthly factors"""
    lines = ('quality', 'atr_stability', 'efficiency', 'trend_strength')
    params = (('lookback', 20),)

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)

    def next(self):
        # ATR stability (coefficient of variation)
        if len(self.atr) >= self.p.lookback:
            recent_atr = [self.atr[-i] for i in range(self.p.lookback)]
            mean_atr = np.mean(recent_atr)
            std_atr = np.std(recent_atr)
            atr_cv = std_atr / mean_atr if mean_atr > 0 else 0.5
        else:
            atr_cv = 0.5

        # Price efficiency
        if len(self.data) >= self.p.lookback:
            net_move = abs(self.data.close[0] - self.data.close[-self.p.lookback])
            total_move = sum(abs(self.data.close[-i] - self.data.close[-i-1])
                           for i in range(self.p.lookback))
            efficiency = net_move / total_move if total_move > 0 else 0
        else:
            efficiency = 0.1

        # Trend strength (ADX)
        trend_strength = self.adx[0] if len(self.adx) > 0 else 25

        # Calculate score
        score = 0
        if atr_cv < 0.25:
            score += 33
        elif atr_cv < 0.375:
            score += 20

        if efficiency > 0.08:
            score += 33
        elif efficiency > 0.04:
            score += 20

        if trend_strength > 25:
            score += 34
        elif trend_strength > 17.5:
            score += 20

        # Base quality
        if score >= 80:
            base_quality = MIN_QUALITY_GOOD
        elif score >= 40:
            base_quality = BASE_QUALITY
        else:
            base_quality = MAX_QUALITY_BAD

        self.lines.quality[0] = base_quality
        self.lines.atr_stability[0] = atr_cv
        self.lines.efficiency[0] = efficiency
        self.lines.trend_strength[0] = trend_strength


# ============================================================
# MAIN STRATEGY
# ============================================================
class QuadLayerStrategy(bt.Strategy):
    """
    GBPUSD H1 Quad-Layer Quality Strategy

    NO SL_CAPPED accounting trick - real stop losses only.
    Uses manual SL/TP checking per bar for proper forex handling.
    """
    params = (
        ('risk_pct', RISK_PERCENT),
        ('sl_atr_mult', SL_ATR_MULT),
        ('tp_ratio', TP_RATIO),
        ('min_atr', MIN_ATR),
        ('max_atr', MAX_ATR),
        ('pip_size', PIP_SIZE),
        ('pip_value', PIP_VALUE),
        ('lot_size_units', 100000),  # 1 lot = 100,000 units
        ('printlog', True),
    )

    def __init__(self):
        # Indicators
        self.atr_pips = ATRPips(self.data)
        self.regime = MarketRegime(self.data)
        self.quality_filter = QualityFilter(self.data)

        # EMA for pullback detection
        self.ema20 = bt.indicators.EMA(self.data.close, period=20)
        self.ema50 = bt.indicators.EMA(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)

        # Track orders and position
        self.order = None
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0
        self.position_direction = None  # 'BUY' or 'SELL'
        self.position_lots = 0

        # Statistics
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        self.gross_profit = 0
        self.gross_loss = 0

        # Monthly tracking (Layer 3)
        self.current_month = None
        self.monthly_pnl = 0
        self.consecutive_losses = 0

    def log(self, txt, dt=None):
        """Logging function"""
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.strftime("%Y-%m-%d %H:%M")} {txt}')

    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.5f}, Size: {order.executed.size:.0f} units')
            else:
                self.log(f'SELL EXECUTED @ {order.executed.price:.5f}, Size: {order.executed.size:.0f} units')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')

        self.order = None

    def notify_trade(self, trade):
        """Trade notification - calculate PnL manually for forex"""
        if not trade.isclosed:
            return

        # The trade.pnl from backtrader is in price units, need to convert to $ for forex
        # For forex: PnL = (exit - entry) * lot_size * pip_value / pip_size
        # But backtrader already calculated this, we just log it
        pnl = trade.pnlcomm  # Use net PnL (after commission)
        self.trade_count += 1

        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
            self.gross_profit += pnl
            result = "WIN"
        else:
            self.losses += 1
            self.consecutive_losses += 1
            self.gross_loss += abs(pnl)
            result = "LOSS"

        self.total_pnl += pnl
        self.monthly_pnl += pnl

        lots = abs(trade.size) / self.p.lot_size_units
        self.log(f'TRADE CLOSED [{result}]: Lots={lots:.2f}, PnL=${pnl:+.2f}, Total=${self.total_pnl:+.2f}')

    def check_session_time(self, dt) -> Tuple[bool, str]:
        """Check if current time is in trading session"""
        hour = dt.hour
        day = dt.weekday()

        # Skip weekends
        if day >= 5:
            return False, ""

        # London 8-10, NY 13-17
        if 8 <= hour <= 10:
            return True, "london"
        elif 13 <= hour <= 17:
            return True, "newyork"

        return False, ""

    def get_risk_multiplier(self, dt, entry_type: str, quality: float) -> Tuple[float, bool]:
        """Calculate risk multiplier (matching MQL5 v3)"""
        day_mult = DAY_MULTIPLIERS.get(dt.weekday(), 0.5)
        hour_mult = HOUR_MULTIPLIERS.get(dt.hour, 0.0)
        entry_mult = ENTRY_MULTIPLIERS.get(entry_type, 1.0)
        quality_mult = quality / 100.0
        month_mult = MONTHLY_RISK.get(dt.month, 0.8)

        if day_mult == 0.0 or hour_mult == 0.0:
            return 0.0, True

        combined = day_mult * hour_mult * entry_mult * quality_mult * month_mult

        # Clamp to [0.30, 1.20] like MQL5 v3
        if combined < 0.30:
            return combined, True

        return max(0.30, min(1.20, combined)), False

    def detect_order_block(self, min_quality: float) -> Optional[dict]:
        """Detect order block POI"""
        if len(self.data) < 5:
            return None

        o, h, l, c = self.data.open, self.data.high, self.data.low, self.data.close

        # Check previous candle
        prev_o = o[-1]
        prev_h = h[-1]
        prev_l = l[-1]
        prev_c = c[-1]

        curr_o = o[0]
        curr_h = h[0]
        curr_l = l[0]
        curr_c = c[0]

        # Bearish Order Block (potential BUY zone)
        is_prev_bearish = prev_c < prev_o
        if is_prev_bearish:
            curr_body = abs(curr_c - curr_o)
            curr_range = curr_h - curr_l
            if curr_range > 0:
                body_ratio = curr_body / curr_range
                if curr_c > curr_o and body_ratio > 0.55 and curr_c > prev_h:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        return {'direction': 'BUY', 'quality': quality, 'type': 'ORDER_BLOCK'}

        # Bullish Order Block (potential SELL zone)
        is_prev_bullish = prev_c > prev_o
        if is_prev_bullish:
            curr_body = abs(curr_c - curr_o)
            curr_range = curr_h - curr_l
            if curr_range > 0:
                body_ratio = curr_body / curr_range
                if curr_c < curr_o and body_ratio > 0.55 and curr_c < prev_l:
                    quality = body_ratio * 100
                    if quality >= min_quality:
                        return {'direction': 'SELL', 'quality': quality, 'type': 'ORDER_BLOCK'}

        return None

    def detect_ema_pullback(self, min_quality: float) -> Optional[dict]:
        """Detect EMA pullback signal"""
        if len(self.data) < 3:
            return None

        o, h, l, c = self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0]

        body = abs(c - o)
        total_range = h - l
        if total_range < 0.0003:
            return None

        body_ratio = body / total_range

        # Filter criteria
        if body_ratio < 0.4:
            return None
        if self.adx[0] < 20:
            return None
        if self.rsi[0] < 30 or self.rsi[0] > 70:
            return None

        is_bullish = c > o
        is_bearish = c < o

        ema20 = self.ema20[0]
        ema50 = self.ema50[0]

        atr_distance = self.atr_pips[0] * self.p.pip_size * 1.5

        # BUY: Uptrend pullback to EMA
        if c > ema20 > ema50 and is_bullish:
            dist = l - ema20
            if dist <= atr_distance:
                touch_quality = max(0, 30 - (dist / self.p.pip_size))
                adx_quality = min(25, (self.adx[0] - 15) * 1.5)
                rsi_quality = 25 if abs(50 - self.rsi[0]) < 20 else 15
                body_quality = min(20, body_ratio * 30)
                quality = min(100, max(55, touch_quality + adx_quality + rsi_quality + body_quality))
                if quality >= min_quality:
                    return {'direction': 'BUY', 'quality': quality, 'type': 'EMA_PULLBACK'}

        # SELL: Downtrend pullback to EMA
        if c < ema20 < ema50 and is_bearish:
            dist = ema20 - h
            if dist <= atr_distance:
                touch_quality = max(0, 30 - (dist / self.p.pip_size))
                adx_quality = min(25, (self.adx[0] - 15) * 1.5)
                rsi_quality = 25 if abs(50 - self.rsi[0]) < 20 else 15
                body_quality = min(20, body_ratio * 30)
                quality = min(100, max(55, touch_quality + adx_quality + rsi_quality + body_quality))
                if quality >= min_quality:
                    return {'direction': 'SELL', 'quality': quality, 'type': 'EMA_PULLBACK'}

        return None

    def check_entry_trigger(self, direction: str) -> Tuple[bool, str]:
        """Check for entry trigger pattern"""
        if len(self.data) < 3:
            return False, ""

        o, h, l, c = self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0]
        prev_o, prev_h, prev_l, prev_c = self.data.open[-1], self.data.high[-1], self.data.low[-1], self.data.close[-1]

        total_range = h - l
        if total_range < 0.0003:
            return False, ""

        body = abs(c - o)
        is_bullish = c > o
        is_bearish = c < o
        prev_body = abs(prev_c - prev_o)

        # Momentum entry
        if body > total_range * 0.5:
            if direction == 'BUY' and is_bullish:
                return True, 'MOMENTUM'
            if direction == 'SELL' and is_bearish:
                return True, 'MOMENTUM'

        # Engulfing entry
        if body > prev_body * 1.2:
            if direction == 'BUY' and is_bullish and prev_c < prev_o:
                return True, 'ENGULF'
            if direction == 'SELL' and is_bearish and prev_c > prev_o:
                return True, 'ENGULF'

        # Lower high (for SELL)
        if direction == 'SELL':
            if h < prev_h and is_bearish:
                return True, 'LOWER_HIGH'

        return False, ""

    def next(self):
        """Main strategy logic with manual SL/TP management"""
        dt = self.datas[0].datetime.datetime(0)

        # Check month change (Layer 3)
        month_key = (dt.year, dt.month)
        if self.current_month != month_key:
            self.current_month = month_key
            self.monthly_pnl = 0
            self.consecutive_losses = 0

        # ============================================================
        # POSITION MANAGEMENT: Check SL/TP manually each bar
        # ============================================================
        if self.position:
            high = self.data.high[0]
            low = self.data.low[0]
            exit_price = None
            exit_reason = ""

            if self.position_direction == 'BUY':
                # Check SL first (worst case)
                if low <= self.sl_price:
                    exit_price = self.sl_price
                    exit_reason = "SL"
                # Then check TP
                elif high >= self.tp_price:
                    exit_price = self.tp_price
                    exit_reason = "TP"
            else:  # SELL
                if high >= self.sl_price:
                    exit_price = self.sl_price
                    exit_reason = "SL"
                elif low <= self.tp_price:
                    exit_price = self.tp_price
                    exit_reason = "TP"

            # Close position if SL/TP hit
            if exit_price:
                if self.position_direction == 'BUY':
                    self.order = self.sell(size=self.position.size)
                else:
                    self.order = self.buy(size=abs(self.position.size))

                self.log(f'EXIT [{exit_reason}] @ {exit_price:.5f}')
                self.position_direction = None
                self.sl_price = 0
                self.tp_price = 0
            return

        # Skip if we have pending orders
        if self.order:
            return

        # ============================================================
        # ENTRY LOGIC
        # ============================================================

        # Check session time
        in_session, session = self.check_session_time(dt)
        if not in_session:
            return

        # Check ATR range
        atr = self.atr_pips[0]
        if atr < self.p.min_atr or atr > self.p.max_atr:
            return

        # Check regime
        regime = self.regime[0]
        if regime == 0:  # Sideways
            return

        # Get quality threshold (Layer 1 + Layer 2)
        base_quality = self.quality_filter.quality[0]
        monthly_adj = get_monthly_quality_adjustment(dt)
        min_quality = base_quality + monthly_adj

        # Layer 3: Intra-month adjustment
        if self.monthly_pnl <= -400:
            return  # Circuit breaker
        elif self.monthly_pnl <= -350:
            min_quality += 15
        elif self.monthly_pnl <= -250:
            min_quality += 10
        elif self.monthly_pnl <= -150:
            min_quality += 5

        if self.consecutive_losses >= 6:
            return  # Day stop
        elif self.consecutive_losses >= 3:
            min_quality += 5

        # Signal detection
        signal = None
        hour = dt.hour

        # EMA Pullback first (unless filtered hour)
        if hour not in SKIP_EMA_PULLBACK_HOURS:
            signal = self.detect_ema_pullback(min_quality)

        # Order Block second
        if not signal and hour not in SKIP_ORDER_BLOCK_HOURS:
            signal = self.detect_order_block(min_quality)

        if not signal:
            return

        # Check regime alignment
        if signal['direction'] == 'BUY' and regime != 1:
            return
        if signal['direction'] == 'SELL' and regime != -1:
            return

        # Check entry trigger
        has_trigger, entry_type = self.check_entry_trigger(signal['direction'])
        if not has_trigger:
            return

        # Calculate risk multiplier
        risk_mult, should_skip = self.get_risk_multiplier(dt, entry_type, signal['quality'])
        if should_skip:
            return

        # Calculate position size in UNITS (not lots for backtrader)
        sl_pips = atr * self.p.sl_atr_mult
        tp_pips = sl_pips * self.p.tp_ratio
        risk_amount = self.broker.getcash() * (self.p.risk_pct / 100.0) * risk_mult
        lot_size = risk_amount / (sl_pips * self.p.pip_value)
        lot_size = max(MIN_LOT, min(MAX_LOT, round(lot_size, 2)))

        # Convert lots to units for backtrader (1 lot = 100,000 units)
        units = int(lot_size * self.p.lot_size_units)

        price = self.data.close[0]

        # Calculate SL/TP prices
        if signal['direction'] == 'BUY':
            self.sl_price = price - (sl_pips * self.p.pip_size)
            self.tp_price = price + (tp_pips * self.p.pip_size)
        else:
            self.sl_price = price + (sl_pips * self.p.pip_size)
            self.tp_price = price - (tp_pips * self.p.pip_size)

        # Execute trade
        self.log(f'SIGNAL: {signal["direction"]} {signal["type"]} Q={signal["quality"]:.0f} '
                f'Qreq={min_quality:.0f} Entry={entry_type} Lots={lot_size:.2f} SL={self.sl_price:.5f} TP={self.tp_price:.5f}')

        if signal['direction'] == 'BUY':
            self.order = self.buy(size=units)
            self.position_direction = 'BUY'
        else:
            self.order = self.sell(size=units)
            self.position_direction = 'SELL'

        self.position_lots = lot_size

    def stop(self):
        """Called at end of backtest"""
        if self.trade_count > 0:
            win_rate = (self.wins / self.trade_count) * 100
        else:
            win_rate = 0

        final_value = self.broker.getvalue()
        net_pnl = final_value - INITIAL_BALANCE

        print("\n" + "="*70)
        print("BACKTEST v2 RESULTS - BACKTRADER (REALISTIC)")
        print("="*70)
        print(f"Total Trades:    {self.trade_count}")
        print(f"Winners:         {self.wins}")
        print(f"Losers:          {self.losses}")
        print(f"Win Rate:        {win_rate:.1f}%")
        print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
        print(f"Final Balance:   ${final_value:,.2f}")
        print(f"Net P/L:         ${net_pnl:+,.2f}")
        print(f"Return:          {(net_pnl/INITIAL_BALANCE)*100:+.1f}%")
        print("="*70)


# ============================================================
# DATA FEED
# ============================================================
async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        print("Database connection failed!")
        return pd.DataFrame()

    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def prepare_data_for_backtrader(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for Backtrader"""
    # Ensure proper column names
    df = df.copy()

    # Map columns
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_map[col] = 'open'
        elif 'high' in col_lower:
            col_map[col] = 'high'
        elif 'low' in col_lower:
            col_map[col] = 'low'
        elif 'close' in col_lower:
            col_map[col] = 'close'
        elif 'volume' in col_lower:
            col_map[col] = 'volume'

    df = df.rename(columns=col_map)

    # Ensure datetime index
    if 'time' in df.columns:
        df.index = pd.to_datetime(df['time'])
        df = df.drop(columns=['time'])
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if 'volume' not in df.columns:
        df['volume'] = 0

    return df[['open', 'high', 'low', 'close', 'volume']]


# ============================================================
# MAIN
# ============================================================
async def run_backtest():
    """Run backtest with Backtrader"""
    print("="*70)
    print("SURGE-WSI H1 GBPUSD Backtest v2 - BACKTRADER EDITION")
    print("="*70)
    print("Realistic simulation with:")
    print(f"  - Spread: {SPREAD_PIPS} pips")
    print(f"  - Commission: ${COMMISSION_PER_LOT}/lot")
    print(f"  - NO SL_CAPPED accounting trick")
    print(f"  - Real stop losses & take profits")
    print("="*70)

    # Fetch data
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    print(f"\nFetching {SYMBOL} H1 data...")
    df = await fetch_data(SYMBOL, "H1", start, end)

    if df.empty:
        print("Error: No data fetched!")
        return

    print(f"Fetched {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Prepare data
    df = prepare_data_for_backtrader(df)

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add data
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    cerebro.adddata(data, name=SYMBOL)

    # Add strategy
    cerebro.addstrategy(QuadLayerStrategy, printlog=True)

    # Set broker parameters
    cerebro.broker.setcash(INITIAL_BALANCE)

    # CRITICAL: Disable margin checking to allow forex leverage
    # In real forex, 100:1 leverage means $50k can control $5M
    cerebro.broker.set_checksubmit(False)

    # Add commission scheme (spread + commission)
    commission = ForexCommission()
    cerebro.broker.addcommissioninfo(commission)

    # Add slippage (0.5 pip)
    cerebro.broker.set_slippage_fixed(0.5 * PIP_SIZE, slip_open=True, slip_match=True)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    print("\nRunning backtest...")
    results = cerebro.run()
    strat = results[0]

    # Print analyzer results
    print("\n[ANALYZERS]")
    print("-"*50)

    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    except:
        pass

    try:
        dd = strat.analyzers.drawdown.get_analysis()
        print(f"Max Drawdown: {dd.max.drawdown:.2f}%")
        print(f"Max DD Money: ${dd.max.moneydown:,.2f}")
    except:
        pass

    try:
        trades = strat.analyzers.trades.get_analysis()
        if 'total' in trades and 'total' in trades.total:
            print(f"\nTrade Analysis:")
            print(f"  Total:  {trades.total.total}")
            if 'pnl' in trades and 'gross' in trades.pnl:
                print(f"  Gross:  ${trades.pnl.gross.total:,.2f}")
            if 'pnl' in trades and 'net' in trades.pnl:
                print(f"  Net:    ${trades.pnl.net.total:,.2f}")
    except:
        pass

    # Optional: Plot
    # cerebro.plot(style='candlestick')

    return cerebro.broker.getvalue()


def main():
    """Main entry point"""
    final_value = asyncio.run(run_backtest())

    if final_value:
        print(f"\n[COMPARISON]")
        print(f"-"*50)
        print(f"Python Backtest v1 (with SL_CAPPED): ~$37,181 profit")
        print(f"MQL5 EA v3 (realistic):              ~$1,428 profit")
        print(f"Backtrader v2 (realistic):           ${final_value - INITIAL_BALANCE:+,.2f}")
        print(f"\nExpected: Results should be closer to MQL5 v3")


if __name__ == "__main__":
    main()
