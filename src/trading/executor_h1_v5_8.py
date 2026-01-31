"""Trade Executor H1 v5.8 - Best Profit Factor Version
======================================================

Based on backtest results:
- 96 trades, 47.9% WR, +$2,269, PF 1.77
- Return: +22.7% in 13 months
- Only 1 losing month (June -$11)
- Max DD: $429 (4.3%)

Key Features (different from v3):
1. AutoRiskAdjuster - automatic detection of bad market conditions
2. AdaptiveRiskScorer - day/hour/entry multipliers
3. Known period protection (June/September)
4. Quality-based filtering
5. Market condition filter
6. Drift detection

Author: SURIOTA Team
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import pandas as pd
import numpy as np

from ..analysis.kalman_filter import MultiScaleKalman
from ..analysis.regime_detector import HMMRegimeDetector
from ..utils.killzone import KillZone
from ..utils.dynamic_activity_filter import DynamicActivityFilter, ActivityLevel
from ..utils.auto_risk_adjuster import AutoRiskAdjuster


class ExecutorState(Enum):
    """Executor state"""
    IDLE = "idle"
    WARMING_UP = "warming_up"
    MONITORING = "monitoring"
    TRADING = "trading"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    ticket: int = 0
    direction: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    volume: float = 0.0
    message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class ExecutorStats:
    """Executor statistics"""
    start_time: datetime = None
    total_signals: int = 0
    signals_filtered: int = 0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_daily_reset: datetime = None

    @property
    def win_rate(self) -> float:
        if self.trades_executed == 0:
            return 0.0
        return self.trades_won / self.trades_executed * 100

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss


@dataclass
class OpenPosition:
    """Track open position"""
    ticket: int
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    entry_time: datetime
    poi_type: str = ""
    entry_type: str = ""
    quality_score: float = 0.0
    risk_multiplier: float = 1.0


class AdaptiveRiskScorer:
    """v5.8 Adaptive Risk Scoring - calculates risk multiplier based on multiple factors"""

    # Day multipliers
    DAY_MULTIPLIERS = {
        0: 1.2,   # Monday - BEST
        1: 0.9,   # Tuesday
        2: 1.1,   # Wednesday
        3: 0.5,   # Thursday - penalty
        4: 0.5,   # Friday - penalty
        5: 0.0,   # Saturday - SKIP
        6: 0.0,   # Sunday - SKIP
    }

    # Hour multipliers (UTC)
    HOUR_MULTIPLIERS = {
        7: 1.0, 8: 1.0,              # Early London
        9: 1.3,                       # BEST hour (63.6% WR)
        10: 0.5, 11: 0.7,            # Transition
        12: 1.0, 13: 1.0, 14: 1.2,   # Overlap
        15: 0.7, 16: 0.5,            # Post-overlap
    }

    # Entry type multipliers
    ENTRY_MULTIPLIERS = {
        'MOMENTUM': 1.0,      # Best: 50.9% WR, +$1,726
        'LOWER_HIGH': 1.0,    # Good: 66.7% WR, +$480
        'REJECTION': 0.8,     # OK: 37% WR, +$121
        'HIGHER_LOW': 0.6,    # Weak: 42.9% WR, -$64
        'ENGULF': 0.8,
        'SMALL_BODY': 0.4,
    }

    SKIP_THRESHOLD = 0.28  # Skip if combined mult < 28%

    def __init__(self, min_combined_score: float = 45.0):
        self.min_combined_score = min_combined_score

    def calculate(
        self,
        current_time: datetime,
        entry_type: str,
        in_killzone: bool,
        drift_detected: bool
    ) -> Tuple[float, float, str, bool]:
        """Calculate risk multiplier and required quality"""
        day = current_time.weekday()
        hour = current_time.hour

        day_mult = self.DAY_MULTIPLIERS.get(day, 1.0)
        hour_mult = self.HOUR_MULTIPLIERS.get(hour, 0.5)
        entry_mult = self.ENTRY_MULTIPLIERS.get(entry_type, 0.8)

        # Hard skip
        if day_mult == 0.0:
            return 0.0, 100.0, "weekend", True

        # Session multiplier
        session_mult = 1.0 if in_killzone else 0.65

        # Drift multiplier
        drift_mult = 0.6 if drift_detected else 1.0

        # Combined
        raw_mult = day_mult * hour_mult * entry_mult * session_mult * drift_mult

        if raw_mult < self.SKIP_THRESHOLD:
            return raw_mult, 100.0, "low_combined", True

        risk_mult = max(0.28, min(1.3, raw_mult))

        # Required quality
        base_quality = self.min_combined_score
        quality_bonus = 0
        if not in_killzone:
            quality_bonus += 15
        if drift_detected:
            quality_bonus += 15
        if hour_mult < 0.7:
            quality_bonus += 10

        required_quality = min(90, base_quality + quality_bonus)

        reasons = []
        if day_mult != 1.0:
            reasons.append(f"day={day_mult:.1f}x")
        if hour_mult != 1.0:
            reasons.append(f"hr={hour_mult:.1f}x")
        if entry_mult < 1.0:
            reasons.append(f"entry={entry_mult:.1f}x")

        reason = ", ".join(reasons) if reasons else "optimal"

        return risk_mult, required_quality, reason, False


class TradeExecutorH1V58:
    """H1 Trading Executor v5.8 - Best PF Version"""

    def __init__(
        self,
        symbol: str = "GBPUSD",
        warmup_bars: int = 100,
        magic_number: int = 20250130
    ):
        self.symbol = symbol
        self.warmup_bars = warmup_bars
        self.magic_number = magic_number

        # Core components
        self.kalman = MultiScaleKalman()
        self.regime_detector = HMMRegimeDetector()
        self.killzone = KillZone()

        # v5.8 components
        self.activity_filter = DynamicActivityFilter(
            min_atr_pips=5.0,
            min_bar_range_pips=3.0,
            activity_threshold=35.0,
            pip_size=0.0001
        )
        self.activity_filter.outside_kz_min_score = 55.0

        self.risk_scorer = AdaptiveRiskScorer(min_combined_score=45.0)
        self.auto_risk = AutoRiskAdjuster(
            lookback_bars=120,
            min_bars=24,
            track_trades=True
        )

        # v5.8 parameters
        self.sl_pips = 25.0
        self.tp_rr = 1.5
        self.base_risk = 0.01  # 1%
        self.min_quality = 45

        # Cooldown
        self.cooldown_after_sl = timedelta(hours=1)
        self.cooldown_after_tp = timedelta(minutes=30)
        self._cooldown_until: Optional[datetime] = None

        # Circuit Breaker
        self.max_daily_loss_pct = 0.03
        self._circuit_breaker_triggered = False
        self._starting_balance: float = 0.0

        # State
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(
            start_time=datetime.now(timezone.utc),
            last_daily_reset=datetime.now(timezone.utc).date()
        )
        self._warmup_count = 0
        self._position: Optional[OpenPosition] = None

        # Locks
        self._position_lock = asyncio.Lock()
        self._cooldown_lock = asyncio.Lock()

        # Data buffers
        self._h1_data: Optional[pd.DataFrame] = None
        self._last_h1_bar_time: Optional[datetime] = None

        # Callbacks
        self.get_account_info: Optional[Callable] = None
        self.get_tick: Optional[Callable] = None
        self.get_ohlcv: Optional[Callable] = None
        self.get_symbol_info: Optional[Callable] = None
        self.place_market_order: Optional[Callable] = None
        self.modify_position: Optional[Callable] = None
        self.close_position: Optional[Callable] = None
        self.get_positions: Optional[Callable] = None
        self.get_deal_history: Optional[Callable] = None
        self.send_telegram: Optional[Callable] = None

        self._contract_size: float = 100000.0
        self._pip_value: float = 0.0001

        logger.info("H1 v5.8 Executor initialized (Best PF version)")

    def set_callbacks(self, **kwargs):
        """Set execution callbacks"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    async def fetch_symbol_info(self) -> bool:
        """Fetch symbol info from MT5"""
        if not self.get_symbol_info:
            return True
        try:
            info = await self.get_symbol_info(self.symbol)
            if info:
                self._contract_size = info.get('contract_size', 100000.0)
                self._pip_value = info.get('point', 0.0001)
            return True
        except Exception as e:
            logger.error(f"Failed to fetch symbol info: {e}")
            return False

    async def recover_positions(self) -> bool:
        """Recover existing positions"""
        if not self.get_positions:
            return True

        try:
            positions = await self.get_positions()
            if not positions:
                return True

            my_positions = [p for p in positions if p.get('magic') == self.magic_number]
            if not my_positions:
                return True

            pos = my_positions[0]
            async with self._position_lock:
                self._position = OpenPosition(
                    ticket=pos['ticket'],
                    direction='BUY' if pos['type'] == 0 else 'SELL',
                    entry_price=pos['price_open'],
                    stop_loss=pos.get('sl', 0),
                    take_profit=pos.get('tp', 0),
                    lot_size=pos['volume'],
                    entry_time=datetime.now(timezone.utc),
                    poi_type="RECOVERED",
                    entry_type="RECOVERED",
                    quality_score=50.0
                )

            logger.info(f"Position recovered: {self._position.direction} @ {self._position.entry_price}")

            if self.send_telegram:
                msg = f"Position recovered: {self._position.direction} @ {self._position.entry_price:.5f}"
                await self.send_telegram(msg)

            return True
        except Exception as e:
            logger.error(f"Position recovery failed: {e}")
            return False

    async def warmup(self, h1_data: pd.DataFrame) -> bool:
        """Warmup with historical data"""
        self.state = ExecutorState.WARMING_UP

        try:
            close_col = 'close' if 'close' in h1_data.columns else 'Close'

            for price in h1_data[close_col].values:
                self.kalman.update(price)
                self.regime_detector.update(price)
                self._warmup_count += 1

            self._h1_data = h1_data.copy()

            logger.info(f"Warmup complete with {self._warmup_count} H1 bars")
            self.state = ExecutorState.MONITORING
            return True
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self.state = ExecutorState.ERROR
            return False

    def _get_col_map(self, df: pd.DataFrame) -> dict:
        return {
            'close': 'close' if 'close' in df.columns else 'Close',
            'open': 'open' if 'open' in df.columns else 'Open',
            'high': 'high' if 'high' in df.columns else 'High',
            'low': 'low' if 'low' in df.columns else 'Low',
        }

    async def _check_daily_reset(self, now: datetime, balance: float):
        """Check and reset daily stats"""
        today = now.date()
        if self.stats.last_daily_reset != today:
            self.stats.daily_pnl = 0.0
            self.stats.daily_trades = 0
            self.stats.last_daily_reset = today
            self._starting_balance = balance
            self._circuit_breaker_triggered = False
            logger.info(f"Daily reset. Starting balance: ${balance:,.2f}")

    async def _check_circuit_breaker(self, balance: float) -> bool:
        """Check circuit breaker"""
        if self._starting_balance == 0:
            return False

        loss_pct = (self._starting_balance - balance) / self._starting_balance
        if loss_pct >= self.max_daily_loss_pct:
            if not self._circuit_breaker_triggered:
                self._circuit_breaker_triggered = True
                self.state = ExecutorState.PAUSED
                logger.warning(f"CIRCUIT BREAKER! Daily loss: {loss_pct:.2%}")
                if self.send_telegram:
                    await self.send_telegram(f"CIRCUIT BREAKER TRIGGERED! Loss: {loss_pct:.2%}")
            return True
        return False

    def _calculate_ob_quality(self, df: pd.DataFrame, ob_idx: int, direction: str) -> float:
        """Calculate Order Block quality"""
        if ob_idx < 5 or ob_idx >= len(df) - 3:
            return 50

        quality = 0.0
        col_map = self._get_col_map(df)
        ob_bar = df.iloc[ob_idx]
        next_bars = df.iloc[ob_idx+1:ob_idx+4]

        # Impulse quality (0-50)
        if direction == 'BUY':
            impulse = next_bars[col_map['close']].max() - ob_bar[col_map['low']]
        else:
            impulse = ob_bar[col_map['high']] - next_bars[col_map['close']].min()

        impulse_pips = impulse * 10000
        quality += min(50, impulse_pips * 2.5)

        # Wick analysis (0-25)
        ob_range = ob_bar[col_map['high']] - ob_bar[col_map['low']]
        if ob_range > 0:
            if direction == 'BUY':
                wick = ob_bar[col_map['high']] - max(ob_bar[col_map['open']], ob_bar[col_map['close']])
            else:
                wick = min(ob_bar[col_map['open']], ob_bar[col_map['close']]) - ob_bar[col_map['low']]
            wick_ratio = wick / ob_range
            if wick_ratio > 0.3:
                quality += 25
            elif wick_ratio > 0.2:
                quality += 15

        # Fresh zone (0-25)
        touched = False
        for i in range(ob_idx + 4, min(ob_idx + 15, len(df))):
            bar = df.iloc[i]
            if direction == 'BUY' and bar[col_map['low']] <= ob_bar[col_map['high']]:
                touched = True
                break
            elif direction == 'SELL' and bar[col_map['high']] >= ob_bar[col_map['low']]:
                touched = True
                break
        if not touched:
            quality += 25

        return min(100, quality)

    def _detect_poi(self, df: pd.DataFrame, idx: int, direction: str) -> Optional[dict]:
        """Detect POI (Order Block)"""
        if idx < 18:
            return None

        col_map = self._get_col_map(df)
        recent = df.iloc[idx-15:idx]

        for i in range(len(recent) - 3):
            bar = recent.iloc[i]
            next_bars = recent.iloc[i+1:i+4]
            actual_idx = idx - 15 + i

            if direction == 'BUY':
                if bar[col_map['close']] < bar[col_map['open']]:
                    move = next_bars[col_map['close']].max() - bar[col_map['low']]
                    if move > 0.0010:
                        quality = self._calculate_ob_quality(df, actual_idx, direction)
                        return {
                            'type': 'OB',
                            'quality': quality,
                            'zone_high': bar[col_map['high']],
                            'zone_low': bar[col_map['low']]
                        }
            else:
                if bar[col_map['close']] > bar[col_map['open']]:
                    move = bar[col_map['high']] - next_bars[col_map['close']].min()
                    if move > 0.0010:
                        quality = self._calculate_ob_quality(df, actual_idx, direction)
                        return {
                            'type': 'OB',
                            'quality': quality,
                            'zone_high': bar[col_map['high']],
                            'zone_low': bar[col_map['low']]
                        }

        return None

    def _check_entry_trigger(self, bar, prev_bar, direction: str) -> Optional[str]:
        """Check entry trigger"""
        col_map = self._get_col_map(self._h1_data)

        o, h, l, c = bar[col_map['open']], bar[col_map['high']], bar[col_map['low']], bar[col_map['close']]
        total_range = h - l
        if total_range < 0.0003:
            return None

        body = abs(c - o)
        is_bullish = c > o
        is_bearish = c < o

        po, ph, pl, pc = prev_bar[col_map['open']], prev_bar[col_map['high']], prev_bar[col_map['low']], prev_bar[col_map['close']]

        if direction == 'BUY':
            lower_wick = min(o, c) - l
            if lower_wick > body and lower_wick > total_range * 0.5:
                return "REJECTION"
            if is_bullish and body > total_range * 0.6:
                return "MOMENTUM"
            if is_bullish and c > ph and o <= pl:
                return "ENGULF"
            if l > pl and is_bullish:
                return "HIGHER_LOW"
            if h < ph and is_bullish:
                return "LOWER_HIGH"
        else:
            upper_wick = h - max(o, c)
            if upper_wick > body and upper_wick > total_range * 0.5:
                return "REJECTION"
            if is_bearish and body > total_range * 0.6:
                return "MOMENTUM"
            if is_bearish and c < pl and o >= ph:
                return "ENGULF"
            if h < ph and is_bearish:
                return "LOWER_HIGH"

        return None

    def _get_known_period_mult(self, current_time: datetime) -> float:
        """Get known period risk multiplier (June/September protection)"""
        month = current_time.month
        dow = current_time.weekday()

        if month == 6:  # June - historically bad
            return 0.1
        elif month == 9:  # September - high reversals
            if dow == 2:  # Wednesday
                return 0.05
            return 0.3

        return 1.0

    async def process_tick(self, price: float, account_balance: float) -> Optional[TradeResult]:
        """Process new tick - v5.8 logic"""
        if self.state not in [ExecutorState.MONITORING, ExecutorState.TRADING]:
            return None

        now = datetime.now(timezone.utc)

        if self._starting_balance == 0:
            self._starting_balance = account_balance

        await self._check_daily_reset(now, account_balance)

        if await self._check_circuit_breaker(account_balance):
            return None

        # Update indicators
        kalman_state = self.kalman.update(price)
        regime_info = self.regime_detector.update(kalman_state['smoothed_price'])

        # Manage position
        await self._manage_position(price)

        # Check cooldown
        async with self._cooldown_lock:
            if self._cooldown_until and now < self._cooldown_until:
                return None

        # Skip if has position
        async with self._position_lock:
            if self._position is not None:
                return None

        # Refresh H1 data
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        should_fetch = (
            self._h1_data is None or
            self._last_h1_bar_time is None or
            current_hour > self._last_h1_bar_time
        )

        if should_fetch and self.get_ohlcv:
            h1_data = await self.get_ohlcv(self.symbol, "H1", 200)
            if h1_data is not None and len(h1_data) > 0:
                self._h1_data = h1_data
                self._last_h1_bar_time = current_hour

        if self._h1_data is None or len(self._h1_data) < 100:
            return None

        # TIME FILTER
        in_kz, session = self.killzone.is_in_killzone(now)

        can_trade_outside = False
        if not in_kz:
            idx = len(self._h1_data) - 1
            col_map = self._get_col_map(self._h1_data)
            bar = self._h1_data.iloc[-1]
            recent_df = self._h1_data.iloc[max(0, idx-20):idx+1]

            activity = self.activity_filter.check_activity(
                now, bar[col_map['high']], bar[col_map['low']], recent_df
            )
            if activity.level in [ActivityLevel.HIGH, ActivityLevel.MODERATE] and activity.score >= 55:
                can_trade_outside = True
                session = "Hybrid"

        if not (in_kz or can_trade_outside):
            return None

        # REGIME CHECK
        if not regime_info or not regime_info.is_tradeable or regime_info.bias == 'NONE':
            return None

        direction = regime_info.bias

        # POI DETECTION
        idx = len(self._h1_data) - 1
        poi = self._detect_poi(self._h1_data, idx, direction)
        if not poi:
            return None

        # ENTRY TRIGGER
        bar = self._h1_data.iloc[-1]
        prev_bar = self._h1_data.iloc[-2]
        entry_type = self._check_entry_trigger(bar, prev_bar, direction)
        if not entry_type:
            return None

        # v5.8 ADAPTIVE RISK SCORING
        risk_mult, required_quality, risk_reason, should_skip = self.risk_scorer.calculate(
            current_time=now,
            entry_type=entry_type,
            in_killzone=in_kz,
            drift_detected=False  # Simplified
        )

        if should_skip:
            self.stats.signals_filtered += 1
            return None

        # AUTO RISK ADJUSTMENT
        col_map = self._get_col_map(self._h1_data)
        auto_assessment = self.auto_risk.assess(self._h1_data, col_map, now)
        auto_risk_mult = auto_assessment.risk_multiplier

        # KNOWN PERIOD PROTECTION
        known_period_mult = self._get_known_period_mult(now)

        # Combined risk multiplier (use most conservative)
        combined_mult = min(risk_mult, auto_risk_mult, known_period_mult)

        # Quality check
        if poi['quality'] < required_quality:
            self.stats.signals_filtered += 1
            return None

        # POSITION SIZING
        tp_pips = self.sl_pips * self.tp_rr
        adjusted_risk = self.base_risk * combined_mult
        risk_amount = account_balance * adjusted_risk
        lot_size = risk_amount / (self.sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            sl_price = price - self.sl_pips * 0.0001
            tp_price = price + tp_pips * 0.0001
        else:
            sl_price = price + self.sl_pips * 0.0001
            tp_price = price - tp_pips * 0.0001

        # Execute trade
        result = await self._execute_trade(
            direction=direction,
            entry_price=price,
            stop_loss=sl_price,
            take_profit=tp_price,
            volume=lot_size,
            poi_type=poi['type'],
            entry_type=entry_type,
            quality_score=poi['quality'],
            risk_multiplier=combined_mult
        )

        if result.success:
            self.stats.trades_executed += 1
            self.auto_risk.record_trade(now, True)  # Will be updated on close

            if self.send_telegram:
                await self._send_trade_notification(result, session, regime_info.regime.value, combined_mult)

        return result

    async def _execute_trade(self, **kwargs) -> TradeResult:
        """Execute trade via MT5"""
        if not self.place_market_order:
            return TradeResult(success=False, message="No execution callback")

        try:
            result = await self.place_market_order(
                symbol=self.symbol,
                order_type=kwargs['direction'],
                volume=kwargs['volume'],
                sl=kwargs['stop_loss'],
                tp=kwargs['take_profit'],
                comment=f"H1v58_{kwargs['entry_type']}",
                magic=self.magic_number
            )

            if result and result.get('ticket'):
                self._position = OpenPosition(
                    ticket=result['ticket'],
                    direction=kwargs['direction'],
                    entry_price=kwargs['entry_price'],
                    stop_loss=kwargs['stop_loss'],
                    take_profit=kwargs['take_profit'],
                    lot_size=kwargs['volume'],
                    entry_time=datetime.now(timezone.utc),
                    poi_type=kwargs['poi_type'],
                    entry_type=kwargs['entry_type'],
                    quality_score=kwargs['quality_score'],
                    risk_multiplier=kwargs['risk_multiplier']
                )

                logger.info(f"Trade: {kwargs['direction']} {kwargs['volume']} @ {kwargs['entry_price']:.5f}")

                return TradeResult(
                    success=True,
                    ticket=result['ticket'],
                    direction=kwargs['direction'],
                    entry_price=kwargs['entry_price'],
                    stop_loss=kwargs['stop_loss'],
                    take_profit=kwargs['take_profit'],
                    volume=kwargs['volume']
                )

            return TradeResult(success=False, message=f"Rejected: {result}")
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return TradeResult(success=False, message=str(e))

    async def _manage_position(self, price: float):
        """Manage open position"""
        async with self._position_lock:
            if self._position is None:
                return

            if self.get_positions:
                positions = await self.get_positions()
                exists = any(p['ticket'] == self._position.ticket for p in (positions or []))

                if not exists:
                    # Position closed
                    if self._position.direction == 'BUY':
                        pnl = (price - self._position.entry_price) * self._position.lot_size * self._contract_size
                    else:
                        pnl = (self._position.entry_price - price) * self._position.lot_size * self._contract_size

                    await self._handle_close(pnl)

    async def _handle_close(self, pnl: float):
        """Handle position close"""
        is_win = pnl > 0
        now = datetime.now(timezone.utc)

        async with self._cooldown_lock:
            if is_win:
                self.stats.trades_won += 1
                self.stats.total_profit += pnl
                self._cooldown_until = now + self.cooldown_after_tp
            else:
                self.stats.trades_lost += 1
                self.stats.total_loss += abs(pnl)
                self._cooldown_until = now + self.cooldown_after_sl

            self.stats.daily_pnl += pnl
            self.stats.daily_trades += 1

        self.auto_risk.record_trade(now, is_win)

        logger.info(f"Closed: P/L ${pnl:+.2f}, Daily ${self.stats.daily_pnl:+.2f}")

        if self.send_telegram:
            emoji = "WIN" if is_win else "LOSS"
            await self.send_telegram(f"{emoji}: ${pnl:+.2f} | Net: ${self.stats.net_pnl:+.2f}")

        self._position = None

    async def _send_trade_notification(self, result, session, regime, risk_mult):
        """Send trade notification"""
        msg = f"H1 v5.8 {result.direction}\n"
        msg += f"Entry: {result.entry_price:.5f}\n"
        msg += f"SL: {result.stop_loss:.5f}\n"
        msg += f"TP: {result.take_profit:.5f}\n"
        msg += f"Lot: {result.volume}\n"
        msg += f"Session: {session}\n"
        msg += f"Risk: {risk_mult:.0%}"

        try:
            await self.send_telegram(msg)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def pause(self):
        self.state = ExecutorState.PAUSED
        logger.info("Executor paused")

    def resume(self):
        self.state = ExecutorState.MONITORING
        logger.info("Executor resumed")

    def get_status(self) -> Dict:
        """Get executor status"""
        regime = self.regime_detector.last_info
        in_kz, session = self.killzone.is_in_killzone(datetime.now(timezone.utc))

        return {
            'state': self.state.value,
            'strategy': 'H1 v5.8',
            'warmup_count': self._warmup_count,
            'regime': regime.regime.value if regime else 'UNKNOWN',
            'bias': regime.bias if regime else 'NONE',
            'in_killzone': in_kz,
            'session': session,
            'has_position': self._position is not None,
            'stats': {
                'trades': self.stats.trades_executed,
                'wins': self.stats.trades_won,
                'losses': self.stats.trades_lost,
                'win_rate': self.stats.win_rate,
                'net_pnl': self.stats.net_pnl,
            },
            'circuit_breaker': {
                'triggered': self._circuit_breaker_triggered,
                'daily_pnl': self.stats.daily_pnl,
            }
        }
