"""Trade Executor - Main Trading Engine
=======================================

Combines all layers for automated trade execution:
1. Data Pipeline → Kalman Filter
2. Regime Detection → Kill Zone check
3. POI Detection → Entry Trigger
4. Risk Manager → Position sizing
5. Execute Trade → MT5/MCP
6. Exit Manager → Manage position

Author: SURIOTA Team
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..analysis.kalman_filter import MultiScaleKalman
from ..analysis.regime_detector import HMMRegimeDetector, MarketRegime
from ..analysis.poi_detector import POIDetector
from .entry_trigger import EntryTrigger, LTFEntrySignal
from .risk_manager import RiskManager
from .exit_manager import ExitManager, PositionState
from .trade_mode_manager import TradeModeManager, TradeMode, TradeModeConfig
from ..analysis.market_filter import MarketFilter, RelaxedEntryFilter
from .adaptive_risk import AdaptiveRiskManager, calculate_atr
from ..utils.intelligent_activity_filter import IntelligentActivityFilter, MarketActivity


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
    max_drawdown: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trades_executed == 0:
            return 0.0
        return self.trades_won / self.trades_executed * 100

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss

    @property
    def profit_factor(self) -> float:
        if self.total_loss == 0:
            return 999.0
        return self.total_profit / self.total_loss


class TradeExecutor:
    """Main trading execution engine"""

    def __init__(
        self,
        symbol: str = "GBPUSD",
        timeframe_htf: str = "H4",
        timeframe_ltf: str = "M5",
        warmup_bars: int = 100,
        magic_number: int = 20250125
    ):
        """Initialize Trade Executor

        Args:
            symbol: Trading symbol
            timeframe_htf: Higher timeframe for POI
            timeframe_ltf: Lower timeframe for entry
            warmup_bars: Bars for warmup
            magic_number: Magic number for trades
        """
        self.symbol = symbol
        self.timeframe_htf = timeframe_htf
        self.timeframe_ltf = timeframe_ltf
        self.warmup_bars = warmup_bars
        self.magic_number = magic_number

        # Components
        self.kalman = MultiScaleKalman()
        self.regime_detector = HMMRegimeDetector()
        self.poi_detector = POIDetector()
        # ZERO LOSING MONTHS CONFIGURATION
        self.entry_trigger = EntryTrigger(
            min_quality_score=75.0,        # ZERO LOSING MONTHS: Higher quality threshold
            max_sl_pips=10.0               # ZERO LOSING MONTHS: Max SL 10 pips
        )
        self.risk_manager = RiskManager(
            max_lot_size=0.5,              # Zero Losing Months config
            daily_loss_limit=80.0,         # Stricter daily limit
            max_sl_pips=10.0,              # ZERO LOSING MONTHS: Max SL 10 pips
            max_loss_per_trade_pct=0.1,    # ZERO LOSING MONTHS: Cap max loss at 0.1%
            monthly_loss_stop_pct=2.0      # Stop at 2% monthly loss
        )
        self.exit_manager = ExitManager()
        self.trade_mode_manager = TradeModeManager()

        # Market Filter - Trend alignment and volatility check
        # Added to match backtest system for better accuracy
        self.market_filter = MarketFilter(
            trend_threshold=0.6,        # 60% directional bars = trending
            min_volatility_pips=15.0,   # Min ATR to trade
            max_volatility_pips=80.0,   # Max ATR (too risky)
            atr_period=14,
            lookback_bars=20
        )

        # Relaxed Entry Filter - DISABLED for Zero Losing Months strategy
        # Stricter quality control produces better results
        self.relaxed_filter = RelaxedEntryFilter(
            min_quality_normal=65.0,       # Higher base quality
            min_quality_relaxed=60.0,      # Still require decent quality
            require_full_confirmation_normal=True,
            require_full_confirmation_relaxed=True  # Always require full confirmation
        )
        self.use_relaxed_filter = False  # DISABLED for zero-loss strategy

        # ZERO LOSING MONTHS: Adaptive Risk Manager Configuration
        # Optimized settings: SL10 + Quality75 + Loss0.1%
        self.adaptive_risk = AdaptiveRiskManager(
            base_max_lot=0.5,              # Conservative base for zero-loss
            base_min_sl=15.0,
            base_max_sl=10.0,              # ZERO LOSING MONTHS: Max SL 10 pips
            base_risk_percent=0.008,       # 0.8% max risk per trade
            max_loss_per_trade_pct=0.1,    # ZERO LOSING MONTHS: Cap max loss at 0.1%
            low_volatility_atr=12.0,       # ATR < 12 = reduce exposure
            high_volatility_atr=40.0,      # ATR > 40 = reduce exposure
            extreme_volatility_atr=55.0,   # ATR > 55 = heavily reduce
            consecutive_loss_threshold=2,  # Stricter: 2 losses = reduce
            drawdown_threshold=0.08,       # Stricter: 8% DD = reduce
            december_max_lot=0.01,         # Skip December (anomaly)
            december_min_quality=99.0      # Essentially skip December
        )
        self.use_adaptive_risk = True  # Enable by default

        # December skip flag (Zero Losing Months)
        self.skip_december = True

        # INTELLIGENT ACTIVITY FILTER - Replaces Kill Zone
        # Using INTEL_60 configuration (best from backtest: 72% WR, 2 losing months)
        self.use_intelligent_filter = True  # Enable intelligent filter
        self.intelligent_filter = IntelligentActivityFilter(
            activity_threshold=60.0,      # INTEL_60 threshold
            min_velocity_pips=2.0,        # Minimum velocity to consider active
            min_atr_pips=5.0,             # Minimum ATR
            pip_size=0.0001
        )
        self._last_intelligent_result = None
        self._last_activity = None  # For status display (same as _last_intelligent_result)

        # Status notification interval (configurable)
        self.status_interval_seconds = 1800  # 30 minutes default

        # Track recent trades for relaxed filter
        self._recent_trades: List[datetime] = []

        # Previous mode for change detection
        self._previous_mode: TradeMode = TradeMode.AUTO

        # State
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(start_time=datetime.now(timezone.utc))
        self._warmup_count = 0
        self._last_signal_time: Optional[datetime] = None
        self._cooldown_hours = 4

        # Previous state tracking for notifications
        self._prev_regime: Optional[str] = None
        self._prev_bias: Optional[str] = None
        self._prev_bull_pois: int = 0
        self._prev_bear_pois: int = 0
        self._prev_at_poi: bool = False
        self._last_status_time: Optional[datetime] = None
        self._last_risk_warning_time: Optional[datetime] = None
        self._risk_warning_cooldown = 3600  # Send warning at most every hour

        # MT5/MCP callbacks
        self.get_account_info: Optional[Callable] = None
        self.get_tick: Optional[Callable] = None
        self.get_ohlcv_htf: Optional[Callable] = None
        self.get_ohlcv_ltf: Optional[Callable] = None
        self.place_market_order: Optional[Callable] = None
        self.modify_position: Optional[Callable] = None
        self.close_position: Optional[Callable] = None
        self.close_partial: Optional[Callable] = None
        self.get_positions: Optional[Callable] = None

        # Telegram callback
        self.send_telegram: Optional[Callable] = None

    def set_callbacks(
        self,
        get_account_info: Callable = None,
        get_tick: Callable = None,
        get_ohlcv_htf: Callable = None,
        get_ohlcv_ltf: Callable = None,
        place_market_order: Callable = None,
        modify_position: Callable = None,
        close_position: Callable = None,
        close_partial: Callable = None,
        get_positions: Callable = None,
        send_telegram: Callable = None
    ):
        """Set execution callbacks"""
        self.get_account_info = get_account_info
        self.get_tick = get_tick
        self.get_ohlcv_htf = get_ohlcv_htf
        self.get_ohlcv_ltf = get_ohlcv_ltf
        self.place_market_order = place_market_order
        self.modify_position = modify_position
        self.close_position = close_position
        self.close_partial = close_partial
        self.get_positions = get_positions
        self.send_telegram = send_telegram

    async def warmup(self, htf_data, ltf_data) -> bool:
        """Perform warmup with historical data

        Args:
            htf_data: HTF OHLCV DataFrame
            ltf_data: LTF OHLCV DataFrame

        Returns:
            True if warmup successful
        """
        self.state = ExecutorState.WARMING_UP

        try:
            # Warmup Kalman filter
            close_col = 'close' if 'close' in ltf_data.columns else 'Close'
            for price in ltf_data[close_col].values:
                self.kalman.update(price)
                self._warmup_count += 1

            # Warmup regime detector
            smoothed_prices = self.kalman.medium.get_smoothed_series(len(ltf_data))
            for price in smoothed_prices:
                self.regime_detector.update(price)

            # Initial POI detection
            self.poi_detector.detect(htf_data)

            logger.info(f"Warmup complete with {self._warmup_count} bars")
            self.state = ExecutorState.MONITORING
            return True

        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self.state = ExecutorState.ERROR
            return False

    @property
    def is_warmup_complete(self) -> bool:
        """Check if warmup is complete"""
        return self._warmup_count >= self.warmup_bars

    def is_in_killzone(self) -> Tuple[bool, str]:
        """Check if in ICT Kill Zone

        Returns:
            Tuple of (in_killzone: bool, session_name: str)
        """
        now = datetime.now(timezone.utc)
        hour = now.hour

        # London Kill Zone: 08:00-12:00 UTC
        if 8 <= hour < 12:
            return True, "London"

        # New York Kill Zone: 13:00-17:00 UTC
        if 13 <= hour < 17:
            return True, "New York"

        # London Close: 15:00-17:00 UTC (overlap with NY)
        if 15 <= hour < 17:
            return True, "London Close"

        return False, "Off-session"

    def is_cooldown_active(self) -> bool:
        """Check if signal cooldown is active"""
        if self._last_signal_time is None:
            return False

        now = datetime.now(timezone.utc)
        hours_since = (now - self._last_signal_time).total_seconds() / 3600
        return hours_since < self._cooldown_hours

    async def _send_condition_notification(
        self,
        regime_info,
        bull_pois: int,
        bear_pois: int,
        at_poi: bool,
        poi_info,
        price: float
    ):
        """Send Telegram notification when conditions change"""
        if not self.send_telegram:
            return

        notifications = []

        # Check regime change
        current_regime = regime_info.regime.value if regime_info else None
        current_bias = regime_info.bias if regime_info else None

        if current_regime and current_regime != self._prev_regime:
            emoji = "🟢" if current_regime == "BULLISH" else ("🔴" if current_regime == "BEARISH" else "⚪")
            notifications.append(f"{emoji} <b>Regime Changed:</b> {current_regime}")
            self._prev_regime = current_regime

        if current_bias and current_bias != self._prev_bias:
            notifications.append(f"📊 <b>Bias:</b> {current_bias}")
            self._prev_bias = current_bias

        # Check POI changes (only notify significant changes)
        if bull_pois != self._prev_bull_pois or bear_pois != self._prev_bear_pois:
            if bull_pois > self._prev_bull_pois:
                notifications.append(f"🎯 <b>New Bullish POI detected!</b> Total: {bull_pois}")
            if bear_pois > self._prev_bear_pois:
                notifications.append(f"🎯 <b>New Bearish POI detected!</b> Total: {bear_pois}")
            self._prev_bull_pois = bull_pois
            self._prev_bear_pois = bear_pois

        # Check if price entered POI zone
        if at_poi and not self._prev_at_poi and poi_info:
            notifications.append(
                f"⚡ <b>Price at POI Zone!</b>\n"
                f"├ Price: {price:.5f}\n"
                f"├ Zone: {poi_info.zone_low:.5f} - {poi_info.zone_high:.5f}\n"
                f"└ Waiting for entry trigger..."
            )
        self._prev_at_poi = at_poi

        # Send notifications
        if notifications:
            msg = "🔔 <b>SURGE-WSI Alert</b>\n\n" + "\n".join(notifications)
            await self.send_telegram(msg)

    async def _send_periodic_status(self, price: float, regime_info, bull_pois: int, bear_pois: int):
        """Send periodic status update (configurable interval)"""
        if not self.send_telegram:
            return

        now = datetime.now(timezone.utc)

        # Send status at configurable interval (default 30 minutes)
        if self._last_status_time is None or (now - self._last_status_time).total_seconds() >= self.status_interval_seconds:
            in_kz, session = self.is_in_killzone()
            regime = regime_info.regime.value if regime_info else "N/A"
            bias = regime_info.bias if regime_info else "N/A"

            # Activity info - with safe access
            activity = self._last_activity
            if activity is not None:
                activity_emoji = activity.get_emoji()
                activity_score = f"{activity.score:.0f}"
                activity_level = activity.activity.value.upper()  # Fixed: was .level, should be .activity
            else:
                activity_emoji = "⚪"
                activity_score = "N/A"
                activity_level = "N/A"

            msg = f"📊 <b>Status Update</b>\n\n"
            msg += f"├ Price: {price:.5f}\n"
            msg += f"├ Session: {session}\n"
            msg += f"├ Kill Zone: {'Yes' if in_kz else 'No'}\n"
            msg += f"├ Activity: {activity_emoji} {activity_level} ({activity_score}/100)\n"
            msg += f"├ Regime: {regime}\n"
            msg += f"├ Bias: {bias}\n"
            msg += f"├ Bullish POIs: {bull_pois}\n"
            msg += f"└ Bearish POIs: {bear_pois}\n"

            # Intelligent filter mode status
            if self.use_intelligent_filter:
                msg += "\n🧠 <b>Intelligent Filter:</b> Active (INTEL_60)\n"
                if activity is not None:
                    if activity.should_trade:
                        msg += f"├ Market: {activity.activity.value.upper()} ✅\n"
                        msg += f"└ Quality threshold: {activity.quality_threshold:.0f}\n"
                    else:
                        msg += f"├ Market: {activity.activity.value.upper()} ⏸️\n"
                        msg += f"└ Waiting for activity...\n"

            # Waiting status
            if activity is not None and not activity.should_trade:
                msg += "\n⏳ <i>Waiting for market activity...</i>"
            elif bias == "SELL" and bear_pois == 0:
                msg += "\n⏳ <i>Waiting for Bearish POI...</i>"
            elif bias == "BUY" and bull_pois == 0:
                msg += "\n⏳ <i>Waiting for Bullish POI...</i>"

            await self.send_telegram(msg)
            self._last_status_time = now

    async def _send_risk_warnings(self):
        """Check and send risk warning notifications"""
        if not self.send_telegram:
            return

        now = datetime.now(timezone.utc)

        # Cooldown check - don't spam warnings
        if self._last_risk_warning_time is not None:
            time_since = (now - self._last_risk_warning_time).total_seconds()
            if time_since < self._risk_warning_cooldown:
                return

        # Get warnings from risk manager
        warnings = self.risk_manager.check_risk_warnings()

        if warnings["has_warning"]:
            msg = "⚠️ <b>RISK WARNING</b>\n\n"
            for warning_msg in warnings["messages"]:
                msg += f"{warning_msg}\n"

            if warnings["risk_reduced"]:
                msg += "\n<i>Position sizes automatically reduced</i>"

            await self.send_telegram(msg)
            self._last_risk_warning_time = now
            logger.warning(f"Risk warning sent: {warnings['messages']}")

    async def process_tick(
        self,
        price: float,
        account_balance: float
    ) -> Optional[TradeResult]:
        """Process new tick and potentially execute trade

        Args:
            price: Current price
            account_balance: Account balance

        Returns:
            TradeResult if trade executed
        """
        if self.state not in [ExecutorState.MONITORING, ExecutorState.TRADING]:
            return None

        # Update Kalman
        kalman_state = self.kalman.update(price)
        smoothed_price = kalman_state['smoothed_price']

        # Update regime
        regime_info = self.regime_detector.update(smoothed_price)

        # Check existing positions
        await self._manage_positions(price)

        # Check and send risk warnings (approaching limits)
        await self._send_risk_warnings()

        # Check if can trade (includes December skip and monthly limit)
        now = datetime.now(timezone.utc)
        can_trade, reason = self.risk_manager.can_open_trade(current_time=now)
        if not can_trade:
            logger.debug(f"Cannot trade: {reason}")
            return None

        # Additional December check (Zero Losing Months)
        if self.skip_december and now.month == 12:
            logger.debug("December trading disabled (Zero Losing Months)")
            return None

        # INTELLIGENT ACTIVITY FILTER - Replaces fixed Kill Zone
        # Uses Kalman velocity, ATR, and momentum to detect market activity
        if self.use_intelligent_filter:
            try:
                # Get current price data for filter
                tick_data = await self.get_tick(self.symbol) if self.get_tick else None
                current_high = tick_data.get('high', price) if tick_data else price
                current_low = tick_data.get('low', price) if tick_data else price

                # Update intelligent filter with Kalman velocity
                self.intelligent_filter.update(current_high, current_low, price)
                self.intelligent_filter.update_kalman_velocity(kalman_state['velocity'])

                # Check activity
                intel_result = self.intelligent_filter.check(now, current_high, current_low, price)

                # Validate result
                if intel_result is None:
                    logger.warning("Intelligent filter returned None, falling back to Kill Zone")
                    in_kz, session = self.is_in_killzone()
                    if not in_kz:
                        return None
                else:
                    self._last_intelligent_result = intel_result
                    self._last_activity = intel_result  # Store for status display

                    if not intel_result.should_trade:
                        logger.debug(f"Intelligent filter: {intel_result.reason}")
                        return None

                    logger.debug(f"Intelligent filter: ACTIVE ({intel_result.activity.value}, score={intel_result.score:.0f})")
            except Exception as e:
                # Fallback to Kill Zone on error
                logger.warning(f"Intelligent filter error: {e}, falling back to Kill Zone")
                in_kz, session = self.is_in_killzone()
                if not in_kz:
                    return None
        else:
            # Fallback to Kill Zone (legacy mode)
            in_kz, session = self.is_in_killzone()
            if not in_kz:
                return None

        # Check cooldown
        if self.is_cooldown_active():
            return None

        # Check regime
        if not regime_info.is_tradeable:
            self.stats.signals_filtered += 1
            return None

        direction = regime_info.bias
        if direction == 'NONE':
            return None

        # Get HTF data for POI and trend filter
        htf_data = None
        if self.get_ohlcv_htf:
            htf_data = await self.get_ohlcv_htf(self.symbol, self.timeframe_htf, 200)
            if htf_data is not None:
                self.poi_detector.detect(htf_data)
                self.poi_detector.update_mitigation(htf_data)
        else:
            logger.debug("No get_ohlcv_htf callback")
            return None

        # Trend Filter - Check alignment with HTF trend
        if htf_data is not None and len(htf_data) >= 20:
            aligned, trend_reason = self.market_filter.check_trend_alignment(htf_data, direction)
            if not aligned:
                logger.debug(f"Trend filter: {trend_reason}")
                self.stats.signals_filtered += 1
                return None

        # Check if price at POI
        poi_result = self.poi_detector.last_result
        if poi_result is None:
            logger.debug("No POI result available")
            return None

        # Count active POIs
        bull_pois = len(poi_result.bullish_pois) if poi_result.bullish_pois else 0
        bear_pois = len(poi_result.bearish_pois) if poi_result.bearish_pois else 0
        logger.debug(f"POIs available: {bull_pois} bullish, {bear_pois} bearish")

        at_poi, poi_info = poi_result.price_at_poi(price, direction)

        # Send condition notifications
        await self._send_condition_notification(
            regime_info, bull_pois, bear_pois, at_poi, poi_info, price
        )

        # Send periodic status
        await self._send_periodic_status(price, regime_info, bull_pois, bear_pois)

        if not at_poi or poi_info is None:
            logger.debug(f"Price {price:.5f} not at any {direction} POI zone")
            return None

        logger.info(f"Price at POI! Zone: {poi_info.zone_high:.5f}-{poi_info.zone_low:.5f}")

        # Get LTF data for entry confirmation
        if self.get_ohlcv_ltf:
            ltf_data = await self.get_ohlcv_ltf(self.symbol, self.timeframe_ltf, 100)
            if ltf_data is None:
                logger.debug("Failed to get LTF data")
                return None
        else:
            logger.debug("No get_ohlcv_ltf callback")
            return None

        # Relaxed Entry Filter - Check if we should relax entry requirements
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(days=7)
        recent_trade_count = sum(1 for t in self._recent_trades if t > cutoff_time)
        min_quality, require_full_confirmation = self.relaxed_filter.get_entry_params(
            recent_trade_count=recent_trade_count,
            lookback_days=7
        )

        # Use adaptive quality from intelligent filter if available
        if self.use_intelligent_filter and self._last_intelligent_result is not None:
            # Intelligent filter provides adaptive quality based on market activity
            adaptive_quality = self._last_intelligent_result.quality_threshold
            min_quality = max(min_quality, adaptive_quality)  # Use higher of the two
            logger.debug(f"Intelligent filter adaptive quality: {adaptive_quality} -> using {min_quality}")

        # Temporarily adjust entry trigger's min_quality_score
        original_min_quality = self.entry_trigger.min_quality_score
        self.entry_trigger.min_quality_score = min_quality

        # Check entry confirmation
        should_enter, signal = self.entry_trigger.check_for_entry(
            ltf_data, direction, poi_info, price,
            require_full_confirmation=require_full_confirmation
        )

        # Restore original min_quality
        self.entry_trigger.min_quality_score = original_min_quality

        if not should_enter or signal is None:
            logger.debug(f"Entry trigger not confirmed for {direction}")
            return None

        # Track this trade time for relaxed filter
        self._recent_trades.append(now)
        # Clean up old trades from tracking
        self._recent_trades = [t for t in self._recent_trades if t > cutoff_time]

        logger.info(f"Entry trigger confirmed! Signal: {signal.signal_type}")

        # Send entry trigger notification
        if self.send_telegram:
            msg = f"🚨 <b>Entry Trigger Confirmed!</b>\n\n"
            msg += f"├ Direction: {direction}\n"
            msg += f"├ Entry: {signal.entry_price:.5f}\n"
            msg += f"├ Stop Loss: {signal.stop_loss:.5f}\n"
            msg += f"├ SL Pips: {signal.sl_pips:.1f}\n"
            msg += f"├ Quality: {signal.quality_score:.0f}%\n"
            msg += f"└ Type: {signal.signal_type}\n"
            msg += "\n⏳ <i>Checking risk & executing...</i>"
            await self.send_telegram(msg)

        self.stats.total_signals += 1

        # Apply adaptive risk settings based on market conditions
        if self.use_adaptive_risk and htf_data is not None:
            # Calculate ATR from HTF data
            atr_pips = calculate_atr(htf_data, period=14) if len(htf_data) >= 14 else signal.sl_pips * 1.5
            regime_confidence = regime_info.probability if regime_info else 0.7

            # Get adaptive parameters
            adaptive_params = self.adaptive_risk.get_adaptive_params(
                current_balance=account_balance,
                atr_pips=atr_pips,
                regime_confidence=regime_confidence,
                current_time=datetime.now(timezone.utc)
            )

            # Apply adaptive limits to risk manager
            self.risk_manager.max_lot_size = adaptive_params.max_lot_size
            self.risk_manager.min_sl_pips = adaptive_params.min_sl_pips
            self.risk_manager.max_sl_pips = adaptive_params.max_sl_pips

            logger.debug(f"Adaptive risk: {adaptive_params.reason} | MaxLot={adaptive_params.max_lot_size} ATR={atr_pips:.1f}")

        # Calculate position size
        risk_params = self.risk_manager.calculate_lot_size(
            account_balance,
            signal.quality_score,
            signal.sl_pips
        )

        # Zero Losing Months: Enforce max loss per trade cap
        # Calculate max lot that respects loss cap
        max_loss_pct = self.adaptive_risk.max_loss_per_trade_pct if hasattr(self.adaptive_risk, 'max_loss_per_trade_pct') else 0.1  # ZERO LOSING MONTHS
        max_lot_from_cap = self.risk_manager.calculate_lot_with_loss_cap(
            account_balance,
            signal.sl_pips,
            max_loss_pct
        )

        # Use the smaller of calculated lot and loss-capped lot
        if risk_params.lot_size > max_lot_from_cap:
            logger.debug(f"Lot capped: {risk_params.lot_size} -> {max_lot_from_cap} (max loss {max_loss_pct}%)")
            risk_params = self.risk_manager.calculate_lot_size(
                account_balance,
                signal.quality_score,
                signal.sl_pips
            )
            # Override lot size with capped value
            risk_params.lot_size = max_lot_from_cap
            risk_params.risk_amount = max_lot_from_cap * signal.sl_pips * self.risk_manager.pip_value

        # Validate risk
        valid, msg = self.risk_manager.validate_risk(
            account_balance,
            risk_params.lot_size,
            signal.sl_pips
        )

        if not valid:
            logger.warning(f"Risk validation failed: {msg}")
            self.stats.signals_filtered += 1
            return None

        # Calculate ATR for mode evaluation
        atr_pips = signal.sl_pips * 1.5  # Approximate ATR from SL distance

        # Evaluate trade mode
        current_mode = self.trade_mode_manager.evaluate_mode(
            current_time=datetime.now(timezone.utc),
            current_balance=account_balance,
            atr_pips=atr_pips
        )

        # Track regime changes
        regime_name = regime_info.regime.value if regime_info else 'UNKNOWN'
        self.trade_mode_manager.record_regime_change(regime_name)

        # Check for mode change and notify
        if current_mode != self._previous_mode:
            self._previous_mode = current_mode
            await self._send_mode_change_alert()

        # Monitoring mode: full pause - don't send signals, just log
        if current_mode == TradeMode.MONITORING:
            logger.debug(f"Monitoring mode (paused): {self.trade_mode_manager.mode_reason}")
            # Don't send signal, just log for analysis
            return TradeResult(
                success=False,
                direction=direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                volume=risk_params.lot_size,
                message=f"Monitoring mode (paused): {self.trade_mode_manager.mode_reason}"
            )

        # Signal-only mode: send signal but don't execute
        if current_mode == TradeMode.SIGNAL_ONLY:
            logger.info(f"Signal-only mode: {self.trade_mode_manager.mode_reason}")
            await self._send_signal_only_alert(signal, risk_params, direction)
            self._last_signal_time = datetime.now(timezone.utc)
            self.stats.total_signals += 1
            return TradeResult(
                success=False,
                direction=direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                volume=risk_params.lot_size,
                message=f"Signal-only mode: {self.trade_mode_manager.mode_reason}"
            )

        # AUTO mode: Execute trade
        result = await self._execute_trade(
            direction=direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            volume=risk_params.lot_size,
            signal=signal
        )

        if result.success:
            self._last_signal_time = datetime.now(timezone.utc)
            self.stats.trades_executed += 1
            self.risk_manager.register_trade_open()

            # Create managed position in exit manager
            self.exit_manager.create_position(
                ticket=result.ticket,
                symbol=self.symbol,
                direction=direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                volume=risk_params.lot_size
            )

            # Send Telegram notification
            if self.send_telegram:
                await self._send_trade_alert(result, signal, risk_params)

        return result

    async def _execute_trade(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        volume: float,
        signal: LTFEntrySignal
    ) -> TradeResult:
        """Execute trade via MT5/MCP

        Args:
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            stop_loss: Stop loss price
            volume: Lot size
            signal: Entry signal details

        Returns:
            TradeResult
        """
        if not self.place_market_order:
            logger.warning("No place_market_order callback set")
            return TradeResult(
                success=False,
                message="No execution callback"
            )

        try:
            # Get TP levels from exit manager config
            sl_distance = abs(entry_price - stop_loss)
            if direction == 'BUY':
                take_profit = entry_price + (sl_distance * self.exit_manager.tp3_rr)
            else:
                take_profit = entry_price - (sl_distance * self.exit_manager.tp3_rr)

            # Execute order
            result = await self.place_market_order(
                symbol=self.symbol,
                order_type=direction,
                volume=volume,
                sl=stop_loss,
                tp=take_profit,
                comment=f"SURGE-WSI {signal.confirmation_type.value}",
                magic=self.magic_number
            )

            if result and result.get('ticket'):
                logger.info(
                    f"Trade executed: {direction} {volume} {self.symbol} "
                    f"@ {entry_price:.5f}, Ticket: {result['ticket']}"
                )
                return TradeResult(
                    success=True,
                    ticket=result['ticket'],
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    volume=volume,
                    message="Trade executed successfully"
                )
            else:
                return TradeResult(
                    success=False,
                    message=f"Order rejected: {result}"
                )

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResult(
                success=False,
                message=str(e)
            )

    async def _manage_positions(self, current_price: float):
        """Manage open positions via exit manager"""
        for pos in self.exit_manager.get_all_positions():
            action, details = self.exit_manager.update_position(
                pos.ticket,
                current_price
            )

            if action is None:
                continue

            try:
                if action == 'CLOSE_PARTIAL_TP1':
                    if self.close_partial:
                        await self.close_partial(pos.ticket, details['close_volume'])
                        self.exit_manager.update_volume_after_partial(
                            pos.ticket, details['close_volume']
                        )

                    if details.get('move_sl_to_be') and self.modify_position:
                        await self.modify_position(pos.ticket, sl=pos.entry_price)
                        self.exit_manager.set_breakeven(pos.ticket)

                elif action == 'CLOSE_PARTIAL_TP2':
                    if self.close_partial:
                        await self.close_partial(pos.ticket, details['close_volume'])
                        self.exit_manager.update_volume_after_partial(
                            pos.ticket, details['close_volume']
                        )

                elif action == 'CLOSE_ALL_TP3':
                    if self.close_position:
                        await self.close_position(pos.ticket)
                        self._handle_trade_close(pos, 'TP3', pos.tp3.price)

                elif action == 'CLOSE_ALL_SL':
                    # SL hit is handled by broker, just clean up
                    self._handle_trade_close(pos, 'SL', current_price)

                elif action == 'UPDATE_TRAILING':
                    if self.modify_position:
                        await self.modify_position(
                            pos.ticket,
                            sl=details['new_sl']
                        )

            except Exception as e:
                logger.error(f"Position management failed: {e}")

    def _handle_trade_close(self, pos: PositionState, result: str, exit_price: float):
        """Handle trade close for statistics

        Updates all tracking: daily, monthly, adaptive risk.
        """
        if pos.direction == 'BUY':
            profit_pips = (exit_price - pos.entry_price) / 0.0001
        else:
            profit_pips = (pos.entry_price - exit_price) / 0.0001

        profit_usd = profit_pips * pos.original_volume * 10  # Approx
        is_win = profit_usd >= 0

        if is_win:
            self.stats.trades_won += 1
            self.stats.total_profit += profit_usd
        else:
            self.stats.trades_lost += 1
            self.stats.total_loss += abs(profit_usd)

        # Record with trade mode manager for consecutive loss tracking
        self.trade_mode_manager.record_trade_result(is_win, profit_usd)

        # Record with adaptive risk manager for performance tracking
        if self.use_adaptive_risk:
            self.adaptive_risk.record_trade_result(is_win, profit_usd)

        # Register trade close (updates both daily and monthly stats)
        self.risk_manager.register_trade_close(profit_usd)
        self.exit_manager.close_position(pos.ticket, result)
        self.exit_manager.remove_position(pos.ticket)

        # Log with monthly info (Zero Losing Months tracking)
        monthly_stats = self.risk_manager.get_monthly_stats()
        logger.info(f"Trade closed: {result}, P/L: ${profit_usd:+.2f}, Monthly: ${monthly_stats['pnl']:+.2f}")

    async def _send_trade_alert(self, result: TradeResult, signal: LTFEntrySignal, risk_params):
        """Send trade alert to Telegram"""
        if not self.send_telegram:
            return

        sl_pips = signal.sl_pips
        rr_levels = signal.get_risk_reward_levels(2.0)

        msg = f"🔔 <b>{result.direction} Signal</b>\n\n"
        msg += f"📊 Symbol: {self.symbol}\n"
        msg += f"💰 Entry: <code>{result.entry_price:.5f}</code>\n"
        msg += f"🛑 SL: <code>{result.stop_loss:.5f}</code> ({sl_pips:.1f} pips)\n"
        msg += f"🎯 TP1: <code>{rr_levels['take_profit']:.5f}</code>\n\n"
        msg += f"📈 Lot: {result.volume}\n"
        msg += f"💵 Risk: ~${risk_params.risk_amount:.2f}\n"
        msg += f"⭐ Quality: {signal.quality_score:.0f}%\n"
        msg += f"🔍 Type: {signal.confirmation_type.value}"

        try:
            await self.send_telegram(msg)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def _send_signal_only_alert(
        self,
        signal: LTFEntrySignal,
        risk_params,
        direction: str
    ):
        """Send signal-only alert (no execution) to Telegram"""
        if not self.send_telegram:
            return

        # Calculate TP levels
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        if direction == 'BUY':
            tp1 = signal.entry_price + (sl_distance * self.exit_manager.tp1_rr)
            tp2 = signal.entry_price + (sl_distance * self.exit_manager.tp2_rr)
            tp3 = signal.entry_price + (sl_distance * self.exit_manager.tp3_rr)
        else:
            tp1 = signal.entry_price - (sl_distance * self.exit_manager.tp1_rr)
            tp2 = signal.entry_price - (sl_distance * self.exit_manager.tp2_rr)
            tp3 = signal.entry_price - (sl_distance * self.exit_manager.tp3_rr)

        regime_info = self.regime_detector.last_info
        regime = regime_info.regime.value if regime_info else "UNKNOWN"

        msg = f"🟡 <b>SIGNAL ONLY - {direction}</b>\n"
        msg += f"<i>(No auto-execution)</i>\n\n"
        msg += f"📊 Symbol: {self.symbol}\n"
        msg += f"💰 Entry: <code>{signal.entry_price:.5f}</code>\n"
        msg += f"🛑 SL: <code>{signal.stop_loss:.5f}</code> ({signal.sl_pips:.1f} pips)\n\n"
        msg += f"<b>Take Profit Levels:</b>\n"
        msg += f"├ TP1 (50%): <code>{tp1:.5f}</code>\n"
        msg += f"├ TP2 (30%): <code>{tp2:.5f}</code>\n"
        msg += f"└ TP3 (20%): <code>{tp3:.5f}</code>\n\n"
        msg += f"📈 Suggested Lot: {risk_params.lot_size}\n"
        msg += f"💵 Risk: ~${risk_params.risk_amount:.2f}\n"
        msg += f"⭐ Quality: {signal.quality_score:.0f}%\n"
        msg += f"🔍 Regime: {regime}\n\n"
        msg += f"⚠️ <b>Mode Reason:</b>\n"
        msg += f"<code>{self.trade_mode_manager.mode_reason}</code>"

        try:
            await self.send_telegram(msg)
            logger.info(f"Signal-only alert sent for {direction} {self.symbol}")
        except Exception as e:
            logger.error(f"Signal-only Telegram send failed: {e}")

    async def _send_mode_change_alert(self):
        """Send mode change notification to Telegram"""
        if not self.send_telegram:
            return

        status = self.trade_mode_manager.get_status()
        current_mode = self.trade_mode_manager.current_mode

        if current_mode == TradeMode.AUTO:
            msg = "🟢 <b>Mode Changed: AUTO TRADE</b>\n\n"
        elif current_mode == TradeMode.RECOVERY:
            msg = f"🟠 <b>Mode Changed: RECOVERY ({int(status['lot_multiplier']*100)}% lot)</b>\n\n"
        elif current_mode == TradeMode.MONITORING:
            msg = "🔴 <b>Mode Changed: MONITORING ONLY</b>\n\n"
            msg += "⏸️ <i>Trading fully paused - observing market only</i>\n"
            msg += "📊 <i>Signals will NOT be sent during this period</i>\n\n"
        else:
            msg = "🟡 <b>Mode Changed: SIGNAL ONLY</b>\n\n"

        msg += f"<b>Reason:</b>\n<code>{status['reason']}</code>\n\n"
        msg += f"├ Daily P/L: ${status['daily_pnl']:+.2f}\n"
        msg += f"├ Daily Trades: {status['daily_trades']}\n"
        msg += f"├ Win Rate: {status['daily_win_rate']:.0f}%\n"
        msg += f"├ Consecutive Losses: {status['consecutive_losses']}\n"
        msg += f"└ Weekly P/L: ${status['weekly_pnl']:+.2f}"

        try:
            await self.send_telegram(msg)
            logger.info(f"Mode change alert sent: {current_mode.value}")
        except Exception as e:
            logger.error(f"Mode change Telegram send failed: {e}")

    def pause(self):
        """Pause trading"""
        self.state = ExecutorState.PAUSED
        logger.info("Executor paused")

    def resume(self):
        """Resume trading"""
        if self.is_warmup_complete:
            self.state = ExecutorState.MONITORING
        else:
            self.state = ExecutorState.WARMING_UP
        logger.info("Executor resumed")

    def stop(self):
        """Stop executor"""
        self.state = ExecutorState.STOPPED
        logger.info("Executor stopped")

    def get_status(self) -> Dict:
        """Get executor status"""
        regime = self.regime_detector.last_info
        in_kz, session = self.is_in_killzone()
        mode_status = self.trade_mode_manager.get_status()

        # Activity info (from IntelligentActivityFilter)
        activity_info = {}
        if self._last_activity:
            activity_info = {
                'level': self._last_activity.activity.value,  # MarketActivity enum
                'score': self._last_activity.score,
                'atr_pips': self._last_activity.atr_pips,
                'should_trade': self._last_activity.should_trade,
                'velocity': self._last_activity.velocity,
                'quality_threshold': self._last_activity.quality_threshold
            }

        # Get monthly stats (Zero Losing Months)
        monthly_stats = self.risk_manager.get_monthly_stats()

        return {
            'state': self.state.value,
            'warmup_complete': self.is_warmup_complete,
            'warmup_count': self._warmup_count,
            'regime': regime.regime.value if regime else 'UNKNOWN',
            'regime_probability': regime.probability if regime else 0,
            'bias': regime.bias if regime else 'NONE',
            'in_killzone': in_kz,
            'session': session,
            'intelligent_filter': self.use_intelligent_filter,
            'activity': activity_info,
            'cooldown_active': self.is_cooldown_active(),
            'open_positions': self.risk_manager.open_positions,
            'daily_pnl': self.risk_manager.daily_pnl,
            'monthly_pnl': monthly_stats['pnl'],
            'monthly_trades': monthly_stats['trades'],
            'trade_mode': mode_status['mode'],
            'mode_reason': mode_status['reason'],
            'consecutive_losses': mode_status['consecutive_losses'],
            'regime_changes_today': mode_status['regime_changes_today'],
            'december_skip': self.skip_december,
            'stats': {
                'total_signals': self.stats.total_signals,
                'trades_executed': self.stats.trades_executed,
                'win_rate': self.stats.win_rate,
                'net_pnl': self.stats.net_pnl,
            }
        }

    def reset(self):
        """Reset executor state"""
        self.kalman.reset()
        self.regime_detector.reset()
        self.poi_detector.reset()
        self.entry_trigger.reset()
        self.exit_manager.reset()
        self.intelligent_filter.reset()
        self.trade_mode_manager = TradeModeManager()
        self._previous_mode = TradeMode.AUTO
        self._last_intelligent_result = None
        self._last_activity = None
        self._last_risk_warning_time = None
        self._warmup_count = 0
        self._last_signal_time = None
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(start_time=datetime.now(timezone.utc))
        logger.info("Executor reset")

    def initialize_daily_stats(self, current_balance: float):
        """Initialize daily/weekly/monthly stats with current balance

        Call this at the start of each trading day.

        Args:
            current_balance: Current account balance
        """
        now = datetime.now(timezone.utc)

        # Reset daily stats
        self.trade_mode_manager.reset_daily_stats(current_balance)

        # Reset weekly stats on Monday
        if now.weekday() == 0:  # Monday
            self.trade_mode_manager.reset_weekly_stats(current_balance)

        # Reset monthly stats on 1st of month (Zero Losing Months)
        if now.day == 1:
            self.risk_manager.start_month(current_balance)
            logger.info(f"Monthly stats initialized for {now.strftime('%B %Y')}")

        logger.info(f"Daily stats initialized with balance ${current_balance:,.2f}")

    def update_balance(self, new_balance: float):
        """Update current balance for mode tracking

        Args:
            new_balance: Current account balance
        """
        self.trade_mode_manager.update_balance(new_balance)
