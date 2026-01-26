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
        self.entry_trigger = EntryTrigger()
        self.risk_manager = RiskManager()
        self.exit_manager = ExitManager()
        self.trade_mode_manager = TradeModeManager()

        # Previous mode for change detection
        self._previous_mode: TradeMode = TradeMode.AUTO

        # State
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(start_time=datetime.now(timezone.utc))
        self._warmup_count = 0
        self._last_signal_time: Optional[datetime] = None
        self._cooldown_hours = 4

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

        # Check if can trade
        can_trade, reason = self.risk_manager.can_open_trade()
        if not can_trade:
            return None

        # Check Kill Zone
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

        # Get HTF data for POI
        if self.get_ohlcv_htf:
            htf_data = await self.get_ohlcv_htf(self.symbol, self.timeframe_htf, 200)
            if htf_data is not None:
                self.poi_detector.detect(htf_data)
                self.poi_detector.update_mitigation(htf_data)

        # Check if price at POI
        poi_result = self.poi_detector.last_result
        if poi_result is None:
            return None

        at_poi, poi_info = poi_result.price_at_poi(price, direction)
        if not at_poi or poi_info is None:
            return None

        # Get LTF data for entry confirmation
        if self.get_ohlcv_ltf:
            ltf_data = await self.get_ohlcv_ltf(self.symbol, self.timeframe_ltf, 100)
            if ltf_data is None:
                return None
        else:
            return None

        # Check entry confirmation
        should_enter, signal = self.entry_trigger.check_for_entry(
            ltf_data, direction, poi_info, price
        )

        if not should_enter or signal is None:
            return None

        self.stats.total_signals += 1

        # Calculate position size
        risk_params = self.risk_manager.calculate_lot_size(
            account_balance,
            signal.quality_score,
            signal.sl_pips
        )

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
        """Handle trade close for statistics"""
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

        self.risk_manager.register_trade_close(profit_usd)
        self.exit_manager.close_position(pos.ticket, result)
        self.exit_manager.remove_position(pos.ticket)

        logger.info(f"Trade closed: {result}, P/L: ${profit_usd:+.2f}")

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

        return {
            'state': self.state.value,
            'warmup_complete': self.is_warmup_complete,
            'warmup_count': self._warmup_count,
            'regime': regime.regime.value if regime else 'UNKNOWN',
            'regime_probability': regime.probability if regime else 0,
            'bias': regime.bias if regime else 'NONE',
            'in_killzone': in_kz,
            'session': session,
            'cooldown_active': self.is_cooldown_active(),
            'open_positions': self.risk_manager.open_positions,
            'daily_pnl': self.risk_manager.daily_pnl,
            'trade_mode': mode_status['mode'],
            'mode_reason': mode_status['reason'],
            'consecutive_losses': mode_status['consecutive_losses'],
            'regime_changes_today': mode_status['regime_changes_today'],
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
        self.trade_mode_manager = TradeModeManager()
        self._previous_mode = TradeMode.AUTO
        self._warmup_count = 0
        self._last_signal_time = None
        self.state = ExecutorState.IDLE
        self.stats = ExecutorStats(start_time=datetime.now(timezone.utc))
        logger.info("Executor reset")

    def initialize_daily_stats(self, current_balance: float):
        """Initialize daily/weekly stats with current balance

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

        logger.info(f"Daily stats initialized with balance ${current_balance:,.2f}")

    def update_balance(self, new_balance: float):
        """Update current balance for mode tracking

        Args:
            new_balance: Current account balance
        """
        self.trade_mode_manager.update_balance(new_balance)
