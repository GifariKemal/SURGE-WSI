"""
ML Trading Bot - Telegram Alerts
================================

Telegram notifications for ML Trading Bot:
- Trade signals and executions
- Position close alerts
- Status updates
- Command handlers

Uses existing TelegramNotifier from src/utils/telegram.py

Author: SURIOTA Team
"""

import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Callable, Any
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.telegram import TelegramNotifier, TelegramFormatter


class MLTelegramNotifier:
    """
    Telegram notification handler for ML Trading Bot

    Features:
    - Trade execution alerts
    - Position close notifications
    - ML regime and signal updates
    - Command handlers (/status, /balance, etc.)
    """

    def __init__(
        self,
        bot_token: str = None,
        chat_id: str = None,
        enabled: bool = True
    ):
        """
        Initialize ML Telegram Notifier

        Args:
            bot_token: Telegram bot token (from .env if not provided)
            chat_id: Telegram chat ID (from .env if not provided)
            enabled: Enable notifications
        """
        # Load from .env if not provided
        load_dotenv()

        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = enabled and os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

        # Core notifier
        self._notifier: Optional[TelegramNotifier] = None

        # Executor reference (set by executor)
        self.executor = None

        # Formatter
        self.fmt = TelegramFormatter

        logger.info(f"MLTelegramNotifier initialized (enabled={self.enabled})")

    async def initialize(self) -> bool:
        """Initialize Telegram bot"""
        if not self.enabled:
            logger.warning("Telegram notifications disabled")
            return False

        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot_token or chat_id not configured")
            return False

        self._notifier = TelegramNotifier(
            bot_token=self.bot_token,
            chat_id=self.chat_id,
            enabled=self.enabled
        )

        success = await self._notifier.initialize()

        if success:
            # Set up command handlers
            self._setup_command_handlers()
            logger.info("Telegram notifier initialized")

        return success

    def _setup_command_handlers(self):
        """Set up command handler callbacks"""
        if not self._notifier:
            return

        self._notifier.on_status = self._handle_status
        self._notifier.on_balance = self._handle_balance
        self._notifier.on_positions = self._handle_positions
        self._notifier.on_regime = self._handle_regime
        self._notifier.on_pause = self._handle_pause
        self._notifier.on_resume = self._handle_resume

    async def start_polling(self):
        """Start polling for commands"""
        if self._notifier:
            await self._notifier.start_polling()

    async def stop_polling(self):
        """Stop polling"""
        if self._notifier:
            await self._notifier.stop_polling()

    async def send(self, message: str, force: bool = False):
        """Send message to Telegram"""
        if self._notifier:
            await self._notifier.send(message, force=force)

    # =========================================================================
    # ML BOT SPECIFIC ALERTS
    # =========================================================================

    async def send_startup(self, account_name: str, balance: float, server: str):
        """Send bot startup notification"""
        msg = f"{self.fmt.BRAIN} <b>ML Trading Bot Started</b>\n\n"
        msg += f"{self.fmt.BRANCH} Account: {account_name}\n"
        msg += f"{self.fmt.BRANCH} Balance: ${balance:,.2f}\n"
        msg += f"{self.fmt.BRANCH} Server: {server}\n"
        msg += f"{self.fmt.LAST} Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
        msg += f"\n{self.fmt.GEAR} <i>Monitoring market for signals...</i>"

        await self.send(msg, force=True)

    async def send_trade_signal(
        self,
        direction: str,
        confidence: float,
        regime: str,
        regime_confidence: float,
        approved: bool,
        reason: str = ""
    ):
        """Send trade signal notification (before execution)"""
        dir_emoji = self.fmt.UP if direction == "BUY" else self.fmt.DOWN

        msg = f"{self.fmt.BELL} <b>ML Signal Detected</b>\n\n"
        msg += f"{dir_emoji} Direction: <b>{direction}</b>\n"
        msg += f"{self.fmt.BRANCH} Confidence: {confidence:.1%}\n"
        msg += f"{self.fmt.BRANCH} Regime: {regime} ({regime_confidence:.1%})\n"

        if approved:
            msg += f"{self.fmt.LAST} Status: {self.fmt.GREEN} APPROVED\n"
        else:
            msg += f"{self.fmt.BRANCH} Status: {self.fmt.RED} FILTERED\n"
            msg += f"{self.fmt.LAST} Reason: {reason}\n"

        await self.send(msg)

    async def send_trade_executed(
        self,
        direction: str,
        symbol: str,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        ticket: int,
        regime: str,
        confidence: float
    ):
        """Send trade execution notification"""
        pip_mult = 10000
        sl_pips = abs(entry - sl) * pip_mult
        tp_pips = abs(tp - entry) * pip_mult
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        dir_emoji = self.fmt.GREEN if direction == "BUY" else self.fmt.RED

        msg = f"{self.fmt.ROCKET} <b>ML TRADE EXECUTED</b>\n\n"
        msg += f"{dir_emoji} <b>{symbol}</b> - <b>{direction}</b>\n"
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f"SL:    {sl:.5f} ({sl_pips:.0f}p)\n"
        msg += f"TP:    {tp:.5f} ({tp_pips:.0f}p)\n"
        msg += f"Lot:   {lot:.2f}\n"
        msg += "</pre>"

        msg += f"\n{self.fmt.BRAIN} <b>ML Analysis</b>\n"
        msg += f"{self.fmt.BRANCH} Regime: {regime}\n"
        msg += f"{self.fmt.BRANCH} Confidence: {confidence:.1%}\n"
        msg += f"{self.fmt.LAST} R:R = 1:{rr:.1f}\n"

        msg += f"\n<code>#{ticket}</code>"

        await self.send(msg, force=True)

    async def send_position_closed(
        self,
        direction: str,
        symbol: str,
        entry: float,
        exit_price: float,
        pnl: float,
        close_reason: str,
        daily_pnl: float,
        win_rate: float
    ):
        """Send position close notification"""
        pips = (exit_price - entry) / 0.0001 if direction == "BUY" else (entry - exit_price) / 0.0001
        emoji = self.fmt.profit_emoji(pnl)

        msg = f"{emoji} <b>POSITION CLOSED - {close_reason}</b>\n\n"
        msg += f"<b>{symbol}</b> - {direction}\n"
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f"Exit:  {exit_price:.5f}\n"
        msg += f"Pips:  {pips:+.1f}\n"
        msg += f"P/L:   ${pnl:+.2f}\n"
        msg += "</pre>"

        msg += f"\n{self.fmt.CHART} <b>Session Stats</b>\n"
        msg += f"{self.fmt.BRANCH} Daily P/L: ${daily_pnl:+.2f}\n"
        msg += f"{self.fmt.LAST} Win Rate: {win_rate:.1f}%"

        await self.send(msg, force=True)

    async def send_circuit_breaker(
        self,
        loss_pct: float,
        daily_pnl: float,
        starting_balance: float,
        current_balance: float
    ):
        """Send circuit breaker alert"""
        msg = f"{self.fmt.WARNING} <b>CIRCUIT BREAKER TRIGGERED</b>\n\n"
        msg += f"{self.fmt.BRANCH} Daily Loss: {loss_pct:.2%}\n"
        msg += f"{self.fmt.BRANCH} P/L Today: ${daily_pnl:+.2f}\n"
        msg += f"{self.fmt.BRANCH} Start Balance: ${starting_balance:,.2f}\n"
        msg += f"{self.fmt.BRANCH} Current: ${current_balance:,.2f}\n"
        msg += f"{self.fmt.LAST} Status: Trading PAUSED\n"
        msg += f"\n{self.fmt.RED} <i>Trading will resume tomorrow</i>"

        await self.send(msg, force=True)

    async def send_daily_summary(
        self,
        date: str,
        trades: int,
        wins: int,
        losses: int,
        gross_profit: float,
        gross_loss: float,
        net_pnl: float,
        balance: float
    ):
        """Send daily summary"""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        emoji = self.fmt.profit_emoji(net_pnl)

        msg = f"{emoji} <b>Daily Summary - {date}</b>\n\n"

        msg += f"{self.fmt.TARGET} <b>Trades</b>\n"
        msg += f"{self.fmt.BRANCH} Total: {trades}\n"
        msg += f"{self.fmt.BRANCH} Winners: {wins} ({win_rate:.0f}%)\n"
        msg += f"{self.fmt.LAST} Losers: {losses}\n"

        msg += f"\n{self.fmt.MONEY} <b>P/L</b>\n"
        msg += f"{self.fmt.BRANCH} Gross Profit: +${gross_profit:.2f}\n"
        msg += f"{self.fmt.BRANCH} Gross Loss: -${gross_loss:.2f}\n"
        msg += f"{self.fmt.BRANCH} Net P/L: ${net_pnl:+.2f}\n"
        msg += f"{self.fmt.LAST} Balance: ${balance:,.2f}"

        await self.send(msg, force=True)

    async def send_ml_status(
        self,
        regime: str,
        regime_confidence: float,
        signal: str,
        signal_confidence: float,
        state: str,
        has_position: bool
    ):
        """Send ML model status update"""
        msg = f"{self.fmt.BRAIN} <b>ML Bot Status</b>\n\n"

        # State
        state_emoji = self.fmt.GREEN if state == "monitoring" else self.fmt.YELLOW
        msg += f"{self.fmt.BRANCH} State: {state_emoji} {state.upper()}\n"

        # Regime
        regime_emoji = self.fmt.regime_emoji(regime)
        msg += f"{self.fmt.BRANCH} Regime: {regime_emoji} {regime} ({regime_confidence:.1%})\n"

        # Signal
        if signal == "BUY":
            sig_emoji = self.fmt.UP
        elif signal == "SELL":
            sig_emoji = self.fmt.DOWN
        else:
            sig_emoji = "➡️"
        msg += f"{self.fmt.BRANCH} Signal: {sig_emoji} {signal} ({signal_confidence:.1%})\n"

        # Position
        pos_status = f"{self.fmt.GREEN} Yes" if has_position else f"{self.fmt.RED} No"
        msg += f"{self.fmt.LAST} Position: {pos_status}"

        await self.send(msg)

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    async def _handle_status(self) -> str:
        """Handle /status command"""
        if not self.executor:
            return f"{self.fmt.WARNING} Executor not connected"

        try:
            status = self.executor.get_status()
            ml = status.get('ml', {})
            stats = status.get('stats', {})

            msg = f"{self.fmt.BRAIN} <b>ML Trading Bot Status</b>\n\n"

            # State
            state = status.get('state', 'unknown')
            state_emoji = self.fmt.GREEN if state == "monitoring" else self.fmt.YELLOW
            msg += f"{self.fmt.BRANCH} State: {state_emoji} {state.upper()}\n"

            # ML Predictions
            regime = ml.get('regime', 'N/A')
            regime_conf = ml.get('regime_confidence', 0)
            signal = ml.get('signal', 'N/A')
            signal_conf = ml.get('signal_confidence', 0)

            msg += f"{self.fmt.BRANCH} Regime: {regime} ({regime_conf:.1%})\n"
            msg += f"{self.fmt.BRANCH} Signal: {signal} ({signal_conf:.1%})\n"

            # Position
            has_pos = status.get('has_position', False)
            if has_pos:
                pos = status.get('position', {})
                msg += f"{self.fmt.BRANCH} Position: {self.fmt.GREEN} {pos.get('direction')} @ {pos.get('entry', 0):.5f}\n"
            else:
                msg += f"{self.fmt.BRANCH} Position: {self.fmt.RED} None\n"

            # Stats
            msg += f"\n{self.fmt.CHART} <b>Stats</b>\n"
            msg += f"{self.fmt.BRANCH} Trades: {stats.get('trades', 0)}\n"
            msg += f"{self.fmt.BRANCH} Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            msg += f"{self.fmt.BRANCH} Net P/L: ${stats.get('net_pnl', 0):+.2f}\n"
            msg += f"{self.fmt.LAST} Daily P/L: ${stats.get('daily_pnl', 0):+.2f}"

            return msg

        except Exception as e:
            logger.error(f"Status error: {e}")
            return f"{self.fmt.CROSS} Error: {e}"

    async def _handle_balance(self) -> str:
        """Handle /balance command"""
        if not self.executor or not self.executor.mt5:
            return f"{self.fmt.WARNING} MT5 not connected"

        try:
            account = self.executor.mt5.get_account_info_sync()
            if not account:
                return f"{self.fmt.CROSS} Failed to get account info"

            return self.fmt.balance_report(
                login=account['login'],
                server=account['server'],
                balance=account['balance'],
                equity=account['equity'],
                profit=account['profit'],
                free_margin=account['free_margin'],
                margin_level=account.get('margin_level', 0)
            )

        except Exception as e:
            logger.error(f"Balance error: {e}")
            return f"{self.fmt.CROSS} Error: {e}"

    async def _handle_positions(self) -> str:
        """Handle /positions command"""
        if not self.executor or not self.executor.mt5:
            return f"{self.fmt.WARNING} MT5 not connected"

        try:
            positions = self.executor.mt5.get_positions_sync()
            return self.fmt.positions_list(positions or [])

        except Exception as e:
            logger.error(f"Positions error: {e}")
            return f"{self.fmt.CROSS} Error: {e}"

    async def _handle_regime(self) -> str:
        """Handle /regime command"""
        if not self.executor:
            return f"{self.fmt.WARNING} Executor not connected"

        try:
            status = self.executor.get_status()
            ml = status.get('ml', {})

            regime = ml.get('regime', 'N/A')
            regime_conf = ml.get('regime_confidence', 0)
            signal = ml.get('signal', 'N/A')
            signal_conf = ml.get('signal_confidence', 0)

            msg = f"{self.fmt.BRAIN} <b>ML Regime Analysis</b>\n\n"

            # Regime
            regime_emoji = self.fmt.regime_emoji(regime)
            bar = self.fmt.progress_bar(regime_conf * 100)
            msg += f"<b>Regime:</b> {regime_emoji} {regime.upper()}\n"
            msg += f"Confidence: {bar} {regime_conf:.1%}\n\n"

            # Signal
            if signal == "BUY":
                sig_emoji = self.fmt.UP
            elif signal == "SELL":
                sig_emoji = self.fmt.DOWN
            else:
                sig_emoji = "➡️"

            sig_bar = self.fmt.progress_bar(signal_conf * 100)
            msg += f"<b>Signal:</b> {sig_emoji} {signal.upper()}\n"
            msg += f"Confidence: {sig_bar} {signal_conf:.1%}"

            return msg

        except Exception as e:
            logger.error(f"Regime error: {e}")
            return f"{self.fmt.CROSS} Error: {e}"

    async def _handle_pause(self):
        """Handle /pause command"""
        if self.executor:
            self.executor.pause()
            logger.info("Trading paused via Telegram")

    async def _handle_resume(self):
        """Handle /resume command"""
        if self.executor:
            self.executor.resume()
            logger.info("Trading resumed via Telegram")


# Convenience function
def create_ml_notifier() -> MLTelegramNotifier:
    """Create ML Telegram notifier from environment"""
    return MLTelegramNotifier()
