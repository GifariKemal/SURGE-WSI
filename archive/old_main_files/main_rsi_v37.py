"""
RSI Mean Reversion v3.7 - Live Demo Runner
===========================================

Run this script to start live demo trading with the RSI v3.7 strategy.

Usage:
    python main_rsi_v37.py

Requirements:
    - MT5 terminal running
    - PostgreSQL database (optional, for logging)
    - Telegram bot configured (optional)
"""
import asyncio
import signal
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    "logs/rsi_v37_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)

from src.data.mt5_connector import MT5Connector
from src.trading.executor_rsi_v37 import RSIMeanReversionV37
from src.utils.telegram import TelegramNotifier, TelegramFormatter


async def async_wrapper(func, *args, **kwargs):
    """Wrap sync function to async"""
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))


class RSIv37Runner:
    """Main runner for RSI v3.7 strategy"""

    # Finex Demo Account Configuration
    MT5_LOGIN = 61045904
    MT5_PASSWORD = "iy#K5L7sF"
    MT5_SERVER = "FinexBisnisSolusi-Demo"
    MT5_TERMINAL_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

    # Telegram Configuration (from .env)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

    def __init__(self, symbol: str = "GBPUSD", demo: bool = True):
        self.symbol = symbol
        self.demo = demo
        self.running = False

        # Initialize components with Finex credentials
        self.mt5 = MT5Connector(
            terminal_path=self.MT5_TERMINAL_PATH,
            login=self.MT5_LOGIN,
            password=self.MT5_PASSWORD,
            server=self.MT5_SERVER
        )
        self.executor = RSIMeanReversionV37(
            symbol=symbol,
            magic_number=20250131
        )

        # Telegram notifier
        self.telegram: TelegramNotifier = None
        self.fmt = TelegramFormatter

        # Track last bar time to detect new bars
        self._last_bar_time: datetime = None

    async def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("RSI MEAN REVERSION v3.7 - LIVE DEMO")
        logger.info("=" * 60)

        # Initialize Telegram
        if self.TELEGRAM_ENABLED and self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID:
            logger.info("Initializing Telegram...")
            self.telegram = TelegramNotifier(
                bot_token=self.TELEGRAM_BOT_TOKEN,
                chat_id=self.TELEGRAM_CHAT_ID,
                enabled=True
            )
            if await self.telegram.initialize():
                logger.info("Telegram initialized successfully")
                # Set up command handlers
                self._setup_telegram_commands()
                # Start polling for commands
                await self.telegram.start_polling()
            else:
                logger.warning("Telegram initialization failed, continuing without it")
                self.telegram = None
        else:
            logger.info("Telegram disabled or not configured")

        # Connect to MT5 with Finex credentials
        logger.info("Connecting to MT5 (Finex Demo)...")
        logger.info(f"Login: {self.MT5_LOGIN} | Server: {self.MT5_SERVER}")
        if not self.mt5.connect(force_login=True):
            logger.error("Failed to connect to MT5")
            return False

        # Get account info
        account = await self.mt5.get_account_info()
        if account:
            logger.info(f"Account: {account.get('login')} | Balance: ${account.get('balance', 0):,.2f}")
            logger.info(f"Server: {account.get('server')} | {'DEMO' if self.demo else 'LIVE'}")
            self._account_balance = account.get('balance', 0)
        else:
            logger.error("Failed to get account info")
            return False

        # Set callbacks (wrap sync methods to async)
        self.executor.set_callbacks(
            get_account_info=self.mt5.get_account_info,
            get_tick=self.mt5.get_tick,
            get_ohlcv=self._async_get_ohlcv,
            get_symbol_info=self._async_get_symbol_info,
            place_market_order=self.mt5.place_market_order,
            close_position=self.mt5.close_position,
            get_positions=self.mt5.get_positions,
            get_deal_history=self._async_get_deal_history,
            send_telegram=self._send_telegram
        )

        # Fetch symbol info
        await self.executor.fetch_symbol_info()

        # Warmup with historical data
        logger.info("Warming up with historical data...")
        if not await self.executor.warmup():
            logger.error("Warmup failed")
            return False

        # Recover existing positions
        await self.executor.recover_positions()

        logger.info("Initialization complete!")
        logger.info("-" * 60)
        self._print_strategy_info()
        logger.info("-" * 60)

        # Send startup notification to Telegram
        await self._send_startup_notification(account)

        return True

    def _setup_telegram_commands(self):
        """Set up Telegram command handlers"""
        if not self.telegram:
            return

        self.telegram.on_status = self._cmd_status
        self.telegram.on_balance = self._cmd_balance
        self.telegram.on_positions = self._cmd_positions
        self.telegram.on_pause = self._cmd_pause
        self.telegram.on_resume = self._cmd_resume
        self.telegram.on_help = self._cmd_help

    async def _cmd_status(self) -> str:
        """Handle /status command"""
        status = self.executor.get_status()
        stats = status.get('stats', {})
        indicators = status.get('indicators', {})

        state_emoji = self.fmt.GREEN if status['state'] == 'monitoring' else self.fmt.YELLOW

        msg = f"{self.fmt.CHART} <b>RSI v3.7 Status</b>\n\n"
        msg += f"{self.fmt.BRANCH} State: {state_emoji} {status['state'].upper()}\n"
        msg += f"{self.fmt.BRANCH} RSI: {indicators.get('rsi', 'N/A')}\n"
        msg += f"{self.fmt.BRANCH} ATR%: {indicators.get('atr_pct', 'N/A')}\n"

        if status['has_position']:
            pos = status['position']
            dir_emoji = self.fmt.UP if pos['direction'] == 'BUY' else self.fmt.DOWN
            msg += f"\n{dir_emoji} <b>Position</b>\n"
            msg += f"{self.fmt.BRANCH} Entry: {pos['entry']:.5f}\n"
            msg += f"{self.fmt.BRANCH} SL: {pos['sl']:.5f}\n"
            msg += f"{self.fmt.LAST} TP: {pos['tp']:.5f}\n"
        else:
            msg += f"{self.fmt.LAST} Position: None\n"

        msg += f"\n{self.fmt.MONEY} <b>Stats</b>\n"
        msg += f"{self.fmt.BRANCH} Trades: {stats.get('trades', 0)}\n"
        msg += f"{self.fmt.BRANCH} Win Rate: {stats.get('win_rate', '0%')}\n"
        msg += f"{self.fmt.BRANCH} Net P/L: {stats.get('net_pnl', '$0')}\n"
        msg += f"{self.fmt.LAST} Daily P/L: {stats.get('daily_pnl', '$0')}"

        return msg

    async def _cmd_balance(self) -> str:
        """Handle /balance command"""
        account = await self.mt5.get_account_info()
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

    async def _cmd_positions(self) -> str:
        """Handle /positions command"""
        positions = await self.mt5.get_positions()
        return self.fmt.positions_list(positions or [])

    async def _cmd_pause(self):
        """Handle /pause command"""
        self.executor.pause()
        logger.info("Trading paused via Telegram")

    async def _cmd_resume(self):
        """Handle /resume command"""
        self.executor.resume()
        logger.info("Trading resumed via Telegram")

    async def _cmd_help(self) -> str:
        """Handle /help command"""
        msg = f"{self.fmt.EAGLE} <b>RSI v3.7 Commands</b>\n\n"
        msg += f"{self.fmt.CHART} <b>Information</b>\n"
        msg += "<code>/status</code> - Bot status & indicators\n"
        msg += "<code>/balance</code> - Account balance\n"
        msg += "<code>/positions</code> - Open positions\n\n"
        msg += f"{self.fmt.GEAR} <b>Control</b>\n"
        msg += "<code>/pause</code> - Pause trading\n"
        msg += "<code>/resume</code> - Resume trading\n\n"
        msg += f"{self.fmt.MEMO} <b>Strategy</b>\n"
        msg += "RSI(10) 42/58 Mean Reversion\n"
        msg += "SL: 1.5x ATR | TP: 2.4-3.6x ATR\n"
        msg += "Hours: 07-22 UTC | Risk: 1%"
        return msg

    async def _send_startup_notification(self, account: dict):
        """Send startup notification to Telegram"""
        if not self.telegram:
            return

        msg = f"{self.fmt.ROCKET} <b>RSI v3.7 Bot Started</b>\n\n"
        msg += f"{self.fmt.BRANCH} Account: {account.get('login')}\n"
        msg += f"{self.fmt.BRANCH} Balance: ${account.get('balance', 0):,.2f}\n"
        msg += f"{self.fmt.BRANCH} Server: {account.get('server')}\n"
        msg += f"{self.fmt.LAST} Symbol: {self.symbol}\n\n"

        # Current indicators
        status = self.executor.get_status()
        indicators = status.get('indicators', {})
        msg += f"{self.fmt.CHART} <b>Current</b>\n"
        msg += f"{self.fmt.BRANCH} RSI: {indicators.get('rsi', 'N/A')}\n"
        msg += f"{self.fmt.LAST} ATR%: {indicators.get('atr_pct', 'N/A')}\n\n"

        msg += f"{self.fmt.GEAR} <i>Monitoring for signals...</i>"

        await self.telegram.send(msg, force=True)

    async def _async_get_ohlcv(self, symbol: str, timeframe: str, bars: int):
        """Async wrapper for get_ohlcv"""
        return await async_wrapper(self.mt5.get_ohlcv, symbol, timeframe, bars)

    async def _async_get_symbol_info(self, symbol: str):
        """Async wrapper for get_symbol_info"""
        return await async_wrapper(self.mt5.get_symbol_info, symbol)

    async def _async_get_deal_history(self, ticket: int):
        """Async wrapper for get_deal_history"""
        return await async_wrapper(self.mt5.get_deal_history, ticket)

    def _print_strategy_info(self):
        """Print strategy parameters"""
        logger.info("STRATEGY PARAMETERS:")
        logger.info(f"  RSI Period: {self.executor.RSI_PERIOD}")
        logger.info(f"  RSI Oversold: {self.executor.RSI_OVERSOLD}")
        logger.info(f"  RSI Overbought: {self.executor.RSI_OVERBOUGHT}")
        logger.info(f"  SL Multiplier: {self.executor.SL_MULT}x ATR")
        logger.info(f"  TP Multiplier: {self.executor.TP_LOW}/{self.executor.TP_MED}/{self.executor.TP_HIGH}x ATR")
        logger.info(f"  Time TP Bonus: +{self.executor.TIME_TP_BONUS}x (12-16 UTC)")
        logger.info(f"  ATR Filter: {self.executor.MIN_ATR_PCT}-{self.executor.MAX_ATR_PCT}%")
        logger.info(f"  Trading Hours: {self.executor.TRADING_START_HOUR}:00-{self.executor.TRADING_END_HOUR}:00 UTC")
        logger.info(f"  Skip Hours: {self.executor.SKIP_HOURS}")
        logger.info(f"  Max Holding: {self.executor.MAX_HOLDING_HOURS} hours")
        logger.info(f"  Risk per Trade: {self.executor.RISK_PER_TRADE * 100}%")
        logger.info(f"  Circuit Breaker: {self.executor.MAX_DAILY_LOSS_PCT * 100}% daily loss")

    async def _send_telegram(self, message: str):
        """Send Telegram notification"""
        if self.telegram:
            await self.telegram.send(message, force=True)
        logger.debug(f"Telegram: {message[:50]}...")

    async def _check_new_bar(self) -> bool:
        """Check if a new H1 bar has formed"""
        now = datetime.now(timezone.utc)
        current_bar_time = now.replace(minute=0, second=0, microsecond=0)

        if self._last_bar_time is None:
            self._last_bar_time = current_bar_time
            return False

        if current_bar_time > self._last_bar_time:
            self._last_bar_time = current_bar_time
            return True

        return False

    async def run(self):
        """Main trading loop"""
        self.running = True
        check_interval = 10  # Check every 10 seconds

        logger.info("Starting main trading loop...")
        logger.info("Press Ctrl+C to stop")

        while self.running:
            try:
                # Check for new H1 bar
                if await self._check_new_bar():
                    logger.info(f"New H1 bar: {self._last_bar_time}")

                    # Get current balance
                    account = await self.mt5.get_account_info()
                    balance = account.get('balance', 10000) if account else 10000

                    # Process new bar
                    result = await self.executor.on_new_bar(
                        bar_time=self._last_bar_time,
                        balance=balance
                    )

                    if result and result.success:
                        logger.info(f"Trade executed: {result.direction} @ {result.entry_price:.5f}")
                        # Send detailed trade notification
                        await self._send_trade_notification(result)

                # Log status periodically (every 5 minutes)
                now = datetime.now(timezone.utc)
                if now.minute % 5 == 0 and now.second < check_interval:
                    status = self.executor.get_status()
                    logger.info(
                        f"Status: {status['state']} | "
                        f"RSI: {status['indicators']['rsi']} | "
                        f"Pos: {status['has_position']} | "
                        f"Daily: {status['stats']['daily_pnl']}"
                    )

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(check_interval)

    async def _send_trade_notification(self, result):
        """Send detailed trade notification to Telegram"""
        if not self.telegram:
            return

        pip_value = 0.0001
        sl_pips = abs(result.entry_price - result.stop_loss) / pip_value
        tp_pips = abs(result.take_profit - result.entry_price) / pip_value
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        dir_emoji = self.fmt.GREEN if result.direction == "BUY" else self.fmt.RED

        msg = f"{self.fmt.TARGET} <b>TRADE EXECUTED</b>\n\n"
        msg += f"{dir_emoji} <b>{self.symbol}</b> - <b>{result.direction}</b>\n"
        msg += "<pre>"
        msg += f"Entry: {result.entry_price:.5f}\n"
        msg += f"SL:    {result.stop_loss:.5f} ({sl_pips:.0f}p)\n"
        msg += f"TP:    {result.take_profit:.5f} ({tp_pips:.0f}p)\n"
        msg += f"Lot:   {result.volume:.2f}\n"
        msg += "</pre>"
        msg += f"\n{self.fmt.BRAIN} <b>Signal</b>\n"
        msg += f"{self.fmt.BRANCH} RSI: {result.rsi_value:.1f}\n"
        msg += f"{self.fmt.BRANCH} ATR%: {result.atr_pct:.0f}\n"
        msg += f"{self.fmt.LAST} R:R = 1:{rr:.1f}"

        await self.telegram.send(msg, force=True)

    async def shutdown(self):
        """Shutdown gracefully"""
        logger.info("Shutting down...")
        self.running = False
        self.executor.stop()

        # Send shutdown notification
        if self.telegram:
            msg = f"{self.fmt.WARNING} <b>RSI v3.7 Bot Stopped</b>\n\n"
            stats = self.executor.get_status().get('stats', {})
            msg += f"{self.fmt.BRANCH} Trades: {stats.get('trades', 0)}\n"
            msg += f"{self.fmt.BRANCH} Win Rate: {stats.get('win_rate', '0%')}\n"
            msg += f"{self.fmt.LAST} Net P/L: {stats.get('net_pnl', '$0')}"
            await self.telegram.send(msg, force=True)
            await self.telegram.stop_polling()

        self.mt5.disconnect()
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    runner = RSIv37Runner(symbol="GBPUSD", demo=True)

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Interrupt received, shutting down...")
        runner.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if await runner.initialize():
            await runner.run()
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
