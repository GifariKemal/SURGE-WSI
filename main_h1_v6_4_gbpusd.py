"""
SURGE-WSI H1 v6.4 GBPUSD Main Entry Point
==========================================

Dual-Layer Quality Filter for ZERO losing months

Usage:
    python main_h1_v6_4_gbpusd.py [--demo] [--live] [--interval 300]

Commands (Telegram):
    /status - Get current status
    /balance - Get account balance
    /regime - Get current market regime
    /condition - Get current market condition
    /stop - Stop trading

Backtest Results (Jan 2024 - Jan 2025):
    - 147 trades, 0.40/day (~2-3 per week)
    - 42.2% WR, PF 3.98
    - +$23,394 profit (+46.8% return on $50K)
    - ZERO losing months (0/13)

Author: SURIOTA Team
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.data.mt5_connector import MT5Connector
from src.trading.executor_h1_v6_4_gbpusd import H1V64GBPUSDExecutor, SYMBOL
from src.utils.telegram import TelegramNotifier, TelegramFormatter


class BrokerAdapter:
    """Adapter to bridge MT5Connector with executor's expected interface"""

    def __init__(self, mt5_connector: MT5Connector):
        self.mt5 = mt5_connector

    async def connect(self) -> bool:
        return self.mt5.connect()

    async def get_balance(self) -> float:
        info = await self.mt5.get_account_info()
        if info:
            return info.get('balance', 0.0)
        return 0.0

    async def get_positions(self, symbol: str = None):
        return await self.mt5.get_positions(symbol)

    async def place_order(
        self,
        symbol: str,
        order_type: str,
        lot_size: float,
        stop_loss: float = 0,
        take_profit: float = 0
    ) -> dict:
        result = await self.mt5.place_market_order(
            symbol=symbol,
            order_type=order_type,
            volume=lot_size,
            sl=stop_loss,
            tp=take_profit,
            comment="SURGE-v6.4"
        )
        if result:
            return {'success': True, 'ticket': result.get('ticket'), **result}
        return {'success': False, 'error': self.mt5.get_last_error()}

    async def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        return await self.mt5.modify_position(ticket, sl, tp)

    async def close_position(self, ticket: int) -> bool:
        return await self.mt5.close_position(ticket)

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/h1_v6_4_gbpusd_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)


class TradingBot:
    """Main trading bot for H1 v6.4 GBPUSD"""

    def __init__(self, mode: str = 'demo', interval: int = 300):
        self.mode = mode
        self.interval = interval
        self.running = False
        self.executor = None
        self.telegram = None
        self.db = None
        self.mt5 = None  # MT5 connector reference
        self.loop_count = 0
        self._last_hourly_status = None

    async def initialize(self):
        """Initialize all components"""
        logger.info(f"Initializing H1 v6.4 GBPUSD Trading Bot")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Interval: {self.interval}s")

        # Initialize database
        self.db = DBHandler(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.user,
            password=config.database.password
        )

        if not await self.db.connect():
            raise RuntimeError("Failed to connect to database")

        # Initialize broker (MT5 Connector with Adapter)
        # Get MT5 credentials from config if available
        mt5_login = getattr(config, 'mt5_login', None)
        mt5_password = getattr(config, 'mt5_password', None)
        mt5_server = getattr(config, 'mt5_server', None)

        mt5_connector = MT5Connector(
            login=mt5_login,
            password=mt5_password,
            server=mt5_server
        )

        if not mt5_connector.connect():
            raise RuntimeError("Failed to connect to MT5. Make sure MT5 terminal is running and logged in.")

        # Store MT5 reference for status logging
        self.mt5 = mt5_connector

        # Check AutoTrading status
        if mt5_connector.is_autotrading_enabled():
            logger.info("AutoTrading: ENABLED")
        else:
            logger.warning("AutoTrading: DISABLED - Enable in MT5 terminal!")

        # Wrap with adapter for executor compatibility
        broker = BrokerAdapter(mt5_connector)

        # Initialize Telegram
        try:
            self.telegram = TelegramNotifier(
                bot_token=config.telegram.bot_token,
                chat_id=config.telegram.chat_id
            )
            if await self.telegram.initialize():
                await self.telegram.start_polling()
                logger.info("Telegram bot initialized")
            else:
                logger.warning("Telegram init returned False")
                self.telegram = None
        except Exception as e:
            logger.warning(f"Telegram init failed: {e}")
            self.telegram = None

        # Initialize executor (MT5 primary, DB fallback)
        self.executor = H1V64GBPUSDExecutor(
            broker_client=broker,
            db_handler=self.db,  # Fallback + logging
            telegram_bot=self.telegram,
            mt5_connector=self.mt5  # Primary data source
        )

        # Register Telegram commands
        if self.telegram:
            self._register_commands()

        logger.info("Initialization complete")
        return True

    def _register_commands(self):
        """Register Telegram command handlers"""

        async def status_handler():
            status = self.executor.get_status()
            now = datetime.now(timezone.utc)
            adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)
            tradeable = {0: "75%+", 5: "70-75%", 10: "60-70%", 15: "<60%"}.get(adj, "Unknown")

            msg = TelegramFormatter.tree_header("H1 v6.4 GBPUSD Status", TelegramFormatter.CHART)
            msg += TelegramFormatter.tree_section("System", TelegramFormatter.GEAR)
            msg += TelegramFormatter.tree_item("Symbol", status['symbol'])
            msg += TelegramFormatter.tree_item("Strategy", "Dual-Layer Quality")
            msg += TelegramFormatter.tree_item("Has Position", "Yes" if status['has_position'] else "No")
            msg += TelegramFormatter.tree_item("Last Signal", status['last_signal'] or 'None', last=True)

            msg += TelegramFormatter.tree_section("Market Condition", TelegramFormatter.BRAIN)
            msg += TelegramFormatter.tree_item("Month", now.strftime('%Y-%m'))
            msg += TelegramFormatter.tree_item("Tradeable", tradeable)
            msg += TelegramFormatter.tree_item("Quality Adj", f"+{adj}")
            msg += TelegramFormatter.tree_item("Min Quality", f"{60 + adj}-{80 + adj}", last=True)
            return msg

        async def balance_handler():
            balance = await self.executor.broker.get_balance()
            return f"{TelegramFormatter.MONEY} <b>Account Balance:</b> <code>${balance:,.2f}</code>"

        async def pause_handler():
            self.running = False

        async def resume_handler():
            self.running = True

        # Register callbacks
        self.telegram.on_status = status_handler
        self.telegram.on_balance = balance_handler
        self.telegram.on_pause = pause_handler
        self.telegram.on_resume = resume_handler

    def _get_session_name(self, hour: int) -> str:
        """Get current trading session name (UTC)"""
        # Asia/Tokyo: 00:00-08:00 UTC (09:00-17:00 JST)
        # London: 07:00-16:00 UTC (08:00-17:00 GMT)
        # New York: 13:00-22:00 UTC (08:00-17:00 EST)
        if 0 <= hour < 7:
            return "Asia"
        elif 7 <= hour < 8:
            return "Asia/London"
        elif 8 <= hour < 12:
            return "London"
        elif 12 <= hour < 13:
            return "London/NY"
        elif 13 <= hour < 16:
            return "NY"
        elif 16 <= hour < 21:
            return "NY-Late"
        else:
            return "Off-hours"

    def _is_kill_zone(self, hour: int) -> bool:
        """Check if current hour is in kill zone (for GBPUSD)"""
        # London KZ: 07:00-11:00 UTC
        # NY KZ: 13:00-17:00 UTC
        return (7 <= hour <= 11) or (13 <= hour <= 17)

    def _is_market_open(self, now: datetime) -> bool:
        """Check if forex market is open"""
        # Forex closes Friday 22:00 UTC, opens Sunday 22:00 UTC
        weekday = now.weekday()
        hour = now.hour

        if weekday == 5:  # Saturday
            return False
        elif weekday == 6:  # Sunday
            return hour >= 22  # Opens at 22:00 UTC
        elif weekday == 4:  # Friday
            return hour < 22  # Closes at 22:00 UTC
        else:
            return True

    def _get_market_status(self, now: datetime) -> str:
        """Get market open/close status"""
        if self._is_market_open(now):
            return "OPEN"
        else:
            return "CLOSED"

    async def _send_hourly_status(self, now: datetime, price: float, balance: float):
        """Send hourly status to Telegram"""
        if not self.telegram:
            return

        # Only send once per hour
        if self._last_hourly_status is not None:
            if (now.hour == self._last_hourly_status.hour and
                now.date() == self._last_hourly_status.date()):
                return

        self._last_hourly_status = now

        try:
            adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)
            session = self._get_session_name(now.hour)
            in_kz = self._is_kill_zone(now.hour)
            market_status = self._get_market_status(now)
            status = self.executor.get_status()

            # Market status emoji
            mkt_emoji = TelegramFormatter.GREEN if market_status == "OPEN" else TelegramFormatter.RED

            msg = TelegramFormatter.tree_header(
                f"H1 v6.4 - {now.strftime('%H:%M')} UTC",
                TelegramFormatter.CLOCK
            )
            msg += TelegramFormatter.tree_section("Market", TelegramFormatter.CHART)
            msg += TelegramFormatter.tree_item("Status", f"{mkt_emoji} {market_status}")
            msg += TelegramFormatter.tree_item("Price", f"{price:.5f}")
            msg += TelegramFormatter.tree_item("Session", session)
            msg += TelegramFormatter.tree_item("Kill Zone", "Yes" if in_kz else "No")
            msg += TelegramFormatter.tree_item("Quality Adj", f"+{adj}", last=True)

            msg += TelegramFormatter.tree_section("Account", TelegramFormatter.MONEY)
            msg += TelegramFormatter.tree_item("Balance", f"${balance:,.2f}")
            msg += TelegramFormatter.tree_item("Position", "Yes" if status['has_position'] else "No", last=True)

            await self.telegram.send(msg)
        except Exception as e:
            logger.warning(f"Failed to send hourly status: {e}")

    async def run(self):
        """Main trading loop"""
        self.running = True
        self.loop_count = 0

        # Send startup message
        if self.telegram:
            autotrading = "ENABLED" if self.mt5.is_autotrading_enabled() else "DISABLED"
            msg = (
                f"{TelegramFormatter.ROCKET} <b>H1 v6.4 GBPUSD Bot Started</b>\n\n"
                f"{TelegramFormatter.BRANCH} Mode: <code>{self.mode.upper()}</code>\n"
                f"{TelegramFormatter.BRANCH} Interval: <code>{self.interval}s</code>\n"
                f"{TelegramFormatter.BRANCH} AutoTrading: <code>{autotrading}</code>\n"
                f"{TelegramFormatter.LAST} Strategy: <code>Dual-Layer Quality Filter</code>\n\n"
                f"Commands: /status /balance /pause /resume"
            )
            await self.telegram.send(msg)

        logger.info("Starting main trading loop")

        while self.running:
            try:
                self.loop_count += 1
                cycle_start = datetime.now(timezone.utc)

                # Get current market info for logging
                tick = self.mt5.get_tick_sync(SYMBOL)
                account = self.mt5.get_account_info_sync()

                if tick and account:
                    price = tick.get('bid', 0)
                    balance = account.get('balance', 0)
                    spread = tick.get('spread', 0)

                    # Get market condition
                    now = datetime.now(timezone.utc)
                    adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)
                    session = self._get_session_name(now.hour)
                    in_kz = "KZ" if self._is_kill_zone(now.hour) else "--"
                    market_status = self._get_market_status(now)
                    status = self.executor.get_status()
                    pos = "POS" if status['has_position'] else "---"

                    # Log status every cycle
                    logger.info(
                        f"[{self.loop_count:04d}] {SYMBOL} {price:.5f} | "
                        f"Sprd: {spread:.1f} | "
                        f"{session:10} | {in_kz} | "
                        f"Mkt: {market_status:6} | "
                        f"Q+{adj:02d} | "
                        f"${balance:,.0f} | {pos}"
                    )

                    # Send hourly status to Telegram
                    await self._send_hourly_status(now, price, balance)

                # Run trading cycle
                await self.executor.run_cycle()

                # Calculate sleep time
                elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_time = max(1, self.interval - elapsed)

                logger.debug(f"Cycle completed in {elapsed:.1f}s, sleeping {sleep_time:.0f}s")
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry

        logger.info("Trading loop stopped")

    async def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down...")

        if self.telegram:
            await self.telegram.send(f"{TelegramFormatter.RED} <b>H1 v6.4 GBPUSD Bot Stopped</b>")
            await self.telegram.stop_polling()

        if self.db:
            await self.db.disconnect()

        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SURGE-WSI H1 v6.4 GBPUSD Trading Bot'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run in demo mode (default)'
    )
    parser.add_argument(
        '--live', action='store_true',
        help='Run in live trading mode'
    )
    parser.add_argument(
        '--interval', type=int, default=300,
        help='Check interval in seconds (default: 300)'
    )

    args = parser.parse_args()

    mode = 'live' if args.live else 'demo'

    # Print banner
    print("=" * 60)
    print("SURGE-WSI H1 v6.4 GBPUSD")
    print("Dual-Layer Quality Filter")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Interval: {args.interval}s")
    print(f"Symbol: {SYMBOL}")
    print("=" * 60)
    print("\nBacktest Performance:")
    print("  - 147 trades, 0.40/day")
    print("  - 42.2% WR, PF 3.98")
    print("  - +$23,394 (+46.8% return)")
    print("  - ZERO losing months")
    print("=" * 60)

    # Create and run bot
    bot = TradingBot(mode=mode, interval=args.interval)

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        bot.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
