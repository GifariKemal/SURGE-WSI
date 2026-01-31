"""SURGE-WSI Main - H1 v6.2 GBPUSD Market Analysis Based Strategy
================================================================

Production-ready H1 GBPUSD trading system based on market analysis:
- 151 trades, 40.4% WR, +$21,938, PF 3.69
- Return: +43.9% in 13 months
- ZERO losing months
- Max DD: 1.7%

Key Features:
- ATR-based SL/TP (dynamic)
- Monthly risk adjustment from market analysis
- London & NY session focus

Usage:
    python main_h1_v6_2_gbpusd.py [--demo] [--live] [--interval 300]

Estimated Trades: ~2-3 per week (0.41 trades/day)

Author: SURIOTA Team
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import config
from src.utils.logger import setup_logger
from src.utils.telegram import TelegramNotifier, TelegramFormatter as TF
from src.data.mt5_connector import MT5Connector
from src.data.db_handler import DBHandler
from src.trading.executor_h1_v6_2_gbpusd import TradeExecutorH1V62GBPUSD, ExecutorState


class SurgeWSIH1V62GBPUSD:
    """H1 v6.2 GBPUSD Trading Application - Market Analysis Based"""

    def __init__(self, mode: str = "demo", verbose: bool = False):
        self.mode = mode
        self.verbose = verbose
        self.autotrading_enabled = False

        log_level = "DEBUG" if verbose else "INFO"
        setup_logger(
            log_level=log_level,
            log_file=f"logs/surge_h1_v6_2_gbpusd_{mode}.log",
            console=True
        )

        logger.info("=" * 60)
        logger.info(f"SURGE-WSI H1 v6.2 GBPUSD Starting - Mode: {mode.upper()}")
        logger.info("=" * 60)

        self.mt5 = MT5Connector(
            login=config.mt5.login,
            password=config.mt5.password,
            server=config.mt5.server,
            terminal_path=config.mt5.terminal_path
        )

        self.db = DBHandler(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.user,
            password=config.database.password
        )

        self.telegram: Optional[TelegramNotifier] = None
        if config.telegram.enabled and config.telegram.bot_token:
            self.telegram = TelegramNotifier(
                bot_token=config.telegram.bot_token,
                chat_id=config.telegram.chat_id
            )

        self.executor = TradeExecutorH1V62GBPUSD(
            symbol="GBPUSD",
            warmup_bars=100,
            magic_number=20250131  # Different magic for v6.2
        )

        self._running = False
        self._last_hourly_status = None

    async def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing components...")

        if not self.mt5.connect():
            logger.error("Failed to connect to MT5")
            return False

        self._db_connected = False
        try:
            if await self.db.connect():
                await self.db.initialize_tables()
                self._db_connected = True
        except Exception as e:
            logger.warning(f"Database not available: {e}")

        self.autotrading_enabled = self.mt5.is_autotrading_enabled()
        if self.autotrading_enabled:
            logger.info("AutoTrading: ENABLED")
        else:
            logger.warning("AutoTrading: DISABLED - Enable in MT5!")

        if self.telegram:
            await self.telegram.initialize()
            self._setup_telegram_callbacks()

        self._setup_executor_callbacks()

        logger.info("Initialization complete")
        return True

    def _setup_executor_callbacks(self):
        """Setup executor MT5 callbacks"""
        self.executor.set_callbacks(
            get_account_info=self._get_account_info,
            get_tick=self._get_tick,
            get_ohlcv=self._get_ohlcv,
            get_symbol_info=self._get_symbol_info,
            place_market_order=self._place_order,
            modify_position=self._modify_position,
            close_position=self._close_position,
            get_positions=self._get_positions,
            get_deal_history=self._get_deal_history,
            send_telegram=self._send_telegram
        )

    def _setup_telegram_callbacks(self):
        """Setup Telegram command callbacks"""
        if not self.telegram:
            return

        self.telegram.on_status = self._telegram_status
        self.telegram.on_balance = self._telegram_balance
        self.telegram.on_positions = self._telegram_positions
        self.telegram.on_regime = self._telegram_regime
        self.telegram.on_pause = self._telegram_pause
        self.telegram.on_resume = self._telegram_resume
        self.telegram.on_test_buy = self._telegram_test_buy
        self.telegram.on_test_sell = self._telegram_test_sell

    # MT5 callbacks
    async def _get_account_info(self):
        return self.mt5.get_account_info_sync()

    async def _get_tick(self, symbol):
        return self.mt5.get_tick_sync(symbol)

    async def _get_ohlcv(self, symbol, timeframe, bars):
        return self.mt5.get_ohlcv(symbol, timeframe, bars)

    async def _get_symbol_info(self, symbol):
        return self.mt5.get_symbol_info(symbol)

    async def _get_deal_history(self, ticket):
        return self.mt5.get_deal_history(ticket)

    async def _place_order(self, symbol, order_type, volume, sl=0, tp=0, comment="", magic=0):
        return await self.mt5.place_market_order(symbol, order_type, volume, sl, tp, comment, magic)

    async def _modify_position(self, ticket, sl=None, tp=None):
        return await self.mt5.modify_position(ticket, sl, tp)

    async def _close_position(self, ticket):
        return await self.mt5.close_position(ticket)

    async def _get_positions(self):
        return self.mt5.get_positions_sync()

    async def _send_telegram(self, message):
        if self.telegram:
            await self.telegram.send(message)

    # Telegram commands
    async def _telegram_status(self):
        status = self.executor.get_status()
        msg = TF.header("H1 v6.2 GBPUSD Status", TF.CHART)
        msg += TF.spacer()
        msg += TF.item("Strategy", "H1 v6.2 (Market Analysis)")
        msg += TF.item("State", status['state'])
        msg += TF.item("Regime", status['regime'])
        msg += TF.item("Bias", status['bias'])
        msg += TF.item("Session", status['session'])
        msg += TF.item("Kill Zone", "Yes" if status['in_killzone'] else "No")
        msg += TF.item("ATR", f"{status['current_atr']:.1f} pips")
        msg += TF.spacer()
        msg += TF.item("Trades", status['stats']['trades'])
        msg += TF.item("Win Rate", f"{status['stats']['win_rate']:.1f}%")
        msg += TF.item("Net P/L", f"${status['stats']['net_pnl']:+.2f}", last=True)
        return msg

    async def _telegram_balance(self):
        account = self.mt5.get_account_info_sync()
        if account:
            return TF.balance(
                login=account['login'],
                balance=account['balance'],
                equity=account['equity'],
                profit=account['profit'],
                free_margin=account['free_margin']
            )
        return "Account info not available"

    async def _telegram_positions(self):
        positions = self.mt5.get_positions_sync()
        if not positions:
            return "No open positions"
        msg = TF.header("Open Positions", TF.MEMO)
        for p in positions:
            emoji = TF.profit_emoji(p['profit'])
            msg += f"{emoji} {p['type']} {p['volume']} @ {p['price_open']:.5f} | P/L: ${p['profit']:+.2f}\n"
        return msg

    async def _telegram_regime(self):
        regime_info = self.executor.regime_detector.last_info
        if regime_info:
            msg = TF.header("Market Regime", TF.CHART)
            msg += TF.item("Regime", regime_info.regime.value)
            msg += TF.item("Probability", f"{regime_info.probability:.1%}")
            msg += TF.item("Bias", regime_info.bias)
            msg += TF.item("Tradeable", "Yes" if regime_info.is_tradeable else "No")
            msg += TF.item("ATR", f"{self.executor._current_atr:.1f} pips", last=True)
            return msg
        return "Regime not available"

    async def _telegram_pause(self):
        self.executor.pause()

    async def _telegram_resume(self):
        self.executor.resume()

    async def _telegram_test_buy(self):
        self.autotrading_enabled = self.mt5.is_autotrading_enabled()
        if not self.autotrading_enabled:
            return "AutoTrading DISABLED in MT5!"
        symbol = "GBPUSD"
        tick = self.mt5.get_tick_sync(symbol)
        if not tick:
            return "Failed to get price"
        try:
            result = await self.mt5.place_market_order(
                symbol=symbol, order_type="BUY", volume=0.01,
                sl=0, tp=0, comment="H1v62_TEST", magic=20250131
            )
            if result and result.get('ticket'):
                return f"TEST BUY executed: Ticket {result.get('ticket')}"
            return f"TEST BUY failed: {result}"
        except Exception as e:
            return f"Error: {e}"

    async def _telegram_test_sell(self):
        self.autotrading_enabled = self.mt5.is_autotrading_enabled()
        if not self.autotrading_enabled:
            return "AutoTrading DISABLED in MT5!"
        symbol = "GBPUSD"
        tick = self.mt5.get_tick_sync(symbol)
        if not tick:
            return "Failed to get price"
        try:
            result = await self.mt5.place_market_order(
                symbol=symbol, order_type="SELL", volume=0.01,
                sl=0, tp=0, comment="H1v62_TEST", magic=20250131
            )
            if result and result.get('ticket'):
                return f"TEST SELL executed: Ticket {result.get('ticket')}"
            return f"TEST SELL failed: {result}"
        except Exception as e:
            return f"Error: {e}"

    async def warmup(self) -> bool:
        """Warmup with H1 historical data"""
        logger.info("Starting warmup...")

        logger.info("Fetching symbol info...")
        await self.executor.fetch_symbol_info()

        h1_data = self.mt5.get_ohlcv("GBPUSD", "H1", 200)

        if h1_data is None:
            logger.error("Failed to get H1 data")
            return False

        if not await self.executor.warmup(h1_data):
            return False

        logger.info("Checking for existing positions...")
        if not await self.executor.recover_positions():
            logger.warning("Position recovery failed, continuing anyway")

        return True

    async def _send_hourly_status(self, price: float, account: dict):
        """Send hourly status notification"""
        if not self.telegram:
            return

        now = datetime.now(timezone.utc)

        if self._last_hourly_status is not None:
            if now.hour == self._last_hourly_status.hour and now.date() == self._last_hourly_status.date():
                return

        self._last_hourly_status = now

        try:
            status = self.executor.get_status()
            msg = TF.header(f"H1 v6.2 GBPUSD - {now.strftime('%H:%M')} UTC", TF.CLOCK)
            msg += TF.spacer()
            msg += TF.item("Price", f"{price:.5f}")
            msg += TF.item("ATR", f"{status['current_atr']:.1f} pips")
            msg += TF.item("Session", status['session'])
            msg += TF.item("Regime", status['regime'])
            msg += TF.item("Bias", status['bias'])
            msg += TF.spacer()
            msg += TF.item("Balance", f"${account.get('balance', 0):,.2f}")
            msg += TF.item("Equity", f"${account.get('equity', 0):,.2f}")
            msg += TF.item("Trades", status['stats']['trades'])
            msg += TF.item("Win Rate", f"{status['stats']['win_rate']:.1f}%", last=True)
            await self.telegram.send(msg)
        except Exception as e:
            logger.warning(f"Failed to send hourly status: {e}")

    async def run(self, interval_seconds: int = 300):
        """Main run loop"""
        self._running = True

        if self.telegram:
            msg = TF.header("H1 v6.2 GBPUSD Started", TF.ROCKET)
            msg += TF.spacer()
            msg += TF.item("Mode", self.mode.upper())
            msg += TF.item("Strategy", "H1 v6.2 (Market Analysis)")
            msg += TF.item("Symbol", "GBPUSD")
            msg += TF.item("Timeframe", "H1")
            msg += TF.item("SL", "1.5x ATR (Dynamic)")
            msg += TF.item("TP", "1.5x SL")
            msg += TF.item("PF", "3.69 (backtest)")
            msg += TF.item("Trades/Day", "~0.4 (2-3/week)")
            autotrading_status = "ENABLED" if self.autotrading_enabled else "DISABLED"
            msg += TF.item("AutoTrading", autotrading_status, last=True)
            if not self.autotrading_enabled:
                msg += "\n\nEnable AutoTrading in MT5!"
            msg += "\n\nCommands: /status /balance /regime"
            await self.telegram.send(msg)
            asyncio.create_task(self.telegram.start_polling())

        loop_count = 0
        consecutive_failures = 0

        try:
            while self._running:
                loop_count += 1
                try:
                    if not self.mt5.is_connected():
                        logger.warning("MT5 connection lost, reconnecting...")
                        consecutive_failures += 1
                        if self.mt5.connect():
                            consecutive_failures = 0
                        else:
                            await asyncio.sleep(30)
                            continue

                    self.autotrading_enabled = self.mt5.is_autotrading_enabled()

                    account = self.mt5.get_account_info_sync()
                    if not account:
                        consecutive_failures += 1
                        await asyncio.sleep(interval_seconds)
                        continue

                    tick = self.mt5.get_tick_sync("GBPUSD")
                    if not tick:
                        consecutive_failures += 1
                        await asyncio.sleep(interval_seconds)
                        continue

                    consecutive_failures = 0
                    price = tick.get('bid', 0)
                    balance = account.get('balance', 0)

                    status = self.executor.get_status()
                    cb_status = "CB" if status['circuit_breaker']['triggered'] else ""
                    logger.info(
                        f"[{loop_count}] GBPUSD: {price:.5f} | "
                        f"ATR: {status['current_atr']:.1f} | "
                        f"Session: {status['session']} | "
                        f"Regime: {status['regime']} | "
                        f"Bias: {status['bias']} {cb_status}"
                    )

                    await self._send_hourly_status(price, account)

                    if self.autotrading_enabled:
                        result = await self.executor.process_tick(price, balance)
                        if result and result.success:
                            logger.info(f"Trade executed: {result.direction} @ {result.entry_price:.5f}")

                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    consecutive_failures += 1

                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Cleanup and shutdown"""
        logger.info("Shutting down...")
        self._running = False

        if self.telegram:
            stats = self.executor.stats
            msg = TF.header("H1 v6.2 GBPUSD Stopped", TF.WARNING)
            msg += TF.spacer()
            msg += TF.item("Trades", stats.trades_executed)
            msg += TF.item("Win Rate", f"{stats.win_rate:.1f}%")
            msg += TF.item("Net P/L", f"${stats.net_pnl:+.2f}", last=True)
            await self.telegram.send(msg)
            await self.telegram.stop_polling()

        self.mt5.disconnect()
        await self.db.disconnect()
        logger.info("Shutdown complete")


async def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="SURGE-WSI H1 v6.2 GBPUSD")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval (default: 300)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    mode = "live" if args.live else "demo"

    app = SurgeWSIH1V62GBPUSD(mode=mode, verbose=args.verbose)

    if not await app.initialize():
        logger.error("Initialization failed")
        return

    if not await app.warmup():
        logger.error("Warmup failed")
        return

    await app.run(interval_seconds=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
