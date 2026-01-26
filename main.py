"""SURGE-WSI Main Orchestrator
==============================

6-Layer Quantitative Trading System:
1. Kalman Filter → Noise reduction
2. HMM Regime → Market state
3. Kill Zone → Time filter
4. POI Detection → Entry zones
5. Entry Trigger → Precise entry
6. Exit Manager → Partial TP

Usage:
    python main.py [--demo] [--backtest]

Author: SURIOTA Team
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import config
from src.utils.logger import setup_logger
from src.utils.telegram import TelegramNotifier, TelegramFormatter as TF
from src.utils.killzone import KillZone
from src.data.mt5_connector import MT5Connector
from src.data.db_handler import DBHandler
from src.data.cache import RedisCache
from src.trading.executor import TradeExecutor, ExecutorState


class SurgeWSI:
    """Main SURGE-WSI Application"""

    def __init__(self, mode: str = "demo"):
        """Initialize SURGE-WSI

        Args:
            mode: Trading mode ('demo', 'live', 'backtest')
        """
        self.mode = mode

        # Setup logger
        setup_logger(
            log_level="INFO" if mode != "backtest" else "WARNING",
            log_file=f"logs/surge_wsi_{mode}.log",
            console=True
        )

        logger.info("=" * 60)
        logger.info(f"SURGE-WSI Starting - Mode: {mode.upper()}")
        logger.info("=" * 60)

        # Components
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

        self.cache = RedisCache(
            host=config.redis.host,
            port=config.redis.port
        )

        self.killzone = KillZone(
            london_start=config.killzone.london_start,
            london_end=config.killzone.london_end,
            new_york_start=config.killzone.new_york_start,
            new_york_end=config.killzone.new_york_end,
            enabled=config.killzone.enabled
        )

        self.telegram: Optional[TelegramNotifier] = None
        if config.telegram.enabled and config.telegram.bot_token:
            self.telegram = TelegramNotifier(
                bot_token=config.telegram.bot_token,
                chat_id=config.telegram.chat_id
            )

        self.executor = TradeExecutor(
            symbol=config.trading.symbol,
            timeframe_htf=config.trading.timeframe_htf,
            timeframe_ltf=config.trading.timeframe_ltf,
            warmup_bars=100,
            magic_number=config.trading.magic_number
        )

        # Set executor risk params
        self.executor.risk_manager.high_quality_threshold = config.risk.high_quality_threshold
        self.executor.risk_manager.high_quality_risk = config.risk.high_quality_risk
        self.executor.risk_manager.medium_quality_threshold = config.risk.medium_quality_threshold
        self.executor.risk_manager.medium_quality_risk = config.risk.medium_quality_risk
        self.executor.risk_manager.low_quality_risk = config.risk.low_quality_risk
        self.executor.risk_manager.daily_profit_target = config.risk.daily_profit_target
        self.executor.risk_manager.daily_loss_limit = config.risk.daily_loss_limit

        # Set exit manager params
        self.executor.exit_manager.tp1_rr = config.exit.tp1_rr
        self.executor.exit_manager.tp1_percent = config.exit.tp1_percent
        self.executor.exit_manager.tp2_rr = config.exit.tp2_rr
        self.executor.exit_manager.tp2_percent = config.exit.tp2_percent
        self.executor.exit_manager.tp3_rr = config.exit.tp3_rr
        self.executor.exit_manager.tp3_percent = config.exit.tp3_percent
        self.executor.exit_manager.trailing_enabled = config.exit.trailing_enabled
        self.executor.exit_manager.trailing_step_pips = config.exit.trailing_step_pips
        self.executor.exit_manager.move_sl_to_be_at_tp1 = config.exit.move_sl_to_be_at_tp1

        # Running state
        self._running = False

    async def initialize(self) -> bool:
        """Initialize all components

        Returns:
            True if successful
        """
        logger.info("Initializing components...")

        # Connect MT5
        if not self.mt5.connect():
            logger.error("Failed to connect to MT5")
            return False

        # Connect database (optional)
        try:
            await self.db.connect()
            await self.db.initialize_tables()
        except Exception as e:
            logger.warning(f"Database not available: {e}")

        # Connect Redis (optional)
        self.cache.connect()

        # Initialize Telegram
        if self.telegram:
            await self.telegram.initialize()
            self._setup_telegram_callbacks()

        # Set executor callbacks
        self._setup_executor_callbacks()

        logger.info("Initialization complete")
        return True

    def _setup_executor_callbacks(self):
        """Setup executor MT5 callbacks"""
        self.executor.set_callbacks(
            get_account_info=self._get_account_info,
            get_tick=self._get_tick,
            get_ohlcv_htf=self._get_ohlcv_htf,
            get_ohlcv_ltf=self._get_ohlcv_ltf,
            place_market_order=self._place_order,
            modify_position=self._modify_position,
            close_position=self._close_position,
            close_partial=self._close_partial,
            get_positions=self._get_positions,
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
        self.telegram.on_pois = self._telegram_pois
        self.telegram.on_pause = self._telegram_pause
        self.telegram.on_resume = self._telegram_resume
        self.telegram.on_close_all = self._telegram_close_all

    # MT5 callbacks
    async def _get_account_info(self):
        return self.mt5.get_account_info_sync()

    async def _get_tick(self, symbol):
        return self.mt5.get_tick_sync(symbol)

    async def _get_ohlcv_htf(self, symbol, timeframe, bars):
        return self.mt5.get_ohlcv(symbol, timeframe, bars)

    async def _get_ohlcv_ltf(self, symbol, timeframe, bars):
        return self.mt5.get_ohlcv(symbol, timeframe, bars)

    async def _place_order(self, symbol, order_type, volume, sl=0, tp=0, comment="", magic=0):
        return await self.mt5.place_market_order(symbol, order_type, volume, sl, tp, comment, magic)

    async def _modify_position(self, ticket, sl=None, tp=None):
        return await self.mt5.modify_position(ticket, sl, tp)

    async def _close_position(self, ticket):
        return await self.mt5.close_position(ticket)

    async def _close_partial(self, ticket, volume):
        return await self.mt5.close_partial(ticket, volume)

    async def _get_positions(self):
        return self.mt5.get_positions_sync()

    async def _send_telegram(self, message):
        if self.telegram:
            await self.telegram.send(message)

    # Telegram command handlers
    async def _telegram_status(self):
        status = self.executor.get_status()
        in_kz, session = self.killzone.is_in_killzone()
        return TF.status(
            state=status['state'],
            regime=status['regime'],
            in_killzone=in_kz,
            session=session,
            positions=status['open_positions'],
            daily_pnl=status['daily_pnl']
        )

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
            msg += TF.item("Tradeable", "Yes" if regime_info.is_tradeable else "No", last=True)
            return msg
        return "Regime not available"

    async def _telegram_pois(self):
        poi_result = self.executor.poi_detector.last_result
        if not poi_result:
            return "POIs not available"

        msg = TF.header("Active POIs", TF.TARGET)
        bull = poi_result.bullish_pois
        bear = poi_result.bearish_pois
        msg += TF.item("Bullish POIs", len(bull))
        msg += TF.item("Bearish POIs", len(bear), last=True)
        return msg

    async def _telegram_pause(self):
        self.executor.pause()

    async def _telegram_resume(self):
        self.executor.resume()

    async def _telegram_close_all(self):
        positions = self.mt5.get_positions_sync()
        if not positions:
            return "No positions to close"

        closed = 0
        for p in positions:
            if await self.mt5.close_position(p['ticket']):
                closed += 1

        return f"Closed {closed}/{len(positions)} positions"

    async def warmup(self):
        """Warmup with historical data"""
        logger.info("Starting warmup...")

        htf_data = self.mt5.get_ohlcv(
            config.trading.symbol,
            config.trading.timeframe_htf,
            200
        )

        ltf_data = self.mt5.get_ohlcv(
            config.trading.symbol,
            config.trading.timeframe_ltf,
            500
        )

        if htf_data is None or ltf_data is None:
            logger.error("Failed to get historical data")
            return False

        return await self.executor.warmup(htf_data, ltf_data)

    async def run(self, interval_seconds: int = 60):
        """Main run loop

        Args:
            interval_seconds: Polling interval
        """
        self._running = True

        # Send startup message
        if self.telegram:
            is_open, market_msg = self.killzone.is_market_open()
            in_kz, session = self.killzone.is_in_killzone()

            msg = TF.header("SURGE-WSI Started", TF.ROCKET)
            msg += TF.spacer()
            msg += TF.item("Mode", self.mode.upper())
            msg += TF.item("Symbol", config.trading.symbol)
            msg += TF.item("Market", market_msg)
            msg += TF.item("Session", session)
            msg += TF.item("Kill Zone", "Yes" if in_kz else "No", last=True)
            msg += TF.spacer()
            msg += "Commands: /status /balance /positions /regime"

            await self.telegram.send(msg)
            # Start polling for commands
            asyncio.create_task(self.telegram.start_polling())

        try:
            while self._running:
                try:
                    # Get account info
                    account = self.mt5.get_account_info_sync()
                    if not account:
                        await asyncio.sleep(interval_seconds)
                        continue

                    # Get current price
                    tick = self.mt5.get_tick_sync(config.trading.symbol)
                    if not tick:
                        await asyncio.sleep(interval_seconds)
                        continue

                    price = tick.get('bid', 0)
                    balance = account.get('balance', 0)

                    # Process tick
                    result = await self.executor.process_tick(price, balance)

                    if result and result.success:
                        logger.info(f"Trade executed: {result.direction} @ {result.entry_price:.5f}")

                except Exception as e:
                    logger.error(f"Error in main loop: {e}")

                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Cleanup and shutdown"""
        logger.info("Shutting down...")
        self._running = False

        # Send shutdown message
        if self.telegram:
            stats = self.executor.stats
            msg = TF.header("SURGE-WSI Stopped", TF.WARNING)
            msg += TF.spacer()
            msg += TF.item("Trades", stats.trades_executed)
            msg += TF.item("Win Rate", f"{stats.win_rate:.1f}%")
            msg += TF.item("Net P/L", f"${stats.net_pnl:+.2f}", last=True)
            await self.telegram.send(msg)
            await self.telegram.stop_polling()

        # Disconnect
        self.mt5.disconnect()
        await self.db.disconnect()
        self.cache.disconnect()

        logger.info("Shutdown complete")


async def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="SURGE-WSI Trading System")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval (seconds)")
    args = parser.parse_args()

    # Determine mode
    if args.live:
        mode = "live"
    elif args.backtest:
        mode = "backtest"
    else:
        mode = "demo"

    # Create and run
    app = SurgeWSI(mode=mode)

    if not await app.initialize():
        logger.error("Initialization failed")
        return

    if not await app.warmup():
        logger.error("Warmup failed")
        return

    await app.run(interval_seconds=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
