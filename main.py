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

# MT5 Timeframe mapping for auto sync
import MetaTrader5 as mt5
TIMEFRAMES_MAP = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class SurgeWSI:
    """Main SURGE-WSI Application"""

    def __init__(self, mode: str = "demo", verbose: bool = False):
        """Initialize SURGE-WSI

        Args:
            mode: Trading mode ('demo', 'live', 'backtest')
            verbose: Enable verbose logging
        """
        self.mode = mode
        self.verbose = verbose
        self.autotrading_enabled = False

        # Setup logger
        log_level = "DEBUG" if verbose else ("INFO" if mode != "backtest" else "WARNING")
        setup_logger(
            log_level=log_level,
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
            warmup_bars=1000,  # Increased from 100 for better analysis
            magic_number=config.trading.magic_number
        )

        # Set executor risk params - ZERO LOSING MONTHS CONFIG
        self.executor.risk_manager.high_quality_threshold = config.risk.high_quality_threshold
        self.executor.risk_manager.high_quality_risk = config.risk.high_quality_risk
        self.executor.risk_manager.medium_quality_threshold = config.risk.medium_quality_threshold
        self.executor.risk_manager.medium_quality_risk = config.risk.medium_quality_risk
        self.executor.risk_manager.low_quality_risk = config.risk.low_quality_risk
        self.executor.risk_manager.daily_profit_target = config.risk.daily_profit_target
        self.executor.risk_manager.daily_loss_limit = config.risk.daily_loss_limit
        self.executor.risk_manager.max_lot_size = config.risk.max_lot_size
        self.executor.risk_manager.min_sl_pips = config.risk.min_sl_pips
        self.executor.risk_manager.max_sl_pips = config.risk.max_sl_pips
        self.executor.risk_manager.max_loss_per_trade_pct = config.risk.max_loss_per_trade_pct
        self.executor.risk_manager.monthly_loss_stop_pct = config.risk.monthly_loss_stop_pct

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
        self._last_hourly_status = None  # Track last hourly notification time

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
        self._db_connected = False
        try:
            if await self.db.connect():
                await self.db.initialize_tables()
                self._db_connected = True
                # Auto sync initial data (separate try-except to not crash if sync fails)
                try:
                    await self._auto_sync_database()
                except Exception as sync_err:
                    logger.warning(f"Database sync failed (non-critical): {sync_err}")
        except Exception as e:
            logger.warning(f"Database not available: {e}")

        # Connect Redis (optional)
        self.cache.connect()

        # Check AutoTrading status
        self.autotrading_enabled = self.mt5.is_autotrading_enabled()
        if self.autotrading_enabled:
            logger.info("AutoTrading: ENABLED - System can execute trades")
        else:
            logger.warning("AutoTrading: DISABLED - Enable in MT5 terminal to trade!")

        # Initialize Telegram
        if self.telegram:
            await self.telegram.initialize()
            self._setup_telegram_callbacks()

        # Set executor callbacks
        self._setup_executor_callbacks()

        logger.info("Initialization complete")
        return True

    async def _auto_sync_database(self):
        """Auto sync recent data from MT5 to database"""
        if not self._db_connected:
            return

        import pandas as pd

        logger.info("Auto syncing data to database...")
        symbol = config.trading.symbol
        timeframes = ["M5", "M15", "H1", "H4", "D1"]
        bars_per_tf = {"M5": 500, "M15": 200, "H1": 100, "H4": 50, "D1": 30}

        total_synced = 0
        for tf in timeframes:
            try:
                tf_enum = TIMEFRAMES_MAP.get(tf)
                if tf_enum is None:
                    continue

                bars = bars_per_tf.get(tf, 100)
                rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, bars)

                if rates is None or len(rates) == 0:
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)
                df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
                df = df[['open', 'high', 'low', 'close', 'tick_volume']]
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)

                # Save to database
                count = await self.db.save_ohlcv(symbol, tf, df)
                total_synced += count if count else 0
                logger.debug(f"Synced {count} bars for {symbol} {tf}")

            except Exception as e:
                logger.warning(f"Failed to sync {tf}: {e}")

        logger.info(f"Auto sync complete: {total_synced} bars synced to database")

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
        self.telegram.on_activity = self._telegram_activity
        self.telegram.on_pause = self._telegram_pause
        self.telegram.on_resume = self._telegram_resume
        self.telegram.on_close_all = self._telegram_close_all
        self.telegram.on_test_buy = self._telegram_test_buy
        self.telegram.on_test_sell = self._telegram_test_sell
        self.telegram.on_autotrading = self._telegram_autotrading

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

    async def _send_hourly_status(self, price: float, account: dict):
        """Send hourly status notification to Telegram"""
        if not self.telegram:
            return

        now = datetime.now(timezone.utc)

        # Check if hour has changed since last notification
        if self._last_hourly_status is not None:
            if now.hour == self._last_hourly_status.hour and now.date() == self._last_hourly_status.date():
                return  # Same hour, skip

        self._last_hourly_status = now

        try:
            # Gather status info (with safe access)
            in_kz, session = self.executor.is_in_killzone()
            regime_info = self.executor.regime_detector.last_info
            daily_stats = self.executor.risk_manager.get_daily_stats() if hasattr(self.executor.risk_manager, 'get_daily_stats') else {'pnl': 0, 'trades': 0}

            # Get intelligent filter status (may be None before first tick)
            intel_result = getattr(self.executor, '_last_intelligent_result', None)
            use_intel = getattr(self.executor, 'use_intelligent_filter', False)

            # Build message
            msg = TF.header(f"Hourly Status - {now.strftime('%H:%M')} UTC", TF.CLOCK)
            msg += TF.spacer()
            msg += TF.item("Price", f"{price:.5f}")

            # Show intelligent filter status or Kill Zone
            if use_intel and intel_result is not None:
                activity_emoji = intel_result.get_emoji()
                msg += TF.item("Activity", f"{activity_emoji} {intel_result.activity.value.upper()} ({intel_result.score:.0f}/100)")
                msg += TF.item("Can Trade", "YES ✅" if intel_result.should_trade else "NO ⏸️")
            else:
                msg += TF.item("Session", session if in_kz else "Outside KZ")
                msg += TF.item("Kill Zone", "Yes" if in_kz else "No")
                if use_intel:
                    msg += TF.item("Activity", "⏳ Warming up...")

            msg += TF.item("Regime", regime_info.regime.value if regime_info else "N/A")
            msg += TF.item("Bias", regime_info.bias if regime_info else "N/A")
            msg += TF.spacer()
            msg += TF.item("Balance", f"${account.get('balance', 0):,.2f}")
            msg += TF.item("Equity", f"${account.get('equity', 0):,.2f}")
            msg += TF.item("Daily P/L", f"${daily_stats.get('pnl', 0):+.2f}")
            msg += TF.item("Trades Today", daily_stats.get('trades', 0), last=True)

            # Add mode indicator
            if use_intel:
                msg += "\n🧠 <i>Mode: INTEL_60</i>"

            await self.telegram.send(msg)
            logger.debug(f"Hourly status sent to Telegram")
        except Exception as e:
            logger.warning(f"Failed to send hourly status: {e}")

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

    async def _telegram_activity(self):
        """Check intelligent activity filter status"""
        intel_result = self.executor._last_intelligent_result
        use_intel = self.executor.use_intelligent_filter

        if not use_intel:
            return "Intelligent Filter is DISABLED\nUsing Kill Zone mode instead."

        if not intel_result:
            return "Activity data not available yet.\nWaiting for first tick..."

        emoji = intel_result.get_emoji()
        msg = TF.header("Intelligent Activity Filter", "🧠")
        msg += TF.spacer()
        msg += TF.item("Mode", "INTEL_60")
        msg += TF.item("Activity", f"{emoji} {intel_result.activity.value.upper()}")
        msg += TF.item("Score", f"{intel_result.score:.0f}/100")
        msg += TF.item("Should Trade", "YES ✅" if intel_result.should_trade else "NO ⏸️")
        msg += TF.spacer()
        msg += TF.item("Velocity", f"{intel_result.velocity:.2f} pips/bar ({intel_result.velocity_score:.0f}/30)")
        msg += TF.item("ATR", f"{intel_result.atr_pips:.1f} pips ({intel_result.atr_score:.0f}/30)")
        msg += TF.item("Range", f"{intel_result.range_pips:.1f} pips ({intel_result.range_score:.0f}/20)")
        msg += TF.item("Momentum", f"{intel_result.momentum:.1f} pips ({intel_result.momentum_score:.0f}/20)")
        msg += TF.spacer()
        msg += TF.item("Quality Threshold", f"{intel_result.quality_threshold:.0f}")
        msg += TF.item("Lot Multiplier", f"{intel_result.lot_multiplier:.1f}x", last=True)

        if not intel_result.should_trade:
            msg += f"\n\n⏳ <i>Reason: {intel_result.reason}</i>"

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

    async def _telegram_test_buy(self):
        """Test BUY order with 0.01 lot"""
        # Always check autotrading status live
        self.autotrading_enabled = self.mt5.is_autotrading_enabled()
        if not self.autotrading_enabled:
            return "AutoTrading DISABLED in MT5!\n\nEnable: Tools > Options > Expert Advisors > Allow Algo Trading\nOr click AutoTrading button (Ctrl+E)"

        symbol = config.trading.symbol
        tick = self.mt5.get_tick_sync(symbol)
        if not tick:
            return "Failed to get price from MT5"

        try:
            logger.info(f"Executing TEST BUY {symbol} 0.01 lot @ {tick['ask']:.5f}")
            result = await self.mt5.place_market_order(
                symbol=symbol,
                order_type="BUY",
                volume=0.01,
                sl=0,
                tp=0,
                comment="TEST_BUY",
                magic=config.trading.magic_number
            )

            if result is None:
                return "TEST BUY Failed: Order rejected by MT5\n\nCheck MT5 Journal for details."

            if result.get('success') or result.get('ticket'):
                msg = TF.header("TEST BUY Executed", TF.ROCKET)
                msg += TF.item("Symbol", symbol)
                msg += TF.item("Volume", "0.01 lot")
                msg += TF.item("Price", f"{result.get('price', tick['ask']):.5f}")
                msg += TF.item("Ticket", result.get('ticket', 'N/A'), last=True)
                logger.info(f"TEST BUY success: ticket={result.get('ticket')}")
                return msg
            else:
                error_msg = result.get('message', result.get('comment', 'Unknown error'))
                return f"TEST BUY Failed: {error_msg}"

        except Exception as e:
            logger.error(f"TEST BUY exception: {e}")
            return f"TEST BUY Error: {e}"

    async def _telegram_test_sell(self):
        """Test SELL order with 0.01 lot"""
        # Always check autotrading status live
        self.autotrading_enabled = self.mt5.is_autotrading_enabled()
        if not self.autotrading_enabled:
            return "AutoTrading DISABLED in MT5!\n\nEnable: Tools > Options > Expert Advisors > Allow Algo Trading\nOr click AutoTrading button (Ctrl+E)"

        symbol = config.trading.symbol
        tick = self.mt5.get_tick_sync(symbol)
        if not tick:
            return "Failed to get price from MT5"

        try:
            logger.info(f"Executing TEST SELL {symbol} 0.01 lot @ {tick['bid']:.5f}")
            result = await self.mt5.place_market_order(
                symbol=symbol,
                order_type="SELL",
                volume=0.01,
                sl=0,
                tp=0,
                comment="TEST_SELL",
                magic=config.trading.magic_number
            )

            if result is None:
                return "TEST SELL Failed: Order rejected by MT5\n\nCheck MT5 Journal for details."

            if result.get('success') or result.get('ticket'):
                msg = TF.header("TEST SELL Executed", TF.ROCKET)
                msg += TF.item("Symbol", symbol)
                msg += TF.item("Volume", "0.01 lot")
                msg += TF.item("Price", f"{result.get('price', tick['bid']):.5f}")
                msg += TF.item("Ticket", result.get('ticket', 'N/A'), last=True)
                logger.info(f"TEST SELL success: ticket={result.get('ticket')}")
                return msg
            else:
                error_msg = result.get('message', result.get('comment', 'Unknown error'))
                return f"TEST SELL Failed: {error_msg}"

        except Exception as e:
            logger.error(f"TEST SELL exception: {e}")
            return f"TEST SELL Error: {e}"

    async def _telegram_autotrading(self):
        """Check AutoTrading status"""
        self.autotrading_enabled = self.mt5.is_autotrading_enabled()

        # Also get terminal info for more details
        terminal_info = mt5.terminal_info()

        msg = TF.header("AutoTrading Status", TF.GEAR)
        if self.autotrading_enabled:
            msg += TF.item("Status", "ENABLED", last=True)
            msg += "\nSystem can execute trades automatically."
        else:
            msg += TF.item("Status", "DISABLED", last=True)
            msg += "\n\nTo Enable AutoTrading:"
            msg += "\n1. Tools > Options > Expert Advisors"
            msg += "\n2. Check 'Allow Algorithmic Trading'"
            msg += "\n3. Or press Ctrl+E"

        if terminal_info:
            msg += f"\n\nTerminal: {terminal_info.name}"
            msg += f"\nTrade Allowed: {terminal_info.trade_allowed}"

        return msg

    async def warmup(self):
        """Warmup with historical data

        Increased bars for optimal performance:
        - HTF (H4): 500 bars = ~83 days for better POI/swing detection
        - LTF (M5): 1000 bars = ~3.5 days for better Kalman/velocity smoothing
        """
        logger.info("Starting warmup...")

        htf_data = self.mt5.get_ohlcv(
            config.trading.symbol,
            config.trading.timeframe_htf,
            500  # Increased from 200 for better POI detection
        )

        ltf_data = self.mt5.get_ohlcv(
            config.trading.symbol,
            config.trading.timeframe_ltf,
            1000  # Increased from 500 for better velocity/momentum
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

            # Check if using intelligent filter
            use_intel = self.executor.use_intelligent_filter

            msg = TF.header("SURGE-WSI Started", TF.ROCKET)
            msg += TF.spacer()
            msg += TF.item("Mode", self.mode.upper())
            msg += TF.item("Symbol", config.trading.symbol)
            msg += TF.item("Market", market_msg)

            if use_intel:
                msg += TF.item("Filter", "🧠 INTEL_60 (Intelligent Activity)")
            else:
                msg += TF.item("Session", session)
                msg += TF.item("Kill Zone", "Yes" if in_kz else "No")

            autotrading_status = "ENABLED" if self.autotrading_enabled else "DISABLED"
            msg += TF.item("AutoTrading", autotrading_status, last=True)
            msg += TF.spacer()
            if not self.autotrading_enabled:
                msg += "⚠️ Enable AutoTrading in MT5!\n\n"
            msg += "Commands:\n/status /balance /activity\n/test_buy /test_sell /autotrading"

            await self.telegram.send(msg)
            # Start polling for commands
            asyncio.create_task(self.telegram.start_polling())

        loop_count = 0
        try:
            while self._running:
                loop_count += 1
                try:
                    # Get account info
                    account = self.mt5.get_account_info_sync()
                    if not account:
                        logger.warning("Failed to get account info")
                        await asyncio.sleep(interval_seconds)
                        continue

                    # Get current price
                    tick = self.mt5.get_tick_sync(config.trading.symbol)
                    if not tick:
                        logger.warning("Failed to get tick")
                        await asyncio.sleep(interval_seconds)
                        continue

                    price = tick.get('bid', 0)
                    balance = account.get('balance', 0)

                    # Log status every loop (was every 5 loops)
                    if loop_count % 1 == 0:  # Every loop = every interval
                        in_kz, session = self.executor.is_in_killzone()
                        regime_info = self.executor.regime_detector.last_info
                        state = self.executor.state.value
                        regime_str = regime_info.regime.value if regime_info else 'N/A'
                        bias_str = regime_info.bias if regime_info else 'N/A'

                        logger.info(
                            f"[Loop {loop_count}] Price: {price:.5f} | "
                            f"State: {state} | Session: {session} | "
                            f"KZ: {'Yes' if in_kz else 'No'} | "
                            f"Regime: {regime_str} | Bias: {bias_str}"
                        )

                    # Hourly status notification to Telegram
                    await self._send_hourly_status(price, account)

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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Determine mode
    if args.live:
        mode = "live"
    elif args.backtest:
        mode = "backtest"
    else:
        mode = "demo"

    # Create and run
    app = SurgeWSI(mode=mode, verbose=args.verbose)

    if not await app.initialize():
        logger.error("Initialization failed")
        return

    if not await app.warmup():
        logger.error("Warmup failed")
        return

    await app.run(interval_seconds=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
