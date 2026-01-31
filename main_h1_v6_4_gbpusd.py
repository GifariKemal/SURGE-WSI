"""
SURGE-WSI H1 v6.4 GBPUSD Main Entry Point
==========================================

Dual-Layer Quality Filter for ZERO losing months

Usage:
    python main_h1_v6_4_gbpusd.py [--demo] [--live] [--interval 300]

Telegram Commands:
    /status     - System status & market conditions
    /balance    - Account balance & equity
    /positions  - View open positions
    /regime     - Current market regime (Bullish/Bearish/Sideways)
    /pois       - Active Points of Interest (Order Blocks, FVG)
    /activity   - Intelligent Activity Filter status
    /mode       - Current trading mode
    /market     - Market Analysis (detailed)
    /pause      - Pause auto trading
    /resume     - Resume auto trading
    /close_all  - Close all open positions
    /test_buy   - Test BUY order (0.01 lot)
    /test_sell  - Test SELL order (0.01 lot)
    /autotrading - Check MT5 AutoTrading status
    /help       - Show all available commands

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
from src.trading.executor_h1_v6_4_gbpusd import (
    H1V64GBPUSDExecutor, SYMBOL, TIMEFRAME, Regime,
    MONTHLY_TRADEABLE_PCT, HOUR_MULTIPLIERS, DAY_MULTIPLIERS
)
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
        self.paused = False  # Paused state (can resume)
        self.executor = None
        self.telegram = None
        self.db = None
        self.mt5 = None  # MT5 connector reference
        self.loop_count = 0
        self._last_hourly_status = None
        self._cached_regime = None
        self._cached_pois = []
        self._cached_market_condition = None

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
        # Get MT5 credentials from config.mt5
        mt5_connector = MT5Connector(
            login=config.mt5.login,
            password=config.mt5.password,
            server=config.mt5.server,
            terminal_path=config.mt5.terminal_path
        )

        logger.info(f"MT5 Config: server={config.mt5.server}, login={config.mt5.login or 'auto'}")

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
            info = self.mt5.get_account_info_sync()
            if info:
                balance = info.get('balance', 0)
                equity = info.get('equity', 0)
                margin = info.get('margin', 0)
                free_margin = info.get('margin_free', 0)
                profit = info.get('profit', 0)

                msg = TelegramFormatter.tree_header("Account Info", TelegramFormatter.MONEY)
                msg += TelegramFormatter.tree_item("Balance", f"${balance:,.2f}")
                msg += TelegramFormatter.tree_item("Equity", f"${equity:,.2f}")
                msg += TelegramFormatter.tree_item("Margin", f"${margin:,.2f}")
                msg += TelegramFormatter.tree_item("Free Margin", f"${free_margin:,.2f}")
                pnl_emoji = TelegramFormatter.CHECK if profit >= 0 else TelegramFormatter.CROSS
                msg += TelegramFormatter.tree_item("Floating P/L", f"{pnl_emoji} ${profit:+,.2f}", last=True)
                return msg
            return f"{TelegramFormatter.CROSS} Failed to get account info"

        async def pause_handler():
            self.paused = True
            return f"{TelegramFormatter.WARNING} <b>Trading PAUSED</b>\n\nUse /resume to continue trading."

        async def resume_handler():
            self.paused = False
            return f"{TelegramFormatter.CHECK} <b>Trading RESUMED</b>\n\nBot is now active."

        async def positions_handler():
            positions = self.mt5.get_positions_sync(SYMBOL)
            if not positions:
                return f"{TelegramFormatter.MEMO} <b>No open positions</b>"

            msg = TelegramFormatter.tree_header(f"Open Positions ({len(positions)})", TelegramFormatter.CHART)
            for p in positions:
                emoji = TelegramFormatter.UP if p['type'] == 'BUY' else TelegramFormatter.DOWN
                pnl_emoji = TelegramFormatter.CHECK if p['profit'] >= 0 else TelegramFormatter.CROSS
                msg += f"\n{emoji} <b>#{p['ticket']}</b> {p['type']} {p['volume']}\n"
                msg += f"   Entry: {p['price_open']:.5f}\n"
                msg += f"   P/L: {pnl_emoji} ${p['profit']:+.2f}\n"
            return msg

        async def regime_handler():
            """Get current market regime"""
            try:
                df = await self.executor.get_ohlcv_data(SYMBOL, TIMEFRAME, 100)
                if df is None or df.empty:
                    return f"{TelegramFormatter.CROSS} Failed to get market data"

                # Map columns
                col_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'open' in col_lower:
                        col_map['open'] = col
                    elif 'high' in col_lower:
                        col_map['high'] = col
                    elif 'low' in col_lower:
                        col_map['low'] = col
                    elif 'close' in col_lower:
                        col_map['close'] = col

                regime, strength = self.executor.detect_regime(df, col_map)
                self._cached_regime = regime

                # Regime emoji
                if regime == Regime.BULLISH:
                    emoji = TelegramFormatter.UP
                    color = TelegramFormatter.GREEN
                elif regime == Regime.BEARISH:
                    emoji = TelegramFormatter.DOWN
                    color = TelegramFormatter.RED
                else:
                    emoji = "⚖️"
                    color = "🟡"

                msg = TelegramFormatter.tree_header("Market Regime", emoji)
                msg += TelegramFormatter.tree_item("Regime", f"{color} {regime.value.upper()}")
                msg += TelegramFormatter.tree_item("Strength", f"{strength*100:.0f}%")
                msg += TelegramFormatter.tree_item("Symbol", SYMBOL)
                msg += TelegramFormatter.tree_item("Timeframe", TIMEFRAME, last=True)

                # Trading recommendation
                if regime == Regime.BULLISH:
                    msg += f"\n{TelegramFormatter.BRAIN} <i>Bias: BUY signals preferred</i>"
                elif regime == Regime.BEARISH:
                    msg += f"\n{TelegramFormatter.BRAIN} <i>Bias: SELL signals preferred</i>"
                else:
                    msg += f"\n{TelegramFormatter.WARNING} <i>Sideway: No clear direction</i>"

                return msg
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def pois_handler():
            """Get active Points of Interest"""
            try:
                df = await self.executor.get_ohlcv_data(SYMBOL, TIMEFRAME, 100)
                if df is None or df.empty:
                    return f"{TelegramFormatter.CROSS} Failed to get market data"

                # Map columns
                col_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'open' in col_lower:
                        col_map['open'] = col
                    elif 'high' in col_lower:
                        col_map['high'] = col
                    elif 'low' in col_lower:
                        col_map['low'] = col
                    elif 'close' in col_lower:
                        col_map['close'] = col

                now = datetime.now(timezone.utc)
                adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)
                min_quality = 60 + adj

                pois = self.executor.detect_order_blocks(df, col_map, min_quality)
                self._cached_pois = pois

                if not pois:
                    return f"{TelegramFormatter.MEMO} <b>No active POIs</b>\n\nMin quality: {min_quality}"

                msg = TelegramFormatter.tree_header(f"Active POIs ({len(pois)})", "🎯")
                for i, poi in enumerate(pois[-5:]):  # Show last 5 POIs
                    emoji = TelegramFormatter.UP if poi['direction'] == 'BUY' else TelegramFormatter.DOWN
                    msg += f"\n{emoji} <b>Order Block</b>\n"
                    msg += f"   Direction: {poi['direction']}\n"
                    msg += f"   Price: {poi['price']:.5f}\n"
                    msg += f"   Quality: {poi['quality']:.0f}%\n"

                return msg
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def activity_handler():
            """Get activity filter status"""
            now = datetime.now(timezone.utc)
            hour = now.hour
            day = now.weekday()
            month = now.month

            hour_mult = HOUR_MULTIPLIERS.get(hour, 0.0)
            day_mult = DAY_MULTIPLIERS.get(day, 0.0)

            # Activity score (0-100)
            activity_score = hour_mult * day_mult * 100

            # Kill zone check
            in_kz, kz_name = self.executor.is_kill_zone(now)

            # Determine activity level
            if activity_score >= 80:
                level = f"{TelegramFormatter.GREEN} HIGH"
            elif activity_score >= 50:
                level = f"🟡 MEDIUM"
            elif activity_score > 0:
                level = f"🟠 LOW"
            else:
                level = f"{TelegramFormatter.RED} INACTIVE"

            msg = TelegramFormatter.tree_header("Activity Filter", "📊")
            msg += TelegramFormatter.tree_section("Time Filters", TelegramFormatter.CLOCK)
            msg += TelegramFormatter.tree_item("UTC Hour", str(hour))
            msg += TelegramFormatter.tree_item("Hour Mult", f"{hour_mult:.1f}x")
            msg += TelegramFormatter.tree_item("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day])
            msg += TelegramFormatter.tree_item("Day Mult", f"{day_mult:.1f}x", last=True)

            msg += TelegramFormatter.tree_section("Activity", "📈")
            msg += TelegramFormatter.tree_item("Score", f"{activity_score:.0f}%")
            msg += TelegramFormatter.tree_item("Level", level)
            msg += TelegramFormatter.tree_item("Kill Zone", f"{TelegramFormatter.CHECK} {kz_name.upper()}" if in_kz else f"{TelegramFormatter.CROSS} No", last=True)

            # Trading recommendation
            if activity_score >= 50 and in_kz:
                msg += f"\n{TelegramFormatter.CHECK} <i>Good time to trade</i>"
            elif activity_score > 0:
                msg += f"\n{TelegramFormatter.WARNING} <i>Reduced activity</i>"
            else:
                msg += f"\n{TelegramFormatter.RED} <i>Trading not recommended</i>"

            return msg

        async def mode_handler():
            """Get current trading mode"""
            now = datetime.now(timezone.utc)
            market_open = self._is_market_open(now)
            autotrading = self.mt5.is_autotrading_enabled()

            # Mode status
            mode_emoji = TelegramFormatter.CHECK if self.mode == 'demo' else TelegramFormatter.WARNING

            msg = TelegramFormatter.tree_header("Trading Mode", TelegramFormatter.GEAR)
            msg += TelegramFormatter.tree_item("Mode", f"{mode_emoji} {self.mode.upper()}")
            msg += TelegramFormatter.tree_item("Bot Status", f"{TelegramFormatter.GREEN} RUNNING" if self.running and not self.paused else f"{TelegramFormatter.RED} PAUSED")
            msg += TelegramFormatter.tree_item("AutoTrading", f"{TelegramFormatter.GREEN} ON" if autotrading else f"{TelegramFormatter.RED} OFF")
            msg += TelegramFormatter.tree_item("Market", f"{TelegramFormatter.GREEN} OPEN" if market_open else f"{TelegramFormatter.RED} CLOSED")
            msg += TelegramFormatter.tree_item("Interval", f"{self.interval}s")
            msg += TelegramFormatter.tree_item("Loop Count", str(self.loop_count), last=True)

            return msg

        async def market_handler():
            """Detailed market analysis"""
            try:
                now = datetime.now(timezone.utc)

                # Get market data
                df = await self.executor.get_ohlcv_data(SYMBOL, TIMEFRAME, 100)
                if df is None or df.empty:
                    return f"{TelegramFormatter.CROSS} Failed to get market data"

                # Map columns
                col_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'open' in col_lower:
                        col_map['open'] = col
                    elif 'high' in col_lower:
                        col_map['high'] = col
                    elif 'low' in col_lower:
                        col_map['low'] = col
                    elif 'close' in col_lower:
                        col_map['close'] = col

                # Calculate ATR
                atr_series = await self.executor.calculate_atr(df, col_map)
                current_atr = atr_series.iloc[-1]

                # Get regime
                regime, strength = self.executor.detect_regime(df, col_map)

                # Get market condition
                market_cond = self.executor.risk_scorer.assess_market_condition(
                    df, col_map, atr_series, now
                )
                self._cached_market_condition = market_cond

                # Get current price
                tick = self.mt5.get_tick_sync(SYMBOL)
                price = tick.get('bid', 0) if tick else df[col_map['close']].iloc[-1]
                spread = tick.get('spread', 0) if tick else 0

                # Monthly profile
                key = (now.year, now.month)
                tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)

                # Regime emoji
                if regime == Regime.BULLISH:
                    regime_str = f"{TelegramFormatter.UP} BULLISH"
                elif regime == Regime.BEARISH:
                    regime_str = f"{TelegramFormatter.DOWN} BEARISH"
                else:
                    regime_str = "⚖️ SIDEWAYS"

                # Condition emoji
                if market_cond.label in ["GOOD", "NORMAL"]:
                    cond_emoji = TelegramFormatter.GREEN
                elif market_cond.label == "CAUTION":
                    cond_emoji = "🟡"
                else:
                    cond_emoji = TelegramFormatter.RED

                msg = TelegramFormatter.tree_header("Market Analysis", TelegramFormatter.BRAIN)

                msg += TelegramFormatter.tree_section("Price Action", TelegramFormatter.CHART)
                msg += TelegramFormatter.tree_item("Symbol", SYMBOL)
                msg += TelegramFormatter.tree_item("Price", f"{price:.5f}")
                msg += TelegramFormatter.tree_item("Spread", f"{spread:.1f} pts")
                msg += TelegramFormatter.tree_item("ATR", f"{current_atr:.1f} pips", last=True)

                msg += TelegramFormatter.tree_section("Technical", "📐")
                msg += TelegramFormatter.tree_item("Regime", regime_str)
                msg += TelegramFormatter.tree_item("Strength", f"{strength*100:.0f}%")
                msg += TelegramFormatter.tree_item("Tech Quality", f"{market_cond.technical_quality:.0f}", last=True)

                msg += TelegramFormatter.tree_section("Quality Filter", "🎯")
                msg += TelegramFormatter.tree_item("Month Profile", f"{tradeable_pct}% tradeable")
                msg += TelegramFormatter.tree_item("Monthly Adj", f"+{market_cond.monthly_adjustment}")
                msg += TelegramFormatter.tree_item("Final Quality", f"{cond_emoji} {market_cond.final_quality:.0f}")
                msg += TelegramFormatter.tree_item("Label", market_cond.label, last=True)

                return msg
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def autotrading_handler():
            enabled = self.mt5.is_autotrading_enabled()
            now = datetime.now(timezone.utc)
            market_open = self._is_market_open(now)

            if enabled:
                at_status = f"{TelegramFormatter.GREEN} ENABLED"
            else:
                at_status = f"{TelegramFormatter.RED} DISABLED"

            if market_open:
                mkt_status = f"{TelegramFormatter.GREEN} OPEN"
            else:
                mkt_status = f"{TelegramFormatter.RED} CLOSED (Weekend)"

            msg = TelegramFormatter.tree_header("Trading Status", TelegramFormatter.GEAR)
            msg += TelegramFormatter.tree_item("AutoTrading", at_status)
            msg += TelegramFormatter.tree_item("Market", mkt_status, last=True)

            if not enabled:
                msg += f"\n{TelegramFormatter.WARNING} <i>Enable AutoTrading di MT5 terminal!</i>"
            if not market_open:
                msg += f"\n{TelegramFormatter.WARNING} <i>Market buka Senin 05:00 WIB</i>"

            return msg

        async def test_buy_handler():
            # Check prerequisites
            if not self.mt5.is_autotrading_enabled():
                return f"{TelegramFormatter.RED} AutoTrading DISABLED di MT5!"

            now = datetime.now(timezone.utc)
            if not self._is_market_open(now):
                return f"{TelegramFormatter.RED} Market CLOSED (weekend)"

            try:
                tick = self.mt5.get_tick_sync(SYMBOL)
                result = await self.mt5.place_market_order(
                    symbol=SYMBOL,
                    order_type='BUY',
                    volume=0.01,
                    sl=0,
                    tp=0,
                    comment='TEST_BUY_v64',
                    magic=20250131
                )
                if result and result.get('ticket'):
                    return (
                        f"{TelegramFormatter.CHECK} <b>TEST BUY Success!</b>\n\n"
                        f"Ticket: <code>#{result.get('ticket')}</code>\n"
                        f"Price: {tick.get('ask'):.5f}\n"
                        f"Volume: 0.01 lot"
                    )
                else:
                    return f"{TelegramFormatter.CROSS} TEST BUY Failed: {self.mt5.get_last_error()}"
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def test_sell_handler():
            # Check prerequisites
            if not self.mt5.is_autotrading_enabled():
                return f"{TelegramFormatter.RED} AutoTrading DISABLED di MT5!"

            now = datetime.now(timezone.utc)
            if not self._is_market_open(now):
                return f"{TelegramFormatter.RED} Market CLOSED (weekend)"

            try:
                tick = self.mt5.get_tick_sync(SYMBOL)
                result = await self.mt5.place_market_order(
                    symbol=SYMBOL,
                    order_type='SELL',
                    volume=0.01,
                    sl=0,
                    tp=0,
                    comment='TEST_SELL_v64',
                    magic=20250131
                )
                if result and result.get('ticket'):
                    return (
                        f"{TelegramFormatter.CHECK} <b>TEST SELL Success!</b>\n\n"
                        f"Ticket: <code>#{result.get('ticket')}</code>\n"
                        f"Price: {tick.get('bid'):.5f}\n"
                        f"Volume: 0.01 lot"
                    )
                else:
                    return f"{TelegramFormatter.CROSS} TEST SELL Failed: {self.mt5.get_last_error()}"
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def close_all_handler():
            positions = self.mt5.get_positions_sync(SYMBOL)
            if not positions:
                return f"{TelegramFormatter.MEMO} No positions to close"

            closed = 0
            failed = 0
            for p in positions:
                try:
                    if await self.mt5.close_position(p['ticket']):
                        closed += 1
                    else:
                        failed += 1
                except:
                    failed += 1

            return (
                f"{TelegramFormatter.CHECK} <b>Close All Complete</b>\n\n"
                f"Closed: {closed}\n"
                f"Failed: {failed}"
            )

        async def help_handler():
            """Show all available commands"""
            msg = TelegramFormatter.tree_header("H1 v6.4 GBPUSD Commands", "📖")

            msg += TelegramFormatter.tree_section("Information", "ℹ️")
            msg += f"  /status - System status & market conditions\n"
            msg += f"  /balance - Account balance & equity\n"
            msg += f"  /positions - View open positions\n"
            msg += f"  /regime - Market regime (Bull/Bear/Side)\n"
            msg += f"  /pois - Active Order Blocks & FVG\n"
            msg += f"  /activity - Activity filter status\n"
            msg += f"  /mode - Current trading mode\n"
            msg += f"  /market - Detailed market analysis\n"
            msg += f"  /autotrading - MT5 AutoTrading status\n"

            msg += TelegramFormatter.tree_section("Control", TelegramFormatter.GEAR)
            msg += f"  /pause - Pause auto trading\n"
            msg += f"  /resume - Resume auto trading\n"

            msg += TelegramFormatter.tree_section("Testing", "🧪")
            msg += f"  /test_buy - Test BUY (0.01 lot)\n"
            msg += f"  /test_sell - Test SELL (0.01 lot)\n"
            msg += f"  /close_all - Close all positions\n"

            msg += f"\n{TelegramFormatter.BRAIN} <i>Strategy: Dual-Layer Quality Filter</i>"

            return msg

        # Register all callbacks
        self.telegram.on_status = status_handler
        self.telegram.on_balance = balance_handler
        self.telegram.on_pause = pause_handler
        self.telegram.on_resume = resume_handler
        self.telegram.on_positions = positions_handler
        self.telegram.on_regime = regime_handler
        self.telegram.on_pois = pois_handler
        self.telegram.on_activity = activity_handler
        self.telegram.on_mode = mode_handler
        self.telegram.on_market = market_handler
        self.telegram.on_autotrading = autotrading_handler
        self.telegram.on_test_buy = test_buy_handler
        self.telegram.on_test_sell = test_sell_handler
        self.telegram.on_close_all = close_all_handler
        self.telegram.on_help = help_handler

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
            autotrading_enabled = self.mt5.is_autotrading_enabled()
            now = datetime.now(timezone.utc)
            market_open = self._is_market_open(now)

            autotrading = f"{TelegramFormatter.GREEN} ENABLED" if autotrading_enabled else f"{TelegramFormatter.RED} DISABLED"
            market_status = f"{TelegramFormatter.GREEN} OPEN" if market_open else f"{TelegramFormatter.RED} CLOSED"

            msg = (
                f"{TelegramFormatter.ROCKET} <b>H1 v6.4 GBPUSD Bot Started</b>\n\n"
                f"{TelegramFormatter.BRANCH} Mode: <code>{self.mode.upper()}</code>\n"
                f"{TelegramFormatter.BRANCH} Interval: <code>{self.interval}s</code>\n"
                f"{TelegramFormatter.BRANCH} AutoTrading: {autotrading}\n"
                f"{TelegramFormatter.BRANCH} Market: {market_status}\n"
                f"{TelegramFormatter.LAST} Strategy: <code>Dual-Layer Quality</code>\n"
            )

            # Add warnings
            warnings = []
            if not autotrading_enabled:
                warnings.append("Enable AutoTrading di MT5!")
            if not market_open:
                warnings.append("Market tutup (weekend)")

            if warnings:
                msg += f"\n{TelegramFormatter.WARNING} <b>Warning:</b>\n"
                for w in warnings:
                    msg += f"  • {w}\n"

            msg += "\n<b>Commands:</b> /help for full list\n"
            msg += "<code>/status /balance /positions /regime</code>\n"
            msg += "<code>/pois /activity /mode /market</code>\n"
            msg += "<code>/test_buy /test_sell /close_all</code>\n"
            msg += "<code>/pause /resume /autotrading</code>"

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

                # Run trading cycle (skip if paused)
                if not self.paused:
                    await self.executor.run_cycle()
                else:
                    logger.debug("Trading paused, skipping cycle")

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
