"""
SURGE-WSI GBPUSD H1 Quad-Layer Strategy
========================================

QUAD-LAYER Quality Filter for ZERO losing months:
- Layer 1: Monthly profile from market analysis (tradeable %)
- Layer 2: Real-time technical indicators (ATR stability, efficiency, ADX)
- Layer 3: Intra-month dynamic risk (consecutive losses, monthly P&L)
- Layer 4: Pattern-Based Choppy Market Detector (rolling WR, direction tracking)

Usage:
    python main.py [--demo] [--live] [--interval 300]

Telegram Commands:
    /status     - System status & all layer info
    /balance    - Account balance & equity
    /positions  - View open positions
    /regime     - Current market regime (Bullish/Bearish/Sideways)
    /pois       - Active Points of Interest (Order Blocks, FVG)
    /activity   - Intelligent Activity Filter status
    /mode       - Current trading mode
    /market     - Market Analysis (detailed)
    /layers     - View all 4 layers status
    /pause      - Pause auto trading
    /resume     - Resume auto trading
    /close_all  - Close all open positions
    /test_buy   - Test BUY order (0.01 lot)
    /test_sell  - Test SELL order (0.01 lot)
    /autotrading - Check MT5 AutoTrading status
    /help       - Show all available commands

Backtest Results (Jan 2025 - Jan 2026):
    - 102 trades, 42.2% WR, PF 3.57
    - +$12,888.80 profit (+25.8% return on $50K)
    - ZERO losing months (0/13)

MT5 Account: MetaQuotes-Demo (NOT Finex)

Author: SURIOTA Team
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Strategy directory
STRATEGY_DIR = Path(__file__).parent
# Project root (2 levels up)
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.data.mt5_connector import MT5Connector

# Import from local executor.py
from executor import (
    H1V64GBPUSDExecutor, SYMBOL, TIMEFRAME, Regime,
    MONTHLY_TRADEABLE_PCT, HOUR_MULTIPLIERS, DAY_MULTIPLIERS,
    PATTERN_FILTER_ENABLED, WARMUP_TRADES, ROLLING_WINDOW,
    ROLLING_WR_HALT, ROLLING_WR_CAUTION,
    # Config for /config command
    RISK_PERCENT, SL_ATR_MULT, TP_RATIO,
    BASE_QUALITY, MIN_QUALITY_GOOD, MAX_QUALITY_BAD
)
from src.utils.telegram import TelegramNotifier, TelegramFormatter
import MetaTrader5 as mt5
import socket
import subprocess

# ============================================================
# CONFIGURATION
# ============================================================
REQUIRED_BROKER = "MetaQuotes"  # Must contain this string
FORBIDDEN_BROKER = "Finex"      # Must NOT contain this string
METAQUOTES_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Service ports
POSTGRES_PORT = 5434
REDIS_PORT = 6381


# ============================================================
# STARTUP CHECKS
# ============================================================
class StartupChecker:
    """Comprehensive startup checks before bot runs"""

    def __init__(self):
        self.checks = {}
        self.all_passed = True
        self.critical_failed = False

    def check_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def check_docker_running(self) -> bool:
        """Check if Docker is running"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def check_docker_containers(self) -> dict:
        """Check Docker container status"""
        containers = {}
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            name = parts[0]
                            status = parts[1]
                            containers[name] = 'Up' in status
        except Exception:
            pass
        return containers

    def check_postgres(self) -> tuple:
        """Check PostgreSQL/TimescaleDB connection"""
        is_open = self.check_port('localhost', POSTGRES_PORT)
        return is_open, f"localhost:{POSTGRES_PORT}"

    def check_redis(self) -> tuple:
        """Check Redis connection"""
        is_open = self.check_port('localhost', REDIS_PORT)
        return is_open, f"localhost:{REDIS_PORT}"

    def check_mt5_terminal(self) -> tuple:
        """Check if MT5 terminal is running"""
        try:
            if mt5.initialize():
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    return True, terminal_info.name
                return True, "Connected"
            return False, "Not running"
        except Exception as e:
            return False, str(e)

    def check_mt5_autotrading(self) -> tuple:
        """Check if AutoTrading is enabled in MT5"""
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info:
                # trade_allowed indicates if AutoTrading is enabled
                if terminal_info.trade_allowed:
                    return True, "Enabled"
                else:
                    return False, "DISABLED - Press Ctrl+E in MT5!"
            return False, "Cannot check"
        except Exception:
            return False, "Cannot check"

    def check_mt5_account(self) -> tuple:
        """Check MT5 account info"""
        try:
            account_info = mt5.account_info()
            if account_info:
                # Check if it's Finex
                if FORBIDDEN_BROKER.lower() in account_info.server.lower():
                    return False, f"WRONG BROKER: {account_info.server}"
                return True, f"{account_info.login} @ {account_info.server}"
            return False, "No account"
        except Exception:
            return False, "Cannot check"

    def check_telegram_config(self) -> tuple:
        """Check if Telegram is configured"""
        token = config.telegram.bot_token
        chat_id = config.telegram.chat_id
        if token and chat_id:
            return True, f"Chat: {chat_id[:10]}..."
        elif not token:
            return False, "Missing TELEGRAM_BOT_TOKEN"
        else:
            return False, "Missing TELEGRAM_CHAT_ID"

    def run_all_checks(self) -> dict:
        """Run all startup checks and return results"""
        print("\n" + "="*60)
        print("STARTUP CHECKS")
        print("="*60 + "\n")

        results = {}

        # 1. Docker
        print("Checking Docker...")
        docker_ok = self.check_docker_running()
        results['docker'] = {
            'name': 'Docker Engine',
            'status': docker_ok,
            'detail': 'Running' if docker_ok else 'Not running',
            'critical': False
        }

        # 2. PostgreSQL/TimescaleDB
        print("Checking PostgreSQL...")
        pg_ok, pg_detail = self.check_postgres()
        results['postgres'] = {
            'name': 'PostgreSQL/TimescaleDB',
            'status': pg_ok,
            'detail': pg_detail if pg_ok else f"Not responding on port {POSTGRES_PORT}",
            'critical': True
        }

        # 3. Redis
        print("Checking Redis...")
        redis_ok, redis_detail = self.check_redis()
        results['redis'] = {
            'name': 'Redis Cache',
            'status': redis_ok,
            'detail': redis_detail if redis_ok else f"Not responding on port {REDIS_PORT}",
            'critical': False  # Can work without Redis
        }

        # 4. MT5 Terminal
        print("Checking MT5 Terminal...")
        mt5_ok, mt5_detail = self.check_mt5_terminal()
        results['mt5_terminal'] = {
            'name': 'MT5 Terminal',
            'status': mt5_ok,
            'detail': mt5_detail,
            'critical': True
        }

        # 5. MT5 Account (broker check)
        if mt5_ok:
            print("Checking MT5 Account...")
            acc_ok, acc_detail = self.check_mt5_account()
            results['mt5_account'] = {
                'name': 'MT5 Account',
                'status': acc_ok,
                'detail': acc_detail,
                'critical': True
            }

            # 6. AutoTrading
            print("Checking AutoTrading...")
            at_ok, at_detail = self.check_mt5_autotrading()
            results['autotrading'] = {
                'name': 'MT5 AutoTrading',
                'status': at_ok,
                'detail': at_detail,
                'critical': True
            }

        # 7. Telegram
        print("Checking Telegram...")
        tg_ok, tg_detail = self.check_telegram_config()
        results['telegram'] = {
            'name': 'Telegram Bot',
            'status': tg_ok,
            'detail': tg_detail,
            'critical': False
        }

        # Print results table
        print("\n" + "-"*60)
        print(f"{'Service':<25} {'Status':<10} {'Details'}")
        print("-"*60)

        all_ok = True
        critical_fail = False

        for key, check in results.items():
            status_icon = "[OK]" if check['status'] else "[FAIL]"
            status_color = "" if check['status'] else "(!)"

            print(f"{check['name']:<25} {status_icon:<10} {check['detail']}")

            if not check['status']:
                all_ok = False
                if check['critical']:
                    critical_fail = True

        print("-"*60)

        # Summary
        if all_ok:
            print("\n[OK] All checks passed!")
        elif critical_fail:
            print("\n[!] CRITICAL checks failed - cannot start bot")
        else:
            print("\n[?] Some non-critical checks failed - bot may work with reduced functionality")

        print("="*60 + "\n")

        self.checks = results
        self.all_passed = all_ok
        self.critical_failed = critical_fail

        return results

    def get_failed_critical(self) -> list:
        """Get list of failed critical checks"""
        failed = []
        for key, check in self.checks.items():
            if check['critical'] and not check['status']:
                failed.append(check)
        return failed

    def print_fix_instructions(self):
        """Print instructions to fix failed checks"""
        failed = self.get_failed_critical()
        if not failed:
            return

        print("\n" + "="*60)
        print("HOW TO FIX")
        print("="*60 + "\n")

        for check in failed:
            name = check['name']

            if 'PostgreSQL' in name:
                print(f"[{name}]")
                print("  Run: docker-compose up -d")
                print("  Or:  docker start surge-timescaledb")
                print()

            elif 'Redis' in name:
                print(f"[{name}]")
                print("  Run: docker-compose up -d")
                print("  Or:  docker start surge-redis")
                print()

            elif 'MT5 Terminal' in name:
                print(f"[{name}]")
                print("  1. Open MetaTrader 5 (NOT Finex!)")
                print("  2. Login to MetaQuotes-Demo account")
                print()

            elif 'MT5 Account' in name:
                print(f"[{name}]")
                print("  1. Close Finex MT5 terminal")
                print("  2. Open MetaQuotes MT5 terminal")
                print("  3. Login to MetaQuotes-Demo account")
                print(f"  Path: {METAQUOTES_TERMINAL_PATH}")
                print()

            elif 'AutoTrading' in name:
                print(f"[{name}]")
                print("  1. In MT5, press Ctrl+E to enable AutoTrading")
                print("  2. Or click the 'AutoTrading' button in toolbar")
                print("  3. Make sure the button shows GREEN, not RED")
                print()

        print("="*60 + "\n")


def ensure_metaquotes_connection() -> bool:
    """
    Ensure we're connected to MetaQuotes-Demo, NOT Finex.

    Returns:
        True if connected to MetaQuotes, False otherwise
    """
    # First, try to connect to whatever is running
    if mt5.initialize():
        account_info = mt5.account_info()
        terminal_info = mt5.terminal_info()

        if account_info and terminal_info:
            server = account_info.server
            terminal_name = terminal_info.name

            logger.info(f"Detected: {terminal_name} - Account: {account_info.login} @ {server}")

            # Check if it's Finex (forbidden)
            if FORBIDDEN_BROKER.lower() in server.lower() or FORBIDDEN_BROKER.lower() in terminal_name.lower():
                logger.warning(f"[!] Finex detected! This strategy requires MetaQuotes-Demo.")
                logger.warning(f"[!] Disconnecting from Finex and trying MetaQuotes...")
                mt5.shutdown()

                # Try to connect to MetaQuotes specifically
                if Path(METAQUOTES_TERMINAL_PATH).exists():
                    logger.info(f"Attempting to connect to MetaQuotes: {METAQUOTES_TERMINAL_PATH}")
                    if mt5.initialize(path=METAQUOTES_TERMINAL_PATH):
                        account_info = mt5.account_info()
                        if account_info:
                            # Verify it's now MetaQuotes
                            if FORBIDDEN_BROKER.lower() not in account_info.server.lower():
                                logger.info(f"[OK] Connected to MetaQuotes: {account_info.login} @ {account_info.server}")
                                return True
                            else:
                                logger.error(f"[!] Still got Finex after switching!")
                                mt5.shutdown()
                                return False
                    else:
                        logger.error(f"[!] Failed to connect to MetaQuotes terminal")
                        logger.error(f"[!] Please open MetaQuotes MT5 and login to MetaQuotes-Demo account")
                        return False
                else:
                    logger.error(f"[!] MetaQuotes terminal not found at: {METAQUOTES_TERMINAL_PATH}")
                    logger.error(f"[!] Please install MetaQuotes MT5 or close Finex and open MetaQuotes manually")
                    return False

            # Check if it's MetaQuotes (required) or at least not Finex
            if REQUIRED_BROKER.lower() in server.lower() or REQUIRED_BROKER.lower() in terminal_name.lower():
                logger.info(f"[OK] MetaQuotes connection verified")
                return True

            # Not Finex, not explicitly MetaQuotes - warn but allow
            logger.warning(f"[?] Unknown broker: {server}")
            logger.warning(f"[?] This strategy is optimized for MetaQuotes-Demo")
            logger.warning(f"[?] Proceeding anyway, but results may differ from backtest")
            return True

    # Initial connection failed, try MetaQuotes directly
    logger.warning("No MT5 terminal detected, trying MetaQuotes...")
    if Path(METAQUOTES_TERMINAL_PATH).exists():
        if mt5.initialize(path=METAQUOTES_TERMINAL_PATH):
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"[OK] Connected to MetaQuotes: {account_info.login} @ {account_info.server}")
                return True

    logger.error("[!] Failed to connect to any MT5 terminal")
    logger.error("[!] Please open MetaQuotes MT5 and login to MetaQuotes-Demo account")
    return False


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
LOG_DIR = STRATEGY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    str(LOG_DIR / "quadlayer_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)


class TradingBot:
    """Main trading bot for H1 v6.4 GBPUSD with QUAD-LAYER filter"""

    def __init__(self, mode: str = 'demo', interval: int = 300):
        self.mode = mode
        self.interval = interval
        self.running = False
        self.paused = False
        self.executor = None
        self.telegram = None
        self.db = None
        self.mt5 = None
        self.loop_count = 0
        self._last_hourly_status = None
        self._cached_regime = None
        self._cached_pois = []
        self._cached_market_condition = None

    async def initialize(self):
        """Initialize all components"""

        # ============================================================
        # STARTUP CHECKS - Run before anything else
        # ============================================================
        checker = StartupChecker()
        checker.run_all_checks()

        if checker.critical_failed:
            checker.print_fix_instructions()
            raise RuntimeError(
                "\n[!] Critical startup checks failed. Please fix the issues above and try again."
            )

        # Warn about non-critical issues
        if not checker.all_passed:
            logger.warning("Some non-critical checks failed - continuing with reduced functionality")

        logger.info(f"Initializing H1 v6.4 GBPUSD Trading Bot (QUAD-LAYER)")
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

        # ============================================================
        # METAQUOTES ENFORCEMENT
        # This strategy REQUIRES MetaQuotes-Demo, NOT Finex
        # ============================================================
        logger.info("Verifying MetaQuotes connection...")

        if not ensure_metaquotes_connection():
            raise RuntimeError(
                "\n" + "="*60 + "\n"
                "[ERROR] This strategy requires MetaQuotes-Demo account!\n"
                "\n"
                "Please do ONE of the following:\n"
                "1. Close Finex MT5 and open MetaQuotes MT5\n"
                "2. Login to MetaQuotes-Demo account\n"
                "3. Install MetaQuotes MT5 from metaquotes.net\n"
                "\n"
                "MetaQuotes Terminal Path:\n"
                f"  {METAQUOTES_TERMINAL_PATH}\n"
                + "="*60
            )

        # Create MT5Connector wrapper (connection already established)
        mt5_connector = MT5Connector(
            login=config.mt5.login,
            password=config.mt5.password,
            server=config.mt5.server,
            terminal_path=config.mt5.terminal_path
        )
        mt5_connector.connected = True  # Already connected by ensure_metaquotes_connection()

        self.mt5 = mt5_connector

        # Final AutoTrading check with clear warning
        if mt5_connector.is_autotrading_enabled():
            logger.info("AutoTrading: ENABLED")
        else:
            logger.error("="*60)
            logger.error("[!] AUTOTRADING IS DISABLED!")
            logger.error("[!] Bot will NOT be able to execute trades!")
            logger.error("[!] Press Ctrl+E in MT5 to enable AutoTrading")
            logger.error("="*60)
            # Don't raise error, just warn - user might want to monitor only

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

        # Initialize executor with QUAD-LAYER filter
        self.executor = H1V64GBPUSDExecutor(
            broker_client=broker,
            db_handler=self.db,
            telegram_bot=self.telegram,
            mt5_connector=self.mt5
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

            # Get Layer 3 & 4 status
            l3_status = status.get('layer3_status', {})
            l4_status = status.get('layer4_status', {})

            msg = TelegramFormatter.tree_header("H1 v6.4 GBPUSD Status", TelegramFormatter.CHART)
            msg += TelegramFormatter.tree_section("System", TelegramFormatter.GEAR)
            msg += TelegramFormatter.tree_item("Symbol", status['symbol'])
            msg += TelegramFormatter.tree_item("Strategy", "QUAD-LAYER Quality")
            msg += TelegramFormatter.tree_item("Has Position", "Yes" if status['has_position'] else "No")
            msg += TelegramFormatter.tree_item("Last Signal", status['last_signal'] or 'None', last=True)

            msg += TelegramFormatter.tree_section("Market (L1+L2)", TelegramFormatter.BRAIN)
            msg += TelegramFormatter.tree_item("Month", now.strftime('%Y-%m'))
            msg += TelegramFormatter.tree_item("Tradeable", tradeable)
            msg += TelegramFormatter.tree_item("Quality Adj", f"+{adj}")
            msg += TelegramFormatter.tree_item("Min Quality", f"{60 + adj}-{80 + adj}", last=True)

            msg += TelegramFormatter.tree_section("Risk (L3)", "🛡️")
            msg += TelegramFormatter.tree_item("Monthly P&L", f"${l3_status.get('monthly_pnl', 0):.2f}")
            msg += TelegramFormatter.tree_item("Consec Losses", str(l3_status.get('consecutive_losses', 0)))
            month_ok = TelegramFormatter.CHECK if not l3_status.get('month_stopped') else TelegramFormatter.RED
            msg += TelegramFormatter.tree_item("Month Active", month_ok, last=True)

            msg += TelegramFormatter.tree_section("Pattern (L4)", "🔍")
            rolling_wr = l4_status.get('rolling_wr', 1.0)
            msg += TelegramFormatter.tree_item("Trade History", f"{l4_status.get('total_history', 0)}/{WARMUP_TRADES}")
            msg += TelegramFormatter.tree_item("Rolling WR", f"{rolling_wr:.0%}")
            if l4_status.get('is_halted'):
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.RED} HALTED", last=True)
            elif l4_status.get('in_recovery'):
                msg += TelegramFormatter.tree_item("Status", f"🟡 RECOVERY", last=True)
            else:
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.GREEN} OK", last=True)

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
                for i, poi in enumerate(pois[-5:]):
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

            hour_mult = HOUR_MULTIPLIERS.get(hour, 0.0)
            day_mult = DAY_MULTIPLIERS.get(day, 0.0)

            activity_score = hour_mult * day_mult * 100

            in_kz, kz_name = self.executor.is_kill_zone(now)

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

            mode_emoji = TelegramFormatter.CHECK if self.mode == 'demo' else TelegramFormatter.WARNING

            msg = TelegramFormatter.tree_header("Trading Mode", TelegramFormatter.GEAR)
            msg += TelegramFormatter.tree_item("Mode", f"{mode_emoji} {self.mode.upper()}")
            msg += TelegramFormatter.tree_item("Bot Status", f"{TelegramFormatter.GREEN} RUNNING" if self.running and not self.paused else f"{TelegramFormatter.RED} PAUSED")
            msg += TelegramFormatter.tree_item("AutoTrading", f"{TelegramFormatter.GREEN} ON" if autotrading else f"{TelegramFormatter.RED} OFF")
            msg += TelegramFormatter.tree_item("Market", f"{TelegramFormatter.GREEN} OPEN" if market_open else f"{TelegramFormatter.RED} CLOSED")
            msg += TelegramFormatter.tree_item("Interval", f"{self.interval}s")
            msg += TelegramFormatter.tree_item("Loop Count", str(self.loop_count), last=True)

            return msg

        async def layers_handler():
            """View all 4 layers status"""
            status = self.executor.get_status()
            now = datetime.now(timezone.utc)

            l3_status = status.get('layer3_status', {})
            l4_status = status.get('layer4_status', {})

            # Layer 1: Monthly Profile
            adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)
            key = (now.year, now.month)
            tradeable = MONTHLY_TRADEABLE_PCT.get(key, 70)

            msg = TelegramFormatter.tree_header("QUAD-LAYER Status", "🛡️")

            # Layer 1
            msg += TelegramFormatter.tree_section("Layer 1: Monthly Profile", "📅")
            msg += TelegramFormatter.tree_item("Month", now.strftime('%Y-%m'))
            msg += TelegramFormatter.tree_item("Tradeable", f"{tradeable}%")
            msg += TelegramFormatter.tree_item("Quality Adj", f"+{adj}", last=True)

            # Layer 2
            msg += TelegramFormatter.tree_section("Layer 2: Technical", "📐")
            msg += TelegramFormatter.tree_item("ATR Range", "8-30 pips")
            msg += TelegramFormatter.tree_item("Efficiency", ">8%")
            msg += TelegramFormatter.tree_item("Trend (ADX)", ">25", last=True)

            # Layer 3
            msg += TelegramFormatter.tree_section("Layer 3: Intra-Month", "📊")
            msg += TelegramFormatter.tree_item("Monthly P&L", f"${l3_status.get('monthly_pnl', 0):.2f}")
            msg += TelegramFormatter.tree_item("Peak", f"${l3_status.get('monthly_peak', 0):.2f}")
            msg += TelegramFormatter.tree_item("Consec Losses", str(l3_status.get('consecutive_losses', 0)))

            if l3_status.get('month_stopped'):
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.RED} MONTH STOPPED", last=True)
            elif l3_status.get('day_stopped'):
                msg += TelegramFormatter.tree_item("Status", f"🟡 DAY STOPPED", last=True)
            else:
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.GREEN} OK", last=True)

            # Layer 4
            msg += TelegramFormatter.tree_section("Layer 4: Pattern Filter", "🔍")
            msg += TelegramFormatter.tree_item("Enabled", f"{TelegramFormatter.CHECK}" if PATTERN_FILTER_ENABLED else f"{TelegramFormatter.CROSS}")
            msg += TelegramFormatter.tree_item("Trade History", f"{l4_status.get('total_history', 0)}")
            msg += TelegramFormatter.tree_item("Warmup", f"{WARMUP_TRADES} trades")
            msg += TelegramFormatter.tree_item("Rolling WR", f"{l4_status.get('rolling_wr', 1.0):.0%}")
            msg += TelegramFormatter.tree_item("BUY WR", f"{l4_status.get('buy_wr', 1.0):.0%}")
            msg += TelegramFormatter.tree_item("SELL WR", f"{l4_status.get('sell_wr', 1.0):.0%}")

            if l4_status.get('is_halted'):
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.RED} HALTED", last=True)
            elif l4_status.get('in_recovery'):
                recovery_wins = l4_status.get('recovery_wins', 0)
                msg += TelegramFormatter.tree_item("Status", f"🟡 RECOVERY ({recovery_wins}/1)", last=True)
            elif l4_status.get('both_fail'):
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.RED} CHOPPY DETECTED", last=True)
            else:
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.GREEN} OK", last=True)

            return msg

        async def market_handler():
            """Detailed market analysis"""
            try:
                now = datetime.now(timezone.utc)

                df = await self.executor.get_ohlcv_data(SYMBOL, TIMEFRAME, 100)
                if df is None or df.empty:
                    return f"{TelegramFormatter.CROSS} Failed to get market data"

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

                atr_series = await self.executor.calculate_atr(df, col_map)
                current_atr = atr_series.iloc[-1]

                regime, strength = self.executor.detect_regime(df, col_map)

                market_cond = self.executor.risk_scorer.assess_market_condition(
                    df, col_map, atr_series, now
                )
                self._cached_market_condition = market_cond

                tick = self.mt5.get_tick_sync(SYMBOL)
                price = tick.get('bid', 0) if tick else df[col_map['close']].iloc[-1]
                spread = tick.get('spread', 0) if tick else 0

                key = (now.year, now.month)
                tradeable_pct = MONTHLY_TRADEABLE_PCT.get(key, 70)

                if regime == Regime.BULLISH:
                    regime_str = f"{TelegramFormatter.UP} BULLISH"
                elif regime == Regime.BEARISH:
                    regime_str = f"{TelegramFormatter.DOWN} BEARISH"
                else:
                    regime_str = "⚖️ SIDEWAYS"

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

        async def trades_handler():
            """View recent trade history from Layer 4"""
            l4_status = self.executor.pattern_filter.get_stats()
            history = self.executor.pattern_filter.trade_history

            if not history:
                return f"{TelegramFormatter.MEMO} <b>No trade history yet</b>\n\nWarmup: {WARMUP_TRADES} trades needed"

            msg = TelegramFormatter.tree_header(f"Trade History ({len(history)})", "📜")

            # Show last 10 trades
            recent = history[-10:]
            wins = sum(1 for _, pnl, _ in recent if pnl > 0)
            losses = len(recent) - wins

            msg += TelegramFormatter.tree_section("Recent Trades", TelegramFormatter.CHART)
            for i, (direction, pnl, ts) in enumerate(reversed(recent)):
                emoji = TelegramFormatter.UP if direction == 'BUY' else TelegramFormatter.DOWN
                pnl_emoji = TelegramFormatter.CHECK if pnl > 0 else TelegramFormatter.CROSS
                time_str = ts.strftime('%m/%d %H:%M') if hasattr(ts, 'strftime') else str(ts)[:10]
                msg += f"  {emoji} {direction} {pnl_emoji} ${pnl:+.2f} ({time_str})\n"

            msg += TelegramFormatter.tree_section("Statistics", "📊")
            msg += TelegramFormatter.tree_item("Total Trades", str(len(history)))
            msg += TelegramFormatter.tree_item("Recent W/L", f"{wins}/{losses}")
            msg += TelegramFormatter.tree_item("Rolling WR", f"{l4_status.get('rolling_wr', 0):.0%}")
            msg += TelegramFormatter.tree_item("BUY WR", f"{l4_status.get('buy_wr', 0):.0%}")
            msg += TelegramFormatter.tree_item("SELL WR", f"{l4_status.get('sell_wr', 0):.0%}", last=True)

            return msg

        async def stats_handler():
            """View trading statistics"""
            status = self.executor.get_status()
            l3_status = status.get('layer3_status', {})
            l4_status = status.get('layer4_status', {})
            history = self.executor.pattern_filter.trade_history

            msg = TelegramFormatter.tree_header("Trading Statistics", "📊")

            if history:
                total = len(history)
                wins = sum(1 for _, pnl, _ in history if pnl > 0)
                losses = total - wins
                total_pnl = sum(pnl for _, pnl, _ in history)
                avg_win = sum(pnl for _, pnl, _ in history if pnl > 0) / wins if wins > 0 else 0
                avg_loss = abs(sum(pnl for _, pnl, _ in history if pnl < 0)) / losses if losses > 0 else 0

                msg += TelegramFormatter.tree_section("Overall", TelegramFormatter.CHART)
                msg += TelegramFormatter.tree_item("Total Trades", str(total))
                msg += TelegramFormatter.tree_item("Wins/Losses", f"{wins}/{losses}")
                msg += TelegramFormatter.tree_item("Win Rate", f"{wins/total*100:.1f}%")
                pnl_emoji = TelegramFormatter.CHECK if total_pnl >= 0 else TelegramFormatter.CROSS
                msg += TelegramFormatter.tree_item("Total P&L", f"{pnl_emoji} ${total_pnl:+,.2f}")
                msg += TelegramFormatter.tree_item("Avg Win", f"${avg_win:,.2f}")
                msg += TelegramFormatter.tree_item("Avg Loss", f"${avg_loss:,.2f}", last=True)

                # By direction
                buy_trades = [(d, p) for d, p, _ in history if d == 'BUY']
                sell_trades = [(d, p) for d, p, _ in history if d == 'SELL']

                msg += TelegramFormatter.tree_section("By Direction", "📈")
                if buy_trades:
                    buy_wins = sum(1 for _, p in buy_trades if p > 0)
                    buy_pnl = sum(p for _, p in buy_trades)
                    msg += TelegramFormatter.tree_item("BUY", f"{buy_wins}/{len(buy_trades)} (${buy_pnl:+.2f})")
                if sell_trades:
                    sell_wins = sum(1 for _, p in sell_trades if p > 0)
                    sell_pnl = sum(p for _, p in sell_trades)
                    msg += TelegramFormatter.tree_item("SELL", f"{sell_wins}/{len(sell_trades)} (${sell_pnl:+.2f})", last=True)
            else:
                msg += f"\n{TelegramFormatter.MEMO} No trade history yet"

            msg += TelegramFormatter.tree_section("Current Month", "📅")
            msg += TelegramFormatter.tree_item("Monthly P&L", f"${l3_status.get('monthly_pnl', 0):.2f}")
            msg += TelegramFormatter.tree_item("Peak", f"${l3_status.get('monthly_peak', 0):.2f}")
            msg += TelegramFormatter.tree_item("Consec Losses", str(l3_status.get('consecutive_losses', 0)), last=True)

            return msg

        async def daily_handler():
            """View today's trading summary"""
            now = datetime.now(timezone.utc)
            today = now.date()

            # Get today's trades from history
            history = self.executor.pattern_filter.trade_history
            today_trades = [(d, p, t) for d, p, t in history
                          if hasattr(t, 'date') and t.date() == today]

            # Get account info
            info = self.mt5.get_account_info_sync()
            balance = info.get('balance', 0) if info else 0
            profit = info.get('profit', 0) if info else 0

            msg = TelegramFormatter.tree_header(f"Daily Summary - {today}", "📅")

            msg += TelegramFormatter.tree_section("Account", TelegramFormatter.MONEY)
            msg += TelegramFormatter.tree_item("Balance", f"${balance:,.2f}")
            pnl_emoji = TelegramFormatter.CHECK if profit >= 0 else TelegramFormatter.CROSS
            msg += TelegramFormatter.tree_item("Floating", f"{pnl_emoji} ${profit:+,.2f}", last=True)

            if today_trades:
                wins = sum(1 for _, p, _ in today_trades if p > 0)
                total_pnl = sum(p for _, p, _ in today_trades)

                msg += TelegramFormatter.tree_section("Today's Trades", TelegramFormatter.CHART)
                msg += TelegramFormatter.tree_item("Trades", str(len(today_trades)))
                msg += TelegramFormatter.tree_item("Wins/Losses", f"{wins}/{len(today_trades)-wins}")
                pnl_emoji = TelegramFormatter.CHECK if total_pnl >= 0 else TelegramFormatter.CROSS
                msg += TelegramFormatter.tree_item("P&L", f"{pnl_emoji} ${total_pnl:+,.2f}", last=True)

                # List trades
                msg += "\n<b>Trades:</b>\n"
                for d, p, t in today_trades:
                    emoji = TelegramFormatter.UP if d == 'BUY' else TelegramFormatter.DOWN
                    pnl_emoji = TelegramFormatter.CHECK if p > 0 else TelegramFormatter.CROSS
                    time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else '--:--'
                    msg += f"  {emoji} {d} {pnl_emoji} ${p:+.2f} ({time_str})\n"
            else:
                msg += f"\n{TelegramFormatter.MEMO} No trades today"

            # Bot status
            msg += TelegramFormatter.tree_section("Bot Status", TelegramFormatter.GEAR)
            msg += TelegramFormatter.tree_item("Loops", str(self.loop_count))
            msg += TelegramFormatter.tree_item("Paused", "Yes" if self.paused else "No", last=True)

            return msg

        async def monthly_handler():
            """View monthly trading summary"""
            now = datetime.now(timezone.utc)
            l3_status = self.executor.intra_month_manager.get_status()

            # Get monthly trades from history
            history = self.executor.pattern_filter.trade_history
            monthly_trades = [(d, p, t) for d, p, t in history
                            if hasattr(t, 'month') and t.month == now.month and t.year == now.year]

            msg = TelegramFormatter.tree_header(f"Monthly Summary - {now.strftime('%Y-%m')}", "📅")

            # Monthly profile
            key = (now.year, now.month)
            tradeable = MONTHLY_TRADEABLE_PCT.get(key, 70)
            adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)

            msg += TelegramFormatter.tree_section("Month Profile", "📊")
            msg += TelegramFormatter.tree_item("Tradeable", f"{tradeable}%")
            msg += TelegramFormatter.tree_item("Quality Adj", f"+{adj}")
            msg += TelegramFormatter.tree_item("Min Quality", f"{60+adj}", last=True)

            msg += TelegramFormatter.tree_section("Performance", TelegramFormatter.CHART)
            msg += TelegramFormatter.tree_item("Monthly P&L", f"${l3_status.get('monthly_pnl', 0):.2f}")
            msg += TelegramFormatter.tree_item("Peak", f"${l3_status.get('monthly_peak', 0):.2f}")

            if monthly_trades:
                wins = sum(1 for _, p, _ in monthly_trades if p > 0)
                total_pnl = sum(p for _, p, _ in monthly_trades)
                msg += TelegramFormatter.tree_item("Trades", str(len(monthly_trades)))
                msg += TelegramFormatter.tree_item("Win Rate", f"{wins/len(monthly_trades)*100:.1f}%", last=True)
            else:
                msg += TelegramFormatter.tree_item("Trades", "0", last=True)

            # Circuit breaker status
            msg += TelegramFormatter.tree_section("Protection", "🛡️")
            msg += TelegramFormatter.tree_item("Consec Losses", str(l3_status.get('consecutive_losses', 0)))
            month_ok = TelegramFormatter.CHECK if not l3_status.get('month_stopped') else TelegramFormatter.RED
            day_ok = TelegramFormatter.CHECK if not l3_status.get('day_stopped') else TelegramFormatter.RED
            msg += TelegramFormatter.tree_item("Month Active", month_ok)
            msg += TelegramFormatter.tree_item("Day Active", day_ok, last=True)

            return msg

        async def config_handler():
            """View current configuration"""
            now = datetime.now(timezone.utc)
            adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)

            msg = TelegramFormatter.tree_header("Configuration", TelegramFormatter.GEAR)

            msg += TelegramFormatter.tree_section("Trading", TelegramFormatter.CHART)
            msg += TelegramFormatter.tree_item("Symbol", SYMBOL)
            msg += TelegramFormatter.tree_item("Timeframe", TIMEFRAME)
            msg += TelegramFormatter.tree_item("Risk %", f"{RISK_PERCENT}%")
            msg += TelegramFormatter.tree_item("SL Mult", f"{SL_ATR_MULT}x ATR")
            msg += TelegramFormatter.tree_item("TP Ratio", f"{TP_RATIO}:1", last=True)

            msg += TelegramFormatter.tree_section("Quality Filter", "🎯")
            msg += TelegramFormatter.tree_item("Base Quality", str(BASE_QUALITY))
            msg += TelegramFormatter.tree_item("Good Market", str(MIN_QUALITY_GOOD))
            msg += TelegramFormatter.tree_item("Bad Market", str(MAX_QUALITY_BAD))
            msg += TelegramFormatter.tree_item("Current Adj", f"+{adj}", last=True)

            msg += TelegramFormatter.tree_section("Layer 4 Config", "🔍")
            msg += TelegramFormatter.tree_item("Enabled", f"{TelegramFormatter.CHECK}" if PATTERN_FILTER_ENABLED else f"{TelegramFormatter.CROSS}")
            msg += TelegramFormatter.tree_item("Warmup", f"{WARMUP_TRADES} trades")
            msg += TelegramFormatter.tree_item("Rolling Window", str(ROLLING_WINDOW))
            msg += TelegramFormatter.tree_item("Halt WR", f"<{ROLLING_WR_HALT*100:.0f}%")
            msg += TelegramFormatter.tree_item("Caution WR", f"<{ROLLING_WR_CAUTION*100:.0f}%", last=True)

            msg += TelegramFormatter.tree_section("Kill Zones (UTC)", TelegramFormatter.CLOCK)
            msg += TelegramFormatter.tree_item("London", "07:00-11:00")
            msg += TelegramFormatter.tree_item("New York", "13:00-17:00", last=True)

            return msg

        async def reset_l4_handler():
            """Reset Layer 4 pattern filter"""
            # Reset the filter
            self.executor.pattern_filter.trade_history = []
            self.executor.pattern_filter.is_halted = False
            self.executor.pattern_filter.in_recovery = False
            self.executor.pattern_filter.recovery_wins = 0
            self.executor.pattern_filter.halt_reason = ""

            return (
                f"{TelegramFormatter.CHECK} <b>Layer 4 Reset Complete</b>\n\n"
                f"Trade history cleared\n"
                f"Halt status: OFF\n"
                f"Recovery mode: OFF\n\n"
                f"{TelegramFormatter.WARNING} <i>Warmup period ({WARMUP_TRADES} trades) will restart</i>"
            )

        async def history_handler():
            """View MT5 trade history"""
            try:
                # Get deals from MT5
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=7)  # Last 7 days

                deals = self.mt5.get_history_deals(start, end, SYMBOL)

                if not deals:
                    return f"{TelegramFormatter.MEMO} <b>No trade history in last 7 days</b>"

                # Filter only OUT deals (closed trades)
                closed_deals = [d for d in deals if d.get('entry') == 'OUT']

                if not closed_deals:
                    return f"{TelegramFormatter.MEMO} <b>No closed trades in last 7 days</b>"

                msg = TelegramFormatter.tree_header(f"MT5 History (7 days)", "📜")

                total_pnl = 0
                wins = 0
                losses = 0

                # Show last 10 deals
                recent_deals = closed_deals[-10:] if len(closed_deals) > 10 else closed_deals

                for deal in reversed(recent_deals):
                    pnl = deal.get('profit', 0)
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    elif pnl < 0:
                        losses += 1

                    emoji = TelegramFormatter.UP if deal.get('type') == 'BUY' else TelegramFormatter.DOWN
                    pnl_emoji = TelegramFormatter.CHECK if pnl >= 0 else TelegramFormatter.CROSS
                    time_str = deal.get('time', '--')
                    if hasattr(time_str, 'strftime'):
                        time_str = time_str.strftime('%m/%d %H:%M')

                    msg += f"{emoji} {pnl_emoji} ${pnl:+.2f} | {deal.get('volume', 0)} lot | {time_str}\n"

                msg += f"\n<b>Summary:</b>\n"
                msg += f"Total: {len(closed_deals)} deals\n"
                msg += f"W/L: {wins}/{losses}\n"
                pnl_emoji = TelegramFormatter.CHECK if total_pnl >= 0 else TelegramFormatter.CROSS
                msg += f"P&L: {pnl_emoji} ${total_pnl:+,.2f}"

                return msg
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def vector_handler():
            """Get vector database status"""
            try:
                v_status = self.executor.get_vector_status()

                msg = TelegramFormatter.tree_header("Vector Database", "🔷")

                if not v_status.get('connected', False):
                    msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.RED} DISCONNECTED")
                    msg += TelegramFormatter.tree_item("Backend", v_status.get('backend', 'unknown'), last=True)
                    msg += f"\n{TelegramFormatter.WARNING} <i>Vector DB not available</i>"
                    return msg

                msg += TelegramFormatter.tree_section("Connection", TelegramFormatter.GEAR)
                msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.GREEN} CONNECTED")
                msg += TelegramFormatter.tree_item("Backend", v_status.get('backend', 'unknown'))
                msg += TelegramFormatter.tree_item("Host", f"{v_status.get('qdrant_host')}:{v_status.get('qdrant_port')}")
                redis_status = TelegramFormatter.CHECK if v_status.get('redis_connected') else TelegramFormatter.CROSS
                msg += TelegramFormatter.tree_item("Redis", redis_status, last=True)

                # Collections
                collections = v_status.get('collections', [])
                if collections:
                    msg += TelegramFormatter.tree_section("Collections", "📊")
                    for i, col in enumerate(collections):
                        is_last = i == len(collections) - 1
                        msg += TelegramFormatter.tree_item(
                            col.get('name', 'unknown'),
                            f"{col.get('vectors_count', 0)} vectors",
                            last=is_last
                        )
                else:
                    msg += f"\n{TelegramFormatter.MEMO} No collections yet"

                # Last sync
                last_sync = v_status.get('last_sync', {})
                if last_sync:
                    msg += TelegramFormatter.tree_section("Last Sync", TelegramFormatter.CLOCK)
                    for key, ts in last_sync.items():
                        msg += f"  {key}: {ts[:19]}\n"

                return msg
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def sync_handler():
            """Trigger manual vector sync"""
            try:
                msg = f"{TelegramFormatter.CLOCK} <b>Syncing to Vector DB...</b>\n"

                synced = await self.executor.sync_vector(force=True)

                if synced > 0:
                    msg = f"{TelegramFormatter.CHECK} <b>Vector Sync Complete</b>\n\n"
                    msg += f"Synced: {synced} vectors\n"
                    msg += f"Symbol: {SYMBOL}\n"
                    msg += f"Timeframe: {TIMEFRAME}"
                else:
                    msg = f"{TelegramFormatter.WARNING} <b>Sync returned 0 vectors</b>\n\n"
                    msg += "This could mean:\n"
                    msg += "- Recently synced (rate limited)\n"
                    msg += "- No data available\n"
                    msg += "- Vector DB not connected"

                return msg
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Sync failed: {e}"

        async def similar_handler():
            """Find similar historical patterns"""
            try:
                patterns = await self.executor.find_similar_patterns(top_k=5)

                if not patterns:
                    msg = f"{TelegramFormatter.MEMO} <b>No similar patterns found</b>\n\n"
                    msg += "Possible reasons:\n"
                    msg += "- Vector DB not connected\n"
                    msg += "- No historical data synced\n"
                    msg += "- Insufficient current data"
                    return msg

                msg = TelegramFormatter.tree_header("Similar Patterns", "🔍")
                msg += f"<i>Top 5 historical matches for current market state</i>\n\n"

                for i, p in enumerate(patterns, 1):
                    score = p.get('score', 0) * 100
                    ts = p.get('timestamp', 'Unknown')
                    close = p.get('close', 0)

                    if score >= 90:
                        score_emoji = TelegramFormatter.CHECK
                    elif score >= 70:
                        score_emoji = "🟡"
                    else:
                        score_emoji = "🟠"

                    msg += f"{i}. {score_emoji} <b>{score:.1f}%</b> match\n"
                    msg += f"   Time: <code>{ts[:16] if len(ts) > 16 else ts}</code>\n"
                    msg += f"   Close: {close:.5f}\n\n"

                msg += f"{TelegramFormatter.BRAIN} <i>Use for context, not predictions</i>"

                return msg
            except Exception as e:
                return f"{TelegramFormatter.CROSS} Error: {e}"

        async def help_handler():
            """Show all available commands"""
            msg = TelegramFormatter.tree_header("H1 v6.4 GBPUSD Commands", "📖")

            msg += TelegramFormatter.tree_section("Information", "ℹ️")
            msg += f"  /status - System status & layer info\n"
            msg += f"  /balance - Account balance & equity\n"
            msg += f"  /positions - Open positions\n"
            msg += f"  /regime - Market regime\n"
            msg += f"  /pois - Active Order Blocks\n"
            msg += f"  /activity - Activity filter\n"
            msg += f"  /mode - Trading mode\n"
            msg += f"  /market - Market analysis\n"
            msg += f"  /layers - All 4 layers status\n"
            msg += f"  /autotrading - MT5 AutoTrading\n"

            msg += TelegramFormatter.tree_section("Statistics", "📊")
            msg += f"  /trades - Recent trade history\n"
            msg += f"  /stats - Trading statistics\n"
            msg += f"  /daily - Today's summary\n"
            msg += f"  /monthly - Monthly summary\n"
            msg += f"  /history - MT5 trade history\n"
            msg += f"  /config - Configuration\n"

            msg += TelegramFormatter.tree_section("Vector DB", "🔷")
            msg += f"  /vector - Vector DB status\n"
            msg += f"  /sync - Trigger manual sync\n"
            msg += f"  /similar - Find similar patterns\n"

            msg += TelegramFormatter.tree_section("Control", TelegramFormatter.GEAR)
            msg += f"  /pause - Pause trading\n"
            msg += f"  /resume - Resume trading\n"
            msg += f"  /reset_l4 - Reset pattern filter\n"

            msg += TelegramFormatter.tree_section("Testing", "🧪")
            msg += f"  /test_buy - Test BUY (0.01)\n"
            msg += f"  /test_sell - Test SELL (0.01)\n"
            msg += f"  /close_all - Close all\n"

            msg += f"\n{TelegramFormatter.BRAIN} <i>QUAD-LAYER Quality Filter</i>"

            return msg

        # Register all callbacks
        # Information commands
        self.telegram.on_status = status_handler
        self.telegram.on_balance = balance_handler
        self.telegram.on_positions = positions_handler
        self.telegram.on_regime = regime_handler
        self.telegram.on_pois = pois_handler
        self.telegram.on_activity = activity_handler
        self.telegram.on_mode = mode_handler
        self.telegram.on_market = market_handler
        self.telegram.on_layers = layers_handler
        self.telegram.on_autotrading = autotrading_handler

        # Statistics commands (NEW)
        self.telegram.on_trades = trades_handler
        self.telegram.on_stats = stats_handler
        self.telegram.on_daily = daily_handler
        self.telegram.on_monthly = monthly_handler
        self.telegram.on_history = history_handler
        self.telegram.on_config = config_handler

        # Control commands
        self.telegram.on_pause = pause_handler
        self.telegram.on_resume = resume_handler
        self.telegram.on_reset_l4 = reset_l4_handler

        # Vector DB commands (NEW)
        self.telegram.on_vector = vector_handler
        self.telegram.on_sync = sync_handler
        self.telegram.on_similar = similar_handler

        # Testing commands
        self.telegram.on_test_buy = test_buy_handler
        self.telegram.on_test_sell = test_sell_handler
        self.telegram.on_close_all = close_all_handler
        self.telegram.on_help = help_handler

    def _get_session_name(self, hour: int) -> str:
        """Get current trading session name (UTC)"""
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
        """Check if current hour is in kill zone"""
        return (7 <= hour <= 11) or (13 <= hour <= 17)

    def _is_market_open(self, now: datetime) -> bool:
        """Check if forex market is open"""
        weekday = now.weekday()
        hour = now.hour

        if weekday == 5:
            return False
        elif weekday == 6:
            return hour >= 22
        elif weekday == 4:
            return hour < 22
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

            # Get Layer 3 & 4 status
            l3_status = status.get('layer3_status', {})
            l4_status = status.get('layer4_status', {})

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
            msg += TelegramFormatter.tree_item("Monthly P&L", f"${l3_status.get('monthly_pnl', 0):.2f}")
            msg += TelegramFormatter.tree_item("Position", "Yes" if status['has_position'] else "No", last=True)

            # Show Layer 4 status if interesting
            if l4_status.get('is_halted') or l4_status.get('in_recovery'):
                msg += TelegramFormatter.tree_section("Pattern Filter", "🔍")
                if l4_status.get('is_halted'):
                    msg += TelegramFormatter.tree_item("Status", f"{TelegramFormatter.RED} HALTED", last=True)
                elif l4_status.get('in_recovery'):
                    msg += TelegramFormatter.tree_item("Status", f"🟡 RECOVERY", last=True)

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
                f"{TelegramFormatter.LAST} Strategy: <code>QUAD-LAYER Quality</code>\n"
            )

            warnings = []
            if not autotrading_enabled:
                warnings.append("Enable AutoTrading di MT5!")
            if not market_open:
                warnings.append("Market tutup (weekend)")

            if warnings:
                msg += f"\n{TelegramFormatter.WARNING} <b>Warning:</b>\n"
                for w in warnings:
                    msg += f"  - {w}\n"

            msg += "\n<b>Commands:</b> /help for full list\n"
            msg += "<code>/status /balance /positions /layers</code>\n"
            msg += "<code>/regime /pois /activity /market</code>\n"
            msg += "<code>/test_buy /test_sell /close_all</code>\n"
            msg += "<code>/pause /resume /autotrading</code>"

            await self.telegram.send(msg)

        logger.info("Starting main trading loop")

        while self.running:
            try:
                self.loop_count += 1
                cycle_start = datetime.now(timezone.utc)

                tick = self.mt5.get_tick_sync(SYMBOL)
                account = self.mt5.get_account_info_sync()

                if tick and account:
                    price = tick.get('bid', 0)
                    balance = account.get('balance', 0)
                    spread = tick.get('spread', 0)

                    now = datetime.now(timezone.utc)
                    adj = self.executor.risk_scorer.get_monthly_quality_adjustment(now)
                    session = self._get_session_name(now.hour)
                    in_kz = "KZ" if self._is_kill_zone(now.hour) else "--"
                    market_status = self._get_market_status(now)
                    status = self.executor.get_status()
                    pos = "POS" if status['has_position'] else "---"

                    # Get Layer 4 status for logging
                    l4_status = status.get('layer4_status', {})
                    l4_str = "OK"
                    if l4_status.get('is_halted'):
                        l4_str = "HALT"
                    elif l4_status.get('in_recovery'):
                        l4_str = "RECV"

                    logger.info(
                        f"[{self.loop_count:04d}] {SYMBOL} {price:.5f} | "
                        f"Sprd: {spread:.1f} | "
                        f"{session:10} | {in_kz} | "
                        f"Mkt: {market_status:6} | "
                        f"Q+{adj:02d} | L4:{l4_str} | "
                        f"${balance:,.0f} | {pos}"
                    )

                    await self._send_hourly_status(now, price, balance)

                if not self.paused:
                    await self.executor.run_cycle()
                else:
                    logger.debug("Trading paused, skipping cycle")

                elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_time = max(1, self.interval - elapsed)

                logger.debug(f"Cycle completed in {elapsed:.1f}s, sleeping {sleep_time:.0f}s")
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)

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
        description='SURGE-WSI H1 v6.4 GBPUSD Trading Bot (QUAD-LAYER)'
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
    print("QUAD-LAYER Quality Filter")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Interval: {args.interval}s")
    print(f"Symbol: {SYMBOL}")
    print("=" * 60)
    print("\nLayers:")
    print("  Layer 1: Monthly Profile (tradeable %)")
    print("  Layer 2: Technical (ATR, Efficiency, ADX)")
    print("  Layer 3: Intra-Month Risk (consecutive losses)")
    print("  Layer 4: Pattern Filter (rolling WR, direction)")
    print("=" * 60)
    print("\nBacktest Performance (Jan 2025 - Jan 2026):")
    print("  - 102 trades, 42.2% WR")
    print("  - PF 3.57")
    print("  - +$12,888.80 (+25.8% return)")
    print("  - ZERO losing months (0/13)")
    print("=" * 60)

    bot = TradingBot(mode=mode, interval=args.interval)

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        bot.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
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
