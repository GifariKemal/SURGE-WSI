"""
SURGE-WSI Pre-Flight Checker
============================

Comprehensive system check before running the trading bot:
1. Docker services (PostgreSQL, Redis, Qdrant)
2. MT5 connection and AutoTrading status
3. Database connectivity
4. Telegram notification test
5. Market status

Author: SURIOTA Team
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


class PreflightChecker:
    """Pre-flight system checker"""

    def __init__(self):
        self.results = {}
        self.all_passed = True

    def log_section(self, title: str):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def log_result(self, name: str, passed: bool, details: str = ""):
        """Log check result"""
        status = "[OK]" if passed else "[FAIL]"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} {name}")
        if details:
            print(f"      {details}")
        self.results[name] = {"passed": passed, "details": details}
        if not passed:
            self.all_passed = False

    async def check_docker_services(self):
        """Check if Docker services are running"""
        self.log_section("DOCKER SERVICES")

        # Check Docker first
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.log_result("Docker Engine", False, "Docker not running")
                return
            self.log_result("Docker Engine", True, "Running")
        except FileNotFoundError:
            self.log_result("Docker Engine", False, "Docker not installed")
            return
        except Exception as e:
            self.log_result("Docker Engine", False, str(e))
            return

        # Check containers - try multiple possible names
        services = [
            ("PostgreSQL/TimescaleDB", ["surge-timescaledb", "timescaledb", "postgres"], 5432),
            ("Redis", ["surge-redis", "redis"], 6379),
            ("Qdrant", ["surge-qdrant", "qdrant"], 6333),
        ]

        for name, container_names, port in services:
            found = False
            for container in container_names:
                try:
                    result = subprocess.run(
                        ["docker", "ps", "--filter", f"name={container}", "--format", "{{.Names}}: {{.Status}}"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    output = result.stdout.strip()

                    if output and "Up" in output:
                        self.log_result(name, True, f"{output.split(':')[0]}, Port: {port}")
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                # Try checking by port
                try:
                    result = subprocess.run(
                        ["docker", "ps", "--format", "{{.Names}}: {{.Ports}}"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if f":{port}" in result.stdout:
                        self.log_result(name, True, f"Port {port} in use")
                        found = True
                except Exception:
                    pass

                if not found:
                    self.log_result(name, False, f"Not found (expected port: {port})")

    async def check_database_connection(self):
        """Check database connectivity"""
        self.log_section("DATABASE CONNECTION")

        try:
            from config import config
            from src.data.db_handler import DBHandler

            db = DBHandler(
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.user,
                password=config.database.password
            )

            if await db.connect():
                self.log_result("Database Connection", True, f"Host: {config.database.host}:{config.database.port}")

                # Check data availability using get_ohlcv
                try:
                    from datetime import datetime, timezone, timedelta
                    end = datetime.now(timezone.utc)
                    start = end - timedelta(days=365)
                    df = await db.get_ohlcv("GBPUSD", "H1", 10000, start, end)
                    bar_count = len(df) if df is not None else 0
                    self.log_result("GBPUSD H1 Data", bar_count > 1000, f"{bar_count:,} bars available")
                except Exception as e:
                    self.log_result("GBPUSD H1 Data", False, str(e))

                await db.disconnect()
            else:
                self.log_result("Database Connection", False, "Connection failed")
        except Exception as e:
            self.log_result("Database Connection", False, str(e))

    async def check_mt5_connection(self):
        """Check MT5 connection and AutoTrading status"""
        self.log_section("MT5 CONNECTION")

        try:
            import MetaTrader5 as mt5

            if mt5.initialize():
                # Terminal info
                terminal = mt5.terminal_info()
                account = mt5.account_info()

                if terminal and account:
                    self.log_result("MT5 Terminal", True, f"{terminal.name}")
                    self.log_result("MT5 Account", True, f"Login: {account.login}, Server: {account.server}")
                    self.log_result("Account Balance", True, f"${account.balance:,.2f}")
                    self.log_result("Account Equity", True, f"${account.equity:,.2f}")

                    # AutoTrading check - CRITICAL
                    autotrading = terminal.trade_allowed
                    self.log_result(
                        "AutoTrading Status",
                        autotrading,
                        "ENABLED - Ready for trading" if autotrading else "DISABLED - Enable in MT5!"
                    )

                    # Check if algo trading allowed for account
                    algo_allowed = account.trade_allowed
                    self.log_result(
                        "Algo Trading Permission",
                        algo_allowed,
                        "Allowed" if algo_allowed else "Not allowed by broker"
                    )

                    # Get GBPUSD info
                    symbol_info = mt5.symbol_info("GBPUSD")
                    if symbol_info:
                        self.log_result("GBPUSD Symbol", True, f"Spread: {symbol_info.spread} points")
                    else:
                        self.log_result("GBPUSD Symbol", False, "Symbol not available")

                    mt5.shutdown()
                else:
                    self.log_result("MT5 Info", False, "Could not get terminal/account info")
                    mt5.shutdown()
            else:
                error = mt5.last_error()
                self.log_result("MT5 Terminal", False, f"Init failed: {error}")
        except ImportError:
            self.log_result("MT5 Module", False, "MetaTrader5 module not installed")
        except Exception as e:
            self.log_result("MT5 Connection", False, str(e))

    async def check_telegram(self):
        """Check Telegram bot connectivity"""
        self.log_section("TELEGRAM NOTIFICATION")

        try:
            from config import config
            from src.utils.telegram import TelegramNotifier

            if not config.telegram.bot_token or not config.telegram.chat_id:
                self.log_result("Telegram Config", False, "Bot token or chat_id not configured")
                return

            telegram = TelegramNotifier(
                bot_token=config.telegram.bot_token,
                chat_id=config.telegram.chat_id
            )

            if await telegram.initialize():
                self.log_result("Telegram Bot", True, "Connected successfully")

                # Send test notification
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = (
                    f"<b>SURGE-WSI Pre-Flight Check</b>\n\n"
                    f"Time: {now}\n"
                    f"Status: System starting...\n"
                    f"Mode: DEMO"
                )
                try:
                    await telegram.send(msg, force=True)  # Force to bypass rate limit
                    self.log_result("Telegram Test Message", True, "Startup notification sent")
                except Exception as e:
                    self.log_result("Telegram Test Message", False, str(e))
            else:
                self.log_result("Telegram Bot", False, "Failed to initialize")
        except Exception as e:
            self.log_result("Telegram", False, str(e))

    async def check_market_status(self):
        """Check current market status"""
        self.log_section("MARKET STATUS")

        try:
            import MetaTrader5 as mt5
            from datetime import datetime

            if mt5.initialize():
                # Get current time
                now = datetime.now(timezone.utc)
                day = now.weekday()
                hour = now.hour

                # Check if market is open
                is_weekend = day >= 5
                is_trading_hours = (8 <= hour <= 10) or (13 <= hour <= 17)

                self.log_result("Current Time (UTC)", True, now.strftime("%Y-%m-%d %H:%M:%S"))
                self.log_result("Day of Week", not is_weekend, f"{'Weekend' if is_weekend else ['Mon','Tue','Wed','Thu','Fri'][day]}")

                if not is_weekend:
                    session = ""
                    if 8 <= hour <= 10:
                        session = "London Session (8-10 UTC)"
                    elif 13 <= hour <= 17:
                        session = "New York Session (13-17 UTC)"
                    else:
                        session = "Outside Kill Zones"

                    self.log_result("Trading Session", is_trading_hours, session)

                # Get latest price
                tick = mt5.symbol_info_tick("GBPUSD")
                if tick:
                    self.log_result("GBPUSD Price", True, f"Bid: {tick.bid:.5f}, Ask: {tick.ask:.5f}")

                mt5.shutdown()
            else:
                self.log_result("Market Status", False, "Cannot check - MT5 not connected")
        except Exception as e:
            self.log_result("Market Status", False, str(e))

    async def run_all_checks(self) -> bool:
        """Run all pre-flight checks"""
        print("\n")
        print("=" * 60)
        print("     SURGE-WSI PRE-FLIGHT SYSTEM CHECK")
        print("=" * 60)
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        await self.check_docker_services()
        await self.check_database_connection()
        await self.check_mt5_connection()
        await self.check_telegram()
        await self.check_market_status()

        # Summary
        self.log_section("SUMMARY")
        passed = sum(1 for r in self.results.values() if r["passed"])
        total = len(self.results)
        print(f"  Checks Passed: {passed}/{total}")

        if self.all_passed:
            print(f"\n  \033[92m[ALL SYSTEMS GO]\033[0m Ready to start trading bot!")
        else:
            print(f"\n  \033[93m[WARNING]\033[0m Some checks failed. Review before proceeding.")

        print("=" * 60)
        print()

        return self.all_passed


async def main():
    """Main entry point"""
    checker = PreflightChecker()
    passed = await checker.run_all_checks()
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
