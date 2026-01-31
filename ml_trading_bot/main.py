"""
ML Trading Bot - Main Entry Point
==================================

Live trading with ML models on MT5.

Usage:
    python -m ml_trading_bot.main
    python -m ml_trading_bot.main --paper  # Paper trading mode (no real orders)
    python -m ml_trading_bot.main --status # Show status only

Author: SURIOTA Team
"""

import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    Path(__file__).parent / "logs" / "ml_bot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


# ========================================================================
# MT5 CREDENTIALS - MetaQuotes Demo
# ========================================================================
MT5_CONFIG = {
    "server": "MetaQuotes-Demo",
    "login": 10009310110,
    "password": "P-WyAnG8",
    "terminal_path": r"C:\Program Files\MetaTrader 5\terminal64.exe"
}


def print_banner():
    """Print startup banner"""
    print("""
+==============================================================+
|                                                              |
|   ML TRADING BOT v1.0                                        |
|   ----------------------                                     |
|                                                              |
|   Symbol:  GBPUSD H1                                         |
|   Models:  Regime (HMM) + Signal (XGBoost+RF)                |
|   Risk:    Half Kelly Criterion                              |
|                                                              |
+==============================================================+
""")


async def run_bot(paper_mode: bool = False):
    """Run the ML trading bot"""
    from ml_trading_bot.executor import MLExecutor
    from ml_trading_bot.notifications import MLTelegramNotifier

    # Initialize executor with improved risk management
    executor = MLExecutor(
        symbol="GBPUSD",
        confidence_threshold=0.55,
        max_daily_loss_pct=0.02,   # 2% daily loss limit (was 3%)
        max_drawdown_pct=0.10,     # 10% max drawdown protection
        base_risk_pct=0.01         # 1% base risk
    )

    # Load ML models
    logger.info("Loading ML models...")
    if not executor.load_models():
        logger.error("Failed to load ML models")
        return

    # Connect to MT5
    logger.info(f"Connecting to MT5: {MT5_CONFIG['server']}...")
    if not executor.connect_mt5(
        login=MT5_CONFIG['login'],
        password=MT5_CONFIG['password'],
        server=MT5_CONFIG['server'],
        terminal_path=MT5_CONFIG.get('terminal_path')
    ):
        logger.error("Failed to connect to MT5")
        return

    # Initialize Telegram notifications
    telegram = MLTelegramNotifier()
    telegram_enabled = await telegram.initialize()

    if telegram_enabled:
        # Connect executor to telegram
        telegram.executor = executor
        executor.send_telegram = telegram.send

        # Start command polling in background
        asyncio.create_task(telegram.start_polling())

        # Send startup notification
        account = executor.mt5.get_account_info_sync()
        if account:
            await telegram.send_startup(
                account_name=account['name'],
                balance=account['balance'],
                server=account['server']
            )
        logger.info("Telegram notifications enabled")
    else:
        logger.warning("Telegram notifications disabled")

    # Paper mode warning
    if paper_mode:
        logger.warning("=" * 50)
        logger.warning("PAPER TRADING MODE - No real orders will be placed")
        logger.warning("=" * 50)
        # In paper mode, we just monitor without executing
        executor.risk_manager = None  # Disable trading

    # Warmup with historical data
    logger.info("Warming up with historical data...")
    if not await executor.warmup():
        logger.error("Warmup failed")
        return

    # Show initial status
    status = executor.get_status()
    logger.info(f"Initial Status:")
    logger.info(f"  State: {status['state']}")
    logger.info(f"  ML Regime: {status['ml'].get('regime', 'N/A')}")
    logger.info(f"  ML Signal: {status['ml'].get('signal', 'N/A')}")

    # Run main loop
    logger.info("Starting main trading loop (Ctrl+C to stop)...")

    try:
        await executor.run(interval_seconds=5)
    finally:
        # Cleanup
        if telegram_enabled:
            await telegram.stop_polling()
        executor.stop()


async def show_status():
    """Show current ML predictions without trading"""
    from ml_trading_bot.executor import MLExecutor

    executor = MLExecutor(symbol="GBPUSD")

    # Load models
    if not executor.load_models():
        logger.error("Failed to load models")
        return

    # Connect to MT5
    if not executor.connect_mt5(
        login=MT5_CONFIG['login'],
        password=MT5_CONFIG['password'],
        server=MT5_CONFIG['server']
    ):
        logger.error("Failed to connect to MT5")
        return

    # Warmup
    await executor.warmup()

    # Get status
    status = executor.get_status()

    print("\n" + "=" * 60)
    print("ML TRADING BOT - STATUS")
    print("=" * 60)

    # Account info
    account = executor.mt5.get_account_info_sync()
    if account:
        print(f"\nAccount: {account['name']}")
        print(f"Balance: ${account['balance']:,.2f}")
        print(f"Equity:  ${account['equity']:,.2f}")
        print(f"Server:  {account['server']}")

    # ML Predictions
    print(f"\n{'-' * 60}")
    print("ML PREDICTIONS")
    print(f"{'-' * 60}")

    ml = status.get('ml', {})
    print(f"Regime:     {ml.get('regime', 'N/A')} ({ml.get('regime_confidence', 0):.1%})")
    print(f"Signal:     {ml.get('signal', 'N/A')} ({ml.get('signal_confidence', 0):.1%})")

    # Position
    print(f"\n{'-' * 60}")
    print("POSITION")
    print(f"{'-' * 60}")

    if status['has_position']:
        pos = status['position']
        print(f"Direction: {pos['direction']}")
        print(f"Entry:     {pos['entry']:.5f}")
        print(f"SL:        {pos['sl']:.5f}")
        print(f"TP:        {pos['tp']:.5f}")
    else:
        print("No open position")

    # Stats
    print(f"\n{'-' * 60}")
    print("SESSION STATS")
    print(f"{'-' * 60}")

    stats = status['stats']
    print(f"Trades:   {stats['trades']}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Net P/L:  ${stats['net_pnl']:+.2f}")

    print("\n" + "=" * 60)

    executor.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ML Trading Bot")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (no real orders)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit"
    )
    args = parser.parse_args()

    # Create logs directory
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    print_banner()
    print(f"Start time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Mode: {'Paper Trading' if args.paper else 'LIVE TRADING'}")
    print()

    if args.status:
        asyncio.run(show_status())
    else:
        # Confirmation for live trading
        if not args.paper:
            print("⚠️  WARNING: LIVE TRADING MODE")
            print("    This will execute REAL trades on your account.")
            print()
            confirm = input("Type 'YES' to confirm: ")
            if confirm != "YES":
                print("Aborted.")
                return

        asyncio.run(run_bot(paper_mode=args.paper))


if __name__ == "__main__":
    main()
