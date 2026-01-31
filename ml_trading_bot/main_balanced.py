"""
ML Trading Bot - Balanced Mode
==============================

Run the balanced trading strategy optimized for:
- $50,000 starting balance
- 1% risk per trade
- Automatic lot sizing
- ~27 trades per year (2+ per month)
- 74.1% historical win rate

Usage:
    python ml_trading_bot/main_balanced.py

Configuration:
    PAPER_TRADING = True/False
    INITIAL_BALANCE = your balance
    RISK_PCT = 0.01 (1%)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from ml_trading_bot.executor.balanced_executor import BalancedExecutor


# ========== CONFIGURATION ==========
PAPER_TRADING = True       # Set to False for live trading
SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50000.0  # Your starting balance
RISK_PCT = 0.01            # 1% risk per trade


def main():
    """Run Balanced Trading Strategy"""

    print("=" * 70)
    print("ML TRADING BOT - BALANCED MODE")
    print("=" * 70)
    print()
    print("Strategy Configuration:")
    print(f"  - Symbol: {SYMBOL}")
    print(f"  - Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"  - Risk per Trade: {RISK_PCT*100:.1f}%")
    print(f"  - Max Risk Amount: ${INITIAL_BALANCE * RISK_PCT:,.2f}")
    print()
    print("Strategy Filters:")
    print("  - Hours: 01:00-14:00 UTC")
    print("  - Days: Mon, Tue, Thu, Fri (Skip Wednesday)")
    print("  - Direction: BUY only")
    print("  - Min Confidence: 0.49")
    print()
    print("Expected Performance (based on 13-month backtest):")
    print("  - ~27 trades per year")
    print("  - ~74% win rate")
    print("  - ~+17% return")
    print("  - ~3.4% max drawdown")
    print()
    print(f"Paper Trading: {PAPER_TRADING}")
    print()

    if not PAPER_TRADING:
        print("*" * 70)
        print("WARNING: LIVE TRADING MODE!")
        print("Real money will be at risk.")
        print("*" * 70)
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    executor = BalancedExecutor(
        symbol=SYMBOL,
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
        paper_trading=PAPER_TRADING
    )

    if not executor.initialize():
        logger.error("Failed to initialize executor")
        return

    try:
        print()
        print("=" * 70)
        print("Executor running. Press Ctrl+C to stop.")
        print("Waiting for trading opportunities...")
        print("=" * 70)
        print()
        executor.run()
    except KeyboardInterrupt:
        print()
        logger.info("Interrupted by user")
    finally:
        executor.stop()


if __name__ == "__main__":
    main()
