"""
RSI Trading Bot - Main Entry Point (v3.7 OPTIMIZED)
====================================================

OPTIMIZED RSI Mean Reversion Strategy
Backtest Result: +618.2% over 6 years (2020-2026)
WR 37.7% | MaxDD 30.7%

v3.7: Skip 12:00 UTC (London lunch break) -> +45.3% improvement, MaxDD 30.7%
v3.6: Max holding period (46h) -> +48.9% improvement, lower drawdown
v3.5: Time-based TP (+0.35x during London+NY overlap) -> +31.0% improvement
v3.4: RSI thresholds 42/58 -> +493.1% (optimal after extensive testing)
v3.3: Added Dynamic TP (volatility-adjusted) -> +75.2% improvement
v3.2: Added Volatility Filter (ATR percentile 20-80) -> +52.1% improvement

Usage:
    python -m ml_trading_bot.main_rsi
    python -m ml_trading_bot.main_rsi --paper
    python -m ml_trading_bot.main_rsi --status

Author: SURIOTA Team
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    Path(__file__).parent / "logs" / "rsi_bot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


# ========================================================================
# MT5 CREDENTIALS FROM ENVIRONMENT
# ========================================================================
def get_mt5_config() -> dict:
    """Get MT5 config from environment variables"""
    login = os.getenv('MT5_LOGIN')
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    terminal_path = os.getenv('MT5_TERMINAL_PATH')

    if not login or not password or not server:
        raise ValueError(
            "MT5 credentials not set!\n"
            "Create ml_trading_bot/.env file with:\n"
            "  MT5_LOGIN=your_login\n"
            "  MT5_PASSWORD=your_password\n"
            "  MT5_SERVER=your_server\n"
            "  MT5_TERMINAL_PATH=path_to_terminal (optional)"
        )

    return {
        "login": int(login),
        "password": password,
        "server": server,
        "terminal_path": terminal_path
    }


# ========================================================================
# OPTIMIZED RSI PARAMETERS (v3.7)
# ========================================================================
RSI_CONFIG = {
    "rsi_period": 10,
    "rsi_oversold": 42.0,
    "rsi_overbought": 58.0,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "session_start": 7,
    "session_end": 22,
    "risk_per_trade": 0.01,
    # Volatility filter (v3.2) - skip extreme volatility
    "min_atr_percentile": 20.0,
    "max_atr_percentile": 80.0,
    # Dynamic TP (v3.3) - adjust TP based on volatility
    "dynamic_tp": True,
    "tp_low_vol_mult": 2.4,   # TP when ATR < 40th percentile
    "tp_high_vol_mult": 3.6,  # TP when ATR > 60th percentile
    # Time-based TP (v3.5) - larger TP during London+NY overlap
    "time_tp_bonus": True,
    "time_tp_start": 12,      # London+NY overlap start
    "time_tp_end": 16,        # London+NY overlap end
    "time_tp_bonus_mult": 0.35,  # Add 0.35x ATR to TP during overlap
    # Max holding period (v3.6) - force close stuck positions
    "max_holding_hours": 46,  # Close at market after 46 hours
    # Skip hours filter (v3.7) - avoid low-liquidity periods
    "skip_hours": [12],       # Skip 12:00 UTC (London lunch break)
}


def print_banner():
    print("""
+==============================================================+
|   RSI TRADING BOT v3.7 OPTIMIZED                             |
|   ----------------------------------                         |
|   Strategy:  RSI Mean Reversion + Dynamic TP + Hour Filter   |
|   Symbol:    GBPUSD H1                                       |
|   Backtest:  +618.2% (2020-2026) | MaxDD 30.7%               |
|                                                              |
|   Entry:     RSI(10) < 42 (BUY) / > 58 (SELL)                |
|   Filter:    ATR percentile 20-80 (skip extreme vol)         |
|   Filter:    Skip 12:00 UTC (London lunch break) (v3.7)      |
|   Exit:      SL: 1.5x ATR | Risk: 1% per trade               |
|   Dynamic:   TP 2.4x(low vol) / 3.0x(med) / 3.6x(high vol)   |
|   Time TP:   +0.35x during London+NY overlap (12-16 UTC)     |
|   Max Hold:  Force close after 46 hours                      |
|   Session:   07:00 - 22:00 UTC (excl. 12:00)                 |
+==============================================================+
""")


async def run_bot(paper_mode: bool = False):
    """Run the RSI trading bot"""
    from ml_trading_bot.executor.rsi_executor import RSIExecutor

    # Get MT5 config from environment
    try:
        mt5_config = get_mt5_config()
    except ValueError as e:
        logger.error(str(e))
        return

    executor = RSIExecutor(
        symbol="GBPUSD",
        rsi_period=RSI_CONFIG["rsi_period"],
        rsi_oversold=RSI_CONFIG["rsi_oversold"],
        rsi_overbought=RSI_CONFIG["rsi_overbought"],
        sl_atr_mult=RSI_CONFIG["sl_atr_mult"],
        tp_atr_mult=RSI_CONFIG["tp_atr_mult"],
        session_start=RSI_CONFIG["session_start"],
        session_end=RSI_CONFIG["session_end"],
        risk_per_trade=RSI_CONFIG["risk_per_trade"],
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        paper_mode=paper_mode,
        # Volatility filter (v3.2)
        min_atr_percentile=RSI_CONFIG["min_atr_percentile"],
        max_atr_percentile=RSI_CONFIG["max_atr_percentile"],
        # Dynamic TP (v3.3)
        dynamic_tp=RSI_CONFIG["dynamic_tp"],
        tp_low_vol_mult=RSI_CONFIG["tp_low_vol_mult"],
        tp_high_vol_mult=RSI_CONFIG["tp_high_vol_mult"],
        # Time-based TP (v3.5)
        time_tp_bonus=RSI_CONFIG["time_tp_bonus"],
        time_tp_start=RSI_CONFIG["time_tp_start"],
        time_tp_end=RSI_CONFIG["time_tp_end"],
        time_tp_bonus_mult=RSI_CONFIG["time_tp_bonus_mult"],
        # Max holding period (v3.6)
        max_holding_hours=RSI_CONFIG["max_holding_hours"],
        # Skip hours filter (v3.7)
        skip_hours=RSI_CONFIG["skip_hours"],
    )

    logger.info(f"Connecting to MT5: {mt5_config['server']}...")
    if not executor.connect_mt5(
        login=mt5_config['login'],
        password=mt5_config['password'],
        server=mt5_config['server'],
        terminal_path=mt5_config.get('terminal_path')
    ):
        logger.error("Failed to connect to MT5")
        return

    # Telegram (optional)
    try:
        from ml_trading_bot.notifications import MLTelegramNotifier
        telegram = MLTelegramNotifier()
        if await telegram.initialize():
            executor.send_telegram = telegram.send
            account = executor.mt5.get_account_info_sync()
            if account:
                mode_str = "PAPER" if paper_mode else "LIVE"
                msg = f"<b>RSI BOT STARTED ({mode_str})</b>\n\n"
                msg += f"Balance: ${account['balance']:,.2f}\n"
                msg += f"Server: {mt5_config['server']}\n"
                msg += f"RSI({RSI_CONFIG['rsi_period']}) < {RSI_CONFIG['rsi_oversold']:.0f} / > {RSI_CONFIG['rsi_overbought']:.0f}"
                await telegram.send(msg, force=True)
            logger.info("Telegram enabled")
    except Exception as e:
        logger.warning(f"Telegram disabled: {e}")

    if paper_mode:
        logger.warning("=" * 50)
        logger.warning("PAPER MODE - No real orders will be placed")
        logger.warning("=" * 50)

    logger.info("Warming up...")
    if not await executor.warmup():
        logger.error("Warmup failed")
        return

    status = executor.get_status()
    logger.info(f"RSI: {status['rsi']:.1f} | In Session: {status['in_session']}")
    logger.info("Starting trading loop (Ctrl+C to stop)...")

    try:
        await executor.run(interval_seconds=5)
    finally:
        executor.stop()


async def show_status():
    """Show current status"""
    from ml_trading_bot.executor.rsi_executor import RSIExecutor

    try:
        mt5_config = get_mt5_config()
    except ValueError as e:
        logger.error(str(e))
        return

    executor = RSIExecutor(
        symbol="GBPUSD",
        rsi_period=RSI_CONFIG["rsi_period"],
        rsi_oversold=RSI_CONFIG["rsi_oversold"],
        rsi_overbought=RSI_CONFIG["rsi_overbought"],
    )

    if not executor.connect_mt5(
        login=mt5_config['login'],
        password=mt5_config['password'],
        server=mt5_config['server'],
        terminal_path=mt5_config.get('terminal_path')
    ):
        logger.error("Failed to connect")
        return

    await executor.warmup()
    status = executor.get_status()

    print("\n" + "=" * 50)
    print("RSI TRADING BOT - STATUS")
    print("=" * 50)

    account = executor.mt5.get_account_info_sync()
    if account:
        print(f"\nAccount: {account.get('name', 'N/A')}")
        print(f"Server: {mt5_config['server']}")
        print(f"Balance: ${account['balance']:,.2f}")

    print(f"\nRSI({RSI_CONFIG['rsi_period']}): {status['rsi']:.1f}")
    print(f"ATR Percentile: {status.get('atr_percentile', 0):.0f}th")
    print(f"Volatility OK: {'YES' if status.get('volatility_ok', False) else 'NO'} (range: 20-80)")
    print(f"In Session: {'YES' if status['in_session'] else 'NO'}")
    print(f"Market Open: {'YES' if status['market_open'] else 'NO'}")

    rsi = status['rsi']
    vol_ok = status.get('volatility_ok', True)

    if rsi < RSI_CONFIG['rsi_oversold'] and vol_ok:
        print(f"\nSignal: BUY (RSI < {RSI_CONFIG['rsi_oversold']} + Vol OK)")
    elif rsi > RSI_CONFIG['rsi_overbought'] and vol_ok:
        print(f"\nSignal: SELL (RSI > {RSI_CONFIG['rsi_overbought']} + Vol OK)")
    elif rsi < RSI_CONFIG['rsi_oversold'] and not vol_ok:
        print(f"\nSignal: BUY blocked (volatility outside range)")
    elif rsi > RSI_CONFIG['rsi_overbought'] and not vol_ok:
        print(f"\nSignal: SELL blocked (volatility outside range)")
    else:
        print(f"\nSignal: NONE (RSI neutral)")

    print("\n" + "=" * 50)
    executor.stop()


def main():
    parser = argparse.ArgumentParser(description="RSI Trading Bot")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--status", action="store_true", help="Show status only")
    args = parser.parse_args()

    (Path(__file__).parent / "logs").mkdir(exist_ok=True)

    print_banner()
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Mode: {'Paper' if args.paper else 'LIVE'}")
    print()

    if args.status:
        asyncio.run(show_status())
    else:
        if not args.paper:
            print("WARNING: LIVE TRADING MODE")
            print("This will execute REAL trades on your account.")
            if input("Type 'YES' to confirm: ") != "YES":
                print("Aborted.")
                return

        asyncio.run(run_bot(paper_mode=args.paper))


if __name__ == "__main__":
    main()
