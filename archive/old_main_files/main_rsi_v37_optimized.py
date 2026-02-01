"""
RSI v3.7 OPTIMIZED - Live Trading Launcher
===========================================

Strategy: RSI Mean Reversion with SIDEWAYS + ConsecLoss3 filters

Backtest Results (Oct 2024 - Jan 2026):
- Return: +72.7%
- Drawdown: 14.4%
- Win Rate: 37.6%
- Losing Months: 2/16

Usage:
    python main_rsi_v37_optimized.py
"""
import asyncio
import sys
from datetime import datetime, timezone
from loguru import logger
import MetaTrader5 as mt5

# Add src to path
sys.path.insert(0, 'src')

from trading.executor_rsi_v37_optimized import RSIMeanReversionV37Optimized


# =============================================================================
# CONFIGURATION
# =============================================================================

MT5_LOGIN = 61045904
MT5_PASSWORD = "iy#K5L7sF"
MT5_SERVER = "FinexBisnisSolusi-Demo"
MT5_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

SYMBOL = "GBPUSD"
MAGIC_NUMBER = 20250201

# Telegram (optional - set to None to disable)
TELEGRAM_BOT_TOKEN = None  # "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = None    # "YOUR_CHAT_ID"


# =============================================================================
# MT5 FUNCTIONS
# =============================================================================

def connect_mt5():
    """Connect to MetaTrader 5"""
    if not mt5.initialize(path=MT5_PATH):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        return False

    account = mt5.account_info()
    logger.info(f"Connected: {account.login} | Balance: ${account.balance:,.2f}")
    return True


async def get_account_info():
    """Get account information"""
    info = mt5.account_info()
    if info:
        return {
            'balance': info.balance,
            'equity': info.equity,
            'profit': info.profit,
            'margin_free': info.margin_free,
        }
    return None


async def get_tick(symbol: str):
    """Get current tick"""
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return {'bid': tick.bid, 'ask': tick.ask, 'time': tick.time}
    return None


async def get_ohlcv(symbol: str, timeframe: str, count: int):
    """Get OHLCV data"""
    import pandas as pd

    tf_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

    if rates is None:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df


async def get_symbol_info(symbol: str):
    """Get symbol info"""
    info = mt5.symbol_info(symbol)
    if info:
        return {
            'contract_size': info.trade_contract_size,
            'point': info.point,
            'digits': info.digits,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
        }
    return None


async def place_market_order(
    symbol: str,
    order_type: str,
    volume: float,
    sl: float,
    tp: float,
    comment: str = "",
    magic: int = 0
):
    """Place market order"""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None

    if order_type.upper() == 'BUY':
        price = tick.ask
        mt5_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        mt5_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Order placed: {order_type} {volume} @ {price:.5f}")
        return {'ticket': result.order, 'price': price}
    else:
        logger.error(f"Order failed: {result}")
        return None


async def close_position(ticket: int):
    """Close position by ticket"""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False

    pos = positions[0]
    symbol = pos.symbol
    volume = pos.volume

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False

    if pos.type == mt5.ORDER_TYPE_BUY:
        price = tick.bid
        close_type = mt5.ORDER_TYPE_SELL
    else:
        price = tick.ask
        close_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": pos.magic,
        "comment": "Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return result and result.retcode == mt5.TRADE_RETCODE_DONE


async def get_positions():
    """Get open positions"""
    positions = mt5.positions_get()
    if not positions:
        return []

    return [
        {
            'ticket': p.ticket,
            'symbol': p.symbol,
            'type': p.type,
            'volume': p.volume,
            'price_open': p.price_open,
            'sl': p.sl,
            'tp': p.tp,
            'profit': p.profit,
            'magic': p.magic,
            'time': p.time,
        }
        for p in positions
    ]


async def get_deal_history(ticket: int):
    """Get deal history for closed position"""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    from_date = now - timedelta(days=7)

    deals = mt5.history_deals_get(from_date, now, position=ticket)
    if not deals or len(deals) < 2:
        return None

    # Get closing deal
    close_deal = deals[-1]
    return {
        'profit': close_deal.profit,
        'swap': close_deal.swap,
        'commission': close_deal.commission,
        'close_reason': 'TP' if close_deal.profit > 0 else 'SL',
    }


async def send_telegram(message: str):
    """Send telegram notification"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    import aiohttp

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram failed: {resp.status}")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


# =============================================================================
# MAIN LOOP
# =============================================================================

async def main():
    """Main trading loop"""
    logger.info("=" * 60)
    logger.info("RSI v3.7 OPTIMIZED - Live Trading")
    logger.info("SIDEWAYS Regime + ConsecLoss3 Filter")
    logger.info("=" * 60)

    # Connect to MT5
    if not connect_mt5():
        return

    try:
        # Initialize executor
        executor = RSIMeanReversionV37Optimized(
            symbol=SYMBOL,
            magic_number=MAGIC_NUMBER
        )

        # Set callbacks
        executor.set_callbacks(
            get_account_info=get_account_info,
            get_tick=get_tick,
            get_ohlcv=get_ohlcv,
            get_symbol_info=get_symbol_info,
            place_market_order=place_market_order,
            close_position=close_position,
            get_positions=get_positions,
            get_deal_history=get_deal_history,
            send_telegram=send_telegram
        )

        # Fetch symbol info
        await executor.fetch_symbol_info()

        # Warmup
        logger.info("Warming up...")
        if not await executor.warmup():
            logger.error("Warmup failed")
            return

        # Recover existing positions
        await executor.recover_positions()

        # Log status
        status = executor.get_status()
        logger.info(f"State: {status['state']}")
        logger.info(f"Regime: {status['current_regime']} (Allowed: {status['regime_allowed']})")
        logger.info(f"ConsecLoss: {status['consecutive_losses']}/{executor.CONSEC_LOSS_LIMIT}")

        # Main loop
        logger.info("\nStarting main loop... Press Ctrl+C to stop")
        last_bar_time = None

        while True:
            try:
                # Get account info
                account = await get_account_info()
                if not account:
                    await asyncio.sleep(5)
                    continue

                balance = account['balance']

                # Get current bar time
                df = await get_ohlcv(SYMBOL, "H1", 2)
                if df is None or len(df) < 2:
                    await asyncio.sleep(5)
                    continue

                current_bar = df.index[-1]

                # Check for new bar
                if last_bar_time is None or current_bar > last_bar_time:
                    last_bar_time = current_bar

                    # Process new bar
                    result = await executor.on_new_bar(current_bar, balance)

                    if result and result.success:
                        logger.info(f"Trade: {result.direction} @ {result.entry_price:.5f}")

                    # Log status periodically
                    status = executor.get_status()
                    logger.debug(
                        f"Bar: {current_bar} | Regime: {status['current_regime']} | "
                        f"Trades: {status['stats']['trades']} | Net: {status['stats']['net_pnl']}"
                    )

                # Wait before next check
                await asyncio.sleep(10)

            except KeyboardInterrupt:
                logger.info("Stopping...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(30)

    finally:
        mt5.shutdown()
        logger.info("MT5 disconnected")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    logger.add(
        "logs/rsi_v37_opt_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG"
    )

    # Run
    asyncio.run(main())
