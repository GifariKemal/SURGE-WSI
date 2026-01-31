"""
Test MT5 Connection
===================

Quick test to verify MT5 connection with MetaQuotes-Demo credentials.

Usage:
    python -m ml_trading_bot.test_mt5_connection
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mt5_connector import MT5Connector


# MT5 Credentials - MetaQuotes Demo
MT5_CONFIG = {
    "server": "MetaQuotes-Demo",
    "login": 10009310110,
    "password": "P-WyAnG8",
    "terminal_path": r"C:\Program Files\MetaTrader 5\terminal64.exe"
}


def test_connection():
    """Test MT5 connection"""
    print("=" * 60)
    print("MT5 CONNECTION TEST")
    print("=" * 60)
    print(f"\nServer: {MT5_CONFIG['server']}")
    print(f"Login:  {MT5_CONFIG['login']}")
    print()

    # Connect with force_login to use specific credentials
    print("Connecting to MT5 (force login)...")
    connector = MT5Connector(
        login=MT5_CONFIG['login'],
        password=MT5_CONFIG['password'],
        server=MT5_CONFIG['server'],
        terminal_path=MT5_CONFIG.get('terminal_path')
    )

    if not connector.connect(force_login=True):
        print("FAILED to connect!")
        print(f"Error: {connector.get_last_error()}")
        return False

    print("Connected!")

    # Account info
    print("\n" + "-" * 60)
    print("ACCOUNT INFO")
    print("-" * 60)

    account = connector.get_account_info_sync()
    if account:
        print(f"Name:     {account['name']}")
        print(f"Server:   {account['server']}")
        print(f"Currency: {account['currency']}")
        print(f"Balance:  ${account['balance']:,.2f}")
        print(f"Equity:   ${account['equity']:,.2f}")
        print(f"Leverage: 1:{account['leverage']}")
    else:
        print("Failed to get account info")

    # Tick data
    print("\n" + "-" * 60)
    print("GBPUSD TICK")
    print("-" * 60)

    tick = connector.get_tick_sync("GBPUSD")
    if tick:
        print(f"Symbol: {tick['symbol']}")
        print(f"Bid:    {tick['bid']:.5f}")
        print(f"Ask:    {tick['ask']:.5f}")
        print(f"Spread: {tick['spread']:.1f} pts")
        print(f"Time:   {tick['time']}")
    else:
        print("Failed to get tick")

    # Symbol info
    print("\n" + "-" * 60)
    print("GBPUSD SYMBOL INFO")
    print("-" * 60)

    info = connector.get_symbol_info("GBPUSD")
    if info:
        print(f"Description:   {info['description']}")
        print(f"Digits:        {info['digits']}")
        print(f"Contract Size: {info['contract_size']}")
        print(f"Volume Min:    {info['volume_min']}")
        print(f"Volume Max:    {info['volume_max']}")
        print(f"Spread:        {info['spread']} pts")
    else:
        print("Failed to get symbol info")

    # H1 Data
    print("\n" + "-" * 60)
    print("LATEST H1 BARS")
    print("-" * 60)

    df = connector.get_ohlcv("GBPUSD", "H1", 5)
    if df is not None:
        print(df.to_string())
    else:
        print("Failed to get OHLCV data")

    # Open positions
    print("\n" + "-" * 60)
    print("OPEN POSITIONS")
    print("-" * 60)

    positions = connector.get_positions_sync("GBPUSD")
    if positions:
        for pos in positions:
            print(f"  Ticket: {pos['ticket']}")
            print(f"  Type:   {pos['type']}")
            print(f"  Volume: {pos['volume']}")
            print(f"  Profit: ${pos['profit']:.2f}")
            print()
    else:
        print("No open positions")

    # Session info
    print("\n" + "-" * 60)
    print("SESSION INFO")
    print("-" * 60)

    session = connector.get_session_info()
    print(f"UTC Time:      {session['utc_time']}")
    print(f"Session:       {session['session']}")
    print(f"Major Session: {session['major_session']}")
    print(f"Day:           {session['day_of_week']}")
    print(f"Is Weekend:    {session['is_weekend']}")

    # Disconnect
    connector.disconnect()
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
