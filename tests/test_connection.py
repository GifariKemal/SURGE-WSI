"""Quick MT5 connection test"""
import asyncio
from src.data.mt5_connector import MT5Connector

async def test():
    mt5 = MT5Connector()
    connected = mt5.connect()
    print(f"Connected: {connected}")

    if connected:
        info = await mt5.get_account_info()
        print(f"Balance: ${info['balance']:,.2f}")
        print(f"Equity: ${info['equity']:,.2f}")
        print(f"Leverage: 1:{info['leverage']}")
        print(f"Server: {info['server']}")

        # Get GBPUSD price
        price = await mt5.get_symbol_price("GBPUSD")
        if price:
            print(f"GBPUSD: Bid={price['bid']:.5f} Ask={price['ask']:.5f}")

        mt5.disconnect()

if __name__ == "__main__":
    asyncio.run(test())
