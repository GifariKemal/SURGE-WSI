"""Test MT5 connection with explicit credentials"""
import MetaTrader5 as mt5

# Try with full credentials including terminal path
result = mt5.initialize(
    path=r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe",
    login=6145904,
    password="iy#K5L7sF",
    server="FinexBisnisSolusi-Demo"
)

print(f"Result: {result}")
print(f"Error: {mt5.last_error()}")

if result:
    info = mt5.account_info()
    print(f"Balance: ${info.balance:,.2f}")
    print(f"Server: {info.server}")
    mt5.shutdown()
