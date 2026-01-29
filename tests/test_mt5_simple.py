"""Test MT5 simple connection without credentials"""
import MetaTrader5 as mt5

# Try without credentials (use terminal's current session)
result = mt5.initialize()
print(f"Init: {result}")
print(f"Error: {mt5.last_error()}")

if result:
    ti = mt5.terminal_info()
    print(f"Terminal: {ti.name if ti else 'None'}")

    ai = mt5.account_info()
    print(f"Account: {ai.login if ai else 'None'}")

    mt5.shutdown()
