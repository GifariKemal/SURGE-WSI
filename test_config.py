"""Test config loading"""
from config import config

print(f"Login: {config.mt5.login}")
print(f"Password: {config.mt5.password}")
print(f"Server: {config.mt5.server}")
print(f"Path: {config.mt5.terminal_path}")
