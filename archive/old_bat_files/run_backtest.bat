@echo off
taskkill /IM terminal64.exe /F 2>nul
timeout /t 3 /nobreak >nul
start "" "C:\Program Files\MetaTrader 5\terminal64.exe" /config:"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\tester.ini"
echo MT5 started with backtest config
