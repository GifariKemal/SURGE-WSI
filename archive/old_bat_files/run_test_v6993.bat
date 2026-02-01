@echo off
echo Killing existing MT5 processes...
taskkill /F /IM terminal64.exe 2>nul
taskkill /F /IM metatester64.exe 2>nul
timeout /t 3 /nobreak >nul

echo Starting MT5 Strategy Tester...
start "" "C:\Program Files\MetaTrader 5\terminal64.exe" /config:"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\tester.ini"
echo MT5 started. Waiting for backtest to complete...

:wait_loop
timeout /t 10 /nobreak >nul
tasklist | findstr /i "terminal64.exe" >nul
if errorlevel 1 (
    echo MT5 has finished
    goto done
)
goto wait_loop

:done
echo Backtest complete!
