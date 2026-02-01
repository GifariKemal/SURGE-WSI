@echo off
title RSI v3.7 OPTIMIZED - Backtest
echo ============================================
echo RSI v3.7 OPTIMIZED - Backtest
echo ============================================
echo.

:: MT5 Credentials
set MT5_PASSWORD=iy#K5L7sF

:: Telegram (untuk kirim report)
set TELEGRAM_BOT_TOKEN=8215295219:AAGwcevN5QKqYIgVnOogB9P1Lo-x6HCoatM
set TELEGRAM_CHAT_ID=-1003549733840

cd /d "%~dp0"

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

python backtest_rsi_v37_optimized.py
pause
