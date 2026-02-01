@echo off
title RSI v3.7 OPTIMIZED - GBPUSD H1
echo ============================================
echo RSI v3.7 OPTIMIZED - Live Trading
echo ============================================
echo.
echo Filters: SIDEWAYS Regime + ConsecLoss3 (2h cooldown)
echo Backtest: +72.7%% Return, 14.4%% DD
echo.

:: ===========================================
:: MT5 Credentials (EDIT THESE)
:: ===========================================
set MT5_PASSWORD=iy#K5L7sF

:: ===========================================
:: Telegram Notifications (EDIT THESE)
:: ===========================================
:: Get bot token from @BotFather on Telegram
:: Get chat ID from @userinfobot or @getidsbot
set TELEGRAM_BOT_TOKEN=8215295219:AAGwcevN5QKqYIgVnOogB9P1Lo-x6HCoatM
set TELEGRAM_CHAT_ID=-1003549733840

:: Optional: Override other MT5 settings
:: set MT5_LOGIN=61045904
:: set MT5_SERVER=FinexBisnisSolusi-Demo

cd /d "%~dp0"

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

:: Check aiohttp for Telegram
python -c "import aiohttp" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing aiohttp for Telegram notifications...
    pip install aiohttp
)

:: Run
echo.
echo Starting bot...
echo.
python rsi_v37_optimized.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Script exited with error code %ERRORLEVEL%
)
pause
