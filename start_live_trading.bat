@echo off
REM ============================================
REM SURGE-WSI LIVE Trading System Startup
REM ============================================
REM
REM WARNING: This script runs LIVE trading with REAL money!
REM Make sure you understand the risks before proceeding.
REM
REM Author: SURIOTA Team
REM ============================================

title SURGE-WSI LIVE Trading
cd /d "%~dp0"

echo.
echo ============================================
echo        *** WARNING: LIVE TRADING ***
echo ============================================
echo.
echo This script will start LIVE trading with REAL money!
echo.
echo Before proceeding, ensure:
echo   [1] You have tested on DEMO account
echo   [2] You understand the risks involved
echo   [3] You have proper risk management in place
echo   [4] MT5 is connected to your LIVE account
echo.
echo ============================================
echo.

set /p CONFIRM="Type 'LIVE' to confirm and start: "
if /i not "%CONFIRM%"=="LIVE" (
    echo.
    echo Cancelled. Use start_trading.bat for demo mode.
    pause
    exit /b 0
)

echo.
echo ============================================
echo         Starting LIVE Trading...
echo ============================================
echo.

REM Check Docker
echo [1/5] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    pause
    exit /b 1
)
echo [OK] Docker is running

REM Start Docker Services
echo.
echo [2/5] Starting Docker services...
docker-compose up -d
timeout /t 10 /nobreak >nul
echo [OK] Docker services started

REM Check TimescaleDB
:check_db
docker exec surge_wsi_timescaledb pg_isready -U surge_wsi -d surge_wsi >nul 2>&1
if errorlevel 1 (
    echo Waiting for TimescaleDB...
    timeout /t 3 /nobreak >nul
    goto check_db
)
echo [OK] TimescaleDB ready

REM Check MT5
echo.
echo [3/5] Checking MT5 Terminal...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] MT5 Terminal running
) else (
    echo [ERROR] MT5 Terminal is not running!
    echo Please start MT5 and login to your LIVE account first.
    pause
    exit /b 1
)

REM Activate Python
echo.
echo [4/5] Activating Python environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
echo [OK] Python ready

REM Run LIVE
echo.
echo [5/5] Starting LIVE trading...
echo.
echo ============================================
echo     *** SURGE-WSI LIVE TRADING ACTIVE ***
echo ============================================
echo.
echo Mode: LIVE (Real Money)
echo.
echo Telegram Commands:
echo   /status    - View system status
echo   /balance   - View account balance
echo   /positions - View open positions
echo   /pause     - PAUSE trading immediately
echo   /resume    - Resume trading
echo   /close_all - Close ALL positions
echo.
echo Press Ctrl+C to stop the system
echo ============================================
echo.

python main.py --live --interval 60

echo.
echo ============================================
echo SURGE-WSI LIVE Trading has stopped.
echo ============================================
pause
