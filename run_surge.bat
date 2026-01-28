@echo off
REM ============================================
REM SURGE-WSI Trading System with Live Logging
REM ============================================
REM
REM Features:
REM - Real-time log display
REM - Auto sync database on startup
REM - Auto trading with notifications
REM
REM Author: SURIOTA Team
REM ============================================

title SURGE-WSI Live Trading

cd /d "%~dp0"

echo.
echo ============================================
echo     SURGE-WSI Trading System
echo     with Live Logging
echo ============================================
echo.

REM Check Docker
echo [1/4] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)
echo [OK] Docker is running

REM Start Docker Services
echo.
echo [2/4] Starting Docker services...
docker-compose up -d
timeout /t 5 /nobreak >nul

REM Check TimescaleDB
:check_db
docker exec surge_wsi_timescaledb pg_isready -U surge_wsi -d surge_wsi >nul 2>&1
if errorlevel 1 (
    echo Waiting for TimescaleDB...
    timeout /t 2 /nobreak >nul
    goto check_db
)
echo [OK] TimescaleDB ready

REM Check Redis
docker exec surge_wsi_redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Redis not responding
) else (
    echo [OK] Redis ready
)

REM Check MT5
echo.
echo [3/4] Checking MT5 Terminal...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] MT5 Terminal already running
) else (
    echo Starting MT5 Terminal...
    start "" "C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"
    echo Waiting for MT5...
    timeout /t 10 /nobreak >nul
    echo [OK] MT5 Terminal started
)

REM Activate venv
echo.
echo [4/4] Starting SURGE-WSI...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo ============================================
echo           SURGE-WSI LIVE MODE
echo ============================================
echo.
echo Data Flow:
echo   MT5 --[price]--> Python --[sync]--> Database
echo.
echo Features:
echo   [x] Auto Sync Database
echo   [x] Auto Trading (if enabled in MT5)
echo   [x] Telegram Notifications
echo.
echo Telegram Commands:
echo   /status    - System status
echo   /balance   - Account balance
echo   /positions - Open positions
echo   /regime    - Market regime
echo   /pois      - Active POIs
echo   /pause     - Pause trading
echo   /resume    - Resume trading
echo   /test_buy  - Test BUY order (0.01 lot)
echo   /test_sell - Test SELL order (0.01 lot)
echo.
echo ============================================
echo Press Ctrl+C to stop
echo ============================================
echo.

REM Run with verbose logging
python main.py --demo --interval 60 --verbose

echo.
echo ============================================
echo SURGE-WSI has stopped.
echo ============================================
pause
