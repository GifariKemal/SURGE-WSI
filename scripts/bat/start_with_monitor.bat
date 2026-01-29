@echo off
REM ============================================
REM SURGE-WSI with Data Monitor
REM ============================================
REM Opens two terminals:
REM 1. Main trading system
REM 2. Data monitor (MT5 + Database)
REM ============================================

title SURGE-WSI Launcher
cd /d "%~dp0"

echo.
echo ============================================
echo     SURGE-WSI with Data Monitor
echo ============================================
echo.

REM ============================================
REM Step 1: Check Docker
REM ============================================
echo [1/4] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)
echo [OK] Docker is running

REM ============================================
REM Step 2: Start Docker Services
REM ============================================
echo.
echo [2/4] Starting Docker services...
docker-compose up -d >nul 2>&1
timeout /t 5 /nobreak >nul

REM Wait for TimescaleDB
:check_db
docker exec surge_wsi_timescaledb pg_isready -U surge_wsi -d surge_wsi >nul 2>&1
if errorlevel 1 (
    echo Waiting for TimescaleDB...
    timeout /t 2 /nobreak >nul
    goto check_db
)
echo [OK] Docker services ready

REM ============================================
REM Step 3: Check MT5
REM ============================================
echo.
echo [3/4] Checking MT5 Terminal...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] MT5 Terminal already running
) else (
    echo Starting MT5 Terminal...
    start "" "C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"
    timeout /t 10 /nobreak >nul
    echo [OK] MT5 Terminal started
)

REM ============================================
REM Step 4: Launch Both Terminals
REM ============================================
echo.
echo [4/4] Launching terminals...

REM Start Data Monitor in new window
echo Starting Data Monitor...
start "SURGE-WSI Data Monitor" cmd /k "cd /d %~dp0 && python data_monitor.py"

REM Wait a moment
timeout /t 3 /nobreak >nul

REM Start Trading System in new window
echo Starting Trading System...
start "SURGE-WSI Trading" cmd /k "cd /d %~dp0 && python main.py --demo --interval 60"

echo.
echo ============================================
echo  Both terminals launched!
echo ============================================
echo.
echo Terminal 1: SURGE-WSI Trading (main system)
echo Terminal 2: Data Monitor (MT5 + Database)
echo.
echo Press any key to close this launcher...
pause >nul
