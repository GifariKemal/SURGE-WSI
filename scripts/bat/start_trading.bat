@echo off
REM ============================================
REM SURGE-WSI Trading System Startup Script
REM ============================================
REM
REM This script will:
REM 1. Start Docker services (TimescaleDB, Redis)
REM 2. Wait for services to be healthy
REM 3. Launch MT5 Terminal (if not running)
REM 4. Run SURGE-WSI trading system
REM 5. Enable Telegram commands
REM
REM Author: SURIOTA Team
REM ============================================

title SURGE-WSI Trading System
cd /d "%~dp0"

echo.
echo ============================================
echo     SURGE-WSI Trading System Launcher
echo ============================================
echo.

REM ============================================
REM Step 1: Check Docker is running
REM ============================================
echo [1/5] Checking Docker...
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
echo [2/5] Starting Docker services...
docker-compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start Docker services!
    pause
    exit /b 1
)
echo [OK] Docker services started

REM Wait for services to be healthy
echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check TimescaleDB
:check_db
docker exec surge_wsi_timescaledb pg_isready -U surge_wsi -d surge_wsi >nul 2>&1
if errorlevel 1 (
    echo Waiting for TimescaleDB...
    timeout /t 3 /nobreak >nul
    goto check_db
)
echo [OK] TimescaleDB ready

REM Check Redis
docker exec surge_wsi_redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Redis not responding, continuing anyway...
) else (
    echo [OK] Redis ready
)

REM ============================================
REM Step 3: Launch MT5 Terminal
REM ============================================
echo.
echo [3/5] Checking MT5 Terminal...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] MT5 Terminal already running
) else (
    echo Starting MT5 Terminal...
    start "" "C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"
    echo Waiting for MT5 to initialize...
    timeout /t 15 /nobreak >nul
    echo [OK] MT5 Terminal started
)

REM ============================================
REM Step 4: Activate Python Environment
REM ============================================
echo.
echo [4/5] Activating Python environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [INFO] No virtual environment found, using system Python
)

REM ============================================
REM Step 5: Run SURGE-WSI
REM ============================================
echo.
echo [5/5] Starting SURGE-WSI Trading System...
echo.
echo ============================================
echo        SURGE-WSI is now RUNNING
echo ============================================
echo.
echo Telegram Commands:
echo   /status    - View system status
echo   /balance   - View account balance
echo   /positions - View open positions
echo   /regime    - View market regime
echo   /pois      - View active POIs
echo   /pause     - Pause trading
echo   /resume    - Resume trading
echo   /close_all - Close all positions
echo.
echo Press Ctrl+C to stop the system
echo ============================================
echo.

REM Run with --demo by default (change to --live for live trading)
python main.py --demo --interval 60

REM If Python exits, show message
echo.
echo ============================================
echo SURGE-WSI has stopped.
echo ============================================
pause
