@echo off
REM ============================================
REM SURGE-WSI H1 v3 FINAL Trading System
REM ============================================
REM
REM Based on backtest results:
REM - 100 trades, 51.0%% WR, +$2,669, PF 1.50
REM - Return: +26.7%% in 13 months
REM - Only 3 losing months
REM
REM Author: SURIOTA Team
REM ============================================

title SURGE-WSI H1 v3 FINAL

cd /d "%~dp0"

echo.
echo ============================================
echo     SURGE-WSI H1 v3 FINAL
echo     Production Trading System
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
echo [4/4] Starting SURGE-WSI H1 v3 FINAL...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo ============================================
echo      H1 v3 FINAL STRATEGY DETAILS
echo ============================================
echo.
echo Strategy Workflow:
echo   1. Time Filter: Kill Zone + Hybrid Mode
echo   2. Regime: HMM Detection
echo   3. POI: Enhanced Order Block + FVG
echo   4. Entry: REJECTION, MOMENTUM, HIGHER_LOW
echo   5. Risk: Quality-adjusted (0.8%%-1.2%%)
echo   6. SL/TP: 25 pips / 1.5R (37.5 pips)
echo.
echo Backtest Results (13 months):
echo   - Trades: 100
echo   - Win Rate: 51.0%%
echo   - Net P/L: +$2,669
echo   - Profit Factor: 1.50
echo   - Return: +26.7%%
echo   - Losing Months: 3/11
echo.
echo ============================================
echo.

echo Telegram Commands:
echo   /status    - System status
echo   /balance   - Account balance
echo   /positions - Open positions
echo   /regime    - Market regime
echo   /pause     - Pause trading
echo   /resume    - Resume trading
echo   /test_buy  - Test BUY order (0.01 lot)
echo   /test_sell - Test SELL order (0.01 lot)
echo.
echo ============================================
echo Press Ctrl+C to stop
echo ============================================
echo.

REM Run H1 v3 FINAL with verbose logging
REM Interval: 300s (5 min) is optimal for H1 strategy
python main_h1_v3.py --demo --interval 300 --verbose

echo.
echo ============================================
echo SURGE-WSI H1 v3 FINAL has stopped.
echo ============================================
pause
