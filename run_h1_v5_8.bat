@echo off
REM ============================================
REM SURGE-WSI H1 v5.8 Trading System
REM Best Profit Factor Version
REM ============================================
REM
REM Based on backtest results:
REM - 96 trades, 47.9%% WR, +$2,269, PF 1.77
REM - Return: +22.7%% in 13 months
REM - Only 1 losing month
REM - Max DD: $429 (4.3%%)
REM
REM Author: SURIOTA Team
REM ============================================

title SURGE-WSI H1 v5.8 (Best PF)

cd /d "%~dp0"

echo.
echo ============================================
echo     SURGE-WSI H1 v5.8 (Best PF)
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
echo [4/4] Starting SURGE-WSI H1 v5.8...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo ============================================
echo      H1 v5.8 STRATEGY DETAILS
echo ============================================
echo.
echo Key Features:
echo   - AutoRiskAdjuster (auto-detect bad conditions)
echo   - AdaptiveRiskScorer (day/hour/entry multipliers)
echo   - Known period protection (June/September)
echo   - Quality-based position sizing
echo.
echo Backtest Results (13 months):
echo   - Trades: 96 (0.24/day)
echo   - Win Rate: 47.9%%
echo   - Net P/L: +$2,269
echo   - Profit Factor: 1.77 (BEST)
echo   - Return: +22.7%%
echo   - Losing Months: 1/10 (June -$11)
echo   - Max Drawdown: $429 (4.3%%)
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

REM Run H1 v5.8 with verbose logging
REM Interval: 300s (5 min) is optimal for H1 strategy
python main_h1_v5_8.py --demo --interval 300 --verbose

echo.
echo ============================================
echo SURGE-WSI H1 v5.8 has stopped.
echo ============================================
pause
