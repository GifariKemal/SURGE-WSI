@echo off
REM ============================================================
REM SURGE-WSI GBPUSD H1 Quad-Layer Strategy - DEMO MODE
REM ============================================================
REM
REM Enhanced startup script with:
REM   - Docker services check
REM   - MT5 connection & AutoTrading verification
REM   - Pre-flight system check
REM   - Telegram notifications
REM   - Comprehensive logging
REM
REM Account: Finex Demo (61045904)
REM ============================================================

setlocal EnableDelayedExpansion

REM ============================================================
REM CONFIGURATION
REM ============================================================
set "MT5_LOGIN=61045904"
set "MT5_PASSWORD=iy#K5L7sF"
set "MT5_SERVER=FinexBisnisSolusi-Demo"

set "TRADING_MODE=demo"
set "CHECK_INTERVAL=300"

REM Colors
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

REM ============================================================
REM HEADER
REM ============================================================
cls
echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  SURGE-WSI GBPUSD H1 Quad-Layer Strategy v6.6%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo   Mode:     %GREEN%DEMO%RESET%
echo   Account:  %MT5_LOGIN%
echo   Server:   %MT5_SERVER%
echo   Interval: %CHECK_INTERVAL%s
echo.
echo %CYAN%============================================================%RESET%
echo.

REM ============================================================
REM STEP 1: CHECK DOCKER SERVICES
REM ============================================================
echo %YELLOW%[STEP 1/4]%RESET% Checking Docker services...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo   %RED%[FAIL]%RESET% Docker is not running!
    echo   Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)
echo   %GREEN%[OK]%RESET% Docker is running

REM Check TimescaleDB
docker ps --filter "name=surge-timescaledb" --format "{{.Status}}" | findstr /C:"Up" >nul
if errorlevel 1 (
    echo   %YELLOW%[WARN]%RESET% TimescaleDB not running, starting...
    docker-compose -f "%~dp0..\..\..\docker-compose.yml" up -d surge-timescaledb
    timeout /t 5 /nobreak >nul
) else (
    echo   %GREEN%[OK]%RESET% TimescaleDB is running
)

REM Check Redis
docker ps --filter "name=surge-redis" --format "{{.Status}}" | findstr /C:"Up" >nul
if errorlevel 1 (
    echo   %YELLOW%[WARN]%RESET% Redis not running, starting...
    docker-compose -f "%~dp0..\..\..\docker-compose.yml" up -d surge-redis
    timeout /t 3 /nobreak >nul
) else (
    echo   %GREEN%[OK]%RESET% Redis is running
)

REM Check Qdrant (optional)
docker ps --filter "name=surge-qdrant" --format "{{.Status}}" | findstr /C:"Up" >nul
if errorlevel 1 (
    echo   %YELLOW%[INFO]%RESET% Qdrant not running (optional)
) else (
    echo   %GREEN%[OK]%RESET% Qdrant is running
)

echo.

REM ============================================================
REM STEP 2: SET ENVIRONMENT VARIABLES
REM ============================================================
echo %YELLOW%[STEP 2/4]%RESET% Setting environment variables...
echo.

REM MT5 credentials
set "MT5_LOGIN=%MT5_LOGIN%"
set "MT5_PASSWORD=%MT5_PASSWORD%"
set "MT5_SERVER=%MT5_SERVER%"

REM Trading mode
set "TRADING_MODE=%TRADING_MODE%"
set "TRADING_ENABLED=false"

echo   %GREEN%[OK]%RESET% MT5_LOGIN=%MT5_LOGIN%
echo   %GREEN%[OK]%RESET% MT5_SERVER=%MT5_SERVER%
echo   %GREEN%[OK]%RESET% TRADING_MODE=%TRADING_MODE%
echo.

REM ============================================================
REM STEP 3: RUN PRE-FLIGHT CHECK
REM ============================================================
echo %YELLOW%[STEP 3/4]%RESET% Running pre-flight system check...
echo.

cd /d "%~dp0"
python preflight_checker.py
set PREFLIGHT_RESULT=%ERRORLEVEL%

if %PREFLIGHT_RESULT% NEQ 0 (
    echo.
    echo %YELLOW%[WARNING]%RESET% Some pre-flight checks failed.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i "!CONTINUE!" NEQ "y" (
        echo Startup cancelled.
        pause
        exit /b 1
    )
)

echo.

REM ============================================================
REM STEP 4: START TRADING BOT
REM ============================================================
echo %YELLOW%[STEP 4/4]%RESET% Starting trading bot...
echo.
echo %CYAN%============================================================%RESET%
echo   TRADING BOT STARTING
echo %CYAN%============================================================%RESET%
echo.
echo   Press Ctrl+C to stop the bot gracefully.
echo.
echo   Telegram Commands:
echo     /status     - System status
echo     /balance    - Account balance
echo     /positions  - Open positions
echo     /market     - Market analysis
echo     /layers     - Filter layers status
echo     /autotrading- Check AutoTrading
echo     /help       - All commands
echo.
echo %CYAN%============================================================%RESET%
echo.

REM Create logs directory if not exists
if not exist "%~dp0logs" mkdir "%~dp0logs"

REM Get current date for log file
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set LOG_DATE=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%
set LOG_FILE=logs\demo_%LOG_DATE%.log

REM Run the trading bot with logging (PowerShell tee equivalent)
powershell -Command "& {python GBPUSD_H1_QuadLayer_Final.py --demo --interval %CHECK_INTERVAL% 2>&1 | Tee-Object -FilePath '%LOG_FILE%' -Append}"

echo.
echo %CYAN%============================================================%RESET%
echo   Trading bot stopped.
echo %CYAN%============================================================%RESET%
echo.

pause
