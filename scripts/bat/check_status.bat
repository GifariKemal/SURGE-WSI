@echo off
REM ============================================
REM SURGE-WSI System Status Check
REM ============================================

title SURGE-WSI Status Check
cd /d "%~dp0"

echo.
echo ============================================
echo     SURGE-WSI System Status Check
echo ============================================
echo.

REM Check Docker
echo [Docker]
docker info >nul 2>&1
if errorlevel 1 (
    echo   Status: NOT RUNNING
) else (
    echo   Status: Running
)

REM Check Docker containers
echo.
echo [Docker Containers]
docker ps --filter "name=surge_wsi" --format "  {{.Names}}: {{.Status}}" 2>nul
if errorlevel 1 (
    echo   No containers found
)

REM Check TimescaleDB
echo.
echo [TimescaleDB]
docker exec surge_wsi_timescaledb pg_isready -U surge_wsi -d surge_wsi >nul 2>&1
if errorlevel 1 (
    echo   Status: NOT READY
) else (
    echo   Status: Ready (localhost:5434)
)

REM Check Redis
echo.
echo [Redis]
docker exec surge_wsi_redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo   Status: NOT READY
) else (
    echo   Status: Ready (localhost:6381)
)

REM Check MT5
echo.
echo [MT5 Terminal]
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo   Status: Running
) else (
    echo   Status: NOT RUNNING
)

REM Check Python
echo.
echo [Python]
python --version 2>nul
if errorlevel 1 (
    echo   Status: NOT FOUND
) else (
    echo   Status: Available
)

REM Show environment
echo.
echo [Environment]
if exist ".env" (
    echo   .env file: Found
    for /f "tokens=1,2 delims==" %%a in ('findstr /R "^TRADING_MODE=" .env') do echo   Trading Mode: %%b
    for /f "tokens=1,2 delims==" %%a in ('findstr /R "^SYMBOL=" .env') do echo   Symbol: %%b
    for /f "tokens=1,2 delims==" %%a in ('findstr /R "^TELEGRAM_ENABLED=" .env') do echo   Telegram: %%b
) else (
    echo   .env file: NOT FOUND
)

echo.
echo ============================================
echo.
echo Quick Commands:
echo   start_trading.bat      - Start DEMO trading
echo   start_live_trading.bat - Start LIVE trading
echo   stop_trading.bat       - Stop all services
echo.
pause
