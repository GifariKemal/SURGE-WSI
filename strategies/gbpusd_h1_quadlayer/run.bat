@echo off
REM ============================================================
REM SURGE-WSI GBPUSD H1 Quad-Layer Strategy
REM ============================================================
REM MT5 Account: MetaQuotes-Demo (NOT Finex!)
REM ============================================================

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Project root is 2 levels up
for %%I in ("%SCRIPT_DIR%\..\.." ) do set "PROJECT_ROOT=%%~fI"

REM Log file with timestamp (using wmic for consistent format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set "TODAY=%DT:~0,8%"
set "LOG_DIR=%SCRIPT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\quadlayer_%TODAY%.log"

REM Create logs directory if not exists
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cls
echo.
echo ============================================================
echo SURGE-WSI GBPUSD H1 Quad-Layer Strategy
echo ============================================================
echo.
echo [!] IMPORTANT: Ensure MT5 is running with MetaQuotes-Demo!
echo [!] DO NOT use Finex account for this strategy.
echo.
echo Quad-Layer Quality Filter:
echo   Layer 1: Monthly Profile (tradeable %%)
echo   Layer 2: Technical (ATR, Efficiency, ADX)
echo   Layer 3: Intra-Month Risk (consecutive losses)
echo   Layer 4: Pattern Filter (rolling WR, direction)
echo.
echo Backtest Results (Jan 2025 - Jan 2026):
echo   - 102 trades, 42.2%% WR, PF 3.57
echo   - +$12,888.80 profit (+25.8%% return)
echo   - ZERO losing months (0/13)
echo.
echo ============================================================
echo.

REM Activate virtual environment if exists
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
)

REM Default settings
set "MODE=--demo"
set "INTERVAL=300"

REM Parse arguments
:parse
if "%~1"=="" goto confirm
if /i "%~1"=="--live" set "MODE=--live"
if /i "%~1"=="--demo" set "MODE=--demo"
if /i "%~1"=="--interval" (
    set "INTERVAL=%~2"
    shift
)
if /i "%~1"=="--yes" goto run
if /i "%~1"=="-y" goto run
shift
goto parse

:confirm
echo Starting bot with:
echo   Mode: %MODE%
echo   Interval: %INTERVAL%s
echo   Log: %LOG_FILE%
echo.
echo Telegram Commands: /help, /status, /layers, /balance
echo.
set /p "CONFIRM=Start trading? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Aborted by user.
    pause
    exit /b 0
)

:run
echo.
echo [%date% %time%] Starting GBPUSD H1 Quad-Layer Strategy...
echo [%date% %time%] Log file: %LOG_FILE%
echo.
echo Press Ctrl+C to stop
echo ============================================================
echo.

REM Change to project root for imports to work
cd /d "%PROJECT_ROOT%"

REM Run with simple logging (append to file)
echo [%date% %time%] Bot started >> "%LOG_FILE%"
python "%SCRIPT_DIR%\main.py" %MODE% --interval %INTERVAL% 2>&1

echo.
echo [%date% %time%] Bot stopped.
echo Log saved to: %LOG_FILE%
pause
