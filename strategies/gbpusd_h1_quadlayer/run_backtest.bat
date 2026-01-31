@echo off
REM ============================================================
REM SURGE-WSI GBPUSD H1 Quad-Layer Strategy - Backtest
REM ============================================================

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Project root is 2 levels up
for %%I in ("%SCRIPT_DIR%\..\.." ) do set "PROJECT_ROOT=%%~fI"

REM Log file with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set "TODAY=%DT:~0,8%"
set "LOG_DIR=%SCRIPT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\backtest_%TODAY%.log"

REM Create logs directory if not exists
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cls
echo.
echo ============================================================
echo SURGE-WSI GBPUSD H1 Quad-Layer - BACKTEST
echo ============================================================
echo.
echo Running 13-month backtest (Jan 2025 - Jan 2026)...
echo Log: %LOG_FILE%
echo.
echo ============================================================
echo.

REM Activate virtual environment if exists
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
)

REM Change to project root for imports to work
cd /d "%PROJECT_ROOT%"

REM Run backtest
echo [%date% %time%] Backtest started >> "%LOG_FILE%"
python "%SCRIPT_DIR%\backtest.py" 2>&1

echo.
echo [%date% %time%] Backtest complete.
echo.
echo Reports saved to:
echo   - %SCRIPT_DIR%\reports\
echo   - Telegram (if configured)
echo.
pause
