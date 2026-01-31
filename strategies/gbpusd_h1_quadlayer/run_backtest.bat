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
set "TODAY=%date:~10,4%%date:~4,2%%date:~7,2%"
set "LOG_DIR=%SCRIPT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\backtest_%TODAY%.log"

REM Create logs directory if not exists
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

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

REM Run backtest with logging
python "%SCRIPT_DIR%\backtest.py" 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"

echo.
echo [%date% %time%] Backtest complete.
echo Log saved to: %LOG_FILE%
echo.
echo Reports saved to:
echo   - strategies\gbpusd_h1_quadlayer\reports\
echo   - Telegram (if configured)
echo.
pause
