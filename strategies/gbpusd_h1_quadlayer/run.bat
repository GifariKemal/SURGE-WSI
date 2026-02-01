@echo off
REM ============================================================
REM SURGE-WSI H1 v6.4 GBPUSD - QUAD-LAYER Filter
REM Run from: strategies/gbpusd_h1_quadlayer/
REM ============================================================

echo.
echo ============================================================
echo SURGE-WSI H1 v6.4 GBPUSD - QUAD-LAYER Filter
echo ============================================================
echo.
echo Layers:
echo   Layer 1: Monthly Profile (tradeable %%)
echo   Layer 2: Technical (ATR, Efficiency, ADX)
echo   Layer 3: Intra-Month Risk (percentage-based thresholds)
echo   Layer 4: Pattern Filter (rolling WR, direction)
echo.
echo Features:
echo   - Telegram Authentication (secured)
echo   - DST-Aware Session Detection
echo   - Scalable Risk Thresholds
echo.
echo Backtest Results (Jan 2025 - Jan 2026):
echo   - 173 trades, 45.7%% WR, PF 4.98
echo   - +$37,180 profit (+74.4%% return)
echo   - ZERO losing months (0/13)
echo.
echo ============================================================
echo.

REM Navigate to project root
cd /d "%~dp0..\.."

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Default settings
set MODE=--demo
set INTERVAL=300

REM Parse arguments
:parse
if "%~1"=="" goto run
if /i "%~1"=="--live" set MODE=--live
if /i "%~1"=="--demo" set MODE=--demo
if /i "%~1"=="--interval" (
    set INTERVAL=%~2
    shift
)
shift
goto parse

:run
echo Starting bot with:
echo   Mode: %MODE%
echo   Interval: %INTERVAL%s
echo.
echo Telegram Commands: /help, /status, /layers, /balance
echo.
echo Press Ctrl+C to stop
echo ============================================================
echo.

python strategies/gbpusd_h1_quadlayer/main_live.py %MODE% --interval %INTERVAL%

pause
