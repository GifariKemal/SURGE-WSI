@echo off
REM ============================================================
REM SURGE-WSI H1 v6.4 GBPUSD Trading Bot Launcher
REM QUAD-LAYER Quality Filter
REM ============================================================

echo.
echo ============================================================
echo SURGE-WSI H1 v6.4 GBPUSD - QUAD-LAYER Filter
echo ============================================================
echo.
echo Layers:
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

python main_h1_v6_4_gbpusd.py %MODE% --interval %INTERVAL%

pause
