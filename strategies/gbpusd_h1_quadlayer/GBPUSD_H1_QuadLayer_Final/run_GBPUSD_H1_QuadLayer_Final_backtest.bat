@echo off
REM ============================================================
REM SURGE-WSI H1 v6.4 GBPUSD - Backtest Runner
REM ============================================================

echo.
echo ============================================================
echo Running GBPUSD H1 Backtest (QUAD-LAYER Filter)
echo ============================================================
echo.

REM Navigate to project root
cd /d "%~dp0..\..\..\"

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run backtest
python strategies/gbpusd_h1_quadlayer/GBPUSD_H1_QuadLayer_Final/backtest_GBPUSD_H1_QuadLayer_Final.py

echo.
echo ============================================================
echo Backtest complete! Check reports/ folder for results.
echo ============================================================

pause
