@echo off
REM ============================================================
REM SURGE-WSI GBPUSD H1 Quad-Layer Strategy v6.6 - DEMO
REM ============================================================
REM
REM Demo mode - no real trades, just signal scanning
REM ============================================================

echo.
echo ============================================================
echo SURGE-WSI GBPUSD H1 Quad-Layer v6.6 - DEMO MODE
echo ============================================================
echo.

cd /d %~dp0
python main.py --demo --interval 300

pause
