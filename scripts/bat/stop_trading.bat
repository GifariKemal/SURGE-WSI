@echo off
REM ============================================
REM SURGE-WSI Trading System Stop Script
REM ============================================

title SURGE-WSI - Stopping
cd /d "%~dp0"

echo.
echo ============================================
echo     Stopping SURGE-WSI Trading System
echo ============================================
echo.

REM Kill Python process if running
echo [1/2] Stopping Python processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq SURGE-WSI*" >nul 2>&1
echo [OK] Python stopped

REM Stop Docker services
echo.
echo [2/2] Stopping Docker services...
docker-compose down
echo [OK] Docker services stopped

echo.
echo ============================================
echo SURGE-WSI has been stopped.
echo ============================================
echo.
pause
