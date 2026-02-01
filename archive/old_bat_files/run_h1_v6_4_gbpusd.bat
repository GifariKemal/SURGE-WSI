@echo off
title SURGE-WSI H1 v6.4 GBPUSD Live Trading
color 0A

echo ============================================================
echo  SURGE-WSI H1 v6.4 GBPUSD - LIVE TRADING
echo ============================================================
echo.
echo  Strategy: Dual-Layer Quality Filter
echo  Symbol: GBPUSD
echo  Timeframe: H1
echo  Risk: 1%% per trade
echo.
echo  Backtest Performance (Jan 2024 - Jan 2025):
echo    - 147 trades, 0.40/day
echo    - 42.2%% WR, PF 3.98
echo    - +$23,394 (+46.8%% return)
echo    - ZERO losing months
echo.
echo ============================================================
echo.
echo  WARNING: This will run in LIVE mode with real money!
echo.
echo ============================================================
echo.

set /p confirm="Type 'YES' to confirm LIVE trading: "
if /i not "%confirm%"=="YES" (
    echo.
    echo Cancelled. Use run_h1_v6_4_gbpusd_demo.bat for demo mode.
    pause
    exit /b
)

echo.
echo Starting LIVE trading...
echo.

cd /d "%~dp0"
python main_h1_v6_4_gbpusd.py --live --interval 60

echo.
echo Trading stopped.
pause
