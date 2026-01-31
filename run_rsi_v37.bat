@echo off
title RSI Mean Reversion v3.7 - Live Demo
echo.
echo ============================================================
echo   RSI MEAN REVERSION v3.7 - LIVE DEMO
echo ============================================================
echo.
echo Strategy: RSI(10) 42/58 Mean Reversion
echo Symbol: GBPUSD H1
echo Backtest: +618.2% Return, 30.7% MaxDD, 37.7% WR
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python main_rsi_v37.py

echo.
echo ============================================================
echo   Trading session ended
echo ============================================================
pause
