@echo off
REM ============================================================
REM SURGE-WSI GBPUSD H1 Quad-Layer Strategy v6.6 - LIVE
REM ============================================================
REM
REM Backtest Performance:
REM   - 97 trades, 43.3% WR, PF 3.78
REM   - +$12,897.43 profit (+25.8% return)
REM   - Max Drawdown: -0.75%
REM   - ZERO losing months (0/13)
REM
REM WARNING: This runs LIVE trading with real money!
REM ============================================================

echo.
echo ============================================================
echo SURGE-WSI GBPUSD H1 Quad-Layer v6.6 - LIVE MODE
echo ============================================================
echo.
echo WARNING: Ini akan trading dengan UANG ASLI!
echo.
echo Tekan Ctrl+C untuk cancel, atau
pause

cd /d %~dp0
python main.py --live --interval 300

pause
