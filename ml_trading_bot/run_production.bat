@echo off
echo ============================================================
echo HYBRID ML + RSI TRADING BOT - PRODUCTION
echo ============================================================
echo.
echo Strategy: HMM Regime Detection + RSI Mean Reversion
echo Validated: 2,684 trades, +173.25%% return, 55.7%% WR
echo.

cd /d %~dp0

REM Check if dry-run flag is passed
if "%1"=="--live" (
    echo MODE: LIVE TRADING
    echo WARNING: Real money will be used!
    echo.
    pause
    python production\live_executor.py --config config\production_config.yaml
) else (
    echo MODE: DRY RUN (no real trades)
    echo.
    python production\live_executor.py --dry-run --config config\production_config.yaml
)

pause
