@echo off
echo ============================================
echo SURGE-WSI Docker Services
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Starting Docker services...
docker-compose up -d

echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo ============================================
echo Service Status:
echo ============================================
docker-compose ps

echo.
echo ============================================
echo Connection Info:
echo ============================================
echo TimescaleDB: localhost:5434
echo   - User: surge_wsi
echo   - Password: surge_wsi_secret
echo   - Database: surge_wsi
echo.
echo Redis: localhost:6381
echo.
echo Adminer (DB UI): http://localhost:8081
echo   - Server: timescaledb
echo   - User: surge_wsi
echo   - Password: surge_wsi_secret
echo ============================================
echo.
pause
