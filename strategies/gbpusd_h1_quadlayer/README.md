# SURGE-WSI GBPUSD H1 Quad-Layer Strategy

## Overview

This strategy uses a **QUAD-LAYER Quality Filter** designed for **ZERO losing months**.

| Metric | Value |
|--------|-------|
| Symbol | GBPUSD |
| Timeframe | H1 |
| Backtest Period | Jan 2025 - Jan 2026 |
| Total Trades | 102 |
| Win Rate | 42.2% |
| Profit Factor | 3.57 |
| Net Profit | +$12,888.80 (+25.8%) |
| Losing Months | 0/13 |

## Quad-Layer Quality Filter

### Layer 1: Monthly Profile
Uses historical `tradeable_pct` from market analysis:
- `< 30%`: +50 quality (NO TRADE)
- `< 40%`: +35 quality (HALT)
- `< 50%`: +25 quality
- `< 60%`: +15 quality
- `< 70%`: +10 quality
- `< 75%`: +5 quality

### Layer 2: Technical Indicators
Real-time assessment:
- ATR Stability
- Price Efficiency
- ADX/Trend Strength

### Layer 3: Intra-Month Dynamic Risk
Adjusts based on current month performance:
- Monthly P&L tracking
- Consecutive loss counter
- Auto-halt if loss > $400

### Layer 4: Pattern-Based Choppy Market Detector
- Rolling win rate tracking
- Direction balance (BUY/SELL)
- Warmup period (first 15 trades)

## MT5 Account

**IMPORTANT**: Use **MetaQuotes-Demo** account, NOT Finex!

```
MT5_SERVER=MetaQuotes-Demo
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Main entry point for live trading |
| `executor.py` | Trading executor with quad-layer filter |
| `backtest.py` | Backtest script |
| `config.env` | Environment variables template |
| `run.bat` | Launcher for live trading |
| `run_backtest.bat` | Launcher for backtest |

## Usage

### Live Trading

```batch
cd strategies\gbpusd_h1_quadlayer
run.bat
```

Or with arguments:
```batch
run.bat --demo --interval 300
run.bat --live --interval 60
```

### Backtest

```batch
cd strategies\gbpusd_h1_quadlayer
run_backtest.bat
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | System status & all layer info |
| `/balance` | Account balance & equity |
| `/positions` | View open positions |
| `/layers` | View all 4 layers status |
| `/market` | Market Analysis (detailed) |
| `/pause` | Pause auto trading |
| `/resume` | Resume auto trading |
| `/close_all` | Close all open positions |
| `/help` | Show all commands |

## Configuration

Copy `config.env` to project root `.env` or set environment variables:

```bash
# MT5 Connection - MetaQuotes Demo
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Logs

Logs are saved to `logs/` folder:
- `quadlayer_YYYYMMDD.log` - Live trading logs
- `backtest_YYYYMMDD.log` - Backtest logs

## Reports

Reports are saved to `reports/` folder and sent to Telegram:
- PNG summary image
- PDF detailed report (MT5 Strategy Tester format)
