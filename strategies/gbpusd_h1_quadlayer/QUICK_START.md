# Quick Start Guide - GBPUSD H1 Quad-Layer Strategy

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | With pip |
| MetaTrader 5 | Latest | MetaQuotes-Demo account |
| PostgreSQL | 14+ | Via Docker (recommended) |
| Redis | 7+ | Via Docker (recommended) |

## Step 1: Setup Environment (5 minutes)

### 1.1 Clone/Copy Project

```batch
:: Copy entire folder to your PC
xcopy /E /I "SURGE-WSI" "C:\Trading\SURGE-WSI"
cd C:\Trading\SURGE-WSI
```

### 1.2 Create Virtual Environment

```batch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 Start Docker Services

```batch
docker-compose up -d
```

This starts:
- TimescaleDB (PostgreSQL) on port 5434
- Redis on port 6381

## Step 2: Configure MT5 (3 minutes)

### 2.1 Open MetaTrader 5

1. Open MT5 (NOT Finex terminal!)
2. Login to **MetaQuotes-Demo** account
3. Enable **AutoTrading** (Ctrl+E or button on toolbar)

### 2.2 Verify Connection

```batch
python -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.account_info())"
```

Should show your MetaQuotes-Demo account info.

## Step 3: Configure Environment (2 minutes)

### 3.1 Edit .env File

Copy `strategies/gbpusd_h1_quadlayer/config.env` to project root `.env`:

```batch
copy strategies\gbpusd_h1_quadlayer\config.env .env
```

Edit `.env` with your credentials:

```ini
# MT5 - Leave empty for auto-connect to running terminal
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=MetaQuotes-Demo
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# Telegram (get from @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3.2 Get Telegram Credentials

1. Open Telegram, search **@BotFather**
2. Send `/newbot`, follow instructions
3. Copy the bot token to `TELEGRAM_BOT_TOKEN`
4. Add bot to your group/channel
5. Get chat ID: `https://api.telegram.org/bot<TOKEN>/getUpdates`

## Step 4: Initialize Database (1 minute)

```batch
python -c "from src.data.db_handler import DBHandler; import asyncio; asyncio.run(DBHandler().initialize())"
```

## Step 5: Run Backtest (Verify Setup)

```batch
cd strategies\gbpusd_h1_quadlayer
run_backtest.bat
```

Expected output:
- 102 trades, 42.2% WR, PF 3.57
- +$12,888.80 profit
- 0/13 losing months
- Report sent to Telegram

## Step 6: Start Live Trading

```batch
cd strategies\gbpusd_h1_quadlayer
run.bat
```

Or with arguments:

```batch
run.bat --demo              :: Demo mode (default)
run.bat --live              :: Live mode (real money!)
run.bat --interval 60       :: Check every 60 seconds
```

## Telegram Commands

Once running, control via Telegram:

| Command | Description |
|---------|-------------|
| `/status` | System status |
| `/balance` | Account balance |
| `/layers` | View all 4 filter layers |
| `/positions` | Open positions |
| `/pause` | Pause trading |
| `/resume` | Resume trading |
| `/help` | All commands |

## Troubleshooting

### MT5 Won't Connect

```
Error: IPC initialize failed
```

**Solution**:
1. Make sure MT5 is running
2. Make sure you're logged in
3. Run MT5 as Administrator

### Database Connection Failed

```
Error: connection refused
```

**Solution**:
```batch
docker-compose up -d
docker ps  :: Verify containers running
```

### Telegram Not Sending

```
Error: Chat not found
```

**Solution**:
1. Add bot to your group
2. Send a message in the group first
3. Get correct chat_id (negative for groups)

## File Locations

| What | Where |
|------|-------|
| Strategy files | `strategies/gbpusd_h1_quadlayer/` |
| Logs | `strategies/gbpusd_h1_quadlayer/logs/` |
| Reports | `strategies/gbpusd_h1_quadlayer/reports/` |
| Config | `.env` (project root) |

## Next Steps

1. Monitor first few trades via Telegram
2. Check logs for any errors
3. Review PDF reports for performance
4. Adjust risk settings in `executor.py` if needed

---

**Total Setup Time: ~15 minutes**

For detailed workflow, see [WORKFLOW.md](WORKFLOW.md)
