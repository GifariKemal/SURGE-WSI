# SURGE-WSI EA (Expert Advisor)

Simplified MQL5 Expert Advisor for MetaTrader 5 based on SURGE-WSI Python trading system.

## Overview

This EA implements the core trading logic of SURGE-WSI without ML/HMM components, using pure Price Action with Supply/Demand zones.

## Features

### Core Strategy
- **Supply/Demand Zone Detection**: RBD (Supply) and DBR (Demand) patterns
- **Zone Quality Scoring**: 0-100 based on departure strength, base tightness, and session quality
- **Entry Confirmation**: Pin Bar and Rejection candle patterns

### Kill Zone Filter
- London Session: 08:00-12:00 UTC
- New York Session: 13:00-17:00 UTC
- London Close Overlap: 15:00-17:00 UTC

### Partial Take Profit (50/30/20)
- TP1 (1:1 R:R): Close 50%, move SL to breakeven
- TP2 (2:1 R:R): Close 30%, enable trailing stop
- TP3 (3:1 R:R): Close remaining 20%

### Risk Management
- Dynamic position sizing based on account risk %
- Recovery mode with reduced lot size after consecutive losses
- Daily loss limit protection
- Maximum drawdown protection

### December Filter
- Dec 1-14: Signal-only mode (no execution)
- Dec 15-31: Full monitoring pause

### Trade Modes
- **AUTO**: Full auto trading with normal risk
- **RECOVERY**: Reduced lot size (50%) after 3 consecutive losses
- **SIGNAL_ONLY**: Only log signals, no execution

## Installation

1. Copy `SURGE_WSI_EA.mq5` to:
   ```
   C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[Terminal ID]\MQL5\Experts\
   ```

2. Open MetaEditor and compile the EA

3. In MetaTrader 5:
   - Drag EA to GBPUSD H1 chart
   - Enable Auto Trading
   - Configure parameters

## Recommended Settings

### Conservative (Demo Testing)
```
RiskPercent = 0.5
MinRRRatio = 2.0
MinQualityScore = 60
MaxTradesPerDay = 2
```

### Standard (Live Trading)
```
RiskPercent = 1.0
MinRRRatio = 1.5
MinQualityScore = 50
MaxTradesPerDay = 3
```

### Aggressive (Experienced)
```
RiskPercent = 1.5
MinRRRatio = 1.0
MinQualityScore = 40
MaxTradesPerDay = 5
```

## Parameters

### Risk Management
| Parameter | Default | Description |
|-----------|---------|-------------|
| RiskPercent | 1.0 | Risk per trade (%) |
| RecoveryRiskPercent | 0.5 | Risk in recovery mode |
| MinRRRatio | 1.5 | Minimum Risk:Reward |
| MaxTradesPerDay | 3 | Maximum trades per day |
| MaxDrawdownPercent | 10.0 | Pause if drawdown exceeds |
| MaxConsecutiveLosses | 3 | Enter recovery after N losses |
| DailyLossLimit | 3.0 | Daily loss limit % |

### Zone Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| SwingLookback | 5 | Bars for swing detection |
| MaxBaseCandles | 5 | Maximum base candles |
| MinImbalanceRatio | 1.5 | Min departure/base ratio |
| MinQualityScore | 50 | Minimum zone quality |
| MaxZoneAge | 200 | Max zone age in bars |
| MaxTestCount | 2 | Max zone retests |

### Partial TP
| Parameter | Default | Description |
|-----------|---------|-------------|
| UsePartialTP | true | Enable partial TP |
| TP1_RR | 1.0 | First TP at R:R |
| TP1_Percent | 50.0 | Close % at TP1 |
| TP2_RR | 2.0 | Second TP at R:R |
| TP2_Percent | 30.0 | Close % at TP2 |
| TP3_RR | 3.0 | Third TP at R:R |

## Comparison: EA vs Python System

| Feature | Python System | EA |
|---------|---------------|-----|
| Supply/Demand Zones | Yes | Yes |
| HMM Regime Detection | Yes | No |
| Kalman Filter | Yes | No |
| Kill Zone | Yes | Yes |
| Partial TP | Yes | Yes |
| December Filter | Yes | Yes |
| Recovery Mode | Yes | Yes |
| Telegram Integration | Yes | No |
| Database Logging | Yes | No |

## Backtest Tips

1. Use H1 timeframe for GBPUSD
2. Set modeling to "Every tick based on real ticks" for accuracy
3. Test period: 2024-01 to present
4. Initial deposit: $10,000 (for realistic lot sizing)

## Changelog

### v1.00 (2025-01-25)
- Initial release
- Core S/D zone detection
- Kill zone filter
- Partial TP (50/30/20)
- December filter
- Recovery mode

## License

Internal use only - SURIOTA Team
