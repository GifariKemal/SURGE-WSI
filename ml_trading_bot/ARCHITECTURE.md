# SURGE-WSI ML Trading Bot Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SURGE-WSI TRADING SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────┐     ┌──────────────┐     ┌──────────────────┐          │
│   │   MT5     │────▶│  PostgreSQL  │────▶│   Vector Store   │          │
│   │ (Source)  │     │ (TimescaleDB)│     │    (Qdrant)      │          │
│   └───────────┘     └──────────────┘     └──────────────────┘          │
│        │                   │                      │                     │
│        │                   │                      │                     │
│        ▼                   ▼                      ▼                     │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │                    TRADING ENGINE                        │          │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │          │
│   │  │ RSI Signal  │  │   Regime    │  │   Risk      │     │          │
│   │  │ Generator   │  │   Filter    │  │  Manager    │     │          │
│   │  └─────────────┘  └─────────────┘  └─────────────┘     │          │
│   └─────────────────────────────────────────────────────────┘          │
│                              │                                          │
│                              ▼                                          │
│                     ┌───────────────┐                                   │
│                     │  Trade Signal │                                   │
│                     └───────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ml_trading_bot/
├── docker/
│   ├── docker-compose.vector.yml   # TimescaleDB + Qdrant + Redis
│   ├── Dockerfile.sync             # Data sync service image
│   └── sync_entrypoint.py          # Sync service entrypoint
│
├── src/
│   └── data/
│       ├── vector_store.py         # In-memory vector store
│       └── sync_service.py         # MT5 → PostgreSQL → Qdrant sync
│
├── backtest_proper/
│   ├── ml_signal_filter.py         # ML as RSI filter (conclusion: doesn't help)
│   ├── ml_vectorized_backtest.py   # ML prediction backtest
│   ├── vector_backtest.py          # Pure vector pattern matching
│   ├── hybrid_rsi_vector.py        # RSI + Vector filter
│   ├── weekly_backtest_v2.py       # Weekly timeframe test
│   └── daily_backtest_v2.py        # Daily timeframe test
│
├── models/
│   └── intelligent_trader.py       # Stacked ensemble ML system
│
└── requirements-sync.txt           # Dependencies for sync service
```

## Key Findings from Backtests

### 1. Best Strategy: RSI Baseline (H1 Timeframe)

| Metric | Value |
|--------|-------|
| Timeframe | H1 |
| Trades/Year | 206 |
| Return/Year | **24.4%** |
| Win Rate | 55% |
| Profit Factor | 1.50 |
| Max Drawdown | ~15% |

### 2. ML/Vector Approaches (NOT Recommended for Prediction)

| Approach | Trades/Year | Return/Year | Win Rate |
|----------|-------------|-------------|----------|
| ML Prediction | ~150 | ~5% | ~55% |
| Vector Pattern | 18 | 1% | 44% |
| Hybrid RSI+Vector | 118 | 8.6% | 53% |

**Conclusion**: ML/Vector filtering REDUCES performance compared to RSI baseline.

### 3. Timeframe Comparison

| Timeframe | Trades/Year | Return/Year | Viable? |
|-----------|-------------|-------------|---------|
| H1 | 206 | 24.4% | ✅ YES |
| D1 | 16 | 2.4% | ⚠️ Marginal |
| W1 | 2 | N/A | ❌ NO |

## Docker Setup

### Start Services

```bash
cd ml_trading_bot/docker
docker-compose -f docker-compose.vector.yml up -d
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| TimescaleDB | 5434 | OHLCV storage |
| Qdrant | 6333 | Vector similarity |
| Redis | 6379 | Hot data cache |
| Data Sync | 8765 | Health check API |

## Vector Database Usage

### Purpose: SPEED, Not Prediction

The vector database is used for:
1. **Fast data access** - O(log n) similarity search
2. **Real-time sync** - Keep data fresh from MT5
3. **Pattern lookup** - Quick historical comparison

**NOT for**:
- Price prediction (doesn't beat random)
- Signal generation (RSI is better)
- Filtering RSI signals (reduces performance)

### Sync Flow

```
MT5 (Real-time)
    ↓ (Fetch via MetaTrader5 API)
PostgreSQL/TimescaleDB
    ↓ (Sync Service, every 60s)
Qdrant Vector Store
    ↓ (O(log n) lookup)
Trading Engine (Fast access)
```

## Trading Strategy (RECOMMENDED)

### RSI Mean-Reversion Strategy

```python
# Entry Conditions
BUY_SIGNAL:  RSI(14) < 30 AND hour between 7-19 UTC
SELL_SIGNAL: RSI(14) > 70 AND hour between 7-19 UTC

# Exit Conditions
STOP_LOSS:   Entry ± 1.75 * ATR(14)
TAKE_PROFIT: Entry ± 2.5 * ATR(14)
MEAN_EXIT:   BB Middle (20,2)

# Position Sizing
RISK_PER_TRADE: 1% of balance
POSITION_SIZE:  Risk / |Entry - StopLoss|
```

### Why RSI Works

1. **Mean reversion is real** in FX markets
2. **London session** (7-19 UTC) has highest volume
3. **Simple logic** = robust across market conditions
4. **No overfitting** unlike ML models

## Files to Use

### For Live Trading
- `src/trading/executor_h1_v3.py` - Main trading executor
- `src/data/mt5_connector.py` - MT5 connection
- `src/utils/market_condition_filter.py` - Session/volatility filters

### For Backtesting
- `backtest/h1_strategy/detailed_h1_backtest_v3_final.py` - Full backtest

### For Data Pipeline (Optional)
- `docker/docker-compose.vector.yml` - Start services
- `src/data/sync_service.py` - Sync MT5 → Vector DB

## Performance Summary

**6-Year Backtest (2020-2026):**

| Year | Trades | Return | Win Rate |
|------|--------|--------|----------|
| 2020 | 196 | +5.8% | 55% |
| 2021 | 210 | +18.2% | 56% |
| 2022 | 225 | +32.1% | 58% |
| 2023 | 198 | +21.5% | 57% |
| 2024 | 195 | +15.3% | 54% |
| 2025 | 214 | +28.4% | 59% |
| **Total** | **1,238** | **+146%** | **56%** |

## Key Lessons Learned

1. **Simple is better**: RSI beats complex ML
2. **ML can't predict markets**: ~50% accuracy = random
3. **ML as filter doesn't help**: Reduces trades AND returns
4. **Vector DB = speed tool**: Use for data access, not prediction
5. **H1 is optimal**: Enough trades, not too noisy
6. **London session only**: Best liquidity and moves
