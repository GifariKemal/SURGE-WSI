# Strategy Workflow - GBPUSD H1 Quad-Layer

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SURGE-WSI Trading Workflow                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   MT5 Terminal ──► Data Fetch ──► Signal Generation ──► Execution   │
│        │               │                │                   │        │
│        │               ▼                ▼                   ▼        │
│        │          TimescaleDB    Quad-Layer Filter     MT5 Order     │
│        │               │                │                   │        │
│        │               ▼                ▼                   ▼        │
│        └────────► Redis Cache    Quality Score ◄──── Risk Manager   │
│                                        │                             │
│                                        ▼                             │
│                                   Telegram Bot                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Trading Cycle (Every 5 Minutes)

### Phase 1: Data Collection

```
┌──────────────────────────────────────────────────────────────┐
│ 1. FETCH DATA                                                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   MT5.copy_rates_from_pos("GBPUSD", H1, 0, 100)              │
│                      │                                        │
│                      ▼                                        │
│   ┌─────────────────────────────────────┐                    │
│   │ OHLCV Data (100 bars)               │                    │
│   │ - Open, High, Low, Close, Volume    │                    │
│   │ - Timestamp                         │                    │
│   └─────────────────────────────────────┘                    │
│                      │                                        │
│                      ▼                                        │
│   Store in TimescaleDB + Redis Cache                         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 2: Technical Analysis

```
┌──────────────────────────────────────────────────────────────┐
│ 2. CALCULATE INDICATORS                                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   RSI (14)                                                   │
│   ├── RSI < 30 → Oversold (BUY signal)                       │
│   ├── RSI > 70 → Overbought (SELL signal)                    │
│   └── 30-70 → Neutral (no signal)                            │
│                                                               │
│   ATR (14)                                                   │
│   ├── Used for Stop Loss calculation                         │
│   ├── Used for position sizing                               │
│   └── Used for market condition assessment                   │
│                                                               │
│   Price Efficiency                                           │
│   ├── (Close - Open) / (High - Low)                          │
│   └── Measures trend strength                                │
│                                                               │
│   ADX (14)                                                   │
│   ├── ADX > 25 → Trending market                             │
│   └── ADX < 20 → Ranging market                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 3: Quad-Layer Quality Filter

```
┌──────────────────────────────────────────────────────────────┐
│ 3. QUAD-LAYER QUALITY FILTER                                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ LAYER 1: Monthly Profile                             │    │
│   │ ─────────────────────────────────────────────────── │    │
│   │ Based on historical tradeable_pct for each month    │    │
│   │                                                      │    │
│   │ tradeable_pct < 30%  → +50 quality (NO TRADE)       │    │
│   │ tradeable_pct < 40%  → +35 quality (HALT)           │    │
│   │ tradeable_pct < 50%  → +25 quality                  │    │
│   │ tradeable_pct < 60%  → +15 quality                  │    │
│   │ tradeable_pct < 70%  → +10 quality                  │    │
│   │ tradeable_pct < 75%  → +5 quality                   │    │
│   │ tradeable_pct >= 75% → +0 quality (good month)      │    │
│   └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ LAYER 2: Technical Indicators                        │    │
│   │ ─────────────────────────────────────────────────── │    │
│   │ Real-time market condition assessment               │    │
│   │                                                      │    │
│   │ GOOD market:                                        │    │
│   │   - ATR stable (0.8-1.2x average)                   │    │
│   │   - Efficiency > 0.3                                │    │
│   │   - ADX > 20                                        │    │
│   │   → Base quality = 60                               │    │
│   │                                                      │    │
│   │ NORMAL market:                                      │    │
│   │   - Mixed conditions                                │    │
│   │   → Base quality = 65                               │    │
│   │                                                      │    │
│   │ BAD market:                                         │    │
│   │   - High volatility or low efficiency               │    │
│   │   → Base quality = 80                               │    │
│   └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ LAYER 3: Intra-Month Dynamic Risk                    │    │
│   │ ─────────────────────────────────────────────────── │    │
│   │ Adjusts based on current month performance          │    │
│   │                                                      │    │
│   │ Monthly P&L Tracking:                               │    │
│   │   loss < -$150 → +5 quality                         │    │
│   │   loss < -$250 → +10 quality                        │    │
│   │   loss < -$350 → +15 quality                        │    │
│   │   loss < -$400 → STOP trading this month            │    │
│   │                                                      │    │
│   │ Consecutive Losses:                                 │    │
│   │   3+ losses → +5 quality                            │    │
│   │   6+ losses → STOP for the day                      │    │
│   └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ LAYER 4: Pattern-Based Choppy Market Detector        │    │
│   │ ─────────────────────────────────────────────────── │    │
│   │ Learns from recent trade patterns                   │    │
│   │                                                      │    │
│   │ Warmup Period:                                      │    │
│   │   First 15 trades → observe only                    │    │
│   │                                                      │    │
│   │ Rolling Win Rate (window=10):                       │    │
│   │   WR < 10% → HALT trading                           │    │
│   │   WR < 25% → reduce size to 60%                     │    │
│   │                                                      │    │
│   │ Direction Balance:                                  │    │
│   │   Both BUY and SELL fail 4+ times → HALT            │    │
│   │                                                      │    │
│   │ Recovery Mode:                                      │    │
│   │   After halt, need 1 win at 50% size to resume      │    │
│   └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ COMBINED QUALITY SCORE                               │    │
│   │ ─────────────────────────────────────────────────── │    │
│   │                                                      │    │
│   │ Total = Layer1 + Layer2 + Layer3 + Layer4           │    │
│   │                                                      │    │
│   │ Example (February, bad month):                      │    │
│   │   Layer 1: +15 (tradeable=55%)                      │    │
│   │   Layer 2: 65 (NORMAL market)                       │    │
│   │   Layer 3: +5 (2 consecutive losses)                │    │
│   │   Layer 4: +0 (no pattern issues)                   │    │
│   │   ────────────────────────────                      │    │
│   │   Total: 85                                         │    │
│   │                                                      │    │
│   │ Signal only passes if quality < required threshold  │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 4: Signal Generation

```
┌──────────────────────────────────────────────────────────────┐
│ 4. SIGNAL GENERATION                                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Check Kill Zone (London/NY sessions only)                  │
│                      │                                        │
│                      ▼                                        │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ RSI Signal Check                                     │    │
│   │                                                      │    │
│   │ BUY Signal:                                         │    │
│   │   - RSI crossed above 30 (oversold recovery)        │    │
│   │   - In Kill Zone (08:00-12:00 or 13:00-17:00 UTC)   │    │
│   │   - Quality score < threshold                       │    │
│   │                                                      │    │
│   │ SELL Signal:                                        │    │
│   │   - RSI crossed below 70 (overbought rejection)     │    │
│   │   - In Kill Zone                                    │    │
│   │   - Quality score < threshold                       │    │
│   └─────────────────────────────────────────────────────┘    │
│                      │                                        │
│                      ▼                                        │
│   Signal passed? ──► Yes ──► Proceed to Risk Management      │
│         │                                                     │
│         No                                                    │
│         │                                                     │
│         ▼                                                     │
│   Wait for next candle                                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 5: Risk Management

```
┌──────────────────────────────────────────────────────────────┐
│ 5. RISK MANAGEMENT                                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ Position Sizing                                      │    │
│   │                                                      │    │
│   │ Base Risk: 1% of account per trade                  │    │
│   │                                                      │    │
│   │ Stop Loss = Entry ± (ATR × 1.5)                     │    │
│   │ Take Profit = Entry ± (SL × 1.5)                    │    │
│   │                                                      │    │
│   │ Lot Size = (Account × Risk%) / (SL_pips × pip_value)│    │
│   │                                                      │    │
│   │ Max Loss Cap: 0.15% per trade                       │    │
│   │   If calculated loss > 0.15%, reduce lot size       │    │
│   └─────────────────────────────────────────────────────┘    │
│                      │                                        │
│                      ▼                                        │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ Validation Checks                                    │    │
│   │                                                      │    │
│   │ ✓ No existing position on GBPUSD                    │    │
│   │ ✓ Daily loss limit not reached                      │    │
│   │ ✓ Monthly loss limit not reached                    │    │
│   │ ✓ Lot size within broker limits                     │    │
│   │ ✓ Sufficient margin available                       │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 6: Order Execution

```
┌──────────────────────────────────────────────────────────────┐
│ 6. ORDER EXECUTION                                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   MT5.order_send({                                           │
│       action: TRADE_ACTION_DEAL,                             │
│       symbol: "GBPUSD",                                      │
│       volume: calculated_lot_size,                           │
│       type: ORDER_TYPE_BUY / ORDER_TYPE_SELL,                │
│       price: current_price,                                  │
│       sl: stop_loss_price,                                   │
│       tp: take_profit_price,                                 │
│       magic: 20250131,                                       │
│       comment: "SURGE-QuadLayer"                             │
│   })                                                         │
│                      │                                        │
│                      ▼                                        │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ Post-Execution                                       │    │
│   │                                                      │    │
│   │ 1. Log trade to database                            │    │
│   │ 2. Update Layer 4 statistics                        │    │
│   │ 3. Send Telegram notification                       │    │
│   │ 4. Update monthly P&L tracking                      │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 7: Position Management

```
┌──────────────────────────────────────────────────────────────┐
│ 7. POSITION MANAGEMENT (while trade is open)                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Every 5 minutes:                                           │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │ Check Position Status                                │    │
│   │                                                      │    │
│   │ If TP hit:                                          │    │
│   │   → Record profit                                   │    │
│   │   → Update win counter                              │    │
│   │   → Reset consecutive loss counter                  │    │
│   │                                                      │    │
│   │ If SL hit:                                          │    │
│   │   → Record loss                                     │    │
│   │   → Increment consecutive loss counter              │    │
│   │   → Update Layer 3 & 4 metrics                      │    │
│   │                                                      │    │
│   │ Position still open:                                │    │
│   │   → Monitor for manual close signals                │    │
│   │   → Check for emergency conditions                  │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Daily Schedule

```
┌────────────────────────────────────────────────────────────┐
│ DAILY TRADING SCHEDULE (UTC)                                │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ 00:00 - 07:59  │  No trading (Asian session)               │
│                │  System monitoring only                    │
│                │                                            │
│ 08:00 - 12:00  │  LONDON SESSION (Kill Zone 1)             │
│                │  Active signal scanning                    │
│                │  High priority period                      │
│                │                                            │
│ 12:00 - 12:59  │  Lunch break                              │
│                │  Reduced activity                          │
│                │                                            │
│ 13:00 - 17:00  │  NEW YORK SESSION (Kill Zone 2)           │
│                │  Active signal scanning                    │
│                │  London-NY overlap (15:00-17:00) = best    │
│                │                                            │
│ 17:00 - 23:59  │  No trading (end of day)                  │
│                │  System monitoring only                    │
│                │                                            │
└────────────────────────────────────────────────────────────┘
```

## Weekly Schedule

```
┌────────────────────────────────────────────────────────────┐
│ WEEKLY TRADING SCHEDULE                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ Monday      │  Normal trading (may skip first few hours)   │
│ Tuesday     │  Normal trading                              │
│ Wednesday   │  Normal trading                              │
│ Thursday    │  Normal trading                              │
│ Friday      │  Reduced after 18:00 UTC (50% position size) │
│ Saturday    │  No trading (market closed)                  │
│ Sunday      │  No trading (market closed)                  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Monthly Workflow

```
┌────────────────────────────────────────────────────────────┐
│ MONTHLY REVIEW WORKFLOW                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Month Start (Day 1)                                     │
│    - Reset monthly P&L counter                             │
│    - Reset consecutive loss counter                        │
│    - Load monthly tradeable_pct for Layer 1                │
│                                                             │
│ 2. During Month                                            │
│    - Monitor Layer 3 dynamic adjustments                   │
│    - Watch for circuit breaker triggers                    │
│                                                             │
│ 3. Month End                                               │
│    - Generate monthly report                               │
│    - Review Layer 4 pattern statistics                     │
│    - Analyze which days/hours performed best               │
│                                                             │
│ 4. Special Months                                          │
│    - December: Almost no trading (historically poor)       │
│    - Summer (Jul-Aug): May have reduced liquidity         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Error Handling

```
┌────────────────────────────────────────────────────────────┐
│ ERROR HANDLING WORKFLOW                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ MT5 Connection Lost:                                       │
│   → Retry connection every 30 seconds                      │
│   → Send Telegram alert after 3 failures                   │
│   → Continue monitoring existing positions                 │
│                                                             │
│ Database Connection Lost:                                  │
│   → Switch to in-memory mode                               │
│   → Queue data for later sync                              │
│   → Send Telegram alert                                    │
│                                                             │
│ Order Execution Failed:                                    │
│   → Log error details                                      │
│   → Send Telegram alert                                    │
│   → Do NOT retry automatically (avoid double orders)       │
│                                                             │
│ Unexpected Exception:                                      │
│   → Log full stack trace                                   │
│   → Send Telegram alert                                    │
│   → Continue main loop (don't crash)                       │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Performance Metrics

Based on 13-month backtest (Jan 2025 - Jan 2026):

| Metric | Value |
|--------|-------|
| Total Trades | 102 |
| Win Rate | 42.2% |
| Profit Factor | 3.57 |
| Net Profit | +$12,888.80 |
| Return | +25.8% |
| Max Drawdown | 0.75% |
| Sharpe Ratio | 7.69 |
| Recovery Factor | 27.97 |
| Losing Months | 0/13 |

## Key Success Factors

1. **Quality over Quantity** - Only 102 trades in 13 months (avg 8/month)
2. **Strict Filtering** - Quad-layer filter blocks 90%+ of potential signals
3. **Capped Losses** - Max 0.15% loss per trade protects capital
4. **Adaptive Risk** - Layer 3 & 4 reduce exposure during drawdowns
5. **No Overtrading** - Kill zones limit trading to best hours only
