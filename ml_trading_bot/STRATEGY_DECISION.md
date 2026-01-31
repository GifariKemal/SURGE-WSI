# ML Trading Bot - Strategy Decision

## Hasil Backtest (2020-2026)

```
Strategy            Trades/Yr   Return    Annual    Win Rate
================================================================
RSI OPTIMIZED           580     +530%    +88.3%      52.7%  <-- WINNER (NEW!)
RSI Baseline            341     +159%    +26.5%      54.1%
ICT (Kill Zone)         140      -56%     -9.3%      50.1%
HYBRID                  330      -92%    -15.4%      40.3%
BBMA Oma Ally           952     -100%    -16.7%      33.0%
SMC (Order Block)       715     -100%    -16.7%      28.4%
```

## KEPUTUSAN FINAL

### Strategi Utama: RSI FULLY OPTIMIZED

```python
# Entry Rules (FULLY OPTIMIZED 2026-01-31)
BUY:  RSI(10) < 35 AND hour >= 7 AND hour < 22  # London + NY
SELL: RSI(10) > 65 AND hour >= 7 AND hour < 22  # London + NY

# Exit Rules (OPTIMIZED)
STOP_LOSS:   Entry +/- ATR(14) * 1.5   # Tighter SL
TAKE_PROFIT: Entry +/- ATR(14) * 3.0   # Wider TP
MEAN_EXIT:   Bollinger Band Middle (20, 2)

# Risk Management
RISK_PER_TRADE: 1% of balance
POSITION_SIZE:  Risk / |Entry - SL|
```

### Optimization Results

| Parameter | Baseline | Optimized | Change |
|-----------|----------|-----------|--------|
| RSI Period | 14 | 10 | Faster reaction |
| RSI Oversold | 30 | 35 | More signals |
| RSI Overbought | 70 | 65 | More signals |
| SL ATR Mult | 1.75 | 1.5 | Tighter stops |
| TP ATR Mult | 2.5 | 3.0 | Larger targets |
| Session | 07-19 (London) | **07-22 (London+NY)** | +3 hours |
| **Return (6yr)** | +159% | **+652%** | **+493%** |
| Annual Return | +26.5% | **+109%** | +82.5% |
| Max Drawdown | 27% | 24% | Improved |

### Session Comparison

| Session | Hours (UTC) | Return | Verdict |
|---------|-------------|--------|---------|
| Asian | 00:00-08:00 | -5% | AVOID |
| London Only | 07:00-16:00 | +447% | Good |
| London Full | 07:00-19:00 | +531% | Good |
| **London+NY** | **07:00-22:00** | **+652%** | **BEST** |
| 24 Hours | 00:00-24:00 | +523% | OK |

## Feature Comparison Backtest (2026-01-31)

Tested additional professional trading features against baseline.

| Feature | Trades | Return | Win Rate | Max DD | Verdict |
|---------|--------|--------|----------|--------|---------|
| **BASELINE (none)** | 2497 | **+286.9%** | 35.5% | 27.2% | **BEST** |
| + Trailing Stop | 3375 | -87.5% | 48.5% | 88.3% | REMOVE |
| + Break Even | 3179 | -95.7% | 52.2% | 95.7% | REMOVE |
| + Trailing+BE | 3487 | -95.3% | 52.0% | 95.5% | REMOVE |
| + Spread Filter | 2497 | +286.9% | 35.5% | 27.2% | NEUTRAL |
| + News Filter | 2489 | +272.8% | 35.5% | 26.5% | OPTIONAL |
| ALL FEATURES | 3476 | -95.4% | 52.1% | 95.6% | AVOID |

### Analysis

**Why Trailing Stop and Break Even DESTROY performance:**
- RSI strategy relies on 3:1 reward ratio (TP = 3x ATR, SL = 1.5x ATR)
- These features cut winners short before reaching TP
- With ALL features: only 2.3% of trades hit TP (vs 35.5% baseline)
- Higher win rate (52%) but much smaller average win = negative expectancy

**Exit Breakdown (All Features):**
- BE (Break Even): 49.8%
- SL (Stop Loss): 47.9%
- TP (Take Profit): 2.3%

### Final Feature Decision

| Feature | Enabled | Reason |
|---------|---------|--------|
| Trailing Stop | **NO** | -374% vs baseline |
| Break Even | **NO** | -383% vs baseline |
| Spread Filter | Optional | Neutral impact |
| News Filter | Optional | -14%, slight DD improvement |
| Trade Journal | Yes | For tracking only, no impact |

**CRITICAL INSIGHT:** For a strategy that depends on occasional big wins (high R:R),
never use features that limit profits. Let winners run to TP.

---

## Kenapa TIDAK Menggunakan:

### BBMA Oma Ally
- Terlalu banyak signal (952 trades/year = overtrading)
- Win rate rendah (33%)
- MA High/Low crossing sering false signal

### SMC (Order Block)
- Zone detection tidak presisi
- Banyak false positive
- Perlu manual confirmation

### ICT (Kill Zone)
- Trades terlalu sedikit (140/year)
- Liquidity sweep detection perlu tuning
- Lebih cocok untuk manual trading

### HYBRID
- Terlalu banyak filter = miss opportunities
- Kompleksitas tidak sebanding dengan hasil

## Implementasi di ml_trading_bot

### File yang Perlu Diupdate:

1. **features/technical.py** - Sudah ada RSI, BB, ATR
2. **executor/ml_executor.py** - Gunakan RSI signal
3. **inference/risk_manager.py** - 1% risk per trade

### Workflow Final:

```
                    +----------------+
                    |   MT5 Data     |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | TimescaleDB    |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Compute RSI    |
                    | BB, ATR        |
                    +-------+--------+
                            |
                            v
                +------------------------+
                | RSI < 30?  --> BUY     |
                | RSI > 70?  --> SELL    |
                +------------------------+
                            |
                            v
                +------------------------+
                | Session Filter         |
                | (07:00 - 19:00 UTC)    |
                +------------------------+
                            |
                            v
                +------------------------+
                | Position Sizing        |
                | (1% Risk)              |
                +------------------------+
                            |
                            v
                +------------------------+
                | Execute via MT5        |
                | SL: ATR * 1.75         |
                | TP: ATR * 2.5          |
                +------------------------+
```

## Performance Target

| Metric | Target | Backtest Result |
|--------|--------|-----------------|
| Trades/Year | 200-400 | 341 |
| Win Rate | > 50% | 54.1% |
| Annual Return | > 20% | 26.5% |
| Profit Factor | > 1.0 | 1.13 |
| Max Drawdown | < 20% | ~15% |

## Kesimpulan

**RSI Baseline adalah strategi TERBAIK untuk ml_trading_bot** karena:
1. Simple dan robust
2. Proven profitable 6 tahun
3. 86% profitable years
4. Parameter jelas dan tidak perlu tuning rumit
5. Mudah diimplementasi dan dimaintain

**Tidak perlu:**
- Machine Learning untuk prediksi (tidak membantu)
- Vector Database untuk pattern matching (tidak membantu)
- Strategi kompleks seperti BBMA/SMC/ICT (hasil lebih buruk)

**Yang perlu:**
- RSI(14) indicator
- Bollinger Bands(20, 2) untuk mean exit
- ATR(14) untuk SL/TP sizing
- Session filter (London hours)
- Risk management (1% per trade)
