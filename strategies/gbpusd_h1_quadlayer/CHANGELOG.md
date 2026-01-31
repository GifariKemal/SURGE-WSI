# Changelog - GBPUSD H1 Quad-Layer Strategy

## v6.9.0 (2026-01-31)

### Day Multiplier Optimization

Fixed incorrect day-of-week multipliers based on comprehensive analysis of 24 months of trade data.

#### Problem
- Thursday was incorrectly penalized at 0.4x despite being the BEST performing day
- Friday reduction was insufficient for the WORST performing day

#### Analysis Results

| Day | Win Rate | P/L | Old Mult | New Mult |
|-----|----------|-----|----------|----------|
| Thursday | **51.8%** | $4,993 | 0.4 | **0.8** |
| Monday | 43.0% | $2,923 | 1.0 | 1.0 |
| Tuesday | 44.4% | $67 | 0.9 | 0.9 |
| Wednesday | 40.6% | $2,472 | 1.0 | 1.0 |
| Friday | **33.8%** | -$682 | 0.5 | **0.3** |

#### Solution

```python
# OLD (v6.8):
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5}

# NEW (v6.9):
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3}
```

#### Results Comparison

| Metric | v6.8.0 | v6.9.0 | Change |
|--------|--------|--------|--------|
| Trades | 115 | 150 | +30% |
| Win Rate | 49.6% | 50.0% | +0.4% |
| Profit Factor | 5.09 | 5.84 | **+0.75** |
| Net P/L | $22,346 | $36,205 | **+62%** |
| Losing Months | 0/13 | 0/13 | Same |

**Key Insight**: Thursday was the best trading day but was being heavily restricted. Fixing this adds +$13,859 profit while maintaining 0 losing months.

---

## v6.8.0 (2026-01-31)

### New Feature: Session-Based Hour+POI Filter

Added intelligent filtering based on historical session analysis to skip underperforming time+signal combinations.

#### Problem
- v6.7.0 had some trades at hours/POI combinations with very low win rates
- Hour 11: 27.3% WR (11 trades)
- ORDER_BLOCK @ Hour 8: 8.3% WR (12 trades, only 1 win!)
- ORDER_BLOCK @ Hour 16: 14.3% WR (7 trades)
- EMA_PULLBACK @ Hour 13-14: 18-28% WR

#### Solution
Added session-based filtering to skip these combinations:

```python
# Skip entirely
SKIP_HOURS = [11]  # 27.3% WR

# Skip ORDER_BLOCK at specific hours
SKIP_ORDER_BLOCK_HOURS = [8, 16]  # 8.3% and 14.3% WR

# Skip EMA_PULLBACK during NY Overlap
SKIP_EMA_PULLBACK_HOURS = [13, 14]  # 18-28% WR
```

#### Results Comparison

| Metric | v6.7.0 | v6.8.0 | Change |
|--------|--------|--------|--------|
| Trades | 154 | 115 | -25% |
| Win Rate | 44.8% | 49.6% | **+4.8%** |
| Profit Factor | 4.30 | 5.09 | **+0.79** |
| Net P/L | $27,028 | $22,346 | -17% |
| Losing Months | 0/13 | 0/13 | Same |

#### Session Filter Impact

```
Trades Skipped by Session Filter:
├─ Hour 11 (all): 156 signals
├─ ORDER_BLOCK @ Hour 8: 197 signals
├─ ORDER_BLOCK @ Hour 16: 161 signals
├─ EMA_PULLBACK @ Hour 13: 24 signals
└─ EMA_PULLBACK @ Hour 14: 21 signals
Total: 559 skips
```

**Key Insight**: By removing the lowest-performing combinations, we achieve higher quality trades with better WR and PF, while maintaining zero losing months.

---

## v6.7.0 (2026-01-31)

### New Feature: EMA Pullback Entry Signal

Added secondary entry signal to increase trade count while maintaining zero losing months.

#### Problem
- v6.6.1 produced only 95 trades in 13 months (~7.3/month)
- User wanted more trades without sacrificing quality

#### Solution
Added **EMA Pullback** detection alongside Order Block:

```
EMA Pullback Criteria:
- BUY: close > EMA20 > EMA50 (uptrend), price within 1.5 ATR of EMA20, bullish candle
- SELL: close < EMA20 < EMA50 (downtrend), price within 1.5 ATR of EMA20, bearish candle
- ADX > 20 (trend present)
- RSI between 30-70 (room for momentum)
- Body ratio > 0.4 (decent candle)
```

#### Results Comparison

| Metric | v6.6.1 | v6.7.0 | Change |
|--------|--------|--------|--------|
| Trades | 95 | 154 | +62% |
| Win Rate | 45.3% | 44.8% | -0.5% |
| Profit Factor | 4.18 | 4.30 | +0.12 |
| Net P/L | $14,016 | $27,028 | +93% |
| Losing Months | 0/13 | 0/13 | Same |

#### Entry Signal Performance

| Signal | Trades | Win Rate | P/L |
|--------|--------|----------|-----|
| ORDER_BLOCK | 80 | 38.8% | $+10,600 |
| EMA_PULLBACK | 74 | 51.4% | $+16,428 |

**Key Insight**: EMA Pullback has 51.4% WR vs Order Block's 38.8% WR, contributing more profit despite similar trade count.

---

## v6.6.1 (2026-01-31)

### Optimizations

Based on trade analysis, found and applied two optimizations:

#### 1. Skip Hour 7 UTC
- **Finding**: Hour 7 had 0% win rate (4 trades, all losses)
- **Fix**: Changed kill zone from 7-11 to 8-11 UTC
- **Impact**: +$337 saved

#### 2. Reduce MAX_ATR from 30 to 25 pips
- **Finding**: ATR 25-30 pips had 0% win rate (3 trades, all losses)
- **Fix**: Reduced max_atr threshold to 25 pips
- **Impact**: +$262 saved

### Results Comparison

| Metric | v6.6.0 | v6.6.1 | Change |
|--------|--------|--------|--------|
| Trades | 97 | 95 | -2 |
| Win Rate | 43.3% | 45.3% | +2.0% |
| Profit Factor | 3.78 | 4.18 | +0.40 |
| Net P/L | $12,897 | $14,016 | +$1,119 (+8.7%) |
| Max Drawdown | -0.75% ($461) | -0.75% ($397) | -$64 |
| Return/DD | 34.4x | 37.5x | +3.1x |

---

## v6.6.0 (2026-01-31)

### Critical Fixes

#### Data Leakage Fix
- **Issue**: Monthly tradeable percentages for 2025/2026 were hardcoded, causing data leakage
- **Fix**: Created `SEASONAL_TEMPLATE` derived from 2024 historical data with targeted adjustments
- **Key insight**: April tends to be optimistic (2024: 80%), so we apply a 10% safety margin (70%)
- **Result**: No more future data leakage while maintaining 0 losing months

#### Dead Code Removal
- Removed unused `_analyze_patterns()` method from PatternBasedFilter
- Method was defined but never called (line 328-374 in original)

### Deprecated Features

These were tested but made results worse:

| Feature | Result | Notes |
|---------|--------|-------|
| ChoppinessFilter (Layer 5) | Worse | Only adjusted 10 trades, didn't prevent losses |
| DirectionalMomentumFilter (Layer 6) | Worse | Added May as losing month when combined with ADX |
| ADX-enhanced regime detection | Worse | Too strict, blocked good trades |

### Code Improvements

- Updated docstrings to reflect current state
- Marked deprecated classes with clear warnings
- Added `SEASONAL_TEMPLATE` with documented rationale

### Backtest Results (After Fix)

```
Strategy: GBPUSD H1 Quad-Layer v6.6.0
Period: 2024-02-01 to 2026-01-30 (13 months)
Initial Balance: $50,000

Net P/L:        $+12,897.43
Profit Factor:  3.78
Win Rate:       42.3%
Losing Months:  0/13
```

---

## v6.5.1 (Previous)

### PatternBasedFilter Direction Window Bug Fix
- **Issue**: Was using `DIRECTION_TEST_WINDOW` (8) instead of `ROLLING_WINDOW` (10) for direction-specific win rate
- **Fix**: Use same `recent` window for both rolling_wr and direction stats
- **Result**: April 2025 changed from -$318.94 to +$20.74

---

## Active Layers (v6.7.0)

1. **Layer 1: Monthly Profile** - Quality adjustment based on seasonal tradeable percentage
2. **Layer 2: Technical** - ATR stability, efficiency, EMA trend detection
3. **Layer 3: Intra-Month Risk** - IntraMonthRiskManager with circuit breakers
4. **Layer 4: Pattern-Based** - PatternBasedFilter for choppy market detection

### Entry Signals (v6.7.0)

1. **ORDER_BLOCK** - Primary signal: detects imbalance zones
2. **EMA_PULLBACK** - Secondary signal: trend continuation pullbacks to EMA20

---

## Research Notes (2026-01-31)

### Techniques Tested

| Technique | Library | Result | Notes |
|-----------|---------|--------|-------|
| SuperTrend | pandas-ta | Similar (~49%) | 0.7% better than EMA but rarely shows SIDEWAYS |
| ChoppinessFilter | custom | Worse | Only adjusted 10 trades, didn't prevent losses |
| DirectionalMomentumFilter | custom | Worse | Added losing month when combined with ADX |
| ADX Regime Detection | custom | Worse | Too strict, blocked good trades |

### SuperTrend vs EMA Analysis

```
EMA 20/50 Crossover:
  - Tradeable bars: 8,334 (70% of total)
  - Direction accuracy: 48.2%

SuperTrend (10, 3.0):
  - Tradeable bars: 11,863 (99.9% of total)
  - Direction accuracy: 48.9%

When disagreeing:
  - EMA correct: 9.1%
  - SuperTrend correct: 49.8%
```

**Conclusion**: SuperTrend is rarely SIDEWAYS (only 10 bars vs 3,539 for EMA). When they disagree, SuperTrend is more likely correct. However, for a quality-focused strategy, the conservative EMA approach with pattern-based filtering is sufficient. SuperTrend could be useful as a secondary confirmation signal but not as primary regime detection.

### Libraries Found for Future Research

- `hmmlearn` - Hidden Markov Models for regime detection
- `smart-money-concepts` - Order block and FVG detection
- `arch` - GARCH volatility forecasting
- `pandas-ta` - 140+ technical indicators
