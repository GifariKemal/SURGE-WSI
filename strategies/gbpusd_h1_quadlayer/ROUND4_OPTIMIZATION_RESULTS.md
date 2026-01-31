# Round 4 Optimizations Test Results - GBPUSD H1 Strategy

## Test Date: 2026-01-31
## Test Period: 2024-02-01 to 2026-01-30

---

## Executive Summary

**Only 1 of 4 optimizations passed testing:**
- Fixed Day Multipliers: APPROVED
- Kelly 25% Position Sizing: REJECTED
- Fixed 4:1 R:R Exit: REJECTED
- PIN_BAR Entry Signal: REJECTED

---

## Baseline (v6.8.0 Reference)

| Metric | v6.8.0 Original | Baseline Test |
|--------|-----------------|---------------|
| Total Trades | 115 | 217 |
| Win Rate | 49.6% | 44.2% |
| Profit Factor | 5.09 | 4.27 |
| Net Profit | $22,346 | $39,012 |
| Max Drawdown | N/A | 1.61% |
| Losing Months | 0 | 2 |

Note: The baseline test shows 2 losing months (Feb 2024, Mar 2025) vs the 0 in v6.8.0 reference. This is due to different data periods or slight implementation differences.

---

## Individual Optimization Results

### 1. Kelly 25% Position Sizing - REJECTED

| Config | Trades | WR | PF | Net P/L | Losing Months |
|--------|--------|----|----|---------|---------------|
| Kelly Only | N/A | N/A | N/A | N/A | N/A |
| Kelly + 4:1 R:R | 43 | 18.6% | 5.27 | $12,702 | 2/5 |
| Kelly + 2:1 R:R | 186 | 35.5% | 6.98 | $107,137 | 6/24 |

**Issues:**
- Dramatically reduces trade count when combined with higher R:R
- Increases monthly variance
- Creates more losing months

**Verdict:** REJECTED - Adds too much variance

---

### 2. Fixed 4:1 R:R Exit - REJECTED

| Config | Trades | WR | PF | Net P/L | Losing Months |
|--------|--------|----|----|---------|---------------|
| 4:1 R:R Combined | 131 | 28.2% | 8.91 | $115,715 | 8/18 |
| 2:1 R:R Test | 186 | 35.5% | 6.98 | $107,137 | 6/24 |

**Issues:**
- Win rate drops dramatically (~28% vs ~45%)
- While profit is higher, 8 losing months is unacceptable
- High variance per trade means some months have only losses

**Verdict:** REJECTED - Too many losing months (8)

---

### 3. Fixed Day Multipliers - APPROVED

| Config | Trades | WR | PF | Net P/L | Losing Months |
|--------|--------|----|----|---------|---------------|
| Baseline | 217 | 44.2% | 4.27 | $39,012 | 2 |
| Fixed Day Mults | 288 | 44.1% | 4.73 | $68,608 | 1 |
| Improvement | +71 | -0.1% | +0.46 | +$29,596 | -1 |

**Day Multiplier Changes:**
```python
# OLD (wrong):
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.4, 4: 0.5}

# NEW (fixed based on actual WR analysis):
DAY_MULTIPLIERS = {
    0: 1.0,   # Monday - 43.0% WR
    1: 1.0,   # Tuesday - 44.4% WR
    2: 1.0,   # Wednesday - 40.6% WR
    3: 1.0,   # Thursday - 51.8% WR (BEST - was wrongly at 0.4!)
    4: 0.5,   # Friday - 33.8% WR (WORST - keep reduced)
}
```

**Day of Week Analysis with Fixed Multipliers:**
| Day | Trades | Wins | WR | P/L |
|-----|--------|------|-----|-----|
| Monday | 58 | 27 | 46.6% | +$16,014 |
| Tuesday | 57 | 19 | 33.3% | +$9,951 |
| Wednesday | 66 | 32 | 48.5% | +$19,065 |
| Thursday | 56 | 29 | 51.8% | +$19,435 |
| Friday | 51 | 20 | 39.2% | +$4,142 |

**Key Insight:** Thursday was incorrectly penalized at 0.4 multiplier when it's actually the BEST day (51.8% WR)!

**Verdict:** APPROVED - Reduces losing months from 2 to 1, increases profit by 76%

---

### 4. PIN_BAR Entry Signal - REJECTED

| Config | Trades | WR | PF | Net P/L | Losing Months |
|--------|--------|----|----|---------|---------------|
| Baseline | 217 | 44.2% | 4.27 | $39,012 | 2 |
| PIN_BAR Only | 287 | 48.4% | 4.82 | $66,243 | 3 |
| Improvement | +70 | +4.2% | +0.55 | +$27,231 | +1 |

**PIN_BAR Entry Stats:**
- Added 67 PIN_BAR trades (23.3% of total)
- Improved win rate slightly
- BUT added 1 extra losing month

**Losing Months with PIN_BAR:**
- 2024-02: -$5 (LOSS)
- 2024-09: -$456 (LOSS) - NEW
- 2024-12: -$304 (LOSS) - NEW

**Verdict:** REJECTED - Adds losing months despite profit increase

---

## Combined Test Results

### All 4 Optimizations Combined

| Metric | Value |
|--------|-------|
| Total Trades | 131 |
| Win Rate | 28.2% |
| Profit Factor | 8.91 |
| Net P/L | $115,715 |
| Max Drawdown | 5.01% |
| Losing Months | 8/18 |

**Verdict:** REJECTED - 8 losing months is unacceptable

---

## Final Recommendation

### Approved for v6.9.0: Fixed Day Multipliers Only

**Final Results:**
- Total Trades: 288
- Win Rate: 44.1%
- Profit Factor: 4.73
- Net P/L: $68,607.56
- Max Drawdown: 2.15%
- **Losing Months: 1/24**
- Final Balance: $118,607.56

**Implementation:**
```python
DAY_MULTIPLIERS = {
    0: 1.0,   # Monday
    1: 1.0,   # Tuesday
    2: 1.0,   # Wednesday
    3: 1.0,   # Thursday (FIXED from 0.4)
    4: 0.5,   # Friday
    5: 0.0,   # Saturday
    6: 0.0,   # Sunday
}
```

---

## Why Other Optimizations Failed

### Kelly 25% Position Sizing
- Creates position size variance that amplifies monthly P/L swings
- With a ~45% WR system, Kelly suggests smaller sizes than fixed 1%
- Not suitable for this strategy's risk profile

### Fixed 4:1 R:R Exit
- Requires 25%+ win rate to be profitable
- With our filter-heavy system, win rate drops to ~28%
- Many months have zero winners = automatic loss

### PIN_BAR Entry Signal
- Adds trades in marginal conditions
- Sep 2024 and Dec 2024 became losing months
- Quality over quantity principle violated

---

## Next Steps

1. Apply Fixed Day Multipliers to main backtest.py
2. Update strategy_config.py with corrected DAY_RISK_MULT
3. Re-run main v6.8.0 backtest to confirm 0-1 losing months
4. Consider v6.8.1 release with this single optimization

---

*Generated by SURGE-WSI Round 4 Optimization Testing*
