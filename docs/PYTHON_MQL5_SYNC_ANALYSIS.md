# Python vs MQL5 Backtest Sync Analysis

## Summary
Both implementations have 150 trades but with **completely different entry times**, resulting in different outcomes:
- Python: 71 wins (47.3% WR), +$32,010
- MQL5: 57 wins (38.0% WR), -$531

## Root Cause: Entry Time Mismatch

| Trade # | Python Entry | MQL5 Entry | Gap |
|---------|-------------|------------|-----|
| 1 | 2025.01.08 09:00 | 2025.01.02 10:00 | 6 days |
| 2 | 2025.01.08 13:00 | 2025.01.02 11:00 | 6 days |
| 3 | 2025.01.09 09:00 | 2025.01.02 15:00 | 7 days |
| ... | ... | ... | ... |

**ALL 150 trades have different entry times!** This means they are trading completely different market conditions.

## Key Differences Found

### 1. Warmup Period
- **Python**: Starts at bar 100 (`for i in range(100, len(df))`)
  - First possible trade: ~Jan 6 (100 hours after Jan 2)
  - Actual first trade: Jan 8 (additional signal filtering)
- **MQL5**: No explicit warmup, starts immediately
  - First trade: Jan 2 at 10:00

### 2. Order Block Detection Window
- **Python**: Looks at last 30 bars of slice (`range(len(df)-30, len(df)-2)`)
- **MQL5**: Always looks at bars 1 and 2 (`iOpen/iClose(_Symbol, PERIOD_H1, 1/2)`)

### 3. EMA Calculation
- **Python**: Uses `ewm(span=20, adjust=False)` on the full series
- **MQL5**: Uses `iMA()` built-in function with same parameters

### 4. Entry Price
- **Python (v6.9)**: Uses Open of next bar (realistic entry)
- **MQL5**: Uses ASK/BID at signal bar close

## Solution Options

### Option A: Sync MQL5 to Python (Recommended)
1. Add 100-bar warmup period to MQL5
2. Match Order Block detection window
3. Both start trading from ~Jan 6 onwards

### Option B: Sync Python to MQL5
1. Remove 100-bar warmup from Python (start at bar 2)
2. Match Order Block detection to use only bars 1-2
3. Both start trading from Jan 2 onwards

### Option C: Accept Differences
Since live trading uses MQL5, the MQL5 results are "realistic".
Python backtest is for development/optimization only.

## Implementation for Option A (Sync MQL5 to Python)

In `GBPUSD_H1_QuadLayer_v69.mq5`, add warmup check:

```mql5
// At start of OnTick()
int totalBars = iBars(_Symbol, PERIOD_H1);
if(totalBars < 100)
{
    if(DebugMode) Print("Warmup: waiting for 100 bars, current=", totalBars);
    return;
}
```

## Expected Result After Sync
- Entry times should match between Python and MQL5
- Win rates should be within 1-2% of each other
- Remaining differences due to:
  - Spread (MQL5 uses real spread, Python uses 0)
  - Execution timing (tick-level vs bar-level)
  - Floating point precision

## Files Analyzed
- Python: `backtest/h1_strategy/detailed_h1_backtest_v6_4_dual_filter.py`
- MQL5: `mql5/Experts/GBPUSD_H1_QuadLayer_v69.mq5`
- Data: MetaQuotes GBPUSD H1 (synced via `scripts/sync_metaquotes.py`)
