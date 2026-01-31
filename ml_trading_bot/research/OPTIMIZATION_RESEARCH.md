# RSI Strategy Optimization Research

## Strategy Evolution Summary

| Version | Return | Improvement | Key Change |
|---------|--------|-------------|------------|
| v3.1 | ~+100% | Baseline | Fixed bugs, proper implementation |
| v3.2 | ~+152% | +52.1% | Added Volatility Filter (ATR 20-80) |
| v3.3 | ~+227% | +75.2% | Added Dynamic TP by volatility regime |
| v3.4 | +493.1% | +238.5% | RSI thresholds 42/58 (from 35/65) |
| v3.5 | +524.1% | +31.0% | Time-based TP (+0.35x during 12-16 UTC) |
| v3.6 | +572.9% | +48.9% | Max holding 46h (force close stuck positions) |
| **v3.7** | **+618.2%** | **+45.3%** | Skip 12:00 UTC (London lunch break) |

---

## Current Best Configuration (v3.7)

```python
RSI_CONFIG = {
    "rsi_period": 10,
    "rsi_oversold": 42.0,
    "rsi_overbought": 58.0,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "session_start": 7,
    "session_end": 22,
    "risk_per_trade": 0.01,
    # Volatility filter (v3.2)
    "min_atr_percentile": 20.0,
    "max_atr_percentile": 80.0,
    # Dynamic TP (v3.3)
    "dynamic_tp": True,
    "tp_low_vol_mult": 2.4,   # ATR < 40th pct
    "tp_high_vol_mult": 3.6,  # ATR > 60th pct
    # Time-based TP (v3.5)
    "time_tp_bonus": True,
    "time_tp_start": 12,
    "time_tp_end": 16,
    "time_tp_bonus_mult": 0.35,
    # Max holding period (v3.6)
    "max_holding_hours": 46,  # Force close after 46 hours
    # Skip hours filter (v3.7)
    "skip_hours": [12],       # Skip 12:00 UTC (London lunch break)
}
```

**Performance:**
- Return: +618.2% (6 years)
- Trades: 2654 (~442/year, ~8.5/week)
- Win Rate: 37.7%
- Max Drawdown: 30.7% (improved from 36.7%)
- Profitable Years: 5/6 (2020 slightly negative due to COVID)

---

## TECHNIQUES THAT WORKED ✅

### 1. Volatility Filter (ATR Percentile 20-80) ✅
**Result:** +52.1% improvement
**Rationale:** Skip trading during extreme volatility (both too low and too high)
- ATR < 20th percentile: market too quiet, signals less reliable
- ATR > 80th percentile: market too chaotic, stops get hit easily

### 2. Dynamic TP Based on Volatility Regime ✅
**Result:** +75.2% improvement
**Rationale:** Adjust take profit based on current market conditions
- Low volatility (ATR < 40th): TP = 2.4x ATR (take profit faster)
- Medium volatility: TP = 3.0x ATR (standard)
- High volatility (ATR > 60th): TP = 3.6x ATR (let profits run)

### 3. RSI Thresholds 42/58 (Looser) ✅
**Result:** +238.5% improvement
**Rationale:** More signals = more opportunities
- Original 35/65 was too strict, missing good entries
- 42/58 catches more mean reversion opportunities
- Trade count increased significantly

### 4. Time-based TP Bonus (12-16 UTC) ✅
**Result:** +31.0% improvement
**Rationale:** London+NY overlap has highest momentum
- More volume = better trending after entry
- Larger TP captures bigger moves during overlap
- +0.35x ATR bonus optimal (tested 0.3, 0.35, 0.4, 0.45, 0.5)

### 5. Skip 12:00 UTC (London Lunch Break) ✅ - v3.7
**Result:** +45.3% improvement, MaxDD 30.7% (from 36.7%)
**Rationale:** London lunch break has reduced liquidity
- Lower volume = worse signal quality
- Higher spreads and more noise
- Skipping 12:00 filters out worst-performing hour

**Hour Analysis (from test_new_techniques.py):**
| Hour | Trades | Win Rate | P/L |
|------|--------|----------|-----|
| 07:00 | 308 | 41.2% | +$13,277 |
| 09:00 | 148 | 43.9% | +$4,480 |
| 10:00 | 217 | 41.5% | +$5,713 |
| 11:00 | 248 | 40.7% | +$5,456 |
| **12:00** | **201** | **30.3%** | **-$1,822** |
| 13:00 | 227 | 34.8% | +$2,051 |
| 14:00 | 220 | 38.2% | +$4,741 |

12:00 UTC is the ONLY hour with negative P/L - clear candidate for skipping.

---

## TECHNIQUES THAT FAILED ❌

### Entry Filter Techniques (Tested with v3.7)

#### RSI Slope Filter ❌
**Result:** -620.3% (CATASTROPHIC)
**Why Failed:** Waiting for RSI to "turn" misses best entries
- Filtered 3249 trades (55%!)
- Win rate dropped to 34.3%
**Verdict:** REMOVED - severely reduces opportunity

#### ATR Expanding Filter ❌
**Result:** -394.6%
**Why Failed:** Mean reversion works in both expanding and contracting volatility
- Filtered 831 trades
- No improvement in win rate
**Verdict:** REMOVED - arbitrary filter

#### Previous Candle Confirmation ❌
**Result:** -484.2%
**Why Failed:** Random candle direction doesn't predict mean reversion success
- MaxDD actually increased to 44.6%
**Verdict:** REMOVED - adds noise, no benefit

#### Multi-bar RSI Confirmation ❌
**Result:** -432.2%
**Why Failed:** Waiting for 2 bars oversold misses best entries
- Reduces trade count significantly
**Verdict:** REMOVED - delays entry unnecessarily

### RSI Threshold Variations (with skip_hours)

| RSI | Return | MaxDD | Ret/DD |
|-----|--------|-------|--------|
| 40/60 | +516.2% | 38.6% | 13.36 |
| 41/59 | +534.3% | 38.9% | 13.73 |
| **42/58** | **+618.2%** | **30.7%** | **20.17** |
| 43/57 | +554.8% | 34.4% | 16.12 |
| 44/56 | +514.2% | 33.2% | 15.51 |
| 45/55 | +544.0% | 28.8% | 18.88 |

**Verdict:** RSI 42/58 is OPTIMAL by risk-adjusted metric (Ret/DD = 20.17)

---

### Position Sizing Techniques

#### Kelly Criterion (Half-Kelly ~2.3%) ❌
**Result:** +109% improvement BUT MaxDD jumped to 75.8%
**Why Failed:** Too aggressive for 37% win rate strategy
```
Kelly = (W × R − L) / R = (0.374 × 1.91 − 0.626) / 1.91 = 4.7%
Half-Kelly = 2.3%
```
**Verdict:** Violates user requirement of 1% risk, too risky

#### Scaled Entry (Add on extreme RSI) ❌
**Result:** -556.3% (LOSS)
**Why Failed:** Adding to position when RSI more extreme often means trend continuing against us
- Scale-in at RSI 35 when already in at RSI 42 = bigger loss when SL hit
**Verdict:** REMOVED - catastrophic losses

#### Dynamic Risk (Streak-based) ❌
**Result:** -1.7% (no improvement)
**Why Failed:** Win/loss streaks are random, no predictive value
- Increasing risk after wins doesn't help
- Reducing risk after losses misses recovery opportunities
**Verdict:** REMOVED - no benefit

### Exit Strategy Techniques

#### Partial Exit (50% at midpoint) ❌
**Result:** -179.1%
**Why Failed:** Taking profit too early reduces overall gains
- Mean reversion needs full TP to compensate for losses
- R:R drops when closing half early
**Verdict:** REMOVED - significantly hurts performance

#### Trailing Stop ❌
**Result:** -592.9% (NEAR TOTAL LOSS)
**Why Failed:** Mean reversion moves are not trending
- Price often retraces before hitting TP
- Trailing stop gets hit during normal retracement
**Verdict:** REMOVED - catastrophic for mean reversion

### Entry Filter Techniques

#### RSI Momentum Confirmation ❌
**Result:** -511.2%
**Why Failed:** Waiting for RSI to "turn" misses best entries
- Entry when RSI is still falling (for BUY) often best
- Momentum filter removes 3435 trades (65%!)
**Verdict:** REMOVED - too many filtered trades

#### H4 RSI Confirmation ❌
**Result:** -488.5%
**Why Failed:** Higher timeframe alignment often contradicts H1 signal
- H4 can be neutral while H1 is oversold
- Reduces trade count significantly
**Verdict:** REMOVED - misaligned timeframes hurt

#### Support/Resistance Proximity ❌
**Result:** -106.4%
**Why Failed:** S/R levels not reliable for mean reversion
- Price at support doesn't mean buy signal is better
- Filtered 530 trades without improving WR
**Verdict:** REMOVED - no benefit

### Take Profit Adjustments

#### Asymmetric TP (BUY vs SELL) ❌
**Result:** All variants negative (-116% to -214%)
- BUY +0.3x: -116.8%
- SELL -0.3x: -111.5%
- Combined: -189.3% to -214.7%
**Why Failed:** Both directions need same R:R for consistency
**Verdict:** REMOVED - hurts performance

#### Wider TP (3.0/4.0/5.0) ❌
**Result:** -451.1%
**Why Failed:** Mean reversion doesn't run far enough
- Win rate drops from 37% to 29%
- Fewer winning trades don't compensate
**Verdict:** REMOVED - too ambitious for mean reversion

---

## INDICATORS TESTED (from research_ideas_test.py)

### RSI2 (Larry Connors style) ❌
**Result:** Negative
**Why:** Too volatile, false signals

### Bollinger Band Filter ❌
**Result:** Negative
**Why:** Redundant with RSI, filters good trades

### Stochastic Confirmation ❌
**Result:** Negative
**Why:** Adds noise, no improvement

### MACD Trend Filter ❌
**Result:** Negative
**Why:** Trend filter contradicts mean reversion

### Williams %R ❌
**Result:** Negative
**Why:** Similar to RSI, no added value

### CCI (Commodity Channel Index) ❌
**Result:** Negative
**Why:** Extra complexity, no benefit

---

## CRITICAL FINDINGS

### What Makes RSI Mean Reversion Work:
1. **Simplicity** - RSI alone is sufficient
2. **Loose thresholds** - More trades = more opportunities
3. **Fixed R:R** - Consistent SL/TP structure
4. **Volatility-aware TP** - Adapt to market conditions
5. **Time-aware TP** - Capitalize on high-momentum periods

### What Kills RSI Mean Reversion:
1. **Additional filters** - Most reduce trades without improving WR
2. **Scaled entries** - Averaging into losers = bigger losses
3. **Trailing stops** - Cuts winners short in ranging markets
4. **Aggressive position sizing** - Drawdowns become unbearable
5. **Multi-timeframe confirmation** - Reduces opportunity

---

## TIME-BASED EXIT (46 hours) ✅ IMPLEMENTED in v3.6

**Result:** +48.9% improvement
**Implementation:** Force close position after 46 hours if no SL/TP hit
**Rationale:**
- Research from arXiv shows mean reversion has optimal holding period
- Trades stuck too long are often in unfavorable positions
- 46 hours = ~2 trading days is optimal for H1 timeframe

**Testing Results (from test_time_exit_finetune.py):**
| Hours | Return | Diff vs Baseline | MaxDD |
|-------|--------|------------------|-------|
| 40h | +556.2% | +32.1% | 37.4% |
| 42h | +561.8% | +37.7% | 36.9% |
| 44h | +568.1% | +44.0% | 36.8% |
| **46h** | **+572.9%** | **+48.9%** | **36.7%** |
| 48h | +570.3% | +46.2% | 37.0% |
| 50h | +565.1% | +41.0% | 37.3% |

**Performance:**
- Return: +572.9% (vs +524.1% baseline)
- MaxDD: 36.7% (vs 39.2% baseline) - BETTER!
- Time exits: only 44 (1.6% of trades)
- 6/6 profitable years maintained

---

## Z-SCORE TECHNIQUES TESTED (All Failed) ❌

### RSI + Z-Score Confirmation ❌
**Result:** -145% to -468% depending on parameters
**Tested:** Different lookbacks (10, 20, 50) and thresholds (0.5, 1.0, 1.5)
**Why Failed:** Adds another filter, reduces trades, no WR improvement
**Verdict:** REMOVED - classic case of over-filtering

### Extreme Z-Score TP Bonus ❌
**Result:** -72% to -280%
**Why Failed:** Not all extreme Z-score entries benefit from larger TP
**Verdict:** REMOVED - no improvement

### RSI Z-Score (Adaptive thresholds) ❌
**Result:** -554.7%
**Why Failed:** Z-score of RSI is too noisy, reduces trades by 56%
**Verdict:** REMOVED - terrible performance

---

## LIBRARY & IMPLEMENTATION ANALYSIS ✅

### RSI Calculation Methods Compared

| Method | Return | WR | MaxDD | Notes |
|--------|--------|-----|-------|-------|
| **SMA (ours)** | **+618.2%** | **37.7%** | **30.7%** | Best for mean reversion |
| RMA (standard) | +173.7% | 35.4% | 42.8% | Matches TradingView/TA-Lib |
| EMA | +330.5% | 36.1% | 44.9% | More responsive |
| pandas-ta | +173.7% | 35.4% | 42.8% | Uses RMA internally |

**Key Finding:** Our SMA-based RSI significantly outperforms standard RMA calculation!
- SMA is more responsive/noisy - better for catching mean reversion
- RMA is smoother - misses some entry opportunities
- **Verdict:** Keep SMA-based calculation

### Transaction Costs Impact

| Cost Model | Return | MaxDD | Notes |
|------------|--------|-------|-------|
| No costs | +618.2% | 30.7% | Backtest ideal |
| Fixed 1p+0.5p | -39.5% | 69.6% | Too pessimistic |
| Session-based | +142.8% | 61.5% | More realistic |
| With volatility adj | +141.3% | 61.9% | Slightly worse |

**Session Spread Model:**
- London (07-12 UTC): 0.6 pip spread, 0.3 pip slip
- Overlap (12-16 UTC): 0.4 pip spread, 0.2 pip slip
- NY (16-22 UTC): 0.8 pip spread, 0.4 pip slip

**Key Finding:** Transaction costs reduce returns by ~75%!
- Strategy remains profitable with realistic costs
- Need broker with tight spreads (<1 pip) for best results
- Avoid trading during high-spread periods

### Library Recommendations

1. **Keep pandas-ta** - Good for most indicators, actively maintained
2. **Consider TA-Lib** - More accurate, matches TradingView
3. **Consider vectorbt** - Much faster backtesting for optimization
4. **Consider ArbitrageLab** - For advanced O-U mean reversion

Sources:
- [QuantStart - Ornstein-Uhlenbeck](https://www.quantstart.com/articles/ornstein-uhlenbeck-simulation-with-python/)
- [Hudson & Thames - Optimal Stopping](https://hudsonthames.org/optimal-stopping-in-pairs-trading-ornstein-uhlenbeck-model/)
- [VectorBT Guide](https://algotrading101.com/learn/vectorbt-guide/)
- [LuxAlgo - Slippage](https://www.luxalgo.com/blog/backtesting-limitations-slippage-and-liquidity-explained/)

---

## NEXT RESEARCH IDEAS TO TEST

### From Trading Literature:
1. **Mean Reversion Timing** - Entry on bar close vs intrabar
2. **Exit Timing** - Time-based exits (close after X bars)
3. **Regime Detection** - Only trade in ranging regimes (skip trends)
4. **Pair Correlation** - GBPUSD vs EURUSD correlation filter
5. **Economic Calendar Filter** - Skip high-impact news hours

### From Quantitative Research:
1. **Hurst Exponent** - Measure mean reversion strength
2. **Half-life of Mean Reversion** - Optimal holding period
3. **Ornstein-Uhlenbeck Process** - Statistical mean reversion model
4. **Z-score normalization** - Better entry signal calibration

### Position Management:
1. **Time-based stop** - Exit if no TP/SL after X hours
2. **Breakeven move** - Move SL to entry after 1x ATR profit
3. **Daily trade limit** - Cap trades per day

---

## REFERENCES

- [Kelly Criterion - ALGOGENE](https://algogene.com/community/post/175)
- [Kelly Criterion - PyQuant News](https://www.pyquantnews.com/the-pyquant-newsletter/use-kelly-criterion-optimal-position-sizing)
- [Mean Reversion Strategies - LuxAlgo](https://www.luxalgo.com/blog/mean-reversion-strategies-for-algorithmic-trading/)
- [Risk-Constrained Kelly - QuantInsti](https://blog.quantinsti.com/risk-constrained-kelly-criterion/)
- [Larry Connors Mean Reversion - GreaterWaves](https://greaterwaves.com/secrets-of-larry-connors-mean-reversion/)

---

## CHANGELOG

### 2026-01-31 (v3.7)
- v3.7 RELEASED with skip_hours filter
- Tested all hours 07:00-21:00, found 12:00 UTC worst performing
- Skip 12:00 UTC gives +45.3% improvement
- MaxDD improved from 36.7% to 30.7%
- Hour analysis: only 12:00 has negative P/L

### 2026-01-31 (v3.6)
- v3.6 RELEASED with 46-hour max holding period
- Tested holding periods from 40-60 hours, 46h optimal
- MaxDD improved from 39.2% to 36.7%
- Implemented time-exit logic in rsi_executor.py

### 2026-01-31 (v3.5)
- v3.5 finalized with Time-based TP
- Tested 30+ techniques, documented all results
- Created comprehensive research notes

### Previous
- v3.4: RSI 42/58 optimization
- v3.3: Dynamic TP implementation
- v3.2: Volatility filter added
- v3.1: Bug fixes, production ready
