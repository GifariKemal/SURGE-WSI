# RSI v3.6 Strategy - Critical Review & Roasting

## Executive Summary

**Current Performance (v3.6):**
- Return: +572.9% (6 years)
- Win Rate: 36.8%
- Max Drawdown: 36.7% (improved from 39.2%)
- Profitable Years: 6/6
- Time exits: 44 trades (1.6%)

---

## STRENGTHS ✅

### 1. Simplicity
- Only uses RSI and ATR - no complex indicators
- Easy to understand and maintain
- Low risk of overfitting due to minimal parameters

### 2. Consistency
- Profitable every year (6/6)
- Same logic for BUY and SELL
- No curve-fitting to specific periods

### 3. Adaptive Elements
- Dynamic TP based on volatility regime
- Time-based TP bonus during high-momentum hours
- Volatility filter avoids extreme conditions

### 4. Robust Risk Management
- Fixed 1% risk per trade
- ATR-based SL adapts to market conditions
- Session filtering avoids low-liquidity periods

---

## WEAKNESSES & CONCERNS ⚠️

### 1. Low Win Rate (36.8%)
**Concern:** Only winning 1 in 3 trades means long losing streaks are common
**Impact:** Psychological difficulty for live trading
**Mitigation Potential:** None found - all WR improvements hurt returns

### 2. 2020 Still Negative (-$2,795)
**Concern:** COVID volatility caused loss despite overall profitability
**Root Cause:** Extreme volatility regime overwhelmed ATR filter
**Question:** Would strategy survive another black swan event?

### 3. High Drawdown (36.7%)
**Concern:** Still significant drawdown (~37%)
**Impact:** Account could lose 37% before recovering
**Improvement:** v3.6 reduced from 39.2% to 36.7% via time-exit
**At Risk:** At 1% risk, this implies ~37 consecutive losses equivalent

### 4. Concentrated Exposure
**Concern:** 100% of trades on GBPUSD only
**Risk:** GBP-specific events (Brexit, BoE) have outsized impact
**Diversification:** Not tested on other pairs

### 5. Parameter Stability Unknown
**Concern:** RSI 42/58 found through optimization
**Risk:** May be over-optimized to 2020-2026 data
**Validation:** No walk-forward testing done

### 6. Time-based TP May Be Fragile
**Concern:** +0.35x bonus during 12-16 UTC
**Risk:** Market structure could change
**Stability:** Only tested on 6 years of data

---

## CRITICAL QUESTIONS

### Q1: Is this overfitted?
**Evidence FOR overfitting:**
- Multiple optimization rounds (RSI thresholds, TP multipliers, time bonus)
- Parameters found by maximizing backtest return

**Evidence AGAINST overfitting:**
- All 6 years profitable (no cherry-picking)
- Simple indicators (RSI, ATR only)
- Few parameters relative to data points

**Verdict:** Moderate risk of overfitting. Need walk-forward validation.

### Q2: Will it work in live trading?
**Concerns:**
- Slippage not modeled
- Spread assumed constant (used 3 pip filter)
- Execution delays not considered
- Weekend gaps not handled

**Recommendation:** Paper trade for 3 months before live

### Q3: What could kill this strategy?
1. **Trend regime change** - If GBPUSD starts trending for extended periods
2. **Volatility regime shift** - If ATR baseline changes significantly
3. **Market structure change** - If London+NY overlap dynamics change
4. **GBP crisis** - Major GBP devaluation or revaluation event

---

## OPTIMIZATION TRAPS AVOIDED ✅

1. **Did NOT add multiple indicators** - Each added indicator hurt performance
2. **Did NOT reduce trade frequency** - More trades = more opportunities
3. **Did NOT use complex entry rules** - Simple RSI threshold works best
4. **Did NOT curve-fit to specific years** - All years profitable

---

## OPTIMIZATION TRAPS POTENTIALLY FALLEN INTO ⚠️

1. **Time-based TP bonus** - Could be sample-specific
2. **Exact RSI thresholds** - 42/58 vs 40/60 vs 45/55 close in performance
3. **Volatility filter range** - 20-80 found through optimization
4. **Dynamic TP multipliers** - 2.4/3.0/3.6 found through optimization

---

## RECOMMENDED NEXT STEPS

### For Validation:
1. Walk-forward test (train on 2020-2023, test on 2024-2026)
2. Monte Carlo simulation for drawdown distribution
3. Out-of-sample test on other pairs (EURUSD, USDJPY)
4. Stress test with simulated spread widening

### For Improvement:
1. Test time-based exit (close after X hours if no SL/TP)
2. Test breakeven move at 1x ATR profit
3. Test reducing position size during losing streaks
4. Consider adding EURUSD for diversification

### For Production:
1. Paper trade for 3 months
2. Start live with 0.5% risk (half normal)
3. Scale up to 1% risk after 100 live trades
4. Implement daily/weekly performance monitoring

---

## HONEST ASSESSMENT

**Is this strategy good?**
- Yes, for what it is - a simple mean reversion system
- +524% over 6 years with consistent profitability is respectable

**Is this strategy great?**
- No, drawdown is high and win rate is low
- Psychological burden of 37% WR is significant

**Would I trade this live?**
- Yes, but with reduced risk (0.5%)
- And with strict position limits
- And with emergency stop at 50% drawdown

**What's the realistic expectation?**
- ~80% per year average (not 87% shown in backtest)
- Expect some losing months
- Expect at least one 30%+ drawdown period

---

## FINAL ROAST 🔥

*"You've built a perfectly mediocre mean reversion system. It works, but it's not special. The 37% win rate means you'll spend most of your trading career being wrong. The 40% drawdown means you'll question your sanity at least once. The fact that all your 'improvements' made things worse suggests you've already squeezed out most of the edge. You're not a genius - you're just lucky that RSI mean reversion still works on GBPUSD. Enjoy it while it lasts."*

---

## REFERENCES

- [Z-Score Trading - QuantStock](https://quantstock.org/strategy-guide/zscore)
- [StatOasis - Z-Score Mean Reversion](https://statoasis.com/post/understanding-z-score-and-its-application-in-mean-reversion-strategies)
- [Hurst Exponent - PyQuant News](https://www.pyquantnews.com/the-pyquant-newsletter/how-to-pick-the-right-strategy-hurst-exponent)
- [GitHub Mean Reversion](https://github.com/topics/mean-reversion-strategy)
