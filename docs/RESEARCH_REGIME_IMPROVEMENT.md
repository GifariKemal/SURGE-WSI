# SURGE-WSI Research: Improving Regime Detection
## Analisis Root Cause & Solusi Berbasis Riset

---

## 1. Executive Summary

### Problem Statement
Backtest 2024 menunjukkan **4 bulan rugi** (Feb, Apr, May, Jun) dengan root cause utama:
- **Regime Detection Accuracy** sangat rendah di bulan rugi (20-44%)
- Dibandingkan bulan profit yang mencapai 51-83% accuracy

### Key Metrics Comparison

| Metric | Losing Months | Winning Months |
|--------|---------------|----------------|
| Regime Accuracy | 20-44% | 51-83% |
| Win Rate | 31.1% | 71.6% |
| Avg ATR | 26.3 pips | 31.8 pips |

---

## 2. Root Cause Analysis

### 2.1 Regime Detection Failure Pattern

```
Feb 2024: Market BULLISH (52.5% up days)
          -> HMM detected BEARISH
          -> 4/5 SELL trades FAILED
          -> Accuracy: 20%

Apr 2024: BUY trades: 0% WR (semua loss)
          SELL trades: 50% WR
          -> Regime salah arah untuk BUY
          -> Accuracy: 33%

May 2024: 6 consecutive losses on May 8th
          -> HMM tidak adaptif terhadap perubahan cepat
          -> Accuracy: 27%

Jun 2024: Market BEARISH (58% down days)
          -> BUY signals masih dihasilkan
          -> Accuracy: 44%
```

### 2.2 Karakteristik Bulan Rugi
1. **Volatilitas Lebih Rendah**: 26.3 vs 31.8 pips ATR
2. **Market Choppy/Sideways**: Trend tidak jelas
3. **Regime Transitions**: Perubahan regime yang cepat
4. **HMM Lag**: Model tidak cukup cepat beradaptasi

---

## 3. Proposed Solutions & Research References

### 3.1 Adaptive HMM / Online Learning

**Problem**: HMM saat ini di-train sekali dan tidak update real-time.

**Solution**: Implement Online HMM Learning yang update secara incremental.

**References**:
- [QuantInsti: Regime-Adaptive Trading with HMM + Random Forest](https://blog.quantinsti.com/regime-adaptive-trading-python/)
- [QuantStart: Market Regime Detection using HMM](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [GitHub: theo-dim/regime_detection_ml](https://github.com/theo-dim/regime_detection_ml) - HMM + SVM ensemble

**Implementation Ideas**:
```python
# Periodic retraining with rolling window
class AdaptiveHMM:
    def __init__(self, retrain_every=50, lookback=200):
        self.retrain_every = retrain_every
        self.lookback = lookback
        self.update_count = 0

    def update(self, new_data):
        self.update_count += 1
        if self.update_count % self.retrain_every == 0:
            # Retrain with recent data only
            recent_data = self.data[-self.lookback:]
            self.model.fit(recent_data)
```

---

### 3.2 Ensemble Methods: HMM + LSTM

**Problem**: Single HMM tidak capture temporal patterns dengan baik.

**Solution**: Combine HMM dengan LSTM untuk prediksi regime.

**References**:
- [GitHub: JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM](https://github.com/JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM)
  - GMM-HMM-LSTM: 76.16% accuracy
  - XGB-HMM-LSTM: 80.70% accuracy
- [GitHub: CryptoMarket_Regime_Classifier](https://github.com/akash-kumar5/CryptoMarket_Regime_Classifier)
  - Two-stage: HMM discovers regimes → LSTM predicts

**Architecture**:
```
Input Data → HMM (Unsupervised Regime Discovery)
          → Features + Regime Labels
          → LSTM (Supervised Regime Prediction)
          → Final Regime Output
```

---

### 3.3 Choppy Market Detection

**Problem**: Sistem trading saat regime sideways/choppy masih aktif.

**Solution**: Add explicit choppy market filter untuk SKIP trading.

**References**:
- [Medium: Detecting Ranging Markets with Choppiness Index](https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58)
- [Medium: Hurst Exponent for Market Predictability](https://medium.com/algorithmic-and-quantitative-trading/the-hurst-filter-explained-how-to-use-the-hurst-exponent-h-to-measure-market-predictability-and-d94371d0f33c)

**Indicators to Add**:

| Indicator | Trending | Choppy | Implementation |
|-----------|----------|--------|----------------|
| **Choppiness Index** | < 38.2 | > 61.8 | `ta.CHOP()` |
| **Hurst Exponent** | > 0.55 | < 0.45 | Custom calculation |
| **ADX** | > 25 | < 20 | `ta.ADX()` |

**Implementation**:
```python
def is_choppy_market(df, period=14):
    """Detect choppy/ranging market conditions"""
    # Choppiness Index
    chop = calculate_choppiness(df, period)

    # Hurst Exponent
    hurst = calculate_hurst(df['close'], period)

    # ADX
    adx = ta.ADX(df['high'], df['low'], df['close'], period)

    # Confluence scoring
    choppy_score = 0
    if chop > 61.8: choppy_score += 1
    if hurst < 0.45: choppy_score += 1
    if adx < 20: choppy_score += 1

    return choppy_score >= 2  # 2 out of 3 indicators agree
```

---

### 3.4 Adaptive Kalman Filter

**Problem**: Fixed Kalman parameters tidak optimal untuk semua market conditions.

**Solution**: Implement Adaptive Kalman yang adjust Q/R berdasarkan market volatility.

**References**:
- [GitHub: Kalman-and-Bayesian-Filters-in-Python (Chapter 14: Adaptive Filtering)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb)
- [QuantInsti: Kalman Filter Tutorial](https://blog.quantinsti.com/kalman-filter/)
- [QuantConnect: Kalman Filter Pairs Trading](https://github.com/QuantConnect/Research/blob/master/Analysis/02%20Kalman%20Filter%20Based%20Pairs%20Trading.ipynb)

**Adaptive Q/R Based on Volatility**:
```python
class AdaptiveKalman:
    def __init__(self):
        self.base_q = 0.01
        self.base_r = 0.1

    def adapt_parameters(self, current_volatility, avg_volatility):
        """Adjust Q and R based on market volatility"""
        vol_ratio = current_volatility / avg_volatility

        if vol_ratio > 1.5:  # High volatility
            # Trust measurements more, increase Q
            self.Q = self.base_q * vol_ratio
            self.R = self.base_r / vol_ratio
        elif vol_ratio < 0.7:  # Low volatility
            # Trust model more, decrease Q
            self.Q = self.base_q * vol_ratio
            self.R = self.base_r * (1/vol_ratio)
        else:
            self.Q = self.base_q
            self.R = self.base_r
```

---

### 3.5 Volatility Regime Detection (GARCH)

**Problem**: ATR-based volatility measurement is lagging.

**Solution**: Use GARCH model untuk predict volatility regime.

**References**:
- [LSEG: Market Regime Detection Statistical & ML](https://github.com/LSEG-API-Samples/Article.RD.Python.MarketRegimeDetectionUsingStatisticalAndMLBasedApproaches)
- [GitHub: Kritzman-Regime-Detection](https://github.com/tianyu-z/Kritzman-Regime-Detection)

**Implementation with arch library**:
```python
from arch import arch_model

def detect_volatility_regime(returns, threshold_low=0.01, threshold_high=0.02):
    """Detect volatility regime using GARCH"""
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    results = model.fit(disp='off')

    # Get conditional volatility
    cond_vol = results.conditional_volatility
    current_vol = cond_vol.iloc[-1]

    if current_vol < threshold_low:
        return 'LOW_VOL'
    elif current_vol > threshold_high:
        return 'HIGH_VOL'
    else:
        return 'NORMAL_VOL'
```

---

## 4. Recommended Implementation Priority

### Phase 1: Quick Wins (1-2 weeks)
1. **Add Choppiness Index Filter** - Skip trading when CHOP > 61.8
2. **Add ADX Threshold** - Skip when ADX < 20
3. **Increase min_quality to 70** for 2024-style choppy periods

### Phase 2: Medium-term (2-4 weeks)
4. **Implement Adaptive HMM** - Retrain every N bars
5. **Add Hurst Exponent** - Filter random walk periods
6. **Adaptive Kalman** - Vol-based Q/R adjustment

### Phase 3: Advanced (1-2 months)
7. **HMM + LSTM Ensemble** - Two-stage regime detection
8. **GARCH Volatility Regime** - Predictive volatility
9. **Machine Learning Entry Scoring** - Replace rule-based quality

---

## 5. GitHub Repositories to Study

### Regime Detection
| Repo | Description | Stars |
|------|-------------|-------|
| [theo-dim/regime_detection_ml](https://github.com/theo-dim/regime_detection_ml) | HMM + SVM regime detection | - |
| [JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM](https://github.com/JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM) | HMM-LSTM ensemble | - |
| [akash-kumar5/CryptoMarket_Regime_Classifier](https://github.com/akash-kumar5/CryptoMarket_Regime_Classifier) | Multi-timeframe HMM + LSTM | - |
| [tianyu-z/Kritzman-Regime-Detection](https://github.com/tianyu-z/Kritzman-Regime-Detection) | Kritzman paper implementation | - |
| [LSEG-API-Samples](https://github.com/LSEG-API-Samples/Article.RD.Python.MarketRegimeDetectionUsingStatisticalAndMLBasedApproaches) | Statistical + ML approaches | - |

### Kalman Filter
| Repo | Description | Stars |
|------|-------------|-------|
| [rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) | Complete Kalman book with code | 16k+ |
| [QuantConnect/Research](https://github.com/QuantConnect/Research) | Kalman pairs trading | - |
| [anthonyng2/Machine-Learning-For-Finance](https://github.com/anthonyng2/Machine-Learning-For-Finance) | ML for finance including Kalman | - |

---

## 6. Python Libraries to Consider

```python
# Regime Detection
pip install hmmlearn          # HMM implementation
pip install pomegranate       # Alternative HMM (faster)

# Deep Learning
pip install tensorflow        # For LSTM
pip install pytorch           # Alternative DL framework

# Technical Analysis
pip install ta                # Technical indicators
pip install ta-lib            # TA-Lib wrapper (faster)

# Volatility Modeling
pip install arch              # GARCH models

# Kalman Filtering
pip install filterpy          # Kalman filters (current)
pip install pykalman          # Alternative Kalman

# Statistical
pip install statsmodels       # Time series analysis
pip install scipy             # Scientific computing
```

---

## 7. Next Steps

1. **Review current regime_detector.py** - Identify improvement points
2. **Implement Choppiness Index** - Quick win filter
3. **Test Adaptive HMM** - Periodic retraining
4. **Study HMM-LSTM repos** - Learn ensemble approaches
5. **Backtest improvements** - Validate on 2024 data

---

## 8. References

### Academic/Blog Articles
- [QuantInsti: Regime-Adaptive Trading Python](https://blog.quantinsti.com/regime-adaptive-trading-python/)
- [QuantStart: HMM Market Regime Detection](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Medium: Choppiness Index](https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58)
- [Medium: Hurst Exponent](https://medium.com/algorithmic-and-quantitative-trading/the-hurst-filter-explained-how-to-use-the-hurst-exponent-h-to-measure-market-predictability-and-d94371d0f33c)
- [QuantInsti: Kalman Filter](https://blog.quantinsti.com/kalman-filter/)

### Papers
- Kritzman, Page, Turkington (2012) - "Regime Shifts: Implications for Dynamic Strategies"
- Hamilton (1989) - "A New Approach to the Economic Analysis of Nonstationary Time Series"

---

*Document created: 2026-01-29*
*Author: SURIOTA Team with Claude AI*
