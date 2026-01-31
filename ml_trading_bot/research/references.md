# ML Trading Bot - Research References

**Compiled:** 2026-01-30
**Purpose:** Reference materials for building AI/ML Trading Bot

---

## 1. Machine Learning Trading - Best Practices

### Key Sources
- [ML_Trading_Bot (GitHub)](https://github.com/Mun-Min/ML_Trading_Bot) - Algorithmic trading bot that learns and adapts to new data
- [Intelligent Trading Bot (GitHub)](https://github.com/asavinov/intelligent-trading-bot) - Auto signal generation with ML and feature engineering
- [Machine Learning for Algorithmic Trading (Book)](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715) - Stefan Jansen, 800+ pages, 150+ notebooks

### Best Practices Summary
1. **Avoid Overfitting** - Strategies that work perfectly on historical data but fail live
2. **Handle Latency** - Add realistic latency simulation to testing
3. **Account for Slippage** - Difference between expected and actual execution prices
4. **Error Handling** - Gracefully handle API failures, connection drops
5. **Stop-Loss Protection** - Always implement automatic stop-loss mechanisms
6. **Testing Methodology** - Backtesting → Forward testing → Paper trading → Live

### Recommended Libraries (2025)
```
numpy, pandas, matplotlib, seaborn
ta-lib, pandas-ta (technical analysis)
scikit-learn, xgboost, lightgbm (ML)
pytorch, tensorflow (deep learning)
backtrader, vectorbt (backtesting)
ccxt, MetaTrader5 (broker APIs)
```

---

## 2. Forex/FX Trading with ML

### Key Repositories
- [AI-Enhanced-HFT (GitHub)](https://github.com/Marco210210/AI-Enhanced-HFT) - Hybrid EA with XGBoost, RF, LSTM for MetaTrader 5
- [LSTM-FX (GitHub)](https://github.com/AdamTibi/LSTM-FX) - End-to-end LSTM for FX prediction
- [Forex-trading-XGBoost-model (GitHub)](https://github.com/MichaelSoegaard/Forex-trading-XGBoost-model) - XGBoost for noisy forex data
- [LSTM-XGBoost-Hybrid-Forecasting (GitHub)](https://github.com/Hupperich-Manuel/LSTM-XGBoost-Hybrid-Forecasting) - Hybrid approach

### AI-Enhanced-HFT Architecture (Reference)
```
┌─────────────────────────────────────────────────┐
│              MetaTrader 5 EA (MQL5)             │
│  • EMA crossover signals (6/24)                 │
│  • ADX validation (threshold: 22)              │
│  • Volume analysis                              │
└─────────────────────┬───────────────────────────┘
                      │ TCP Socket
┌─────────────────────▼───────────────────────────┐
│              Python AI Server                   │
│  • XGBoost (best performer, Sharpe > 9)        │
│  • Random Forest (stable)                       │
│  • LSTM (underperformed on noisy data)         │
└─────────────────────────────────────────────────┘
```

### Key Findings
- **XGBoost** outperformed LSTM for forex due to noise sensitivity
- Training on **H1 timeframe** for short-term prediction
- Dynamic lot sizing based on signal-AI agreement
- ADX differential for early exit when trends weaken

---

## 3. Market Regime Detection

### Key Sources
- [Market Regime Detection with HMM (QuantStart)](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Regime Detection ML (GitHub)](https://github.com/theo-dim/regime_detection_ml) - HMM + SVM for regime detection
- [Market Regime Detection (QuantInsti)](https://blog.quantinsti.com/market-regime-detection-hidden-markov-model-project-fahim/)
- [Market Regime Detection (LSEG DevPortal)](https://developers.lseg.com/en/article-catalog/article/market-regime-detection)

### Hidden Markov Model (HMM) Approach
```python
# Typical HMM implementation
from hmmlearn.hmm import GaussianHMM

# Model with 2-3 states
model = GaussianHMM(n_components=3, covariance_type="full")

# States often correspond to:
# - State 0: Low volatility (bull)
# - State 1: High volatility (bear)
# - State 2: Transition/choppy

# Train on daily returns
model.fit(returns.reshape(-1, 1))
regimes = model.predict(returns.reshape(-1, 1))
```

### Regime-Based Trading Strategy
1. Detect current regime using HMM
2. Train specialist ML models for each regime
3. Use appropriate model based on regime prediction
4. Disallow trades during high-volatility regimes

### Research Findings
- HMM outperformed other models for regime detection (2006-2023 study)
- Simple regime-based strategy beat buy-and-hold
- 3 states typical: bull, bear, neutral

---

## 4. Risk Management & Position Sizing

### Kelly Criterion
- [Risk-Constrained Kelly (QuantInsti)](https://blog.quantinsti.com/risk-constrained-kelly-criterion/)
- [Kelly Criterion in Trading (QuantStart)](https://www.quantstart.com/articles/Money-Management-via-the-Kelly-Criterion/)
- [Kelly Criterion Applications (QuantConnect)](https://www.quantconnect.com/research/18312/kelly-criterion-applications-in-trading-systems/)

### Formula
```
K% = W - [(1-W) / R]

Where:
- W = Win rate (as decimal)
- R = Reward/Risk ratio (avg win ÷ avg loss)

For strategies with Sharpe ratio:
f = μ / σ²

Where:
- μ = Mean excess returns
- σ = Standard deviation of returns
```

### Practical Guidelines
| Approach | Risk % | Notes |
|:---|---:|:---|
| Full Kelly | 15-20% | Max growth, 50-70% drawdown possible |
| Half Kelly | 7.5-10% | 75% growth, 50% less drawdown |
| Quarter Kelly | 3-5% | Conservative, stable equity curve |

### Key Insights
- Full Kelly too aggressive for most traders
- Half Kelly recommended for algorithmic trading
- Rebalance allocation once per day
- Account for estimation errors in win probability

---

## 5. Feature Engineering for Trading

### Key Sources
- [Feature Engineering for Trading (LuxAlgo)](https://www.luxalgo.com/blog/feature-engineering-in-trading-turning-data-into-insights/)
- [Alpha Factor Research (Stefan Jansen)](https://stefan-jansen.github.io/machine-learning-for-trading/04_alpha_factor_research/)
- [Technical Indicators for ML (TowardsDataScience)](https://towardsdatascience.com/implementation-of-technical-indicators-into-a-machine-learning-framework-for-quantitative-trading-44a05be8e06/)

### Best Practices
1. **Less is more**: 25 quality indicators > 150 random ones
2. **Use proven indicators**: RSI, MACD, Bollinger Bands, ATR
3. **Add incrementally**: One indicator at a time, measure impact
4. **Avoid overfitting**: Keep features relevant and interpretable

### Recommended Features
```python
# Price-based
- Returns (1h, 4h, 1d, 1w)
- Log returns
- Percent change

# Volatility
- ATR (14, 20)
- Bollinger Band width
- Historical volatility

# Trend
- ADX (14)
- EMA crossovers (9/21, 50/200)
- MACD

# Momentum
- RSI (14)
- Stochastic
- ROC (Rate of Change)

# Volume (if available)
- OBV
- Volume MA ratio
```

### Data Split
- Training: 70%
- Validation: 15%
- Test: 15%
- **Use time-series split** to prevent look-ahead bias

---

## 6. Backtesting Frameworks

### Comparison
- [VectorBT vs Backtrader (Greyhound Analytics)](https://greyhoundanalytics.com/blog/vectorbt-vs-backtrader/)
- [Battle-Tested Backtesters (Medium)](https://medium.com/@trading.dude/battle-tested-backtesters-comparing-vectorbt-zipline-and-backtrader-for-financial-strategy-dee33d33a9e0)

| Framework | Speed | Ease of Use | Live Trading | Status |
|:---|:---:|:---:|:---:|:---|
| VectorBT | Fast | Medium | No | Maintained (PRO paid) |
| Backtrader | Slow | Easy | Yes | Abandoned (2018) |
| Zipline | Medium | Medium | No | Community fork |
| NautilusTrader | Fast | Hard | Yes | Active |

### Recommendation
- **Research/Optimization**: VectorBT (speed)
- **Live Trading**: Backtrader or custom
- **High Frequency**: NautilusTrader

---

## 7. Reference Architecture: Intelligent Trading Bot

From: [asavinov/intelligent-trading-bot](https://github.com/asavinov/intelligent-trading-bot)

### Pipeline
```
1. Data Acquisition → Download from Binance/Yahoo
2. Data Merging → Align multi-source data
3. Feature Engineering → Generate derived features
4. Label Generation → Compute future targets
5. Model Training → Train ML models
6. Prediction → Apply models
7. Signal Generation → Aggregate predictions
8. Output → Telegram alerts / Execute trades
```

### Folder Structure
```
├── common/          # Shared utilities
├── configs/         # Configuration files
├── docs/            # Documentation
├── inputs/          # Raw data storage
├── outputs/         # Results & models
├── scripts/         # Batch processing
├── service/         # Online trading service
└── tests/           # Unit tests
```

### Label Generators
- `highlow`: Threshold-based price movements
- `highlow2`: Future increases with tolerance
- `topbot2`: Local maxima/minima with volatility thresholds

---

## 8. Model Performance Benchmarks

### From AI-Enhanced-HFT Research
| Model | Profit Factor | Sharpe Ratio | Notes |
|:---|---:|---:|:---|
| XGBoost | > 1.6 | > 9 | Best performer |
| Random Forest | ~1.4 | ~7 | Stable |
| LSTM | < 1.2 | < 5 | Noise sensitivity |

### From Literature
| Model | Accuracy | Use Case |
|:---|---:|:---|
| SVM | 65-85% | Trend prediction |
| CNN | 70-90% | Chart pattern recognition |
| LSTM | ~60-70% | Sequence prediction |
| XGBoost | 55-70% | Feature-based classification |

---

## 9. Key Takeaways for Our Implementation

### Priority 1: Market Condition Filter
- Use HMM for regime detection (bull/bear/choppy)
- Already have 11 years of volatility/trend data
- Filter out low-quality trading days

### Priority 2: Risk Management
- Implement Kelly Criterion (Half Kelly)
- Dynamic position sizing based on regime
- ATR-based stop loss/take profit

### Priority 3: Feature Engineering
- Start with 15-20 proven indicators
- Use our existing ATR, ADX, Choppiness data
- Add session-based features

### Priority 4: ML Models
- Start with XGBoost (proven forex performance)
- Add Random Forest for ensemble
- LSTM optional for sequence patterns

### Priority 5: Integration
- Socket communication with MT5
- Real-time prediction service
- Telegram alerts

---

## Next Steps

1. Create detailed implementation plan
2. Set up training pipeline with 11-year data
3. Train regime detection model (HMM)
4. Train signal prediction model (XGBoost)
5. Integrate with existing MT5 executor
6. Backtest and validate
7. Paper trade before live

