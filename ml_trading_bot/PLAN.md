# ML Trading Bot - Implementation Plan

**Project:** SURGE ML Trading Bot
**Created:** 2026-01-30
**Data Available:** 11 years (2015-2026), 67,940 H1 bars, 2,862 daily profiles

---

## Project Structure

```
ml_trading_bot/
├── PLAN.md                    # This file
├── README.md                  # Project documentation
│
├── research/                  # Research references
│   └── references.md
│
├── config/                    # Configuration
│   ├── model_config.yaml      # Model hyperparameters
│   └── feature_config.yaml    # Feature definitions
│
├── features/                  # Feature Engineering
│   ├── __init__.py
│   ├── technical.py           # Technical indicators
│   ├── regime.py              # Regime features
│   └── session.py             # Session-based features
│
├── models/                    # ML Models
│   ├── __init__.py
│   ├── regime_detector.py     # HMM for regime detection
│   ├── signal_classifier.py   # XGBoost/RF for signals
│   └── ensemble.py            # Model ensemble
│
├── training/                  # Training Scripts
│   ├── __init__.py
│   ├── data_loader.py         # Load from TimescaleDB
│   ├── train_regime.py        # Train regime model
│   ├── train_signal.py        # Train signal model
│   └── evaluate.py            # Model evaluation
│
├── inference/                 # Real-time Inference
│   ├── __init__.py
│   ├── predictor.py           # Real-time predictions
│   └── risk_manager.py        # Position sizing
│
├── backtest/                  # Backtesting
│   ├── __init__.py
│   └── ml_backtest.py         # Backtest ML strategies
│
├── saved_models/              # Trained model artifacts
│   ├── regime_hmm.pkl
│   ├── signal_xgb.pkl
│   └── scaler.pkl
│
└── notebooks/                 # Jupyter notebooks
    ├── 01_data_exploration.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_model_training.ipynb
    └── 04_backtest_analysis.ipynb
```

---

## Implementation Phases

### Phase 1: Data Pipeline & Feature Engineering [HIGH PRIORITY]

**Goal:** Build robust feature engineering pipeline using existing 11-year data

**Tasks:**
1. [ ] Create `data_loader.py` - Load H1 data from TimescaleDB
2. [ ] Create `features/technical.py` - Technical indicators
3. [ ] Create `features/regime.py` - Regime-based features
4. [ ] Create `features/session.py` - Session timing features
5. [ ] Create feature configuration YAML

**Features to Implement:**

| Category | Features | Source |
|:---|:---|:---|
| **Price** | returns_1h, returns_4h, returns_1d, log_returns | Raw OHLC |
| **Volatility** | atr_14, atr_ratio, bb_width, historical_vol | pandas-ta |
| **Trend** | adx_14, plus_di, minus_di, ema_cross, macd | pandas-ta |
| **Momentum** | rsi_14, stoch_k, stoch_d, roc_10 | pandas-ta |
| **Session** | is_asian, is_london, is_ny, hour_of_day | Timestamp |
| **Calendar** | day_of_week, is_monday, is_friday | Timestamp |
| **Existing** | quality_score, volatility_regime, trend_regime | Our profiles |

**Output:** Feature matrix ready for training

---

### Phase 2: Regime Detection Model [HIGH PRIORITY]

**Goal:** Build HMM-based regime detector to identify market states

**Tasks:**
1. [ ] Create `models/regime_detector.py`
2. [ ] Implement `RegimeDetector` class with HMM
3. [ ] Train on 11-year data
4. [ ] Validate regime transitions
5. [ ] Save trained model

**Model Design:**
```python
class RegimeDetector:
    """
    Hidden Markov Model for market regime detection

    States:
    - 0: Low volatility / Trending (good for trading)
    - 1: High volatility / Crisis (reduce size or skip)
    - 2: Choppy / Ranging (use mean reversion)
    """

    def __init__(self, n_states=3):
        self.model = GaussianHMM(n_components=n_states)

    def fit(self, returns, volatility):
        """Train on historical returns and volatility"""
        pass

    def predict(self, current_data):
        """Predict current regime"""
        pass

    def get_regime_probabilities(self):
        """Get probability of each regime"""
        pass
```

**Validation Criteria:**
- Regime 0 should have lower volatility, positive returns
- Regime 1 should capture Brexit (Jun 2016), COVID (Mar 2020)
- Regime 2 should identify ranging/choppy periods

---

### Phase 3: Signal Classification Model [HIGH PRIORITY]

**Goal:** Build XGBoost classifier for buy/sell/hold signals

**Tasks:**
1. [ ] Create `models/signal_classifier.py`
2. [ ] Implement label generation (target variable)
3. [ ] Feature selection with importance ranking
4. [ ] Train XGBoost with cross-validation
5. [ ] Evaluate on test set (2024-2026)

**Label Generation:**
```python
def generate_labels(df, lookahead=24, threshold=0.003):
    """
    Generate trading labels

    Args:
        lookahead: Hours to look ahead (24 = 1 day)
        threshold: Min price change for signal (0.3% = 30 pips)

    Returns:
        labels: 1 (buy), -1 (sell), 0 (hold)
    """
    future_return = df['close'].pct_change(lookahead).shift(-lookahead)

    labels = np.where(future_return > threshold, 1,
             np.where(future_return < -threshold, -1, 0))

    return labels
```

**Model Configuration:**
```yaml
# model_config.yaml
xgboost:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.01
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 1.0

random_forest:
  n_estimators: 300
  max_depth: 10
  min_samples_split: 20
  min_samples_leaf: 10
```

**Target Metrics:**
- Accuracy > 55% (better than random)
- Precision > 50% for buy/sell signals
- Recall > 40% for capturing trends
- Sharpe ratio > 1.5 in backtest

---

### Phase 4: Risk Management Module [HIGH PRIORITY]

**Goal:** Implement dynamic position sizing and risk controls

**Tasks:**
1. [ ] Create `inference/risk_manager.py`
2. [ ] Implement Kelly Criterion (Half Kelly)
3. [ ] Regime-based position scaling
4. [ ] ATR-based SL/TP calculation
5. [ ] Daily loss limit

**Risk Manager Design:**
```python
class RiskManager:
    """
    Dynamic risk management based on market conditions
    """

    def __init__(self, account_risk=0.02, max_daily_loss=0.05):
        self.account_risk = account_risk  # 2% per trade
        self.max_daily_loss = max_daily_loss  # 5% daily limit

    def calculate_position_size(self,
                                 signal_confidence: float,
                                 regime: int,
                                 atr_pips: float,
                                 account_balance: float) -> float:
        """
        Calculate position size using Half Kelly with regime adjustment

        Args:
            signal_confidence: Model confidence (0-1)
            regime: Current market regime (0, 1, 2)
            atr_pips: Current ATR in pips
            account_balance: Current account balance

        Returns:
            lot_size: Position size in lots
        """
        # Base position from account risk
        base_risk = account_balance * self.account_risk
        sl_pips = atr_pips * 1.5
        pip_value = 10  # For standard lot GBPUSD
        base_lots = base_risk / (sl_pips * pip_value)

        # Kelly adjustment based on confidence
        kelly_factor = signal_confidence * 0.5  # Half Kelly

        # Regime adjustment
        regime_multiplier = {
            0: 1.0,    # Normal - full size
            1: 0.5,    # High vol - half size
            2: 0.7     # Choppy - reduced size
        }[regime]

        final_lots = base_lots * kelly_factor * regime_multiplier

        return round(final_lots, 2)

    def calculate_sl_tp(self, atr_pips: float, regime: int):
        """Calculate stop loss and take profit"""
        sl_mult = {0: 1.2, 1: 2.0, 2: 1.5}[regime]
        tp_mult = {0: 2.0, 1: 2.5, 2: 1.5}[regime]

        return atr_pips * sl_mult, atr_pips * tp_mult
```

**Risk Parameters:**
| Regime | Position Size | SL Multiplier | TP Multiplier |
|:---|---:|---:|---:|
| Normal (0) | 100% | 1.2x ATR | 2.0x ATR |
| High Vol (1) | 50% | 2.0x ATR | 2.5x ATR |
| Choppy (2) | 70% | 1.5x ATR | 1.5x ATR |

---

### Phase 5: Backtesting Framework [MEDIUM PRIORITY]

**Goal:** Validate ML models with realistic backtesting

**Tasks:**
1. [ ] Create `backtest/ml_backtest.py`
2. [ ] Implement walk-forward validation
3. [ ] Add transaction costs and slippage
4. [ ] Generate performance reports
5. [ ] Compare with baseline strategy

**Walk-Forward Validation:**
```
Training Window: 2 years rolling
Test Window: 3 months forward
Retrain: Every 3 months

Example:
├── Train: 2015-01 to 2016-12 → Test: 2017-01 to 2017-03
├── Train: 2015-04 to 2017-03 → Test: 2017-04 to 2017-06
├── Train: 2015-07 to 2017-06 → Test: 2017-07 to 2017-09
└── ... continue rolling
```

**Performance Metrics:**
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Recovery Factor

---

### Phase 6: Integration with MT5 [MEDIUM PRIORITY]

**Goal:** Connect ML models to existing MT5 executor

**Tasks:**
1. [ ] Create `inference/predictor.py`
2. [ ] Integrate with existing `src/trading/executor.py`
3. [ ] Add real-time feature calculation
4. [ ] Implement prediction caching

**Integration Architecture:**
```
┌─────────────────────────────────────────────────┐
│            Existing SURGE System                │
│                                                 │
│  ┌─────────────┐    ┌──────────────────────┐   │
│  │  MT5        │    │  src/trading/        │   │
│  │  Connector  │◄──►│  executor.py         │   │
│  └─────────────┘    └──────────┬───────────┘   │
│                                │               │
│                     ┌──────────▼───────────┐   │
│                     │  ML Predictor        │   │
│                     │  (New Integration)   │   │
│                     └──────────┬───────────┘   │
│                                │               │
│  ┌─────────────────────────────▼────────────┐  │
│  │           ml_trading_bot/                │  │
│  │  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │   Regime    │  │    Signal       │   │  │
│  │  │   Detector  │  │    Classifier   │   │  │
│  │  └─────────────┘  └─────────────────┘   │  │
│  │  ┌─────────────────────────────────┐    │  │
│  │  │        Risk Manager             │    │  │
│  │  └─────────────────────────────────┘    │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

### Phase 7: Monitoring & Alerts [LOW PRIORITY]

**Goal:** Add Telegram alerts and performance dashboard

**Tasks:**
1. [ ] Create Telegram bot for alerts
2. [ ] Signal notifications
3. [ ] Daily performance summary
4. [ ] Anomaly alerts (high loss, regime change)

---

## Development Schedule

| Phase | Priority | Dependencies | Est. Files |
|:---|:---:|:---|---:|
| **1. Features** | HIGH | None | 5 |
| **2. Regime** | HIGH | Phase 1 | 2 |
| **3. Signal** | HIGH | Phase 1, 2 | 3 |
| **4. Risk** | HIGH | Phase 2, 3 | 2 |
| **5. Backtest** | MEDIUM | Phase 1-4 | 2 |
| **6. Integration** | MEDIUM | Phase 1-5 | 2 |
| **7. Monitoring** | LOW | Phase 6 | 3 |

---

## File Implementation Order

### Round 1: Core Infrastructure
```
1. ml_trading_bot/__init__.py
2. ml_trading_bot/config/model_config.yaml
3. ml_trading_bot/config/feature_config.yaml
4. ml_trading_bot/training/data_loader.py
```

### Round 2: Feature Engineering
```
5. ml_trading_bot/features/__init__.py
6. ml_trading_bot/features/technical.py
7. ml_trading_bot/features/regime.py
8. ml_trading_bot/features/session.py
```

### Round 3: Models
```
9. ml_trading_bot/models/__init__.py
10. ml_trading_bot/models/regime_detector.py
11. ml_trading_bot/models/signal_classifier.py
12. ml_trading_bot/models/ensemble.py
```

### Round 4: Training & Evaluation
```
13. ml_trading_bot/training/train_regime.py
14. ml_trading_bot/training/train_signal.py
15. ml_trading_bot/training/evaluate.py
```

### Round 5: Inference & Risk
```
16. ml_trading_bot/inference/__init__.py
17. ml_trading_bot/inference/predictor.py
18. ml_trading_bot/inference/risk_manager.py
```

### Round 6: Backtesting
```
19. ml_trading_bot/backtest/__init__.py
20. ml_trading_bot/backtest/ml_backtest.py
```

---

## Success Criteria

### Model Performance
- [ ] Regime detector accuracy > 70% on known events (Brexit, COVID)
- [ ] Signal classifier accuracy > 55%
- [ ] Backtest Sharpe ratio > 1.5
- [ ] Max drawdown < 20%

### Integration
- [ ] Real-time prediction latency < 100ms
- [ ] Seamless MT5 integration
- [ ] Proper error handling

### Risk Management
- [ ] No single trade > 2% account risk
- [ ] Daily loss limit enforced
- [ ] Regime-based position adjustment working

---

## Technology Stack

```yaml
Core:
  - Python 3.11+
  - asyncio

Data:
  - pandas
  - numpy
  - TimescaleDB (existing)

ML:
  - scikit-learn
  - xgboost
  - hmmlearn
  - joblib (model persistence)

Technical Analysis:
  - pandas-ta (existing)

Visualization:
  - matplotlib
  - plotly

Testing:
  - pytest
```

---

## Next Action

**Start with Phase 1:** Create data loader and feature engineering pipeline.

```bash
# Files to create first:
ml_trading_bot/__init__.py
ml_trading_bot/config/model_config.yaml
ml_trading_bot/training/data_loader.py
ml_trading_bot/features/technical.py
```
