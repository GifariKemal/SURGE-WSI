# SURGE ML Trading Bot

Machine Learning-based trading system for GBPUSD with market regime detection, signal classification, and dynamic risk management.

## Overview

This module extends the SURGE trading system with ML capabilities:

1. **Regime Detection** - Hidden Markov Model (HMM) identifies market states
2. **Signal Classification** - XGBoost/Random Forest predicts buy/sell signals
3. **Risk Management** - Half Kelly position sizing with regime adjustments

## Data Foundation

Built on 11 years of GBPUSD H1 data (2015-2026):
- **67,940 H1 bars** in TimescaleDB
- **133 monthly profiles** with pre-computed metrics
- **2,862 daily profiles** with quality scores

Key events captured:
- Brexit Vote (June 2016) - ATR 47.7 pips
- COVID Crash (March 2020) - ATR 48.3 pips
- UK Mini-Budget Crisis (Sep 2022) - ATR 34.5 pips

## Architecture

```
ml_trading_bot/
├── config/                    # Configuration files
│   └── model_config.yaml      # Model hyperparameters
│
├── features/                  # Feature Engineering
│   ├── technical.py           # Technical indicators (50+ features)
│   ├── regime.py              # Regime-based features
│   └── session.py             # Session timing features
│
├── models/                    # ML Models
│   ├── regime_detector.py     # HMM for 3-state regime detection
│   └── signal_classifier.py   # XGBoost + Random Forest ensemble
│
├── inference/                 # Real-time Inference
│   └── risk_manager.py        # Position sizing & risk controls
│
├── training/                  # Training Scripts
│   └── data_loader.py         # Load from TimescaleDB
│
└── saved_models/              # Trained model artifacts
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ml_trading_bot/requirements.txt
```

### 2. Load Data

```python
from ml_trading_bot.training.data_loader import DataLoader

loader = DataLoader()
df = loader.prepare_training_data("2015-01-01", "2024-12-31")
```

### 3. Compute Features

```python
from ml_trading_bot.features.technical import TechnicalFeatures
from ml_trading_bot.features.session import SessionFeatures
from ml_trading_bot.features.regime import RegimeFeatures

# Add all features
tech = TechnicalFeatures()
df = tech.add_all_features(df)

session = SessionFeatures()
df = session.add_all_features(df)

regime = RegimeFeatures()
df = regime.add_all_features(df)
```

### 4. Train Regime Detector

```python
from ml_trading_bot.models.regime_detector import RegimeDetector

detector = RegimeDetector(n_states=3)
detector.fit(df)
detector.save("saved_models/regime_hmm.pkl")
```

### 5. Train Signal Classifier

```python
from ml_trading_bot.models.signal_classifier import SignalClassifier

classifier = SignalClassifier(lookahead_hours=24, threshold_pct=0.003)
classifier.fit(train_df)
classifier.save("saved_models/signal_xgb.pkl")
```

### 6. Calculate Position Size

```python
from ml_trading_bot.inference.risk_manager import RiskManager

rm = RiskManager(account_risk_pct=0.02, kelly_fraction=0.5)

params = rm.calculate_position_size(
    account_balance=10000,
    signal_confidence=0.75,
    regime=0,  # trending_low_vol
    atr_pips=15.5
)

print(f"Lot size: {params.lot_size}")
print(f"SL: {params.stop_loss_pips} pips")
print(f"TP: {params.take_profit_pips} pips")
```

## Regime Detection

The HMM identifies 3 market states:

| State | Name | Characteristics | Action |
|:---:|:---|:---|:---|
| 0 | `trending_low_vol` | Low volatility, clear trends | Full position |
| 1 | `crisis_high_vol` | Extreme volatility (Brexit, COVID) | Half position or skip |
| 2 | `ranging_choppy` | Sideways, choppy action | Reduced position |

## Signal Classification

XGBoost + Random Forest ensemble predicts:
- **1 (Buy)**: Price expected to rise >0.3% in 24h
- **-1 (Sell)**: Price expected to fall >0.3% in 24h
- **0 (Hold)**: No clear direction

Features used:
- 50+ technical indicators (ATR, ADX, RSI, MACD, etc.)
- Session timing (Asian, London, NY)
- Pre-computed regime features

## Risk Management

Position sizing uses Half Kelly with adjustments:

```
Position Size = Base Lots × Kelly Factor × Regime Mult × Confidence Factor

Where:
- Base Lots = (Account × Risk%) / (SL pips × Pip Value)
- Kelly Factor = Confidence × 0.5 (Half Kelly)
- Regime Mult = [1.0, 0.5, 0.7] for regimes [0, 1, 2]
- Confidence Factor = scaled 0.5-1.0
```

## Configuration

Edit `config/model_config.yaml`:

```yaml
# Key settings
data:
  symbol: "GBPUSD"
  timeframe: "H1"

labels:
  lookahead_hours: 24
  threshold_pips: 30

risk:
  account_risk_pct: 0.02
  kelly_fraction: 0.5
  max_daily_loss_pct: 0.05
```

## Integration with SURGE

The ML models integrate with existing SURGE components:

```python
# In src/trading/executor.py

from ml_trading_bot.models.regime_detector import RegimeDetector
from ml_trading_bot.models.signal_classifier import SignalClassifier
from ml_trading_bot.inference.risk_manager import RiskManager

# Load models
detector = RegimeDetector().load("saved_models/regime_hmm.pkl")
classifier = SignalClassifier().load("saved_models/signal_xgb.pkl")
risk_mgr = RiskManager()

# Get current regime
regime_info = detector.get_current_regime(recent_data)

# Get signal
signal_info = classifier.get_signal(recent_data, confidence_threshold=0.6)

# Calculate position
if signal_info['signal'] != 0:
    params = risk_mgr.calculate_position_size(
        account_balance=account.balance,
        signal_confidence=signal_info['confidence'],
        regime=regime_info['regime'],
        atr_pips=current_atr
    )
```

## Testing

Run individual module tests:

```bash
# Test data loader
python -m ml_trading_bot.training.data_loader

# Test features
python -m ml_trading_bot.features.technical
python -m ml_trading_bot.features.session
python -m ml_trading_bot.features.regime

# Test models
python -m ml_trading_bot.models.regime_detector
python -m ml_trading_bot.models.signal_classifier

# Test risk manager
python -m ml_trading_bot.inference.risk_manager
```

## References

Research and libraries used:
- [Machine Learning for Algorithmic Trading](https://github.com/stefan-jansen/machine-learning-for-trading) - Stefan Jansen
- [AI-Enhanced-HFT](https://github.com/Marco210210/AI-Enhanced-HFT) - XGBoost/LSTM with MT5
- [hmmlearn](https://hmmlearn.readthedocs.io/) - Hidden Markov Models
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting

## Status

| Component | Status |
|:---|:---:|
| Data Loader | Ready |
| Technical Features | Ready |
| Session Features | Ready |
| Regime Features | Ready |
| Regime Detector (HMM) | Ready |
| Signal Classifier (XGBoost) | Ready |
| Risk Manager | Ready |
| Integration | Pending |
| Backtesting | Pending |
| Live Trading | Pending |

## Next Steps

1. [ ] Train models on full 11-year dataset
2. [ ] Backtest with walk-forward validation
3. [ ] Integrate with MT5 executor
4. [ ] Add Telegram alerts
5. [ ] Paper trade validation
6. [ ] Live deployment
