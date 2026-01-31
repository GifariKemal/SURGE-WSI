# Production Trading System - Hybrid ML + RSI Strategy

## Overview

Production-ready trading bot based on **walk-forward validated** backtest results:

| Metric | Result |
|--------|--------|
| Total Trades | 2,684 |
| Total Return | +173.25% |
| Win Rate | 55.7% |
| Profitable Windows | 14/19 (73.7%) |
| Max Drawdown | -15.04% |
| Test Period | 9 years (2017-2026) |

## Strategy Components

### 1. HMM Regime Detection
- **3 market states**: Trending, Crisis, Ranging
- **Skip trading** during crisis regime (high volatility)
- Trained on 11 years of data (67,940 H1 bars)

### 2. RSI Mean Reversion Signals
- **Buy**: RSI < 35 with bullish reversal candle
- **Sell**: RSI > 65 with bearish reversal candle
- Parameters optimized per-window using Optuna

### 3. Risk Management
- 1% risk per trade
- ATR-based SL (1.75x) and TP (2.5x)
- Exit at Bollinger Band mean
- Daily loss limit: 3%
- Max drawdown protection: 15%

## File Structure

```
ml_trading_bot/
├── config/
│   └── production_config.yaml    # Main configuration
├── production/
│   ├── live_executor.py          # Live trading bot
│   └── train_production_model.py # HMM training script
├── saved_models/
│   └── regime_hmm.pkl            # Trained HMM model
├── backtest_proper/
│   ├── walk_forward.py           # Walk-forward validation
│   ├── vectorbt_engine.py        # Backtesting engine
│   └── hybrid_optuna_backtest.py # Main backtest script
├── run_production.bat            # Startup script
└── PRODUCTION_README.md          # This file
```

## Quick Start

### 1. Dry Run (Recommended First)
```batch
run_production.bat
```
This runs the bot without executing real trades.

### 2. Live Trading
```batch
run_production.bat --live
```
**WARNING**: This will execute real trades with real money!

### 3. Manual Start
```bash
cd ml_trading_bot
python production/live_executor.py --dry-run
```

## Configuration

Edit `config/production_config.yaml` to adjust:

```yaml
signals:
  rsi_oversold: 35      # Buy threshold
  rsi_overbought: 65    # Sell threshold

risk:
  risk_per_trade: 0.01  # 1% per trade
  sl_atr_multiplier: 1.75
  tp_atr_multiplier: 2.50

session:
  start_hour: 7         # UTC
  end_hour: 19          # UTC
```

## Retraining the Model

If market conditions change significantly:

```bash
cd ml_trading_bot
python production/train_production_model.py
```

This will:
1. Load all historical data
2. Train new HMM model
3. Save to `saved_models/regime_hmm.pkl`

## Validation Details

### Walk-Forward Process
1. Train on 2 years of data
2. Test on next 6 months
3. Expand training window
4. Repeat for all available data

### Per-Window Results (H1 = Jan-Jun, H2 = Jul-Dec)
```
2017 H1: -2.02%     2017 H2: +13.56%
2018 H1: -0.28%     2018 H2: +24.71%
2019 H1: +7.58%     2019 H2: -10.27%
2020 H1: +1.59%     2020 H2: +6.88%
2021 H1: +4.06%     2021 H2: +8.71%
2022 H1: +19.45%    2022 H2: +25.90%
2023 H1: +6.34%     2023 H2: +24.37%
2024 H1: +23.22%    2024 H2: -0.09%
2025 H1: +4.55%     2025 H2: +17.38%
```

### Why This Works
1. **Simple RSI rules** - More robust than complex ML signals
2. **HMM regime filter** - Skips dangerous market conditions
3. **Optuna optimization** - Auto-tunes parameters
4. **Walk-forward validation** - Avoids overfitting

## Monitoring

### Logs
- Location: `logs/live_executor.log`
- Contains: Trade entries, exits, regime changes, errors

### Telegram Notifications
Configure in `.env`:
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Troubleshooting

### "Regime detector not found"
Run: `python production/train_production_model.py`

### "MT5 initialization failed"
1. Check MT5 is installed and running
2. Verify credentials in `.env`
3. Ensure symbol "GBPUSD" is available

### "Cannot trade: Crisis regime"
This is expected behavior - the bot skips high-volatility periods.

## Disclaimer

This trading system is provided for educational purposes. Past performance does not guarantee future results. Always test with demo accounts first and never risk more than you can afford to lose.

---

*Generated from walk-forward validated backtest on 2026-01-31*
