"""
Test ML Models on Latest Data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ml_trading_bot.models.regime_detector import RegimeDetector
from ml_trading_bot.models.signal_classifier import SignalClassifier
from ml_trading_bot.inference.risk_manager import RiskManager
from ml_trading_bot.training.data_loader import DataLoader
from ml_trading_bot.features.technical import TechnicalFeatures
from ml_trading_bot.features.session import SessionFeatures
from ml_trading_bot.features.regime import RegimeFeatures

def main():
    print("=" * 60)
    print("TESTING ML MODELS ON LATEST DATA")
    print("=" * 60)

    # Load saved models
    print("\n1. Loading saved models...")
    models_dir = Path(__file__).parent / "saved_models"

    detector = RegimeDetector()
    detector.load(str(models_dir / "regime_hmm.pkl"))

    classifier = SignalClassifier()
    classifier.load(str(models_dir / "signal_classifier.pkl"))

    risk_mgr = RiskManager()

    # Load data for indicator warmup (3 months)
    print("\n2. Loading data with warmup period...")
    loader = DataLoader()
    df = loader.load_ohlcv_sync("2025-11-01", "2026-01-30")
    print(f"   Loaded {len(df)} bars for warmup")

    # Merge with profiles
    daily_profiles = loader.load_daily_profiles()
    df = loader.merge_ohlcv_with_daily_profiles(df, daily_profiles)

    # Compute features
    print("\n3. Computing features...")
    tech = TechnicalFeatures()
    df = tech.add_all_features(df)

    session = SessionFeatures()
    df = session.add_all_features(df)

    regime = RegimeFeatures()
    df = regime.add_all_features(df)

    print(f"   Shape after features: {df.shape}")

    # Filter to recent data
    df_recent = df[df.index >= "2026-01-20"]
    print(f"   Recent data (Jan 20-30): {len(df_recent)} bars")

    # Get latest data point
    latest = df_recent.iloc[-1:]
    latest_time = df_recent.index[-1]

    print(f"\n4. Latest data point: {latest_time}")
    close_price = latest["close"].values[0]
    print(f"   Close: {close_price:.5f}")

    # Get current ATR
    current_atr = latest["atr_14"].values[0] if "atr_14" in latest.columns else 0.0015
    atr_pips = current_atr * 10000
    print(f"   ATR (14): {atr_pips:.1f} pips")

    # Predict regime
    print("\n" + "=" * 60)
    print("REGIME DETECTION")
    print("=" * 60)

    regime_info = detector.get_current_regime(df_recent)
    print(f"Current Regime: {regime_info['regime_name'].upper()}")
    print(f"Confidence: {regime_info['confidence']:.1%}")
    print(f"Risk Multiplier: {regime_info['risk_multiplier']}x")
    print(f"\nProbabilities:")
    for name, prob in regime_info["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {name:20s} {prob:6.1%} |{bar}")

    # Predict signal
    print("\n" + "=" * 60)
    print("SIGNAL PREDICTION")
    print("=" * 60)

    signal_info = classifier.get_signal(df_recent, confidence_threshold=0.55)
    print(f"Signal: {signal_info['signal_name']}")
    print(f"Confidence: {signal_info['confidence']:.1%}")
    print(f"\nProbabilities:")
    for name, prob in signal_info["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {name:6s} {prob:6.1%} |{bar}")

    # Calculate position size
    print("\n" + "=" * 60)
    print("RISK MANAGEMENT")
    print("=" * 60)

    account_balance = 10000

    params = risk_mgr.calculate_position_size(
        account_balance=account_balance,
        signal_confidence=signal_info["confidence"],
        regime=regime_info["regime"],
        atr_pips=atr_pips
    )

    print(f"Account Balance: ${account_balance:,.2f}")
    print(f"\nTrade Parameters:")
    print(f"  Approved: {params.approved}")
    if params.approved:
        print(f"  Lot Size: {params.lot_size}")
        print(f"  Stop Loss: {params.stop_loss_pips:.1f} pips")
        print(f"  Take Profit: {params.take_profit_pips:.1f} pips")
        print(f"  Risk Amount: ${params.risk_amount:.2f} ({params.risk_pct:.2%})")
    else:
        print(f"  Reason: {params.reason}")

    # Show last 24 hours predictions
    print("\n" + "=" * 60)
    print("LAST 24 HOURS SIGNALS")
    print("=" * 60)

    last_24h = df_recent.tail(24)
    regimes = detector.predict(last_24h)
    signals = classifier.predict(last_24h)
    proba = classifier.predict_proba(last_24h)

    regime_names = {0: "TREND", 1: "CRISIS", 2: "RANGE"}
    signal_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    print(f"\n{'Time (UTC)':16s} {'Close':>10s} {'Regime':>7s} {'Signal':>6s} {'Conf':>6s}")
    print("-" * 50)

    for i in range(len(last_24h)):
        idx = last_24h.index[i]
        close = last_24h["close"].iloc[i]
        reg = regime_names[regimes[i]]
        sig = signal_names[signals[i]]

        if signals[i] == 1:
            conf = proba["buy"][i]
        elif signals[i] == -1:
            conf = proba["sell"][i]
        else:
            conf = proba["hold"][i]

        time_str = idx.strftime("%m-%d %H:%M")
        print(f"{time_str:16s} {close:10.5f} {reg:>7s} {sig:>6s} {conf:>5.1%}")

    # Summary
    buy_count = sum(1 for s in signals if s == 1)
    sell_count = sum(1 for s in signals if s == -1)
    hold_count = sum(1 for s in signals if s == 0)

    print(f"\nLast 24h Summary:")
    print(f"  BUY signals:  {buy_count}")
    print(f"  SELL signals: {sell_count}")
    print(f"  HOLD signals: {hold_count}")

    trend_count = sum(1 for r in regimes if r == 0)
    crisis_count = sum(1 for r in regimes if r == 1)
    range_count = sum(1 for r in regimes if r == 2)

    print(f"\nRegime Distribution:")
    print(f"  TREND:  {trend_count} ({trend_count/len(regimes)*100:.0f}%)")
    print(f"  CRISIS: {crisis_count} ({crisis_count/len(regimes)*100:.0f}%)")
    print(f"  RANGE:  {range_count} ({range_count/len(regimes)*100:.0f}%)")

    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
