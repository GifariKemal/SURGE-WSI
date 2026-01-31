"""
Train All ML Models
===================

Complete training pipeline for:
1. Regime Detector (HMM)
2. Signal Classifier (XGBoost + Random Forest)

Using 11 years of GBPUSD H1 data (2015-2026)
"""

import time
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

# Timing decorator
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] completed in {elapsed:.1f}s")
        return result
    return wrapper


class ModelTrainer:
    """Complete training pipeline for ML Trading Bot"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.models_dir.mkdir(exist_ok=True)

        # Timing stats
        self.timings = {}

    @timed
    def load_data(self) -> pd.DataFrame:
        """Load 11 years of H1 data with profiles"""
        from ml_trading_bot.training.data_loader import DataLoader

        print("\n" + "=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)

        loader = DataLoader()

        # Load OHLCV from database
        print("Loading H1 OHLCV data from TimescaleDB...")
        df = loader.load_ohlcv_sync("2015-01-01", "2026-01-31")
        print(f"  Loaded {len(df):,} bars")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

        # Load and merge daily profiles
        print("\nMerging with daily profiles...")
        daily_profiles = loader.load_daily_profiles()
        df = loader.merge_ohlcv_with_daily_profiles(df, daily_profiles)

        return df

    @timed
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features"""
        from ml_trading_bot.features.technical import TechnicalFeatures
        from ml_trading_bot.features.session import SessionFeatures
        from ml_trading_bot.features.regime import RegimeFeatures

        print("\n" + "=" * 60)
        print("STEP 2: COMPUTING FEATURES")
        print("=" * 60)

        initial_cols = len(df.columns)

        # Technical features
        print("\nAdding technical indicators...")
        tech = TechnicalFeatures()
        df = tech.add_all_features(df)
        print(f"  Added {len(df.columns) - initial_cols} technical features")

        # Session features
        print("\nAdding session features...")
        session_cols = len(df.columns)
        session = SessionFeatures()
        df = session.add_all_features(df)
        print(f"  Added {len(df.columns) - session_cols} session features")

        # Regime features
        print("\nAdding regime features...")
        regime_cols = len(df.columns)
        regime = RegimeFeatures()
        df = regime.add_all_features(df)
        print(f"  Added {len(df.columns) - regime_cols} regime features")

        print(f"\nTotal features: {len(df.columns)}")
        print(f"Final shape: {df.shape}")

        return df

    @timed
    def train_regime_detector(self, df: pd.DataFrame):
        """Train HMM regime detector"""
        from ml_trading_bot.models.regime_detector import RegimeDetector

        print("\n" + "=" * 60)
        print("STEP 3: TRAINING REGIME DETECTOR (HMM)")
        print("=" * 60)

        detector = RegimeDetector(n_states=3, n_iter=100)
        detector.fit(df)

        # Statistics
        print("\nRegime Statistics:")
        stats = detector.get_regime_statistics(df)
        print(stats.to_string(index=False))

        # Test on known events
        print("\nValidation on known events:")

        # Brexit
        brexit = df['2016-06-20':'2016-06-30']
        if len(brexit) > 0:
            regime = detector.get_current_regime(brexit)
            print(f"  Brexit (Jun 2016): {regime['regime_name']} ({regime['confidence']:.1%})")

        # COVID
        covid = df['2020-03-01':'2020-03-31']
        if len(covid) > 0:
            regime = detector.get_current_regime(covid)
            print(f"  COVID (Mar 2020): {regime['regime_name']} ({regime['confidence']:.1%})")

        # Normal period
        normal = df['2024-07-01':'2024-07-31']
        if len(normal) > 0:
            regime = detector.get_current_regime(normal)
            print(f"  Normal (Jul 2024): {regime['regime_name']} ({regime['confidence']:.1%})")

        # Save model
        save_path = self.models_dir / "regime_hmm.pkl"
        detector.save(str(save_path))

        return detector

    @timed
    def train_signal_classifier(self, df: pd.DataFrame):
        """Train XGBoost + Random Forest signal classifier"""
        from ml_trading_bot.models.signal_classifier import SignalClassifier

        print("\n" + "=" * 60)
        print("STEP 4: TRAINING SIGNAL CLASSIFIER")
        print("=" * 60)

        # Split train/test (train on 2015-2023, test on 2024+)
        # Handle timezone-aware index
        train_end = pd.Timestamp("2024-01-01", tz='UTC') if df.index.tz else pd.Timestamp("2024-01-01")
        train_df = df[df.index < train_end]
        test_df = df[df.index >= train_end]

        print(f"Train set: {len(train_df):,} samples (2015-2023)")
        print(f"Test set:  {len(test_df):,} samples (2024-2026)")

        # Initialize classifier
        classifier = SignalClassifier(
            lookahead_hours=24,
            threshold_pct=0.003,
            use_xgboost=True,
            use_random_forest=True
        )

        # Train
        print("\nTraining ensemble (XGBoost + Random Forest)...")
        classifier.fit(train_df, verbose=True)

        # Evaluate on test set
        print("\nEvaluating on test set (2024-2026)...")
        metrics = classifier.evaluate(test_df, verbose=True)

        # Feature importance
        print("\nTop 15 Important Features:")
        importance = classifier.get_feature_importance(15)
        print(importance.to_string(index=False))

        # Save model
        save_path = self.models_dir / "signal_classifier.pkl"
        classifier.save(str(save_path))

        return classifier, metrics

    def run(self):
        """Run complete training pipeline"""
        total_start = time.time()

        print("\n" + "#" * 60)
        print("#" + " " * 58 + "#")
        print("#    ML TRADING BOT - FULL TRAINING PIPELINE" + " " * 14 + "#")
        print("#    11 Years Data (2015-2026)" + " " * 28 + "#")
        print("#" + " " * 58 + "#")
        print("#" * 60)

        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Load data
            step1_start = time.time()
            df = self.load_data()
            self.timings['load_data'] = time.time() - step1_start

            # Step 2: Compute features
            step2_start = time.time()
            df = self.compute_features(df)
            self.timings['compute_features'] = time.time() - step2_start

            # Step 3: Train regime detector
            step3_start = time.time()
            regime_detector = self.train_regime_detector(df)
            self.timings['train_regime'] = time.time() - step3_start

            # Step 4: Train signal classifier
            step4_start = time.time()
            signal_classifier, metrics = self.train_signal_classifier(df)
            self.timings['train_signal'] = time.time() - step4_start

            # Summary
            total_time = time.time() - total_start

            print("\n" + "=" * 60)
            print("TRAINING COMPLETE!")
            print("=" * 60)

            print("\nTiming Summary:")
            print(f"  Load data:        {self.timings['load_data']:.1f}s")
            print(f"  Compute features: {self.timings['compute_features']:.1f}s")
            print(f"  Train regime:     {self.timings['train_regime']:.1f}s")
            print(f"  Train signal:     {self.timings['train_signal']:.1f}s")
            print(f"  " + "-" * 30)
            print(f"  TOTAL:            {total_time:.1f}s ({total_time/60:.1f} minutes)")

            print("\nSaved Models:")
            print(f"  {self.models_dir / 'regime_hmm.pkl'}")
            print(f"  {self.models_dir / 'signal_classifier.pkl'}")

            print("\nKey Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  F1 Score: {metrics['f1_macro']:.2%}")

            return {
                'success': True,
                'total_time': total_time,
                'timings': self.timings,
                'metrics': metrics
            }

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    trainer = ModelTrainer()
    result = trainer.run()

    if result['success']:
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training failed!")
        print("=" * 60)
