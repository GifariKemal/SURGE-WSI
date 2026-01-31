"""
Train Production HMM Model
===========================

Trains the HMM regime detector on all available data
and saves it for production use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import pandas as pd
import numpy as np

from models.regime_detector import RegimeDetector

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}


def load_all_data() -> pd.DataFrame:
    """Load all available GBPUSD H1 data"""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = 'GBPUSD'
        AND timeframe = 'H1'
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features needed for HMM"""
    df = df.copy()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    # ADX
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0), 0
    )
    df['minus_dm'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0), 0
    )
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx_14'] = df['dx'].rolling(14).mean()

    return df.fillna(method='ffill').fillna(0)


def main():
    print("=" * 60)
    print("TRAINING PRODUCTION HMM MODEL")
    print("=" * 60)

    # Load data
    print("\n1. Loading all historical data...")
    df = load_all_data()
    print(f"   Loaded {len(df):,} bars")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Add features
    print("\n2. Computing features...")
    df = add_features(df)

    # Train HMM
    print("\n3. Training HMM Regime Detector...")
    detector = RegimeDetector(n_states=3, n_iter=200)
    detector.fit(df)

    # Get statistics
    print("\n4. Regime Statistics:")
    stats = detector.get_regime_statistics(df)
    print(stats.to_string(index=False))

    # Test on known events
    print("\n5. Testing on known events:")

    # COVID March 2020
    covid = df['2020-03-01':'2020-03-31']
    if len(covid) > 0:
        regime = detector.get_current_regime(covid)
        print(f"   COVID (Mar 2020): {regime['regime_name']} (conf: {regime['confidence']:.1%})")

    # Normal period
    normal = df['2024-06-01':'2024-06-30']
    if len(normal) > 0:
        regime = detector.get_current_regime(normal)
        print(f"   Normal (Jun 2024): {regime['regime_name']} (conf: {regime['confidence']:.1%})")

    # Recent
    recent = df.tail(100)
    regime = detector.get_current_regime(recent)
    print(f"   Current: {regime['regime_name']} (conf: {regime['confidence']:.1%})")

    # Save model
    print("\n6. Saving model...")
    save_dir = Path(__file__).parent.parent / "saved_models"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "regime_hmm.pkl"
    detector.save(str(save_path))

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print(f"Saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
