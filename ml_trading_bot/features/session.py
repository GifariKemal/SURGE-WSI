"""
Session-based Features
======================

Features based on trading session timing (Asian, London, NY)
and calendar effects (day of week, month).
"""

import pandas as pd
import numpy as np
from typing import Optional


class SessionFeatures:
    """
    Generate session and calendar-based features

    Trading Sessions (UTC):
    - Asian: 00:00 - 08:00
    - London: 08:00 - 16:00
    - New York: 13:00 - 21:00
    - London/NY Overlap: 13:00 - 16:00
    """

    # Session definitions (UTC hours)
    SESSIONS = {
        'asian': (0, 8),
        'london': (8, 16),
        'new_york': (13, 21),
        'overlap': (13, 16)
    }

    # Best/worst hours from historical analysis
    BEST_HOURS = [10, 14, 15]  # London morning, overlap
    WORST_HOURS = [0, 1, 22, 23]  # Asian night, late NY

    def __init__(self):
        pass

    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session identification features
        """
        df = df.copy()

        # Extract hour from index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        hour = df.index.hour

        # Session flags
        df['is_asian'] = ((hour >= 0) & (hour < 8)).astype(int)
        df['is_london'] = ((hour >= 8) & (hour < 16)).astype(int)
        df['is_new_york'] = ((hour >= 13) & (hour < 21)).astype(int)
        df['is_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)

        # Session as categorical (one-hot encoded)
        df['session_asian_only'] = ((hour >= 0) & (hour < 8)).astype(int)
        df['session_london_only'] = ((hour >= 8) & (hour < 13)).astype(int)
        df['session_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)
        df['session_ny_only'] = ((hour >= 16) & (hour < 21)).astype(int)
        df['session_off_hours'] = ((hour >= 21) | (hour < 0)).astype(int)

        # Best/worst hour flags
        df['is_best_hour'] = hour.isin(self.BEST_HOURS).astype(int)
        df['is_worst_hour'] = hour.isin(self.WORST_HOURS).astype(int)

        # Hour of day (cyclical encoding for ML)
        df['hour'] = hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features
        """
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        # Day of week
        dow = df.index.dayofweek  # 0=Monday, 6=Sunday

        df['day_of_week'] = dow
        df['is_monday'] = (dow == 0).astype(int)
        df['is_tuesday'] = (dow == 1).astype(int)
        df['is_wednesday'] = (dow == 2).astype(int)
        df['is_thursday'] = (dow == 3).astype(int)
        df['is_friday'] = (dow == 4).astype(int)

        # Cyclical encoding
        df['dow_sin'] = np.sin(2 * np.pi * dow / 5)  # 5 trading days
        df['dow_cos'] = np.cos(2 * np.pi * dow / 5)

        # Month features
        month = df.index.month
        df['month'] = month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Quarter
        df['quarter'] = df.index.quarter

        # Day of month
        dom = df.index.day
        df['day_of_month'] = dom
        df['is_month_start'] = (dom <= 3).astype(int)
        df['is_month_end'] = (dom >= 28).astype(int)

        # Week of month
        df['week_of_month'] = (dom - 1) // 7 + 1

        return df

    def add_time_based_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling statistics by time of day
        """
        df = df.copy()

        if 'close' not in df.columns and 'Close' not in df.columns:
            return df

        close = df['close'] if 'close' in df.columns else df['Close']
        hour = df.index.hour

        # Returns by session (rolling averages)
        returns = close.pct_change()

        # Session returns (rolling 20-day average for same hour)
        df['hour_avg_return'] = returns.groupby(hour).transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )

        # Session volatility
        df['hour_avg_volatility'] = returns.groupby(hour).transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all session and calendar features

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            DataFrame with session features added
        """
        print("  Adding session features...")
        df = self.add_session_features(df)

        print("  Adding calendar features...")
        df = self.add_calendar_features(df)

        print("  Adding time-based statistics...")
        df = self.add_time_based_stats(df)

        return df

    def get_feature_names(self) -> list:
        """Get list of all session feature names"""
        return [
            # Session flags
            'is_asian', 'is_london', 'is_new_york', 'is_overlap',
            'session_asian_only', 'session_london_only', 'session_overlap',
            'session_ny_only', 'session_off_hours',
            'is_best_hour', 'is_worst_hour',
            'hour', 'hour_sin', 'hour_cos',

            # Calendar
            'day_of_week', 'is_monday', 'is_tuesday', 'is_wednesday',
            'is_thursday', 'is_friday', 'dow_sin', 'dow_cos',
            'month', 'month_sin', 'month_cos', 'quarter',
            'day_of_month', 'is_month_start', 'is_month_end', 'week_of_month',

            # Time-based stats
            'hour_avg_return', 'hour_avg_volatility'
        ]


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to compute session features

    Usage:
        from ml_trading_bot.features.session import compute_session_features
        df_with_features = compute_session_features(ohlcv_df)
    """
    session = SessionFeatures()
    return session.add_all_features(df)


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from training.data_loader import DataLoader

    print("=" * 60)
    print("Testing Session Features")
    print("=" * 60)

    # Load sample data
    loader = DataLoader()
    df = loader.load_ohlcv_sync("2024-01-01", "2024-01-31")

    print(f"\nInput shape: {df.shape}")

    # Compute features
    session = SessionFeatures()
    df_features = session.add_all_features(df)

    print(f"\nOutput shape: {df_features.shape}")
    print(f"\nSample session distribution:")
    print(df_features[['is_asian', 'is_london', 'is_new_york', 'is_overlap']].sum())

    print(f"\nSample day distribution:")
    print(df_features['day_of_week'].value_counts().sort_index())

    print("\n" + "=" * 60)
    print("Session features test complete!")
    print("=" * 60)
