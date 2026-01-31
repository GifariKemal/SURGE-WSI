"""
Data Loader for ML Trading Bot
==============================

Loads H1 OHLCV data from TimescaleDB and existing market profiles.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json
import asyncio
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.data.db_handler import DBHandler


class DataLoader:
    """
    Load and prepare data for ML training

    Data sources:
    1. TimescaleDB: Raw H1 OHLCV data (67,940 bars)
    2. JSON profiles: Pre-computed market condition profiles
    """

    def __init__(self, symbol: str = "GBPUSD", timeframe: str = "H1"):
        self.symbol = symbol
        self.timeframe = timeframe

        # Load DB config from environment
        self.db = DBHandler(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5434")),
            database=os.getenv("POSTGRES_DB", "surge_wsi"),
            user=os.getenv("POSTGRES_USER", "surge_wsi"),
            password=os.getenv("POSTGRES_PASSWORD", "surge_wsi_secret")
        )

        # Paths to existing data
        self.project_root = Path(__file__).parent.parent.parent
        self.profiles_dir = self.project_root / "backtest" / "market_analysis"

    async def load_ohlcv(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-01-31"
    ) -> pd.DataFrame:
        """
        Load H1 OHLCV data from TimescaleDB

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime (UTC)
        """
        try:
            await self.db.connect()

            df = await self.db.get_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100000,  # More than we need
                start_time=datetime.fromisoformat(start_date),
                end_time=datetime.fromisoformat(end_date)
            )

            if df.empty:
                raise ValueError(f"No data found for {self.symbol} between {start_date} and {end_date}")

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

            print(f"Loaded {len(df):,} {self.timeframe} bars from {df.index[0]} to {df.index[-1]}")

            return df

        finally:
            await self.db.disconnect()

    def load_ohlcv_sync(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-01-31"
    ) -> pd.DataFrame:
        """Synchronous wrapper for load_ohlcv"""
        return asyncio.run(self.load_ohlcv(start_date, end_date))

    def load_monthly_profiles(self) -> Dict[str, Any]:
        """
        Load pre-computed monthly profiles

        Returns:
            Dict with monthly profiles (133 months)
        """
        profiles_path = self.profiles_dir / "monthly_profiles.json"

        if not profiles_path.exists():
            raise FileNotFoundError(f"Monthly profiles not found: {profiles_path}")

        with open(profiles_path, 'r') as f:
            profiles = json.load(f)

        print(f"Loaded {len(profiles)} monthly profiles")
        return profiles

    def load_daily_profiles(self) -> Dict[str, Any]:
        """
        Load pre-computed daily profiles

        Returns:
            Dict with daily profiles (2,862 days)
        """
        profiles_path = self.profiles_dir / "daily_profiles.json"

        if not profiles_path.exists():
            raise FileNotFoundError(f"Daily profiles not found: {profiles_path}")

        with open(profiles_path, 'r') as f:
            profiles = json.load(f)

        print(f"Loaded {len(profiles)} daily profiles")
        return profiles

    def profiles_to_dataframe(self, profiles: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert profile dict to DataFrame

        Args:
            profiles: Dict of profiles (monthly or daily)

        Returns:
            DataFrame with profile data
        """
        records = []
        for date_key, profile in profiles.items():
            record = {'date': date_key}
            record.update(profile)
            records.append(record)

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        return df

    def merge_ohlcv_with_daily_profiles(
        self,
        ohlcv: pd.DataFrame,
        daily_profiles: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Merge H1 OHLCV data with daily profile features

        This adds pre-computed daily metrics to each H1 bar:
        - quality_score
        - volatility_regime
        - trend_regime
        - atr_pips
        - adx
        - etc.

        Args:
            ohlcv: H1 OHLCV DataFrame
            daily_profiles: Daily profiles dict

        Returns:
            Merged DataFrame with daily features on H1 bars
        """
        # Convert profiles to DataFrame
        profiles_df = self.profiles_to_dataframe(daily_profiles)

        # Add date column to OHLCV for merging
        ohlcv = ohlcv.copy()
        ohlcv['date'] = ohlcv.index.date
        ohlcv['date'] = pd.to_datetime(ohlcv['date'])

        # Select important profile columns
        profile_cols = [
            'quality_score', 'atr_pips', 'adx', 'plus_di', 'minus_di',
            'choppiness', 'price_efficiency', 'daily_range_pips',
            'volatility_regime', 'trend_regime', 'trend_direction',
            'asian_range', 'london_range', 'ny_range',
            'is_tradeable', 'risk_multiplier'
        ]

        # Filter existing columns
        available_cols = [c for c in profile_cols if c in profiles_df.columns]
        profiles_subset = profiles_df[available_cols].copy()
        profiles_subset.index = profiles_subset.index.normalize()

        # Merge on date
        merged = ohlcv.merge(
            profiles_subset,
            left_on='date',
            right_index=True,
            how='left'
        )

        # Drop temporary date column
        merged.drop(columns=['date'], inplace=True)

        # Forward fill missing values (for any days without profiles)
        merged[available_cols] = merged[available_cols].fillna(method='ffill')

        print(f"Merged {len(merged):,} bars with {len(available_cols)} profile features")

        return merged

    def prepare_training_data(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2023-12-31",
        include_profiles: bool = True
    ) -> pd.DataFrame:
        """
        Prepare complete training dataset

        Args:
            start_date: Training start date
            end_date: Training end date
            include_profiles: Whether to merge daily profiles

        Returns:
            Complete DataFrame ready for feature engineering
        """
        # Load OHLCV
        ohlcv = self.load_ohlcv_sync(start_date, end_date)

        if include_profiles:
            # Load and merge daily profiles
            daily_profiles = self.load_daily_profiles()
            data = self.merge_ohlcv_with_daily_profiles(ohlcv, daily_profiles)
        else:
            data = ohlcv

        return data

    def split_train_test(
        self,
        data: pd.DataFrame,
        test_start: str = "2024-01-01"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-based)

        Args:
            data: Full dataset
            test_start: Start of test period

        Returns:
            (train_df, test_df)
        """
        test_start_dt = pd.to_datetime(test_start)

        train = data[data.index < test_start_dt].copy()
        test = data[data.index >= test_start_dt].copy()

        print(f"Train set: {len(train):,} samples ({train.index[0]} to {train.index[-1]})")
        print(f"Test set:  {len(test):,} samples ({test.index[0]} to {test.index[-1]})")

        return train, test

    def get_walk_forward_splits(
        self,
        data: pd.DataFrame,
        train_months: int = 24,
        test_months: int = 3
    ) -> list:
        """
        Generate walk-forward validation splits

        Args:
            data: Full dataset
            train_months: Training window size in months
            test_months: Test window size in months

        Returns:
            List of (train_df, test_df) tuples
        """
        splits = []
        start_date = data.index.min()
        end_date = data.index.max()

        current_train_start = start_date
        current_train_end = start_date + pd.DateOffset(months=train_months)

        while current_train_end + pd.DateOffset(months=test_months) <= end_date:
            test_start = current_train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            train_mask = (data.index >= current_train_start) & (data.index < current_train_end)
            test_mask = (data.index >= test_start) & (data.index < test_end)

            train_df = data[train_mask].copy()
            test_df = data[test_mask].copy()

            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))

            # Move forward by test_months
            current_train_start += pd.DateOffset(months=test_months)
            current_train_end += pd.DateOffset(months=test_months)

        print(f"Generated {len(splits)} walk-forward splits")
        return splits


def load_data_quick(
    start: str = "2015-01-01",
    end: str = "2026-01-31"
) -> pd.DataFrame:
    """
    Quick function to load data for notebooks/testing

    Usage:
        from ml_trading_bot.training.data_loader import load_data_quick
        df = load_data_quick("2020-01-01", "2024-12-31")
    """
    loader = DataLoader()
    return loader.prepare_training_data(start, end, include_profiles=True)


if __name__ == "__main__":
    # Test data loading
    print("=" * 60)
    print("Testing ML Trading Bot Data Loader")
    print("=" * 60)

    loader = DataLoader()

    # Load OHLCV
    print("\n1. Loading H1 OHLCV data...")
    ohlcv = loader.load_ohlcv_sync("2024-01-01", "2024-03-31")
    print(f"   Shape: {ohlcv.shape}")
    print(f"   Columns: {list(ohlcv.columns)}")

    # Load profiles
    print("\n2. Loading monthly profiles...")
    monthly = loader.load_monthly_profiles()
    print(f"   Months: {len(monthly)}")

    print("\n3. Loading daily profiles...")
    daily = loader.load_daily_profiles()
    print(f"   Days: {len(daily)}")

    # Merge data
    print("\n4. Merging OHLCV with daily profiles...")
    merged = loader.merge_ohlcv_with_daily_profiles(ohlcv, daily)
    print(f"   Shape: {merged.shape}")
    print(f"   Columns: {list(merged.columns)}")

    # Split data
    print("\n5. Splitting train/test...")
    train, test = loader.split_train_test(merged, "2024-03-01")

    print("\n" + "=" * 60)
    print("Data loading test complete!")
    print("=" * 60)
