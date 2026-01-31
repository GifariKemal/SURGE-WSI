"""
Regime-based Features
=====================

Features derived from market regime analysis.
Uses existing daily profiles and adds regime-specific features.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class RegimeFeatures:
    """
    Generate regime-based features from existing profiles and real-time data

    Regime Categories:
    - Volatility: HIGH, NORMAL, LOW
    - Trend: TRENDING, RANGING, MIXED
    - Quality: 0-100 score
    """

    # Volatility thresholds (ATR percentiles)
    VOL_HIGH_THRESHOLD = 0.7
    VOL_LOW_THRESHOLD = 0.3

    # ADX thresholds
    ADX_STRONG = 25
    ADX_WEAK = 20

    # Choppiness thresholds
    CHOP_RANGING = 61.8
    CHOP_TRENDING = 38.2

    def __init__(self):
        pass

    def add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility regime features based on ATR
        """
        df = df.copy()

        # Check if ATR already exists (from profiles or technical)
        if 'atr_14' not in df.columns and 'atr_pips' not in df.columns:
            # Calculate ATR if not present
            import pandas_ta as ta
            high = df['high'] if 'high' in df.columns else df['High']
            low = df['low'] if 'low' in df.columns else df['Low']
            close = df['close'] if 'close' in df.columns else df['Close']
            df['atr_14'] = ta.atr(high, low, close, length=14)

        atr_col = 'atr_14' if 'atr_14' in df.columns else 'atr_pips'

        # Calculate ATR percentile
        df['atr_percentile'] = df[atr_col].rolling(500, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Volatility regime encoding
        df['vol_regime_high'] = (df['atr_percentile'] >= self.VOL_HIGH_THRESHOLD).astype(int)
        df['vol_regime_low'] = (df['atr_percentile'] <= self.VOL_LOW_THRESHOLD).astype(int)
        df['vol_regime_normal'] = (
            (df['atr_percentile'] > self.VOL_LOW_THRESHOLD) &
            (df['atr_percentile'] < self.VOL_HIGH_THRESHOLD)
        ).astype(int)

        # Volatility z-score
        df['vol_zscore'] = (
            (df[atr_col] - df[atr_col].rolling(100).mean()) /
            (df[atr_col].rolling(100).std() + 0.00001)
        )

        # Volatility change
        df['vol_change'] = df[atr_col].pct_change(24)  # Daily change

        # Volatility expansion/contraction
        df['vol_expanding'] = (df[atr_col] > df[atr_col].shift(24)).astype(int)

        return df

    def add_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend regime features based on ADX and directional movement
        """
        df = df.copy()

        # Check for ADX
        if 'adx_14' not in df.columns and 'adx' not in df.columns:
            import pandas_ta as ta
            high = df['high'] if 'high' in df.columns else df['High']
            low = df['low'] if 'low' in df.columns else df['Low']
            close = df['close'] if 'close' in df.columns else df['Close']
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None:
                for col in adx_df.columns:
                    if 'ADX' in col.upper() and 'DM' not in col.upper():
                        df['adx_14'] = adx_df[col]
                        break

        adx_col = 'adx_14' if 'adx_14' in df.columns else 'adx'

        if adx_col in df.columns:
            # Trend strength encoding
            df['trend_strong'] = (df[adx_col] >= self.ADX_STRONG).astype(int)
            df['trend_weak'] = (df[adx_col] < self.ADX_WEAK).astype(int)

            # ADX change
            df['adx_change'] = df[adx_col] - df[adx_col].shift(24)
            df['adx_increasing'] = (df['adx_change'] > 0).astype(int)

        # Trend direction from DI
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            df['trend_bullish'] = (df['plus_di'] > df['minus_di']).astype(int)
            df['trend_bearish'] = (df['plus_di'] < df['minus_di']).astype(int)
        elif 'di_diff' in df.columns:
            df['trend_bullish'] = (df['di_diff'] > 0).astype(int)
            df['trend_bearish'] = (df['di_diff'] < 0).astype(int)

        return df

    def add_choppiness_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add choppiness regime features
        """
        df = df.copy()

        # Calculate Choppiness Index if not present
        if 'choppiness' not in df.columns:
            high = df['high'] if 'high' in df.columns else df['High']
            low = df['low'] if 'low' in df.columns else df['Low']
            close = df['close'] if 'close' in df.columns else df['Close']

            # Choppiness Index calculation
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)

            atr_sum = tr.rolling(14).sum()
            high_low_diff = high.rolling(14).max() - low.rolling(14).min()

            df['choppiness'] = 100 * np.log10(atr_sum / (high_low_diff + 0.00001)) / np.log10(14)

        # Choppiness regime
        df['is_ranging'] = (df['choppiness'] > self.CHOP_RANGING).astype(int)
        df['is_trending'] = (df['choppiness'] < self.CHOP_TRENDING).astype(int)
        df['is_mixed'] = (
            (df['choppiness'] >= self.CHOP_TRENDING) &
            (df['choppiness'] <= self.CHOP_RANGING)
        ).astype(int)

        return df

    def add_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add quality-based features from existing profiles
        """
        df = df.copy()

        # If quality_score exists from daily profiles
        if 'quality_score' in df.columns:
            # Quality regime
            df['quality_high'] = (df['quality_score'] >= 60).astype(int)
            df['quality_low'] = (df['quality_score'] < 40).astype(int)

            # Quality z-score
            df['quality_zscore'] = (
                (df['quality_score'] - df['quality_score'].rolling(20).mean()) /
                (df['quality_score'].rolling(20).std() + 0.00001)
            )

        # If risk_multiplier exists
        if 'risk_multiplier' in df.columns:
            df['risk_reduced'] = (df['risk_multiplier'] < 1.0).astype(int)

        # If is_tradeable exists
        if 'is_tradeable' in df.columns:
            df['tradeable'] = df['is_tradeable'].astype(int)

        return df

    def add_regime_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features for regime transitions
        """
        df = df.copy()

        # Volatility regime transitions
        if 'vol_regime_high' in df.columns:
            df['entering_high_vol'] = (
                (df['vol_regime_high'] == 1) &
                (df['vol_regime_high'].shift(1) == 0)
            ).astype(int)
            df['exiting_high_vol'] = (
                (df['vol_regime_high'] == 0) &
                (df['vol_regime_high'].shift(1) == 1)
            ).astype(int)

        # Trend transitions
        if 'trend_strong' in df.columns:
            df['entering_strong_trend'] = (
                (df['trend_strong'] == 1) &
                (df['trend_strong'].shift(1) == 0)
            ).astype(int)
            df['exiting_strong_trend'] = (
                (df['trend_strong'] == 0) &
                (df['trend_strong'].shift(1) == 1)
            ).astype(int)

        # Direction changes
        if 'trend_bullish' in df.columns:
            df['bull_to_bear'] = (
                (df['trend_bullish'] == 0) &
                (df['trend_bullish'].shift(1) == 1)
            ).astype(int)
            df['bear_to_bull'] = (
                (df['trend_bullish'] == 1) &
                (df['trend_bullish'].shift(1) == 0)
            ).astype(int)

        return df

    def compute_composite_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite regime score for trading decisions
        """
        df = df.copy()

        # Composite score (higher = better for trading)
        score = 50  # Base score

        # Volatility component (-20 to +10)
        if 'vol_regime_high' in df.columns:
            vol_score = np.where(df['vol_regime_high'] == 1, -20,
                       np.where(df['vol_regime_low'] == 1, -5, 10))
            score = score + vol_score

        # Trend component (-10 to +20)
        if 'trend_strong' in df.columns:
            trend_score = np.where(df['trend_strong'] == 1, 20,
                         np.where(df['trend_weak'] == 1, -10, 5))
            score = score + trend_score

        # Choppiness component (-20 to +10)
        if 'is_ranging' in df.columns:
            chop_score = np.where(df['is_ranging'] == 1, -20,
                        np.where(df['is_trending'] == 1, 10, 0))
            score = score + chop_score

        df['composite_regime_score'] = score
        df['composite_regime_good'] = (score >= 60).astype(int)
        df['composite_regime_bad'] = (score < 40).astype(int)

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all regime features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with regime features added
        """
        print("  Adding volatility regime features...")
        df = self.add_volatility_regime(df)

        print("  Adding trend regime features...")
        df = self.add_trend_regime(df)

        print("  Adding choppiness regime features...")
        df = self.add_choppiness_regime(df)

        print("  Adding quality features...")
        df = self.add_quality_features(df)

        print("  Adding regime transition features...")
        df = self.add_regime_transitions(df)

        print("  Computing composite regime score...")
        df = self.compute_composite_regime(df)

        return df

    def get_feature_names(self) -> list:
        """Get list of all regime feature names"""
        return [
            # Volatility regime
            'atr_percentile', 'vol_regime_high', 'vol_regime_low', 'vol_regime_normal',
            'vol_zscore', 'vol_change', 'vol_expanding',

            # Trend regime
            'trend_strong', 'trend_weak', 'adx_change', 'adx_increasing',
            'trend_bullish', 'trend_bearish',

            # Choppiness
            'choppiness', 'is_ranging', 'is_trending', 'is_mixed',

            # Quality
            'quality_high', 'quality_low', 'quality_zscore',
            'risk_reduced', 'tradeable',

            # Transitions
            'entering_high_vol', 'exiting_high_vol',
            'entering_strong_trend', 'exiting_strong_trend',
            'bull_to_bear', 'bear_to_bull',

            # Composite
            'composite_regime_score', 'composite_regime_good', 'composite_regime_bad'
        ]


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to compute regime features

    Usage:
        from ml_trading_bot.features.regime import compute_regime_features
        df_with_features = compute_regime_features(ohlcv_df)
    """
    regime = RegimeFeatures()
    return regime.add_all_features(df)


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from training.data_loader import DataLoader

    print("=" * 60)
    print("Testing Regime Features")
    print("=" * 60)

    # Load sample data with profiles
    loader = DataLoader()
    df = loader.prepare_training_data("2024-01-01", "2024-06-30")

    print(f"\nInput shape: {df.shape}")

    # Compute features
    regime = RegimeFeatures()
    df_features = regime.add_all_features(df)

    print(f"\nOutput shape: {df_features.shape}")

    print(f"\nVolatility regime distribution:")
    print(f"  High: {df_features['vol_regime_high'].sum()}")
    print(f"  Normal: {df_features['vol_regime_normal'].sum()}")
    print(f"  Low: {df_features['vol_regime_low'].sum()}")

    print(f"\nComposite regime score stats:")
    print(df_features['composite_regime_score'].describe())

    print("\n" + "=" * 60)
    print("Regime features test complete!")
    print("=" * 60)
