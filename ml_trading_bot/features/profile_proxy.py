"""
Profile Proxy Features
======================

Computes daily profile-like features from H1 data for live trading.
These features are normally merged from daily profile database during training,
but need to be computed on-the-fly for live execution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone


class ProfileProxyFeatures:
    """
    Generate profile features that are normally loaded from daily profiles.

    This allows the ML model to work with live data without requiring
    pre-computed daily profiles.
    """

    def __init__(self, pip_size: float = 0.0001):
        self.pip_size = pip_size

    def _get_col(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Get column with case-insensitive lookup"""
        if name in df.columns:
            return df[name]
        if name.lower() in df.columns:
            return df[name.lower()]
        if name.upper() in df.columns:
            return df[name.upper()]
        raise KeyError(f"Column '{name}' not found")

    def add_atr_pips(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR in pips (converted from price)"""
        if 'atr_pips' not in df.columns:
            if 'atr_14' in df.columns:
                df['atr_pips'] = df['atr_14'] / self.pip_size
            elif 'atr_20' in df.columns:
                df['atr_pips'] = df['atr_20'] / self.pip_size
            else:
                # Compute ATR manually
                high = self._get_col(df, 'high')
                low = self._get_col(df, 'low')
                close = self._get_col(df, 'close')

                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)

                df['atr_pips'] = tr.rolling(14).mean() / self.pip_size

        return df

    def add_adx_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX as 'adx' (proxy from adx_14 if available)"""
        if 'adx' not in df.columns:
            if 'adx_14' in df.columns:
                df['adx'] = df['adx_14']
            else:
                # Already computed by technical features
                df['adx'] = 25.0  # Default neutral value

        return df

    def add_daily_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily range in pips"""
        if 'daily_range_pips' not in df.columns:
            high = self._get_col(df, 'high')
            low = self._get_col(df, 'low')

            # Rolling 24 hours (24 H1 bars)
            daily_high = high.rolling(24, min_periods=1).max()
            daily_low = low.rolling(24, min_periods=1).min()
            df['daily_range_pips'] = (daily_high - daily_low) / self.pip_size

        return df

    def add_session_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session ranges (Asian, London, NY)"""
        high = self._get_col(df, 'high')
        low = self._get_col(df, 'low')

        # Get hour from index
        if hasattr(df.index, 'hour'):
            hours = df.index.hour
        else:
            hours = pd.to_datetime(df.index).hour

        # Session masks
        asian_mask = (hours >= 0) & (hours < 8)   # 00:00-08:00 UTC
        london_mask = (hours >= 8) & (hours < 16)  # 08:00-16:00 UTC
        ny_mask = (hours >= 13) & (hours < 21)     # 13:00-21:00 UTC

        # Compute ranges using rolling window within session hours
        if 'asian_range' not in df.columns:
            # Use rolling window for approximate session range
            df['asian_range'] = (
                high.rolling(8, min_periods=1).max() -
                low.rolling(8, min_periods=1).min()
            ) / self.pip_size

        if 'london_range' not in df.columns:
            df['london_range'] = (
                high.rolling(8, min_periods=1).max() -
                low.rolling(8, min_periods=1).min()
            ) / self.pip_size

        if 'ny_range' not in df.columns:
            df['ny_range'] = (
                high.rolling(8, min_periods=1).max() -
                low.rolling(8, min_periods=1).min()
            ) / self.pip_size

        return df

    def add_price_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price efficiency (net move / total path)"""
        if 'price_efficiency' not in df.columns:
            close = self._get_col(df, 'close')
            high = self._get_col(df, 'high')
            low = self._get_col(df, 'low')

            # Net change over 24 hours
            net_change = (close - close.shift(24)).abs()

            # Total path (sum of bar ranges)
            bar_range = high - low
            total_path = bar_range.rolling(24, min_periods=1).sum()

            # Efficiency = net_change / total_path
            df['price_efficiency'] = net_change / (total_path + 0.00001)
            df['price_efficiency'] = df['price_efficiency'].clip(0, 1)

        return df

    def add_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add quality score (composite trading quality metric)

        Quality is based on:
        - Volatility (ATR) - higher is better up to a point
        - Trend strength (ADX) - higher is better
        - Price efficiency - higher is better
        - Choppiness - lower is better
        """
        if 'quality_score' not in df.columns:
            score = pd.Series(50.0, index=df.index)  # Base score

            # ATR component (0-25 pts)
            if 'atr_pips' in df.columns:
                atr = df['atr_pips']
                # Optimal ATR around 15-25 pips
                atr_score = pd.Series(0.0, index=df.index)
                atr_score = np.where(atr >= 10, atr_score + 10, atr_score)
                atr_score = np.where(atr >= 15, atr_score + 10, atr_score)
                atr_score = np.where((atr >= 20) & (atr <= 40), atr_score + 5, atr_score)
                score += atr_score

            # ADX component (0-25 pts)
            adx_col = 'adx' if 'adx' in df.columns else 'adx_14'
            if adx_col in df.columns:
                adx = df[adx_col]
                adx_score = (adx / 100 * 25).clip(0, 25)
                score += adx_score

            # Efficiency component (0-15 pts)
            if 'price_efficiency' in df.columns:
                eff = df['price_efficiency']
                eff_score = eff * 15
                score += eff_score

            # Choppiness penalty (0 to -15 pts)
            if 'choppiness' in df.columns:
                chop = df['choppiness']
                # Penalize high choppiness (ranging markets)
                chop_penalty = np.where(chop > 61.8, -15, 0)
                chop_penalty = np.where((chop > 50) & (chop <= 61.8), -7, chop_penalty)
                score += chop_penalty

            df['quality_score'] = score.clip(0, 100)

        return df

    def add_risk_multiplier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk multiplier based on quality and market conditions"""
        if 'risk_multiplier' not in df.columns:
            multiplier = pd.Series(1.0, index=df.index)

            # Reduce risk in low quality
            if 'quality_score' in df.columns:
                multiplier = np.where(df['quality_score'] < 40, 0.5, multiplier)
                multiplier = np.where(df['quality_score'] < 50, 0.7, multiplier)

            # Reduce risk in high choppiness
            if 'choppiness' in df.columns:
                multiplier = np.where(df['choppiness'] > 61.8, multiplier * 0.5, multiplier)

            # Reduce risk in very high volatility
            if 'atr_pips' in df.columns:
                multiplier = np.where(df['atr_pips'] > 50, multiplier * 0.7, multiplier)

            df['risk_multiplier'] = pd.Series(multiplier, index=df.index).clip(0.3, 1.5)

        return df

    def add_quality_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality-derived features"""
        if 'quality_score' in df.columns:
            if 'quality_high' not in df.columns:
                df['quality_high'] = (df['quality_score'] >= 60).astype(int)
            if 'quality_low' not in df.columns:
                df['quality_low'] = (df['quality_score'] < 40).astype(int)
            if 'quality_zscore' not in df.columns:
                mean = df['quality_score'].rolling(20, min_periods=1).mean()
                std = df['quality_score'].rolling(20, min_periods=1).std()
                df['quality_zscore'] = (df['quality_score'] - mean) / (std + 0.00001)

        if 'risk_multiplier' in df.columns:
            if 'risk_reduced' not in df.columns:
                df['risk_reduced'] = (df['risk_multiplier'] < 1.0).astype(int)

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all profile proxy features"""
        df = df.copy()

        print("  Adding profile proxy features...")
        df = self.add_atr_pips(df)
        df = self.add_adx_proxy(df)
        df = self.add_daily_range(df)
        df = self.add_session_ranges(df)
        df = self.add_price_efficiency(df)
        df = self.add_quality_score(df)
        df = self.add_risk_multiplier(df)
        df = self.add_quality_derivatives(df)

        return df


def add_profile_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to add all profile proxy features"""
    proxy = ProfileProxyFeatures()
    return proxy.add_all_features(df)
