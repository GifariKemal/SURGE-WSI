"""
Technical Indicator Features
============================

Computes technical indicators for ML models using pandas-ta.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, List


class TechnicalFeatures:
    """
    Generate technical indicator features from OHLCV data

    Features include:
    - Volatility: ATR, Bollinger Bands, Historical Volatility
    - Trend: ADX, EMA crossovers, MACD
    - Momentum: RSI, Stochastic, ROC
    - Price patterns: Returns, log returns
    """

    def __init__(
        self,
        atr_period: int = 14,
        adx_period: int = 14,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_fast: int = 9,
        ema_slow: int = 21,
        stoch_k: int = 14,
        stoch_d: int = 3
    ):
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d

    def _get_col(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Get column with case-insensitive lookup"""
        if name in df.columns:
            return df[name]
        if name.lower() in df.columns:
            return df[name.lower()]
        if name.upper() in df.columns:
            return df[name.upper()]
        if name.capitalize() in df.columns:
            return df[name.capitalize()]
        raise KeyError(f"Column '{name}' not found in DataFrame")

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return features at multiple timeframes
        """
        close = self._get_col(df, 'close')

        # Simple returns
        df['returns_1h'] = close.pct_change(1)
        df['returns_4h'] = close.pct_change(4)
        df['returns_1d'] = close.pct_change(24)
        df['returns_1w'] = close.pct_change(120)

        # Log returns
        df['log_returns_1h'] = np.log(close / close.shift(1))
        df['log_returns_1d'] = np.log(close / close.shift(24))

        # Cumulative returns
        df['cum_returns_1d'] = close.pct_change(24).rolling(24).sum()

        return df

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility features
        """
        high = self._get_col(df, 'high')
        low = self._get_col(df, 'low')
        close = self._get_col(df, 'close')

        # ATR at different periods
        df['atr_14'] = ta.atr(high, low, close, length=14)
        df['atr_20'] = ta.atr(high, low, close, length=20)

        # ATR ratio (current vs average)
        df['atr_ratio'] = df['atr_14'] / df['atr_14'].rolling(50).mean()

        # Bollinger Bands
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        if bb is not None and not bb.empty:
            bb_cols = bb.columns.tolist()
            # Find the bandwidth column
            bbb_col = [c for c in bb_cols if 'BBB' in c.upper()]
            bbp_col = [c for c in bb_cols if 'BBP' in c.upper()]

            if bbb_col:
                df['bb_width'] = bb[bbb_col[0]]
            if bbp_col:
                df['bb_position'] = bb[bbp_col[0]]

        # Historical volatility
        df['hist_vol_20'] = close.pct_change().rolling(20).std() * np.sqrt(24)  # Annualized for H1
        df['hist_vol_50'] = close.pct_change().rolling(50).std() * np.sqrt(24)

        # Volatility percentile (rolling window reduced for smaller datasets)
        df['vol_percentile'] = df['hist_vol_20'].rolling(200, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # True Range
        df['true_range'] = ta.true_range(high, low, close)
        df['tr_ratio'] = df['true_range'] / df['true_range'].rolling(50).mean()

        return df

    def add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend features
        """
        high = self._get_col(df, 'high')
        low = self._get_col(df, 'low')
        close = self._get_col(df, 'close')

        # ADX and Directional Movement
        adx = ta.adx(high, low, close, length=self.adx_period)
        if adx is not None and not adx.empty:
            adx_cols = adx.columns.tolist()
            for col in adx_cols:
                col_lower = col.lower()
                if 'adx' in col_lower and 'dmp' not in col_lower and 'dmn' not in col_lower:
                    df['adx_14'] = adx[col]
                elif 'dmp' in col_lower or '+di' in col_lower:
                    df['plus_di'] = adx[col]
                elif 'dmn' in col_lower or '-di' in col_lower:
                    df['minus_di'] = adx[col]

        # DI difference (trend direction indicator)
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            df['di_diff'] = df['plus_di'] - df['minus_di']
            df['di_ratio'] = df['plus_di'] / (df['minus_di'] + 0.0001)

        # EMAs
        df['ema_9'] = ta.ema(close, length=9)
        df['ema_21'] = ta.ema(close, length=21)
        df['ema_50'] = ta.ema(close, length=50)
        df['ema_200'] = ta.ema(close, length=200)

        # EMA crossovers
        df['ema_9_21_diff'] = df['ema_9'] - df['ema_21']
        df['ema_50_200_diff'] = df['ema_50'] - df['ema_200']

        # Price position relative to EMAs
        df['price_vs_ema_21'] = (close - df['ema_21']) / df['ema_21']
        df['price_vs_ema_50'] = (close - df['ema_50']) / df['ema_50']

        # MACD
        macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd is not None and not macd.empty:
            macd_cols = macd.columns.tolist()
            for col in macd_cols:
                col_lower = col.lower()
                if 'macdh' in col_lower or 'hist' in col_lower:
                    df['macd_hist'] = macd[col]
                elif 'macds' in col_lower or 'signal' in col_lower:
                    df['macd_signal'] = macd[col]
                elif 'macd' in col_lower:
                    df['macd'] = macd[col]

        # Linear regression slope
        df['linreg_slope'] = ta.slope(close, length=20)

        return df

    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum features
        """
        high = self._get_col(df, 'high')
        low = self._get_col(df, 'low')
        close = self._get_col(df, 'close')

        # RSI
        df['rsi_14'] = ta.rsi(close, length=14)
        df['rsi_7'] = ta.rsi(close, length=7)

        # RSI zones
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)

        # Stochastic
        stoch = ta.stoch(high, low, close, k=self.stoch_k, d=self.stoch_d)
        if stoch is not None and not stoch.empty:
            stoch_cols = stoch.columns.tolist()
            for col in stoch_cols:
                col_lower = col.lower()
                if 'stochk' in col_lower or '_k' in col_lower:
                    df['stoch_k'] = stoch[col]
                elif 'stochd' in col_lower or '_d' in col_lower:
                    df['stoch_d'] = stoch[col]

        # Rate of Change
        df['roc_10'] = ta.roc(close, length=10)
        df['roc_20'] = ta.roc(close, length=20)

        # Momentum
        df['mom_10'] = ta.mom(close, length=10)

        # Williams %R
        df['willr'] = ta.willr(high, low, close, length=14)

        # CCI
        df['cci'] = ta.cci(high, low, close, length=20)

        return df

    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price pattern features
        """
        open_ = self._get_col(df, 'open')
        high = self._get_col(df, 'high')
        low = self._get_col(df, 'low')
        close = self._get_col(df, 'close')

        # Candle body and wick
        df['body_size'] = abs(close - open_)
        df['upper_wick'] = high - pd.concat([close, open_], axis=1).max(axis=1)
        df['lower_wick'] = pd.concat([close, open_], axis=1).min(axis=1) - low
        df['body_to_range'] = df['body_size'] / (high - low + 0.00001)

        # Bullish/Bearish candle
        df['is_bullish'] = (close > open_).astype(int)

        # Higher highs, lower lows
        df['higher_high'] = (high > high.shift(1)).astype(int)
        df['lower_low'] = (low < low.shift(1)).astype(int)

        # Consecutive patterns
        df['consecutive_bullish'] = df['is_bullish'].rolling(3).sum()
        df['consecutive_bearish'] = (1 - df['is_bullish']).rolling(3).sum()

        # Distance from recent high/low
        df['dist_from_20h_high'] = (high.rolling(20).max() - close) / close
        df['dist_from_20h_low'] = (close - low.rolling(20).min()) / close

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all technical features added
        """
        df = df.copy()

        print("  Adding return features...")
        df = self.add_returns(df)

        print("  Adding volatility features...")
        df = self.add_volatility(df)

        print("  Adding trend features...")
        df = self.add_trend(df)

        print("  Adding momentum features...")
        df = self.add_momentum(df)

        print("  Adding price pattern features...")
        df = self.add_price_patterns(df)

        # Drop rows with NaN from indicator calculation
        initial_len = len(df)
        df.dropna(inplace=True)
        print(f"  Dropped {initial_len - len(df)} rows with NaN values")

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names generated by this class
        """
        return [
            # Returns
            'returns_1h', 'returns_4h', 'returns_1d', 'returns_1w',
            'log_returns_1h', 'log_returns_1d', 'cum_returns_1d',

            # Volatility
            'atr_14', 'atr_20', 'atr_ratio', 'bb_width', 'bb_position',
            'hist_vol_20', 'hist_vol_50', 'vol_percentile', 'true_range', 'tr_ratio',

            # Trend
            'adx_14', 'plus_di', 'minus_di', 'di_diff', 'di_ratio',
            'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'ema_9_21_diff', 'ema_50_200_diff',
            'price_vs_ema_21', 'price_vs_ema_50',
            'macd', 'macd_signal', 'macd_hist', 'linreg_slope',

            # Momentum
            'rsi_14', 'rsi_7', 'rsi_oversold', 'rsi_overbought',
            'stoch_k', 'stoch_d', 'roc_10', 'roc_20', 'mom_10', 'willr', 'cci',

            # Price patterns
            'body_size', 'upper_wick', 'lower_wick', 'body_to_range',
            'is_bullish', 'higher_high', 'lower_low',
            'consecutive_bullish', 'consecutive_bearish',
            'dist_from_20h_high', 'dist_from_20h_low'
        ]


def compute_technical_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to compute all technical features

    Usage:
        from ml_trading_bot.features.technical import compute_technical_features
        df_with_features = compute_technical_features(ohlcv_df)
    """
    tech = TechnicalFeatures(**kwargs)
    return tech.add_all_features(df)


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from training.data_loader import DataLoader

    print("=" * 60)
    print("Testing Technical Features")
    print("=" * 60)

    # Load sample data
    loader = DataLoader()
    df = loader.load_ohlcv_sync("2024-01-01", "2024-03-31")

    print(f"\nInput shape: {df.shape}")

    # Compute features
    tech = TechnicalFeatures()
    df_features = tech.add_all_features(df)

    print(f"\nOutput shape: {df_features.shape}")
    print(f"\nNew features ({len(tech.get_feature_names())}):")
    for i, name in enumerate(tech.get_feature_names(), 1):
        if name in df_features.columns:
            print(f"  {i:2}. {name}")

    print("\n" + "=" * 60)
    print("Technical features test complete!")
    print("=" * 60)
