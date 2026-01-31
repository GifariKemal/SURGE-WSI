#!/usr/bin/env python3
"""
Generate 2024 Market Profiles

Fetches GBPUSD H1 data for 2024 from database and generates:
- Monthly profiles (Jan 2024 - Dec 2024)
- Daily profiles (all trading days in 2024)

Then merges with existing 2025-2026 data and regenerates all reports.
"""

import sys
import io
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from loguru import logger

from config import config
from src.data.db_handler import DBHandler


# =====================================================================
# DATA CLASSES (copied from analyzers to avoid import issues)
# =====================================================================

@dataclass
class MonthProfile:
    """Market profile for a specific month"""
    month: str

    avg_atr_pips: float = 0.0
    atr_percentile: float = 0.0
    avg_daily_range_pips: float = 0.0
    range_percentile: float = 0.0
    volatility_regime: str = ""

    avg_adx: float = 0.0
    adx_percentile: float = 0.0
    trend_days_pct: float = 0.0
    dominant_direction: str = ""
    trend_regime: str = ""

    avg_choppiness: float = 0.0
    chop_percentile: float = 0.0
    ranging_days_pct: float = 0.0

    monthly_change_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    price_efficiency: float = 0.0

    avg_hourly_range_pips: float = 0.0
    best_hours: List[int] = field(default_factory=list)
    worst_hours: List[int] = field(default_factory=list)

    best_days: List[str] = field(default_factory=list)
    worst_days: List[str] = field(default_factory=list)

    recommended_risk_mult: float = 1.0
    recommended_quality_threshold: float = 50.0
    trading_notes: str = ""


@dataclass
class DayProfile:
    """Comprehensive profile for a single trading day"""
    date: str
    day_of_week: str

    atr_pips: float = 0.0
    daily_range_pips: float = 0.0
    volatility_percentile: float = 50.0
    is_low_volatility: bool = False
    is_high_volatility: bool = False

    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    trend_direction: str = "NEUTRAL"
    trend_strength: str = "WEAK"
    is_trending: bool = False

    choppiness: float = 50.0
    is_ranging: bool = False
    is_choppy: bool = False

    open_price: float = 0.0
    close_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    price_change_pips: float = 0.0

    num_reversals: int = 0
    price_efficiency: float = 0.0
    max_intraday_dd_pips: float = 0.0

    asian_range: float = 0.0
    london_range: float = 0.0
    ny_range: float = 0.0
    best_session: str = ""

    best_hour: int = -1
    worst_hour: int = -1
    best_hour_range: float = 0.0
    worst_hour_range: float = 0.0

    quality_score: float = 50.0
    tradeable: bool = True
    risk_multiplier: float = 1.0
    notes: List[str] = field(default_factory=list)


# =====================================================================
# ANALYZER CLASS
# =====================================================================

class MarketAnalyzer:
    """Combined market condition analyzer"""

    def __init__(self):
        self.monthly_profiles: Dict[str, MonthProfile] = {}
        self.daily_profiles: Dict[str, DayProfile] = {}

    async def fetch_data(self, symbol: str, timeframe: str,
                         start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data"""
        db = DBHandler(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.user,
            password=config.database.password
        )
        if not await db.connect():
            return pd.DataFrame()
        df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
        await db.disconnect()
        return df

    def _get_col_map(self, df: pd.DataFrame) -> dict:
        return {
            'open': 'open' if 'open' in df.columns else 'Open',
            'high': 'high' if 'high' in df.columns else 'High',
            'low': 'low' if 'low' in df.columns else 'Low',
            'close': 'close' if 'close' in df.columns else 'Close',
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        col_map = self._get_col_map(df)
        close = df[col_map['close']]
        high = df[col_map['high']]
        low = df[col_map['low']]

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pips'] = df['atr'] * 10000

        # Choppiness Index
        atr_sum = tr.rolling(14).sum()
        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        price_range = highest_high - lowest_low
        price_range = price_range.replace(0, np.nan)
        df['choppiness'] = 100 * np.log10(atr_sum / price_range) / np.log10(14)

        # ADX and DI
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Bar range
        df['bar_range_pips'] = (high - low) * 10000

        return df

    def analyze_month(self, df: pd.DataFrame, month: str) -> MonthProfile:
        """Analyze a specific month"""
        col_map = self._get_col_map(df)
        profile = MonthProfile(month=month)

        if df.empty:
            return profile

        profile.avg_atr_pips = df['atr_pips'].mean()
        profile.avg_daily_range_pips = df['bar_range_pips'].mean()

        profile.avg_adx = df['adx'].mean()
        profile.trend_days_pct = (df['adx'] > 25).mean() * 100

        plus_di_avg = df['plus_di'].mean()
        minus_di_avg = df['minus_di'].mean()
        if plus_di_avg > minus_di_avg + 5:
            profile.dominant_direction = "BULLISH"
        elif minus_di_avg > plus_di_avg + 5:
            profile.dominant_direction = "BEARISH"
        else:
            profile.dominant_direction = "MIXED"

        profile.avg_choppiness = df['choppiness'].mean()
        profile.ranging_days_pct = (df['choppiness'] > 61.8).mean() * 100

        if profile.avg_atr_pips > 12:
            profile.volatility_regime = "HIGH"
        elif profile.avg_atr_pips > 8:
            profile.volatility_regime = "NORMAL"
        else:
            profile.volatility_regime = "LOW"

        if profile.avg_adx > 25 and profile.ranging_days_pct < 40:
            profile.trend_regime = "TRENDING"
        elif profile.avg_adx < 20 or profile.ranging_days_pct > 60:
            profile.trend_regime = "RANGING"
        else:
            profile.trend_regime = "MIXED"

        profile.monthly_change_pips = (df[col_map['close']].iloc[-1] - df[col_map['close']].iloc[0]) * 10000

        total_movement = df['bar_range_pips'].sum()
        net_movement = abs(profile.monthly_change_pips)
        profile.price_efficiency = net_movement / total_movement if total_movement > 0 else 0

        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        hourly_ranges = df_copy.groupby('hour')['bar_range_pips'].mean()

        profile.avg_hourly_range_pips = hourly_ranges.mean()
        profile.best_hours = hourly_ranges.nlargest(3).index.tolist()
        profile.worst_hours = hourly_ranges.nsmallest(3).index.tolist()

        df_copy['dow'] = df_copy.index.dayofweek
        dow_ranges = df_copy.groupby('dow')['bar_range_pips'].mean()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        best_dow_idx = dow_ranges.nlargest(2).index.tolist()
        worst_dow_idx = dow_ranges.nsmallest(2).index.tolist()
        profile.best_days = [dow_names[i] for i in best_dow_idx if i < len(dow_names)]
        profile.worst_days = [dow_names[i] for i in worst_dow_idx if i < len(dow_names)]

        profile.recommended_risk_mult = self._calculate_risk_mult(profile)
        profile.recommended_quality_threshold = 50.0
        profile.trading_notes = self._generate_notes(profile)

        return profile

    def _calculate_risk_mult(self, profile: MonthProfile) -> float:
        mult = 1.0
        if profile.volatility_regime == "LOW":
            mult *= 0.7
        elif profile.volatility_regime == "HIGH":
            mult *= 0.9
        if profile.trend_regime == "RANGING":
            mult *= 0.6
        elif profile.trend_regime == "TRENDING":
            mult *= 1.1
        if profile.avg_choppiness > 65:
            mult *= 0.7
        elif profile.avg_choppiness < 50:
            mult *= 1.1
        if profile.price_efficiency < 0.1:
            mult *= 0.6
        elif profile.price_efficiency > 0.3:
            mult *= 1.1
        return round(max(0.3, min(1.2, mult)), 2)

    def _generate_notes(self, profile: MonthProfile) -> str:
        notes = []
        if profile.volatility_regime == "LOW":
            notes.append("LOW VOLATILITY: Reduce position size, tighter SL")
        if profile.trend_regime == "RANGING":
            notes.append("RANGING MARKET: Prefer mean-reversion, avoid breakouts")
        elif profile.trend_regime == "TRENDING":
            notes.append("TRENDING: Momentum entries preferred")
        if profile.avg_choppiness > 65:
            notes.append("HIGH CHOPPINESS: Skip marginal setups")
        if profile.price_efficiency < 0.1:
            notes.append("LOW EFFICIENCY: Many false moves expected")
        if profile.best_hours:
            notes.append(f"Best hours: {profile.best_hours}")
        return "; ".join(notes) if notes else "Normal conditions"

    def count_reversals(self, closes: pd.Series) -> int:
        if len(closes) < 3:
            return 0
        directions = np.sign(closes.diff().dropna())
        directions = directions[directions != 0]
        if len(directions) < 2:
            return 0
        return int(np.sum(np.diff(directions) != 0))

    def analyze_day(self, df: pd.DataFrame, date_str: str,
                    lookback_data: Optional[pd.DataFrame] = None) -> DayProfile:
        """Analyze a single trading day"""
        col_map = self._get_col_map(df)
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        profile = DayProfile(date=date_str, day_of_week="")

        if df.empty or len(df) < 5:
            return profile

        first_bar = df.iloc[0]
        profile.day_of_week = dow_names[first_bar.name.dayofweek]

        profile.open_price = df[col_map['open']].iloc[0]
        profile.close_price = df[col_map['close']].iloc[-1]
        profile.high_price = df[col_map['high']].max()
        profile.low_price = df[col_map['low']].min()
        profile.daily_range_pips = (profile.high_price - profile.low_price) * 10000
        profile.price_change_pips = (profile.close_price - profile.open_price) * 10000

        profile.atr_pips = df['atr_pips'].mean()
        profile.is_low_volatility = profile.atr_pips < 10
        profile.is_high_volatility = profile.atr_pips > 18

        if lookback_data is not None and len(lookback_data) > 0:
            lookback_by_day = lookback_data.groupby(lookback_data.index.date)
            daily_ranges = []
            for _, day_df in lookback_by_day:
                if len(day_df) > 0:
                    day_range = (day_df[col_map['high']].max() - day_df[col_map['low']].min()) * 10000
                    daily_ranges.append(day_range)
            if daily_ranges:
                profile.volatility_percentile = (
                    np.sum(np.array(daily_ranges) < profile.daily_range_pips) / len(daily_ranges)
                ) * 100

        profile.adx = df['adx'].mean()
        profile.plus_di = df['plus_di'].mean()
        profile.minus_di = df['minus_di'].mean()
        profile.is_trending = profile.adx > 25

        if profile.plus_di > profile.minus_di + 5:
            profile.trend_direction = "BULLISH"
        elif profile.minus_di > profile.plus_di + 5:
            profile.trend_direction = "BEARISH"
        else:
            profile.trend_direction = "NEUTRAL"

        if profile.adx > 35:
            profile.trend_strength = "STRONG"
        elif profile.adx > 25:
            profile.trend_strength = "MODERATE"
        else:
            profile.trend_strength = "WEAK"

        profile.choppiness = df['choppiness'].mean()
        profile.is_ranging = profile.choppiness > 61.8
        profile.is_choppy = profile.choppiness > 55

        closes = df[col_map['close']]
        profile.num_reversals = self.count_reversals(closes)

        total_movement = df['bar_range_pips'].sum()
        net_movement = abs(profile.price_change_pips)
        profile.price_efficiency = net_movement / total_movement if total_movement > 0 else 0

        cum_returns = closes.pct_change().cumsum()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns - rolling_max
        profile.max_intraday_dd_pips = abs(drawdowns.min()) * closes.iloc[0] * 10000 if len(drawdowns) > 0 else 0

        asian_mask = (df.index.hour >= 0) & (df.index.hour < 7)
        london_mask = (df.index.hour >= 7) & (df.index.hour < 12)
        ny_mask = (df.index.hour >= 12) & (df.index.hour < 21)

        if asian_mask.any():
            asian_df = df[asian_mask]
            profile.asian_range = (asian_df[col_map['high']].max() - asian_df[col_map['low']].min()) * 10000
        if london_mask.any():
            london_df = df[london_mask]
            profile.london_range = (london_df[col_map['high']].max() - london_df[col_map['low']].min()) * 10000
        if ny_mask.any():
            ny_df = df[ny_mask]
            profile.ny_range = (ny_df[col_map['high']].max() - ny_df[col_map['low']].min()) * 10000

        sessions = {'Asian': profile.asian_range, 'London': profile.london_range, 'NY': profile.ny_range}
        profile.best_session = max(sessions, key=sessions.get)

        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        hourly_ranges = df_copy.groupby('hour')['bar_range_pips'].mean()

        if not hourly_ranges.empty:
            profile.best_hour = int(hourly_ranges.idxmax())
            profile.worst_hour = int(hourly_ranges.idxmin())
            profile.best_hour_range = hourly_ranges.max()
            profile.worst_hour_range = hourly_ranges.min()

        profile.quality_score = self._calculate_day_quality(profile)
        profile.tradeable = self._is_tradeable(profile)
        profile.risk_multiplier = self._calculate_day_risk(profile)
        profile.notes = self._generate_day_notes(profile)

        return profile

    def _calculate_day_quality(self, profile: DayProfile) -> float:
        score = 50.0
        if profile.volatility_percentile > 70:
            score += 15
        elif profile.volatility_percentile < 30:
            score -= 10
        if profile.is_high_volatility:
            score += 5
        elif profile.is_low_volatility:
            score -= 10
        if profile.is_trending:
            score += 15
        elif profile.adx < 18:
            score -= 15
        if profile.trend_strength == "STRONG":
            score += 10
        if profile.is_ranging:
            score -= 20
        elif profile.choppiness < 45:
            score += 10
        if profile.price_efficiency > 0.4:
            score += 10
        elif profile.price_efficiency < 0.15:
            score -= 15
        if profile.num_reversals > 10:
            score -= 15
        elif profile.num_reversals > 7:
            score -= 10
        elif profile.num_reversals < 4:
            score += 5
        if profile.day_of_week == "Friday":
            score -= 10
        elif profile.day_of_week in ["Tuesday", "Wednesday"]:
            score += 5
        return max(0, min(100, score))

    def _is_tradeable(self, profile: DayProfile) -> bool:
        return (
            profile.quality_score >= 40 and
            profile.adx >= 15 and
            profile.choppiness < 70 and
            profile.daily_range_pips >= 30
        )

    def _calculate_day_risk(self, profile: DayProfile) -> float:
        if profile.quality_score >= 75:
            return 1.3
        elif profile.quality_score >= 60:
            return 1.1
        elif profile.quality_score >= 50:
            return 1.0
        elif profile.quality_score >= 40:
            return 0.7
        else:
            return 0.4

    def _generate_day_notes(self, profile: DayProfile) -> List[str]:
        notes = []
        if profile.is_low_volatility:
            notes.append("LOW_VOLATILITY")
        elif profile.is_high_volatility:
            notes.append("HIGH_VOLATILITY")
        if profile.is_ranging:
            notes.append("RANGING_MARKET")
        elif profile.is_choppy:
            notes.append("CHOPPY")
        if profile.trend_strength == "STRONG":
            notes.append(f"STRONG_{profile.trend_direction}")
        if profile.num_reversals > 10:
            notes.append("MANY_REVERSALS")
        if profile.price_efficiency < 0.15:
            notes.append("LOW_EFFICIENCY")
        elif profile.price_efficiency > 0.4:
            notes.append("CLEAN_TREND")
        if not profile.tradeable:
            notes.append("SKIP_RECOMMENDED")
        return notes if notes else ["NORMAL"]


async def generate_2024_profiles():
    """Generate both monthly and daily profiles for 2024"""
    print("\n" + "=" * 60)
    print("GENERATING 2024 MARKET PROFILES")
    print("=" * 60)

    analyzer = MarketAnalyzer()

    # Fetch H1 data for 2024 (with extra lookback)
    print("\n[1/4] Fetching H1 data for 2024...")
    lookback_start = datetime(2023, 12, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    df = await analyzer.fetch_data("GBPUSD", "H1", lookback_start, end_date)

    if df.empty:
        print("ERROR: No 2024 data found in database!")
        return {}, {}

    print(f"      Loaded {len(df)} bars")

    # Calculate indicators
    print("\n[2/4] Calculating indicators...")
    df = analyzer.calculate_indicators(df)

    # Filter to 2024 only for monthly analysis
    df_2024 = df[df.index >= datetime(2024, 1, 1, tzinfo=timezone.utc)]

    # Analyze each month
    print("\n[3/4] Analyzing monthly conditions...")
    monthly_profiles = {}

    df_2024_copy = df_2024.copy()
    df_2024_copy['month'] = df_2024_copy.index.to_period('M').astype(str)
    months = df_2024_copy['month'].unique()

    for month in months:
        month_df = df_2024_copy[df_2024_copy['month'] == month]
        profile = analyzer.analyze_month(month_df, month)
        monthly_profiles[month] = profile
        print(f"      {month}: {profile.volatility_regime} vol, {profile.trend_regime}, "
              f"ATR={profile.avg_atr_pips:.1f}, risk={profile.recommended_risk_mult}x")

    # Analyze daily profiles
    print("\n[4/4] Analyzing daily conditions...")
    daily_profiles = {}

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    current = start

    while current <= end:
        if current.weekday() < 5:  # Skip weekends
            day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            day_data = df[(df.index >= day_start) & (df.index < day_end)]

            # Lookback data
            lookback_end = day_start
            lookback_start_dt = lookback_end - timedelta(days=28)
            lookback_data = df[(df.index >= lookback_start_dt) & (df.index < lookback_end)]

            if len(day_data) >= 5:
                date_str = current.strftime("%Y-%m-%d")
                profile = analyzer.analyze_day(day_data, date_str, lookback_data)
                daily_profiles[date_str] = profile

        current += timedelta(days=1)

    print(f"\n      Generated {len(monthly_profiles)} monthly + {len(daily_profiles)} daily profiles")

    return monthly_profiles, daily_profiles


def merge_and_save_profiles(monthly_2024: dict, daily_2024: dict):
    """Merge 2024 profiles with existing 2025-2026 data"""
    print("\n" + "=" * 60)
    print("MERGING WITH EXISTING DATA")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "backtest" / "market_analysis"

    # Load existing monthly profiles
    monthly_file = data_dir / "monthly_profiles.json"
    with open(monthly_file, 'r') as f:
        existing_monthly = json.load(f)

    # Load existing daily profiles
    daily_file = data_dir / "daily_profiles.json"
    with open(daily_file, 'r') as f:
        existing_daily = json.load(f)

    # Convert 2024 monthly profiles to dict format
    monthly_2024_dict = {}
    for month, profile in monthly_2024.items():
        monthly_2024_dict[month] = {
            'volatility_regime': profile.volatility_regime,
            'trend_regime': profile.trend_regime,
            'avg_atr_pips': profile.avg_atr_pips,
            'avg_adx': profile.avg_adx,
            'avg_choppiness': profile.avg_choppiness,
            'price_efficiency': profile.price_efficiency,
            'recommended_risk_mult': profile.recommended_risk_mult,
            'recommended_quality_threshold': profile.recommended_quality_threshold,
            'best_hours': profile.best_hours,
            'worst_hours': profile.worst_hours,
            'best_days': profile.best_days,
            'worst_days': profile.worst_days,
            'trading_notes': profile.trading_notes,
        }

    # Merge monthly (2024 + existing)
    merged_monthly = {**monthly_2024_dict, **existing_monthly}
    merged_monthly = dict(sorted(merged_monthly.items()))

    # Convert daily profiles to serializable format
    daily_2024_dict = {}
    for date, profile in daily_2024.items():
        p_dict = asdict(profile)
        # Convert numpy types
        for key, value in p_dict.items():
            if isinstance(value, (np.bool_, np.integer, np.floating)):
                p_dict[key] = value.item()
            elif isinstance(value, np.ndarray):
                p_dict[key] = value.tolist()
        daily_2024_dict[date] = p_dict

    # Merge daily (2024 + existing)
    merged_daily = {**daily_2024_dict, **existing_daily}
    merged_daily = dict(sorted(merged_daily.items()))

    # Save merged files
    print(f"\nSaving merged monthly profiles ({len(merged_monthly)} months)...")
    with open(monthly_file, 'w') as f:
        json.dump(merged_monthly, f, indent=2)

    print(f"Saving merged daily profiles ({len(merged_daily)} days)...")
    with open(daily_file, 'w') as f:
        json.dump(merged_daily, f, indent=2)

    print("\nMerge complete!")
    print(f"  Monthly: {len(monthly_2024_dict)} (2024) + {len(existing_monthly)} (2025-26) = {len(merged_monthly)}")
    print(f"  Daily:   {len(daily_2024_dict)} (2024) + {len(existing_daily)} (2025-26) = {len(merged_daily)}")

    return merged_monthly, merged_daily


def regenerate_reports():
    """Regenerate all markdown reports"""
    print("\n" + "=" * 60)
    print("REGENERATING MARKDOWN REPORTS")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_market_reports.py"), "--all"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )

    if result.returncode == 0:
        print(result.stdout)
    else:
        print("Error generating reports:")
        print(result.stderr)


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("=" * 60)
    print("2024 MARKET PROFILE GENERATOR")
    print("=" * 60)

    # Generate 2024 profiles (both monthly and daily)
    monthly_2024, daily_2024 = await generate_2024_profiles()

    if not monthly_2024 or not daily_2024:
        print("\nERROR: Failed to generate 2024 profiles!")
        print("Make sure the database contains GBPUSD H1 data for 2024.")
        return

    # Merge with existing data
    merge_and_save_profiles(monthly_2024, daily_2024)

    # Regenerate all reports
    regenerate_reports()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print("\nNew reports available at: reports/market_analysis/")
    print("Period now covers: January 2024 - January 2026 (25 months)")


if __name__ == "__main__":
    asyncio.run(main())
