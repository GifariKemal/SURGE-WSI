#!/usr/bin/env python3
"""
Generate 2015-2019 Market Profiles
==================================

Generates monthly and daily profiles for 2015-2019 to complete the 10-year dataset.

Key events:
- 2015: Fed rate hike cycle begins
- 2016 June: BREXIT VOTE (GBP -10% in one day!)
- 2017: Article 50 triggered
- 2018: Trade war volatility
- 2019: Brexit uncertainty, extension after extension

Author: SURIOTA Team
"""

import sys
import io
from pathlib import Path

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


# Reuse dataclasses from historical script
@dataclass
class MonthProfile:
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


class ProfileAnalyzer:
    """Analyzer for generating market profiles"""

    def __init__(self):
        self.db: Optional[DBHandler] = None

    async def connect(self):
        self.db = DBHandler(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.user,
            password=config.database.password
        )
        return await self.db.connect()

    async def disconnect(self):
        if self.db:
            await self.db.disconnect()

    async def fetch_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        df = await self.db.get_ohlcv("GBPUSD", "H1", 100000, start, end)
        return df

    def _get_col(self, df: pd.DataFrame, col: str) -> str:
        if col in df.columns:
            return col
        if col.capitalize() in df.columns:
            return col.capitalize()
        if col.upper() in df.columns:
            return col.upper()
        return col

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        close_col = self._get_col(df, 'close')
        high_col = self._get_col(df, 'high')
        low_col = self._get_col(df, 'low')
        open_col = self._get_col(df, 'open')

        df = df.rename(columns={
            close_col: 'close',
            high_col: 'high',
            low_col: 'low',
            open_col: 'open'
        })

        close = df['close']
        high = df['high']
        low = df['low']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pips'] = df['atr'] * 10000

        atr_sum = tr.rolling(14).sum()
        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        price_range = highest_high - lowest_low
        price_range = price_range.replace(0, np.nan)
        df['choppiness'] = 100 * np.log10(atr_sum / price_range) / np.log10(14)

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
        df['bar_range_pips'] = (high - low) * 10000

        return df

    def analyze_month(self, df: pd.DataFrame, month: str) -> MonthProfile:
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

        profile.monthly_change_pips = (df['close'].iloc[-1] - df['close'].iloc[0]) * 10000

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
        profile.recommended_risk_mult = round(max(0.3, min(1.2, mult)), 2)
        profile.recommended_quality_threshold = 50.0

        notes = []
        if profile.volatility_regime == "LOW":
            notes.append("LOW VOLATILITY: Reduce position size")
        if profile.trend_regime == "RANGING":
            notes.append("RANGING MARKET: Prefer mean-reversion")
        elif profile.trend_regime == "TRENDING":
            notes.append("TRENDING: Momentum entries preferred")
        if profile.avg_choppiness > 65:
            notes.append("HIGH CHOPPINESS: Skip marginal setups")
        if profile.price_efficiency < 0.1:
            notes.append("LOW EFFICIENCY: Many false moves expected")
        if profile.best_hours:
            notes.append(f"Best hours: {profile.best_hours}")
        profile.trading_notes = "; ".join(notes) if notes else "Normal conditions"

        return profile

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
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        profile = DayProfile(date=date_str, day_of_week="")

        if df.empty or len(df) < 5:
            return profile

        first_bar = df.iloc[0]
        profile.day_of_week = dow_names[first_bar.name.dayofweek]

        profile.open_price = df['open'].iloc[0]
        profile.close_price = df['close'].iloc[-1]
        profile.high_price = df['high'].max()
        profile.low_price = df['low'].min()
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
                    day_range = (day_df['high'].max() - day_df['low'].min()) * 10000
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

        closes = df['close']
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
            profile.asian_range = (asian_df['high'].max() - asian_df['low'].min()) * 10000
        if london_mask.any():
            london_df = df[london_mask]
            profile.london_range = (london_df['high'].max() - london_df['low'].min()) * 10000
        if ny_mask.any():
            ny_df = df[ny_mask]
            profile.ny_range = (ny_df['high'].max() - ny_df['low'].min()) * 10000

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
        profile.quality_score = max(0, min(100, score))

        profile.tradeable = (
            profile.quality_score >= 40 and
            profile.adx >= 15 and
            profile.choppiness < 70 and
            profile.daily_range_pips >= 30
        )

        if profile.quality_score >= 75:
            profile.risk_multiplier = 1.3
        elif profile.quality_score >= 60:
            profile.risk_multiplier = 1.1
        elif profile.quality_score >= 50:
            profile.risk_multiplier = 1.0
        elif profile.quality_score >= 40:
            profile.risk_multiplier = 0.7
        else:
            profile.risk_multiplier = 0.4

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
        profile.notes = notes if notes else ["NORMAL"]

        return profile


async def generate_year_profiles(analyzer: ProfileAnalyzer, year: int, all_data: pd.DataFrame) -> tuple:
    print(f"\n  Processing {year}...")

    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    lookback_start = start - timedelta(days=30)
    year_data = all_data[(all_data.index >= lookback_start) & (all_data.index <= end)]

    if year_data.empty:
        print(f"    No data for {year}")
        return {}, {}

    monthly_profiles = {}
    year_only_data = year_data[year_data.index >= start]
    year_only_data = year_only_data.copy()
    year_only_data['month'] = year_only_data.index.to_period('M').astype(str)

    for month in year_only_data['month'].unique():
        month_df = year_only_data[year_only_data['month'] == month]
        profile = analyzer.analyze_month(month_df, month)
        monthly_profiles[month] = profile
        print(f"    {month}: {profile.volatility_regime} vol, {profile.trend_regime}, "
              f"ATR={profile.avg_atr_pips:.1f}")

    daily_profiles = {}
    current = start

    while current <= end:
        if current.weekday() < 5:
            day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            day_data = all_data[(all_data.index >= day_start) & (all_data.index < day_end)]

            lookback_end = day_start
            lookback_start_dt = lookback_end - timedelta(days=28)
            lookback_data = all_data[(all_data.index >= lookback_start_dt) & (all_data.index < lookback_end)]

            if len(day_data) >= 5:
                date_str = current.strftime("%Y-%m-%d")
                profile = analyzer.analyze_day(day_data, date_str, lookback_data)
                daily_profiles[date_str] = profile

        current += timedelta(days=1)

    print(f"    Generated {len(monthly_profiles)} monthly + {len(daily_profiles)} daily profiles")

    return monthly_profiles, daily_profiles


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("=" * 70)
    print("GENERATE 2015-2019 MARKET PROFILES")
    print("Completing 10-year dataset for AI/Bot training")
    print("=" * 70)

    analyzer = ProfileAnalyzer()
    if not await analyzer.connect():
        print("Failed to connect to database!")
        return

    try:
        print("\n[1/3] Fetching 2015-2019 H1 data...")
        start = datetime(2014, 12, 1, tzinfo=timezone.utc)  # Extra for lookback
        end = datetime(2019, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        all_data = await analyzer.fetch_data(start, end)
        if all_data.empty:
            print("No data found for 2015-2019!")
            return

        print(f"      Loaded {len(all_data)} bars")

        print("\n[2/3] Calculating indicators...")
        all_data = analyzer.calculate_indicators(all_data)

        print("\n[3/3] Generating profiles per year...")
        all_monthly = {}
        all_daily = {}

        for year in [2015, 2016, 2017, 2018, 2019]:
            monthly, daily = await generate_year_profiles(analyzer, year, all_data)
            all_monthly.update(monthly)
            all_daily.update(daily)

        # Load existing data (2020-2026)
        print("\n[4/4] Merging with existing 2020-2026 data...")
        data_dir = PROJECT_ROOT / "backtest" / "market_analysis"

        with open(data_dir / "monthly_profiles.json", 'r') as f:
            existing_monthly = json.load(f)

        with open(data_dir / "daily_profiles.json", 'r') as f:
            existing_daily = json.load(f)

        new_monthly = {}
        for month, profile in all_monthly.items():
            new_monthly[month] = {
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

        new_daily = {}
        for date, profile in all_daily.items():
            p_dict = asdict(profile)
            for key, value in p_dict.items():
                if isinstance(value, (np.bool_, np.integer, np.floating)):
                    p_dict[key] = value.item()
                elif isinstance(value, np.ndarray):
                    p_dict[key] = value.tolist()
            new_daily[date] = p_dict

        merged_monthly = {**new_monthly, **existing_monthly}
        merged_monthly = dict(sorted(merged_monthly.items()))

        merged_daily = {**new_daily, **existing_daily}
        merged_daily = dict(sorted(merged_daily.items()))

        print(f"\nSaving merged data...")
        print(f"  Monthly: {len(new_monthly)} (2015-2019) + {len(existing_monthly)} (2020-2026) = {len(merged_monthly)}")
        print(f"  Daily: {len(new_daily)} (2015-2019) + {len(existing_daily)} (2020-2026) = {len(merged_daily)}")

        with open(data_dir / "monthly_profiles.json", 'w') as f:
            json.dump(merged_monthly, f, indent=2)

        with open(data_dir / "daily_profiles.json", 'w') as f:
            json.dump(merged_daily, f, indent=2)

        print("\n" + "=" * 70)
        print("REGENERATING MARKDOWN REPORTS...")
        print("=" * 70)

        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_market_reports.py"), "--all"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                print(line)
        else:
            print("Error:", result.stderr)

    finally:
        await analyzer.disconnect()

    print("\n" + "=" * 70)
    print("10-YEAR DATASET COMPLETE!")
    print("=" * 70)
    print("\nData now covers: January 2015 - January 2026 (11 years)")
    print("Reports available at: reports/market_analysis/")
    print("\nKey events captured:")
    print("  - 2015: Fed rate hike cycle")
    print("  - 2016 Jun: BREXIT VOTE")
    print("  - 2017: Article 50")
    print("  - 2018: Trade war")
    print("  - 2019: Brexit uncertainty")
    print("  - 2020 Mar: COVID CRASH")
    print("  - 2022 Sep: UK Mini-Budget Crisis")


if __name__ == "__main__":
    asyncio.run(main())
