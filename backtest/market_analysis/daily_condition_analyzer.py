"""
Daily Condition Analyzer
========================
Analyzes market conditions per day to understand intraday patterns
and identify problematic trading days.

This is the per-DAY counterpart to monthly_condition_analyzer.py.

Key insights we want to uncover:
1. Why did specific days in June 2025 lose money?
2. What market conditions (ADX, choppiness, volatility) characterize bad days?
3. Which hours within bad days were the problem?
4. Can we detect "bad day" conditions in real-time?

Created by: Claude AI for SURGE-WSI
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json
from loguru import logger

from config import config
from src.data.db_handler import DBHandler


@dataclass
class DayProfile:
    """Comprehensive profile for a single trading day"""
    date: str
    day_of_week: str

    # Volatility metrics
    atr_pips: float = 0.0
    daily_range_pips: float = 0.0
    volatility_percentile: float = 50.0  # vs last 20 days
    is_low_volatility: bool = False
    is_high_volatility: bool = False

    # Trend metrics
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    trend_direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    trend_strength: str = "WEAK"      # STRONG, MODERATE, WEAK
    is_trending: bool = False

    # Choppiness metrics
    choppiness: float = 50.0
    is_ranging: bool = False
    is_choppy: bool = False

    # Price action
    open_price: float = 0.0
    close_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    price_change_pips: float = 0.0

    # Intraday analysis
    num_reversals: int = 0
    price_efficiency: float = 0.0  # net_move / total_movement
    max_intraday_dd_pips: float = 0.0

    # Session analysis (ranges in pips)
    asian_range: float = 0.0
    london_range: float = 0.0
    ny_range: float = 0.0
    best_session: str = ""

    # Hour analysis
    best_hour: int = -1
    worst_hour: int = -1
    best_hour_range: float = 0.0
    worst_hour_range: float = 0.0

    # Quality assessment
    quality_score: float = 50.0
    tradeable: bool = True
    risk_multiplier: float = 1.0
    notes: List[str] = field(default_factory=list)


class DailyConditionAnalyzer:
    """Analyzes market conditions on a daily basis"""

    def __init__(self):
        self.h1_data: Optional[pd.DataFrame] = None
        self.daily_profiles: Dict[str, DayProfile] = {}

    async def fetch_data(self, symbol: str, timeframe: str,
                         start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from database"""
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

    def count_reversals(self, closes: pd.Series) -> int:
        """Count price direction reversals"""
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

        # Basic info
        first_bar = df.iloc[0]
        profile.day_of_week = dow_names[first_bar.name.dayofweek]

        # Price action
        profile.open_price = df[col_map['open']].iloc[0]
        profile.close_price = df[col_map['close']].iloc[-1]
        profile.high_price = df[col_map['high']].max()
        profile.low_price = df[col_map['low']].min()
        profile.daily_range_pips = (profile.high_price - profile.low_price) * 10000
        profile.price_change_pips = (profile.close_price - profile.open_price) * 10000

        # Volatility
        profile.atr_pips = df['atr_pips'].mean()
        profile.is_low_volatility = profile.atr_pips < 10
        profile.is_high_volatility = profile.atr_pips > 18

        # Calculate volatility percentile vs lookback
        if lookback_data is not None and len(lookback_data) > 0:
            # Group lookback by day and calculate daily ranges
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

        # Trend
        profile.adx = df['adx'].mean()
        profile.plus_di = df['plus_di'].mean()
        profile.minus_di = df['minus_di'].mean()
        profile.is_trending = profile.adx > 25

        # Trend direction
        if profile.plus_di > profile.minus_di + 5:
            profile.trend_direction = "BULLISH"
        elif profile.minus_di > profile.plus_di + 5:
            profile.trend_direction = "BEARISH"
        else:
            profile.trend_direction = "NEUTRAL"

        # Trend strength
        if profile.adx > 35:
            profile.trend_strength = "STRONG"
        elif profile.adx > 25:
            profile.trend_strength = "MODERATE"
        else:
            profile.trend_strength = "WEAK"

        # Choppiness
        profile.choppiness = df['choppiness'].mean()
        profile.is_ranging = profile.choppiness > 61.8
        profile.is_choppy = profile.choppiness > 55

        # Intraday analysis
        closes = df[col_map['close']]
        profile.num_reversals = self.count_reversals(closes)

        # Price efficiency
        total_movement = df['bar_range_pips'].sum()
        net_movement = abs(profile.price_change_pips)
        profile.price_efficiency = net_movement / total_movement if total_movement > 0 else 0

        # Max intraday drawdown
        cum_returns = closes.pct_change().cumsum()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns - rolling_max
        profile.max_intraday_dd_pips = abs(drawdowns.min()) * closes.iloc[0] * 10000 if len(drawdowns) > 0 else 0

        # Session analysis (UTC times)
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

        # Best session
        sessions = {'Asian': profile.asian_range, 'London': profile.london_range, 'NY': profile.ny_range}
        profile.best_session = max(sessions, key=sessions.get)

        # Hour analysis
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        hourly_ranges = df_copy.groupby('hour')['bar_range_pips'].mean()

        if not hourly_ranges.empty:
            profile.best_hour = int(hourly_ranges.idxmax())
            profile.worst_hour = int(hourly_ranges.idxmin())
            profile.best_hour_range = hourly_ranges.max()
            profile.worst_hour_range = hourly_ranges.min()

        # Quality assessment
        profile.quality_score = self._calculate_quality_score(profile)
        profile.tradeable = self._is_tradeable(profile)
        profile.risk_multiplier = self._calculate_risk_mult(profile)
        profile.notes = self._generate_notes(profile)

        return profile

    def _calculate_quality_score(self, profile: DayProfile) -> float:
        """Calculate overall quality score (0-100)"""
        score = 50.0

        # Volatility component
        if profile.volatility_percentile > 70:
            score += 15  # High vol day - good for trends
        elif profile.volatility_percentile < 30:
            score -= 10  # Low vol - limited opportunity

        if profile.is_high_volatility:
            score += 5
        elif profile.is_low_volatility:
            score -= 10

        # Trend component
        if profile.is_trending:
            score += 15
        elif profile.adx < 18:
            score -= 15

        if profile.trend_strength == "STRONG":
            score += 10

        # Choppiness component
        if profile.is_ranging:
            score -= 20
        elif profile.choppiness < 45:
            score += 10

        # Efficiency component
        if profile.price_efficiency > 0.4:
            score += 10  # Clean trend day
        elif profile.price_efficiency < 0.15:
            score -= 15  # Too much noise

        # Reversal penalty
        if profile.num_reversals > 10:
            score -= 15
        elif profile.num_reversals > 7:
            score -= 10
        elif profile.num_reversals < 4:
            score += 5

        # Day of week adjustment
        if profile.day_of_week == "Friday":
            score -= 10
        elif profile.day_of_week in ["Tuesday", "Wednesday"]:
            score += 5

        return max(0, min(100, score))

    def _is_tradeable(self, profile: DayProfile) -> bool:
        """Determine if day is tradeable"""
        return (
            profile.quality_score >= 40 and
            profile.adx >= 15 and
            profile.choppiness < 70 and
            profile.daily_range_pips >= 30
        )

    def _calculate_risk_mult(self, profile: DayProfile) -> float:
        """Calculate recommended risk multiplier"""
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

    def _generate_notes(self, profile: DayProfile) -> List[str]:
        """Generate trading notes for the day"""
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

    async def analyze_period(self, symbol: str, start: datetime, end: datetime) -> Dict[str, DayProfile]:
        """Analyze all trading days in a period"""
        print(f"\nFetching {symbol} H1 data from {start.date()} to {end.date()}...")

        # Fetch data with extra lookback for indicators
        lookback_start = start - timedelta(days=30)
        self.h1_data = await self.fetch_data(symbol, "H1", lookback_start, end)

        if self.h1_data.empty:
            print("ERROR: No data retrieved!")
            return {}

        print(f"Loaded {len(self.h1_data)} H1 bars")

        # Calculate indicators
        print("Calculating indicators...")
        self.h1_data = self.calculate_indicators(self.h1_data)

        # Analyze each day
        print("Analyzing daily profiles...")
        profiles = {}

        current = start
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1)

                # Get day's data
                day_data = self.h1_data[(self.h1_data.index >= day_start) &
                                         (self.h1_data.index < day_end)]

                # Get lookback data (last 20 trading days)
                lookback_end = day_start
                lookback_start = lookback_end - timedelta(days=28)  # ~20 trading days
                lookback_data = self.h1_data[(self.h1_data.index >= lookback_start) &
                                              (self.h1_data.index < lookback_end)]

                if len(day_data) >= 5:  # Need at least 5 hours of data
                    date_str = current.strftime("%Y-%m-%d")
                    profile = self.analyze_day(day_data, date_str, lookback_data)
                    profiles[date_str] = profile

            current += timedelta(days=1)

        self.daily_profiles = profiles
        return profiles

    def get_month_summary(self, month_prefix: str) -> Dict:
        """Get summary statistics for a specific month"""
        month_profiles = [p for d, p in self.daily_profiles.items() if d.startswith(month_prefix)]

        if not month_profiles:
            return {}

        tradeable = [p for p in month_profiles if p.tradeable]
        non_tradeable = [p for p in month_profiles if not p.tradeable]

        return {
            'total_days': len(month_profiles),
            'tradeable_days': len(tradeable),
            'non_tradeable_days': len(non_tradeable),
            'tradeable_pct': round(len(tradeable) / len(month_profiles) * 100, 1),
            'avg_quality': round(np.mean([p.quality_score for p in month_profiles]), 1),
            'avg_atr': round(np.mean([p.atr_pips for p in month_profiles]), 1),
            'avg_adx': round(np.mean([p.adx for p in month_profiles]), 1),
            'avg_choppiness': round(np.mean([p.choppiness for p in month_profiles]), 1),
            'avg_efficiency': round(np.mean([p.price_efficiency for p in month_profiles]), 3),
            'avg_reversals': round(np.mean([p.num_reversals for p in month_profiles]), 1),
            'best_day': max(month_profiles, key=lambda p: p.quality_score).date if month_profiles else "",
            'worst_day': min(month_profiles, key=lambda p: p.quality_score).date if month_profiles else "",
        }

    def export_to_json(self, filename: str):
        """Export profiles to JSON"""
        output = {}
        for date, profile in self.daily_profiles.items():
            p_dict = asdict(profile)
            # Convert numpy types to native Python types
            for key, value in p_dict.items():
                if isinstance(value, (np.bool_, np.integer, np.floating)):
                    p_dict[key] = value.item()
                elif isinstance(value, np.ndarray):
                    p_dict[key] = value.tolist()
            output[date] = p_dict

        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Exported to {output_path}")
        return output_path


async def main():
    """Main analysis function"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("=" * 70)
    print("DAILY CONDITION ANALYZER")
    print("Per-day market condition analysis for SURGE-WSI")
    print("=" * 70)

    analyzer = DailyConditionAnalyzer()

    # Analyze Jan 2025 to Jan 2026
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    profiles = await analyzer.analyze_period("GBPUSD", start, end)

    if not profiles:
        print("No profiles generated!")
        return

    print(f"\nAnalyzed {len(profiles)} trading days")

    # Monthly summaries
    print("\n" + "=" * 70)
    print("MONTHLY SUMMARIES")
    print("=" * 70)

    months = [
        "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
        "2026-01"
    ]

    for month in months:
        summary = analyzer.get_month_summary(month)
        if summary:
            print(f"\n{month}:")
            print(f"  Days: {summary['total_days']} ({summary['tradeable_days']} tradeable, {summary['tradeable_pct']}%)")
            print(f"  Avg Quality: {summary['avg_quality']}, Avg ADX: {summary['avg_adx']:.1f}, "
                  f"Avg Chop: {summary['avg_choppiness']:.1f}")
            print(f"  Avg Efficiency: {summary['avg_efficiency']:.3f}, Avg Reversals: {summary['avg_reversals']:.1f}")
            print(f"  Best Day: {summary['best_day']}, Worst Day: {summary['worst_day']}")

    # Special focus on June 2025 (our losing month)
    print("\n" + "=" * 70)
    print("SPECIAL FOCUS: JUNE 2025 (LOSING MONTH)")
    print("=" * 70)

    june_profiles = [(d, p) for d, p in sorted(profiles.items()) if d.startswith("2025-06")]

    print("\nAll June 2025 trading days:")
    print(f"{'Date':12} {'DoW':10} {'Quality':8} {'ADX':6} {'Chop':6} {'Eff':6} {'Rev':4} {'Tradeable':10} Notes")
    print("-" * 85)

    for date, profile in june_profiles:
        notes_str = ", ".join(profile.notes[:2]) if profile.notes else ""
        tradeable_str = "YES" if profile.tradeable else "SKIP"
        print(f"{date:12} {profile.day_of_week[:8]:10} {profile.quality_score:7.1f} "
              f"{profile.adx:5.1f} {profile.choppiness:5.1f} {profile.price_efficiency:5.3f} "
              f"{profile.num_reversals:3d} {tradeable_str:10} {notes_str}")

    # Identify problematic days in June
    june_skip_days = [p for d, p in june_profiles if not p.tradeable]
    print(f"\nJune days recommended to SKIP: {len(june_skip_days)}")
    for p in june_skip_days:
        print(f"  {p.date} ({p.day_of_week[:3]}): Score={p.quality_score:.0f}, {', '.join(p.notes)}")

    # Day of week analysis
    print("\n" + "=" * 70)
    print("DAY OF WEEK ANALYSIS (ALL DATA)")
    print("=" * 70)

    dow_stats = {}
    for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        dow_profiles = [p for p in profiles.values() if p.day_of_week == dow]
        if dow_profiles:
            dow_stats[dow] = {
                'count': len(dow_profiles),
                'avg_quality': np.mean([p.quality_score for p in dow_profiles]),
                'tradeable_pct': len([p for p in dow_profiles if p.tradeable]) / len(dow_profiles) * 100,
                'avg_efficiency': np.mean([p.price_efficiency for p in dow_profiles]),
                'avg_reversals': np.mean([p.num_reversals for p in dow_profiles]),
            }

    print(f"\n{'Day':12} {'Count':6} {'Avg Quality':12} {'Tradeable %':12} {'Avg Eff':10} {'Avg Rev':8}")
    print("-" * 65)
    for dow, stats in dow_stats.items():
        print(f"{dow:12} {stats['count']:5d} {stats['avg_quality']:11.1f} "
              f"{stats['tradeable_pct']:11.1f}% {stats['avg_efficiency']:9.3f} {stats['avg_reversals']:7.1f}")

    # Export data
    analyzer.export_to_json("daily_profiles.json")

    # Export monthly summaries
    monthly_summaries = {}
    for month in months:
        summary = analyzer.get_month_summary(month)
        if summary:
            monthly_summaries[month] = summary

    summary_path = Path(__file__).parent / "daily_summaries.json"
    with open(summary_path, 'w') as f:
        json.dump(monthly_summaries, f, indent=2)
    print(f"Exported summaries to {summary_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
