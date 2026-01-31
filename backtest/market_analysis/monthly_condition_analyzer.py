"""Monthly & Daily Market Condition Analyzer
=============================================

Deep analysis of market conditions per month and per day to understand:
1. WHY certain months/days are profitable vs losing
2. WHAT market characteristics define each period
3. HOW to create dynamic rules based on actual patterns

This creates a "Market DNA" profile for each period that can be used
to dynamically adjust trading parameters.

Analysis includes:
- Volatility regime (ATR, range)
- Trend strength (ADX, directional movement)
- Choppiness (ranging vs trending)
- Momentum (velocity, acceleration)
- Seasonal patterns
- Hour-of-day patterns
- Day-of-week patterns

Author: SURIOTA Team
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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger
import json

from config import config
from src.data.db_handler import DBHandler


@dataclass
class MonthProfile:
    """Market profile for a specific month"""
    month: str                      # YYYY-MM

    # Volatility metrics
    avg_atr_pips: float = 0.0
    atr_percentile: float = 0.0     # vs all months (0-100)
    avg_daily_range_pips: float = 0.0
    range_percentile: float = 0.0
    volatility_regime: str = ""     # HIGH, NORMAL, LOW

    # Trend metrics
    avg_adx: float = 0.0
    adx_percentile: float = 0.0
    trend_days_pct: float = 0.0     # % of days with ADX > 25
    dominant_direction: str = ""    # BULLISH, BEARISH, MIXED
    trend_regime: str = ""          # TRENDING, RANGING, MIXED

    # Choppiness metrics
    avg_choppiness: float = 0.0
    chop_percentile: float = 0.0
    ranging_days_pct: float = 0.0   # % of days with chop > 61.8

    # Price action
    monthly_change_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    price_efficiency: float = 0.0   # |net move| / total range

    # Activity metrics
    avg_hourly_range_pips: float = 0.0
    best_hours: List[int] = field(default_factory=list)
    worst_hours: List[int] = field(default_factory=list)

    # Day patterns
    best_days: List[str] = field(default_factory=list)
    worst_days: List[str] = field(default_factory=list)

    # Trading recommendation
    recommended_risk_mult: float = 1.0
    recommended_quality_threshold: float = 50.0
    trading_notes: str = ""


@dataclass
class DayProfile:
    """Market profile for a specific day"""
    date: str                       # YYYY-MM-DD
    day_of_week: str               # Monday, Tuesday, etc.

    # Volatility
    atr_pips: float = 0.0
    daily_range_pips: float = 0.0
    is_low_volatility: bool = False

    # Trend
    adx: float = 0.0
    is_trending: bool = False
    direction: str = ""             # BULLISH, BEARISH, NEUTRAL

    # Choppiness
    choppiness: float = 0.0
    is_ranging: bool = False

    # Hour analysis
    best_hour: int = 0
    worst_hour: int = 0
    active_hours: List[int] = field(default_factory=list)

    # Price action
    open_price: float = 0.0
    close_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    change_pips: float = 0.0

    # Quality score for trading
    quality_score: float = 50.0
    recommended_risk: float = 1.0


class MarketConditionAnalyzer:
    """Comprehensive market condition analyzer"""

    def __init__(self):
        self.monthly_profiles: Dict[str, MonthProfile] = {}
        self.daily_profiles: Dict[str, DayProfile] = {}
        self.hourly_stats: Dict[int, Dict] = {}
        self.dow_stats: Dict[str, Dict] = {}

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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
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

        # ADX
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

        # Momentum (simple)
        df['momentum'] = close.diff(5)
        df['momentum_pct'] = close.pct_change(5) * 100

        return df

    def _get_col_map(self, df: pd.DataFrame) -> dict:
        return {
            'open': 'open' if 'open' in df.columns else 'Open',
            'high': 'high' if 'high' in df.columns else 'High',
            'low': 'low' if 'low' in df.columns else 'Low',
            'close': 'close' if 'close' in df.columns else 'Close',
        }

    def analyze_month(self, df: pd.DataFrame, month: str) -> MonthProfile:
        """Analyze a specific month's market conditions"""
        col_map = self._get_col_map(df)

        profile = MonthProfile(month=month)

        if df.empty:
            return profile

        # Volatility metrics
        profile.avg_atr_pips = df['atr_pips'].mean()
        profile.avg_daily_range_pips = df['bar_range_pips'].mean()

        # Trend metrics
        profile.avg_adx = df['adx'].mean()
        profile.trend_days_pct = (df['adx'] > 25).mean() * 100

        # Direction analysis
        plus_di_avg = df['plus_di'].mean()
        minus_di_avg = df['minus_di'].mean()
        if plus_di_avg > minus_di_avg + 5:
            profile.dominant_direction = "BULLISH"
        elif minus_di_avg > plus_di_avg + 5:
            profile.dominant_direction = "BEARISH"
        else:
            profile.dominant_direction = "MIXED"

        # Choppiness metrics
        profile.avg_choppiness = df['choppiness'].mean()
        profile.ranging_days_pct = (df['choppiness'] > 61.8).mean() * 100

        # Regime classification
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

        # Price action
        profile.monthly_change_pips = (df[col_map['close']].iloc[-1] - df[col_map['close']].iloc[0]) * 10000

        # Calculate price efficiency
        total_movement = df['bar_range_pips'].sum()
        net_movement = abs(profile.monthly_change_pips)
        profile.price_efficiency = net_movement / total_movement if total_movement > 0 else 0

        # Hour analysis
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        hourly_ranges = df_copy.groupby('hour')['bar_range_pips'].mean()

        profile.avg_hourly_range_pips = hourly_ranges.mean()
        profile.best_hours = hourly_ranges.nlargest(3).index.tolist()
        profile.worst_hours = hourly_ranges.nsmallest(3).index.tolist()

        # Day of week analysis
        df_copy['dow'] = df_copy.index.dayofweek
        dow_ranges = df_copy.groupby('dow')['bar_range_pips'].mean()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        best_dow_idx = dow_ranges.nlargest(2).index.tolist()
        worst_dow_idx = dow_ranges.nsmallest(2).index.tolist()
        profile.best_days = [dow_names[i] for i in best_dow_idx if i < len(dow_names)]
        profile.worst_days = [dow_names[i] for i in worst_dow_idx if i < len(dow_names)]

        # Trading recommendations
        profile.recommended_risk_mult = self._calculate_risk_multiplier(profile)
        profile.recommended_quality_threshold = self._calculate_quality_threshold(profile)
        profile.trading_notes = self._generate_notes(profile)

        return profile

    def _calculate_risk_multiplier(self, profile: MonthProfile) -> float:
        """Calculate recommended risk multiplier based on conditions"""
        mult = 1.0

        # Volatility adjustment
        if profile.volatility_regime == "LOW":
            mult *= 0.7
        elif profile.volatility_regime == "HIGH":
            mult *= 0.9  # Slightly reduce for high vol

        # Trend adjustment
        if profile.trend_regime == "RANGING":
            mult *= 0.6
        elif profile.trend_regime == "TRENDING":
            mult *= 1.1

        # Choppiness adjustment
        if profile.avg_choppiness > 65:
            mult *= 0.7
        elif profile.avg_choppiness < 50:
            mult *= 1.1

        # Efficiency adjustment
        if profile.price_efficiency < 0.1:
            mult *= 0.6  # Very inefficient = ranging
        elif profile.price_efficiency > 0.3:
            mult *= 1.1  # Efficient = good trends

        return round(max(0.3, min(1.2, mult)), 2)

    def _calculate_quality_threshold(self, profile: MonthProfile) -> float:
        """Calculate recommended quality threshold"""
        threshold = 50.0

        if profile.volatility_regime == "LOW":
            threshold += 15

        if profile.trend_regime == "RANGING":
            threshold += 10

        if profile.avg_choppiness > 65:
            threshold += 10

        return min(80, threshold)

    def _generate_notes(self, profile: MonthProfile) -> str:
        """Generate trading notes for the month"""
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

    def analyze_day(self, df: pd.DataFrame, date: str) -> DayProfile:
        """Analyze a specific day's market conditions"""
        col_map = self._get_col_map(df)
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        profile = DayProfile(date=date)

        if df.empty:
            return profile

        # Basic info
        first_bar = df.iloc[0]
        profile.day_of_week = dow_names[first_bar.name.dayofweek]

        # Volatility
        profile.atr_pips = df['atr_pips'].mean()
        profile.daily_range_pips = (df[col_map['high']].max() - df[col_map['low']].min()) * 10000
        profile.is_low_volatility = profile.atr_pips < 8

        # Trend
        profile.adx = df['adx'].mean()
        profile.is_trending = profile.adx > 25

        plus_di = df['plus_di'].mean()
        minus_di = df['minus_di'].mean()
        if plus_di > minus_di + 3:
            profile.direction = "BULLISH"
        elif minus_di > plus_di + 3:
            profile.direction = "BEARISH"
        else:
            profile.direction = "NEUTRAL"

        # Choppiness
        profile.choppiness = df['choppiness'].mean()
        profile.is_ranging = profile.choppiness > 61.8

        # Hour analysis
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        hourly_ranges = df_copy.groupby('hour')['bar_range_pips'].mean()

        if not hourly_ranges.empty:
            profile.best_hour = hourly_ranges.idxmax()
            profile.worst_hour = hourly_ranges.idxmin()
            profile.active_hours = hourly_ranges[hourly_ranges > hourly_ranges.median()].index.tolist()

        # Price action
        profile.open_price = df[col_map['open']].iloc[0]
        profile.close_price = df[col_map['close']].iloc[-1]
        profile.high_price = df[col_map['high']].max()
        profile.low_price = df[col_map['low']].min()
        profile.change_pips = (profile.close_price - profile.open_price) * 10000

        # Quality score
        profile.quality_score = self._calculate_day_quality(profile)
        profile.recommended_risk = self._calculate_day_risk(profile)

        return profile

    def _calculate_day_quality(self, profile: DayProfile) -> float:
        """Calculate quality score for a day (0-100)"""
        score = 50.0

        # Volatility
        if profile.is_low_volatility:
            score -= 15
        elif profile.atr_pips > 12:
            score += 10

        # Trend
        if profile.is_trending:
            score += 15
        elif profile.adx < 18:
            score -= 10

        # Choppiness
        if profile.is_ranging:
            score -= 15
        elif profile.choppiness < 50:
            score += 10

        # Day of week (based on analysis)
        if profile.day_of_week in ['Thursday', 'Friday']:
            score -= 10
        elif profile.day_of_week == 'Monday':
            score += 5

        return max(0, min(100, score))

    def _calculate_day_risk(self, profile: DayProfile) -> float:
        """Calculate recommended risk for a day"""
        if profile.quality_score >= 70:
            return 1.1
        elif profile.quality_score >= 50:
            return 1.0
        elif profile.quality_score >= 35:
            return 0.7
        else:
            return 0.4

    def generate_monthly_report(self, profiles: Dict[str, MonthProfile]) -> str:
        """Generate comprehensive monthly report"""
        report = []
        report.append("=" * 80)
        report.append("MONTHLY MARKET CONDITION ANALYSIS")
        report.append("=" * 80)
        report.append("")

        for month, profile in sorted(profiles.items()):
            report.append(f"\n{'='*60}")
            report.append(f"MONTH: {month}")
            report.append(f"{'='*60}")

            report.append(f"\nVOLATILITY:")
            report.append(f"  Regime: {profile.volatility_regime}")
            report.append(f"  Avg ATR: {profile.avg_atr_pips:.1f} pips")
            report.append(f"  Avg Daily Range: {profile.avg_daily_range_pips:.1f} pips")

            report.append(f"\nTREND:")
            report.append(f"  Regime: {profile.trend_regime}")
            report.append(f"  Avg ADX: {profile.avg_adx:.1f}")
            report.append(f"  Trending Days: {profile.trend_days_pct:.0f}%")
            report.append(f"  Direction: {profile.dominant_direction}")

            report.append(f"\nCHOPPINESS:")
            report.append(f"  Avg Choppiness: {profile.avg_choppiness:.1f}")
            report.append(f"  Ranging Days: {profile.ranging_days_pct:.0f}%")

            report.append(f"\nPRICE ACTION:")
            report.append(f"  Monthly Change: {profile.monthly_change_pips:+.0f} pips")
            report.append(f"  Price Efficiency: {profile.price_efficiency:.1%}")

            report.append(f"\nTIMING:")
            report.append(f"  Best Hours: {profile.best_hours}")
            report.append(f"  Worst Hours: {profile.worst_hours}")
            report.append(f"  Best Days: {profile.best_days}")
            report.append(f"  Worst Days: {profile.worst_days}")

            report.append(f"\nRECOMMENDATIONS:")
            report.append(f"  Risk Multiplier: {profile.recommended_risk_mult}x")
            report.append(f"  Quality Threshold: {profile.recommended_quality_threshold}")
            report.append(f"  Notes: {profile.trading_notes}")

        return "\n".join(report)

    def export_to_json(self, profiles: Dict[str, MonthProfile], filename: str):
        """Export profiles to JSON for use in trading system"""
        data = {}
        for month, profile in profiles.items():
            data[month] = {
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

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported to {filename}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

    print("\n" + "=" * 80)
    print("MONTHLY MARKET CONDITION ANALYZER")
    print("Deep analysis for dynamic trading rules")
    print("=" * 80)

    analyzer = MarketConditionAnalyzer()

    # Fetch H1 data for full period
    print("\n[1/4] Fetching H1 data...")
    df = await analyzer.fetch_data(
        "GBPUSD", "H1",
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 31, tzinfo=timezone.utc)
    )

    if df.empty:
        print("ERROR: No data")
        return

    print(f"      Loaded {len(df)} bars")

    # Calculate indicators
    print("\n[2/4] Calculating indicators...")
    df = analyzer.calculate_indicators(df)

    # Analyze each month
    print("\n[3/4] Analyzing monthly conditions...")
    monthly_profiles = {}

    # Group by month
    df['month'] = df.index.to_period('M').astype(str)
    months = df['month'].unique()

    for month in months:
        month_df = df[df['month'] == month]
        profile = analyzer.analyze_month(month_df, month)
        monthly_profiles[month] = profile
        print(f"      {month}: {profile.volatility_regime} vol, {profile.trend_regime}, risk={profile.recommended_risk_mult}x")

    # Generate report
    print("\n[4/4] Generating report...")
    report = analyzer.generate_monthly_report(monthly_profiles)
    print(report)

    # Export to JSON
    output_file = Path(__file__).parent / "monthly_profiles.json"
    analyzer.export_to_json(monthly_profiles, str(output_file))

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY: OPTIMAL VS CHALLENGING MONTHS")
    print("=" * 80)

    # Sort by recommended risk
    sorted_months = sorted(monthly_profiles.items(),
                          key=lambda x: x[1].recommended_risk_mult,
                          reverse=True)

    print("\nBEST MONTHS (highest risk multiplier):")
    for month, profile in sorted_months[:3]:
        print(f"  {month}: {profile.recommended_risk_mult}x - {profile.trend_regime}, {profile.volatility_regime} vol")

    print("\nCHALLENGING MONTHS (lowest risk multiplier):")
    for month, profile in sorted_months[-3:]:
        print(f"  {month}: {profile.recommended_risk_mult}x - {profile.trend_regime}, {profile.volatility_regime} vol")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
