"""
Markdown Report Generator for GBPUSD Market Analysis

Generates comprehensive markdown reports from market analysis data.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import json


class MarkdownReportGenerator:
    """Generator for markdown-formatted market analysis reports."""

    MONTH_NAMES = {
        "01": "January", "02": "February", "03": "March", "04": "April",
        "05": "May", "06": "June", "07": "July", "08": "August",
        "09": "September", "10": "October", "11": "November", "12": "December"
    }

    def __init__(self, monthly_profiles: Dict, daily_profiles: Dict):
        """
        Initialize generator with profile data.

        Args:
            monthly_profiles: Dict of monthly market profiles keyed by YYYY-MM
            daily_profiles: Dict of daily market profiles keyed by YYYY-MM-DD
        """
        self.monthly_profiles = monthly_profiles
        self.daily_profiles = daily_profiles

    # =========================================================================
    # TABLE FORMATTING HELPERS
    # =========================================================================

    @staticmethod
    def format_table(headers: List[str], rows: List[List[Any]],
                     alignments: Optional[List[str]] = None) -> str:
        """
        Format data as a markdown table.

        Args:
            headers: List of column headers
            rows: List of row data (each row is a list)
            alignments: List of alignments ('left', 'center', 'right')

        Returns:
            Formatted markdown table string
        """
        if not alignments:
            alignments = ['left'] * len(headers)

        # Create header row
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"

        # Create separator row with alignments
        sep_parts = []
        for align in alignments:
            if align == 'center':
                sep_parts.append(':---:')
            elif align == 'right':
                sep_parts.append('---:')
            else:
                sep_parts.append(':---')
        separator_row = "| " + " | ".join(sep_parts) + " |"

        # Create data rows
        data_rows = []
        for row in rows:
            data_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join([header_row, separator_row] + data_rows)

    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """Format a number with specified decimal places."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format a number as percentage."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}%"

    @staticmethod
    def format_pips(value: float, decimals: int = 1) -> str:
        """Format a value as pips."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"

    @staticmethod
    def bool_to_str(value: bool) -> str:
        """Convert boolean to Yes/No string."""
        return "Yes" if value else "No"

    @staticmethod
    def format_hour(hour: int) -> str:
        """Format hour as HH:00 UTC."""
        return f"{hour:02d}:00"

    @staticmethod
    def format_hours_list(hours: List[int]) -> str:
        """Format a list of hours."""
        return ", ".join(f"{h:02d}:00" for h in hours)

    # =========================================================================
    # MONTHLY REPORT GENERATION
    # =========================================================================

    def generate_monthly_report(self, year_month: str) -> str:
        """
        Generate a comprehensive monthly report.

        Args:
            year_month: Month in YYYY-MM format

        Returns:
            Markdown formatted report string
        """
        if year_month not in self.monthly_profiles:
            return f"# Error\n\nNo data available for {year_month}"

        profile = self.monthly_profiles[year_month]
        year, month = year_month.split("-")
        month_name = self.MONTH_NAMES.get(month, month)

        # Get daily data for this month
        daily_data = self._get_daily_data_for_month(year_month)

        # Build report sections
        sections = [
            self._build_monthly_header(year, month_name, year_month),
            self._build_executive_summary(profile, daily_data),
            self._build_volatility_analysis(profile, daily_data),
            self._build_trend_analysis(profile, daily_data),
            self._build_choppiness_analysis(profile, daily_data),
            self._build_session_analysis(profile, daily_data),
            self._build_day_of_week_analysis(profile, daily_data),
            self._build_daily_breakdown_table(daily_data),
            self._build_risk_recommendations(profile, daily_data),
            self._build_trading_notes(profile),
            self._build_previous_month_comparison(year_month, profile),
        ]

        return "\n\n".join(sections)

    def _get_daily_data_for_month(self, year_month: str) -> List[Dict]:
        """Get all daily profiles for a specific month."""
        return [
            profile for date, profile in self.daily_profiles.items()
            if date.startswith(year_month)
        ]

    def _build_monthly_header(self, year: str, month_name: str,
                              year_month: str) -> str:
        """Build the header section of the monthly report."""
        return f"""# GBPUSD Market Analysis - {month_name} {year}

**Report Period:** {year_month}-01 to {year_month}-31
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
**Timeframe:** H1 (Primary), D1 (Context)

---"""

    def _build_executive_summary(self, profile: Dict, daily_data: List[Dict]) -> str:
        """Build executive summary section."""
        trading_days = len(daily_data)
        tradeable_days = sum(1 for d in daily_data if d.get('tradeable', False))
        tradeable_pct = (tradeable_days / trading_days * 100) if trading_days > 0 else 0
        avg_quality = sum(d.get('quality_score', 0) for d in daily_data) / trading_days if trading_days > 0 else 0

        table = self.format_table(
            headers=["Metric", "Value", "Assessment"],
            rows=[
                ["Volatility Regime", profile.get('volatility_regime', 'N/A'),
                 self._assess_volatility(profile.get('volatility_regime'))],
                ["Trend Regime", profile.get('trend_regime', 'N/A'),
                 self._assess_trend(profile.get('trend_regime'))],
                ["Trading Days", str(trading_days), "-"],
                ["Tradeable Days", f"{tradeable_days} ({self.format_percentage(tradeable_pct)})",
                 self._assess_tradeable_pct(tradeable_pct)],
                ["Avg Quality Score", self.format_number(avg_quality, 1),
                 self._assess_quality(avg_quality)],
                ["Risk Multiplier", f"{profile.get('recommended_risk_mult', 1.0):.2f}x",
                 self._assess_risk_mult(profile.get('recommended_risk_mult', 1.0))],
                ["Quality Threshold", self.format_number(profile.get('recommended_quality_threshold', 50), 0), "-"],
            ],
            alignments=['left', 'center', 'left']
        )

        return f"""## 1. Executive Summary

{table}"""

    def _build_volatility_analysis(self, profile: Dict, daily_data: List[Dict]) -> str:
        """Build volatility analysis section."""
        atr_values = [d.get('atr_pips', 0) for d in daily_data if d.get('atr_pips')]
        range_values = [d.get('daily_range_pips', 0) for d in daily_data if d.get('daily_range_pips')]

        if atr_values:
            atr_avg = sum(atr_values) / len(atr_values)
            atr_min = min(atr_values)
            atr_max = max(atr_values)
            atr_std = self._calculate_std(atr_values)
        else:
            atr_avg = atr_min = atr_max = atr_std = 0

        if range_values:
            range_avg = sum(range_values) / len(range_values)
            range_min = min(range_values)
            range_max = max(range_values)
        else:
            range_avg = range_min = range_max = 0

        # Volatility distribution
        high_vol = sum(1 for d in daily_data if d.get('is_high_volatility', False))
        low_vol = sum(1 for d in daily_data if d.get('is_low_volatility', False))
        normal_vol = len(daily_data) - high_vol - low_vol

        atr_table = self.format_table(
            headers=["Statistic", "ATR (pips)", "Daily Range (pips)"],
            rows=[
                ["Average", self.format_pips(atr_avg), self.format_pips(range_avg)],
                ["Minimum", self.format_pips(atr_min), self.format_pips(range_min)],
                ["Maximum", self.format_pips(atr_max), self.format_pips(range_max)],
                ["Std Dev", self.format_pips(atr_std), "-"],
            ],
            alignments=['left', 'right', 'right']
        )

        dist_table = self.format_table(
            headers=["Volatility Level", "Days", "Percentage"],
            rows=[
                ["High", str(high_vol), self.format_percentage(high_vol/len(daily_data)*100 if daily_data else 0)],
                ["Normal", str(normal_vol), self.format_percentage(normal_vol/len(daily_data)*100 if daily_data else 0)],
                ["Low", str(low_vol), self.format_percentage(low_vol/len(daily_data)*100 if daily_data else 0)],
            ],
            alignments=['left', 'center', 'right']
        )

        return f"""## 2. Volatility Analysis

### ATR Statistics
{atr_table}

### Volatility Distribution
{dist_table}"""

    def _build_trend_analysis(self, profile: Dict, daily_data: List[Dict]) -> str:
        """Build trend analysis section."""
        adx_values = [d.get('adx', 0) for d in daily_data if d.get('adx')]
        plus_di = [d.get('plus_di', 0) for d in daily_data if d.get('plus_di')]
        minus_di = [d.get('minus_di', 0) for d in daily_data if d.get('minus_di')]
        efficiency = [d.get('price_efficiency', 0) for d in daily_data if d.get('price_efficiency')]

        adx_table = self.format_table(
            headers=["Metric", "Average", "Min", "Max"],
            rows=[
                ["ADX",
                 self.format_number(sum(adx_values)/len(adx_values) if adx_values else 0),
                 self.format_number(min(adx_values) if adx_values else 0),
                 self.format_number(max(adx_values) if adx_values else 0)],
                ["+DI",
                 self.format_number(sum(plus_di)/len(plus_di) if plus_di else 0),
                 self.format_number(min(plus_di) if plus_di else 0),
                 self.format_number(max(plus_di) if plus_di else 0)],
                ["-DI",
                 self.format_number(sum(minus_di)/len(minus_di) if minus_di else 0),
                 self.format_number(min(minus_di) if minus_di else 0),
                 self.format_number(max(minus_di) if minus_di else 0)],
            ],
            alignments=['left', 'right', 'right', 'right']
        )

        # Trend direction distribution
        bullish = sum(1 for d in daily_data if d.get('trend_direction') == 'BULLISH')
        bearish = sum(1 for d in daily_data if d.get('trend_direction') == 'BEARISH')
        neutral = sum(1 for d in daily_data if d.get('trend_direction') == 'NEUTRAL')

        direction_table = self.format_table(
            headers=["Direction", "Days", "Percentage"],
            rows=[
                ["Bullish", str(bullish), self.format_percentage(bullish/len(daily_data)*100 if daily_data else 0)],
                ["Bearish", str(bearish), self.format_percentage(bearish/len(daily_data)*100 if daily_data else 0)],
                ["Neutral", str(neutral), self.format_percentage(neutral/len(daily_data)*100 if daily_data else 0)],
            ],
            alignments=['left', 'center', 'right']
        )

        avg_efficiency = sum(efficiency)/len(efficiency) if efficiency else 0

        return f"""## 3. Trend Analysis

### ADX Statistics
{adx_table}

### Trend Direction Distribution
{direction_table}

### Price Efficiency
- **Average Efficiency:** {self.format_percentage(avg_efficiency * 100)}
- **Assessment:** {self._assess_efficiency(avg_efficiency)}"""

    def _build_choppiness_analysis(self, profile: Dict, daily_data: List[Dict]) -> str:
        """Build choppiness analysis section."""
        chop_values = [d.get('choppiness', 0) for d in daily_data if d.get('choppiness')]

        if chop_values:
            chop_avg = sum(chop_values) / len(chop_values)
            chop_min = min(chop_values)
            chop_max = max(chop_values)
        else:
            chop_avg = chop_min = chop_max = 0

        ranging = sum(1 for d in daily_data if d.get('is_ranging', False))
        choppy = sum(1 for d in daily_data if d.get('is_choppy', False))
        trending = len(daily_data) - ranging - choppy

        chop_table = self.format_table(
            headers=["Statistic", "Value"],
            rows=[
                ["Average Choppiness", self.format_number(chop_avg)],
                ["Minimum", self.format_number(chop_min)],
                ["Maximum", self.format_number(chop_max)],
            ],
            alignments=['left', 'right']
        )

        structure_table = self.format_table(
            headers=["Market Structure", "Days", "Percentage"],
            rows=[
                ["Trending", str(trending), self.format_percentage(trending/len(daily_data)*100 if daily_data else 0)],
                ["Ranging", str(ranging), self.format_percentage(ranging/len(daily_data)*100 if daily_data else 0)],
                ["Choppy", str(choppy), self.format_percentage(choppy/len(daily_data)*100 if daily_data else 0)],
            ],
            alignments=['left', 'center', 'right']
        )

        return f"""## 4. Choppiness Analysis

### Choppiness Index Statistics
{chop_table}

### Market Structure Distribution
{structure_table}"""

    def _build_session_analysis(self, profile: Dict, daily_data: List[Dict]) -> str:
        """Build session analysis section."""
        asian = [d.get('asian_range', 0) for d in daily_data if d.get('asian_range')]
        london = [d.get('london_range', 0) for d in daily_data if d.get('london_range')]
        ny = [d.get('ny_range', 0) for d in daily_data if d.get('ny_range')]

        session_table = self.format_table(
            headers=["Session", "Avg Range (pips)", "Best For Trading"],
            rows=[
                ["Asian (00:00-08:00)",
                 self.format_pips(sum(asian)/len(asian) if asian else 0),
                 "Low" if sum(asian)/len(asian) < 30 else "Medium" if asian else "N/A"],
                ["London (08:00-16:00)",
                 self.format_pips(sum(london)/len(london) if london else 0),
                 "High" if sum(london)/len(london) > 40 else "Medium" if london else "N/A"],
                ["New York (13:00-21:00)",
                 self.format_pips(sum(ny)/len(ny) if ny else 0),
                 "High" if sum(ny)/len(ny) > 50 else "Medium" if ny else "N/A"],
            ],
            alignments=['left', 'right', 'center']
        )

        best_hours = profile.get('best_hours', [])
        worst_hours = profile.get('worst_hours', [])

        hours_table = self.format_table(
            headers=["Category", "Hours (UTC)", "Notes"],
            rows=[
                ["Best Hours", self.format_hours_list(best_hours[:3]) if best_hours else "N/A",
                 "Highest average range"],
                ["Worst Hours", self.format_hours_list(worst_hours[:3]) if worst_hours else "N/A",
                 "Lowest average range"],
            ],
            alignments=['left', 'left', 'left']
        )

        return f"""## 5. Session Analysis

### Session Range Comparison
{session_table}

### Optimal Trading Hours
{hours_table}"""

    def _build_day_of_week_analysis(self, profile: Dict, daily_data: List[Dict]) -> str:
        """Build day of week analysis section."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_stats = {}

        for day in days:
            day_data = [d for d in daily_data if d.get('day_of_week') == day]
            if day_data:
                avg_quality = sum(d.get('quality_score', 0) for d in day_data) / len(day_data)
                tradeable_pct = sum(1 for d in day_data if d.get('tradeable', False)) / len(day_data) * 100
                avg_range = sum(d.get('daily_range_pips', 0) for d in day_data) / len(day_data)
            else:
                avg_quality = tradeable_pct = avg_range = 0
            day_stats[day] = {'quality': avg_quality, 'tradeable': tradeable_pct,
                             'range': avg_range, 'count': len(day_data)}

        rows = []
        for day in days:
            stats = day_stats[day]
            best_marker = " *" if day[:3] in profile.get('best_days', []) else ""
            worst_marker = " **" if day[:3] in profile.get('worst_days', []) else ""
            rows.append([
                f"{day}{best_marker}{worst_marker}",
                str(stats['count']),
                self.format_number(stats['quality'], 1),
                self.format_percentage(stats['tradeable']),
                self.format_pips(stats['range'])
            ])

        table = self.format_table(
            headers=["Day", "Count", "Avg Quality", "Tradeable %", "Avg Range"],
            rows=rows,
            alignments=['left', 'center', 'right', 'right', 'right']
        )

        best_days = profile.get('best_days', [])
        worst_days = profile.get('worst_days', [])

        return f"""## 6. Day of Week Analysis

{table}

*\\* Best days for trading | \\*\\* Worst days for trading*

- **Best Days:** {', '.join(best_days) if best_days else 'N/A'}
- **Worst Days:** {', '.join(worst_days) if worst_days else 'N/A'}"""

    def _build_daily_breakdown_table(self, daily_data: List[Dict]) -> str:
        """Build daily breakdown table."""
        if not daily_data:
            return "## 7. Daily Breakdown\n\nNo daily data available."

        # Sort by date
        sorted_data = sorted(daily_data, key=lambda x: x.get('date', ''))

        rows = []
        for d in sorted_data:
            date = d.get('date', 'N/A')
            day = d.get('day_of_week', 'N/A')[:3]
            quality = self.format_number(d.get('quality_score', 0), 0)
            atr = self.format_pips(d.get('atr_pips', 0))
            adx = self.format_number(d.get('adx', 0), 1)
            chop = self.format_number(d.get('choppiness', 0), 1)
            eff = self.format_percentage(d.get('price_efficiency', 0) * 100)
            tradeable = "Yes" if d.get('tradeable', False) else "No"
            trend = d.get('trend_direction', 'N/A')[:4] if d.get('trend_direction') else "N/A"

            rows.append([date, day, quality, atr, adx, chop, eff, tradeable, trend])

        table = self.format_table(
            headers=["Date", "Day", "Quality", "ATR", "ADX", "Chop", "Eff%", "Trade", "Trend"],
            rows=rows,
            alignments=['left', 'center', 'right', 'right', 'right', 'right', 'right', 'center', 'center']
        )

        return f"""## 7. Daily Breakdown

{table}"""

    def _build_risk_recommendations(self, profile: Dict, daily_data: List[Dict]) -> str:
        """Build risk recommendations section."""
        avg_atr = profile.get('avg_atr_pips', 15)
        risk_mult = profile.get('recommended_risk_mult', 1.0)
        quality_threshold = profile.get('recommended_quality_threshold', 50)

        position_table = self.format_table(
            headers=["Risk Level", "Position Size", "Stop Loss (ATR)", "Take Profit (ATR)"],
            rows=[
                ["Conservative", f"{risk_mult * 0.5:.2f}x base", "1.5x", "2.0x"],
                ["Standard", f"{risk_mult:.2f}x base", "1.2x", "1.8x"],
                ["Aggressive", f"{risk_mult * 1.5:.2f}x base", "1.0x", "1.5x"],
            ],
            alignments=['left', 'center', 'center', 'center']
        )

        sl_pips = avg_atr * 1.2
        tp_pips = avg_atr * 1.8

        return f"""## 8. Risk Recommendations

### Position Sizing
{position_table}

### Stop Loss / Take Profit Guidelines
| Parameter | Pips | Based On |
|:---|---:|:---|
| Suggested Stop Loss | {self.format_pips(sl_pips)} | 1.2x ATR |
| Suggested Take Profit | {self.format_pips(tp_pips)} | 1.8x ATR |
| Average ATR | {self.format_pips(avg_atr)} | Monthly average |

### Quality Filters
- **Minimum Quality Score:** {quality_threshold}
- **Skip days with:** Quality < {quality_threshold}, High choppiness, Low efficiency"""

    def _build_trading_notes(self, profile: Dict) -> str:
        """Build trading notes section."""
        notes = profile.get('trading_notes', 'No specific notes.')

        return f"""## 9. Trading Notes

{notes}

### Key Observations
- Volatility Regime: **{profile.get('volatility_regime', 'N/A')}**
- Trend Regime: **{profile.get('trend_regime', 'N/A')}**
- Price Efficiency: **{self.format_percentage(profile.get('price_efficiency', 0) * 100)}**"""

    def _build_previous_month_comparison(self, year_month: str, profile: Dict) -> str:
        """Build comparison with previous month."""
        # Calculate previous month
        year, month = map(int, year_month.split("-"))
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1
        prev_key = f"{prev_year}-{prev_month:02d}"

        if prev_key not in self.monthly_profiles:
            return "## 10. Month-over-Month Comparison\n\n*No previous month data available for comparison.*"

        prev = self.monthly_profiles[prev_key]

        def change_indicator(current, previous):
            if current > previous:
                return "^"
            elif current < previous:
                return "v"
            return "-"

        def format_change(current, previous, fmt="pips"):
            diff = current - previous
            sign = "+" if diff > 0 else ""
            if fmt == "pips":
                return f"{sign}{diff:.1f}"
            elif fmt == "pct":
                return f"{sign}{diff:.1f}%"
            return f"{sign}{diff:.2f}"

        table = self.format_table(
            headers=["Metric", "This Month", "Previous", "Change"],
            rows=[
                ["Avg ATR (pips)",
                 self.format_pips(profile.get('avg_atr_pips', 0)),
                 self.format_pips(prev.get('avg_atr_pips', 0)),
                 format_change(profile.get('avg_atr_pips', 0), prev.get('avg_atr_pips', 0))],
                ["Avg ADX",
                 self.format_number(profile.get('avg_adx', 0)),
                 self.format_number(prev.get('avg_adx', 0)),
                 format_change(profile.get('avg_adx', 0), prev.get('avg_adx', 0))],
                ["Choppiness",
                 self.format_number(profile.get('avg_choppiness', 0)),
                 self.format_number(prev.get('avg_choppiness', 0)),
                 format_change(profile.get('avg_choppiness', 0), prev.get('avg_choppiness', 0))],
                ["Efficiency",
                 self.format_percentage(profile.get('price_efficiency', 0) * 100),
                 self.format_percentage(prev.get('price_efficiency', 0) * 100),
                 format_change(profile.get('price_efficiency', 0) * 100,
                              prev.get('price_efficiency', 0) * 100, "pct")],
                ["Risk Mult",
                 f"{profile.get('recommended_risk_mult', 1.0):.2f}x",
                 f"{prev.get('recommended_risk_mult', 1.0):.2f}x",
                 format_change(profile.get('recommended_risk_mult', 1.0),
                              prev.get('recommended_risk_mult', 1.0))],
            ],
            alignments=['left', 'right', 'right', 'right']
        )

        return f"""## 10. Month-over-Month Comparison

{table}"""

    # =========================================================================
    # DAILY REPORT GENERATION
    # =========================================================================

    def generate_daily_report(self, date: str) -> str:
        """
        Generate a daily market report.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Markdown formatted report string
        """
        if date not in self.daily_profiles:
            return f"# Error\n\nNo data available for {date}"

        profile = self.daily_profiles[date]

        sections = [
            self._build_daily_header(date, profile),
            self._build_quick_summary(profile),
            self._build_price_action(profile),
            self._build_technical_indicators(profile),
            self._build_daily_session_analysis(profile),
            self._build_daily_recommendations(profile),
        ]

        return "\n\n".join(sections)

    def _build_daily_header(self, date: str, profile: Dict) -> str:
        """Build daily report header."""
        day_of_week = profile.get('day_of_week', 'Unknown')

        return f"""# GBPUSD Daily Analysis - {date}

**Day:** {day_of_week}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

---"""

    def _build_quick_summary(self, profile: Dict) -> str:
        """Build quick summary section for daily report."""
        table = self.format_table(
            headers=["Metric", "Value", "Assessment"],
            rows=[
                ["Quality Score", self.format_number(profile.get('quality_score', 0), 0),
                 self._assess_quality(profile.get('quality_score', 0))],
                ["Tradeable", self.bool_to_str(profile.get('tradeable', False)),
                 "Recommended" if profile.get('tradeable', False) else "Skip"],
                ["Risk Multiplier", f"{profile.get('risk_multiplier', 1.0):.1f}x", "-"],
                ["Trend Direction", profile.get('trend_direction', 'N/A'), "-"],
                ["Trend Strength", profile.get('trend_strength', 'N/A'), "-"],
                ["Best Session", profile.get('best_session', 'N/A'), "-"],
                ["Best Hour", self.format_hour(profile.get('best_hour', 0)) if profile.get('best_hour') else "N/A", "-"],
            ],
            alignments=['left', 'center', 'left']
        )

        return f"""## Quick Summary

{table}"""

    def _build_price_action(self, profile: Dict) -> str:
        """Build price action section."""
        table = self.format_table(
            headers=["Metric", "Value"],
            rows=[
                ["Open", f"{profile.get('open_price', 0):.5f}"],
                ["High", f"{profile.get('high_price', 0):.5f}"],
                ["Low", f"{profile.get('low_price', 0):.5f}"],
                ["Close", f"{profile.get('close_price', 0):.5f}"],
                ["Daily Range", f"{self.format_pips(profile.get('daily_range_pips', 0))} pips"],
                ["Net Change", f"{self.format_pips(profile.get('price_change_pips', 0))} pips"],
            ],
            alignments=['left', 'right']
        )

        return f"""## Price Action

{table}"""

    def _build_technical_indicators(self, profile: Dict) -> str:
        """Build technical indicators section."""
        vol_table = self.format_table(
            headers=["Indicator", "Value", "Status"],
            rows=[
                ["ATR (pips)", self.format_pips(profile.get('atr_pips', 0)),
                 "High" if profile.get('is_high_volatility') else "Low" if profile.get('is_low_volatility') else "Normal"],
                ["Volatility Percentile", self.format_percentage(profile.get('volatility_percentile', 0)), "-"],
            ],
            alignments=['left', 'right', 'center']
        )

        trend_table = self.format_table(
            headers=["Indicator", "Value", "Status"],
            rows=[
                ["ADX", self.format_number(profile.get('adx', 0)), profile.get('trend_strength', 'N/A')],
                ["+DI", self.format_number(profile.get('plus_di', 0)), "-"],
                ["-DI", self.format_number(profile.get('minus_di', 0)), "-"],
                ["Direction", profile.get('trend_direction', 'N/A'), "-"],
            ],
            alignments=['left', 'right', 'center']
        )

        chop_table = self.format_table(
            headers=["Indicator", "Value", "Status"],
            rows=[
                ["Choppiness Index", self.format_number(profile.get('choppiness', 0)),
                 "Ranging" if profile.get('is_ranging') else "Choppy" if profile.get('is_choppy') else "Trending"],
            ],
            alignments=['left', 'right', 'center']
        )

        eff_table = self.format_table(
            headers=["Indicator", "Value"],
            rows=[
                ["Price Efficiency", self.format_percentage(profile.get('price_efficiency', 0) * 100)],
                ["Reversals", str(profile.get('num_reversals', 0))],
                ["Max Intraday DD", f"{self.format_pips(profile.get('max_intraday_dd_pips', 0))} pips"],
            ],
            alignments=['left', 'right']
        )

        return f"""## Technical Indicators

### Volatility
{vol_table}

### Trend
{trend_table}

### Choppiness
{chop_table}

### Price Efficiency
{eff_table}"""

    def _build_daily_session_analysis(self, profile: Dict) -> str:
        """Build session analysis for daily report."""
        table = self.format_table(
            headers=["Session", "Range (pips)", "Best"],
            rows=[
                ["Asian", self.format_pips(profile.get('asian_range', 0)),
                 "Yes" if profile.get('best_session') == 'Asian' else ""],
                ["London", self.format_pips(profile.get('london_range', 0)),
                 "Yes" if profile.get('best_session') == 'London' else ""],
                ["New York", self.format_pips(profile.get('ny_range', 0)),
                 "Yes" if profile.get('best_session') == 'NY' else ""],
            ],
            alignments=['left', 'right', 'center']
        )

        best_hour = profile.get('best_hour')
        worst_hour = profile.get('worst_hour')

        return f"""## Session Analysis

{table}

- **Best Hour:** {self.format_hour(best_hour) if best_hour is not None else 'N/A'} ({self.format_pips(profile.get('best_hour_range', 0))} pips)
- **Worst Hour:** {self.format_hour(worst_hour) if worst_hour is not None else 'N/A'} ({self.format_pips(profile.get('worst_hour_range', 0))} pips)"""

    def _build_daily_recommendations(self, profile: Dict) -> str:
        """Build recommendations section for daily report."""
        notes = profile.get('notes', [])
        notes_str = "\n".join(f"- {note}" for note in notes) if notes else "- No specific alerts"

        quality = profile.get('quality_score', 0)
        if quality >= 70:
            assessment = "HIGH QUALITY - Good trading day"
        elif quality >= 50:
            assessment = "MODERATE QUALITY - Trade with caution"
        else:
            assessment = "LOW QUALITY - Consider skipping"

        return f"""## Trading Recommendations

### Quality Assessment
**{assessment}**

### Notes & Alerts
{notes_str}"""

    # =========================================================================
    # DAILY SUMMARY GENERATION
    # =========================================================================

    def generate_daily_summary(self, year_month: str) -> str:
        """
        Generate a summary of all daily reports for a month.

        Args:
            year_month: Month in YYYY-MM format

        Returns:
            Markdown formatted summary string
        """
        year, month = year_month.split("-")
        month_name = self.MONTH_NAMES.get(month, month)

        daily_data = self._get_daily_data_for_month(year_month)
        if not daily_data:
            return f"# Daily Summary - {month_name} {year}\n\nNo daily data available."

        sorted_data = sorted(daily_data, key=lambda x: x.get('date', ''))

        rows = []
        for d in sorted_data:
            date = d.get('date', 'N/A')
            day = d.get('day_of_week', 'N/A')[:3]
            quality = self.format_number(d.get('quality_score', 0), 0)
            tradeable = "Yes" if d.get('tradeable', False) else "No"
            atr = self.format_pips(d.get('atr_pips', 0))
            trend = d.get('trend_direction', 'N/A')[:4] if d.get('trend_direction') else "N/A"
            best_session = d.get('best_session', 'N/A')
            range_pips = self.format_pips(d.get('daily_range_pips', 0))

            # Link to daily report
            link = f"[{date}](./{date}.md)"
            rows.append([link, day, quality, tradeable, atr, range_pips, trend, best_session])

        table = self.format_table(
            headers=["Date", "Day", "Quality", "Trade", "ATR", "Range", "Trend", "Best Session"],
            rows=rows,
            alignments=['left', 'center', 'right', 'center', 'right', 'right', 'center', 'center']
        )

        # Statistics
        total_days = len(daily_data)
        tradeable_days = sum(1 for d in daily_data if d.get('tradeable', False))
        avg_quality = sum(d.get('quality_score', 0) for d in daily_data) / total_days
        avg_atr = sum(d.get('atr_pips', 0) for d in daily_data) / total_days

        return f"""# Daily Summary - {month_name} {year}

## Overview

| Metric | Value |
|:---|---:|
| Total Trading Days | {total_days} |
| Tradeable Days | {tradeable_days} ({self.format_percentage(tradeable_days/total_days*100)}) |
| Average Quality | {self.format_number(avg_quality, 1)} |
| Average ATR | {self.format_pips(avg_atr)} pips |

## Daily Reports

{table}

---

*Click on date to view detailed daily report.*"""

    # =========================================================================
    # INDEX GENERATION
    # =========================================================================

    def generate_index(self) -> str:
        """Generate the main index page."""
        months = sorted(self.monthly_profiles.keys())

        rows = []
        for ym in months:
            profile = self.monthly_profiles[ym]
            year, month = ym.split("-")
            month_name = self.MONTH_NAMES.get(month, month)

            daily_data = self._get_daily_data_for_month(ym)
            trading_days = len(daily_data)
            tradeable_days = sum(1 for d in daily_data if d.get('tradeable', False))
            tradeable_pct = tradeable_days / trading_days * 100 if trading_days > 0 else 0

            link = f"[{month_name} {year}](./monthly/{ym}_{month_name.lower()}.md)"
            rows.append([
                link,
                profile.get('volatility_regime', 'N/A'),
                profile.get('trend_regime', 'N/A'),
                self.format_pips(profile.get('avg_atr_pips', 0)),
                self.format_number(profile.get('avg_adx', 0)),
                str(trading_days),
                self.format_percentage(tradeable_pct),
                f"{profile.get('recommended_risk_mult', 1.0):.2f}x"
            ])

        table = self.format_table(
            headers=["Month", "Volatility", "Trend", "Avg ATR", "Avg ADX", "Days", "Tradeable%", "Risk Mult"],
            rows=rows,
            alignments=['left', 'center', 'center', 'right', 'right', 'center', 'right', 'center']
        )

        # Determine period from data
        first_month = months[0] if months else "N/A"
        last_month = months[-1] if months else "N/A"
        first_year, first_m = first_month.split("-") if "-" in first_month else ("N/A", "01")
        last_year, last_m = last_month.split("-") if "-" in last_month else ("N/A", "01")
        first_month_name = self.MONTH_NAMES.get(first_m, first_m)
        last_month_name = self.MONTH_NAMES.get(last_m, last_m)

        return f"""# GBPUSD Market Analysis Index

**Period:** {first_month_name} {first_year} - {last_month_name} {last_year}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

---

## Monthly Reports Summary

{table}

---

## Quick Navigation

### Monthly Reports
{self._generate_monthly_links(months)}

### Daily Reports
{self._generate_daily_links(months)}

---

## Data Sources

- Primary Timeframe: H1
- Supporting: D1, H4
- Source: MT5 / TimescaleDB

## Report Types

1. **Monthly Reports** - Comprehensive monthly analysis with all metrics
2. **Daily Reports** - Individual day breakdown with session analysis
3. **Daily Summaries** - Quick overview of all days in a month"""

    def _generate_monthly_links(self, months: List[str]) -> str:
        """Generate monthly report links."""
        links = []
        for ym in months:
            year, month = ym.split("-")
            month_name = self.MONTH_NAMES.get(month, month)
            links.append(f"- [{month_name} {year}](./monthly/{ym}_{month_name.lower()}.md)")
        return "\n".join(links)

    def _generate_daily_links(self, months: List[str]) -> str:
        """Generate daily summary links."""
        links = []
        for ym in months:
            year, month = ym.split("-")
            month_name = self.MONTH_NAMES.get(month, month)
            links.append(f"- [{month_name} {year}](./daily/{ym}/summary.md)")
        return "\n".join(links)

    # =========================================================================
    # HELPER ASSESSMENT METHODS
    # =========================================================================

    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    @staticmethod
    def _assess_volatility(regime: str) -> str:
        """Assess volatility regime."""
        if regime == "HIGH":
            return "Reduce position size"
        elif regime == "LOW":
            return "Increase position size"
        return "Normal sizing"

    @staticmethod
    def _assess_trend(regime: str) -> str:
        """Assess trend regime."""
        if regime == "TRENDING":
            return "Momentum entries preferred"
        elif regime == "RANGING":
            return "Mean reversion preferred"
        return "Mixed strategies"

    @staticmethod
    def _assess_tradeable_pct(pct: float) -> str:
        """Assess tradeable percentage."""
        if pct >= 70:
            return "Excellent"
        elif pct >= 50:
            return "Good"
        elif pct >= 30:
            return "Fair"
        return "Poor"

    @staticmethod
    def _assess_quality(score: float) -> str:
        """Assess quality score."""
        if score >= 70:
            return "High Quality"
        elif score >= 50:
            return "Moderate"
        elif score >= 30:
            return "Low Quality"
        return "Very Low"

    @staticmethod
    def _assess_risk_mult(mult: float) -> str:
        """Assess risk multiplier."""
        if mult >= 1.0:
            return "Full risk"
        elif mult >= 0.7:
            return "Moderate reduction"
        return "Significant reduction"

    @staticmethod
    def _assess_efficiency(eff: float) -> str:
        """Assess price efficiency."""
        if eff >= 0.1:
            return "Good directional moves"
        elif eff >= 0.05:
            return "Moderate efficiency"
        return "Many false moves expected"
