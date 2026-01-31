#!/usr/bin/env python3
"""
GBPUSD Market Analysis Report Generator

Generates comprehensive markdown reports from market analysis JSON data.

Usage:
    python generate_market_reports.py --all              # Generate all reports
    python generate_market_reports.py --month 2025-01    # Generate specific month
    python generate_market_reports.py --update           # Update current month only
    python generate_market_reports.py --index            # Regenerate index only
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.markdown_report_generator import MarkdownReportGenerator


class MarketReportGenerator:
    """Main class for generating market analysis reports."""

    MONTH_NAMES = {
        "01": "january", "02": "february", "03": "march", "04": "april",
        "05": "may", "06": "june", "07": "july", "08": "august",
        "09": "september", "10": "october", "11": "november", "12": "december"
    }

    def __init__(self, project_root: Path = None):
        """Initialize the report generator."""
        self.project_root = project_root or PROJECT_ROOT
        self.data_dir = self.project_root / "backtest" / "market_analysis"
        self.output_dir = self.project_root / "reports" / "market_analysis"

        # Load data
        self.monthly_profiles = self._load_json("monthly_profiles.json")
        self.daily_profiles = self._load_json("daily_profiles.json")

        # Initialize markdown generator
        self.md_generator = MarkdownReportGenerator(
            self.monthly_profiles,
            self.daily_profiles
        )

        # Ensure directories exist
        self._ensure_directories()

    def _load_json(self, filename: str) -> dict:
        """Load JSON data file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        # Main directories
        (self.output_dir / "monthly").mkdir(parents=True, exist_ok=True)

        # Daily directories for each month
        for year_month in self.monthly_profiles.keys():
            (self.output_dir / "daily" / year_month).mkdir(parents=True, exist_ok=True)

    def _write_report(self, filepath: Path, content: str):
        """Write report content to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Created: {filepath.relative_to(self.project_root)}")

    # =========================================================================
    # MAIN GENERATION METHODS
    # =========================================================================

    def generate_all(self):
        """Generate all reports."""
        print("=" * 60)
        print("GBPUSD Market Analysis Report Generator")
        print("=" * 60)

        # Generate index and README
        self.generate_index()
        self.generate_readme()

        # Generate monthly reports
        self.generate_all_monthly()

        # Generate daily reports
        self.generate_all_daily()

        print("\n" + "=" * 60)
        print("Report generation complete!")
        print("=" * 60)
        self._print_summary()

    def generate_all_monthly(self):
        """Generate all monthly reports."""
        print("\n[Monthly Reports]")
        for year_month in sorted(self.monthly_profiles.keys()):
            self.generate_monthly_report(year_month)

    def generate_all_daily(self):
        """Generate all daily reports and summaries."""
        print("\n[Daily Reports]")
        for year_month in sorted(self.monthly_profiles.keys()):
            self.generate_daily_reports_for_month(year_month)

    def generate_monthly_report(self, year_month: str):
        """Generate a single monthly report."""
        if year_month not in self.monthly_profiles:
            print(f"  Warning: No data for {year_month}")
            return

        year, month = year_month.split("-")
        month_name = self.MONTH_NAMES.get(month, month)
        filename = f"{year_month}_{month_name}.md"
        filepath = self.output_dir / "monthly" / filename

        content = self.md_generator.generate_monthly_report(year_month)
        self._write_report(filepath, content)

    def generate_daily_reports_for_month(self, year_month: str):
        """Generate all daily reports for a specific month."""
        # Get daily data for this month
        daily_dates = [
            date for date in self.daily_profiles.keys()
            if date.startswith(year_month)
        ]

        if not daily_dates:
            print(f"  No daily data for {year_month}")
            return

        print(f"\n  {year_month}: {len(daily_dates)} days")

        # Generate individual daily reports
        for date in sorted(daily_dates):
            filename = f"{date}.md"
            filepath = self.output_dir / "daily" / year_month / filename
            content = self.md_generator.generate_daily_report(date)
            self._write_report(filepath, content)

        # Generate monthly daily summary
        summary_filepath = self.output_dir / "daily" / year_month / "summary.md"
        summary_content = self.md_generator.generate_daily_summary(year_month)
        self._write_report(summary_filepath, summary_content)

    def generate_index(self):
        """Generate the main index file."""
        print("\n[Index]")
        filepath = self.output_dir / "index.md"
        content = self.md_generator.generate_index()
        self._write_report(filepath, content)

    def generate_readme(self):
        """Generate the README file."""
        print("\n[README]")
        filepath = self.output_dir / "README.md"
        content = self._generate_readme_content()
        self._write_report(filepath, content)

    def _generate_readme_content(self) -> str:
        """Generate README content."""
        return f"""# GBPUSD Market Analysis Reports

This directory contains comprehensive market analysis reports for GBPUSD covering the period from January 2025 to January 2026.

## Overview

- **Period:** January 2025 - January 2026 (13 months)
- **Primary Timeframe:** H1
- **Supporting Timeframes:** D1, H4
- **Total Trading Days:** ~260

## Directory Structure

```
reports/market_analysis/
|-- README.md                  # This file
|-- index.md                   # Main navigation & summary
|
|-- monthly/                   # 13 Monthly Reports
|   |-- 2025-01_january.md
|   |-- 2025-02_february.md
|   |-- ...
|   |-- 2026-01_january.md
|
|-- daily/                     # Daily Reports by Month
    |-- 2025-01/
    |   |-- summary.md         # Monthly daily overview
    |   |-- 2025-01-02.md      # Individual daily reports
    |   |-- 2025-01-03.md
    |   |-- ...
    |-- 2025-02/
    |-- ...
    |-- 2026-01/
```

## Report Types

### Monthly Reports
Comprehensive monthly analysis including:
- Executive Summary
- Volatility Analysis (ATR, ranges, distribution)
- Trend Analysis (ADX, directional movement)
- Choppiness Analysis
- Session Analysis (Asian, London, NY)
- Day of Week Analysis
- Daily Breakdown Table
- Risk Recommendations
- Month-over-Month Comparison

### Daily Reports
Individual day breakdown including:
- Quick Summary (Quality, Tradeable, Trend)
- Price Action (OHLC, Range)
- Technical Indicators (ATR, ADX, Choppiness, Efficiency)
- Session Analysis
- Trading Recommendations

### Daily Summaries
Overview of all trading days in a month with links to individual reports.

## Quick Start

1. Start with [index.md](./index.md) for an overview of all months
2. Click on a month to view detailed monthly analysis
3. Navigate to daily/ folders for day-by-day breakdown

## Regenerating Reports

To regenerate reports from source data:

```bash
# Generate all reports
python scripts/generate_market_reports.py --all

# Generate specific month
python scripts/generate_market_reports.py --month 2025-01

# Update current month only
python scripts/generate_market_reports.py --update

# Regenerate index only
python scripts/generate_market_reports.py --index
```

## Data Sources

Reports are generated from:
- `backtest/market_analysis/monthly_profiles.json`
- `backtest/market_analysis/daily_profiles.json`

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*
"""

    def _print_summary(self):
        """Print generation summary."""
        monthly_count = len(self.monthly_profiles)

        daily_count = len(self.daily_profiles)

        monthly_dir = self.output_dir / "monthly"
        daily_dir = self.output_dir / "daily"

        monthly_files = list(monthly_dir.glob("*.md")) if monthly_dir.exists() else []
        daily_files = []
        if daily_dir.exists():
            for month_dir in daily_dir.iterdir():
                if month_dir.is_dir():
                    daily_files.extend(list(month_dir.glob("*.md")))

        print(f"""
Summary:
--------
Monthly Profiles: {monthly_count}
Daily Profiles:   {daily_count}
Monthly Reports:  {len(monthly_files)}
Daily Reports:    {len(daily_files)}
Output Directory: {self.output_dir.relative_to(self.project_root)}
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate GBPUSD Market Analysis Reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_market_reports.py --all           Generate all reports
  python generate_market_reports.py --month 2025-01 Generate specific month
  python generate_market_reports.py --update        Update current month
  python generate_market_reports.py --index         Regenerate index only
        """
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Generate all reports (monthly + daily + index)'
    )

    parser.add_argument(
        '--month', '-m',
        type=str,
        metavar='YYYY-MM',
        help='Generate reports for specific month (e.g., 2025-01)'
    )

    parser.add_argument(
        '--update', '-u',
        action='store_true',
        help='Update current month only'
    )

    parser.add_argument(
        '--index', '-i',
        action='store_true',
        help='Regenerate index and README only'
    )

    parser.add_argument(
        '--daily-only',
        type=str,
        metavar='YYYY-MM',
        help='Generate only daily reports for specific month'
    )

    parser.add_argument(
        '--monthly-only',
        action='store_true',
        help='Generate only monthly reports (no daily)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = MarketReportGenerator()

    # Handle arguments
    if args.all:
        generator.generate_all()

    elif args.month:
        print(f"Generating reports for {args.month}...")
        generator.generate_monthly_report(args.month)
        generator.generate_daily_reports_for_month(args.month)
        generator.generate_index()

    elif args.update:
        current_month = datetime.now().strftime('%Y-%m')
        print(f"Updating current month: {current_month}")
        generator.generate_monthly_report(current_month)
        generator.generate_daily_reports_for_month(current_month)
        generator.generate_index()

    elif args.index:
        generator.generate_index()
        generator.generate_readme()

    elif args.daily_only:
        generator.generate_daily_reports_for_month(args.daily_only)

    elif args.monthly_only:
        generator.generate_all_monthly()
        generator.generate_index()
        generator.generate_readme()

    else:
        # Default: generate all
        generator.generate_all()


if __name__ == "__main__":
    main()
