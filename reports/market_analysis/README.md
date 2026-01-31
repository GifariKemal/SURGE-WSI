# GBPUSD Market Analysis Reports

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

*Generated: 2026-01-30 11:21 UTC*
