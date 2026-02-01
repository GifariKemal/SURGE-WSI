"""
Debug script to compare Python backtest signals vs expected MQL5 signals.
This helps identify why MQL5 EA results differ from Python.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import MetaTrader5 as mt5

# Constants matching MQL5 EA
SYMBOL = 'GBPUSD'
GMT_OFFSET = 5  # Finex broker GMT offset

# MQL5 Hour Multipliers (UTC hours)
HOUR_MULTIPLIERS = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 0-5
    0.5, 0.0, 1.0, 1.0, 0.9, 0.0,  # 6-11
    0.7, 1.0, 1.0, 1.0, 0.9, 0.7,  # 12-17
    0.3, 0.0, 0.0, 0.0, 0.0, 0.0   # 18-23
]

# MQL5 Day Multipliers (0=Sun, 1=Mon, ..., 6=Sat)
DAY_MULTIPLIERS = [0.0, 1.0, 0.9, 1.0, 0.8, 0.3, 0.0]

# Kill zones (UTC)
LONDON_START, LONDON_END = 8, 10
NY_START, NY_END = 13, 17

# Monthly Tradeable Percentages
MONTHLY_PCT = [65, 55, 70, 70, 62, 68, 78, 65, 72, 58, 66, 60]

def get_monthly_quality_adj(month):
    """Match MQL5 GetMonthlyQualityAdjustment"""
    pct = MONTHLY_PCT[month - 1]
    if pct < 30: return 50
    elif pct < 40: return 35
    elif pct < 50: return 25
    elif pct < 60: return 15
    elif pct < 70: return 10
    elif pct < 75: return 5
    return 0

def server_to_utc(server_hour, gmt_offset):
    """Convert server hour to UTC"""
    utc = server_hour - gmt_offset
    if utc < 0: utc += 24
    if utc >= 24: utc -= 24
    return utc

def analyze_signals():
    """Analyze signals for 2025 and compare Python vs MQL5 logic"""

    if not mt5.initialize():
        print("MT5 init failed")
        return

    # Fetch data
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, start, end)
    if rates is None:
        print("No data")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    print(f"Loaded {len(df)} bars")
    print(f"\nAnalyzing with Finex GMT+{GMT_OFFSET} offset...")
    print("="*70)

    # Track statistics
    stats = {
        'total_bars': 0,
        'in_killzone': 0,
        'hour_blocked': 0,
        'day_blocked': 0,
        'valid_trading_hours': 0,
    }

    hour_distribution = {h: 0 for h in range(24)}
    utc_hour_distribution = {h: 0 for h in range(24)}

    for i in range(100, len(df)):  # Skip warmup
        row = df.iloc[i]
        t = row['time']

        stats['total_bars'] += 1

        # Server time (what MT5 sees)
        server_hour = t.hour
        server_day = t.weekday()  # Python: 0=Mon, 6=Sun

        # Convert to MQL5 day_of_week (0=Sun, 1=Mon, ..., 6=Sat)
        mql5_day = (server_day + 1) % 7

        # Convert to UTC
        utc_hour = server_to_utc(server_hour, GMT_OFFSET)

        hour_distribution[server_hour] += 1
        utc_hour_distribution[utc_hour] += 1

        # Check day multiplier
        day_mult = DAY_MULTIPLIERS[mql5_day]
        if day_mult <= 0:
            stats['day_blocked'] += 1
            continue

        # Check hour multiplier (using UTC)
        hour_mult = HOUR_MULTIPLIERS[utc_hour]
        if hour_mult <= 0:
            stats['hour_blocked'] += 1
            continue

        # Check kill zone (using UTC)
        in_london = LONDON_START <= utc_hour <= LONDON_END
        in_ny = NY_START <= utc_hour <= NY_END

        if in_london or in_ny:
            stats['in_killzone'] += 1
            stats['valid_trading_hours'] += 1

    print("\n[FILTER STATISTICS]")
    print("-"*50)
    print(f"Total bars analyzed: {stats['total_bars']}")
    print(f"Day blocked (weekend): {stats['day_blocked']}")
    print(f"Hour blocked (multiplier=0): {stats['hour_blocked']}")
    print(f"In kill zone: {stats['in_killzone']}")
    print(f"Valid trading hours: {stats['valid_trading_hours']}")

    print("\n[SERVER HOUR DISTRIBUTION] (Finex time)")
    print("-"*50)
    for h in range(24):
        bar = '#' * (hour_distribution[h] // 50)
        print(f"Hour {h:02d}: {hour_distribution[h]:4d} {bar}")

    print("\n[UTC HOUR DISTRIBUTION] (after GMT conversion)")
    print("-"*50)
    for h in range(24):
        bar = '#' * (utc_hour_distribution[h] // 50)
        kz = ""
        if LONDON_START <= h <= LONDON_END:
            kz = " [LONDON]"
        elif NY_START <= h <= NY_END:
            kz = " [NY]"
        mult = HOUR_MULTIPLIERS[h]
        print(f"Hour {h:02d}: {utc_hour_distribution[h]:4d} mult={mult:.1f}{kz} {bar}")

    print("\n[KILL ZONE MAPPING]")
    print("-"*50)
    print(f"London (UTC 8-10)  → Finex time {8+GMT_OFFSET}-{10+GMT_OFFSET}")
    print(f"New York (UTC 13-17) → Finex time {13+GMT_OFFSET}-{17+GMT_OFFSET}")

    print("\n[POTENTIAL ISSUES]")
    print("-"*50)

    # Check if kill zones are reasonable
    finex_london_start = 8 + GMT_OFFSET
    finex_ny_end = 17 + GMT_OFFSET

    if finex_ny_end > 23:
        print(f"WARNING:  NY session ends at {finex_ny_end}:00 Finex time (wraps to next day!)")
        print("   This might cause issues with daily bar transitions.")

    # Check data timezone
    first_bar = df.iloc[0]['time']
    print(f"\nFirst bar timestamp: {first_bar}")
    print(f"If this is UTC, GMT offset should be 0, not {GMT_OFFSET}")
    print(f"If this is Finex server time, GMT offset {GMT_OFFSET} is correct")

    mt5.shutdown()

    return stats

if __name__ == '__main__':
    analyze_signals()
