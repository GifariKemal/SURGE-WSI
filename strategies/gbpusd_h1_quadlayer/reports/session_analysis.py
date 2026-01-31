"""
GBPUSD H1 QuadLayer Strategy - Session-Specific Performance Analysis
=====================================================================
Analyzes trading performance by hour, session, day of week, and POI type.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load trades data
df = pd.read_csv(r'C:\Users\Administrator\Music\SURGE-WSI\strategies\gbpusd_h1_quadlayer\reports\quadlayer_trades.csv')

# Convert entry_time to datetime and extract components
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['hour'] = df['entry_time'].dt.hour
df['day_of_week'] = df['entry_time'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['entry_time'].dt.day_name()
df['is_win'] = df['pnl'] > 0

# Define sessions based on hour
def get_session_type(hour):
    if 8 <= hour <= 11:
        return 'London (8-11)'
    elif 13 <= hour <= 14:
        return 'NY_Overlap (13-14)'
    elif 15 <= hour <= 17:
        return 'NY (15-17)'
    else:
        return 'Other'

df['session_type'] = df['hour'].apply(get_session_type)

print("="*80)
print("GBPUSD H1 QUADLAYER STRATEGY - SESSION PERFORMANCE ANALYSIS")
print("="*80)
print(f"\nTotal Trades: {len(df)}")
print(f"Date Range: {df['entry_time'].min().strftime('%Y-%m-%d')} to {df['entry_time'].max().strftime('%Y-%m-%d')}")
print(f"Overall Win Rate: {df['is_win'].mean()*100:.1f}%")
print(f"Total PnL: ${df['pnl'].sum():,.2f}")

# =============================================================================
# 1. HOURLY ANALYSIS (8-17 UTC)
# =============================================================================
print("\n" + "="*80)
print("1. HOURLY PERFORMANCE ANALYSIS (8-17 UTC)")
print("="*80)

hourly_stats = []
for hour in range(8, 18):
    hour_df = df[df['hour'] == hour]
    if len(hour_df) > 0:
        wins = hour_df['is_win'].sum()
        losses = len(hour_df) - wins
        win_rate = hour_df['is_win'].mean() * 100
        total_pnl = hour_df['pnl'].sum()
        avg_win = hour_df[hour_df['is_win']]['pnl'].mean() if wins > 0 else 0
        avg_loss = hour_df[~hour_df['is_win']]['pnl'].mean() if losses > 0 else 0

        hourly_stats.append({
            'Hour': f"{hour:02d}:00",
            'Trades': len(hour_df),
            'Wins': wins,
            'Losses': losses,
            'Win Rate': f"{win_rate:.1f}%",
            'Total PnL': total_pnl,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Expectancy': (avg_win * (win_rate/100)) + (avg_loss * (1 - win_rate/100))
        })

hourly_df = pd.DataFrame(hourly_stats)
print("\n" + hourly_df.to_string(index=False))

# Identify worst hours
print("\n>>> WORST PERFORMING HOURS (by Win Rate):")
worst_hours = sorted(hourly_stats, key=lambda x: float(x['Win Rate'].replace('%', '')))[:3]
for h in worst_hours:
    print(f"    Hour {h['Hour']}: {h['Win Rate']} win rate, ${h['Total PnL']:.2f} PnL ({h['Trades']} trades)")

print("\n>>> BEST PERFORMING HOURS (by Win Rate):")
best_hours = sorted(hourly_stats, key=lambda x: float(x['Win Rate'].replace('%', '')), reverse=True)[:3]
for h in best_hours:
    print(f"    Hour {h['Hour']}: {h['Win Rate']} win rate, ${h['Total PnL']:.2f} PnL ({h['Trades']} trades)")

# =============================================================================
# 2. SESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("2. SESSION PERFORMANCE ANALYSIS")
print("="*80)

session_order = ['London (8-11)', 'NY_Overlap (13-14)', 'NY (15-17)']
session_stats = []

for session in session_order:
    sess_df = df[df['session_type'] == session]
    if len(sess_df) > 0:
        wins = sess_df['is_win'].sum()
        losses = len(sess_df) - wins
        win_rate = sess_df['is_win'].mean() * 100
        total_pnl = sess_df['pnl'].sum()
        avg_win = sess_df[sess_df['is_win']]['pnl'].mean() if wins > 0 else 0
        avg_loss = sess_df[~sess_df['is_win']]['pnl'].mean() if losses > 0 else 0
        profit_factor = abs(sess_df[sess_df['is_win']]['pnl'].sum() / sess_df[~sess_df['is_win']]['pnl'].sum()) if losses > 0 else float('inf')

        session_stats.append({
            'Session': session,
            'Trades': len(sess_df),
            'Wins': wins,
            'Losses': losses,
            'Win Rate': f"{win_rate:.1f}%",
            'Total PnL': f"${total_pnl:,.2f}",
            'Avg Win': f"${avg_win:.2f}",
            'Avg Loss': f"${avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.2f}"
        })

session_df = pd.DataFrame(session_stats)
print("\n" + session_df.to_string(index=False))

# =============================================================================
# 3. DAY OF WEEK ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("3. DAY OF WEEK ANALYSIS")
print("="*80)

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
day_stats = []

for i, day_name in enumerate(day_names):
    day_df = df[df['day_of_week'] == i]
    if len(day_df) > 0:
        wins = day_df['is_win'].sum()
        losses = len(day_df) - wins
        win_rate = day_df['is_win'].mean() * 100
        total_pnl = day_df['pnl'].sum()
        avg_pnl = day_df['pnl'].mean()

        day_stats.append({
            'Day': day_name,
            'Trades': len(day_df),
            'Wins': wins,
            'Losses': losses,
            'Win Rate': f"{win_rate:.1f}%",
            'Total PnL': f"${total_pnl:,.2f}",
            'Avg PnL': f"${avg_pnl:.2f}"
        })

day_df = pd.DataFrame(day_stats)
print("\n" + day_df.to_string(index=False))

# =============================================================================
# 4. POI TYPE BY SESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("4. POI TYPE BY SESSION ANALYSIS")
print("="*80)

poi_types = df['poi_type'].unique()

for poi in poi_types:
    print(f"\n--- {poi} ---")
    poi_df = df[df['poi_type'] == poi]

    poi_session_stats = []
    for session in session_order:
        sess_df = poi_df[poi_df['session_type'] == session]
        if len(sess_df) > 0:
            wins = sess_df['is_win'].sum()
            win_rate = sess_df['is_win'].mean() * 100
            total_pnl = sess_df['pnl'].sum()

            poi_session_stats.append({
                'Session': session,
                'Trades': len(sess_df),
                'Wins': wins,
                'Win Rate': f"{win_rate:.1f}%",
                'Total PnL': f"${total_pnl:,.2f}"
            })

    if poi_session_stats:
        poi_session_df = pd.DataFrame(poi_session_stats)
        print(poi_session_df.to_string(index=False))

# =============================================================================
# 5. DETAILED HOUR + POI TYPE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("5. DETAILED HOUR + POI TYPE CROSS-ANALYSIS")
print("="*80)

print("\n--- ORDER_BLOCK Performance by Hour ---")
ob_df = df[df['poi_type'] == 'ORDER_BLOCK']
ob_hourly = []
for hour in range(8, 18):
    hour_df = ob_df[ob_df['hour'] == hour]
    if len(hour_df) >= 2:  # At least 2 trades
        win_rate = hour_df['is_win'].mean() * 100
        total_pnl = hour_df['pnl'].sum()
        ob_hourly.append({
            'Hour': f"{hour:02d}:00",
            'Trades': len(hour_df),
            'Win Rate': f"{win_rate:.1f}%",
            'PnL': f"${total_pnl:,.2f}"
        })

if ob_hourly:
    print(pd.DataFrame(ob_hourly).to_string(index=False))

print("\n--- EMA_PULLBACK Performance by Hour ---")
ema_df = df[df['poi_type'] == 'EMA_PULLBACK']
ema_hourly = []
for hour in range(8, 18):
    hour_df = ema_df[ema_df['hour'] == hour]
    if len(hour_df) >= 2:  # At least 2 trades
        win_rate = hour_df['is_win'].mean() * 100
        total_pnl = hour_df['pnl'].sum()
        ema_hourly.append({
            'Hour': f"{hour:02d}:00",
            'Trades': len(hour_df),
            'Win Rate': f"{win_rate:.1f}%",
            'PnL': f"${total_pnl:,.2f}"
        })

if ema_hourly:
    print(pd.DataFrame(ema_hourly).to_string(index=False))

# =============================================================================
# 6. UNDERPERFORMING COMBINATIONS
# =============================================================================
print("\n" + "="*80)
print("6. UNDERPERFORMING COMBINATIONS (Win Rate < 50% with 3+ trades)")
print("="*80)

# Hour + POI combinations
print("\n--- Underperforming Hour + POI Type Combinations ---")
underperformers = []
for hour in range(8, 18):
    for poi in ['ORDER_BLOCK', 'EMA_PULLBACK']:
        combo_df = df[(df['hour'] == hour) & (df['poi_type'] == poi)]
        if len(combo_df) >= 3:
            win_rate = combo_df['is_win'].mean() * 100
            total_pnl = combo_df['pnl'].sum()
            if win_rate < 50:
                underperformers.append({
                    'Hour': f"{hour:02d}:00",
                    'POI Type': poi,
                    'Trades': len(combo_df),
                    'Win Rate': f"{win_rate:.1f}%",
                    'PnL': f"${total_pnl:,.2f}"
                })

if underperformers:
    underperf_df = pd.DataFrame(underperformers)
    underperf_df = underperf_df.sort_values(by='Win Rate')
    print(underperf_df.to_string(index=False))
else:
    print("No severely underperforming combinations found.")

# Day + Session combinations
print("\n--- Underperforming Day + Session Combinations ---")
day_session_underperf = []
for i, day_name in enumerate(day_names):
    for session in session_order:
        combo_df = df[(df['day_of_week'] == i) & (df['session_type'] == session)]
        if len(combo_df) >= 3:
            win_rate = combo_df['is_win'].mean() * 100
            total_pnl = combo_df['pnl'].sum()
            if win_rate < 50:
                day_session_underperf.append({
                    'Day': day_name,
                    'Session': session,
                    'Trades': len(combo_df),
                    'Win Rate': f"{win_rate:.1f}%",
                    'PnL': f"${total_pnl:,.2f}"
                })

if day_session_underperf:
    ds_df = pd.DataFrame(day_session_underperf)
    ds_df = ds_df.sort_values(by='Win Rate')
    print(ds_df.to_string(index=False))

# =============================================================================
# 7. RECOMMENDATIONS
# =============================================================================
print("\n" + "="*80)
print("7. OPTIMIZATION RECOMMENDATIONS")
print("="*80)

print("\n" + "-"*60)
print("A. HOURS TO CONSIDER SKIPPING (like hour 7 was skipped)")
print("-"*60)

# Analyze each hour's performance
skip_candidates = []
for hour in range(8, 18):
    hour_df = df[df['hour'] == hour]
    if len(hour_df) >= 4:
        win_rate = hour_df['is_win'].mean() * 100
        total_pnl = hour_df['pnl'].sum()
        if win_rate < 45 or total_pnl < -200:
            skip_candidates.append({
                'Hour': hour,
                'Win Rate': win_rate,
                'PnL': total_pnl,
                'Trades': len(hour_df)
            })

if skip_candidates:
    for sc in sorted(skip_candidates, key=lambda x: x['Win Rate']):
        print(f"  CONSIDER SKIP: Hour {sc['Hour']:02d}:00 - {sc['Win Rate']:.1f}% win rate, ${sc['PnL']:.2f} PnL ({sc['Trades']} trades)")
else:
    print("  No hours with consistently poor performance requiring skip.")

print("\n" + "-"*60)
print("B. BEST SESSIONS FOR EACH ENTRY TYPE")
print("-"*60)

for poi in ['ORDER_BLOCK', 'EMA_PULLBACK']:
    print(f"\n  {poi}:")
    poi_df = df[df['poi_type'] == poi]

    best_session = None
    best_win_rate = 0
    best_pnl = float('-inf')

    for session in session_order:
        sess_df = poi_df[poi_df['session_type'] == session]
        if len(sess_df) >= 5:
            win_rate = sess_df['is_win'].mean() * 100
            total_pnl = sess_df['pnl'].sum()
            print(f"    - {session}: {win_rate:.1f}% win rate, ${total_pnl:,.2f} PnL ({len(sess_df)} trades)")
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_session = session
                best_win_rate = win_rate

    if best_session:
        print(f"    >>> BEST: {best_session}")

print("\n" + "-"*60)
print("C. DAY-SPECIFIC RECOMMENDATIONS")
print("-"*60)

for i, day_name in enumerate(day_names):
    day_df = df[df['day_of_week'] == i]
    if len(day_df) >= 5:
        win_rate = day_df['is_win'].mean() * 100
        total_pnl = day_df['pnl'].sum()
        avg_pnl = day_df['pnl'].mean()

        if win_rate < 45:
            print(f"  WARNING: {day_name} - {win_rate:.1f}% win rate (consider reduced position size)")
        elif win_rate >= 55:
            print(f"  STRONG: {day_name} - {win_rate:.1f}% win rate (consider normal/increased position)")
        else:
            print(f"  NEUTRAL: {day_name} - {win_rate:.1f}% win rate")

print("\n" + "-"*60)
print("D. SPECIFIC HOUR + POI RECOMMENDATIONS")
print("-"*60)

# Find best combinations
best_combos = []
for hour in range(8, 18):
    for poi in ['ORDER_BLOCK', 'EMA_PULLBACK']:
        combo_df = df[(df['hour'] == hour) & (df['poi_type'] == poi)]
        if len(combo_df) >= 4:
            win_rate = combo_df['is_win'].mean() * 100
            total_pnl = combo_df['pnl'].sum()
            best_combos.append({
                'Hour': hour,
                'POI': poi,
                'Win Rate': win_rate,
                'PnL': total_pnl,
                'Trades': len(combo_df)
            })

# Top performers
print("\n  TOP PERFORMING COMBINATIONS (Win Rate >= 55%, 4+ trades):")
for combo in sorted(best_combos, key=lambda x: x['PnL'], reverse=True)[:5]:
    if combo['Win Rate'] >= 55:
        print(f"    Hour {combo['Hour']:02d}:00 + {combo['POI']}: {combo['Win Rate']:.1f}% WR, ${combo['PnL']:.2f} ({combo['Trades']} trades)")

# Worst performers to avoid
print("\n  COMBINATIONS TO AVOID (Win Rate < 45%, 4+ trades):")
for combo in sorted(best_combos, key=lambda x: x['Win Rate'])[:5]:
    if combo['Win Rate'] < 45:
        print(f"    Hour {combo['Hour']:02d}:00 + {combo['POI']}: {combo['Win Rate']:.1f}% WR, ${combo['PnL']:.2f} ({combo['Trades']} trades)")

# =============================================================================
# 8. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("8. EXECUTIVE SUMMARY")
print("="*80)

# Best session
best_session_data = max(session_stats, key=lambda x: float(x['Total PnL'].replace('$', '').replace(',', '')))
print(f"\n  BEST SESSION: {best_session_data['Session']}")
print(f"    - Win Rate: {best_session_data['Win Rate']}")
print(f"    - Total PnL: {best_session_data['Total PnL']}")
print(f"    - Profit Factor: {best_session_data['Profit Factor']}")

# Best day
best_day_data = max(day_stats, key=lambda x: float(x['Total PnL'].replace('$', '').replace(',', '')))
print(f"\n  BEST DAY: {best_day_data['Day']}")
print(f"    - Win Rate: {best_day_data['Win Rate']}")
print(f"    - Total PnL: {best_day_data['Total PnL']}")

# Worst day
worst_day_data = min(day_stats, key=lambda x: float(x['Total PnL'].replace('$', '').replace(',', '')))
print(f"\n  WORST DAY: {worst_day_data['Day']}")
print(f"    - Win Rate: {worst_day_data['Win Rate']}")
print(f"    - Total PnL: {worst_day_data['Total PnL']}")

# Best hour
best_hour_data = max(hourly_stats, key=lambda x: x['Total PnL'])
print(f"\n  BEST HOUR: {best_hour_data['Hour']}")
print(f"    - Win Rate: {best_hour_data['Win Rate']}")
print(f"    - Total PnL: ${best_hour_data['Total PnL']:.2f}")

# Worst hour
worst_hour_data = min(hourly_stats, key=lambda x: x['Total PnL'])
print(f"\n  WORST HOUR: {worst_hour_data['Hour']}")
print(f"    - Win Rate: {worst_hour_data['Win Rate']}")
print(f"    - Total PnL: ${worst_hour_data['Total PnL']:.2f}")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
