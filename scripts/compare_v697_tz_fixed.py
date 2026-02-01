"""Compare Python vs MQL5 v6.97 trades with timezone adjustment"""
import re
import pandas as pd
from datetime import datetime, timedelta

# MQL5 server is GMT+1, Python is UTC
# So MQL5 11:00 = Python 10:00
MQL5_OFFSET_HOURS = 1  # Subtract this from MQL5 time to get UTC

# Read MQL5 log
log_path = r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260201.log'
with open(log_path, 'r', encoding='utf-16') as f:
    content = f.read()

# Try v6.991 first, fallback to v6.99
v697_start = content.rfind('=== GBPUSD H1 QuadLayer v6.991 (Full Sync) initialized ===')
if v697_start == -1:
    v697_start = content.rfind('=== GBPUSD H1 QuadLayer v6.99 (Full Sync) initialized ===')
v697_content = content[v697_start:]

entry_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(BUY|SELL) executed: (\w+)'
exit_pattern = r'Trade closed: (BUY|SELL) \| P&L: \$([+-]?[\d.]+)'

entries = re.findall(entry_pattern, v697_content)
exits = re.findall(exit_pattern, v697_content)

mql5_trades = []
for i, (entry, exit) in enumerate(zip(entries, exits)):
    pnl = float(exit[1])
    # Convert MQL5 server time to UTC
    dt = datetime.strptime(entry[0], '%Y.%m.%d %H:%M:%S')
    dt_utc = dt - timedelta(hours=MQL5_OFFSET_HOURS)
    mql5_trades.append({
        'entry_time': entry[0],
        'entry_time_utc': dt_utc.strftime('%Y.%m.%d %H:%M:%S'),
        'direction': entry[1],
        'pnl': pnl,
        'result': 'WIN' if pnl > 0 else 'LOSS'
    })

mql5_df = pd.DataFrame(mql5_trades)
mql5_wins = len(mql5_df[mql5_df['pnl'] > 0])

# Load Python trades
py_df = pd.read_csv('backtest/results/h1_v6_9_quad_filter_trades.csv')
py_df['entry_time_str'] = pd.to_datetime(py_df['entry_time']).dt.strftime('%Y.%m.%d %H:%M:%S')
py_df['result'] = py_df['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')
py_wins = len(py_df[py_df['pnl'] > 0])

print('='*80)
print('COMPARISON: Python vs MQL5 v6.97 (Timezone Adjusted)')
print('MQL5 server time adjusted by -1 hour to match Python UTC')
print('='*80)
print(f'Python:  {len(py_df)} trades, {py_wins} wins, {py_wins/len(py_df)*100:.1f}% WR, ${py_df["pnl"].sum():,.2f}')
print(f'MQL5:    {len(mql5_df)} trades, {mql5_wins} wins, {mql5_wins/len(mql5_df)*100:.1f}% WR, ${mql5_df["pnl"].sum():,.2f}')

# Find matching trades by entry time (UTC adjusted)
print()
print('='*80)
print('MATCHING TRADES (same UTC time)')
print('='*80)
matches = 0
result_matches = 0
result_mismatches = []

for _, py_row in py_df.iterrows():
    mql_match = mql5_df[mql5_df['entry_time_utc'] == py_row['entry_time_str']]
    if len(mql_match) > 0:
        matches += 1
        mql_row = mql_match.iloc[0]
        if py_row['result'] == mql_row['result']:
            result_matches += 1
        else:
            result_mismatches.append({
                'time': py_row['entry_time_str'],
                'direction': py_row['direction'],
                'py_result': py_row['result'],
                'py_pnl': py_row['pnl'],
                'mql_result': mql_row['result'],
                'mql_pnl': mql_row['pnl']
            })

print(f'Trades with same UTC time: {matches}/{len(py_df)} ({matches/len(py_df)*100:.1f}%)')
print(f'Result matches: {result_matches}')
print(f'Result mismatches: {len(result_mismatches)}')

if result_mismatches:
    print()
    print('Trades with different results:')
    print('-'*80)
    for d in result_mismatches[:20]:
        print(f"{d['time']} {d['direction']:4} | Py: {d['py_result']:4} ${d['py_pnl']:>9.2f} | MQL: {d['mql_result']:4} ${d['mql_pnl']:>8.2f}")

# Find trades in Python but not in MQL5
print()
print('='*80)
print('TRADES IN PYTHON BUT NOT IN MQL5')
print('='*80)
py_only = []
for _, py_row in py_df.iterrows():
    mql_match = mql5_df[mql5_df['entry_time_utc'] == py_row['entry_time_str']]
    if len(mql_match) == 0:
        py_only.append(py_row)

print(f'Python-only trades: {len(py_only)}')
for row in py_only[:15]:
    print(f"{row['entry_time_str']} {row['direction']:4} | {row['result']:4} ${row['pnl']:>9.2f}")
if len(py_only) > 15:
    print(f'... and {len(py_only)-15} more')

# Find trades in MQL5 but not in Python
print()
print('='*80)
print('TRADES IN MQL5 BUT NOT IN PYTHON')
print('='*80)
mql_only = []
for _, mql_row in mql5_df.iterrows():
    py_match = py_df[py_df['entry_time_str'] == mql_row['entry_time_utc']]
    if len(py_match) == 0:
        mql_only.append(mql_row)

print(f'MQL5-only trades: {len(mql_only)}')
for row in mql_only[:15]:
    print(f"{row['entry_time_utc']} {row['direction']:4} | {row['result']:4} ${row['pnl']:>8.2f}")
if len(mql_only) > 15:
    print(f'... and {len(mql_only)-15} more')
