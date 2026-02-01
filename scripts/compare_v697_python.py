"""Compare Python vs MQL5 v6.97 trades"""
import re
import pandas as pd

# Read MQL5 log
log_path = r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260201.log'
with open(log_path, 'r', encoding='utf-16') as f:
    content = f.read()

v697_start = content.rfind('=== GBPUSD H1 QuadLayer v6.97 (Full Sync) initialized ===')
v697_content = content[v697_start:]

entry_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(BUY|SELL) executed: (\w+)'
exit_pattern = r'Trade closed: (BUY|SELL) \| P&L: \$([+-]?[\d.]+)'

entries = re.findall(entry_pattern, v697_content)
exits = re.findall(exit_pattern, v697_content)

mql5_trades = []
for i, (entry, exit) in enumerate(zip(entries, exits)):
    pnl = float(exit[1])
    mql5_trades.append({
        'entry_time': entry[0],
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
print('COMPARISON: Python vs MQL5 v6.97')
print('='*80)
print(f'Python:  {len(py_df)} trades, {py_wins} wins, {py_wins/len(py_df)*100:.1f}% WR, ${py_df["pnl"].sum():,.2f}')
print(f'MQL5:    {len(mql5_df)} trades, {mql5_wins} wins, {mql5_wins/len(mql5_df)*100:.1f}% WR, ${mql5_df["pnl"].sum():,.2f}')
print()
print('First 10 Python trades:')
print('-'*50)
for i in range(10):
    row = py_df.iloc[i]
    print(f"{i+1:2}: {row['entry_time_str']} {row['direction']:4} => {row['result']}")

print()
print('First 10 MQL5 trades:')
print('-'*50)
for i in range(min(10, len(mql5_df))):
    row = mql5_df.iloc[i]
    print(f"{i+1:2}: {row['entry_time']} {row['direction']:4} => {row['result']}")

# Find matching trades by entry time
print()
print('='*80)
print('MATCHING TRADES (same entry time)')
print('='*80)
matches = 0
mismatches = 0
for _, py_row in py_df.iterrows():
    mql_match = mql5_df[mql5_df['entry_time'] == py_row['entry_time_str']]
    if len(mql_match) > 0:
        matches += 1
        if py_row['result'] != mql_match.iloc[0]['result']:
            mismatches += 1
            print(f"{py_row['entry_time_str']} | Py: {py_row['result']} ${py_row['pnl']:>8.2f} | MQL: {mql_match.iloc[0]['result']} ${mql_match.iloc[0]['pnl']:>8.2f}")

print()
print(f'Trades with same entry time: {matches}/{len(py_df)}')
print(f'Result mismatches: {mismatches}')
