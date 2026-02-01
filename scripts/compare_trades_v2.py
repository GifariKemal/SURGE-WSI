"""Compare Python vs MQL5 trades - v6.95"""
import re
import pandas as pd
from datetime import datetime

# Read MQL5 log
log_path = r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260201.log'
with open(log_path, 'r', encoding='utf-16') as f:
    content = f.read()

# Find the v6.94 run section (which had 150 trades)
v694_start = content.rfind('=== GBPUSD H1 QuadLayer v6.94 (Full Sync) initialized ===')
v695_start = content.rfind('=== GBPUSD H1 QuadLayer v6.95 (Full Sync) initialized ===')

if v694_start == -1:
    print("ERROR: Could not find v6.94 run")
    exit(1)

# Extract content from v6.94 run only (stop at v6.95 start)
if v695_start > v694_start:
    v695_content = content[v694_start:v695_start]
else:
    v695_content = content[v694_start:]

print(f"Analyzing v6.94 run...")

# Find all trade entries and exits in v6.95 run
# Entry format: 2025.01.02 11:00:00   SELL executed: EMA_PULLBACK_MOMENTUM | Lot: 0.38 | SL: 19.38 pips | TP: 29.07 pips
# Exit format: 2025.01.02 12:33:40   Trade closed: SELL | P&L: $110.58 | Monthly: $222.38 | Consec Losses: 0

entry_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(BUY|SELL) executed: (\w+)'
exit_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+Trade closed: (BUY|SELL) \| P&L: \$([+-]?[\d.]+)'

entries = re.findall(entry_pattern, v695_content)
exits = re.findall(exit_pattern, v695_content)

print(f'MQL5 v6.95 Trades: {len(entries)} entries, {len(exits)} exits')

# Build MQL5 trades list
mql5_trades = []
for i, (entry, exit) in enumerate(zip(entries, exits)):
    pnl = float(exit[2])
    mql5_trades.append({
        'trade_num': i + 1,
        'entry_time': entry[0],
        'direction': entry[1],
        'signal_type': entry[2],
        'exit_time': exit[0],
        'pnl': pnl,
        'result': 'WIN' if pnl > 0 else 'LOSS'
    })

mql5_df = pd.DataFrame(mql5_trades)
mql5_wins = len(mql5_df[mql5_df['pnl'] > 0])
mql5_losses = len(mql5_df[mql5_df['pnl'] <= 0])
mql5_total_pnl = mql5_df['pnl'].sum()
print(f'MQL5 Wins: {mql5_wins}, Losses: {mql5_losses}, WR: {mql5_wins/len(mql5_df)*100:.1f}%')
print(f'MQL5 Total P/L: ${mql5_total_pnl:,.2f}')

# Load Python trades
py_df = pd.read_csv('backtest/results/h1_v6_9_quad_filter_trades.csv')
py_df['entry_time'] = pd.to_datetime(py_df['entry_time'])
py_df['entry_time_str'] = py_df['entry_time'].dt.strftime('%Y.%m.%d %H:%M:%S')
py_df['result'] = py_df['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')

py_wins = len(py_df[py_df['pnl'] > 0])
py_losses = len(py_df[py_df['pnl'] <= 0])
py_total_pnl = py_df['pnl'].sum()
print(f'\nPython Wins: {py_wins}, Losses: {py_losses}, WR: {py_wins/len(py_df)*100:.1f}%')
print(f'Python Total P/L: ${py_total_pnl:,.2f}')

# Compare trades
print('\n' + '='*80)
print('TRADE-BY-TRADE COMPARISON')
print('='*80)

# First, let's compare by order (trade 1, trade 2, etc.)
print('\nComparing by trade order...')
print('-'*80)
print(f"{'#':>3} | {'Entry Time':^20} | {'Dir':^4} | {'Py Result':^8} | {'Py P/L':>10} | {'MQL Result':^8} | {'MQL P/L':>10} | Match")
print('-'*80)

differences = []
for i in range(min(len(py_df), len(mql5_df))):
    py_row = py_df.iloc[i]
    mql_row = mql5_df.iloc[i]

    py_result = py_row['result']
    mql_result = mql_row['result']
    match = 'YES' if py_result == mql_result else 'NO'

    # Check if entry times match
    py_time = py_row['entry_time_str']
    mql_time = mql_row['entry_time']
    time_match = py_time == mql_time

    if py_result != mql_result or not time_match:
        differences.append({
            'trade_num': i + 1,
            'py_time': py_time,
            'mql_time': mql_time,
            'direction': py_row['direction'],
            'py_result': py_result,
            'py_pnl': py_row['pnl'],
            'mql_result': mql_result,
            'mql_pnl': mql_row['pnl'],
            'time_match': time_match
        })
        print(f"{i+1:3} | {py_time:^20} | {py_row['direction']:^4} | {py_result:^8} | ${py_row['pnl']:>9.2f} | {mql_result:^8} | ${mql_row['pnl']:>9.2f} | {match}")

print('-'*80)
print(f'Found {len(differences)} differences')

# Summary by category
print('\n' + '='*80)
print('DIFFERENCE ANALYSIS')
print('='*80)

time_mismatches = sum(1 for d in differences if not d['time_match'])
result_mismatches = sum(1 for d in differences if d['time_match'] and d['py_result'] != d['mql_result'])

print(f'\nEntry time mismatches: {time_mismatches}')
print(f'Same time, different result: {result_mismatches}')

py_win_mql_loss = sum(1 for d in differences if d['py_result'] == 'WIN' and d['mql_result'] == 'LOSS')
py_loss_mql_win = sum(1 for d in differences if d['py_result'] == 'LOSS' and d['mql_result'] == 'WIN')
print(f'\nPython WIN but MQL5 LOSS: {py_win_mql_loss}')
print(f'Python LOSS but MQL5 WIN: {py_loss_mql_win}')

# Show specific trades where they differ
if len(differences) > 0:
    print('\n' + '='*80)
    print('DETAILED DIFFERENCES (first 30)')
    print('='*80)
    for d in differences[:30]:
        time_note = '' if d['time_match'] else ' [TIME MISMATCH]'
        print(f"\nTrade #{d['trade_num']}{time_note}")
        print(f"  Python:  {d['py_time']} {d['direction']:4} => {d['py_result']:4} ${d['py_pnl']:>9.2f}")
        print(f"  MQL5:    {d['mql_time']} {d['direction']:4} => {d['mql_result']:4} ${d['mql_pnl']:>9.2f}")
