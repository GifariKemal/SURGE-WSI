"""Compare Python vs MQL5 trades"""
import re
import pandas as pd

# Parse MQL5 log
with open(r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260201.log', 'r', encoding='utf-16') as f:
    content = f.read()

# Find all trade entries and exits
entry_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(BUY|SELL) executed: (\w+)'
exit_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+Trade closed: (BUY|SELL) \| P&L: \$([\-\d.]+)'

entries = re.findall(entry_pattern, content)
exits = re.findall(exit_pattern, content)

print(f'MQL5 Trades: {len(entries)} entries, {len(exits)} exits')

mql5_trades = []
for entry, exit in zip(entries, exits):
    mql5_trades.append({
        'entry_time': entry[0],
        'direction': entry[1],
        'signal_type': entry[2],
        'pnl': float(exit[2]),
        'result': 'WIN' if float(exit[2]) > 0 else 'LOSS'
    })

mql5_df = pd.DataFrame(mql5_trades)
mql5_wins = len(mql5_df[mql5_df['pnl'] > 0])
mql5_losses = len(mql5_df[mql5_df['pnl'] <= 0])
print(f'MQL5 Wins: {mql5_wins}, Losses: {mql5_losses}, WR: {mql5_wins/len(mql5_df)*100:.1f}%')

# Load Python trades
py_df = pd.read_csv('backtest/results/h1_v6_9_quad_filter_trades.csv')
py_df['entry_time'] = pd.to_datetime(py_df['entry_time'])
py_df['entry_time_str'] = py_df['entry_time'].dt.strftime('%Y.%m.%d %H:%M:%S')
py_df['result'] = py_df['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')

py_wins = len(py_df[py_df['pnl'] > 0])
py_losses = len(py_df[py_df['pnl'] <= 0])
print(f'Python Wins: {py_wins}, Losses: {py_losses}, WR: {py_wins/len(py_df)*100:.1f}%')

# Compare trades
print('\n' + '='*70)
print('COMPARING TRADES - Finding differences')
print('='*70)

# Match by entry time
differences = []
for i, py_row in py_df.iterrows():
    py_time = py_row['entry_time_str']
    # Find matching MQL5 trade (same time or close)
    mql5_match = mql5_df[mql5_df['entry_time'] == py_time]

    if len(mql5_match) > 0:
        mql5_row = mql5_match.iloc[0]
        if py_row['result'] != mql5_row['result']:
            differences.append({
                'time': py_time,
                'direction': py_row['direction'],
                'signal': py_row.get('signal_type', 'N/A'),
                'py_result': py_row['result'],
                'py_pnl': py_row['pnl'],
                'mql5_result': mql5_row['result'],
                'mql5_pnl': mql5_row['pnl']
            })

print(f'\nFound {len(differences)} trades with different results:')
print('-'*70)
for d in differences[:20]:
    print(f"{d['time']} {d['direction']:4} | Python: {d['py_result']:4} ${d['py_pnl']:>8.2f} | MQL5: {d['mql5_result']:4} ${d['mql5_pnl']:>8.2f}")

if len(differences) > 20:
    print(f"... and {len(differences)-20} more")

# Summary
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
py_win_mql5_loss = sum(1 for d in differences if d['py_result'] == 'WIN' and d['mql5_result'] == 'LOSS')
py_loss_mql5_win = sum(1 for d in differences if d['py_result'] == 'LOSS' and d['mql5_result'] == 'WIN')
print(f'Python WIN but MQL5 LOSS: {py_win_mql5_loss}')
print(f'Python LOSS but MQL5 WIN: {py_loss_mql5_win}')
