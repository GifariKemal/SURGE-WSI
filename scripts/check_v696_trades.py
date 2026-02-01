"""Check v6.96 backtest trades"""
import re

log_path = r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260201.log'
with open(log_path, 'r', encoding='utf-16') as f:
    content = f.read()

# Find the v6.96 run section
v696_start = content.rfind('=== GBPUSD H1 QuadLayer v6.991 (Full Sync) initialized ===')
if v696_start == -1:
    print("ERROR: Could not find v6.96 run")
    exit(1)

v696_content = content[v696_start:]

# Find all trade entries
entry_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(BUY|SELL) executed: (\w+)'
exit_pattern = r'Trade closed: (BUY|SELL) \| P&L: \$([+-]?[\d.]+)'

entries = re.findall(entry_pattern, v696_content)
exits = re.findall(exit_pattern, v696_content)

print(f'MQL5 v6.96 Trades: {len(entries)} entries, {len(exits)} exits')
print('='*60)
print('First 15 trades:')
print('-'*60)
for i, (entry, exit) in enumerate(zip(entries[:15], exits[:15])):
    pnl = float(exit[1])
    result = 'WIN' if pnl > 0 else 'LOSS'
    print(f"{i+1:3}: {entry[0]} {entry[1]:4} {entry[2]:20} => {result} ${pnl:>8.2f}")

# Summary
wins = sum(1 for e in exits if float(e[1]) > 0)
losses = len(exits) - wins
total_pnl = sum(float(e[1]) for e in exits)
print('-'*60)
print(f'Total: {len(entries)} trades, {wins} wins, {losses} losses')
print(f'Win Rate: {wins/len(entries)*100:.1f}%')
print(f'Net P/L: ${total_pnl:,.2f}')
