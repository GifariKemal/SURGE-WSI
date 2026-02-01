"""Check v6.991 backtest results"""
import re

log_path = r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260201.log'
with open(log_path, 'r', encoding='utf-16') as f:
    content = f.read()

# Find v6.991 section
v6991_start = content.rfind('=== GBPUSD H1 QuadLayer v6.991 (Full Sync) initialized ===')
if v6991_start == -1:
    print('v6.991 NOT FOUND! Checking for any version...')
    # Find all versions in log
    versions = re.findall(r'QuadLayer v(\d+\.\d+)', content)
    if versions:
        print(f'Versions found: {set(versions)}')
    # Check last init message
    last_init = content.rfind('initialized ===')
    if last_init != -1:
        start = max(0, last_init - 100)
        print(f'Last init: {content[start:last_init+30]}')
else:
    print('v6.991 FOUND! Checking results...')
    v6991_content = content[v6991_start:]

    # Find trades
    entry_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(BUY|SELL) executed'
    entries = re.findall(entry_pattern, v6991_content)

    # Find exits
    exit_pattern = r'Trade closed: (BUY|SELL) \| P&L: \$([+-]?[\d.]+)'
    exits = re.findall(exit_pattern, v6991_content)

    print(f'Trades: {len(entries)}')
    if len(entries) > 0:
        wins = sum(1 for e in exits if float(e[1]) > 0)
        total_pnl = sum(float(e[1]) for e in exits)
        print(f'Wins: {wins}, WR: {wins/len(entries)*100:.1f}%')
        print(f'Total PnL: ${total_pnl:,.2f}')

        print('\nFirst 10 trades:')
        for i, e in enumerate(entries[:10]):
            exit_pnl = float(exits[i][1]) if i < len(exits) else 0
            result = 'WIN' if exit_pnl > 0 else 'LOSS'
            print(f'{i+1}: {e[0]} {e[1]:4} => {result} ${exit_pnl:.2f}')
