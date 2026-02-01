"""Analyze when MQL5 Layer 4 halts occur"""
import re

log_path = r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260201.log'
with open(log_path, 'r', encoding='utf-16') as f:
    content = f.read()

v_start = content.rfind('=== GBPUSD H1 QuadLayer v6.991 (Full Sync) initialized ===')
if v_start == -1:
    v_start = content.rfind('=== GBPUSD H1 QuadLayer v6.992')
if v_start == -1:
    print("Version not found")
    exit()

v_content = content[v_start:]

# Count trades, wins, losses, and halts with timestamps
trade_count = 0
win_count = 0
loss_count = 0
halt_count = 0
first_halt_trade = None

for line in v_content.split('\n'):
    if 'executed' in line and ('BUY' in line or 'SELL' in line):
        trade_count += 1
    elif 'Trade closed' in line:
        if 'P&L: $-' in line or 'P&L: $0' in line:
            loss_count += 1
        else:
            win_count += 1
    elif 'LAYER4: HALT' in line:
        halt_count += 1
        if first_halt_trade is None:
            first_halt_trade = trade_count

print(f"Total trades: {trade_count}")
print(f"Wins: {win_count}")
print(f"Losses: {loss_count}")
print(f"Total halts: {halt_count}")
print(f"First halt after trade #: {first_halt_trade}")

# Calculate running win rate
print("\nRunning win rate after each trade:")
trade_idx = 0
wins = 0
losses = 0
for line in v_content.split('\n'):
    if 'Trade closed' in line:
        trade_idx += 1
        if 'P&L: $-' in line or 'P&L: $0' in line:
            losses += 1
        else:
            wins += 1

        if trade_idx <= 20 or trade_idx % 10 == 0:
            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            rolling_wr = 0
            if total >= 10:
                # Calculate last 10 trades WR
                # This is approximate since we don't have the actual history
                pass
            print(f"  Trade {trade_idx}: {wins}W/{losses}L = {wr:.1f}% overall")
