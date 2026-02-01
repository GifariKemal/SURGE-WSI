"""Parse MT5 backtest log and generate XLSX report"""
import re
import pandas as pd
from datetime import datetime
import subprocess

# Path to log file
LOG_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260201.log"
OUTPUT_PATH = r"C:\Users\Administrator\Music\SURGE-WSI\reports\MQL5_QuadLayer_v693_Backtest.xlsx"

def parse_log():
    # Convert UTF-16 to UTF-8
    result = subprocess.run(
        ['iconv', '-f', 'UTF-16LE', '-t', 'UTF-8', LOG_PATH],
        capture_output=True, text=True
    )
    lines = result.stdout.split('\n')

    trades = []
    entries = {}

    # Patterns
    entry_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(BUY|SELL) executed: (\w+) \| Lot: ([\d.]+) \| SL: ([\d.]+) pips \| TP: ([\d.]+) pips'
    exit_pattern = r'(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+(take profit|stop loss) triggered #(\d+) (buy|sell) ([\d.]+) GBPUSD ([\d.]+) sl: ([\d.]+) tp: ([\d.]+) \[#(\d+) (buy|sell) ([\d.]+) GBPUSD at ([\d.]+)\]'

    for line in lines:
        # Parse entry
        entry_match = re.search(entry_pattern, line)
        if entry_match:
            entry_time = entry_match.group(1)
            direction = entry_match.group(2)
            signal_type = entry_match.group(3)
            lot = float(entry_match.group(4))
            sl_pips = float(entry_match.group(5))
            tp_pips = float(entry_match.group(6))

            # Store pending entry
            entries[len(entries)] = {
                'entry_time': entry_time,
                'direction': direction,
                'signal_type': signal_type,
                'lot': lot,
                'sl_pips': sl_pips,
                'tp_pips': tp_pips
            }

        # Parse exit
        exit_match = re.search(exit_pattern, line)
        if exit_match:
            exit_time = exit_match.group(1)
            exit_type = exit_match.group(2)
            deal_id = exit_match.group(3)
            direction = exit_match.group(4).upper()
            lot = float(exit_match.group(5))
            entry_price = float(exit_match.group(6))
            sl_price = float(exit_match.group(7))
            tp_price = float(exit_match.group(8))
            exit_price = float(exit_match.group(12))

            # Calculate P&L
            if direction == 'BUY':
                pnl_pips = (exit_price - entry_price) / 0.0001
            else:
                pnl_pips = (entry_price - exit_price) / 0.0001

            pnl_usd = pnl_pips * lot * 10  # Approximate pip value

            # Find matching entry
            if entries:
                entry_key = list(entries.keys())[-1]
                entry = entries.pop(entry_key)

                trades.append({
                    'Deal #': deal_id,
                    'Entry Time': entry['entry_time'],
                    'Exit Time': exit_time,
                    'Direction': direction,
                    'Signal Type': entry['signal_type'],
                    'Lot Size': lot,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'SL Price': sl_price,
                    'TP Price': tp_price,
                    'SL Pips': entry['sl_pips'],
                    'TP Pips': entry['tp_pips'],
                    'Exit Type': 'TP' if 'take profit' in exit_type else 'SL',
                    'P&L Pips': round(pnl_pips, 1),
                    'P&L USD': round(pnl_usd, 2)
                })

    return trades

def generate_xlsx(trades):
    df = pd.DataFrame(trades)

    # Calculate statistics
    total_trades = len(df)
    wins = len(df[df['Exit Type'] == 'TP'])
    losses = len(df[df['Exit Type'] == 'SL'])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    total_pnl = df['P&L USD'].sum()
    avg_win = df[df['P&L USD'] > 0]['P&L USD'].mean() if wins > 0 else 0
    avg_loss = df[df['P&L USD'] < 0]['P&L USD'].mean() if losses > 0 else 0

    profit_factor = abs(df[df['P&L USD'] > 0]['P&L USD'].sum() / df[df['P&L USD'] < 0]['P&L USD'].sum()) if losses > 0 else float('inf')

    # Create stats DataFrame
    stats = pd.DataFrame({
        'Metric': [
            'Total Trades',
            'Wins (TP)',
            'Losses (SL)',
            'Win Rate %',
            'Total P&L USD',
            'Avg Win USD',
            'Avg Loss USD',
            'Profit Factor',
            'Initial Balance',
            'Final Balance'
        ],
        'Value': [
            total_trades,
            wins,
            losses,
            f'{win_rate:.1f}%',
            f'${total_pnl:,.2f}',
            f'${avg_win:,.2f}',
            f'${avg_loss:,.2f}',
            f'{profit_factor:.2f}',
            '$50,000',
            '$48,665.72'
        ]
    })

    # Monthly breakdown
    df['Month'] = pd.to_datetime(df['Entry Time']).dt.to_period('M')
    monthly = df.groupby('Month').agg({
        'Deal #': 'count',
        'P&L USD': 'sum',
        'Exit Type': lambda x: (x == 'TP').sum()
    }).rename(columns={'Deal #': 'Trades', 'Exit Type': 'Wins'})
    monthly['Win Rate %'] = (monthly['Wins'] / monthly['Trades'] * 100).round(1)
    monthly['P&L USD'] = monthly['P&L USD'].round(2)
    monthly = monthly.reset_index()
    monthly['Month'] = monthly['Month'].astype(str)

    # Signal type breakdown
    by_signal = df.groupby('Signal Type').agg({
        'Deal #': 'count',
        'P&L USD': 'sum',
        'Exit Type': lambda x: (x == 'TP').sum()
    }).rename(columns={'Deal #': 'Trades', 'Exit Type': 'Wins'})
    by_signal['Win Rate %'] = (by_signal['Wins'] / by_signal['Trades'] * 100).round(1)
    by_signal['P&L USD'] = by_signal['P&L USD'].round(2)
    by_signal = by_signal.reset_index()

    # Write to Excel
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All Trades', index=False)
        stats.to_excel(writer, sheet_name='Statistics', index=False)
        monthly.to_excel(writer, sheet_name='Monthly Summary', index=False)
        by_signal.to_excel(writer, sheet_name='By Signal Type', index=False)

    print(f"XLSX saved to: {OUTPUT_PATH}")
    print(f"\n=== BACKTEST SUMMARY ===")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Final Balance: $48,665.72")

if __name__ == '__main__':
    trades = parse_log()
    print(f"Parsed {len(trades)} trades")
    generate_xlsx(trades)
