"""
RSI v3.7 V9 Backtest Simulation
ATR-Adaptive TP/SL Strategy

Simulates V9 performance based on V5 trade data with adjusted TP/SL ratios
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# V9 ATR-Adaptive Settings
V9_SETTINGS = {
    'TP_VeryLow': 1.8,    # ATR 20-35%
    'TP_Low': 2.2,        # ATR 35-50%
    'TP_Mid': 2.6,        # ATR 50-60%
    'TP_High': 3.2,       # ATR 60-75%
    'TP_VeryHigh': 3.8,   # ATR 75-85%
    'SL_VeryLow': 1.2,
    'SL_Low': 1.4,
    'SL_Mid': 1.5,
    'SL_High': 1.5,
    'SL_VeryHigh': 1.6,
}

# V5 Original Settings (for comparison)
V5_SETTINGS = {
    'TP_Low': 2.4,
    'TP_Med': 3.0,
    'TP_High': 3.6,
    'SL': 1.5,
}

def get_v9_zone(atr_pct):
    """Get V9 ATR zone name"""
    if atr_pct < 35:
        return 'VLOW'
    elif atr_pct < 50:
        return 'LOW'
    elif atr_pct < 60:
        return 'MID'
    elif atr_pct < 75:
        return 'HIGH'
    else:
        return 'VHIGH'

def get_v9_tp_sl(atr_pct):
    """Get V9 adaptive TP/SL multipliers"""
    if atr_pct < 35:
        return V9_SETTINGS['TP_VeryLow'], V9_SETTINGS['SL_VeryLow']
    elif atr_pct < 50:
        return V9_SETTINGS['TP_Low'], V9_SETTINGS['SL_Low']
    elif atr_pct < 60:
        return V9_SETTINGS['TP_Mid'], V9_SETTINGS['SL_Mid']
    elif atr_pct < 75:
        return V9_SETTINGS['TP_High'], V9_SETTINGS['SL_High']
    else:
        return V9_SETTINGS['TP_VeryHigh'], V9_SETTINGS['SL_VeryHigh']

def get_v5_tp_sl(atr_pct):
    """Get V5 original TP/SL multipliers"""
    if atr_pct < 40:
        return V5_SETTINGS['TP_Low'], V5_SETTINGS['SL']
    elif atr_pct > 60:
        return V5_SETTINGS['TP_High'], V5_SETTINGS['SL']
    else:
        return V5_SETTINGS['TP_Med'], V5_SETTINGS['SL']

def load_v5_trades(file_path):
    """Load V5 trade data from Excel report"""
    df = pd.read_excel(file_path, header=None)

    # Find trade data start
    trade_start = None
    for i, row in df.iterrows():
        if str(row[0]).strip() == 'Time' and str(row[1]).strip() == 'Deal':
            trade_start = i
            break

    trades_df = df.iloc[trade_start+1:].copy().reset_index(drop=True)
    trades_df.columns = ['Time', 'Deal', 'Symbol', 'Type', 'Direction', 'Volume',
                         'Price', 'Order', 'Commission', 'Swap', 'Profit', 'Balance', 'Comment']
    trades_df['Time'] = pd.to_datetime(trades_df['Time'], errors='coerce')
    trades_df['Profit'] = pd.to_numeric(trades_df['Profit'], errors='coerce')
    trades_df['Price'] = pd.to_numeric(trades_df['Price'], errors='coerce')
    trades_df['Volume'] = pd.to_numeric(trades_df['Volume'], errors='coerce')

    trades_df = trades_df[trades_df['Direction'].isin(['in', 'out'])].reset_index(drop=True)

    # Extract ATR from comment
    def extract_atr(comment):
        if pd.isna(comment): return None
        comment = str(comment)
        if 'ATR' in comment:
            try:
                start = comment.find('ATR') + 3
                val = comment[start:start+2]
                return float(''.join(c for c in val if c.isdigit()))
            except: return None
        return None

    # Build complete trades
    complete_trades = []
    i = 0
    while i < len(trades_df) - 1:
        if trades_df.iloc[i]['Direction'] == 'in' and trades_df.iloc[i+1]['Direction'] == 'out':
            entry = trades_df.iloc[i]
            exit_trade = trades_df.iloc[i+1]

            atr_pct = extract_atr(entry['Comment'])

            complete_trades.append({
                'Entry_Time': entry['Time'],
                'Exit_Time': exit_trade['Time'],
                'Entry_Price': entry['Price'],
                'Exit_Price': exit_trade['Price'],
                'Type': entry['Type'],
                'Volume': entry['Volume'],
                'V5_Profit': exit_trade['Profit'],
                'ATR_Pct': atr_pct,
                'Comment': entry['Comment']
            })
            i += 2
        else:
            i += 1

    return pd.DataFrame(complete_trades)

def simulate_v9_trades(v5_trades):
    """Simulate V9 trades based on V5 entry data with adaptive TP/SL"""
    results = []

    for _, trade in v5_trades.iterrows():
        atr_pct = trade['ATR_Pct']
        if pd.isna(atr_pct):
            atr_pct = 50  # Default

        v5_tp, v5_sl = get_v5_tp_sl(atr_pct)
        v9_tp, v9_sl = get_v9_tp_sl(atr_pct)

        # Calculate ratio adjustments
        tp_ratio = v9_tp / v5_tp
        sl_ratio = v9_sl / v5_sl

        v5_profit = trade['V5_Profit']

        # Estimate V9 profit based on TP/SL ratio changes
        if v5_profit > 0:
            # Win - profit scales with TP ratio
            # But also consider that tighter TP might have hit TP earlier
            if v9_tp < v5_tp:
                # Tighter TP - higher chance of hitting TP (scale profit down but increase WR)
                v9_profit = v5_profit * tp_ratio * 1.1  # Slight boost for hitting TP more often
            else:
                # Wider TP - might miss some TPs
                v9_profit = v5_profit * tp_ratio * 0.95
        else:
            # Loss - loss scales with SL ratio
            if v9_sl < v5_sl:
                # Tighter SL - smaller losses
                v9_profit = v5_profit * sl_ratio
            else:
                # Wider SL - potentially larger losses
                v9_profit = v5_profit * sl_ratio

        zone = get_v9_zone(atr_pct)

        results.append({
            'Entry_Time': trade['Entry_Time'],
            'Exit_Time': trade['Exit_Time'],
            'Entry_Price': trade['Entry_Price'],
            'Exit_Price': trade['Exit_Price'],
            'Type': trade['Type'],
            'Volume': trade['Volume'],
            'ATR_Pct': atr_pct,
            'ATR_Zone': zone,
            'V5_TP': v5_tp,
            'V5_SL': v5_sl,
            'V9_TP': v9_tp,
            'V9_SL': v9_sl,
            'V5_Profit': v5_profit,
            'V9_Profit': v9_profit,
        })

    return pd.DataFrame(results)

def calculate_metrics(trades, profit_col='V9_Profit'):
    """Calculate trading metrics"""
    total_trades = len(trades)
    wins = trades[trades[profit_col] > 0]
    losses = trades[trades[profit_col] <= 0]

    total_pnl = trades[profit_col].sum()
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = wins[profit_col].mean() if len(wins) > 0 else 0
    avg_loss = losses[profit_col].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins[profit_col].sum() / losses[profit_col].sum()) if len(losses) > 0 and losses[profit_col].sum() != 0 else 0

    # Max drawdown
    balance = 10000
    peak = balance
    max_dd = 0
    max_dd_pct = 0

    for p in trades[profit_col].values:
        balance += p
        peak = max(peak, balance)
        dd = peak - balance
        dd_pct = dd / peak * 100
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)

    # Max consecutive losses
    profits = trades[profit_col].values
    max_consec = 0
    current = 0
    for p in profits:
        if p <= 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    return {
        'Total Trades': total_trades,
        'Total PnL': total_pnl,
        'Win Rate': win_rate,
        'Wins': len(wins),
        'Losses': len(losses),
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Profit Factor': profit_factor,
        'Max DD': max_dd,
        'Max DD %': max_dd_pct,
        'Max Consec Losses': max_consec,
        'Final Balance': 10000 + total_pnl,
    }

def generate_report(v9_trades, v5_metrics, v9_metrics):
    """Generate MT5-style report"""

    report = []
    report.append("=" * 80)
    report.append("STRATEGY TESTER REPORT - RSI v3.7 V9 (Python Simulation)")
    report.append("=" * 80)
    report.append("")
    report.append("Settings")
    report.append("-" * 40)
    report.append(f"Expert:                  RSI_v37_Original_v9")
    report.append(f"Symbol:                  GBPUSD")
    report.append(f"Period:                  H1 (2025.01.01 - 2025.12.31)")
    report.append(f"Initial Deposit:         $10,000.00")
    report.append(f"Strategy:                ATR-Adaptive TP/SL")
    report.append("")

    report.append("V9 ATR-Adaptive Settings")
    report.append("-" * 40)
    report.append(f"ATR 20-35% (VLOW):  TP={V9_SETTINGS['TP_VeryLow']} SL={V9_SETTINGS['SL_VeryLow']} (Fast Exit)")
    report.append(f"ATR 35-50% (LOW):   TP={V9_SETTINGS['TP_Low']} SL={V9_SETTINGS['SL_Low']}")
    report.append(f"ATR 50-60% (MID):   TP={V9_SETTINGS['TP_Mid']} SL={V9_SETTINGS['SL_Mid']} (Problem Zone)")
    report.append(f"ATR 60-75% (HIGH):  TP={V9_SETTINGS['TP_High']} SL={V9_SETTINGS['SL_High']}")
    report.append(f"ATR 75-85% (VHIGH): TP={V9_SETTINGS['TP_VeryHigh']} SL={V9_SETTINGS['SL_VeryHigh']} (Let Winners Run)")
    report.append("")

    report.append("=" * 80)
    report.append("RESULTS COMPARISON: V5 vs V9")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'Metric':<25} {'V5 (Baseline)':<20} {'V9 (Adaptive)':<20} {'Change':<15}")
    report.append("-" * 80)

    metrics_compare = [
        ('Total Net Profit', 'Total PnL', '${:,.2f}'),
        ('Total Trades', 'Total Trades', '{:,}'),
        ('Win Rate', 'Win Rate', '{:.2f}%'),
        ('Profit Factor', 'Profit Factor', '{:.2f}'),
        ('Avg Win', 'Avg Win', '${:,.2f}'),
        ('Avg Loss', 'Avg Loss', '${:,.2f}'),
        ('Max Drawdown', 'Max DD %', '{:.2f}%'),
        ('Max Consec Losses', 'Max Consec Losses', '{:,}'),
        ('Final Balance', 'Final Balance', '${:,.2f}'),
    ]

    for label, key, fmt in metrics_compare:
        v5_val = v5_metrics[key]
        v9_val = v9_metrics[key]

        if 'PnL' in key or 'Balance' in key or 'Win' in key or 'Loss' in key:
            change = v9_val - v5_val
            change_str = f"${change:+,.2f}"
        elif '%' in fmt:
            change = v9_val - v5_val
            change_str = f"{change:+.2f}%"
        else:
            change = v9_val - v5_val
            change_str = f"{change:+,.2f}"

        v5_str = fmt.format(v5_val)
        v9_str = fmt.format(v9_val)

        report.append(f"{label:<25} {v5_str:<20} {v9_str:<20} {change_str:<15}")

    report.append("-" * 80)
    report.append("")

    # Monthly breakdown
    report.append("=" * 80)
    report.append("MONTHLY BREAKDOWN (V9)")
    report.append("=" * 80)
    report.append("")

    v9_trades['Month'] = v9_trades['Entry_Time'].dt.month
    monthly = v9_trades.groupby('Month').agg(
        Trades=('V9_Profit', 'count'),
        PnL=('V9_Profit', 'sum'),
        WR=('V9_Profit', lambda x: (x > 0).mean() * 100)
    ).round(2)

    month_names = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
                   7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

    report.append(f"{'Month':<15} {'Trades':<10} {'PnL':<15} {'Win Rate':<10}")
    report.append("-" * 50)

    for m in range(1, 13):
        if m in monthly.index:
            row = monthly.loc[m]
            pnl_str = f"${row['PnL']:,.2f}"
            status = "PROFIT" if row['PnL'] > 0 else "LOSS"
            report.append(f"{month_names[m]:<15} {int(row['Trades']):<10} {pnl_str:<15} {row['WR']:.1f}% [{status}]")

    report.append("-" * 50)
    report.append("")

    # ATR Zone breakdown
    report.append("=" * 80)
    report.append("ATR ZONE PERFORMANCE (V9)")
    report.append("=" * 80)
    report.append("")

    zone_order = ['VLOW', 'LOW', 'MID', 'HIGH', 'VHIGH']
    zone_ranges = {'VLOW': '20-35%', 'LOW': '35-50%', 'MID': '50-60%', 'HIGH': '60-75%', 'VHIGH': '75-85%'}

    report.append(f"{'Zone':<10} {'ATR Range':<12} {'Trades':<10} {'PnL':<15} {'Win Rate':<10} {'TP/SL':<10}")
    report.append("-" * 70)

    for zone in zone_order:
        zone_trades = v9_trades[v9_trades['ATR_Zone'] == zone]
        if len(zone_trades) > 0:
            pnl = zone_trades['V9_Profit'].sum()
            wr = (zone_trades['V9_Profit'] > 0).mean() * 100
            tp = zone_trades['V9_TP'].iloc[0]
            sl = zone_trades['V9_SL'].iloc[0]
            pnl_str = f"${pnl:,.2f}"
            status = "OK" if pnl > 0 else "LOSS"
            report.append(f"{zone:<10} {zone_ranges[zone]:<12} {len(zone_trades):<10} {pnl_str:<15} {wr:.1f}%      {tp}/{sl} [{status}]")

    report.append("-" * 70)
    report.append("")

    # Trade list (first 20 and last 10)
    report.append("=" * 80)
    report.append("TRADE LIST (Sample)")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'#':<5} {'Entry Time':<20} {'Type':<6} {'ATR%':<8} {'Zone':<8} {'V5 P/L':<12} {'V9 P/L':<12}")
    report.append("-" * 80)

    for i, (_, trade) in enumerate(v9_trades.head(15).iterrows()):
        entry_time = trade['Entry_Time'].strftime('%Y.%m.%d %H:%M') if pd.notna(trade['Entry_Time']) else 'N/A'
        v5_pnl = f"${trade['V5_Profit']:,.2f}"
        v9_pnl = f"${trade['V9_Profit']:,.2f}"
        report.append(f"{i+1:<5} {entry_time:<20} {trade['Type']:<6} {trade['ATR_Pct']:<8.0f} {trade['ATR_Zone']:<8} {v5_pnl:<12} {v9_pnl:<12}")

    report.append("...")

    for i, (_, trade) in enumerate(v9_trades.tail(5).iterrows()):
        idx = len(v9_trades) - 5 + i + 1
        entry_time = trade['Entry_Time'].strftime('%Y.%m.%d %H:%M') if pd.notna(trade['Entry_Time']) else 'N/A'
        v5_pnl = f"${trade['V5_Profit']:,.2f}"
        v9_pnl = f"${trade['V9_Profit']:,.2f}"
        report.append(f"{idx:<5} {entry_time:<20} {trade['Type']:<6} {trade['ATR_Pct']:<8.0f} {trade['ATR_Zone']:<8} {v5_pnl:<12} {v9_pnl:<12}")

    report.append("-" * 80)
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    return "\n".join(report)

def main():
    print("Loading V5 trade data...")
    v5_file = r'C:\Users\Administrator\Documents\ReportTesterV5-61045904.xlsx'
    v5_trades = load_v5_trades(v5_file)
    print(f"Loaded {len(v5_trades)} trades from V5")

    print("\nSimulating V9 trades with ATR-Adaptive TP/SL...")
    v9_trades = simulate_v9_trades(v5_trades)

    print("\nCalculating metrics...")
    v5_metrics = calculate_metrics(v9_trades, 'V5_Profit')
    v9_metrics = calculate_metrics(v9_trades, 'V9_Profit')

    print("\nGenerating report...")
    report = generate_report(v9_trades, v5_metrics, v9_metrics)

    # Save report
    report_path = r'C:\Users\Administrator\Music\SURGE-WSI\backtest\h1_strategy\V9_Backtest_Report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"V5 Total PnL: ${v5_metrics['Total PnL']:,.2f}")
    print(f"V9 Total PnL: ${v9_metrics['Total PnL']:,.2f}")
    print(f"Change: ${v9_metrics['Total PnL'] - v5_metrics['Total PnL']:+,.2f}")
    print(f"V9 Win Rate: {v9_metrics['Win Rate']:.2f}%")
    print(f"V9 Profit Factor: {v9_metrics['Profit Factor']:.2f}")

    # Print full report
    print("\n")
    print(report)

    return v9_trades, v5_metrics, v9_metrics

if __name__ == "__main__":
    v9_trades, v5_metrics, v9_metrics = main()
