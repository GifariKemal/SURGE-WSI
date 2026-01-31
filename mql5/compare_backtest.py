"""Compare MT5 vs PostgreSQL backtest results"""
import pandas as pd
import json

# Load MT5 data
with open("mql5/quadlayer_v69_mt5_full_summary.json") as f:
    mt5_data = json.load(f)

mt5_trades = pd.read_csv("mql5/quadlayer_v69_mt5_full_trades.csv")
mt5_total = mt5_data["total_trades"]
mt5_winners = len(mt5_trades[mt5_trades["pnl"] > 0])
mt5_losers = len(mt5_trades[mt5_trades["pnl"] <= 0])
mt5_tp = len(mt5_trades[mt5_trades["exit_reason"] == "TP"])
mt5_sl = len(mt5_trades[mt5_trades["exit_reason"] == "SL"])
mt5_avg_win = mt5_trades[mt5_trades["pnl"] > 0]["pnl"].mean() if mt5_winners > 0 else 0
mt5_avg_loss = abs(mt5_trades[mt5_trades["pnl"] <= 0]["pnl"].mean()) if mt5_losers > 0 else 0
mt5_rr = mt5_avg_win / mt5_avg_loss if mt5_avg_loss > 0 else 0
mt5_wr = mt5_data["win_rate"]
mt5_pf = mt5_data["profit_factor"]
mt5_net = mt5_data["net_pnl"]
mt5_losing = mt5_data["losing_months"]
mt5_months = mt5_data["total_months"]

# PostgreSQL data
pg_trades = pd.read_csv("strategies/gbpusd_h1_quadlayer/reports/quadlayer_trades.csv")
pg_total = len(pg_trades)
pg_winners = len(pg_trades[pg_trades["pnl"] > 0])
pg_losers = len(pg_trades[pg_trades["pnl"] <= 0])
pg_wr = pg_winners / pg_total * 100
pg_total_win = pg_trades[pg_trades["pnl"] > 0]["pnl"].sum()
pg_total_loss = abs(pg_trades[pg_trades["pnl"] <= 0]["pnl"].sum())
pg_pf = pg_total_win / pg_total_loss if pg_total_loss > 0 else 999
pg_net = pg_total_win - pg_total_loss
pg_avg_win = pg_trades[pg_trades["pnl"] > 0]["pnl"].mean()
pg_avg_loss = abs(pg_trades[pg_trades["pnl"] <= 0]["pnl"].mean())
pg_rr = pg_avg_win / pg_avg_loss
pg_tp = len(pg_trades[pg_trades["exit_reason"] == "TP"])
pg_sl = len(pg_trades[pg_trades["exit_reason"].str.contains("SL")])
pg_trades["month"] = pd.to_datetime(pg_trades["entry_time"]).dt.strftime("%Y-%m")
pg_monthly = pg_trades.groupby("month")["pnl"].sum()
pg_profitable_months = (pg_monthly > 0).sum()
pg_total_months = len(pg_monthly)
mt5_profitable = mt5_months - mt5_losing

print()
print("  Perbandingan Backtest: MT5 MetaQuotes vs PostgreSQL")
print("  +-------------------+------------------+------------------+-----------+")
print(f"  | {'Metrik':<17} | {'MT5 MetaQuotes':^16} | {'PostgreSQL':^16} | {'Selisih':^9} |")
print("  +-------------------+------------------+------------------+-----------+")

def row(name, mt5, pg, diff):
    print(f"  | {name:<17} | {str(mt5):^16} | {str(pg):^16} | {str(diff):^9} |")
    print("  +-------------------+------------------+------------------+-----------+")

row("Total Trades", mt5_total, pg_total, f"+{pg_total - mt5_total}")
row("Winners", mt5_winners, pg_winners, f"+{pg_winners - mt5_winners}")
row("Losers", mt5_losers, pg_losers, f"+{pg_losers - mt5_losers}")
row("Win Rate", f"{mt5_wr:.1f}%", f"{pg_wr:.1f}%", f"{pg_wr - mt5_wr:+.1f}%")
row("Profit Factor", f"{mt5_pf:.2f}", f"{pg_pf:.2f}", f"+{pg_pf - mt5_pf:.2f}")
row("Net P/L", f"${mt5_net:+,.0f}", f"${pg_net:+,.0f}", f"+${pg_net - mt5_net:,.0f}")
row("Avg Win", f"${mt5_avg_win:.0f}", f"${pg_avg_win:.0f}", f"+${pg_avg_win - mt5_avg_win:.0f}")
row("Avg Loss", f"${mt5_avg_loss:.0f}", f"${pg_avg_loss:.0f}", f"${pg_avg_loss - mt5_avg_loss:+.0f}")
row("R:R Ratio", f"1:{mt5_rr:.2f}", f"1:{pg_rr:.2f}", f"+{pg_rr - mt5_rr:.2f}")
row("TP Exits", f"{mt5_tp} ({mt5_tp/mt5_total*100:.0f}%)", f"{pg_tp} ({pg_tp/pg_total*100:.0f}%)", f"+{pg_tp - mt5_tp}")
row("SL Exits", f"{mt5_sl} ({mt5_sl/mt5_total*100:.0f}%)", f"{pg_sl} ({pg_sl/pg_total*100:.0f}%)", f"+{pg_sl - mt5_sl}")
print(f"  | Profitable Months | {mt5_profitable}/{mt5_months} ({mt5_profitable/mt5_months*100:.0f}%)          | {pg_profitable_months}/{pg_total_months} ({pg_profitable_months/pg_total_months*100:.0f}%)         |           |")
print("  +-------------------+------------------+------------------+-----------+")
print()
print("  CATATAN PERBEDAAN:")
print("  - MT5 backtest menggunakan deteksi sinyal SEDERHANA (Order Block + EMA Pullback)")
print("  - PostgreSQL menggunakan strategi LENGKAP dengan Capped SL & Multiple Entry Types")
print("  - Avg Loss PostgreSQL jauh lebih kecil karena menggunakan SL_CAPPED")
