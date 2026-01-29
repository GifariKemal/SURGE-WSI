"""Final Comparison: All Configurations
======================================

Compare: Conservative, Balanced, Aggressive (Adaptive), No Cap

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.trading.adaptive_risk import AdaptiveRiskManager, calculate_atr
from backtest.backtester import Backtester


async def fetch_data(symbol, timeframe, start, end):
    db = DBHandler(host=config.database.host, port=config.database.port,
                   database=config.database.database, user=config.database.user,
                   password=config.database.password)
    if not await db.connect():
        return pd.DataFrame()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=50000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def run_backtest(htf_df, ltf_df, max_lot, use_adaptive=False, adaptive_mgr=None):
    months = []
    for year in [2025, 2026]:
        for month in range(1, 13):
            if year == 2025 or (year == 2026 and month == 1):
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
                months.append((start, end))

    balance = 10000.0
    total_trades = 0
    total_wins = 0
    max_dd = 0
    monthly = []

    for start_date, end_date in months:
        warmup = start_date - timedelta(days=30)
        htf_m = htf_df[(htf_df.index >= warmup) & (htf_df.index <= end_date)]
        ltf_m = ltf_df[(ltf_df.index >= warmup) & (ltf_df.index <= end_date)]
        if htf_m.empty or ltf_m.empty:
            continue

        htf = htf_m.reset_index().rename(columns={'index': 'time'})
        ltf = ltf_m.reset_index().rename(columns={'index': 'time'})

        atr = calculate_atr(htf, 14)

        if use_adaptive and adaptive_mgr:
            params = adaptive_mgr.get_adaptive_params(balance, atr, 0.7, start_date)
            actual_lot = params.max_lot_size
            reason = params.reason
        else:
            actual_lot = max_lot
            reason = "Fixed"

        bt = Backtester(symbol="GBPUSD", start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"), initial_balance=balance,
                        pip_value=10.0, spread_pips=1.5, use_killzone=True,
                        use_trend_filter=True, use_relaxed_filter=True, use_hybrid_mode=False)

        bt.risk_manager.max_lot_size = actual_lot
        bt.risk_manager.min_sl_pips = 15.0
        bt.risk_manager.max_sl_pips = 50.0

        bt.load_data(htf, ltf)
        result = bt.run()

        if use_adaptive and adaptive_mgr:
            for t in result.trade_list:
                adaptive_mgr.record_trade_result(t.pnl > 0, t.pnl)

        monthly.append({'month': start_date.strftime("%b %Y"), 'pnl': result.net_profit,
                        'lot': actual_lot, 'reason': reason})
        balance = result.final_balance
        total_trades += result.total_trades
        total_wins += result.winning_trades
        if result.max_drawdown_percent > max_dd:
            max_dd = result.max_drawdown_percent

    return {
        'balance': balance,
        'return': (balance - 10000) / 100,
        'trades': total_trades,
        'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'max_dd': max_dd,
        'monthly': monthly
    }


async def main():
    print("\n" + "=" * 70)
    print("FINAL CONFIGURATION COMPARISON")
    print("=" * 70)

    print("\nFetching data...")
    htf = await fetch_data("GBPUSD", "H4", datetime(2024, 10, 1, tzinfo=timezone.utc),
                           datetime(2026, 1, 31, tzinfo=timezone.utc))
    ltf = await fetch_data("GBPUSD", "M15", datetime(2024, 10, 1, tzinfo=timezone.utc),
                           datetime(2026, 1, 31, tzinfo=timezone.utc))

    if htf.empty or ltf.empty:
        print("No data")
        return

    configs = [
        ("Conservative", 0.3, False),
        ("Balanced", 0.5, False),
        ("Aggressive+Adaptive", 0.75, True),
        ("No Cap", 5.0, False),
    ]

    results = []
    for name, max_lot, use_adaptive in configs:
        print(f"\nTesting: {name}...")
        adaptive_mgr = None
        if use_adaptive:
            adaptive_mgr = AdaptiveRiskManager(
                base_max_lot=0.75, low_volatility_atr=12.0,
                high_volatility_atr=40.0, extreme_volatility_atr=55.0,
                consecutive_loss_threshold=3, drawdown_threshold=0.10
            )
        r = run_backtest(htf, ltf, max_lot, use_adaptive, adaptive_mgr)
        r['name'] = name
        r['max_lot'] = max_lot
        results.append(r)
        print(f"  Return: {r['return']:+.1f}%, DD: {r['max_dd']:.1f}%")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print("\n{:<22} {:>8} {:>10} {:>10} {:>10}".format(
        "Config", "MaxLot", "Return", "MaxDD", "RiskAdj"
    ))
    print("-" * 60)
    for r in results:
        risk_adj = r['return'] / r['max_dd'] if r['max_dd'] > 0 else 0
        print("{:<22} {:>8.2f} {:>+9.1f}% {:>9.1f}% {:>9.2f}".format(
            r['name'], r['max_lot'], r['return'], r['max_dd'], risk_adj
        ))

    print("\n" + "-" * 60)
    print("MONTHLY P/L COMPARISON")
    print("-" * 60)
    print("{:<10}".format("Month"), end="")
    for r in results:
        print("{:>12}".format(r['name'][:10]), end="")
    print()

    for i in range(13):
        m = results[0]['monthly'][i]['month']
        print("{:<10}".format(m), end="")
        for r in results:
            print("{:>+11.0f}$".format(r['monthly'][i]['pnl']), end="")
        print()

    # Recommendation
    best = max(results, key=lambda x: x['return'] / x['max_dd'] if x['max_dd'] > 0 else 0)
    print("\n" + "=" * 70)
    print(f"RECOMMENDED: {best['name']}")
    print(f"  Return: {best['return']:+.1f}%")
    print(f"  Max DD: {best['max_dd']:.1f}%")
    print(f"  Risk-Adjusted: {best['return']/best['max_dd']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
