"""Deep Analysis: Why 2024 Has Losing Months
============================================

Comprehensive analysis to find root cause:
1. Trade-by-trade analysis
2. Market conditions
3. HMM regime accuracy
4. Entry quality scores
5. Risk management effectiveness
6. Compare winning vs losing months

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger
from collections import defaultdict

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester, BacktestTrade


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        return pd.DataFrame()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=100000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def analyze_trades_detailed(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, year: int) -> dict:
    """Run backtest and collect detailed trade information"""

    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'skip_december': True,
    }

    months = []
    for month in range(1, 13):
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        months.append((start, end))

    all_trades = []
    monthly_stats = []

    for start_date, end_date in months:
        month_name = start_date.strftime("%b %Y")
        is_december = start_date.month == 12

        if is_december and CONFIG.get('skip_december', True):
            monthly_stats.append({
                'month': month_name,
                'month_num': start_date.month,
                'skipped': True
            })
            continue

        warmup_days = 30
        month_start_warmup = start_date - timedelta(days=warmup_days)

        htf_month = htf_df[(htf_df.index >= month_start_warmup) & (htf_df.index <= end_date)]
        ltf_month = ltf_df[(ltf_df.index >= month_start_warmup) & (ltf_df.index <= end_date)]

        if htf_month.empty or ltf_month.empty:
            monthly_stats.append({
                'month': month_name,
                'month_num': start_date.month,
                'skipped': True
            })
            continue

        htf = htf_month.reset_index()
        htf.rename(columns={'index': 'time'}, inplace=True)
        ltf = ltf_month.reset_index()
        ltf.rename(columns={'index': 'time'}, inplace=True)

        bt = Backtester(
            symbol="GBPUSD",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_balance=10000,
            pip_value=10.0,
            spread_pips=1.5,
            use_killzone=True,
            use_trend_filter=True,
            use_relaxed_filter=False,
            use_hybrid_mode=False,
            use_choppiness_filter=False
        )

        bt.risk_manager.max_lot_size = CONFIG['max_lot']
        bt.entry_trigger.min_quality_score = CONFIG['min_quality']

        bt.load_data(htf, ltf)
        result = bt.run()

        # Collect detailed trade info
        month_trades = []
        for trade in result.trade_list:
            trade_info = {
                'month': month_name,
                'month_num': start_date.month,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': trade.direction,
                'regime': trade.regime,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'sl_price': trade.sl_price,
                'tp1_price': trade.tp1_price,
                'lot_size': trade.initial_volume,
                'pnl': trade.pnl,
                'is_win': trade.pnl > 0,
                'quality_score': getattr(trade, 'quality_score', 0),
                'poi_type': getattr(trade, 'poi_type', 'unknown'),
                'exit_reason': getattr(trade, 'status', 'unknown'),
            }

            # Calculate risk-reward
            if trade.direction == 'BUY':
                risk_pips = (trade.entry_price - trade.sl_price) / 0.0001
                actual_pips = (trade.exit_price - trade.entry_price) / 0.0001
            else:
                risk_pips = (trade.sl_price - trade.entry_price) / 0.0001
                actual_pips = (trade.entry_price - trade.exit_price) / 0.0001

            trade_info['risk_pips'] = risk_pips
            trade_info['actual_pips'] = actual_pips
            trade_info['rr_achieved'] = actual_pips / risk_pips if risk_pips > 0 else 0

            month_trades.append(trade_info)
            all_trades.append(trade_info)

        # Calculate monthly stats
        wins = [t for t in month_trades if t['is_win']]
        losses = [t for t in month_trades if not t['is_win']]

        total_pnl = sum(t['pnl'] for t in month_trades)

        monthly_stats.append({
            'month': month_name,
            'month_num': start_date.month,
            'skipped': False,
            'total_trades': len(month_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(month_trades) * 100 if month_trades else 0,
            'total_pnl': total_pnl,
            'is_losing_month': total_pnl < 0,
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
            'avg_risk_pips': np.mean([t['risk_pips'] for t in month_trades]) if month_trades else 0,
            'avg_rr': np.mean([t['rr_achieved'] for t in month_trades]) if month_trades else 0,
            'regime_bullish': len([t for t in month_trades if t['regime'] == 'BULLISH']),
            'regime_bearish': len([t for t in month_trades if t['regime'] == 'BEARISH']),
            'buy_trades': len([t for t in month_trades if t['direction'] == 'BUY']),
            'sell_trades': len([t for t in month_trades if t['direction'] == 'SELL']),
            'buy_wins': len([t for t in wins if t['direction'] == 'BUY']),
            'sell_wins': len([t for t in wins if t['direction'] == 'SELL']),
        })

    return {
        'all_trades': all_trades,
        'monthly_stats': monthly_stats
    }


async def main():
    print("\n" + "=" * 80)
    print("DEEP ANALYSIS: ROOT CAUSE OF 2024 LOSING MONTHS")
    print("=" * 80)

    print("\nFetching data...")
    start = datetime(2023, 11, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Analyze both years
    print("\n" + "-" * 80)
    print("Analyzing 2024...")
    data_2024 = analyze_trades_detailed(htf_df, ltf_df, 2024)

    print("Analyzing 2025...")
    data_2025 = analyze_trades_detailed(htf_df, ltf_df, 2025)

    # ============================================================
    # SECTION 1: MONTHLY COMPARISON
    # ============================================================
    print("\n" + "=" * 80)
    print("SECTION 1: MONTHLY P&L COMPARISON")
    print("=" * 80)

    print("\n2024 Monthly Results:")
    print("{:<10} {:>8} {:>8} {:>8} {:>10} {:>12}".format(
        "Month", "Trades", "Wins", "WinRate", "P/L", "Status"
    ))
    print("-" * 60)

    losing_months_2024 = []
    winning_months_2024 = []

    for m in data_2024['monthly_stats']:
        if m.get('skipped'):
            print(f"{m['month']:<10} {'SKIPPED':>8}")
            continue
        status = "LOSING" if m['is_losing_month'] else "OK"
        print("{:<10} {:>8} {:>8} {:>7.1f}% {:>+9,.0f}$ {:>12}".format(
            m['month'], m['total_trades'], m['wins'], m['win_rate'],
            m['total_pnl'], status
        ))
        if m['is_losing_month']:
            losing_months_2024.append(m)
        else:
            winning_months_2024.append(m)

    print("\n2025 Monthly Results:")
    print("{:<10} {:>8} {:>8} {:>8} {:>10} {:>12}".format(
        "Month", "Trades", "Wins", "WinRate", "P/L", "Status"
    ))
    print("-" * 60)

    for m in data_2025['monthly_stats']:
        if m.get('skipped'):
            print(f"{m['month']:<10} {'SKIPPED':>8}")
            continue
        status = "LOSING" if m['is_losing_month'] else "OK"
        print("{:<10} {:>8} {:>8} {:>7.1f}% {:>+9,.0f}$ {:>12}".format(
            m['month'], m['total_trades'], m['wins'], m['win_rate'],
            m['total_pnl'], status
        ))

    # ============================================================
    # SECTION 2: WINNING VS LOSING MONTHS COMPARISON
    # ============================================================
    print("\n" + "=" * 80)
    print("SECTION 2: WINNING VS LOSING MONTHS CHARACTERISTICS")
    print("=" * 80)

    if losing_months_2024:
        print("\n--- LOSING MONTHS (2024) ---")
        print(f"Months: {[m['month'] for m in losing_months_2024]}")

        avg_trades_losing = np.mean([m['total_trades'] for m in losing_months_2024])
        avg_winrate_losing = np.mean([m['win_rate'] for m in losing_months_2024])
        avg_rr_losing = np.mean([m['avg_rr'] for m in losing_months_2024])
        avg_risk_losing = np.mean([m['avg_risk_pips'] for m in losing_months_2024])

        print(f"Avg Trades/Month: {avg_trades_losing:.1f}")
        print(f"Avg Win Rate: {avg_winrate_losing:.1f}%")
        print(f"Avg Risk (pips): {avg_risk_losing:.1f}")
        print(f"Avg R:R Achieved: {avg_rr_losing:.2f}")

        # Direction analysis
        total_buy = sum(m['buy_trades'] for m in losing_months_2024)
        total_sell = sum(m['sell_trades'] for m in losing_months_2024)
        buy_wins = sum(m['buy_wins'] for m in losing_months_2024)
        sell_wins = sum(m['sell_wins'] for m in losing_months_2024)

        print(f"BUY trades: {total_buy} (wins: {buy_wins}, rate: {buy_wins/total_buy*100 if total_buy else 0:.1f}%)")
        print(f"SELL trades: {total_sell} (wins: {sell_wins}, rate: {sell_wins/total_sell*100 if total_sell else 0:.1f}%)")

    if winning_months_2024:
        print("\n--- WINNING MONTHS (2024) ---")
        print(f"Months: {[m['month'] for m in winning_months_2024]}")

        avg_trades_winning = np.mean([m['total_trades'] for m in winning_months_2024])
        avg_winrate_winning = np.mean([m['win_rate'] for m in winning_months_2024])
        avg_rr_winning = np.mean([m['avg_rr'] for m in winning_months_2024])
        avg_risk_winning = np.mean([m['avg_risk_pips'] for m in winning_months_2024])

        print(f"Avg Trades/Month: {avg_trades_winning:.1f}")
        print(f"Avg Win Rate: {avg_winrate_winning:.1f}%")
        print(f"Avg Risk (pips): {avg_risk_winning:.1f}")
        print(f"Avg R:R Achieved: {avg_rr_winning:.2f}")

        # Direction analysis
        total_buy = sum(m['buy_trades'] for m in winning_months_2024)
        total_sell = sum(m['sell_trades'] for m in winning_months_2024)
        buy_wins = sum(m['buy_wins'] for m in winning_months_2024)
        sell_wins = sum(m['sell_wins'] for m in winning_months_2024)

        print(f"BUY trades: {total_buy} (wins: {buy_wins}, rate: {buy_wins/total_buy*100 if total_buy else 0:.1f}%)")
        print(f"SELL trades: {total_sell} (wins: {sell_wins}, rate: {sell_wins/total_sell*100 if total_sell else 0:.1f}%)")

    # ============================================================
    # SECTION 3: TRADE-BY-TRADE ANALYSIS OF LOSING MONTHS
    # ============================================================
    print("\n" + "=" * 80)
    print("SECTION 3: TRADE-BY-TRADE ANALYSIS (LOSING MONTHS)")
    print("=" * 80)

    losing_month_names = [m['month'] for m in losing_months_2024]
    losing_trades = [t for t in data_2024['all_trades'] if t['month'] in losing_month_names]

    for month_name in losing_month_names:
        month_trades = [t for t in losing_trades if t['month'] == month_name]
        print(f"\n--- {month_name} ---")
        print("{:>3} {:>12} {:>6} {:>8} {:>8} {:>8} {:>10}".format(
            "#", "Date", "Dir", "Regime", "RiskPip", "R:R", "P/L"
        ))
        print("-" * 65)

        for i, t in enumerate(month_trades, 1):
            date_str = t['entry_time'].strftime("%m/%d %H:%M") if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])[:11]
            status = "WIN" if t['is_win'] else "LOSS"
            print("{:>3} {:>12} {:>6} {:>8} {:>7.1f} {:>+7.2f} {:>+9,.0f}$ {}".format(
                i, date_str, t['direction'], t['regime'],
                t['risk_pips'], t['rr_achieved'], t['pnl'], status
            ))

    # ============================================================
    # SECTION 4: PATTERN ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("SECTION 4: PATTERN ANALYSIS")
    print("=" * 80)

    # Analyze by direction + regime combination
    print("\n--- Win Rate by Direction + Regime (2024 Losing Months) ---")
    combos = defaultdict(lambda: {'wins': 0, 'total': 0})
    for t in losing_trades:
        key = f"{t['direction']}_{t['regime']}"
        combos[key]['total'] += 1
        if t['is_win']:
            combos[key]['wins'] += 1

    for combo, stats in sorted(combos.items()):
        wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {combo}: {stats['wins']}/{stats['total']} = {wr:.1f}%")

    print("\n--- Win Rate by Direction + Regime (2024 Winning Months) ---")
    winning_month_names = [m['month'] for m in winning_months_2024]
    winning_trades = [t for t in data_2024['all_trades'] if t['month'] in winning_month_names]

    combos_win = defaultdict(lambda: {'wins': 0, 'total': 0})
    for t in winning_trades:
        key = f"{t['direction']}_{t['regime']}"
        combos_win[key]['total'] += 1
        if t['is_win']:
            combos_win[key]['wins'] += 1

    for combo, stats in sorted(combos_win.items()):
        wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {combo}: {stats['wins']}/{stats['total']} = {wr:.1f}%")

    # ============================================================
    # SECTION 5: RISK ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("SECTION 5: RISK & LOSS ANALYSIS")
    print("=" * 80)

    losing_trades_all = [t for t in losing_trades if not t['is_win']]

    print("\n--- Losing Trades Analysis (2024 Losing Months) ---")
    print(f"Total losing trades: {len(losing_trades_all)}")

    if losing_trades_all:
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades_all])
        max_loss = max([abs(t['pnl']) for t in losing_trades_all])
        total_loss = sum([abs(t['pnl']) for t in losing_trades_all])

        print(f"Average loss: ${avg_loss:.2f}")
        print(f"Max single loss: ${max_loss:.2f}")
        print(f"Total losses: ${total_loss:.2f}")

        # Analyze by risk size
        print("\n--- Losses by Risk Size ---")
        small_risk = [t for t in losing_trades_all if t['risk_pips'] <= 25]
        medium_risk = [t for t in losing_trades_all if 25 < t['risk_pips'] <= 40]
        large_risk = [t for t in losing_trades_all if t['risk_pips'] > 40]

        print(f"Small risk (<=25 pips): {len(small_risk)} losses, avg ${np.mean([abs(t['pnl']) for t in small_risk]) if small_risk else 0:.2f}")
        print(f"Medium risk (25-40 pips): {len(medium_risk)} losses, avg ${np.mean([abs(t['pnl']) for t in medium_risk]) if medium_risk else 0:.2f}")
        print(f"Large risk (>40 pips): {len(large_risk)} losses, avg ${np.mean([abs(t['pnl']) for t in large_risk]) if large_risk else 0:.2f}")

    # ============================================================
    # SECTION 6: KEY INSIGHTS
    # ============================================================
    print("\n" + "=" * 80)
    print("SECTION 6: KEY INSIGHTS & ROOT CAUSE")
    print("=" * 80)

    # Calculate key metrics
    if losing_months_2024 and winning_months_2024:
        wr_diff = avg_winrate_winning - avg_winrate_losing
        print(f"\n1. WIN RATE GAP: {wr_diff:.1f}%")
        print(f"   Losing months: {avg_winrate_losing:.1f}%")
        print(f"   Winning months: {avg_winrate_winning:.1f}%")

        # Check if specific direction is problematic
        print("\n2. DIRECTION ANALYSIS:")

        # Losing months
        buy_wr_losing = sum(m['buy_wins'] for m in losing_months_2024) / sum(m['buy_trades'] for m in losing_months_2024) * 100 if sum(m['buy_trades'] for m in losing_months_2024) > 0 else 0
        sell_wr_losing = sum(m['sell_wins'] for m in losing_months_2024) / sum(m['sell_trades'] for m in losing_months_2024) * 100 if sum(m['sell_trades'] for m in losing_months_2024) > 0 else 0

        # Winning months
        buy_wr_winning = sum(m['buy_wins'] for m in winning_months_2024) / sum(m['buy_trades'] for m in winning_months_2024) * 100 if sum(m['buy_trades'] for m in winning_months_2024) > 0 else 0
        sell_wr_winning = sum(m['sell_wins'] for m in winning_months_2024) / sum(m['sell_trades'] for m in winning_months_2024) * 100 if sum(m['sell_trades'] for m in winning_months_2024) > 0 else 0

        print(f"   BUY Win Rate  - Losing: {buy_wr_losing:.1f}%, Winning: {buy_wr_winning:.1f}% (gap: {buy_wr_winning-buy_wr_losing:.1f}%)")
        print(f"   SELL Win Rate - Losing: {sell_wr_losing:.1f}%, Winning: {sell_wr_winning:.1f}% (gap: {sell_wr_winning-sell_wr_losing:.1f}%)")

    # Specific analysis for combos
    print("\n3. PROBLEMATIC COMBINATIONS:")
    for combo, stats in sorted(combos.items()):
        wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
        if wr < 40 and stats['total'] >= 3:
            print(f"   {combo}: {wr:.1f}% win rate ({stats['total']} trades)")

    # ============================================================
    # SECTION 7: RECOMMENDATIONS
    # ============================================================
    print("\n" + "=" * 80)
    print("SECTION 7: POTENTIAL SOLUTIONS")
    print("=" * 80)

    solutions = []

    # Check win rate issue
    if losing_months_2024:
        if avg_winrate_losing < 40:
            solutions.append({
                'issue': 'Low win rate in losing months',
                'detail': f'{avg_winrate_losing:.1f}% vs {avg_winrate_winning:.1f}% in winning months',
                'solution': 'Increase min_quality_score threshold or add regime confirmation'
            })

    # Check direction imbalance
    for combo, stats in combos.items():
        wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
        if wr < 35 and stats['total'] >= 3:
            solutions.append({
                'issue': f'Poor performance on {combo}',
                'detail': f'{wr:.1f}% win rate on {stats["total"]} trades',
                'solution': f'Consider filtering out {combo} trades or add extra confirmation'
            })

    for i, sol in enumerate(solutions, 1):
        print(f"\n{i}. ISSUE: {sol['issue']}")
        print(f"   Detail: {sol['detail']}")
        print(f"   Solution: {sol['solution']}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
