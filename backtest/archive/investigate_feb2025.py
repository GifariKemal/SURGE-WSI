"""Investigate February 2025 Loss
================================

Detailed analysis of why February 2025 was a losing month.

Results from monthly backtest:
- Trades: 8
- Win Rate: 50%
- Profit Factor: 0.78
- Net P/L: -$79.21
- Max DD: 3.28%

Author: SURIOTA Team
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester


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
        logger.error("Failed to connect to database")
        return pd.DataFrame()

    df = await db.get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=50000,
        start_time=start,
        end_time=end
    )
    await db.disconnect()
    return df


async def investigate_february():
    """Detailed investigation of February 2025"""

    print("\n" + "=" * 70)
    print("FEBRUARY 2025 LOSS INVESTIGATION")
    print("=" * 70)

    # February 2025 dates
    feb_start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    feb_end = datetime(2025, 2, 28, tzinfo=timezone.utc)

    # Get data with warmup
    warmup_start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    print("\nFetching data...")
    htf_df = await fetch_data("GBPUSD", "H4", warmup_start, feb_end)
    ltf_df = await fetch_data("GBPUSD", "M15", warmup_start, feb_end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available")
        return

    print(f"H4 bars: {len(htf_df)}, M15 bars: {len(ltf_df)}")

    # Prepare data
    htf = htf_df.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)

    ltf = ltf_df.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)

    # Run backtest
    print("\nRunning backtest for February 2025...")
    bt = Backtester(
        symbol="GBPUSD",
        start_date=feb_start.strftime("%Y-%m-%d"),
        end_date=feb_end.strftime("%Y-%m-%d"),
        initial_balance=10000.0,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=True,
        use_trend_filter=True,
        use_relaxed_filter=True,
        use_hybrid_mode=False
    )

    bt.load_data(htf, ltf)
    result = bt.run()

    # Get detailed trades
    trades_df = bt.get_trades_df()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning: {result.winning_trades} | Losing: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Net P/L: ${result.net_profit:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")

    if trades_df.empty:
        print("\nNo trades in February 2025!")
        return

    # Detailed trade analysis
    print("\n" + "=" * 70)
    print("TRADE-BY-TRADE ANALYSIS")
    print("=" * 70)

    for idx, trade in trades_df.iterrows():
        status = "WIN" if trade['pnl'] > 0 else "LOSS"
        print(f"\n--- Trade #{idx + 1} ({status}) ---")
        print(f"Entry Time:  {trade['entry_time']}")
        print(f"Exit Time:   {trade['exit_time']}")
        print(f"Direction:   {trade['direction']}")
        print(f"Entry Price: {trade['entry_price']:.5f}")
        print(f"Exit Price:  {trade['exit_price']:.5f}")
        print(f"SL Price:    {trade['sl_price']:.5f}")
        print(f"TP1 Price:   {trade['tp1_price']:.5f}")
        print(f"TP2 Price:   {trade['tp2_price']:.5f}")
        print(f"TP3 Price:   {trade['tp3_price']:.5f}")
        print(f"Status:      {trade['status']}")
        print(f"Regime:      {trade['regime']}")
        print(f"Quality:     {trade['quality_score']:.1f}")
        print(f"Volume:      {trade['initial_volume']}")
        print(f"Pips:        {trade['pips']:.1f}")
        print(f"P/L:         ${trade['pnl']:.2f}")
        print(f"TP1 Hit:     {trade['tp1_hit']}")
        print(f"TP2 Hit:     {trade['tp2_hit']}")
        print(f"TP3 Hit:     {trade['tp3_hit']}")

    # Analysis
    print("\n" + "=" * 70)
    print("LOSS ANALYSIS")
    print("=" * 70)

    # Separate wins and losses
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    print(f"\nWinning Trades ({len(wins)}):")
    if not wins.empty:
        print(f"  Total Win: ${wins['pnl'].sum():.2f}")
        print(f"  Avg Win:   ${wins['pnl'].mean():.2f}")
        print(f"  Best Win:  ${wins['pnl'].max():.2f}")
        print(f"  Avg Pips:  {wins['pips'].mean():.1f}")

    print(f"\nLosing Trades ({len(losses)}):")
    if not losses.empty:
        print(f"  Total Loss: ${abs(losses['pnl'].sum()):.2f}")
        print(f"  Avg Loss:   ${abs(losses['pnl'].mean()):.2f}")
        print(f"  Worst Loss: ${abs(losses['pnl'].min()):.2f}")
        print(f"  Avg Pips:   {losses['pips'].mean():.1f}")

    # By regime
    print("\n" + "-" * 40)
    print("BY REGIME")
    print("-" * 40)
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        regime_wins = regime_trades[regime_trades['pnl'] > 0]
        print(f"\n{regime.upper()}:")
        print(f"  Trades: {len(regime_trades)}")
        print(f"  Win Rate: {len(regime_wins)/len(regime_trades)*100:.1f}%")
        print(f"  P/L: ${regime_trades['pnl'].sum():.2f}")

    # By direction
    print("\n" + "-" * 40)
    print("BY DIRECTION")
    print("-" * 40)
    for direction in trades_df['direction'].unique():
        dir_trades = trades_df[trades_df['direction'] == direction]
        dir_wins = dir_trades[dir_trades['pnl'] > 0]
        print(f"\n{direction}:")
        print(f"  Trades: {len(dir_trades)}")
        print(f"  Win Rate: {len(dir_wins)/len(dir_trades)*100:.1f}%")
        print(f"  P/L: ${dir_trades['pnl'].sum():.2f}")

    # Exit status analysis
    print("\n" + "-" * 40)
    print("BY EXIT STATUS")
    print("-" * 40)
    for status in trades_df['status'].unique():
        status_trades = trades_df[trades_df['status'] == status]
        print(f"\n{status}:")
        print(f"  Count: {len(status_trades)}")
        print(f"  P/L: ${status_trades['pnl'].sum():.2f}")

    # Duration analysis
    print("\n" + "-" * 40)
    print("TRADE DURATION ANALYSIS")
    print("-" * 40)
    trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
    wins_with_duration = trades_df[trades_df['pnl'] > 0]
    losses_with_duration = trades_df[trades_df['pnl'] <= 0]
    avg_win_duration = wins_with_duration['duration'].mean() if not wins_with_duration.empty else timedelta(0)
    avg_loss_duration = losses_with_duration['duration'].mean() if not losses_with_duration.empty else timedelta(0)
    print(f"Average Win Duration:  {avg_win_duration}")
    print(f"Average Loss Duration: {avg_loss_duration}")

    # Problem identification
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    total_win = wins['pnl'].sum() if not wins.empty else 0
    total_loss = abs(losses['pnl'].sum()) if not losses.empty else 0

    if total_loss > total_win:
        loss_excess = total_loss - total_win
        print(f"\nLosses exceeded wins by: ${loss_excess:.2f}")

        if not losses.empty:
            avg_loss_size = abs(losses['pnl'].mean())
            avg_win_size = wins['pnl'].mean() if not wins.empty else 0

            if avg_loss_size > avg_win_size:
                print(f"\nPROBLEM: Average loss (${avg_loss_size:.2f}) > Average win (${avg_win_size:.2f})")
                print("CAUSE: Losses are too large relative to wins")
                print("POSSIBLE REASONS:")
                print("  - SL too wide")
                print("  - Not reaching TP levels")
                print("  - Trailing stop not effective")

            # Check if issue is win rate
            if result.win_rate < 55:
                print(f"\nPROBLEM: Win rate ({result.win_rate:.1f}%) below profitable threshold")
                print("CAUSE: Not enough winning trades")
                print("POSSIBLE REASONS:")
                print("  - Poor market conditions (choppy/ranging)")
                print("  - Regime detection lagging")
                print("  - Entry timing issues")

            # Check TP hit rates
            tp1_hits = trades_df['tp1_hit'].sum()
            tp2_hits = trades_df['tp2_hit'].sum()
            tp3_hits = trades_df['tp3_hit'].sum()

            print(f"\nTP HIT ANALYSIS:")
            print(f"  TP1 Hit: {tp1_hits}/{len(trades_df)} ({tp1_hits/len(trades_df)*100:.1f}%)")
            print(f"  TP2 Hit: {tp2_hits}/{len(trades_df)} ({tp2_hits/len(trades_df)*100:.1f}%)")
            print(f"  TP3 Hit: {tp3_hits}/{len(trades_df)} ({tp3_hits/len(trades_df)*100:.1f}%)")

            if tp1_hits < len(trades_df) * 0.5:
                print("\nPROBLEM: TP1 hit rate below 50%")
                print("CAUSE: Price not reaching TP1 before reversing")

    # Market context - analyze price action in February
    print("\n" + "-" * 40)
    print("FEBRUARY 2025 MARKET CONTEXT")
    print("-" * 40)

    feb_ltf = ltf[(pd.to_datetime(ltf['time']) >= feb_start) & (pd.to_datetime(ltf['time']) <= feb_end)]
    if not feb_ltf.empty:
        feb_high = feb_ltf['high'].max()
        feb_low = feb_ltf['low'].min()
        feb_range = (feb_high - feb_low) / 0.0001  # in pips
        feb_open = feb_ltf.iloc[0]['open']
        feb_close = feb_ltf.iloc[-1]['close']
        feb_change = (feb_close - feb_open) / 0.0001

        print(f"Month Range: {feb_range:.0f} pips")
        print(f"Month Change: {feb_change:+.0f} pips ({('BULLISH' if feb_change > 0 else 'BEARISH')})")
        print(f"High: {feb_high:.5f}")
        print(f"Low: {feb_low:.5f}")

        # Volatility analysis
        feb_ltf['range'] = (feb_ltf['high'] - feb_ltf['low']) / 0.0001
        avg_bar_range = feb_ltf['range'].mean()
        print(f"Avg Bar Range: {avg_bar_range:.1f} pips")

        if avg_bar_range < 10:
            print("\nMARKET CONDITION: LOW VOLATILITY")
            print("Low volatility can cause whipsaws and false breakouts")

    # Position sizing analysis
    print("\n" + "-" * 40)
    print("POSITION SIZING ANALYSIS")
    print("-" * 40)
    print("\nVolume per trade:")
    for idx, trade in trades_df.iterrows():
        status = "WIN" if trade['pnl'] > 0 else "LOSS"
        print(f"  Trade #{idx+1}: {trade['initial_volume']:.2f} lot | {trade['pips']:.1f} pips | ${trade['pnl']:.2f} ({status})")

    max_vol_trade = trades_df.loc[trades_df['initial_volume'].idxmax()]
    print(f"\nLargest position: Trade #{trades_df['initial_volume'].idxmax()+1}")
    print(f"  Volume: {max_vol_trade['initial_volume']:.2f} lot")
    print(f"  P/L: ${max_vol_trade['pnl']:.2f}")
    print(f"  SL Distance: {abs(max_vol_trade['entry_price'] - max_vol_trade['sl_price'])/0.0001:.1f} pips")

    if max_vol_trade['pnl'] < 0:
        print("\n  WARNING: Largest position was a LOSING trade!")
        print("  This is a major contributor to the loss.")

    # Save detailed trades to CSV
    output_path = Path(__file__).parent / "results" / "feb2025_trades.csv"
    output_path.parent.mkdir(exist_ok=True)
    trades_df.to_csv(output_path, index=False)
    print(f"\nDetailed trades saved to: {output_path}")

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(investigate_february())
