"""H1 Day Trading Backtest - High Frequency Mode
================================================

Optimized for more frequent trades (3-8 signals per day target).
Less strict filters than the detailed backtest.

Features:
- Kalman Filter (noise reduction)
- HMM Regime Detection (BULLISH/BEARISH)
- Extended Kill Zones (London/New York overlap)
- Relaxed POI Detection
- Simple Entry Triggers (momentum-based)
- Risk Management (1% per trade)

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import KalmanNoiseReducer
from src.analysis.regime_detector import HMMRegimeDetector


@dataclass
class DayTrade:
    """Single day trade record"""
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot_size: float = 0.1
    pnl: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""
    exit_reason: str = ""
    session: str = ""
    regime: str = ""
    velocity: float = 0.0


@dataclass
class DayTradeStats:
    """Backtest statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    trades_per_day: float = 0.0
    london_trades: int = 0
    london_pnl: float = 0.0
    newyork_trades: int = 0
    newyork_pnl: float = 0.0
    overlap_trades: int = 0
    overlap_pnl: float = 0.0
    tp_exits: int = 0
    sl_exits: int = 0
    monthly_stats: Dict = field(default_factory=dict)


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data from database"""
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

    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def get_session(dt: datetime) -> str:
    """Get trading session name"""
    hour = dt.hour

    # London: 07:00 - 16:00 UTC
    # New York: 12:00 - 21:00 UTC
    # Overlap: 12:00 - 16:00 UTC

    if 12 <= hour < 16:
        return "Overlap"
    elif 7 <= hour < 16:
        return "London"
    elif 12 <= hour < 21:
        return "NewYork"
    return "Asian"


def is_trading_session(dt: datetime) -> bool:
    """Check if in trading session"""
    hour = dt.hour
    weekday = dt.weekday()

    # Skip weekends
    if weekday >= 5:
        return False

    # Skip Friday after 18:00
    if weekday == 4 and hour >= 18:
        return False

    # Extended hours: 07:00 - 21:00 UTC (London + NY)
    return 7 <= hour < 21


def run_backtest(df: pd.DataFrame, symbol: str = "GBPUSD") -> tuple:
    """Run H1 day trading backtest"""

    # Configuration for GBPUSD
    pip_size = 0.0001
    spread_pips = 1.5  # Typical spread

    # Day trading parameters
    sl_pips = 20.0  # Tighter SL for day trading
    tp_pips = 30.0  # 1.5R ratio
    risk_per_trade = 0.01  # 1% risk
    min_bars_between_trades = 2  # 2 hours minimum

    # Initialize components
    kalman = KalmanNoiseReducer()
    regime_detector = HMMRegimeDetector()

    # Column detection
    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Warmup (first 50 bars)
    print("      Warming up indicators...")
    for _, row in df.head(50).iterrows():
        kalman.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    # Trading state
    trades: List[DayTrade] = []
    position: Optional[DayTrade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    last_trade_idx = -100

    print("      Processing bars...")
    total_bars = len(df) - 50

    # Calculate ATR for each bar
    df['tr'] = np.maximum(
        df[col_map['high']] - df[col_map['low']],
        np.maximum(
            abs(df[col_map['high']] - df[col_map['close']].shift(1)),
            abs(df[col_map['low']] - df[col_map['close']].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    # Process bars
    for idx in range(50, len(df)):
        bar = df.iloc[idx]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]
        atr = bar['atr'] if not pd.isna(bar['atr']) else 0.001

        # Update indicators
        state = kalman.update(price)
        regime_info = regime_detector.update(price)

        # Progress update
        if (idx - 50) % 500 == 0:
            pct = (idx - 50) / total_bars * 100
            print(f"      Progress: {pct:.0f}% ({idx-50}/{total_bars} bars)")

        # Manage open position
        if position:
            # Check SL
            if position.direction == 'BUY' and low <= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.sl - position.entry_price) / pip_size - spread_pips
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'LOSS'
                position.exit_reason = 'SL'
                balance += position.pnl
                trades.append(position)
                position = None
                continue

            elif position.direction == 'SELL' and high >= position.sl:
                position.exit_time = current_time
                position.exit_price = position.sl
                position.pnl_pips = (position.entry_price - position.sl) / pip_size - spread_pips
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'LOSS'
                position.exit_reason = 'SL'
                balance += position.pnl
                trades.append(position)
                position = None
                continue

            # Check TP
            if position.direction == 'BUY' and high >= position.tp:
                position.exit_time = current_time
                position.exit_price = position.tp
                position.pnl_pips = (position.tp - position.entry_price) / pip_size - spread_pips
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN'
                position.exit_reason = 'TP'
                balance += position.pnl
                trades.append(position)
                position = None
                continue

            elif position.direction == 'SELL' and low <= position.tp:
                position.exit_time = current_time
                position.exit_price = position.tp
                position.pnl_pips = (position.entry_price - position.tp) / pip_size - spread_pips
                position.pnl = position.pnl_pips * position.lot_size * 10
                position.result = 'WIN'
                position.exit_reason = 'TP'
                balance += position.pnl
                trades.append(position)
                position = None
                continue

        # Update drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Skip if already in position
        if position:
            continue

        # Minimum bars between trades
        if idx - last_trade_idx < min_bars_between_trades:
            continue

        # Check trading session
        if not is_trading_session(current_time):
            continue

        # Check regime
        if not regime_info or not regime_info.is_tradeable:
            continue
        if regime_info.bias == 'NONE':
            continue

        # Get Kalman velocity
        if not state:
            continue

        velocity = state.velocity / pip_size  # Convert to pips

        # Minimum velocity for entry (momentum filter)
        min_velocity = 3.0  # 3 pips minimum velocity

        # Entry conditions based on regime + velocity
        direction = None

        if regime_info.bias == 'BUY' and velocity > min_velocity:
            direction = 'BUY'
        elif regime_info.bias == 'SELL' and velocity < -min_velocity:
            direction = 'SELL'

        if not direction:
            continue

        # Dynamic SL/TP based on ATR
        atr_pips = atr / pip_size
        actual_sl = min(sl_pips, max(15.0, atr_pips * 1.0))
        actual_tp = actual_sl * 1.5

        # Risk-based lot size (1% risk)
        risk_amount = balance * risk_per_trade
        lot_size = risk_amount / (actual_sl * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            entry = price + spread_pips * pip_size
            sl_price = entry - actual_sl * pip_size
            tp_price = entry + actual_tp * pip_size
        else:
            entry = price
            sl_price = entry + actual_sl * pip_size
            tp_price = entry - actual_tp * pip_size

        # Open position
        session = get_session(current_time)
        position = DayTrade(
            entry_time=current_time,
            direction=direction,
            entry_price=entry,
            sl=sl_price,
            tp=tp_price,
            lot_size=lot_size,
            session=session,
            regime=regime_info.regime.value if hasattr(regime_info.regime, 'value') else str(regime_info.regime),
            velocity=velocity
        )
        last_trade_idx = idx

    # Close any remaining position at last bar
    if position:
        last_bar = df.iloc[-1]
        position.exit_time = last_bar.name if isinstance(last_bar.name, datetime) else pd.Timestamp(last_bar.name).to_pydatetime()
        position.exit_price = last_bar[col_map['close']]
        if position.direction == 'BUY':
            position.pnl_pips = (position.exit_price - position.entry_price) / pip_size - spread_pips
        else:
            position.pnl_pips = (position.entry_price - position.exit_price) / pip_size - spread_pips
        position.pnl = position.pnl_pips * position.lot_size * 10
        position.result = 'WIN' if position.pnl > 0 else 'LOSS'
        position.exit_reason = 'END_OF_TEST'
        balance += position.pnl
        trades.append(position)

    # Calculate statistics
    stats = calculate_stats(trades, balance, max_dd, df)

    return trades, stats, balance


def calculate_stats(trades: List[DayTrade], final_balance: float, max_dd: float, df: pd.DataFrame) -> DayTradeStats:
    """Calculate comprehensive statistics"""
    stats = DayTradeStats()

    if not trades:
        return stats

    stats.total_trades = len(trades)
    stats.wins = sum(1 for t in trades if t.result == 'WIN')
    stats.losses = sum(1 for t in trades if t.result == 'LOSS')
    stats.win_rate = stats.wins / stats.total_trades * 100 if stats.total_trades > 0 else 0

    stats.total_pnl = sum(t.pnl for t in trades)
    stats.total_pips = sum(t.pnl_pips for t in trades)
    stats.max_drawdown = max_dd
    stats.max_drawdown_pct = max_dd / 10000 * 100

    winning_trades = [t.pnl for t in trades if t.pnl > 0]
    losing_trades = [abs(t.pnl) for t in trades if t.pnl < 0]

    stats.avg_win = np.mean(winning_trades) if winning_trades else 0
    stats.avg_loss = np.mean(losing_trades) if losing_trades else 0
    stats.largest_win = max(winning_trades) if winning_trades else 0
    stats.largest_loss = max(losing_trades) if losing_trades else 0

    total_wins = sum(winning_trades)
    total_losses = sum(losing_trades)
    stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Average trade duration
    durations = []
    for t in trades:
        if t.exit_time and t.entry_time:
            duration = (t.exit_time - t.entry_time).total_seconds() / 3600
            durations.append(duration)
    stats.avg_trade_duration = np.mean(durations) if durations else 0

    # Trades per day
    if len(df) > 0:
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        if total_days > 0:
            stats.trades_per_day = stats.total_trades / total_days

    # By session
    london_trades = [t for t in trades if t.session == 'London']
    newyork_trades = [t for t in trades if t.session == 'NewYork']
    overlap_trades = [t for t in trades if t.session == 'Overlap']

    stats.london_trades = len(london_trades)
    stats.london_pnl = sum(t.pnl for t in london_trades)
    stats.newyork_trades = len(newyork_trades)
    stats.newyork_pnl = sum(t.pnl for t in newyork_trades)
    stats.overlap_trades = len(overlap_trades)
    stats.overlap_pnl = sum(t.pnl for t in overlap_trades)

    # By exit reason
    stats.tp_exits = sum(1 for t in trades if t.exit_reason == 'TP')
    stats.sl_exits = sum(1 for t in trades if t.exit_reason == 'SL')

    # Monthly breakdown
    for t in trades:
        month_key = t.entry_time.strftime('%Y-%m')
        if month_key not in stats.monthly_stats:
            stats.monthly_stats[month_key] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pips': 0.0}
        stats.monthly_stats[month_key]['trades'] += 1
        if t.result == 'WIN':
            stats.monthly_stats[month_key]['wins'] += 1
        stats.monthly_stats[month_key]['pnl'] += t.pnl
        stats.monthly_stats[month_key]['pips'] += t.pnl_pips

    return stats


def print_report(stats: DayTradeStats, final_balance: float):
    """Print detailed report"""
    print()
    print("=" * 70)
    print("H1 DAY TRADING BACKTEST RESULTS")
    print("=" * 70)
    print()

    print("OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Initial Balance:     $10,000.00")
    print(f"Final Balance:       ${final_balance:,.2f}")
    print(f"Net P/L:             ${stats.total_pnl:+,.2f}")
    print(f"Return:              {(final_balance/10000-1)*100:+.1f}%")
    print(f"Total Pips:          {stats.total_pips:+.1f}")
    print()

    print("TRADE STATISTICS")
    print("-" * 40)
    print(f"Total Trades:        {stats.total_trades}")
    print(f"Trades/Day:          {stats.trades_per_day:.2f}")
    print(f"Wins:                {stats.wins}")
    print(f"Losses:              {stats.losses}")
    print(f"Win Rate:            {stats.win_rate:.1f}%")
    print(f"Profit Factor:       {stats.profit_factor:.2f}")
    print()

    print("P/L ANALYSIS")
    print("-" * 40)
    print(f"Average Win:         ${stats.avg_win:,.2f}")
    print(f"Average Loss:        ${stats.avg_loss:,.2f}")
    print(f"Largest Win:         ${stats.largest_win:,.2f}")
    print(f"Largest Loss:        ${stats.largest_loss:,.2f}")
    print(f"Max Drawdown:        ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)")
    print(f"Avg Duration:        {stats.avg_trade_duration:.1f} hours")
    print()

    print("SESSION BREAKDOWN")
    print("-" * 40)
    print(f"{'Session':<12} {'Trades':>8} {'P/L':>12}")
    print(f"{'London':<12} {stats.london_trades:>8} ${stats.london_pnl:>+10.2f}")
    print(f"{'New York':<12} {stats.newyork_trades:>8} ${stats.newyork_pnl:>+10.2f}")
    print(f"{'Overlap':<12} {stats.overlap_trades:>8} ${stats.overlap_pnl:>+10.2f}")
    print()

    print("EXIT REASONS")
    print("-" * 40)
    print(f"Take Profit:         {stats.tp_exits}")
    print(f"Stop Loss:           {stats.sl_exits}")
    print()

    print("MONTHLY PERFORMANCE")
    print("-" * 70)
    print(f"{'Month':<10} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Pips':>10} {'P/L':>12}")
    print("-" * 70)

    losing_months = 0
    for month, data in sorted(stats.monthly_stats.items()):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        status = "[-]" if data['pnl'] < 0 else "[+]"
        if data['pnl'] < 0:
            losing_months += 1
        print(f"{month:<10} {data['trades']:>8} {data['wins']:>8} {wr:>7.1f}% {data['pips']:>+9.1f} ${data['pnl']:>+10.2f} {status}")

    print("-" * 70)
    print(f"Losing months: {losing_months}/{len(stats.monthly_stats)}")
    print()


async def send_telegram_report(stats: DayTradeStats, final_balance: float):
    """Send report to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        msg = f"""*H1 DAY TRADING BACKTEST*
Period: Jan 2025 - Jan 2026 (13 months)
Symbol: GBPUSD | Timeframe: H1

*PERFORMANCE*
• Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
• Win Rate: {stats.win_rate:.1f}%
• Net P/L: ${stats.total_pnl:+,.2f} ({stats.total_pips:+.1f} pips)
• Return: {(final_balance/10000-1)*100:+.1f}%
• Profit Factor: {stats.profit_factor:.2f}
• Max DD: ${stats.max_drawdown:,.2f}

*SESSION BREAKDOWN*
• London: {stats.london_trades} trades, ${stats.london_pnl:+.2f}
• New York: {stats.newyork_trades} trades, ${stats.newyork_pnl:+.2f}
• Overlap: {stats.overlap_trades} trades, ${stats.overlap_pnl:+.2f}

Final Balance: ${final_balance:,.2f}
"""
        await telegram.send(msg)
        logger.info("Report sent to Telegram!")

    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI H1 DAY TRADING BACKTEST")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("Strategy: Regime + Velocity Momentum")
    print("=" * 70)

    # Fetch data
    print("\n[1/3] Fetching H1 data...")
    symbol = "GBPUSD"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    df = await fetch_data(symbol, "H1", start, end)
    if df.empty:
        print("ERROR: No data available")
        return

    print(f"      Loaded {len(df)} H1 bars")
    print(f"      Period: {df.index[0]} to {df.index[-1]}")

    # Run backtest
    print("\n[2/3] Running backtest...")
    trades, stats, final_balance = run_backtest(df, symbol)

    # Print report
    print_report(stats, final_balance)

    # Send to Telegram
    print("[3/3] Sending report to Telegram...")
    await send_telegram_report(stats, final_balance)

    print("=" * 70)
    print("BACKTEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
