"""M15 Day Trading Backtest - High Frequency Quality Trades
==========================================================

Uses M15 timeframe for more frequent entries while maintaining
quality filters for better win rate.

Strategy:
- H1 for regime direction (HTF trend)
- M15 for entry timing (more opportunities)
- Momentum + Price Action filters
- Tighter SL/TP for day trading

Target: 2-4 signals per day with 55%+ win rate

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
    setup: str = ""


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

    if weekday >= 5:
        return False
    if weekday == 4 and hour >= 18:
        return False

    # Core sessions: 08:00 - 18:00 UTC (prime time)
    return 8 <= hour < 18


def is_momentum_candle(bar: pd.Series, col_map: dict, direction: str) -> bool:
    """Check if candle shows momentum in direction"""
    o = bar[col_map['open']]
    h = bar[col_map['high']]
    l = bar[col_map['low']]
    c = bar[col_map['close']]

    total_range = h - l
    if total_range < 0.0003:  # Min 3 pips
        return False

    body = abs(c - o)
    body_pct = body / total_range

    if direction == 'BUY':
        is_bullish = c > o
        if not is_bullish:
            return False
        # Strong bullish candle: body > 50% of range
        return body_pct > 0.5

    else:  # SELL
        is_bearish = c < o
        if not is_bearish:
            return False
        # Strong bearish candle: body > 50% of range
        return body_pct > 0.5


def is_pullback_entry(df: pd.DataFrame, idx: int, col_map: dict, direction: str) -> bool:
    """
    Check for pullback entry - price pulled back against trend then resumed

    BUY: Price made lower low but closing higher (potential reversal)
    SELL: Price made higher high but closing lower
    """
    if idx < 3:
        return False

    curr = df.iloc[idx]
    prev1 = df.iloc[idx-1]
    prev2 = df.iloc[idx-2]

    if direction == 'BUY':
        # Current bar is bullish
        if curr[col_map['close']] <= curr[col_map['open']]:
            return False

        # Had a pullback (lower lows in recent bars)
        pullback = (prev1[col_map['low']] < prev2[col_map['low']] or
                   curr[col_map['low']] < prev1[col_map['low']])

        # Now recovering
        recovering = curr[col_map['close']] > prev1[col_map['close']]

        return pullback and recovering

    else:  # SELL
        # Current bar is bearish
        if curr[col_map['close']] >= curr[col_map['open']]:
            return False

        # Had a rally (higher highs in recent bars)
        rally = (prev1[col_map['high']] > prev2[col_map['high']] or
                curr[col_map['high']] > prev1[col_map['high']])

        # Now falling
        falling = curr[col_map['close']] < prev1[col_map['close']]

        return rally and falling


def detect_breakout(df: pd.DataFrame, idx: int, col_map: dict, direction: str, lookback: int = 12) -> bool:
    """Detect breakout above/below recent range"""
    if idx < lookback + 1:
        return False

    curr = df.iloc[idx]
    recent = df.iloc[idx-lookback:idx]

    if direction == 'BUY':
        # Close above recent highs
        recent_high = recent[col_map['high']].max()
        return curr[col_map['close']] > recent_high

    else:  # SELL
        # Close below recent lows
        recent_low = recent[col_map['low']].min()
        return curr[col_map['close']] < recent_low


def run_backtest(df_m15: pd.DataFrame, df_h1: pd.DataFrame, symbol: str = "GBPUSD") -> tuple:
    """Run M15 day trading backtest with H1 regime"""

    # Configuration
    pip_size = 0.0001
    spread_pips = 1.5

    # Day trading parameters for M15
    sl_pips = 15.0  # Tighter SL for M15
    tp_pips = 22.5  # 1.5R ratio
    risk_per_trade = 0.01

    # Cooldown: 4 bars = 1 hour on M15
    min_bars_between_trades = 4

    # Initialize components
    kalman_ltf = KalmanNoiseReducer()  # For M15 entries
    kalman_htf = KalmanNoiseReducer()  # For H1 regime
    regime_detector = HMMRegimeDetector()

    # Column detection
    col_map = {
        'close': 'close' if 'close' in df_m15.columns else 'Close',
        'open': 'open' if 'open' in df_m15.columns else 'Open',
        'high': 'high' if 'high' in df_m15.columns else 'High',
        'low': 'low' if 'low' in df_m15.columns else 'Low',
    }

    # Warmup H1 regime detector
    print("      Warming up H1 regime detector...")
    for _, row in df_h1.head(100).iterrows():
        kalman_htf.update(row[col_map['close']])
        regime_detector.update(row[col_map['close']])

    # Warmup M15 Kalman
    print("      Warming up M15 indicators...")
    for _, row in df_m15.head(50).iterrows():
        kalman_ltf.update(row[col_map['close']])

    # Trading state
    trades: List[DayTrade] = []
    position: Optional[DayTrade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    last_trade_idx = -100

    # Track H1 bars for regime updates
    h1_idx = 100  # Start after warmup

    print("      Processing M15 bars...")
    total_bars = len(df_m15) - 50

    # Calculate ATR
    df_m15['tr'] = np.maximum(
        df_m15[col_map['high']] - df_m15[col_map['low']],
        np.maximum(
            abs(df_m15[col_map['high']] - df_m15[col_map['close']].shift(1)),
            abs(df_m15[col_map['low']] - df_m15[col_map['close']].shift(1))
        )
    )
    df_m15['atr'] = df_m15['tr'].rolling(14).mean()

    # Current regime
    current_regime = None
    last_h1_update = None

    for idx in range(50, len(df_m15)):
        bar = df_m15.iloc[idx]
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]
        atr = bar['atr'] if not pd.isna(bar['atr']) else 0.0006

        # Update M15 Kalman
        state_ltf = kalman_ltf.update(price)

        # Update H1 regime every hour (every 4 M15 bars)
        current_h1_time = current_time.replace(minute=0, second=0, microsecond=0)
        if last_h1_update != current_h1_time and h1_idx < len(df_h1):
            # Find matching H1 bar
            h1_bar = None
            while h1_idx < len(df_h1):
                h1_time = df_h1.index[h1_idx]
                if h1_time.tzinfo is None:
                    h1_time = h1_time.replace(tzinfo=timezone.utc)
                if h1_time <= current_time:
                    h1_bar = df_h1.iloc[h1_idx]
                    h1_idx += 1
                else:
                    break

            if h1_bar is not None:
                h1_price = h1_bar[col_map['close']]
                kalman_htf.update(h1_price)
                current_regime = regime_detector.update(h1_price)
                last_h1_update = current_h1_time

        # Progress update
        if (idx - 50) % 2000 == 0:
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

        # Skip if in position
        if position:
            continue

        # Cooldown
        if idx - last_trade_idx < min_bars_between_trades:
            continue

        # Check trading session
        if not is_trading_session(current_time):
            continue

        # Check regime
        if not current_regime or not current_regime.is_tradeable:
            continue
        if current_regime.bias == 'NONE':
            continue

        direction = current_regime.bias

        # Entry confirmation - need at least one of:
        # 1. Momentum candle in direction
        # 2. Pullback entry
        # 3. Breakout

        setup = None

        if is_momentum_candle(bar, col_map, direction):
            setup = "MOMENTUM"
        elif is_pullback_entry(df_m15, idx, col_map, direction):
            setup = "PULLBACK"
        elif detect_breakout(df_m15, idx, col_map, direction):
            setup = "BREAKOUT"

        if not setup:
            continue

        # M15 velocity confirmation
        if state_ltf:
            velocity_pips = state_ltf.velocity / pip_size
            min_vel = 1.5  # Lower threshold for M15

            if direction == 'BUY' and velocity_pips < min_vel:
                continue
            if direction == 'SELL' and velocity_pips > -min_vel:
                continue

        # Dynamic SL based on ATR
        atr_pips = atr / pip_size
        actual_sl = min(sl_pips, max(10.0, atr_pips * 1.2))
        actual_tp = actual_sl * 1.5

        # Risk-based lot size
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
            regime=current_regime.regime.value if hasattr(current_regime.regime, 'value') else str(current_regime.regime),
            setup=setup
        )
        last_trade_idx = idx

    # Close remaining position
    if position:
        last_bar = df_m15.iloc[-1]
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
    stats = calculate_stats(trades, balance, max_dd, df_m15)

    return trades, stats, balance


def calculate_stats(trades: List[DayTrade], final_balance: float, max_dd: float, df: pd.DataFrame) -> DayTradeStats:
    """Calculate statistics"""
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

    # Duration
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

    # Exit reasons
    stats.tp_exits = sum(1 for t in trades if t.exit_reason == 'TP')
    stats.sl_exits = sum(1 for t in trades if t.exit_reason == 'SL')

    # Monthly
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


def print_report(stats: DayTradeStats, final_balance: float, trades: List[DayTrade]):
    """Print report"""
    print()
    print("=" * 70)
    print("M15 DAY TRADING BACKTEST RESULTS")
    print("HTF: H1 (Regime) | LTF: M15 (Entry)")
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

    # Setup breakdown
    setup_counts = {}
    setup_pnl = {}
    for t in trades:
        if t.setup not in setup_counts:
            setup_counts[t.setup] = 0
            setup_pnl[t.setup] = 0
        setup_counts[t.setup] += 1
        setup_pnl[t.setup] += t.pnl

    print("SETUP BREAKDOWN")
    print("-" * 40)
    print(f"{'Setup':<12} {'Trades':>8} {'P/L':>12}")
    for setup in sorted(setup_counts.keys()):
        print(f"{setup:<12} {setup_counts[setup]:>8} ${setup_pnl[setup]:>+10.2f}")
    print()

    print("SESSION BREAKDOWN")
    print("-" * 40)
    print(f"{'Session':<12} {'Trades':>8} {'P/L':>12}")
    print(f"{'London':<12} {stats.london_trades:>8} ${stats.london_pnl:>+10.2f}")
    print(f"{'New York':<12} {stats.newyork_trades:>8} ${stats.newyork_pnl:>+10.2f}")
    print(f"{'Overlap':<12} {stats.overlap_trades:>8} ${stats.overlap_pnl:>+10.2f}")
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


async def send_telegram_report(stats: DayTradeStats, final_balance: float, trades: List[DayTrade]):
    """Send report to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        # Setup breakdown
        setup_counts = {}
        for t in trades:
            setup_counts[t.setup] = setup_counts.get(t.setup, 0) + 1

        setup_str = ", ".join([f"{k}: {v}" for k, v in setup_counts.items()])

        msg = f"""*M15 DAY TRADING BACKTEST*
Period: Jan 2025 - Jan 2026
HTF: H1 (Regime) | LTF: M15 (Entry)

*PERFORMANCE*
Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
Win Rate: {stats.win_rate:.1f}%
Net P/L: ${stats.total_pnl:+,.2f}
Return: {(final_balance/10000-1)*100:+.1f}%
Profit Factor: {stats.profit_factor:.2f}
Max DD: ${stats.max_drawdown:,.2f}

*SETUP BREAKDOWN*
{setup_str}

*SESSION BREAKDOWN*
London: {stats.london_trades} trades, ${stats.london_pnl:+.2f}
NY: {stats.newyork_trades} trades, ${stats.newyork_pnl:+.2f}
Overlap: {stats.overlap_trades} trades, ${stats.overlap_pnl:+.2f}

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
    print("SURGE-WSI M15 DAY TRADING BACKTEST")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("Strategy: H1 Regime + M15 Entry (Momentum/Pullback/Breakout)")
    print("=" * 70)

    symbol = "GBPUSD"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    # Fetch both timeframes
    print("\n[1/4] Fetching M15 data...")
    df_m15 = await fetch_data(symbol, "M15", start, end)
    if df_m15.empty:
        print("ERROR: No M15 data available")
        return
    print(f"      Loaded {len(df_m15)} M15 bars")

    print("\n[2/4] Fetching H1 data...")
    df_h1 = await fetch_data(symbol, "H1", start, end)
    if df_h1.empty:
        print("ERROR: No H1 data available")
        return
    print(f"      Loaded {len(df_h1)} H1 bars")

    # Run backtest
    print("\n[3/4] Running backtest...")
    trades, stats, final_balance = run_backtest(df_m15, df_h1, symbol)

    # Print report
    print_report(stats, final_balance, trades)

    # Send to Telegram
    print("[4/4] Sending report to Telegram...")
    await send_telegram_report(stats, final_balance, trades)

    print("=" * 70)
    print("BACKTEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
