"""H1 Balanced Backtest - Quality + Frequency
=============================================

Balanced approach: Keep quality filters but allow more opportunities.

Key changes from strict version:
1. Keep POI detection but wider tolerance
2. Simpler entry triggers (momentum instead of rejection only)
3. Shorter cooldown (1 hour instead of 2)
4. Trade during extended hours (7-20 UTC)

Target: 1-2 trades per day with 55%+ win rate

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
class Trade:
    """Trade record"""
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
    poi_type: str = ""
    entry_type: str = ""


@dataclass
class Stats:
    """Statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades_per_day: float = 0.0
    monthly_stats: Dict = field(default_factory=dict)


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        return pd.DataFrame()
    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


def get_session(dt: datetime) -> str:
    """Get trading session"""
    hour = dt.hour
    if 12 <= hour < 16:
        return "Overlap"
    elif 7 <= hour < 12:
        return "London"
    elif 16 <= hour < 20:
        return "NewYork"
    return "Other"


def is_trading_hour(dt: datetime) -> bool:
    """Check trading hours"""
    hour = dt.hour
    weekday = dt.weekday()
    if weekday >= 5:
        return False
    if weekday == 4 and hour >= 18:
        return False
    return 7 <= hour < 20


def detect_poi(df: pd.DataFrame, idx: int, direction: str, col_map: dict, lookback: int = 15) -> Optional[dict]:
    """
    Simplified POI detection:
    - Order Block: last opposing candle before strong move
    - Looks for recent swing points
    """
    if idx < lookback + 3:
        return None

    recent = df.iloc[idx-lookback:idx]

    for i in range(len(recent) - 3):
        bar = recent.iloc[i]
        next_bars = recent.iloc[i+1:i+4]

        is_bullish = bar[col_map['close']] > bar[col_map['open']]
        is_bearish = bar[col_map['close']] < bar[col_map['open']]

        if direction == 'BUY':
            # Bullish OB: bearish candle before up move
            if is_bearish:
                move = next_bars[col_map['close']].max() - bar[col_map['low']]
                if move > 0.0010:  # 10 pips minimum move
                    return {
                        'type': 'OB',
                        'high': bar[col_map['high']],
                        'low': bar[col_map['low']],
                        'quality': min(100, move * 10000)
                    }
        else:
            # Bearish OB: bullish candle before down move
            if is_bullish:
                move = bar[col_map['high']] - next_bars[col_map['close']].min()
                if move > 0.0010:
                    return {
                        'type': 'OB',
                        'high': bar[col_map['high']],
                        'low': bar[col_map['low']],
                        'quality': min(100, move * 10000)
                    }

    # Also check for FVG
    for i in range(len(recent) - 2):
        bar1 = recent.iloc[i]
        bar3 = recent.iloc[i+2]

        if direction == 'BUY':
            gap = bar3[col_map['low']] - bar1[col_map['high']]
            if gap > 0.0004:  # 4 pips gap
                return {
                    'type': 'FVG',
                    'high': bar3[col_map['low']],
                    'low': bar1[col_map['high']],
                    'quality': min(100, gap * 10000 * 2)
                }
        else:
            gap = bar1[col_map['low']] - bar3[col_map['high']]
            if gap > 0.0004:
                return {
                    'type': 'FVG',
                    'high': bar1[col_map['low']],
                    'low': bar3[col_map['high']],
                    'quality': min(100, gap * 10000 * 2)
                }

    return None


def check_entry_signal(bar: pd.Series, prev_bar: pd.Series, direction: str, col_map: dict, velocity: float) -> Optional[str]:
    """
    Multiple entry types:
    1. Momentum: Strong candle in direction
    2. Rejection: Long wick showing rejection
    3. Engulfing: Current candle engulfs previous
    """
    o = bar[col_map['open']]
    h = bar[col_map['high']]
    l = bar[col_map['low']]
    c = bar[col_map['close']]

    total_range = h - l
    if total_range < 0.0003:
        return None

    body = abs(c - o)
    is_bull = c > o
    is_bear = c < o

    # Previous bar
    po = prev_bar[col_map['open']]
    pc = prev_bar[col_map['close']]

    if direction == 'BUY':
        # 1. Momentum: Strong bullish candle + positive velocity
        if is_bull and body > total_range * 0.5 and velocity > 2:
            return "MOMENTUM"

        # 2. Rejection: Long lower wick
        lower_wick = min(o, c) - l
        if lower_wick > total_range * 0.5:
            return "REJECTION"

        # 3. Engulfing: Bullish engulfing
        if is_bull and c > max(po, pc) and o < min(po, pc):
            return "ENGULF"

    else:  # SELL
        # 1. Momentum: Strong bearish candle + negative velocity
        if is_bear and body > total_range * 0.5 and velocity < -2:
            return "MOMENTUM"

        # 2. Rejection: Long upper wick
        upper_wick = h - max(o, c)
        if upper_wick > total_range * 0.5:
            return "REJECTION"

        # 3. Engulfing: Bearish engulfing
        if is_bear and c < min(po, pc) and o > max(po, pc):
            return "ENGULF"

    return None


def run_backtest(df: pd.DataFrame) -> tuple:
    """Run balanced H1 backtest"""

    pip_size = 0.0001
    spread_pips = 1.5
    sl_pips = 25.0
    tp_pips = 37.5  # 1.5R
    risk_per_trade = 0.01
    cooldown_hours = 1  # Shorter cooldown

    kalman = KalmanNoiseReducer()
    regime = HMMRegimeDetector()

    col_map = {
        'close': 'close' if 'close' in df.columns else 'Close',
        'open': 'open' if 'open' in df.columns else 'Open',
        'high': 'high' if 'high' in df.columns else 'High',
        'low': 'low' if 'low' in df.columns else 'Low',
    }

    # Warmup
    print("      Warming up...")
    for _, row in df.head(100).iterrows():
        kalman.update(row[col_map['close']])
        regime.update(row[col_map['close']])

    trades: List[Trade] = []
    position: Optional[Trade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    print("      Processing bars...")
    total_bars = len(df) - 100

    for idx in range(100, len(df)):
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1] if idx > 0 else bar
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        state = kalman.update(price)
        regime_info = regime.update(price)

        velocity = state.velocity / pip_size if state else 0

        if (idx - 100) % 500 == 0:
            pct = (idx - 100) / total_bars * 100
            print(f"      Progress: {pct:.0f}%")

        # Manage position
        if position:
            if position.direction == 'BUY':
                if low <= position.sl:
                    position.exit_time = current_time
                    position.exit_price = position.sl
                    position.pnl_pips = (position.sl - position.entry_price) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'LOSS'
                    position.exit_reason = 'SL'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=cooldown_hours)
                    continue
                elif high >= position.tp:
                    position.exit_time = current_time
                    position.exit_price = position.tp
                    position.pnl_pips = (position.tp - position.entry_price) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN'
                    position.exit_reason = 'TP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=cooldown_hours)
                    continue
            else:
                if high >= position.sl:
                    position.exit_time = current_time
                    position.exit_price = position.sl
                    position.pnl_pips = (position.entry_price - position.sl) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'LOSS'
                    position.exit_reason = 'SL'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=cooldown_hours)
                    continue
                elif low <= position.tp:
                    position.exit_time = current_time
                    position.exit_price = position.tp
                    position.pnl_pips = (position.entry_price - position.tp) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN'
                    position.exit_reason = 'TP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=cooldown_hours)
                    continue

            # Regime flip exit
            if regime_info and regime_info.is_tradeable:
                if position.direction == 'BUY' and regime_info.bias == 'SELL':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (price - position.entry_price) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=cooldown_hours)
                    continue
                elif position.direction == 'SELL' and regime_info.bias == 'BUY':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (position.entry_price - price) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + timedelta(hours=cooldown_hours)
                    continue

        # Update drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

        # Skip if in position or cooldown
        if position:
            continue
        if cooldown_until and current_time < cooldown_until:
            continue

        # Entry checks
        if not is_trading_hour(current_time):
            continue

        if not regime_info or not regime_info.is_tradeable:
            continue
        if regime_info.bias == 'NONE':
            continue

        direction = regime_info.bias

        # Require POI
        poi = detect_poi(df, idx, direction, col_map)
        if not poi:
            continue

        # Price near POI zone (wider tolerance)
        poi_tolerance = 0.0015  # 15 pips
        if direction == 'BUY':
            if not (poi['low'] - poi_tolerance <= price <= poi['high'] + poi_tolerance):
                continue
        else:
            if not (poi['low'] - poi_tolerance <= price <= poi['high'] + poi_tolerance):
                continue

        # Entry signal
        entry_type = check_entry_signal(bar, prev_bar, direction, col_map, velocity)
        if not entry_type:
            continue

        # Position size
        risk_amount = balance * risk_per_trade
        lot_size = risk_amount / (sl_pips * 10)
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            entry = price + spread_pips * pip_size
            sl_price = entry - sl_pips * pip_size
            tp_price = entry + tp_pips * pip_size
        else:
            entry = price
            sl_price = entry + sl_pips * pip_size
            tp_price = entry - tp_pips * pip_size

        session = get_session(current_time)
        position = Trade(
            entry_time=current_time,
            direction=direction,
            entry_price=entry,
            sl=sl_price,
            tp=tp_price,
            lot_size=lot_size,
            session=session,
            regime=regime_info.regime.value if hasattr(regime_info.regime, 'value') else str(regime_info.regime),
            poi_type=poi['type'],
            entry_type=entry_type
        )

    # Close remaining
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
        position.exit_reason = 'END'
        balance += position.pnl
        trades.append(position)

    # Statistics
    stats = calculate_stats(trades, balance, max_dd, df)
    return trades, stats, balance


def calculate_stats(trades: List[Trade], final_balance: float, max_dd: float, df: pd.DataFrame) -> Stats:
    """Calculate stats"""
    stats = Stats()
    if not trades:
        return stats

    stats.total_trades = len(trades)
    stats.wins = sum(1 for t in trades if t.result == 'WIN')
    stats.losses = sum(1 for t in trades if t.result == 'LOSS')
    stats.win_rate = stats.wins / stats.total_trades * 100 if stats.total_trades > 0 else 0
    stats.total_pnl = sum(t.pnl for t in trades)
    stats.total_pips = sum(t.pnl_pips for t in trades)
    stats.max_drawdown = max_dd

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [abs(t.pnl) for t in trades if t.pnl < 0]
    stats.avg_win = np.mean(wins) if wins else 0
    stats.avg_loss = np.mean(losses) if losses else 0
    stats.profit_factor = sum(wins) / sum(losses) if losses else float('inf')

    if len(df) > 0:
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            stats.trades_per_day = stats.total_trades / days

    for t in trades:
        month = t.entry_time.strftime('%Y-%m')
        if month not in stats.monthly_stats:
            stats.monthly_stats[month] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        stats.monthly_stats[month]['trades'] += 1
        if t.result == 'WIN':
            stats.monthly_stats[month]['wins'] += 1
        stats.monthly_stats[month]['pnl'] += t.pnl

    return stats


def print_report(stats: Stats, final_balance: float, trades: List[Trade]):
    """Print report"""
    print()
    print("=" * 70)
    print("H1 BALANCED BACKTEST RESULTS")
    print("=" * 70)
    print()

    print("PERFORMANCE")
    print("-" * 40)
    print(f"Initial:         $10,000.00")
    print(f"Final:           ${final_balance:,.2f}")
    print(f"Net P/L:         ${stats.total_pnl:+,.2f} ({(final_balance/10000-1)*100:+.1f}%)")
    print(f"Total Pips:      {stats.total_pips:+.1f}")
    print()

    print("STATISTICS")
    print("-" * 40)
    print(f"Trades:          {stats.total_trades} ({stats.trades_per_day:.2f}/day)")
    print(f"Wins/Losses:     {stats.wins}/{stats.losses}")
    print(f"Win Rate:        {stats.win_rate:.1f}%")
    print(f"Profit Factor:   {stats.profit_factor:.2f}")
    print(f"Max Drawdown:    ${stats.max_drawdown:,.2f}")
    print(f"Avg Win:         ${stats.avg_win:,.2f}")
    print(f"Avg Loss:        ${stats.avg_loss:,.2f}")
    print()

    # Entry type breakdown
    entry_types = {}
    for t in trades:
        et = t.entry_type
        if et not in entry_types:
            entry_types[et] = {'count': 0, 'wins': 0, 'pnl': 0}
        entry_types[et]['count'] += 1
        if t.result == 'WIN':
            entry_types[et]['wins'] += 1
        entry_types[et]['pnl'] += t.pnl

    print("ENTRY TYPE BREAKDOWN")
    print("-" * 50)
    print(f"{'Type':<12} {'Trades':>8} {'WR%':>8} {'P/L':>12}")
    for et, data in sorted(entry_types.items()):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"{et:<12} {data['count']:>8} {wr:>7.1f}% ${data['pnl']:>+10.2f}")
    print()

    # POI breakdown
    poi_types = {}
    for t in trades:
        pt = t.poi_type
        if pt not in poi_types:
            poi_types[pt] = {'count': 0, 'wins': 0, 'pnl': 0}
        poi_types[pt]['count'] += 1
        if t.result == 'WIN':
            poi_types[pt]['wins'] += 1
        poi_types[pt]['pnl'] += t.pnl

    print("POI TYPE BREAKDOWN")
    print("-" * 50)
    print(f"{'Type':<12} {'Trades':>8} {'WR%':>8} {'P/L':>12}")
    for pt, data in sorted(poi_types.items()):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"{pt:<12} {data['count']:>8} {wr:>7.1f}% ${data['pnl']:>+10.2f}")
    print()

    print("MONTHLY PERFORMANCE")
    print("-" * 60)
    print(f"{'Month':<10} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    print("-" * 60)

    losing = 0
    for month, data in sorted(stats.monthly_stats.items()):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        status = "[-]" if data['pnl'] < 0 else "[+]"
        if data['pnl'] < 0:
            losing += 1
        print(f"{month:<10} {data['trades']:>8} {data['wins']:>6} {wr:>7.1f}% ${data['pnl']:>+10.2f} {status}")

    print("-" * 60)
    print(f"Losing months: {losing}/{len(stats.monthly_stats)}")
    print()


async def send_telegram_report(stats: Stats, final_balance: float, trades: List[Trade]):
    """Send to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        msg = f"""*H1 BALANCED BACKTEST*
Period: Jan 2025 - Jan 2026 (13 months)

*PERFORMANCE*
• Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
• Win Rate: {stats.win_rate:.1f}%
• Net P/L: ${stats.total_pnl:+,.2f} ({(final_balance/10000-1)*100:+.1f}%)
• Profit Factor: {stats.profit_factor:.2f}
• Max DD: ${stats.max_drawdown:,.2f}

*ENTRY TYPES*
"""
        entry_types = {}
        for t in trades:
            if t.entry_type not in entry_types:
                entry_types[t.entry_type] = {'count': 0, 'pnl': 0}
            entry_types[t.entry_type]['count'] += 1
            entry_types[t.entry_type]['pnl'] += t.pnl

        for et, data in entry_types.items():
            msg += f"• {et}: {data['count']} trades, ${data['pnl']:+.2f}\n"

        msg += f"\nFinal Balance: ${final_balance:,.2f}"

        await telegram.send(msg)
        logger.info("Report sent to Telegram!")

    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    print("\n" + "=" * 70)
    print("SURGE-WSI H1 BALANCED BACKTEST")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("Strategy: Regime + POI + Multiple Entry Types")
    print("=" * 70)

    print("\n[1/3] Fetching H1 data...")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)
    df = await fetch_data("GBPUSD", "H1", start, end)

    if df.empty:
        print("ERROR: No data")
        return

    print(f"      Loaded {len(df)} bars")

    print("\n[2/3] Running backtest...")
    trades, stats, final_balance = run_backtest(df)

    print_report(stats, final_balance, trades)

    print("[3/3] Sending to Telegram...")
    await send_telegram_report(stats, final_balance, trades)

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
