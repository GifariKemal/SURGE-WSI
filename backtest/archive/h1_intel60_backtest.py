"""H1 + INTEL_70 Backtest
==========================

Main: H1 timeframe for trading
Filter: Intelligent Activity Filter (INTEL_70) - replaces Kill Zone
Support: H4 for regime, M5 for velocity detection

This matches the live system logic but uses H1 for trading signals.

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
from src.utils.intelligent_activity_filter import IntelligentActivityFilter, MarketActivity


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
    activity_level: str = ""
    activity_score: float = 0.0
    regime: str = ""
    poi_type: str = ""
    entry_type: str = ""


@dataclass
class Stats:
    """Backtest statistics"""
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

    # By activity level
    surging_trades: int = 0
    surging_pnl: float = 0.0
    active_trades: int = 0
    active_pnl: float = 0.0
    moderate_trades: int = 0
    moderate_pnl: float = 0.0

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


def detect_poi(df: pd.DataFrame, idx: int, direction: str, col_map: dict, lookback: int = 15) -> Optional[dict]:
    """Detect POI (Order Block or FVG)"""
    if idx < lookback + 3:
        return None

    recent = df.iloc[idx-lookback:idx]

    # Order Block detection
    for i in range(len(recent) - 3):
        bar = recent.iloc[i]
        next_bars = recent.iloc[i+1:i+4]

        is_bullish = bar[col_map['close']] > bar[col_map['open']]
        is_bearish = bar[col_map['close']] < bar[col_map['open']]

        if direction == 'BUY' and is_bearish:
            move = next_bars[col_map['close']].max() - bar[col_map['low']]
            if move > 0.0010:
                return {'type': 'OB', 'high': bar[col_map['high']], 'low': bar[col_map['low']], 'quality': min(100, move * 10000)}

        if direction == 'SELL' and is_bullish:
            move = bar[col_map['high']] - next_bars[col_map['close']].min()
            if move > 0.0010:
                return {'type': 'OB', 'high': bar[col_map['high']], 'low': bar[col_map['low']], 'quality': min(100, move * 10000)}

    # FVG detection
    for i in range(len(recent) - 2):
        bar1 = recent.iloc[i]
        bar3 = recent.iloc[i+2]

        if direction == 'BUY':
            gap = bar3[col_map['low']] - bar1[col_map['high']]
            if gap > 0.0003:
                return {'type': 'FVG', 'high': bar3[col_map['low']], 'low': bar1[col_map['high']], 'quality': min(100, gap * 20000)}
        else:
            gap = bar1[col_map['low']] - bar3[col_map['high']]
            if gap > 0.0003:
                return {'type': 'FVG', 'high': bar1[col_map['low']], 'low': bar3[col_map['high']], 'quality': min(100, gap * 20000)}

    return None


def check_entry(bar: pd.Series, prev_bar: pd.Series, direction: str, col_map: dict, velocity: float) -> Optional[str]:
    """Check entry trigger"""
    o, h, l, c = bar[col_map['open']], bar[col_map['high']], bar[col_map['low']], bar[col_map['close']]
    total_range = h - l
    if total_range < 0.0003:
        return None

    body = abs(c - o)
    is_bull = c > o
    is_bear = c < o
    po, ph, pl, pc = prev_bar[col_map['open']], prev_bar[col_map['high']], prev_bar[col_map['low']], prev_bar[col_map['close']]

    if direction == 'BUY':
        # Rejection
        lower_wick = min(o, c) - l
        if lower_wick > body and lower_wick > total_range * 0.5:
            return "REJECTION"
        # Momentum
        if is_bull and body > total_range * 0.6:
            return "MOMENTUM"
        # Higher low
        if l > pl and is_bull:
            return "HIGHER_LOW"
    else:
        # Rejection
        upper_wick = h - max(o, c)
        if upper_wick > body and upper_wick > total_range * 0.5:
            return "REJECTION"
        # Momentum
        if is_bear and body > total_range * 0.6:
            return "MOMENTUM"
        # Lower high
        if h < ph and is_bear:
            return "LOWER_HIGH"

    return None


def run_backtest(df_h1: pd.DataFrame, df_h4: pd.DataFrame, df_m5: pd.DataFrame) -> tuple:
    """Run H1 + INTEL_70 backtest"""

    pip_size = 0.0001
    spread_pips = 1.5
    sl_pips = 25.0
    tp_pips = 37.5  # 1.5R
    risk_per_trade = 0.01

    # Cooldowns
    cooldown_after_sl = timedelta(hours=1)
    cooldown_after_tp = timedelta(minutes=30)

    col_map = {
        'close': 'close' if 'close' in df_h1.columns else 'Close',
        'open': 'open' if 'open' in df_h1.columns else 'Open',
        'high': 'high' if 'high' in df_h1.columns else 'High',
        'low': 'low' if 'low' in df_h1.columns else 'Low',
    }

    # Initialize components
    kalman_m5 = KalmanNoiseReducer()  # M5 for velocity (more granular)
    kalman_h1 = KalmanNoiseReducer()  # H1 for trend
    regime_h1 = HMMRegimeDetector()   # H1 for regime (same TF as trading)

    # INTEL_70 filter (higher threshold for quality)
    intel_filter = IntelligentActivityFilter(
        activity_threshold=70.0,  # INTEL_70 - more selective
        min_velocity_pips=2.5,    # Higher min velocity
        min_atr_pips=6.0,         # Higher min ATR
        pip_size=pip_size
    )

    # Warmup M5 Kalman
    print("      Warming up M5 Kalman...")
    for _, row in df_m5.head(200).iterrows():
        kalman_m5.update(row[col_map['close']])

    # Warmup H1 regime
    print("      Warming up H1 regime...")
    for _, row in df_h1.head(100).iterrows():
        regime_h1.update(row[col_map['close']])

    # Warmup H1 Kalman and INTEL filter
    print("      Warming up H1 + INTEL...")
    for _, row in df_h1.head(50).iterrows():
        state = kalman_h1.update(row[col_map['close']])
        intel_filter.update(row[col_map['high']], row[col_map['low']], row[col_map['close']])
        if state:
            intel_filter.update_kalman_velocity(state.velocity)

    # Trading state
    trades: List[Trade] = []
    position: Optional[Trade] = None
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    cooldown_until = None

    # Track M5 index
    m5_idx = 200
    current_regime = None

    print("      Processing H1 bars...")
    total_bars = len(df_h1) - 50

    for idx in range(50, len(df_h1)):
        bar = df_h1.iloc[idx]
        prev_bar = df_h1.iloc[idx-1] if idx > 0 else bar
        current_time = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        price = bar[col_map['close']]
        high = bar[col_map['high']]
        low = bar[col_map['low']]

        # Update H1 Kalman
        state_h1 = kalman_h1.update(price)
        velocity_h1 = state_h1.velocity if state_h1 else 0

        # Update M5 Kalman (sync to current H1 time)
        state_m5 = None
        while m5_idx < len(df_m5):
            m5_time = df_m5.index[m5_idx]
            if m5_time.tzinfo is None:
                m5_time = m5_time.replace(tzinfo=timezone.utc)
            if m5_time <= current_time:
                m5_bar = df_m5.iloc[m5_idx]
                state_m5 = kalman_m5.update(m5_bar[col_map['close']])
                m5_idx += 1
            else:
                break

        # Get M5 velocity for INTEL filter (more sensitive to short-term moves)
        velocity_m5 = state_m5.velocity if state_m5 else 0

        # Update H1 regime (same timeframe as trading)
        current_regime = regime_h1.update(price)

        # Check INTEL_70 activity (using M5 velocity for sensitivity)
        activity = intel_filter.check(current_time, high, low, price, velocity_m5)

        # Progress
        if (idx - 50) % 500 == 0:
            pct = (idx - 50) / total_bars * 100
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
                    cooldown_until = current_time + cooldown_after_sl
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
                    cooldown_until = current_time + cooldown_after_tp
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
                    cooldown_until = current_time + cooldown_after_sl
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
                    cooldown_until = current_time + cooldown_after_tp
                    continue

            # Regime flip exit
            if current_regime and current_regime.is_tradeable:
                if position.direction == 'BUY' and current_regime.bias == 'SELL':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (price - position.entry_price) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
                    continue
                elif position.direction == 'SELL' and current_regime.bias == 'BUY':
                    position.exit_time = current_time
                    position.exit_price = price
                    position.pnl_pips = (position.entry_price - price) / pip_size - spread_pips
                    position.pnl = position.pnl_pips * position.lot_size * 10
                    position.result = 'WIN' if position.pnl > 0 else 'LOSS'
                    position.exit_reason = 'REGIME_FLIP'
                    balance += position.pnl
                    trades.append(position)
                    position = None
                    cooldown_until = current_time + cooldown_after_tp
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

        # ========== INTEL_70 FILTER ==========
        if not activity.should_trade:
            continue

        # Check H4 regime
        if not current_regime or not current_regime.is_tradeable:
            continue
        if current_regime.bias == 'NONE':
            continue

        direction = current_regime.bias

        # Detect POI on H1
        poi = detect_poi(df_h1, idx, direction, col_map)
        if not poi:
            continue

        # Price near POI
        poi_tolerance = 0.0015
        if direction == 'BUY':
            if not (poi['low'] - poi_tolerance <= price <= poi['high'] + poi_tolerance):
                continue
        else:
            if not (poi['low'] - poi_tolerance <= price <= poi['high'] + poi_tolerance):
                continue

        # Entry trigger
        entry_type = check_entry(bar, prev_bar, direction, col_map, velocity_h1 / pip_size)
        if not entry_type:
            continue

        # Only use REJECTION entry (the only profitable type from previous test)
        if entry_type != "REJECTION":
            continue

        # Position size (use INTEL lot multiplier)
        risk_amount = balance * risk_per_trade
        lot_size = risk_amount / (sl_pips * 10) * activity.lot_multiplier
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))

        if direction == 'BUY':
            entry = price + spread_pips * pip_size
            sl_price = entry - sl_pips * pip_size
            tp_price = entry + tp_pips * pip_size
        else:
            entry = price
            sl_price = entry + sl_pips * pip_size
            tp_price = entry - tp_pips * pip_size

        position = Trade(
            entry_time=current_time,
            direction=direction,
            entry_price=entry,
            sl=sl_price,
            tp=tp_price,
            lot_size=lot_size,
            activity_level=activity.activity.value,
            activity_score=activity.score,
            regime=current_regime.regime.value if hasattr(current_regime.regime, 'value') else str(current_regime.regime),
            poi_type=poi['type'],
            entry_type=entry_type
        )

    # Close remaining
    if position:
        last_bar = df_h1.iloc[-1]
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

    stats = calculate_stats(trades, balance, max_dd, df_h1)
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

    # By activity level
    surging = [t for t in trades if t.activity_level == 'surging']
    active = [t for t in trades if t.activity_level == 'active']
    moderate = [t for t in trades if t.activity_level == 'moderate']

    stats.surging_trades = len(surging)
    stats.surging_pnl = sum(t.pnl for t in surging)
    stats.active_trades = len(active)
    stats.active_pnl = sum(t.pnl for t in active)
    stats.moderate_trades = len(moderate)
    stats.moderate_pnl = sum(t.pnl for t in moderate)

    # Monthly
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
    print("H1 + INTEL_70 BACKTEST RESULTS")
    print("Main: H1 | Filter: INTEL_70 | Regime: H1 | Velocity: M5")
    print("=" * 70)
    print()

    print("PERFORMANCE")
    print("-" * 50)
    print(f"Initial:         $10,000.00")
    print(f"Final:           ${final_balance:,.2f}")
    print(f"Net P/L:         ${stats.total_pnl:+,.2f} ({(final_balance/10000-1)*100:+.1f}%)")
    print(f"Total Pips:      {stats.total_pips:+.1f}")
    print()

    print("STATISTICS")
    print("-" * 50)
    print(f"Trades:          {stats.total_trades} ({stats.trades_per_day:.2f}/day)")
    print(f"Wins/Losses:     {stats.wins}/{stats.losses}")
    print(f"Win Rate:        {stats.win_rate:.1f}%")
    print(f"Profit Factor:   {stats.profit_factor:.2f}")
    print(f"Max Drawdown:    ${stats.max_drawdown:,.2f}")
    print()

    # Activity breakdown
    print("INTEL_70 ACTIVITY BREAKDOWN")
    print("-" * 50)
    print(f"{'Level':<12} {'Trades':>8} {'P/L':>12}")
    print(f"{'SURGING':<12} {stats.surging_trades:>8} ${stats.surging_pnl:>+10.2f}")
    print(f"{'ACTIVE':<12} {stats.active_trades:>8} ${stats.active_pnl:>+10.2f}")
    print(f"{'MODERATE':<12} {stats.moderate_trades:>8} ${stats.moderate_pnl:>+10.2f}")
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
    print("-" * 60)
    print(f"{'Type':<15} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12}")
    for et, data in sorted(entry_types.items(), key=lambda x: -x[1]['pnl']):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"{et:<15} {data['count']:>8} {data['wins']:>6} {wr:>7.1f}% ${data['pnl']:>+10.2f}")
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


async def send_telegram(stats: Stats, final_balance: float, trades: List[Trade]):
    """Send to Telegram"""
    from src.utils.telegram import TelegramNotifier

    try:
        telegram = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id
        )

        msg = f"""*H1 + INTEL_70 BACKTEST*
Period: Jan 2025 - Jan 2026

*CONFIGURATION*
• Main TF: H1
• Filter: INTEL_70 (threshold=60)
• Regime: H4
• Velocity: M5

*PERFORMANCE*
• Trades: {stats.total_trades} ({stats.trades_per_day:.2f}/day)
• Win Rate: {stats.win_rate:.1f}%
• Net P/L: ${stats.total_pnl:+,.2f} ({(final_balance/10000-1)*100:+.1f}%)
• Profit Factor: {stats.profit_factor:.2f}
• Max DD: ${stats.max_drawdown:,.2f}

*INTEL_70 ACTIVITY*
• SURGING: {stats.surging_trades} trades, ${stats.surging_pnl:+.2f}
• ACTIVE: {stats.active_trades} trades, ${stats.active_pnl:+.2f}
• MODERATE: {stats.moderate_trades} trades, ${stats.moderate_pnl:+.2f}

*MONTHLY*
Losing months: {sum(1 for m in stats.monthly_stats.values() if m['pnl'] < 0)}/{len(stats.monthly_stats)}

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
    print("SURGE-WSI H1 + INTEL_70 BACKTEST")
    print("Period: 13 Months (Jan 2025 - Jan 2026)")
    print("=" * 70)

    symbol = "GBPUSD"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 31, tzinfo=timezone.utc)

    # Fetch all timeframes
    print("\n[1/5] Fetching H1 data...")
    df_h1 = await fetch_data(symbol, "H1", start, end)
    if df_h1.empty:
        print("ERROR: No H1 data")
        return
    print(f"      Loaded {len(df_h1)} H1 bars")

    print("\n[2/5] Fetching H4 data (regime)...")
    df_h4 = await fetch_data(symbol, "H4", start, end)
    if df_h4.empty:
        print("ERROR: No H4 data")
        return
    print(f"      Loaded {len(df_h4)} H4 bars")

    print("\n[3/5] Fetching M5 data (velocity)...")
    df_m5 = await fetch_data(symbol, "M5", start, end)
    if df_m5.empty:
        print("ERROR: No M5 data")
        return
    print(f"      Loaded {len(df_m5)} M5 bars")

    print("\n[4/5] Running backtest...")
    trades, stats, final_balance = run_backtest(df_h1, df_h4, df_m5)

    print_report(stats, final_balance, trades)

    print("[5/5] Sending to Telegram...")
    await send_telegram(stats, final_balance, trades)

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
