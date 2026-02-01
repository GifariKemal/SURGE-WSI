"""
SURGE-WSI DUAL-PAIR PORTFOLIO BACKTEST
=====================================
Session-Balanced Trading:
- GBPJPY: Asian Session (Tokyo 00:00-08:00 UTC)
- GBPUSD: London + NY Session (08:00-20:00 UTC)

This balances trading across 24 hours, with each pair
trading during its most active/liquid session.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Literal
import pandas as pd
import numpy as np
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.gbpusd_h1_quadlayer.trading_filters import IntraMonthRiskManager, PatternBasedFilter

# ============================================================
# DUAL-PAIR CONFIGURATION
# ============================================================

@dataclass
class PairConfig:
    """Configuration for each trading pair"""
    symbol: str
    pip_size: float
    pip_value: float  # USD per pip per lot
    session_hours: list  # UTC hours when this pair trades
    session_name: str

# APPROACH: Non-overlapping sessions with optimal pair-session matching
#
# GBPJPY - Tokyo-London Overlap (when both JPY and GBP are active)
# Tokyo: 00:00-09:00 UTC, London opens: 08:00 UTC
# Best overlap: 07:00-10:00 UTC + late Asian 00:00-07:00
GBPJPY_CONFIG = PairConfig(
    symbol="GBPJPY",
    pip_size=0.01,        # JPY pairs use 0.01
    pip_value=6.5,        # ~$6.5 per pip per lot
    session_hours=list(range(0, 11)),  # 00:00-10:59 UTC (Asian + Tokyo-London overlap)
    session_name="Tokyo-London"
)

# GBPUSD - Pure London + NY (after JPY closes)
# Avoid early London to prevent overlap
GBPUSD_CONFIG = PairConfig(
    symbol="GBPUSD",
    pip_size=0.0001,      # USD pairs use 0.0001
    pip_value=10.0,       # $10 per pip per lot
    session_hours=list(range(11, 21)),  # 11:00-20:59 UTC (London-NY, no Asian overlap)
    session_name="London-NY"
)

# ============================================================
# RISK PARAMETERS (shared)
# ============================================================
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = 0.5
SL_ATR_MULT = 1.2
TP_RATIO = 3.5
MAX_LOT = 5.0

# Entry signals
USE_ORDER_BLOCK = True
USE_EMA_PULLBACK = True

# Triple-layer filter
USE_TRIPLE_LAYER_FILTER = True

# ============================================================
# BACKTEST PERIOD
# ============================================================
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 2, 1)

# ============================================================
# TRADE DATA STRUCTURE
# ============================================================
@dataclass
class Trade:
    """Trade record"""
    pair: str
    session: str
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""
    quality_score: float = 0.0
    entry_type: str = ""
    poi_type: str = ""

# ============================================================
# DATA FETCHING
# ============================================================
async def fetch_mt5_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from MT5 directly"""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return pd.DataFrame()

    mt5_tf = mt5.TIMEFRAME_H1
    rates = mt5.copy_rates_range(symbol, mt5_tf, start, end)

    if rates is None or len(rates) == 0:
        logger.error(f"No data for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    }, inplace=True)

    logger.info(f"[MT5] Loaded {len(df)} bars for {symbol}")
    return df

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators"""
    df = df.copy()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # EMAs
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Structure
    df['higher_high'] = df['high'] > df['high'].shift(1)
    df['higher_low'] = df['low'] > df['low'].shift(1)
    df['lower_high'] = df['high'] < df['high'].shift(1)
    df['lower_low'] = df['low'] < df['low'].shift(1)

    # Trend
    df['trend'] = 0
    df.loc[df['ema_9'] > df['ema_21'], 'trend'] = 1
    df.loc[df['ema_9'] < df['ema_21'], 'trend'] = -1

    return df

# ============================================================
# SIGNAL DETECTION
# ============================================================
def detect_signals(df: pd.DataFrame, pair_config: PairConfig) -> list:
    """Detect trading signals for a pair"""
    signals = []

    for i in range(50, len(df)):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        hour = bar.name.hour

        # Session filter - only trade during pair's optimal session
        if hour not in pair_config.session_hours:
            continue

        atr = bar['atr']
        if pd.isna(atr) or atr <= 0:
            continue

        atr_pips = atr / pair_config.pip_size

        # Skip extreme ATR
        if atr_pips < 10 or atr_pips > 50:
            continue

        # EMA alignment check
        bullish_ema = bar['ema_9'] > bar['ema_21'] > bar['ema_50']
        bearish_ema = bar['ema_9'] < bar['ema_21'] < bar['ema_50']

        # RSI filter
        rsi = bar['rsi']
        rsi_bullish = 40 < rsi < 70
        rsi_bearish = 30 < rsi < 60

        signal = None

        # ORDER BLOCK signals
        if USE_ORDER_BLOCK:
            # Bullish Order Block
            if (bullish_ema and rsi_bullish and
                prev_bar['close'] < prev_bar['open'] and  # Previous bearish
                bar['close'] > bar['open'] and           # Current bullish
                bar['close'] > prev_bar['high']):        # Engulfing
                signal = {
                    'time': bar.name,
                    'direction': 'BUY',
                    'entry_price': bar['close'],
                    'atr': atr,
                    'atr_pips': atr_pips,
                    'entry_type': 'ENGULF',
                    'poi_type': 'ORDER_BLOCK',
                    'session': pair_config.session_name
                }

            # Bearish Order Block
            elif (bearish_ema and rsi_bearish and
                  prev_bar['close'] > prev_bar['open'] and  # Previous bullish
                  bar['close'] < bar['open'] and           # Current bearish
                  bar['close'] < prev_bar['low']):         # Engulfing
                signal = {
                    'time': bar.name,
                    'direction': 'SELL',
                    'entry_price': bar['close'],
                    'atr': atr,
                    'atr_pips': atr_pips,
                    'entry_type': 'ENGULF',
                    'poi_type': 'ORDER_BLOCK',
                    'session': pair_config.session_name
                }

        # EMA PULLBACK signals
        if USE_EMA_PULLBACK and signal is None:
            # Bullish pullback
            if (bullish_ema and rsi_bullish and
                bar['low'] <= bar['ema_21'] * 1.002 and
                bar['close'] > bar['ema_9'] and
                bar['close'] > bar['open']):
                signal = {
                    'time': bar.name,
                    'direction': 'BUY',
                    'entry_price': bar['close'],
                    'atr': atr,
                    'atr_pips': atr_pips,
                    'entry_type': 'MOMENTUM',
                    'poi_type': 'EMA_PULLBACK',
                    'session': pair_config.session_name
                }

            # Bearish pullback
            elif (bearish_ema and rsi_bearish and
                  bar['high'] >= bar['ema_21'] * 0.998 and
                  bar['close'] < bar['ema_9'] and
                  bar['close'] < bar['open']):
                signal = {
                    'time': bar.name,
                    'direction': 'SELL',
                    'entry_price': bar['close'],
                    'atr': atr,
                    'atr_pips': atr_pips,
                    'entry_type': 'MOMENTUM',
                    'poi_type': 'EMA_PULLBACK',
                    'session': pair_config.session_name
                }

        if signal:
            signals.append(signal)

    return signals

# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest_for_pair(
    df: pd.DataFrame,
    pair_config: PairConfig,
    risk_manager: IntraMonthRiskManager,
    pattern_filter: PatternBasedFilter,
    balance: float
) -> tuple[list, float]:
    """Run backtest for a single pair"""

    trades = []
    signals = detect_signals(df, pair_config)
    current_balance = balance

    logger.info(f"[{pair_config.symbol}] Found {len(signals)} signals in {pair_config.session_name} session")

    in_trade = False
    last_trade_time = None

    for signal in signals:
        # Skip if already in trade
        if in_trade:
            continue

        # Minimum time between trades (4 hours)
        if last_trade_time and (signal['time'] - last_trade_time).total_seconds() < 4 * 3600:
            continue

        # Quality filter
        if USE_TRIPLE_LAYER_FILTER:
            # Check risk manager (Layer 3)
            risk_ok, risk_adj, risk_reason = risk_manager.new_trade_check(signal['time'])
            if not risk_ok:
                continue

            # Check pattern filter (Layer 4)
            pattern_ok, pattern_size, pattern_reason = pattern_filter.check_trade_allowed()
            if not pattern_ok:
                continue

            quality = 70 + risk_adj

            if quality >= 100:
                continue
        else:
            quality = 70.0
            pattern_size = 1.0

        # Calculate position size
        atr_pips = signal['atr_pips']
        sl_pips = atr_pips * SL_ATR_MULT
        tp_pips = sl_pips * TP_RATIO

        risk_amount = current_balance * (RISK_PERCENT / 100)
        lot_size = risk_amount / (sl_pips * pair_config.pip_value)
        lot_size = min(lot_size, MAX_LOT)
        lot_size = round(lot_size, 2)

        if lot_size < 0.01:
            continue

        # Find exit
        entry_idx = df.index.get_loc(signal['time'])
        entry_price = signal['entry_price']
        direction = signal['direction']

        if direction == 'BUY':
            sl_price = entry_price - (sl_pips * pair_config.pip_size)
            tp_price = entry_price + (tp_pips * pair_config.pip_size)
        else:
            sl_price = entry_price + (sl_pips * pair_config.pip_size)
            tp_price = entry_price - (tp_pips * pair_config.pip_size)

        # Simulate trade
        exit_price = None
        exit_time = None
        exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + 48, len(df))):
            future_bar = df.iloc[j]

            if direction == 'BUY':
                if future_bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_time = future_bar.name
                    exit_reason = 'SL'
                    break
                elif future_bar['high'] >= tp_price:
                    exit_price = tp_price
                    exit_time = future_bar.name
                    exit_reason = 'TP'
                    break
            else:
                if future_bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_time = future_bar.name
                    exit_reason = 'SL'
                    break
                elif future_bar['low'] <= tp_price:
                    exit_price = tp_price
                    exit_time = future_bar.name
                    exit_reason = 'TP'
                    break

        # Timeout - close at market
        if exit_price is None:
            if entry_idx + 48 < len(df):
                exit_bar = df.iloc[entry_idx + 48]
                exit_price = exit_bar['close']
                exit_time = exit_bar.name
                exit_reason = 'TIMEOUT'
            else:
                continue

        # Calculate P&L
        if direction == 'BUY':
            pips_gained = (exit_price - entry_price) / pair_config.pip_size
        else:
            pips_gained = (entry_price - exit_price) / pair_config.pip_size

        pnl = pips_gained * lot_size * pair_config.pip_value
        current_balance += pnl

        # Record trade
        trade = Trade(
            pair=pair_config.symbol,
            session=signal['session'],
            entry_time=signal['time'],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            lot_size=lot_size,
            pnl=pnl,
            exit_reason=exit_reason,
            quality_score=quality,
            entry_type=signal['entry_type'],
            poi_type=signal['poi_type']
        )
        trades.append(trade)

        # Update filters
        risk_manager.record_trade(pnl, exit_time, direction)
        pattern_filter.record_trade(direction, pnl, exit_time)

        last_trade_time = signal['time']
        in_trade = False  # Trade completed

    return trades, current_balance

# ============================================================
# MAIN BACKTEST
# ============================================================
async def run_dual_pair_backtest():
    """Run backtest for both pairs with session filtering"""

    print("=" * 70)
    print("SURGE-WSI DUAL-PAIR PORTFOLIO BACKTEST")
    print("=" * 70)
    print(f"GBPJPY: Asian Session (00:00-08:00 UTC)")
    print(f"GBPUSD: London+NY Session (08:00-21:00 UTC)")
    print("=" * 70)
    print()

    # Fetch data for both pairs
    print("Fetching data...")
    df_gbpjpy = await fetch_mt5_data("GBPJPY", START_DATE, END_DATE)
    df_gbpusd = await fetch_mt5_data("GBPUSD", START_DATE, END_DATE)

    if df_gbpjpy.empty or df_gbpusd.empty:
        print("ERROR: Could not fetch data")
        return

    # Add indicators
    df_gbpjpy = compute_indicators(df_gbpjpy)
    df_gbpusd = compute_indicators(df_gbpusd)

    # Initialize filters for each pair (separate instances)
    risk_jpy = IntraMonthRiskManager()
    pattern_jpy = PatternBasedFilter()
    risk_usd = IntraMonthRiskManager()
    pattern_usd = PatternBasedFilter()

    # Split balance 50/50 for each pair
    pair_balance = INITIAL_BALANCE / 2

    # Run backtests
    print("\nRunning backtests...")
    trades_jpy, final_bal_jpy = run_backtest_for_pair(
        df_gbpjpy, GBPJPY_CONFIG, risk_jpy, pattern_jpy, pair_balance
    )
    trades_usd, final_bal_usd = run_backtest_for_pair(
        df_gbpusd, GBPUSD_CONFIG, risk_usd, pattern_usd, pair_balance
    )

    # Combine results
    all_trades = trades_jpy + trades_usd
    all_trades.sort(key=lambda t: t.entry_time)

    total_final = final_bal_jpy + final_bal_usd
    total_pnl = total_final - INITIAL_BALANCE

    # Calculate metrics
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0

    # Per-pair breakdown
    jpy_wins = [t for t in trades_jpy if t.pnl > 0]
    usd_wins = [t for t in trades_usd if t.pnl > 0]
    jpy_pnl = sum(t.pnl for t in trades_jpy)
    usd_pnl = sum(t.pnl for t in trades_usd)
    jpy_wr = len(jpy_wins) / len(trades_jpy) * 100 if trades_jpy else 0
    usd_wr = len(usd_wins) / len(trades_usd) * 100 if trades_usd else 0

    # Monthly breakdown
    monthly_pnl = {}
    for trade in all_trades:
        month_key = trade.entry_time.strftime('%Y-%m')
        if month_key not in monthly_pnl:
            monthly_pnl[month_key] = {'total': 0, 'jpy': 0, 'usd': 0}
        monthly_pnl[month_key]['total'] += trade.pnl
        if trade.pair == 'GBPJPY':
            monthly_pnl[month_key]['jpy'] += trade.pnl
        else:
            monthly_pnl[month_key]['usd'] += trade.pnl

    losing_months = sum(1 for m in monthly_pnl.values() if m['total'] < 0)

    # Print results
    print("\n" + "=" * 70)
    print("DUAL-PAIR PORTFOLIO RESULTS")
    print("=" * 70)

    print("\n[PORTFOLIO SUMMARY]")
    print("-" * 50)
    print(f"  Initial Balance:     ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Balance:       ${total_final:,.2f}")
    print(f"  Net P/L:             ${total_pnl:+,.2f}")
    print(f"  Total Return:        {(total_pnl/INITIAL_BALANCE)*100:+.1f}%")

    print("\n[TRADE STATISTICS]")
    print("-" * 50)
    print(f"  Total Trades:        {len(all_trades)}")
    print(f"  Win Rate:            {win_rate:.1f}%")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Avg Win:             ${avg_win:,.2f}")
    print(f"  Avg Loss:            ${avg_loss:,.2f}")

    print("\n[PER-PAIR BREAKDOWN]")
    print("-" * 50)
    print(f"  GBPJPY (Asian):")
    print(f"    Trades: {len(trades_jpy)}, WR: {jpy_wr:.1f}%, P/L: ${jpy_pnl:+,.2f}")
    print(f"  GBPUSD (London+NY):")
    print(f"    Trades: {len(trades_usd)}, WR: {usd_wr:.1f}%, P/L: ${usd_pnl:+,.2f}")

    print("\n[MONTHLY BREAKDOWN]")
    print("-" * 50)
    for month, data in sorted(monthly_pnl.items()):
        status = "WIN " if data['total'] >= 0 else "LOSS"
        print(f"  [{status}] {month}: ${data['total']:+,.2f} (JPY: ${data['jpy']:+,.2f}, USD: ${data['usd']:+,.2f})")

    print("\n" + "=" * 70)
    print(f"[OK] Profit: ${total_pnl:+,.2f}" if total_pnl > 0 else f"[X] Loss: ${total_pnl:,.2f}")
    print(f"[OK] PF: {profit_factor:.2f}" if profit_factor >= 2 else f"[X] PF: {profit_factor:.2f}")
    print(f"[OK] Losing Months: {losing_months}/{len(monthly_pnl)}" if losing_months <= 2 else f"[X] Losing Months: {losing_months}/{len(monthly_pnl)}")
    print("=" * 70)

    # Session coverage analysis
    print("\n[SESSION COVERAGE ANALYSIS]")
    print("-" * 50)
    jpy_hours = set()
    usd_hours = set()
    for t in trades_jpy:
        jpy_hours.add(t.entry_time.hour)
    for t in trades_usd:
        usd_hours.add(t.entry_time.hour)
    print(f"  GBPJPY trading hours: {sorted(jpy_hours)}")
    print(f"  GBPUSD trading hours: {sorted(usd_hours)}")
    overlap = jpy_hours & usd_hours
    if overlap:
        print(f"  Overlapping hours: {sorted(overlap)}")
    else:
        print(f"  No hour overlap - perfect session separation!")

    total_coverage = len(jpy_hours | usd_hours)
    print(f"  Total hour coverage: {total_coverage}/24 ({total_coverage/24*100:.0f}%)")

    # Save trades
    trades_df = pd.DataFrame([{
        'pair': t.pair,
        'session': t.session,
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'lot_size': t.lot_size,
        'pnl': t.pnl,
        'exit_reason': t.exit_reason,
        'quality': t.quality_score,
        'entry_type': t.entry_type,
        'poi_type': t.poi_type
    } for t in all_trades])

    output_path = os.path.join(os.path.dirname(__file__), 'reports', 'dual_pair_portfolio_trades.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")

    # Send to Telegram
    try:
        from src.utils.telegram import TelegramNotifier
        notifier = TelegramNotifier()
        await notifier.initialize()

        message = f"""📊 *DUAL-PAIR PORTFOLIO BACKTEST*

*Period:* {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}

*Portfolio Results:*
💰 Net P/L: ${total_pnl:+,.2f}
📈 Return: {(total_pnl/INITIAL_BALANCE)*100:+.1f}%
🎯 Win Rate: {win_rate:.1f}%
📊 Profit Factor: {profit_factor:.2f}
📅 Losing Months: {losing_months}/{len(monthly_pnl)}

*Per-Pair Breakdown:*
🇯🇵 GBPJPY (Asian): {len(trades_jpy)} trades, {jpy_wr:.1f}% WR, ${jpy_pnl:+,.2f}
🇺🇸 GBPUSD (London+NY): {len(trades_usd)} trades, {usd_wr:.1f}% WR, ${usd_pnl:+,.2f}

*Session Coverage:* {total_coverage}/24 hours ({total_coverage/24*100:.0f}%)
"""
        await notifier.send_message(message)
        print("[TELEGRAM] Results sent!")
    except Exception as e:
        print(f"[TELEGRAM] Error: {e}")

    return {
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'losing_months': losing_months,
        'jpy_pnl': jpy_pnl,
        'usd_pnl': usd_pnl
    }

if __name__ == "__main__":
    asyncio.run(run_dual_pair_backtest())
