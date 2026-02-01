"""
SURGE-WSI MULTI-PAIR BACKTEST (FULL LOGIC)
==========================================
Runs the EXACT same logic as backtest.py for both pairs:
- GBPUSD with full Quad-Layer filter
- GBPJPY with full Quad-Layer filter

Uses SHARED BALANCE - both pairs risk 0.5% of same balance.
Each pair has independent filters.
"""

import sys
import io
from pathlib import Path

STRATEGY_DIR = Path(__file__).parent
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGY_DIR.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5

from config import config
from src.utils.telegram import TelegramNotifier

# Import ALL the trading filters from the original
from gbpusd_h1_quadlayer.trading_filters import (
    IntraMonthRiskManager,
    PatternBasedFilter,
    get_monthly_quality_adjustment,
    MONTHLY_TRADEABLE_PCT,
)

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAIR CONFIGURATIONS
# ============================================================
@dataclass
class PairConfig:
    symbol: str
    pip_size: float
    pip_value: float

PAIRS = {
    'GBPUSD': PairConfig('GBPUSD', 0.0001, 10.0),
    'GBPJPY': PairConfig('GBPJPY', 0.01, 6.5),
}

# ============================================================
# SHARED PARAMETERS (from backtest.py)
# ============================================================
INITIAL_BALANCE = 50_000.0
RISK_PERCENT = 0.5
SL_ATR_MULT = 1.5   # From proven config
TP_RATIO = 1.5      # From proven config (1.5:1 R:R)
MAX_LOT = 5.0

MIN_ATR_PIPS = 10
MAX_ATR_PIPS = 50

# Day multipliers (from original)
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}

# Hour multipliers (from original)
HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.5, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.0,
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0,
}

ENTRY_MULTIPLIERS = {'MOMENTUM': 1.0, 'LOWER_HIGH': 1.0, 'ENGULF': 0.8}

MONTHLY_RISK = {
    1: 0.9, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.7, 6: 0.85,
    7: 1.0, 8: 0.75, 9: 0.9, 10: 0.6, 11: 0.75, 12: 0.8,
}

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 2, 1)

# ============================================================
# TRADE DATA
# ============================================================
@dataclass
class Trade:
    pair: str
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""
    quality: float = 0.0
    entry_type: str = ""
    poi_type: str = ""

# ============================================================
# DATA FETCHING & INDICATORS
# ============================================================
def fetch_data(symbol: str) -> pd.DataFrame:
    """Fetch H1 data from MT5"""
    if not mt5.initialize():
        print("MT5 init failed")
        return pd.DataFrame()

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, START_DATE, END_DATE)
    if rates is None:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    print(f"  {symbol}: {len(df)} bars")
    return df

def add_indicators(df: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    """Add all technical indicators"""
    df = df.copy()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pips'] = df['atr'] / pip_size

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

    # ADX for regime detection
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df

# ============================================================
# REGIME DETECTION
# ============================================================
class Regime(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"

def detect_regime(df: pd.DataFrame) -> Regime:
    """Detect market regime"""
    if len(df) < 50:
        return Regime.SIDEWAYS

    bar = df.iloc[-1]
    adx = bar.get('adx', 0)
    plus_di = bar.get('plus_di', 0)
    minus_di = bar.get('minus_di', 0)

    if pd.isna(adx) or adx < 20:
        return Regime.SIDEWAYS

    if plus_di > minus_di:
        return Regime.BULLISH
    else:
        return Regime.BEARISH

# ============================================================
# MARKET CONDITION ASSESSMENT
# ============================================================
def assess_quality(df: pd.DataFrame, current_time: datetime) -> Tuple[float, str]:
    """
    Assess market quality (Layer 1 + Layer 2)
    Returns (quality_score, condition_label)
    """
    bar = df.iloc[-1]

    # Layer 1: Monthly profile
    month = current_time.month
    tradeable_pct = MONTHLY_TRADEABLE_PCT.get(month, 65)

    # Calculate monthly adjustment based on tradeable_pct
    if tradeable_pct < 30:
        monthly_adj = 50  # NO TRADE
    elif tradeable_pct < 40:
        monthly_adj = 35  # HALT
    elif tradeable_pct < 50:
        monthly_adj = 25  # extreme
    elif tradeable_pct < 60:
        monthly_adj = 15  # very poor
    elif tradeable_pct < 70:
        monthly_adj = 10  # below avg
    elif tradeable_pct < 75:
        monthly_adj = 5   # slight
    else:
        monthly_adj = 0

    # Layer 2: Technical quality
    atr_pips = bar.get('atr_pips', 20)

    # ATR stability
    if len(df) >= 14:
        recent_atr = df['atr_pips'].iloc[-14:]
        atr_std = recent_atr.std() / recent_atr.mean() if recent_atr.mean() > 0 else 1.0
    else:
        atr_std = 0.5

    # Base quality from technical
    if atr_std < 0.25:
        base_quality = 60  # GOOD
        label = "GOOD"
    elif atr_std < 0.4:
        base_quality = 65  # NORMAL
        label = "NORMAL"
    else:
        base_quality = 80  # BAD
        label = "BAD"

    # Adjust for poor month
    if monthly_adj >= 15:
        label = "POOR_MONTH"

    final_quality = base_quality + monthly_adj

    return final_quality, label

# ============================================================
# SIGNAL DETECTION (from original backtest.py)
# ============================================================
def detect_signals(df: pd.DataFrame, pair_config: PairConfig) -> List[dict]:
    """Detect trading signals using full logic from backtest.py"""
    signals = []

    for i in range(100, len(df)):
        bar = df.iloc[i]
        prev = df.iloc[i-1]
        current_time = df.index[i]

        # Weekend filter
        if current_time.weekday() >= 5:
            continue

        hour = current_time.hour
        day = current_time.weekday()

        # Hour filter (from original)
        if HOUR_MULTIPLIERS.get(hour, 0) == 0:
            continue

        # Day filter
        if DAY_MULTIPLIERS.get(day, 0) == 0:
            continue

        # ATR filter
        atr_pips = bar.get('atr_pips', 0)
        if pd.isna(atr_pips) or atr_pips < MIN_ATR_PIPS or atr_pips > MAX_ATR_PIPS:
            continue

        # Regime check
        regime = detect_regime(df.iloc[:i+1])
        if regime == Regime.SIDEWAYS:
            continue

        # Quality assessment
        quality, label = assess_quality(df.iloc[:i+1], current_time)

        # EMA alignment
        bullish_ema = bar['ema_9'] > bar['ema_21'] > bar['ema_50']
        bearish_ema = bar['ema_9'] < bar['ema_21'] < bar['ema_50']

        # RSI filter
        rsi = bar['rsi']
        rsi_bullish = 40 < rsi < 70
        rsi_bearish = 30 < rsi < 60

        signal = None

        # ORDER BLOCK detection
        # Bullish engulfing
        if regime == Regime.BULLISH and bullish_ema and rsi_bullish:
            if prev['close'] < prev['open'] and bar['close'] > bar['open']:
                body = abs(bar['close'] - bar['open'])
                prev_body = abs(prev['close'] - prev['open'])
                if body > prev_body * 1.2 and bar['close'] > prev['high']:
                    signal = {
                        'time': current_time,
                        'direction': 'BUY',
                        'price': bar['close'],
                        'atr_pips': atr_pips,
                        'entry_type': 'ENGULF',
                        'poi_type': 'ORDER_BLOCK',
                        'quality': quality,
                        'hour': hour,
                        'day': day,
                        'month': current_time.month
                    }

        # Bearish engulfing
        if regime == Regime.BEARISH and bearish_ema and rsi_bearish and signal is None:
            if prev['close'] > prev['open'] and bar['close'] < bar['open']:
                body = abs(bar['close'] - bar['open'])
                prev_body = abs(prev['close'] - prev['open'])
                if body > prev_body * 1.2 and bar['close'] < prev['low']:
                    signal = {
                        'time': current_time,
                        'direction': 'SELL',
                        'price': bar['close'],
                        'atr_pips': atr_pips,
                        'entry_type': 'ENGULF',
                        'poi_type': 'ORDER_BLOCK',
                        'quality': quality,
                        'hour': hour,
                        'day': day,
                        'month': current_time.month
                    }

        # EMA PULLBACK detection
        if signal is None:
            # Bullish pullback
            if regime == Regime.BULLISH and bullish_ema and rsi_bullish:
                if bar['low'] <= bar['ema_21'] * 1.002:
                    if bar['close'] > bar['ema_9'] and bar['close'] > bar['open']:
                        total_range = bar['high'] - bar['low']
                        body = abs(bar['close'] - bar['open'])
                        if total_range > 0 and body > total_range * 0.5:
                            signal = {
                                'time': current_time,
                                'direction': 'BUY',
                                'price': bar['close'],
                                'atr_pips': atr_pips,
                                'entry_type': 'MOMENTUM',
                                'poi_type': 'EMA_PULLBACK',
                                'quality': quality,
                                'hour': hour,
                                'day': day,
                                'month': current_time.month
                            }

            # Bearish pullback
            if regime == Regime.BEARISH and bearish_ema and rsi_bearish and signal is None:
                if bar['high'] >= bar['ema_21'] * 0.998:
                    if bar['close'] < bar['ema_9'] and bar['close'] < bar['open']:
                        total_range = bar['high'] - bar['low']
                        body = abs(bar['close'] - bar['open'])
                        if total_range > 0 and body > total_range * 0.5:
                            signal = {
                                'time': current_time,
                                'direction': 'SELL',
                                'price': bar['close'],
                                'atr_pips': atr_pips,
                                'entry_type': 'MOMENTUM',
                                'poi_type': 'EMA_PULLBACK',
                                'quality': quality,
                                'hour': hour,
                                'day': day,
                                'month': current_time.month
                            }

        if signal:
            signals.append(signal)

    return signals

# ============================================================
# RISK MULTIPLIER (from original)
# ============================================================
def get_risk_multiplier(signal: dict, quality: float) -> Tuple[float, bool]:
    """Calculate combined risk multiplier"""
    day_mult = DAY_MULTIPLIERS.get(signal['day'], 0.5)
    hour_mult = HOUR_MULTIPLIERS.get(signal['hour'], 0.0)
    entry_mult = ENTRY_MULTIPLIERS.get(signal['entry_type'], 0.8)
    quality_mult = quality / 100.0
    month_mult = MONTHLY_RISK.get(signal['month'], 0.8)

    if day_mult == 0 or hour_mult == 0:
        return 0.0, True

    combined = day_mult * hour_mult * entry_mult * quality_mult * month_mult
    if combined < 0.30:
        return combined, True

    return max(0.30, min(1.2, combined)), False

# ============================================================
# MAIN BACKTEST
# ============================================================
async def run_multi_pair_backtest():
    """Run multi-pair backtest with shared balance and full filters"""

    print("=" * 70)
    print("MULTI-PAIR BACKTEST (FULL LOGIC)")
    print("=" * 70)
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Pairs: GBPUSD, GBPJPY")
    print(f"Risk: 0.5% per trade (shared balance)")
    print("=" * 70)

    # Fetch data
    print("\nLoading data...")
    data = {}
    for symbol, cfg in PAIRS.items():
        df = fetch_data(symbol)
        if df.empty:
            print(f"ERROR: No data for {symbol}")
            return
        data[symbol] = add_indicators(df, cfg.pip_size)

    # Detect signals
    print("\nDetecting signals...")
    signals = {}
    for symbol, cfg in PAIRS.items():
        signals[symbol] = detect_signals(data[symbol], cfg)
        print(f"  {symbol}: {len(signals[symbol])} raw signals")

    # Initialize state
    shared_balance = INITIAL_BALANCE

    pair_state = {}
    for symbol in PAIRS:
        pair_state[symbol] = {
            'risk_mgr': IntraMonthRiskManager(),
            'pattern_flt': PatternBasedFilter(),
            'last_trade': None,
            'current_month': None,
            'trades': []
        }

    # Combine and sort all signals
    all_signals = []
    for symbol, sigs in signals.items():
        for s in sigs:
            s['pair'] = symbol
            all_signals.append(s)
    all_signals.sort(key=lambda x: x['time'])

    print(f"\nProcessing {len(all_signals)} signals chronologically...")

    for signal in all_signals:
        pair = signal['pair']
        cfg = PAIRS[pair]
        state = pair_state[pair]
        df = data[pair]

        current_time = signal['time']
        direction = signal['direction']
        quality = signal['quality']

        # Cooldown check (2 hours)
        if state['last_trade']:
            if (current_time - state['last_trade']).total_seconds() < 2 * 3600:
                continue

        # Month reset check
        month_key = (current_time.year, current_time.month)
        if month_key != state['current_month']:
            state['current_month'] = month_key
            state['pattern_flt'].reset_for_month(current_time.month)

        # Layer 3: Risk manager
        can_trade, intra_adj, reason = state['risk_mgr'].new_trade_check(current_time)
        if not can_trade:
            continue

        # Layer 4: Pattern filter
        pattern_ok, size_mult, p_reason = state['pattern_flt'].check_trade_allowed()
        if not pattern_ok:
            continue

        # Quality threshold
        final_quality = quality + intra_adj
        if final_quality >= 100:
            continue

        # Risk multiplier
        risk_mult, skip = get_risk_multiplier(signal, final_quality)
        if skip or risk_mult < 0.3:
            continue

        # Calculate lot size from SHARED balance
        atr_pips = signal['atr_pips']
        sl_pips = atr_pips * SL_ATR_MULT
        tp_pips = sl_pips * TP_RATIO

        risk_amount = shared_balance * (RISK_PERCENT / 100) * risk_mult
        lot_size = risk_amount / (sl_pips * cfg.pip_value)
        lot_size = min(lot_size, MAX_LOT)
        lot_size = round(lot_size * size_mult, 2)

        if lot_size < 0.01:
            continue

        # Find exit
        entry_price = signal['price']
        entry_idx = df.index.get_loc(signal['time'])

        if direction == 'BUY':
            sl_price = entry_price - (sl_pips * cfg.pip_size)
            tp_price = entry_price + (tp_pips * cfg.pip_size)
        else:
            sl_price = entry_price + (sl_pips * cfg.pip_size)
            tp_price = entry_price - (tp_pips * cfg.pip_size)

        exit_price = None
        exit_time = None
        exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + 48, len(df))):
            bar = df.iloc[j]

            if direction == 'BUY':
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_time = df.index[j]
                    exit_reason = 'SL'
                    break
                if bar['high'] >= tp_price:
                    exit_price = tp_price
                    exit_time = df.index[j]
                    exit_reason = 'TP'
                    break
            else:
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_time = df.index[j]
                    exit_reason = 'SL'
                    break
                if bar['low'] <= tp_price:
                    exit_price = tp_price
                    exit_time = df.index[j]
                    exit_reason = 'TP'
                    break

        if exit_price is None:
            if entry_idx + 48 < len(df):
                bar = df.iloc[entry_idx + 48]
                exit_price = bar['close']
                exit_time = df.index[entry_idx + 48]
                exit_reason = 'TIMEOUT'
            else:
                continue

        # Calculate PnL
        if direction == 'BUY':
            pips = (exit_price - entry_price) / cfg.pip_size
        else:
            pips = (entry_price - exit_price) / cfg.pip_size

        pnl = pips * lot_size * cfg.pip_value

        # Cap losses
        max_loss = shared_balance * 0.02
        if pnl < -max_loss:
            pnl = -max_loss
            exit_reason = 'SL_CAPPED'

        # Update shared balance
        shared_balance += pnl

        # Update filters
        state['risk_mgr'].record_trade(pnl, exit_time, direction)
        state['pattern_flt'].record_trade(direction, pnl, exit_time)
        state['last_trade'] = current_time

        # Record trade
        trade = Trade(
            pair=pair,
            entry_time=current_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            lot_size=lot_size,
            pnl=pnl,
            exit_reason=exit_reason,
            quality=final_quality,
            entry_type=signal['entry_type'],
            poi_type=signal['poi_type']
        )
        state['trades'].append(trade)

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total_pnl = shared_balance - INITIAL_BALANCE
    all_trades = []

    for pair, state in pair_state.items():
        trades = state['trades']
        all_trades.extend(trades)

        if trades:
            pnl = sum(t.pnl for t in trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            wr = wins / len(trades) * 100

            gp = sum(t.pnl for t in trades if t.pnl > 0)
            gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
            pf = gp / gl if gl > 0 else 999

            print(f"\n{pair}:")
            print(f"  Trades: {len(trades)}")
            print(f"  Win Rate: {wr:.1f}%")
            print(f"  PF: {pf:.2f}")
            print(f"  P/L: ${pnl:+,.2f}")

    # Portfolio summary
    if all_trades:
        wins = sum(1 for t in all_trades if t.pnl > 0)
        total_wr = wins / len(all_trades) * 100

        gp = sum(t.pnl for t in all_trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in all_trades if t.pnl <= 0))
        total_pf = gp / gl if gl > 0 else 999

        # Monthly
        monthly = {}
        for t in all_trades:
            m = t.entry_time.strftime('%Y-%m')
            monthly[m] = monthly.get(m, 0) + t.pnl

        losing_months = sum(1 for v in monthly.values() if v < 0)

        print("\n" + "=" * 70)
        print("[PORTFOLIO]")
        print("=" * 70)
        print(f"  Initial:      ${INITIAL_BALANCE:,.0f}")
        print(f"  Final:        ${shared_balance:,.0f}")
        print(f"  Net P/L:      ${total_pnl:+,.2f}")
        print(f"  Return:       {(total_pnl/INITIAL_BALANCE)*100:+.1f}%")
        print(f"  Total Trades: {len(all_trades)}")
        print(f"  Win Rate:     {total_wr:.1f}%")
        print(f"  PF:           {total_pf:.2f}")
        print(f"  Losing Months: {losing_months}/{len(monthly)}")

        print("\n[MONTHLY]")
        for m, pnl in sorted(monthly.items()):
            status = "WIN " if pnl >= 0 else "LOSS"
            print(f"  [{status}] {m}: ${pnl:+,.2f}")

        print("\n" + "=" * 70)
        if total_pnl > 10000 and total_pf > 2:
            print("[OK] PROFITABLE - Ready for live demo!")
        elif total_pnl > 0:
            print("[OK] Profitable but consider optimization")
        else:
            print("[X] Not profitable - needs tuning")
        print("=" * 70)

    # Telegram
    try:
        from src.config.mt5_config import MT5Config
        mt5_config = MT5Config()
        notifier = TelegramNotifier(mt5_config.telegram_token, mt5_config.telegram_chat_id)
        await notifier.initialize()

        # Per pair summary
        pair_lines = []
        for pair, state in pair_state.items():
            trades = state['trades']
            if trades:
                pnl = sum(t.pnl for t in trades)
                wr = sum(1 for t in trades if t.pnl > 0) / len(trades) * 100
                pair_lines.append(f"  {pair}: {len(trades)} trades, {wr:.0f}% WR, ${pnl:+,.0f}")

        msg = f"""📊 *MULTI-PAIR BACKTEST (FULL)*

*Period:* {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}

*Portfolio:*
  Initial: $50,000
  Final: ${shared_balance:,.0f}
  P/L: ${total_pnl:+,.0f}
  Return: {(total_pnl/INITIAL_BALANCE)*100:+.1f}%

*Stats:*
  Trades: {len(all_trades)}
  WR: {total_wr:.0f}%
  PF: {total_pf:.2f}
  Losing Months: {losing_months}/{len(monthly)}

*Per Pair:*
{chr(10).join(pair_lines)}
"""
        await notifier.send_message(msg)
        print("\n[TELEGRAM] Sent!")
    except Exception as e:
        print(f"\n[TELEGRAM] Error: {e}")

    return total_pnl

if __name__ == "__main__":
    asyncio.run(run_multi_pair_backtest())
