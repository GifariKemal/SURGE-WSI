"""
Signal Debug Logger
====================
Logs detailed signal detection data for comparison with MQL5 EA.
Outputs CSV file that can be compared with MQL5 logs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

# Configuration
SYMBOL = "GBPUSD"
INITIAL_BALANCE = 50_000.0
PIP_SIZE = 0.0001
MIN_ATR = 8.0
MAX_ATR = 25.0

# Quality thresholds
BASE_QUALITY = 65
MIN_QUALITY = 60

# Session filter
SKIP_ORDER_BLOCK_HOURS = [8, 16]
SKIP_EMA_PULLBACK_HOURS = [13, 14]

# Day multipliers (Monday=0 in Python)
DAY_MULTIPLIERS = {0: 1.0, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.0, 6: 0.0}

# Hour multipliers
HOUR_MULTIPLIERS = {
    7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0,
    13: 0.9, 14: 0.9, 15: 1.0, 16: 1.0, 17: 0.8
}

# Session definitions
LONDON_START, LONDON_END = 8, 10
NY_START, NY_END = 13, 17


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr / PIP_SIZE


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(period).mean()
    return adx


def check_ema_pullback(df: pd.DataFrame, idx: int, atr_pips: float,
                       ema20: pd.Series, rsi: pd.Series, adx: pd.Series,
                       min_quality: float) -> Dict:
    """
    Check EMA Pullback signal and return detailed info.
    """
    result = {
        'detected': False,
        'direction': None,
        'quality': 0,
        'body_ratio': 0,
        'adx_value': 0,
        'rsi_value': 0,
        'ema_distance': 0,
        'touch_quality': 0,
        'skip_reason': None
    }

    if idx < 3:
        result['skip_reason'] = 'idx < 3'
        return result

    current = df.iloc[idx]
    prev = df.iloc[idx - 1]

    o, h, l, c = current['Open'], current['High'], current['Low'], current['Close']

    body = abs(c - o)
    total_range = h - l

    if total_range < PIP_SIZE:
        result['skip_reason'] = 'range too small'
        return result

    body_ratio = body / total_range
    result['body_ratio'] = round(body_ratio, 4)

    if body_ratio < 0.4:
        result['skip_reason'] = f'body_ratio {body_ratio:.3f} < 0.4'
        return result

    adx_val = adx.iloc[idx] if not pd.isna(adx.iloc[idx]) else 0
    result['adx_value'] = round(adx_val, 2)

    if adx_val < 20:
        result['skip_reason'] = f'ADX {adx_val:.1f} < 20'
        return result

    rsi_val = rsi.iloc[idx] if not pd.isna(rsi.iloc[idx]) else 50
    result['rsi_value'] = round(rsi_val, 2)

    if rsi_val < 30 or rsi_val > 70:
        result['skip_reason'] = f'RSI {rsi_val:.1f} outside 30-70'
        return result

    ema = ema20.iloc[idx]
    ema_dist_pips = abs(c - ema) / PIP_SIZE
    result['ema_distance'] = round(ema_dist_pips, 2)

    # Check BUY signal (price touched EMA from above)
    if l <= ema <= h and c > ema:
        touch_quality = 30 if l <= ema else 15
        adx_q = min(25, (adx_val - 15) * 1.5)
        rsi_q = 25 if abs(50 - rsi_val) < 20 else 15
        body_q = min(20, body_ratio * 30)
        quality = min(100, max(55, touch_quality + adx_q + rsi_q + body_q))

        result['touch_quality'] = round(touch_quality, 2)
        result['quality'] = round(quality, 2)

        if quality >= min_quality:
            result['detected'] = True
            result['direction'] = 'BUY'
        else:
            result['skip_reason'] = f'quality {quality:.1f} < {min_quality}'
        return result

    # Check SELL signal (price touched EMA from below)
    if l <= ema <= h and c < ema:
        touch_quality = 30 if h >= ema else 15
        adx_q = min(25, (adx_val - 15) * 1.5)
        rsi_q = 25 if abs(50 - rsi_val) < 20 else 15
        body_q = min(20, body_ratio * 30)
        quality = min(100, max(55, touch_quality + adx_q + rsi_q + body_q))

        result['touch_quality'] = round(touch_quality, 2)
        result['quality'] = round(quality, 2)

        if quality >= min_quality:
            result['detected'] = True
            result['direction'] = 'SELL'
        else:
            result['skip_reason'] = f'quality {quality:.1f} < {min_quality}'
        return result

    result['skip_reason'] = 'no EMA touch'
    return result


def check_order_block(df: pd.DataFrame, idx: int, min_quality: float) -> Dict:
    """
    Check Order Block signal and return detailed info.
    """
    result = {
        'detected': False,
        'direction': None,
        'quality': 0,
        'body_ratio': 0,
        'skip_reason': None
    }

    if idx < 5:
        result['skip_reason'] = 'idx < 5'
        return result

    current = df.iloc[idx]
    next1 = df.iloc[idx + 1] if idx + 1 < len(df) else None

    if next1 is None:
        result['skip_reason'] = 'no next bar'
        return result

    o_curr, h_curr, l_curr, c_curr = current['Open'], current['High'], current['Low'], current['Close']
    o_next, h_next, l_next, c_next = next1['Open'], next1['High'], next1['Low'], next1['Close']

    next_body = abs(c_next - o_next)
    next_range = h_next - l_next

    if next_range < PIP_SIZE:
        result['skip_reason'] = 'next range too small'
        return result

    body_ratio = next_body / next_range
    result['body_ratio'] = round(body_ratio, 4)

    # Bullish OB: bearish candle followed by strong bullish engulf
    if c_curr < o_curr:  # Current is bearish
        if c_next > o_next and body_ratio > 0.55 and c_next > h_curr:
            quality = body_ratio * 100
            result['quality'] = round(quality, 2)

            if quality >= min_quality:
                result['detected'] = True
                result['direction'] = 'BUY'
            else:
                result['skip_reason'] = f'quality {quality:.1f} < {min_quality}'
            return result

    # Bearish OB: bullish candle followed by strong bearish engulf
    if c_curr > o_curr:  # Current is bullish
        if c_next < o_next and body_ratio > 0.55 and c_next < l_curr:
            quality = body_ratio * 100
            result['quality'] = round(quality, 2)

            if quality >= min_quality:
                result['detected'] = True
                result['direction'] = 'SELL'
            else:
                result['skip_reason'] = f'quality {quality:.1f} < {min_quality}'
            return result

    result['skip_reason'] = 'no OB pattern'
    return result


def check_entry_trigger(current: pd.Series, prev: pd.Series, direction: str) -> Tuple[bool, str]:
    """Check for entry trigger pattern."""
    o, h, l, c = current['Open'], current['High'], current['Low'], current['Close']
    po, ph, pl, pc = prev['Open'], prev['High'], prev['Low'], prev['Close']

    body = abs(c - o)
    range_ = h - l
    body_ratio = body / range_ if range_ > 0 else 0

    if direction == 'BUY':
        # Bullish engulfing
        if c > o and c > pc and o < po and body_ratio > 0.6:
            return True, 'ENGULF'
        # Hammer
        lower_wick = min(o, c) - l
        if c > o and lower_wick > body * 1.5:
            return True, 'HAMMER'
        # Momentum
        if c > o and c > h - (range_ * 0.3):
            return True, 'MOMENTUM'
        # Higher high
        if h > ph:
            return True, 'HIGHER_HIGH'
    else:
        # Bearish engulfing
        if c < o and c < pc and o > po and body_ratio > 0.6:
            return True, 'ENGULF'
        # Shooting star
        upper_wick = h - max(o, c)
        if c < o and upper_wick > body * 1.5:
            return True, 'SHOOTING_STAR'
        # Momentum
        if c < o and c < l + (range_ * 0.3):
            return True, 'MOMENTUM'
        # Lower low
        if l < pl:
            return True, 'LOWER_LOW'

    return False, ''


def get_regime(df: pd.DataFrame, idx: int) -> str:
    """Determine market regime."""
    if idx < 20:
        return 'NEUTRAL'

    close = df['Close'].iloc[idx]
    ema20 = df['Close'].iloc[idx-20:idx].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df['Close'].iloc[idx-50:idx].ewm(span=50, adjust=False).mean().iloc[-1] if idx >= 50 else ema20

    if close > ema20 > ema50:
        return 'BULLISH'
    elif close < ema20 < ema50:
        return 'BEARISH'
    return 'NEUTRAL'


def run_signal_logger():
    """Main function to log all signals."""

    # Load MT5 data
    data_path = Path(__file__).parent.parent / "data" / "mt5_export" / "GBPUSD_H1_latest.csv"
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Rename columns
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })

    # Filter date range
    start = datetime(2025, 1, 1)
    end = datetime(2026, 1, 31)
    df = df[(df.index >= start) & (df.index <= end)]

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Calculate indicators
    print("Calculating indicators...")
    atr_series = calculate_atr(df)
    ema20_series = calculate_ema(df['Close'], 20)
    ema50_series = calculate_ema(df['Close'], 50)
    rsi_series = calculate_rsi(df['Close'])
    adx_series = calculate_adx(df)

    # Prepare output
    signals = []

    print("Scanning for signals...")

    for i in range(50, len(df) - 1):
        current_bar = df.iloc[i]
        current_time = df.index[i]

        # Skip weekends
        if current_time.weekday() >= 5:
            continue

        hour = current_time.hour
        day = current_time.weekday()

        # Get ATR
        current_atr = atr_series.iloc[i]
        if pd.isna(current_atr) or current_atr < MIN_ATR or current_atr > MAX_ATR:
            continue

        # Get regime
        regime = get_regime(df, i)

        # Get multipliers
        day_mult = DAY_MULTIPLIERS.get(day, 0.5)
        hour_mult = HOUR_MULTIPLIERS.get(hour, 0.0)

        if day_mult == 0.0 or hour_mult == 0.0:
            continue

        # Determine session
        if LONDON_START <= hour <= LONDON_END:
            session = 'london'
        elif NY_START <= hour <= NY_END:
            session = 'newyork'
        else:
            continue

        # Base quality threshold
        min_quality = BASE_QUALITY

        # Check EMA Pullback (skip hours 13, 14)
        ema_result = None
        if hour not in SKIP_EMA_PULLBACK_HOURS:
            ema_result = check_ema_pullback(df, i, current_atr, ema20_series,
                                            rsi_series, adx_series, min_quality)

        # Check Order Block (skip hours 8, 16)
        ob_result = None
        if hour not in SKIP_ORDER_BLOCK_HOURS:
            ob_result = check_order_block(df, i, min_quality)

        # Determine which signal to use
        signal_type = None
        signal_dir = None
        signal_quality = 0
        entry_type = None

        # EMA Pullback first
        if ema_result and ema_result['detected']:
            # Check regime alignment
            if (ema_result['direction'] == 'BUY' and regime == 'BULLISH') or \
               (ema_result['direction'] == 'SELL' and regime == 'BEARISH'):
                # Check entry trigger
                prev_bar = df.iloc[i-1]
                has_trigger, trigger_type = check_entry_trigger(current_bar, prev_bar, ema_result['direction'])
                if has_trigger:
                    signal_type = 'EMA_PULLBACK'
                    signal_dir = ema_result['direction']
                    signal_quality = ema_result['quality']
                    entry_type = trigger_type

        # Order Block if no EMA signal
        if not signal_type and ob_result and ob_result['detected']:
            if (ob_result['direction'] == 'BUY' and regime == 'BULLISH') or \
               (ob_result['direction'] == 'SELL' and regime == 'BEARISH'):
                prev_bar = df.iloc[i-1]
                has_trigger, trigger_type = check_entry_trigger(current_bar, prev_bar, ob_result['direction'])
                if has_trigger:
                    signal_type = 'ORDER_BLOCK'
                    signal_dir = ob_result['direction']
                    signal_quality = ob_result['quality']
                    entry_type = trigger_type

        # Log signal info
        if signal_type or (ema_result and ema_result['quality'] > 50) or (ob_result and ob_result['quality'] > 50):
            signals.append({
                'timestamp': current_time.strftime('%Y.%m.%d %H:%M'),
                'hour': hour,
                'day': day,
                'session': session,
                'regime': regime,
                'atr': round(current_atr, 2),
                'close': round(current_bar['Close'], 5),
                'ema20': round(ema20_series.iloc[i], 5),
                'rsi': round(rsi_series.iloc[i], 2) if not pd.isna(rsi_series.iloc[i]) else 0,
                'adx': round(adx_series.iloc[i], 2) if not pd.isna(adx_series.iloc[i]) else 0,
                # EMA Pullback details
                'ema_detected': ema_result['detected'] if ema_result else False,
                'ema_dir': ema_result['direction'] if ema_result else '',
                'ema_quality': ema_result['quality'] if ema_result else 0,
                'ema_body_ratio': ema_result['body_ratio'] if ema_result else 0,
                'ema_skip': ema_result['skip_reason'] if ema_result else 'hour_skip',
                # Order Block details
                'ob_detected': ob_result['detected'] if ob_result else False,
                'ob_dir': ob_result['direction'] if ob_result else '',
                'ob_quality': ob_result['quality'] if ob_result else 0,
                'ob_body_ratio': ob_result['body_ratio'] if ob_result else 0,
                'ob_skip': ob_result['skip_reason'] if ob_result else 'hour_skip',
                # Final signal
                'signal_type': signal_type or '',
                'signal_dir': signal_dir or '',
                'signal_quality': signal_quality,
                'entry_type': entry_type or '',
                'executed': 1 if signal_type else 0
            })

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "signal_debug_python.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_signals = pd.DataFrame(signals)
    df_signals.to_csv(output_path, index=False)

    print(f"\nSignal log saved to: {output_path}")
    print(f"Total potential signals logged: {len(signals)}")
    print(f"Executed signals: {df_signals['executed'].sum()}")

    # Summary by type
    executed = df_signals[df_signals['executed'] == 1]
    print(f"\nBy signal type:")
    print(executed['signal_type'].value_counts().to_string())

    # First 10 executed signals
    print(f"\nFirst 10 executed signals:")
    print(executed[['timestamp', 'signal_type', 'signal_dir', 'signal_quality', 'entry_type']].head(10).to_string())


if __name__ == "__main__":
    run_signal_logger()
