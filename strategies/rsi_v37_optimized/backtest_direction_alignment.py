"""
RSI v3.7 IMPROVED v2 - Direction Alignment Filter
==================================================
Testing on Losing Months: Nov 2024 & Apr 2025

Key Insight from previous analysis:
- H4=BULL + SELL trades = LOSS
- H4=BEAR + SELL trades = PROFIT
- Problem: Trading AGAINST H4 trend

New Filter Strategy:
1. Direction Alignment: Only trade WITH the H4 trend, not against it
   - H4=BULL -> Only take BUY signals
   - H4=BEAR -> Only take SELL signals
   - H4=SIDEWAYS -> Take both (mean reversion)

2. Trend Strength Filter: Use ADX differently
   - ADX > 30 + H4 trending -> Trade with trend only
   - ADX < 20 -> Trade mean reversion (both directions)
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# Configuration
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '61045904'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'iy#K5L7sF')
MT5_SERVER = os.getenv('MT5_SERVER', 'FinexBisnisSolusi-Demo')
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe")

SYMBOL = "GBPUSD"
INITIAL_BALANCE = 10000.0

# Strategy Parameters
RSI_PERIOD = 10
RSI_OVERSOLD = 42
RSI_OVERBOUGHT = 58
ATR_PERIOD = 14
SL_MULT = 1.5
TP_LOW = 2.4
TP_MED = 3.0
TP_HIGH = 3.6
MAX_HOLDING_HOURS = 46
MIN_ATR_PCT = 20
MAX_ATR_PCT = 80
RISK_PER_TRADE = 0.01

# Original Filters
USE_REGIME_FILTER = True
ALLOWED_REGIMES = ['SIDEWAYS']
USE_CONSEC_LOSS_FILTER = True
CONSEC_LOSS_LIMIT = 3

# NEW: Direction Alignment Filter
USE_DIRECTION_ALIGNMENT = True
# Options: 'strict', 'flexible', 'adaptive'
# strict: Only trade with H4 trend (BULL=BUY only, BEAR=SELL only, SIDEWAYS=both)
# flexible: Block trades against strong H4 trend (allow neutral)
# adaptive: Use ADX to decide - high ADX=trend following, low ADX=mean reversion
ALIGNMENT_MODE = 'adaptive'

# ADX Settings for adaptive mode
ADX_PERIOD = 14
ADX_TRENDING = 25      # Above this = trending market
ADX_RANGING = 18       # Below this = ranging market


def connect_mt5():
    if not MT5_PASSWORD:
        print("ERROR: MT5_PASSWORD not set")
        return False
    if not mt5.initialize(path=MT5_PATH):
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return False
    return True


def get_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df


def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


def calculate_regime(df):
    """Calculate market regime"""
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')
    return df


def prepare_h1_data(df_h1, df_h4=None):
    """Prepare H1 data with all indicators"""

    # RSI
    delta = df_h1['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df_h1['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df_h1['high'] - df_h1['low'],
                    np.maximum(abs(df_h1['high'] - df_h1['close'].shift(1)),
                              abs(df_h1['low'] - df_h1['close'].shift(1))))
    df_h1['atr'] = pd.Series(tr, index=df_h1.index).rolling(ATR_PERIOD).mean()

    # ATR Percentile
    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df_h1['atr_pct'] = df_h1['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # H1 Regime
    df_h1 = calculate_regime(df_h1)

    # H1 ADX
    df_h1['adx'], df_h1['plus_di'], df_h1['minus_di'] = calculate_adx(df_h1, ADX_PERIOD)

    # H4 Data
    if df_h4 is not None:
        df_h4 = calculate_regime(df_h4)
        df_h4['h4_adx'], _, _ = calculate_adx(df_h4, ADX_PERIOD)
        df_h4['h4_regime'] = df_h4['regime']

        # Map to H1
        df_h1['h4_regime'] = 'UNKNOWN'
        df_h1['h4_adx'] = 0.0

        for idx in df_h1.index:
            h4_before = df_h4[df_h4.index <= idx]
            if len(h4_before) > 0:
                df_h1.loc[idx, 'h4_regime'] = h4_before['h4_regime'].iloc[-1]
                df_h1.loc[idx, 'h4_adx'] = h4_before['h4_adx'].iloc[-1]
    else:
        df_h1['h4_regime'] = df_h1['regime']
        df_h1['h4_adx'] = df_h1['adx']

    df_h1['hour'] = df_h1.index.hour
    df_h1['weekday'] = df_h1.index.weekday

    return df_h1.ffill().fillna(0)


def check_direction_alignment(signal, h4_regime, h4_adx, mode='adaptive'):
    """
    Check if signal direction aligns with H4 trend

    Returns: (allowed, reason)
    """
    # signal: 1 = BUY, -1 = SELL

    if mode == 'strict':
        # Only trade with H4 trend
        if h4_regime == 'BULL':
            if signal == 1:
                return True, "BUY aligned with H4 BULL"
            else:
                return False, "SELL blocked - H4 is BULL"
        elif h4_regime == 'BEAR':
            if signal == -1:
                return True, "SELL aligned with H4 BEAR"
            else:
                return False, "BUY blocked - H4 is BEAR"
        else:  # SIDEWAYS
            return True, "H4 SIDEWAYS - both directions OK"

    elif mode == 'flexible':
        # Block only strong counter-trend trades
        if h4_regime == 'BULL' and signal == -1:
            return False, "SELL blocked - H4 BULL"
        elif h4_regime == 'BEAR' and signal == 1:
            return False, "BUY blocked - H4 BEAR"
        return True, "Trade allowed"

    elif mode == 'adaptive':
        # Use ADX to decide strategy
        if h4_adx > ADX_TRENDING:
            # Strong trend - only trade with trend
            if h4_regime == 'BULL':
                if signal == 1:
                    return True, f"BUY with BULL trend (ADX={h4_adx:.0f})"
                else:
                    return False, f"SELL blocked - strong BULL (ADX={h4_adx:.0f})"
            elif h4_regime == 'BEAR':
                if signal == -1:
                    return True, f"SELL with BEAR trend (ADX={h4_adx:.0f})"
                else:
                    return False, f"BUY blocked - strong BEAR (ADX={h4_adx:.0f})"
            else:
                return True, f"SIDEWAYS with ADX={h4_adx:.0f}"

        elif h4_adx < ADX_RANGING:
            # Weak trend - mean reversion OK
            return True, f"Mean reversion OK (ADX={h4_adx:.0f} < {ADX_RANGING})"

        else:
            # Middle ground - be cautious
            if h4_regime == 'BULL' and signal == -1:
                return False, f"SELL blocked - moderate BULL (ADX={h4_adx:.0f})"
            elif h4_regime == 'BEAR' and signal == 1:
                return False, f"BUY blocked - moderate BEAR (ADX={h4_adx:.0f})"
            return True, f"Trade allowed (ADX={h4_adx:.0f})"

    return True, "Default allow"


def run_backtest(df, filter_config, target_months=None):
    """
    Run backtest with configurable filters

    filter_config: dict with keys:
        - use_regime_filter: bool
        - use_consec_filter: bool
        - use_direction_alignment: bool
        - alignment_mode: str
    """
    balance = INITIAL_BALANCE
    position = None
    consecutive_losses = 0
    trades = []

    stats = {
        'signals_generated': 0,
        'filtered_regime': 0,
        'filtered_alignment': 0,
        'filtered_consec': 0,
        'trades_executed': 0,
    }

    alignment_blocked = []

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        month_str = current_time.strftime('%Y-%m')

        if target_months and month_str not in target_months:
            continue

        if row['weekday'] >= 5:
            continue

        # Position management
        if position:
            exit_reason = None
            exit_price = None
            pnl = 0
            bars_held = i - position['entry_idx']

            if bars_held >= MAX_HOLDING_HOURS:
                exit_price = row['close']
                pnl = (exit_price - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - exit_price) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL HIT'
                    elif row['high'] >= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP HIT'
                else:
                    if row['high'] >= position['sl']:
                        exit_price = position['sl']
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL HIT'
                    elif row['low'] <= position['tp']:
                        exit_price = position['tp']
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP HIT'

            if exit_reason:
                balance += pnl
                result = 'WIN' if pnl > 0 else 'LOSS'

                if pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                pips = (exit_price - position['entry']) / 0.0001 if position['dir'] == 1 else (position['entry'] - exit_price) / 0.0001

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': 'BUY' if position['dir'] == 1 else 'SELL',
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pips': pips,
                    'pnl': pnl,
                    'result': result,
                    'exit_reason': exit_reason,
                    'h4_regime': position.get('h4_regime', ''),
                    'h4_adx': position.get('h4_adx', 0),
                    'alignment_note': position.get('alignment_note', ''),
                })

                position = None

        # Entry logic
        if not position:
            hour = row['hour']
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            # RSI signal
            rsi = row['rsi']
            signal = 0
            if rsi < RSI_OVERSOLD:
                signal = 1
            elif rsi > RSI_OVERBOUGHT:
                signal = -1

            if not signal:
                continue

            stats['signals_generated'] += 1

            skip_trade = False

            # Filter 1: Original regime filter
            if filter_config.get('use_regime_filter', True):
                if row['regime'] not in ALLOWED_REGIMES:
                    skip_trade = True
                    stats['filtered_regime'] += 1

            # Filter 2: Direction alignment (NEW)
            alignment_note = ""
            if not skip_trade and filter_config.get('use_direction_alignment', False):
                mode = filter_config.get('alignment_mode', 'adaptive')
                allowed, reason = check_direction_alignment(
                    signal,
                    row['h4_regime'],
                    row['h4_adx'],
                    mode
                )
                alignment_note = reason
                if not allowed:
                    skip_trade = True
                    stats['filtered_alignment'] += 1
                    alignment_blocked.append({
                        'time': current_time,
                        'signal': 'BUY' if signal == 1 else 'SELL',
                        'h4_regime': row['h4_regime'],
                        'h4_adx': row['h4_adx'],
                        'reason': reason,
                    })

            # Filter 3: Consecutive loss filter
            if not skip_trade and filter_config.get('use_consec_filter', True):
                if consecutive_losses >= CONSEC_LOSS_LIMIT:
                    skip_trade = True
                    consecutive_losses = 0
                    stats['filtered_consec'] += 1

            if skip_trade:
                continue

            # Execute trade
            entry = row['close']
            atr = row['atr'] if row['atr'] > 0 else entry * 0.002

            if atr_pct < 40:
                tp_mult = TP_LOW
            elif atr_pct > 60:
                tp_mult = TP_HIGH
            else:
                tp_mult = TP_MED

            if 12 <= hour < 16:
                tp_mult += 0.35

            if signal == 1:
                sl = entry - atr * SL_MULT
                tp = entry + atr * tp_mult
            else:
                sl = entry + atr * SL_MULT
                tp = entry - atr * tp_mult

            risk = balance * RISK_PER_TRADE
            size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

            position = {
                'dir': signal,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'size': size,
                'entry_idx': i,
                'entry_time': current_time,
                'h4_regime': row['h4_regime'],
                'h4_adx': row['h4_adx'],
                'alignment_note': alignment_note,
            }

            stats['trades_executed'] += 1

    return trades, balance, stats, alignment_blocked


def print_results(name, trades, balance, stats=None):
    """Print backtest results"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    if len(trades) == 0:
        print("No trades executed")
        if stats:
            print(f"Signals generated: {stats.get('signals_generated', 0)}")
        return

    df = pd.DataFrame(trades)
    wins = len(df[df['result'] == 'WIN'])
    losses = len(df[df['result'] == 'LOSS'])
    total_pnl = df['pnl'].sum()
    win_rate = wins / len(df) * 100

    print(f"Trades: {len(df)}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P/L: ${total_pnl:+,.2f}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Return: {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:+.2f}%")

    # SL/TP stats
    sl_hits = len(df[df['exit_reason'] == 'SL HIT'])
    tp_hits = len(df[df['exit_reason'] == 'TP HIT'])
    print(f"\nSL Hit Rate: {sl_hits/len(df)*100:.1f}%")
    print(f"TP Hit Rate: {tp_hits/len(df)*100:.1f}%")

    # Direction breakdown
    buys = df[df['direction'] == 'BUY']
    sells = df[df['direction'] == 'SELL']
    print(f"\nBY DIRECTION:")
    if len(buys) > 0:
        buy_wins = len(buys[buys['result'] == 'WIN'])
        print(f"  BUY:  {len(buys)} trades, {buy_wins}W ({buy_wins/len(buys)*100:.0f}%), ${buys['pnl'].sum():+,.2f}")
    if len(sells) > 0:
        sell_wins = len(sells[sells['result'] == 'WIN'])
        print(f"  SELL: {len(sells)} trades, {sell_wins}W ({sell_wins/len(sells)*100:.0f}%), ${sells['pnl'].sum():+,.2f}")

    if stats:
        print(f"\nFILTER STATS:")
        print(f"  Signals generated: {stats.get('signals_generated', 0)}")
        print(f"  Filtered by regime: {stats.get('filtered_regime', 0)}")
        print(f"  Filtered by alignment: {stats.get('filtered_alignment', 0)}")
        print(f"  Filtered by consec loss: {stats.get('filtered_consec', 0)}")


def main():
    print("=" * 70)
    print("RSI v3.7 IMPROVED v2 - Direction Alignment Filter")
    print("=" * 70)
    print("\nKey Insight: Don't trade AGAINST the H4 trend!")
    print("  - H4=BULL + SELL = BAD (counter-trend)")
    print("  - H4=BEAR + BUY = BAD (counter-trend)")
    print("  - H4=SIDEWAYS = OK for mean reversion")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

        df_h1 = get_data(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
        df_h4 = get_data(SYMBOL, mt5.TIMEFRAME_H4, start_date, end_date)

        print(f"H1: {len(df_h1)} bars | H4: {len(df_h4)} bars")

        df_h1 = prepare_h1_data(df_h1, df_h4)

        losing_months = ['2024-11', '2025-04']

        print(f"\n{'#'*70}")
        print(f"TESTING ON LOSING MONTHS: {', '.join(losing_months)}")
        print(f"{'#'*70}")

        # ============================================
        # Test 1: Original (baseline)
        # ============================================
        config_original = {
            'use_regime_filter': True,
            'use_consec_filter': True,
            'use_direction_alignment': False,
        }
        trades_orig, balance_orig, stats_orig, _ = run_backtest(
            df_h1, config_original, losing_months
        )
        print_results("ORIGINAL (Baseline)", trades_orig, balance_orig, stats_orig)

        # ============================================
        # Test 2: Strict Direction Alignment
        # ============================================
        config_strict = {
            'use_regime_filter': False,  # Disable original regime filter
            'use_consec_filter': True,
            'use_direction_alignment': True,
            'alignment_mode': 'strict',
        }
        trades_strict, balance_strict, stats_strict, blocked_strict = run_backtest(
            df_h1, config_strict, losing_months
        )
        print_results("STRICT Alignment (no regime filter)", trades_strict, balance_strict, stats_strict)

        # ============================================
        # Test 3: Flexible Direction Alignment
        # ============================================
        config_flexible = {
            'use_regime_filter': False,
            'use_consec_filter': True,
            'use_direction_alignment': True,
            'alignment_mode': 'flexible',
        }
        trades_flex, balance_flex, stats_flex, blocked_flex = run_backtest(
            df_h1, config_flexible, losing_months
        )
        print_results("FLEXIBLE Alignment", trades_flex, balance_flex, stats_flex)

        # ============================================
        # Test 4: Adaptive (ADX-based)
        # ============================================
        config_adaptive = {
            'use_regime_filter': False,
            'use_consec_filter': True,
            'use_direction_alignment': True,
            'alignment_mode': 'adaptive',
        }
        trades_adaptive, balance_adaptive, stats_adaptive, blocked_adaptive = run_backtest(
            df_h1, config_adaptive, losing_months
        )
        print_results("ADAPTIVE (ADX-based)", trades_adaptive, balance_adaptive, stats_adaptive)

        # ============================================
        # Test 5: Combined (Regime + Alignment)
        # ============================================
        config_combined = {
            'use_regime_filter': True,
            'use_consec_filter': True,
            'use_direction_alignment': True,
            'alignment_mode': 'flexible',
        }
        trades_combined, balance_combined, stats_combined, blocked_combined = run_backtest(
            df_h1, config_combined, losing_months
        )
        print_results("COMBINED (Regime + Flexible Align)", trades_combined, balance_combined, stats_combined)

        # ============================================
        # COMPARISON SUMMARY
        # ============================================
        print("\n" + "#" * 70)
        print("COMPARISON SUMMARY - LOSING MONTHS (Nov 2024 + Apr 2025)")
        print("#" * 70)

        results = [
            ("Original", trades_orig, balance_orig),
            ("Strict Align", trades_strict, balance_strict),
            ("Flexible Align", trades_flex, balance_flex),
            ("Adaptive (ADX)", trades_adaptive, balance_adaptive),
            ("Combined", trades_combined, balance_combined),
        ]

        print(f"\n{'Config':<18} {'Trades':>7} {'Wins':>5} {'Win%':>7} {'SL%':>6} {'P/L':>12} {'Return':>9}")
        print("-" * 70)

        best_pnl = -999999
        best_name = ""

        for name, trades, balance in results:
            if len(trades) > 0:
                df = pd.DataFrame(trades)
                wins = len(df[df['result'] == 'WIN'])
                win_rate = wins / len(df) * 100
                sl_rate = len(df[df['exit_reason'] == 'SL HIT']) / len(df) * 100
                pnl = df['pnl'].sum()
                ret = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

                if pnl > best_pnl:
                    best_pnl = pnl
                    best_name = name

                print(f"{name:<18} {len(df):>7} {wins:>5} {win_rate:>6.1f}% {sl_rate:>5.1f}% ${pnl:>+10,.2f} {ret:>+8.2f}%")
            else:
                print(f"{name:<18} {'0':>7} {'-':>5} {'-':>7} {'-':>6} {'-':>12} {'-':>9}")

        # Best result
        orig_pnl = pd.DataFrame(trades_orig)['pnl'].sum() if trades_orig else 0

        print(f"\n{'='*70}")
        print(f"BEST RESULT: {best_name}")
        print(f"Improvement vs Original: ${best_pnl - orig_pnl:+,.2f}")
        if orig_pnl < 0:
            reduction = ((orig_pnl - best_pnl) / abs(orig_pnl)) * 100
            print(f"Loss Reduction: {reduction:.1f}%")
        print(f"{'='*70}")

        # ============================================
        # Analyze blocked trades
        # ============================================
        if blocked_adaptive:
            print("\n" + "=" * 70)
            print("TRADES BLOCKED BY ADAPTIVE FILTER")
            print("=" * 70)

            # Check what would have happened
            blocked_df = pd.DataFrame(blocked_adaptive)
            print(f"\nTotal signals blocked: {len(blocked_df)}")

            # Group by H4 regime
            print(f"\nBlocked by H4 regime:")
            for regime in blocked_df['h4_regime'].unique():
                subset = blocked_df[blocked_df['h4_regime'] == regime]
                print(f"  H4={regime}: {len(subset)} blocked")

            print(f"\nSample blocked trades:")
            for _, row in blocked_df.head(5).iterrows():
                print(f"  {row['time'].strftime('%Y-%m-%d %H:%M')} {row['signal']} blocked")
                print(f"    H4={row['h4_regime']}, ADX={row['h4_adx']:.1f}")
                print(f"    Reason: {row['reason']}")

        # ============================================
        # Test on ALL months (full backtest)
        # ============================================
        print("\n" + "#" * 70)
        print("FULL BACKTEST (All 2024-2025 data)")
        print("#" * 70)

        # Original on all data
        trades_orig_all, balance_orig_all, _, _ = run_backtest(df_h1, config_original, None)
        print_results("Original (Full)", trades_orig_all, balance_orig_all)

        # Best config on all data
        trades_best_all, balance_best_all, _, _ = run_backtest(df_h1, config_adaptive, None)
        print_results("Adaptive (Full)", trades_best_all, balance_best_all)

        # Quick comparison
        if trades_orig_all and trades_best_all:
            orig_all_pnl = pd.DataFrame(trades_orig_all)['pnl'].sum()
            best_all_pnl = pd.DataFrame(trades_best_all)['pnl'].sum()

            print(f"\n{'='*70}")
            print("FULL BACKTEST COMPARISON")
            print(f"{'='*70}")
            print(f"Original P/L: ${orig_all_pnl:+,.2f}")
            print(f"Adaptive P/L: ${best_all_pnl:+,.2f}")
            print(f"Difference:   ${best_all_pnl - orig_all_pnl:+,.2f}")

    finally:
        mt5.shutdown()
        print("\n" + "=" * 70)
        print("Backtest complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
