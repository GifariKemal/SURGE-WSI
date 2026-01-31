"""
Research-Based RSI Improvements Test
=====================================

Ideas from trading literature and academic research:
1. RSI2 (2-period RSI) - Larry Connors style
2. Bollinger Bands + RSI - Price at BB + RSI confirmation
3. RSI + Price Position - Near swing high/low
4. RSI Slope/Momentum - RSI turning before entry
5. Stochastic + RSI - Double confirmation
6. MACD Histogram - Momentum confirmation
7. RSI Exit at 50 - Exit when RSI crosses mean
8. Williams %R - Additional oversold/overbought
9. CCI (Commodity Channel Index) - Mean reversion signal
10. Price Rate of Change (ROC) - Momentum filter

Sources:
- MQL5: RSI2 Mean-Reversion Strategies
- Trading Rush: Bollinger Bands + RSI Backtested
- QuantifiedStrategies: RSI Trading Strategy

Base: RSI 42/58 + Vol Filter 20-80 + Dynamic TP = +493.1%
"""

import pandas as pd
import numpy as np
import psycopg2
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}


def load_and_prepare_data():
    """Load data and calculate all indicators"""
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)
    conn.close()

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # ========== RSI Variants ==========
    # RSI(10) - Current
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi10'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # RSI(2) - Larry Connors style
    gain2 = delta.where(delta > 0, 0).rolling(2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
    df['rsi2'] = 100 - (100 / (1 + gain2 / (loss2 + 1e-10)))

    # RSI(14) - Traditional
    gain14 = delta.where(delta > 0, 0).rolling(14).mean()
    loss14 = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi14'] = 100 - (100 / (1 + gain14 / (loss14 + 1e-10)))

    # RSI Slope (momentum)
    df['rsi_slope'] = df['rsi10'].diff(3)
    df['rsi_turning_up'] = (df['rsi10'] > df['rsi10'].shift(1)) & (df['rsi10'].shift(1) <= df['rsi10'].shift(2))
    df['rsi_turning_down'] = (df['rsi10'] < df['rsi10'].shift(1)) & (df['rsi10'].shift(1) >= df['rsi10'].shift(2))

    # ========== ATR ==========
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # ========== Bollinger Bands ==========
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['at_bb_lower'] = df['close'] <= df['bb_lower']
    df['at_bb_upper'] = df['close'] >= df['bb_upper']

    # ========== Stochastic ==========
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ========== MACD ==========
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_rising'] = df['macd_hist'] > df['macd_hist'].shift(1)
    df['macd_hist_falling'] = df['macd_hist'] < df['macd_hist'].shift(1)

    # ========== Williams %R ==========
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)

    # ========== CCI (Commodity Channel Index) ==========
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

    # ========== ROC (Rate of Change) ==========
    df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100

    # ========== Moving Averages ==========
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['above_sma200'] = df['close'] > df['sma_200']
    df['below_sma200'] = df['close'] < df['sma_200']

    # ========== Swing High/Low ==========
    df['swing_high'] = df['high'].rolling(20).max()
    df['swing_low'] = df['low'].rolling(20).min()
    df['near_swing_low'] = (df['close'] - df['swing_low']) / df['atr'] < 1.5
    df['near_swing_high'] = (df['swing_high'] - df['close']) / df['atr'] < 1.5

    # ========== Volume ==========
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['high_volume'] = df['volume'] > df['vol_sma'] * 1.5

    # ========== Time Features ==========
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)


def run_backtest(
    df,
    # Base parameters (v3.4 current best)
    rsi_col='rsi10',
    rsi_oversold=42,
    rsi_overbought=58,
    min_atr_pct=20,
    max_atr_pct=80,
    sl_mult=1.5,
    tp_low=2.4,
    tp_med=3.0,
    tp_high=3.6,
    # Additional filters to test
    require_bb_touch=False,  # Price at Bollinger Band
    require_stoch_confirm=False,  # Stochastic also oversold/overbought
    stoch_os=20,
    stoch_ob=80,
    require_macd_confirm=False,  # MACD histogram direction
    require_williams_confirm=False,  # Williams %R confirmation
    williams_os=-80,
    williams_ob=-20,
    require_cci_confirm=False,  # CCI confirmation
    cci_os=-100,
    cci_ob=100,
    require_rsi_turning=False,  # RSI starting to turn
    require_near_swing=False,  # Near swing high/low
    require_high_volume=False,  # High volume
    use_rsi2=False,  # Use RSI(2) instead of RSI(10)
    rsi2_os=10,
    rsi2_ob=90,
    exit_at_rsi_50=False,  # Exit when RSI crosses 50
):
    """Run backtest with various filters"""
    balance = 10000.0
    wins = losses = 0
    position = None
    yearly_pnl = {}

    rsi_data = df['rsi2'] if use_rsi2 else df[rsi_col]
    os_level = rsi2_os if use_rsi2 else rsi_oversold
    ob_level = rsi2_ob if use_rsi2 else rsi_overbought

    for i in range(200, len(df)):
        row = df.iloc[i]

        if row['weekday'] >= 5:
            continue

        # Check position
        if position:
            current_rsi = rsi_data.iloc[i]
            closed = False
            pnl = 0

            # RSI 50 exit
            if exit_at_rsi_50:
                if position['dir'] == 1 and current_rsi >= 50:
                    pnl = (row['close'] - position['entry']) * position['size']
                    closed = True
                elif position['dir'] == -1 and current_rsi <= 50:
                    pnl = (position['entry'] - row['close']) * position['size']
                    closed = True

            # Normal SL/TP exit
            if not closed:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        losses += 1
                        closed = True
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        wins += 1
                        closed = True
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        losses += 1
                        closed = True
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        wins += 1
                        closed = True

            if closed:
                if exit_at_rsi_50 and pnl > 0:
                    wins += 1
                elif exit_at_rsi_50 and pnl <= 0:
                    losses += 1

                balance += pnl
                year = df.index[i].year
                yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                position = None

        # New signal
        if position is None:
            if row['hour'] < 7 or row['hour'] >= 22:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                continue

            rsi = rsi_data.iloc[i]
            signal = 0

            if rsi < os_level:
                signal = 1  # BUY
            elif rsi > ob_level:
                signal = -1  # SELL

            if signal == 0:
                continue

            # Apply additional filters
            if require_bb_touch:
                if signal == 1 and not row['at_bb_lower']:
                    continue
                if signal == -1 and not row['at_bb_upper']:
                    continue

            if require_stoch_confirm:
                if signal == 1 and row['stoch_k'] >= stoch_os:
                    continue
                if signal == -1 and row['stoch_k'] <= stoch_ob:
                    continue

            if require_macd_confirm:
                if signal == 1 and not row['macd_hist_rising']:
                    continue
                if signal == -1 and not row['macd_hist_falling']:
                    continue

            if require_williams_confirm:
                if signal == 1 and row['williams_r'] >= williams_os:
                    continue
                if signal == -1 and row['williams_r'] <= williams_ob:
                    continue

            if require_cci_confirm:
                if signal == 1 and row['cci'] >= cci_os:
                    continue
                if signal == -1 and row['cci'] <= cci_ob:
                    continue

            if require_rsi_turning:
                if signal == 1 and not row['rsi_turning_up']:
                    continue
                if signal == -1 and not row['rsi_turning_down']:
                    continue

            if require_near_swing:
                if signal == 1 and not row['near_swing_low']:
                    continue
                if signal == -1 and not row['near_swing_high']:
                    continue

            if require_high_volume:
                if not row['high_volume']:
                    continue

            # Calculate entry
            entry = row['close']
            atr = row['atr'] if row['atr'] > 0 else entry * 0.002

            tp_mult = tp_low if atr_pct < 40 else (tp_high if atr_pct > 60 else tp_med)

            if signal == 1:
                sl = entry - atr * sl_mult
                tp = entry + atr * tp_mult
            else:
                sl = entry + atr * sl_mult
                tp = entry - atr * tp_mult

            risk = balance * 0.01
            size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

            position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size}

    trades = wins + losses
    wr = wins / trades * 100 if trades else 0
    ret = (balance - 10000) / 100
    prof_years = sum(1 for v in yearly_pnl.values() if v > 0)

    return {
        'return': ret,
        'trades': trades,
        'wr': wr,
        'prof_years': prof_years,
        'yearly': yearly_pnl
    }


def main():
    print('=' * 100)
    print('RESEARCH-BASED RSI IMPROVEMENTS TEST')
    print('=' * 100)

    print('\n[1/2] Loading data and calculating indicators...')
    df = load_and_prepare_data()
    print(f'      Loaded {len(df):,} bars with {len(df.columns)} indicators')

    print('\n[2/2] Running backtests...')

    # Baseline
    baseline = run_backtest(df, rsi_oversold=42, rsi_overbought=58)
    print(f'\n      BASELINE (RSI 42/58): +{baseline["return"]:.1f}% | {baseline["trades"]} trades | WR {baseline["wr"]:.1f}%')

    results = []

    # =====================================================================
    # TEST INDIVIDUAL FILTERS
    # =====================================================================
    tests = [
        ('Baseline RSI 42/58', {}),

        # RSI Variants
        ('RSI2 (2-period) 10/90', {'use_rsi2': True, 'rsi2_os': 10, 'rsi2_ob': 90}),
        ('RSI2 (2-period) 5/95', {'use_rsi2': True, 'rsi2_os': 5, 'rsi2_ob': 95}),

        # Bollinger Bands
        ('+ BB Touch Required', {'require_bb_touch': True}),

        # Stochastic
        ('+ Stochastic < 20 / > 80', {'require_stoch_confirm': True}),
        ('+ Stochastic < 30 / > 70', {'require_stoch_confirm': True, 'stoch_os': 30, 'stoch_ob': 70}),

        # MACD
        ('+ MACD Histogram Rising/Falling', {'require_macd_confirm': True}),

        # Williams %R
        ('+ Williams %R < -80 / > -20', {'require_williams_confirm': True}),

        # CCI
        ('+ CCI < -100 / > 100', {'require_cci_confirm': True}),
        ('+ CCI < -50 / > 50', {'require_cci_confirm': True, 'cci_os': -50, 'cci_ob': 50}),

        # RSI Turning
        ('+ RSI Turning (momentum)', {'require_rsi_turning': True}),

        # Near Swing
        ('+ Near Swing High/Low', {'require_near_swing': True}),

        # High Volume
        ('+ High Volume (> 1.5x avg)', {'require_high_volume': True}),

        # Exit at RSI 50
        ('+ Exit at RSI 50', {'exit_at_rsi_50': True}),
    ]

    print('\n' + '-' * 100)
    print('INDIVIDUAL FILTER TESTS')
    print('-' * 100)
    print(f"\n{'Filter':<40} {'Return':>10} {'Diff':>10} {'Trades':>8} {'WR':>8} {'Years':>8}")
    print('-' * 90)

    for name, params in tests:
        result = run_backtest(df, rsi_oversold=42, rsi_overbought=58, **params)
        diff = result['return'] - baseline['return']
        improved = diff > 0

        marker = ' ++ BETTER' if diff > 10 else (' + better' if diff > 0 else '')
        results.append((name, result, diff, improved))

        print(f"{name:<40} {result['return']:>+9.1f}% {diff:>+9.1f}% {result['trades']:>8} {result['wr']:>7.1f}% {result['prof_years']:>6}/6{marker}")

    # =====================================================================
    # TEST COMBINATIONS OF IMPROVING FILTERS
    # =====================================================================
    improving = [(n, r, d) for n, r, d, i in results if i and d > 5]

    if improving:
        print('\n' + '-' * 100)
        print('COMBINATION TESTS (filters that improved > 5%)')
        print('-' * 100)

        # Try combining top improvements
        combo_tests = []

        # Find which filters improved
        improved_filters = {
            'stoch': any('Stochastic' in n for n, _, d in improving),
            'macd': any('MACD' in n for n, _, d in improving),
            'williams': any('Williams' in n for n, _, d in improving),
            'cci': any('CCI' in n for n, _, d in improving),
            'rsi_turning': any('Turning' in n for n, _, d in improving),
            'swing': any('Swing' in n for n, _, d in improving),
            'volume': any('Volume' in n for n, _, d in improving),
            'rsi_50_exit': any('Exit at RSI 50' in n for n, _, d in improving),
        }

        print(f"\n{'Combination':<50} {'Return':>10} {'Trades':>8} {'WR':>8}")
        print('-' * 80)

        # Test combinations of what worked
        if improved_filters['stoch']:
            result = run_backtest(df, rsi_oversold=42, rsi_overbought=58,
                                 require_stoch_confirm=True, stoch_os=30, stoch_ob=70)
            print(f"{'Stochastic 30/70':<50} {result['return']:>+9.1f}% {result['trades']:>8} {result['wr']:>7.1f}%")

        if improved_filters['cci']:
            result = run_backtest(df, rsi_oversold=42, rsi_overbought=58,
                                 require_cci_confirm=True, cci_os=-50, cci_ob=50)
            print(f"{'CCI -50/50':<50} {result['return']:>+9.1f}% {result['trades']:>8} {result['wr']:>7.1f}%")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)

    print(f'\nBASELINE: +{baseline["return"]:.1f}% ({baseline["trades"]} trades)')

    improving_filters = [(n, d) for n, r, d, i in results if d > 0]
    hurting_filters = [(n, d) for n, r, d, i in results if d < 0]

    if improving_filters:
        print('\nFILTERS THAT IMPROVE:')
        for name, diff in sorted(improving_filters, key=lambda x: x[1], reverse=True)[:5]:
            print(f'  {name}: +{diff:.1f}%')

    if hurting_filters:
        print('\nFILTERS THAT HURT (do not use):')
        for name, diff in sorted(hurting_filters, key=lambda x: x[1])[:5]:
            print(f'  {name}: {diff:.1f}%')

    # Best overall
    best = max(results, key=lambda x: x[1]['return'])
    print(f'\nBEST OVERALL: {best[0]} with +{best[1]["return"]:.1f}%')


if __name__ == "__main__":
    main()
