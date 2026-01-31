"""
Half-Life of Mean Reversion Test
================================
Calculate the half-life using Ornstein-Uhlenbeck process to determine
optimal holding period for mean reversion trades.

Theory:
- Half-life = time for price deviation to reduce by 50%
- Ornstein-Uhlenbeck: dX = theta * (mu - X) * dt + sigma * dW
- Half-life = -ln(2) / theta

Research shows this should give us the optimal max_holding period.
"""
import pandas as pd
import numpy as np
import psycopg2
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}


def calculate_halflife(series):
    """
    Calculate half-life of mean reversion using OLS regression.

    Model: y(t) - y(t-1) = alpha + beta * y(t-1) + epsilon
    Half-life = -ln(2) / beta
    """
    lagged = series.shift(1)
    delta = series - lagged

    # Remove NaN
    lagged = lagged.dropna()
    delta = delta.dropna()

    # Align indices
    common_idx = lagged.index.intersection(delta.index)
    lagged = lagged.loc[common_idx]
    delta = delta.loc[common_idx]

    # OLS regression: delta = alpha + beta * lagged
    slope, intercept, r_value, p_value, std_err = stats.linregress(lagged, delta)

    # Half-life calculation
    if slope < 0:
        half_life = -np.log(2) / slope
    else:
        half_life = np.inf  # Not mean-reverting

    return half_life, slope, p_value


def calculate_ou_params(series):
    """
    Calculate Ornstein-Uhlenbeck parameters.
    dX = theta * (mu - X) * dt + sigma * dW
    """
    lagged = series.shift(1)
    delta = series - lagged

    # Remove NaN
    lagged = lagged.dropna()
    delta = delta.dropna()

    common_idx = lagged.index.intersection(delta.index)
    lagged = lagged.loc[common_idx]
    delta = delta.loc[common_idx]

    # OLS: delta = alpha + beta * lagged
    slope, intercept, _, p_value, _ = stats.linregress(lagged, delta)

    # OU parameters
    theta = -slope  # Mean reversion speed
    mu = intercept / theta if theta != 0 else 0  # Long-term mean
    sigma = delta.std()  # Volatility

    half_life = np.log(2) / theta if theta > 0 else np.inf

    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life': half_life,
        'p_value': p_value
    }


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT time, open, high, low, close
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)
    conn.close()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    print('=' * 100)
    print('HALF-LIFE OF MEAN REVERSION ANALYSIS')
    print('=' * 100)

    # 1. Calculate half-life on raw price
    print('\n1. HALF-LIFE ON RAW PRICE')
    print('-' * 80)
    hl, slope, pval = calculate_halflife(df['close'])
    print(f'   Half-life: {hl:.1f} hours')
    print(f'   Slope (beta): {slope:.6f}')
    print(f'   P-value: {pval:.4f}')
    print(f'   Mean-reverting: {"YES" if slope < 0 and pval < 0.05 else "NO"}')

    # 2. Calculate half-life on RSI
    print('\n2. HALF-LIFE ON RSI')
    print('-' * 80)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # RSI deviation from 50 (equilibrium)
    df['rsi_dev'] = df['rsi'] - 50

    hl_rsi, slope_rsi, pval_rsi = calculate_halflife(df['rsi_dev'].dropna())
    print(f'   Half-life (RSI deviation from 50): {hl_rsi:.1f} hours')
    print(f'   Slope (beta): {slope_rsi:.6f}')
    print(f'   P-value: {pval_rsi:.4f}')
    print(f'   Mean-reverting: {"YES" if slope_rsi < 0 and pval_rsi < 0.05 else "NO"}')

    # 3. OU Parameters
    print('\n3. ORNSTEIN-UHLENBECK PARAMETERS')
    print('-' * 80)

    ou_params = calculate_ou_params(df['rsi_dev'].dropna())
    print(f'   Theta (mean reversion speed): {ou_params["theta"]:.6f}')
    print(f'   Mu (long-term mean): {ou_params["mu"]:.4f}')
    print(f'   Sigma (volatility): {ou_params["sigma"]:.4f}')
    print(f'   Half-life: {ou_params["half_life"]:.1f} hours')
    print(f'   P-value: {ou_params["p_value"]:.4f}')

    # 4. Rolling half-life analysis
    print('\n4. ROLLING HALF-LIFE ANALYSIS (250-hour windows)')
    print('-' * 80)

    window = 250
    half_lives = []

    for i in range(window, len(df['rsi_dev']), 50):
        window_data = df['rsi_dev'].iloc[i-window:i]
        hl, slope, pval = calculate_halflife(window_data)
        if 0 < hl < 200:  # Filter out extreme values
            half_lives.append(hl)

    if half_lives:
        print(f'   Mean half-life: {np.mean(half_lives):.1f} hours')
        print(f'   Median half-life: {np.median(half_lives):.1f} hours')
        print(f'   Std half-life: {np.std(half_lives):.1f} hours')
        print(f'   Min half-life: {np.min(half_lives):.1f} hours')
        print(f'   Max half-life: {np.max(half_lives):.1f} hours')

    # 5. Test different max_holding periods based on half-life
    print('\n5. BACKTEST WITH HALF-LIFE BASED HOLDING PERIODS')
    print('-' * 80)

    # ATR calculation
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.7 parameters
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80
    TIME_TP_BONUS = 0.35
    SKIP_HOURS = [12]

    def backtest(max_holding=46):
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                if (i - position['entry_idx']) >= max_holding:
                    if position['dir'] == 1:
                        pnl = (row['close'] - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - row['close']) * position['size']
                    balance += pnl
                    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
                else:
                    if position['dir'] == 1:
                        if row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            position = None
                    else:
                        if row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            losses += 1
                            position = None
                        elif row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['size']
                            balance += pnl
                            yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                            wins += 1
                            position = None

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            if not position:
                if hour < 7 or hour >= 22:
                    continue
                if hour in SKIP_HOURS:
                    continue

                atr_pct = row['atr_pct']
                if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                    continue

                rsi = row['rsi']
                signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

                if signal:
                    entry = row['close']
                    atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                    base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                    if 12 <= hour < 16:
                        tp_mult = base_tp + TIME_TP_BONUS
                    else:
                        tp_mult = base_tp
                    sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                    tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                    risk = balance * 0.01
                    size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                    position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

        trades = wins + losses
        wr = wins / trades * 100 if trades else 0
        ret = (balance - 10000) / 100
        return ret, trades, wr, max_dd

    # Test different holding periods
    print(f'\n   {"Max Holding":^15} | {"Return":^10} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8}')
    print('   ' + '-' * 60)

    baseline_ret = None
    for max_holding in [20, 30, 40, 46, 50, 60, 80, 100, 120]:
        ret, trades, wr, max_dd = backtest(max_holding=max_holding)
        if max_holding == 46:
            baseline_ret = ret
        diff = f'({ret - baseline_ret:+.1f})' if baseline_ret else ''
        marker = ' <<<' if max_holding == 46 else ''
        print(f'   {max_holding:>8} hours  | +{ret:>7.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}%{marker}')

    # 6. Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    optimal_hl = np.median(half_lives) if half_lives else 46

    print(f"""
1. HALF-LIFE CALCULATION:
   - RSI mean reversion half-life: ~{optimal_hl:.0f} hours
   - This suggests optimal holding period around {optimal_hl:.0f}-{optimal_hl*2:.0f} hours
   - Current v3.7 uses 46 hours - {"ALIGNED" if 40 <= optimal_hl <= 60 else "NEEDS ADJUSTMENT"}

2. INTERPRETATION:
   - Half-life = time for 50% of deviation to dissipate
   - Max holding should be ~2x half-life for full reversion
   - If half-life is shorter, we're holding too long
   - If half-life is longer, we may be exiting too early

3. RECOMMENDATION:
   - Based on half-life analysis, optimal max_holding = {optimal_hl*1.5:.0f} hours
   - Test range: {optimal_hl:.0f} to {optimal_hl*2:.0f} hours
""")


if __name__ == "__main__":
    main()
