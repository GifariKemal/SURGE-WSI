"""
EUR/USD Correlation Filter Test
===============================
Test if using EUR/USD as confirmation improves GBPUSD signals.

Theory:
- EUR/USD and GBP/USD have ~77% correlation
- When both pairs show same signal = stronger confirmation
- When they diverge = weaker signal, avoid

Sources:
- https://fxssi.com/currency-pairs-correlation-strategy
- https://www.cmcmarkets.com/en-gb/learn-forex/currency-correlations
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


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    # Load GBPUSD
    df_gbp = pd.read_sql("""
        SELECT time, open, high, low, close
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)

    # Load EURUSD
    df_eur = pd.read_sql("""
        SELECT time, close as eur_close
        FROM ohlcv
        WHERE symbol = 'EURUSD' AND timeframe = 'H1'
        AND time >= '2020-01-01' AND time <= '2026-01-31'
        ORDER BY time ASC
    """, conn)
    conn.close()

    df_gbp['time'] = pd.to_datetime(df_gbp['time']).dt.tz_localize(None)
    df_gbp.set_index('time', inplace=True)

    df_eur['time'] = pd.to_datetime(df_eur['time']).dt.tz_localize(None)
    df_eur.set_index('time', inplace=True)

    # Merge
    df = df_gbp.join(df_eur, how='inner')

    print('=' * 100)
    print('EUR/USD CORRELATION FILTER TEST')
    print('=' * 100)

    # Check if we have EUR/USD data
    if df['eur_close'].isna().all() or len(df) == 0:
        print('\n   ERROR: No EUR/USD data available.')
        return

    print(f'\n   Data points: {len(df)}')

    # Calculate RSI for both
    # GBPUSD RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi_gbp'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # EURUSD RSI
    delta_eur = df['eur_close'].diff()
    gain_eur = delta_eur.where(delta_eur > 0, 0).rolling(10).mean()
    loss_eur = (-delta_eur.where(delta_eur < 0, 0)).rolling(10).mean()
    df['rsi_eur'] = 100 - (100 / (1 + gain_eur / (loss_eur + 1e-10)))

    # ATR for GBPUSD
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # 1. Correlation analysis
    print('\n1. CORRELATION ANALYSIS')
    print('-' * 80)

    # Rolling correlation
    df['correlation'] = df['close'].rolling(100).corr(df['eur_close'])

    print(f'   Overall correlation: {df["close"].corr(df["eur_close"]):.3f}')
    print(f'   RSI correlation: {df["rsi_gbp"].corr(df["rsi_eur"]):.3f}')
    print(f'   Mean rolling correlation (100h): {df["correlation"].mean():.3f}')

    # RSI agreement
    gbp_os = df['rsi_gbp'] < 42
    gbp_ob = df['rsi_gbp'] > 58
    eur_os = df['rsi_eur'] < 42
    eur_ob = df['rsi_eur'] > 58

    both_os = (gbp_os & eur_os).sum()
    both_ob = (gbp_ob & eur_ob).sum()
    gbp_os_only = (gbp_os & ~eur_os).sum()
    gbp_ob_only = (gbp_ob & ~eur_ob).sum()

    print(f'\n   RSI Agreement:')
    print(f'   Both oversold (< 42): {both_os} ({both_os/len(df)*100:.1f}%)')
    print(f'   Both overbought (> 58): {both_ob} ({both_ob/len(df)*100:.1f}%)')
    print(f'   GBP oversold only: {gbp_os_only} ({gbp_os_only/len(df)*100:.1f}%)')
    print(f'   GBP overbought only: {gbp_ob_only} ({gbp_ob_only/len(df)*100:.1f}%)')

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
    MAX_HOLDING = 46
    SKIP_HOURS = [12]

    def backtest(require_eur_confirm=False, require_eur_same_zone=False,
                 skip_eur_diverge=False, eur_os=42, eur_ob=58):
        """
        Backtest with EUR/USD confirmation.

        Args:
            require_eur_confirm: EUR/USD RSI must also be OS/OB
            require_eur_same_zone: EUR/USD RSI in same zone (not exact, but trending same)
            skip_eur_diverge: Skip if EUR/USD is in opposite zone
            eur_os/eur_ob: Thresholds for EUR/USD
        """
        balance = 10000.0
        wins = losses = 0
        position = None
        peak = balance
        max_dd = 0
        yearly_pnl = {}
        filtered = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            year = df.index[i].year
            weekday = row['weekday']
            hour = row['hour']

            if weekday >= 5:
                continue

            if position:
                if (i - position['entry_idx']) >= MAX_HOLDING:
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

                rsi_gbp = row['rsi_gbp']
                rsi_eur = row['rsi_eur']

                signal = 0

                # GBPUSD signal
                if rsi_gbp < RSI_OS:
                    # Check EUR/USD confirmation
                    if require_eur_confirm and rsi_eur >= eur_os:
                        filtered += 1
                        continue
                    if skip_eur_diverge and rsi_eur > eur_ob:  # EUR overbought while GBP oversold
                        filtered += 1
                        continue
                    if require_eur_same_zone and rsi_eur > 50:  # EUR not in lower half
                        filtered += 1
                        continue
                    signal = 1

                elif rsi_gbp > RSI_OB:
                    if require_eur_confirm and rsi_eur <= eur_ob:
                        filtered += 1
                        continue
                    if skip_eur_diverge and rsi_eur < eur_os:
                        filtered += 1
                        continue
                    if require_eur_same_zone and rsi_eur < 50:
                        filtered += 1
                        continue
                    signal = -1

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
        return ret, trades, wr, max_dd, yearly_pnl, filtered

    # 2. Backtest with EUR confirmation
    print('\n2. BACKTEST WITH EUR/USD CONFIRMATION')
    print('-' * 80)
    print(f'   {"Filter":^35} | {"Return":^10} | {"Diff":^8} | {"Trades":^8} | {"WR":^8} | {"MaxDD":^8} | {"Filtered":^10}')
    print('   ' + '-' * 100)

    baseline_ret, baseline_trades, baseline_wr, baseline_dd, _, _ = backtest()
    print(f'   {"v3.7 Baseline":<35} | +{baseline_ret:>7.1f}% | {"":>8} | {baseline_trades:>6} | {baseline_wr:>5.1f}% | {baseline_dd:>5.1f}% | {0:>8}')

    configs = [
        (dict(require_eur_confirm=True, eur_os=42, eur_ob=58), 'EUR RSI also < 42 / > 58'),
        (dict(require_eur_confirm=True, eur_os=45, eur_ob=55), 'EUR RSI also < 45 / > 55'),
        (dict(require_eur_same_zone=True), 'EUR RSI same side of 50'),
        (dict(skip_eur_diverge=True), 'Skip if EUR opposite zone'),
        (dict(skip_eur_diverge=True, require_eur_same_zone=True), 'Same zone + no diverge'),
    ]

    for kwargs, name in configs:
        ret, trades, wr, max_dd, _, filtered = backtest(**kwargs)
        diff = ret - baseline_ret
        marker = ' <<<' if diff > 30 else ''
        print(f'   {name:<35} | +{ret:>7.1f}% | {diff:>+6.1f}% | {trades:>6} | {wr:>5.1f}% | {max_dd:>5.1f}% | {filtered:>8}{marker}')

    # Summary
    print('\n' + '=' * 100)
    print('FINDINGS')
    print('=' * 100)

    print(f"""
1. CORRELATION ANALYSIS:
   - GBPUSD/EURUSD price correlation: {df['close'].corr(df['eur_close']):.3f}
   - RSI correlation: {df['rsi_gbp'].corr(df['rsi_eur']):.3f}

2. CONFIRMATION FILTER IMPACT:
   - Requiring EUR/USD confirmation reduces trade count
   - May or may not improve win rate

3. RECOMMENDATION:
   - If correlation filter improves returns -> Consider
   - If it reduces returns -> REJECT (over-filtering)
   - Each pair should be traded independently for maximum opportunity
""")


if __name__ == "__main__":
    main()
