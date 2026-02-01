"""
Combined Filters Test for RSI v3.7
==================================
Combine Regime + Multiple Filters to eliminate losing months.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

MT5_LOGIN = 61045904
MT5_PASSWORD = "iy#K5L7sF"
MT5_SERVER = "FinexBisnisSolusi-Demo"
MT5_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

def connect_mt5():
    if not mt5.initialize(path=MT5_PATH):
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return False
    return True

def get_h1_data(symbol, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def prepare_indicators(df):
    """Calculate all indicators needed for filtering."""

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean()

    # ATR Percentile
    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # SMAs for regime
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # Regime (SMA method - best for RSI)
    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    # Choppiness Index
    atr_sum = df['atr'].rolling(14).sum()
    high_max = df['high'].rolling(14).max()
    low_min = df['low'].rolling(14).min()
    range_14 = high_max - low_min
    df['chop'] = 100 * np.log10(atr_sum / range_14) / np.log10(14)

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm_arr = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm_arr = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    smoothed_tr = pd.Series(tr, index=df.index).rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm_arr, index=df.index).rolling(14).mean() / smoothed_tr
    minus_di = 100 * pd.Series(minus_dm_arr, index=df.index).rolling(14).mean() / smoothed_tr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(14).mean()

    # Volatility (24h)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(24).std() * 100
    df['vol_ma'] = df['volatility'].rolling(120).mean()
    df['vol_ratio'] = df['volatility'] / df['vol_ma']

    # Momentum
    df['momentum'] = (df['close'] / df['close'].shift(20) - 1) * 100

    # RSI divergence (price vs RSI)
    df['price_change_5'] = df['close'].pct_change(5) * 100
    df['rsi_change_5'] = df['rsi'].diff(5)

    # Time
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)

def run_filtered_backtest(df, filters, test_start='2024-10-01', test_end='2026-02-01'):
    """
    Run RSI v3.7 with multiple filters.

    filters dict can contain:
    - regime: list of allowed regimes ['SIDEWAYS'] or None
    - chop_min/chop_max: Choppiness range
    - adx_max: Max ADX for sideways
    - vol_ratio_max: Max volatility ratio
    - month_loss_limit: Stop trading if month loss exceeds this
    - consec_loss_limit: Pause after N consecutive losses
    - momentum_range: (min, max) momentum allowed
    """
    RSI_OS = 42
    RSI_OB = 58
    SL_MULT = 1.5
    TP_LOW = 2.4
    TP_MED = 3.0
    TP_HIGH = 3.6
    MAX_HOLDING = 46
    MIN_ATR_PCT = 20
    MAX_ATR_PCT = 80

    balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    trades_taken = 0
    trades_filtered = 0
    consecutive_losses = 0
    filter_reasons = {}

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        in_test = current_time >= pd.Timestamp(test_start) and current_time < pd.Timestamp(test_end)

        month_str = current_time.strftime('%Y-%m')
        weekday = row['weekday']
        hour = row['hour']

        if weekday >= 5:
            continue

        # Position management
        if position:
            exit_reason = None
            pnl = 0

            if (i - position['entry_idx']) >= MAX_HOLDING:
                pnl = (row['close'] - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - row['close']) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL'
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP'
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL'
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP'

            if exit_reason:
                balance += pnl
                if pnl > 0:
                    wins += 1
                    consecutive_losses = 0
                else:
                    losses += 1
                    consecutive_losses += 1

                if month_str not in monthly_pnl:
                    monthly_pnl[month_str] = 0
                monthly_pnl[month_str] += pnl
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Entry logic with filters
        if not position and in_test:
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            # ===== APPLY FILTERS =====
            skip_trade = False
            skip_reason = None

            # 1. Regime filter
            if filters.get('regime'):
                if row['regime'] not in filters['regime']:
                    skip_trade = True
                    skip_reason = 'regime'

            # 2. Choppiness filter
            if not skip_trade and filters.get('chop_min') is not None:
                if row['chop'] < filters['chop_min']:
                    skip_trade = True
                    skip_reason = 'chop_low'
            if not skip_trade and filters.get('chop_max') is not None:
                if row['chop'] > filters['chop_max']:
                    skip_trade = True
                    skip_reason = 'chop_high'

            # 3. ADX filter (low ADX = sideways = good for mean reversion)
            if not skip_trade and filters.get('adx_max') is not None:
                if row['adx'] > filters['adx_max']:
                    skip_trade = True
                    skip_reason = 'adx_high'

            # 4. Volatility ratio filter
            if not skip_trade and filters.get('vol_ratio_max') is not None:
                if row['vol_ratio'] > filters['vol_ratio_max']:
                    skip_trade = True
                    skip_reason = 'vol_high'

            # 5. Monthly loss limit
            if not skip_trade and filters.get('month_loss_limit') is not None:
                current_month_pnl = monthly_pnl.get(month_str, 0)
                if current_month_pnl < -filters['month_loss_limit']:
                    skip_trade = True
                    skip_reason = 'month_loss'

            # 6. Consecutive loss limit
            if not skip_trade and filters.get('consec_loss_limit') is not None:
                if consecutive_losses >= filters['consec_loss_limit']:
                    skip_trade = True
                    skip_reason = 'consec_loss'
                    consecutive_losses = 0  # Reset

            # 7. Momentum filter
            if not skip_trade and filters.get('momentum_range') is not None:
                mom_min, mom_max = filters['momentum_range']
                if row['momentum'] < mom_min or row['momentum'] > mom_max:
                    skip_trade = True
                    skip_reason = 'momentum'

            if skip_trade:
                trades_filtered += 1
                if skip_reason not in filter_reasons:
                    filter_reasons[skip_reason] = 0
                filter_reasons[skip_reason] += 1
                continue

            # ===== END FILTERS =====

            rsi = row['rsi']
            signal = 1 if rsi < RSI_OS else (-1 if rsi > RSI_OB else 0)

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                tp_mult = base_tp + 0.35 if 12 <= hour < 16 else base_tp
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}
                trades_taken += 1

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - 10000) / 10000 * 100
    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    losing_months = sum(1 for m in monthly_pnl.values() if m < 0)

    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profitable_months': profitable_months,
        'losing_months': losing_months,
        'total_months': len(monthly_pnl),
        'monthly_pnl': monthly_pnl,
        'trades_filtered': trades_filtered,
        'filter_reasons': filter_reasons
    }

def main():
    print("=" * 70)
    print("COMBINED FILTERS TEST")
    print("Regime + Choppiness + ADX + Volatility + Loss Limits")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data("GBPUSD", start_date, end_date)

        if df is None:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")

        print("Calculating indicators...")
        df = prepare_indicators(df)

        # Baseline
        print("\n" + "=" * 70)
        print("BASELINE (No Filters)")
        print("=" * 70)
        baseline = run_filtered_backtest(df.copy(), {})
        print(f"Return: {baseline['total_return']:+.1f}% | Trades: {baseline['total_trades']} | Loss Months: {baseline['losing_months']}")

        results = [{'name': 'Baseline', 'filters': {}, **baseline}]

        # Test filter combinations
        filter_configs = [
            # Single filters
            {'name': 'Regime:SIDEWAYS', 'filters': {'regime': ['SIDEWAYS']}},
            {'name': 'Chop:38-62', 'filters': {'chop_min': 38, 'chop_max': 62}},
            {'name': 'ADX<25', 'filters': {'adx_max': 25}},
            {'name': 'ADX<20', 'filters': {'adx_max': 20}},
            {'name': 'VolRatio<1.5', 'filters': {'vol_ratio_max': 1.5}},
            {'name': 'VolRatio<1.2', 'filters': {'vol_ratio_max': 1.2}},
            {'name': 'MonthLoss$200', 'filters': {'month_loss_limit': 200}},
            {'name': 'MonthLoss$150', 'filters': {'month_loss_limit': 150}},
            {'name': 'ConsecLoss3', 'filters': {'consec_loss_limit': 3}},
            {'name': 'ConsecLoss2', 'filters': {'consec_loss_limit': 2}},
            {'name': 'Mom:-2to2', 'filters': {'momentum_range': (-2, 2)}},

            # Regime + one other filter
            {'name': 'SIDEWAYS+ADX<25', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 25}},
            {'name': 'SIDEWAYS+Chop38-62', 'filters': {'regime': ['SIDEWAYS'], 'chop_min': 38, 'chop_max': 62}},
            {'name': 'SIDEWAYS+VolR<1.5', 'filters': {'regime': ['SIDEWAYS'], 'vol_ratio_max': 1.5}},
            {'name': 'SIDEWAYS+MonthLoss$200', 'filters': {'regime': ['SIDEWAYS'], 'month_loss_limit': 200}},
            {'name': 'SIDEWAYS+Consec3', 'filters': {'regime': ['SIDEWAYS'], 'consec_loss_limit': 3}},

            # Triple combinations
            {'name': 'SIDEWAYS+ADX<25+Chop', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 25, 'chop_min': 38, 'chop_max': 62}},
            {'name': 'SIDEWAYS+ADX<25+VolR', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 25, 'vol_ratio_max': 1.5}},
            {'name': 'SIDEWAYS+ADX<25+ML$200', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 25, 'month_loss_limit': 200}},
            {'name': 'SIDEWAYS+Chop+ML$200', 'filters': {'regime': ['SIDEWAYS'], 'chop_min': 38, 'chop_max': 62, 'month_loss_limit': 200}},
            {'name': 'SIDEWAYS+VolR+ML$200', 'filters': {'regime': ['SIDEWAYS'], 'vol_ratio_max': 1.5, 'month_loss_limit': 200}},
            {'name': 'SIDEWAYS+Consec3+ML$200', 'filters': {'regime': ['SIDEWAYS'], 'consec_loss_limit': 3, 'month_loss_limit': 200}},

            # Quad combinations
            {'name': 'SIDE+ADX+Chop+ML', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 25, 'chop_min': 38, 'chop_max': 62, 'month_loss_limit': 200}},
            {'name': 'SIDE+ADX+VolR+ML', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 25, 'vol_ratio_max': 1.5, 'month_loss_limit': 200}},
            {'name': 'SIDE+ADX+Consec+ML', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 25, 'consec_loss_limit': 3, 'month_loss_limit': 200}},

            # Aggressive combinations
            {'name': 'SIDE+ADX<20+ML$150', 'filters': {'regime': ['SIDEWAYS'], 'adx_max': 20, 'month_loss_limit': 150}},
            {'name': 'SIDE+VolR<1.2+ML$150', 'filters': {'regime': ['SIDEWAYS'], 'vol_ratio_max': 1.2, 'month_loss_limit': 150}},
            {'name': 'SIDE+Consec2+ML$150', 'filters': {'regime': ['SIDEWAYS'], 'consec_loss_limit': 2, 'month_loss_limit': 150}},
            {'name': 'ALL_FILTERS', 'filters': {
                'regime': ['SIDEWAYS'], 'adx_max': 25, 'chop_min': 38, 'chop_max': 62,
                'vol_ratio_max': 1.5, 'month_loss_limit': 200, 'consec_loss_limit': 3
            }},
        ]

        print("\n" + "=" * 70)
        print("TESTING FILTER COMBINATIONS")
        print("=" * 70)

        for cfg in filter_configs:
            result = run_filtered_backtest(df.copy(), cfg['filters'])
            status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
            print(f"{cfg['name']:<25}: {status:<8} | Ret={result['total_return']:+6.1f}% | "
                  f"Trades={result['total_trades']:>3} | Filtered={result['trades_filtered']}")
            results.append({'name': cfg['name'], **result})

        # Sort by losing months, then by return
        results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        print("\n" + "=" * 70)
        print("TOP 10 CONFIGURATIONS")
        print("=" * 70)

        for i, r in enumerate(results[:10], 1):
            print(f"\n{i}. {r['name']}")
            print(f"   Losing: {r['losing_months']} | Return: {r['total_return']:+.1f}% | DD: {r['max_drawdown']:.1f}%")
            print(f"   Trades: {r['total_trades']} | Filtered: {r.get('trades_filtered', 0)}")

        # Show best config monthly P/L
        best = results[0]
        print("\n" + "=" * 70)
        print(f"BEST: {best['name']}")
        print("=" * 70)
        print("Monthly P/L:")
        for m, p in sorted(best['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            print(f"  {m}: ${p:+,.2f}{marker}")

        # Check for 0-loss
        zero_loss = [r for r in results if r['losing_months'] == 0]
        if zero_loss:
            print("\n" + "=" * 70)
            print("ZERO-LOSS CONFIGURATIONS FOUND!")
            print("=" * 70)
            for r in zero_loss:
                print(f"  {r['name']}: Return={r['total_return']:+.1f}%, Trades={r['total_trades']}")
        else:
            # Show configs with 1-2 losing months
            low_loss = [r for r in results if r['losing_months'] <= 2 and r['total_trades'] > 50]
            if low_loss:
                print("\n" + "=" * 70)
                print("BEST PRACTICAL CONFIGS (1-2 loss, >50 trades)")
                print("=" * 70)
                for r in low_loss[:5]:
                    print(f"  {r['name']}: Loss={r['losing_months']}, Return={r['total_return']:+.1f}%, Trades={r['total_trades']}")

    finally:
        mt5.shutdown()
        print("\nDone!")

if __name__ == "__main__":
    main()
