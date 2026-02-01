"""
Pattern-Based Filter from Losing Month Analysis
================================================
Apply filters based on discovered patterns:
- Avoid bad hours (15, 16, 18)
- Avoid late week (Wed, Thu, Fri)
- Require more extreme RSI
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
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean()

    def atr_pct_func(x):
        if len(x) == 0: return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)

def run_pattern_backtest(df, filters, test_start='2024-10-01', test_end='2026-02-01'):
    """Run backtest with pattern-based filters."""

    # Base RSI thresholds (can be modified by filters)
    RSI_OS = filters.get('rsi_os', 42)
    RSI_OB = filters.get('rsi_ob', 58)
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
    trades_filtered = 0
    consecutive_losses = 0

    for i in range(200, len(df) - 20):
        row = df.iloc[i]
        current_time = df.index[i]
        in_test = current_time >= pd.Timestamp(test_start) and current_time < pd.Timestamp(test_end)

        month_str = current_time.strftime('%Y-%m')
        weekday = row['weekday']
        hour = row['hour']

        if weekday >= 5:
            continue

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

        if not position and in_test:
            # Base time filter
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            skip_trade = False

            # 1. Regime filter (SIDEWAYS only)
            if filters.get('regime_sideways') and row['regime'] != 'SIDEWAYS':
                skip_trade = True

            # 2. Bad hours filter
            if not skip_trade and filters.get('avoid_hours'):
                if hour in filters['avoid_hours']:
                    skip_trade = True

            # 3. Bad weekdays filter
            if not skip_trade and filters.get('avoid_weekdays'):
                if weekday in filters['avoid_weekdays']:
                    skip_trade = True

            # 4. Consecutive loss filter
            if not skip_trade and filters.get('consec_loss'):
                if consecutive_losses >= filters['consec_loss']:
                    skip_trade = True
                    consecutive_losses = 0

            # 5. Only early week (Mon-Tue)
            if not skip_trade and filters.get('early_week_only'):
                if weekday > 1:  # 0=Mon, 1=Tue
                    skip_trade = True

            if skip_trade:
                trades_filtered += 1
                continue

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
        'trades_filtered': trades_filtered
    }

def main():
    print("=" * 70)
    print("PATTERN-BASED FILTER TEST")
    print("Based on Losing Month Analysis")
    print("=" * 70)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data("GBPUSD", start_date, end_date)

        if df is None:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")
        df = prepare_indicators(df)

        configs = [
            # Baseline
            {'name': 'Baseline', 'filters': {}},
            {'name': 'SIDEWAYS', 'filters': {'regime_sideways': True}},
            {'name': 'SIDE+Consec3', 'filters': {'regime_sideways': True, 'consec_loss': 3}},

            # Pattern-based filters
            {'name': 'AvoidHours15,16,18', 'filters': {'avoid_hours': [15, 16, 18]}},
            {'name': 'AvoidWed,Thu,Fri', 'filters': {'avoid_weekdays': [2, 3, 4]}},
            {'name': 'EarlyWeekOnly', 'filters': {'early_week_only': True}},

            # Combinations
            {'name': 'SIDE+AvoidHours', 'filters': {'regime_sideways': True, 'avoid_hours': [15, 16, 18]}},
            {'name': 'SIDE+AvoidDays', 'filters': {'regime_sideways': True, 'avoid_weekdays': [2, 3, 4]}},
            {'name': 'SIDE+EarlyWeek', 'filters': {'regime_sideways': True, 'early_week_only': True}},

            {'name': 'SIDE+C3+AvoidHours', 'filters': {'regime_sideways': True, 'consec_loss': 3, 'avoid_hours': [15, 16, 18]}},
            {'name': 'SIDE+C3+AvoidDays', 'filters': {'regime_sideways': True, 'consec_loss': 3, 'avoid_weekdays': [2, 3, 4]}},
            {'name': 'SIDE+C3+EarlyWeek', 'filters': {'regime_sideways': True, 'consec_loss': 3, 'early_week_only': True}},

            # Stricter RSI thresholds
            {'name': 'RSI38/62', 'filters': {'rsi_os': 38, 'rsi_ob': 62}},
            {'name': 'RSI35/65', 'filters': {'rsi_os': 35, 'rsi_ob': 65}},
            {'name': 'SIDE+RSI38/62', 'filters': {'regime_sideways': True, 'rsi_os': 38, 'rsi_ob': 62}},
            {'name': 'SIDE+C3+RSI38/62', 'filters': {'regime_sideways': True, 'consec_loss': 3, 'rsi_os': 38, 'rsi_ob': 62}},

            # All patterns combined
            {'name': 'SIDE+C3+Hours+Days', 'filters': {
                'regime_sideways': True, 'consec_loss': 3,
                'avoid_hours': [15, 16, 18], 'avoid_weekdays': [2, 3, 4]
            }},
            {'name': 'SIDE+C3+EarlyWeek+RSI', 'filters': {
                'regime_sideways': True, 'consec_loss': 3,
                'early_week_only': True, 'rsi_os': 38, 'rsi_ob': 62
            }},
            {'name': 'ALL_PATTERNS', 'filters': {
                'regime_sideways': True, 'consec_loss': 3,
                'avoid_hours': [15, 16, 18], 'avoid_weekdays': [2, 3, 4],
                'rsi_os': 38, 'rsi_ob': 62
            }},
        ]

        print(f"\nTesting {len(configs)} configurations...")
        print("-" * 70)

        results = []
        for cfg in configs:
            result = run_pattern_backtest(df.copy(), cfg['filters'])
            result['name'] = cfg['name']
            results.append(result)

            status = "0-LOSS!" if result['losing_months'] == 0 else f"{result['losing_months']}-loss"
            print(f"{cfg['name']:<25}: {status:<8} | Ret={result['total_return']:+6.1f}% | "
                  f"Trades={result['total_trades']:>3} | WR={result['win_rate']:.1f}%")

        # Sort
        results.sort(key=lambda x: (x['losing_months'], -x['total_return']))

        print("\n" + "=" * 70)
        print("TOP 10 RESULTS")
        print("=" * 70)

        for i, r in enumerate(results[:10], 1):
            print(f"{i:>2}. {r['name']:<28} Loss={r['losing_months']} | Ret={r['total_return']:+6.1f}% | "
                  f"Trades={r['total_trades']:>3} | WR={r['win_rate']:.1f}%")

        # Best details
        best = results[0]
        print("\n" + "=" * 70)
        print(f"BEST: {best['name']}")
        print("=" * 70)
        print("Monthly P/L:")
        for m, p in sorted(best['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            print(f"  {m}: ${p:+,.2f}{marker}")

        # Show 0-loss or 1-loss
        zero_loss = [r for r in results if r['losing_months'] == 0 and r['total_trades'] > 20]
        one_loss = [r for r in results if r['losing_months'] == 1 and r['total_trades'] > 50]

        if zero_loss:
            print("\n>>> ZERO-LOSS (>20 trades) <<<")
            for r in zero_loss:
                print(f"  {r['name']}: Return={r['total_return']:+.1f}%, Trades={r['total_trades']}, WR={r['win_rate']:.1f}%")

        if one_loss:
            print("\n>>> 1-LOSS (>50 trades) <<<")
            for r in one_loss[:5]:
                print(f"  {r['name']}: Return={r['total_return']:+.1f}%, Trades={r['total_trades']}, WR={r['win_rate']:.1f}%")

    finally:
        mt5.shutdown()
        print("\nDone!")

if __name__ == "__main__":
    main()
