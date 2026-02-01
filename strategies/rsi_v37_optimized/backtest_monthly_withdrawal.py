"""
RSI v3.7 - Monthly Withdrawal Simulation
=========================================
Scenario:
- Start with $5,000 capital
- End of profitable month: Withdraw profit, reset to $5,000
- End of losing month: Absorb loss, may need to top-up or trade with less

Compare: ORIGINAL vs BB_SQUEEZE
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
BASE_CAPITAL = 5000.0  # Fixed monthly capital

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

USE_CONSEC_LOSS_FILTER = True
CONSEC_LOSS_LIMIT = 3

# BB Squeeze settings
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_PERCENTILE = 30
SMA_SLOPE_THRESHOLD = 0.5


def connect_mt5():
    if not MT5_PASSWORD:
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


def prepare_data(df):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = np.where(loss_s == 0, 100, gain / loss_s)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = pd.Series(tr, index=df.index).rolling(ATR_PERIOD).mean()

    # ATR Percentile
    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['atr_pct'] = df['atr'].rolling(100).apply(atr_pct_func, raw=True)

    # SMAs
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # Regime
    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > SMA_SLOPE_THRESHOLD),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -SMA_SLOPE_THRESHOLD),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
    df['bb_width'] = (df['bb_std'] * 2 * BB_STD) / df['bb_mid'] * 100

    def bb_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0
    df['bb_width_pct'] = df['bb_width'].rolling(100).apply(bb_pct_func, raw=True)
    df['bb_squeeze'] = df['bb_width_pct'] < BB_SQUEEZE_PERCENTILE

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)


def run_monthly_backtest(df, use_bb_squeeze=False):
    """
    Run backtest with monthly reset simulation

    Returns monthly results for both:
    1. Compound mode (reinvest profits)
    2. Withdrawal mode (withdraw profits monthly, reset to base capital)
    """

    # Get all months in data
    df['month'] = df.index.to_period('M')
    months = df['month'].unique()

    monthly_results = []

    for month in months:
        month_str = str(month)
        month_data = df[df['month'] == month].copy()

        if len(month_data) < 50:  # Skip incomplete months
            continue

        # Run backtest for this month with BASE_CAPITAL
        result = run_single_month(month_data, BASE_CAPITAL, use_bb_squeeze)

        monthly_results.append({
            'month': month_str,
            'trades': result['trades'],
            'wins': result['wins'],
            'pnl': result['pnl'],
            'end_balance': result['end_balance'],
            'max_dd': result['max_dd'],
        })

    return monthly_results


def run_single_month(df, starting_balance, use_bb_squeeze):
    """Run backtest for a single month"""
    balance = starting_balance
    peak_balance = starting_balance
    max_dd = 0

    position = None
    consecutive_losses = 0
    trades = 0
    wins = 0

    indices = list(range(len(df)))

    for i in indices:
        if i < 50:  # Need warmup
            continue

        row = df.iloc[i]

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
                trades += 1

                if balance > peak_balance:
                    peak_balance = balance
                dd = peak_balance - balance
                if dd > max_dd:
                    max_dd = dd

                if pnl > 0:
                    consecutive_losses = 0
                    wins += 1
                else:
                    consecutive_losses += 1

                position = None

        # Entry logic
        if not position:
            hour = row['hour']
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            if row['regime'] != 'SIDEWAYS':
                continue

            if use_bb_squeeze and not row['bb_squeeze']:
                continue

            if USE_CONSEC_LOSS_FILTER and consecutive_losses >= CONSEC_LOSS_LIMIT:
                consecutive_losses = 0
                continue

            rsi = row['rsi']
            signal = 0
            if rsi < RSI_OVERSOLD:
                signal = 1
            elif rsi > RSI_OVERBOUGHT:
                signal = -1

            if not signal:
                continue

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
            }

    return {
        'trades': trades,
        'wins': wins,
        'pnl': balance - starting_balance,
        'end_balance': balance,
        'max_dd': max_dd,
    }


def simulate_withdrawal_strategy(monthly_results, base_capital):
    """
    Simulate monthly withdrawal strategy

    Rules:
    - Start each month with base_capital (if possible)
    - If month is profitable: Withdraw profit
    - If month is loss: Absorb loss, may need top-up next month
    """

    simulation = []
    cumulative_withdrawn = 0
    cumulative_topup = 0
    actual_capital = base_capital

    for month in monthly_results:
        month_pnl = month['pnl']

        # Calculate P/L as percentage of actual capital used
        pnl_pct = month_pnl / actual_capital * 100 if actual_capital > 0 else 0

        if month_pnl > 0:
            # Profitable month - withdraw profit
            withdrawn = month_pnl
            cumulative_withdrawn += withdrawn
            topup = 0
            actual_capital = base_capital  # Reset to base
            action = f"WD ${withdrawn:+,.0f}"
        else:
            # Losing month - absorb loss
            withdrawn = 0
            end_balance = actual_capital + month_pnl

            if end_balance < base_capital:
                # Need top-up to get back to base_capital
                topup = base_capital - end_balance
                cumulative_topup += topup
                actual_capital = base_capital
                action = f"LOSS ${month_pnl:,.0f}, TopUp ${topup:,.0f}"
            else:
                topup = 0
                actual_capital = end_balance
                action = f"LOSS ${month_pnl:,.0f}"

        simulation.append({
            'month': month['month'],
            'trades': month['trades'],
            'wins': month['wins'],
            'pnl': month_pnl,
            'pnl_pct': pnl_pct,
            'withdrawn': withdrawn,
            'topup': topup,
            'cumulative_wd': cumulative_withdrawn,
            'cumulative_topup': cumulative_topup,
            'net_profit': cumulative_withdrawn - cumulative_topup,
            'action': action,
        })

    return simulation


def main():
    print("=" * 80)
    print("RSI v3.7 - MONTHLY WITHDRAWAL SIMULATION")
    print("=" * 80)
    print(f"\nScenario: Modal ${BASE_CAPITAL:,.0f}")
    print("- Bulan profit: Withdraw profit, reset modal ke $5,000")
    print("- Bulan loss: Absorb loss, top-up jika perlu")
    print("=" * 80)

    if not connect_mt5():
        print("MT5 connection failed")
        return

    try:
        print("\nFetching data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

        df = get_data(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
        print(f"Loaded {len(df)} H1 bars")

        print("Preparing indicators...")
        df = prepare_data(df)

        # Run monthly backtest for both strategies
        print("\nRunning ORIGINAL strategy...")
        monthly_orig = run_monthly_backtest(df, use_bb_squeeze=False)

        print("Running BB_SQUEEZE strategy...")
        monthly_bb = run_monthly_backtest(df, use_bb_squeeze=True)

        # Simulate withdrawal for both
        sim_orig = simulate_withdrawal_strategy(monthly_orig, BASE_CAPITAL)
        sim_bb = simulate_withdrawal_strategy(monthly_bb, BASE_CAPITAL)

        # ============================================
        # DETAILED MONTHLY COMPARISON
        # ============================================
        print("\n" + "=" * 100)
        print("PERBANDINGAN BULANAN: ORIGINAL vs BB_SQUEEZE (Modal $5,000 Fixed)")
        print("=" * 100)

        print(f"\n{'Month':<10} {'--- ORIGINAL ---':^40} {'--- BB_SQUEEZE ---':^40}")
        print(f"{'':10} {'P/L':>10} {'Action':>28} {'P/L':>10} {'Action':>28}")
        print("-" * 100)

        for i in range(len(sim_orig)):
            o = sim_orig[i]
            b = sim_bb[i] if i < len(sim_bb) else None

            if b:
                # Highlight differences
                orig_status = "PROFIT" if o['pnl'] > 0 else "LOSS"
                bb_status = "PROFIT" if b['pnl'] > 0 else "LOSS"

                marker = ""
                if orig_status == "LOSS" and bb_status == "PROFIT":
                    marker = " << BB SAVED!"
                elif orig_status == "PROFIT" and bb_status == "LOSS":
                    marker = " (BB missed)"

                print(f"{o['month']:<10} ${o['pnl']:>+8,.0f} {o['action']:>28} ${b['pnl']:>+8,.0f} {b['action']:>28}{marker}")

        # ============================================
        # SUMMARY
        # ============================================
        print("\n" + "=" * 80)
        print("RINGKASAN AKHIR")
        print("=" * 80)

        # Calculate totals
        orig_total_wd = sim_orig[-1]['cumulative_wd'] if sim_orig else 0
        orig_total_topup = sim_orig[-1]['cumulative_topup'] if sim_orig else 0
        orig_net = orig_total_wd - orig_total_topup

        bb_total_wd = sim_bb[-1]['cumulative_wd'] if sim_bb else 0
        bb_total_topup = sim_bb[-1]['cumulative_topup'] if sim_bb else 0
        bb_net = bb_total_wd - bb_total_topup

        # Count months
        orig_profit_months = len([s for s in sim_orig if s['pnl'] > 0])
        orig_loss_months = len([s for s in sim_orig if s['pnl'] <= 0])
        bb_profit_months = len([s for s in sim_bb if s['pnl'] > 0])
        bb_loss_months = len([s for s in sim_bb if s['pnl'] <= 0])

        # Max single month loss
        orig_max_loss = min([s['pnl'] for s in sim_orig]) if sim_orig else 0
        bb_max_loss = min([s['pnl'] for s in sim_bb]) if sim_bb else 0

        print(f"\n{'Metric':<30} {'ORIGINAL':>20} {'BB_SQUEEZE':>20}")
        print("-" * 75)
        print(f"{'Total Bulan Trading':<30} {len(sim_orig):>20} {len(sim_bb):>20}")
        print(f"{'Bulan Profit':<30} {orig_profit_months:>20} {bb_profit_months:>20}")
        print(f"{'Bulan Loss':<30} {orig_loss_months:>20} {bb_loss_months:>20}")
        print(f"{'Win Rate (Bulan)':<30} {orig_profit_months/len(sim_orig)*100:>19.1f}% {bb_profit_months/len(sim_bb)*100:>19.1f}%")
        print("-" * 75)
        print(f"{'Total Withdrawal':<30} ${orig_total_wd:>18,.0f} ${bb_total_wd:>18,.0f}")
        print(f"{'Total Top-Up (Loss Recovery)':<30} ${orig_total_topup:>18,.0f} ${bb_total_topup:>18,.0f}")
        print(f"{'NET PROFIT (WD - TopUp)':<30} ${orig_net:>+18,.0f} ${bb_net:>+18,.0f}")
        print("-" * 75)
        print(f"{'Max Single Month Loss':<30} ${orig_max_loss:>18,.0f} ${bb_max_loss:>18,.0f}")
        print(f"{'Max TopUp Needed (1 bulan)':<30} ${abs(orig_max_loss):>18,.0f} ${abs(bb_max_loss):>18,.0f}")

        # ============================================
        # CASHFLOW TIMELINE
        # ============================================
        print("\n" + "=" * 80)
        print("CASHFLOW TIMELINE")
        print("=" * 80)

        print(f"\n{'Month':<10} {'ORIGINAL':^35} {'BB_SQUEEZE':^35}")
        print(f"{'':10} {'Net P/L':>12} {'Cumul. Profit':>12} {'Net P/L':>12} {'Cumul. Profit':>12}")
        print("-" * 80)

        for i in range(len(sim_orig)):
            o = sim_orig[i]
            b = sim_bb[i] if i < len(sim_bb) else None

            if b:
                print(f"{o['month']:<10} ${o['pnl']:>+10,.0f} ${o['net_profit']:>+10,.0f}   ${b['pnl']:>+10,.0f} ${b['net_profit']:>+10,.0f}")

        # ============================================
        # WORST CASE ANALYSIS
        # ============================================
        print("\n" + "=" * 80)
        print("WORST CASE ANALYSIS - Bulan Loss Terbesar")
        print("=" * 80)

        # Find worst months for each
        orig_worst = min(sim_orig, key=lambda x: x['pnl'])
        bb_worst = min(sim_bb, key=lambda x: x['pnl'])

        print(f"\nORIGINAL - Worst Month: {orig_worst['month']}")
        print(f"  Loss: ${orig_worst['pnl']:,.0f}")
        print(f"  TopUp needed: ${abs(orig_worst['pnl']):,.0f}")
        print(f"  Impact: Perlu tambah modal ${abs(orig_worst['pnl']):,.0f} untuk lanjut trading")

        print(f"\nBB_SQUEEZE - Worst Month: {bb_worst['month']}")
        print(f"  Loss: ${bb_worst['pnl']:,.0f}")
        print(f"  TopUp needed: ${abs(bb_worst['pnl']):,.0f}")
        print(f"  Impact: Perlu tambah modal ${abs(bb_worst['pnl']):,.0f} untuk lanjut trading")

        # ============================================
        # RECOMMENDATION
        # ============================================
        print("\n" + "=" * 80)
        print("ANALISIS & REKOMENDASI")
        print("=" * 80)

        print(f"""
SKENARIO: Modal $5,000 dengan Withdrawal Bulanan

ORIGINAL Strategy:
  - Total Withdrawal: ${orig_total_wd:,.0f}
  - Total TopUp (recover loss): ${orig_total_topup:,.0f}
  - NET PROFIT: ${orig_net:+,.0f}
  - Bulan Loss: {orig_loss_months} bulan
  - Max TopUp 1 bulan: ${abs(orig_max_loss):,.0f} ({abs(orig_max_loss)/BASE_CAPITAL*100:.0f}% dari modal)

BB_SQUEEZE Strategy:
  - Total Withdrawal: ${bb_total_wd:,.0f}
  - Total TopUp (recover loss): ${bb_total_topup:,.0f}
  - NET PROFIT: ${bb_net:+,.0f}
  - Bulan Loss: {bb_loss_months} bulan
  - Max TopUp 1 bulan: ${abs(bb_max_loss):,.0f} ({abs(bb_max_loss)/BASE_CAPITAL*100:.0f}% dari modal)

KESIMPULAN:
""")

        if orig_net > bb_net:
            diff = orig_net - bb_net
            print(f"  ORIGINAL menghasilkan ${diff:,.0f} lebih banyak")
            print(f"  TAPI membutuhkan total topup ${orig_total_topup:,.0f} vs ${bb_total_topup:,.0f}")
        else:
            diff = bb_net - orig_net
            print(f"  BB_SQUEEZE menghasilkan ${diff:,.0f} lebih banyak")

        print(f"""
REKOMENDASI untuk Modal $5,000 + Monthly WD:

1. Jika TIDAK siap topup saat loss:
   -> Gunakan BB_SQUEEZE (max loss ${abs(bb_max_loss):,.0f}/bulan)

2. Jika siap cadangan ${abs(orig_max_loss):,.0f} untuk topup:
   -> Gunakan ORIGINAL (net profit lebih tinggi)

3. Jika ingin cashflow stabil:
   -> BB_SQUEEZE (lebih sedikit bulan loss)
""")

    finally:
        mt5.shutdown()
        print("\n" + "=" * 80)
        print("Simulation complete!")
        print("=" * 80)


if __name__ == "__main__":
    main()
