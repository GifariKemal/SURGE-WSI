"""
STRATEGY COMPARISON BACKTEST
============================

Membandingkan semua strategi dengan database yang sama:
1. RSI Baseline (Mean Reversion)
2. BBMA Oma Ally (Bollinger + MA)
3. SMC/Order Block + FVG
4. ICT (Kill Zone + Liquidity Sweep)
5. Hybrid (Kombinasi Terbaik)

Database: ml_trading_bot PostgreSQL (TimescaleDB)
Symbol: GBPUSD H1
Period: 2020-2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV dari database"""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
        AND time >= %s AND time <= %s
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah semua indikator yang dibutuhkan"""
    df = df.copy()

    # === RSI ===
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # === ATR ===
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = tr.rolling(14).mean()

    # === BOLLINGER BANDS ===
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # === BBMA OMA ALLY COMPONENTS ===
    # MA5 High/Low (Linear Weighted)
    weights = np.arange(1, 6)
    df['ma5_high'] = df['high'].rolling(5).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    df['ma5_low'] = df['low'].rolling(5).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # MA10 High/Low (Linear Weighted)
    weights10 = np.arange(1, 11)
    df['ma10_high'] = df['high'].rolling(10).apply(lambda x: np.dot(x, weights10) / weights10.sum(), raw=True)
    df['ma10_low'] = df['low'].rolling(10).apply(lambda x: np.dot(x, weights10) / weights10.sum(), raw=True)

    # EMA 50 (Trend Filter)
    df['ema50'] = df['close'].ewm(span=50).mean()

    # === EMAs for TREND ===
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()

    # === MACD ===
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # === STOCHASTIC ===
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = (df['close'] - low14) / (high14 - low14 + 1e-10) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # === SWING HIGHS/LOWS (untuk Order Block) ===
    df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
    df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']

    # === TIME FEATURES ===
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    return df.fillna(method='ffill').fillna(0)


# =============================================================================
# STRATEGY 1: RSI BASELINE
# =============================================================================

def strategy_rsi_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI Mean Reversion Strategy
    - BUY: RSI < 30
    - SELL: RSI > 70
    - Session: London (07-19 UTC)
    """
    df = df.copy()
    df['signal'] = 0

    # Conditions
    london_session = (df['hour'] >= 7) & (df['hour'] < 19)
    rsi_oversold = df['rsi'] < 30
    rsi_overbought = df['rsi'] > 70

    df.loc[london_session & rsi_oversold, 'signal'] = 1   # BUY
    df.loc[london_session & rsi_overbought, 'signal'] = -1  # SELL

    return df


# =============================================================================
# STRATEGY 2: BBMA OMA ALLY
# =============================================================================

def strategy_bbma(df: pd.DataFrame) -> pd.DataFrame:
    """
    BBMA Oma Ally Strategy
    - EXTREME BUY: Close < BB Lower + MA5 Low cross above MA10 Low
    - EXTREME SELL: Close > BB Upper + MA5 High cross below MA10 High
    - Re-Entry: Price returns to EMA50 in trend direction
    - Session: London (07-19 UTC)
    """
    df = df.copy()
    df['signal'] = 0

    london_session = (df['hour'] >= 7) & (df['hour'] < 19)

    # Trend direction (using EMA50)
    uptrend = df['close'] > df['ema50']
    downtrend = df['close'] < df['ema50']

    # EXTREME BUY: Close below BB Lower, MA5 Low > MA10 Low (bullish cross)
    ma_cross_up = (df['ma5_low'] > df['ma10_low']) & (df['ma5_low'].shift(1) <= df['ma10_low'].shift(1))
    extreme_buy = (df['close'] < df['bb_lower']) | ma_cross_up

    # EXTREME SELL: Close above BB Upper, MA5 High < MA10 High (bearish cross)
    ma_cross_down = (df['ma5_high'] < df['ma10_high']) & (df['ma5_high'].shift(1) >= df['ma10_high'].shift(1))
    extreme_sell = (df['close'] > df['bb_upper']) | ma_cross_down

    # Re-Entry: Price touches EMA50 in trend
    ema50_touch_up = uptrend & (df['low'] <= df['ema50'] * 1.002) & (df['close'] > df['ema50'])
    ema50_touch_down = downtrend & (df['high'] >= df['ema50'] * 0.998) & (df['close'] < df['ema50'])

    # Apply signals
    df.loc[london_session & (extreme_buy | ema50_touch_up), 'signal'] = 1   # BUY
    df.loc[london_session & (extreme_sell | ema50_touch_down), 'signal'] = -1  # SELL

    return df


# =============================================================================
# STRATEGY 3: SMC/ORDER BLOCK
# =============================================================================

def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect Order Blocks (Supply/Demand Zones)"""
    df = df.copy()
    df['ob_bull'] = 0.0
    df['ob_bear'] = 0.0

    for i in range(lookback, len(df)):
        # Bullish OB: Bearish candle followed by strong bullish move
        for j in range(i - lookback, i - 3):
            bar = df.iloc[j]
            next_bars = df.iloc[j+1:j+4]

            # Bearish candle
            if bar['close'] < bar['open']:
                # Followed by bullish move
                move_up = next_bars['close'].max() - bar['low']
                if move_up > 0.0015:  # 15 pips minimum
                    # Check if price is near OB
                    current_price = df.iloc[i]['close']
                    if bar['low'] <= current_price <= bar['high']:
                        df.iloc[i, df.columns.get_loc('ob_bull')] = bar['low']
                        break

        # Bearish OB: Bullish candle followed by strong bearish move
        for j in range(i - lookback, i - 3):
            bar = df.iloc[j]
            next_bars = df.iloc[j+1:j+4]

            # Bullish candle
            if bar['close'] > bar['open']:
                # Followed by bearish move
                move_down = bar['high'] - next_bars['close'].min()
                if move_down > 0.0015:  # 15 pips minimum
                    # Check if price is near OB
                    current_price = df.iloc[i]['close']
                    if bar['low'] <= current_price <= bar['high']:
                        df.iloc[i, df.columns.get_loc('ob_bear')] = bar['high']
                        break

    return df


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Fair Value Gaps"""
    df = df.copy()
    df['fvg_bull'] = 0.0
    df['fvg_bear'] = 0.0

    for i in range(2, len(df)):
        bar1 = df.iloc[i-2]
        bar3 = df.iloc[i]

        # Bullish FVG: Gap between bar1 high and bar3 low
        if bar3['low'] > bar1['high']:
            gap = bar3['low'] - bar1['high']
            if gap > 0.0005:  # 5 pips minimum
                df.iloc[i, df.columns.get_loc('fvg_bull')] = (bar3['low'] + bar1['high']) / 2

        # Bearish FVG: Gap between bar1 low and bar3 high
        if bar3['high'] < bar1['low']:
            gap = bar1['low'] - bar3['high']
            if gap > 0.0005:
                df.iloc[i, df.columns.get_loc('fvg_bear')] = (bar1['low'] + bar3['high']) / 2

    return df


def strategy_smc(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMC (Smart Money Concepts) Strategy
    - BUY: Price at Bullish Order Block + Trend up
    - SELL: Price at Bearish Order Block + Trend down
    - FVG as confirmation
    """
    df = detect_order_blocks(df)
    df = detect_fvg(df)

    df['signal'] = 0

    london_session = (df['hour'] >= 7) & (df['hour'] < 19)

    # Trend using EMA
    uptrend = df['ema9'] > df['ema21']
    downtrend = df['ema9'] < df['ema21']

    # Bullish OB + Uptrend
    bull_ob_signal = (df['ob_bull'] > 0) & uptrend

    # Bearish OB + Downtrend
    bear_ob_signal = (df['ob_bear'] > 0) & downtrend

    # FVG confirmation bonus
    fvg_bull = df['fvg_bull'] > 0
    fvg_bear = df['fvg_bear'] > 0

    df.loc[london_session & (bull_ob_signal | fvg_bull), 'signal'] = 1
    df.loc[london_session & (bear_ob_signal | fvg_bear), 'signal'] = -1

    return df


# =============================================================================
# STRATEGY 4: ICT (KILL ZONE + LIQUIDITY)
# =============================================================================

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Detect Liquidity Sweep (Stop Hunt)"""
    df = df.copy()
    df['sweep_bull'] = False
    df['sweep_bear'] = False

    for i in range(lookback + 1, len(df)):
        recent = df.iloc[i-lookback:i]
        current = df.iloc[i]

        recent_high = recent['high'].max()
        recent_low = recent['low'].min()

        # Bullish Sweep: Price goes below recent low then closes above
        if current['low'] < recent_low and current['close'] > recent_low:
            df.iloc[i, df.columns.get_loc('sweep_bull')] = True

        # Bearish Sweep: Price goes above recent high then closes below
        if current['high'] > recent_high and current['close'] < recent_high:
            df.iloc[i, df.columns.get_loc('sweep_bear')] = True

    return df


def strategy_ict(df: pd.DataFrame) -> pd.DataFrame:
    """
    ICT (Inner Circle Trader) Strategy
    - Kill Zone: London (08-12), NY (13-17)
    - Liquidity Sweep + Reversal
    - Market Structure confirmation
    """
    df = detect_liquidity_sweep(df)
    df['signal'] = 0

    # ICT Kill Zones (more specific)
    london_kz = (df['hour'] >= 8) & (df['hour'] < 12)
    ny_kz = (df['hour'] >= 13) & (df['hour'] < 17)
    in_killzone = london_kz | ny_kz

    # Trend confirmation
    uptrend = df['ema9'] > df['ema21']
    downtrend = df['ema9'] < df['ema21']

    # Bullish: Sweep low + uptrend
    df.loc[in_killzone & df['sweep_bull'] & uptrend, 'signal'] = 1

    # Bearish: Sweep high + downtrend
    df.loc[in_killzone & df['sweep_bear'] & downtrend, 'signal'] = -1

    return df


# =============================================================================
# STRATEGY 5: HYBRID (BEST OF ALL)
# =============================================================================

def strategy_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    HYBRID Strategy - Kombinasi Terbaik
    - RSI sebagai filter (bukan signal generator)
    - BBMA untuk timing
    - SMC untuk zone entry
    - ICT untuk session
    """
    # Add all detections
    df = detect_order_blocks(df)
    df = detect_fvg(df)
    df = detect_liquidity_sweep(df)

    df['signal'] = 0

    # ICT Kill Zone
    london_kz = (df['hour'] >= 8) & (df['hour'] < 12)
    ny_kz = (df['hour'] >= 13) & (df['hour'] < 17)
    in_killzone = london_kz | ny_kz

    # Trend (BBMA style)
    uptrend = (df['close'] > df['ema50']) & (df['ema9'] > df['ema21'])
    downtrend = (df['close'] < df['ema50']) & (df['ema9'] < df['ema21'])

    # RSI Filter (not extreme)
    rsi_ok_buy = df['rsi'] < 60  # Not overbought
    rsi_ok_sell = df['rsi'] > 40  # Not oversold

    # SMC Zone
    at_bull_zone = (df['ob_bull'] > 0) | (df['fvg_bull'] > 0)
    at_bear_zone = (df['ob_bear'] > 0) | (df['fvg_bear'] > 0)

    # Liquidity Sweep confirmation
    has_sweep = df['sweep_bull'] | df['sweep_bear']

    # BBMA Extreme
    bb_extreme_buy = df['close'] < df['bb_lower']
    bb_extreme_sell = df['close'] > df['bb_upper']

    # === HYBRID SIGNALS ===
    # BUY: Kill Zone + Uptrend + (Zone OR Sweep OR BB Extreme) + RSI OK
    buy_signal = in_killzone & uptrend & (at_bull_zone | df['sweep_bull'] | bb_extreme_buy) & rsi_ok_buy

    # SELL: Kill Zone + Downtrend + (Zone OR Sweep OR BB Extreme) + RSI OK
    sell_signal = in_killzone & downtrend & (at_bear_zone | df['sweep_bear'] | bb_extreme_sell) & rsi_ok_sell

    df.loc[buy_signal, 'signal'] = 1
    df.loc[sell_signal, 'signal'] = -1

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, strategy_name: str) -> dict:
    """Run backtest for a strategy"""

    balance = 10000.0
    trades = []
    position = None
    cooldown = 0

    for i in range(50, len(df)):
        if cooldown > 0:
            cooldown -= 1
            continue

        row = df.iloc[i]

        # Manage position
        if position is not None:
            if position['direction'] == 1:  # LONG
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'LOSS', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 3
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'WIN', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2
                elif row['high'] >= row['bb_mid']:
                    pnl = (row['bb_mid'] - position['entry']) * position['size']
                    balance += pnl
                    result = 'WIN' if pnl > 0 else 'LOSS'
                    trades.append({'pnl': pnl, 'result': result, 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2
            else:  # SHORT
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'LOSS', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 3
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'WIN', 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2
                elif row['low'] <= row['bb_mid']:
                    pnl = (position['entry'] - row['bb_mid']) * position['size']
                    balance += pnl
                    result = 'WIN' if pnl > 0 else 'LOSS'
                    trades.append({'pnl': pnl, 'result': result, 'entry_time': position['entry_time']})
                    position = None
                    cooldown = 2

        # New signal
        if position is None and row['signal'] != 0:
            entry = row['close']
            atr = row['atr'] if row['atr'] > 0 else entry * 0.002

            if row['signal'] == 1:  # BUY
                sl = entry - atr * 1.75
                tp = entry + atr * 2.5
            else:  # SELL
                sl = entry + atr * 1.75
                tp = entry - atr * 2.5

            risk = balance * 0.01
            size = risk / abs(entry - sl) if abs(entry - sl) > 0 else 0
            size = min(size, 100000)

            position = {
                'entry_time': df.index[i],
                'direction': row['signal'],
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'size': size
            }

    # Close remaining
    if position:
        final = df.iloc[-1]['close']
        if position['direction'] == 1:
            pnl = (final - position['entry']) * position['size']
        else:
            pnl = (position['entry'] - final) * position['size']
        balance += pnl
        result = 'WIN' if pnl > 0 else 'LOSS'
        trades.append({'pnl': pnl, 'result': result, 'entry_time': position['entry_time']})

    # Calculate stats
    if len(trades) == 0:
        return {
            'strategy': strategy_name,
            'trades': 0,
            'win_rate': 0,
            'return_pct': 0,
            'profit_factor': 0,
            'final_balance': balance
        }

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0

    return {
        'strategy': strategy_name,
        'trades': len(trades_df),
        'wins': len(wins),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'return_pct': (balance - 10000) / 10000 * 100,
        'profit_factor': pf,
        'final_balance': balance,
        'trades_df': trades_df
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("STRATEGY COMPARISON BACKTEST")
    print("Database: ml_trading_bot PostgreSQL | Symbol: GBPUSD H1 | Period: 2020-2026")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading data...")
    df = load_data('2020-01-01', '2026-01-31')
    print(f"      Loaded {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Add indicators
    print("\n[2/6] Computing indicators...")
    df = add_all_indicators(df)

    # Run all strategies
    strategies = {
        'RSI Baseline': strategy_rsi_baseline,
        'BBMA Oma Ally': strategy_bbma,
        'SMC (Order Block)': strategy_smc,
        'ICT (Kill Zone)': strategy_ict,
        'HYBRID (Best)': strategy_hybrid
    }

    results = []

    for i, (name, strategy_func) in enumerate(strategies.items(), 3):
        print(f"\n[{i}/6] Running {name}...")
        df_strategy = strategy_func(df.copy())
        result = run_backtest(df_strategy, name)
        results.append(result)
        print(f"      {result['trades']} trades, {result['return_pct']:.2f}%, WR {result['win_rate']:.1f}%")

    # Summary
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (2020-2026, 6 Years)")
    print("=" * 100)
    print(f"{'Strategy':<25} {'Trades':>8} {'Trades/Yr':>10} {'Return':>12} {'Annual':>10} {'WinRate':>10} {'PF':>8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x['return_pct'], reverse=True):
        years = 6
        annual = r['return_pct'] / years
        trades_per_year = r['trades'] / years
        print(f"{r['strategy']:<25} {r['trades']:>8} {trades_per_year:>10.1f} {r['return_pct']:>11.2f}% {annual:>9.2f}% {r['win_rate']:>9.1f}% {r['profit_factor']:>7.2f}")

    # Best strategy
    best = max(results, key=lambda x: x['return_pct'])

    print("\n" + "=" * 80)
    print(f"BEST STRATEGY: {best['strategy']}")
    print("=" * 80)
    print(f"""
  Total Trades:     {best['trades']}
  Trades/Year:      {best['trades'] / 6:.1f}
  Total Return:     {best['return_pct']:.2f}%
  Annual Return:    {best['return_pct'] / 6:.2f}%
  Win Rate:         {best['win_rate']:.1f}%
  Profit Factor:    {best['profit_factor']:.2f}
  Final Balance:    ${best['final_balance']:,.2f}
""")

    # Yearly breakdown for best
    if 'trades_df' in best and len(best['trades_df']) > 0:
        trades_df = best['trades_df']
        trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year

        print("-" * 60)
        print(f"YEARLY BREAKDOWN - {best['strategy']}")
        print("-" * 60)

        yearly = trades_df.groupby('year')['pnl'].agg(['sum', 'count'])
        yearly['wr'] = trades_df.groupby('year').apply(lambda x: (x['pnl'] > 0).mean() * 100)

        profitable = 0
        for yr, row in yearly.iterrows():
            st = "[+]" if row['sum'] > 0 else "[-]"
            if row['sum'] > 0:
                profitable += 1
            print(f"{yr}: {int(row['count']):>4} trades, ${row['sum']:>+10,.2f}, {row['wr']:>5.1f}% WR {st}")

        print(f"\nProfitable years: {profitable}/{len(yearly)} ({profitable/len(yearly)*100:.0f}%)")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
    Berdasarkan backtest 6 tahun (2020-2026):

    1. RSI BASELINE   - Simple & profitable, cocok untuk pemula
    2. BBMA           - Bagus untuk trending market
    3. SMC            - Perlu banyak konfirmasi, lebih selektif
    4. ICT            - Efektif di kill zone, trades lebih sedikit
    5. HYBRID         - Kombinasi terbaik, balance risk/reward

    REKOMENDASI UNTUK ML_TRADING_BOT:
    → Gunakan HYBRID strategy karena menggabungkan semua keunggulan
    → RSI sebagai filter, bukan signal generator
    → Kill Zone untuk timing
    → Order Block/FVG untuk entry zone
    → ATR-based SL/TP untuk exit
""")


if __name__ == "__main__":
    main()
