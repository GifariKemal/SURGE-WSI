"""
Optimized Mean Reversion Strategy
Based on Config 5 which achieved 665 trades, 51.89% return, 53.8% WR
Goal: Optimize to improve profit factor and reduce drawdown
"""

import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}

def get_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = 'GBPUSD'
        AND timeframe = 'H1'
        AND time >= %s AND time <= %s
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volatility filter
    df['volatility'] = df['atr'] / df['close'] * 100
    df['vol_ma'] = df['volatility'].rolling(50).mean()
    df['vol_extreme'] = df['volatility'] > df['vol_ma'] * 2  # Skip extreme vol

    # Price momentum
    df['momentum'] = df['close'].pct_change(5) * 100

    # Candle patterns
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = df['candle_body'] / df['candle_range']

    return df

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10000,
    risk_per_trade: float = 0.01,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    require_reversal_candle: bool = True,
    skip_extreme_vol: bool = True,
    exit_at_mean: bool = True,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 2.5,
    cooldown_bars: int = 3,
    trading_hours: tuple = (7, 19),
    max_holding_bars: int = 24,  # Force exit after 24 bars
) -> dict:

    balance = initial_balance
    equity_curve = [balance]
    trades = []
    position = None
    bars_since_trade = cooldown_bars
    position_bars = 0

    for i in range(50, len(df)):
        row = df.iloc[i]

        hour = row['time'].hour
        in_trading_hours = trading_hours[0] <= hour < trading_hours[1]

        # Manage position
        if position is not None:
            position_bars += 1

            # Force exit after max holding bars
            if position_bars >= max_holding_bars:
                exit_price = row['close']
                if position['direction'] == 'BUY':
                    pnl = (exit_price - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - exit_price) * position['size']
                balance += pnl
                position['exit'] = exit_price
                position['exit_time'] = row['time']
                position['pnl'] = pnl
                position['exit_reason'] = 'TIME'
                trades.append(position)
                position = None
                position_bars = 0
                bars_since_trade = 0
                equity_curve.append(balance)
                continue

            if position['direction'] == 'BUY':
                # SL
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    position['exit'] = position['sl']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'SL'
                    trades.append(position)
                    position = None
                    position_bars = 0
                    bars_since_trade = 0
                # TP
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    position['exit'] = position['tp']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'TP'
                    trades.append(position)
                    position = None
                    position_bars = 0
                    bars_since_trade = 0
                # Mean exit
                elif exit_at_mean and row['high'] >= row['bb_mid']:
                    exit_price = row['bb_mid']
                    pnl = (exit_price - position['entry']) * position['size']
                    balance += pnl
                    position['exit'] = exit_price
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'MEAN'
                    trades.append(position)
                    position = None
                    position_bars = 0
                    bars_since_trade = 0
            else:  # SELL
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    position['exit'] = position['sl']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'SL'
                    trades.append(position)
                    position = None
                    position_bars = 0
                    bars_since_trade = 0
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    position['exit'] = position['tp']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'TP'
                    trades.append(position)
                    position = None
                    position_bars = 0
                    bars_since_trade = 0
                elif exit_at_mean and row['low'] <= row['bb_mid']:
                    exit_price = row['bb_mid']
                    pnl = (position['entry'] - exit_price) * position['size']
                    balance += pnl
                    position['exit'] = exit_price
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'MEAN'
                    trades.append(position)
                    position = None
                    position_bars = 0
                    bars_since_trade = 0

        equity_curve.append(balance)

        # New signals
        if position is None and bars_since_trade >= cooldown_bars and in_trading_hours:
            signal = None

            if pd.isna(row['rsi']):
                bars_since_trade += 1
                continue

            # Skip extreme volatility
            if skip_extreme_vol and row['vol_extreme']:
                bars_since_trade += 1
                continue

            # BUY: RSI oversold
            if row['rsi'] < rsi_oversold:
                # Reversal candle: close > open (bullish)
                if not require_reversal_candle or row['close'] > row['open']:
                    signal = 'BUY'

            # SELL: RSI overbought
            elif row['rsi'] > rsi_overbought:
                if not require_reversal_candle or row['close'] < row['open']:
                    signal = 'SELL'

            if signal:
                atr = row['atr']
                entry = row['close']

                if signal == 'BUY':
                    sl = entry - (atr * atr_sl_mult)
                    tp = entry + (atr * atr_tp_mult)
                else:
                    sl = entry + (atr * atr_sl_mult)
                    tp = entry - (atr * atr_tp_mult)

                risk_amount = balance * risk_per_trade
                pip_risk = abs(entry - sl)
                size = risk_amount / pip_risk if pip_risk > 0 else 0
                size = min(size, 100000)

                position = {
                    'entry_time': row['time'],
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'rsi': row['rsi'],
                }
                bars_since_trade = 0
                position_bars = 0
        else:
            bars_since_trade += 1

    # Close remaining
    if position is not None:
        final_price = df.iloc[-1]['close']
        if position['direction'] == 'BUY':
            pnl = (final_price - position['entry']) * position['size']
        else:
            pnl = (position['entry'] - final_price) * position['size']
        balance += pnl
        position['exit'] = final_price
        position['exit_time'] = df.iloc[-1]['time']
        position['pnl'] = pnl
        position['exit_reason'] = 'CLOSE'
        trades.append(position)

    # Stats
    if len(trades) == 0:
        return {'trades': 0}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    total_pnl = trades_df['pnl'].sum()
    win_rate = len(wins) / len(trades_df) * 100

    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
    profit_factor = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # Yearly breakdown
    trades_df['year'] = trades_df['entry_time'].dt.year
    yearly = trades_df.groupby('year').agg({
        'pnl': ['sum', 'count']
    })
    yearly.columns = ['pnl', 'trades']

    exit_stats = trades_df.groupby('exit_reason').agg({
        'pnl': ['sum', 'count']
    })
    exit_stats.columns = ['total_pnl', 'count']

    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'return_pct': (balance - initial_balance) / initial_balance * 100,
        'final_balance': balance,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'trades_df': trades_df,
        'yearly': yearly,
        'exit_stats': exit_stats,
    }


def main():
    print("=" * 60)
    print("OPTIMIZED MEAN REVERSION STRATEGY")
    print("=" * 60)

    print("\nFetching 6 years of data (2020-2026)...")
    df = get_historical_data('2020-01-01', '2026-01-31')
    print(f"Total bars: {len(df)}")
    df = calculate_indicators(df)

    configs = [
        ("Baseline (Previous Best)", {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'require_reversal_candle': True,
            'skip_extreme_vol': False,
            'exit_at_mean': True,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.0,
            'cooldown_bars': 3,
        }),
        ("Optimized SL/TP", {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'require_reversal_candle': True,
            'skip_extreme_vol': True,
            'exit_at_mean': True,
            'atr_sl_mult': 1.2,  # Tighter SL
            'atr_tp_mult': 2.5,  # Wider TP
            'cooldown_bars': 3,
        }),
        ("Higher RSI Threshold", {
            'rsi_oversold': 25,  # More extreme
            'rsi_overbought': 75,
            'require_reversal_candle': True,
            'skip_extreme_vol': True,
            'exit_at_mean': True,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.0,
            'cooldown_bars': 3,
        }),
        ("More Trades (RSI 35/65)", {
            'rsi_oversold': 35,  # More signals
            'rsi_overbought': 65,
            'require_reversal_candle': True,
            'skip_extreme_vol': True,
            'exit_at_mean': True,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.0,
            'cooldown_bars': 2,  # Less cooldown
        }),
        ("No Mean Exit (Pure SL/TP)", {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'require_reversal_candle': True,
            'skip_extreme_vol': True,
            'exit_at_mean': False,  # No mean exit
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.5,
            'cooldown_bars': 3,
        }),
        ("Aggressive (RSI 35/65 + Short CD)", {
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'require_reversal_candle': False,  # No reversal needed
            'skip_extreme_vol': True,
            'exit_at_mean': True,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.0,
            'cooldown_bars': 2,
        }),
    ]

    results = []
    for name, params in configs:
        print(f"\n{'-'*60}")
        print(f"{name}")
        print('-'*60)

        result = run_backtest(df, **params)
        result['name'] = name
        results.append(result)

        print(f"Trades: {result['trades']}")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Return: {result['return_pct']:.2f}%")
        print(f"P/F: {result['profit_factor']:.2f}")
        print(f"Max DD: {result['max_drawdown']:.2f}%")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Config':<30} {'Trades':>8} {'WR%':>8} {'Return%':>10} {'P/F':>8} {'MaxDD%':>10}")
    print("-" * 80)

    for r in results:
        status = "[OK]" if r['trades'] >= 120 and r['return_pct'] > 0 else "    "
        print(f"{r['name']:<30} {r['trades']:>8} {r['win_rate']:>7.1f}% {r['return_pct']:>9.2f}% {r['profit_factor']:>7.2f} {r['max_drawdown']:>9.2f}% {status}")

    # Best config
    valid = [r for r in results if r['trades'] >= 120 and r['return_pct'] > 0]
    if valid:
        best = max(valid, key=lambda x: x['return_pct'])
        print(f"\n[OK] BEST CONFIG: {best['name']}")
        print(f"    {best['trades']} trades, {best['return_pct']:.2f}% return, {best['win_rate']:.1f}% WR")

        print("\nYearly Breakdown:")
        print(best['yearly'].to_string())

        print("\nExit Reasons:")
        print(best['exit_stats'].to_string())
    else:
        print("\n[X] No config met criteria (120+ trades AND profitable)")


if __name__ == "__main__":
    main()
