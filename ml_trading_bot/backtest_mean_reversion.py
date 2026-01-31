"""
Mean Reversion Strategy with RSI
Goal: 120+ trades with profitability
Key concept: Buy oversold in ranging market, sell overbought in ranging market
Higher win rate expected due to mean reversion nature
"""

import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}

def get_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch GBPUSD H1 data from TimescaleDB"""
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
    """Calculate technical indicators"""
    # EMAs for trend context
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

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

    # ADX for ranging detection
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0),
        0
    )
    df['minus_dm'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0),
        0
    )
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic RSI for better oversold/overbought signals
    rsi_period = 14
    df['rsi_min'] = df['rsi'].rolling(rsi_period).min()
    df['rsi_max'] = df['rsi'].rolling(rsi_period).max()
    df['stoch_rsi'] = (df['rsi'] - df['rsi_min']) / (df['rsi_max'] - df['rsi_min']) * 100

    # Volatility regime
    df['volatility'] = df['atr'] / df['close'] * 100
    df['vol_ma'] = df['volatility'].rolling(50).mean()
    df['normal_volatility'] = (df['volatility'] <= df['vol_ma'] * 1.3) & (df['volatility'] >= df['vol_ma'] * 0.7)

    # Ranging detection
    df['is_ranging'] = df['adx'] < 25

    # Price at extremes (for mean reversion)
    df['at_bb_lower'] = df['close'] <= df['bb_lower']
    df['at_bb_upper'] = df['close'] >= df['bb_upper']

    return df

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10000,
    risk_per_trade: float = 0.01,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    require_ranging: bool = True,
    max_adx: float = 30,
    use_bollinger: bool = True,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 2.0,
    cooldown_bars: int = 4,
    trading_hours: tuple = (7, 19),
) -> dict:
    """Run mean reversion backtest"""

    balance = initial_balance
    equity_curve = [balance]
    trades = []
    position = None
    bars_since_trade = cooldown_bars

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # Check trading hours
        hour = row['time'].hour
        if hour < trading_hours[0] or hour >= trading_hours[1]:
            if position is None:
                bars_since_trade += 1
            continue

        # Manage existing position
        if position is not None:
            if position['direction'] == 'BUY':
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    position['exit'] = position['sl']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'SL'
                    trades.append(position)
                    position = None
                    bars_since_trade = 0
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    position['exit'] = position['tp']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'TP'
                    trades.append(position)
                    position = None
                    bars_since_trade = 0
                # Additional exit: close at mean (BB mid)
                elif row['high'] >= row['bb_mid']:
                    exit_price = row['bb_mid']
                    pnl = (exit_price - position['entry']) * position['size']
                    balance += pnl
                    position['exit'] = exit_price
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'MEAN'
                    trades.append(position)
                    position = None
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
                    bars_since_trade = 0
                # Additional exit: close at mean (BB mid)
                elif row['low'] <= row['bb_mid']:
                    exit_price = row['bb_mid']
                    pnl = (position['entry'] - exit_price) * position['size']
                    balance += pnl
                    position['exit'] = exit_price
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'MEAN'
                    trades.append(position)
                    position = None
                    bars_since_trade = 0

        equity_curve.append(balance)

        # Check for new signals
        if position is None and bars_since_trade >= cooldown_bars:
            signal = None

            # Skip if indicators are NaN
            if pd.isna(row['adx']) or pd.isna(row['rsi']):
                bars_since_trade += 1
                continue

            # ADX filter for ranging market
            if require_ranging and row['adx'] > max_adx:
                bars_since_trade += 1
                continue

            # Skip high/low volatility
            if not row['normal_volatility']:
                bars_since_trade += 1
                continue

            # Mean reversion signals
            # BUY when RSI oversold AND (optional) at BB lower
            if row['rsi'] < rsi_oversold:
                if not use_bollinger or row['at_bb_lower']:
                    # Confirm reversal - current close > open
                    if row['close'] > row['open']:
                        signal = 'BUY'

            # SELL when RSI overbought AND (optional) at BB upper
            elif row['rsi'] > rsi_overbought:
                if not use_bollinger or row['at_bb_upper']:
                    # Confirm reversal - current close < open
                    if row['close'] < row['open']:
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

                # Position sizing
                risk_amount = balance * risk_per_trade
                pip_risk = abs(entry - sl)
                size = risk_amount / pip_risk if pip_risk > 0 else 0
                size = min(size, 100000)  # Max 1 lot

                position = {
                    'entry_time': row['time'],
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'atr': atr,
                    'adx': row['adx'],
                    'rsi': row['rsi'],
                }
                bars_since_trade = 0
        else:
            bars_since_trade += 1

    # Close any remaining position
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

    # Statistics
    if len(trades) == 0:
        return {'trades': 0, 'message': 'No trades'}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    total_pnl = trades_df['pnl'].sum()
    win_rate = len(wins) / len(trades_df) * 100

    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
    profit_factor = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # Exit reason breakdown
    exit_stats = trades_df.groupby('exit_reason').agg({
        'pnl': ['sum', 'count', 'mean']
    })
    exit_stats.columns = ['total_pnl', 'count', 'avg_pnl']

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
        'exit_stats': exit_stats,
    }


def main():
    print("=" * 60)
    print("MEAN REVERSION BACKTEST - RSI + BOLLINGER BANDS")
    print("=" * 60)

    # Fetch 6 years of data
    print("\nFetching data from 2020-01-01 to 2026-01-31...")
    df = get_historical_data('2020-01-01', '2026-01-31')
    print(f"Total bars: {len(df)}")

    df = calculate_indicators(df)

    # Config 1: Standard mean reversion
    print("\n" + "-" * 60)
    print("CONFIG 1: RSI 30/70 + BB in Ranging (ADX < 25)")
    print("-" * 60)

    result1 = run_backtest(
        df,
        rsi_oversold=30,
        rsi_overbought=70,
        require_ranging=True,
        max_adx=25,
        use_bollinger=True,
    )
    print(f"Total trades: {result1['trades']}")
    print(f"Win rate: {result1['win_rate']:.1f}%")
    print(f"Total P&L: ${result1['total_pnl']:.2f}")
    print(f"Return: {result1['return_pct']:.2f}%")
    print(f"Profit Factor: {result1['profit_factor']:.2f}")
    print(f"Max Drawdown: {result1['max_drawdown']:.2f}%")

    # Config 2: Relaxed ADX filter
    print("\n" + "-" * 60)
    print("CONFIG 2: RSI 30/70 + BB (ADX < 35)")
    print("-" * 60)

    result2 = run_backtest(
        df,
        rsi_oversold=30,
        rsi_overbought=70,
        require_ranging=True,
        max_adx=35,
        use_bollinger=True,
    )
    print(f"Total trades: {result2['trades']}")
    print(f"Win rate: {result2['win_rate']:.1f}%")
    print(f"Total P&L: ${result2['total_pnl']:.2f}")
    print(f"Return: {result2['return_pct']:.2f}%")
    print(f"Profit Factor: {result2['profit_factor']:.2f}")
    print(f"Max Drawdown: {result2['max_drawdown']:.2f}%")

    # Config 3: RSI only (no BB requirement)
    print("\n" + "-" * 60)
    print("CONFIG 3: RSI 25/75 only (no BB) + ADX < 30")
    print("-" * 60)

    result3 = run_backtest(
        df,
        rsi_oversold=25,
        rsi_overbought=75,
        require_ranging=True,
        max_adx=30,
        use_bollinger=False,
    )
    print(f"Total trades: {result3['trades']}")
    print(f"Win rate: {result3['win_rate']:.1f}%")
    print(f"Total P&L: ${result3['total_pnl']:.2f}")
    print(f"Return: {result3['return_pct']:.2f}%")
    print(f"Profit Factor: {result3['profit_factor']:.2f}")
    print(f"Max Drawdown: {result3['max_drawdown']:.2f}%")

    # Config 4: More extreme RSI levels
    print("\n" + "-" * 60)
    print("CONFIG 4: RSI 20/80 (extreme) + ADX < 35")
    print("-" * 60)

    result4 = run_backtest(
        df,
        rsi_oversold=20,
        rsi_overbought=80,
        require_ranging=True,
        max_adx=35,
        use_bollinger=False,
    )
    print(f"Total trades: {result4['trades']}")
    print(f"Win rate: {result4['win_rate']:.1f}%")
    print(f"Total P&L: ${result4['total_pnl']:.2f}")
    print(f"Return: {result4['return_pct']:.2f}%")
    print(f"Profit Factor: {result4['profit_factor']:.2f}")
    print(f"Max Drawdown: {result4['max_drawdown']:.2f}%")

    # Config 5: All market conditions (no ranging filter)
    print("\n" + "-" * 60)
    print("CONFIG 5: RSI 30/70 All Markets (no ADX filter)")
    print("-" * 60)

    result5 = run_backtest(
        df,
        rsi_oversold=30,
        rsi_overbought=70,
        require_ranging=False,
        use_bollinger=False,
        cooldown_bars=3,
    )
    print(f"Total trades: {result5['trades']}")
    print(f"Win rate: {result5['win_rate']:.1f}%")
    print(f"Total P&L: ${result5['total_pnl']:.2f}")
    print(f"Return: {result5['return_pct']:.2f}%")
    print(f"Profit Factor: {result5['profit_factor']:.2f}")
    print(f"Max Drawdown: {result5['max_drawdown']:.2f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results = [
        ('Config 1: Strict MR', result1),
        ('Config 2: Relaxed ADX', result2),
        ('Config 3: RSI only', result3),
        ('Config 4: Extreme RSI', result4),
        ('Config 5: All Markets', result5),
    ]

    for name, r in results:
        if r['trades'] > 0:
            status = "[OK]" if r['trades'] >= 120 else "[LOW]"
            profit_status = "[OK]" if r['return_pct'] > 0 else "[LOSS]"
            print(f"{name}: {r['trades']} trades {status}, {r['return_pct']:.2f}% {profit_status}, WR: {r['win_rate']:.1f}%")

    # Check for best result
    valid = [(n, r) for n, r in results if r['trades'] >= 120 and r['return_pct'] > 0]
    if valid:
        best = max(valid, key=lambda x: x[1]['return_pct'])
        print(f"\n[OK] Best config: {best[0]} with {best[1]['return_pct']:.2f}% return")
        print("\nExit reason breakdown:")
        print(best[1]['exit_stats'].to_string())
    else:
        print("\n[X] No config achieved 120+ profitable trades")


if __name__ == "__main__":
    main()
