"""
5-Year Optimized EMA Crossover Strategy
Goal: 120+ trades with profitability
Period: 2020-2026 (5+ years of data)
Key insights applied:
- EMA crossover is the most reliable signal
- Ranging regime performs best
- Use tighter risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,  # Docker TimescaleDB port
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
    # EMA crossover (fast/slow)
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # EMA crossover signals
    df['ema_cross_up'] = (df['ema_8'] > df['ema_21']) & (df['ema_8'].shift(1) <= df['ema_21'].shift(1))
    df['ema_cross_down'] = (df['ema_8'] < df['ema_21']) & (df['ema_8'].shift(1) >= df['ema_21'].shift(1))

    # Trend filter - price above/below EMA50
    df['trend_bullish'] = df['close'] > df['ema_50']
    df['trend_bearish'] = df['close'] < df['ema_50']

    # ATR for dynamic stops
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # ADX for trend strength
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

    # Volatility regime (simplified)
    df['volatility'] = df['atr'] / df['close'] * 100
    df['vol_ma'] = df['volatility'].rolling(50).mean()
    df['high_volatility'] = df['volatility'] > df['vol_ma'] * 1.5

    # Ranging detection (ADX < 25)
    df['is_ranging'] = df['adx'] < 25

    return df

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10000,
    risk_per_trade: float = 0.01,  # 1% risk per trade
    atr_sl_mult: float = 1.5,  # SL at 1.5x ATR
    atr_tp_mult: float = 2.5,  # TP at 2.5x ATR (1.67 R:R)
    use_trend_filter: bool = True,
    use_ranging_filter: bool = False,  # Disabled to get more trades
    max_adx: float = 35,  # Allow trades up to ADX 35
    cooldown_bars: int = 3,  # 3 bars between trades
    trading_hours: tuple = (7, 20),  # Extended London + NY session
) -> dict:
    """Run backtest with optimized parameters"""

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
            # Check stop loss
            if position['direction'] == 'BUY':
                if row['low'] <= position['sl']:
                    # Stop loss hit
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
                    # Take profit hit
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    position['exit'] = position['tp']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'TP'
                    trades.append(position)
                    position = None
                    bars_since_trade = 0
            else:  # SELL
                if row['high'] >= position['sl']:
                    # Stop loss hit
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
                    # Take profit hit
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    position['exit'] = position['tp']
                    position['exit_time'] = row['time']
                    position['pnl'] = pnl
                    position['exit_reason'] = 'TP'
                    trades.append(position)
                    position = None
                    bars_since_trade = 0

        equity_curve.append(balance)

        # Check for new signals if no position
        if position is None and bars_since_trade >= cooldown_bars:
            signal = None

            # ADX filter
            if pd.isna(row['adx']) or row['adx'] > max_adx:
                bars_since_trade += 1
                continue

            # Ranging filter (optional)
            if use_ranging_filter and not row['is_ranging']:
                bars_since_trade += 1
                continue

            # Skip high volatility periods (crisis)
            if row['high_volatility']:
                bars_since_trade += 1
                continue

            # EMA crossover signals
            if row['ema_cross_up']:
                # Trend filter
                if use_trend_filter and not row['trend_bullish']:
                    bars_since_trade += 1
                    continue
                signal = 'BUY'
            elif row['ema_cross_down']:
                # Trend filter
                if use_trend_filter and not row['trend_bearish']:
                    bars_since_trade += 1
                    continue
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

                # Position sizing based on risk
                risk_amount = balance * risk_per_trade
                pip_risk = abs(entry - sl)
                size = risk_amount / pip_risk if pip_risk > 0 else 0

                # Cap position size
                max_size = 100000  # 1 standard lot
                size = min(size, max_size)

                position = {
                    'entry_time': row['time'],
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'atr': atr,
                    'adx': row['adx'],
                    'regime': 'ranging' if row['is_ranging'] else 'trending'
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

    # Calculate statistics
    if len(trades) == 0:
        return {'trades': 0, 'message': 'No trades generated'}

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

    # Monthly breakdown
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly = trades_df.groupby('month').agg({
        'pnl': ['sum', 'count'],
    })
    monthly.columns = ['pnl', 'trades']
    monthly['win_rate'] = trades_df.groupby('month').apply(lambda x: (x['pnl'] > 0).sum() / len(x) * 100)

    # Regime breakdown
    regime_stats = trades_df.groupby('regime').agg({
        'pnl': ['sum', 'count', 'mean'],
    })
    regime_stats.columns = ['total_pnl', 'trades', 'avg_pnl']

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
        'monthly': monthly,
        'regime_stats': regime_stats,
        'equity_curve': equity_curve,
    }


def main():
    print("=" * 60)
    print("5-YEAR OPTIMIZED EMA CROSSOVER BACKTEST")
    print("=" * 60)

    # Fetch 5+ years of data
    print("\nFetching data from 2020-01-01 to 2026-01-31...")
    df = get_historical_data('2020-01-01', '2026-01-31')
    print(f"Total bars: {len(df)}")

    # Calculate indicators
    print("Calculating indicators...")
    df = calculate_indicators(df)

    # Configuration 1: EMA with trend filter, no ranging requirement
    print("\n" + "-" * 60)
    print("CONFIG 1: EMA Crossover + Trend Filter (ADX < 35)")
    print("-" * 60)

    result1 = run_backtest(
        df,
        use_trend_filter=True,
        use_ranging_filter=False,
        max_adx=35,
        atr_sl_mult=1.5,
        atr_tp_mult=2.5,
        cooldown_bars=3,
    )

    print(f"Total trades: {result1['trades']}")
    print(f"Win rate: {result1['win_rate']:.1f}%")
    print(f"Total P&L: ${result1['total_pnl']:.2f}")
    print(f"Return: {result1['return_pct']:.2f}%")
    print(f"Profit Factor: {result1['profit_factor']:.2f}")
    print(f"Max Drawdown: {result1['max_drawdown']:.2f}%")

    # Configuration 2: No trend filter, more trades
    print("\n" + "-" * 60)
    print("CONFIG 2: EMA Crossover Only (No Trend Filter, ADX < 40)")
    print("-" * 60)

    result2 = run_backtest(
        df,
        use_trend_filter=False,
        use_ranging_filter=False,
        max_adx=40,
        atr_sl_mult=1.5,
        atr_tp_mult=2.0,  # Tighter TP
        cooldown_bars=2,  # Less cooldown
    )

    print(f"Total trades: {result2['trades']}")
    print(f"Win rate: {result2['win_rate']:.1f}%")
    print(f"Total P&L: ${result2['total_pnl']:.2f}")
    print(f"Return: {result2['return_pct']:.2f}%")
    print(f"Profit Factor: {result2['profit_factor']:.2f}")
    print(f"Max Drawdown: {result2['max_drawdown']:.2f}%")

    # Configuration 3: Optimized for higher R:R
    print("\n" + "-" * 60)
    print("CONFIG 3: EMA + Higher R:R (1:3)")
    print("-" * 60)

    result3 = run_backtest(
        df,
        use_trend_filter=True,
        use_ranging_filter=False,
        max_adx=35,
        atr_sl_mult=1.0,  # Tighter SL
        atr_tp_mult=3.0,  # 1:3 R:R
        cooldown_bars=3,
    )

    print(f"Total trades: {result3['trades']}")
    print(f"Win rate: {result3['win_rate']:.1f}%")
    print(f"Total P&L: ${result3['total_pnl']:.2f}")
    print(f"Return: {result3['return_pct']:.2f}%")
    print(f"Profit Factor: {result3['profit_factor']:.2f}")
    print(f"Max Drawdown: {result3['max_drawdown']:.2f}%")

    # Configuration 4: Tighter R:R for higher win rate
    print("\n" + "-" * 60)
    print("CONFIG 4: EMA + Tighter R:R (1:1.5) for Higher WR")
    print("-" * 60)

    result4 = run_backtest(
        df,
        use_trend_filter=True,
        use_ranging_filter=False,
        max_adx=35,
        atr_sl_mult=1.5,
        atr_tp_mult=2.25,  # 1:1.5 R:R
        cooldown_bars=2,
    )

    print(f"Total trades: {result4['trades']}")
    print(f"Win rate: {result4['win_rate']:.1f}%")
    print(f"Total P&L: ${result4['total_pnl']:.2f}")
    print(f"Return: {result4['return_pct']:.2f}%")
    print(f"Profit Factor: {result4['profit_factor']:.2f}")
    print(f"Max Drawdown: {result4['max_drawdown']:.2f}%")

    # Find best config
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION ANALYSIS")
    print("=" * 60)

    results = [
        ('Config 1: Trend Filter', result1),
        ('Config 2: No Filter', result2),
        ('Config 3: High R:R', result3),
        ('Config 4: Tight R:R', result4),
    ]

    # Filter for 120+ trades and positive return
    valid_results = [(name, r) for name, r in results if r['trades'] >= 120 and r['return_pct'] > 0]

    if valid_results:
        best = max(valid_results, key=lambda x: x[1]['return_pct'])
        print(f"\nBest config meeting criteria (120+ trades, profitable):")
        print(f"  {best[0]}")
        print(f"  Trades: {best[1]['trades']}, Return: {best[1]['return_pct']:.2f}%")

        # Show monthly breakdown
        print("\nMonthly P&L Breakdown:")
        print(best[1]['monthly'].tail(12).to_string())

        # Show regime breakdown
        print("\nRegime Breakdown:")
        print(best[1]['regime_stats'].to_string())
    else:
        print("\nNo configuration met the criteria (120+ trades AND profitable)")
        print("Results summary:")
        for name, r in results:
            status = "[OK]" if r['trades'] >= 120 else "[LOW]"
            profit_status = "[OK]" if r['return_pct'] > 0 else "[LOSS]"
            print(f"  {name}: {r['trades']} trades {status}, {r['return_pct']:.2f}% {profit_status}")


if __name__ == "__main__":
    main()
