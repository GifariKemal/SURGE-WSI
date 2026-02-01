"""
RSI v3.7 OPTIMIZED - Backtest
=============================
Combined Filters Test: SIDEWAYS Regime + ConsecLoss3

Results (Oct 2024 - Jan 2026):
- Return: +72.7%
- Drawdown: 14.4%
- Win Rate: 37.6%
- Losing Months: 2/16

Usage:
    python backtest_rsi_v37_optimized.py
"""
import os
import asyncio
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

MT5_LOGIN = int(os.getenv('MT5_LOGIN', '61045904'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', 'FinexBisnisSolusi-Demo')
MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe")

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

SYMBOL = "GBPUSD"
INITIAL_BALANCE = 10000.0

# Strategy Parameters (must match live bot)
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

# Filter Settings
USE_REGIME_FILTER = True
ALLOWED_REGIMES = ['SIDEWAYS']
USE_CONSEC_LOSS_FILTER = True
CONSEC_LOSS_LIMIT = 3


# =============================================================================
# TELEGRAM FORMATTER
# =============================================================================

class TelegramFormatter:
    """Backtest report formatter for Telegram"""

    # Emojis
    ROCKET = "🚀"
    CHECK = "✅"
    CROSS = "❌"
    CHART = "📊"
    MONEY = "💰"
    TARGET = "🎯"
    CALENDAR = "📅"
    WARNING = "⚠️"
    BRANCH = "├"
    LAST = "└"

    @classmethod
    def backtest_report(cls, symbol: str, timeframe: str, period: str,
                        initial_balance: float, final_balance: float,
                        total_return: float, max_drawdown: float,
                        total_trades: int, wins: int, losses: int,
                        win_rate: float, profitable_months: int,
                        losing_months: int, filtered: int,
                        monthly_pnl: dict) -> str:
        """Format complete backtest report"""

        # Header with emoji based on performance
        if total_return >= 50:
            emoji = cls.ROCKET
        elif total_return > 0:
            emoji = cls.CHECK
        else:
            emoji = cls.CROSS

        msg = f"{emoji} <b>RSI v3.7 OPTIMIZED Backtest</b>\n\n"

        # Data section
        msg += f"{cls.CHART} <b>Data</b>\n"
        msg += f"{cls.BRANCH} Symbol: {symbol}\n"
        msg += f"{cls.BRANCH} Timeframe: {timeframe}\n"
        msg += f"{cls.LAST} Period: {period}\n"

        # Performance section
        msg += f"\n{cls.TARGET} <b>Performance</b>\n"
        msg += f"{cls.BRANCH} Initial: ${initial_balance:,.2f}\n"
        msg += f"{cls.BRANCH} Final: ${final_balance:,.2f}\n"
        ret_emoji = cls.CHECK if total_return > 0 else cls.CROSS
        msg += f"{cls.BRANCH} Return: {ret_emoji} <b>{total_return:+.1f}%</b>\n"
        msg += f"{cls.LAST} Max DD: {max_drawdown:.1f}%\n"

        # Trades section
        msg += f"\n{cls.CHART} <b>Trades</b>\n"
        msg += f"{cls.BRANCH} Total: {total_trades}\n"
        msg += f"{cls.BRANCH} Winners: {wins} ({win_rate:.1f}%)\n"
        msg += f"{cls.BRANCH} Losers: {losses}\n"
        msg += f"{cls.LAST} Filtered: {filtered}\n"

        # Months section
        msg += f"\n{cls.CALENDAR} <b>Months</b>\n"
        msg += f"{cls.BRANCH} Profitable: {profitable_months}\n"
        loss_emoji = cls.CHECK if losing_months <= 2 else cls.WARNING
        msg += f"{cls.LAST} Losing: {loss_emoji} {losing_months}\n"

        # Monthly breakdown
        msg += f"\n{cls.MONEY} <b>Monthly P/L</b>\n"
        msg += "<pre>"
        for m, p in sorted(monthly_pnl.items()):
            marker = " ❌" if p < 0 else ""
            msg += f"{m}: ${p:+,.0f}{marker}\n"
        msg += "</pre>"

        # Filters info
        msg += f"\n<i>Filters: SIDEWAYS + ConsecLoss3</i>"

        return msg

    @classmethod
    def backtest_summary(cls, total_return: float, max_drawdown: float,
                         total_trades: int, win_rate: float,
                         losing_months: int) -> str:
        """Format short backtest summary"""

        if total_return >= 50:
            emoji = cls.ROCKET
        elif total_return > 0:
            emoji = cls.CHECK
        else:
            emoji = cls.CROSS

        msg = f"{emoji} <b>Backtest Complete</b>\n\n"
        msg += f"Return: <b>{total_return:+.1f}%</b>\n"
        msg += f"Drawdown: {max_drawdown:.1f}%\n"
        msg += f"Trades: {total_trades} ({win_rate:.1f}% WR)\n"
        msg += f"Losing Months: {losing_months}"

        return msg


# =============================================================================
# TELEGRAM FUNCTIONS
# =============================================================================

async def send_telegram(message: str):
    """Send Telegram notification"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, timeout=10) as resp:
                if resp.status == 200:
                    print("Telegram: Report sent!")
                    return True
                else:
                    print(f"Telegram failed: {resp.status}")
                    return False
    except ImportError:
        print("Telegram: aiohttp not installed (pip install aiohttp)")
        return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def send_telegram_sync(message: str):
    """Synchronous wrapper for send_telegram"""
    try:
        return asyncio.run(send_telegram(message))
    except:
        # Fallback for environments where asyncio.run doesn't work
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(send_telegram(message))
        finally:
            loop.close()


# =============================================================================
# MT5 FUNCTIONS
# =============================================================================

def connect_mt5():
    if not MT5_PASSWORD:
        print("ERROR: MT5_PASSWORD not set")
        print("Use: set MT5_PASSWORD=your_password")
        return False
    if not mt5.initialize(path=MT5_PATH):
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        return False
    acc = mt5.account_info()
    print(f"Connected: {acc.login} | Balance: ${acc.balance:,.2f}")
    return True


def get_h1_data(symbol, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df


# =============================================================================
# INDICATORS
# =============================================================================

def prepare_indicators(df):
    """Calculate all indicators."""

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

    # SMAs for regime
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = (df['sma_20'] / df['sma_20'].shift(10) - 1) * 100

    # Regime Detection
    conditions = [
        (df['sma_20'] > df['sma_50']) & (df['sma_slope'] > 0.5),
        (df['sma_20'] < df['sma_50']) & (df['sma_slope'] < -0.5),
    ]
    df['regime'] = np.select(conditions, ['BULL', 'BEAR'], default='SIDEWAYS')

    # Time features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    return df.ffill().fillna(0)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df, test_start='2024-10-01', test_end='2026-02-01'):
    """Run RSI v3.7 OPTIMIZED backtest with SIDEWAYS + ConsecLoss3 filters."""

    balance = INITIAL_BALANCE
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0
    monthly_pnl = {}
    trades_taken = 0
    trades_filtered = 0
    consecutive_losses = 0

    trade_log = []

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

            if (i - position['entry_idx']) >= MAX_HOLDING_HOURS:
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

                trade_log.append({
                    'time': current_time,
                    'direction': 'BUY' if position['dir'] == 1 else 'SELL',
                    'entry': position['entry'],
                    'exit': row['close'],
                    'pnl': pnl,
                    'reason': exit_reason,
                    'balance': balance
                })

                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Entry logic with filters
        if not position and in_test:
            # Trading hours filter
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            # ATR percentile filter
            atr_pct = row['atr_pct']
            if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                continue

            skip_trade = False

            # Regime filter
            if USE_REGIME_FILTER:
                if row['regime'] not in ALLOWED_REGIMES:
                    skip_trade = True
                    trades_filtered += 1

            # Consecutive loss filter
            if not skip_trade and USE_CONSEC_LOSS_FILTER:
                if consecutive_losses >= CONSEC_LOSS_LIMIT:
                    skip_trade = True
                    trades_filtered += 1
                    consecutive_losses = 0  # Reset after pause

            if skip_trade:
                continue

            # RSI signal
            rsi = row['rsi']
            signal = 1 if rsi < RSI_OVERSOLD else (-1 if rsi > RSI_OVERBOUGHT else 0)

            if signal:
                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002

                # Dynamic TP
                base_tp = TP_LOW if atr_pct < 40 else (TP_HIGH if atr_pct > 60 else TP_MED)
                tp_mult = base_tp + 0.35 if 12 <= hour < 16 else base_tp

                # Calculate levels
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult

                # Position sizing
                risk = balance * RISK_PER_TRADE
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0

                position = {
                    'dir': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'entry_idx': i,
                    'regime': row['regime'],
                    'rsi': rsi
                }
                trades_taken += 1

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    losing_months = sum(1 for m in monthly_pnl.values() if m < 0)

    return {
        'balance': balance,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profitable_months': profitable_months,
        'losing_months': losing_months,
        'total_months': len(monthly_pnl),
        'monthly_pnl': monthly_pnl,
        'trades_filtered': trades_filtered,
        'trade_log': trade_log
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("RSI v3.7 OPTIMIZED - Backtest")
    print("SIDEWAYS Regime + ConsecLoss3 Filter")
    print("=" * 60)

    if not connect_mt5():
        return

    try:
        print(f"\nFetching {SYMBOL} H1 data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_h1_data(SYMBOL, start_date, end_date)

        if df is None or len(df) == 0:
            print("Failed to get data")
            return

        print(f"Loaded {len(df)} bars")

        print("Calculating indicators...")
        df = prepare_indicators(df)

        print("\nRunning backtest (Oct 2024 - Jan 2026)...")
        result = run_backtest(df.copy())

        # Results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nPerformance:")
        print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
        print(f"  Final Balance:   ${result['balance']:,.2f}")
        print(f"  Total Return:    {result['total_return']:+.1f}%")
        print(f"  Max Drawdown:    {result['max_drawdown']:.1f}%")

        print(f"\nTrades:")
        print(f"  Total Trades:    {result['total_trades']}")
        print(f"  Winners:         {result['wins']} ({result['win_rate']:.1f}%)")
        print(f"  Losers:          {result['losses']}")
        print(f"  Filtered:        {result['trades_filtered']}")

        print(f"\nMonths:")
        print(f"  Profitable:      {result['profitable_months']}")
        print(f"  Losing:          {result['losing_months']}")
        print(f"  Total:           {result['total_months']}")

        print("\nMonthly P/L:")
        for m, p in sorted(result['monthly_pnl'].items()):
            marker = " <-- LOSS" if p < 0 else ""
            print(f"  {m}: ${p:+,.2f}{marker}")

        # Show losing months detail
        losing = [(m, p) for m, p in result['monthly_pnl'].items() if p < 0]
        if losing:
            print(f"\nLosing Months Detail:")
            for m, p in losing:
                month_trades = [t for t in result['trade_log'] if t['time'].strftime('%Y-%m') == m]
                month_wins = sum(1 for t in month_trades if t['pnl'] > 0)
                month_losses = sum(1 for t in month_trades if t['pnl'] <= 0)
                print(f"  {m}: {len(month_trades)} trades, {month_wins}W/{month_losses}L, ${p:+.2f}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Strategy: RSI v3.7 OPTIMIZED")
        print(f"Filters:  SIDEWAYS Regime + ConsecLoss3")
        print(f"Return:   {result['total_return']:+.1f}%")
        print(f"Drawdown: {result['max_drawdown']:.1f}%")
        print(f"Losing:   {result['losing_months']} months")

        # Send Telegram report
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            print("\nSending Telegram report...")

            # Full report
            report = TelegramFormatter.backtest_report(
                symbol=SYMBOL,
                timeframe="H1",
                period="Oct 2024 - Jan 2026",
                initial_balance=INITIAL_BALANCE,
                final_balance=result['balance'],
                total_return=result['total_return'],
                max_drawdown=result['max_drawdown'],
                total_trades=result['total_trades'],
                wins=result['wins'],
                losses=result['losses'],
                win_rate=result['win_rate'],
                profitable_months=result['profitable_months'],
                losing_months=result['losing_months'],
                filtered=result['trades_filtered'],
                monthly_pnl=result['monthly_pnl']
            )
            send_telegram_sync(report)
        else:
            print("\nTelegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")

    finally:
        mt5.shutdown()
        print("\nDone!")


if __name__ == "__main__":
    main()
