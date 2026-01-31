"""
v3.7 RSI Mean Reversion - Backtest with Telegram Report
========================================================
Run backtest and send results to Telegram.
"""
import asyncio
import pandas as pd
import numpy as np
import psycopg2
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.telegram import TelegramNotifier, TelegramFormatter

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

DB_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'surge_wsi',
    'user': 'surge_wsi',
    'password': 'surge_wsi_secret'
}

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def run_backtest():
    """Run v3.7 backtest and return results"""
    print("Loading data from database...")

    # Use context manager for proper connection handling
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql("""
            SELECT time, open, high, low, close
            FROM ohlcv
            WHERE symbol = 'GBPUSD' AND timeframe = 'H1'
            AND time >= '2020-01-01' AND time <= '2026-01-31'
            ORDER BY time ASC
        """, conn)

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    print(f"Loaded {len(df)} bars")

    # RSI(10) - with safe division
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = np.where(loss == 0, 100, gain / loss)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR(14)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()

    # ATR Percentile - optimized (10x faster than lambda)
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: (x.argsort().argsort()[-1] + 1) / len(x) * 100, raw=True)

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.ffill().fillna(0)

    # v3.7 Parameters
    SL_MULT = 1.5
    TP_LOW, TP_MED, TP_HIGH = 2.4, 3.0, 3.6
    MIN_ATR_PCT, MAX_ATR_PCT = 20, 80
    TIME_TP_BONUS = 0.35
    MAX_HOLDING = 46
    SKIP_HOURS = [12]
    RSI_OS, RSI_OB = 42, 58

    print("Running backtest...")
    balance = 10000.0
    initial_balance = 10000.0
    wins = losses = 0
    position = None
    peak = balance
    max_dd = 0

    trades_list = []
    monthly_pnl = {}
    yearly_pnl = {}

    for i in range(200, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        year = current_time.year
        month = current_time.strftime('%Y-%m')
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
                else:
                    losses += 1

                if month not in monthly_pnl:
                    monthly_pnl[month] = 0
                monthly_pnl[month] += pnl

                if year not in yearly_pnl:
                    yearly_pnl[year] = 0
                yearly_pnl[year] += pnl

                trades_list.append({'pnl': pnl, 'exit': exit_reason})
                position = None

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        if not position:
            if hour < 7 or hour >= 22 or hour in SKIP_HOURS:
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
                tp_mult = base_tp + TIME_TP_BONUS if 12 <= hour < 16 else base_tp
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * tp_mult if signal == 1 else entry - atr * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

    # Calculate results
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - initial_balance) / initial_balance * 100

    wins_list = [t['pnl'] for t in trades_list if t['pnl'] > 0]
    losses_list = [t['pnl'] for t in trades_list if t['pnl'] <= 0]
    gross_profit = sum(wins_list) if wins_list else 0
    gross_loss = abs(sum(losses_list)) if losses_list else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    avg_win = np.mean(wins_list) if wins_list else 0
    avg_loss = abs(np.mean(losses_list)) if losses_list else 0

    tp_exits = sum(1 for t in trades_list if t['exit'] == 'TP')
    sl_exits = sum(1 for t in trades_list if t['exit'] == 'SL')
    timeout_exits = sum(1 for t in trades_list if t['exit'] == 'TIMEOUT')

    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    total_months = len(monthly_pnl)

    results = {
        'period': '2020-01 to 2026-01',
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'timeout_exits': timeout_exits,
        'profitable_months': profitable_months,
        'total_months': total_months,
        'yearly_pnl': yearly_pnl,
    }

    print(f"Backtest complete: +{total_return:.1f}% return")
    return results


def format_backtest_message(results: dict) -> str:
    """Format backtest results for Telegram"""
    fmt = TelegramFormatter

    msg = f"{fmt.ROCKET} <b>RSI v3.7 Backtest Report</b>\n"
    msg += f"<i>GBPUSD H1 | {results['period']}</i>\n"

    # Performance
    msg += f"\n{fmt.TARGET} <b>Performance</b>\n"
    msg += f"{fmt.BRANCH} Initial: ${results['initial_balance']:,.0f}\n"
    msg += f"{fmt.BRANCH} Final: ${results['final_balance']:,.2f}\n"

    ret_emoji = fmt.ROCKET if results['total_return'] > 100 else fmt.CHECK
    msg += f"{fmt.BRANCH} Return: {ret_emoji} <b>+{results['total_return']:.1f}%</b>\n"
    msg += f"{fmt.LAST} Max DD: {results['max_drawdown']:.1f}%\n"

    # Trades
    msg += f"\n{fmt.CHART} <b>Trades</b>\n"
    msg += f"{fmt.BRANCH} Total: {results['total_trades']}\n"
    msg += f"{fmt.BRANCH} Winners: {results['wins']} ({results['win_rate']:.1f}%)\n"
    msg += f"{fmt.BRANCH} Losers: {results['losses']}\n"
    msg += f"{fmt.LAST} Profit Factor: {results['profit_factor']:.2f}\n"

    # Risk/Reward
    msg += f"\n{fmt.SHIELD} <b>Risk/Reward</b>\n"
    msg += f"{fmt.BRANCH} Avg Win: ${results['avg_win']:.0f}\n"
    msg += f"{fmt.BRANCH} Avg Loss: ${results['avg_loss']:.0f}\n"
    msg += f"{fmt.LAST} R:R = 1:{results['rr_ratio']:.2f}\n"

    # Exit Analysis
    msg += f"\n{fmt.MEMO} <b>Exit Analysis</b>\n"
    tp_pct = results['tp_exits'] / results['total_trades'] * 100
    sl_pct = results['sl_exits'] / results['total_trades'] * 100
    to_pct = results['timeout_exits'] / results['total_trades'] * 100
    msg += f"{fmt.BRANCH} TP: {results['tp_exits']} ({tp_pct:.0f}%)\n"
    msg += f"{fmt.BRANCH} SL: {results['sl_exits']} ({sl_pct:.0f}%)\n"
    msg += f"{fmt.LAST} Timeout: {results['timeout_exits']} ({to_pct:.0f}%)\n"

    # Consistency
    msg += f"\n{fmt.MONEY} <b>Consistency</b>\n"
    msg += f"{fmt.LAST} Profitable Months: {results['profitable_months']}/{results['total_months']} ({results['profitable_months']/results['total_months']*100:.0f}%)\n"

    # Yearly Performance
    msg += f"\n{fmt.BRAIN} <b>Yearly P/L</b>\n"
    yearly = results['yearly_pnl']
    years = sorted(yearly.keys())
    for i, year in enumerate(years):
        pnl = yearly[year]
        emoji = fmt.CHECK if pnl > 0 else fmt.CROSS
        is_last = (i == len(years) - 1)
        connector = fmt.LAST if is_last else fmt.BRANCH
        msg += f"{connector} {year}: {emoji} ${pnl:+,.0f}\n"

    # Footer
    msg += f"\n<code>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</code>"

    return msg


async def send_to_telegram(message: str):
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured!")
        return False

    notifier = TelegramNotifier(
        bot_token=TELEGRAM_BOT_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
        enabled=True
    )

    if await notifier.initialize():
        await notifier.send(message, force=True)
        print("Message sent to Telegram!")
        return True
    else:
        print("Failed to initialize Telegram")
        return False


async def main():
    print("=" * 60)
    print("RSI v3.7 BACKTEST WITH TELEGRAM REPORT")
    print("=" * 60)

    # Run backtest
    results = run_backtest()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Return: +{results['total_return']:.1f}%")
    print(f"Max DD: {results['max_drawdown']:.1f}%")
    print(f"Trades: {results['total_trades']} | WR: {results['win_rate']:.1f}%")
    print(f"PF: {results['profit_factor']:.2f} | R:R: 1:{results['rr_ratio']:.2f}")

    # Format and send to Telegram
    print("\nSending to Telegram...")
    message = format_backtest_message(results)
    await send_to_telegram(message)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
