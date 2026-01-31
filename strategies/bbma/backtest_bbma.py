"""
BBMA Strategy Backtest
======================
Bollinger Bands + Moving Average Strategy
Inspired by Malaysian BBMA technique (Sifu Zero, Omar Ali style)

Strategy Logic:
- Bollinger Bands (20, 2.0) for volatility & extreme detection
- EMA 5, 10, 50 for trend direction
- Entry: Price rejects from BB extreme, then retests MA
- Exit: Opposite BB band or trailing stop

Timeframe: H1
Pair: GBPUSD (can adapt to XAUUSD)
"""

import sys
import io
from pathlib import Path

# Strategy directory (where this file is located)
STRATEGY_DIR = Path(__file__).parent
# Project root is 2 levels up
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple
import asyncio
import os

from src.data.db_handler import DBHandler
from config import config


# Column mapping for database (db_handler returns title case columns)
COL_MAP = {'time': 'time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
PIP_SIZE = 0.0001


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: str  # 'BUY' or 'SELL'
    sl: float
    tp: float
    signal_type: str  # 'BB_REENTRY', 'MA_CROSS', 'EXTREME_REJECTION'
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    exit_reason: Optional[str] = None


class BBMABacktest:
    """
    BBMA Strategy Backtest Engine

    Entry Conditions:
    1. EXTREME_REJECTION: Price touches outer BB and reverses
    2. BB_REENTRY: After rejection, price retests EMA 5/10 zone
    3. MA_CROSS: EMA 5 crosses EMA 10 with confirmation

    Exit Conditions:
    1. Take Profit at opposite BB band
    2. Stop Loss (ATR-based or fixed pips)
    3. Trailing Stop after 50% profit reached
    """

    def __init__(self):
        # Bollinger Bands settings
        self.BB_PERIOD = 20
        self.BB_STD = 2.0

        # Moving Average settings
        self.EMA_FAST = 5
        self.EMA_MID = 10
        self.EMA_SLOW = 50

        # Risk Management
        self.RISK_PERCENT = 1.0  # 1% per trade (conservative)
        self.MAX_DAILY_RISK = 2.0  # 2% max daily loss
        self.RR_RATIO = 2.0  # Risk:Reward ratio
        self.MAX_ATR_PIPS = 30.0  # Skip if ATR too high
        self.MIN_ATR_PIPS = 5.0  # Skip if ATR too low

        # Trading hours (London + NY overlap)
        self.TRADE_HOURS = list(range(7, 18))  # 7-17 UTC

        # Position sizing
        self.INITIAL_BALANCE = 50000.0
        self.LOT_SIZE = 0.5  # Fixed lot for simplicity

        # Pip value
        self.PIP_VALUE = 10.0  # $10 per pip per lot for GBPUSD

        self.trades: List[Trade] = []
        self.daily_pnl = {}

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all BBMA indicators"""

        # Normalize column names to lowercase for consistency
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Reset index to make time a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'time'})

        # Bollinger Bands
        df['BB_MID'] = df['close'].rolling(window=self.BB_PERIOD).mean()
        df['BB_STD'] = df['close'].rolling(window=self.BB_PERIOD).std()
        df['BB_UPPER'] = df['BB_MID'] + (df['BB_STD'] * self.BB_STD)
        df['BB_LOWER'] = df['BB_MID'] - (df['BB_STD'] * self.BB_STD)

        # BB Width (volatility indicator)
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID'] * 100

        # %B indicator (position within BB)
        df['PCT_B'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])

        # EMAs
        df['EMA5'] = df['close'].ewm(span=self.EMA_FAST, adjust=False).mean()
        df['EMA10'] = df['close'].ewm(span=self.EMA_MID, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=self.EMA_SLOW, adjust=False).mean()

        # ATR for volatility filter
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_PIPS'] = df['ATR'] / PIP_SIZE

        # Trend direction (EMA alignment)
        df['TREND_UP'] = (df['EMA5'] > df['EMA10']) & (df['EMA10'] > df['EMA50'])
        df['TREND_DOWN'] = (df['EMA5'] < df['EMA10']) & (df['EMA10'] < df['EMA50'])

        # EMA Cross signals
        df['EMA_CROSS_UP'] = (df['EMA5'] > df['EMA10']) & (df['EMA5'].shift(1) <= df['EMA10'].shift(1))
        df['EMA_CROSS_DOWN'] = (df['EMA5'] < df['EMA10']) & (df['EMA5'].shift(1) >= df['EMA10'].shift(1))

        # BB extreme touches
        df['TOUCH_UPPER'] = df['high'] >= df['BB_UPPER']
        df['TOUCH_LOWER'] = df['low'] <= df['BB_LOWER']

        # Candle patterns
        df['BODY'] = abs(df['close'] - df['open'])
        df['UPPER_WICK'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['LOWER_WICK'] = df[['close', 'open']].min(axis=1) - df['low']
        df['RANGE'] = df['high'] - df['low']

        # Rejection candles (long wick + small body)
        df['REJECTION_UP'] = (df['UPPER_WICK'] > df['BODY'] * 2) & (df['TOUCH_UPPER'])
        df['REJECTION_DOWN'] = (df['LOWER_WICK'] > df['BODY'] * 2) & (df['TOUCH_LOWER'])

        return df

    def detect_reentry_signal(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """
        Detect BB Reentry signal

        Logic:
        1. Recent rejection from BB extreme (within last 5 bars)
        2. Price now pulling back to EMA zone
        3. Trend confirmation from EMA50
        """
        if idx < 20:
            return None

        row = df.iloc[idx]

        # Check ATR filter
        if row['ATR_PIPS'] > self.MAX_ATR_PIPS or row['ATR_PIPS'] < self.MIN_ATR_PIPS:
            return None

        # Check trading hours
        hour = row['time'].hour if hasattr(row['time'], 'hour') else pd.Timestamp(row['time']).hour
        if hour not in self.TRADE_HOURS:
            return None

        # Look for recent BB rejection (last 5 bars)
        lookback = df.iloc[max(0, idx-5):idx+1]

        # BUY: Recent rejection from lower BB, now at EMA zone
        if lookback['REJECTION_DOWN'].any():
            # Price should be near EMA5/10 zone
            ema_zone_low = min(row['EMA5'], row['EMA10'])
            ema_zone_high = max(row['EMA5'], row['EMA10'])

            if row['low'] <= ema_zone_high and row['close'] > ema_zone_low:
                # Confirm trend with EMA50
                if row['close'] > row['EMA50']:
                    # Calculate SL/TP
                    atr = row['ATR']
                    sl = row['close'] - (atr * 1.5)
                    tp = row['BB_UPPER']  # Target opposite BB

                    # Ensure RR is acceptable
                    risk = row['close'] - sl
                    reward = tp - row['close']
                    if reward / risk >= 1.5:
                        return {
                            'direction': 'BUY',
                            'signal_type': 'BB_REENTRY',
                            'entry': row['close'],
                            'sl': sl,
                            'tp': tp,
                            'rr': reward / risk
                        }

        # SELL: Recent rejection from upper BB, now at EMA zone
        if lookback['REJECTION_UP'].any():
            ema_zone_low = min(row['EMA5'], row['EMA10'])
            ema_zone_high = max(row['EMA5'], row['EMA10'])

            if row['high'] >= ema_zone_low and row['close'] < ema_zone_high:
                # Confirm trend with EMA50
                if row['close'] < row['EMA50']:
                    atr = row['ATR']
                    sl = row['close'] + (atr * 1.5)
                    tp = row['BB_LOWER']

                    risk = sl - row['close']
                    reward = row['close'] - tp
                    if reward / risk >= 1.5:
                        return {
                            'direction': 'SELL',
                            'signal_type': 'BB_REENTRY',
                            'entry': row['close'],
                            'sl': sl,
                            'tp': tp,
                            'rr': reward / risk
                        }

        return None

    def detect_extreme_rejection(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """
        Detect immediate BB extreme rejection

        Logic:
        1. Price touches outer BB
        2. Strong rejection candle (long wick)
        3. Against current momentum (mean reversion)
        """
        if idx < 20:
            return None

        row = df.iloc[idx]

        # Check ATR filter
        if row['ATR_PIPS'] > self.MAX_ATR_PIPS or row['ATR_PIPS'] < self.MIN_ATR_PIPS:
            return None

        hour = row['time'].hour if hasattr(row['time'], 'hour') else pd.Timestamp(row['time']).hour
        if hour not in self.TRADE_HOURS:
            return None

        atr = row['ATR']

        # BUY: Strong rejection from lower BB
        if row['REJECTION_DOWN'] and row['close'] > row['open']:
            # Extra confirmation: BB squeeze or expansion
            if row['BB_WIDTH'] > 0.5:  # Some volatility
                sl = row['low'] - (atr * 0.5)
                tp = row['BB_MID']  # Conservative: target mid BB

                risk = row['close'] - sl
                reward = tp - row['close']
                if reward / risk >= 1.2:
                    return {
                        'direction': 'BUY',
                        'signal_type': 'EXTREME_REJECTION',
                        'entry': row['close'],
                        'sl': sl,
                        'tp': tp,
                        'rr': reward / risk
                    }

        # SELL: Strong rejection from upper BB
        if row['REJECTION_UP'] and row['close'] < row['open']:
            if row['BB_WIDTH'] > 0.5:
                sl = row['high'] + (atr * 0.5)
                tp = row['BB_MID']

                risk = sl - row['close']
                reward = row['close'] - tp
                if reward / risk >= 1.2:
                    return {
                        'direction': 'SELL',
                        'signal_type': 'EXTREME_REJECTION',
                        'entry': row['close'],
                        'sl': sl,
                        'tp': tp,
                        'rr': reward / risk
                    }

        return None

    def detect_ma_cross(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """
        Detect EMA cross signal

        Logic:
        1. EMA5 crosses EMA10
        2. Price is on correct side of EMA50 (trend filter)
        3. BB not squeezed (need volatility for trend)
        """
        if idx < 20:
            return None

        row = df.iloc[idx]

        # Check ATR filter
        if row['ATR_PIPS'] > self.MAX_ATR_PIPS or row['ATR_PIPS'] < self.MIN_ATR_PIPS:
            return None

        hour = row['time'].hour if hasattr(row['time'], 'hour') else pd.Timestamp(row['time']).hour
        if hour not in self.TRADE_HOURS:
            return None

        atr = row['ATR']

        # BUY: EMA5 crosses above EMA10, above EMA50
        if row['EMA_CROSS_UP'] and row['close'] > row['EMA50']:
            if row['BB_WIDTH'] > 0.3:  # Not squeezed
                sl = row['EMA50'] - (atr * 0.5)
                tp = row['close'] + (atr * 3)  # 3x ATR target

                risk = row['close'] - sl
                reward = tp - row['close']
                if reward / risk >= 2.0:
                    return {
                        'direction': 'BUY',
                        'signal_type': 'MA_CROSS',
                        'entry': row['close'],
                        'sl': sl,
                        'tp': tp,
                        'rr': reward / risk
                    }

        # SELL: EMA5 crosses below EMA10, below EMA50
        if row['EMA_CROSS_DOWN'] and row['close'] < row['EMA50']:
            if row['BB_WIDTH'] > 0.3:
                sl = row['EMA50'] + (atr * 0.5)
                tp = row['close'] - (atr * 3)

                risk = sl - row['close']
                reward = row['close'] - tp
                if reward / risk >= 2.0:
                    return {
                        'direction': 'SELL',
                        'signal_type': 'MA_CROSS',
                        'entry': row['close'],
                        'sl': sl,
                        'tp': tp,
                        'rr': reward / risk
                    }

        return None

    def check_daily_risk(self, date: datetime) -> bool:
        """Check if daily risk limit exceeded"""
        date_key = date.date() if hasattr(date, 'date') else date
        daily_loss = self.daily_pnl.get(date_key, 0)
        max_daily_loss = self.INITIAL_BALANCE * (self.MAX_DAILY_RISK / 100)
        return daily_loss > -max_daily_loss

    def run_backtest(self, df: pd.DataFrame) -> dict:
        """Run the BBMA backtest"""

        print("\n" + "="*60)
        print("BBMA STRATEGY BACKTEST")
        print("Bollinger Bands + Moving Average")
        print("="*60)

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Remove NaN rows
        df = df.dropna().reset_index(drop=True)

        print(f"\nData period: {df['time'].min()} to {df['time'].max()}")
        print(f"Total bars: {len(df)}")

        balance = self.INITIAL_BALANCE
        peak_balance = balance
        max_drawdown = 0

        open_trade: Optional[Trade] = None

        for idx in range(50, len(df)):  # Start after warmup
            row = df.iloc[idx]
            current_time = row['time']

            # Check daily risk
            if not self.check_daily_risk(current_time):
                continue

            # If we have an open trade, check exit conditions
            if open_trade:
                # Check SL hit
                if open_trade.direction == 'BUY':
                    if row['low'] <= open_trade.sl:
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.sl
                        open_trade.exit_reason = 'SL'
                        open_trade.pnl_pips = (open_trade.sl - open_trade.entry_price) / PIP_SIZE
                    elif row['high'] >= open_trade.tp:
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.tp
                        open_trade.exit_reason = 'TP'
                        open_trade.pnl_pips = (open_trade.tp - open_trade.entry_price) / PIP_SIZE
                else:  # SELL
                    if row['high'] >= open_trade.sl:
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.sl
                        open_trade.exit_reason = 'SL'
                        open_trade.pnl_pips = (open_trade.entry_price - open_trade.sl) / PIP_SIZE
                    elif row['low'] <= open_trade.tp:
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.tp
                        open_trade.exit_reason = 'TP'
                        open_trade.pnl_pips = (open_trade.entry_price - open_trade.tp) / PIP_SIZE

                # If trade closed
                if open_trade.exit_time:
                    open_trade.pnl = open_trade.pnl_pips * self.LOT_SIZE * self.PIP_VALUE
                    balance += open_trade.pnl

                    # Track daily P&L
                    date_key = current_time.date() if hasattr(current_time, 'date') else current_time
                    self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0) + open_trade.pnl

                    # Track drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance * 100
                    if dd > max_drawdown:
                        max_drawdown = dd

                    self.trades.append(open_trade)
                    open_trade = None

            # If no open trade, look for signals
            if not open_trade:
                # Priority: Reentry > Extreme Rejection > MA Cross
                signal = self.detect_reentry_signal(df, idx)
                if not signal:
                    signal = self.detect_extreme_rejection(df, idx)
                if not signal:
                    signal = self.detect_ma_cross(df, idx)

                if signal:
                    open_trade = Trade(
                        entry_time=current_time,
                        entry_price=signal['entry'],
                        direction=signal['direction'],
                        sl=signal['sl'],
                        tp=signal['tp'],
                        signal_type=signal['signal_type']
                    )

        # Close any remaining trade at last bar
        if open_trade:
            row = df.iloc[-1]
            open_trade.exit_time = row['time']
            open_trade.exit_price = row['close']
            open_trade.exit_reason = 'END'
            if open_trade.direction == 'BUY':
                open_trade.pnl_pips = (row['close'] - open_trade.entry_price) / PIP_SIZE
            else:
                open_trade.pnl_pips = (open_trade.entry_price - row['close']) / PIP_SIZE
            open_trade.pnl = open_trade.pnl_pips * self.LOT_SIZE * self.PIP_VALUE
            balance += open_trade.pnl
            self.trades.append(open_trade)

        # Calculate statistics
        return self.calculate_statistics(balance, max_drawdown)

    def calculate_statistics(self, final_balance: float, max_drawdown: float) -> dict:
        """Calculate and display backtest statistics"""

        if not self.trades:
            print("\nNo trades generated!")
            return {}

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl and t.pnl > 0]
        losses = [t for t in self.trades if t.pnl and t.pnl < 0]

        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        net_profit = final_balance - self.INITIAL_BALANCE
        roi = net_profit / self.INITIAL_BALANCE * 100

        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0

        # Monthly breakdown
        monthly_pnl = {}
        for trade in self.trades:
            if trade.exit_time and trade.pnl:
                month_key = trade.exit_time.strftime('%Y-%m')
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl

        losing_months = sum(1 for pnl in monthly_pnl.values() if pnl < 0)

        # Signal type breakdown
        signal_stats = {}
        for trade in self.trades:
            if trade.signal_type not in signal_stats:
                signal_stats[trade.signal_type] = {'count': 0, 'wins': 0, 'pnl': 0}
            signal_stats[trade.signal_type]['count'] += 1
            if trade.pnl and trade.pnl > 0:
                signal_stats[trade.signal_type]['wins'] += 1
            signal_stats[trade.signal_type]['pnl'] += trade.pnl if trade.pnl else 0

        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        print(f"\n{'PERFORMANCE METRICS':^60}")
        print("-"*60)
        print(f"Initial Balance:     ${self.INITIAL_BALANCE:,.2f}")
        print(f"Final Balance:       ${final_balance:,.2f}")
        print(f"Net Profit:          ${net_profit:,.2f} ({roi:+.1f}%)")
        print(f"Max Drawdown:        {max_drawdown:.1f}%")

        print(f"\n{'TRADE STATISTICS':^60}")
        print("-"*60)
        print(f"Total Trades:        {total_trades}")
        print(f"Wins:                {len(wins)} ({win_rate:.1f}%)")
        print(f"Losses:              {len(losses)}")
        print(f"Profit Factor:       {profit_factor:.2f}")
        print(f"Avg Win:             ${avg_win:,.2f}")
        print(f"Avg Loss:            ${avg_loss:,.2f}")

        print(f"\n{'SIGNAL BREAKDOWN':^60}")
        print("-"*60)
        for sig_type, stats in sorted(signal_stats.items()):
            wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
            print(f"{sig_type:20} | {stats['count']:3} trades | {wr:5.1f}% WR | ${stats['pnl']:+,.2f}")

        print(f"\n{'MONTHLY BREAKDOWN':^60}")
        print("-"*60)
        for month, pnl in sorted(monthly_pnl.items()):
            status = "+" if pnl >= 0 else "-"
            print(f"{month}: ${pnl:+,.2f} {status}")

        print(f"\nLosing Months: {losing_months}/{len(monthly_pnl)}")

        print("\n" + "="*60)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'losing_months': losing_months,
            'signal_stats': signal_stats,
            'monthly_pnl': monthly_pnl
        }

    def save_trades(self, filepath: str):
        """Save trades to CSV"""
        if not self.trades:
            return

        data = []
        for t in self.trades:
            data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'signal_type': t.signal_type,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'sl': t.sl,
                'tp': t.tp,
                'pnl_pips': t.pnl_pips,
                'pnl': t.pnl,
                'exit_reason': t.exit_reason
            })

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\nTrades saved to: {filepath}")


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        return pd.DataFrame()
    df = await db.get_ohlcv(symbol, timeframe, 100000, start, end)
    await db.disconnect()
    return df


async def main():
    """Main entry point"""

    print("Fetching GBPUSD H1 data...")

    # Date range: Jan 2025 - Jan 2026 (same as QuadLayer)
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 1, 31)

    df = await fetch_data('GBPUSD', 'H1', start_date, end_date)

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} bars")

    # Run backtest
    backtest = BBMABacktest()
    results = backtest.run_backtest(df)

    # Save trades
    backtest.save_trades('strategies/bbma/reports/bbma_trades.csv')

    # Print comparison header
    print("\n" + "="*60)
    print("COMPARISON: BBMA vs QUADLAYER")
    print("="*60)
    print("""
+------------------------------------------------------------+
|                    BBMA STRATEGY                           |
|  - Bollinger Bands + EMA (5/10/50)                        |
|  - Mean reversion at BB extremes                          |
|  - Trend following with EMA cross                         |
+------------------------------------------------------------+
|                    QUADLAYER v6.9                          |
|  - 4-Layer Quality Filter                                 |
|  - Monthly Profile + Technical + Risk + Pattern           |
|  - Order Block + EMA Pullback entries                     |
|  - Result: 150 trades, 50% WR, PF 5.84, +$36,205         |
|  - ZERO losing months                                     |
+------------------------------------------------------------+
    """)


if __name__ == '__main__':
    asyncio.run(main())
