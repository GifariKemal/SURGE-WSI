"""
BBMA + QUADLAYER HYBRID Strategy Backtest
==========================================
Combining BBMA entry signals with QuadLayer quality filters

Hypothesis:
- BBMA provides good mean-reversion entries
- QuadLayer filters prevent entries during poor market conditions
- Combination = Better win rate and consistency

Strategy:
- Entry: BBMA signals (BB rejection, reentry, MA cross)
- Filter: QuadLayer 4-Layer quality system
  * Layer 1: Monthly Profile (tradeable %)
  * Layer 2: Technical (ATR, Efficiency, ADX)
  * Layer 3: Intra-Month Risk (consecutive losses)
  * Layer 4: Pattern Filter (rolling WR)

Timeframe: H1
Pair: GBPUSD
"""

import sys
import io
from pathlib import Path

# Strategy directory (where this file is located)
STRATEGY_DIR = Path(__file__).parent
# Project root is 2 levels up
PROJECT_ROOT = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGY_DIR.parent))

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


PIP_SIZE = 0.0001

# ============================================
# QUADLAYER CONFIGURATION (from v6.9)
# ============================================

# Monthly profile data (tradeable percentage)
MONTHLY_PROFILE = {
    1: 67,   # January
    2: 55,   # February - POOR
    3: 70,   # March
    4: 80,   # April - EXCELLENT
    5: 68,   # May
    6: 75,   # June
    7: 72,   # July
    8: 60,   # August - Summer lull
    9: 78,   # September - Good
    10: 73,  # October
    11: 65,  # November
    12: 50,  # December - Holiday - POOR
}

# Day multipliers (Thursday best, Friday worst)
DAY_MULTIPLIERS = {
    0: 1.0,   # Monday
    1: 0.9,   # Tuesday
    2: 1.0,   # Wednesday
    3: 0.8,   # Thursday (best)
    4: 0.3,   # Friday (worst)
    5: 0.0,   # Saturday - skip
    6: 0.0    # Sunday - skip
}

# Hour multipliers (Kill zones)
HOUR_MULTIPLIERS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,  # Asian - skip
    6: 0.5,                                             # Pre-London
    7: 0.0,                                             # Skip (bad hour)
    8: 1.0, 9: 1.0, 10: 0.9,                           # London open
    11: 0.0,                                            # Skip (bad hour)
    12: 0.7, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,  # NY overlap
    18: 0.3, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0   # Late - skip
}


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: str
    sl: float
    tp: float
    signal_type: str
    quality_score: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    exit_reason: Optional[str] = None


class BBMAQuadLayerHybrid:
    """
    BBMA + QuadLayer Hybrid Strategy

    BBMA Signals:
    1. BB_REENTRY: Pullback to EMA after BB rejection
    2. EXTREME_REJECTION: Strong reversal at BB extreme
    3. MA_CROSS: EMA cross with trend confirmation

    QuadLayer Filters:
    1. Monthly Profile (skip poor months)
    2. Technical Quality (ATR, Efficiency, ADX)
    3. Intra-Month Risk (consecutive losses)
    4. Pattern Filter (rolling WR)
    """

    def __init__(self):
        # BBMA settings
        self.BB_PERIOD = 20
        self.BB_STD = 2.0
        self.EMA_FAST = 5
        self.EMA_MID = 10
        self.EMA_SLOW = 50

        # QuadLayer settings
        self.MIN_QUALITY = 55  # Minimum quality score to trade
        self.MAX_ATR_PIPS = 25.0  # Synced with QuadLayer v6.9
        self.MIN_ATR_PIPS = 5.0

        # Risk Management (QuadLayer style)
        self.INITIAL_BALANCE = 50000.0
        self.LOT_SIZE = 0.5
        self.PIP_VALUE = 10.0
        self.MAX_LOSS_PCT = 0.15  # SL_CAPPED at 0.15%

        # Session filters
        self.LONDON_HOURS = list(range(8, 12))
        self.NY_HOURS = list(range(13, 18))

        self.trades: List[Trade] = []

        # Simple internal risk/pattern tracking (instead of importing classes)
        self.consecutive_losses = 0
        self.monthly_pnl_tracker = {}
        self.recent_trades = []  # For pattern filter

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate BBMA + QuadLayer indicators"""

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()
        df = df.rename(columns={'index': 'time'})

        # BBMA indicators
        df['BB_MID'] = df['close'].rolling(window=self.BB_PERIOD).mean()
        df['BB_STD'] = df['close'].rolling(window=self.BB_PERIOD).std()
        df['BB_UPPER'] = df['BB_MID'] + (df['BB_STD'] * self.BB_STD)
        df['BB_LOWER'] = df['BB_MID'] - (df['BB_STD'] * self.BB_STD)
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID'] * 100

        df['EMA5'] = df['close'].ewm(span=self.EMA_FAST, adjust=False).mean()
        df['EMA10'] = df['close'].ewm(span=self.EMA_MID, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=self.EMA_SLOW, adjust=False).mean()

        # ATR
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_PIPS'] = df['ATR'] / PIP_SIZE

        # QuadLayer: ATR Stability
        df['ATR_MEAN'] = df['ATR'].rolling(window=20).mean()
        df['ATR_STD'] = df['ATR'].rolling(window=20).std()
        df['ATR_STABILITY'] = 1 - (df['ATR_STD'] / df['ATR_MEAN']).clip(0, 1)

        # QuadLayer: Efficiency Ratio
        change = abs(df['close'] - df['close'].shift(10))
        volatility = df['TR'].rolling(window=10).sum()
        df['EFFICIENCY'] = (change / volatility).clip(0, 1)

        # ADX for trend strength
        df['ADX'] = self._calculate_adx(df)

        # EMA Cross signals
        df['EMA_CROSS_UP'] = (df['EMA5'] > df['EMA10']) & (df['EMA5'].shift(1) <= df['EMA10'].shift(1))
        df['EMA_CROSS_DOWN'] = (df['EMA5'] < df['EMA10']) & (df['EMA5'].shift(1) >= df['EMA10'].shift(1))

        # BB touches
        df['TOUCH_UPPER'] = df['high'] >= df['BB_UPPER']
        df['TOUCH_LOWER'] = df['low'] <= df['BB_LOWER']

        # Candle patterns
        df['BODY'] = abs(df['close'] - df['open'])
        df['UPPER_WICK'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['LOWER_WICK'] = df[['close', 'open']].min(axis=1) - df['low']

        # Rejection candles
        df['REJECTION_UP'] = (df['UPPER_WICK'] > df['BODY'] * 2) & (df['TOUCH_UPPER'])
        df['REJECTION_DOWN'] = (df['LOWER_WICK'] > df['BODY'] * 2) & (df['TOUCH_LOWER'])

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        tr = np.maximum(
            high - low,
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
        )

        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx

    def calculate_quality(self, df: pd.DataFrame, idx: int, current_time: datetime) -> float:
        """
        Calculate QuadLayer quality score

        Layer 1: Monthly Profile (0-30 points)
        Layer 2: Technical (0-40 points)
        Layer 3: Day/Hour (0-20 points)
        Layer 4: Pattern (0-10 points) - via filter
        """
        row = df.iloc[idx]

        # Layer 1: Monthly Profile
        month = current_time.month
        tradeable_pct = MONTHLY_PROFILE.get(month, 70)
        if tradeable_pct >= 75:
            monthly_score = 30
        elif tradeable_pct >= 65:
            monthly_score = 20
        elif tradeable_pct >= 55:
            monthly_score = 10
        else:
            monthly_score = 0  # Skip poor months

        # Layer 2: Technical
        atr_stability = row.get('ATR_STABILITY', 0.5)
        efficiency = row.get('EFFICIENCY', 0.5)
        adx = row.get('ADX', 25)

        tech_score = 0
        tech_score += min(15, atr_stability * 20)  # Max 15
        tech_score += min(15, efficiency * 20)      # Max 15
        if adx and 20 <= adx <= 40:
            tech_score += 10
        elif adx and 15 <= adx < 20:
            tech_score += 5

        # Layer 3: Day/Hour
        day = current_time.weekday()
        hour = current_time.hour

        day_mult = DAY_MULTIPLIERS.get(day, 0)
        hour_mult = HOUR_MULTIPLIERS.get(hour, 0)

        if day_mult == 0 or hour_mult == 0:
            return 0  # Skip this bar

        timing_score = 20 * day_mult * hour_mult

        # Total score
        total_score = monthly_score + tech_score + timing_score

        return total_score

    def detect_bbma_signal(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """Detect BBMA entry signals"""
        if idx < 20:
            return None

        row = df.iloc[idx]

        # ATR filter
        if row['ATR_PIPS'] > self.MAX_ATR_PIPS or row['ATR_PIPS'] < self.MIN_ATR_PIPS:
            return None

        atr = row['ATR']
        lookback = df.iloc[max(0, idx-5):idx+1]

        # Signal 1: BB Reentry (best signal)
        if lookback['REJECTION_DOWN'].any():
            ema_zone = max(row['EMA5'], row['EMA10'])
            if row['low'] <= ema_zone and row['close'] > row['EMA50']:
                sl = row['close'] - (atr * 1.5)
                tp = row['BB_UPPER']
                risk = row['close'] - sl
                reward = tp - row['close']
                if reward / risk >= 1.5:
                    return {'direction': 'BUY', 'signal_type': 'BB_REENTRY',
                            'entry': row['close'], 'sl': sl, 'tp': tp, 'rr': reward/risk}

        if lookback['REJECTION_UP'].any():
            ema_zone = min(row['EMA5'], row['EMA10'])
            if row['high'] >= ema_zone and row['close'] < row['EMA50']:
                sl = row['close'] + (atr * 1.5)
                tp = row['BB_LOWER']
                risk = sl - row['close']
                reward = row['close'] - tp
                if reward / risk >= 1.5:
                    return {'direction': 'SELL', 'signal_type': 'BB_REENTRY',
                            'entry': row['close'], 'sl': sl, 'tp': tp, 'rr': reward/risk}

        # Signal 2: Extreme Rejection
        if row['REJECTION_DOWN'] and row['close'] > row['open'] and row['BB_WIDTH'] > 0.5:
            sl = row['low'] - (atr * 0.5)
            tp = row['BB_MID']
            risk = row['close'] - sl
            reward = tp - row['close']
            if reward / risk >= 1.2:
                return {'direction': 'BUY', 'signal_type': 'EXTREME_REJECTION',
                        'entry': row['close'], 'sl': sl, 'tp': tp, 'rr': reward/risk}

        if row['REJECTION_UP'] and row['close'] < row['open'] and row['BB_WIDTH'] > 0.5:
            sl = row['high'] + (atr * 0.5)
            tp = row['BB_MID']
            risk = sl - row['close']
            reward = row['close'] - tp
            if reward / risk >= 1.2:
                return {'direction': 'SELL', 'signal_type': 'EXTREME_REJECTION',
                        'entry': row['close'], 'sl': sl, 'tp': tp, 'rr': reward/risk}

        # Signal 3: MA Cross (trend)
        if row['EMA_CROSS_UP'] and row['close'] > row['EMA50'] and row['BB_WIDTH'] > 0.3:
            sl = row['EMA50'] - (atr * 0.5)
            tp = row['close'] + (atr * 3)
            risk = row['close'] - sl
            reward = tp - row['close']
            if reward / risk >= 2.0:
                return {'direction': 'BUY', 'signal_type': 'MA_CROSS',
                        'entry': row['close'], 'sl': sl, 'tp': tp, 'rr': reward/risk}

        if row['EMA_CROSS_DOWN'] and row['close'] < row['EMA50'] and row['BB_WIDTH'] > 0.3:
            sl = row['EMA50'] + (atr * 0.5)
            tp = row['close'] - (atr * 3)
            risk = sl - row['close']
            reward = row['close'] - tp
            if reward / risk >= 2.0:
                return {'direction': 'SELL', 'signal_type': 'MA_CROSS',
                        'entry': row['close'], 'sl': sl, 'tp': tp, 'rr': reward/risk}

        return None

    def run_backtest(self, df: pd.DataFrame) -> dict:
        """Run the hybrid backtest"""

        print("\n" + "="*60)
        print("BBMA + QUADLAYER HYBRID BACKTEST")
        print("="*60)

        df = self.calculate_indicators(df)
        df = df.dropna().reset_index(drop=True)

        print(f"\nData period: {df['time'].min()} to {df['time'].max()}")
        print(f"Total bars: {len(df)}")

        balance = self.INITIAL_BALANCE
        peak_balance = balance
        max_drawdown = 0
        open_trade: Optional[Trade] = None
        monthly_pnl = {}
        signals_rejected = {'quality': 0, 'risk_manager': 0, 'pattern': 0}

        for idx in range(50, len(df)):
            row = df.iloc[idx]
            current_time = row['time']
            current_month = current_time.strftime('%Y-%m')

            # Initialize month tracking
            if current_month not in monthly_pnl:
                monthly_pnl[current_month] = 0
                self.consecutive_losses = 0  # Reset consecutive losses at new month

            # Check open trade
            if open_trade:
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
                else:
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

                if open_trade.exit_time:
                    open_trade.pnl = open_trade.pnl_pips * self.LOT_SIZE * self.PIP_VALUE
                    balance += open_trade.pnl
                    monthly_pnl[current_month] += open_trade.pnl

                    # Update internal tracking
                    is_win = open_trade.pnl > 0
                    if is_win:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1

                    # Track for pattern filter
                    self.recent_trades.append((open_trade.direction, is_win))
                    if len(self.recent_trades) > 10:
                        self.recent_trades.pop(0)

                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance * 100
                    if dd > max_drawdown:
                        max_drawdown = dd

                    self.trades.append(open_trade)
                    open_trade = None

            # Look for new signals
            if not open_trade:
                signal = self.detect_bbma_signal(df, idx)

                if signal:
                    # Layer 1-3: Quality check
                    quality = self.calculate_quality(df, idx, current_time)
                    if quality < self.MIN_QUALITY:
                        signals_rejected['quality'] += 1
                        continue

                    # Layer 3: Risk manager check (halt after 3 consecutive losses)
                    if self.consecutive_losses >= 3:
                        signals_rejected['risk_manager'] += 1
                        continue

                    # Layer 4: Pattern filter (check rolling win rate)
                    if len(self.recent_trades) >= 5:
                        wins = sum(1 for _, w in self.recent_trades if w)
                        wr = wins / len(self.recent_trades)
                        if wr < 0.2:  # Less than 20% WR in last 10 trades
                            signals_rejected['pattern'] += 1
                            continue

                    # Apply SL_CAPPED (max 0.15% loss)
                    max_sl_usd = balance * (self.MAX_LOSS_PCT / 100)
                    risk_pips = abs(signal['entry'] - signal['sl']) / PIP_SIZE
                    max_loss = risk_pips * self.LOT_SIZE * self.PIP_VALUE

                    if max_loss > max_sl_usd:
                        # Adjust SL to cap loss
                        capped_pips = max_sl_usd / (self.LOT_SIZE * self.PIP_VALUE)
                        if signal['direction'] == 'BUY':
                            signal['sl'] = signal['entry'] - (capped_pips * PIP_SIZE)
                        else:
                            signal['sl'] = signal['entry'] + (capped_pips * PIP_SIZE)

                    open_trade = Trade(
                        entry_time=current_time,
                        entry_price=signal['entry'],
                        direction=signal['direction'],
                        sl=signal['sl'],
                        tp=signal['tp'],
                        signal_type=signal['signal_type'],
                        quality_score=quality
                    )

        # Close remaining trade
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

        return self.calculate_statistics(balance, max_drawdown, monthly_pnl, signals_rejected)

    def calculate_statistics(self, final_balance: float, max_drawdown: float,
                             monthly_pnl: dict, signals_rejected: dict) -> dict:
        """Calculate and display statistics"""

        if not self.trades:
            print("\nNo trades generated!")
            return {}

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl and t.pnl > 0]
        losses = [t for t in self.trades if t.pnl and t.pnl < 0]

        win_rate = len(wins) / total_trades * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        net_profit = final_balance - self.INITIAL_BALANCE
        roi = net_profit / self.INITIAL_BALANCE * 100

        losing_months = sum(1 for pnl in monthly_pnl.values() if pnl < 0)

        # Signal breakdown
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

        print(f"\n{'QUADLAYER FILTER IMPACT':^60}")
        print("-"*60)
        print(f"Rejected by Quality:     {signals_rejected['quality']}")
        print(f"Rejected by Risk Mgr:    {signals_rejected['risk_manager']}")
        print(f"Rejected by Pattern:     {signals_rejected['pattern']}")
        total_rejected = sum(signals_rejected.values())
        print(f"Total Signals Filtered:  {total_rejected}")

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
            'losing_months': losing_months
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
                'quality_score': t.quality_score,
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

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 1, 31)

    df = await fetch_data('GBPUSD', 'H1', start_date, end_date)

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} bars")

    # Run backtest
    backtest = BBMAQuadLayerHybrid()
    results = backtest.run_backtest(df)

    # Save trades
    backtest.save_trades('strategies/bbma/reports/bbma_quadlayer_hybrid_trades.csv')

    # Comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print("""
+------------------------------------------------------------+
|                    PURE BBMA                               |
|  - 88 trades, 17.0% WR, PF 0.46                           |
|  - Net: -$3,854 (-7.7%)                                   |
|  - Losing Months: 10/13                                   |
+------------------------------------------------------------+
|                    BBMA + QUADLAYER HYBRID                 |
|  - Filtered by 4-Layer Quality System                     |
|  - Should have better WR and fewer losing months          |
+------------------------------------------------------------+
|                    PURE QUADLAYER v6.9                     |
|  - 150 trades, 50% WR, PF 5.84                           |
|  - Net: +$36,205 (+72.4%)                                 |
|  - ZERO losing months                                     |
+------------------------------------------------------------+
    """)


if __name__ == '__main__':
    asyncio.run(main())
