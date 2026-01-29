"""Backtest XAUUSD 13 Months (Jan 2025 - Jan 2026)
=================================================

Run backtest for XAUUSD (Gold) and send report to Telegram.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger
from dataclasses import dataclass
from typing import List, Dict, Optional

from config import config
from src.data.db_handler import DBHandler

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


# ============================================================
# XAUUSD SPECIFIC SETTINGS
# ============================================================
SYMBOL = "XAUUSD"
PIP_SIZE = 0.1  # 1 pip = $0.10 for gold (some brokers use 0.01)
PIP_VALUE_PER_LOT = 1.0  # $1 per pip per 0.01 lot (standard)
SPREAD_PIPS = 25  # Typical spread for gold (2.5 price points = 25 pips)

# Backtest period
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 31, tzinfo=timezone.utc)


async def fetch_mt5_data(symbol: str, timeframe_str: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetch historical data from MT5"""
    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 not installed")
        return None

    if not mt5.initialize():
        logger.error(f"MT5 initialize failed: {mt5.last_error()}")
        return None

    # Map timeframe
    tf_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
    }
    tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_H4)

    logger.info(f"Fetching {symbol} {timeframe_str} from MT5: {start} to {end}")

    rates = mt5.copy_rates_range(symbol, tf, start, end)

    if rates is None or len(rates) == 0:
        logger.error(f"No data returned for {symbol} {timeframe_str}")
        mt5.shutdown()
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'tick_volume': 'volume'
    }, inplace=True)

    logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe_str}")
    mt5.shutdown()

    return df[['open', 'high', 'low', 'close', 'volume']]


async def save_to_db(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save data to database"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    count = await db.save_ohlcv(symbol, timeframe, df)
    await db.disconnect()
    logger.info(f"Saved {count} bars to database for {symbol} {timeframe}")
    return count


async def load_from_db(symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Load data from database"""
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=500000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


# ============================================================
# SIMPLE BACKTESTER FOR GOLD
# ============================================================
class GoldBacktester:
    """Simple backtester for XAUUSD"""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,  # 1%
        pip_value: float = 1.0,  # Per 0.01 lot
        spread_pips: float = 25,
        max_sl_pips: float = 500,  # 50 price points
        use_intelligent_filter: bool = True,
        intelligent_threshold: float = 60
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.max_sl_pips = max_sl_pips
        self.use_intelligent_filter = use_intelligent_filter
        self.intelligent_threshold = intelligent_threshold

        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]

        # Import analysis modules
        from src.analysis.kalman_filter import KalmanNoiseReducer
        from src.analysis.regime_detector import HMMRegimeDetector
        from src.analysis.poi_detector import POIDetector
        from src.utils.intelligent_activity_filter import IntelligentActivityFilter

        self.kalman = KalmanNoiseReducer()
        self.regime = HMMRegimeDetector()
        self.poi = POIDetector()
        self.intel_filter = IntelligentActivityFilter(activity_threshold=intelligent_threshold)

    def run(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Run backtest"""
        if htf_df is None or ltf_df is None or htf_df.empty or ltf_df.empty:
            return self._empty_result()

        # Reset state
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.kalman.reset()
        self.regime.reset()
        self.poi.reset()

        # Warmup
        warmup_bars = min(200, len(htf_df) // 3)
        for i in range(warmup_bars):
            price = htf_df.iloc[i]['close']
            state = self.kalman.update(price)
            if state:
                self.regime.update(state.smoothed_price)

        # Main loop
        position = None

        for i in range(warmup_bars, len(htf_df)):
            row = htf_df.iloc[i]
            price = row['close']
            high = row['high']
            low = row['low']
            bar_time = htf_df.index[i]

            # Update analysis
            state = self.kalman.update(price)
            if not state:
                continue

            smoothed = state.smoothed_price
            regime_info = self.regime.update(smoothed)

            # Update POI with recent HTF data
            if i > 50:
                poi_data = htf_df.iloc[i-50:i+1].copy()
                self.poi.detect(poi_data)

            # Check existing position
            if position:
                # Check SL/TP
                if position['direction'] == 'BUY':
                    if low <= position['sl']:
                        # Stop loss hit
                        pnl = (position['sl'] - position['entry'] - self.spread_pips * PIP_SIZE) / PIP_SIZE * position['lot'] * self.pip_value
                        self._close_trade(position, position['sl'], pnl, bar_time, 'SL')
                        position = None
                    elif high >= position['tp']:
                        # Take profit hit
                        pnl = (position['tp'] - position['entry'] - self.spread_pips * PIP_SIZE) / PIP_SIZE * position['lot'] * self.pip_value
                        self._close_trade(position, position['tp'], pnl, bar_time, 'TP')
                        position = None
                else:  # SELL
                    if high >= position['sl']:
                        # Stop loss hit
                        pnl = (position['entry'] - position['sl'] - self.spread_pips * PIP_SIZE) / PIP_SIZE * position['lot'] * self.pip_value
                        self._close_trade(position, position['sl'], pnl, bar_time, 'SL')
                        position = None
                    elif low <= position['tp']:
                        # Take profit hit
                        pnl = (position['entry'] - position['tp'] - self.spread_pips * PIP_SIZE) / PIP_SIZE * position['lot'] * self.pip_value
                        self._close_trade(position, position['tp'], pnl, bar_time, 'TP')
                        position = None
                continue

            # Check for new entry
            if not regime_info or not regime_info.is_tradeable:
                continue

            direction = regime_info.bias
            if direction == 'NONE':
                continue

            # Check intelligent filter
            if self.use_intelligent_filter:
                # Get Kalman velocity in pips
                velocity_pips = abs(state.velocity) / PIP_SIZE if state else 0

                # Call intel filter check method
                intel_result = self.intel_filter.check(
                    dt=bar_time,
                    current_high=high,
                    current_low=low,
                    current_close=price,
                    kalman_velocity=state.velocity if state else None
                )
                if not intel_result.should_trade:
                    continue

            # Check POI
            poi_result = self.poi.last_result
            if not poi_result:
                continue

            at_poi, poi_info = poi_result.price_at_poi(price, direction, tolerance_pips=30)
            if not at_poi:
                continue

            # Calculate position size
            sl_pips = min(self.max_sl_pips, 300)  # 30 price points default
            risk_amount = self.balance * self.risk_per_trade
            lot_size = risk_amount / (sl_pips * self.pip_value)
            lot_size = max(0.01, min(lot_size, 1.0))  # Clamp

            # Entry
            if direction == 'BUY':
                entry = price + self.spread_pips * PIP_SIZE
                sl = entry - sl_pips * PIP_SIZE
                tp = entry + sl_pips * 2 * PIP_SIZE  # 2:1 RR
            else:
                entry = price
                sl = entry + sl_pips * PIP_SIZE
                tp = entry - sl_pips * 2 * PIP_SIZE

            position = {
                'direction': direction,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'lot': lot_size,
                'time': bar_time
            }

        # Close any remaining position
        if position:
            final_price = htf_df.iloc[-1]['close']
            if position['direction'] == 'BUY':
                pnl = (final_price - position['entry']) / PIP_SIZE * position['lot'] * self.pip_value
            else:
                pnl = (position['entry'] - final_price) / PIP_SIZE * position['lot'] * self.pip_value
            self._close_trade(position, final_price, pnl, htf_df.index[-1], 'EOD')

        return self._calculate_results()

    def _close_trade(self, position: Dict, exit_price: float, pnl: float, time, reason: str):
        """Record closed trade"""
        self.balance += pnl
        self.equity_curve.append(self.balance)

        self.trades.append({
            'direction': position['direction'],
            'entry': position['entry'],
            'exit': exit_price,
            'sl': position['sl'],
            'tp': position['tp'],
            'lot': position['lot'],
            'pnl': pnl,
            'reason': reason,
            'entry_time': position['time'],
            'exit_time': time
        })

    def _calculate_results(self) -> Dict:
        """Calculate backtest results"""
        if not self.trades:
            return self._empty_result()

        total_pnl = sum(t['pnl'] for t in self.trades)
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

        # Drawdown
        peak = self.initial_balance
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'pnl_pct': total_pnl / self.initial_balance * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0,
            'max_drawdown': max_dd,
            'final_balance': self.balance,
            'trades': self.trades
        }

    def _empty_result(self) -> Dict:
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pnl': 0, 'pnl_pct': 0, 'avg_win': 0, 'avg_loss': 0,
            'profit_factor': 0, 'max_drawdown': 0, 'final_balance': self.initial_balance,
            'trades': []
        }


def run_monthly_backtest(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, year: int, month: int,
                         initial_balance: float = 10000.0) -> Dict:
    """Run backtest for a single month"""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)

    # Add warmup
    warmup_start = start - timedelta(days=30)

    # Filter data
    htf_m = htf_df[(htf_df.index >= warmup_start) & (htf_df.index <= end)]
    ltf_m = ltf_df[(ltf_df.index >= warmup_start) & (ltf_df.index <= end)]

    if htf_m.empty or ltf_m.empty:
        return {'month': month, 'year': year, 'skipped': True, 'pnl': 0, 'trades': 0}

    bt = GoldBacktester(
        initial_balance=initial_balance,
        risk_per_trade=0.01,
        pip_value=PIP_VALUE_PER_LOT,
        spread_pips=SPREAD_PIPS,
        use_intelligent_filter=True,
        intelligent_threshold=60
    )

    result = bt.run(htf_m, ltf_m)
    result['month'] = month
    result['year'] = year
    result['skipped'] = False

    return result


async def send_telegram_report(results: List[Dict], htf_count: int, ltf_count: int):
    """Send detailed report to Telegram"""
    if not TELEGRAM_AVAILABLE:
        logger.warning("Telegram not available")
        return

    bot_token = config.telegram.bot_token
    chat_id = config.telegram.chat_id

    if not bot_token or not chat_id:
        logger.warning("Telegram not configured")
        return

    bot = Bot(token=bot_token)

    # Calculate totals
    total_trades = sum(r.get('total_trades', 0) for r in results if not r.get('skipped'))
    total_wins = sum(r.get('wins', 0) for r in results if not r.get('skipped'))
    total_pnl = sum(r.get('total_pnl', 0) for r in results if not r.get('skipped'))

    # Monthly breakdown
    monthly_pnl = []
    losing_months = 0
    for r in results:
        if not r.get('skipped'):
            pnl = r.get('total_pnl', 0)
            monthly_pnl.append(pnl)
            if pnl < 0:
                losing_months += 1

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_monthly = np.mean(monthly_pnl) if monthly_pnl else 0

    # Build message
    msg = f"""<b>📊 XAUUSD BACKTEST REPORT</b>
<b>Period: Jan 2025 - Jan 2026 (13 months)</b>

━━━━━━━━━━━━━━━━━━━━━━━━
<b>📈 DATA INFO</b>
━━━━━━━━━━━━━━━━━━━━━━━━
  Symbol: XAUUSD (Gold)
  HTF Bars: {htf_count:,}
  LTF Bars: {ltf_count:,}
  Config: INTEL_60

━━━━━━━━━━━━━━━━━━━━━━━━
<b>💰 OVERALL PERFORMANCE</b>
━━━━━━━━━━━━━━━━━━━━━━━━
  Total Trades: {total_trades}
  Win Rate: {win_rate:.1f}%
  Total P&L: ${total_pnl:,.2f}
  Losing Months: {losing_months}/13

━━━━━━━━━━━━━━━━━━━━━━━━
<b>📅 MONTHLY BREAKDOWN</b>
━━━━━━━━━━━━━━━━━━━━━━━━"""

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

    for i, r in enumerate(results):
        if r.get('skipped'):
            msg += f"\n  {months[i]} {r['year']}: <i>No data</i>"
        else:
            pnl = r.get('total_pnl', 0)
            trades = r.get('total_trades', 0)
            wr = r.get('win_rate', 0)
            emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            msg += f"\n  {emoji} {months[i]} {r['year']}: ${pnl:+,.2f} ({trades} trades, {wr:.0f}% WR)"

    msg += f"""

━━━━━━━━━━━━━━━━━━━━━━━━
<b>📊 STATISTICS</b>
━━━━━━━━━━━━━━━━━━━━━━━━
  Avg Monthly P&L: ${avg_monthly:,.2f}
  Best Month: ${max(monthly_pnl) if monthly_pnl else 0:,.2f}
  Worst Month: ${min(monthly_pnl) if monthly_pnl else 0:,.2f}
  Profitable Months: {13 - losing_months}/13

━━━━━━━━━━━━━━━━━━━━━━━━
<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>
"""

    try:
        await bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML')
        logger.info("Report sent to Telegram")
    except Exception as e:
        logger.error(f"Failed to send Telegram: {e}")


async def main():
    print("=" * 60)
    print("XAUUSD BACKTEST - 13 MONTHS (Jan 2025 - Jan 2026)")
    print("=" * 60)
    print()

    # Step 1: Fetch data from MT5
    print("[1/4] Fetching XAUUSD data from MT5...")

    # Extend range for warmup
    fetch_start = START_DATE - timedelta(days=60)

    htf_df = await fetch_mt5_data(SYMBOL, 'H4', fetch_start, END_DATE)
    ltf_df = await fetch_mt5_data(SYMBOL, 'M15', fetch_start, END_DATE)

    if htf_df is None or ltf_df is None:
        print("ERROR: Failed to fetch data from MT5")
        print("Make sure MT5 is running and XAUUSD is available")
        return

    print(f"  HTF (H4): {len(htf_df)} bars")
    print(f"  LTF (M15): {len(ltf_df)} bars")
    print()

    # Step 2: Save to database
    print("[2/4] Saving to database...")
    await save_to_db(htf_df, SYMBOL, 'H4')
    await save_to_db(ltf_df, SYMBOL, 'M15')
    print()

    # Step 3: Run backtest
    print("[3/4] Running backtest...")
    print()

    results = []
    balance = 10000.0

    # Jan 2025 - Dec 2025
    for month in range(1, 13):
        result = run_monthly_backtest(htf_df, ltf_df, 2025, month, balance)
        results.append(result)

        if not result.get('skipped'):
            balance = result.get('final_balance', balance)
            pnl = result.get('total_pnl', 0)
            trades = result.get('total_trades', 0)
            emoji = "+" if pnl > 0 else ""
            print(f"  2025-{month:02d}: {emoji}${pnl:.2f} ({trades} trades)")
        else:
            print(f"  2025-{month:02d}: SKIPPED (no data)")

    # Jan 2026
    result = run_monthly_backtest(htf_df, ltf_df, 2026, 1, balance)
    results.append(result)
    if not result.get('skipped'):
        pnl = result.get('total_pnl', 0)
        trades = result.get('total_trades', 0)
        emoji = "+" if pnl > 0 else ""
        print(f"  2026-01: {emoji}${pnl:.2f} ({trades} trades)")
    else:
        print(f"  2026-01: SKIPPED (no data)")

    print()

    # Summary
    total_pnl = sum(r.get('total_pnl', 0) for r in results if not r.get('skipped'))
    total_trades = sum(r.get('total_trades', 0) for r in results if not r.get('skipped'))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Final Balance: ${balance:,.2f}")
    print()

    # Step 4: Send to Telegram
    print("[4/4] Sending report to Telegram...")
    await send_telegram_report(results, len(htf_df), len(ltf_df))

    print()
    print("DONE!")


if __name__ == "__main__":
    asyncio.run(main())
