"""
Multi-Year Backtest for GBPUSD H1 QuadLayer Strategy
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Initialize MT5
if not mt5.initialize():
    print('MT5 initialization failed')
    exit()

# Constants
SYMBOL = 'GBPUSD'
PIP_SIZE = 0.0001
INITIAL_BALANCE = 50000
RISK_PERCENT = 1.0
SL_ATR_MULT = 1.5
TP_RATIO = 1.5

# Monthly tradeable percentage (seasonal template)
MONTHLY_TRADEABLE_PCT = {
    1: 65, 2: 55, 3: 70, 4: 70, 5: 62, 6: 68,
    7: 78, 8: 60, 9: 72, 10: 58, 11: 66, 12: 45
}

def get_monthly_quality_adj(month: int) -> int:
    pct = MONTHLY_TRADEABLE_PCT.get(month, 60)
    if pct < 50: return 25
    elif pct < 60: return 15
    elif pct < 70: return 10
    elif pct < 75: return 5
    return 0


@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    pnl: float = 0.0
    exit_time: Optional[datetime] = None


def fetch_data(year: int) -> pd.DataFrame:
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)

    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean() / PIP_SIZE


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def is_kill_zone(hour: int) -> Tuple[bool, str]:
    if 8 <= hour <= 10:
        return True, 'london'
    elif 13 <= hour <= 17:
        return True, 'newyork'
    return False, ''


def run_backtest(year: int) -> dict:
    df = fetch_data(year)
    if df.empty:
        return {'year': year, 'error': 'No data'}

    # Calculate indicators
    atr = calculate_atr(df)
    ema21 = calculate_ema(df['close'], 21)
    ema50 = calculate_ema(df['close'], 50)
    ema200 = calculate_ema(df['close'], 200)

    trades: List[Trade] = []
    balance = INITIAL_BALANCE
    monthly_pnl = {}

    for i in range(200, len(df) - 24):
        row = df.iloc[i]
        current_time = df.index[i]
        hour = current_time.hour
        month = current_time.month

        # Kill zone check
        in_kz, session = is_kill_zone(hour)
        if not in_kz:
            continue

        # Get values
        close = row['close']
        current_atr = atr.iloc[i]
        current_ema21 = ema21.iloc[i]
        current_ema50 = ema50.iloc[i]
        current_ema200 = ema200.iloc[i]

        if pd.isna(current_atr) or current_atr <= 0:
            continue

        # Quality calculation
        base_quality = 65
        monthly_adj = get_monthly_quality_adj(month)
        total_quality = base_quality + monthly_adj

        # Signal quality requirement
        required_quality = 75

        # Simple signal: EMA alignment + pullback
        signal = None
        signal_quality = 0

        # Bullish
        if close > current_ema21 > current_ema50 > current_ema200:
            if df['low'].iloc[i] <= current_ema21 * 1.001:  # Pullback to EMA21
                signal = 'BUY'
                signal_quality = 80

        # Bearish
        elif close < current_ema21 < current_ema50 < current_ema200:
            if df['high'].iloc[i] >= current_ema21 * 0.999:  # Pullback to EMA21
                signal = 'SELL'
                signal_quality = 80

        if signal is None or signal_quality < total_quality:
            continue

        # Calculate SL/TP
        sl_pips = current_atr * SL_ATR_MULT
        sl_pips = max(10, min(50, sl_pips))
        tp_pips = sl_pips * TP_RATIO

        # Calculate lot size
        risk_amount = balance * (RISK_PERCENT / 100)
        lot_size = risk_amount / (sl_pips * 10)
        lot_size = max(0.01, min(5.0, round(lot_size, 2)))

        # Entry
        entry_price = close
        if signal == 'BUY':
            sl = entry_price - sl_pips * PIP_SIZE
            tp = entry_price + tp_pips * PIP_SIZE
        else:
            sl = entry_price + sl_pips * PIP_SIZE
            tp = entry_price - tp_pips * PIP_SIZE

        # Simulate trade outcome (look ahead)
        pnl = 0
        exit_time = None
        for j in range(i + 1, min(i + 48, len(df))):
            future_high = df['high'].iloc[j]
            future_low = df['low'].iloc[j]

            if signal == 'BUY':
                if future_low <= sl:
                    pnl = -sl_pips * 10 * lot_size
                    exit_time = df.index[j]
                    break
                elif future_high >= tp:
                    pnl = tp_pips * 10 * lot_size
                    exit_time = df.index[j]
                    break
            else:
                if future_high >= sl:
                    pnl = -sl_pips * 10 * lot_size
                    exit_time = df.index[j]
                    break
                elif future_low <= tp:
                    pnl = tp_pips * 10 * lot_size
                    exit_time = df.index[j]
                    break

        if pnl != 0:
            trades.append(Trade(
                entry_time=current_time,
                direction=signal,
                entry_price=entry_price,
                sl=sl, tp=tp,
                lot_size=lot_size,
                pnl=pnl,
                exit_time=exit_time
            ))
            balance += pnl

            # Track monthly P&L
            month_key = f'{current_time.year}-{current_time.month:02d}'
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl

    # Calculate stats
    if not trades:
        return {'year': year, 'trades': 0, 'error': 'No trades'}

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    total_profit = sum(t.pnl for t in trades if t.pnl > 0)
    total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

    # Drawdown
    equity = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    losing_months = sum(1 for v in monthly_pnl.values() if v < 0)

    return {
        'year': year,
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) * 100,
        'profit_factor': total_profit / total_loss if total_loss > 0 else 999,
        'net_pnl': sum(t.pnl for t in trades),
        'return_pct': (sum(t.pnl for t in trades) / INITIAL_BALANCE) * 100,
        'max_dd': max_dd,
        'max_dd_pct': (max_dd / INITIAL_BALANCE) * 100,
        'losing_months': losing_months,
        'total_months': len(monthly_pnl)
    }


if __name__ == '__main__':
    print('='*80)
    print('MULTI-YEAR BACKTEST - GBPUSD H1 QuadLayer Strategy (Simplified)')
    print('='*80)
    print(f'Initial Balance: ${INITIAL_BALANCE:,}')
    print(f'Risk per Trade: {RISK_PERCENT}%')
    print(f'SL: {SL_ATR_MULT}x ATR, TP: {TP_RATIO}:1 RR')
    print()

    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    results = []

    print(f"{'Year':<6} {'Trades':>7} {'WR':>7} {'PF':>7} {'Net P/L':>12} {'Return':>8} {'Max DD':>8} {'Losing':<10}")
    print('-'*80)

    for year in years:
        result = run_backtest(year)
        results.append(result)

        if 'error' in result:
            print(f'{year:<6} {result["error"]}')
        else:
            print(f'{year:<6} {result["trades"]:>7} {result["win_rate"]:>6.1f}% {result["profit_factor"]:>7.2f} ${result["net_pnl"]:>+10,.0f} {result["return_pct"]:>+7.1f}% {result["max_dd_pct"]:>7.2f}% {result["losing_months"]}/{result["total_months"]}')

    print('-'*80)
    print()
    print('='*80)
    print('SUMMARY')
    print('='*80)

    valid = [r for r in results if 'error' not in r]
    if valid:
        total_trades = sum(r['trades'] for r in valid)
        avg_wr = sum(r['win_rate'] for r in valid) / len(valid)
        avg_pf = sum(r['profit_factor'] for r in valid) / len(valid)
        total_pnl = sum(r['net_pnl'] for r in valid)
        avg_dd = sum(r['max_dd_pct'] for r in valid) / len(valid)
        total_losing = sum(r['losing_months'] for r in valid)
        total_months = sum(r['total_months'] for r in valid)

        profitable_years = sum(1 for r in valid if r['net_pnl'] > 0)

        print(f'Years Tested:    {len(valid)}')
        print(f'Profitable Years: {profitable_years}/{len(valid)}')
        print(f'Total Trades:    {total_trades}')
        print(f'Avg Win Rate:    {avg_wr:.1f}%')
        print(f'Avg PF:          {avg_pf:.2f}')
        print(f'Total P/L:       ${total_pnl:+,.0f}')
        print(f'Avg Max DD:      {avg_dd:.2f}%')
        print(f'Losing Months:   {total_losing}/{total_months}')
        print()

        # Compound calculation
        print('COMPOUND GROWTH (reinvesting all profits):')
        print('-'*40)
        compound_balance = INITIAL_BALANCE
        for r in valid:
            if r['net_pnl'] > 0:
                old_balance = compound_balance
                compound_balance += r['net_pnl']
                print(f"  {r['year']}: ${old_balance:,.0f} -> ${compound_balance:,.0f} (+${r['net_pnl']:,.0f})")
            else:
                old_balance = compound_balance
                compound_balance += r['net_pnl']
                print(f"  {r['year']}: ${old_balance:,.0f} -> ${compound_balance:,.0f} (${r['net_pnl']:,.0f})")

        total_return = ((compound_balance / INITIAL_BALANCE) - 1) * 100
        print(f'\nFinal Balance: ${compound_balance:,.0f} ({total_return:+.0f}% total return)')

    mt5.shutdown()
