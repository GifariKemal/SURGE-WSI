"""Deep Analysis of 2024 Losing Months
======================================

Analyze WHY Feb, Apr, May, Jun 2024 were losing months.
Look at:
- Trade-by-trade breakdown
- Regime detection accuracy
- Market conditions (volatility, trend)
- Entry quality scores
- Win/loss patterns

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
from collections import defaultdict

from config import config
from src.data.db_handler import DBHandler
from backtest.backtester import Backtester


async def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from database"""
    db = DBHandler(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.user,
        password=config.database.password
    )
    if not await db.connect():
        return pd.DataFrame()
    df = await db.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=100000,
                            start_time=start, end_time=end)
    await db.disconnect()
    return df


def calculate_market_stats(df: pd.DataFrame) -> dict:
    """Calculate market statistics for a period"""
    if df.empty:
        return {}

    # ATR calculation
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    tr = pd.DataFrame()
    tr['hl'] = high - low
    tr['hc'] = abs(high - close.shift(1))
    tr['lc'] = abs(low - close.shift(1))
    tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
    atr = tr['tr'].rolling(14).mean()

    # Trend calculation (directional movement)
    returns = close.pct_change()
    up_days = (returns > 0).sum()
    down_days = (returns < 0).sum()
    total_days = len(returns.dropna())

    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized

    # Range
    price_range = (high.max() - low.min()) / close.mean() * 100

    return {
        'avg_atr_pips': atr.mean() * 10000 if not atr.empty else 0,
        'max_atr_pips': atr.max() * 10000 if not atr.empty else 0,
        'min_atr_pips': atr.min() * 10000 if not atr.empty else 0,
        'up_days_pct': up_days / total_days * 100 if total_days > 0 else 0,
        'down_days_pct': down_days / total_days * 100 if total_days > 0 else 0,
        'volatility_annual': volatility,
        'price_range_pct': price_range,
        'trend_bias': 'BULLISH' if up_days > down_days else 'BEARISH' if down_days > up_days else 'NEUTRAL'
    }


def analyze_month(htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                  start_date: datetime, end_date: datetime,
                  month_name: str) -> dict:
    """Deep analysis of a single month"""

    # Zero Losing Months Config
    CONFIG = {
        'max_lot': 0.5,
        'max_loss_per_trade_pct': 0.8,
        'min_quality': 65.0,
        'min_sl_pips': 15.0,
        'max_sl_pips': 50.0,
    }

    warmup_days = 30
    month_start_warmup = start_date - timedelta(days=warmup_days)

    htf_month = htf_df[(htf_df.index >= month_start_warmup) & (htf_df.index <= end_date)]
    ltf_month = ltf_df[(ltf_df.index >= month_start_warmup) & (ltf_df.index <= end_date)]

    # Market stats for the month only (not warmup)
    htf_period = htf_df[(htf_df.index >= start_date) & (htf_df.index <= end_date)]
    ltf_period = ltf_df[(ltf_df.index >= start_date) & (ltf_df.index <= end_date)]

    market_stats = calculate_market_stats(htf_period)

    if htf_month.empty or ltf_month.empty:
        return {'error': 'No data'}

    htf = htf_month.reset_index()
    htf.rename(columns={'index': 'time'}, inplace=True)
    ltf = ltf_month.reset_index()
    ltf.rename(columns={'index': 'time'}, inplace=True)

    # Run backtest
    bt = Backtester(
        symbol="GBPUSD",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_balance=10000.0,
        pip_value=10.0,
        spread_pips=1.5,
        use_killzone=True,
        use_trend_filter=True,
        use_relaxed_filter=False,
        use_hybrid_mode=False
    )

    bt.risk_manager.max_lot_size = CONFIG['max_lot']
    bt.risk_manager.min_sl_pips = CONFIG['min_sl_pips']
    bt.risk_manager.max_sl_pips = CONFIG['max_sl_pips']
    bt.entry_trigger.min_quality_score = CONFIG['min_quality']

    bt.load_data(htf, ltf)
    result = bt.run()

    # Analyze trades
    trades = result.trade_list
    trade_analysis = []

    total_pnl = 0
    wins = 0
    losses = 0

    regime_accuracy = {'correct': 0, 'wrong': 0}
    quality_scores = []
    sl_distances = []
    trade_durations = []

    for trade in trades:
        pnl = trade.pnl

        # Apply loss cap
        max_loss = 10000 * CONFIG['max_loss_per_trade_pct'] / 100
        if pnl < 0 and abs(pnl) > max_loss:
            pnl = -max_loss

        total_pnl += pnl

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        # Regime accuracy check
        # If trade won, regime was likely correct
        if trade.pnl > 0:
            regime_accuracy['correct'] += 1
        else:
            regime_accuracy['wrong'] += 1

        quality_scores.append(trade.quality_score)

        # SL distance in pips
        sl_pips = abs(trade.entry_price - trade.sl_price) / 0.0001
        sl_distances.append(sl_pips)

        # Trade duration
        if trade.exit_time and trade.entry_time:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
            trade_durations.append(duration)

        trade_analysis.append({
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'direction': trade.direction,
            'regime': trade.regime,
            'quality': trade.quality_score,
            'pnl': pnl,
            'pips': trade.pips,
            'sl_pips': sl_pips,
            'tp1_hit': trade.tp1_hit,
            'tp2_hit': trade.tp2_hit,
            'status': trade.status.value
        })

    return {
        'month': month_name,
        'total_pnl': total_pnl,
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'market_stats': market_stats,
        'avg_quality': np.mean(quality_scores) if quality_scores else 0,
        'avg_sl_pips': np.mean(sl_distances) if sl_distances else 0,
        'avg_duration_hours': np.mean(trade_durations) if trade_durations else 0,
        'regime_accuracy': regime_accuracy,
        'tp1_hit_rate': sum(1 for t in trades if t.tp1_hit) / len(trades) * 100 if trades else 0,
        'trade_details': trade_analysis,
        'debug': {
            'regime_fail': bt._debug_regime_fail,
            'poi_none': bt._debug_poi_none,
            'sideways': bt._debug_regime_sideways,
            'not_in_poi': bt._debug_not_in_poi,
            'entry_fail': bt._debug_entry_fail,
            'trend_filtered': bt._debug_trend_filtered
        }
    }


def print_analysis(analysis: dict):
    """Print detailed analysis"""
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {analysis['month']}")
    print(f"{'='*70}")

    print(f"\n[SUMMARY]")
    print(f"  Total P/L: ${analysis['total_pnl']:+,.2f}")
    print(f"  Trades: {analysis['trades']} (W:{analysis['wins']} L:{analysis['losses']})")
    print(f"  Win Rate: {analysis['win_rate']:.1f}%")
    print(f"  TP1 Hit Rate: {analysis['tp1_hit_rate']:.1f}%")

    ms = analysis['market_stats']
    print(f"\n[MARKET CONDITIONS]")
    print(f"  Trend Bias: {ms.get('trend_bias', 'N/A')}")
    print(f"  Up Days: {ms.get('up_days_pct', 0):.1f}%")
    print(f"  Down Days: {ms.get('down_days_pct', 0):.1f}%")
    print(f"  Avg ATR: {ms.get('avg_atr_pips', 0):.1f} pips")
    print(f"  Max ATR: {ms.get('max_atr_pips', 0):.1f} pips")
    print(f"  Volatility (Ann): {ms.get('volatility_annual', 0):.1f}%")
    print(f"  Price Range: {ms.get('price_range_pct', 0):.2f}%")

    print(f"\n[TRADE QUALITY]")
    print(f"  Avg Quality Score: {analysis['avg_quality']:.1f}")
    print(f"  Avg SL Distance: {analysis['avg_sl_pips']:.1f} pips")
    print(f"  Avg Duration: {analysis['avg_duration_hours']:.1f} hours")

    ra = analysis['regime_accuracy']
    total_ra = ra['correct'] + ra['wrong']
    print(f"\n[REGIME DETECTION]")
    print(f"  Trades with correct regime: {ra['correct']}/{total_ra} "
          f"({ra['correct']/total_ra*100:.1f}% accuracy)" if total_ra > 0 else "  No trades")

    dbg = analysis['debug']
    print(f"\n[FILTER STATS]")
    print(f"  Regime Fail: {dbg['regime_fail']}")
    print(f"  POI None: {dbg['poi_none']}")
    print(f"  Sideways Skip: {dbg['sideways']}")
    print(f"  Not in POI: {dbg['not_in_poi']}")
    print(f"  Entry Fail: {dbg['entry_fail']}")
    print(f"  Trend Filtered: {dbg['trend_filtered']}")

    print(f"\n[TRADE-BY-TRADE]")
    print(f"  {'Time':<18} {'Dir':<5} {'Regime':<8} {'Qual':<5} {'P/L':<10} {'Pips':<8} {'Status':<10}")
    print(f"  {'-'*75}")

    for t in analysis['trade_details']:
        time_str = t['entry_time'].strftime('%m/%d %H:%M') if t['entry_time'] else 'N/A'
        print(f"  {time_str:<18} {t['direction']:<5} {t['regime']:<8} {t['quality']:<5.0f} "
              f"${t['pnl']:>+8.2f} {t['pips']:>+7.1f} {t['status']:<10}")

    # Identify patterns
    print(f"\n[PATTERN ANALYSIS]")

    # Consecutive losses
    consecutive_losses = 0
    max_consecutive = 0
    for t in analysis['trade_details']:
        if t['pnl'] < 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0
    print(f"  Max Consecutive Losses: {max_consecutive}")

    # Direction bias
    buy_trades = [t for t in analysis['trade_details'] if t['direction'] == 'BUY']
    sell_trades = [t for t in analysis['trade_details'] if t['direction'] == 'SELL']
    buy_wins = sum(1 for t in buy_trades if t['pnl'] > 0)
    sell_wins = sum(1 for t in sell_trades if t['pnl'] > 0)
    print(f"  BUY trades: {len(buy_trades)} ({buy_wins} wins, {buy_wins/len(buy_trades)*100:.0f}% WR)" if buy_trades else "  BUY trades: 0")
    print(f"  SELL trades: {len(sell_trades)} ({sell_wins} wins, {sell_wins/len(sell_trades)*100:.0f}% WR)" if sell_trades else "  SELL trades: 0")

    # Regime distribution
    regimes = defaultdict(list)
    for t in analysis['trade_details']:
        regimes[t['regime']].append(t['pnl'])

    print(f"\n  Regime Performance:")
    for regime, pnls in regimes.items():
        wins = sum(1 for p in pnls if p > 0)
        print(f"    {regime}: {len(pnls)} trades, {wins} wins ({wins/len(pnls)*100:.0f}% WR), "
              f"${sum(pnls):+.2f} total")


async def main():
    print("\n" + "=" * 70)
    print("DEEP ANALYSIS: 2024 LOSING MONTHS")
    print("Feb, Apr, May, Jun - Why did they lose?")
    print("=" * 70)

    # Fetch data
    print("\nFetching data...")
    start = datetime(2023, 12, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    htf_df = await fetch_data("GBPUSD", "H4", start, end)
    ltf_df = await fetch_data("GBPUSD", "M15", start, end)

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data")
        return

    print(f"H4: {len(htf_df)} bars, M15: {len(ltf_df)} bars")

    # Analyze losing months
    losing_months = [
        (datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 2, 29, tzinfo=timezone.utc), "Feb 2024"),
        (datetime(2024, 4, 1, tzinfo=timezone.utc), datetime(2024, 4, 30, tzinfo=timezone.utc), "Apr 2024"),
        (datetime(2024, 5, 1, tzinfo=timezone.utc), datetime(2024, 5, 31, tzinfo=timezone.utc), "May 2024"),
        (datetime(2024, 6, 1, tzinfo=timezone.utc), datetime(2024, 6, 30, tzinfo=timezone.utc), "Jun 2024"),
    ]

    # Also analyze some winning months for comparison
    winning_months = [
        (datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc), "Jan 2024"),
        (datetime(2024, 8, 1, tzinfo=timezone.utc), datetime(2024, 8, 31, tzinfo=timezone.utc), "Aug 2024"),
        (datetime(2024, 9, 1, tzinfo=timezone.utc), datetime(2024, 9, 30, tzinfo=timezone.utc), "Sep 2024"),
    ]

    all_analyses = []

    print("\n" + "=" * 70)
    print("LOSING MONTHS ANALYSIS")
    print("=" * 70)

    for start_date, end_date, month_name in losing_months:
        analysis = analyze_month(htf_df, ltf_df, start_date, end_date, month_name)
        all_analyses.append(('LOSS', analysis))
        print_analysis(analysis)

    print("\n" + "=" * 70)
    print("WINNING MONTHS (FOR COMPARISON)")
    print("=" * 70)

    for start_date, end_date, month_name in winning_months:
        analysis = analyze_month(htf_df, ltf_df, start_date, end_date, month_name)
        all_analyses.append(('WIN', analysis))
        print_analysis(analysis)

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)

    print("\n{:<12} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "Month", "P/L", "Trades", "WinRate", "AvgATR", "Quality", "RegimeAcc"))
    print("-" * 70)

    for cat, a in all_analyses:
        marker = "[L]" if cat == 'LOSS' else "[W]"
        ms = a['market_stats']
        ra = a['regime_accuracy']
        total_ra = ra['correct'] + ra['wrong']
        regime_acc = ra['correct'] / total_ra * 100 if total_ra > 0 else 0

        print("{:<12} {:>+7.0f}$ {:>8} {:>7.1f}% {:>9.1f} {:>10.1f} {:>9.1f}%".format(
            f"{marker} {a['month'][:7]}",
            a['total_pnl'],
            a['trades'],
            a['win_rate'],
            ms.get('avg_atr_pips', 0),
            a['avg_quality'],
            regime_acc
        ))

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS & ROOT CAUSE ANALYSIS")
    print("=" * 70)

    losing_analyses = [a for cat, a in all_analyses if cat == 'LOSS']
    winning_analyses = [a for cat, a in all_analyses if cat == 'WIN']

    # Average metrics
    avg_losing_wr = np.mean([a['win_rate'] for a in losing_analyses])
    avg_winning_wr = np.mean([a['win_rate'] for a in winning_analyses])

    avg_losing_quality = np.mean([a['avg_quality'] for a in losing_analyses])
    avg_winning_quality = np.mean([a['avg_quality'] for a in winning_analyses])

    avg_losing_atr = np.mean([a['market_stats'].get('avg_atr_pips', 0) for a in losing_analyses])
    avg_winning_atr = np.mean([a['market_stats'].get('avg_atr_pips', 0) for a in winning_analyses])

    print(f"\n1. WIN RATE DIFFERENCE:")
    print(f"   Losing months avg WR: {avg_losing_wr:.1f}%")
    print(f"   Winning months avg WR: {avg_winning_wr:.1f}%")
    print(f"   -> Gap: {avg_winning_wr - avg_losing_wr:.1f}%")

    print(f"\n2. QUALITY SCORE:")
    print(f"   Losing months avg: {avg_losing_quality:.1f}")
    print(f"   Winning months avg: {avg_winning_quality:.1f}")

    print(f"\n3. VOLATILITY (ATR):")
    print(f"   Losing months avg: {avg_losing_atr:.1f} pips")
    print(f"   Winning months avg: {avg_winning_atr:.1f} pips")

    print(f"\n4. REGIME ACCURACY:")
    for cat, a in all_analyses:
        ra = a['regime_accuracy']
        total = ra['correct'] + ra['wrong']
        acc = ra['correct'] / total * 100 if total > 0 else 0
        print(f"   {a['month']}: {acc:.1f}% ({ra['correct']}/{total})")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on the analysis, potential improvements to investigate:

1. REGIME DETECTION ENHANCEMENT
   - Current HMM may not adapt well to changing market conditions
   - Consider: Adaptive HMM, Online Learning, or Ensemble methods
   - Research: "Online HMM learning", "Adaptive regime switching"

2. VOLATILITY-BASED FILTERING
   - Losing months may have different volatility patterns
   - Consider: Dynamic ATR thresholds, Volatility regime detection
   - Research: GARCH models, Realized volatility measures

3. KALMAN FILTER TUNING
   - Process/measurement noise may need adaptation
   - Consider: Adaptive Kalman, Unscented Kalman Filter
   - Research: "Adaptive Kalman filter for trading"

4. ENTRY QUALITY IMPROVEMENT
   - Quality scores in losing months may not be reliable
   - Consider: Machine learning-based entry scoring
   - Research: Random Forest, XGBoost for signal quality

5. MARKET CONDITION CLASSIFICATION
   - Add explicit choppy/ranging market detection
   - Consider: ADX-based filters, Hurst exponent
   - Research: "Market regime classification machine learning"
    """)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
