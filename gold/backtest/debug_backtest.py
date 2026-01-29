"""Debug Gold Backtest - Find why 0 trades
=========================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from config import config
from src.data.db_handler import DBHandler
from gold.config.gold_settings import get_gold_config
from src.analysis.kalman_filter import KalmanNoiseReducer
from src.analysis.regime_detector import HMMRegimeDetector
from src.analysis.poi_detector import POIDetector


async def load_data():
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()
    htf = await db.get_ohlcv("XAUUSD", "H4", 500)
    await db.disconnect()
    return htf


async def main():
    print("=" * 60)
    print("DEBUG: Why 0 trades?")
    print("=" * 60)
    print()

    cfg = get_gold_config()

    # Load data
    htf_df = await load_data()
    htf_df.columns = [c.lower() for c in htf_df.columns]
    print(f"Loaded {len(htf_df)} H4 bars")
    print(f"Date range: {htf_df.index.min()} to {htf_df.index.max()}")
    print()

    # Initialize components
    kalman = KalmanNoiseReducer()
    regime = HMMRegimeDetector()
    poi = POIDetector(
        swing_length=10,
        ob_min_strength=cfg.poi.min_ob_strength,
        max_poi_age_bars=cfg.poi.max_ob_age_bars,
        pip_size=cfg.symbol.pip_size  # Gold uses 0.1
    )

    # Counters for debugging
    session_pass = 0
    session_fail = 0
    regime_pass = 0
    regime_fail = 0
    activity_pass = 0
    activity_fail = 0
    poi_pass = 0
    poi_fail = 0

    # Calculate ATR
    htf_df['tr'] = np.maximum(
        htf_df['high'] - htf_df['low'],
        np.maximum(
            abs(htf_df['high'] - htf_df['close'].shift(1)),
            abs(htf_df['low'] - htf_df['close'].shift(1))
        )
    )
    htf_df['atr'] = htf_df['tr'].rolling(14).mean()

    # Warmup
    warmup = 100
    for i in range(warmup):
        state = kalman.update(htf_df.iloc[i]['close'])
        if state:
            regime.update(state.smoothed_price)

    print(f"Warmup done ({warmup} bars)")
    print()

    # Check each bar
    for i in range(warmup, len(htf_df)):
        bar = htf_df.iloc[i]
        bar_time = htf_df.index[i]
        price = bar['close']
        atr = bar['atr'] if not pd.isna(bar['atr']) else 187 * cfg.symbol.pip_size

        # Update kalman
        state = kalman.update(price)
        if not state:
            continue

        # Update regime
        regime_info = regime.update(state.smoothed_price)

        # Update POI
        if i > 30:
            poi_data = htf_df.iloc[i-30:i+1].copy()
            poi.detect(poi_data)

        # Check 1: Trading session
        hour = bar_time.hour
        weekday = bar_time.weekday()
        in_session = (weekday < 5 and
                      cfg.session.primary_session_start <= hour < cfg.session.primary_session_end)

        if not in_session:
            session_fail += 1
            continue
        session_pass += 1

        # Check 2: Regime
        if not regime_info or not regime_info.is_tradeable:
            regime_fail += 1
            continue
        regime_pass += 1

        direction = regime_info.bias
        if direction == "NONE":
            regime_fail += 1
            continue

        # Check 3: Activity
        velocity_pips = abs(state.velocity) / cfg.symbol.pip_size
        atr_pips = atr / cfg.symbol.pip_size

        vel_ok = velocity_pips >= cfg.intel_filter.min_velocity_pips * 0.5
        atr_ok = atr_pips >= cfg.intel_filter.min_atr_pips * 0.5

        if not (vel_ok and atr_ok):
            activity_fail += 1
            continue
        activity_pass += 1

        # Check 4: POI
        poi_result = poi.last_result
        if not poi_result:
            poi_fail += 1
            continue

        at_poi, poi_info = poi_result.price_at_poi(price, direction, tolerance_pips=cfg.poi.tolerance_pips, pip_size=cfg.symbol.pip_size)
        if not at_poi:
            poi_fail += 1
            continue
        poi_pass += 1

        print(f"POTENTIAL TRADE at {bar_time}: {direction} @ {price:.2f}")

    print()
    print("=" * 60)
    print("FILTER STATISTICS")
    print("=" * 60)
    print(f"Total bars checked: {len(htf_df) - warmup}")
    print()
    print(f"Session filter:")
    print(f"  Pass: {session_pass}")
    print(f"  Fail: {session_fail}")
    print()
    print(f"Regime filter (after session):")
    print(f"  Pass: {regime_pass}")
    print(f"  Fail: {regime_fail}")
    print()
    print(f"Activity filter (after regime):")
    print(f"  Pass: {activity_pass}")
    print(f"  Fail: {activity_fail}")
    print()
    print(f"POI filter (after activity):")
    print(f"  Pass: {poi_pass}")
    print(f"  Fail: {poi_fail}")
    print()

    # Check regime distribution
    print("=" * 60)
    print("REGIME CHECK")
    print("=" * 60)
    regime_info = regime.last_info
    if regime_info:
        print(f"Current regime: {regime_info.regime}")
        print(f"Bias: {regime_info.bias}")
        print(f"Probability: {regime_info.probability:.2%}")
        print(f"Is tradeable: {regime_info.is_tradeable}")
    else:
        print("No regime info")
    print()

    # Check POI
    print("=" * 60)
    print("POI CHECK")
    print("=" * 60)
    poi_result = poi.last_result
    if poi_result:
        print(f"Bullish POIs: {len(poi_result.bullish_pois)}")
        print(f"Bearish POIs: {len(poi_result.bearish_pois)}")
        for p in poi_result.bullish_pois[:3]:
            print(f"  Bull POI: {p}")
        for p in poi_result.bearish_pois[:3]:
            print(f"  Bear POI: {p}")
    else:
        print("No POI result")
    print()

    # Check sample velocities
    print("=" * 60)
    print("VELOCITY CHECK (last 20 bars)")
    print("=" * 60)
    for i in range(-20, 0):
        bar = htf_df.iloc[i]
        state = kalman.update(bar['close'])
        if state:
            vel_pips = abs(state.velocity) / cfg.symbol.pip_size
            print(f"  {htf_df.index[i]}: velocity={vel_pips:.1f} pips (min req: {cfg.intel_filter.min_velocity_pips * 0.5:.1f})")


if __name__ == "__main__":
    asyncio.run(main())
