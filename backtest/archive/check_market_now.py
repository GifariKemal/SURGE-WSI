"""Check Current Market Conditions
===================================

Analyze current market for trading opportunity at 15:00 UTC.

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
from datetime import datetime, timedelta, timezone
from loguru import logger

from config import config
from src.data.db_handler import DBHandler
from src.analysis.kalman_filter import MultiScaleKalman
from src.analysis.regime_detector import HMMRegimeDetector, MarketRegime
from src.analysis.poi_detector import POIDetector
from src.utils.intelligent_activity_filter import IntelligentActivityFilter, MarketActivity

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


async def main():
    print("\n" + "="*60)
    print("MARKET ANALYSIS - TRADING OPPORTUNITY CHECK")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("="*60)

    # Fetch recent data
    db = DBHandler(
        host=config.database.host, port=config.database.port,
        database=config.database.database, user=config.database.user,
        password=config.database.password
    )
    await db.connect()

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)

    htf_df = await db.get_ohlcv('GBPUSD', 'H4', 500, start_time, end_time)
    ltf_df = await db.get_ohlcv('GBPUSD', 'M15', 2000, start_time, end_time)

    await db.disconnect()

    # Normalize columns
    htf_df.columns = [c.lower() for c in htf_df.columns]
    ltf_df.columns = [c.lower() for c in ltf_df.columns]

    print(f"\nData loaded: H4={len(htf_df)}, M15={len(ltf_df)}")

    if htf_df.empty or ltf_df.empty:
        print("ERROR: No data available!")
        return

    # Current price
    current_price = ltf_df['close'].iloc[-1]
    current_high = ltf_df['high'].iloc[-1]
    current_low = ltf_df['low'].iloc[-1]
    last_time = ltf_df.index[-1]

    print(f"\nCurrent Price: {current_price:.5f}")
    print(f"Last Data: {last_time}")

    # Initialize components
    kalman = MultiScaleKalman()
    regime_detector = HMMRegimeDetector()
    poi_detector = POIDetector()
    intelligent_filter = IntelligentActivityFilter(
        activity_threshold=60.0,
        min_velocity_pips=2.0,
        min_atr_pips=5.0,
        pip_size=0.0001
    )

    # Warmup Kalman and Intelligent Filter
    print("\nWarming up analysis components...")
    for _, row in ltf_df.iterrows():
        kalman_state = kalman.update(row['close'])
        intelligent_filter.update(row['high'], row['low'], row['close'])
        intelligent_filter.update_kalman_velocity(kalman_state['velocity'])

    # Warmup Regime Detector
    for _, row in htf_df.iterrows():
        regime_detector.update(row['close'])

    # Detect POIs
    htf_reset = htf_df.reset_index().rename(columns={'index': 'time'})
    poi_detector.detect(htf_reset)

    # Get current analysis
    print("\n" + "="*60)
    print("CURRENT MARKET CONDITIONS")
    print("="*60)

    # 1. Intelligent Filter
    now = datetime.now(timezone.utc)
    intel_result = intelligent_filter.check(now, current_high, current_low, current_price)

    print(f"\n🧠 INTELLIGENT FILTER (INTEL_60):")
    print(f"   Activity: {intel_result.activity.value.upper()} ({intel_result.score:.0f}/100)")
    print(f"   Should Trade: {'YES ✅' if intel_result.should_trade else 'NO ❌'}")
    print(f"   Velocity: {intel_result.velocity:.2f} pips/bar")
    print(f"   ATR: {intel_result.atr_pips:.1f} pips")
    print(f"   Quality Threshold: {intel_result.quality_threshold:.0f}")
    print(f"   Reason: {intel_result.reason}")

    # 2. Regime
    regime_info = regime_detector.last_info
    print(f"\n📊 REGIME DETECTOR:")
    if regime_info:
        print(f"   Regime: {regime_info.regime.value}")
        print(f"   Bias: {regime_info.bias}")
        print(f"   Probability: {regime_info.probability:.1%}")
        print(f"   Tradeable: {'YES ✅' if regime_info.is_tradeable else 'NO ❌'}")
    else:
        print("   No regime info available")

    # 3. POIs
    poi_result = poi_detector.last_result
    print(f"\n🎯 POI DETECTION:")
    if poi_result:
        bull_pois = poi_result.bullish_pois if poi_result.bullish_pois else []
        bear_pois = poi_result.bearish_pois if poi_result.bearish_pois else []

        print(f"   Bullish POIs: {len(bull_pois)}")
        for i, poi in enumerate(bull_pois[:3]):  # Show top 3
            zone_high = poi.get('top', poi.get('high', 0))
            zone_low = poi.get('bottom', poi.get('low', 0))
            strength = poi.get('strength', 0)
            print(f"     {i+1}. Zone: {zone_low:.5f} - {zone_high:.5f} (str: {strength:.0f})")

        print(f"   Bearish POIs: {len(bear_pois)}")
        for i, poi in enumerate(bear_pois[:3]):  # Show top 3
            zone_high = poi.get('top', poi.get('high', 0))
            zone_low = poi.get('bottom', poi.get('low', 0))
            strength = poi.get('strength', 0)
            print(f"     {i+1}. Zone: {zone_low:.5f} - {zone_high:.5f} (str: {strength:.0f})")

        # Check if price at POI
        if regime_info and regime_info.bias in ['BUY', 'SELL']:
            at_poi, poi_info = poi_result.price_at_poi(current_price, regime_info.bias)
            print(f"\n   Price at POI: {'YES ✅' if at_poi else 'NO ❌'}")
            if at_poi and poi_info:
                print(f"   Active Zone: {poi_info.get('bottom', 0):.5f} - {poi_info.get('top', 0):.5f}")
    else:
        print("   No POI data available")

    # 4. Trading Opportunity Summary
    print("\n" + "="*60)
    print("TRADING OPPORTUNITY @ 15:00 UTC")
    print("="*60)

    can_trade = True
    reasons = []

    if not intel_result.should_trade:
        can_trade = False
        reasons.append(f"Market not active ({intel_result.activity.value})")

    if not regime_info or not regime_info.is_tradeable:
        can_trade = False
        reasons.append("Regime not tradeable")

    if regime_info:
        direction = regime_info.bias
        if direction == 'BUY' and len(bull_pois) == 0:
            can_trade = False
            reasons.append("No bullish POI available")
        elif direction == 'SELL' and len(bear_pois) == 0:
            can_trade = False
            reasons.append("No bearish POI available")

        if poi_result:
            at_poi, _ = poi_result.price_at_poi(current_price, direction)
            if not at_poi:
                reasons.append(f"Price not at {direction} POI zone yet")

    if can_trade:
        print(f"\n✅ TRADING POSSIBLE!")
        print(f"   Direction: {regime_info.bias if regime_info else 'N/A'}")
        print(f"   Quality Threshold: {intel_result.quality_threshold:.0f}")
    else:
        print(f"\n⏳ WAITING FOR CONDITIONS...")
        for r in reasons:
            print(f"   - {r}")

    # What to watch
    print("\n📋 WHAT TO WATCH:")
    if regime_info:
        if regime_info.bias == 'BUY':
            print("   - Wait for price to retrace to Bullish POI zone")
            print("   - Look for rejection candle / MSS at POI")
        elif regime_info.bias == 'SELL':
            print("   - Wait for price to retrace to Bearish POI zone")
            print("   - Look for rejection candle / MSS at POI")
        else:
            print("   - Wait for clear regime direction")
    else:
        print("   - Wait for regime to stabilize")

    # Send to Telegram
    if TELEGRAM_AVAILABLE:
        print("\nSending to Telegram...")

        msg = "📊 <b>MARKET CHECK - 15:00 UTC</b>\n"
        msg += f"<i>{now.strftime('%Y-%m-%d %H:%M')} UTC</i>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        msg += f"💹 <b>GBPUSD:</b> {current_price:.5f}\n\n"

        # Activity
        emoji = intel_result.get_emoji()
        msg += f"🧠 <b>Activity:</b> {emoji} {intel_result.activity.value.upper()}\n"
        msg += f"├ Score: {intel_result.score:.0f}/100\n"
        msg += f"├ Velocity: {intel_result.velocity:.2f} p/bar\n"
        msg += f"└ ATR: {intel_result.atr_pips:.1f} pips\n\n"

        # Regime
        if regime_info:
            regime_emoji = "🟢" if regime_info.regime == MarketRegime.BULLISH else ("🔴" if regime_info.regime == MarketRegime.BEARISH else "⚪")
            msg += f"📈 <b>Regime:</b> {regime_emoji} {regime_info.regime.value}\n"
            msg += f"├ Bias: {regime_info.bias}\n"
            msg += f"└ Prob: {regime_info.probability:.0%}\n\n"

        # POIs
        msg += f"🎯 <b>POIs:</b>\n"
        msg += f"├ Bullish: {len(bull_pois)}\n"
        msg += f"└ Bearish: {len(bear_pois)}\n\n"

        # Opportunity
        if can_trade:
            msg += f"✅ <b>READY TO TRADE</b>\n"
            msg += f"Direction: {regime_info.bias}\n"
        else:
            msg += f"⏳ <b>WAITING...</b>\n"
            for r in reasons[:3]:
                msg += f"• {r}\n"

        try:
            bot = Bot(token=config.telegram.bot_token)
            await bot.send_message(chat_id=config.telegram.chat_id, text=msg, parse_mode='HTML')
            print("Sent!")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*60)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(main())
