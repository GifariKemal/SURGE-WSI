"""Send INTEL_60 Live Configuration Notification"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime
from config import config

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


async def main():
    if not TELEGRAM_AVAILABLE:
        print("Telegram not available")
        return

    msg = "🚀 <b>LIVE TRADING CONFIG UPDATED</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    msg += "🧠 <b>Mode:</b> INTEL_60 (Intelligent Activity Filter)\n\n"

    msg += "📊 <b>Backtest Results (13 months):</b>\n"
    msg += "<pre>\n"
    msg += "Jan 2025 - Jan 2026\n"
    msg += "───────────────────\n"
    msg += "Trades:     29 (2.2/month)\n"
    msg += "Win Rate:   72% (HIGHEST)\n"
    msg += "Losing:     2 months\n"
    msg += "Return:     +13.8%\n"
    msg += "</pre>\n\n"

    msg += "⚙️ <b>Configuration:</b>\n"
    msg += "├ Activity Threshold: 60\n"
    msg += "├ Min Velocity: 2.0 pips\n"
    msg += "├ Min ATR: 5.0 pips\n"
    msg += "├ Kill Zone: DISABLED\n"
    msg += "├ Trend Filter: ON\n"
    msg += "└ December Skip: ON\n\n"

    msg += "🔄 <b>vs Kill Zone ON:</b>\n"
    msg += "├ +21% more trades\n"
    msg += "├ +10% higher win rate\n"
    msg += "├ 1 fewer losing month\n"
    msg += "└ Similar return\n\n"

    msg += "💡 <b>How it works:</b>\n"
    msg += "Trades when market is MOVING\n"
    msg += "(uses Kalman velocity + ATR)\n"
    msg += "Skips when market is QUIET\n"
    msg += "(no fixed time restrictions)\n\n"

    msg += f"<i>Activated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</i>"

    try:
        bot = Bot(token=config.telegram.bot_token)
        await bot.send_message(chat_id=config.telegram.chat_id, text=msg, parse_mode='HTML')
        print("✅ Notification sent to Telegram!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
