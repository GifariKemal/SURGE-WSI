"""Send Risk Configuration Comparison Report to Telegram
======================================================

Sends detailed comparison of risk settings to Telegram.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows encoding for emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import pandas as pd
from datetime import datetime
from config import config


async def send_to_telegram(message: str):
    """Send message to Telegram"""
    import aiohttp

    bot_token = config.telegram.bot_token
    chat_id = config.telegram.chat_id

    if not bot_token or not chat_id:
        print("Telegram not configured")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                print("Message sent to Telegram!")
                return True
            else:
                print(f"Failed to send: {resp.status}")
                return False


async def main():
    """Generate and send risk comparison report"""

    print("=" * 60)
    print("RISK CONFIGURATION COMPARISON REPORT")
    print("=" * 60)

    # Build report
    report = """
<b>📊 RISK CONFIGURATION COMPARISON</b>
<i>SURGE-WSI Backtest Analysis</i>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🔧 CONFIGURATION RESULTS</b>
<i>13-Month Backtest (Jan 2025 - Jan 2026)</i>

<code>Config        MaxLot Return   MaxDD  RiskAdj
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conservative   0.30  +16.0%   6.1%   2.65
Balanced       0.50  +24.3%   9.8%   2.49
Moderate       0.60  +29.8%  11.6%   2.58
Aggressive     0.75  +37.4%  14.1%   2.65
No Cap         5.00  +52.0%  19.5%   2.67</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🤖 ADAPTIVE vs FIXED COMPARISON</b>

<code>Metric          Fixed   Adaptive
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return         +29.6%    +27.0%
Max Drawdown    11.6%     10.4%
Risk-Adjusted    2.56      2.60</code>

✅ <b>Adaptive Benefits:</b>
• Lower drawdown (-1.2%)
• Better risk-adjusted return
• Auto-protection during volatility

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📈 KEY IMPROVEMENTS</b>

<b>December 2025 (Holiday Period):</b>
• Fixed: <code>-$761</code> loss
• Adaptive: <code>-$602</code> loss
• <b>Saved: $159 (21%)</b>

<b>January 2026 (Extreme Vol):</b>
• Fixed: <code>-$152</code> loss
• Adaptive: <code>+$2</code> profit
• <b>Saved: $154 (100%)</b>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>⚙️ CURRENT SETTINGS</b>

<code>adaptive_risk:
  enabled: true
  base_max_lot: 0.6
  low_vol_atr: 8.0 pips
  high_vol_atr: 25.0 pips
  extreme_vol_atr: 40.0 pips
  drawdown_threshold: 10%</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🎯 RECOMMENDATIONS</b>

1️⃣ <b>Conservative Traders:</b>
   MaxLot 0.3 | Return ~16% | DD ~6%

2️⃣ <b>Balanced Traders:</b>
   MaxLot 0.5 | Return ~24% | DD ~10%

3️⃣ <b>Aggressive Traders:</b>
   MaxLot 0.75 | Return ~37% | DD ~14%

<i>💡 Adaptive mode recommended for all profiles</i>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Generated: {timestamp}</i>
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(report)

    # Send to Telegram
    print("\nSending to Telegram...")
    await send_to_telegram(report)

    # Send second message with monthly breakdown
    monthly_report = """
<b>📅 MONTHLY ADAPTIVE ADJUSTMENTS</b>

<code>Month     ATR    Adjustment
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Jan 2025  33.1   HighVol -20%
Feb 2025  29.7   HighVol -20%
Mar 2025  31.3   HighVol -20%
Apr 2025  33.4   HighVol -20%
May 2025  35.0   HighVol -20%
Jun 2025  39.9   HighVol -20%
Jul 2025  37.6   HighVol -20%
Aug 2025  28.8   HighVol -20%
Sep 2025  24.2   Normal   0%
Oct 2025  35.7   HighVol -20%
Nov 2025  29.0   HighVol -20%
Dec 2025  26.1   HighVol+Holiday
Jan 2026  46.3   EXTREME -60%</code>

<i>ATR = Average True Range in pips</i>
<i>Adjustments reduce max lot size</i>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🛡️ PROTECTION TRIGGERS</b>

• <b>3 consecutive losses</b> → -30% lot
• <b>10% drawdown</b> → -50% lot
• <b>Friday 18:00+ UTC</b> → -50% lot
• <b>Dec 15-31</b> → -70% lot
• <b>ATR > 40 pips</b> → -60% lot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>🦅 SURGE-WSI Adaptive Risk System</i>
"""

    print(monthly_report)
    await send_to_telegram(monthly_report)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
