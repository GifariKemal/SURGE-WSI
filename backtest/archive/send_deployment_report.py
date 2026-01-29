"""Send Zero Losing Months Deployment Report
============================================

Notify about the live system update with Zero Losing Months configuration.

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
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
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return resp.status == 200


async def main():
    print("Sending deployment notification...")

    report1 = """<b>SURGE-WSI LIVE SYSTEM UPDATE</b>
<i>Zero Losing Months Configuration Deployed</i>

<b>BACKTEST RESULTS (13 Months)</b>
<code>
Return:          +37.9%
Final Balance:   $13,789.37
Losing Months:   0 (ZERO!)
Win Rate:        59.4%
Max Drawdown:    8.3%
</code>

<b>KEY CHANGES</b>
<code>
1. Max Lot:      0.75 -> 0.5
2. Max Loss/Trade: NEW 0.8%
3. Min Quality:  60 -> 65
4. Daily Limit:  $150 -> $80
5. December:     SKIP (anomaly)
6. Consec Loss:  3 -> 2
</code>

<b>PROTECTION LAYERS</b>
<code>
* Per-trade loss cap: 0.8%
* Daily loss limit: 0.8%
* Monthly loss stop: 2.0%
* December trading: DISABLED
* Stricter quality: 65+
</code>

<i>{ts}</i>""".format(ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    await send_to_telegram(report1)
    print("Report 1 sent!")

    report2 = """<b>CONFIGURATION DETAILS</b>

<b>Risk Manager:</b>
<code>
max_lot_size: 0.5
max_loss_per_trade_pct: 0.8
daily_loss_limit: $80
monthly_loss_stop_pct: 2.0
min_sl_pips: 15
max_sl_pips: 50
</code>

<b>Adaptive Risk:</b>
<code>
base_max_lot: 0.5
consecutive_loss_threshold: 2
drawdown_threshold: 8%
december_max_lot: 0.01 (skip)
</code>

<b>Entry Trigger:</b>
<code>
min_quality_score: 65
relaxed_filter: DISABLED
require_full_confirmation: TRUE
</code>

<b>STATUS: LIVE</b>
<i>System ready for trading</i>

<b>SURGE-WSI Zero-Loss Mode</b>
<i>"Better to miss trades than lose months"</i>"""

    await send_to_telegram(report2)
    print("Report 2 sent!")

    print("\nDeployment notification complete!")


if __name__ == "__main__":
    asyncio.run(main())
