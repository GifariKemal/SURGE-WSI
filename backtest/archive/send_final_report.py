"""Send Final Aggressive+Adaptive Report to Telegram
===================================================

Author: SURIOTA Team
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
from datetime import datetime
from config import config


async def send_to_telegram(message: str):
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
    print("Sending final report to Telegram...")

    report1 = """
<b>🚀 SURGE-WSI FINAL CONFIGURATION</b>
<i>Aggressive + Adaptive Risk Management</i>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 BACKTEST RESULTS (13 Months)</b>
<i>Jan 2025 - Jan 2026</i>

<code>┌─────────────────────────────┐
│ Starting:    $10,000.00     │
│ Final:       $13,533.65     │
│ Profit:      $3,533.65      │
│ Return:      +35.34%        │
└─────────────────────────────┘</code>

<b>📈 STATISTICS</b>
<code>├ Total Trades:    104
├ Win Rate:        54.8%
├ Profit Factor:   5.19
├ Max Drawdown:    13.91%
└ Trades/Month:    8.0</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>⚙️ CONFIGURATION</b>

<code>Mode: Aggressive + Adaptive
Base Max Lot: 0.75
Min SL: 15 pips
Max SL: 50 pips</code>

<b>Adaptive Thresholds (GBPUSD):</b>
<code>├ Low Vol:     ATR < 12 pips
├ Normal:      ATR 12-40 pips
├ High Vol:    ATR 40-55 pips
└ Extreme:     ATR > 55 pips</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🛡️ AUTO-PROTECTION TRIGGERS</b>

<code>├ 3 consecutive losses → -30%
├ 10% drawdown → -50%
├ Friday 18:00+ UTC → -50%
├ Dec 15-31 → -70%
├ ATR > 40 pips → -20%
└ ATR > 55 pips → -60%</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Generated: {timestamp}</i>
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    await send_to_telegram(report1)
    print("Report 1 sent!")

    report2 = """
<b>📊 CONFIG COMPARISON</b>

<code>Config            Return  MaxDD  RiskAdj
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conservative       +16%    6%    2.63
Balanced           +24%   10%    2.49
Aggressive+Adapt   +35%   14%    2.56 ✅
No Cap             +35%   14%    2.49</code>

<b>✅ Aggressive+Adaptive WINS!</b>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>💰 MONTHLY P/L</b>

<code>Month      P/L      Balance
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Jan 2025  +$477    $10,477
Feb 2025   -$66    $10,411
Mar 2025   +$93    $10,504
Apr 2025  +$694    $11,198
May 2025   -$99    $11,099
Jun 2025  +$161    $11,261
Jul 2025 +$1,331   $12,592 🚀
Aug 2025   +$97    $12,689
Sep 2025  +$884    $13,573 🚀
Oct 2025 +$1,027   $14,601 🚀
Nov 2025   +$53    $14,654
Dec 2025  -$954    $13,700 ⚠️
Jan 2026  -$166    $13,534</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🎯 KEY METRICS</b>

<code>├ Profitable Months: 9/13 (69%)
├ Best Month: Jul +$1,331
├ Worst Month: Dec -$954
├ Avg Monthly: +$272
└ Risk-Adjusted: 2.56</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>🦅 SURGE-WSI Trading System</b>
<i>Aggressive + Adaptive Mode ACTIVE</i>
"""

    await send_to_telegram(report2)
    print("Report 2 sent!")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
