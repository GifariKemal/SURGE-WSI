# News Event Filter Integration Guide

## Overview

The News Event Filter (`src/utils/news_filter.py`) prevents trading during high-impact economic events for GBP and USD currencies. This helps avoid unpredictable market movements around major news releases.

## Data Sources

### 1. Hardcoded Recurring Events (Always Available)

The filter includes hardcoded schedules for major recurring events:

| Event | Currency | Schedule | Time (UTC) |
|-------|----------|----------|------------|
| Non-Farm Payrolls (NFP) | USD | First Friday of month | 13:30 |
| FOMC Rate Decision | USD | 8 meetings/year | 18:00-19:00 |
| BOE Rate Decision | GBP | 8 meetings/year | 11:00-12:00 |
| US CPI | USD | Mid-month (~12th) | 13:30 |
| UK GDP | GBP | Mid-month (~13th) | 07:00 |
| US Initial Jobless Claims | USD | Every Thursday | 13:30 |

### 2. JBlanked API (Optional)

Free API with economic calendar data from Forex Factory and other sources.

- **Website**: https://www.jblanked.com/news/api/docs/calendar/
- **Free Tier**: 1 request per day (limited)
- **Endpoints**:
  - `/forex-factory/calendar/today/`
  - `/forex-factory/calendar/week/`
  - `/forex-factory/calendar/range/?from=YYYY-MM-DD&to=YYYY-MM-DD`
- **Authentication**: API key in header (`Authorization: Api-Key YOUR_KEY`)
- **Rate Limit**: 1 request/second (paid), 1 request/day (free)

### 3. Finnhub API (Limited Free Tier)

- **Website**: https://finnhub.io/docs/api/economic-calendar
- **Note**: Economic calendar endpoint requires paid subscription
- **Free endpoints**: Country data only

### 4. Web Scraping (Not Recommended)

Several open-source scrapers exist but are not production-ready:
- github.com/ehsanrs2/forexfactory-scraper
- github.com/AtaCanYmc/ForexFactoryScrapper

**Issues**: Cloudflare protection, rate limiting, ToS violations.

## Basic Usage

```python
from src.utils.news_filter import NewsEventFilter, get_upcoming_news

# Create filter
filter = NewsEventFilter(
    currencies=['GBP', 'USD'],
    buffer_before=30,    # Stop trading 30 min before event
    buffer_after=15,     # Resume trading 15 min after event
    jblanked_api_key=None  # Optional: Add API key for more events
)

# Initialize (fetches events)
await filter.initialize()

# Check if should skip trading
result = filter.should_skip_trading()
if result.should_skip:
    print(f"SKIP: {result.reason}")
    print(f"Resume at: {result.next_safe_time}")
else:
    print("OK to trade")

# Get upcoming events
events = await filter.get_upcoming_events(hours_ahead=24)
for event in events:
    print(f"{event.time} | {event.currency} | {event.event}")
```

## Integration with H1 v6.4 Executor

### Option 1: Add as Layer 5

Add news filtering as an additional layer in the quad-layer quality filter:

```python
# In src/trading/executor_h1_v6_4_gbpusd.py

from src.utils.news_filter import NewsEventFilter

class H1V64GBPUSDExecutor:
    def __init__(self, ...):
        # ... existing init ...

        # Layer 5: News Event Filter
        self.news_filter = NewsEventFilter(
            currencies=['GBP', 'USD'],
            buffer_before=30,
            buffer_after=15
        )

    async def analyze_market(self, balance: float) -> Optional[TradeSignal]:
        # ... existing checks ...

        # LAYER 5: Check news events
        news_result = self.news_filter.should_skip_trading()
        if news_result.should_skip:
            logger.info(f"[Layer5] Trade blocked: {news_result.reason}")
            return None

        # ... rest of analysis ...
```

### Option 2: Check Before Each Cycle

```python
async def run_cycle(self):
    # Check news before analyzing
    news_result = self.news_filter.should_skip_trading()
    if news_result.should_skip:
        logger.info(f"Skipping cycle - news: {news_result.reason}")
        return

    # ... existing cycle code ...
```

### Option 3: Standalone Pre-Check

```python
from src.utils.news_filter import should_skip_trading_sync

# Quick synchronous check
skip, reason = should_skip_trading_sync(buffer_minutes=30)
if skip:
    print(f"Skip trading: {reason}")
```

## Configuration Options

```python
filter = NewsEventFilter(
    # Currencies to monitor
    currencies=['GBP', 'USD'],

    # Buffer times (minutes)
    buffer_before=30,  # Stop trading X min before event
    buffer_after=15,   # Resume trading X min after event

    # Optional: JBlanked API key for more events
    jblanked_api_key="your-api-key",

    # Optional: Enable web scraping fallback
    use_scraping=False,

    # Impact levels to consider (default: HIGH only)
    impact_filter=[NewsImpact.HIGH, NewsImpact.MEDIUM]
)
```

## Telegram Commands Integration

Add to `TelegramNotifier`:

```python
# In telegram handler
self.on_news: Optional[Callable] = None

async def _handle_news(self, update, context):
    if self.on_news:
        result = await self.on_news()
        await update.message.reply_text(result, parse_mode='HTML')
```

Add to executor:

```python
async def _handle_news_command(self) -> str:
    events = await self.news_filter.get_upcoming_events(hours_ahead=72)

    msg = "📰 <b>Upcoming News Events</b>\n\n"
    for event in events[:10]:
        time_str = event.time.strftime('%a %H:%M')
        msg += f"├ {time_str} | {event.currency} | {event.event}\n"

    result = self.news_filter.should_skip_trading()
    msg += f"\n<b>Status:</b> {'⚠️ SKIP' if result.should_skip else '✅ OK'}\n"
    if result.reason:
        msg += f"<i>{result.reason}</i>"

    return msg
```

## API Documentation Summary

### JBlanked News API

| Endpoint | Description | Rate Limit |
|----------|-------------|------------|
| `/today/` | Today's events | 1/sec (paid) |
| `/week/` | This week's events | 1/sec (paid) |
| `/range/?from=&to=` | Date range events | 1/sec (paid) |

**Response Format**:
```json
[
  {
    "date": "2025-02-07T13:30:00Z",
    "currency": "USD",
    "name": "Non-Farm Payrolls",
    "impact": "High",
    "forecast": "180K",
    "previous": "175K",
    "actual": null
  }
]
```

### Alternative Free Options

1. **Trading Economics**: No free API tier
2. **Finnhub**: Economic calendar requires paid subscription
3. **FXStreet**: API documentation available but pricing unclear

## Best Practices

1. **Use hardcoded events for critical dates** (NFP, FOMC, BOE)
2. **Cache API responses** to minimize requests
3. **Graceful degradation**: If API fails, fallback to hardcoded
4. **Buffer times**: 30 min before, 15 min after is recommended
5. **Log all skipped trades** for analysis
6. **Periodic refresh**: Update events cache every 30-60 minutes

## Troubleshooting

### No Events Showing

1. Check date range - hardcoded events may be in the future
2. Verify currency filter includes your pairs
3. Check impact filter settings

### API Rate Limiting

1. JBlanked free tier is very limited (1 request/day)
2. Consider paid subscription or rely on hardcoded events
3. Use longer cache durations

### Missing Events

1. Hardcoded events cover major recurring items only
2. Ad-hoc events (speeches, emergency meetings) need API or manual addition
3. Consider web scraping for comprehensive coverage

## Sources

- [JBlanked News API Documentation](https://www.jblanked.com/news/api/docs/calendar/)
- [Finnhub Economic Calendar](https://finnhub.io/docs/api/economic-calendar)
- [Federal Reserve FOMC Calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- [Forex Factory Calendar](https://www.forexfactory.com/calendar/)
