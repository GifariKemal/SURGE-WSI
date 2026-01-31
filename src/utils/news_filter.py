"""News Event Filter for Forex Trading
======================================

Avoid trading during high-impact economic events for GBP and USD.

This module provides multiple strategies for getting economic calendar data:
1. API-based: JBlanked API (free tier), Finnhub (limited free)
2. Web scraping: Forex Factory scraper (fallback)
3. Hardcoded schedules: NFP, FOMC, BOE, and other recurring events

The filter checks upcoming news events and determines if trading
should be paused based on configurable buffer times.

Author: SURIOTA Team
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from loguru import logger


class NewsImpact(Enum):
    """News impact level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NewsEvent:
    """Economic news event"""
    time: datetime
    currency: str
    event: str
    impact: NewsImpact
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            'time': self.time.isoformat(),
            'currency': self.currency,
            'event': self.event,
            'impact': self.impact.value,
            'forecast': self.forecast,
            'previous': self.previous,
            'actual': self.actual,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NewsEvent':
        return cls(
            time=datetime.fromisoformat(data['time']),
            currency=data['currency'],
            event=data['event'],
            impact=NewsImpact(data['impact']),
            forecast=data.get('forecast'),
            previous=data.get('previous'),
            actual=data.get('actual'),
            source=data.get('source', 'unknown')
        )


@dataclass
class NewsFilterResult:
    """Result of news filter check"""
    should_skip: bool
    reason: str
    upcoming_events: List[NewsEvent] = field(default_factory=list)
    next_safe_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            'should_skip': self.should_skip,
            'reason': self.reason,
            'upcoming_events': [e.to_dict() for e in self.upcoming_events],
            'next_safe_time': self.next_safe_time.isoformat() if self.next_safe_time else None
        }


# ============================================================
# RECURRING EVENT SCHEDULES
# Known high-impact events with predictable schedules
# ============================================================

# NFP (Non-Farm Payrolls) - First Friday of month, 13:30 UTC
# FOMC - 8 meetings per year, dates published in advance
# BOE - 8 meetings per year
# UK GDP - Monthly release

# 2025-2026 FOMC Meeting Dates (from Federal Reserve)
FOMC_DATES_2025 = [
    datetime(2025, 1, 28, 19, 0, tzinfo=timezone.utc),   # Jan 28-29
    datetime(2025, 3, 18, 18, 0, tzinfo=timezone.utc),   # Mar 18-19
    datetime(2025, 5, 6, 18, 0, tzinfo=timezone.utc),    # May 6-7
    datetime(2025, 6, 17, 18, 0, tzinfo=timezone.utc),   # Jun 17-18
    datetime(2025, 7, 29, 18, 0, tzinfo=timezone.utc),   # Jul 29-30
    datetime(2025, 9, 16, 18, 0, tzinfo=timezone.utc),   # Sep 16-17
    datetime(2025, 11, 4, 18, 0, tzinfo=timezone.utc),   # Nov 4-5
    datetime(2025, 12, 16, 19, 0, tzinfo=timezone.utc),  # Dec 16-17
]

FOMC_DATES_2026 = [
    datetime(2026, 1, 27, 19, 0, tzinfo=timezone.utc),   # Jan 27-28
    datetime(2026, 3, 17, 18, 0, tzinfo=timezone.utc),   # Mar 17-18
    datetime(2026, 4, 28, 18, 0, tzinfo=timezone.utc),   # Apr 28-29
    datetime(2026, 6, 16, 18, 0, tzinfo=timezone.utc),   # Jun 16-17
    datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),   # Jul 28-29
    datetime(2026, 9, 15, 18, 0, tzinfo=timezone.utc),   # Sep 15-16
    datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc),   # Nov 3-4
    datetime(2026, 12, 15, 19, 0, tzinfo=timezone.utc),  # Dec 15-16
]

# BOE (Bank of England) Meeting Dates - 8 per year
BOE_DATES_2025 = [
    datetime(2025, 2, 6, 12, 0, tzinfo=timezone.utc),
    datetime(2025, 3, 20, 12, 0, tzinfo=timezone.utc),
    datetime(2025, 5, 8, 11, 0, tzinfo=timezone.utc),
    datetime(2025, 6, 19, 11, 0, tzinfo=timezone.utc),
    datetime(2025, 8, 7, 11, 0, tzinfo=timezone.utc),
    datetime(2025, 9, 18, 11, 0, tzinfo=timezone.utc),
    datetime(2025, 11, 6, 12, 0, tzinfo=timezone.utc),
    datetime(2025, 12, 18, 12, 0, tzinfo=timezone.utc),
]

BOE_DATES_2026 = [
    datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc),
    datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    datetime(2026, 5, 7, 11, 0, tzinfo=timezone.utc),
    datetime(2026, 6, 18, 11, 0, tzinfo=timezone.utc),
    datetime(2026, 8, 6, 11, 0, tzinfo=timezone.utc),
    datetime(2026, 9, 17, 11, 0, tzinfo=timezone.utc),
    datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc),
    datetime(2026, 12, 17, 12, 0, tzinfo=timezone.utc),
]


def get_nfp_date(year: int, month: int) -> datetime:
    """
    Calculate NFP release date (first Friday of month at 13:30 UTC)

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        datetime of NFP release
    """
    # Find first day of month
    first_day = datetime(year, month, 1, tzinfo=timezone.utc)

    # Find first Friday (weekday 4)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)

    # NFP released at 13:30 UTC (8:30 AM ET)
    return first_friday.replace(hour=13, minute=30)


def get_recurring_events(
    start_date: datetime,
    end_date: datetime,
    currencies: List[str] = None
) -> List[NewsEvent]:
    """
    Get hardcoded recurring high-impact events

    Args:
        start_date: Start of date range
        end_date: End of date range
        currencies: Filter by currencies (default: ['GBP', 'USD'])

    Returns:
        List of NewsEvent objects
    """
    currencies = currencies or ['GBP', 'USD']
    events = []

    # Ensure UTC timezone
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # NFP events (USD)
    if 'USD' in currencies:
        current = start_date.replace(day=1)
        while current <= end_date:
            nfp_date = get_nfp_date(current.year, current.month)
            if start_date <= nfp_date <= end_date:
                events.append(NewsEvent(
                    time=nfp_date,
                    currency='USD',
                    event='Non-Farm Payrolls',
                    impact=NewsImpact.HIGH,
                    source='hardcoded'
                ))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    # FOMC events (USD)
    if 'USD' in currencies:
        all_fomc = FOMC_DATES_2025 + FOMC_DATES_2026
        for fomc_date in all_fomc:
            if start_date <= fomc_date <= end_date:
                events.append(NewsEvent(
                    time=fomc_date,
                    currency='USD',
                    event='FOMC Interest Rate Decision',
                    impact=NewsImpact.HIGH,
                    source='hardcoded'
                ))

    # BOE events (GBP)
    if 'GBP' in currencies:
        all_boe = BOE_DATES_2025 + BOE_DATES_2026
        for boe_date in all_boe:
            if start_date <= boe_date <= end_date:
                events.append(NewsEvent(
                    time=boe_date,
                    currency='GBP',
                    event='BOE Interest Rate Decision',
                    impact=NewsImpact.HIGH,
                    source='hardcoded'
                ))

    # US CPI (typically mid-month, around 13:30 UTC)
    # This is approximate - actual dates vary
    if 'USD' in currencies:
        current = start_date.replace(day=1)
        while current <= end_date:
            # Approximate CPI date (usually 10th-15th of month)
            cpi_date = current.replace(day=12, hour=13, minute=30)
            if start_date <= cpi_date <= end_date:
                events.append(NewsEvent(
                    time=cpi_date,
                    currency='USD',
                    event='US CPI (Approximate)',
                    impact=NewsImpact.HIGH,
                    source='hardcoded'
                ))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    # UK GDP (typically mid-month)
    if 'GBP' in currencies:
        current = start_date.replace(day=1)
        while current <= end_date:
            # Approximate GDP date
            gdp_date = current.replace(day=13, hour=7, minute=0)
            if start_date <= gdp_date <= end_date:
                events.append(NewsEvent(
                    time=gdp_date,
                    currency='GBP',
                    event='UK GDP (Approximate)',
                    impact=NewsImpact.HIGH,
                    source='hardcoded'
                ))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    # Weekly: US Initial Jobless Claims (Thursday 13:30 UTC)
    if 'USD' in currencies:
        current = start_date
        while current <= end_date:
            # Find next Thursday
            days_until_thursday = (3 - current.weekday()) % 7
            if days_until_thursday == 0 and current.hour >= 14:
                days_until_thursday = 7
            thursday = current + timedelta(days=days_until_thursday)
            thursday = thursday.replace(hour=13, minute=30, second=0, microsecond=0)

            if start_date <= thursday <= end_date:
                events.append(NewsEvent(
                    time=thursday,
                    currency='USD',
                    event='US Initial Jobless Claims',
                    impact=NewsImpact.MEDIUM,
                    source='hardcoded'
                ))
            current = thursday + timedelta(days=1)

    # Sort by time
    events.sort(key=lambda x: x.time)

    return events


# ============================================================
# API-BASED NEWS FETCHING
# ============================================================

class JBlankedAPIClient:
    """
    JBlanked News API Client

    Free tier: 1 request per day (limited)
    Documentation: https://www.jblanked.com/news/api/docs/calendar/

    Endpoints:
    - /forex-factory/calendar/today/
    - /forex-factory/calendar/week/
    - /forex-factory/calendar/range/?from=YYYY-MM-DD&to=YYYY-MM-DD
    """

    BASE_URL = "https://www.jblanked.com/news/api/forex-factory/calendar"

    def __init__(self, api_key: str = None):
        """
        Initialize JBlanked API client

        Args:
            api_key: API key (generate at jblanked.com profile)
        """
        self.api_key = api_key
        self._last_request_time: Optional[datetime] = None
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}

    def _get_headers(self) -> dict:
        """Get request headers with authentication"""
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'SURGE-WSI/1.0'
        }
        if self.api_key:
            headers['Authorization'] = f'Api-Key {self.api_key}'
        return headers

    async def _fetch(self, endpoint: str, params: dict = None) -> Optional[List[dict]]:
        """
        Fetch data from API

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            List of event dictionaries or None on error
        """
        url = f"{self.BASE_URL}{endpoint}"
        cache_key = f"{endpoint}:{json.dumps(params or {})}"

        # Check cache (5 minute expiry for free tier conservation)
        now = datetime.now(timezone.utc)
        if cache_key in self._cache:
            if self._cache_expiry.get(cache_key, now) > now:
                logger.debug(f"JBlanked cache hit: {endpoint}")
                return self._cache[cache_key]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache for 5 minutes
                        self._cache[cache_key] = data
                        self._cache_expiry[cache_key] = now + timedelta(minutes=5)
                        self._last_request_time = now
                        return data
                    elif response.status == 429:
                        logger.warning("JBlanked API rate limit exceeded")
                        return None
                    else:
                        logger.warning(f"JBlanked API error: {response.status}")
                        return None
        except asyncio.TimeoutError:
            logger.warning("JBlanked API timeout")
            return None
        except Exception as e:
            logger.error(f"JBlanked API error: {e}")
            return None

    async def get_today_events(self) -> List[NewsEvent]:
        """Get today's events"""
        data = await self._fetch("/today/")
        return self._parse_events(data) if data else []

    async def get_week_events(self) -> List[NewsEvent]:
        """Get this week's events"""
        data = await self._fetch("/week/")
        return self._parse_events(data) if data else []

    async def get_range_events(
        self,
        from_date: datetime,
        to_date: datetime
    ) -> List[NewsEvent]:
        """Get events for date range"""
        params = {
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d')
        }
        data = await self._fetch("/range/", params)
        return self._parse_events(data) if data else []

    def _parse_events(self, data: List[dict]) -> List[NewsEvent]:
        """Parse API response into NewsEvent objects"""
        events = []

        for item in data:
            try:
                # Parse datetime from API format
                date_str = item.get('date', '') or item.get('Date', '')
                if not date_str:
                    continue

                # Handle various date formats
                try:
                    event_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except:
                    try:
                        event_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    except:
                        continue

                # Parse impact
                impact_str = str(item.get('impact', '') or item.get('Impact', '')).lower()
                if 'high' in impact_str or impact_str == '3':
                    impact = NewsImpact.HIGH
                elif 'medium' in impact_str or 'med' in impact_str or impact_str == '2':
                    impact = NewsImpact.MEDIUM
                else:
                    impact = NewsImpact.LOW

                events.append(NewsEvent(
                    time=event_time,
                    currency=item.get('currency', '') or item.get('Currency', ''),
                    event=item.get('name', '') or item.get('Name', '') or item.get('event', ''),
                    impact=impact,
                    forecast=item.get('forecast') or item.get('Forecast'),
                    previous=item.get('previous') or item.get('Previous'),
                    actual=item.get('actual') or item.get('Actual'),
                    source='jblanked'
                ))
            except Exception as e:
                logger.debug(f"Failed to parse event: {e}")
                continue

        return events


# ============================================================
# WEB SCRAPING FALLBACK (Forex Factory)
# ============================================================

class ForexFactoryScraper:
    """
    Forex Factory Calendar Scraper (Fallback)

    WARNING: Web scraping may violate ToS. Use API when possible.
    This is provided as a fallback when APIs are unavailable.

    Based on: github.com/pohzipohzi/forexfactory_econcal.py
    """

    BASE_URL = "https://www.forexfactory.com/calendar"

    def __init__(self):
        self._cache: Dict[str, List[NewsEvent]] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)

    async def get_events(
        self,
        start_date: datetime = None,
        currencies: List[str] = None
    ) -> List[NewsEvent]:
        """
        Scrape events from Forex Factory

        Note: This is a stub. Full implementation would require:
        - BeautifulSoup or lxml for HTML parsing
        - Handling dynamic content (JavaScript)
        - Potentially using Selenium or Playwright
        - Respecting robots.txt and rate limits

        For production use, consider:
        - github.com/ehsanrs2/forexfactory-scraper (uses undetected-chromedriver)
        - github.com/AtaCanYmc/ForexFactoryScrapper (uses BeautifulSoup)

        Args:
            start_date: Start date for events
            currencies: Filter currencies

        Returns:
            List of NewsEvent (empty if scraping not implemented)
        """
        logger.warning(
            "Forex Factory scraping not fully implemented. "
            "Consider using JBlanked API or hardcoded schedules."
        )
        return []


# ============================================================
# MAIN NEWS FILTER CLASS
# ============================================================

class NewsEventFilter:
    """
    News Event Filter for Forex Trading

    Provides unified interface for checking upcoming news events
    and determining if trading should be paused.

    Data sources (in priority order):
    1. JBlanked API (if API key provided)
    2. Hardcoded recurring events (always available)
    3. Web scraping fallback (optional, not fully implemented)

    Usage:
        filter = NewsEventFilter(
            currencies=['GBP', 'USD'],
            buffer_before=30,  # 30 min before event
            buffer_after=15    # 15 min after event
        )

        # Initialize
        await filter.initialize()

        # Check if should skip trading
        result = await filter.should_skip_trading()
        if result.should_skip:
            print(f"Skip trading: {result.reason}")
    """

    def __init__(
        self,
        currencies: List[str] = None,
        buffer_before: int = 30,
        buffer_after: int = 15,
        jblanked_api_key: str = None,
        use_scraping: bool = False,
        impact_filter: List[NewsImpact] = None
    ):
        """
        Initialize News Event Filter

        Args:
            currencies: Currencies to filter (default: ['GBP', 'USD'])
            buffer_before: Minutes before event to stop trading
            buffer_after: Minutes after event to resume trading
            jblanked_api_key: JBlanked API key (optional)
            use_scraping: Enable web scraping fallback
            impact_filter: Only consider these impact levels (default: HIGH only)
        """
        self.currencies = currencies or ['GBP', 'USD']
        self.buffer_before = buffer_before
        self.buffer_after = buffer_after
        self.impact_filter = impact_filter or [NewsImpact.HIGH]

        # API clients
        self._jblanked = JBlankedAPIClient(api_key=jblanked_api_key) if jblanked_api_key else None
        self._scraper = ForexFactoryScraper() if use_scraping else None

        # Cache
        self._events_cache: List[NewsEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=30)

        # State
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize filter and fetch initial events

        Returns:
            True if successful
        """
        try:
            await self.refresh_events()
            self._initialized = True
            logger.info(
                f"NewsEventFilter initialized: {len(self._events_cache)} events, "
                f"currencies={self.currencies}, buffer={self.buffer_before}/{self.buffer_after}min"
            )
            return True
        except Exception as e:
            logger.error(f"NewsEventFilter initialization failed: {e}")
            return False

    async def refresh_events(self, force: bool = False) -> int:
        """
        Refresh events cache

        Args:
            force: Force refresh even if cache is valid

        Returns:
            Number of events fetched
        """
        now = datetime.now(timezone.utc)

        # Check cache validity
        if not force and self._last_fetch:
            if now - self._last_fetch < self._cache_duration:
                return len(self._events_cache)

        events = []

        # Fetch from JBlanked API
        if self._jblanked:
            try:
                api_events = await self._jblanked.get_week_events()
                events.extend(api_events)
                logger.debug(f"JBlanked: {len(api_events)} events")
            except Exception as e:
                logger.warning(f"JBlanked fetch failed: {e}")

        # Add hardcoded recurring events
        end_date = now + timedelta(days=7)
        recurring = get_recurring_events(now, end_date, self.currencies)

        # Merge, avoiding duplicates
        existing_keys = {(e.time.date(), e.currency, e.event.lower()[:20]) for e in events}
        for event in recurring:
            key = (event.time.date(), event.currency, event.event.lower()[:20])
            if key not in existing_keys:
                events.append(event)
                existing_keys.add(key)

        logger.debug(f"Recurring: {len(recurring)} events (merged: {len(events)} total)")

        # Filter by currency and impact
        events = [
            e for e in events
            if e.currency.upper() in [c.upper() for c in self.currencies]
            and e.impact in self.impact_filter
        ]

        # Sort by time
        events.sort(key=lambda x: x.time)

        self._events_cache = events
        self._last_fetch = now

        logger.info(f"News events refreshed: {len(events)} high-impact events for {self.currencies}")

        return len(events)

    async def get_upcoming_events(
        self,
        hours_ahead: int = 24
    ) -> List[NewsEvent]:
        """
        Get high-impact news events for the next N hours

        Args:
            hours_ahead: Hours to look ahead

        Returns:
            List of upcoming NewsEvent objects
        """
        # Refresh cache if needed
        await self.refresh_events()

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        return [
            event for event in self._events_cache
            if now <= event.time <= cutoff
        ]

    def should_skip_trading(
        self,
        current_time: datetime = None,
        news_events: List[NewsEvent] = None,
        buffer_before: int = None,
        buffer_after: int = None
    ) -> NewsFilterResult:
        """
        Check if trading should be skipped due to upcoming news

        Args:
            current_time: Time to check (default: now UTC)
            news_events: Events to check (default: cached events)
            buffer_before: Override buffer before event (minutes)
            buffer_after: Override buffer after event (minutes)

        Returns:
            NewsFilterResult with skip decision and reason
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        events = news_events if news_events is not None else self._events_cache
        buffer_mins_before = buffer_before if buffer_before is not None else self.buffer_before
        buffer_mins_after = buffer_after if buffer_after is not None else self.buffer_after

        # Find events within buffer window
        conflicting_events = []

        for event in events:
            event_start = event.time - timedelta(minutes=buffer_mins_before)
            event_end = event.time + timedelta(minutes=buffer_mins_after)

            if event_start <= current_time <= event_end:
                conflicting_events.append(event)

        if conflicting_events:
            # Find next safe time
            latest_event = max(conflicting_events, key=lambda e: e.time)
            next_safe = latest_event.time + timedelta(minutes=buffer_mins_after + 1)

            # Build reason string
            event_names = [f"{e.currency} {e.event}" for e in conflicting_events[:3]]
            if len(conflicting_events) > 3:
                event_names.append(f"and {len(conflicting_events) - 3} more")

            reason = f"High-impact news: {', '.join(event_names)}"

            return NewsFilterResult(
                should_skip=True,
                reason=reason,
                upcoming_events=conflicting_events,
                next_safe_time=next_safe
            )

        # Check for very close upcoming events (within warning threshold)
        warning_threshold = timedelta(minutes=buffer_mins_before + 15)
        upcoming = [
            e for e in events
            if 0 < (e.time - current_time).total_seconds() < warning_threshold.total_seconds()
        ]

        if upcoming:
            return NewsFilterResult(
                should_skip=False,
                reason=f"News approaching: {upcoming[0].currency} {upcoming[0].event} at {upcoming[0].time.strftime('%H:%M')} UTC",
                upcoming_events=upcoming,
                next_safe_time=None
            )

        return NewsFilterResult(
            should_skip=False,
            reason="No high-impact news in trading window",
            upcoming_events=[],
            next_safe_time=None
        )

    async def check_trading_window(
        self,
        hours_ahead: int = 4
    ) -> NewsFilterResult:
        """
        Convenience method: Check if trading is safe for the next N hours

        Args:
            hours_ahead: Hours to check ahead

        Returns:
            NewsFilterResult
        """
        events = await self.get_upcoming_events(hours_ahead)
        return self.should_skip_trading(news_events=events)

    def get_status(self) -> dict:
        """Get filter status"""
        now = datetime.now(timezone.utc)
        upcoming = [e for e in self._events_cache if e.time > now]

        return {
            'initialized': self._initialized,
            'currencies': self.currencies,
            'buffer_before': self.buffer_before,
            'buffer_after': self.buffer_after,
            'cached_events': len(self._events_cache),
            'upcoming_events': len(upcoming),
            'last_fetch': self._last_fetch.isoformat() if self._last_fetch else None,
            'has_api_key': self._jblanked is not None,
            'next_event': upcoming[0].to_dict() if upcoming else None
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

async def get_upcoming_news(
    hours_ahead: int = 24,
    currencies: List[str] = None
) -> List[dict]:
    """
    Get high-impact news events for next N hours

    This is a standalone function that creates a temporary filter.
    For repeated use, create a NewsEventFilter instance instead.

    Args:
        hours_ahead: Hours to look ahead
        currencies: Currencies to filter (default: ['GBP', 'USD'])

    Returns:
        List of event dictionaries
    """
    currencies = currencies or ['GBP', 'USD']

    # Use hardcoded events (always available)
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=hours_ahead)

    events = get_recurring_events(now, end, currencies)

    return [e.to_dict() for e in events]


def should_skip_trading_sync(
    current_time: datetime = None,
    news_events: List[dict] = None,
    buffer_minutes: int = 30
) -> Tuple[bool, str]:
    """
    Synchronous version: Check if we should skip trading due to upcoming news

    Args:
        current_time: Time to check (default: now UTC)
        news_events: List of event dictionaries
        buffer_minutes: Minutes buffer around events

    Returns:
        Tuple of (should_skip: bool, reason: str)
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    # Convert dicts to NewsEvent objects if needed
    if news_events:
        events = [
            NewsEvent.from_dict(e) if isinstance(e, dict) else e
            for e in news_events
        ]
    else:
        # Get hardcoded events
        end = current_time + timedelta(hours=4)
        events = get_recurring_events(current_time, end)

    # Check each event
    for event in events:
        event_start = event.time - timedelta(minutes=buffer_minutes)
        event_end = event.time + timedelta(minutes=buffer_minutes // 2)

        if event_start <= current_time <= event_end:
            return True, f"Near {event.currency} {event.event} at {event.time.strftime('%H:%M')} UTC"

    return False, "No conflicting news events"


# ============================================================
# MAIN (Testing)
# ============================================================

async def main():
    """Test the news filter"""
    print("=" * 60)
    print("NEWS EVENT FILTER TEST")
    print("=" * 60)

    # Test hardcoded events
    now = datetime.now(timezone.utc)
    print(f"\nCurrent time (UTC): {now.strftime('%Y-%m-%d %H:%M')}")

    # Get upcoming events
    events = await get_upcoming_news(hours_ahead=168)  # 1 week
    print(f"\nUpcoming high-impact events (next 7 days):")
    for event in events[:10]:
        print(f"  {event['time'][:16]} | {event['currency']:3s} | {event['event']}")
    if len(events) > 10:
        print(f"  ... and {len(events) - 10} more")

    # Test filter
    print("\n" + "=" * 60)
    print("TESTING FILTER")
    print("=" * 60)

    filter = NewsEventFilter(
        currencies=['GBP', 'USD'],
        buffer_before=30,
        buffer_after=15
    )
    await filter.initialize()

    # Check trading window
    result = await filter.check_trading_window(hours_ahead=4)
    print(f"\nShould skip trading: {result.should_skip}")
    print(f"Reason: {result.reason}")
    if result.upcoming_events:
        print(f"Events: {[e.event for e in result.upcoming_events]}")
    if result.next_safe_time:
        print(f"Next safe time: {result.next_safe_time.strftime('%Y-%m-%d %H:%M')} UTC")

    # Test specific times
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC TIMES")
    print("=" * 60)

    # Test during NFP
    nfp_time = get_nfp_date(now.year, now.month)
    print(f"\nNext NFP: {nfp_time.strftime('%Y-%m-%d %H:%M')} UTC")

    # Test 15 minutes before NFP
    test_time = nfp_time - timedelta(minutes=15)
    result = filter.should_skip_trading(current_time=test_time)
    print(f"15 min before NFP: skip={result.should_skip}, reason={result.reason}")

    # Test 45 minutes before NFP
    test_time = nfp_time - timedelta(minutes=45)
    result = filter.should_skip_trading(current_time=test_time)
    print(f"45 min before NFP: skip={result.should_skip}, reason={result.reason}")

    print("\n" + "=" * 60)
    print("FILTER STATUS")
    print("=" * 60)
    print(json.dumps(filter.get_status(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
