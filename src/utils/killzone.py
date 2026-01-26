"""ICT Kill Zone Detection
=========================

Layer 3 of 6-Layer Architecture

ICT Kill Zones are specific time windows with highest probability
of institutional activity:

- London Kill Zone: 08:00-12:00 UTC
- New York Kill Zone: 13:00-17:00 UTC
- London Close: 15:00-17:00 UTC

Author: SURIOTA Team
"""
from dataclasses import dataclass
from datetime import datetime, timezone, time
from typing import Tuple, Optional
from enum import Enum


class Session(Enum):
    """Trading session types"""
    LONDON = "London"
    NEW_YORK = "New York"
    LONDON_CLOSE = "London Close"
    ASIA = "Asia"
    OFF_SESSION = "Off Session"


@dataclass
class SessionInfo:
    """Session information"""
    session: Session
    in_killzone: bool
    start_time: time
    end_time: time
    time_remaining_minutes: int
    optimal_for_trading: bool
    quality_score: float = 0.0  # 0-100 quality score

    def to_dict(self):
        return {
            'session': self.session.value,
            'in_killzone': self.in_killzone,
            'start_time': self.start_time.strftime('%H:%M'),
            'end_time': self.end_time.strftime('%H:%M'),
            'time_remaining': self.time_remaining_minutes,
            'optimal_for_trading': self.optimal_for_trading,
            'quality_score': self.quality_score,
        }


@dataclass
class SessionQualityScore:
    """Session quality score with breakdown"""
    total_score: float  # 0-100
    session_score: float  # 0-40 based on session type
    day_score: float  # 0-30 based on day of week
    time_score: float  # 0-30 based on time within session
    factors: dict  # Detailed breakdown

    def is_tradeable(self, min_score: float = 50.0) -> bool:
        """Check if session quality meets minimum threshold"""
        return self.total_score >= min_score

    def get_lot_adjustment(self) -> float:
        """Get lot size adjustment based on quality

        Returns:
            Multiplier (0.5 - 1.5)
        """
        if self.total_score >= 80:
            return 1.2  # 20% boost for high quality
        elif self.total_score >= 60:
            return 1.0  # Normal lot
        elif self.total_score >= 40:
            return 0.75  # 25% reduction
        else:
            return 0.5  # 50% reduction

    def to_dict(self):
        return {
            'total_score': round(self.total_score, 1),
            'session_score': round(self.session_score, 1),
            'day_score': round(self.day_score, 1),
            'time_score': round(self.time_score, 1),
            'lot_adjustment': self.get_lot_adjustment(),
            'is_tradeable': self.is_tradeable(),
            'factors': self.factors
        }


class KillZone:
    """ICT Kill Zone detector"""

    def __init__(
        self,
        london_start: int = 8,
        london_end: int = 12,
        new_york_start: int = 13,
        new_york_end: int = 17,
        london_close_start: int = 15,
        london_close_end: int = 17,
        enabled: bool = True
    ):
        """Initialize Kill Zone detector

        Args:
            london_start: London KZ start hour (UTC)
            london_end: London KZ end hour (UTC)
            new_york_start: NY KZ start hour (UTC)
            new_york_end: NY KZ end hour (UTC)
            london_close_start: London close start (UTC)
            london_close_end: London close end (UTC)
            enabled: Enable kill zone filtering
        """
        self.london_start = london_start
        self.london_end = london_end
        self.new_york_start = new_york_start
        self.new_york_end = new_york_end
        self.london_close_start = london_close_start
        self.london_close_end = london_close_end
        self.enabled = enabled

    def is_in_killzone(self, dt: datetime = None) -> Tuple[bool, str]:
        """Check if current time is in a kill zone

        Args:
            dt: Datetime to check (default: now UTC)

        Returns:
            Tuple of (in_killzone: bool, session_name: str)
        """
        if not self.enabled:
            return True, "Kill Zone disabled"

        if dt is None:
            dt = datetime.now(timezone.utc)

        hour = dt.hour

        # London Kill Zone
        if self.london_start <= hour < self.london_end:
            return True, "London"

        # New York Kill Zone
        if self.new_york_start <= hour < self.new_york_end:
            # Check for London Close overlap
            if self.london_close_start <= hour < self.london_close_end:
                return True, "London Close"
            return True, "New York"

        # Asia session (not a kill zone but useful to know)
        if 0 <= hour < 7:
            return False, "Asia"

        return False, "Off Session"

    def get_session_info(self, dt: datetime = None) -> SessionInfo:
        """Get detailed session information

        Args:
            dt: Datetime to check (default: now UTC)

        Returns:
            SessionInfo with details
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        hour = dt.hour
        minute = dt.minute

        # Determine current session
        in_kz, session_name = self.is_in_killzone(dt)

        if session_name == "London":
            session = Session.LONDON
            start = time(self.london_start, 0)
            end = time(self.london_end, 0)
            remaining = (self.london_end - hour) * 60 - minute
        elif session_name == "New York":
            session = Session.NEW_YORK
            start = time(self.new_york_start, 0)
            end = time(self.new_york_end, 0)
            remaining = (self.new_york_end - hour) * 60 - minute
        elif session_name == "London Close":
            session = Session.LONDON_CLOSE
            start = time(self.london_close_start, 0)
            end = time(self.london_close_end, 0)
            remaining = (self.london_close_end - hour) * 60 - minute
        elif session_name == "Asia":
            session = Session.ASIA
            start = time(0, 0)
            end = time(7, 0)
            remaining = (7 - hour) * 60 - minute if hour < 7 else 0
        else:
            session = Session.OFF_SESSION
            start = time(0, 0)
            end = time(0, 0)
            remaining = 0

        # Check if optimal for trading
        # Best: London Close (highest volatility)
        # Good: London, New York
        # Not optimal: Asia, Off Session
        optimal = in_kz and session in [Session.LONDON, Session.NEW_YORK, Session.LONDON_CLOSE]

        return SessionInfo(
            session=session,
            in_killzone=in_kz,
            start_time=start,
            end_time=end,
            time_remaining_minutes=max(0, remaining),
            optimal_for_trading=optimal
        )

    def get_next_killzone(self, dt: datetime = None) -> Tuple[str, int]:
        """Get next kill zone and time until it starts

        Args:
            dt: Current datetime (default: now UTC)

        Returns:
            Tuple of (session_name: str, minutes_until: int)
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        hour = dt.hour
        minute = dt.minute

        # Already in a kill zone
        in_kz, _ = self.is_in_killzone(dt)
        if in_kz:
            return "Currently in Kill Zone", 0

        # Calculate time until each session
        sessions = [
            ("London", self.london_start),
            ("New York", self.new_york_start),
        ]

        next_session = None
        min_minutes = float('inf')

        for name, start_hour in sessions:
            if hour < start_hour:
                minutes = (start_hour - hour) * 60 - minute
            else:
                # Next day
                minutes = (24 - hour + start_hour) * 60 - minute

            if minutes < min_minutes:
                min_minutes = minutes
                next_session = name

        return next_session, int(min_minutes)

    def is_market_open(self, dt: datetime = None) -> Tuple[bool, str]:
        """Check if forex market is open

        Args:
            dt: Datetime to check (default: now UTC)

        Returns:
            Tuple of (is_open: bool, message: str)
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        hour = dt.hour

        # Saturday - always closed
        if weekday == 5:
            hours_until = (24 - hour) + 22
            return False, f"Market CLOSED (Saturday). Opens Sunday 22:00 UTC (~{hours_until}h)"

        # Sunday before 22:00 - closed
        if weekday == 6 and hour < 22:
            hours_until = 22 - hour
            return False, f"Market CLOSED (Sunday). Opens 22:00 UTC (~{hours_until}h)"

        # Friday after 22:00 - closed
        if weekday == 4 and hour >= 22:
            hours_until = (24 - hour) + 24 + 22
            return False, f"Market CLOSED (Weekend). Opens Sunday 22:00 UTC (~{hours_until}h)"

        return True, "Market OPEN"

    def get_trading_recommendation(self, dt: datetime = None) -> str:
        """Get trading recommendation based on time

        Args:
            dt: Datetime to check

        Returns:
            Recommendation string
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        is_open, _ = self.is_market_open(dt)
        if not is_open:
            return "WAIT - Market closed"

        in_kz, session = self.is_in_killzone(dt)

        if session == "London Close":
            return "OPTIMAL - London Close (highest volatility)"
        elif session == "London":
            return "GOOD - London session"
        elif session == "New York":
            return "GOOD - New York session"
        elif session == "Asia":
            return "AVOID - Asia session (low volatility)"
        else:
            return "WAIT - Off session hours"

    def get_session_quality_score(self, dt: datetime = None) -> SessionQualityScore:
        """Calculate comprehensive session quality score

        Scoring breakdown:
        - Session type: 0-40 points
        - Day of week: 0-30 points
        - Time within session: 0-30 points

        Args:
            dt: Datetime to check (default: now UTC)

        Returns:
            SessionQualityScore with detailed breakdown
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        factors = {}
        hour = dt.hour
        minute = dt.minute
        weekday = dt.weekday()

        # ============================
        # SESSION SCORE (0-40 points)
        # ============================
        in_kz, session_name = self.is_in_killzone(dt)

        session_scores = {
            "London Close": 40,  # Best - overlap with high volatility
            "London": 35,
            "New York": 35,
            "Asia": 10,
            "Off Session": 5,
        }

        session_score = session_scores.get(session_name, 5)
        factors['session'] = f"{session_name} ({session_score}/40)"

        # ============================
        # DAY SCORE (0-30 points)
        # ============================
        # Tuesday-Thursday are best days for trading
        # Monday: Start of week, gaps possible
        # Friday: Weekend risk, early close
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_scores = {
            0: 20,  # Monday - OK but gaps
            1: 30,  # Tuesday - Best
            2: 30,  # Wednesday - Best
            3: 30,  # Thursday - Best
            4: 15,  # Friday - Weekend risk
            5: 0,   # Saturday - Market closed
            6: 0,   # Sunday - Market mostly closed
        }

        day_score = day_scores.get(weekday, 0)
        factors['day'] = f"{day_names[weekday]} ({day_score}/30)"

        # ============================
        # TIME SCORE (0-30 points)
        # ============================
        # Score based on position within session
        time_score = 0

        if session_name == "London":
            # Best: 09:00-11:00 (after open volatility settles)
            if 9 <= hour <= 10:
                time_score = 30
            elif hour == 8:
                # Opening hour - more volatile
                time_score = 20
            elif hour == 11:
                # Approaching NY overlap
                time_score = 25
            else:
                time_score = 15

        elif session_name == "New York":
            # Best: 14:00-16:00 (NY overlap with London)
            if 14 <= hour <= 15:
                time_score = 30
            elif hour == 13:
                # Opening hour
                time_score = 20
            elif hour == 16:
                # Near close
                time_score = 20
            else:
                time_score = 15

        elif session_name == "London Close":
            # Best overlap period
            if 15 <= hour < 16:
                time_score = 30
            else:
                time_score = 25

        elif session_name == "Asia":
            # Low volatility - consistent low score
            time_score = 10

        else:
            # Off session
            time_score = 5

        factors['time'] = f"{hour:02d}:{minute:02d} ({time_score}/30)"

        # ============================
        # SPECIAL ADJUSTMENTS
        # ============================

        # Friday afternoon penalty (after 15:00 UTC)
        if weekday == 4 and hour >= 15:
            penalty = 15
            day_score = max(0, day_score - penalty)
            factors['friday_penalty'] = f"-{penalty} (weekend risk)"

        # Month-end adjustment (last 2 days of month)
        if dt.day >= 28:
            # Month-end can have unusual flows
            session_score = max(0, session_score - 5)
            factors['month_end'] = "-5 (month-end flows)"

        # NFP Friday (first Friday of month) - high volatility risk
        if weekday == 4 and dt.day <= 7:
            session_score = max(0, session_score - 10)
            factors['nfp_risk'] = "-10 (potential NFP day)"

        # ============================
        # CALCULATE TOTAL
        # ============================
        total_score = session_score + day_score + time_score
        total_score = max(0, min(100, total_score))

        factors['total'] = f"{total_score}/100"

        return SessionQualityScore(
            total_score=total_score,
            session_score=session_score,
            day_score=day_score,
            time_score=time_score,
            factors=factors
        )

    def get_session_info_with_quality(self, dt: datetime = None) -> SessionInfo:
        """Get session info with quality score included

        Args:
            dt: Datetime to check

        Returns:
            SessionInfo with quality_score populated
        """
        info = self.get_session_info(dt)
        quality = self.get_session_quality_score(dt)
        info.quality_score = quality.total_score
        return info
