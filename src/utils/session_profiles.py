"""Session Profiles v1 - Session-Specific Trading Parameters
==============================================================

Each trading session (London, New York, Overlap) has different characteristics.
This module provides optimized parameters for each session.

Research basis:
- NBER: "Intra-day Seasonality in FX Markets" - U-shaped activity in London/Tokyo
- Market microstructure: Spread narrower during high activity
- Volatility patterns differ by session

Author: SURIOTA Team
"""
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Optional, Tuple
from enum import Enum
from loguru import logger


class TradingSession(Enum):
    """Trading sessions"""
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"  # London + NY overlap - best time
    ASIA = "asia"
    OFF_HOURS = "off_hours"


@dataclass
class SessionParameters:
    """Trading parameters for a specific session"""
    session: TradingSession
    name: str

    # Entry parameters
    min_poi_quality: float      # Minimum POI quality score (0-100)
    min_confluence_score: float # Minimum confluence score (0-100)

    # Risk parameters
    risk_multiplier: float      # Multiply base risk (1.0 = normal)
    max_sl_pips: float          # Maximum stop loss
    min_sl_pips: float          # Minimum stop loss

    # Market condition thresholds
    chop_threshold: float       # Choppiness threshold (skip if above)
    adx_threshold: float        # ADX threshold (skip if below)

    # Behavior
    allow_trading: bool         # Whether trading is allowed
    prefer_momentum: bool       # Prefer momentum entries vs reversals
    tp_rr: float               # Take profit R:R ratio

    # Session-specific notes
    notes: str = ""

    def __str__(self):
        status = "✅" if self.allow_trading else "🚫"
        return (
            f"{status} {self.name}: risk={self.risk_multiplier:.1f}x, "
            f"quality>={self.min_poi_quality:.0f}, RR={self.tp_rr}"
        )


# ============================================================================
# OPTIMIZED SESSION PROFILES (Based on backtest analysis)
# ============================================================================

LONDON_PROFILE = SessionParameters(
    session=TradingSession.LONDON,
    name="London Session",

    # London: High liquidity, clear trends
    min_poi_quality=45,         # Standard quality
    min_confluence_score=45,

    # Risk: Normal
    risk_multiplier=1.0,
    max_sl_pips=40,
    min_sl_pips=15,

    # More lenient - high activity
    chop_threshold=68,          # Allow slightly choppier
    adx_threshold=16,           # Lower ADX ok due to activity

    allow_trading=True,
    prefer_momentum=True,       # Trends form well
    tp_rr=1.5,

    notes="Best for momentum plays. High liquidity reduces slippage."
)

NEW_YORK_PROFILE = SessionParameters(
    session=TradingSession.NEW_YORK,
    name="New York Session",

    # NY: Volatile, news-driven
    min_poi_quality=50,         # Slightly higher quality
    min_confluence_score=50,

    # Risk: Slightly reduced due to news volatility
    risk_multiplier=0.9,
    max_sl_pips=45,             # Allow wider SL for volatility
    min_sl_pips=18,

    # Standard thresholds
    chop_threshold=65,
    adx_threshold=18,

    allow_trading=True,
    prefer_momentum=True,
    tp_rr=1.5,

    notes="Watch for news events. Can be volatile."
)

OVERLAP_PROFILE = SessionParameters(
    session=TradingSession.OVERLAP,
    name="London/NY Overlap",

    # Overlap: BEST time - highest liquidity
    min_poi_quality=40,         # Can be more lenient
    min_confluence_score=40,

    # Risk: Can be slightly higher
    risk_multiplier=1.1,        # 10% extra risk allowed
    max_sl_pips=40,
    min_sl_pips=15,

    # More lenient - best conditions
    chop_threshold=70,          # High liquidity smooths chop
    adx_threshold=15,           # Lower ADX ok

    allow_trading=True,
    prefer_momentum=True,
    tp_rr=1.5,

    notes="BEST trading window. Highest liquidity. Clear moves."
)

ASIA_PROFILE = SessionParameters(
    session=TradingSession.ASIA,
    name="Asian Session",

    # Asia: Low volatility, ranging
    min_poi_quality=60,         # Higher quality required
    min_confluence_score=60,

    # Risk: Reduced
    risk_multiplier=0.7,
    max_sl_pips=30,             # Tighter SL
    min_sl_pips=12,

    # Stricter - low activity
    chop_threshold=55,          # More sensitive to chop
    adx_threshold=22,           # Need stronger trend

    allow_trading=True,         # Allow but cautiously
    prefer_momentum=False,      # Reversals work better
    tp_rr=1.3,                  # Lower TP due to ranging

    notes="Low volatility. Consider mean-reversion strategies."
)

OFF_HOURS_PROFILE = SessionParameters(
    session=TradingSession.OFF_HOURS,
    name="Off Hours",

    # Off hours: Only trade exceptional setups
    min_poi_quality=75,         # Very high quality only
    min_confluence_score=70,

    # Risk: Minimal
    risk_multiplier=0.5,
    max_sl_pips=25,
    min_sl_pips=10,

    # Very strict
    chop_threshold=50,
    adx_threshold=25,

    allow_trading=False,        # Disabled by default
    prefer_momentum=False,
    tp_rr=1.2,

    notes="Low liquidity, wide spreads. Avoid if possible."
)


class SessionProfileManager:
    """
    Manages session profiles and returns optimal parameters
    based on current time.
    """

    def __init__(self, hybrid_mode: bool = True):
        """
        Initialize session profile manager.

        Args:
            hybrid_mode: If True, allow trading outside kill zones
                        with stricter parameters
        """
        self.hybrid_mode = hybrid_mode

        # Session times (UTC)
        self.sessions = {
            # London: 07:00-10:00 UTC (but extends to overlap)
            TradingSession.LONDON: (time(7, 0), time(12, 0)),

            # Overlap: 12:00-16:00 UTC
            TradingSession.OVERLAP: (time(12, 0), time(16, 0)),

            # New York: 12:00-21:00 UTC (full session)
            TradingSession.NEW_YORK: (time(16, 0), time(21, 0)),

            # Asia: 23:00-07:00 UTC
            TradingSession.ASIA: (time(23, 0), time(7, 0)),
        }

        # Profile mapping
        self.profiles = {
            TradingSession.LONDON: LONDON_PROFILE,
            TradingSession.NEW_YORK: NEW_YORK_PROFILE,
            TradingSession.OVERLAP: OVERLAP_PROFILE,
            TradingSession.ASIA: ASIA_PROFILE,
            TradingSession.OFF_HOURS: OFF_HOURS_PROFILE,
        }

        logger.info(f"SessionProfileManager initialized: hybrid_mode={hybrid_mode}")

    def get_current_session(self, dt: datetime = None) -> TradingSession:
        """Get current trading session."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Ensure UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        current_time = dt.time()

        # Check overlap first (highest priority)
        overlap_start, overlap_end = self.sessions[TradingSession.OVERLAP]
        if overlap_start <= current_time < overlap_end:
            return TradingSession.OVERLAP

        # Check London (before overlap)
        london_start, london_end = self.sessions[TradingSession.LONDON]
        if london_start <= current_time < london_end:
            return TradingSession.LONDON

        # Check New York (after overlap)
        ny_start, ny_end = self.sessions[TradingSession.NEW_YORK]
        if ny_start <= current_time < ny_end:
            return TradingSession.NEW_YORK

        # Check Asia (wraps around midnight)
        asia_start, asia_end = self.sessions[TradingSession.ASIA]
        if current_time >= asia_start or current_time < asia_end:
            return TradingSession.ASIA

        return TradingSession.OFF_HOURS

    def get_profile(self, dt: datetime = None) -> SessionParameters:
        """Get trading parameters for current session."""
        session = self.get_current_session(dt)
        return self.profiles[session]

    def get_adjusted_parameters(
        self,
        dt: datetime = None,
        base_risk: float = 0.01,
        is_thursday: bool = None,
        drift_detected: bool = False
    ) -> dict:
        """
        Get fully adjusted trading parameters.

        Returns dict with all parameters adjusted for:
        - Current session
        - Day of week (Thursday caution)
        - Drift detection status
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        profile = self.get_profile(dt)

        # Start with profile values
        params = {
            'session': profile.session.value,
            'session_name': profile.name,
            'allow_trading': profile.allow_trading,

            # Entry thresholds
            'min_poi_quality': profile.min_poi_quality,
            'min_confluence': profile.min_confluence_score,

            # Risk parameters
            'risk_pct': base_risk * profile.risk_multiplier,
            'max_sl_pips': profile.max_sl_pips,
            'min_sl_pips': profile.min_sl_pips,
            'tp_rr': profile.tp_rr,

            # Market condition thresholds
            'chop_threshold': profile.chop_threshold,
            'adx_threshold': profile.adx_threshold,

            # Behavior
            'prefer_momentum': profile.prefer_momentum,
        }

        # Thursday adjustment
        if is_thursday is None:
            is_thursday = dt.weekday() == 3

        if is_thursday:
            params['risk_pct'] *= 0.6  # Reduce risk on Thursday
            params['min_poi_quality'] += 10  # Require higher quality
            params['min_confluence'] += 5
            params['thursday_adjusted'] = True
        else:
            params['thursday_adjusted'] = False

        # Drift adjustment
        if drift_detected:
            params['risk_pct'] *= 0.5  # Halve risk during drift
            params['min_poi_quality'] += 15
            params['min_confluence'] += 10
            params['drift_adjusted'] = True
        else:
            params['drift_adjusted'] = False

        return params

    def is_optimal_trading_time(self, dt: datetime = None) -> Tuple[bool, str]:
        """Check if current time is optimal for trading."""
        session = self.get_current_session(dt)
        profile = self.profiles[session]

        if session == TradingSession.OVERLAP:
            return True, "Optimal: London/NY Overlap - Best liquidity"
        elif session == TradingSession.LONDON:
            return True, "Good: London Session - Strong trends"
        elif session == TradingSession.NEW_YORK:
            return True, "Good: New York Session - Watch for news"
        elif session == TradingSession.ASIA:
            if self.hybrid_mode:
                return True, "Caution: Asian Session - Low volatility"
            return False, "Skip: Asian Session (hybrid mode off)"
        else:
            if self.hybrid_mode:
                return True, "Caution: Off Hours - Strict criteria"
            return False, "Skip: Off Hours (hybrid mode off)"

    def get_session_stats_template(self) -> dict:
        """Get template for tracking session statistics."""
        return {
            session.value: {
                'trades': 0,
                'wins': 0,
                'pnl': 0.0,
                'avg_quality': 0.0
            }
            for session in TradingSession
        }


# Convenience function
def get_session_parameters(dt: datetime = None) -> SessionParameters:
    """Quick function to get current session parameters."""
    manager = SessionProfileManager()
    return manager.get_profile(dt)


if __name__ == "__main__":
    from datetime import timedelta

    print("\n" + "=" * 70)
    print("SESSION PROFILES TEST")
    print("=" * 70)

    manager = SessionProfileManager(hybrid_mode=True)

    # Test different times
    test_times = [
        datetime(2025, 1, 30, 8, 0, tzinfo=timezone.utc),   # London
        datetime(2025, 1, 30, 13, 0, tzinfo=timezone.utc),  # Overlap
        datetime(2025, 1, 30, 18, 0, tzinfo=timezone.utc),  # NY
        datetime(2025, 1, 30, 4, 0, tzinfo=timezone.utc),   # Asia
        datetime(2025, 1, 30, 22, 0, tzinfo=timezone.utc),  # Off hours
    ]

    for dt in test_times:
        session = manager.get_current_session(dt)
        profile = manager.get_profile(dt)
        optimal, reason = manager.is_optimal_trading_time(dt)

        print(f"\n{dt.strftime('%H:%M UTC')}:")
        print(f"  Session: {session.value}")
        print(f"  Profile: {profile}")
        print(f"  Optimal: {'✅' if optimal else '❌'} - {reason}")

    # Test Thursday adjustment
    print("\n" + "-" * 50)
    print("Thursday Adjustment Test:")
    thursday = datetime(2025, 1, 30, 13, 0, tzinfo=timezone.utc)  # Thursday overlap
    params = manager.get_adjusted_parameters(thursday, base_risk=0.01)
    print(f"  Risk: {params['risk_pct']:.3f} (base 0.01 * session * thursday)")
    print(f"  Min POI Quality: {params['min_poi_quality']}")
    print(f"  Thursday adjusted: {params['thursday_adjusted']}")

    # Test drift adjustment
    print("\n" + "-" * 50)
    print("Drift Adjustment Test:")
    params = manager.get_adjusted_parameters(thursday, base_risk=0.01, drift_detected=True)
    print(f"  Risk: {params['risk_pct']:.3f} (with drift)")
    print(f"  Min POI Quality: {params['min_poi_quality']}")
    print(f"  Drift adjusted: {params['drift_adjusted']}")

    print("\n" + "=" * 70)
