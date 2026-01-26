"""Exit Manager - Partial TP Strategy
=====================================

Layer 6 of 6-Layer Architecture

Implements partial take profit strategy:
- TP1 (1:1 RR): Close 50%, move SL to breakeven
- TP2 (2:1 RR): Close 30%
- TP3 (3:1 RR): Close remaining 20% or trail

Features:
- Breakeven management
- Trailing stop after TP1
- Dynamic exit based on market structure

Author: SURIOTA Team
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
from datetime import datetime, timezone
from loguru import logger


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TP3_HIT = "tp3_hit"
    SL_HIT = "sl_hit"
    CLOSED = "closed"
    TRAILING = "trailing"


@dataclass
class TPLevel:
    """Take Profit level configuration"""
    rr_ratio: float
    close_percent: float
    price: float = 0.0
    hit: bool = False
    hit_time: datetime = None
    volume_closed: float = 0.0


@dataclass
class PositionState:
    """Managed position state"""
    ticket: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    original_sl: float
    current_sl: float
    original_volume: float
    current_volume: float

    # TP Levels
    tp1: TPLevel = None
    tp2: TPLevel = None
    tp3: TPLevel = None

    # State
    status: PositionStatus = PositionStatus.OPEN
    breakeven_set: bool = False
    trailing_active: bool = False
    trailing_distance: float = 0.0

    # Tracking
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    current_profit_pips: float = 0.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    @property
    def sl_pips(self) -> float:
        """SL distance in pips"""
        return abs(self.entry_price - self.original_sl) / 0.0001

    @property
    def current_sl_pips(self) -> float:
        """Current SL distance in pips"""
        return abs(self.entry_price - self.current_sl) / 0.0001

    @property
    def is_in_profit(self) -> bool:
        """Check if position is in profit"""
        return self.current_profit_pips > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'ticket': self.ticket,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'current_sl': self.current_sl,
            'original_volume': self.original_volume,
            'current_volume': self.current_volume,
            'status': self.status.value,
            'breakeven_set': self.breakeven_set,
            'trailing_active': self.trailing_active,
            'profit_pips': self.current_profit_pips,
            'tp1_hit': self.tp1.hit if self.tp1 else False,
            'tp2_hit': self.tp2.hit if self.tp2 else False,
            'tp3_hit': self.tp3.hit if self.tp3 else False,
        }


class ExitManager:
    """Manages position exits with partial TP strategy"""

    def __init__(
        self,
        tp1_rr: float = 1.0,
        tp1_percent: float = 0.5,
        tp2_rr: float = 2.0,
        tp2_percent: float = 0.3,
        tp3_rr: float = 3.0,
        tp3_percent: float = 0.2,
        trailing_enabled: bool = True,
        trailing_start_rr: float = 1.5,
        trailing_step_pips: float = 10.0,
        trailing_atr_multiplier: float = 1.5,
        use_atr_trailing: bool = True,
        move_sl_to_be_at_tp1: bool = True,
        trail_after_tp1: bool = False
    ):
        """Initialize Exit Manager

        Args:
            tp1_rr: R:R ratio for TP1
            tp1_percent: Percentage to close at TP1
            tp2_rr: R:R ratio for TP2
            tp2_percent: Percentage to close at TP2
            tp3_rr: R:R ratio for TP3
            tp3_percent: Percentage to close at TP3
            trailing_enabled: Enable trailing stop
            trailing_start_rr: Start trailing after this R:R
            trailing_step_pips: Fixed trailing step size (fallback)
            trailing_atr_multiplier: ATR multiplier for dynamic trailing
            use_atr_trailing: Use ATR-based trailing (True) or fixed pips (False)
            move_sl_to_be_at_tp1: Move SL to breakeven at TP1
            trail_after_tp1: Start trailing after TP1 (otherwise after TP2)
        """
        self.tp1_rr = tp1_rr
        self.tp1_percent = tp1_percent
        self.tp2_rr = tp2_rr
        self.tp2_percent = tp2_percent
        self.tp3_rr = tp3_rr
        self.tp3_percent = tp3_percent
        self.trailing_enabled = trailing_enabled
        self.trailing_start_rr = trailing_start_rr
        self.trailing_step_pips = trailing_step_pips
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.use_atr_trailing = use_atr_trailing
        self.move_sl_to_be_at_tp1 = move_sl_to_be_at_tp1
        self.trail_after_tp1 = trail_after_tp1

        # Current ATR value (updated externally)
        self._current_atr_pips: float = 20.0  # Default fallback

        # Managed positions
        self._positions: Dict[int, PositionState] = {}

    def set_current_atr(self, atr_pips: float):
        """Update current ATR for dynamic trailing

        Args:
            atr_pips: Current ATR in pips
        """
        if atr_pips > 0:
            self._current_atr_pips = atr_pips
            logger.debug(f"ATR updated: {atr_pips:.1f} pips")

    def get_trailing_distance_pips(self) -> float:
        """Get trailing distance based on ATR or fixed pips

        Returns:
            Trailing distance in pips
        """
        if self.use_atr_trailing:
            # ATR-based: Use ATR * multiplier
            # Minimum 10 pips, maximum 50 pips for safety
            atr_distance = self._current_atr_pips * self.trailing_atr_multiplier
            return max(10.0, min(50.0, atr_distance))
        else:
            return self.trailing_step_pips

    def create_position(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        volume: float
    ) -> PositionState:
        """Create managed position with TP levels

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            stop_loss: Stop loss price
            volume: Position volume

        Returns:
            PositionState with calculated TP levels
        """
        sl_distance = abs(entry_price - stop_loss)

        # Calculate TP levels
        if direction == 'BUY':
            tp1_price = entry_price + (sl_distance * self.tp1_rr)
            tp2_price = entry_price + (sl_distance * self.tp2_rr)
            tp3_price = entry_price + (sl_distance * self.tp3_rr)
        else:
            tp1_price = entry_price - (sl_distance * self.tp1_rr)
            tp2_price = entry_price - (sl_distance * self.tp2_rr)
            tp3_price = entry_price - (sl_distance * self.tp3_rr)

        position = PositionState(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            original_sl=stop_loss,
            current_sl=stop_loss,
            original_volume=volume,
            current_volume=volume,
            tp1=TPLevel(rr_ratio=self.tp1_rr, close_percent=self.tp1_percent, price=tp1_price),
            tp2=TPLevel(rr_ratio=self.tp2_rr, close_percent=self.tp2_percent, price=tp2_price),
            tp3=TPLevel(rr_ratio=self.tp3_rr, close_percent=self.tp3_percent, price=tp3_price),
        )

        self._positions[ticket] = position
        logger.info(
            f"Position created: {ticket} {direction} @ {entry_price:.5f}, "
            f"TP1={tp1_price:.5f}, TP2={tp2_price:.5f}, TP3={tp3_price:.5f}"
        )

        return position

    def update_position(
        self,
        ticket: int,
        current_price: float
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Update position state based on current price

        Args:
            ticket: Position ticket
            current_price: Current market price

        Returns:
            Tuple of (action: str, details: Dict) or (None, None)
            Actions: 'CLOSE_PARTIAL_TP1', 'CLOSE_PARTIAL_TP2', 'CLOSE_ALL_TP3',
                    'MOVE_SL_BE', 'UPDATE_TRAILING', 'CLOSE_SL'
        """
        if ticket not in self._positions:
            return None, None

        pos = self._positions[ticket]

        # Calculate current profit in pips
        if pos.direction == 'BUY':
            pos.current_profit_pips = (current_price - pos.entry_price) / 0.0001
        else:
            pos.current_profit_pips = (pos.entry_price - current_price) / 0.0001

        # Update MFE/MAE
        pos.max_favorable_excursion = max(pos.max_favorable_excursion, pos.current_profit_pips)
        pos.max_adverse_excursion = min(pos.max_adverse_excursion, pos.current_profit_pips)

        # Check for SL hit
        sl_hit = self._check_sl_hit(pos, current_price)
        if sl_hit:
            pos.status = PositionStatus.SL_HIT
            return 'CLOSE_ALL_SL', {'ticket': ticket, 'price': current_price}

        # Check TP levels
        action, details = self._check_tp_levels(pos, current_price)
        if action:
            return action, details

        # Check trailing stop
        if pos.trailing_active and self.trailing_enabled:
            action, details = self._update_trailing(pos, current_price)
            if action:
                return action, details

        return None, None

    def _check_sl_hit(self, pos: PositionState, current_price: float) -> bool:
        """Check if SL is hit"""
        if pos.direction == 'BUY':
            return current_price <= pos.current_sl
        else:
            return current_price >= pos.current_sl

    def _check_tp_levels(
        self,
        pos: PositionState,
        current_price: float
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Check and handle TP level hits"""

        # TP1 check
        if not pos.tp1.hit:
            tp1_hit = (pos.direction == 'BUY' and current_price >= pos.tp1.price) or \
                      (pos.direction == 'SELL' and current_price <= pos.tp1.price)

            if tp1_hit:
                pos.tp1.hit = True
                pos.tp1.hit_time = datetime.now(timezone.utc)
                pos.status = PositionStatus.TP1_HIT

                # Calculate volume to close
                close_volume = round(pos.original_volume * self.tp1_percent, 2)
                close_volume = max(0.01, close_volume)
                pos.tp1.volume_closed = close_volume

                # Start trailing after TP1 if enabled
                start_trailing = self.trailing_enabled and self.trail_after_tp1
                if start_trailing:
                    pos.trailing_active = True
                    pos.trailing_distance = self.get_trailing_distance_pips() * 0.0001
                    logger.info(f"Trailing started after TP1 (ATR-based: {self.use_atr_trailing})")

                logger.info(f"TP1 hit for {pos.ticket} @ {current_price:.5f}")

                # Prepare actions: close partial + move to BE
                actions = {
                    'ticket': pos.ticket,
                    'close_volume': close_volume,
                    'price': current_price,
                    'move_sl_to_be': self.move_sl_to_be_at_tp1,
                    'new_sl': pos.entry_price if self.move_sl_to_be_at_tp1 else None,
                    'start_trailing': start_trailing
                }

                return 'CLOSE_PARTIAL_TP1', actions

        # TP2 check
        if pos.tp1.hit and not pos.tp2.hit:
            tp2_hit = (pos.direction == 'BUY' and current_price >= pos.tp2.price) or \
                      (pos.direction == 'SELL' and current_price <= pos.tp2.price)

            if tp2_hit:
                pos.tp2.hit = True
                pos.tp2.hit_time = datetime.now(timezone.utc)
                pos.status = PositionStatus.TP2_HIT

                close_volume = round(pos.original_volume * self.tp2_percent, 2)
                close_volume = max(0.01, close_volume)
                pos.tp2.volume_closed = close_volume

                # Start trailing if enabled
                if self.trailing_enabled:
                    pos.trailing_active = True
                    pos.trailing_distance = self.trailing_step_pips * 0.0001

                logger.info(f"TP2 hit for {pos.ticket} @ {current_price:.5f}")

                return 'CLOSE_PARTIAL_TP2', {
                    'ticket': pos.ticket,
                    'close_volume': close_volume,
                    'price': current_price,
                    'start_trailing': self.trailing_enabled
                }

        # TP3 check
        if pos.tp2.hit and not pos.tp3.hit:
            tp3_hit = (pos.direction == 'BUY' and current_price >= pos.tp3.price) or \
                      (pos.direction == 'SELL' and current_price <= pos.tp3.price)

            if tp3_hit:
                pos.tp3.hit = True
                pos.tp3.hit_time = datetime.now(timezone.utc)
                pos.status = PositionStatus.TP3_HIT

                logger.info(f"TP3 hit for {pos.ticket} @ {current_price:.5f}")

                return 'CLOSE_ALL_TP3', {
                    'ticket': pos.ticket,
                    'price': current_price
                }

        return None, None

    def _update_trailing(
        self,
        pos: PositionState,
        current_price: float
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Update trailing stop with ATR-based distance

        Uses dynamic trailing distance based on current ATR,
        with minimum step requirement to avoid too frequent updates.
        """
        if not pos.trailing_active:
            return None, None

        # Get dynamic trailing distance
        trailing_pips = self.get_trailing_distance_pips()
        trailing_distance = trailing_pips * 0.0001

        # Minimum move required before updating (avoid micro-adjustments)
        min_move_pips = max(5.0, trailing_pips * 0.3)  # At least 30% of trail or 5 pips
        min_move = min_move_pips * 0.0001

        # Calculate new SL based on trailing distance
        if pos.direction == 'BUY':
            new_sl = current_price - trailing_distance

            # Only update if new SL is significantly higher
            if new_sl > pos.current_sl + min_move:
                old_sl = pos.current_sl
                pos.current_sl = new_sl
                pos.trailing_distance = trailing_distance

                # Calculate locked profit
                locked_profit_pips = (new_sl - pos.entry_price) / 0.0001
                logger.info(
                    f"Trailing SL updated: {old_sl:.5f} -> {new_sl:.5f} "
                    f"(ATR: {self._current_atr_pips:.1f}, Trail: {trailing_pips:.1f} pips, "
                    f"Locked: {locked_profit_pips:+.1f} pips)"
                )
                return 'UPDATE_TRAILING', {
                    'ticket': pos.ticket,
                    'new_sl': new_sl,
                    'trailing_pips': trailing_pips,
                    'locked_profit_pips': locked_profit_pips
                }
        else:
            new_sl = current_price + trailing_distance

            # Only update if new SL is significantly lower
            if new_sl < pos.current_sl - min_move:
                old_sl = pos.current_sl
                pos.current_sl = new_sl
                pos.trailing_distance = trailing_distance

                # Calculate locked profit
                locked_profit_pips = (pos.entry_price - new_sl) / 0.0001
                logger.info(
                    f"Trailing SL updated: {old_sl:.5f} -> {new_sl:.5f} "
                    f"(ATR: {self._current_atr_pips:.1f}, Trail: {trailing_pips:.1f} pips, "
                    f"Locked: {locked_profit_pips:+.1f} pips)"
                )
                return 'UPDATE_TRAILING', {
                    'ticket': pos.ticket,
                    'new_sl': new_sl,
                    'trailing_pips': trailing_pips,
                    'locked_profit_pips': locked_profit_pips
                }

        return None, None

    def set_breakeven(self, ticket: int, buffer_pips: float = 1.0):
        """Move SL to breakeven (entry + buffer)

        Args:
            ticket: Position ticket
            buffer_pips: Buffer above/below entry
        """
        if ticket not in self._positions:
            return

        pos = self._positions[ticket]
        buffer = buffer_pips * 0.0001

        if pos.direction == 'BUY':
            pos.current_sl = pos.entry_price + buffer
        else:
            pos.current_sl = pos.entry_price - buffer

        pos.breakeven_set = True
        logger.info(f"Position {ticket} moved to breakeven: {pos.current_sl:.5f}")

    def update_volume_after_partial(self, ticket: int, closed_volume: float):
        """Update position volume after partial close

        Args:
            ticket: Position ticket
            closed_volume: Volume that was closed
        """
        if ticket not in self._positions:
            return

        pos = self._positions[ticket]
        pos.current_volume = max(0, pos.current_volume - closed_volume)
        logger.debug(f"Position {ticket} volume updated: {pos.current_volume}")

    def close_position(self, ticket: int, result: str = 'MANUAL'):
        """Mark position as closed

        Args:
            ticket: Position ticket
            result: Close reason
        """
        if ticket not in self._positions:
            return

        pos = self._positions[ticket]
        pos.status = PositionStatus.CLOSED
        pos.current_volume = 0

        logger.info(f"Position {ticket} closed: {result}")

    def remove_position(self, ticket: int):
        """Remove position from management"""
        if ticket in self._positions:
            del self._positions[ticket]

    def get_position(self, ticket: int) -> Optional[PositionState]:
        """Get position state"""
        return self._positions.get(ticket)

    def get_all_positions(self) -> List[PositionState]:
        """Get all managed positions"""
        return list(self._positions.values())

    def get_tp_levels(self, ticket: int) -> Optional[Dict]:
        """Get TP levels for position

        Args:
            ticket: Position ticket

        Returns:
            Dict with TP level details
        """
        if ticket not in self._positions:
            return None

        pos = self._positions[ticket]
        return {
            'tp1': {
                'price': pos.tp1.price,
                'rr': pos.tp1.rr_ratio,
                'percent': pos.tp1.close_percent,
                'hit': pos.tp1.hit
            },
            'tp2': {
                'price': pos.tp2.price,
                'rr': pos.tp2.rr_ratio,
                'percent': pos.tp2.close_percent,
                'hit': pos.tp2.hit
            },
            'tp3': {
                'price': pos.tp3.price,
                'rr': pos.tp3.rr_ratio,
                'percent': pos.tp3.close_percent,
                'hit': pos.tp3.hit
            },
            'current_sl': pos.current_sl,
            'breakeven_set': pos.breakeven_set,
            'trailing_active': pos.trailing_active
        }

    def reset(self):
        """Clear all managed positions"""
        self._positions.clear()
