"""MT5 Connector - MetaTrader 5 Integration for SURGE-WSI

This module provides connection and data retrieval functionality
for interacting with MetaTrader 5 terminal.

Supports both direct MT5 API and MCP server integration.

Author: SURIOTA Team
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
import pandas as pd
from loguru import logger


class MT5Connector:
    """MetaTrader 5 Connector with MCP support"""

    # MT5 Timeframe constants mapping
    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }

    # Order type constants
    ORDER_TYPES = {
        "BUY": mt5.ORDER_TYPE_BUY,
        "SELL": mt5.ORDER_TYPE_SELL,
        "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
        "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
        "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
        "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
    }

    def __init__(
        self,
        terminal_path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        use_mcp: bool = False
    ):
        """Initialize MT5 Connector

        Args:
            terminal_path: Path to MT5 terminal executable
            login: MT5 account login
            password: MT5 account password
            server: MT5 server name
            use_mcp: Use MCP server instead of direct API
        """
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server
        self.use_mcp = use_mcp
        self.connected = False
        self._last_error = None

        # MCP callback functions (set externally when use_mcp=True)
        self.mcp_get_account_info = None
        self.mcp_get_symbol_price = None
        self.mcp_get_candles_latest = None
        self.mcp_get_all_positions = None
        self.mcp_place_market_order = None
        self.mcp_modify_position = None
        self.mcp_close_position = None

    def connect(self) -> bool:
        """Initialize MT5 connection

        Returns:
            True if connection successful, False otherwise
        """
        if self.use_mcp:
            # MCP mode - assume connected if callbacks are set
            self.connected = self.mcp_get_account_info is not None
            if self.connected:
                logger.info("MT5 connected via MCP")
            return self.connected

        try:
            # First try connecting without credentials (use already logged-in terminal)
            # This is the recommended approach - let MT5 terminal handle authentication
            if mt5.initialize():
                self.connected = True
                terminal_info = mt5.terminal_info()
                account_info = mt5.account_info()
                logger.info(f"MT5 connected: {terminal_info.name} - Build {terminal_info.build}")
                if account_info:
                    logger.info(f"Account: {account_info.login} ({account_info.server})")
                return True

            # If that failed, try with terminal path only (no credentials)
            if self.terminal_path:
                if mt5.initialize(path=self.terminal_path):
                    self.connected = True
                    terminal_info = mt5.terminal_info()
                    account_info = mt5.account_info()
                    logger.info(f"MT5 connected: {terminal_info.name} - Build {terminal_info.build}")
                    if account_info:
                        logger.info(f"Account: {account_info.login} ({account_info.server})")
                    return True

            # Last resort: try with full credentials
            if self.login and self.password:
                init_args = {"login": self.login, "password": self.password}
                if self.server:
                    init_args["server"] = self.server
                if self.terminal_path:
                    init_args["path"] = self.terminal_path

                if mt5.initialize(**init_args):
                    self.connected = True
                    terminal_info = mt5.terminal_info()
                    logger.info(f"MT5 connected: {terminal_info.name} - Build {terminal_info.build}")
                    return True

            self._last_error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {self._last_error}")
            return False

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            self._last_error = str(e)
            return False

    def disconnect(self) -> None:
        """Shutdown MT5 connection"""
        if self.connected and not self.use_mcp:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def ensure_connected(self) -> bool:
        """Ensure MT5 is connected, reconnect if needed"""
        if not self.connected:
            return self.connect()

        if self.use_mcp:
            return True

        # Verify connection is still alive
        try:
            mt5.terminal_info()
            return True
        except:
            self.connected = False
            return self.connect()

    def is_autotrading_enabled(self) -> bool:
        """Check if AutoTrading is enabled in MT5 terminal

        Returns:
            True if AutoTrading is enabled (or MCP mode), False otherwise
        """
        if self.use_mcp:
            return True

        if not self.ensure_connected():
            return False

        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("Failed to get terminal info for AutoTrading check")
                return False
            return bool(terminal_info.trade_allowed)
        except Exception as e:
            logger.error(f"AutoTrading check failed: {e}")
            return False

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account balance and equity information

        Returns:
            Dict with account info or None if not connected
        """
        if self.use_mcp and self.mcp_get_account_info:
            try:
                return await self.mcp_get_account_info()
            except Exception as e:
                logger.error(f"MCP get_account_info failed: {e}")
                return None

        if not self.ensure_connected():
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "server": info.server,
            "name": info.name,
            "currency": info.currency,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "leverage": info.leverage,
            "trade_mode": info.trade_mode,
        }

    def get_account_info_sync(self) -> Optional[Dict[str, Any]]:
        """Synchronous version of get_account_info"""
        if not self.ensure_connected():
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "server": info.server,
            "name": info.name,
            "currency": info.currency,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "leverage": info.leverage,
            "trade_mode": info.trade_mode,
        }

    async def get_tick(self, symbol: str = "GBPUSD") -> Optional[Dict[str, Any]]:
        """Get latest tick data for symbol

        Args:
            symbol: Trading symbol (default: GBPUSD)

        Returns:
            Dict with tick data or None if error
        """
        if self.use_mcp and self.mcp_get_symbol_price:
            try:
                return await self.mcp_get_symbol_price(symbol)
            except Exception as e:
                logger.error(f"MCP get_symbol_price failed: {e}")
                return None

        if not self.ensure_connected():
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"Failed to get tick for {symbol}")
            return None

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round((tick.ask - tick.bid) / self._get_point(symbol), 1),
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time),
            "time_msc": tick.time_msc,
        }

    def get_tick_sync(self, symbol: str = "GBPUSD") -> Optional[Dict[str, Any]]:
        """Synchronous version of get_tick"""
        if not self.ensure_connected():
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round((tick.ask - tick.bid) / self._get_point(symbol), 1),
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time),
        }

    def _get_point(self, symbol: str) -> float:
        """Get point value for symbol"""
        info = mt5.symbol_info(symbol)
        return info.point if info else 0.0001

    def get_symbol_info(self, symbol: str = "GBPUSD") -> Optional[Dict[str, Any]]:
        """Get symbol information

        Args:
            symbol: Trading symbol (default: GBPUSD)

        Returns:
            Dict with symbol info or None if error
        """
        if not self.ensure_connected():
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Failed to get info for {symbol}")
            return None

        return {
            "symbol": symbol,
            "description": info.description,
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "trade_mode": info.trade_mode,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "contract_size": info.trade_contract_size,
            "swap_long": info.swap_long,
            "swap_short": info.swap_short,
        }

    def get_ohlcv(
        self,
        symbol: str = "GBPUSD",
        timeframe: str = "H1",
        bars: int = 100,
        start_pos: int = 0
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV (candlestick) data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            bars: Number of bars to retrieve
            start_pos: Starting position (0 = current bar)

        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.ensure_connected():
            return None

        tf = self.TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, bars)

        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get OHLCV for {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'spread': 'Spread',
            'real_volume': 'RealVolume'
        }, inplace=True)

        return df

    def get_ohlcv_range(
        self,
        symbol: str = "GBPUSD",
        timeframe: str = "H1",
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV data for date range

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start datetime
            end_date: End datetime (default: now)

        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.ensure_connected():
            return None

        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        tf = self.TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get OHLCV range for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'spread': 'Spread',
            'real_volume': 'RealVolume'
        }, inplace=True)

        return df

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of position dicts
        """
        if self.use_mcp and self.mcp_get_all_positions:
            try:
                positions = await self.mcp_get_all_positions()
                if symbol:
                    positions = [p for p in positions if p.get('symbol') == symbol]
                return positions
            except Exception as e:
                logger.error(f"MCP get_all_positions failed: {e}")
                return []

        if not self.ensure_connected():
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        return [self._position_to_dict(p) for p in positions]

    def get_positions_sync(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Synchronous version of get_positions"""
        if not self.ensure_connected():
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        return [self._position_to_dict(p) for p in positions]

    def _position_to_dict(self, position) -> Dict[str, Any]:
        """Convert MT5 position to dictionary"""
        return {
            "ticket": position.ticket,
            "symbol": position.symbol,
            "type": "BUY" if position.type == 0 else "SELL",
            "volume": position.volume,
            "price_open": position.price_open,
            "price_current": position.price_current,
            "sl": position.sl,
            "tp": position.tp,
            "profit": position.profit,
            "swap": getattr(position, 'swap', 0.0),
            "commission": getattr(position, 'commission', 0.0),
            "time": datetime.fromtimestamp(position.time),
            "magic": position.magic,
            "comment": getattr(position, 'comment', ''),
        }

    async def place_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: float = 0,
        tp: float = 0,
        comment: str = "SURGE-WSI",
        magic: int = 20250125
    ) -> Optional[Dict[str, Any]]:
        """Place market order

        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            volume: Lot size
            sl: Stop loss price (0 = none)
            tp: Take profit price (0 = none)
            comment: Order comment
            magic: Magic number

        Returns:
            Order result dict or None if failed
        """
        if self.use_mcp and self.mcp_place_market_order:
            try:
                result = await self.mcp_place_market_order(
                    symbol=symbol,
                    volume=volume,
                    type=order_type
                )
                # Modify to add SL/TP if specified
                if result and (sl > 0 or tp > 0) and self.mcp_modify_position:
                    await self.mcp_modify_position(
                        id=result.get('ticket'),
                        stop_loss=sl if sl > 0 else None,
                        take_profit=tp if tp > 0 else None
                    )
                return result
            except Exception as e:
                logger.error(f"MCP place_market_order failed: {e}")
                return None

        if not self.ensure_connected():
            return None

        # Check AutoTrading before sending order
        if not self.is_autotrading_enabled():
            logger.error(
                "AutoTrading is DISABLED in MT5 terminal. "
                "Enable it via Tools → Options → Expert Advisors → Allow Algorithmic Trading, "
                "or click the AutoTrading button in the toolbar."
            )
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None

        # Determine price
        if order_type.upper() == "BUY":
            price = tick.ask
            mt5_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            mt5_type = mt5.ORDER_TYPE_SELL

        # Detect supported filling type from symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            filling_mode = symbol_info.filling_mode
            if filling_mode & 1:  # FOK supported
                filling_type = mt5.ORDER_FILLING_FOK
            elif filling_mode & 2:  # IOC supported
                filling_type = mt5.ORDER_FILLING_IOC
            else:  # RETURN
                filling_type = mt5.ORDER_FILLING_RETURN
        else:
            filling_type = mt5.ORDER_FILLING_IOC

        # MT5 comment field is limited to 31 characters
        if len(comment) > 31:
            logger.warning(f"Order comment truncated: '{comment}' -> '{comment[:31]}'")
            comment = comment[:31]

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_type,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        if sl > 0:
            request["sl"] = sl
        if tp > 0:
            request["tp"] = tp

        # Send order with filling type fallback
        filling_types = [filling_type]
        # Add fallback filling types
        for ft in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
            if ft not in filling_types:
                filling_types.append(ft)

        # retcode=1 is also treated as success by some MT5 builds
        SUCCESS_RETCODES = {mt5.TRADE_RETCODE_DONE, 1}

        result = None
        for i, ft in enumerate(filling_types):
            request["type_filling"] = ft
            ft_name = {0: "FOK", 1: "FOK", 2: "IOC", 4: "RETURN"}.get(ft, str(ft))
            result = mt5.order_send(request)

            if result is None:
                logger.error(f"Order send returned None (filling={ft_name}): {mt5.last_error()}")
                continue

            if result.retcode in SUCCESS_RETCODES:
                break  # Success

            # AutoTrading disabled in MT5 terminal
            if result.retcode == 10027:
                logger.error(
                    "AutoTrading is DISABLED in MT5. "
                    "Enable it: Tools → Options → Expert Advisors → Allow Algorithmic Trading, "
                    "or click the AutoTrading button in the toolbar. "
                    f"(retcode={result.retcode}, comment='{result.comment}')"
                )
                return None

            logger.warning(
                f"Order rejected (retcode={result.retcode}, comment='{result.comment}'). "
                f"Request: {order_type} {volume} {symbol}, filling={ft_name}"
            )

            # Don't retry filling types for non-filling errors
            if result.retcode not in (10030, 10033):  # INVALID_FILL, INVALID_ORDER
                logger.error(
                    f"Order failed (not a filling issue): retcode={result.retcode}, "
                    f"comment='{result.comment}'"
                )
                return None

        if result is None or result.retcode not in SUCCESS_RETCODES:
            logger.error(
                f"Order failed all filling types. "
                f"Last retcode={result.retcode if result else 'None'}, "
                f"comment='{result.comment if result else 'None'}'"
            )
            return None

        logger.info(f"Order placed: {order_type} {volume} {symbol} @ {result.price}")

        return {
            "ticket": result.order,
            "volume": volume,
            "price": price,
            "symbol": symbol,
            "type": order_type,
            "retcode": result.retcode,
        }

    async def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> bool:
        """Modify position SL/TP

        Args:
            ticket: Position ticket
            sl: New stop loss (None = no change)
            tp: New take profit (None = no change)

        Returns:
            True if successful
        """
        if self.use_mcp and self.mcp_modify_position:
            try:
                await self.mcp_modify_position(
                    id=ticket,
                    stop_loss=sl,
                    take_profit=tp
                )
                return True
            except Exception as e:
                logger.error(f"MCP modify_position failed: {e}")
                return False

        if not self.ensure_connected():
            return False

        # Get current position
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return False

        position = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify failed: {mt5.last_error()}")
            return False

        logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        return True

    async def close_position(self, ticket: int) -> bool:
        """Close position by ticket

        Args:
            ticket: Position ticket

        Returns:
            True if successful
        """
        if self.use_mcp and self.mcp_close_position:
            try:
                await self.mcp_close_position(id=ticket)
                return True
            except Exception as e:
                logger.error(f"MCP close_position failed: {e}")
                return False

        if not self.ensure_connected():
            return False

        # Get position
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return False

        position = positions[0]

        # Get price
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return False

        # Reverse order type
        if position.type == 0:  # BUY -> close with SELL
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:  # SELL -> close with BUY
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "SURGE-WSI Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {mt5.last_error()}")
            return False

        logger.info(f"Position {ticket} closed @ {price}")
        return True

    async def close_partial(self, ticket: int, volume: float) -> bool:
        """Partially close position

        Args:
            ticket: Position ticket
            volume: Volume to close

        Returns:
            True if successful
        """
        if not self.ensure_connected():
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return False

        position = positions[0]

        # Validate volume
        if volume >= position.volume:
            return await self.close_position(ticket)

        # Get price
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return False

        if position.type == 0:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "SURGE-WSI Partial",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Partial close failed: {mt5.last_error()}")
            return False

        logger.info(f"Position {ticket} partially closed: {volume} lots @ {price}")
        return True

    def get_history_deals(
        self,
        from_date: datetime = None,
        to_date: datetime = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get historical deals (closed trades)

        Args:
            from_date: Start date
            to_date: End date
            symbol: Filter by symbol (optional)

        Returns:
            List of deal dicts
        """
        if not self.ensure_connected():
            return []

        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()

        deals = mt5.history_deals_get(from_date, to_date)

        if deals is None:
            return []

        result = []
        for deal in deals:
            if symbol and deal.symbol != symbol:
                continue
            result.append(self._deal_to_dict(deal))

        return result

    def _deal_to_dict(self, deal) -> Dict[str, Any]:
        """Convert MT5 deal to dictionary"""
        deal_types = {0: "BUY", 1: "SELL", 2: "BALANCE", 3: "CREDIT"}
        entry_types = {0: "IN", 1: "OUT", 2: "INOUT", 3: "CLOSE"}

        return {
            "ticket": deal.ticket,
            "order": deal.order,
            "symbol": deal.symbol,
            "type": deal_types.get(deal.type, "UNKNOWN"),
            "entry": entry_types.get(deal.entry, "UNKNOWN"),
            "volume": deal.volume,
            "price": deal.price,
            "profit": deal.profit,
            "swap": deal.swap,
            "commission": deal.commission,
            "time": datetime.fromtimestamp(deal.time),
            "magic": deal.magic,
            "comment": deal.comment,
        }

    def get_session_info(self, symbol: str = "GBPUSD") -> Dict[str, Any]:
        """Get current trading session information

        Args:
            symbol: Trading symbol

        Returns:
            Dict with session info
        """
        now = datetime.now(timezone.utc)
        hour = now.hour

        # Determine session
        if 22 <= hour or hour < 7:
            session = "Sydney/Tokyo"
            major_session = False
        elif 7 <= hour < 8:
            session = "Tokyo/London Overlap"
            major_session = True
        elif 8 <= hour < 12:
            session = "London"
            major_session = True
        elif 12 <= hour < 16:
            session = "London/New York Overlap"
            major_session = True
        elif 16 <= hour < 21:
            session = "New York"
            major_session = True
        else:
            session = "Off-hours"
            major_session = False

        return {
            "utc_time": now,
            "session": session,
            "major_session": major_session,
            "day_of_week": now.strftime("%A"),
            "is_weekend": now.weekday() >= 5,
        }

    def get_last_error(self) -> Optional[str]:
        """Get last error message"""
        if self._last_error:
            return str(self._last_error)
        error = mt5.last_error()
        return str(error) if error else None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Utility functions for quick access
def quick_tick(symbol: str = "GBPUSD") -> Optional[Dict]:
    """Quick function to get tick data"""
    with MT5Connector() as connector:
        return connector.get_tick_sync(symbol)


def quick_ohlcv(symbol: str = "GBPUSD", timeframe: str = "H1", bars: int = 100) -> Optional[pd.DataFrame]:
    """Quick function to get OHLCV data"""
    with MT5Connector() as connector:
        return connector.get_ohlcv(symbol, timeframe, bars)
