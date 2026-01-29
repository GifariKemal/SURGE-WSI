"""Telegram Bot Integration
==========================

Features:
- Signal notifications with SURIOTA compact style
- Position updates with tree-style formatting
- Bidirectional commands:
  - /status, /balance, /positions, /regime, /pois
  - /pause, /resume, /close_all

Styling Reference:
- Tree-style: Backtest reports, daily summaries
- Compact: Signal alerts, trade execution, quick status

Author: SURIOTA Team
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Callable, Any, List
from loguru import logger

try:
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed")


class TelegramFormatter:
    """Message formatter for Telegram with SURIOTA styling"""

    # ============================================================
    # EMOJI REFERENCE
    # ============================================================
    # Brand
    EAGLE = "🦅"        # SURIOTA brand

    # Status
    ROCKET = "🚀"       # Big profit (>$50)
    CHECK = "✅"        # Profit/success
    CROSS = "❌"        # Loss/failure
    WARNING = "⚠️"      # Warning/caution
    BELL = "🔔"         # Alert/notification

    # Trading
    UP = "📈"           # BUY/uptrend
    DOWN = "📉"         # SELL/downtrend
    TARGET = "🎯"       # Target/result
    SHIELD = "🛡"       # Stop loss/protection

    # Info
    CHART = "📊"        # Chart/stats
    MONEY = "💰"        # Money/profit
    GEAR = "⚙️"         # Config/settings
    MEMO = "📝"         # Notes/info
    CLOCK = "⏰"        # Time
    BRAIN = "🧠"        # Analysis/AI

    # Circles for status
    GREEN = "🟢"        # Active/good
    RED = "🔴"          # Inactive/bad
    YELLOW = "🟡"       # Warning/signal-only
    BLUE = "🔵"         # Info

    # Tree connectors
    BRANCH = "├"
    LAST = "└"
    PIPE = "│"

    # ============================================================
    # HELPER METHODS
    # ============================================================

    @classmethod
    def progress_bar(cls, percent: float, width: int = 10) -> str:
        """Create progress bar: ████████░░"""
        filled = int(percent / 100 * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    @classmethod
    def direction_indicator(cls, direction: str) -> str:
        """Get direction with color indicator"""
        if direction.upper() == "BUY":
            return f"{cls.GREEN} {direction.upper()}"
        else:
            return f"{cls.RED} {direction.upper()}"

    @classmethod
    def profit_emoji(cls, profit: float) -> str:
        """Get profit emoji based on amount"""
        if profit >= 50:
            return cls.ROCKET
        elif profit > 0:
            return cls.CHECK
        return cls.CROSS

    @classmethod
    def regime_emoji(cls, regime: str) -> str:
        """Get regime emoji"""
        regime_lower = regime.lower()
        if "bullish" in regime_lower:
            return cls.UP
        elif "bearish" in regime_lower:
            return cls.DOWN
        return "➡️"

    # ============================================================
    # TREE-STYLE FORMATTERS (Reports, Summaries)
    # ============================================================

    @classmethod
    def tree_header(cls, title: str, emoji: str = None) -> str:
        """Create tree-style header"""
        if emoji:
            return f"{emoji} <b>{title}</b>\n"
        return f"<b>{title}</b>\n"

    @classmethod
    def tree_section(cls, title: str, emoji: str = None) -> str:
        """Create tree-style section"""
        if emoji:
            return f"\n{emoji} <b>{title}</b>\n"
        return f"\n<b>{title}</b>\n"

    @classmethod
    def tree_item(cls, label: str, value: Any, last: bool = False) -> str:
        """Create tree-style item"""
        connector = cls.LAST if last else cls.BRANCH
        return f"{connector} {label}: <code>{value}</code>\n"

    @classmethod
    def tree_spacer(cls) -> str:
        """Empty line"""
        return "\n"

    # ============================================================
    # COMPACT SURIOTA STYLE (Signals, Alerts)
    # ============================================================

    @classmethod
    def compact_signal(
        cls,
        direction: str,
        symbol: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float = None,
        tp3: float = None,
        confidence: float = 0,
        regime: str = None,
        risk_percent: float = 1.0,
        signal_id: str = None
    ) -> str:
        """Format compact SURIOTA-style signal alert

        Example output:
        🦅 SURIOTA SIGNAL
        🟢 GBPUSD • BUY
        Entry: 1.27150-1.27200
         TP1:   1.27500 +35p
         TP2:   1.27800 +65p
         SL:    1.26900 -25p

        🧠 Analysis
        BULLISH 📈 • ████████░░ 80%
        TF: H1+H4 • ATR: 15.2p

        ⚠️ R:R 1:2.5 • Risk 1.0%
        #SIG001 • 25-01 14:30
        """
        # Calculate pips
        pip_mult = 10000 if "JPY" not in symbol else 100
        sl_pips = abs(entry - sl) * pip_mult
        tp1_pips = abs(tp1 - entry) * pip_mult
        tp2_pips = abs(tp2 - entry) * pip_mult if tp2 else 0
        tp3_pips = abs(tp3 - entry) * pip_mult if tp3 else 0

        # Calculate R:R
        rr = tp1_pips / sl_pips if sl_pips > 0 else 0

        # Direction indicator
        dir_emoji = cls.GREEN if direction.upper() == "BUY" else cls.RED
        dir_arrow = "+" if direction.upper() == "BUY" else "-"

        msg = f"{cls.EAGLE} <b>SURIOTA SIGNAL</b>\n"
        msg += f"{dir_emoji} <b>{symbol}</b> • <b>{direction.upper()}</b>\n"

        # Entry and levels in pre block for alignment
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f" TP1:  {tp1:.5f} {dir_arrow}{tp1_pips:.0f}p\n"
        if tp2:
            msg += f" TP2:  {tp2:.5f} {dir_arrow}{tp2_pips:.0f}p\n"
        if tp3:
            msg += f" TP3:  {tp3:.5f} {dir_arrow}{tp3_pips:.0f}p\n"
        msg += f" SL:   {sl:.5f} -{sl_pips:.0f}p\n"
        msg += "</pre>\n"

        # Analysis section
        if regime or confidence > 0:
            msg += f"{cls.BRAIN} <b>Analysis</b>\n"
            analysis_parts = []
            if regime:
                regime_emoji = cls.regime_emoji(regime)
                analysis_parts.append(f"{regime.upper()} {regime_emoji}")
            if confidence > 0:
                bar = cls.progress_bar(confidence)
                analysis_parts.append(f"{bar} {confidence:.0f}%")
            msg += " • ".join(analysis_parts) + "\n"

        # Risk info
        msg += f"\n{cls.WARNING} R:R 1:{rr:.1f} • Risk {risk_percent}%\n"

        # Footer with ID and timestamp
        timestamp = datetime.now().strftime("%d-%m %H:%M")
        if signal_id:
            msg += f"<code>#{signal_id}</code> • {timestamp}"
        else:
            msg += f"<code>{timestamp}</code>"

        return msg

    @classmethod
    def compact_execution(
        cls,
        direction: str,
        symbol: str,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        ticket: int = None
    ) -> str:
        """Format compact trade execution alert"""
        pip_mult = 10000 if "JPY" not in symbol else 100
        sl_pips = abs(entry - sl) * pip_mult
        tp_pips = abs(tp - entry) * pip_mult
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        dir_emoji = cls.GREEN if direction.upper() == "BUY" else cls.RED

        msg = f"{cls.TARGET} <b>TRADE EXECUTED</b>\n"
        msg += f"{dir_emoji} <b>{symbol}</b> • <b>{direction.upper()}</b>\n"
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f"SL:    {sl:.5f} ({sl_pips:.0f}p)\n"
        msg += f"TP:    {tp:.5f} ({tp_pips:.0f}p)\n"
        msg += f"Lot:   {lot:.2f}\n"
        msg += "</pre>"
        msg += f"R:R 1:{rr:.1f}"
        if ticket:
            msg += f" • <code>#{ticket}</code>"

        return msg

    @classmethod
    def compact_close(
        cls,
        direction: str,
        symbol: str,
        entry: float,
        exit_price: float,
        pnl: float,
        pips: float,
        result: str,
        duration: str = None
    ) -> str:
        """Format compact position close alert"""
        emoji = cls.profit_emoji(pnl)
        dir_emoji = cls.UP if direction.upper() == "BUY" else cls.DOWN

        msg = f"{emoji} <b>POSITION CLOSED</b>\n"
        msg += f"{dir_emoji} <b>{symbol}</b> • {direction.upper()}\n"
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f"Exit:  {exit_price:.5f}\n"
        msg += f"Pips:  {pips:+.1f}\n"
        msg += f"P/L:   ${pnl:+.2f}\n"
        msg += "</pre>"
        msg += f"Result: <b>{result}</b>"
        if duration:
            msg += f" • {duration}"

        return msg

    # ============================================================
    # STATUS & BALANCE (Tree-Style)
    # ============================================================

    @classmethod
    def status_report(
        cls,
        state: str,
        mode: str,
        regime: str,
        in_killzone: bool,
        session: str,
        positions: int,
        daily_pnl: float,
        weekly_pnl: float = 0
    ) -> str:
        """Format detailed status report (tree-style)"""
        # State emoji
        if state.lower() == "running":
            state_emoji = cls.GREEN
        elif state.lower() == "paused":
            state_emoji = cls.YELLOW
        else:
            state_emoji = cls.RED

        # Mode emoji
        if mode.lower() == "auto":
            mode_emoji = cls.GREEN
            mode_text = "AUTO TRADE"
        elif mode.lower() == "recovery":
            mode_emoji = cls.YELLOW
            mode_text = "RECOVERY"
        elif mode.lower() == "signal":
            mode_emoji = cls.YELLOW
            mode_text = "SIGNAL ONLY"
        else:
            mode_emoji = cls.RED
            mode_text = "MONITORING"

        msg = cls.tree_header("SURGE-WSI Status", cls.CHART)

        msg += cls.tree_section("System", cls.GEAR)
        msg += cls.tree_item("State", f"{state_emoji} {state.upper()}")
        msg += cls.tree_item("Mode", f"{mode_emoji} {mode_text}")
        msg += cls.tree_item("Positions", positions, last=True)

        msg += cls.tree_section("Market", cls.BRAIN)
        msg += cls.tree_item("Regime", f"{cls.regime_emoji(regime)} {regime.upper()}")
        kz_status = f"{cls.GREEN} Yes" if in_killzone else f"{cls.RED} No"
        msg += cls.tree_item("Kill Zone", f"{kz_status} ({session})", last=True)

        msg += cls.tree_section("Performance", cls.MONEY)
        daily_emoji = cls.CHECK if daily_pnl >= 0 else cls.CROSS
        weekly_emoji = cls.CHECK if weekly_pnl >= 0 else cls.CROSS
        msg += cls.tree_item("Daily P/L", f"{daily_emoji} ${daily_pnl:+.2f}")
        msg += cls.tree_item("Weekly P/L", f"{weekly_emoji} ${weekly_pnl:+.2f}", last=True)

        return msg

    @classmethod
    def balance_report(
        cls,
        login: int,
        server: str,
        balance: float,
        equity: float,
        profit: float,
        free_margin: float,
        margin_level: float = 0
    ) -> str:
        """Format balance report (tree-style)"""
        msg = cls.tree_header("Account Balance", cls.MONEY)

        msg += cls.tree_section("Account", cls.MEMO)
        msg += cls.tree_item("Login", login)
        msg += cls.tree_item("Server", server, last=True)

        msg += cls.tree_section("Balance", cls.CHART)
        msg += cls.tree_item("Balance", f"${balance:,.2f}")
        msg += cls.tree_item("Equity", f"${equity:,.2f}")

        profit_emoji = cls.CHECK if profit >= 0 else cls.CROSS
        msg += cls.tree_item("Open P/L", f"{profit_emoji} ${profit:+.2f}")
        msg += cls.tree_item("Free Margin", f"${free_margin:,.2f}")

        if margin_level > 0:
            msg += cls.tree_item("Margin Level", f"{margin_level:.0f}%", last=True)
        else:
            msg = msg.rstrip('\n') + '\n'  # Remove last tree_item's newline fix

        return msg

    # ============================================================
    # BACKTEST REPORT (Tree-Style)
    # ============================================================

    @classmethod
    def backtest_report(
        cls,
        symbol: str,
        timeframe: str,
        period: str,
        total_trades: int,
        winners: int,
        losers: int,
        win_rate: float,
        gross_profit: float,
        gross_loss: float,
        net_pnl: float,
        profit_factor: float,
        max_drawdown: float,
        avg_duration: str = None
    ) -> str:
        """Format backtest report (tree-style)

        Example:
        🚀 SURGE-AI Backtest Report

        📊 Data
        ├ Symbol: GBPUSD
        ├ Timeframe: H1
        ├ Period: 2025-01 to 2025-12
        └ Bars: 7,425

        🎯 Performance
        ...
        """
        msg = cls.tree_header("SURGE-WSI Backtest Report", cls.ROCKET)

        msg += cls.tree_section("Data", cls.CHART)
        msg += cls.tree_item("Symbol", symbol)
        msg += cls.tree_item("Timeframe", timeframe)
        msg += cls.tree_item("Period", period, last=True)

        msg += cls.tree_section("Performance", cls.TARGET)
        msg += cls.tree_item("Total Trades", total_trades)
        msg += cls.tree_item("Winners", f"{winners} ({win_rate:.1f}%)")
        msg += cls.tree_item("Losers", losers)
        if avg_duration:
            msg += cls.tree_item("Avg Duration", avg_duration, last=True)
        else:
            msg = msg.rstrip('\n') + '\n'

        msg += cls.tree_section("Profit/Loss", cls.MONEY)
        msg += cls.tree_item("Gross Profit", f"${gross_profit:,.2f}")
        msg += cls.tree_item("Gross Loss", f"${gross_loss:,.2f}")

        pnl_emoji = cls.ROCKET if net_pnl >= 50 else (cls.CHECK if net_pnl > 0 else cls.CROSS)
        msg += cls.tree_item("Net P/L", f"{pnl_emoji} ${net_pnl:+,.2f}")
        msg += cls.tree_item("Profit Factor", f"{profit_factor:.2f}")
        msg += cls.tree_item("Max Drawdown", f"${max_drawdown:,.2f}", last=True)

        return msg

    # ============================================================
    # DAILY SUMMARY (Tree-Style)
    # ============================================================

    @classmethod
    def daily_summary(
        cls,
        date: str,
        trades: int,
        winners: int,
        losers: int,
        gross_profit: float,
        gross_loss: float,
        net_pnl: float,
        balance: float
    ) -> str:
        """Format daily summary report"""
        win_rate = (winners / trades * 100) if trades > 0 else 0
        pnl_emoji = cls.profit_emoji(net_pnl)

        msg = cls.tree_header(f"Daily Summary - {date}", pnl_emoji)

        msg += cls.tree_section("Trades", cls.TARGET)
        msg += cls.tree_item("Total", trades)
        msg += cls.tree_item("Winners", f"{winners} ({win_rate:.0f}%)")
        msg += cls.tree_item("Losers", losers, last=True)

        msg += cls.tree_section("P/L", cls.MONEY)
        msg += cls.tree_item("Gross Profit", f"+${gross_profit:.2f}")
        msg += cls.tree_item("Gross Loss", f"-${gross_loss:.2f}")
        msg += cls.tree_item("Net P/L", f"${net_pnl:+.2f}")
        msg += cls.tree_item("Balance", f"${balance:,.2f}", last=True)

        return msg

    # ============================================================
    # MODE & ALERT MESSAGES
    # ============================================================

    @classmethod
    def mode_change(
        cls,
        new_mode: str,
        reason: str,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0
    ) -> str:
        """Format mode change notification"""
        mode_lower = new_mode.lower()

        if mode_lower == "auto":
            emoji = cls.GREEN
            mode_text = "AUTO TRADE"
            desc = "Full auto execution enabled"
        elif mode_lower == "recovery":
            emoji = cls.YELLOW
            mode_text = "RECOVERY"
            desc = "Reduced lot size active"
        elif mode_lower == "signal":
            emoji = cls.YELLOW
            mode_text = "SIGNAL ONLY"
            desc = "Signals only, no auto-execution"
        else:
            emoji = cls.RED
            mode_text = "MONITORING"
            desc = "Full pause - observing only"

        msg = f"{emoji} <b>Mode: {mode_text}</b>\n"
        msg += f"<i>{desc}</i>\n"
        msg += cls.tree_spacer()
        msg += f"<b>Reason:</b>\n<code>{reason}</code>\n"
        msg += cls.tree_spacer()
        msg += cls.tree_item("Daily P/L", f"${daily_pnl:+.2f}")
        msg += cls.tree_item("Weekly P/L", f"${weekly_pnl:+.2f}", last=True)

        return msg

    @classmethod
    def signal_only_alert(
        cls,
        direction: str,
        symbol: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        confidence: float,
        lot: float,
        regime: str = None,
        reason: str = None
    ) -> str:
        """Format signal-only alert (no auto-execution)"""
        pip_mult = 10000 if "JPY" not in symbol else 100
        sl_pips = abs(entry - sl) * pip_mult
        tp1_pips = abs(tp1 - entry) * pip_mult
        tp2_pips = abs(tp2 - entry) * pip_mult
        tp3_pips = abs(tp3 - entry) * pip_mult

        dir_emoji = cls.UP if direction.upper() == "BUY" else cls.DOWN

        msg = f"{cls.BELL} <b>SIGNAL ONLY</b>\n"
        msg += f"<i>No auto-execution</i>\n"
        msg += cls.tree_spacer()
        msg += f"{dir_emoji} <b>{symbol} • {direction.upper()}</b>\n"
        msg += "<pre>"
        msg += f"Entry: {entry:.5f}\n"
        msg += f"SL:    {sl:.5f} ({sl_pips:.0f}p)\n"
        msg += f"\nPartial TP:\n"
        msg += f" TP1 (50%): {tp1:.5f} ({tp1_pips:.0f}p)\n"
        msg += f" TP2 (30%): {tp2:.5f} ({tp2_pips:.0f}p)\n"
        msg += f" TP3 (20%): {tp3:.5f} ({tp3_pips:.0f}p)\n"
        msg += "</pre>"

        # Analysis
        msg += f"\n{cls.BRAIN} <b>Analysis</b>\n"
        if regime:
            msg += f"Regime: {regime.upper()} {cls.regime_emoji(regime)}\n"
        bar = cls.progress_bar(confidence)
        msg += f"Confidence: {bar} {confidence:.0f}%\n"
        msg += f"Suggested Lot: {lot:.2f}\n"

        # Reason
        if reason:
            msg += f"\n{cls.WARNING} <b>Mode Reason:</b>\n"
            msg += f"<code>{reason}</code>"

        return msg

    @classmethod
    def partial_tp_hit(
        cls,
        tp_level: int,
        symbol: str,
        price: float,
        pnl: float,
        pips: float,
        remaining_percent: int
    ) -> str:
        """Format partial TP hit notification"""
        msg = f"{cls.TARGET} <b>TP{tp_level} HIT</b>\n"
        msg += f"{symbol} @ {price:.5f}\n"
        msg += f"<code>+{pips:.1f} pips • +${pnl:.2f}</code>\n"

        if tp_level == 1:
            msg += f"\n{cls.SHIELD} SL moved to breakeven"
        if remaining_percent > 0:
            msg += f"\n{remaining_percent}% position remaining"

        return msg

    # ============================================================
    # POSITIONS LIST
    # ============================================================

    @classmethod
    def positions_list(cls, positions: List[Dict]) -> str:
        """Format positions list"""
        if not positions:
            return f"{cls.MEMO} <b>No Open Positions</b>"

        msg = cls.tree_header(f"Open Positions ({len(positions)})", cls.CHART)

        for i, pos in enumerate(positions):
            is_last = (i == len(positions) - 1)
            dir_emoji = cls.UP if pos.get('type', '').upper() == 'BUY' else cls.DOWN
            pnl = pos.get('profit', 0)
            pnl_emoji = cls.CHECK if pnl >= 0 else cls.CROSS

            msg += cls.tree_spacer()
            msg += f"{dir_emoji} <b>{pos.get('symbol', 'N/A')}</b>\n"
            msg += f"   {cls.BRANCH} Entry: {pos.get('price_open', 0):.5f}\n"
            msg += f"   {cls.BRANCH} Lot: {pos.get('volume', 0):.2f}\n"
            msg += f"   {cls.BRANCH} SL: {pos.get('sl', 0):.5f}\n"
            msg += f"   {cls.BRANCH} TP: {pos.get('tp', 0):.5f}\n"
            msg += f"   {cls.LAST} P/L: {pnl_emoji} ${pnl:+.2f}\n"

        return msg

    # ============================================================
    # HELP MESSAGE
    # ============================================================

    @classmethod
    def help_message(cls) -> str:
        """Format help message"""
        msg = f"{cls.EAGLE} <b>SURGE-WSI Commands</b>\n"

        msg += cls.tree_section("Information", cls.CHART)
        msg += "<code>/status</code> - System status\n"
        msg += "<code>/balance</code> - Account balance\n"
        msg += "<code>/positions</code> - Open positions\n"
        msg += "<code>/regime</code> - Market regime\n"
        msg += "<code>/pois</code> - Active POIs\n"
        msg += "<code>/activity</code> - Market activity (INTEL_60)\n"
        msg += "<code>/mode</code> - Trading mode\n"

        msg += cls.tree_section("Control", cls.GEAR)
        msg += "<code>/pause</code> - Pause trading\n"
        msg += "<code>/resume</code> - Resume trading\n"
        msg += "<code>/close_all</code> - Close all positions\n"

        msg += cls.tree_section("Mode Override", cls.WARNING)
        msg += "<code>/force_auto</code> - Force AUTO mode\n"
        msg += "<code>/force_signal</code> - Force SIGNAL-ONLY\n"

        msg += cls.tree_spacer()
        msg += "<b>Trading Modes:</b>\n"
        msg += f"{cls.GREEN} AUTO - Full auto execution\n"
        msg += f"{cls.YELLOW} RECOVERY - Reduced lot size\n"
        msg += f"{cls.YELLOW} SIGNAL - Signal only\n"
        msg += f"{cls.RED} MONITORING - Full pause\n"

        msg += cls.tree_spacer()
        msg += "<b>Hybrid Mode:</b>\n"
        msg += "Trade in Kill Zones OR when\n"
        msg += "market shows high activity\n"

        return msg

    # ============================================================
    # LEGACY COMPATIBILITY
    # ============================================================

    @classmethod
    def header(cls, title: str, emoji: str = None) -> str:
        return cls.tree_header(title, emoji)

    @classmethod
    def section(cls, title: str, emoji: str = None) -> str:
        return cls.tree_section(title, emoji)

    @classmethod
    def item(cls, label: str, value: Any, last: bool = False) -> str:
        return cls.tree_item(label, value, last)

    @classmethod
    def spacer(cls) -> str:
        return cls.tree_spacer()

    @classmethod
    def signal_alert(
        cls,
        direction: str,
        symbol: str,
        entry: float,
        sl: float,
        tp: float,
        confidence: float,
        lot: float,
        regime: str = None
    ) -> str:
        """Legacy signal alert - now uses compact style"""
        return cls.compact_signal(
            direction=direction,
            symbol=symbol,
            entry=entry,
            sl=sl,
            tp1=tp,
            confidence=confidence,
            regime=regime
        )

    @classmethod
    def position_closed(
        cls,
        direction: str,
        entry: float,
        exit_price: float,
        pnl: float,
        result: str
    ) -> str:
        """Legacy position closed - now uses compact style"""
        pips = abs(exit_price - entry) / 0.0001
        return cls.compact_close(
            direction=direction,
            symbol="GBPUSD",
            entry=entry,
            exit_price=exit_price,
            pnl=pnl,
            pips=pips,
            result=result
        )

    @classmethod
    def status(
        cls,
        state: str,
        regime: str,
        in_killzone: bool,
        session: str,
        positions: int,
        daily_pnl: float
    ) -> str:
        """Legacy status - now uses detailed report"""
        return cls.status_report(
            state=state,
            mode="auto",
            regime=regime,
            in_killzone=in_killzone,
            session=session,
            positions=positions,
            daily_pnl=daily_pnl
        )

    @classmethod
    def balance(
        cls,
        login: int,
        balance: float,
        equity: float,
        profit: float,
        free_margin: float
    ) -> str:
        """Legacy balance - now uses detailed report"""
        return cls.balance_report(
            login=login,
            server="N/A",
            balance=balance,
            equity=equity,
            profit=profit,
            free_margin=free_margin
        )

    @classmethod
    def mode_change_alert(
        cls,
        new_mode: str,
        reason: str,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0
    ) -> str:
        """Legacy mode change"""
        return cls.mode_change(new_mode, reason, daily_pnl, weekly_pnl)


class TelegramNotifier:
    """Telegram bot with command support"""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True
    ):
        """Initialize Telegram notifier

        Args:
            bot_token: Bot token from BotFather
            chat_id: Chat ID to send messages to
            enabled: Enable notifications
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled

        self._bot: Optional[Bot] = None
        self._app: Optional[Application] = None

        # Rate limiting (max 20 messages per minute for Telegram API)
        self._message_times: List[datetime] = []
        self._rate_limit_messages = 15  # Conservative limit
        self._rate_limit_window = 60  # seconds

        # Command callbacks (set by executor)
        self.on_status: Optional[Callable] = None
        self.on_balance: Optional[Callable] = None
        self.on_positions: Optional[Callable] = None
        self.on_regime: Optional[Callable] = None
        self.on_pois: Optional[Callable] = None
        self.on_activity: Optional[Callable] = None  # Intelligent Activity Filter status
        self.on_mode: Optional[Callable] = None
        self.on_pause: Optional[Callable] = None
        self.on_resume: Optional[Callable] = None
        self.on_close_all: Optional[Callable] = None
        self.on_force_auto: Optional[Callable] = None
        self.on_force_signal: Optional[Callable] = None
        self.on_test_buy: Optional[Callable] = None
        self.on_test_sell: Optional[Callable] = None
        self.on_autotrading: Optional[Callable] = None

    async def initialize(self) -> bool:
        """Initialize bot

        Returns:
            True if successful
        """
        if not TELEGRAM_AVAILABLE:
            logger.error("python-telegram-bot not installed")
            return False

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured")
            return False

        try:
            self._bot = Bot(token=self.bot_token)
            await self._bot.get_me()  # Test connection
            logger.info("Telegram bot initialized")
            return True
        except Exception as e:
            logger.error(f"Telegram init failed: {e}")
            return False

    async def start_polling(self):
        """Start polling for commands"""
        if not TELEGRAM_AVAILABLE or not self._bot:
            return

        try:
            self._app = Application.builder().token(self.bot_token).build()

            # Add command handlers
            self._app.add_handler(CommandHandler("status", self._handle_status))
            self._app.add_handler(CommandHandler("balance", self._handle_balance))
            self._app.add_handler(CommandHandler("positions", self._handle_positions))
            self._app.add_handler(CommandHandler("regime", self._handle_regime))
            self._app.add_handler(CommandHandler("pois", self._handle_pois))
            self._app.add_handler(CommandHandler("activity", self._handle_activity))
            self._app.add_handler(CommandHandler("mode", self._handle_mode))
            self._app.add_handler(CommandHandler("pause", self._handle_pause))
            self._app.add_handler(CommandHandler("resume", self._handle_resume))
            self._app.add_handler(CommandHandler("close_all", self._handle_close_all))
            self._app.add_handler(CommandHandler("force_auto", self._handle_force_auto))
            self._app.add_handler(CommandHandler("force_signal", self._handle_force_signal))
            self._app.add_handler(CommandHandler("test_buy", self._handle_test_buy))
            self._app.add_handler(CommandHandler("test_sell", self._handle_test_sell))
            self._app.add_handler(CommandHandler("autotrading", self._handle_autotrading))
            self._app.add_handler(CommandHandler("help", self._handle_help))

            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling()

            logger.info("Telegram polling started")
        except Exception as e:
            logger.error(f"Failed to start polling: {e}")

    async def stop_polling(self):
        """Stop polling"""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send(self, message: str, force: bool = False):
        """Send message to chat with rate limiting

        Args:
            message: HTML formatted message
            force: Bypass rate limiting for critical messages
        """
        if not self.enabled or not self._bot:
            return

        # Rate limiting check
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._rate_limit_window)
        self._message_times = [t for t in self._message_times if t > cutoff]

        if not force and len(self._message_times) >= self._rate_limit_messages:
            logger.warning(f"Telegram rate limit reached ({len(self._message_times)}/{self._rate_limit_messages}), skipping message")
            return

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            self._message_times.append(now)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    # Command handlers - with exception handling
    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            if self.on_status:
                status = await self.on_status()
                await update.message.reply_text(status, parse_mode='HTML')
            else:
                await update.message.reply_text("Status not available")
        except Exception as e:
            logger.error(f"Error handling /status: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        try:
            if self.on_balance:
                balance = await self.on_balance()
                await update.message.reply_text(balance, parse_mode='HTML')
            else:
                await update.message.reply_text("Balance not available")
        except Exception as e:
            logger.error(f"Error handling /balance: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            if self.on_positions:
                positions = await self.on_positions()
                await update.message.reply_text(positions, parse_mode='HTML')
            else:
                await update.message.reply_text("Positions not available")
        except Exception as e:
            logger.error(f"Error handling /positions: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /regime command"""
        try:
            if self.on_regime:
                regime = await self.on_regime()
                await update.message.reply_text(regime, parse_mode='HTML')
            else:
                await update.message.reply_text("Regime not available")
        except Exception as e:
            logger.error(f"Error handling /regime: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_pois(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pois command"""
        try:
            if self.on_pois:
                pois = await self.on_pois()
                await update.message.reply_text(pois, parse_mode='HTML')
            else:
                await update.message.reply_text("POIs not available")
        except Exception as e:
            logger.error(f"Error handling /pois: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_activity(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /activity command - Intelligent Activity Filter status"""
        try:
            if self.on_activity:
                activity = await self.on_activity()
                await update.message.reply_text(activity, parse_mode='HTML')
            else:
                await update.message.reply_text("Activity filter status not available")
        except Exception as e:
            logger.error(f"Error handling /activity: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mode command"""
        try:
            if self.on_mode:
                mode = await self.on_mode()
                await update.message.reply_text(mode, parse_mode='HTML')
            else:
                await update.message.reply_text("Mode status not available")
        except Exception as e:
            logger.error(f"Error handling /mode: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_force_auto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /force_auto command"""
        try:
            if self.on_force_auto:
                await self.on_force_auto()
                await update.message.reply_text(
                    f"{TelegramFormatter.GREEN} <b>Forced AUTO mode</b>\n<i>Use with caution!</i>",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text("Cannot force auto mode")
        except Exception as e:
            logger.error(f"Error handling /force_auto: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_force_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /force_signal command"""
        try:
            if self.on_force_signal:
                await self.on_force_signal()
                await update.message.reply_text(
                    f"{TelegramFormatter.YELLOW} <b>Forced SIGNAL-ONLY mode</b>",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text("Cannot force signal-only mode")
        except Exception as e:
            logger.error(f"Error handling /force_signal: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command"""
        try:
            if self.on_pause:
                await self.on_pause()
                await update.message.reply_text(
                    f"⏸ <b>Trading Paused</b>\n<i>Use /resume to continue</i>",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text("Cannot pause")
        except Exception as e:
            logger.error(f"Error handling /pause: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        try:
            if self.on_resume:
                await self.on_resume()
                await update.message.reply_text(
                    f"▶️ <b>Trading Resumed</b>",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text("Cannot resume")
        except Exception as e:
            logger.error(f"Error handling /resume: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close_all command"""
        try:
            if self.on_close_all:
                result = await self.on_close_all()
                await update.message.reply_text(result, parse_mode='HTML')
            else:
                await update.message.reply_text("Cannot close positions")
        except Exception as e:
            logger.error(f"Error handling /close_all: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_test_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test_buy command"""
        try:
            if self.on_test_buy:
                result = await self.on_test_buy()
                await update.message.reply_text(result, parse_mode='HTML')
            else:
                await update.message.reply_text("Test buy not available")
        except Exception as e:
            logger.error(f"Error handling /test_buy: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_test_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test_sell command"""
        try:
            if self.on_test_sell:
                result = await self.on_test_sell()
                await update.message.reply_text(result, parse_mode='HTML')
            else:
                await update.message.reply_text("Test sell not available")
        except Exception as e:
            logger.error(f"Error handling /test_sell: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_autotrading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /autotrading command"""
        try:
            if self.on_autotrading:
                result = await self.on_autotrading()
                await update.message.reply_text(result, parse_mode='HTML')
            else:
                await update.message.reply_text("AutoTrading check not available")
        except Exception as e:
            logger.error(f"Error handling /autotrading: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        try:
            await update.message.reply_text(
                TelegramFormatter.help_message(),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error handling /help: {e}")
            await update.message.reply_text(f"Error: {e}")
