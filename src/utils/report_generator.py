"""
SURGE-WSI Report Generator
===========================

Generates professional trading reports with MT5 Strategy Tester style + shadcn UI:
- Visual charts (PNG) for Telegram
- PDF reports matching MT5 Strategy Tester format

Design: MT5 Report structure + shadcn UI styling

Author: SURIOTA Team
"""

import io
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, Image, PageBreak, HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from loguru import logger


# ================================================================
# SHADCN UI COLOR PALETTE
# ================================================================
class ShadcnColors:
    """shadcn UI inspired color palette"""
    # Backgrounds
    BG_PRIMARY = '#09090b'      # zinc-950
    BG_SECONDARY = '#18181b'    # zinc-900
    BG_CARD = '#1c1c1f'
    BG_MUTED = '#27272a'        # zinc-800

    # For PDF (light theme)
    PDF_BG = '#ffffff'
    PDF_BG_HEADER = '#f4f4f5'   # zinc-100
    PDF_BG_ALT = '#fafafa'      # zinc-50
    PDF_BORDER = '#e4e4e7'      # zinc-200
    PDF_TEXT = '#09090b'        # zinc-950
    PDF_TEXT_MUTED = '#71717a'  # zinc-500
    PDF_SUCCESS = '#22c55e'     # green-500
    PDF_DESTRUCTIVE = '#ef4444' # red-500
    PDF_ACCENT = '#3b82f6'      # blue-500

    # Dark theme
    BORDER = '#3f3f46'
    TEXT_PRIMARY = '#fafafa'
    TEXT_SECONDARY = '#a1a1aa'
    TEXT_MUTED = '#71717a'
    SUCCESS = '#22c55e'
    DESTRUCTIVE = '#ef4444'
    WARNING = '#f59e0b'
    INFO = '#3b82f6'
    ACCENT = '#8b5cf6'


# ================================================================
# DATA CLASSES
# ================================================================
@dataclass
class BacktestStats:
    """Comprehensive backtest statistics matching MT5 format"""
    # Settings
    expert_name: str = "SURGE-WSI H1 v6.4"
    symbol: str = "GBPUSD"
    period: str = "H1"
    date_range: str = ""
    company: str = "SURGE Trading System"
    currency: str = "USD"
    initial_deposit: float = 50000.0
    leverage: str = "1:100"

    # Core Results
    total_net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expected_payoff: float = 0.0
    recovery_factor: float = 0.0
    sharpe_ratio: float = 0.0

    # Drawdown
    balance_dd_absolute: float = 0.0
    balance_dd_maximal: float = 0.0
    balance_dd_maximal_pct: float = 0.0
    balance_dd_relative_pct: float = 0.0
    equity_dd_absolute: float = 0.0
    equity_dd_maximal: float = 0.0
    equity_dd_maximal_pct: float = 0.0

    # Trade Stats
    total_trades: int = 0
    total_deals: int = 0
    short_trades: int = 0
    short_won_pct: float = 0.0
    long_trades: int = 0
    long_won_pct: float = 0.0
    profit_trades: int = 0
    profit_trades_pct: float = 0.0
    loss_trades: int = 0
    loss_trades_pct: float = 0.0

    # Trade Analysis
    largest_profit_trade: float = 0.0
    largest_loss_trade: float = 0.0
    average_profit_trade: float = 0.0
    average_loss_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_wins_money: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_losses_money: float = 0.0
    avg_consecutive_wins: int = 0
    avg_consecutive_losses: int = 0

    # Time Analysis
    min_holding_time: str = "0:00:00"
    max_holding_time: str = "0:00:00"
    avg_holding_time: str = "0:00:00"

    # Additional
    bars: int = 0
    win_rate: float = 0.0
    losing_months: int = 0
    total_months: int = 13


class BacktestReportGenerator:
    """Generate professional backtest reports"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = ShadcnColors()

    def calculate_stats(self, trades: List[Dict], summary: Dict) -> BacktestStats:
        """Calculate comprehensive statistics from trades"""
        stats = BacktestStats()

        # Basic info
        stats.symbol = summary.get('symbol', 'GBPUSD')
        stats.initial_deposit = summary.get('initial_balance', 50000)
        stats.total_net_profit = summary.get('net_pnl', 0)
        stats.profit_factor = summary.get('profit_factor', 0)
        stats.win_rate = summary.get('win_rate', 0)
        stats.total_trades = summary.get('total_trades', len(trades))
        stats.total_deals = stats.total_trades * 2  # Entry + Exit
        stats.bars = summary.get('bars', 6403)
        stats.losing_months = summary.get('losing_months', 0)

        # Calculate from trades
        if trades:
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) <= 0]

            # Gross profit/loss
            stats.gross_profit = sum(t.get('pnl', 0) for t in wins)
            stats.gross_loss = abs(sum(t.get('pnl', 0) for t in losses))

            # Trade counts
            stats.profit_trades = len(wins)
            stats.loss_trades = len(losses)
            stats.profit_trades_pct = (stats.profit_trades / stats.total_trades * 100) if stats.total_trades > 0 else 0
            stats.loss_trades_pct = (stats.loss_trades / stats.total_trades * 100) if stats.total_trades > 0 else 0

            # Direction analysis
            long_trades = [t for t in trades if t.get('direction', '').upper() == 'BUY']
            short_trades = [t for t in trades if t.get('direction', '').upper() == 'SELL']
            stats.long_trades = len(long_trades)
            stats.short_trades = len(short_trades)
            long_wins = len([t for t in long_trades if t.get('pnl', 0) > 0])
            short_wins = len([t for t in short_trades if t.get('pnl', 0) > 0])
            stats.long_won_pct = (long_wins / stats.long_trades * 100) if stats.long_trades > 0 else 0
            stats.short_won_pct = (short_wins / stats.short_trades * 100) if stats.short_trades > 0 else 0

            # Trade analysis
            if wins:
                stats.largest_profit_trade = max(t.get('pnl', 0) for t in wins)
                stats.average_profit_trade = stats.gross_profit / len(wins)
            if losses:
                stats.largest_loss_trade = min(t.get('pnl', 0) for t in losses)
                stats.average_loss_trade = -stats.gross_loss / len(losses)

            # Expected payoff
            stats.expected_payoff = stats.total_net_profit / stats.total_trades if stats.total_trades > 0 else 0

            # Consecutive wins/losses
            consecutive = self._calculate_consecutive(trades)
            stats.max_consecutive_wins = consecutive['max_wins']
            stats.max_consecutive_wins_money = consecutive['max_wins_money']
            stats.max_consecutive_losses = consecutive['max_losses']
            stats.max_consecutive_losses_money = consecutive['max_losses_money']
            stats.avg_consecutive_wins = consecutive['avg_wins']
            stats.avg_consecutive_losses = consecutive['avg_losses']

            # Drawdown
            dd = self._calculate_drawdown(trades, stats.initial_deposit)
            stats.balance_dd_absolute = dd['absolute']
            stats.balance_dd_maximal = dd['maximal']
            stats.balance_dd_maximal_pct = dd['maximal_pct']
            stats.equity_dd_absolute = dd['absolute']
            stats.equity_dd_maximal = dd['maximal']
            stats.equity_dd_maximal_pct = dd['maximal_pct']

            # Recovery factor
            if stats.balance_dd_maximal > 0:
                stats.recovery_factor = stats.total_net_profit / stats.balance_dd_maximal

            # Sharpe ratio (simplified)
            pnls = [t.get('pnl', 0) for t in trades]
            if len(pnls) > 1:
                avg_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                if std_pnl > 0:
                    stats.sharpe_ratio = (avg_pnl / std_pnl) * np.sqrt(252)  # Annualized

            # Holding time
            times = self._calculate_holding_times(trades)
            stats.min_holding_time = times['min']
            stats.max_holding_time = times['max']
            stats.avg_holding_time = times['avg']

        # Date range
        if trades:
            first = trades[0].get('entry_time', '')
            last = trades[-1].get('entry_time', '')
            if isinstance(first, str) and first:
                first = first[:10]
            if isinstance(last, str) and last:
                last = last[:10]
            stats.date_range = f"H1 ({first} - {last})"
            stats.period = stats.date_range

        return stats

    def _calculate_consecutive(self, trades: List[Dict]) -> Dict:
        """Calculate consecutive wins/losses statistics"""
        result = {
            'max_wins': 0, 'max_wins_money': 0,
            'max_losses': 0, 'max_losses_money': 0,
            'avg_wins': 0, 'avg_losses': 0
        }

        if not trades:
            return result

        current_wins = 0
        current_losses = 0
        current_wins_money = 0
        current_losses_money = 0
        win_streaks = []
        loss_streaks = []

        for t in trades:
            pnl = t.get('pnl', 0)
            if pnl > 0:
                current_wins += 1
                current_wins_money += pnl
                if current_losses > 0:
                    loss_streaks.append(current_losses)
                    if current_losses > result['max_losses']:
                        result['max_losses'] = current_losses
                        result['max_losses_money'] = current_losses_money
                    current_losses = 0
                    current_losses_money = 0
            else:
                current_losses += 1
                current_losses_money += pnl
                if current_wins > 0:
                    win_streaks.append(current_wins)
                    if current_wins > result['max_wins']:
                        result['max_wins'] = current_wins
                        result['max_wins_money'] = current_wins_money
                    current_wins = 0
                    current_wins_money = 0

        # Final streaks
        if current_wins > 0:
            win_streaks.append(current_wins)
            if current_wins > result['max_wins']:
                result['max_wins'] = current_wins
                result['max_wins_money'] = current_wins_money
        if current_losses > 0:
            loss_streaks.append(current_losses)
            if current_losses > result['max_losses']:
                result['max_losses'] = current_losses
                result['max_losses_money'] = current_losses_money

        result['avg_wins'] = int(np.mean(win_streaks)) if win_streaks else 0
        result['avg_losses'] = int(np.mean(loss_streaks)) if loss_streaks else 0

        return result

    def _calculate_drawdown(self, trades: List[Dict], initial: float) -> Dict:
        """Calculate drawdown statistics"""
        if not trades:
            return {'absolute': 0, 'maximal': 0, 'maximal_pct': 0}

        equity = initial
        peak = initial
        max_dd = 0
        max_dd_pct = 0

        for t in trades:
            equity += t.get('pnl', 0)
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return {
            'absolute': max_dd,
            'maximal': max_dd,
            'maximal_pct': max_dd_pct
        }

    def _calculate_holding_times(self, trades: List[Dict]) -> Dict:
        """Calculate holding time statistics"""
        result = {'min': '0:00:00', 'max': '0:00:00', 'avg': '0:00:00'}

        durations = []
        for t in trades:
            entry = t.get('entry_time', '')
            exit_time = t.get('exit_time', '')
            if entry and exit_time:
                try:
                    if isinstance(entry, str):
                        entry = datetime.fromisoformat(entry.replace('+00:00', ''))
                    if isinstance(exit_time, str):
                        exit_time = datetime.fromisoformat(exit_time.replace('+00:00', ''))
                    duration = (exit_time - entry).total_seconds()
                    durations.append(duration)
                except:
                    pass

        if durations:
            min_sec = min(durations)
            max_sec = max(durations)
            avg_sec = np.mean(durations)

            result['min'] = str(timedelta(seconds=int(min_sec)))
            result['max'] = str(timedelta(seconds=int(max_sec)))
            result['avg'] = str(timedelta(seconds=int(avg_sec)))

        return result

    # ================================================================
    # PDF GENERATION - MT5 STYLE + SHADCN UI
    # ================================================================
    def generate_pdf_report(
        self,
        trades: List[Dict],
        monthly_stats: List[Dict],
        summary: Dict,
        filename: str = None
    ) -> Optional[str]:
        """Generate comprehensive PDF report in MT5 Strategy Tester style"""
        if not REPORTLAB_AVAILABLE:
            logger.warning("reportlab not available")
            return None

        try:
            # Calculate stats
            stats = self.calculate_stats(trades, summary)

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_report_{timestamp}.pdf"

            filepath = self.output_dir / filename

            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=A4,
                rightMargin=1.2*cm,
                leftMargin=1.2*cm,
                topMargin=1*cm,
                bottomMargin=1*cm
            )

            elements = []
            styles = self._get_styles()

            # ====== TITLE ======
            elements.append(Paragraph("Strategy Tester Report", styles['title']))
            elements.append(Paragraph("SURGE-WSI Trading System", styles['subtitle']))
            elements.append(Spacer(1, 8))
            elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(self.colors.PDF_BORDER)))
            elements.append(Spacer(1, 12))

            # ====== SETTINGS SECTION ======
            elements.append(Paragraph("Settings", styles['section']))
            settings_data = [
                ['Expert:', stats.expert_name, '', ''],
                ['Symbol:', stats.symbol, '', ''],
                ['Period:', stats.period, '', ''],
                ['Company:', stats.company, 'Currency:', stats.currency],
                ['Initial Deposit:', f"${stats.initial_deposit:,.2f}", 'Leverage:', stats.leverage],
            ]
            elements.append(self._create_settings_table(settings_data))
            elements.append(Spacer(1, 15))

            # ====== RESULTS SECTION ======
            elements.append(Paragraph("Results", styles['section']))

            # Row 1: Net Profit, DD Absolute, Equity DD
            results_row1 = [
                ['Total Net Profit:', f"${stats.total_net_profit:+,.2f}",
                 'Balance Drawdown Absolute:', f"${stats.balance_dd_absolute:,.2f}",
                 'Equity Drawdown Absolute:', f"${stats.equity_dd_absolute:,.2f}"],
                ['Gross Profit:', f"${stats.gross_profit:,.2f}",
                 'Balance Drawdown Maximal:', f"${stats.balance_dd_maximal:,.2f} ({stats.balance_dd_maximal_pct:.2f}%)",
                 'Equity Drawdown Maximal:', f"${stats.equity_dd_maximal:,.2f} ({stats.equity_dd_maximal_pct:.2f}%)"],
                ['Gross Loss:', f"${-stats.gross_loss:,.2f}",
                 'Balance Drawdown Relative:', f"{stats.balance_dd_maximal_pct:.2f}% (${stats.balance_dd_maximal:,.2f})",
                 'Equity Drawdown Relative:', f"{stats.equity_dd_maximal_pct:.2f}% (${stats.equity_dd_maximal:,.2f})"],
            ]
            elements.append(self._create_results_table(results_row1))
            elements.append(Spacer(1, 8))

            # Row 2: Metrics
            results_row2 = [
                ['Profit Factor:', f"{stats.profit_factor:.6f}",
                 'Expected Payoff:', f"${stats.expected_payoff:.2f}",
                 'Margin Level:', '-'],
                ['Recovery Factor:', f"{stats.recovery_factor:.6f}",
                 'Sharpe Ratio:', f"{stats.sharpe_ratio:.6f}",
                 'Z-Score:', '-'],
            ]
            elements.append(self._create_results_table(results_row2))
            elements.append(Spacer(1, 8))

            # Row 3: Holding Times
            results_row3 = [
                ['Minimal position holding time:', stats.min_holding_time,
                 'Maximal position holding time:', stats.max_holding_time,
                 'Average position holding time:', stats.avg_holding_time],
            ]
            elements.append(self._create_results_table(results_row3))
            elements.append(Spacer(1, 8))

            # Row 4: Trade Stats
            results_row4 = [
                ['Total Trades:', str(stats.total_trades),
                 f'Short Trades (won %):', f"{stats.short_trades} ({stats.short_won_pct:.2f}%)",
                 f'Long Trades (won %):', f"{stats.long_trades} ({stats.long_won_pct:.2f}%)"],
                ['Total Deals:', str(stats.total_deals),
                 'Profit Trades (% of total):', f"{stats.profit_trades} ({stats.profit_trades_pct:.2f}%)",
                 'Loss Trades (% of total):', f"{stats.loss_trades} ({stats.loss_trades_pct:.2f}%)"],
                ['', '',
                 'Largest profit trade:', f"${stats.largest_profit_trade:,.2f}",
                 'Largest loss trade:', f"${stats.largest_loss_trade:,.2f}"],
                ['', '',
                 'Average profit trade:', f"${stats.average_profit_trade:,.2f}",
                 'Average loss trade:', f"${stats.average_loss_trade:,.2f}"],
                ['', '',
                 f'Maximum consecutive wins ($):', f"{stats.max_consecutive_wins} (${stats.max_consecutive_wins_money:,.2f})",
                 f'Maximum consecutive losses ($):', f"{stats.max_consecutive_losses} (${stats.max_consecutive_losses_money:,.2f})"],
                ['', '',
                 'Average consecutive wins:', str(stats.avg_consecutive_wins),
                 'Average consecutive losses:', str(stats.avg_consecutive_losses)],
            ]
            elements.append(self._create_results_table(results_row4))
            elements.append(Spacer(1, 20))

            # ====== MONTHLY PERFORMANCE ======
            elements.append(Paragraph("Monthly Performance", styles['section']))

            monthly_data = [['Month', 'P/L', 'Trades', 'Win Rate', 'Status']]
            for m in monthly_stats:
                month = m.get('month', '')
                pnl = m.get('pnl', 0)
                trades_count = m.get('trades', '-')
                wr = m.get('win_rate', '-')
                status = '✓ WIN' if pnl >= 0 else '✗ LOSS'
                monthly_data.append([
                    month,
                    f"${pnl:+,.2f}",
                    str(trades_count) if trades_count != '-' else '-',
                    f"{wr:.1f}%" if isinstance(wr, (int, float)) else wr,
                    status
                ])

            # Summary row
            total_pnl = sum(m.get('pnl', 0) for m in monthly_stats)
            winning_months = len([m for m in monthly_stats if m.get('pnl', 0) >= 0])
            monthly_data.append(['TOTAL', f"${total_pnl:+,.2f}", '-', '-', f"{winning_months}/{len(monthly_stats)} Months"])

            elements.append(self._create_monthly_table(monthly_data))
            elements.append(Spacer(1, 20))

            # ====== ORDERS/TRADES LIST ======
            elements.append(PageBreak())
            elements.append(Paragraph("Orders", styles['section']))

            orders_header = ['#', 'Open Time', 'Type', 'Volume', 'Entry', 'Exit', 'S/L', 'T/P', 'P/L', 'Exit Reason']
            orders_data = [orders_header]

            for i, t in enumerate(trades[:50], 1):  # Limit to 50 trades for PDF
                entry_time = t.get('entry_time', '')
                if isinstance(entry_time, str):
                    entry_time = entry_time[:19]  # Trim timezone

                direction = t.get('direction', '-')
                lot = t.get('lot_size', t.get('volume', 0.01))
                entry_price = t.get('entry_price', 0)
                exit_price = t.get('exit_price', 0)
                sl = t.get('sl', '-')
                tp = t.get('tp', '-')
                pnl = t.get('pnl', 0)
                reason = t.get('exit_reason', '-')

                orders_data.append([
                    str(i),
                    entry_time,
                    direction,
                    f"{lot:.2f}",
                    f"{entry_price:.5f}" if entry_price else '-',
                    f"{exit_price:.5f}" if exit_price else '-',
                    f"{sl:.5f}" if isinstance(sl, float) else str(sl),
                    f"{tp:.5f}" if isinstance(tp, float) else str(tp),
                    f"${pnl:+,.2f}",
                    reason
                ])

            if len(trades) > 50:
                orders_data.append(['...', f'+{len(trades)-50} more trades', '', '', '', '', '', '', '', ''])

            elements.append(self._create_orders_table(orders_data))

            # ====== FOOTER ======
            elements.append(Spacer(1, 30))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor(self.colors.PDF_BORDER)))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(
                f"Generated by SURGE-WSI Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles['footer']
            ))

            # Build PDF
            doc.build(elements)
            logger.info(f"PDF report saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_styles(self) -> Dict:
        """Get custom PDF styles with shadcn UI look"""
        styles = getSampleStyleSheet()

        custom = {
            'title': ParagraphStyle(
                'Title', parent=styles['Heading1'],
                fontSize=22, spaceAfter=4,
                textColor=colors.HexColor(self.colors.PDF_TEXT),
                fontName='Helvetica-Bold'
            ),
            'subtitle': ParagraphStyle(
                'Subtitle', parent=styles['Normal'],
                fontSize=11, spaceAfter=8,
                textColor=colors.HexColor(self.colors.PDF_TEXT_MUTED),
            ),
            'section': ParagraphStyle(
                'Section', parent=styles['Heading2'],
                fontSize=13, spaceBefore=12, spaceAfter=8,
                textColor=colors.HexColor(self.colors.PDF_TEXT),
                fontName='Helvetica-Bold',
                borderColor=colors.HexColor(self.colors.PDF_BORDER),
                borderWidth=0, borderPadding=0,
            ),
            'normal': ParagraphStyle(
                'CustomNormal', parent=styles['Normal'],
                fontSize=9, textColor=colors.HexColor(self.colors.PDF_TEXT),
            ),
            'footer': ParagraphStyle(
                'Footer', parent=styles['Normal'],
                fontSize=8, textColor=colors.HexColor(self.colors.PDF_TEXT_MUTED),
                alignment=TA_CENTER
            ),
        }
        return custom

    def _create_settings_table(self, data: List) -> Table:
        """Create settings table with shadcn styling"""
        table = Table(data, colWidths=[2.5*cm, 6*cm, 2.5*cm, 6*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor(self.colors.PDF_TEXT_MUTED)),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor(self.colors.PDF_TEXT_MUTED)),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor(self.colors.PDF_TEXT)),
            ('TEXTCOLOR', (3, 0), (3, -1), colors.HexColor(self.colors.PDF_TEXT)),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTNAME', (3, 0), (3, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        return table

    def _create_results_table(self, data: List) -> Table:
        """Create results table with MT5 style layout"""
        table = Table(data, colWidths=[3.2*cm, 3*cm, 3.2*cm, 3.2*cm, 3.2*cm, 3*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor(self.colors.PDF_TEXT_MUTED)),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor(self.colors.PDF_TEXT_MUTED)),
            ('TEXTCOLOR', (4, 0), (4, -1), colors.HexColor(self.colors.PDF_TEXT_MUTED)),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor(self.colors.PDF_TEXT)),
            ('TEXTCOLOR', (3, 0), (3, -1), colors.HexColor(self.colors.PDF_TEXT)),
            ('TEXTCOLOR', (5, 0), (5, -1), colors.HexColor(self.colors.PDF_TEXT)),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.colors.PDF_BORDER)),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(self.colors.PDF_BG)),
        ]))
        return table

    def _create_monthly_table(self, data: List) -> Table:
        """Create monthly performance table"""
        table = Table(data, colWidths=[3*cm, 3.5*cm, 2.5*cm, 2.5*cm, 4*cm])
        style = [
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors.PDF_BG_HEADER)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(self.colors.PDF_TEXT)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.colors.PDF_BORDER)),
            ('ROWHEIGHT', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            # Last row (summary) styling
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor(self.colors.PDF_BG_HEADER)),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]

        # Color P/L cells
        for i, row in enumerate(data[1:], 1):
            if len(row) > 1:
                pnl_str = row[1]
                if '+' in pnl_str:
                    style.append(('TEXTCOLOR', (1, i), (1, i), colors.HexColor(self.colors.PDF_SUCCESS)))
                elif '-' in pnl_str and pnl_str != '-':
                    style.append(('TEXTCOLOR', (1, i), (1, i), colors.HexColor(self.colors.PDF_DESTRUCTIVE)))

        table.setStyle(TableStyle(style))
        return table

    def _create_orders_table(self, data: List) -> Table:
        """Create orders/trades table"""
        col_widths = [0.8*cm, 3*cm, 1.3*cm, 1.3*cm, 2*cm, 2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2*cm]
        table = Table(data, colWidths=col_widths)

        style = [
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors.PDF_BG_HEADER)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(self.colors.PDF_TEXT)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.colors.PDF_BORDER)),
            ('ROWHEIGHT', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]

        # Alternate row colors & P/L coloring
        for i, row in enumerate(data[1:], 1):
            if i % 2 == 0:
                style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor(self.colors.PDF_BG_ALT)))
            # Color P/L
            if len(row) > 8:
                pnl_str = row[8]
                if '+' in str(pnl_str):
                    style.append(('TEXTCOLOR', (8, i), (8, i), colors.HexColor(self.colors.PDF_SUCCESS)))
                elif '-' in str(pnl_str) and pnl_str != '-':
                    style.append(('TEXTCOLOR', (8, i), (8, i), colors.HexColor(self.colors.PDF_DESTRUCTIVE)))

        table.setStyle(TableStyle(style))
        return table

    # ================================================================
    # IMAGE GENERATION FOR TELEGRAM
    # ================================================================
    def generate_telegram_summary_image(
        self,
        summary: Dict,
        monthly_stats: List[Dict],
        filename: str = None
    ) -> Optional[str]:
        """Generate compact summary image for Telegram"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Segoe UI', 'Arial'],
                'font.size': 10,
            })

            fig = plt.figure(figsize=(12, 7), facecolor=self.colors.BG_PRIMARY)
            gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25,
                         left=0.06, right=0.94, top=0.88, bottom=0.08)

            # Header
            fig.text(0.06, 0.95, "SURGE-WSI v6.4 GBPUSD", fontsize=16, fontweight='bold',
                    color=self.colors.TEXT_PRIMARY, ha='left')
            fig.text(0.94, 0.95, datetime.now().strftime("%Y-%m-%d %H:%M"),
                    fontsize=10, color=self.colors.TEXT_MUTED, ha='right')

            # Row 1: Main P/L + Metrics + Win Rate
            ax1 = fig.add_subplot(gs[0, 0])
            pnl = summary.get('net_pnl', 0)
            self._plot_main_pnl_card(ax1, pnl, summary.get('return_pct', 0))

            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_key_metrics_compact(ax2, summary)

            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_win_rate_donut(ax3, summary)

            # Row 2: Monthly Performance
            ax_monthly = fig.add_subplot(gs[1, :])
            self._plot_monthly_performance(ax_monthly, monthly_stats)

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"telegram_report_{timestamp}.png"

            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=120, facecolor=self.colors.BG_PRIMARY,
                       edgecolor='none', bbox_inches='tight', pad_inches=0.15)
            plt.close(fig)

            logger.info(f"Telegram report image saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate telegram image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_main_pnl_card(self, ax, pnl: float, return_pct: float):
        """Plot main P/L card"""
        ax.set_facecolor(self.colors.BG_CARD)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_color(self.colors.BORDER)

        color = self.colors.SUCCESS if pnl >= 0 else self.colors.DESTRUCTIVE
        ax.text(0.5, 0.65, f"${pnl:+,.2f}", fontsize=24, color=color,
               ha='center', va='center', fontweight='bold')
        ax.text(0.5, 0.35, f"{return_pct:+.1f}% Return", fontsize=11,
               color=self.colors.TEXT_SECONDARY, ha='center', va='center')

    def _plot_key_metrics_compact(self, ax, summary: Dict):
        """Plot compact key metrics"""
        ax.set_facecolor(self.colors.BG_CARD)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_color(self.colors.BORDER)

        metrics = [
            ('Profit Factor', f"{summary.get('profit_factor', 0):.2f}"),
            ('Total Trades', f"{summary.get('total_trades', 0)}"),
            ('Losing Months', f"{summary.get('losing_months', 0)}/13"),
        ]

        y_positions = [0.75, 0.5, 0.25]
        for (label, value), y in zip(metrics, y_positions):
            ax.text(0.1, y, label, fontsize=9, color=self.colors.TEXT_MUTED, ha='left', va='center')
            ax.text(0.9, y, value, fontsize=11, color=self.colors.TEXT_PRIMARY,
                   ha='right', va='center', fontweight='bold')

    def _plot_win_rate_donut(self, ax, summary: Dict):
        """Plot win rate donut chart"""
        ax.set_facecolor(self.colors.BG_CARD)

        wins = summary.get('winners', 43)
        losses = summary.get('losers', 59)
        win_rate = summary.get('win_rate', 42.2)

        sizes = [wins, losses]
        colors_pie = [self.colors.SUCCESS, self.colors.DESTRUCTIVE]

        wedges, _ = ax.pie(sizes, colors=colors_pie, startangle=90,
                          wedgeprops=dict(width=0.35, edgecolor=self.colors.BG_CARD))

        ax.text(0, 0.08, f'{win_rate:.1f}%', ha='center', va='center',
               fontsize=20, fontweight='bold', color=self.colors.TEXT_PRIMARY)
        ax.text(0, -0.18, 'Win Rate', ha='center', va='center',
               fontsize=9, color=self.colors.TEXT_MUTED)
        ax.text(0, -0.55, f'{wins}W / {losses}L', ha='center', va='center',
               fontsize=10, color=self.colors.TEXT_SECONDARY)
        ax.set_title('Win/Loss', fontsize=12, color=self.colors.TEXT_PRIMARY,
                    fontweight='bold', pad=10)

    def _plot_monthly_performance(self, ax, monthly_stats: List[Dict]):
        """Plot monthly performance bar chart"""
        ax.set_facecolor(self.colors.BG_CARD)

        months = []
        pnls = []
        bar_colors = []

        for stat in monthly_stats:
            month_str = stat.get('month', '')
            pnl = stat.get('pnl', 0)
            try:
                parts = month_str.split('-')
                if len(parts) >= 2:
                    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    months.append(f"{month_names[int(parts[1])]}\n'{parts[0][2:]}")
                else:
                    months.append(month_str[-5:])
            except:
                months.append(month_str[-5:])

            pnls.append(pnl)
            bar_colors.append(self.colors.SUCCESS if pnl >= 0 else self.colors.DESTRUCTIVE)

        x = np.arange(len(months))
        bars = ax.bar(x, pnls, 0.7, color=bar_colors, edgecolor='none', alpha=0.9)

        for bar, pnl in zip(bars, pnls):
            height = bar.get_height()
            ax.annotate(f'${pnl:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5 if height >= 0 else -12), textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8, color=self.colors.TEXT_SECONDARY, fontweight='medium')

        ax.set_xticks(x)
        ax.set_xticklabels(months, fontsize=8)
        ax.axhline(y=0, color=self.colors.BORDER, linestyle='-', linewidth=0.5)
        ax.set_title('Monthly Performance', fontsize=12, color=self.colors.TEXT_PRIMARY,
                    fontweight='bold', loc='left', pad=10)
        ax.set_ylabel('P/L ($)', fontsize=9, color=self.colors.TEXT_MUTED)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors.BORDER)
        ax.spines['bottom'].set_color(self.colors.BORDER)
        ax.tick_params(axis='both', colors=self.colors.TEXT_MUTED, labelsize=8)
        ax.yaxis.grid(True, alpha=0.2)
