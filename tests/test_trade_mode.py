"""Test Trade Mode Manager
=========================

Tests the auto vs signal-only mode switching logic.
"""
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading import TradeModeManager, TradeMode, TradeModeConfig


def test_december_mode():
    """Test December switches to signal-only"""
    config = TradeModeConfig()
    manager = TradeModeManager(config)

    # December date
    dec_time = datetime(2025, 12, 15, 10, 0)
    mode = manager.evaluate_mode(dec_time, 10000, atr_pips=30)

    print(f"December 15 mode: {mode.value}")
    print(f"Reason: {manager.mode_reason}")
    assert mode == TradeMode.SIGNAL_ONLY
    print("[OK] December mode test passed")


def test_consecutive_losses():
    """Test consecutive losses trigger signal-only"""
    config = TradeModeConfig(max_consecutive_losses=3)
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)

    # Record 3 consecutive losses
    manager.record_trade_result(is_win=False, pnl=-50)
    manager.record_trade_result(is_win=False, pnl=-50)
    manager.record_trade_result(is_win=False, pnl=-50)

    mode = manager.evaluate_mode(datetime.now(), 9850, atr_pips=30)

    print(f"\nAfter 3 losses mode: {mode.value}")
    print(f"Reason: {manager.mode_reason}")
    assert mode == TradeMode.SIGNAL_ONLY
    print("[OK] Consecutive losses test passed")


def test_daily_loss_limit():
    """Test daily loss limit triggers signal-only"""
    config = TradeModeConfig(daily_loss_limit_pct=2.0)
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)

    # Record 2.5% loss
    manager.record_trade_result(is_win=False, pnl=-250)

    mode = manager.evaluate_mode(datetime.now(), 9750, atr_pips=30)

    print(f"\nAfter 2.5% loss mode: {mode.value}")
    print(f"Reason: {manager.mode_reason}")
    assert mode == TradeMode.SIGNAL_ONLY
    print("[OK] Daily loss limit test passed")


def test_high_volatility():
    """Test high volatility triggers signal-only"""
    config = TradeModeConfig(high_volatility_atr_pips=50)
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)

    # High ATR
    mode = manager.evaluate_mode(datetime.now(), 10000, atr_pips=60)

    print(f"\nHigh volatility (ATR=60) mode: {mode.value}")
    print(f"Reason: {manager.mode_reason}")
    assert mode == TradeMode.SIGNAL_ONLY
    print("[OK] High volatility test passed")


def test_regime_instability():
    """Test regime instability triggers signal-only"""
    config = TradeModeConfig(max_regime_changes_per_day=5)
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)

    # Simulate many regime changes
    regimes = ['BULLISH', 'BEARISH', 'SIDEWAYS', 'BULLISH', 'BEARISH', 'SIDEWAYS', 'BULLISH']
    for regime in regimes:
        manager.record_regime_change(regime)

    mode = manager.evaluate_mode(datetime.now(), 10000, atr_pips=30)

    print(f"\nAfter {manager.daily_regime_changes} regime changes: {mode.value}")
    print(f"Reason: {manager.mode_reason}")
    assert mode == TradeMode.SIGNAL_ONLY
    print("[OK] Regime instability test passed")


def test_normal_conditions():
    """Test normal conditions stay in auto mode"""
    config = TradeModeConfig()
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)

    # Normal conditions (not December, low volatility, no losses)
    june_time = datetime(2025, 6, 15, 10, 0)
    mode = manager.evaluate_mode(june_time, 10000, atr_pips=30)

    print(f"\nNormal conditions mode: {mode.value}")
    print(f"Reason: {manager.mode_reason}")
    assert mode == TradeMode.AUTO
    print("[OK] Normal conditions test passed")


def test_force_mode():
    """Test force mode override"""
    config = TradeModeConfig()
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)

    # Force signal-only mode
    manager.force_mode(TradeMode.SIGNAL_ONLY, duration_hours=1)

    # Even in normal conditions, should return forced mode
    june_time = datetime(2025, 6, 15, 10, 0)
    mode = manager.evaluate_mode(june_time, 10000, atr_pips=30)

    print(f"\nForced mode: {mode.value}")
    print(f"Reason: {manager.mode_reason}")
    assert mode == TradeMode.SIGNAL_ONLY
    print("[OK] Force mode test passed")


def test_status_output():
    """Test status output format"""
    config = TradeModeConfig()
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)
    manager.reset_weekly_stats(10000)

    # Record some activity
    manager.record_trade_result(is_win=True, pnl=100)
    manager.record_trade_result(is_win=True, pnl=50)
    manager.record_trade_result(is_win=False, pnl=-30)

    status = manager.get_status()

    print("\n" + "="*50)
    print("STATUS OUTPUT:")
    print("="*50)
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("[OK] Status output test passed")


def test_telegram_format():
    """Test Telegram message format"""
    config = TradeModeConfig()
    manager = TradeModeManager(config)
    manager.reset_daily_stats(10000)

    # Record some activity
    manager.record_trade_result(is_win=True, pnl=100)
    manager.record_trade_result(is_win=False, pnl=-30)
    manager.evaluate_mode(datetime.now(), 10070, atr_pips=30)

    msg = manager.format_mode_message()

    print("\n" + "="*50)
    print("TELEGRAM MESSAGE:")
    print("="*50)
    # Handle emojis that Windows console can't print
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode())
    print("[OK] Telegram format test passed")


if __name__ == "__main__":
    print("="*50)
    print("TRADE MODE MANAGER TESTS")
    print("="*50)

    test_normal_conditions()
    test_december_mode()
    test_consecutive_losses()
    test_daily_loss_limit()
    test_high_volatility()
    test_regime_instability()
    test_force_mode()
    test_status_output()
    test_telegram_format()

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
