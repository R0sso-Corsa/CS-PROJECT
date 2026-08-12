"""Risk management tests."""
import numpy as np
import pandas as pd
import pytest
from src.risk.guardrails import RiskGuardrails, RiskReport
from src.risk.position_sizing import PositionSizer, KellyCriterion


def test_drawdown_breach():
    rg = RiskGuardrails(initial_capital=100_000.0, max_drawdown_pct=0.05)
    assert rg.check_equity(100_000.0) is True
    assert rg.check_equity(90_000.0) is True   # 10% — should trip
    assert rg.check_equity(94_000.0) is False  # 6% drawdown


def test_can_open_limits():
    rg = RiskGuardrails(initial_capital=100_000.0, max_open_positions=2)
    assert rg.can_open("BTC", 1)
    rg.open_position("BTC", 1, 50_000.0, 0.01)
    assert rg.can_open("ETH", 1)
    rg.open_position("ETH", -1, 3_000.0, 0.01)
    assert not rg.can_open("SOL", 1)


def test_kelly_criterion():
    k, _ = KellyCriterion.calculate(win_prob=0.6, win_loss_ratio=2.0)
    assert 0 < k < 0.25


def test_position_sizer_outputs():
    ps = PositionSizer(capital=100_000.0, max_position_pct=0.20, method="risk_parity")
    result = ps.size(signal_confidence=0.7, atr=1_000.0, current_price=50_000.0, stop_loss=48_000.0)
    assert "units" in result
    assert result["units"] > 0
    assert result["method"] == "risk_parity"
