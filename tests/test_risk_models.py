import pytest
from optimization.risk_models import FixedPctRisk, FixedDollarRisk, KellyRisk

def test_fixed_pct_risk():
    model = FixedPctRisk(pct=0.02)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    # risk = 1000 * 0.02 = 20; price_risk = 2.0; size = 20/2 = 10
    assert abs(size - 10.0) < 0.01

def test_fixed_dollar_risk():
    model = FixedDollarRisk(amount=20.0)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    assert abs(size - 10.0) < 0.01

def test_kelly_risk_positive():
    model = KellyRisk(win_rate=0.55, avg_win=2.0, avg_loss=1.0)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    assert size > 0

def test_kelly_risk_capped_at_25pct():
    # Kelly fraction can be very high — should be capped
    model = KellyRisk(win_rate=0.9, avg_win=5.0, avg_loss=1.0)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    max_risk = 1000.0 * 0.25  # 25% cap
    max_size = max_risk / abs(100.0 - 98.0)
    assert size <= max_size + 0.01

def test_zero_price_risk_returns_zero():
    model = FixedPctRisk(pct=0.02)
    size = model.position_size(capital=1000.0, entry=100.0, sl=100.0)
    assert size == 0.0
