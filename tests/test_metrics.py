import pandas as pd
import numpy as np
import pytest
from backtest.metrics import compute_metrics

@pytest.fixture
def winning_trades():
    return pd.DataFrame({
        "entry_price": [100.0, 200.0, 150.0],
        "exit_price":  [110.0, 210.0, 165.0],
        "pnl":         [10.0,  10.0,  15.0],
        "direction":   [1, 1, 1],
        "equity":      [1010.0, 1020.0, 1035.0],
        "r_multiple":  [2.0, 2.0, 3.0],
    })

@pytest.fixture
def mixed_trades():
    return pd.DataFrame({
        "entry_price": [100.0, 200.0, 150.0, 120.0],
        "exit_price":  [110.0, 190.0, 165.0, 115.0],
        "pnl":         [10.0, -10.0, 15.0, -5.0],
        "direction":   [1, 1, 1, 1],
        "equity":      [1010.0, 1000.0, 1015.0, 1010.0],
        "r_multiple":  [2.0, -1.0, 3.0, -1.0],
    })

def test_win_rate_all_winners(winning_trades):
    m = compute_metrics(winning_trades, starting_capital=1000.0)
    assert m["win_rate"] == 1.0

def test_win_rate_mixed(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    assert m["win_rate"] == 0.5

def test_profit_factor(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    gross_profit = 10.0 + 15.0
    gross_loss = 10.0 + 5.0
    assert abs(m["profit_factor"] - gross_profit / gross_loss) < 0.01

def test_max_drawdown(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    # equity = [1010, 1000, 1015, 1010]; peak = [1010, 1010, 1015, 1015]
    # drawdown at i=1: (1010-1000)/1010 ≈ 0.0099
    # drawdown at i=3: (1015-1010)/1015 ≈ 0.0049
    # max_drawdown ≈ 0.0099
    assert abs(m["max_drawdown"] - (10.0 / 1010.0)) < 0.001

def test_sharpe_ratio_positive_returns(winning_trades):
    m = compute_metrics(winning_trades, starting_capital=1000.0)
    assert m["sharpe_ratio"] > 0

def test_returns_all_required_keys(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    for key in ["win_rate", "profit_factor", "max_drawdown", "sharpe_ratio", "total_return", "num_trades"]:
        assert key in m, f"Missing key: {key}"
