# tests/test_order_blocks.py
import pandas as pd
import numpy as np
import pytest
from strategies.order_blocks import OrderBlockStrategy

@pytest.fixture
def ob_strategy():
    return OrderBlockStrategy(block_size_pct=0.003, invalidation_pct=0.005)

@pytest.fixture
def ob_df():
    """Bearish candle followed by strong bullish impulse (order block setup)."""
    data = {
        "open":  [100, 101, 100, 98,  97,  100, 103, 106],
        "high":  [101, 102, 101, 100, 98,  102, 105, 108],
        "low":   [99,  100, 99,  97,  96,  99,  102, 105],
        "close": [101, 100, 99,  98,  100, 103, 106, 107],  # i=4 bearish then bullish impulse
        "volume":[1000]*8,
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=8, freq="1h"))

def test_returns_required_columns(ob_strategy, ob_df):
    result = ob_strategy.generate_signals(ob_df)
    assert "signal" in result.columns
    assert "sl" in result.columns
    assert "tp" in result.columns

def test_signal_values_valid(ob_strategy, ob_df):
    result = ob_strategy.generate_signals(ob_df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(ob_strategy):
    params = ob_strategy.get_params()
    assert "block_size_pct" in params
    assert "invalidation_pct" in params

def test_no_signal_on_flat_market(ob_strategy):
    flat = pd.DataFrame({
        "open":  [100.0] * 20, "high": [100.5] * 20,
        "low":   [99.5] * 20,  "close":[100.0] * 20, "volume":[1000.0] * 20,
    }, index=pd.date_range("2024-01-01", periods=20, freq="1h"))
    result = ob_strategy.generate_signals(flat)
    assert (result["signal"] == 0).all()
