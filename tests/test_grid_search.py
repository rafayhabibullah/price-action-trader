import pandas as pd
import numpy as np
import pytest
from optimization.grid_search import GridSearch
from strategies.support_resistance import SupportResistanceStrategy

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open":  closes * 0.999, "high": closes * 1.005,
        "low":   closes * 0.995, "close": closes,
        "volume": np.ones(n) * 1000,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

def test_grid_search_returns_dataframe(sample_data):
    strategies = [SupportResistanceStrategy(lookback=10, touch_count=2, zone_width_pct=0.005)]
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=strategies,
        data={"TEST/USDT": {"1h": sample_data}},
    )
    assert isinstance(results, pd.DataFrame)

def test_grid_search_has_required_columns(sample_data):
    strategies = [SupportResistanceStrategy(lookback=10, touch_count=2, zone_width_pct=0.005)]
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=strategies,
        data={"TEST/USDT": {"1h": sample_data}},
    )
    for col in ["strategy", "asset", "timeframe", "sharpe_ratio", "score"]:
        assert col in results.columns

def test_grid_search_sorted_by_score(sample_data):
    strategies = [
        SupportResistanceStrategy(lookback=10, touch_count=2, zone_width_pct=0.005),
        SupportResistanceStrategy(lookback=20, touch_count=2, zone_width_pct=0.005),
    ]
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=strategies,
        data={"TEST/USDT": {"1h": sample_data}},
    )
    if len(results) > 1:
        scores = results["score"].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
