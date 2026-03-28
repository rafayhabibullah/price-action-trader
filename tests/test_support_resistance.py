# tests/test_support_resistance.py
import pandas as pd
import numpy as np
import pytest
from strategies.support_resistance import SupportResistanceStrategy

@pytest.fixture
def sr_strategy():
    return SupportResistanceStrategy(lookback=20, touch_count=2, zone_width_pct=0.005)

@pytest.fixture
def bullish_sr_df():
    """Price bounces at support twice, then breaks above resistance."""
    np.random.seed(42)
    n = 60
    prices = [100.0]
    for i in range(n - 1):
        if i in (10, 20):   # two touches of ~95 support
            prices.append(95.2)
        elif i == 30:        # breakout
            prices.append(112.0)
        else:
            prices.append(prices[-1] + np.random.uniform(-1, 1))
    close = pd.Series(prices)
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    return df

def test_returns_required_columns(sr_strategy, bullish_sr_df):
    result = sr_strategy.generate_signals(bullish_sr_df)
    assert "signal" in result.columns
    assert "sl" in result.columns
    assert "tp" in result.columns

def test_signal_values_are_valid(sr_strategy, bullish_sr_df):
    result = sr_strategy.generate_signals(bullish_sr_df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_sl_set_when_signal(sr_strategy, bullish_sr_df):
    result = sr_strategy.generate_signals(bullish_sr_df)
    signal_rows = result[result["signal"] != 0]
    if not signal_rows.empty:
        assert signal_rows["sl"].notna().all()

def test_get_params(sr_strategy):
    params = sr_strategy.get_params()
    assert "lookback" in params
    assert "touch_count" in params
    assert "zone_width_pct" in params
