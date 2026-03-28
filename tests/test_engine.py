import pandas as pd
import numpy as np
import pytest
from backtest.engine import BacktestEngine
from strategies.base import BaseStrategy


class AlwaysBuyStrategy(BaseStrategy):
    def _generate_raw_signals(self, df):
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan
        for i in range(5, len(df)):
            result.iloc[i, result.columns.get_loc("signal")] = 1
            result.iloc[i, result.columns.get_loc("sl")] = df["close"].iloc[i] * 0.98
            result.iloc[i, result.columns.get_loc("tp")] = df["close"].iloc[i] * 1.04
        return result

    def get_params(self):
        return {}


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open":  closes * 0.999,
        "high":  closes * 1.005,
        "low":   closes * 0.995,
        "close": closes,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))


def test_run_returns_result_dict(sample_df):
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    result = engine.run(AlwaysBuyStrategy(), sample_df)
    assert "trades" in result
    assert "metrics" in result


def test_metrics_have_required_keys(sample_df):
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    result = engine.run(AlwaysBuyStrategy(), sample_df)
    for key in ["win_rate", "sharpe_ratio", "max_drawdown", "profit_factor", "num_trades"]:
        assert key in result["metrics"]


def test_no_negative_equity(sample_df):
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    result = engine.run(AlwaysBuyStrategy(), sample_df)
    if not result["trades"].empty:
        assert (result["trades"]["equity"] > 0).all()
