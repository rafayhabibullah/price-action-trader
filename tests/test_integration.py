"""
Smoke test: runs a full backtest pipeline end-to-end using synthetic data.
No API calls — all data is generated in-memory.
"""
import numpy as np
import pandas as pd
import pytest
from backtest.engine import BacktestEngine
from optimization.grid_search import GridSearch
from reporting.results import ResultsReporter
from strategies.support_resistance import SupportResistanceStrategy
from strategies.order_blocks import OrderBlockStrategy
from strategies.fair_value_gaps import FairValueGapStrategy
from strategies.candlestick_patterns import CandlestickPatternStrategy
from strategies.market_structure import MarketStructureStrategy


def make_synthetic_data(n=300, seed=42):
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.8)
    closes = np.maximum(closes, 1.0)
    return pd.DataFrame({
        "open":   closes * np.random.uniform(0.998, 1.0, n),
        "high":   closes * np.random.uniform(1.001, 1.01, n),
        "low":    closes * np.random.uniform(0.99, 0.999, n),
        "close":  closes,
        "volume": np.random.uniform(1000, 10000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))


def test_full_pipeline_runs_without_error(tmp_path):
    df = make_synthetic_data()
    strategies = [
        SupportResistanceStrategy(lookback=20, touch_count=2, zone_width_pct=0.005),
        OrderBlockStrategy(),
        FairValueGapStrategy(),
        CandlestickPatternStrategy(),
        MarketStructureStrategy(),
    ]
    data = {"SYNTHETIC/USDT": {"1h": df}}
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(strategies=strategies, data=data, n_workers=1)
    assert isinstance(results, pd.DataFrame)
    if not results.empty:
        assert "score" in results.columns
        assert results["score"].iloc[0] >= results["score"].iloc[-1]


def test_single_backtest_computes_metrics():
    df = make_synthetic_data()
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    for StratClass in [
        SupportResistanceStrategy,
        OrderBlockStrategy,
        FairValueGapStrategy,
        CandlestickPatternStrategy,
        MarketStructureStrategy,
    ]:
        result = engine.run(StratClass(), df)
        assert "metrics" in result
        assert "trades" in result
        assert "win_rate" in result["metrics"]


def test_reporter_saves_results(tmp_path):
    df = make_synthetic_data()
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=[SupportResistanceStrategy()],
        data={"SYNTHETIC/USDT": {"1h": df}},
        n_workers=1,
    )
    reporter = ResultsReporter(reports_dir=tmp_path)
    out_dir = reporter.save(results, run_id="smoke_test")
    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "leaderboard.html").exists()
