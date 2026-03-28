import pandas as pd
import pytest
from pathlib import Path
from reporting.results import ResultsReporter

@pytest.fixture
def sample_results():
    return pd.DataFrame({
        "strategy": ["OrderBlock", "FVG", "SR"],
        "asset": ["BTC/USDT", "AAPL", "ETH/USDT"],
        "timeframe": ["4h", "1d", "1h"],
        "score": [0.85, 0.72, 0.61],
        "sharpe_ratio": [1.5, 1.2, 0.9],
        "win_rate": [0.6, 0.55, 0.5],
        "profit_factor": [2.1, 1.8, 1.4],
        "max_drawdown": [0.08, 0.12, 0.15],
        "total_return": [0.45, 0.30, 0.20],
        "num_trades": [42, 31, 28],
        "params": ["{}", "{}", "{}"],
    })

def test_save_csv(sample_results, tmp_path):
    reporter = ResultsReporter(reports_dir=tmp_path)
    reporter.save(sample_results, run_id="test")
    csv_path = tmp_path / "test" / "leaderboard.csv"
    assert csv_path.exists()

def test_save_html(sample_results, tmp_path):
    reporter = ResultsReporter(reports_dir=tmp_path)
    reporter.save(sample_results, run_id="test")
    html_path = tmp_path / "test" / "leaderboard.html"
    assert html_path.exists()

def test_top_n(sample_results, tmp_path):
    reporter = ResultsReporter(reports_dir=tmp_path)
    top = reporter.top(sample_results, n=2)
    assert len(top) == 2
    assert top.iloc[0]["score"] >= top.iloc[1]["score"]
