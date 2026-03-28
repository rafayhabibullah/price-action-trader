import pandas as pd
import numpy as np
import pytest
from optimization.walk_forward import WalkForwardEngine, WalkForwardResult


def _make_ohlcv(n_bars=500):
    """Create synthetic OHLCV data with a slight upward trend."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="h")
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    df = pd.DataFrame({
        "open": close - np.random.rand(n_bars) * 0.3,
        "high": close + np.abs(np.random.randn(n_bars)) * 0.5,
        "low": close - np.abs(np.random.randn(n_bars)) * 0.5,
        "close": close,
        "volume": np.random.randint(100, 10000, n_bars).astype(float),
    }, index=dates)
    return df


class TestWalkForwardResult:
    def test_result_has_required_fields(self):
        r = WalkForwardResult(
            window_id=0,
            train_start=pd.Timestamp("2025-01-01"),
            train_end=pd.Timestamp("2025-06-01"),
            test_start=pd.Timestamp("2025-06-01"),
            test_end=pd.Timestamp("2025-09-01"),
            best_strategy="SupportResistanceStrategy",
            best_asset="BTC/USDT",
            best_timeframe="1h",
            train_score=0.45,
            train_metrics={"sharpe_ratio": 1.2, "win_rate": 0.55, "profit_factor": 1.8, "max_drawdown": 0.1, "total_return": 0.15, "num_trades": 20},
            oos_score=0.35,
            oos_metrics={"sharpe_ratio": 0.9, "win_rate": 0.50, "profit_factor": 1.5, "max_drawdown": 0.12, "total_return": 0.08, "num_trades": 10},
        )
        assert r.window_id == 0
        assert r.best_strategy == "SupportResistanceStrategy"
        assert r.oos_score == 0.35
        assert r.oos_metrics["sharpe_ratio"] == 0.9


class TestWalkForwardEngine:
    def test_generate_windows_count(self):
        engine = WalkForwardEngine(
            starting_capital=1000,
            risk_per_trade=0.02,
            train_bars=200,
            test_bars=50,
            step_bars=50,
        )
        df = _make_ohlcv(500)
        windows = engine._generate_windows(df)
        # 500 bars: offsets 0,50,100,150,200,250 where offset+250<=500
        assert len(windows) == 6

    def test_generate_windows_boundaries(self):
        engine = WalkForwardEngine(
            starting_capital=1000,
            risk_per_trade=0.02,
            train_bars=200,
            test_bars=50,
            step_bars=50,
        )
        df = _make_ohlcv(500)
        windows = engine._generate_windows(df)
        first = windows[0]
        assert len(first["train"]) == 200
        assert len(first["test"]) == 50
        assert first["train"].index[-1] < first["test"].index[0]

    def test_run_returns_results(self):
        """Integration: run walk-forward on synthetic data."""
        from strategies.support_resistance import SupportResistanceStrategy
        engine = WalkForwardEngine(
            starting_capital=1000,
            risk_per_trade=0.02,
            train_bars=200,
            test_bars=50,
            step_bars=50,
        )
        df = _make_ohlcv(500)
        data = {"SYN/USDT": {"1h": df}}
        strategies = [SupportResistanceStrategy()]
        results = engine.run(strategies, data)
        assert isinstance(results, list)
        # Results may be empty if no strategy produces >= 3 trades on synthetic data
        for r in results:
            assert isinstance(r, WalkForwardResult)

    def test_aggregate_oos_metrics(self):
        engine = WalkForwardEngine(
            starting_capital=1000,
            risk_per_trade=0.02,
            train_bars=200,
            test_bars=50,
            step_bars=50,
        )
        results = [
            WalkForwardResult(
                window_id=i,
                train_start=pd.Timestamp("2025-01-01"),
                train_end=pd.Timestamp("2025-06-01"),
                test_start=pd.Timestamp("2025-06-01"),
                test_end=pd.Timestamp("2025-09-01"),
                best_strategy="S",
                best_asset="A",
                best_timeframe="1h",
                train_score=0.5,
                train_metrics={"sharpe_ratio": 1.5, "win_rate": 0.6, "profit_factor": 2.0, "max_drawdown": 0.1, "total_return": 0.2, "num_trades": 20},
                oos_score=score,
                oos_metrics={"sharpe_ratio": sh, "win_rate": wr, "profit_factor": 1.5, "max_drawdown": dd, "total_return": 0.1, "num_trades": 10},
            )
            for i, (score, sh, wr, dd) in enumerate([
                (0.35, 0.9, 0.50, 0.12),
                (0.40, 1.1, 0.55, 0.08),
                (0.30, 0.7, 0.45, 0.15),
            ])
        ]
        agg = engine.aggregate(results)
        assert "mean_oos_score" in agg
        assert "mean_oos_sharpe" in agg
        assert "oos_degradation" in agg
        assert abs(agg["mean_oos_score"] - 0.35) < 1e-9
        assert abs(agg["mean_train_score"] - 0.5) < 1e-9
        assert abs(agg["oos_degradation"] - (1 - 0.35 / 0.5)) < 1e-9
