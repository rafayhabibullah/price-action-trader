"""
Anchored Walk-Forward Testing Engine.

Splits data into rolling train/test windows, optimizes on each train window,
then evaluates the winning strategy on the held-out test window.
Aggregates out-of-sample (OOS) metrics to detect overfitting.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from strategies.base import BaseStrategy
from backtest.engine import BacktestEngine
from optimization.grid_search import score_metrics


@dataclass
class WalkForwardResult:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_strategy: str
    best_asset: str
    best_timeframe: str
    train_score: float
    train_metrics: dict
    oos_score: float
    oos_metrics: dict


class WalkForwardEngine:
    def __init__(
        self,
        starting_capital: float,
        risk_per_trade: float = 0.02,
        train_bars: int = 200,
        test_bars: int = 50,
        step_bars: int = 50,
    ):
        self.starting_capital = starting_capital
        self.risk_per_trade = risk_per_trade
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars

    def _generate_windows(self, df: pd.DataFrame) -> list[dict]:
        """Generate rolling train/test window slices from a single DataFrame."""
        n = len(df)
        windows = []
        offset = 0
        while offset + self.train_bars + self.test_bars <= n:
            train_end = offset + self.train_bars
            test_end = train_end + self.test_bars
            windows.append({
                "train": df.iloc[offset:train_end],
                "test": df.iloc[train_end:test_end],
            })
            offset += self.step_bars
        return windows

    def run(
        self,
        strategies: list[BaseStrategy],
        data: dict[str, dict[str, pd.DataFrame]],
    ) -> list[WalkForwardResult]:
        """
        Run walk-forward analysis.

        For each asset/timeframe combo, generates rolling windows.
        On each train window: test all strategies, pick best by composite score.
        On each test window: evaluate the winner out-of-sample.
        """
        results = []
        window_id = 0

        for asset, tf_data in data.items():
            for timeframe, df in tf_data.items():
                if len(df) < self.train_bars + self.test_bars:
                    continue

                windows = self._generate_windows(df)

                for w in windows:
                    train_df = w["train"]
                    test_df = w["test"]

                    # -- Train phase: find best strategy --
                    best_score = -1
                    best_strat = None
                    best_train_metrics = None

                    for strat in strategies:
                        engine = BacktestEngine(
                            starting_capital=self.starting_capital,
                            risk_per_trade=self.risk_per_trade,
                        )
                        result = engine.run(strat, train_df)
                        m = result["metrics"]
                        if m["num_trades"] < 3:
                            continue
                        sc = score_metrics(m)
                        if sc > best_score:
                            best_score = sc
                            best_strat = strat
                            best_train_metrics = m

                    if best_strat is None:
                        continue

                    # -- Test phase: evaluate winner OOS --
                    engine = BacktestEngine(
                        starting_capital=self.starting_capital,
                        risk_per_trade=self.risk_per_trade,
                    )
                    oos_result = engine.run(best_strat, test_df)
                    oos_m = oos_result["metrics"]
                    oos_score = score_metrics(oos_m)

                    results.append(WalkForwardResult(
                        window_id=window_id,
                        train_start=train_df.index[0],
                        train_end=train_df.index[-1],
                        test_start=test_df.index[0],
                        test_end=test_df.index[-1],
                        best_strategy=best_strat.__class__.__name__,
                        best_asset=asset,
                        best_timeframe=timeframe,
                        train_score=best_score,
                        train_metrics=best_train_metrics,
                        oos_score=oos_score,
                        oos_metrics=oos_m,
                    ))
                    window_id += 1

        return results

    def aggregate(self, results: list[WalkForwardResult]) -> dict:
        """Aggregate walk-forward results into summary statistics."""
        if not results:
            return {
                "num_windows": 0,
                "mean_train_score": 0,
                "mean_oos_score": 0,
                "mean_oos_sharpe": 0,
                "mean_oos_win_rate": 0,
                "mean_oos_return": 0,
                "oos_degradation": 0,
                "oos_positive_pct": 0,
            }

        train_scores = [r.train_score for r in results]
        oos_scores = [r.oos_score for r in results]
        oos_sharpes = [r.oos_metrics["sharpe_ratio"] for r in results]
        oos_win_rates = [r.oos_metrics["win_rate"] for r in results]
        oos_returns = [r.oos_metrics["total_return"] for r in results]

        mean_train = np.mean(train_scores)
        mean_oos = np.mean(oos_scores)

        return {
            "num_windows": len(results),
            "mean_train_score": float(mean_train),
            "mean_oos_score": float(mean_oos),
            "mean_oos_sharpe": float(np.mean(oos_sharpes)),
            "mean_oos_win_rate": float(np.mean(oos_win_rates)),
            "mean_oos_return": float(np.mean(oos_returns)),
            "oos_degradation": float(1 - mean_oos / mean_train) if mean_train > 0 else 0,
            "oos_positive_pct": float(np.mean([1 if s > 0 else 0 for s in oos_returns])),
        }
