import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from strategies.base import BaseStrategy
from backtest.engine import BacktestEngine


def _run_single(args):
    strategy, asset, timeframe, df, starting_capital, risk_per_trade = args
    engine = BacktestEngine(starting_capital=starting_capital, risk_per_trade=risk_per_trade)
    result = engine.run(strategy, df)
    m = result["metrics"]
    # Composite score: 0.4*Sharpe + 0.3*ProfitFactor + 0.2*WinRate + 0.1*(1-MaxDrawdown)
    sharpe = max(m["sharpe_ratio"], 0)
    pf = min(m["profit_factor"], 10)  # cap profit factor to avoid inf skewing score
    score = (0.4 * sharpe / 3.0 +          # normalize Sharpe (typical range 0-3)
             0.3 * pf / 10.0 +              # normalize PF (0-10)
             0.2 * m["win_rate"] +          # already 0-1
             0.1 * (1 - m["max_drawdown"])) # already 0-1
    return {
        "strategy": strategy.__class__.__name__,
        "params": strategy.get_params(),
        "asset": asset,
        "timeframe": timeframe,
        "num_trades": m["num_trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "sharpe_ratio": m["sharpe_ratio"],
        "max_drawdown": m["max_drawdown"],
        "total_return": m["total_return"],
        "score": score,
    }


class GridSearch:
    def __init__(self, starting_capital: float, risk_per_trade: float = 0.02):
        self.starting_capital = starting_capital
        self.risk_per_trade = risk_per_trade

    def run(
        self,
        strategies: list[BaseStrategy],
        data: dict[str, dict[str, pd.DataFrame]],
        n_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        data: {asset: {timeframe: ohlcv_df}}
        Returns sorted leaderboard DataFrame.
        """
        tasks = []
        for strategy in strategies:
            for asset, tf_data in data.items():
                for timeframe, df in tf_data.items():
                    if len(df) < 50:
                        continue
                    tasks.append((strategy, asset, timeframe, df,
                                  self.starting_capital, self.risk_per_trade))

        workers = n_workers or max(1, cpu_count() - 1)
        if workers == 1 or len(tasks) <= 4:
            rows = [_run_single(t) for t in tasks]
        else:
            with Pool(workers) as pool:
                rows = pool.map(_run_single, tasks)

        if not rows:
            return pd.DataFrame()

        df_results = pd.DataFrame(rows)
        df_results.sort_values("score", ascending=False, inplace=True)
        df_results.reset_index(drop=True, inplace=True)
        return df_results
