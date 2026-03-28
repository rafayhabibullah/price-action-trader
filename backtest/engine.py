import pandas as pd
from strategies.base import BaseStrategy
from backtest.positions import simulate_trades
from backtest.metrics import compute_metrics


class BacktestEngine:
    def __init__(self, starting_capital: float, risk_per_trade: float, max_bars: int = 100):
        self.starting_capital = starting_capital
        self.risk_per_trade = risk_per_trade
        self.max_bars = max_bars

    def run(self, strategy: BaseStrategy, df: pd.DataFrame) -> dict:
        """
        Run a single backtest.
        Returns {"trades": pd.DataFrame, "metrics": dict}.
        """
        signals_df = strategy.generate_signals(df)
        trades = simulate_trades(
            signals_df,
            starting_capital=self.starting_capital,
            risk_per_trade=self.risk_per_trade,
            max_bars=self.max_bars,
        )
        metrics = compute_metrics(trades, starting_capital=self.starting_capital)
        return {"trades": trades, "metrics": metrics}
