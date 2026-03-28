import pandas as pd
import numpy as np


def compute_metrics(trades: pd.DataFrame, starting_capital: float) -> dict:
    """
    Compute performance metrics from a trades DataFrame.

    Required columns: pnl, equity, r_multiple
    """
    if trades.empty:
        return {
            "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
            "sharpe_ratio": 0.0, "total_return": 0.0, "num_trades": 0,
        }

    num_trades = len(trades)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    win_rate = len(wins) / num_trades

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    equity = trades["equity"].values
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    max_drawdown = float(drawdowns.max())

    # Sharpe ratio (annualised, assumes daily returns — adjust per timeframe in engine)
    equity_series = pd.Series(np.concatenate([[starting_capital], equity]))
    returns = equity_series.pct_change().dropna()
    if returns.std() > 0:
        sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    total_return = float((equity[-1] - starting_capital) / starting_capital)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return,
        "num_trades": num_trades,
    }
