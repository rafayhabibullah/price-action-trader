import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_equity_curve(trades: pd.DataFrame, title: str, out_path: Path) -> None:
    """Plot equity curve from a trades DataFrame."""
    if trades.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    equity = trades["equity"]
    axes[0].plot(range(len(equity)), equity.values, color="steelblue", linewidth=1.5)
    axes[0].set_title(title)
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(alpha=0.3)

    pnl_colors = ["green" if p > 0 else "red" for p in trades["pnl"]]
    axes[1].bar(range(len(trades)), trades["pnl"].values, color=pnl_colors, alpha=0.7)
    axes[1].set_ylabel("P&L per Trade ($)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_leaderboard_bar(results: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    """Bar chart of top N strategies by composite score."""
    top = results.head(top_n).copy()
    labels = [f"{r['strategy']}\n{r['asset']}\n{r['timeframe']}" for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(range(len(top)), top["score"].values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Composite Score")
    ax.set_title(f"Top {top_n} Strategy Combinations")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
