import time
import click
import pandas as pd
from pathlib import Path
from datetime import datetime

from config.settings import STARTING_CAPITAL, MAX_RISK_PER_TRADE, TIMEFRAMES, REPORTS_DIR
from strategies.support_resistance import SupportResistanceStrategy
from strategies.order_blocks import OrderBlockStrategy
from strategies.fair_value_gaps import FairValueGapStrategy
from strategies.candlestick_patterns import CandlestickPatternStrategy
from strategies.market_structure import MarketStructureStrategy
from backtest.engine import BacktestEngine
from optimization.grid_search import GridSearch
from reporting.results import ResultsReporter
from reporting.charts import plot_equity_curve, plot_leaderboard_bar


ALL_STRATEGIES = [
    SupportResistanceStrategy(),
    OrderBlockStrategy(),
    FairValueGapStrategy(),
    CandlestickPatternStrategy(),
    MarketStructureStrategy(),
]


@click.group()
def cli():
    """Price Action Trader — backtest and optimize institutional strategies."""


@cli.command()
@click.option("--capital", default=STARTING_CAPITAL, type=float, help="Starting capital in USD")
@click.option("--risk", default=MAX_RISK_PER_TRADE, type=float, help="Risk per trade (fraction)")
@click.option("--universe", default="auto", help="'auto' to discover assets, or comma-separated symbols")
@click.option("--timeframes", default=",".join(TIMEFRAMES), help="Comma-separated timeframes")
@click.option("--limit", default=500, type=int, help="Candles per asset/timeframe")
def optimize(capital, risk, universe, timeframes, limit):
    """Run full optimization across all strategies, assets, and timeframes."""
    from data.fetcher import CryptoFetcher, StockFetcher
    from data.cache import CacheManager
    from data.universe import UniverseManager
    from config.settings import CACHE_DIR

    cache = CacheManager(CACHE_DIR)
    fetcher_crypto = CryptoFetcher()
    fetcher_stock = StockFetcher()
    mgr = UniverseManager()

    tfs = [t.strip() for t in timeframes.split(",")]

    if universe == "auto":
        click.echo("Discovering asset universe...")
        crypto_assets = mgr.get_crypto_universe()[:30]
        stock_assets = mgr.get_stock_universe()
    else:
        symbols = [s.strip() for s in universe.split(",")]
        crypto_assets = [s for s in symbols if "/" in s]
        stock_assets = [s for s in symbols if "/" not in s]

    click.echo(f"Fetching data for {len(crypto_assets)} crypto + {len(stock_assets)} stock assets...")

    data: dict[str, dict[str, pd.DataFrame]] = {}

    for asset in crypto_assets:
        data[asset] = {}
        for tf in tfs:
            cached = cache.read(asset, tf)
            if cached is not None:
                data[asset][tf] = cached
                continue
            try:
                df = fetcher_crypto.fetch(asset, tf, limit=limit)
                if not df.empty:
                    cache.write(asset, tf, df)
                    data[asset][tf] = df
            except Exception as e:
                click.echo(f"  Skipping {asset}/{tf}: {e}")

    for asset in stock_assets:
        data[asset] = {}
        for tf in tfs:
            cached = cache.read(asset, tf)
            if cached is not None:
                data[asset][tf] = cached
                continue
            try:
                df = fetcher_stock.fetch(asset, tf, outputsize=limit)
                if not df.empty:
                    cache.write(asset, tf, df)
                    data[asset][tf] = df
                time.sleep(8)  # Twelve Data free tier: 8 credits/min
            except Exception as e:
                click.echo(f"  Skipping {asset}/{tf}: {e}")
                time.sleep(8)

    click.echo(f"Running grid search ({len(ALL_STRATEGIES)} strategies × {len(tfs)} timeframes)...")
    gs = GridSearch(starting_capital=capital, risk_per_trade=risk)
    results = gs.run(strategies=ALL_STRATEGIES, data=data)

    if results.empty:
        click.echo("No results generated. Check data and API keys.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    reporter = ResultsReporter(REPORTS_DIR)
    out_dir = reporter.save(results, run_id=run_id)
    plot_leaderboard_bar(results, out_dir / "leaderboard_chart.png")

    click.echo(f"\nTop 10 Results:")
    top10 = reporter.top(results, n=10)
    click.echo(top10[["strategy", "asset", "timeframe", "score", "sharpe_ratio", "win_rate", "total_return"]].to_string(index=False))
    click.echo(f"\nFull results saved to {out_dir}")


@cli.command()
@click.option("--strategy", required=True, type=click.Choice(
    ["support_resistance", "order_blocks", "fair_value_gaps", "candlestick_patterns", "market_structure"]
))
@click.option("--asset", required=True, help="Asset symbol e.g. BTC/USDT or AAPL")
@click.option("--timeframe", default="1h", type=click.Choice(["5m", "15m", "1h", "4h", "1d"]))
@click.option("--capital", default=STARTING_CAPITAL, type=float)
@click.option("--risk", default=MAX_RISK_PER_TRADE, type=float)
@click.option("--limit", default=500, type=int)
def backtest(strategy, asset, timeframe, capital, risk, limit):
    """Backtest a single strategy on one asset and timeframe."""
    from data.fetcher import CryptoFetcher, StockFetcher
    from data.cache import CacheManager
    from config.settings import CACHE_DIR

    cache = CacheManager(CACHE_DIR)
    strategy_map = {
        "support_resistance": SupportResistanceStrategy(),
        "order_blocks": OrderBlockStrategy(),
        "fair_value_gaps": FairValueGapStrategy(),
        "candlestick_patterns": CandlestickPatternStrategy(),
        "market_structure": MarketStructureStrategy(),
    }
    strat = strategy_map[strategy]

    df = cache.read(asset, timeframe)
    if df is None:
        click.echo(f"Fetching {asset} {timeframe}...")
        tf_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}
        if "/" in asset:
            df = CryptoFetcher().fetch(asset, timeframe, limit=limit)
        else:
            df = StockFetcher().fetch(asset, tf_map.get(timeframe, timeframe), outputsize=limit)
        if not df.empty:
            cache.write(asset, timeframe, df)

    if df is None or df.empty:
        click.echo("No data available.")
        return

    engine = BacktestEngine(starting_capital=capital, risk_per_trade=risk)
    result = engine.run(strat, df)
    m = result["metrics"]
    trades = result["trades"]

    click.echo(f"\n{'='*50}")
    click.echo(f"Strategy: {strategy} | Asset: {asset} | TF: {timeframe}")
    click.echo(f"{'='*50}")
    click.echo(f"Trades:        {m['num_trades']}")
    click.echo(f"Win Rate:      {m['win_rate']:.1%}")
    click.echo(f"Profit Factor: {m['profit_factor']:.2f}")
    click.echo(f"Sharpe Ratio:  {m['sharpe_ratio']:.2f}")
    click.echo(f"Max Drawdown:  {m['max_drawdown']:.1%}")
    click.echo(f"Total Return:  {m['total_return']:.1%}")

    run_id = f"{strategy}_{asset.replace('/', '_')}_{timeframe}"
    reporter = ResultsReporter(REPORTS_DIR)
    out_dir = REPORTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if not trades.empty:
        plot_equity_curve(trades, title=f"{strategy} — {asset} {timeframe}", out_path=out_dir / "equity.png")
        trades.to_csv(out_dir / "trades.csv", index=False)
        click.echo(f"\nTrades and chart saved to {out_dir}")


@cli.command()
@click.option("--top", default=20, type=int, help="Number of top results to show")
@click.option("--reports-dir", default=str(REPORTS_DIR), help="Reports directory")
def report(top, reports_dir):
    """Display leaderboard from the most recent optimization run."""
    rdir = Path(reports_dir)
    if not rdir.exists():
        click.echo("No reports directory found.")
        return

    runs = sorted([d for d in rdir.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        click.echo("No runs found.")
        return

    latest = runs[0]
    csv_path = latest / "leaderboard.csv"
    if not csv_path.exists():
        click.echo(f"No leaderboard found in {latest}")
        return

    results = pd.read_csv(csv_path)
    click.echo(f"\nLeaderboard from run: {latest.name}")
    top_results = results.head(top)
    click.echo(top_results[["strategy", "asset", "timeframe", "score", "sharpe_ratio", "win_rate", "total_return"]].to_string(index=False))


@cli.command("walk-forward")
@click.option("--capital", default=STARTING_CAPITAL, type=float, help="Starting capital in USD")
@click.option("--risk", default=MAX_RISK_PER_TRADE, type=float, help="Risk per trade (fraction)")
@click.option("--universe", default="auto", help="'auto' or comma-separated symbols")
@click.option("--timeframes", default="1h,4h,1d", help="Comma-separated timeframes (1h+ recommended)")
@click.option("--train-bars", default=200, type=int, help="Training window size in bars")
@click.option("--test-bars", default=50, type=int, help="Test window size in bars")
@click.option("--step-bars", default=50, type=int, help="Step size between windows in bars")
def walk_forward(capital, risk, universe, timeframes, train_bars, test_bars, step_bars):
    """Run walk-forward analysis to validate strategies out-of-sample."""
    from data.fetcher import CryptoFetcher, StockFetcher
    from data.cache import CacheManager
    from data.universe import UniverseManager
    from config.settings import CACHE_DIR
    from optimization.walk_forward import WalkForwardEngine

    cache = CacheManager(CACHE_DIR)
    fetcher_crypto = CryptoFetcher()
    fetcher_stock = StockFetcher()
    mgr = UniverseManager()

    tfs = [t.strip() for t in timeframes.split(",")]

    if universe == "auto":
        click.echo("Discovering asset universe...")
        crypto_assets = mgr.get_crypto_universe()[:30]
        stock_assets = mgr.get_stock_universe()
    else:
        symbols = [s.strip() for s in universe.split(",")]
        crypto_assets = [s for s in symbols if "/" in s]
        stock_assets = [s for s in symbols if "/" not in s]

    click.echo(f"Loading data for {len(crypto_assets)} crypto + {len(stock_assets)} stock assets...")

    data: dict[str, dict[str, pd.DataFrame]] = {}

    for asset in crypto_assets:
        data[asset] = {}
        for tf in tfs:
            cached = cache.read(asset, tf)
            if cached is not None and len(cached) >= train_bars + test_bars:
                data[asset][tf] = cached
            else:
                click.echo(f"  Skipping {asset}/{tf}: insufficient cached data ({len(cached) if cached is not None else 0} bars, need {train_bars + test_bars})")

    for asset in stock_assets:
        data[asset] = {}
        for tf in tfs:
            cached = cache.read(asset, tf)
            if cached is not None and len(cached) >= train_bars + test_bars:
                data[asset][tf] = cached
            else:
                click.echo(f"  Skipping {asset}/{tf}: insufficient cached data ({len(cached) if cached is not None else 0} bars, need {train_bars + test_bars})")

    click.echo(f"\nRunning walk-forward (train={train_bars}, test={test_bars}, step={step_bars})...")
    click.echo(f"Strategies: {len(ALL_STRATEGIES)} | Assets: {sum(1 for a in data if data[a])} | Timeframes: {tfs}")

    wf = WalkForwardEngine(
        starting_capital=capital,
        risk_per_trade=risk,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
    )
    results = wf.run(strategies=ALL_STRATEGIES, data=data)

    if not results:
        click.echo("No walk-forward results generated. Check data availability.")
        return

    agg = wf.aggregate(results)

    click.echo(f"\n{'='*60}")
    click.echo(f"WALK-FORWARD RESULTS ({agg['num_windows']} windows)")
    click.echo(f"{'='*60}")
    click.echo(f"Mean Train Score:       {agg['mean_train_score']:.4f}")
    click.echo(f"Mean OOS Score:         {agg['mean_oos_score']:.4f}")
    click.echo(f"OOS Degradation:        {agg['oos_degradation']:.1%}")
    click.echo(f"Mean OOS Sharpe:        {agg['mean_oos_sharpe']:.2f}")
    click.echo(f"Mean OOS Win Rate:      {agg['mean_oos_win_rate']:.1%}")
    click.echo(f"Mean OOS Return:        {agg['mean_oos_return']:.1%}")
    click.echo(f"OOS Positive Windows:   {agg['oos_positive_pct']:.0%}")

    # Per-window detail
    click.echo(f"\n{'='*60}")
    click.echo(f"PER-WINDOW DETAIL")
    click.echo(f"{'='*60}")
    rows = []
    for r in results:
        rows.append({
            "window": r.window_id,
            "asset": r.best_asset,
            "tf": r.best_timeframe,
            "strategy": r.best_strategy[:20],
            "train_score": round(r.train_score, 4),
            "oos_score": round(r.oos_score, 4),
            "oos_sharpe": round(r.oos_metrics["sharpe_ratio"], 2),
            "oos_wr": f"{r.oos_metrics['win_rate']:.0%}",
            "oos_return": f"{r.oos_metrics['total_return']:.1%}",
            "oos_trades": r.oos_metrics["num_trades"],
        })
    detail_df = pd.DataFrame(rows)
    click.echo(detail_df.to_string(index=False))

    # Save results
    run_id = f"wf_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPORTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(out_dir / "walk_forward_detail.csv", index=False)
    pd.DataFrame([agg]).to_csv(out_dir / "walk_forward_summary.csv", index=False)
    click.echo(f"\nResults saved to {out_dir}")

    # Overfitting warning
    if agg["oos_degradation"] > 0.5:
        click.echo("\n** WARNING: OOS degradation > 50% — likely overfitting. Consider simpler strategies or longer train windows.")
    elif agg["oos_degradation"] > 0.3:
        click.echo("\n** CAUTION: OOS degradation 30-50% — moderate overfitting risk. Validate with more data.")
    else:
        click.echo("\n** GOOD: OOS degradation < 30% — strategies show reasonable out-of-sample stability.")


if __name__ == "__main__":
    cli()
