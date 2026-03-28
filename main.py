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

    tf_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}
    for asset in stock_assets:
        data[asset] = {}
        for tf in tfs:
            cached = cache.read(asset, tf)
            if cached is not None:
                data[asset][tf] = cached
                continue
            try:
                df = fetcher_stock.fetch(asset, tf_map.get(tf, tf), outputsize=limit)
                if not df.empty:
                    cache.write(asset, tf, df)
                    data[asset][tf] = df
            except Exception as e:
                click.echo(f"  Skipping {asset}/{tf}: {e}")

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


if __name__ == "__main__":
    cli()
