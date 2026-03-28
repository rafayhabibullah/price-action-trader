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


STRATEGY_MAP = {
    "support_resistance": SupportResistanceStrategy,
    "order_blocks": OrderBlockStrategy,
    "fair_value_gaps": FairValueGapStrategy,
    "candlestick_patterns": CandlestickPatternStrategy,
    "market_structure": MarketStructureStrategy,
}


@cli.command()
@click.option("--capital", default=STARTING_CAPITAL, type=float, help="Starting capital in USD")
@click.option("--risk", default=MAX_RISK_PER_TRADE, type=float, help="Risk per trade (fraction of capital)")
@click.option("--max-concurrent", default=5, type=int, help="Max positions open at once")
@click.option("--days", default=60, type=int, help="Lookback period in days")
@click.option("--universe", default="auto", help="'auto' or comma-separated symbols")
@click.option("--timeframes", default="1h,4h,1d", help="Comma-separated timeframes")
@click.option("--strategies", "strategy_names", default="all",
              help="Comma-separated strategy names or 'all'. "
                   "Available: support_resistance,order_blocks,fair_value_gaps,candlestick_patterns,market_structure")
@click.option("--max-bars", default=100, type=int, help="Max bars before force-closing a trade")
def portfolio(capital, risk, max_concurrent, days, universe, timeframes, strategy_names, max_bars):
    """Run portfolio-level backtest across all assets on a shared timeline."""
    from data.cache import CacheManager
    from data.universe import UniverseManager
    from config.settings import CACHE_DIR
    from backtest.portfolio import run_portfolio_backtest

    cache = CacheManager(CACHE_DIR)
    tfs = [t.strip() for t in timeframes.split(",")]

    # Select strategies
    if strategy_names == "all":
        strategies = ALL_STRATEGIES
    else:
        names = [s.strip() for s in strategy_names.split(",")]
        strategies = []
        for name in names:
            if name not in STRATEGY_MAP:
                click.echo(f"Unknown strategy: {name}. Available: {', '.join(STRATEGY_MAP)}")
                return
            strategies.append(STRATEGY_MAP[name]())
    click.echo(f"Strategies: {[s.__class__.__name__ for s in strategies]}")

    # Resolve universe
    if universe == "auto":
        mgr = UniverseManager()
        crypto_assets = mgr.get_crypto_universe()[:30]
        stock_assets = mgr.get_stock_universe()
    else:
        symbols = [s.strip() for s in universe.split(",")]
        crypto_assets = [s for s in symbols if "/" in s]
        stock_assets = [s for s in symbols if "/" not in s]
    all_assets = crypto_assets + stock_assets

    # Load cached data
    cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=days)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for asset in all_assets:
        data[asset] = {}
        for tf in tfs:
            df = cache.read(asset, tf)
            if df is None or df.empty:
                continue
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            df_filtered = df[df.index >= cutoff]
            if len(df_filtered) >= 30:
                data[asset][tf] = df_filtered

    loaded = sum(1 for a in data for tf in data[a])
    click.echo(f"Loaded {loaded} asset/timeframe combos ({days}-day lookback)")

    if loaded == 0:
        click.echo("No data available. Run fetch_historical.py first.")
        return

    # Use large capital + tiny risk to get flat dollar sizing
    # The base must be large enough that PnL never moves it more than ~1%
    flat_risk_dollar = capital * risk
    big_capital = 1_000_000_000.0
    tiny_risk = flat_risk_dollar / big_capital

    click.echo(f"Running portfolio backtest (max {max_concurrent} concurrent, ${flat_risk_dollar:.0f} risk/trade)...")

    result = run_portfolio_backtest(
        strategies=strategies,
        data=data,
        starting_capital=big_capital,
        risk_per_trade=tiny_risk,
        max_concurrent=max_concurrent,
        max_bars=max_bars,
    )

    trades = result["trades"]
    eq = result["equity_curve"]

    if trades.empty:
        click.echo("No trades generated.")
        return

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    total_pnl = trades["pnl"].sum()
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")

    cum_pnl = trades.sort_values("exit_time")["pnl"].cumsum()
    eq_series = capital + cum_pnl
    max_dd = ((eq_series.cummax() - eq_series) / eq_series.cummax()).max()

    click.echo(f"\n{'='*70}")
    click.echo(f"PORTFOLIO BACKTEST  |  ${capital:,.0f}  |  ${flat_risk_dollar:.0f}/trade  |  Max {max_concurrent} concurrent  |  {days}d")
    click.echo(f"{'='*70}")
    click.echo(f"Total Trades:       {len(trades)}")
    click.echo(f"Win Rate:           {len(wins)}/{len(trades)} = {len(wins)/len(trades):.1%}")
    click.echo(f"Profit Factor:      {pf:.2f}")
    click.echo(f"Avg R-Multiple:     {trades['r_multiple'].mean():.2f}R")
    click.echo(f"Avg Win:            ${wins['pnl'].mean():.2f}")
    click.echo(f"Avg Loss:           ${losses['pnl'].mean():.2f}")
    click.echo(f"Total PnL:          ${total_pnl:,.2f}")
    click.echo(f"Final Equity:       ${capital + total_pnl:,.2f}")
    click.echo(f"Return:             {total_pnl/capital:.1%}")
    click.echo(f"Max Drawdown:       {max_dd:.1%}")
    if not eq.empty:
        click.echo(f"Max Concurrent:     {eq['open_positions'].max()}")
        click.echo(f"Avg Concurrent:     {eq['open_positions'].mean():.1f}")

    # Per-asset breakdown
    click.echo(f"\n{'='*70}")
    click.echo("PER-ASSET BREAKDOWN")
    click.echo(f"{'='*70}")
    rows = []
    for asset in sorted(trades["asset"].unique()):
        at = trades[trades["asset"] == asset]
        w = len(at[at["pnl"] > 0])
        rows.append({
            "asset": asset,
            "trades": len(at),
            "W": w,
            "L": len(at) - w,
            "wr": f"{w/len(at):.0%}",
            "avg_R": round(at["r_multiple"].mean(), 1),
            "pnl": round(at["pnl"].sum(), 2),
        })
    click.echo(pd.DataFrame(rows).to_string(index=False))

    # Per-strategy breakdown
    click.echo(f"\n{'='*70}")
    click.echo("PER-STRATEGY BREAKDOWN")
    click.echo(f"{'='*70}")
    rows = []
    for strat in sorted(trades["strategy"].unique()):
        st = trades[trades["strategy"] == strat]
        w = len(st[st["pnl"] > 0])
        rows.append({
            "strategy": strat,
            "trades": len(st),
            "W": w,
            "L": len(st) - w,
            "wr": f"{w/len(st):.0%}",
            "avg_R": round(st["r_multiple"].mean(), 1),
            "pnl": round(st["pnl"].sum(), 2),
        })
    click.echo(pd.DataFrame(rows).to_string(index=False))

    # Save results
    run_id = f"portfolio_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPORTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    if not eq.empty:
        eq.to_csv(out_dir / "equity_curve.csv", index=False)
    click.echo(f"\nResults saved to {out_dir}")


@cli.command("paper-trade")
def paper_trade():
    """Run one paper trading scan cycle (local test)."""
    from dataclasses import asdict
    import json
    from trading.alpaca_client import AlpacaClient
    from trading.scanner import scan_all
    from trading.position_manager import run_cycle
    from trading import config as tconfig
    from strategies.support_resistance import SupportResistanceStrategy
    from strategies.fair_value_gaps import FairValueGapStrategy
    from strategies.candlestick_patterns import CandlestickPatternStrategy
    from strategies.market_structure import MarketStructureStrategy

    strategies = [
        SupportResistanceStrategy(),
        FairValueGapStrategy(),
        CandlestickPatternStrategy(),
        MarketStructureStrategy(),
    ]

    client = AlpacaClient(
        api_key=tconfig.ALPACA_API_KEY,
        secret_key=tconfig.ALPACA_SECRET_KEY,
        base_url=tconfig.ALPACA_BASE_URL,
    )

    click.echo("Running paper trading cycle...")
    result = run_cycle(client, scan_fn=scan_all, strategies=strategies)
    click.echo(json.dumps(asdict(result), indent=2))


@cli.command("paper-status")
def paper_status():
    """Show current paper trading account status and open positions."""
    from trading.alpaca_client import AlpacaClient
    from trading import config as tconfig
    from trading.symbols import to_internal

    client = AlpacaClient(
        api_key=tconfig.ALPACA_API_KEY,
        secret_key=tconfig.ALPACA_SECRET_KEY,
        base_url=tconfig.ALPACA_BASE_URL,
    )

    acct = client.get_account()
    click.echo(f"\n{'='*60}")
    click.echo(f"ALPACA PAPER ACCOUNT")
    click.echo(f"{'='*60}")
    click.echo(f"Equity:        ${float(acct.get('equity', 0)):>12,.2f}")
    click.echo(f"Cash:          ${float(acct.get('cash', 0)):>12,.2f}")
    click.echo(f"Buying Power:  ${float(acct.get('buying_power', 0)):>12,.2f}")
    click.echo(f"Market Open:   {client.is_market_open()}")

    positions = client.get_positions()
    click.echo(f"\n{'='*60}")
    click.echo(f"OPEN POSITIONS ({len(positions)})")
    click.echo(f"{'='*60}")
    if not positions:
        click.echo("  No open positions.")
    else:
        for pos in positions:
            sym = pos.get("symbol", "")
            side = pos.get("side", "")
            qty = pos.get("qty", "")
            pnl = float(pos.get("unrealized_pl", 0))
            entry = float(pos.get("avg_entry_price", 0))
            click.echo(
                f"  {to_internal(sym):12s}  {side:5s}  qty:{qty:>10}  "
                f"entry:${entry:>10,.2f}  uPnL:${pnl:>+10,.2f}"
            )

    orders = client.get_orders(status="open")
    click.echo(f"\n{'='*60}")
    click.echo(f"OPEN ORDERS ({len(orders)})")
    click.echo(f"{'='*60}")
    if not orders:
        click.echo("  No open orders.")
    else:
        for order in orders[:20]:
            sym = order.get("symbol", "")
            side = order.get("side", "")
            otype = order.get("order_class", order.get("type", ""))
            status = order.get("status", "")
            click.echo(f"  {to_internal(sym):12s}  {side:5s}  {otype:10s}  {status}")


if __name__ == "__main__":
    cli()
