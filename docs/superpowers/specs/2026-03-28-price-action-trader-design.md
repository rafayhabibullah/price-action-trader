# Price Action Trader — Design Spec
**Date:** 2026-03-28
**Status:** Approved
**Scope:** Backtesting system for institutional price action strategies across stocks and crypto

---

## Overview

A new standalone Python project (`price-action-trader`) that researches, backtests, and optimizes institutional-grade price action trading strategies across stocks and crypto. Starting capital is $1,000. The system discovers the best-performing strategy/asset/timeframe/risk-model combinations through exhaustive optimization.

Data is sourced from **Twelve Data** (stocks, ETFs, forex) and **CCXT/Binance** (crypto). The project does not modify any existing trading projects but may reference patterns from `algo-trading-system` and `trading-system`.

---

## Project Structure

```
price-action-trader/
├── data/
│   ├── fetcher.py          # CCXT + Twelve Data API clients
│   ├── cache.py            # Local parquet cache (avoid re-fetching)
│   └── universe.py         # Asset universe discovery & filtering
├── strategies/
│   ├── base.py             # Abstract strategy interface
│   ├── support_resistance.py
│   ├── order_blocks.py
│   ├── fair_value_gaps.py
│   ├── candlestick_patterns.py
│   └── market_structure.py
├── backtest/
│   ├── engine.py           # Vectorized backtest runner
│   ├── positions.py        # Trade lifecycle (entry, SL, TP, trailing)
│   └── metrics.py          # Sharpe, drawdown, win rate, profit factor
├── optimization/
│   ├── grid_search.py      # Param × timeframe × asset grid
│   ├── risk_models.py      # Fixed %, Fixed $, Kelly Criterion
│   └── exit_strategies.py  # Fixed R:R, trailing stop, partial TP
├── reporting/
│   ├── results.py          # Ranked leaderboard output
│   └── charts.py           # Equity curves, trade plots
├── config/
│   └── settings.py         # API keys, capital, default params
└── main.py                 # CLI entrypoint
```

---

## Data Layer

### Asset Universe Discovery
- **Stocks:** Twelve Data screener filters US equities, ETFs, and forex by volume, market cap, and volatility. Targets liquid, tradeable instruments.
- **Crypto:** CCXT scans all USDT pairs on Binance, filters by 24h volume threshold to keep only liquid pairs.
- Universe refreshed weekly, stored in `data/cache/universe.parquet`.

### Timeframes
Backtested across: `5m`, `15m`, `1h`, `4h`, `1d`. Optimizer determines best timeframe per strategy.

### Caching
- Raw OHLCV data stored as `data/cache/<asset>/<timeframe>.parquet`
- On subsequent runs, only new candles are fetched (incremental update)
- Keeps backtests fast and reproducible

### Data Quality
- Gaps, low-volume candles, and market-hours boundaries handled explicitly
- Crypto: 24/7 continuous data
- Stocks: market hours only, weekends/holidays excluded

---

## Strategy Layer

All strategies implement a common interface:

```python
class BaseStrategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Returns df with 'signal' column: 1=buy, -1=sell, 0=hold
        ...
    def get_params(self) -> dict:
        # Returns dict of optimizable parameters and their ranges
        ...
```

### Strategies

| Strategy | Institutional Logic | Key Parameters |
|---|---|---|
| **Support & Resistance** | Price levels where institutions accumulate/distribute; tested via multiple touches | `lookback`, `touch_count`, `zone_width` |
| **Order Blocks** | Last bearish candle before a bullish impulse — where banks placed large orders (Smart Money Concept) | `block_size`, `invalidation_pct`, `mitigation_type` |
| **Fair Value Gaps (FVG)** | 3-candle imbalances where price left a gap; price returns to fill (ICT concept) | `gap_min_size`, `fill_pct`, `direction_bias` |
| **Candlestick Patterns** | Engulfing, pin bars, inside bars detected at key structure levels | `confirmation_candles`, `min_body_ratio` |
| **Market Structure** | Break of Structure (BOS), Change of Character (CHoCH), HH/HL/LL/LH tracking | `swing_lookback`, `structure_type` |

### Signal Confluence (Optional)
Strategies can be combined — e.g., only take an Order Block entry if Market Structure confirms a BOS on the higher timeframe. This filters low-probability trades, mirroring institutional desk filtering.

---

## Backtest Engine

### Execution Model
- **Vectorized** execution using pandas/numpy — entire price history processed in one pass (10-100x faster than bar-by-bar loops)
- No look-ahead bias: signals fire on candle close, fills execute on next open
- Starting capital: $1,000, compounded across trades

### Trade Lifecycle
```
Signal fires on candle close
  → Entry at next open
  → Set Stop Loss (SL) and Take Profit (TP)
  → Monitor each subsequent candle for:
      - Fixed R:R hit (1:2 or 1:3)
      - Trailing stop trigger
      - Partial TP (50% at 1R, trail remainder to breakeven)
      - SL hit (loss)
      - Max bars held (time-based exit)
  → Record trade: entry, exit, P&L, R-multiple, duration
```

---

## Optimization Layer

### Grid Search Dimensions
- 5 strategies × 5 timeframes × 3 risk models × 3 exit strategies = **225 combinations per asset**
- Asset universe: ~50 stocks + ~30 crypto pairs = **~11,250 total backtests**
- Parallel execution via `multiprocessing.Pool` — completes in minutes

### Risk Models
1. **Fixed % risk** — risk X% of current capital per trade (default 2%)
2. **Fixed $ risk** — risk fixed dollar amount per trade
3. **Kelly Criterion** — mathematically optimal position sizing based on historical win rate and R:R

### Exit Strategies
1. **Fixed R:R** — fixed take profit at 1:2 or 1:3 risk-to-reward
2. **Trailing Stop** — stop moves up with price, locks in profit
3. **Partial TP** — take 50% off at 1R, move SL to breakeven, trail remainder

### Composite Ranking Score
```
Score = 0.4 × Sharpe + 0.3 × ProfitFactor + 0.2 × WinRate + 0.1 × (1 - MaxDrawdown)
```
Mirrors how quant funds rank strategy candidates — risk-adjusted return over raw profit.

---

## Reporting

### Output
- Results saved to `reports/<run_date>/` as CSV + HTML
- **Leaderboard:** top 20 strategy/asset/timeframe combinations ranked by composite score
- **Per-trade log:** entry/exit price, P&L, R-multiple, holding duration
- **Equity curves:** matplotlib charts per top strategy
- **Summary stats:** Sharpe ratio, max drawdown, CAGR, win rate, profit factor, avg R:R

### CLI Interface
```bash
# Full optimization run across all assets, strategies, timeframes
python main.py optimize --capital 1000 --universe auto

# Backtest a specific strategy/asset/timeframe
python main.py backtest --strategy order_blocks --asset BTC/USDT --timeframe 4h

# Display leaderboard from last run
python main.py report --top 20
```

---

## Configuration

```python
# config/settings.py
TWELVE_DATA_API_KEY = "..."
CCXT_EXCHANGE = "binance"        # configurable to any CCXT-supported exchange
STARTING_CAPITAL = 1000          # USD
MAX_RISK_PER_TRADE = 0.02        # 2% default, overridden by optimizer
UNIVERSE_REFRESH_DAYS = 7        # How often to refresh asset universe
CACHE_DIR = "data/cache"
REPORTS_DIR = "reports"
```

---

## Key Design Decisions

1. **Vectorized over event-driven backtest** — speed is critical when running 11,250+ backtests. Vectorized pandas is sufficient for strategy validation; event-driven (like Zipline) adds complexity without benefit at this stage.
2. **Parquet caching** — avoids API rate limits and makes reruns instant. Incremental updates keep data fresh.
3. **Composite score** — prevents optimizing for a single metric (e.g., high win rate with terrible drawdown). Matches institutional evaluation criteria.
4. **New standalone project** — keeps concerns clean and doesn't risk destabilizing existing `algo-trading-system` or `trading-system` projects.

---

## Out of Scope (This Phase)
- Live trading / broker integration
- LLM agents
- Portfolio-level optimization (each strategy backtested independently)
- Options or futures
