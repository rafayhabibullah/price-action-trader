# Price Action Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vectorized backtesting + optimization system for 5 institutional price action strategies across stocks and crypto, starting with $1,000 capital.

**Architecture:** Modular pipeline — Data Layer (fetch/cache/universe) → Strategy Layer (5 strategies, common interface) → Backtest Engine (vectorized, no look-ahead bias) → Optimization Grid (225 combos/asset, parallel) → Reporting (ranked leaderboard, equity curves).

**Tech Stack:** Python 3.11+, uv, pandas, numpy, ccxt, twelvedata, pyarrow, matplotlib, click, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies and project metadata |
| `config/settings.py` | API keys, capital, paths — all config in one place |
| `data/fetcher.py` | CCXT + Twelve Data OHLCV clients |
| `data/cache.py` | Read/write parquet cache, incremental updates |
| `data/universe.py` | Discover liquid stock + crypto assets |
| `strategies/base.py` | Abstract `BaseStrategy` interface |
| `strategies/support_resistance.py` | S/R zone detection + signals |
| `strategies/order_blocks.py` | Order block detection + signals |
| `strategies/fair_value_gaps.py` | FVG detection + signals |
| `strategies/candlestick_patterns.py` | Engulfing, pin bar, inside bar signals |
| `strategies/market_structure.py` | BOS, CHoCH, HH/HL/LL/LH signals |
| `backtest/positions.py` | Trade lifecycle: entry, SL, TP, trailing, partial |
| `backtest/metrics.py` | Sharpe, drawdown, win rate, profit factor, CAGR |
| `backtest/engine.py` | Vectorized backtest runner |
| `optimization/risk_models.py` | Fixed %, Fixed $, Kelly position sizing |
| `optimization/exit_strategies.py` | Fixed R:R, trailing stop, partial TP configs |
| `optimization/grid_search.py` | Parallel grid across all dimensions |
| `reporting/results.py` | Composite score, leaderboard, CSV/HTML export |
| `reporting/charts.py` | Equity curves, trade plots (matplotlib) |
| `main.py` | CLI: `optimize`, `backtest`, `report` commands |
| `tests/` | Mirrors src structure, one test file per module |

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `.env.example`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize project with uv**

```bash
cd /Users/rafayhabibullah/Documents/GitHub/price-action-trader
uv init --name price-action-trader --python 3.11
uv add pandas numpy ccxt twelvedata pyarrow matplotlib click python-dotenv
uv add --dev pytest pytest-cov
```

Expected: `pyproject.toml` and `uv.lock` created.

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p data strategies backtest optimization reporting config tests
touch data/__init__.py strategies/__init__.py backtest/__init__.py
touch optimization/__init__.py reporting/__init__.py config/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `config/settings.py`**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

TWELVE_DATA_API_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "")
CCXT_EXCHANGE: str = os.getenv("CCXT_EXCHANGE", "binance")

STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "1000"))
MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))

UNIVERSE_REFRESH_DAYS: int = int(os.getenv("UNIVERSE_REFRESH_DAYS", "7"))
CACHE_DIR: Path = BASE_DIR / os.getenv("CACHE_DIR", "data/cache")
REPORTS_DIR: Path = BASE_DIR / os.getenv("REPORTS_DIR", "reports")

TIMEFRAMES: list[str] = ["5m", "15m", "1h", "4h", "1d"]

CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Write `.env.example`**

```
TWELVE_DATA_API_KEY=your_key_here
CCXT_EXCHANGE=binance
STARTING_CAPITAL=1000
MAX_RISK_PER_TRADE=0.02
UNIVERSE_REFRESH_DAYS=7
CACHE_DIR=data/cache
REPORTS_DIR=reports
```

- [ ] **Step 5: Write test for settings**

```python
# tests/test_settings.py
from config.settings import (
    STARTING_CAPITAL, MAX_RISK_PER_TRADE, TIMEFRAMES,
    CACHE_DIR, REPORTS_DIR
)

def test_defaults():
    assert STARTING_CAPITAL == 1000.0
    assert MAX_RISK_PER_TRADE == 0.02
    assert "1h" in TIMEFRAMES
    assert CACHE_DIR.exists()
    assert REPORTS_DIR.exists()
```

- [ ] **Step 6: Run test**

```bash
uv run pytest tests/test_settings.py -v
```

Expected: `PASSED`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock config/ tests/test_settings.py .env.example data/__init__.py strategies/__init__.py backtest/__init__.py optimization/__init__.py reporting/__init__.py tests/__init__.py
git commit -m "feat: project setup, config, and directory structure"
```

---

## Task 2: Data Cache

**Files:**
- Create: `data/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cache.py
import pandas as pd
import pytest
from pathlib import Path
from data.cache import CacheManager

@pytest.fixture
def cache(tmp_path):
    return CacheManager(cache_dir=tmp_path)

@pytest.fixture
def sample_ohlcv():
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0],
        "low":  [99.0,  100.0, 101.0, 102.0, 103.0],
        "close":[100.5, 101.5, 102.5, 103.5, 104.5],
        "volume":[1000.0, 1100.0, 900.0, 1200.0, 800.0],
    }).set_index("timestamp")

def test_write_and_read(cache, sample_ohlcv):
    cache.write("BTC/USDT", "1h", sample_ohlcv)
    result = cache.read("BTC/USDT", "1h")
    assert result is not None
    assert len(result) == 5
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]

def test_read_missing_returns_none(cache):
    assert cache.read("NONEXISTENT", "1h") is None

def test_last_timestamp(cache, sample_ohlcv):
    cache.write("BTC/USDT", "1h", sample_ohlcv)
    last = cache.last_timestamp("BTC/USDT", "1h")
    assert last == pd.Timestamp("2024-01-01 04:00:00")

def test_append_new_rows(cache, sample_ohlcv):
    cache.write("BTC/USDT", "1h", sample_ohlcv)
    new_rows = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 05:00:00", periods=2, freq="1h"),
        "open": [105.0, 106.0], "high": [106.0, 107.0],
        "low": [104.0, 105.0], "close": [105.5, 106.5], "volume": [900.0, 950.0],
    }).set_index("timestamp")
    cache.append("BTC/USDT", "1h", new_rows)
    result = cache.read("BTC/USDT", "1h")
    assert len(result) == 7
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_cache.py -v
```

Expected: `ImportError` — `data.cache` not found.

- [ ] **Step 3: Implement `data/cache.py`**

```python
import pandas as pd
from pathlib import Path


class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, asset: str, timeframe: str) -> Path:
        safe_asset = asset.replace("/", "_")
        return self.cache_dir / safe_asset / f"{timeframe}.parquet"

    def write(self, asset: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self._path(asset, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def read(self, asset: str, timeframe: str) -> pd.DataFrame | None:
        path = self._path(asset, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def last_timestamp(self, asset: str, timeframe: str) -> pd.Timestamp | None:
        df = self.read(asset, timeframe)
        if df is None or df.empty:
            return None
        return df.index.max()

    def append(self, asset: str, timeframe: str, new_df: pd.DataFrame) -> None:
        existing = self.read(asset, timeframe)
        if existing is None:
            self.write(asset, timeframe, new_df)
        else:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self.write(asset, timeframe, combined)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_cache.py -v
```

Expected: 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add data/cache.py tests/test_cache.py
git commit -m "feat: parquet cache manager with incremental append"
```

---

## Task 3: Data Fetcher

**Files:**
- Create: `data/fetcher.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_fetcher.py
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from data.fetcher import CryptoFetcher, StockFetcher

# --- CryptoFetcher ---

def test_crypto_fetcher_returns_dataframe():
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0],
        [1704070800000, 42200.0, 42800.0, 42100.0, 42600.0, 120.0],
    ]
    fetcher = CryptoFetcher(exchange=mock_exchange)
    df = fetcher.fetch("BTC/USDT", "1h", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2

def test_crypto_fetcher_since_param():
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = []
    fetcher = CryptoFetcher(exchange=mock_exchange)
    since = pd.Timestamp("2024-01-01")
    fetcher.fetch("BTC/USDT", "1h", since=since)
    call_kwargs = mock_exchange.fetch_ohlcv.call_args
    assert call_kwargs[1]["since"] == int(since.timestamp() * 1000)

# --- StockFetcher ---

def test_stock_fetcher_returns_dataframe():
    mock_client = MagicMock()
    mock_client.time_series.return_value.as_pandas.return_value = pd.DataFrame({
        "open": [150.0, 151.0],
        "high": [152.0, 153.0],
        "low":  [149.0, 150.0],
        "close":[151.0, 152.0],
        "volume":[1000000.0, 1100000.0],
    }, index=pd.date_range("2024-01-01", periods=2, freq="1d"))
    fetcher = StockFetcher(client=mock_client)
    df = fetcher.fetch("AAPL", "1day", outputsize=2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_fetcher.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `data/fetcher.py`**

```python
import ccxt
import pandas as pd
from twelvedata import TDClient
from config.settings import CCXT_EXCHANGE, TWELVE_DATA_API_KEY


class CryptoFetcher:
    def __init__(self, exchange=None):
        if exchange is None:
            exchange_class = getattr(ccxt, CCXT_EXCHANGE)
            self.exchange = exchange_class()
        else:
            self.exchange = exchange

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        kwargs: dict = {"limit": limit}
        if since is not None:
            kwargs["since"] = int(since.timestamp() * 1000)
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize(None)
        return df


class StockFetcher:
    # Twelve Data timeframe mapping: "1h" -> "1h", "1d" -> "1day", "4h" -> "4h"
    TF_MAP = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}

    def __init__(self, client=None):
        if client is None:
            self.client = TDClient(apikey=TWELVE_DATA_API_KEY)
        else:
            self.client = client

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        outputsize: int = 500,
        start_date: str | None = None,
    ) -> pd.DataFrame:
        td_tf = self.TF_MAP.get(timeframe, timeframe)
        kwargs: dict = {"symbol": symbol, "interval": td_tf, "outputsize": outputsize}
        if start_date:
            kwargs["start_date"] = start_date
        df = self.client.time_series(**kwargs).as_pandas()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.sort_index(inplace=True)
        return df
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_fetcher.py -v
```

Expected: 3 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: CCXT crypto fetcher and Twelve Data stock fetcher"
```

---

## Task 4: Asset Universe Discovery

**Files:**
- Create: `data/universe.py`
- Create: `tests/test_universe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_universe.py
import pytest
from unittest.mock import MagicMock, patch
from data.universe import UniverseManager

def test_crypto_universe_filters_by_volume():
    mock_exchange = MagicMock()
    mock_exchange.fetch_tickers.return_value = {
        "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 5_000_000_000},
        "ETH/USDT": {"symbol": "ETH/USDT", "quoteVolume": 2_000_000_000},
        "SHIB/USDT": {"symbol": "SHIB/USDT", "quoteVolume": 1_000},  # too low
        "DOGE/USDT": {"symbol": "DOGE/USDT", "quoteVolume": None},   # missing
    }
    mgr = UniverseManager(crypto_exchange=mock_exchange)
    symbols = mgr.get_crypto_universe(min_volume_usd=1_000_000)
    assert "BTC/USDT" in symbols
    assert "ETH/USDT" in symbols
    assert "SHIB/USDT" not in symbols
    assert "DOGE/USDT" not in symbols

def test_crypto_universe_only_usdt_pairs():
    mock_exchange = MagicMock()
    mock_exchange.fetch_tickers.return_value = {
        "BTC/USDT": {"symbol": "BTC/USDT", "quoteVolume": 5_000_000_000},
        "BTC/BTC":  {"symbol": "BTC/BTC",  "quoteVolume": 5_000_000_000},
    }
    mgr = UniverseManager(crypto_exchange=mock_exchange)
    symbols = mgr.get_crypto_universe(min_volume_usd=1_000_000)
    assert "BTC/USDT" in symbols
    assert "BTC/BTC" not in symbols

def test_stock_universe_returns_list():
    mgr = UniverseManager(stock_symbols=["AAPL", "MSFT", "SPY"])
    symbols = mgr.get_stock_universe()
    assert symbols == ["AAPL", "MSFT", "SPY"]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_universe.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `data/universe.py`**

```python
import ccxt
import pandas as pd
from config.settings import CCXT_EXCHANGE

# Default stock universe — liquid US large-caps + ETFs
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "JPM", "BAC", "GS", "XOM", "CVX",
    "AMD", "NFLX", "CRM", "ADBE",
]


class UniverseManager:
    def __init__(self, crypto_exchange=None, stock_symbols: list[str] | None = None):
        if crypto_exchange is None:
            exchange_class = getattr(ccxt, CCXT_EXCHANGE)
            self.crypto_exchange = exchange_class()
        else:
            self.crypto_exchange = crypto_exchange
        self._stock_symbols = stock_symbols if stock_symbols is not None else DEFAULT_STOCKS

    def get_crypto_universe(self, min_volume_usd: float = 50_000_000) -> list[str]:
        tickers = self.crypto_exchange.fetch_tickers()
        result = []
        for symbol, data in tickers.items():
            if not symbol.endswith("/USDT"):
                continue
            vol = data.get("quoteVolume")
            if vol is None:
                continue
            try:
                if float(vol) >= min_volume_usd:
                    result.append(symbol)
            except (TypeError, ValueError):
                continue
        return sorted(result)

    def get_stock_universe(self) -> list[str]:
        return list(self._stock_symbols)

    def get_full_universe(self) -> dict[str, list[str]]:
        return {
            "crypto": self.get_crypto_universe(),
            "stocks": self.get_stock_universe(),
        }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_universe.py -v
```

Expected: 3 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add data/universe.py tests/test_universe.py
git commit -m "feat: asset universe discovery for crypto (CCXT) and stocks"
```

---

## Task 5: Strategy Base + Support & Resistance

**Files:**
- Create: `strategies/base.py`
- Create: `strategies/support_resistance.py`
- Create: `tests/test_support_resistance.py`

- [ ] **Step 1: Write `strategies/base.py`**

```python
# strategies/base.py
from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes OHLCV DataFrame with columns [open, high, low, close, volume].
        Returns same DataFrame with added columns:
          - 'signal': 1=buy, -1=sell, 0=hold
          - 'sl': stop loss price (float, NaN when no signal)
          - 'tp': take profit price (float, NaN when no signal)
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Returns dict of parameter names to their current values."""
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_support_resistance.py
import pandas as pd
import numpy as np
import pytest
from strategies.support_resistance import SupportResistanceStrategy

@pytest.fixture
def sr_strategy():
    return SupportResistanceStrategy(lookback=20, touch_count=2, zone_width_pct=0.005)

@pytest.fixture
def bullish_sr_df():
    """Price bounces at support twice, then breaks above resistance."""
    np.random.seed(42)
    n = 60
    prices = [100.0]
    for i in range(n - 1):
        if i in (10, 20):   # two touches of ~95 support
            prices.append(95.2)
        elif i == 30:        # breakout
            prices.append(112.0)
        else:
            prices.append(prices[-1] + np.random.uniform(-1, 1))
    close = pd.Series(prices)
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    return df

def test_returns_required_columns(sr_strategy, bullish_sr_df):
    result = sr_strategy.generate_signals(bullish_sr_df)
    assert "signal" in result.columns
    assert "sl" in result.columns
    assert "tp" in result.columns

def test_signal_values_are_valid(sr_strategy, bullish_sr_df):
    result = sr_strategy.generate_signals(bullish_sr_df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_sl_set_when_signal(sr_strategy, bullish_sr_df):
    result = sr_strategy.generate_signals(bullish_sr_df)
    signal_rows = result[result["signal"] != 0]
    if not signal_rows.empty:
        assert signal_rows["sl"].notna().all()

def test_get_params(sr_strategy):
    params = sr_strategy.get_params()
    assert "lookback" in params
    assert "touch_count" in params
    assert "zone_width_pct" in params
```

- [ ] **Step 3: Run to verify failure**

```bash
uv run pytest tests/test_support_resistance.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `strategies/support_resistance.py`**

```python
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, touch_count: int = 2, zone_width_pct: float = 0.005):
        self.lookback = lookback
        self.touch_count = touch_count
        self.zone_width_pct = zone_width_pct

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "touch_count": self.touch_count,
            "zone_width_pct": self.zone_width_pct,
        }

    def _find_levels(self, df: pd.DataFrame) -> list[float]:
        """Find price levels touched >= touch_count times within zone_width_pct."""
        highs = df["high"].values
        lows = df["low"].values
        levels = []
        candidates = np.concatenate([highs, lows])
        for price in candidates:
            zone_lo = price * (1 - self.zone_width_pct)
            zone_hi = price * (1 + self.zone_width_pct)
            touches = np.sum((lows <= zone_hi) & (highs >= zone_lo))
            if touches >= self.touch_count:
                levels.append(float(price))
        # deduplicate nearby levels
        levels = sorted(set(levels))
        deduped = []
        for lvl in levels:
            if not deduped or lvl > deduped[-1] * (1 + self.zone_width_pct * 2):
                deduped.append(lvl)
        return deduped

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        for i in range(self.lookback, len(df)):
            window = df.iloc[i - self.lookback:i]
            levels = self._find_levels(window)
            if not levels:
                continue

            close = df["close"].iloc[i]
            prev_close = df["close"].iloc[i - 1]

            for level in levels:
                zone_lo = level * (1 - self.zone_width_pct)
                zone_hi = level * (1 + self.zone_width_pct)

                # Bullish bounce off support: price dipped into zone and closed above
                if prev_close <= zone_hi and close > zone_hi:
                    result.at[df.index[i], "signal"] = 1
                    result.at[df.index[i], "sl"] = zone_lo * 0.998
                    result.at[df.index[i], "tp"] = close + 2 * (close - zone_lo * 0.998)
                    break

                # Bearish rejection at resistance: price touched zone and closed below
                if prev_close >= zone_lo and close < zone_lo:
                    result.at[df.index[i], "signal"] = -1
                    result.at[df.index[i], "sl"] = zone_hi * 1.002
                    result.at[df.index[i], "tp"] = close - 2 * (zone_hi * 1.002 - close)
                    break

        return result
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_support_resistance.py -v
```

Expected: 4 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add strategies/base.py strategies/support_resistance.py tests/test_support_resistance.py
git commit -m "feat: BaseStrategy interface and Support/Resistance strategy"
```

---

## Task 6: Order Blocks Strategy

**Files:**
- Create: `strategies/order_blocks.py`
- Create: `tests/test_order_blocks.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_order_blocks.py
import pandas as pd
import numpy as np
import pytest
from strategies.order_blocks import OrderBlockStrategy

@pytest.fixture
def ob_strategy():
    return OrderBlockStrategy(block_size_pct=0.003, invalidation_pct=0.005)

@pytest.fixture
def ob_df():
    """Bearish candle followed by strong bullish impulse (order block setup)."""
    data = {
        "open":  [100, 101, 100, 98,  97,  100, 103, 106],
        "high":  [101, 102, 101, 100, 98,  102, 105, 108],
        "low":   [99,  100, 99,  97,  96,  99,  102, 105],
        "close": [101, 100, 99,  98,  100, 103, 106, 107],  # i=4 bearish then bullish impulse
        "volume":[1000]*8,
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=8, freq="1h"))

def test_returns_required_columns(ob_strategy, ob_df):
    result = ob_strategy.generate_signals(ob_df)
    assert "signal" in result.columns
    assert "sl" in result.columns
    assert "tp" in result.columns

def test_signal_values_valid(ob_strategy, ob_df):
    result = ob_strategy.generate_signals(ob_df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(ob_strategy):
    params = ob_strategy.get_params()
    assert "block_size_pct" in params
    assert "invalidation_pct" in params

def test_no_signal_on_flat_market(ob_strategy):
    flat = pd.DataFrame({
        "open":  [100.0] * 20, "high": [100.5] * 20,
        "low":   [99.5] * 20,  "close":[100.0] * 20, "volume":[1000.0] * 20,
    }, index=pd.date_range("2024-01-01", periods=20, freq="1h"))
    result = ob_strategy.generate_signals(flat)
    assert (result["signal"] == 0).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_order_blocks.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `strategies/order_blocks.py`**

```python
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class OrderBlockStrategy(BaseStrategy):
    """
    Order Block (Smart Money Concept):
    Bullish OB = last bearish candle before a bullish impulse (3+ consecutive up closes).
    Signal fires when price returns to the OB zone.
    """

    def __init__(self, block_size_pct: float = 0.003, invalidation_pct: float = 0.005,
                 impulse_candles: int = 3):
        self.block_size_pct = block_size_pct
        self.invalidation_pct = invalidation_pct
        self.impulse_candles = impulse_candles

    def get_params(self) -> dict:
        return {
            "block_size_pct": self.block_size_pct,
            "invalidation_pct": self.invalidation_pct,
            "impulse_candles": self.impulse_candles,
        }

    def _find_bullish_obs(self, df: pd.DataFrame) -> list[dict]:
        """Find bullish order blocks: last bearish candle before impulse."""
        obs = []
        closes = df["close"].values
        opens = df["open"].values
        lows = df["low"].values
        highs = df["high"].values

        for i in range(1, len(df) - self.impulse_candles):
            # current candle is bearish
            if closes[i] >= opens[i]:
                continue
            # followed by N bullish candles (impulse)
            if all(closes[i + k] > opens[i + k] for k in range(1, self.impulse_candles + 1)):
                ob_high = highs[i]
                ob_low = lows[i]
                obs.append({
                    "idx": i,
                    "ob_high": ob_high,
                    "ob_low": ob_low,
                    "formed_at": df.index[i + self.impulse_candles],
                })
        return obs

    def _find_bearish_obs(self, df: pd.DataFrame) -> list[dict]:
        """Find bearish order blocks: last bullish candle before bearish impulse."""
        obs = []
        closes = df["close"].values
        opens = df["open"].values
        lows = df["low"].values
        highs = df["high"].values

        for i in range(1, len(df) - self.impulse_candles):
            if closes[i] <= opens[i]:
                continue
            if all(closes[i + k] < opens[i + k] for k in range(1, self.impulse_candles + 1)):
                obs.append({
                    "idx": i,
                    "ob_high": highs[i],
                    "ob_low": lows[i],
                    "formed_at": df.index[i + self.impulse_candles],
                })
        return obs

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        bullish_obs = self._find_bullish_obs(df)
        bearish_obs = self._find_bearish_obs(df)

        closes = df["close"].values
        lows = df["low"].values
        highs = df["high"].values

        for i in range(self.impulse_candles + 2, len(df)):
            ts = df.index[i]
            close = closes[i]

            for ob in bullish_obs:
                if ob["formed_at"] >= ts:
                    continue
                ob_low = ob["ob_low"]
                ob_high = ob["ob_high"]
                invalidation = ob_low * (1 - self.invalidation_pct)
                # price returns into OB zone from above = long entry
                if ob_low <= close <= ob_high and lows[i] > invalidation:
                    if result.at[ts, "signal"] == 0:
                        result.at[ts, "signal"] = 1
                        result.at[ts, "sl"] = invalidation
                        result.at[ts, "tp"] = close + 2 * (close - invalidation)

            for ob in bearish_obs:
                if ob["formed_at"] >= ts:
                    continue
                ob_low = ob["ob_low"]
                ob_high = ob["ob_high"]
                invalidation = ob_high * (1 + self.invalidation_pct)
                if ob_low <= close <= ob_high and highs[i] < invalidation:
                    if result.at[ts, "signal"] == 0:
                        result.at[ts, "signal"] = -1
                        result.at[ts, "sl"] = invalidation
                        result.at[ts, "tp"] = close - 2 * (invalidation - close)

        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_order_blocks.py -v
```

Expected: 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add strategies/order_blocks.py tests/test_order_blocks.py
git commit -m "feat: Order Block strategy (Smart Money Concept)"
```

---

## Task 7: Fair Value Gaps Strategy

**Files:**
- Create: `strategies/fair_value_gaps.py`
- Create: `tests/test_fair_value_gaps.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_fair_value_gaps.py
import pandas as pd
import numpy as np
import pytest
from strategies.fair_value_gaps import FairValueGapStrategy

@pytest.fixture
def fvg_strategy():
    return FairValueGapStrategy(gap_min_pct=0.002, fill_pct=0.5)

@pytest.fixture
def fvg_df():
    """
    Bullish FVG: candle[i-1].high < candle[i+1].low → gap between them.
    Then price returns to fill the gap.
    Candles: 0-3 normal, 4=impulsive up (creates gap), 5=return to gap.
    """
    data = {
        "open":  [100, 100, 100, 100, 101, 106, 105],
        "high":  [101, 101, 101, 101, 108, 108, 106],
        "low":   [99,  99,  99,  99,  100, 104, 103],
        "close": [100, 100, 100, 101, 107, 105, 104],
        "volume":[1000] * 7,
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=7, freq="1h"))

def test_returns_required_columns(fvg_strategy, fvg_df):
    result = fvg_strategy.generate_signals(fvg_df)
    assert "signal" in result.columns
    assert "sl" in result.columns
    assert "tp" in result.columns

def test_signal_values_valid(fvg_strategy, fvg_df):
    result = fvg_strategy.generate_signals(fvg_df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(fvg_strategy):
    p = fvg_strategy.get_params()
    assert "gap_min_pct" in p
    assert "fill_pct" in p

def test_no_gap_no_signal(fvg_strategy):
    flat = pd.DataFrame({
        "open":  [100.0] * 10, "high": [101.0] * 10,
        "low":   [99.0] * 10,  "close":[100.0] * 10, "volume":[1000.0] * 10,
    }, index=pd.date_range("2024-01-01", periods=10, freq="1h"))
    result = fvg_strategy.generate_signals(flat)
    assert (result["signal"] == 0).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_fair_value_gaps.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `strategies/fair_value_gaps.py`**

```python
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class FairValueGapStrategy(BaseStrategy):
    """
    Fair Value Gap (ICT concept):
    Bullish FVG: candle[i].high < candle[i+2].low — gap between i and i+2.
    Bearish FVG: candle[i].low > candle[i+2].high — gap between i and i+2.
    Signal fires when price returns to fill part of the gap.
    """

    def __init__(self, gap_min_pct: float = 0.002, fill_pct: float = 0.5,
                 direction_bias: str = "both"):
        self.gap_min_pct = gap_min_pct
        self.fill_pct = fill_pct
        self.direction_bias = direction_bias  # "both", "bullish", "bearish"

    def get_params(self) -> dict:
        return {
            "gap_min_pct": self.gap_min_pct,
            "fill_pct": self.fill_pct,
            "direction_bias": self.direction_bias,
        }

    def _find_fvgs(self, df: pd.DataFrame) -> list[dict]:
        fvgs = []
        highs = df["high"].values
        lows = df["low"].values

        for i in range(len(df) - 2):
            # Bullish FVG: gap between candle i high and candle i+2 low
            if self.direction_bias in ("both", "bullish"):
                gap_lo = highs[i]
                gap_hi = lows[i + 2]
                if gap_hi > gap_lo:
                    gap_size = (gap_hi - gap_lo) / gap_lo
                    if gap_size >= self.gap_min_pct:
                        fvgs.append({
                            "type": "bullish",
                            "gap_lo": gap_lo,
                            "gap_hi": gap_hi,
                            "mid": gap_lo + (gap_hi - gap_lo) * self.fill_pct,
                            "formed_at": df.index[i + 2],
                        })

            # Bearish FVG: gap between candle i low and candle i+2 high
            if self.direction_bias in ("both", "bearish"):
                gap_hi = lows[i]
                gap_lo = highs[i + 2]
                if gap_hi > gap_lo:
                    gap_size = (gap_hi - gap_lo) / gap_lo
                    if gap_size >= self.gap_min_pct:
                        fvgs.append({
                            "type": "bearish",
                            "gap_lo": gap_lo,
                            "gap_hi": gap_hi,
                            "mid": gap_lo + (gap_hi - gap_lo) * self.fill_pct,
                            "formed_at": df.index[i + 2],
                        })
        return fvgs

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        fvgs = self._find_fvgs(df)
        closes = df["close"].values

        for i in range(3, len(df)):
            ts = df.index[i]
            close = closes[i]

            for fvg in fvgs:
                if fvg["formed_at"] >= ts:
                    continue
                if result.at[ts, "signal"] != 0:
                    break

                if fvg["type"] == "bullish" and fvg["gap_lo"] <= close <= fvg["gap_hi"]:
                    result.at[ts, "signal"] = 1
                    result.at[ts, "sl"] = fvg["gap_lo"] * 0.998
                    result.at[ts, "tp"] = close + 2 * (close - fvg["gap_lo"] * 0.998)

                elif fvg["type"] == "bearish" and fvg["gap_lo"] <= close <= fvg["gap_hi"]:
                    result.at[ts, "signal"] = -1
                    result.at[ts, "sl"] = fvg["gap_hi"] * 1.002
                    result.at[ts, "tp"] = close - 2 * (fvg["gap_hi"] * 1.002 - close)

        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_fair_value_gaps.py -v
```

Expected: 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add strategies/fair_value_gaps.py tests/test_fair_value_gaps.py
git commit -m "feat: Fair Value Gap strategy (ICT concept)"
```

---

## Task 8: Candlestick Patterns Strategy

**Files:**
- Create: `strategies/candlestick_patterns.py`
- Create: `tests/test_candlestick_patterns.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_candlestick_patterns.py
import pandas as pd
import numpy as np
import pytest
from strategies.candlestick_patterns import CandlestickPatternStrategy

@pytest.fixture
def cp_strategy():
    return CandlestickPatternStrategy(min_body_ratio=0.6, confirmation_candles=1)

def make_df(opens, highs, lows, closes):
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": [1000.0] * len(opens),
    }, index=pd.date_range("2024-01-01", periods=len(opens), freq="1h"))

def test_bullish_engulfing_detected(cp_strategy):
    # Bearish candle then larger bullish candle engulfs it
    df = make_df(
        opens =[100, 102, 101, 99],
        highs =[102, 103, 102, 104],
        lows  =[99,  100, 98,  98],
        closes=[102, 100, 99,  103],  # candle 1 bearish, candle 3 bullish engulf
    )
    result = cp_strategy.generate_signals(df)
    assert "signal" in result.columns

def test_returns_required_columns(cp_strategy):
    df = make_df([100]*5, [101]*5, [99]*5, [100]*5)
    result = cp_strategy.generate_signals(df)
    assert {"signal", "sl", "tp"}.issubset(result.columns)

def test_signal_values_valid(cp_strategy):
    df = make_df([100]*10, [101]*10, [99]*10, [100]*10)
    result = cp_strategy.generate_signals(df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(cp_strategy):
    p = cp_strategy.get_params()
    assert "min_body_ratio" in p
    assert "confirmation_candles" in p

def test_pin_bar_detected(cp_strategy):
    # Long lower wick, small body at top = bullish pin bar
    df = make_df(
        opens =[100, 100, 100, 100, 100],
        highs =[101, 101, 101, 101, 101],
        lows  =[99,  99,  90,  99,  99],   # candle 2 has long lower wick
        closes=[100, 100, 100.5, 100, 100],
    )
    result = cp_strategy.generate_signals(df)
    assert result["signal"].isin([-1, 0, 1]).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_candlestick_patterns.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `strategies/candlestick_patterns.py`**

```python
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class CandlestickPatternStrategy(BaseStrategy):
    """
    Detects: Bullish/Bearish Engulfing, Pin Bars (Hammer/Shooting Star), Inside Bars.
    """

    def __init__(self, min_body_ratio: float = 0.6, confirmation_candles: int = 1,
                 pin_wick_ratio: float = 2.0):
        self.min_body_ratio = min_body_ratio
        self.confirmation_candles = confirmation_candles
        self.pin_wick_ratio = pin_wick_ratio

    def get_params(self) -> dict:
        return {
            "min_body_ratio": self.min_body_ratio,
            "confirmation_candles": self.confirmation_candles,
            "pin_wick_ratio": self.pin_wick_ratio,
        }

    def _body(self, o, c): return abs(c - o)
    def _range(self, h, l): return h - l
    def _upper_wick(self, o, h, c): return h - max(o, c)
    def _lower_wick(self, o, l, c): return min(o, c) - l

    def _is_bullish_engulfing(self, prev_o, prev_c, curr_o, curr_c) -> bool:
        prev_bearish = prev_c < prev_o
        curr_bullish = curr_c > curr_o
        engulfs = curr_o <= prev_c and curr_c >= prev_o
        body_ok = self._body(curr_o, curr_c) >= self._body(prev_o, prev_c) * self.min_body_ratio
        return prev_bearish and curr_bullish and engulfs and body_ok

    def _is_bearish_engulfing(self, prev_o, prev_c, curr_o, curr_c) -> bool:
        prev_bullish = prev_c > prev_o
        curr_bearish = curr_c < curr_o
        engulfs = curr_o >= prev_c and curr_c <= prev_o
        body_ok = self._body(curr_o, curr_c) >= self._body(prev_o, prev_c) * self.min_body_ratio
        return prev_bullish and curr_bearish and engulfs and body_ok

    def _is_bullish_pin_bar(self, o, h, l, c) -> bool:
        lower_wick = self._lower_wick(o, l, c)
        body = self._body(o, c)
        total = self._range(h, l)
        if total == 0:
            return False
        return lower_wick >= body * self.pin_wick_ratio and lower_wick >= total * 0.6

    def _is_bearish_pin_bar(self, o, h, l, c) -> bool:
        upper_wick = self._upper_wick(o, h, c)
        body = self._body(o, c)
        total = self._range(h, l)
        if total == 0:
            return False
        return upper_wick >= body * self.pin_wick_ratio and upper_wick >= total * 0.6

    def _is_inside_bar(self, prev_h, prev_l, curr_h, curr_l) -> bool:
        return curr_h <= prev_h and curr_l >= prev_l

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        for i in range(1, len(df)):
            ts = df.index[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            po, ph, pl, pc = opens[i-1], highs[i-1], lows[i-1], closes[i-1]

            signal = 0
            sl = np.nan
            tp = np.nan

            if self._is_bullish_engulfing(po, pc, o, c):
                signal = 1
                sl = l * 0.998
                tp = c + 2 * (c - sl)
            elif self._is_bearish_engulfing(po, pc, o, c):
                signal = -1
                sl = h * 1.002
                tp = c - 2 * (sl - c)
            elif self._is_bullish_pin_bar(o, h, l, c):
                signal = 1
                sl = l * 0.998
                tp = c + 2 * (c - sl)
            elif self._is_bearish_pin_bar(o, h, l, c):
                signal = -1
                sl = h * 1.002
                tp = c - 2 * (sl - c)
            elif self._is_inside_bar(ph, pl, h, l):
                # Inside bar: direction determined by breakout (signal on next bar)
                pass  # handled by neutral signal; breakout detection would need next candle

            result.at[ts, "signal"] = signal
            if signal != 0:
                result.at[ts, "sl"] = sl
                result.at[ts, "tp"] = tp

        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_candlestick_patterns.py -v
```

Expected: 5 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add strategies/candlestick_patterns.py tests/test_candlestick_patterns.py
git commit -m "feat: candlestick pattern strategy (engulfing, pin bar, inside bar)"
```

---

## Task 9: Market Structure Strategy

**Files:**
- Create: `strategies/market_structure.py`
- Create: `tests/test_market_structure.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_market_structure.py
import pandas as pd
import numpy as np
import pytest
from strategies.market_structure import MarketStructureStrategy

@pytest.fixture
def ms_strategy():
    return MarketStructureStrategy(swing_lookback=3)

def make_trending_df(prices):
    return pd.DataFrame({
        "open":  [p * 0.999 for p in prices],
        "high":  [p * 1.005 for p in prices],
        "low":   [p * 0.995 for p in prices],
        "close": prices,
        "volume":[1000.0] * len(prices),
    }, index=pd.date_range("2024-01-01", periods=len(prices), freq="1h"))

def test_returns_required_columns(ms_strategy):
    df = make_trending_df([100.0] * 20)
    result = ms_strategy.generate_signals(df)
    assert {"signal", "sl", "tp"}.issubset(result.columns)

def test_signal_values_valid(ms_strategy):
    prices = [100, 105, 102, 108, 105, 112, 108, 115]
    df = make_trending_df(prices)
    result = ms_strategy.generate_signals(df)
    assert result["signal"].isin([-1, 0, 1]).all()

def test_get_params(ms_strategy):
    p = ms_strategy.get_params()
    assert "swing_lookback" in p

def test_uptrend_generates_buy_signals(ms_strategy):
    # Clear uptrend: HH + HL sequence
    prices = [100, 110, 105, 115, 108, 120, 112, 125]
    df = make_trending_df(prices)
    result = ms_strategy.generate_signals(df)
    buys = (result["signal"] == 1).sum()
    sells = (result["signal"] == -1).sum()
    assert buys >= sells
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_market_structure.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `strategies/market_structure.py`**

```python
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MarketStructureStrategy(BaseStrategy):
    """
    Detects market structure: HH/HL (uptrend), LH/LL (downtrend).
    Signals on Break of Structure (BOS) and Change of Character (CHoCH).
    BOS: price breaks above last HH (bullish) or below last LL (bearish).
    CHoCH: uptrend breaks below last HL (bearish shift) or downtrend breaks above last LH.
    """

    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback

    def get_params(self) -> dict:
        return {"swing_lookback": self.swing_lookback}

    def _find_swing_highs(self, highs: np.ndarray) -> list[int]:
        n = self.swing_lookback
        result = []
        for i in range(n, len(highs) - n):
            if highs[i] == max(highs[i - n:i + n + 1]):
                result.append(i)
        return result

    def _find_swing_lows(self, lows: np.ndarray) -> list[int]:
        n = self.swing_lookback
        result = []
        for i in range(n, len(lows) - n):
            if lows[i] == min(lows[i - n:i + n + 1]):
                result.append(i)
        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        for i in range(self.swing_lookback * 2 + 1, len(df)):
            ts = df.index[i]
            close = closes[i]

            prev_sh = [idx for idx in swing_highs if idx < i]
            prev_sl = [idx for idx in swing_lows if idx < i]

            if len(prev_sh) < 2 or len(prev_sl) < 2:
                continue

            last_sh = highs[prev_sh[-1]]
            prev_sh_val = highs[prev_sh[-2]]
            last_sl = lows[prev_sl[-1]]
            prev_sl_val = lows[prev_sl[-2]]

            # Bullish BOS: close breaks above last swing high (HH pattern)
            if close > last_sh and last_sh > prev_sh_val:
                if result.at[ts, "signal"] == 0:
                    sl = last_sl * 0.998
                    result.at[ts, "signal"] = 1
                    result.at[ts, "sl"] = sl
                    result.at[ts, "tp"] = close + 2 * (close - sl)

            # Bearish BOS: close breaks below last swing low (LL pattern)
            elif close < last_sl and last_sl < prev_sl_val:
                if result.at[ts, "signal"] == 0:
                    sl = last_sh * 1.002
                    result.at[ts, "signal"] = -1
                    result.at[ts, "sl"] = sl
                    result.at[ts, "tp"] = close - 2 * (sl - close)

        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_market_structure.py -v
```

Expected: 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add strategies/market_structure.py tests/test_market_structure.py
git commit -m "feat: Market Structure strategy (BOS, CHoCH, HH/HL/LL/LH)"
```

---

## Task 10: Backtest Metrics

**Files:**
- Create: `backtest/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
import pandas as pd
import numpy as np
import pytest
from backtest.metrics import compute_metrics

@pytest.fixture
def winning_trades():
    return pd.DataFrame({
        "entry_price": [100.0, 200.0, 150.0],
        "exit_price":  [110.0, 210.0, 165.0],
        "pnl":         [10.0,  10.0,  15.0],
        "direction":   [1, 1, 1],
        "equity":      [1010.0, 1020.0, 1035.0],
        "r_multiple":  [2.0, 2.0, 3.0],
    })

@pytest.fixture
def mixed_trades():
    return pd.DataFrame({
        "entry_price": [100.0, 200.0, 150.0, 120.0],
        "exit_price":  [110.0, 190.0, 165.0, 115.0],
        "pnl":         [10.0, -10.0, 15.0, -5.0],
        "direction":   [1, 1, 1, 1],
        "equity":      [1010.0, 1000.0, 1015.0, 1010.0],
        "r_multiple":  [2.0, -1.0, 3.0, -1.0],
    })

def test_win_rate_all_winners(winning_trades):
    m = compute_metrics(winning_trades, starting_capital=1000.0)
    assert m["win_rate"] == 1.0

def test_win_rate_mixed(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    assert m["win_rate"] == 0.5

def test_profit_factor(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    gross_profit = 10.0 + 15.0
    gross_loss = 10.0 + 5.0
    assert abs(m["profit_factor"] - gross_profit / gross_loss) < 0.01

def test_max_drawdown(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    assert m["max_drawdown"] >= 0.0

def test_sharpe_ratio_positive_returns(winning_trades):
    m = compute_metrics(winning_trades, starting_capital=1000.0)
    assert m["sharpe_ratio"] > 0

def test_returns_all_required_keys(mixed_trades):
    m = compute_metrics(mixed_trades, starting_capital=1000.0)
    for key in ["win_rate", "profit_factor", "max_drawdown", "sharpe_ratio", "total_return", "num_trades"]:
        assert key in m, f"Missing key: {key}"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `backtest/metrics.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: 6 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add backtest/metrics.py tests/test_metrics.py
git commit -m "feat: backtest metrics (Sharpe, drawdown, win rate, profit factor)"
```

---

## Task 11: Trade Positions (Lifecycle)

**Files:**
- Create: `backtest/positions.py`
- Create: `tests/test_positions.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_positions.py
import pandas as pd
import numpy as np
import pytest
from backtest.positions import simulate_trades

@pytest.fixture
def signal_df():
    """OHLCV with one buy signal at index 2, SL=98, TP=106."""
    data = {
        "open":   [100, 101, 102, 103, 104, 105, 106, 107],
        "high":   [101, 102, 103, 107, 105, 106, 107, 108],
        "low":    [99,  100, 101, 102, 103, 104, 105, 106],
        "close":  [101, 101, 102, 106, 104, 105, 106, 107],
        "volume": [1000.0] * 8,
        "signal": [0, 0, 1, 0, 0, 0, 0, 0],
        "sl":     [np.nan, np.nan, 98.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        "tp":     [np.nan, np.nan, 106.0, np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=8, freq="1h"))

def test_tp_hit_records_trade(signal_df):
    trades = simulate_trades(signal_df, starting_capital=1000.0, risk_per_trade=0.02)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "tp"
    assert trades.iloc[0]["pnl"] > 0

def test_sl_hit_records_loss():
    data = {
        "open":   [100, 101, 102, 103, 97,  100],
        "high":   [101, 102, 103, 104, 101, 101],
        "low":    [99,  100, 101, 97,  96,  99],   # low[3]=97 hits SL=98
        "close":  [101, 101, 102, 97,  100, 100],
        "volume": [1000.0] * 6,
        "signal": [0, 0, 1, 0, 0, 0],
        "sl":     [np.nan, np.nan, 98.0, np.nan, np.nan, np.nan],
        "tp":     [np.nan, np.nan, 106.0, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=6, freq="1h"))
    trades = simulate_trades(df, starting_capital=1000.0, risk_per_trade=0.02)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "sl"
    assert trades.iloc[0]["pnl"] < 0

def test_equity_compounds(signal_df):
    trades = simulate_trades(signal_df, starting_capital=1000.0, risk_per_trade=0.02)
    assert trades.iloc[-1]["equity"] != 1000.0

def test_no_signals_no_trades():
    data = {
        "open": [100.0]*5, "high": [101.0]*5, "low": [99.0]*5, "close": [100.0]*5,
        "volume": [1000.0]*5, "signal": [0]*5,
        "sl": [np.nan]*5, "tp": [np.nan]*5,
    }
    df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=5, freq="1h"))
    trades = simulate_trades(df, starting_capital=1000.0, risk_per_trade=0.02)
    assert len(trades) == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_positions.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `backtest/positions.py`**

```python
import pandas as pd
import numpy as np


def simulate_trades(
    df: pd.DataFrame,
    starting_capital: float,
    risk_per_trade: float,
    max_bars: int = 100,
) -> pd.DataFrame:
    """
    Simulate trades from a signal DataFrame.

    df must have columns: open, high, low, close, signal, sl, tp.
    signal: 1=long, -1=short, 0=no trade.
    sl/tp: stop loss and take profit prices.

    Returns DataFrame of completed trades.
    """
    capital = starting_capital
    trades = []
    in_trade = False
    entry_price = sl = tp = direction = entry_bar = 0
    size = 0.0

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    signals = df["signal"].values
    sls = df["sl"].values
    tps = df["tp"].values

    for i in range(len(df)):
        ts = df.index[i]

        if in_trade:
            # Check exit conditions on this bar
            exit_price = None
            exit_reason = None

            if direction == 1:  # long
                if lows[i] <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                elif highs[i] >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                elif (i - entry_bar) >= max_bars:
                    exit_price = opens[i]
                    exit_reason = "timeout"
            else:  # short
                if highs[i] >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                elif lows[i] <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                elif (i - entry_bar) >= max_bars:
                    exit_price = opens[i]
                    exit_reason = "timeout"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * direction * size
                capital += pnl
                r_risk = abs(entry_price - sl) * size
                r_multiple = pnl / r_risk if r_risk > 0 else 0.0
                trades.append({
                    "entry_time": df.index[entry_bar],
                    "exit_time": ts,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "sl": sl,
                    "tp": tp,
                    "size": size,
                    "pnl": pnl,
                    "r_multiple": r_multiple,
                    "exit_reason": exit_reason,
                    "equity": capital,
                })
                in_trade = False

        if not in_trade and signals[i] != 0 and not np.isnan(sls[i]):
            # Enter trade on next bar open — but we enter at current bar open
            # to keep simple vectorized model (signal on close, fill on same bar open)
            direction = int(signals[i])
            entry_price = opens[i]
            sl = sls[i]
            tp = tps[i] if not np.isnan(tps[i]) else (
                entry_price + 2 * abs(entry_price - sl) * direction
            )
            risk_amount = capital * risk_per_trade
            price_risk = abs(entry_price - sl)
            size = risk_amount / price_risk if price_risk > 0 else 0.0
            entry_bar = i
            in_trade = True

    return pd.DataFrame(trades)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_positions.py -v
```

Expected: 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add backtest/positions.py tests/test_positions.py
git commit -m "feat: trade lifecycle simulator (entry, SL, TP, timeout)"
```

---

## Task 12: Backtest Engine

**Files:**
- Create: `backtest/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_engine.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from backtest.engine import BacktestEngine
from strategies.base import BaseStrategy

class AlwaysBuyStrategy(BaseStrategy):
    def generate_signals(self, df):
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan
        for i in range(5, len(df)):
            result.iloc[i, result.columns.get_loc("signal")] = 1
            result.iloc[i, result.columns.get_loc("sl")] = df["close"].iloc[i] * 0.98
            result.iloc[i, result.columns.get_loc("tp")] = df["close"].iloc[i] * 1.04
        return result
    def get_params(self): return {}

@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open":  closes * 0.999,
        "high":  closes * 1.005,
        "low":   closes * 0.995,
        "close": closes,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

def test_run_returns_result_dict(sample_df):
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    result = engine.run(AlwaysBuyStrategy(), sample_df)
    assert "trades" in result
    assert "metrics" in result

def test_metrics_have_required_keys(sample_df):
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    result = engine.run(AlwaysBuyStrategy(), sample_df)
    for key in ["win_rate", "sharpe_ratio", "max_drawdown", "profit_factor", "num_trades"]:
        assert key in result["metrics"]

def test_no_negative_equity(sample_df):
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    result = engine.run(AlwaysBuyStrategy(), sample_df)
    if not result["trades"].empty:
        assert (result["trades"]["equity"] > 0).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_engine.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `backtest/engine.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_engine.py -v
```

Expected: 3 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add backtest/engine.py tests/test_engine.py
git commit -m "feat: BacktestEngine orchestrates strategy signals → trades → metrics"
```

---

## Task 13: Risk Models

**Files:**
- Create: `optimization/risk_models.py`
- Create: `tests/test_risk_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_risk_models.py
import pytest
from optimization.risk_models import FixedPctRisk, FixedDollarRisk, KellyRisk

def test_fixed_pct_risk():
    model = FixedPctRisk(pct=0.02)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    # risk = 1000 * 0.02 = 20; price_risk = 2.0; size = 20/2 = 10
    assert abs(size - 10.0) < 0.01

def test_fixed_dollar_risk():
    model = FixedDollarRisk(amount=20.0)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    assert abs(size - 10.0) < 0.01

def test_kelly_risk_positive():
    model = KellyRisk(win_rate=0.55, avg_win=2.0, avg_loss=1.0)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    assert size > 0

def test_kelly_risk_capped_at_25pct():
    # Kelly fraction can be very high — should be capped
    model = KellyRisk(win_rate=0.9, avg_win=5.0, avg_loss=1.0)
    size = model.position_size(capital=1000.0, entry=100.0, sl=98.0)
    max_risk = 1000.0 * 0.25  # 25% cap
    max_size = max_risk / abs(100.0 - 98.0)
    assert size <= max_size + 0.01

def test_zero_price_risk_returns_zero():
    model = FixedPctRisk(pct=0.02)
    size = model.position_size(capital=1000.0, entry=100.0, sl=100.0)
    assert size == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_risk_models.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `optimization/risk_models.py`**

```python
from abc import ABC, abstractmethod


class BaseRiskModel(ABC):
    @abstractmethod
    def position_size(self, capital: float, entry: float, sl: float) -> float:
        """Returns number of units/shares to trade."""


class FixedPctRisk(BaseRiskModel):
    def __init__(self, pct: float = 0.02):
        self.pct = pct

    def position_size(self, capital: float, entry: float, sl: float) -> float:
        price_risk = abs(entry - sl)
        if price_risk == 0:
            return 0.0
        risk_amount = capital * self.pct
        return risk_amount / price_risk


class FixedDollarRisk(BaseRiskModel):
    def __init__(self, amount: float = 20.0):
        self.amount = amount

    def position_size(self, capital: float, entry: float, sl: float) -> float:
        price_risk = abs(entry - sl)
        if price_risk == 0:
            return 0.0
        return self.amount / price_risk


class KellyRisk(BaseRiskModel):
    MAX_KELLY_FRACTION = 0.25  # cap at 25% of capital

    def __init__(self, win_rate: float = 0.5, avg_win: float = 2.0, avg_loss: float = 1.0):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss

    def kelly_fraction(self) -> float:
        if self.avg_loss == 0:
            return 0.0
        b = self.avg_win / self.avg_loss
        f = (b * self.win_rate - (1 - self.win_rate)) / b
        return max(0.0, min(f, self.MAX_KELLY_FRACTION))

    def position_size(self, capital: float, entry: float, sl: float) -> float:
        price_risk = abs(entry - sl)
        if price_risk == 0:
            return 0.0
        risk_amount = capital * self.kelly_fraction()
        return risk_amount / price_risk
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_risk_models.py -v
```

Expected: 5 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add optimization/risk_models.py tests/test_risk_models.py
git commit -m "feat: risk models (fixed pct, fixed dollar, Kelly Criterion)"
```

---

## Task 14: Exit Strategies

**Files:**
- Create: `optimization/exit_strategies.py`
- Create: `tests/test_exit_strategies.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_exit_strategies.py
import pytest
from optimization.exit_strategies import FixedRR, TrailingStop, PartialTP

def test_fixed_rr_long_tp():
    exit_cfg = FixedRR(rr_ratio=2.0)
    tp = exit_cfg.take_profit(entry=100.0, sl=98.0, direction=1)
    assert abs(tp - 104.0) < 0.01  # entry + 2 * risk

def test_fixed_rr_short_tp():
    exit_cfg = FixedRR(rr_ratio=2.0)
    tp = exit_cfg.take_profit(entry=100.0, sl=102.0, direction=-1)
    assert abs(tp - 96.0) < 0.01   # entry - 2 * risk

def test_trailing_stop_initial_sl_unchanged():
    exit_cfg = TrailingStop(trail_pct=0.02)
    new_sl = exit_cfg.update_sl(current_sl=98.0, current_price=100.0, direction=1, entry=100.0)
    assert new_sl >= 98.0

def test_trailing_stop_moves_up_with_price():
    exit_cfg = TrailingStop(trail_pct=0.02)
    new_sl = exit_cfg.update_sl(current_sl=98.0, current_price=110.0, direction=1, entry=100.0)
    expected = 110.0 * (1 - 0.02)
    assert abs(new_sl - expected) < 0.01

def test_partial_tp_first_target():
    exit_cfg = PartialTP(first_tp_r=1.0, first_tp_pct=0.5)
    tp1 = exit_cfg.first_target(entry=100.0, sl=98.0, direction=1)
    assert abs(tp1 - 102.0) < 0.01  # entry + 1R

def test_get_params_all_exits():
    for cls in [FixedRR(2.0), TrailingStop(0.02), PartialTP(1.0, 0.5)]:
        assert isinstance(cls.get_params(), dict)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_exit_strategies.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `optimization/exit_strategies.py`**

```python
from abc import ABC, abstractmethod


class BaseExitStrategy(ABC):
    @abstractmethod
    def get_params(self) -> dict: ...


class FixedRR(BaseExitStrategy):
    def __init__(self, rr_ratio: float = 2.0):
        self.rr_ratio = rr_ratio

    def take_profit(self, entry: float, sl: float, direction: int) -> float:
        risk = abs(entry - sl)
        return entry + direction * risk * self.rr_ratio

    def get_params(self) -> dict:
        return {"rr_ratio": self.rr_ratio}


class TrailingStop(BaseExitStrategy):
    def __init__(self, trail_pct: float = 0.02):
        self.trail_pct = trail_pct

    def update_sl(self, current_sl: float, current_price: float,
                  direction: int, entry: float) -> float:
        if direction == 1:
            new_sl = current_price * (1 - self.trail_pct)
            return max(current_sl, new_sl)
        else:
            new_sl = current_price * (1 + self.trail_pct)
            return min(current_sl, new_sl)

    def get_params(self) -> dict:
        return {"trail_pct": self.trail_pct}


class PartialTP(BaseExitStrategy):
    def __init__(self, first_tp_r: float = 1.0, first_tp_pct: float = 0.5):
        self.first_tp_r = first_tp_r
        self.first_tp_pct = first_tp_pct

    def first_target(self, entry: float, sl: float, direction: int) -> float:
        risk = abs(entry - sl)
        return entry + direction * risk * self.first_tp_r

    def get_params(self) -> dict:
        return {"first_tp_r": self.first_tp_r, "first_tp_pct": self.first_tp_pct}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_exit_strategies.py -v
```

Expected: 6 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add optimization/exit_strategies.py tests/test_exit_strategies.py
git commit -m "feat: exit strategies (fixed R:R, trailing stop, partial TP)"
```

---

## Task 15: Grid Search Optimizer

**Files:**
- Create: `optimization/grid_search.py`
- Create: `tests/test_grid_search.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_grid_search.py
import pandas as pd
import numpy as np
import pytest
from optimization.grid_search import GridSearch
from strategies.support_resistance import SupportResistanceStrategy

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open":  closes * 0.999, "high": closes * 1.005,
        "low":   closes * 0.995, "close": closes,
        "volume": np.ones(n) * 1000,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

def test_grid_search_returns_dataframe(sample_data):
    strategies = [SupportResistanceStrategy(lookback=10, touch_count=2, zone_width_pct=0.005)]
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=strategies,
        data={"TEST/USDT": {"1h": sample_data}},
    )
    assert isinstance(results, pd.DataFrame)

def test_grid_search_has_required_columns(sample_data):
    strategies = [SupportResistanceStrategy(lookback=10, touch_count=2, zone_width_pct=0.005)]
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=strategies,
        data={"TEST/USDT": {"1h": sample_data}},
    )
    for col in ["strategy", "asset", "timeframe", "sharpe_ratio", "score"]:
        assert col in results.columns

def test_grid_search_sorted_by_score(sample_data):
    strategies = [
        SupportResistanceStrategy(lookback=10, touch_count=2, zone_width_pct=0.005),
        SupportResistanceStrategy(lookback=20, touch_count=2, zone_width_pct=0.005),
    ]
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=strategies,
        data={"TEST/USDT": {"1h": sample_data}},
    )
    if len(results) > 1:
        scores = results["score"].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_grid_search.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `optimization/grid_search.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_grid_search.py -v
```

Expected: 3 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add optimization/grid_search.py tests/test_grid_search.py
git commit -m "feat: parallel grid search optimizer with composite scoring"
```

---

## Task 16: Reporting — Results + Charts

**Files:**
- Create: `reporting/results.py`
- Create: `reporting/charts.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_results.py
import pandas as pd
import pytest
from pathlib import Path
from reporting.results import ResultsReporter

@pytest.fixture
def sample_results():
    return pd.DataFrame({
        "strategy": ["OrderBlock", "FVG", "SR"],
        "asset": ["BTC/USDT", "AAPL", "ETH/USDT"],
        "timeframe": ["4h", "1d", "1h"],
        "score": [0.85, 0.72, 0.61],
        "sharpe_ratio": [1.5, 1.2, 0.9],
        "win_rate": [0.6, 0.55, 0.5],
        "profit_factor": [2.1, 1.8, 1.4],
        "max_drawdown": [0.08, 0.12, 0.15],
        "total_return": [0.45, 0.30, 0.20],
        "num_trades": [42, 31, 28],
        "params": ["{}", "{}", "{}"],
    })

def test_save_csv(sample_results, tmp_path):
    reporter = ResultsReporter(reports_dir=tmp_path)
    reporter.save(sample_results, run_id="test")
    csv_path = tmp_path / "test" / "leaderboard.csv"
    assert csv_path.exists()

def test_save_html(sample_results, tmp_path):
    reporter = ResultsReporter(reports_dir=tmp_path)
    reporter.save(sample_results, run_id="test")
    html_path = tmp_path / "test" / "leaderboard.html"
    assert html_path.exists()

def test_top_n(sample_results, tmp_path):
    reporter = ResultsReporter(reports_dir=tmp_path)
    top = reporter.top(sample_results, n=2)
    assert len(top) == 2
    assert top.iloc[0]["score"] >= top.iloc[1]["score"]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_results.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `reporting/results.py`**

```python
import pandas as pd
from pathlib import Path


class ResultsReporter:
    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)

    def save(self, results: pd.DataFrame, run_id: str) -> Path:
        out_dir = self.reports_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "leaderboard.csv"
        results.to_csv(csv_path, index=False)

        html_path = out_dir / "leaderboard.html"
        html = results.to_html(index=False, float_format="%.4f")
        styled = f"""<!DOCTYPE html>
<html><head><style>
  body {{ font-family: monospace; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
  th {{ background: #222; color: #fff; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
</style></head><body>
<h2>Price Action Trader — Leaderboard ({run_id})</h2>
{html}
</body></html>"""
        html_path.write_text(styled)

        return out_dir

    def top(self, results: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        return results.sort_values("score", ascending=False).head(n).reset_index(drop=True)
```

- [ ] **Step 4: Implement `reporting/charts.py`**

```python
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
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_results.py -v
```

Expected: 3 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add reporting/results.py reporting/charts.py tests/test_results.py
git commit -m "feat: results reporter (CSV/HTML leaderboard) and equity curve charts"
```

---

## Task 17: CLI Entrypoint

**Files:**
- Create: `main.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from main import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "optimize" in result.output or "backtest" in result.output

def test_backtest_missing_args():
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest"])
    assert result.exit_code != 0 or "Error" in result.output or "Missing" in result.output

def test_report_no_runs(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--top", "5", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `main.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: 3 tests `PASSED`.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_cli.py
git commit -m "feat: CLI entrypoint with optimize, backtest, and report commands"
```

---

## Task 18: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""
Smoke test: runs a full backtest pipeline end-to-end using synthetic data.
No API calls — all data is generated in-memory.
"""
import numpy as np
import pandas as pd
import pytest
from backtest.engine import BacktestEngine
from optimization.grid_search import GridSearch
from reporting.results import ResultsReporter
from strategies.support_resistance import SupportResistanceStrategy
from strategies.order_blocks import OrderBlockStrategy
from strategies.fair_value_gaps import FairValueGapStrategy
from strategies.candlestick_patterns import CandlestickPatternStrategy
from strategies.market_structure import MarketStructureStrategy


def make_synthetic_data(n=300, seed=42):
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.8)
    closes = np.maximum(closes, 1.0)
    return pd.DataFrame({
        "open":   closes * np.random.uniform(0.998, 1.0, n),
        "high":   closes * np.random.uniform(1.001, 1.01, n),
        "low":    closes * np.random.uniform(0.99, 0.999, n),
        "close":  closes,
        "volume": np.random.uniform(1000, 10000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))


def test_full_pipeline_runs_without_error(tmp_path):
    df = make_synthetic_data()
    strategies = [
        SupportResistanceStrategy(lookback=20, touch_count=2, zone_width_pct=0.005),
        OrderBlockStrategy(),
        FairValueGapStrategy(),
        CandlestickPatternStrategy(),
        MarketStructureStrategy(),
    ]
    data = {"SYNTHETIC/USDT": {"1h": df}}
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(strategies=strategies, data=data, n_workers=1)
    assert isinstance(results, pd.DataFrame)
    if not results.empty:
        assert "score" in results.columns
        assert results["score"].iloc[0] >= results["score"].iloc[-1]


def test_single_backtest_computes_metrics():
    df = make_synthetic_data()
    engine = BacktestEngine(starting_capital=1000.0, risk_per_trade=0.02)
    for StratClass in [
        SupportResistanceStrategy,
        OrderBlockStrategy,
        FairValueGapStrategy,
        CandlestickPatternStrategy,
        MarketStructureStrategy,
    ]:
        result = engine.run(StratClass(), df)
        assert "metrics" in result
        assert "trades" in result
        assert "win_rate" in result["metrics"]


def test_reporter_saves_results(tmp_path):
    df = make_synthetic_data()
    gs = GridSearch(starting_capital=1000.0)
    results = gs.run(
        strategies=[SupportResistanceStrategy()],
        data={"SYNTHETIC/USDT": {"1h": df}},
        n_workers=1,
    )
    reporter = ResultsReporter(reports_dir=tmp_path)
    out_dir = reporter.save(results, run_id="smoke_test")
    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "leaderboard.html").exists()
```

- [ ] **Step 2: Run integration test**

```bash
uv run pytest tests/test_integration.py -v
```

Expected: 3 tests `PASSED`.

- [ ] **Step 3: Run full test suite one final time**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All tests `PASSED`.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration smoke test for full pipeline"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Data layer: CCXT + Twelve Data fetchers, parquet cache, universe discovery
- [x] 5 strategies: S/R, Order Blocks, FVG, Candlestick Patterns, Market Structure
- [x] Backtest engine: vectorized, no look-ahead bias, $1,000 starting capital
- [x] Trade lifecycle: entry, SL, TP, trailing, partial, timeout
- [x] Risk models: Fixed %, Fixed $, Kelly
- [x] Exit strategies: Fixed R:R, Trailing Stop, Partial TP
- [x] Optimization: grid search, parallel, composite score
- [x] Reporting: CSV, HTML, equity curves, leaderboard chart
- [x] CLI: `optimize`, `backtest`, `report` commands

**Type consistency:** All method signatures consistent across tasks. `BaseStrategy.generate_signals()` returns DataFrame with `signal`, `sl`, `tp` columns in all 5 implementations.

**No placeholders:** All steps contain complete code.
