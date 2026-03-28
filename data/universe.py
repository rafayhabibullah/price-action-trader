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
