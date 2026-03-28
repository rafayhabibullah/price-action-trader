import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY: str = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.environ.get(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"
)
ALPACA_DATA_URL_STOCKS: str = "https://data.alpaca.markets/v2"
ALPACA_DATA_URL_CRYPTO: str = "https://data.alpaca.markets/v1beta3"

ASSETS_CRYPTO: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
ASSETS_STOCKS: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "SPY", "QQQ"]
TIMEFRAMES: list[str] = ["1h", "4h", "1d"]

MAX_CONCURRENT: int = int(os.environ.get("MAX_CONCURRENT", "15"))
RISK_PER_TRADE: float = float(os.environ.get("RISK_PER_TRADE", "0.02"))
MAX_BARS: int = int(os.environ.get("MAX_BARS", "100"))
CANDLE_HISTORY: int = int(os.environ.get("CANDLE_HISTORY", "200"))

TF_HOURS: dict[str, float] = {"1h": 1.0, "4h": 4.0, "1d": 24.0}
MIN_SL_DISTANCE_PCT: float = 0.001  # 0.1% minimum SL distance
