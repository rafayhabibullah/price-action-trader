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


def init_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
