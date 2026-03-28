def to_alpaca(symbol: str) -> str:
    """Convert internal symbol to Alpaca format.

    "BTC/USDT" -> "BTC/USD"
    "AAPL"     -> "AAPL"
    """
    if "/" in symbol and symbol.endswith("/USDT"):
        return symbol.replace("/USDT", "/USD")
    return symbol


def to_internal(alpaca_symbol: str) -> str:
    """Convert Alpaca symbol back to internal format.

    "BTC/USD" -> "BTC/USDT"
    "AAPL"    -> "AAPL"
    """
    if "/" in alpaca_symbol and alpaca_symbol.endswith("/USD"):
        return alpaca_symbol.replace("/USD", "/USDT")
    return alpaca_symbol


def is_crypto(symbol: str) -> bool:
    """Return True if the internal symbol is a crypto asset."""
    return "/" in symbol
