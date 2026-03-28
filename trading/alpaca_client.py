import requests
import pandas as pd
from trading.symbols import is_crypto


TF_MAP = {"1h": "1Hour", "4h": "4Hour", "1d": "1Day"}


class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json",
        }
        self._data_url_stocks = "https://data.alpaca.markets/v2"
        self._data_url_crypto = "https://data.alpaca.markets/v1beta3"

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        """GET with one retry on 5xx."""
        for attempt in range(2):
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=30)
                if r.status_code < 500:
                    try:
                        return r.json()
                    except (ValueError, Exception):
                        return {}
                if attempt == 1:
                    return {}
            except requests.RequestException:
                if attempt == 1:
                    return {}
        return {}

    def _post(self, url: str, payload: dict) -> dict:
        for attempt in range(2):
            try:
                r = requests.post(url, headers=self.headers, json=payload, timeout=30)
                if r.status_code < 500:
                    try:
                        return r.json()
                    except (ValueError, Exception):
                        return {}
                if attempt == 1:
                    return {}
            except requests.RequestException:
                if attempt == 1:
                    return {}
        return {}

    def _delete(self, url: str) -> dict:
        for attempt in range(2):
            try:
                r = requests.delete(url, headers=self.headers, timeout=30)
                if r.status_code < 500:
                    try:
                        return r.json()
                    except (ValueError, Exception):
                        return {}
                if attempt == 1:
                    return {}
            except requests.RequestException:
                if attempt == 1:
                    return {}
        return {}

    def get_account(self) -> dict:
        return self._get(f"{self.base_url}/account")

    def is_market_open(self) -> bool:
        result = self._get(f"{self.base_url}/clock")
        return bool(result.get("is_open", False))

    def get_positions(self) -> list[dict]:
        result = self._get(f"{self.base_url}/positions")
        return result if isinstance(result, list) else []

    def get_orders(self, status: str = "open") -> list[dict]:
        result = self._get(f"{self.base_url}/orders", params={"status": status})
        return result if isinstance(result, list) else []

    def close_position(self, symbol: str) -> dict:
        return self._delete(f"{self.base_url}/positions/{symbol}")

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"{self.base_url}/orders/{order_id}")

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        take_profit: float,
        stop_loss: float,
    ) -> dict:
        # Alpaca: stocks require "day" TIF for market orders; crypto supports "gtc"
        time_in_force = "gtc" if is_crypto(symbol) else "day"
        payload = {
            "symbol": symbol,
            "qty": str(round(qty, 8)),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "take_profit": {"limit_price": str(round(take_profit, 4))},
            "stop_loss": {"stop_price": str(round(stop_loss, 4))},
        }
        return self._post(f"{self.base_url}/orders", payload)

    def get_bars(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV bars for a symbol. Returns empty DataFrame on error."""
        tf = TF_MAP.get(timeframe, "1Hour")

        if is_crypto(symbol):
            url = f"{self._data_url_crypto}/crypto/bars"
            params = {"symbols": symbol, "timeframe": tf, "limit": limit, "sort": "asc"}
        else:
            url = f"{self._data_url_stocks}/stocks/bars"
            params = {"symbols": symbol, "timeframe": tf, "limit": limit, "sort": "asc"}

        data = self._get(url, params=params)
        bars_map = data.get("bars", {}) if isinstance(data, dict) else {}
        bars = bars_map.get(symbol, [])

        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(None)
        df = df.set_index("t").rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        return df[["open", "high", "low", "close", "volume"]].astype(float)
