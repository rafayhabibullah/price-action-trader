import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from trading.alpaca_client import AlpacaClient


@pytest.fixture
def client():
    return AlpacaClient(
        api_key="test-key",
        secret_key="test-secret",
        base_url="https://paper-api.alpaca.markets/v2",
    )


def test_get_account(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "equity": "100000.00",
        "cash": "100000.00",
        "buying_power": "200000.00",
        "status": "ACTIVE",
    }
    with patch("requests.get", return_value=mock_response):
        result = client.get_account()
    assert result["equity"] == "100000.00"
    assert result["status"] == "ACTIVE"


def test_get_positions_empty(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    with patch("requests.get", return_value=mock_response):
        result = client.get_positions()
    assert result == []


def test_get_bars_stocks_returns_dataframe(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "bars": {
            "AAPL": [
                {"t": "2024-01-01T10:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 1000.0},
                {"t": "2024-01-01T11:00:00Z", "o": 101.0, "h": 103.0, "l": 100.0, "c": 102.0, "v": 1200.0},
            ]
        }
    }
    with patch("requests.get", return_value=mock_response):
        df = client.get_bars("AAPL", "1h", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_get_bars_crypto_returns_dataframe(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "bars": {
            "BTC/USD": [
                {"t": "2024-01-01T10:00:00Z", "o": 40000.0, "h": 41000.0, "l": 39000.0, "c": 40500.0, "v": 10.0},
            ]
        }
    }
    with patch("requests.get", return_value=mock_response):
        df = client.get_bars("BTC/USD", "1h", limit=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["close"].iloc[0] == 40500.0


def test_place_bracket_order(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "order-123", "status": "pending_new"}
    with patch("requests.post", return_value=mock_response):
        result = client.place_bracket_order(
            symbol="AAPL",
            side="buy",
            qty=10,
            take_profit=120.0,
            stop_loss=95.0,
        )
    assert result["id"] == "order-123"


def test_close_position(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "order-456", "status": "pending_new"}
    with patch("requests.delete", return_value=mock_response):
        result = client.close_position("AAPL")
    assert result["id"] == "order-456"


def test_is_market_open_true(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"is_open": True}
    with patch("requests.get", return_value=mock_response):
        assert client.is_market_open() is True


def test_is_market_open_false(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"is_open": False}
    with patch("requests.get", return_value=mock_response):
        assert client.is_market_open() is False


def test_get_bars_retries_on_500(client):
    fail = MagicMock()
    fail.status_code = 500
    fail.json.return_value = {"message": "internal error"}
    success = MagicMock()
    success.status_code = 200
    success.json.return_value = {"bars": {"AAPL": [
        {"t": "2024-01-01T10:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 1000.0},
    ]}}
    with patch("requests.get", side_effect=[fail, success]):
        df = client.get_bars("AAPL", "1h", limit=1)
    assert len(df) == 1


def test_get_bars_returns_empty_on_repeated_500(client):
    fail = MagicMock()
    fail.status_code = 500
    fail.json.return_value = {"message": "internal error"}
    with patch("requests.get", side_effect=[fail, fail]):
        df = client.get_bars("AAPL", "1h", limit=1)
    assert df.empty


def test_place_bracket_order_retries_on_500(client):
    fail = MagicMock()
    fail.status_code = 500
    fail.json.return_value = {"message": "internal error"}
    success = MagicMock()
    success.status_code = 200
    success.json.return_value = {"id": "order-123", "status": "pending_new"}
    with patch("requests.post", side_effect=[fail, success]):
        result = client.place_bracket_order(
            symbol="AAPL", side="buy", qty=10, take_profit=120.0, stop_loss=95.0
        )
    assert result["id"] == "order-123"


def test_close_position_retries_on_500(client):
    fail = MagicMock()
    fail.status_code = 500
    fail.json.return_value = {"message": "internal error"}
    success = MagicMock()
    success.status_code = 200
    success.json.return_value = {"id": "order-456", "status": "pending_new"}
    with patch("requests.delete", side_effect=[fail, success]):
        result = client.close_position("AAPL")
    assert result["id"] == "order-456"


def test_place_bracket_order_stock_uses_day_tif(client):
    """Stock bracket orders must use time_in_force=day."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "order-123", "status": "pending_new"}
    with patch("requests.post", return_value=mock_response) as mock_post:
        client.place_bracket_order(
            symbol="AAPL",
            side="buy",
            qty=10,
            take_profit=120.0,
            stop_loss=95.0,
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["time_in_force"] == "day"
    assert payload["order_class"] == "bracket"


def test_place_bracket_order_crypto_uses_gtc_tif(client):
    """Crypto bracket orders must use time_in_force=gtc."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "order-123", "status": "pending_new"}
    with patch("requests.post", return_value=mock_response) as mock_post:
        client.place_bracket_order(
            symbol="BTC/USD",
            side="buy",
            qty=0.001,
            take_profit=60000.0,
            stop_loss=45000.0,
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["time_in_force"] == "gtc"
