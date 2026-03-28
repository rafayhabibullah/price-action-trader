import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from trading.position_manager import run_cycle, CycleResult
from trading.scanner import Signal


def _make_mock_client(
    equity="100000.00",
    positions=None,
    orders=None,
    market_open=True,
):
    client = MagicMock()
    client.get_account.return_value = {
        "equity": equity,
        "cash": equity,
        "buying_power": str(float(equity) * 2),
    }
    client.get_positions.return_value = positions or []
    client.get_orders.return_value = orders or []
    client.is_market_open.return_value = market_open
    client.place_bracket_order.return_value = {"id": "order-123", "status": "pending_new"}
    client.close_position.return_value = {"id": "order-456", "status": "pending_new"}
    client.get_bars.return_value = __import__("pandas").DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )
    return client


def _make_mock_scanner(signals=None):
    scanner = MagicMock()
    scanner.return_value = signals or []
    return scanner


def test_run_cycle_returns_cycle_result():
    client = _make_mock_client()
    result = run_cycle(client, scan_fn=lambda *a, **kw: [], strategies=[], config_override={})
    assert isinstance(result, CycleResult)
    assert result.account_equity == 100000.0
    assert result.orders_placed == 0
    assert result.positions_closed == 0


def test_run_cycle_places_order_for_signal():
    client = _make_mock_client()
    signals = [
        Signal(
            asset="BTC/USDT",
            timeframe="1h",
            strategy="SupportResistanceStrategy",
            direction=1,
            entry_price=50000.0,
            sl=49000.0,
            tp=52000.0,
        )
    ]
    result = run_cycle(
        client,
        scan_fn=lambda *a, **kw: signals,
        strategies=[],
        config_override={},
    )
    assert result.orders_placed == 1
    client.place_bracket_order.assert_called_once()
    call_kwargs = client.place_bracket_order.call_args
    assert call_kwargs.kwargs["symbol"] == "BTC/USD"
    assert call_kwargs.kwargs["side"] == "buy"
    assert call_kwargs.kwargs["stop_loss"] == 49000.0
    assert call_kwargs.kwargs["take_profit"] == 52000.0


def test_run_cycle_skips_signal_for_existing_position():
    positions = [{"symbol": "BTC/USD", "qty": "0.001", "side": "long",
                  "created_at": "2024-01-01T00:00:00Z", "unrealized_pl": "10"}]
    client = _make_mock_client(positions=positions)
    signals = [
        Signal("BTC/USDT", "1h", "SR", 1, 50000.0, 49000.0, 52000.0),
    ]
    result = run_cycle(
        client,
        scan_fn=lambda *a, **kw: signals,
        strategies=[],
        config_override={},
    )
    assert result.orders_placed == 0


def test_run_cycle_respects_max_concurrent():
    from datetime import datetime, timezone, timedelta
    recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    positions = [
        {"symbol": f"ASSET{i}/USD", "qty": "1", "side": "long",
         "created_at": recent_time, "unrealized_pl": "0"}
        for i in range(15)  # already at max
    ]
    client = _make_mock_client(positions=positions)
    signals = [Signal("BTC/USDT", "1h", "SR", 1, 50000.0, 49000.0, 52000.0)]
    result = run_cycle(
        client,
        scan_fn=lambda *a, **kw: signals,
        strategies=[],
        config_override={"MAX_CONCURRENT": 15},
    )
    assert result.orders_placed == 0


def test_run_cycle_closes_timed_out_position():
    old_time = (
        __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        - __import__("datetime").timedelta(hours=101)
    ).isoformat()
    positions = [
        {"symbol": "BTC/USD", "qty": "0.001", "side": "long",
         "created_at": old_time, "unrealized_pl": "5"}
    ]
    client = _make_mock_client(positions=positions)
    result = run_cycle(
        client,
        scan_fn=lambda *a, **kw: [],
        strategies=[],
        config_override={"MAX_BARS": 100},
    )
    assert result.positions_closed == 1
    client.close_position.assert_called_once_with("BTC/USD")


def test_position_sizing_short():
    """Short signal: side=sell, stop_loss > entry."""
    client = _make_mock_client(equity="100000.00")
    signals = [
        Signal(
            asset="AAPL",
            timeframe="1h",
            strategy="SupportResistanceStrategy",
            direction=-1,
            entry_price=150.0,
            sl=153.0,
            tp=144.0,
        )
    ]
    result = run_cycle(
        client,
        scan_fn=lambda *a, **kw: signals,
        strategies=[],
        config_override={},
    )
    assert result.orders_placed == 1
    call_kwargs = client.place_bracket_order.call_args.kwargs
    assert call_kwargs["side"] == "sell"
    # qty = (100000 * 0.02) / (153 - 150) = 2000 / 3 = 666 shares
    assert call_kwargs["qty"] == 666


def test_run_cycle_handles_malformed_created_at():
    """Timeout check should handle malformed created_at gracefully."""
    positions = [
        {"symbol": "BTC/USD", "qty": "0.001", "side": "long",
         "created_at": "not-a-valid-date", "unrealized_pl": "5"}
    ]
    client = _make_mock_client(positions=positions)
    result = run_cycle(
        client,
        scan_fn=lambda *a, **kw: [],
        strategies=[],
        config_override={},
    )
    assert len(result.errors) == 1
    assert "timeout check error" in result.errors[0]
    assert result.positions_closed == 0


def test_run_cycle_handles_order_placement_exception():
    """Order placement exception should be logged and cycle continues."""
    client = _make_mock_client()
    client.place_bracket_order.side_effect = Exception("insufficient funds")
    signals = [
        Signal("BTC/USDT", "1h", "SR", 1, 50000.0, 49000.0, 52000.0)
    ]
    result = run_cycle(
        client,
        scan_fn=lambda *a, **kw: signals,
        strategies=[],
        config_override={},
    )
    assert result.orders_placed == 0
    assert len(result.errors) == 1
    assert "order error BTC/USDT" in result.errors[0]
