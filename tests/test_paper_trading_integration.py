"""
Integration test: runs a full cycle against the real Alpaca paper API.
Skipped automatically in CI (no ALPACA_API_KEY set).

To run locally:
    pytest tests/test_paper_trading_integration.py -v -s
"""
import os
import pytest
from dataclasses import asdict

pytestmark = pytest.mark.skipif(
    not os.environ.get("ALPACA_API_KEY"),
    reason="ALPACA_API_KEY not set — skipping integration test",
)


def test_full_cycle_returns_valid_result():
    from trading.alpaca_client import AlpacaClient
    from trading.scanner import scan_all
    from trading.position_manager import run_cycle
    from trading import config
    from strategies.support_resistance import SupportResistanceStrategy
    from strategies.fair_value_gaps import FairValueGapStrategy
    from strategies.candlestick_patterns import CandlestickPatternStrategy
    from strategies.market_structure import MarketStructureStrategy

    client = AlpacaClient(
        api_key=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.ALPACA_BASE_URL,
    )

    strategies = [
        SupportResistanceStrategy(),
        FairValueGapStrategy(),
        CandlestickPatternStrategy(),
        MarketStructureStrategy(),
    ]

    result = run_cycle(client, scan_fn=scan_all, strategies=strategies)

    assert result.account_equity > 0, "Should have positive equity"
    assert result.account_equity <= 200_000, "Equity should be reasonable"
    assert result.orders_placed >= 0
    assert result.positions_closed >= 0
    assert isinstance(result.errors, list)

    import json
    print("\n" + json.dumps(asdict(result), indent=2))


def test_account_is_active():
    from trading.alpaca_client import AlpacaClient
    from trading import config

    client = AlpacaClient(
        api_key=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.ALPACA_BASE_URL,
    )

    acct = client.get_account()
    assert acct.get("status") == "ACTIVE"
    assert float(acct.get("equity", 0)) > 0
