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
