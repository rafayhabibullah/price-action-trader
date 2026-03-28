import importlib
import config.settings as settings_module
from pathlib import Path


def test_defaults(monkeypatch):
    for var in ("STARTING_CAPITAL", "MAX_RISK_PER_TRADE", "CACHE_DIR", "REPORTS_DIR"):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(settings_module)
    from config.settings import STARTING_CAPITAL, MAX_RISK_PER_TRADE, TIMEFRAMES, CACHE_DIR, REPORTS_DIR
    assert STARTING_CAPITAL == 1000.0
    assert MAX_RISK_PER_TRADE == 0.02
    assert "1h" in TIMEFRAMES
    assert isinstance(CACHE_DIR, Path)
    assert isinstance(REPORTS_DIR, Path)
