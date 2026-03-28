import pandas as pd
import numpy as np
import pytest
from backtest.positions import simulate_trades

@pytest.fixture
def signal_df():
    """OHLCV with one buy signal at index 2, SL=98, TP=106."""
    data = {
        "open":   [100, 101, 102, 103, 104, 105, 106, 107],
        "high":   [101, 102, 103, 107, 105, 106, 107, 108],
        "low":    [99,  100, 101, 102, 103, 104, 105, 106],
        "close":  [101, 101, 102, 106, 104, 105, 106, 107],
        "volume": [1000.0] * 8,
        "signal": [0, 0, 1, 0, 0, 0, 0, 0],
        "sl":     [np.nan, np.nan, 98.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        "tp":     [np.nan, np.nan, 106.0, np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=8, freq="1h"))

def test_tp_hit_records_trade(signal_df):
    trades = simulate_trades(signal_df, starting_capital=1000.0, risk_per_trade=0.02)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "tp"
    assert trades.iloc[0]["pnl"] > 0

def test_sl_hit_records_loss():
    data = {
        "open":   [100, 101, 102, 103, 97,  100],
        "high":   [101, 102, 103, 104, 101, 101],
        "low":    [99,  100, 101, 97,  96,  99],   # low[3]=97 hits SL=98
        "close":  [101, 101, 102, 97,  100, 100],
        "volume": [1000.0] * 6,
        "signal": [0, 0, 1, 0, 0, 0],
        "sl":     [np.nan, np.nan, 98.0, np.nan, np.nan, np.nan],
        "tp":     [np.nan, np.nan, 106.0, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=6, freq="1h"))
    trades = simulate_trades(df, starting_capital=1000.0, risk_per_trade=0.02)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "sl"
    assert trades.iloc[0]["pnl"] < 0

def test_equity_compounds(signal_df):
    trades = simulate_trades(signal_df, starting_capital=1000.0, risk_per_trade=0.02)
    assert trades.iloc[-1]["equity"] != 1000.0

def test_no_signals_no_trades():
    data = {
        "open": [100.0]*5, "high": [101.0]*5, "low": [99.0]*5, "close": [100.0]*5,
        "volume": [1000.0]*5, "signal": [0]*5,
        "sl": [np.nan]*5, "tp": [np.nan]*5,
    }
    df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=5, freq="1h"))
    trades = simulate_trades(df, starting_capital=1000.0, risk_per_trade=0.02)
    assert len(trades) == 0
