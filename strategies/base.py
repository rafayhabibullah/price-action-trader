# strategies/base.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    @abstractmethod
    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subclass implements this instead of generate_signals.
        Takes OHLCV DataFrame with columns [open, high, low, close, volume].
        Returns same DataFrame with added columns:
          - 'signal': 1=buy, -1=sell, 0=hold
          - 'sl': stop loss price (float, NaN when no signal)
          - 'tp': take profit price (float, NaN when no signal)
        """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with quality filters applied."""
        result = self._generate_raw_signals(df)
        return self._apply_filters(df, result)

    def _apply_filters(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Apply EMA50 trend + Volume > 1.5x avg filters."""
        out = signals_df

        # Filter 1: EMA50 trend alignment
        # Only longs when price > EMA50, shorts when price < EMA50
        ema50 = df["close"].ewm(span=50).mean()
        trend_mask = (
            ((out["signal"] == 1) & (df["close"] <= ema50)) |
            ((out["signal"] == -1) & (df["close"] >= ema50))
        )
        out.loc[trend_mask, "signal"] = 0
        out.loc[trend_mask, "sl"] = np.nan
        out.loc[trend_mask, "tp"] = np.nan

        # Filter 2: Volume > 1.5x 20-period average
        vol_ma = df["volume"].rolling(20).mean()
        vol_mask = (out["signal"] != 0) & (df["volume"] < vol_ma * 1.5)
        out.loc[vol_mask, "signal"] = 0
        out.loc[vol_mask, "sl"] = np.nan
        out.loc[vol_mask, "tp"] = np.nan

        return out

    @abstractmethod
    def get_params(self) -> dict:
        """Returns dict of parameter names to their current values."""
