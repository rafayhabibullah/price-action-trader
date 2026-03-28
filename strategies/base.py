# strategies/base.py
from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes OHLCV DataFrame with columns [open, high, low, close, volume].
        Returns same DataFrame with added columns:
          - 'signal': 1=buy, -1=sell, 0=hold
          - 'sl': stop loss price (float, NaN when no signal)
          - 'tp': take profit price (float, NaN when no signal)
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Returns dict of parameter names to their current values."""
