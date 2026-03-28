import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class OrderBlockStrategy(BaseStrategy):
    """
    Order Block (Smart Money Concept):
    Bullish OB = last bearish candle before a bullish impulse (3+ consecutive up closes).
    Signal fires when price returns to the OB zone.
    """

    def __init__(self, block_size_pct: float = 0.003, invalidation_pct: float = 0.005,
                 impulse_candles: int = 3):
        self.block_size_pct = block_size_pct
        self.invalidation_pct = invalidation_pct
        self.impulse_candles = impulse_candles

    def get_params(self) -> dict:
        return {
            "block_size_pct": self.block_size_pct,
            "invalidation_pct": self.invalidation_pct,
            "impulse_candles": self.impulse_candles,
        }

    def _find_bullish_obs(self, df: pd.DataFrame) -> list[dict]:
        """Find bullish order blocks: last bearish candle before impulse."""
        obs = []
        closes = df["close"].values
        opens = df["open"].values
        lows = df["low"].values
        highs = df["high"].values

        for i in range(1, len(df) - self.impulse_candles):
            # current candle is bearish
            if closes[i] >= opens[i]:
                continue
            # followed by N bullish candles (impulse)
            if all(closes[i + k] > opens[i + k] for k in range(1, self.impulse_candles + 1)):
                ob_high = highs[i]
                ob_low = lows[i]
                # Filter out OBs too narrow to be meaningful
                if ob_low > 0 and (ob_high - ob_low) / ob_low < self.block_size_pct:
                    continue
                obs.append({
                    "idx": i,
                    "ob_high": ob_high,
                    "ob_low": ob_low,
                    "formed_at": df.index[i + self.impulse_candles],
                })
        return obs

    def _find_bearish_obs(self, df: pd.DataFrame) -> list[dict]:
        """Find bearish order blocks: last bullish candle before bearish impulse."""
        obs = []
        closes = df["close"].values
        opens = df["open"].values
        lows = df["low"].values
        highs = df["high"].values

        for i in range(1, len(df) - self.impulse_candles):
            if closes[i] <= opens[i]:
                continue
            if all(closes[i + k] < opens[i + k] for k in range(1, self.impulse_candles + 1)):
                ob_high = highs[i]
                ob_low = lows[i]
                if ob_low > 0 and (ob_high - ob_low) / ob_low < self.block_size_pct:
                    continue
                obs.append({
                    "idx": i,
                    "ob_high": ob_high,
                    "ob_low": ob_low,
                    "formed_at": df.index[i + self.impulse_candles],
                })
        return obs

    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        bullish_obs = self._find_bullish_obs(df)
        bearish_obs = self._find_bearish_obs(df)

        closes = df["close"].values
        lows = df["low"].values
        highs = df["high"].values

        invalidated_bull = set()
        invalidated_bear = set()

        for i in range(self.impulse_candles + 2, len(df)):
            ts = df.index[i]
            close = closes[i]

            for j, ob in enumerate(bullish_obs):
                if ob["formed_at"] >= ts or j in invalidated_bull:
                    continue
                ob_low = ob["ob_low"]
                ob_high = ob["ob_high"]
                invalidation = ob_low * (1 - self.invalidation_pct)
                # Permanently retire if price closes below invalidation
                if lows[i] < invalidation:
                    invalidated_bull.add(j)
                    continue
                # Price returns into OB zone = long entry
                if ob_low <= close <= ob_high:
                    if result.at[ts, "signal"] == 0:
                        result.at[ts, "signal"] = 1
                        result.at[ts, "sl"] = invalidation
                        result.at[ts, "tp"] = close + 2 * (close - invalidation)

            for j, ob in enumerate(bearish_obs):
                if ob["formed_at"] >= ts or j in invalidated_bear:
                    continue
                ob_low = ob["ob_low"]
                ob_high = ob["ob_high"]
                invalidation = ob_high * (1 + self.invalidation_pct)
                # Permanently retire if price closes above invalidation
                if highs[i] > invalidation:
                    invalidated_bear.add(j)
                    continue
                if ob_low <= close <= ob_high:
                    if result.at[ts, "signal"] == 0:
                        result.at[ts, "signal"] = -1
                        result.at[ts, "sl"] = invalidation
                        result.at[ts, "tp"] = close - 2 * (invalidation - close)

        return result
