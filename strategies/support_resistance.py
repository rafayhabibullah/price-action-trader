import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, touch_count: int = 2, zone_width_pct: float = 0.005):
        self.lookback = lookback
        self.touch_count = touch_count
        self.zone_width_pct = zone_width_pct

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "touch_count": self.touch_count,
            "zone_width_pct": self.zone_width_pct,
        }

    def _find_levels(self, df: pd.DataFrame) -> list[float]:
        """Find price levels touched >= touch_count times within zone_width_pct."""
        highs = df["high"].values
        lows = df["low"].values
        levels = []
        candidates = np.concatenate([highs, lows])
        for price in candidates:
            zone_lo = price * (1 - self.zone_width_pct)
            zone_hi = price * (1 + self.zone_width_pct)
            touches = np.sum((lows <= zone_hi) & (highs >= zone_lo))
            if touches >= self.touch_count:
                levels.append(float(price))
        # deduplicate nearby levels
        levels = sorted(set(levels))
        # Keep lowest level in each cluster (sorted ascending; first occurrence wins).
        # This favours the most conservative support boundary.
        deduped = []
        for lvl in levels:
            if not deduped or lvl > deduped[-1] * (1 + self.zone_width_pct * 2):
                deduped.append(lvl)
        return deduped

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        for i in range(self.lookback, len(df)):
            window = df.iloc[i - self.lookback:i]
            levels = self._find_levels(window)
            if not levels:
                continue

            close = df["close"].iloc[i]
            prev_close = df["close"].iloc[i - 1]

            for level in levels:
                zone_lo = level * (1 - self.zone_width_pct)
                zone_hi = level * (1 + self.zone_width_pct)

                # Bullish bounce off support: price dipped into zone and closed above
                if prev_close <= zone_hi and close > zone_hi:
                    result.at[df.index[i], "signal"] = 1
                    result.at[df.index[i], "sl"] = zone_lo * 0.998
                    result.at[df.index[i], "tp"] = close + 2 * (close - zone_lo * 0.998)
                    break

                # Bearish rejection at resistance: price touched zone and closed below
                if prev_close >= zone_lo and close < zone_lo:
                    result.at[df.index[i], "signal"] = -1
                    result.at[df.index[i], "sl"] = zone_hi * 1.002
                    result.at[df.index[i], "tp"] = close - 2 * (zone_hi * 1.002 - close)
                    break

        return result
