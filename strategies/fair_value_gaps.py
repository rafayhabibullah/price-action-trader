import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class FairValueGapStrategy(BaseStrategy):
    """
    Fair Value Gap (ICT concept):
    Bullish FVG: candle[i].high < candle[i+2].low — gap between i and i+2.
    Bearish FVG: candle[i].low > candle[i+2].high — gap between i and i+2.
    Signal fires when price returns to fill part of the gap.
    """

    def __init__(self, gap_min_pct: float = 0.002, fill_pct: float = 0.5,
                 direction_bias: str = "both"):
        self.gap_min_pct = gap_min_pct
        self.fill_pct = fill_pct
        self.direction_bias = direction_bias  # "both", "bullish", "bearish"

    def get_params(self) -> dict:
        return {
            "gap_min_pct": self.gap_min_pct,
            "fill_pct": self.fill_pct,
            "direction_bias": self.direction_bias,
        }

    def _find_fvgs(self, df: pd.DataFrame) -> list[dict]:
        fvgs = []
        highs = df["high"].values
        lows = df["low"].values

        for i in range(len(df) - 2):
            # Bullish FVG: gap between candle i high and candle i+2 low
            if self.direction_bias in ("both", "bullish"):
                gap_lo = highs[i]
                gap_hi = lows[i + 2]
                if gap_hi > gap_lo:
                    gap_size = (gap_hi - gap_lo) / gap_lo
                    if gap_size >= self.gap_min_pct:
                        fvgs.append({
                            "type": "bullish",
                            "gap_lo": gap_lo,
                            "gap_hi": gap_hi,
                            "mid": gap_lo + (gap_hi - gap_lo) * self.fill_pct,
                            "formed_at": df.index[i + 2],
                        })

            # Bearish FVG: gap between candle i low and candle i+2 high
            if self.direction_bias in ("both", "bearish"):
                gap_hi = lows[i]
                gap_lo = highs[i + 2]
                if gap_hi > gap_lo:
                    gap_size = (gap_hi - gap_lo) / gap_lo
                    if gap_size >= self.gap_min_pct:
                        fvgs.append({
                            "type": "bearish",
                            "gap_lo": gap_lo,
                            "gap_hi": gap_hi,
                            "mid": gap_lo + (gap_hi - gap_lo) * self.fill_pct,
                            "formed_at": df.index[i + 2],
                        })
        return fvgs

    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        fvgs = self._find_fvgs(df)
        closes = df["close"].values

        for i in range(3, len(df)):
            ts = df.index[i]
            close = closes[i]

            for fvg in fvgs:
                if fvg["formed_at"] >= ts:
                    continue
                if result.at[ts, "signal"] != 0:
                    break

                if fvg["type"] == "bullish" and fvg["gap_lo"] <= close <= fvg["gap_hi"]:
                    result.at[ts, "signal"] = 1
                    result.at[ts, "sl"] = fvg["gap_lo"] * 0.998
                    result.at[ts, "tp"] = close + 2 * (close - fvg["gap_lo"] * 0.998)

                elif fvg["type"] == "bearish" and fvg["gap_lo"] <= close <= fvg["gap_hi"]:
                    result.at[ts, "signal"] = -1
                    result.at[ts, "sl"] = fvg["gap_hi"] * 1.002
                    result.at[ts, "tp"] = close - 2 * (fvg["gap_hi"] * 1.002 - close)

        return result
