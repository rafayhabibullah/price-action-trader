import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MarketStructureStrategy(BaseStrategy):
    """
    Detects market structure: HH/HL (uptrend), LH/LL (downtrend).
    Signals on Break of Structure (BOS) and Change of Character (CHoCH).
    BOS: price breaks above last HH (bullish) or below last LL (bearish).
    CHoCH: uptrend breaks below last HL (bearish shift) or downtrend breaks above last LH.
    """

    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback

    def get_params(self) -> dict:
        return {"swing_lookback": self.swing_lookback}

    def _find_swing_highs(self, highs: np.ndarray) -> list[int]:
        n = self.swing_lookback
        result = []
        for i in range(n, len(highs) - n):
            if highs[i] == max(highs[i - n:i + n + 1]):
                result.append(i)
        return result

    def _find_swing_lows(self, lows: np.ndarray) -> list[int]:
        n = self.swing_lookback
        result = []
        for i in range(n, len(lows) - n):
            if lows[i] == min(lows[i - n:i + n + 1]):
                result.append(i)
        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        for i in range(self.swing_lookback * 2 + 1, len(df)):
            ts = df.index[i]
            close = closes[i]

            prev_sh = [idx for idx in swing_highs if idx < i]
            prev_sl = [idx for idx in swing_lows if idx < i]

            if len(prev_sh) < 2 or len(prev_sl) < 2:
                continue

            last_sh = highs[prev_sh[-1]]
            prev_sh_val = highs[prev_sh[-2]]
            last_sl = lows[prev_sl[-1]]
            prev_sl_val = lows[prev_sl[-2]]

            # Bullish BOS: close breaks above last swing high (HH pattern)
            if close > last_sh and last_sh > prev_sh_val:
                if result.at[ts, "signal"] == 0:
                    sl = last_sl * 0.998
                    result.at[ts, "signal"] = 1
                    result.at[ts, "sl"] = sl
                    result.at[ts, "tp"] = close + 2 * (close - sl)

            # Bearish BOS: close breaks below last swing low (LL pattern)
            elif close < last_sl and last_sl < prev_sl_val:
                if result.at[ts, "signal"] == 0:
                    sl = last_sh * 1.002
                    result.at[ts, "signal"] = -1
                    result.at[ts, "sl"] = sl
                    result.at[ts, "tp"] = close - 2 * (sl - close)

        return result
