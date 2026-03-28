import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class CandlestickPatternStrategy(BaseStrategy):
    """
    Detects: Bullish/Bearish Engulfing, Pin Bars (Hammer/Shooting Star), Inside Bars.
    """

    def __init__(self, min_body_ratio: float = 0.6, confirmation_candles: int = 1,
                 pin_wick_ratio: float = 2.0):
        self.min_body_ratio = min_body_ratio
        self.confirmation_candles = confirmation_candles
        self.pin_wick_ratio = pin_wick_ratio

    def get_params(self) -> dict:
        return {
            "min_body_ratio": self.min_body_ratio,
            "confirmation_candles": self.confirmation_candles,
        }

    def _body(self, o, c): return abs(c - o)
    def _range(self, h, l): return h - l
    def _upper_wick(self, o, h, c): return h - max(o, c)
    def _lower_wick(self, o, l, c): return min(o, c) - l

    def _is_bullish_engulfing(self, prev_o, prev_c, curr_o, curr_c) -> bool:
        prev_bearish = prev_c < prev_o
        curr_bullish = curr_c > curr_o
        engulfs = curr_o <= prev_c and curr_c >= prev_o
        body_ok = self._body(curr_o, curr_c) >= self._body(prev_o, prev_c) * self.min_body_ratio
        return prev_bearish and curr_bullish and engulfs and body_ok

    def _is_bearish_engulfing(self, prev_o, prev_c, curr_o, curr_c) -> bool:
        prev_bullish = prev_c > prev_o
        curr_bearish = curr_c < curr_o
        engulfs = curr_o >= prev_c and curr_c <= prev_o
        body_ok = self._body(curr_o, curr_c) >= self._body(prev_o, prev_c) * self.min_body_ratio
        return prev_bullish and curr_bearish and engulfs and body_ok

    def _is_bullish_pin_bar(self, o, h, l, c) -> bool:
        lower_wick = self._lower_wick(o, l, c)
        body = self._body(o, c)
        total = self._range(h, l)
        if total == 0:
            return False
        return lower_wick >= body * self.pin_wick_ratio and lower_wick >= total * 0.6

    def _is_bearish_pin_bar(self, o, h, l, c) -> bool:
        upper_wick = self._upper_wick(o, h, c)
        body = self._body(o, c)
        total = self._range(h, l)
        if total == 0:
            return False
        return upper_wick >= body * self.pin_wick_ratio and upper_wick >= total * 0.6

    def _is_inside_bar(self, prev_h, prev_l, curr_h, curr_l) -> bool:
        return curr_h <= prev_h and curr_l >= prev_l

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result["sl"] = np.nan
        result["tp"] = np.nan

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        for i in range(1, len(df)):
            ts = df.index[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            po, ph, pl, pc = opens[i-1], highs[i-1], lows[i-1], closes[i-1]

            signal = 0
            sl = np.nan
            tp = np.nan

            if self._is_bullish_engulfing(po, pc, o, c):
                signal = 1
                sl = l * 0.998
                tp = c + 2 * (c - sl)
            elif self._is_bearish_engulfing(po, pc, o, c):
                signal = -1
                sl = h * 1.002
                tp = c - 2 * (sl - c)
            elif self._is_bullish_pin_bar(o, h, l, c):
                signal = 1
                sl = l * 0.998
                tp = c + 2 * (c - sl)
            elif self._is_bearish_pin_bar(o, h, l, c):
                signal = -1
                sl = h * 1.002
                tp = c - 2 * (sl - c)

            result.at[ts, "signal"] = signal
            if signal != 0:
                result.at[ts, "sl"] = sl
                result.at[ts, "tp"] = tp

        return result
