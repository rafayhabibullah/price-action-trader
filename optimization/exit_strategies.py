from abc import ABC, abstractmethod


class BaseExitStrategy(ABC):
    @abstractmethod
    def get_params(self) -> dict:
        ...


class FixedRR(BaseExitStrategy):
    def __init__(self, rr_ratio: float = 2.0):
        self.rr_ratio = rr_ratio

    def take_profit(self, entry: float, sl: float, direction: int) -> float:
        risk = abs(entry - sl)
        return entry + direction * risk * self.rr_ratio

    def get_params(self) -> dict:
        return {"rr_ratio": self.rr_ratio}


class TrailingStop(BaseExitStrategy):
    def __init__(self, trail_pct: float = 0.02):
        self.trail_pct = trail_pct

    def update_sl(self, current_sl: float, current_price: float,
                  direction: int, entry: float) -> float:
        if direction == 1:
            new_sl = current_price * (1 - self.trail_pct)
            return max(current_sl, new_sl)
        else:
            new_sl = current_price * (1 + self.trail_pct)
            return min(current_sl, new_sl)

    def get_params(self) -> dict:
        return {"trail_pct": self.trail_pct}


class PartialTP(BaseExitStrategy):
    def __init__(self, first_tp_r: float = 1.0, first_tp_pct: float = 0.5):
        self.first_tp_r = first_tp_r
        self.first_tp_pct = first_tp_pct

    def first_target(self, entry: float, sl: float, direction: int) -> float:
        risk = abs(entry - sl)
        return entry + direction * risk * self.first_tp_r

    def get_params(self) -> dict:
        return {"first_tp_r": self.first_tp_r, "first_tp_pct": self.first_tp_pct}
